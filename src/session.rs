// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! `ATTACH` for DataFusion.
//!
//! # Why this module exists
//!
//! DataFusion has no `ATTACH`. It has extension points for *tables*
//! ([`TableProviderFactory`], reached by `CREATE EXTERNAL TABLE`) and for
//! *functions* ([`FunctionFactory`], reached by `CREATE FUNCTION`) — but nothing
//! that mounts a whole remote catalog from SQL. `SessionContext::register_catalog`
//! exists, and is the right destination; there is simply no statement wired to it.
//!
//! The gap is narrower than it looks, because `ATTACH` **parses**. sqlparser-rs
//! understands it, so it reaches DataFusion's planner as an ordinary
//! [`sqlparser::ast::Statement`] and is refused there:
//!
//! ```text
//! This feature is not implemented: Unsupported SQL statement: ATTACH 'example' AS ex
//! ```
//!
//! That is a planning error, not a parse error — which means the statement can be
//! recognised before planning and handled instead of refused. That is all this
//! module does: parse, intercept `ATTACH`/`DETACH`, delegate everything else
//! untouched.
//!
//! # Syntax
//!
//! ```sql
//! ATTACH 'example?location=vgi-fixture-worker' AS ex;
//! DETACH ex;
//! ```
//!
//! The target string is `<catalog>?<key>=<value>&…`, and `location` is required.
//! DuckDB writes the same thing as
//! `ATTACH 'example' AS ex (TYPE vgi, LOCATION 'vgi-fixture-worker')`, with the
//! options in a trailing parenthesised list.
//!
//! **That trailing list is why options live in the string here.** sqlparser's
//! `ATTACH` grammar models exactly two DuckDB options, `READ_ONLY` and `TYPE`
//! ([`AttachDuckDBDatabaseOption`]) — anything else is a hard parse error:
//!
//! ```text
//! ATTACH 'example' AS ex (TYPE VGI, LOCATION 'worker');
//! SQL error: ParserError("Expected: expected one of: ), READ_ONLY, TYPE,
//!                        found: LOCATION at Line: 1, Column: 35")
//! ```
//!
//! A parse error cannot be intercepted — by then there is no statement to
//! recognise. So the option list is the one piece of DuckDB's spelling that
//! cannot be reached from here without changing sqlparser itself. `TYPE VGI` does
//! parse, and is accepted and ignored, so `ATTACH '…' AS ex (TYPE VGI)` works if
//! you prefer to write it.
//!
//! [`TableProviderFactory`]: datafusion::catalog::TableProviderFactory
//! [`FunctionFactory`]: datafusion::execution::context::FunctionFactory
//! [`AttachDuckDBDatabaseOption`]: datafusion::sql::sqlparser::ast::AttachDuckDBDatabaseOption

use std::collections::{BTreeMap, HashMap};
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex, OnceLock};

use datafusion::catalog::MemoryCatalogProvider;
use datafusion::common::{plan_datafusion_err, plan_err, Result as DFResult};
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::async_udf::AsyncScalarUDF;
use datafusion::logical_expr::AggregateUDF;
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::{DFParser, Statement as DFStatement};
use datafusion::sql::sqlparser::ast::{
    Expr, Statement as SQLStatement, Value, ValueWithSpan, VisitMut,
};

use crate::{VgiAggregateUdf, VgiCatalogProvider, VgiConnection, VgiScalarUdf, VgiTableFunction};

/// Private struct-field prefix used to carry a SQL named table-function
/// argument through DataFusion's positional-only `TableFunctionArgs` API.
pub(crate) const NAMED_ARG_PREFIX: &str = "__vgi_datafusion_named_arg__";

#[derive(Debug, Clone)]
struct SqlMacro {
    parameters: Vec<String>,
    definition: String,
}

type SessionMacros = HashMap<String, HashMap<String, SqlMacro>>;

fn macro_registry() -> &'static Mutex<SessionMacros> {
    static REGISTRY: OnceLock<Mutex<SessionMacros>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Run one SQL statement, handling `ATTACH` and `DETACH` for VGI catalogs.
///
/// Every other statement is planned and executed by DataFusion exactly as
/// [`SessionContext::sql`] would — this is a pre-pass, not a replacement
/// front end.
///
/// ```no_run
/// # use datafusion::prelude::SessionContext;
/// # async fn go() -> datafusion::common::Result<()> {
/// let ctx = SessionContext::new();
/// vgi_datafusion::sql(&ctx, "ATTACH 'example?location=vgi-fixture-worker' AS ex").await?;
/// let df = vgi_datafusion::sql(&ctx, "SELECT count(*) FROM ex.main.ten_thousand").await?;
/// df.show().await?;
/// # Ok(())
/// # }
/// ```
pub async fn sql(ctx: &SessionContext, query: &str) -> DFResult<DataFrame> {
    // DuckDB's own ATTACH spelling has to be handled before sqlparser sees it:
    // its grammar models exactly two options (READ_ONLY, TYPE), so
    // `(TYPE vgi, LOCATION '...')` is a *parse* error, and a parse error leaves
    // no statement to intercept. See `parse_duckdb_attach`.
    if let Some(spec) = parse_duckdb_attach(query) {
        attach(ctx, &spec?).await?;
        return ctx.read_empty();
    }
    let state = ctx.state();
    let dialect = state.config_options().sql_parser.dialect;
    let statement = state.sql_to_statement(query, &dialect)?;

    match classify(&statement)? {
        Some(Intercepted::Attach { target, alias }) => {
            let spec = AttachSpec::parse(&target, &alias)?;
            attach(ctx, &spec).await?;
            ctx.read_empty()
        }
        Some(Intercepted::Detach { alias }) => {
            detach(ctx, &alias)?;
            ctx.read_empty()
        }
        None => {
            let mut statement = statement;
            rewrite_vgi_sql(ctx, &mut statement)?;
            let plan = state.statement_to_plan(statement).await?;
            ctx.execute_logical_plan(plan).await
        }
    }
}

/// Parse DuckDB's `ATTACH '<catalog>' AS <alias> (TYPE vgi, LOCATION '<loc>')`.
///
/// `None` when the statement is not an ATTACH, so everything else is untouched.
///
/// # Why this is hand-parsed rather than intercepted on the AST
///
/// Every other statement this module handles is recognised *after* sqlparser
/// has parsed it. That is impossible here: sqlparser's ATTACH grammar models
/// exactly two DuckDB options, `READ_ONLY` and `TYPE`
/// ([`AttachDuckDBDatabaseOption`]), so a `LOCATION` option is a hard parse
/// error —
///
/// ```text
/// ParserError("Expected: expected one of: ), READ_ONLY, TYPE, found: LOCATION")
/// ```
///
/// — and by then there is no statement left to intercept. Since this is the
/// spelling the entire `.test` corpus uses (248 of its ATTACHes), and since a
/// failed ATTACH makes every later record in the file fail too, the option list
/// is parsed directly.
///
/// The grammar accepted is deliberately small: the keyword, a quoted catalog
/// name, `AS`, an alias, and an optional parenthesised list of `key value`
/// pairs where a value may be quoted, bare, or numeric. `TYPE` is accepted and
/// ignored; `LOCATION`/`PATH` supplies the worker.
///
/// [`AttachDuckDBDatabaseOption`]: datafusion::sql::sqlparser::ast::AttachDuckDBDatabaseOption
fn parse_duckdb_attach(sql: &str) -> Option<DFResult<AttachSpec>> {
    let trimmed = sql.trim().trim_end_matches(';');
    let mut rest = trimmed.strip_prefix_ci("ATTACH")?;
    rest = rest.trim_start();
    // `IF NOT EXISTS` is legal DuckDB; re-attaching is idempotent here anyway.
    if let Some(r) = rest.strip_prefix_ci("IF NOT EXISTS") {
        rest = r.trim_start();
    }

    let (catalog, rest) = match take_quoted(rest) {
        Some(v) => v,
        // A bare (unquoted) target is not a form the corpus uses, and guessing
        // would risk swallowing an ATTACH meant for something else.
        None => {
            return Some(plan_err!(
                "ATTACH target must be a quoted string: {trimmed}"
            ))
        }
    };
    let rest = rest.trim_start();
    let rest = match rest.strip_prefix_ci("AS") {
        Some(r) => r.trim_start(),
        None => return Some(plan_err!("ATTACH is missing `AS <alias>`: {trimmed}")),
    };

    let (alias, rest) = take_ident(rest);
    if alias.is_empty() {
        return Some(plan_err!("ATTACH is missing an alias: {trimmed}"));
    }

    let mut options = BTreeMap::new();
    let rest = rest.trim_start();
    if let Some(inner) = rest.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        for pair in split_top_level_commas(inner) {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let (key, value) = take_ident(pair);
            let value = value.trim();
            let value = value
                .strip_prefix('\'')
                .and_then(|v| v.strip_suffix('\''))
                .unwrap_or(value);
            options.insert(key.to_ascii_lowercase(), value.to_string());
        }
    } else if !rest.is_empty() {
        return Some(plan_err!("unexpected text after ATTACH alias: {rest}"));
    }

    // `TYPE vgi` is the DuckDB way of naming the storage extension; it carries
    // no information here, where the only storage is VGI.
    options.remove("type");

    match options
        .remove("location")
        .or_else(|| options.remove("path"))
    {
        Some(location) if !location.is_empty() => Some(Ok(AttachSpec {
            catalog,
            alias,
            location,
            options,
        })),
        // No LOCATION option: this is the query-string spelling
        // (`'example?location=…'`), where the worker rides in the target
        // itself. Both forms are supported, so hand it to that parser rather
        // than failing — including for its error message, which explains the
        // form the caller was probably reaching for.
        _ => Some(AttachSpec::parse(&catalog, &alias)),
    }
}

/// Take a single-quoted string, returning it and the remainder.
fn take_quoted(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    let rest = s.strip_prefix('\'')?;
    let end = rest.find('\'')?;
    Some((rest[..end].to_string(), &rest[end + 1..]))
}

/// Take a bare identifier, returning it and the remainder.
fn take_ident(s: &str) -> (String, &str) {
    let s = s.trim_start();
    let end = s
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .unwrap_or(s.len());
    (s[..end].to_string(), &s[end..])
}

/// Split on commas that are not inside quotes.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_quote = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' => in_quote = !in_quote,
            ',' if !in_quote => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Case-insensitive prefix matching, for SQL keywords.
trait StripPrefixCi {
    fn strip_prefix_ci(&self, prefix: &str) -> Option<&str>;
}

impl StripPrefixCi for str {
    fn strip_prefix_ci(&self, prefix: &str) -> Option<&str> {
        let head = self.get(..prefix.len())?;
        head.eq_ignore_ascii_case(prefix)
            .then(|| &self[prefix.len()..])
    }
}

/// Collapse `catalog.schema.f(args)` into a single dotted identifier so the
/// call reaches the function it names.
///
/// # Why a rewrite is needed for table functions but not scalars
///
/// Both kinds live in one flat, session-wide map — DataFusion's catalog holds
/// tables only, and `SchemaProvider` has no function surface at all. The two
/// paths then disagree about what a qualified name means:
///
/// * A **scalar** call flattens the whole path into the lookup key, so
///   `SELECT example.main.double(1)` looks up `"example.main.double"`. Register
///   the UDF under that name and it simply resolves — no rewrite.
/// * A **table function** call takes only the *first* identifier
///   (`sql/src/relation/mod.rs`: `name.0.first()`), so
///   `FROM example.main.sequence(10)` looks up `"example"`. It reaches a
///   function — the wrong one — and `TableFunctionArgs` carries no name for the
///   implementation to notice.
///
/// So the name is collapsed here, before planning, into one *unquoted*
/// identifier whose value is the dotted path. Quoting is not an alternative:
/// the lookup key keeps the quote marks (`'"example.main.sequence"' not found`).
///
/// # The guard that matters
///
/// Only a relation **with arguments** is rewritten. A qualified name without
/// parentheses is an ordinary table reference — `ex.data.ten_thousand_table` —
/// which already resolves through the catalog and must not be touched.
///
/// Upstream tracks the underlying gap as apache/datafusion#18021; a PR that
/// added schema-scoped table functions (#18022) was closed in favour of #15095,
/// which is itself dormant.
fn rewrite_vgi_sql(ctx: &SessionContext, statement: &mut DFStatement) -> DFResult<()> {
    use datafusion::sql::sqlparser::ast::{
        Expr as SQLExpr, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName,
        ObjectNamePart, SelectItem, SetExpr, TableFactor, VisitorMut,
    };

    struct Rewrite<'a> {
        ctx: &'a SessionContext,
    }

    impl VisitorMut for Rewrite<'_> {
        type Break = Box<datafusion::common::DataFusionError>;

        fn pre_visit_table_factor(&mut self, tf: &mut TableFactor) -> ControlFlow<Self::Break> {
            if let TableFactor::Table {
                name,
                args: Some(args),
                ..
            } = tf
            {
                if name.0.len() > 1 {
                    let dotted = name
                        .0
                        .iter()
                        .filter_map(|part| part.as_ident().map(|i| i.value.clone()))
                        .collect::<Vec<_>>()
                        .join(".");
                    if !dotted.is_empty() {
                        *name = ObjectName(vec![ObjectNamePart::Identifier(Ident::new(dotted))]);
                    }
                }

                // DataFusion's TableFactor::Table planner rejects named
                // FunctionArgs before a TableFunctionImpl can see them. Carry
                // each one as a private, single-field struct literal instead;
                // the VGI implementation unwraps it back into Arguments::named.
                // This preserves names and ordering without changing DataFusion.
                for argument in &mut args.args {
                    let FunctionArg::Named { name, arg, .. } = argument else {
                        continue;
                    };
                    let FunctionArgExpr::Expr(value) = arg else {
                        continue;
                    };
                    *argument = FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::Struct {
                        values: vec![SQLExpr::Named {
                            expr: Box::new(value.clone()),
                            name: Ident::new(format!("{NAMED_ARG_PREFIX}{}", name.value)),
                        }],
                        fields: vec![],
                    }));
                }
            }
            ControlFlow::Continue(())
        }

        fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<Self::Break> {
            match expand_scalar_macro(self.ctx, expr) {
                Ok(Some(expanded)) => *expr = expanded,
                Ok(None) => {}
                Err(error) => return ControlFlow::Break(Box::new(error)),
            }
            ControlFlow::Continue(())
        }
    }

    fn expand_scalar_macro(ctx: &SessionContext, expr: &SQLExpr) -> DFResult<Option<SQLExpr>> {
        let SQLExpr::Function(function) = expr else {
            return Ok(None);
        };
        let path = function
            .name
            .0
            .iter()
            .filter_map(|part| part.as_ident().map(|ident| ident.value.as_str()))
            .collect::<Vec<_>>();
        let [catalog_name, schema_name, macro_name] = path.as_slice() else {
            return Ok(None);
        };

        let key = format!("{catalog_name}.{schema_name}.{macro_name}").to_ascii_lowercase();
        let info = macro_registry()
            .lock()
            .ok()
            .and_then(|sessions| sessions.get(&ctx.session_id())?.get(&key).cloned());
        let Some(info) = info else {
            return Ok(None);
        };

        let FunctionArguments::List(arguments) = &function.args else {
            return plan_err!(
                "VGI scalar macro `{}` requires an argument list",
                function.name
            );
        };
        if arguments.duplicate_treatment.is_some() || !arguments.clauses.is_empty() {
            return plan_err!(
                "VGI scalar macro `{}` does not accept DISTINCT or argument clauses",
                function.name
            );
        }
        let actual = arguments
            .args
            .iter()
            .map(|argument| match argument {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Ok(expr.clone()),
                other => plan_err!(
                    "VGI scalar macro `{}` only accepts positional expression arguments, found {other}",
                    function.name
                ),
            })
            .collect::<DFResult<Vec<_>>>()?;
        if actual.len() != info.parameters.len() {
            return plan_err!(
                "VGI scalar macro `{}` expects {} argument(s), received {}",
                function.name,
                info.parameters.len(),
                actual.len()
            );
        }

        let mut parsed = DFParser::parse_sql(&format!("SELECT {}", info.definition))?;
        let Some(DFStatement::Statement(statement)) = parsed.pop_front() else {
            return plan_err!(
                "VGI scalar macro `{}` has an empty definition",
                function.name
            );
        };
        let SQLStatement::Query(query) = statement.as_ref() else {
            return plan_err!("VGI scalar macro `{}` is not an expression", function.name);
        };
        let SetExpr::Select(select) = query.body.as_ref() else {
            return plan_err!("VGI scalar macro `{}` is not an expression", function.name);
        };
        let [SelectItem::UnnamedExpr(expanded)] = select.projection.as_slice() else {
            return plan_err!("VGI scalar macro `{}` is not one expression", function.name);
        };
        let mut expanded = expanded.clone();

        struct Substitute<'a> {
            parameters: &'a [String],
            actual: &'a [SQLExpr],
        }
        impl VisitorMut for Substitute<'_> {
            type Break = ();

            fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
                let SQLExpr::Identifier(identifier) = expr else {
                    return ControlFlow::Continue(());
                };
                if let Some(index) = self
                    .parameters
                    .iter()
                    .position(|parameter| parameter.eq_ignore_ascii_case(&identifier.value))
                {
                    *expr = self.actual[index].clone();
                }
                ControlFlow::Continue(())
            }
        }

        let _ = expanded.visit(&mut Substitute {
            parameters: &info.parameters,
            actual: &actual,
        });
        Ok(Some(expanded))
    }

    match statement {
        DFStatement::Statement(inner) => {
            // `visit` walks the whole SQL AST — CTEs, subqueries, joins — so
            // nested calls are covered without hand-rolling the recursion.
            if let ControlFlow::Break(error) = inner.as_mut().visit(&mut Rewrite { ctx }) {
                return Err(*error);
            }
        }
        // DataFusion parses EXPLAIN into its own wrapper rather than a
        // sqlparser Statement. Rewrite the wrapped statement exactly as if it
        // had been submitted directly; otherwise named table-function
        // arguments reach DataFusion's positional-only planner unchanged.
        DFStatement::Explain(explain) => rewrite_vgi_sql(ctx, explain.statement.as_mut())?,
        _ => {}
    }
    Ok(())
}

/// A statement this module handles itself.
enum Intercepted {
    Attach { target: String, alias: String },
    Detach { alias: String },
}

/// Recognise `ATTACH` / `DETACH`; `None` means "hand it back to DataFusion".
///
/// Both dialects are matched. `ATTACH 'x' AS y` parses as
/// [`SQLStatement::AttachDatabase`] under the default dialect and as
/// [`SQLStatement::AttachDuckDBDatabase`] under `DuckDB`, so a harness that
/// switches `datafusion.sql_parser.dialect` keeps working.
fn classify(statement: &DFStatement) -> DFResult<Option<Intercepted>> {
    let DFStatement::Statement(inner) = statement else {
        return Ok(None);
    };
    Ok(match inner.as_ref() {
        SQLStatement::AttachDatabase {
            schema_name,
            database_file_name,
            ..
        } => Some(Intercepted::Attach {
            target: string_literal(database_file_name)?,
            alias: schema_name.value.clone(),
        }),
        SQLStatement::AttachDuckDBDatabase {
            database_path,
            database_alias,
            ..
        } => {
            // The DuckDB arm carries the path as an `Ident` whose `value` is the
            // literal's contents, quote style recorded separately.
            let alias = database_alias
                .as_ref()
                .map(|i| i.value.clone())
                .unwrap_or_else(|| database_path.value.clone());
            Some(Intercepted::Attach {
                target: database_path.value.clone(),
                alias,
            })
        }
        SQLStatement::DetachDuckDBDatabase { database_alias, .. } => Some(Intercepted::Detach {
            alias: database_alias.value.clone(),
        }),
        _ => None,
    })
}

fn string_literal(expr: &Expr) -> DFResult<String> {
    match expr {
        Expr::Value(ValueWithSpan {
            value: Value::SingleQuotedString(s) | Value::DoubleQuotedString(s),
            ..
        }) => Ok(s.clone()),
        other => plan_err!("ATTACH target must be a string literal, found: {other}"),
    }
}

/// A parsed `ATTACH` target.
#[derive(Debug, PartialEq, Eq)]
pub struct AttachSpec {
    /// The VGI catalog name to attach — the part before `?`.
    pub catalog: String,
    /// The name it is mounted under in DataFusion.
    pub alias: String,
    /// Where the worker lives (the `location` option).
    pub location: String,
    /// Every other option, verbatim.
    pub options: BTreeMap<String, String>,
}

impl AttachSpec {
    /// Parse `'<catalog>?<key>=<value>&…'`.
    ///
    /// Values are taken verbatim — no percent-decoding — so a subprocess command
    /// with arguments can be written plainly:
    /// `'example?location=uv run --project ~/vgi-python vgi-fixture-worker'`.
    /// The consequence is that `?`, `&` and `=` cannot appear inside a value.
    pub fn parse(target: &str, alias: &str) -> DFResult<Self> {
        let (catalog, query) = match target.split_once('?') {
            Some((c, q)) => (c, q),
            None => (target, ""),
        };
        if catalog.is_empty() {
            return plan_err!("ATTACH target names no catalog: {target:?}");
        }

        let mut options = BTreeMap::new();
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let Some((k, v)) = pair.split_once('=') else {
                return plan_err!("ATTACH option {pair:?} is not key=value");
            };
            options.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }

        let location = options.remove("location").ok_or_else(|| {
            plan_datafusion_err!(
                "ATTACH target {target:?} has no `location`; \
                 write ATTACH '{catalog}?location=<worker>' AS {alias}"
            )
        })?;
        if location.is_empty() {
            return plan_err!("ATTACH `location` is empty");
        }

        Ok(Self {
            catalog: catalog.to_string(),
            alias: alias.to_string(),
            location,
            options,
        })
    }

    /// The transport implied by `location`.
    ///
    /// Delegates to [`VgiConnection::from_location`], so every scheme the
    /// client speaks — bare command, `http://`, `unix://`, `tcp://`, `launch:`
    /// — is reachable from SQL, and the spelling matches the DuckDB
    /// extension's `LOCATION` exactly.
    pub fn connection(&self) -> DFResult<VgiConnection> {
        VgiConnection::from_location(&self.location)
    }
}

async fn attach(ctx: &SessionContext, spec: &AttachSpec) -> DFResult<()> {
    let conn = spec.connection()?;
    let provider = VgiCatalogProvider::discover(conn.clone(), &spec.catalog).await?;
    register_table_functions(ctx, &conn, spec, &provider);
    register_scalar_functions(ctx, &conn, spec, &provider);
    register_aggregate_functions(ctx, &conn, spec, &provider);
    register_scalar_macros(ctx, spec, &provider);
    ctx.register_catalog(&spec.alias, provider);
    Ok(())
}

fn register_scalar_macros(ctx: &SessionContext, spec: &AttachSpec, provider: &VgiCatalogProvider) {
    let Ok(mut sessions) = macro_registry().lock() else {
        return;
    };
    let macros = sessions.entry(ctx.session_id()).or_default();
    let alias_prefix = format!("{}.", spec.alias.to_ascii_lowercase());
    macros.retain(|name, _| !name.starts_with(&alias_prefix));
    for (schema, info) in provider.scalar_macros() {
        macros.insert(
            format!("{}.{}.{}", spec.alias, schema, info.name).to_ascii_lowercase(),
            SqlMacro {
                parameters: info.parameters.clone(),
                definition: info.definition.clone(),
            },
        );
    }
}

/// Publish the catalog's table functions so they are callable **with
/// arguments**: `SELECT * FROM ex.main.sequence(10)`.
///
/// # Why they need a second surface at all
///
/// A [`CatalogProvider`] yields tables, and a table has no arguments. Most VGI
/// table functions take some, and their output schema depends on them, so they
/// cannot be reached that way — only the zero-argument ones can. DataFusion's
/// answer is a separate registry, so a function that takes arguments is
/// published here as well.
///
/// # Two names, two audiences
///
/// That registry is flat and session-wide, so the hierarchy has to live *inside*
/// the key. Each function is registered twice, pointing at the same
/// implementation:
///
/// * **`catalog.schema.function`** — the fully qualified path as a single key.
///   With [`rewrite_qualified_table_functions`] in front of the planner, this
///   makes ordinary qualified SQL work, and because the schema is part of the
///   key it disambiguates correctly: a worker publishing the same function name
///   in two schemas gets two distinct entries.
/// * **`<alias>_function`** — a short name for callers who would rather not
///   qualify.
///
/// The short name is **first-wins**: it cannot carry a schema, so two schemas
/// publishing one name collide, and clobbering a name someone else registered
/// would be worse than not publishing. The qualified name always works.
///
/// # Why the short name is *not* the worker's `global_function_prefix`
///
/// That prefix looks like the right answer and is not. It belongs to
/// `CatalogAttachResult::global_functions` — a set the worker explicitly
/// nominates for publication into the host's global namespace, four of them on
/// the reference fixture worker against 143 table functions in one schema.
/// Applying `vgi_example_` to all of them would claim every function is one of
/// the worker's declared globals.
///
/// Publishing that set under its own prefix is a real feature — the extension's
/// `RegisterVgiGlobalFunctions` — and a follow-up here; the prefix is reserved
/// for it. [`VgiCatalogProvider::global_function_prefix`] already carries it.
fn register_table_functions(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
    provider: &VgiCatalogProvider,
) {
    let state = ctx.state();

    for (schema_name, schema) in provider.vgi_schemas() {
        for function in datafusion::catalog::SchemaProvider::table_names(schema.as_ref()) {
            let metadata = schema.table_function_metadata(&function).cloned();
            let make = || {
                Arc::new(VgiTableFunction::new(
                    conn.clone(),
                    &spec.catalog,
                    schema_name,
                    &function,
                    metadata.clone(),
                ))
            };

            for name in publish_names(&spec.alias, schema_name, &function) {
                // First-wins throughout: the two shorter forms cannot carry a
                // schema, so a name published in two schemas collides on them,
                // and the fully qualified form is always there as the
                // unambiguous way to say which one you meant.
                if !state.table_functions().contains_key(&name) {
                    ctx.register_udtf(&name, make());
                }
            }
        }
    }
}

/// The names one worker function is published under.
///
/// DataFusion's function registries are flat and session-wide — its catalog
/// holds tables only — so the hierarchy has to live inside the key, and one
/// function gets several keys:
///
/// 1. `alias.schema.function` — fully qualified and unambiguous. Two schemas
///    publishing the same name stay distinct here and nowhere else.
/// 2. `alias.function` — what the corpus overwhelmingly writes (95 distinct
///    call sites against 17 fully-qualified ones), because in DuckDB the
///    default schema is implied.
/// 3. `alias_function` — for callers who would rather not qualify at all.
/// 4. `function` — the bare name.
///
/// Only the first is guaranteed; the rest are first-wins, since none of them
/// can express which schema was meant.
///
/// The bare name exists because the corpus leans on it heavily: a file does
/// `USE example` and then calls `vgi_sum(...)` unqualified, which in DuckDB
/// resolves through the current catalog. DataFusion keeps functions in a flat
/// registry that `USE` does not touch, so without a bare registration those
/// calls cannot resolve at all — 102 records in the aggregate group alone.
///
/// First-wins matters most here: the registry already holds DataFusion's
/// built-ins, so a worker function named `sum` or `abs` is skipped rather than
/// shadowing the engine's own.
fn publish_names(alias: &str, schema: &str, function: &str) -> Vec<String> {
    vec![
        format!("{alias}.{schema}.{function}"),
        format!("{alias}.{function}"),
        format!("{alias}_{function}"),
        function.to_string(),
    ]
}

/// Publish the catalog's scalar functions into DataFusion's function registry.
///
/// Same two-name scheme as [`register_table_functions`], and the qualified name
/// needs **no** rewrite here: a scalar call flattens its whole qualified path
/// into the lookup key, so registering under `catalog.schema.function` makes
/// `SELECT ex.main.double(1)` resolve directly.
///
/// Each name is a separate [`VgiScalarUdf`] because DataFusion keys the registry
/// on the UDF's own `name()`; they differ only in that key and dispatch to the
/// same worker function.
fn register_scalar_functions(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
    provider: &VgiCatalogProvider,
) {
    let state = ctx.state();

    for (schema_name, schema) in provider.vgi_schemas() {
        for (function, specs) in schema.scalars() {
            let register = |name: String| {
                if state.scalar_functions().contains_key(&name) {
                    return;
                }
                let udf = VgiScalarUdf::discovered(
                    conn.clone(),
                    &spec.catalog,
                    schema_name,
                    function,
                    &name,
                    specs.clone(),
                );
                ctx.register_udf(AsyncScalarUDF::new(Arc::new(udf)).into_scalar_udf());
            };
            for name in publish_names(&spec.alias, schema_name, function) {
                register(name);
            }
        }
    }
}

/// Publish the catalog's aggregate functions.
///
/// Same three-name scheme as the others, and like a scalar the qualified name
/// needs no rewrite — an aggregate call flattens its whole path into the lookup
/// key, so `ex.main.my_agg(x)` resolves against a registration under that name.
fn register_aggregate_functions(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
    provider: &VgiCatalogProvider,
) {
    let state = ctx.state();

    for (schema_name, schema) in provider.vgi_schemas() {
        for function in schema.aggregates() {
            for name in publish_names(&spec.alias, schema_name, function) {
                if state.aggregate_functions().contains_key(&name) {
                    continue;
                }
                ctx.register_udaf(AggregateUDF::new_from_impl(VgiAggregateUdf::new(
                    conn.clone(),
                    &spec.catalog,
                    schema_name,
                    function,
                    &name,
                )));
            }
        }
    }
}

/// Make an attached catalog unreachable.
///
/// [`CatalogProviderList`] has `register_catalog` but no deregister, so the alias
/// is rebound to an empty catalog rather than removed. The observable effect —
/// the tables are gone — matches `DETACH`; `SHOW TABLES` and `information_schema`
/// will still list the alias itself with no schemas under it.
///
/// The VGI extension has a comparable limitation in the other direction: its own
/// `DETACH` cannot unregister a COPY format or a secret provider, because DuckDB
/// offers no unload API for either.
///
/// [`CatalogProviderList`]: datafusion::catalog::CatalogProviderList
fn detach(ctx: &SessionContext, alias: &str) -> DFResult<()> {
    if ctx.catalog(alias).is_none() {
        return plan_err!("no catalog attached as {alias:?}");
    }
    if let Ok(mut sessions) = macro_registry().lock() {
        if let Some(macros) = sessions.get_mut(&ctx.session_id()) {
            let prefix = format!("{}.", alias.to_ascii_lowercase());
            macros.retain(|name, _| !name.starts_with(&prefix));
        }
    }
    ctx.register_catalog(alias, Arc::new(MemoryCatalogProvider::new()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(target: &str) -> AttachSpec {
        AttachSpec::parse(target, "ex").expect("parses")
    }

    #[test]
    fn parses_catalog_and_location() {
        let s = spec("example?location=vgi-fixture-worker");
        assert_eq!(s.catalog, "example");
        assert_eq!(s.alias, "ex");
        assert_eq!(s.location, "vgi-fixture-worker");
        assert!(s.options.is_empty());
    }

    #[test]
    fn keeps_extra_options_and_lowercases_keys() {
        let s = spec("example?location=w&Pool=false&cache=true");
        assert_eq!(s.options.get("pool").map(String::as_str), Some("false"));
        assert_eq!(s.options.get("cache").map(String::as_str), Some("true"));
        // `location` is consumed, not left in the bag.
        assert!(!s.options.contains_key("location"));
    }

    #[test]
    fn location_may_carry_spaces_for_a_subprocess_argv() {
        let s = spec("example?location=uv run --project /tmp/p vgi-fixture-worker");
        assert_eq!(s.location, "uv run --project /tmp/p vgi-fixture-worker");
        assert_eq!(s.connection().unwrap().label(), "uv");
    }

    #[test]
    fn http_location_selects_the_http_transport() {
        assert_eq!(
            spec("example?location=http://127.0.0.1:8080")
                .connection()
                .unwrap()
                .label(),
            "http://127.0.0.1:8080"
        );
    }

    #[test]
    fn missing_location_names_the_fix() {
        let err = AttachSpec::parse("example", "ex").unwrap_err().to_string();
        assert!(err.contains("location=<worker>"), "unhelpful: {err}");
    }

    #[test]
    fn rejects_malformed_options() {
        assert!(AttachSpec::parse("example?location=w&bare", "ex").is_err());
        assert!(AttachSpec::parse("?location=w", "ex").is_err());
        assert!(AttachSpec::parse("example?location=", "ex").is_err());
    }
}
