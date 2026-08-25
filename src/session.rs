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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use datafusion::arrow::array::{ArrayRef, BinaryArray, LargeBinaryArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::MemoryCatalogProvider;
use datafusion::common::{plan_datafusion_err, plan_err, Result as DFResult};
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::async_udf::AsyncScalarUDF;
use datafusion::logical_expr::AggregateUDF;
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::{DFParser, Statement as DFStatement};
use datafusion::sql::sqlparser::ast::{
    Expr, Statement as SQLStatement, Value, ValueWithSpan, VisitMut, Visitor,
};
use datafusion::sql::sqlparser::dialect::SnowflakeDialect;

use crate::{
    VgiAggregateUdf, VgiCatalogProvider, VgiConnection, VgiRuntime, VgiScalarUdf, VgiTableFunction,
};

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

fn runtime_registry() -> &'static Mutex<HashMap<String, Weak<VgiRuntime>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Weak<VgiRuntime>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Default)]
struct RegisteredNames {
    scalar: HashSet<String>,
    aggregate: HashSet<String>,
    table: HashSet<String>,
}

type SessionRegistrations = HashMap<String, HashMap<String, RegisteredNames>>;

fn registration_registry() -> &'static Mutex<SessionRegistrations> {
    static REGISTRY: OnceLock<Mutex<SessionRegistrations>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

enum RegistrationKind {
    Scalar,
    Aggregate,
    Table,
}

fn record_registration(ctx: &SessionContext, alias: &str, kind: RegistrationKind, name: String) {
    let mut sessions = registration_registry().lock().unwrap();
    let registered = sessions
        .entry(ctx.session_id())
        .or_default()
        .entry(alias.to_ascii_lowercase())
        .or_default();
    match kind {
        RegistrationKind::Scalar => registered.scalar.insert(name),
        RegistrationKind::Aggregate => registered.aggregate.insert(name),
        RegistrationKind::Table => registered.table.insert(name),
    };
}

fn deregister_alias_functions(ctx: &SessionContext, alias: &str) {
    let registered = registration_registry()
        .lock()
        .ok()
        .and_then(|mut sessions| {
            sessions
                .get_mut(&ctx.session_id())?
                .remove(&alias.to_ascii_lowercase())
        });
    let Some(registered) = registered else {
        return;
    };
    for name in registered.scalar {
        ctx.deregister_udf(&name);
    }
    for name in registered.aggregate {
        ctx.deregister_udaf(&name);
    }
    for name in registered.table {
        ctx.deregister_udtf(&name);
    }
}

fn session_runtime(ctx: &SessionContext) -> Arc<VgiRuntime> {
    if let Some(runtime) = ctx.copied_config().get_extension::<VgiRuntime>() {
        return runtime;
    }
    let session_id = ctx.session_id();
    let mut runtimes = runtime_registry().lock().unwrap();
    runtimes.retain(|_, runtime| runtime.strong_count() > 0);
    if let Some(runtime) = runtimes.get(&session_id).and_then(Weak::upgrade) {
        return runtime;
    }
    let runtime = Arc::new(VgiRuntime::default());
    runtimes.insert(session_id, Arc::downgrade(&runtime));
    runtime
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
    let statement = match state.sql_to_statement(query, &dialect) {
        Ok(statement) => statement,
        Err(original) if contains_time_travel_clause(query) => {
            // The default and DuckDB dialects parse `AT` as a table alias.
            // Snowflake's dialect enables sqlparser's TableVersion AST while
            // preserving the rest of the statement for DataFusion planning.
            let mut statements = DFParser::parse_sql_with_dialect(query, &SnowflakeDialect {})
                .map_err(|_| original)?;
            if statements.len() != 1 {
                return plan_err!("the context currently only supports a single SQL statement");
            }
            statements.pop_front().expect("length checked")
        }
        Err(error) => return Err(error),
    };

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
            let temporary_tables = rewrite_time_travel(ctx, &mut statement).await?;
            let rewritten = rewrite_vgi_sql(ctx, &mut statement);
            let plan = match rewritten {
                Ok(()) => state.statement_to_plan(statement).await,
                Err(error) => Err(error),
            };
            // Logical plans retain their provider Arcs, so these private names
            // only need to exist while the SQL planner resolves the relation.
            for table in temporary_tables {
                let _ = ctx.deregister_table(table);
            }
            let plan = plan?;
            ctx.execute_logical_plan(plan).await
        }
    }
}

fn contains_time_travel_clause(sql: &str) -> bool {
    let upper = sql.to_ascii_uppercase();
    upper.contains(" AT (") || upper.contains(" AT(")
}

/// Replace each VGI `table AT (...)` relation with a private, version-bound
/// provider for the duration of logical planning.
async fn rewrite_time_travel(
    ctx: &SessionContext,
    statement: &mut DFStatement,
) -> DFResult<Vec<String>> {
    use datafusion::sql::sqlparser::ast::{
        Expr as SQLExpr, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName,
        ObjectNamePart, TableAlias, TableFactor, TableVersion, UnaryOperator, VisitorMut,
    };

    #[derive(Debug)]
    struct HistoricalTable {
        hidden: String,
        catalog: String,
        schema: String,
        table: String,
        at: vgi_client::At,
    }

    fn expr_value(expr: &SQLExpr, unit: &str) -> DFResult<String> {
        match (unit, expr) {
            (_, SQLExpr::Value(value)) if matches!(value.value, Value::Null) => {
                plan_err!("VGI time travel value must not be NULL")
            }
            ("VERSION", SQLExpr::Value(value)) => match &value.value {
                Value::Number(number, _) => Ok(number.to_string()),
                _ => plan_err!("VGI AT VERSION value must be an integer literal"),
            },
            (
                "VERSION",
                SQLExpr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr,
                },
            ) => match expr.as_ref() {
                SQLExpr::Value(value) => match &value.value {
                    Value::Number(number, _) => Ok(format!("-{number}")),
                    _ => plan_err!("VGI AT VERSION value must be an integer literal"),
                },
                _ => plan_err!("VGI AT VERSION value must be an integer literal"),
            },
            ("TIMESTAMP", SQLExpr::TypedString(value))
                if value
                    .data_type
                    .to_string()
                    .to_ascii_uppercase()
                    .starts_with("TIMESTAMP") =>
            {
                value.value.clone().into_string().ok_or_else(|| {
                    plan_datafusion_err!("VGI AT TIMESTAMP value must be a timestamp literal")
                })
            }
            ("TIMESTAMP", SQLExpr::Value(value)) => value.clone().into_string().ok_or_else(|| {
                plan_datafusion_err!("VGI AT TIMESTAMP value must be a timestamp literal")
            }),
            ("VERSION", _) => plan_err!("VGI AT VERSION value must be an integer literal"),
            ("TIMESTAMP", _) => {
                plan_err!("VGI AT TIMESTAMP value must be a timestamp literal")
            }
            _ => plan_err!("VGI time travel unit must be VERSION or TIMESTAMP"),
        }
    }

    fn parse_at(version: TableVersion) -> DFResult<vgi_client::At> {
        let TableVersion::Function(SQLExpr::Function(function)) = version else {
            return plan_err!("VGI time travel uses AT (VERSION => ...) or AT (TIMESTAMP => ...)");
        };
        if !function.name.to_string().eq_ignore_ascii_case("AT") {
            return plan_err!("VGI time travel only supports the AT clause");
        }
        let FunctionArguments::List(arguments) = function.args else {
            return plan_err!("VGI AT requires exactly one named argument");
        };
        if arguments.args.len() != 1 || !arguments.clauses.is_empty() {
            return plan_err!("VGI AT requires exactly one named argument");
        }
        let (name, value) = match &arguments.args[0] {
            FunctionArg::Named {
                name,
                arg: FunctionArgExpr::Expr(value),
                ..
            } => (name.value.as_str(), value),
            FunctionArg::ExprNamed {
                name: SQLExpr::Identifier(name),
                arg: FunctionArgExpr::Expr(value),
                ..
            } => (name.value.as_str(), value),
            _ => return plan_err!("VGI AT requires VERSION => value or TIMESTAMP => value"),
        };
        let unit = name.to_ascii_uppercase();
        let value = expr_value(value, &unit)?;
        Ok(vgi_client::At { unit, value })
    }

    static NEXT_HIDDEN_TABLE: AtomicU64 = AtomicU64::new(0);
    struct Extract {
        tables: Vec<HistoricalTable>,
    }
    impl VisitorMut for Extract {
        type Break = Box<datafusion::common::DataFusionError>;

        fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<Self::Break> {
            let TableFactor::Table {
                name,
                alias,
                version,
                ..
            } = factor
            else {
                return ControlFlow::Continue(());
            };
            if version.is_none() {
                return ControlFlow::Continue(());
            }
            let parts = name
                .0
                .iter()
                .filter_map(|part| part.as_ident().cloned())
                .collect::<Vec<_>>();
            let [catalog, schema, table] = parts.as_slice() else {
                return ControlFlow::Break(Box::new(plan_datafusion_err!(
                    "VGI time travel requires a fully qualified catalog.schema.table name"
                )));
            };
            let at = match parse_at(version.take().expect("matched Some above")) {
                Ok(at) => at,
                Err(error) => return ControlFlow::Break(Box::new(error)),
            };
            let hidden = format!(
                "__vgi_time_travel_{}",
                NEXT_HIDDEN_TABLE.fetch_add(1, Ordering::Relaxed)
            );
            if alias.is_none() {
                *alias = Some(TableAlias {
                    explicit: false,
                    name: table.clone(),
                    columns: vec![],
                    at: None,
                });
            }
            *name = ObjectName(vec![ObjectNamePart::Identifier(Ident::new(&hidden))]);
            self.tables.push(HistoricalTable {
                hidden,
                catalog: catalog.value.clone(),
                schema: schema.value.clone(),
                table: table.value.clone(),
                at,
            });
            ControlFlow::Continue(())
        }
    }

    let mut extracted = Extract { tables: Vec::new() };
    match statement {
        DFStatement::Statement(inner) => {
            if let ControlFlow::Break(error) = inner.as_mut().visit(&mut extracted) {
                return Err(*error);
            }
        }
        DFStatement::Explain(explain) => {
            return Box::pin(rewrite_time_travel(ctx, explain.statement.as_mut())).await;
        }
        _ => return Ok(Vec::new()),
    }

    let mut providers = Vec::with_capacity(extracted.tables.len());
    for table in extracted.tables {
        let catalog = ctx
            .catalog(&table.catalog)
            .ok_or_else(|| plan_datafusion_err!("catalog `{}` does not exist", table.catalog))?;
        let provider = catalog
            .downcast_ref::<VgiCatalogProvider>()
            .ok_or_else(|| {
                plan_datafusion_err!(
                    "time travel relation `{}.{}` is not in a VGI catalog",
                    table.schema,
                    table.table
                )
            })?;
        let bound = provider
            .table_at(&table.schema, &table.table, table.at)
            .await?;
        providers.push((table.hidden, bound));
    }

    let mut registered = Vec::with_capacity(providers.len());
    for (hidden, provider) in providers {
        ctx.register_table(hidden.clone(), provider)?;
        registered.push(hidden);
    }
    Ok(registered)
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
    let (alias, rest) = match rest.strip_prefix_ci("AS") {
        Some(r) => {
            let (alias, rest) = take_ident(r.trim_start());
            if alias.is_empty() {
                return Some(plan_err!("ATTACH is missing an alias: {trimmed}"));
            }
            (alias, rest)
        }
        None => (catalog.clone(), rest),
    };

    let mut options = BTreeMap::new();
    let rest = rest.trim_start();
    if let Some(inner) = rest.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        for pair in split_top_level_commas(inner) {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let (key, value) = take_ident(pair);
            if key.is_empty() {
                return Some(plan_err!("ATTACH option has no name: {pair}"));
            }
            let value = value
                .trim_start()
                .strip_prefix(":=")
                .or_else(|| value.trim_start().strip_prefix('='))
                .unwrap_or(value.trim_start())
                .trim();
            if value.is_empty() {
                return Some(plan_err!("ATTACH option `{key}` has no value"));
            }
            options.insert(key.to_ascii_lowercase(), value.to_string());
        }
    } else if !rest.is_empty() {
        return Some(plan_err!("unexpected text after ATTACH alias: {rest}"));
    }

    // `TYPE vgi` is the DuckDB way of naming the storage extension; it carries
    // no information here, where the only storage is VGI.
    if let Some(kind) = options.remove("type") {
        let kind = match option_string(&kind) {
            Ok(kind) => kind,
            Err(error) => return Some(Err(error)),
        };
        if !kind.eq_ignore_ascii_case("vgi") {
            return Some(plan_err!("unsupported ATTACH TYPE {kind:?}; expected VGI"));
        }
    }

    match options
        .remove("location")
        .or_else(|| options.remove("path"))
    {
        Some(location) => match option_string(&location) {
            Ok(location) if !location.is_empty() => Some(Ok(AttachSpec {
                catalog,
                alias,
                location,
                options,
            })),
            Ok(_) => Some(plan_err!("ATTACH `location` is empty")),
            Err(error) => Some(Err(error)),
        },
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
    let mut out = String::new();
    let mut chars = rest.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if ch == '\'' {
            if chars.peek().is_some_and(|(_, next)| *next == '\'') {
                out.push('\'');
                chars.next();
                continue;
            }
            return Some((out, &rest[index + ch.len_utf8()..]));
        }
        out.push(ch);
    }
    None
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
    let mut quote = None;
    let mut depth = 0usize;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if let Some(q) = quote {
            if c == q {
                if chars.peek().is_some_and(|(_, next)| *next == q) {
                    chars.next();
                } else {
                    quote = None;
                }
            } else if c == '\\' && q == '"' {
                chars.next();
            }
            continue;
        }
        match c {
            '\'' | '"' => quote = Some(c),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

fn option_string(raw: &str) -> DFResult<String> {
    let raw = raw.trim();
    if raw.starts_with('\'') {
        let Some((value, rest)) = take_quoted(raw) else {
            return plan_err!("unterminated ATTACH string value");
        };
        if !rest.trim().is_empty() {
            return plan_err!("unexpected text after ATTACH string value: {rest}");
        }
        Ok(value)
    } else if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
        Ok(raw[1..raw.len() - 1].replace("\"\"", "\""))
    } else {
        Ok(raw.to_string())
    }
}

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
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
#[derive(PartialEq, Eq)]
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

impl std::fmt::Debug for AttachSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let option_names = self.options.keys().collect::<Vec<_>>();
        f.debug_struct("AttachSpec")
            .field("catalog", &self.catalog)
            .field("alias", &self.alias)
            .field("location", &self.location)
            .field("option_names", &option_names)
            .finish()
    }
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
            options.insert(k.trim().to_ascii_lowercase(), sql_string(v.trim()));
        }

        let location = options.remove("location").ok_or_else(|| {
            plan_datafusion_err!(
                "ATTACH target {target:?} has no `location`; \
                 write ATTACH '{catalog}?location=<worker>' AS {alias}"
            )
        })?;
        let location = option_string(&location)?;
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
        use std::time::Duration;
        use vgi_client::{PoolConfig, WorkerPool};

        let bool_option = |name: &str, default: bool| -> DFResult<bool> {
            let Some(raw) = self.options.get(name) else {
                return Ok(default);
            };
            option_string(raw)?.parse::<bool>().map_err(|_| {
                datafusion::common::plan_datafusion_err!(
                    "ATTACH option `{name}` must be true or false"
                )
            })
        };
        let integer_option = |name: &str| -> DFResult<Option<u64>> {
            self.options
                .get(name)
                .map(|raw| {
                    option_string(raw)?.parse::<u64>().map_err(|_| {
                        datafusion::common::plan_datafusion_err!(
                            "ATTACH option `{name}` must be a non-negative integer"
                        )
                    })
                })
                .transpose()
        };

        let location = vgi_client::VgiLocation::parse(&self.location).map_err(crate::to_df)?;
        let worker_debug = bool_option("worker_debug", false)?;
        let launcher_idle_timeout =
            integer_option("launcher_idle_timeout")?.map(Duration::from_secs);
        let launcher_state_dir = self
            .options
            .get("launcher_state_dir")
            .map(|raw| option_string(raw))
            .transpose()?;
        if launcher_state_dir.as_deref() == Some("") {
            return plan_err!("launcher_state_dir, if set, must not be empty");
        }
        if (launcher_idle_timeout.is_some() || launcher_state_dir.is_some())
            && !matches!(location, vgi_client::VgiLocation::Launch(_))
        {
            return plan_err!(
                "launcher_idle_timeout / launcher_state_dir are only valid for `launch:` LOCATIONs"
            );
        }
        let use_pool = bool_option("pool", true)?;
        let pool = if !use_pool {
            WorkerPool::new(PoolConfig::disabled())
        } else {
            let defaults = PoolConfig::default();
            WorkerPool::new(PoolConfig {
                max_idle: integer_option("pool_max")?
                    .map(|v| v as usize)
                    .unwrap_or(defaults.max_idle),
                idle_timeout: integer_option("pool_timeout")?
                    .map(Duration::from_secs)
                    .unwrap_or(defaults.idle_timeout),
            })
        };
        let cache_enabled = bool_option("cache", true)?;
        let mut connection = VgiConnection::pooled(location, pool)
            .with_cache_enabled(cache_enabled)
            .with_connection_options(vgi_client::ConnectionOptions {
                worker_debug,
                launcher_idle_timeout,
                launcher_state_dir: launcher_state_dir.map(Into::into),
                rpc_timeout: None,
            });

        let bearer = self
            .options
            .get("bearer_token")
            .map(|v| option_string(v))
            .transpose()?;
        let refresh = self
            .options
            .get("oauth_refresh_token")
            .map(|v| option_string(v))
            .transpose()?;
        if bearer.is_some() && refresh.is_some() {
            return plan_err!("cannot specify both bearer_token and oauth_refresh_token");
        }
        if let Some(token) = bearer {
            connection =
                connection.with_auth(Arc::new(vgi_client::auth::BearerAuth::new(token)))?;
        } else if let Some(token) = refresh {
            let auth = vgi_client::auth::OAuthAuth::new(
                Box::new(vgi_client::auth::oauth::UreqTransport),
                Box::new(vgi_client::auth::StderrInteraction),
            )
            .with_refresh_token(token);
            connection = connection.with_auth(Arc::new(auth))?;
        }
        Ok(connection)
    }
}

async fn attach(ctx: &SessionContext, spec: &AttachSpec) -> DFResult<()> {
    let runtime = session_runtime(ctx);
    let companions_enabled = spec
        .options
        .get("attach_companions")
        .map(|value| option_string(value))
        .transpose()?
        .map(|value| value.parse::<bool>())
        .transpose()
        .map_err(|_| {
            plan_datafusion_err!("ATTACH option `attach_companions` must be true or false")
        })?
        .unwrap_or(true);
    let companions = attach_one(ctx, spec, Arc::clone(&runtime)).await?;
    if !companions_enabled {
        return Ok(());
    }

    let mut pending = std::collections::VecDeque::new();
    for companion in companions {
        pending.push_back((companion, 1usize));
    }
    let mut seen =
        std::collections::HashSet::from([format!("{}\0{}", spec.location, spec.catalog)]);
    while let Some((companion, depth)) = pending.pop_front() {
        if depth > 8 {
            if companion.required {
                return plan_err!("required VGI companion catalog nesting exceeds 8 levels");
            }
            continue;
        }
        let companion_spec = match companion_spec(spec, &companion) {
            Ok(Some(spec)) => spec,
            Ok(None) => continue,
            Err(error) if companion.required => return Err(error),
            Err(_) => continue,
        };
        if ctx.catalog(&companion_spec.alias).is_some() {
            if companion.required {
                return plan_err!(
                    "required VGI companion alias `{}` is already registered",
                    companion_spec.alias
                );
            }
            continue;
        }
        let key = format!("{}\0{}", companion_spec.location, companion_spec.catalog);
        if !seen.insert(key) {
            if companion.required {
                return plan_err!("required VGI companion catalog cycle detected");
            }
            continue;
        }
        let nested = attach_one(ctx, &companion_spec, Arc::clone(&runtime)).await?;
        for child in nested {
            pending.push_back((child, depth + 1));
        }
    }
    Ok(())
}

fn companion_spec(
    parent: &AttachSpec,
    info: &vgi_client::dtos::AttachCatalogInfo,
) -> DFResult<Option<AttachSpec>> {
    if !info.db_type.eq_ignore_ascii_case("vgi") {
        return if info.required {
            plan_err!(
                "required companion `{}` has type `{}`; vgi-datafusion only attaches VGI companions",
                info.alias,
                info.db_type
            )
        } else {
            Ok(None)
        };
    }
    if !info.secret_ref.is_empty() {
        return plan_err!(
            "VGI companion `{}` requires secret reference `{}`; companion secrets are not exposed by DataFusion",
            info.alias,
            info.secret_ref
        );
    }
    let alias = if info.alias.is_empty() {
        info.target.split('?').next().unwrap_or(&info.target)
    } else {
        &info.alias
    };
    let mut options = info
        .options
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), sql_string(value)))
        .collect::<BTreeMap<_, _>>();
    let (catalog, embedded) = info.target.split_once('?').unwrap_or((&info.target, ""));
    for pair in embedded.split('&').filter(|pair| !pair.is_empty()) {
        if let Some((name, value)) = pair.split_once('=') {
            options
                .entry(name.to_ascii_lowercase())
                .or_insert_with(|| sql_string(value));
        }
    }
    let location = options
        .remove("location")
        .map(|value| option_string(&value))
        .transpose()?
        .unwrap_or_else(|| parent.location.clone());
    Ok(Some(AttachSpec {
        catalog: catalog.to_string(),
        alias: alias.to_string(),
        location,
        options,
    }))
}

async fn attach_one(
    ctx: &SessionContext,
    spec: &AttachSpec,
    runtime: Arc<VgiRuntime>,
) -> DFResult<Vec<vgi_client::dtos::AttachCatalogInfo>> {
    let started = std::time::Instant::now();
    let mut conn = spec.connection()?.with_runtime(runtime);
    crate::diagnostics::register(ctx, Arc::clone(conn.runtime()));
    let options = build_attach_options(ctx, &conn, spec).await?;
    conn = conn.with_catalog_attach_options(&spec.catalog, options);
    let provider = VgiCatalogProvider::discover(conn.clone(), &spec.catalog).await?;
    conn.runtime().set_catalog_metadata(
        &spec.alias,
        crate::runtime::VgiCatalogMetadata {
            functions: provider.functions().cloned().collect(),
            macros: provider.metadata_macros().cloned().collect(),
            global_function_prefix: provider.global_function_prefix().to_string(),
            global_functions: provider.global_functions().to_vec(),
        },
    );
    // Re-attaching an alias refreshes its flat function registrations as well
    // as its catalog provider.
    deregister_alias_functions(ctx, &spec.alias);
    register_table_functions(ctx, &conn, spec, &provider);
    register_scalar_functions(ctx, &conn, spec, &provider);
    register_aggregate_functions(ctx, &conn, spec, &provider);
    register_global_functions(ctx, &conn, spec, &provider)?;
    register_scalar_macros(ctx, spec, &provider);
    ctx.register_catalog(&spec.alias, provider.clone());
    register_views(ctx, spec, &provider).await;
    let mut event = crate::VgiEvent::new("catalog.attached");
    event.catalog = Some(spec.alias.clone());
    event.duration = Some(started.elapsed());
    event.message = Some(format!("worker catalog `{}`", spec.catalog));
    conn.runtime().emit(event);
    Ok(provider.companion_catalogs().to_vec())
}

/// Plan worker-declared views against their owning schema and install ordinary
/// DataFusion `ViewTable`s. A broken definition does not make ATTACH fail: the
/// view remains discoverable and reports its retained error when queried.
async fn register_views(ctx: &SessionContext, spec: &AttachSpec, provider: &VgiCatalogProvider) {
    use datafusion::datasource::ViewTable;
    use datafusion::sql::sqlparser::ast::{
        Ident, ObjectName, ObjectNamePart, TableFactor, VisitorMut,
    };

    struct Qualify<'a> {
        catalog: &'a str,
        schema: &'a str,
    }

    impl VisitorMut for Qualify<'_> {
        type Break = Box<datafusion::common::DataFusionError>;

        fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<Self::Break> {
            let TableFactor::Table { name, .. } = factor else {
                return ControlFlow::Continue(());
            };
            if name.0.len() == 1 {
                let Some(local) = name.0[0].as_ident().map(|ident| ident.value.clone()) else {
                    return ControlFlow::Continue(());
                };
                *name = ObjectName(vec![
                    ObjectNamePart::Identifier(Ident::new(self.catalog)),
                    ObjectNamePart::Identifier(Ident::new(self.schema)),
                    ObjectNamePart::Identifier(Ident::new(local)),
                ]);
            }
            ControlFlow::Continue(())
        }
    }

    for (schema_name, schema) in provider.vgi_schemas() {
        for (view_name, info) in schema.views() {
            let planned =
                async {
                    let state = ctx.state();
                    let dialect = state.config_options().sql_parser.dialect;
                    let mut statement = state.sql_to_statement(&info.definition, &dialect)?;
                    if let DFStatement::Statement(inner) = &mut statement {
                        if let ControlFlow::Break(error) = inner.as_mut().visit(&mut Qualify {
                            catalog: &spec.alias,
                            schema: schema_name,
                        }) {
                            return Err(*error);
                        }
                    }
                    rewrite_vgi_sql(ctx, &mut statement)?;
                    let plan = state.statement_to_plan(statement).await?;
                    Ok::<
                        Arc<dyn datafusion::catalog::TableProvider>,
                        datafusion::common::DataFusionError,
                    >(Arc::new(ViewTable::new(
                        plan,
                        Some(info.definition.clone()),
                    )))
                }
                .await
                .map_err(|error| error.to_string());
            schema.install_view(view_name, planned);
        }
    }
}

fn metadata_volatility(
    info: &vgi_client::dtos::FunctionInfo,
) -> datafusion::logical_expr::Volatility {
    use datafusion::logical_expr::Volatility;
    match info.stability.as_ref().map(|value| value.0.as_str()) {
        Some(value) if value.eq_ignore_ascii_case("VOLATILE") => Volatility::Volatile,
        Some(value) if value.eq_ignore_ascii_case("CONSISTENT_WITHIN_QUERY") => Volatility::Stable,
        _ => Volatility::Immutable,
    }
}

fn metadata_secrets(info: &vgi_client::dtos::FunctionInfo) -> Vec<vgi_client::SecretLookupRequest> {
    info.required_secrets
        .iter()
        .map(|secret| vgi_client::SecretLookupRequest {
            secret_type: secret.secret_type.clone(),
            scope: secret.scope.clone(),
            name: secret.secret_name.clone(),
        })
        .collect()
}

fn register_global_functions(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
    provider: &VgiCatalogProvider,
) -> DFResult<()> {
    let prefix = provider.global_function_prefix();
    for info in provider.global_functions() {
        let name = if prefix.is_empty() {
            info.name.to_ascii_lowercase()
        } else {
            format!("{prefix}_{}", info.name).to_ascii_lowercase()
        };
        let kind = info.function_type.0.to_ascii_lowercase();
        let state = ctx.state();
        let collision = match kind.as_str() {
            "scalar" => state.scalar_functions().contains_key(&name),
            "aggregate" => state.aggregate_functions().contains_key(&name),
            "table" | "table_buffering" => state.table_functions().contains_key(&name),
            _ => {
                return plan_err!(
                    "worker nominated global function `{}` with unsupported type `{}`",
                    info.name,
                    info.function_type.0
                )
            }
        };
        if collision {
            return plan_err!("global VGI function `{name}` collides with an existing function");
        }
        match kind.as_str() {
            "scalar" => {
                let specs = vgi_client::ArgSpecs::parse(&info.arguments.0).map_err(crate::to_df)?;
                ctx.register_udf(
                    AsyncScalarUDF::new(Arc::new(VgiScalarUdf::discovered_with_volatility(
                        conn.clone(),
                        &spec.catalog,
                        &info.schema_name,
                        &info.name,
                        &name,
                        specs,
                        metadata_volatility(info),
                    )))
                    .into_scalar_udf(),
                );
                record_registration(ctx, &spec.alias, RegistrationKind::Scalar, name);
            }
            "aggregate" => {
                let specs = vgi_client::ArgSpecs::parse(&info.arguments.0).map_err(crate::to_df)?;
                ctx.register_udaf(AggregateUDF::new_from_impl(
                    VgiAggregateUdf::new_with_volatility(
                        conn.clone(),
                        &spec.catalog,
                        &info.schema_name,
                        &info.name,
                        &name,
                        metadata_volatility(info),
                    )
                    .with_arg_specs(specs)
                    .with_required_secrets(metadata_secrets(info)),
                ));
                record_registration(ctx, &spec.alias, RegistrationKind::Aggregate, name);
            }
            "table" | "table_buffering" => {
                let metadata = crate::catalog::TableFunctionMetadata {
                    specs: vgi_client::ArgSpecs::parse(&info.arguments.0).map_err(crate::to_df)?,
                    buffered: kind == "table_buffering",
                    input_from_args: info.input_from_args,
                };
                ctx.register_udtf(
                    &name,
                    Arc::new(VgiTableFunction::new(
                        conn.clone(),
                        &spec.catalog,
                        &info.schema_name,
                        &info.name,
                        Some(metadata),
                    )),
                );
                record_registration(ctx, &spec.alias, RegistrationKind::Table, name);
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

const IMPLEMENTED_LOCAL_OPTIONS: &[&str] = &[
    "cache",
    "pool",
    "pool_max",
    "pool_timeout",
    "worker_debug",
    "launcher_idle_timeout",
    "launcher_state_dir",
    "data_version_spec",
    "implementation_version",
    "bearer_token",
    "oauth_refresh_token",
    "attach_companions",
];

const UNAVAILABLE_LOCAL_OPTIONS: &[&str] = &["secrets", "attach_companion_secrets"];

async fn build_attach_options(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
) -> DFResult<vgi_client::AttachOptions> {
    for name in UNAVAILABLE_LOCAL_OPTIONS {
        if spec.options.contains_key(*name) {
            return plan_err!("ATTACH option `{name}` is not supported by vgi-datafusion yet");
        }
    }

    let string = |name: &str| -> DFResult<Option<String>> {
        spec.options
            .get(name)
            .map(|raw| option_string(raw))
            .transpose()
    };
    let mut out = vgi_client::AttachOptions {
        options: None,
        data_version_spec: string("data_version_spec")?,
        implementation_version: string("implementation_version")?,
    };
    let worker_values = spec
        .options
        .iter()
        .filter(|(name, _)| !IMPLEMENTED_LOCAL_OPTIONS.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    if worker_values.is_empty() {
        return Ok(out);
    }

    let c = conn.clone();
    let catalog_name = spec.catalog.clone();
    let catalog = tokio::task::spawn_blocking(move || {
        let mut client = c.connect()?;
        client
            .catalogs()
            .map_err(crate::to_df)?
            .into_iter()
            .find(|info| info.name == catalog_name)
            .ok_or_else(|| {
                datafusion::common::DataFusionError::Plan(format!(
                    "worker exposes no catalog named `{catalog_name}`"
                ))
            })
    })
    .await
    .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))??;
    let specs = vgi_client::decode_attach_option_specs(&catalog).map_err(crate::to_df)?;
    let accepted = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    for name in worker_values.keys() {
        if !specs
            .iter()
            .any(|spec| spec.name.eq_ignore_ascii_case(name))
        {
            return plan_err!(
                "unknown ATTACH option `{name}` for catalog `{}`; accepted worker options: {}",
                spec.catalog,
                if accepted.is_empty() {
                    "(none)"
                } else {
                    &accepted
                }
            );
        }
    }
    let missing = specs
        .iter()
        .filter(|decl| {
            decl.required
                && !worker_values
                    .keys()
                    .any(|name| decl.name.eq_ignore_ascii_case(name))
        })
        .map(|decl| format!("`{}`", decl.name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return plan_err!(
            "catalog `{}` requires ATTACH option(s) {}",
            spec.catalog,
            missing.join(", ")
        );
    }

    let mut fields = Vec::with_capacity(worker_values.len());
    let mut arrays = Vec::with_capacity(worker_values.len());
    for (name, raw) in worker_values {
        let declared = specs
            .iter()
            .find(|decl| decl.name.eq_ignore_ascii_case(&name))
            .expect("validated above");
        let array = evaluate_attach_option(ctx, &declared.name, &raw, &declared.data_type).await?;
        fields.push(Field::new(&declared.name, declared.data_type.clone(), true));
        arrays.push(array);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(|error| {
        datafusion::common::DataFusionError::Plan(format!(
            "could not build VGI ATTACH options: {error}"
        ))
    })?;
    out.options = Some(vgi_client::encode_attach_options(&batch).map_err(crate::to_df)?);
    Ok(out)
}

async fn evaluate_attach_option(
    ctx: &SessionContext,
    name: &str,
    raw: &str,
    data_type: &DataType,
) -> DFResult<ArrayRef> {
    validate_constant_option(name, raw)?;
    if matches!(data_type, DataType::Binary | DataType::LargeBinary) {
        if let Some(bytes) = blob_literal(raw)? {
            return Ok(match data_type {
                DataType::Binary => Arc::new(BinaryArray::from(vec![Some(bytes.as_slice())])),
                DataType::LargeBinary => {
                    Arc::new(LargeBinaryArray::from(vec![Some(bytes.as_slice())]))
                }
                _ => unreachable!(),
            });
        }
    }

    let query = format!("SELECT {raw} AS __vgi_attach_value");
    let batches = ctx.sql(&query).await?.collect().await?;
    if batches.len() != 1 || batches[0].num_rows() != 1 || batches[0].num_columns() != 1 {
        return plan_err!("ATTACH option `{name}` must evaluate to exactly one value");
    }
    cast(batches[0].column(0), data_type).map_err(|error| {
        datafusion::common::DataFusionError::Plan(format!(
            "cannot cast ATTACH option `{name}` to {data_type}: {error}"
        ))
    })
}

fn validate_constant_option(name: &str, raw: &str) -> DFResult<()> {
    let mut statements = DFParser::parse_sql(&format!("SELECT {raw}"))?;
    let Some(DFStatement::Statement(statement)) = statements.pop_front() else {
        return plan_err!("ATTACH option `{name}` is not a SQL value expression");
    };
    struct ConstantsOnly;
    impl Visitor for ConstantsOnly {
        type Break = ();

        fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
            match expr {
                Expr::Identifier(_)
                | Expr::CompoundIdentifier(_)
                | Expr::Function(_)
                | Expr::Subquery(_)
                | Expr::Exists { .. }
                | Expr::InSubquery { .. } => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            }
        }
    }
    if let ControlFlow::Break(()) =
        datafusion::sql::sqlparser::ast::Visit::visit(statement.as_ref(), &mut ConstantsOnly)
    {
        return plan_err!(
            "ATTACH option `{name}` must be a constant literal, cast, list, or struct expression"
        );
    }
    Ok(())
}

fn blob_literal(raw: &str) -> DFResult<Option<Vec<u8>>> {
    let lower = raw.trim().to_ascii_lowercase();
    let suffix = if lower.ends_with("::blob") {
        "::blob"
    } else if lower.ends_with("::binary") {
        "::binary"
    } else {
        return Ok(None);
    };
    let value = option_string(raw.trim()[..raw.trim().len() - suffix.len()].trim())?;
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 3 < bytes.len() && bytes[i] == b'\\' && bytes[i + 1] == b'x' {
            let hex = std::str::from_utf8(&bytes[i + 2..i + 4]).map_err(|_| {
                datafusion::common::DataFusionError::Plan("invalid BLOB escape".to_string())
            })?;
            out.push(u8::from_str_radix(hex, 16).map_err(|_| {
                datafusion::common::DataFusionError::Plan(format!("invalid BLOB escape `\\x{hex}`"))
            })?);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Ok(Some(out))
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
    for (schema_name, schema) in provider.vgi_schemas() {
        for function in schema.table_function_names() {
            let metadata = schema.table_function_metadata(function).cloned();
            let make = || {
                Arc::new(VgiTableFunction::new(
                    conn.clone(),
                    &spec.catalog,
                    schema_name,
                    function.as_str(),
                    metadata.clone(),
                ))
            };

            for name in publish_names(&spec.alias, schema_name, function) {
                // First-wins throughout: the two shorter forms cannot carry a
                // schema, so a name published in two schemas collides on them,
                // and the fully qualified form is always there as the
                // unambiguous way to say which one you meant.
                if !ctx.state().table_functions().contains_key(&name) {
                    ctx.register_udtf(&name, make());
                    record_registration(ctx, &spec.alias, RegistrationKind::Table, name);
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
    for (schema_name, schema) in provider.vgi_schemas() {
        for (function, overloads, volatility) in schema.scalars() {
            let register = |name: String| {
                if ctx.state().scalar_functions().contains_key(&name) {
                    return;
                }
                let udf = VgiScalarUdf::discovered_overloads_with_volatility(
                    conn.clone(),
                    &spec.catalog,
                    schema_name,
                    function,
                    &name,
                    overloads.clone(),
                    *volatility,
                );
                ctx.register_udf(AsyncScalarUDF::new(Arc::new(udf)).into_scalar_udf());
                record_registration(ctx, &spec.alias, RegistrationKind::Scalar, name);
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
    for (schema_name, schema) in provider.vgi_schemas() {
        for (function, specs, volatility, _supports_window, required_secrets) in schema.aggregates()
        {
            for name in publish_names(&spec.alias, schema_name, function) {
                if ctx.state().aggregate_functions().contains_key(&name) {
                    continue;
                }
                ctx.register_udaf(AggregateUDF::new_from_impl(
                    VgiAggregateUdf::new_with_volatility(
                        conn.clone(),
                        &spec.catalog,
                        schema_name,
                        function,
                        &name,
                        *volatility,
                    )
                    .with_arg_specs(specs.clone())
                    .with_required_secrets(required_secrets.clone()),
                ));
                record_registration(ctx, &spec.alias, RegistrationKind::Aggregate, name);
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
    deregister_alias_functions(ctx, alias);
    session_runtime(ctx).remove_catalog_metadata(alias);
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
        assert_eq!(s.options.get("pool").map(String::as_str), Some("'false'"));
        assert_eq!(s.options.get("cache").map(String::as_str), Some("'true'"));
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

    #[test]
    fn launcher_options_are_scoped_and_validated() {
        let accepted = spec(
            "example?location=launch:worker&launcher_idle_timeout=0&launcher_state_dir=/tmp/vgi-state&worker_debug=true",
        );
        accepted.connection().expect("valid launcher options");

        let err = spec("example?location=worker&launcher_idle_timeout=60")
            .connection()
            .unwrap_err()
            .to_string();
        assert!(err.contains("only valid for `launch:`"), "{err}");

        let err = spec("example?location=launch:worker&launcher_state_dir=")
            .connection()
            .unwrap_err()
            .to_string();
        assert!(err.contains("must not be empty"), "{err}");

        let err = spec("example?location=launch:worker&launcher_idle_timeout=-1")
            .connection()
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-negative integer"), "{err}");
    }

    #[test]
    fn duckdb_attach_defaults_alias_and_keeps_nested_values() {
        let parsed = parse_duckdb_attach(
            "ATTACH 'example' (TYPE vgi, LOCATION 'worker', opt_list [1, 2], opt_struct {'a': 3, 'b': 'x,y'})",
        )
        .expect("recognized")
        .expect("parsed");
        assert_eq!(parsed.alias, "example");
        assert_eq!(parsed.location, "worker");
        assert_eq!(parsed.options.get("opt_list").unwrap(), "[1, 2]");
        assert_eq!(
            parsed.options.get("opt_struct").unwrap(),
            "{'a': 3, 'b': 'x,y'}"
        );
    }

    #[test]
    fn attach_debug_redacts_values() {
        let parsed = parse_duckdb_attach(
            "ATTACH 'example' (TYPE vgi, LOCATION 'https://example', bearer_token 'sentinel-secret')",
        )
        .unwrap()
        .unwrap();
        let debug = format!("{parsed:?}");
        assert!(!debug.contains("sentinel-secret"), "{debug}");
        assert!(debug.contains("bearer_token"), "{debug}");
    }
}
