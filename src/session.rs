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

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::catalog::{CatalogProvider, MemoryCatalogProvider};
use datafusion::common::{plan_datafusion_err, plan_err, Result as DFResult};
use datafusion::dataframe::DataFrame;
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::Statement as DFStatement;
use datafusion::sql::sqlparser::ast::{Expr, Statement as SQLStatement, Value, ValueWithSpan};

use crate::{VgiCatalogProvider, VgiConnection, VgiTableFunction};

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
    let state = ctx.state();
    let dialect = state.config_options().sql_parser.dialect.clone();
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
            let plan = state.statement_to_plan(statement).await?;
            ctx.execute_logical_plan(plan).await
        }
    }
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
    ctx.register_catalog(&spec.alias, provider);
    Ok(())
}

/// Publish the catalog's table functions so they are callable **with
/// arguments**: `SELECT * FROM ex_sequence(10)`.
///
/// # Why they need a second surface at all
///
/// A [`CatalogProvider`](datafusion::catalog::CatalogProvider) yields tables,
/// and a table has no arguments. Most VGI table functions take some, and their
/// output schema depends on them, so they cannot be reached that way — only the
/// zero-argument ones can. DataFusion's answer is a separate registry, so a
/// function that takes arguments is published here as well.
///
/// # Why the name is prefixed
///
/// That registry is **flat and global** — `register_udtf(name, …)`, with no
/// catalog or schema qualification, and DataFusion resolves
/// `SELECT * FROM a.b.f(1)` as a table reference rather than a call. So the
/// worker's own coordinates cannot be spelled, and bare names would collide:
/// the reference fixture worker alone publishes `test_same_name_cached` in two
/// different schemas.
///
/// Names are therefore `<alias>_<function>`, which is exactly what the DuckDB
/// extension does when a worker asks for global functions — see
/// `VgiGlobalFunctionName` and the `global_function_prefix`. Registration is
/// **first-wins** for the same reason it is there: it is advisory, and
/// clobbering a name someone else registered would be worse than not
/// publishing.
///
/// A function is published whether or not it takes arguments. A zero-argument
/// one is then reachable both ways — `ex.main.f` and `ex_f()` — which costs
/// nothing and saves the caller having to know which kind it is.
fn register_table_functions(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
    provider: &VgiCatalogProvider,
) {
    let state = ctx.state();
    for schema_name in provider.schema_names() {
        let Some(schema) = provider.schema(&schema_name) else {
            continue;
        };
        for function in schema.table_names() {
            let visible = format!("{}_{}", spec.alias, function);
            if state.table_functions().contains_key(&visible) {
                continue; // first attach wins
            }
            ctx.register_udtf(
                &visible,
                Arc::new(VgiTableFunction::new(
                    conn.clone(),
                    &spec.catalog,
                    &schema_name,
                    &function,
                )),
            );
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
