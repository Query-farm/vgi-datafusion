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
use datafusion::catalog::{MemoryCatalogProvider, TableFunctionImpl};
use datafusion::common::{plan_datafusion_err, plan_err, Result as DFResult};
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::async_udf::AsyncScalarUDF;
use datafusion::logical_expr::registry::FunctionRegistry;
use datafusion::logical_expr::{AggregateUDF, ScalarUDF};
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::{DFParser, ResetStatement, Statement as DFStatement};
use datafusion::sql::sqlparser::ast::{
    BinaryOperator, Expr, FunctionArg, FunctionArgExpr, FunctionArguments, ObjectName,
    ObjectNamePart, Query, SelectItem, Set as SQLSet, SetExpr, Statement as SQLStatement,
    TableFactor, Value, ValueWithSpan, VisitMut, Visitor, VisitorMut,
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
    kind: SqlMacroKind,
    parameters: Vec<String>,
    defaults: HashMap<String, ArrayRef>,
    body: SqlMacroBody,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SqlMacroKind {
    Scalar,
    Table,
}

fn sql_macro_kind(value: &str) -> Result<SqlMacroKind, String> {
    match value.to_ascii_lowercase().as_str() {
        "scalar" | "scalar_macro" | "macro" => Ok(SqlMacroKind::Scalar),
        "table" | "table_macro" => Ok(SqlMacroKind::Table),
        other => Err(format!("unsupported macro type {other:?}")),
    }
}

#[derive(Debug, Clone)]
enum SqlMacroBody {
    Scalar(Expr),
    Table(Box<Query>),
    Invalid(String),
}

/// Names from one worker schema that an unqualified SQL macro reference may
/// resolve to. Keeping the sets separate prevents a table name from changing a
/// scalar call (or a scalar name from changing a CTE/table reference).
#[derive(Debug, Default)]
struct SqlMacroNamespace {
    expression_functions: HashSet<String>,
    relation_functions: HashSet<String>,
    relations: HashSet<String>,
}

/// Every object namespace advertised by one attached VGI catalog, keyed by
/// worker schema. Catalog-owned SQL may use either `object` (owning schema) or
/// `schema.object` (cross-schema); both must resolve independently of the
/// caller's current DataFusion catalog/schema.
#[derive(Debug, Default)]
struct SqlCatalogNamespace {
    schemas: HashMap<String, SqlMacroNamespace>,
}

type SessionMacros = HashMap<String, HashMap<String, Arc<SqlMacro>>>;

fn macro_registry() -> &'static Mutex<SessionMacros> {
    static REGISTRY: OnceLock<Mutex<SessionMacros>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn parse_macro_body(kind: SqlMacroKind, definition: &str) -> SqlMacroBody {
    let parsed = if kind == SqlMacroKind::Scalar {
        DFParser::parse_sql(&format!("SELECT {definition}"))
    } else {
        DFParser::parse_sql(definition)
    };
    let mut statements = match parsed {
        Ok(statements) => statements,
        Err(error) => return SqlMacroBody::Invalid(error.to_string()),
    };
    if statements.len() != 1 {
        return SqlMacroBody::Invalid("definition must contain exactly one statement".to_string());
    }
    let Some(DFStatement::Statement(statement)) = statements.pop_front() else {
        return SqlMacroBody::Invalid("definition is not a SQL query".to_string());
    };
    let SQLStatement::Query(query) = statement.as_ref() else {
        return SqlMacroBody::Invalid("definition is not a SQL query".to_string());
    };
    if kind == SqlMacroKind::Table {
        return SqlMacroBody::Table(query.clone());
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return SqlMacroBody::Invalid("scalar definition is not a SELECT expression".to_string());
    };
    let [SelectItem::UnnamedExpr(expression)] = select.projection.as_slice() else {
        return SqlMacroBody::Invalid(
            "scalar definition must contain exactly one unnamed expression".to_string(),
        );
    };
    // The temporary SELECT is only a parser wrapper. Reject any trailing query
    // clause instead of silently throwing away part of the worker definition.
    // Comparing sqlparser's normalized render keeps this exhaustive as new
    // SELECT clauses are added upstream.
    if query.to_string() != format!("SELECT {expression}") {
        return SqlMacroBody::Invalid(
            "scalar definition must contain only one expression".to_string(),
        );
    }
    SqlMacroBody::Scalar(expression.clone())
}

struct CatalogSqlQualifier<'a> {
    catalog: &'a str,
    owning_schema: &'a str,
    namespace: &'a SqlCatalogNamespace,
    cte_scopes: Vec<HashSet<String>>,
}

impl CatalogSqlQualifier<'_> {
    /// Resolve a one-part name in the owning schema or a two-part name in its
    /// explicit worker schema. Three-part names are already catalog-qualified.
    fn object_key(&self, name: &ObjectName) -> Option<(String, String)> {
        let identifiers = name
            .0
            .iter()
            .map(ObjectNamePart::as_ident)
            .collect::<Option<Vec<_>>>()?;
        match identifiers.as_slice() {
            [object] => Some((
                self.owning_schema.to_ascii_lowercase(),
                object.value.to_ascii_lowercase(),
            )),
            [schema, object] => Some((
                schema.value.to_ascii_lowercase(),
                object.value.to_ascii_lowercase(),
            )),
            _ => None,
        }
    }

    fn qualify_name(&self, name: &mut ObjectName) {
        let catalog =
            ObjectNamePart::Identifier(datafusion::sql::sqlparser::ast::Ident::new(self.catalog));
        match name.0.len() {
            1 => {
                let Some(local) = name.0.pop() else {
                    return;
                };
                name.0 = vec![
                    catalog,
                    ObjectNamePart::Identifier(datafusion::sql::sqlparser::ast::Ident::new(
                        self.owning_schema,
                    )),
                    local,
                ];
            }
            2 => name.0.insert(0, catalog),
            _ => {}
        }
    }

    fn cte_in_scope(&self, name: &str) -> bool {
        self.cte_scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(&name.to_ascii_lowercase()))
    }
}

impl VisitorMut for CatalogSqlQualifier<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<Self::Break> {
        self.cte_scopes.push(HashSet::new());
        let Some(with) = &mut query.with else {
            return ControlFlow::Continue(());
        };
        if with.recursive {
            self.cte_scopes
                .last_mut()
                .expect("query scope was just pushed")
                .extend(
                    with.cte_tables
                        .iter()
                        .map(|cte| cte.alias.name.value.to_ascii_lowercase()),
                );
            return ControlFlow::Continue(());
        }

        // sqlparser's generic visitor enters every CTE only after this
        // callback. Walk non-recursive definitions once here, in order, while
        // exposing only preceding aliases. The later generic traversal is
        // harmless because qualification is idempotent.
        for cte in &mut with.cte_tables {
            if let ControlFlow::Break(()) = cte.query.visit(self) {
                return ControlFlow::Break(());
            }
            self.cte_scopes
                .last_mut()
                .expect("query scope remains active")
                .insert(cte.alias.name.value.to_ascii_lowercase());
        }
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<Self::Break> {
        self.cte_scopes.pop();
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<Self::Break> {
        let TableFactor::Table { name, args, .. } = factor else {
            return ControlFlow::Continue(());
        };
        let Some((schema, object)) = self.object_key(name) else {
            return ControlFlow::Continue(());
        };
        let Some(namespace) = self.namespace.schemas.get(&schema) else {
            return ControlFlow::Continue(());
        };
        let worker_object = if args.is_some() {
            namespace.relation_functions.contains(&object)
        } else {
            (name.0.len() != 1 || !self.cte_in_scope(&object))
                && namespace.relations.contains(&object)
        };
        if worker_object {
            self.qualify_name(name);
        }
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<Self::Break> {
        let Expr::Function(function) = expr else {
            return ControlFlow::Continue(());
        };
        let Some((schema, object)) = self.object_key(&function.name) else {
            return ControlFlow::Continue(());
        };
        if self
            .namespace
            .schemas
            .get(&schema)
            .is_some_and(|namespace| namespace.expression_functions.contains(&object))
        {
            self.qualify_name(&mut function.name);
        }
        ControlFlow::Continue(())
    }
}

fn qualify_catalog_sql<T: VisitMut>(
    sql: &mut T,
    catalog: &str,
    owning_schema: &str,
    namespace: &SqlCatalogNamespace,
) {
    let _ = sql.visit(&mut CatalogSqlQualifier {
        catalog,
        owning_schema,
        namespace,
        cte_scopes: Vec::new(),
    });
}

fn qualify_macro_body(
    body: &mut SqlMacroBody,
    catalog: &str,
    schema: &str,
    namespace: &SqlCatalogNamespace,
) {
    match body {
        SqlMacroBody::Scalar(expression) => {
            qualify_catalog_sql(expression, catalog, schema, namespace);
        }
        SqlMacroBody::Table(query) => {
            qualify_catalog_sql(query.as_mut(), catalog, schema, namespace);
        }
        SqlMacroBody::Invalid(_) => {}
    }
}

fn macro_key(name: &ObjectName) -> Option<String> {
    let path = name
        .0
        .iter()
        .map(|part| part.as_ident().map(|ident| ident.value.as_str()))
        .collect::<Option<Vec<_>>>()?;
    if !matches!(path.len(), 2 | 3) {
        return None;
    }
    Some(path.join(".").to_ascii_lowercase())
}

fn lookup_macro(ctx: &SessionContext, name: &ObjectName) -> Option<Arc<SqlMacro>> {
    let key = macro_key(name)?;
    macro_registry()
        .lock()
        .ok()
        .and_then(|sessions| sessions.get(&ctx.session_id())?.get(&key).cloned())
}

fn macro_default_expr(value: &ArrayRef) -> DFResult<Expr> {
    let scalar = datafusion::common::ScalarValue::try_from_array(value.as_ref(), 0)?;
    datafusion::sql::unparser::expr_to_sql(&datafusion::logical_expr::Expr::Literal(scalar, None))
}

fn bind_macro_arguments(
    name: &ObjectName,
    info: &SqlMacro,
    arguments: &[FunctionArg],
) -> DFResult<Vec<Expr>> {
    let mut actual = vec![None; info.parameters.len()];
    let mut positional = 0usize;
    let mut saw_named = false;

    for argument in arguments {
        let (named, expression) = match argument {
            FunctionArg::Named {
                name: argument_name,
                arg,
                ..
            } => {
                let FunctionArgExpr::Expr(expression) = arg else {
                    return plan_err!("VGI macro `{name}` does not accept wildcard arguments");
                };
                (Some(argument_name.value.as_str()), expression.clone())
            }
            FunctionArg::ExprNamed {
                name: Expr::Identifier(argument_name),
                arg,
                ..
            } => {
                let FunctionArgExpr::Expr(expression) = arg else {
                    return plan_err!("VGI macro `{name}` does not accept wildcard arguments");
                };
                (Some(argument_name.value.as_str()), expression.clone())
            }
            FunctionArg::ExprNamed { .. } => {
                return plan_err!("VGI macro `{name}` requires identifier argument names");
            }
            FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            })) => match left.as_ref() {
                Expr::Identifier(argument_name)
                    if info
                        .parameters
                        .iter()
                        .any(|parameter| parameter.eq_ignore_ascii_case(&argument_name.value)) =>
                {
                    (Some(argument_name.value.as_str()), right.as_ref().clone())
                }
                _ => (
                    None,
                    Expr::BinaryOp {
                        left: left.clone(),
                        op: BinaryOperator::Eq,
                        right: right.clone(),
                    },
                ),
            },
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expression)) => (None, expression.clone()),
            FunctionArg::Unnamed(_) => {
                return plan_err!("VGI macro `{name}` does not accept wildcard arguments");
            }
        };

        if let Some(argument_name) = named {
            saw_named = true;
            let Some(index) = info
                .parameters
                .iter()
                .position(|parameter| parameter.eq_ignore_ascii_case(argument_name))
            else {
                return plan_err!("VGI macro `{name}` has no parameter named `{argument_name}`");
            };
            if actual[index].replace(expression).is_some() {
                return plan_err!(
                    "VGI macro `{name}` received parameter `{}` more than once",
                    info.parameters[index]
                );
            }
        } else {
            if saw_named {
                return plan_err!(
                    "VGI macro `{name}` does not accept positional arguments after named arguments"
                );
            }
            if positional >= actual.len() {
                return plan_err!(
                    "VGI macro `{name}` expects at most {} argument(s), received more",
                    actual.len()
                );
            }
            actual[positional] = Some(expression);
            positional += 1;
        }
    }

    let mut missing = Vec::new();
    for (index, parameter) in info.parameters.iter().enumerate() {
        if actual[index].is_none() {
            if let Some(default) = info.defaults.get(&parameter.to_ascii_lowercase()) {
                actual[index] = Some(macro_default_expr(default)?);
            } else {
                missing.push(parameter.clone());
            }
        }
    }
    if !missing.is_empty() {
        return plan_err!(
            "VGI macro `{name}` is missing required argument(s): {}",
            missing.join(", ")
        );
    }
    Ok(actual.into_iter().map(Option::unwrap).collect())
}

fn substitute_macro<T: VisitMut>(template: &mut T, parameters: &[String], actual: &[Expr]) {
    struct Substitute<'a> {
        parameters: &'a [String],
        actual: &'a [Expr],
    }
    impl VisitorMut for Substitute<'_> {
        type Break = ();

        fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<()> {
            let Expr::Identifier(identifier) = expr else {
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
    let _ = template.visit(&mut Substitute { parameters, actual });
}

fn expand_scalar_macro(
    ctx: &SessionContext,
    expr: &Expr,
    active: &mut Vec<String>,
) -> DFResult<Option<Expr>> {
    let Expr::Function(function) = expr else {
        return Ok(None);
    };
    let Some(info) = lookup_macro(ctx, &function.name) else {
        return Ok(None);
    };
    if info.kind != SqlMacroKind::Scalar {
        return plan_err!(
            "VGI table macro `{}` cannot be used as a scalar",
            function.name
        );
    }
    let SqlMacroBody::Scalar(template) = &info.body else {
        return match &info.body {
            SqlMacroBody::Invalid(error) => {
                plan_err!("invalid VGI macro `{}`: {error}", function.name)
            }
            _ => plan_err!("VGI macro `{}` has an invalid scalar body", function.name),
        };
    };
    let FunctionArguments::List(arguments) = &function.args else {
        return plan_err!(
            "VGI scalar macro `{}` requires an argument list",
            function.name
        );
    };
    if !matches!(function.parameters, FunctionArguments::None)
        || arguments.duplicate_treatment.is_some()
        || !arguments.clauses.is_empty()
        || function.filter.is_some()
        || function.null_treatment.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
    {
        return plan_err!(
            "VGI scalar macro `{}` does not accept DISTINCT, FILTER, OVER, or argument clauses",
            function.name
        );
    }
    let actual = bind_macro_arguments(&function.name, &info, &arguments.args)?;
    let key = macro_key(&function.name).expect("a registered macro has a qualified key");
    if let Some(cycle_start) = active.iter().position(|candidate| candidate == &key) {
        let mut cycle = active[cycle_start..].to_vec();
        cycle.push(key);
        return plan_err!(
            "recursive VGI scalar macro expansion: {}",
            cycle.join(" -> ")
        );
    }
    active.push(key);
    let mut expanded = template.clone();
    substitute_macro(&mut expanded, &info.parameters, &actual);

    // sqlparser does not revisit an expression installed by a post-visit hook,
    // so explicitly expand macro calls declared inside this macro's body.
    struct ExpandNested<'a> {
        ctx: &'a SessionContext,
        active: &'a mut Vec<String>,
    }
    impl VisitorMut for ExpandNested<'_> {
        type Break = Box<datafusion::common::DataFusionError>;

        fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<Self::Break> {
            match expand_scalar_macro(self.ctx, expr, self.active) {
                Ok(Some(replacement)) => *expr = replacement,
                Ok(None) => {}
                Err(error) => return ControlFlow::Break(Box::new(error)),
            }
            ControlFlow::Continue(())
        }
    }
    let nested = expanded.visit(&mut ExpandNested { ctx, active });
    active.pop();
    if let ControlFlow::Break(error) = nested {
        return Err(*error);
    }
    Ok(Some(expanded))
}

fn expand_table_macro(ctx: &SessionContext, factor: &TableFactor) -> DFResult<Option<TableFactor>> {
    let TableFactor::Table {
        name,
        alias,
        args: Some(arguments),
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
    } = factor
    else {
        return Ok(None);
    };
    let Some(info) = lookup_macro(ctx, name) else {
        return Ok(None);
    };
    if info.kind != SqlMacroKind::Table {
        return plan_err!("VGI scalar macro `{name}` cannot be used as a relation");
    }
    if arguments.settings.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || *with_ordinality
        || !partitions.is_empty()
        || json_path.is_some()
        || !index_hints.is_empty()
    {
        return plan_err!(
            "VGI table macro `{name}` does not accept SETTINGS, hints, versions, ordinality, partitions, JSON paths, or index hints"
        );
    }
    let SqlMacroBody::Table(template) = &info.body else {
        return match &info.body {
            SqlMacroBody::Invalid(error) => plan_err!("invalid VGI macro `{name}`: {error}"),
            _ => plan_err!("VGI macro `{name}` has an invalid table body"),
        };
    };
    let actual = bind_macro_arguments(name, &info, &arguments.args)?;
    let mut subquery = template.clone();
    substitute_macro(subquery.as_mut(), &info.parameters, &actual);
    Ok(Some(TableFactor::Derived {
        lateral: false,
        subquery,
        alias: alias.clone(),
        sample: sample.clone(),
    }))
}

fn runtime_registry() -> &'static Mutex<HashMap<String, Weak<VgiRuntime>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Weak<VgiRuntime>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

type SessionLifecycleLocks = HashMap<String, Weak<tokio::sync::Mutex<()>>>;

fn lifecycle_lock_registry() -> &'static Mutex<SessionLifecycleLocks> {
    static REGISTRY: OnceLock<Mutex<SessionLifecycleLocks>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn session_lifecycle_lock(ctx: &SessionContext) -> Arc<tokio::sync::Mutex<()>> {
    let session_id = ctx.session_id();
    let mut locks = lifecycle_lock_registry().lock().unwrap();
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(&session_id).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    locks.insert(session_id, Arc::downgrade(&lock));
    lock
}

#[derive(Default)]
struct RegisteredNames {
    scalar: HashSet<String>,
    scalar_owners: HashMap<String, Weak<ScalarUDF>>,
    /// Published aggregate name -> whether its minimum positional arity is
    /// zero. Such invocations need a private row-witness argument because
    /// DataFusion's empty accumulator input does not carry a row count.
    aggregate: HashMap<String, bool>,
    aggregate_owners: HashMap<String, Weak<AggregateUDF>>,
    /// Published table-function name -> worker-declared named arguments.
    ///
    /// Besides DETACH bookkeeping, this lets the SQL compatibility pass
    /// distinguish DuckDB's `name=value` named-argument spelling from an
    /// ordinary equality expression without guessing from syntax alone.
    table: HashMap<String, HashSet<String>>,
    table_owners: HashMap<String, Weak<dyn TableFunctionImpl>>,
}

type SessionRegistrations = HashMap<String, HashMap<String, RegisteredNames>>;

fn registration_registry() -> &'static Mutex<SessionRegistrations> {
    static REGISTRY: OnceLock<Mutex<SessionRegistrations>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

type SessionDefaultSchemas = HashMap<String, HashMap<String, String>>;

fn default_schema_registry() -> &'static Mutex<SessionDefaultSchemas> {
    static REGISTRY: OnceLock<Mutex<SessionDefaultSchemas>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_default_schema(ctx: &SessionContext, alias: &str, schema: &str) {
    default_schema_registry()
        .lock()
        .unwrap()
        .entry(ctx.session_id())
        .or_default()
        .insert(alias.to_ascii_lowercase(), schema.to_string());
}

fn attached_default_schema(ctx: &SessionContext, alias: &str) -> Option<String> {
    default_schema_registry()
        .lock()
        .ok()?
        .get(&ctx.session_id())?
        .get(&alias.to_ascii_lowercase())
        .cloned()
}

fn remove_default_schema(ctx: &SessionContext, alias: &str) {
    if let Ok(mut sessions) = default_schema_registry().lock() {
        if let Some(aliases) = sessions.get_mut(&ctx.session_id()) {
            aliases.remove(&alias.to_ascii_lowercase());
        }
    }
}

enum RegistrationKind {
    Scalar(Weak<ScalarUDF>),
    Aggregate {
        nullary: bool,
        owner: Weak<AggregateUDF>,
    },
    Table {
        named_arguments: HashSet<String>,
        owner: Weak<dyn TableFunctionImpl>,
    },
}

fn record_registration(ctx: &SessionContext, alias: &str, kind: RegistrationKind, name: String) {
    let mut sessions = registration_registry().lock().unwrap();
    let registered = sessions
        .entry(ctx.session_id())
        .or_default()
        .entry(alias.to_ascii_lowercase())
        .or_default();
    match kind {
        RegistrationKind::Scalar(owner) => {
            registered.scalar.insert(name.clone());
            registered.scalar_owners.insert(name, owner).is_none()
        }
        RegistrationKind::Aggregate { nullary, owner } => {
            registered.aggregate.insert(name.clone(), nullary);
            registered.aggregate_owners.insert(name, owner).is_none()
        }
        RegistrationKind::Table {
            named_arguments,
            owner,
        } => {
            registered.table.insert(name.clone(), named_arguments);
            registered.table_owners.insert(name, owner).is_none()
        }
    };
}

fn register_scalar_if_absent(
    ctx: &SessionContext,
    alias: &str,
    name: String,
    udf: ScalarUDF,
) -> bool {
    let owner = Arc::new(udf);
    let state = ctx.state_ref();
    let mut state = state.write();
    if state.scalar_functions().contains_key(&name) {
        return false;
    }
    state
        .register_udf(Arc::clone(&owner))
        .expect("live SessionState accepts scalar UDF registration");
    record_registration(
        ctx,
        alias,
        RegistrationKind::Scalar(Arc::downgrade(&owner)),
        name,
    );
    true
}

fn register_aggregate_if_absent(
    ctx: &SessionContext,
    alias: &str,
    name: String,
    udaf: AggregateUDF,
    nullary: bool,
) -> bool {
    let owner = Arc::new(udaf);
    let state = ctx.state_ref();
    let mut state = state.write();
    if state.aggregate_functions().contains_key(&name) {
        return false;
    }
    state
        .register_udaf(Arc::clone(&owner))
        .expect("live SessionState accepts aggregate UDF registration");
    record_registration(
        ctx,
        alias,
        RegistrationKind::Aggregate {
            nullary,
            owner: Arc::downgrade(&owner),
        },
        name,
    );
    true
}

fn register_table_if_absent(
    ctx: &SessionContext,
    alias: &str,
    name: String,
    function: Arc<dyn TableFunctionImpl>,
    named_arguments: HashSet<String>,
) -> bool {
    let state = ctx.state_ref();
    let mut state = state.write();
    if state.table_functions().contains_key(&name) {
        return false;
    }
    state.register_udtf(&name, Arc::clone(&function));
    record_registration(
        ctx,
        alias,
        RegistrationKind::Table {
            named_arguments,
            owner: Arc::downgrade(&function),
        },
        name,
    );
    true
}

fn named_arguments(specs: &vgi_client::ArgSpecs) -> HashSet<String> {
    specs
        .0
        .iter()
        .filter(|spec| spec.is_named)
        .map(|spec| spec.name.to_ascii_lowercase())
        .collect()
}

fn is_declared_named_table_argument(
    ctx: &SessionContext,
    function_name: &str,
    argument_name: &str,
) -> bool {
    registration_registry()
        .lock()
        .ok()
        .and_then(|sessions| {
            sessions.get(&ctx.session_id()).map(|aliases| {
                aliases.values().any(|registered| {
                    registered.table.iter().any(|(name, arguments)| {
                        name.eq_ignore_ascii_case(function_name)
                            && arguments.contains(&argument_name.to_ascii_lowercase())
                    })
                })
            })
        })
        .unwrap_or(false)
}

fn is_declared_nullary_aggregate(ctx: &SessionContext, function_name: &str) -> bool {
    registration_registry()
        .lock()
        .ok()
        .and_then(|sessions| {
            sessions.get(&ctx.session_id()).map(|aliases| {
                aliases.values().any(|registered| {
                    registered
                        .aggregate
                        .iter()
                        .any(|(name, nullary)| *nullary && name.eq_ignore_ascii_case(function_name))
                })
            })
        })
        .unwrap_or(false)
}

fn deregister_alias_functions(ctx: &SessionContext, alias: &str) {
    let state = ctx.state_ref();
    let mut state = state.write();
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
        let owned = registered
            .scalar_owners
            .get(&name)
            .and_then(Weak::upgrade)
            .is_some_and(|owner| {
                state
                    .scalar_functions()
                    .get(&name)
                    .is_some_and(|current| Arc::ptr_eq(current, &owner))
            });
        if owned {
            state
                .deregister_udf(&name)
                .expect("live SessionState accepts scalar UDF deregistration");
        }
    }
    for name in registered.aggregate.into_keys() {
        let owned = registered
            .aggregate_owners
            .get(&name)
            .and_then(Weak::upgrade)
            .is_some_and(|owner| {
                state
                    .aggregate_functions()
                    .get(&name)
                    .is_some_and(|current| Arc::ptr_eq(current, &owner))
            });
        if owned {
            state
                .deregister_udaf(&name)
                .expect("live SessionState accepts aggregate UDF deregistration");
        }
    }
    for name in registered.table.into_keys() {
        let owned = registered
            .table_owners
            .get(&name)
            .and_then(Weak::upgrade)
            .is_some_and(|owner| {
                state
                    .table_functions()
                    .get(&name)
                    .is_some_and(|current| Arc::ptr_eq(current.function(), &owner))
            });
        if owned {
            state
                .deregister_udtf(&name)
                .expect("live SessionState accepts table UDF deregistration");
        }
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

fn ensure_vgi_settings(ctx: &SessionContext) {
    let state = ctx.state_ref();
    let mut state = state.write();
    if state
        .config()
        .options()
        .extensions
        .get::<crate::VgiSettings>()
        .is_none()
    {
        state
            .config_mut()
            .options_mut()
            .extensions
            .insert(crate::VgiSettings::default());
    }
}

fn sync_vgi_settings(ctx: &SessionContext, runtime: &VgiRuntime) {
    if let Some(settings) = ctx
        .copied_config()
        .options()
        .extensions
        .get::<crate::VgiSettings>()
    {
        runtime.replace_session_settings(settings.clone());
    }
}

fn declared_vgi_setting(runtime: &VgiRuntime, name: &str) -> bool {
    runtime.catalog_metadata().iter().any(|(_, metadata)| {
        metadata
            .settings
            .iter()
            .any(|setting| setting.name.eq_ignore_ascii_case(name))
    })
}

fn setting_object_name(name: &ObjectName) -> Option<(bool, String)> {
    match name.0.as_slice() {
        [part] => part
            .as_ident()
            .map(|ident| (false, ident.value.to_ascii_lowercase())),
        [prefix, part]
            if prefix
                .as_ident()
                .is_some_and(|ident| ident.value.eq_ignore_ascii_case("vgi")) =>
        {
            part.as_ident()
                .map(|ident| (true, ident.value.to_ascii_lowercase()))
        }
        _ => None,
    }
}

/// Rewrite DuckDB's unqualified worker setting spelling onto DataFusion's
/// required third-party configuration namespace.
fn qualify_vgi_set(statement: &mut DFStatement, runtime: &VgiRuntime) {
    let DFStatement::Statement(statement) = statement else {
        return;
    };
    let SQLStatement::Set(SQLSet::SingleAssignment { variable, .. }) = statement.as_mut() else {
        return;
    };
    let Some((prefixed, name)) = setting_object_name(variable) else {
        return;
    };
    if !prefixed && declared_vgi_setting(runtime, &name) {
        variable.0.insert(
            0,
            ObjectNamePart::Identifier(datafusion::sql::sqlparser::ast::Ident::new("vgi")),
        );
    }
}

fn reset_vgi_setting(
    ctx: &SessionContext,
    runtime: &VgiRuntime,
    statement: &DFStatement,
) -> DFResult<bool> {
    let DFStatement::Reset(ResetStatement::Variable(variable)) = statement else {
        return Ok(false);
    };
    let Some((prefixed, name)) = setting_object_name(variable) else {
        return Ok(false);
    };
    let configured = ctx
        .copied_config()
        .options()
        .extensions
        .get::<crate::VgiSettings>()
        .is_some_and(|settings| settings.get(&name).is_some());
    if !prefixed && !declared_vgi_setting(runtime, &name) && !configured {
        return Ok(false);
    }
    if prefixed && !declared_vgi_setting(runtime, &name) && !configured {
        return plan_err!("unknown VGI setting `{name}`");
    }
    let state = ctx.state_ref();
    let mut state = state.write();
    let settings = state
        .config_mut()
        .options_mut()
        .extensions
        .get_mut::<crate::VgiSettings>()
        .expect("VGI settings extension was installed");
    settings.reset_value(&name);
    runtime.replace_session_settings(settings.clone());
    Ok(true)
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
    ensure_vgi_settings(ctx);
    let runtime = session_runtime(ctx);
    // Session-scoped diagnostics such as `vgi_catalogs(location)` are useful
    // before the first ATTACH. Registration is idempotent, so install them at
    // the adapter SQL boundary rather than waiting for attach discovery.
    crate::diagnostics::register(ctx, Arc::clone(&runtime));
    sync_vgi_settings(ctx, &runtime);

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
    let mut statement = match state.sql_to_statement(query, &dialect) {
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

    if reset_vgi_setting(ctx, &runtime, &statement)? {
        return ctx.read_empty();
    }
    qualify_vgi_set(&mut statement, &runtime);

    match classify(&statement)? {
        Some(Intercepted::Attach { target, alias }) => {
            let spec = AttachSpec::parse(&target, &alias)?;
            attach(ctx, &spec).await?;
            ctx.read_empty()
        }
        Some(Intercepted::Detach { alias }) => {
            detach(ctx, &alias).await?;
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
            let dataframe = ctx.execute_logical_plan(plan).await?;
            // SET mutates DataFusion's live SessionConfig while planning the
            // statement. Refresh the runtime after every ordinary statement;
            // for queries this is a cheap clone and for SET it makes the new
            // values available to the next VGI bind.
            sync_vgi_settings(ctx, &runtime);
            Ok(dataframe)
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
/// Relations with arguments use the flattened function key described above.
/// Ordinary three-part table names are untouched. The one table-name rewrite is
/// DuckDB's two-part `catalog.table` shorthand: for a known attached alias it
/// becomes `catalog.<worker-default-schema>.table`.
///
/// Upstream tracks the underlying gap as apache/datafusion#18021; a PR that
/// added schema-scoped table functions (#18022) was closed in favour of #15095,
/// which is itself dormant.
fn rewrite_vgi_sql(ctx: &SessionContext, statement: &mut DFStatement) -> DFResult<()> {
    use datafusion::sql::sqlparser::ast::{
        BinaryOperator, Expr as SQLExpr, FunctionArg, FunctionArgExpr, Ident, ObjectName,
        ObjectNamePart, TableFactor, VisitorMut,
    };

    struct Rewrite<'a> {
        ctx: &'a SessionContext,
        scalar_macros: Vec<String>,
        table_macro_frames: Vec<Option<String>>,
        active_table_macros: HashSet<String>,
    }

    impl VisitorMut for Rewrite<'_> {
        type Break = Box<datafusion::common::DataFusionError>;

        fn pre_visit_table_factor(&mut self, tf: &mut TableFactor) -> ControlFlow<Self::Break> {
            let macro_key = match tf {
                TableFactor::Table {
                    name,
                    args: Some(_),
                    ..
                } if lookup_macro(self.ctx, name).is_some() => macro_key(name),
                _ => None,
            };
            if let Some(key) = &macro_key {
                if self.active_table_macros.contains(key) {
                    return ControlFlow::Break(Box::new(plan_datafusion_err!(
                        "recursive VGI table macro expansion involving `{key}`"
                    )));
                }
            }
            match expand_table_macro(self.ctx, tf) {
                Ok(Some(expanded)) => {
                    let key = macro_key.expect("an expanded table macro has a registered key");
                    self.active_table_macros.insert(key.clone());
                    self.table_macro_frames.push(Some(key));
                    *tf = expanded;
                    return ControlFlow::Continue(());
                }
                Ok(None) => {}
                Err(error) => return ControlFlow::Break(Box::new(error)),
            }
            self.table_macro_frames.push(None);
            if let TableFactor::Table { name, args, .. } = tf {
                if args.is_none() {
                    if let [catalog, _table] = name.0.as_slice() {
                        if let Some(catalog) = catalog.as_ident() {
                            if let Some(schema) = attached_default_schema(self.ctx, &catalog.value)
                            {
                                name.0
                                    .insert(1, ObjectNamePart::Identifier(Ident::new(schema)));
                            }
                        }
                    }
                    return ControlFlow::Continue(());
                }
                let args = args.as_mut().expect("checked table-function arguments");
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

                let function_name = name
                    .0
                    .first()
                    .and_then(ObjectNamePart::as_ident)
                    .map(|ident| ident.value.clone())
                    .unwrap_or_default();

                // DataFusion's TableFactor::Table planner rejects named
                // FunctionArgs before a TableFunctionImpl can see them. Carry
                // each one as a private, single-field struct literal instead;
                // the VGI implementation unwraps it back into Arguments::named.
                // This preserves names and ordering without changing DataFusion.
                for argument in &mut args.args {
                    let named = match argument {
                        FunctionArg::Named { name, arg, .. } => match arg {
                            FunctionArgExpr::Expr(value) => {
                                Some((name.value.clone(), value.clone()))
                            }
                            _ => None,
                        },
                        FunctionArg::ExprNamed {
                            name: SQLExpr::Identifier(name),
                            arg: FunctionArgExpr::Expr(value),
                            ..
                        } => Some((name.value.clone(), value.clone())),
                        // DuckDB also accepts `f(option=value)`. The generic
                        // DataFusion dialect parses that as a positional
                        // equality expression, so recognize it only when VGI
                        // discovery says `option` really is a named parameter.
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::BinaryOp {
                            left,
                            op: BinaryOperator::Eq,
                            right,
                        })) => match left.as_ref() {
                            SQLExpr::Identifier(name)
                                if is_declared_named_table_argument(
                                    self.ctx,
                                    &function_name,
                                    &name.value,
                                ) =>
                            {
                                Some((name.value.clone(), right.as_ref().clone()))
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some((name, value)) = named else {
                        continue;
                    };
                    *argument = FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::Struct {
                        values: vec![SQLExpr::Named {
                            expr: Box::new(value),
                            name: Ident::new(format!("{NAMED_ARG_PREFIX}{name}")),
                        }],
                        fields: vec![],
                    }));
                }
            }
            ControlFlow::Continue(())
        }

        fn post_visit_table_factor(&mut self, _tf: &mut TableFactor) -> ControlFlow<Self::Break> {
            if let Some(Some(key)) = self.table_macro_frames.pop() {
                self.active_table_macros.remove(&key);
            }
            ControlFlow::Continue(())
        }

        fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<Self::Break> {
            match expand_scalar_macro(self.ctx, expr, &mut self.scalar_macros) {
                Ok(Some(expanded)) => *expr = expanded,
                Ok(None) => {}
                Err(error) => return ControlFlow::Break(Box::new(error)),
            }
            // DataFusion supplies `&[]` to a zero-argument accumulator, with
            // no batch row count. Inject a reserved one-field struct as a row
            // witness. Its type marks this invocation (rather than the whole
            // UDAF) so an aggregate with all-default positional parameters can
            // still distinguish `f()` from the real one-argument call `f(x)`.
            if let SQLExpr::Function(function) = expr {
                let function_name = function.name.to_string();
                // DataFusion gives scalar UDFs precedence when a scalar and
                // aggregate share a registry key (even when SQL includes an
                // OVER clause). Preserve that resolution rule rather than
                // turning an invalid zero-argument scalar call into a valid
                // one-argument call.
                let scalar_collision = self
                    .ctx
                    .state()
                    .scalar_functions()
                    .keys()
                    .any(|name| name.eq_ignore_ascii_case(&function_name));
                if !scalar_collision && is_declared_nullary_aggregate(self.ctx, &function_name) {
                    if let FunctionArguments::List(arguments) = &mut function.args {
                        if arguments.args.is_empty() {
                            arguments
                                .args
                                .push(FunctionArg::Unnamed(FunctionArgExpr::Expr(
                                    SQLExpr::Struct {
                                        values: vec![SQLExpr::Named {
                                            expr: Box::new(SQLExpr::Value(
                                                Value::Number("1".into(), false).into(),
                                            )),
                                            name: Ident::new(crate::aggregate::ROW_WITNESS_FIELD),
                                        }],
                                        fields: vec![],
                                    },
                                )));
                        }
                    }
                }
            }
            ControlFlow::Continue(())
        }
    }

    match statement {
        DFStatement::Statement(inner) => {
            // `visit` walks the whole SQL AST — CTEs, subqueries, joins — so
            // nested calls are covered without hand-rolling the recursion.
            if let ControlFlow::Break(error) = inner.as_mut().visit(&mut Rewrite {
                ctx,
                scalar_macros: Vec::new(),
                table_macro_frames: Vec::new(),
                active_table_macros: HashSet::new(),
            }) {
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
        let rpc_timeout = integer_option("rpc_timeout")?
            .map(|seconds| {
                if seconds == 0 {
                    return plan_err!("ATTACH option `rpc_timeout` must be greater than zero");
                }
                Ok(Duration::from_secs(seconds))
            })
            .transpose()?;
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
        let allow_local_format_paths = bool_option("allow_local_format_paths", false)?;
        let mut connection = VgiConnection::pooled(location, pool)
            .with_cache_enabled(cache_enabled)
            .with_local_format_paths(allow_local_format_paths)
            .with_connection_options(vgi_client::ConnectionOptions {
                worker_debug,
                launcher_idle_timeout,
                launcher_state_dir: launcher_state_dir.map(Into::into),
                rpc_timeout,
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

    fn connection_with_runtime(&self, runtime: Arc<VgiRuntime>) -> DFResult<VgiConnection> {
        let mut connection = self.connection()?;
        let attachment_timeout = connection.connection_options.rpc_timeout;
        connection = connection.with_runtime(runtime);
        // An ATTACH-local deadline is more specific than the session default.
        // Keep VgiConnection::with_runtime replacement semantics intact for
        // API callers that deliberately swap runtimes more than once.
        if attachment_timeout.is_some() {
            connection.connection_options.rpc_timeout = attachment_timeout;
        }
        Ok(connection)
    }
}

async fn attach(ctx: &SessionContext, spec: &AttachSpec) -> DFResult<()> {
    let lifecycle = session_lifecycle_lock(ctx);
    let _guard = lifecycle.lock().await;
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
    // Parse local policy before discovery or registry mutation. In particular,
    // a malformed re-ATTACH must leave the existing attachment usable.
    let global_functions_enabled = spec
        .options
        .get("global_functions")
        .map(|value| option_string(value))
        .transpose()?
        .map(|value| match value.trim() {
            value if value.eq_ignore_ascii_case("true") => Ok(true),
            value if value.eq_ignore_ascii_case("false") => Ok(false),
            _ => plan_err!("ATTACH option `global_functions` must be true or false"),
        })
        .transpose()?
        .unwrap_or(true);
    let started = std::time::Instant::now();
    let mut conn = spec.connection_with_runtime(runtime)?;
    crate::diagnostics::register(ctx, Arc::clone(conn.runtime()));
    let options = build_attach_options(ctx, &conn, spec).await?;
    conn = conn.with_catalog_attach_options(&spec.catalog, options);
    let provider =
        VgiCatalogProvider::discover_as(conn.clone(), &spec.catalog, &spec.alias).await?;
    // Every fallible declaration conversion happens before the old alias is
    // removed. Applying this prepared set below is then registry-only and
    // cannot strand a failed re-ATTACH between old metadata and new functions.
    let prepared_global_functions = if global_functions_enabled {
        prepare_global_functions(&provider)?
    } else {
        Vec::new()
    };
    // Re-attaching an alias refreshes its flat function registrations as well
    // as its catalog provider.
    deregister_alias_functions(ctx, &spec.alias);
    register_table_functions(ctx, &conn, spec, &provider);
    register_scalar_functions(ctx, &conn, spec, &provider);
    register_aggregate_functions(ctx, &conn, spec, &provider);
    let published_global_functions = if global_functions_enabled {
        register_global_functions(ctx, &conn, spec, &prepared_global_functions)
    } else {
        Vec::new()
    };
    register_macros(ctx, spec, &provider);
    ctx.register_catalog(&spec.alias, provider.clone());
    register_views(ctx, spec, &provider).await;
    record_default_schema(ctx, &spec.alias, provider.default_schema());
    // Capture diagnostics after view planning so duckdb_columns() can expose a
    // view's actual DataFusion output fields as well as its VGI comments.
    conn.runtime().set_catalog_metadata(
        &spec.alias,
        crate::runtime::VgiCatalogMetadata {
            connection: conn.metadata_connection(),
            worker_catalog: spec.catalog.clone(),
            comment: provider.catalog_comment().map(str::to_string),
            tags: provider.catalog_tags().to_vec(),
            resolved_data_version: provider.resolved_data_version().map(str::to_string),
            resolved_implementation_version: provider
                .resolved_implementation_version()
                .map(str::to_string),
            schemas: provider.schema_infos().to_vec(),
            tables: provider.tables().cloned().collect(),
            table_branches: HashMap::new(),
            views: provider.metadata_views(),
            functions: provider.functions().cloned().collect(),
            macros: provider.metadata_macros().cloned().collect(),
            settings: provider.settings().to_vec(),
            global_function_prefix: provider.global_function_prefix().to_string(),
            global_functions: provider.global_functions().to_vec(),
            published_global_functions,
        },
    );
    let mut event = crate::VgiEvent::new("catalog.attached");
    event.catalog = Some(spec.alias.clone());
    event.duration = Some(started.elapsed());
    event.message = Some(format!("worker catalog `{}`", spec.catalog));
    conn.runtime().emit(event);
    Ok(provider.companion_catalogs().to_vec())
}

/// Build the complete namespace used to resolve catalog-owned SQL without
/// guessing whether an arbitrary identifier belongs to the worker.
fn sql_catalog_namespace(provider: &VgiCatalogProvider) -> SqlCatalogNamespace {
    let schemas = provider
        .vgi_schemas()
        .map(|(schema_name, schema)| {
            let mut namespace = SqlMacroNamespace::default();
            namespace.expression_functions.extend(
                schema
                    .scalars()
                    .iter()
                    .map(|(name, _, _)| name.to_ascii_lowercase()),
            );
            namespace.expression_functions.extend(
                schema
                    .aggregates()
                    .iter()
                    .map(|(name, ..)| name.to_ascii_lowercase()),
            );
            namespace.relation_functions.extend(
                schema
                    .table_function_names()
                    .map(|name| name.to_ascii_lowercase()),
            );
            namespace.relations.extend(
                schema
                    .table_names_only()
                    .into_iter()
                    .map(|name| name.to_ascii_lowercase()),
            );
            namespace
                .relations
                .extend(schema.views().map(|(name, _)| name.to_ascii_lowercase()));
            for info in schema.metadata_macros() {
                match sql_macro_kind(&info.macro_type.0) {
                    Ok(SqlMacroKind::Scalar) => {
                        namespace
                            .expression_functions
                            .insert(info.name.to_ascii_lowercase());
                    }
                    Ok(SqlMacroKind::Table) => {
                        namespace
                            .relation_functions
                            .insert(info.name.to_ascii_lowercase());
                    }
                    Err(_) => {}
                }
            }
            (schema_name.to_ascii_lowercase(), namespace)
        })
        .collect();
    SqlCatalogNamespace { schemas }
}

/// Plan worker-declared views against their owning schema and install ordinary
/// DataFusion `ViewTable`s. A broken definition does not make ATTACH fail: the
/// view remains discoverable and reports its retained error when queried.
async fn register_views(ctx: &SessionContext, spec: &AttachSpec, provider: &VgiCatalogProvider) {
    use datafusion::datasource::ViewTable;
    let namespace = sql_catalog_namespace(provider);

    for (schema_name, schema) in provider.vgi_schemas() {
        for (view_name, info) in schema.views() {
            let planned =
                async {
                    let state = ctx.state();
                    let dialect = state.config_options().sql_parser.dialect;
                    let mut statement = state.sql_to_statement(&info.definition, &dialect)?;
                    if let DFStatement::Statement(inner) = &mut statement {
                        qualify_catalog_sql(inner.as_mut(), &spec.alias, schema_name, &namespace);
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

enum PreparedGlobalKind {
    Scalar {
        overloads: Vec<vgi_client::ArgSpecs>,
        volatility: datafusion::logical_expr::Volatility,
    },
    Aggregate {
        specs: vgi_client::ArgSpecs,
        nullary: bool,
    },
    Table(crate::catalog::TableFunctionMetadata),
}

struct PreparedGlobalFunction<'a> {
    info: &'a vgi_client::dtos::FunctionInfo,
    name: String,
    kind: PreparedGlobalKind,
}

fn prepare_global_functions(
    provider: &VgiCatalogProvider,
) -> DFResult<Vec<PreparedGlobalFunction<'_>>> {
    let prefix = provider.global_function_prefix();
    let mut prepared: Vec<PreparedGlobalFunction<'_>> = Vec::new();
    for info in provider.global_functions() {
        let name = if prefix.is_empty() {
            info.name.to_ascii_lowercase()
        } else {
            format!("{prefix}_{}", info.name).to_ascii_lowercase()
        };
        let function_type = info.function_type.0.to_ascii_lowercase();
        let specs = vgi_client::ArgSpecs::parse(&info.arguments.0).map_err(crate::to_df)?;
        if function_type == "scalar" {
            // One DataFusion ScalarUDF owns all overloads for one concrete VGI
            // dispatch target. A normalization collision between distinct
            // worker targets remains first-wins below rather than being merged
            // into an implementation that could dispatch to only one of them.
            if let Some(existing) = prepared.iter_mut().find(|candidate| {
                candidate.name == name
                    && candidate.info.name.eq_ignore_ascii_case(&info.name)
                    && candidate
                        .info
                        .schema_name
                        .eq_ignore_ascii_case(&info.schema_name)
                    && matches!(&candidate.kind, PreparedGlobalKind::Scalar { .. })
            }) {
                let PreparedGlobalKind::Scalar {
                    overloads,
                    volatility,
                } = &mut existing.kind
                else {
                    unreachable!("scalar group predicate checked")
                };
                overloads.push(specs);
                *volatility = most_volatile(*volatility, metadata_volatility(info));
                continue;
            }
            prepared.push(PreparedGlobalFunction {
                info,
                name,
                kind: PreparedGlobalKind::Scalar {
                    overloads: vec![specs],
                    volatility: metadata_volatility(info),
                },
            });
            continue;
        }
        let kind = match function_type.as_str() {
            "aggregate" => {
                let nullary = specs.minimum_positional_arity() == 0;
                PreparedGlobalKind::Aggregate { specs, nullary }
            }
            "table" | "table_buffering" => {
                PreparedGlobalKind::Table(crate::catalog::TableFunctionMetadata {
                    specs,
                    buffered: function_type == "table_buffering",
                    input_from_args: info.input_from_args,
                    stream_cache_eligible: info.max_workers != Some(1) && !info.has_finalize,
                })
            }
            _ => {
                return plan_err!(
                    "worker nominated global function `{}` with unsupported type `{}`",
                    info.name,
                    info.function_type.0
                )
            }
        };
        prepared.push(PreparedGlobalFunction { info, name, kind });
    }
    Ok(prepared)
}

fn most_volatile(
    left: datafusion::logical_expr::Volatility,
    right: datafusion::logical_expr::Volatility,
) -> datafusion::logical_expr::Volatility {
    use datafusion::logical_expr::Volatility;
    match (left, right) {
        (Volatility::Volatile, _) | (_, Volatility::Volatile) => Volatility::Volatile,
        (Volatility::Stable, _) | (_, Volatility::Stable) => Volatility::Stable,
        _ => Volatility::Immutable,
    }
}

fn register_global_functions(
    ctx: &SessionContext,
    conn: &VgiConnection,
    spec: &AttachSpec,
    functions: &[PreparedGlobalFunction<'_>],
) -> Vec<String> {
    let mut published = Vec::new();
    for function in functions {
        let info = function.info;
        let name = &function.name;
        let registered = match &function.kind {
            PreparedGlobalKind::Scalar {
                overloads,
                volatility,
            } => {
                let udf = AsyncScalarUDF::new(Arc::new(
                    VgiScalarUdf::discovered_overloads_with_volatility(
                        conn.clone(),
                        &spec.catalog,
                        &info.schema_name,
                        &info.name,
                        name,
                        overloads.clone(),
                        *volatility,
                    ),
                ))
                .into_scalar_udf();
                register_scalar_if_absent(ctx, &spec.alias, name.clone(), udf)
            }
            PreparedGlobalKind::Aggregate { specs, nullary } => {
                let udaf = AggregateUDF::new_from_impl(
                    VgiAggregateUdf::new_with_volatility(
                        conn.clone(),
                        &spec.catalog,
                        &info.schema_name,
                        &info.name,
                        name,
                        metadata_volatility(info),
                    )
                    .with_arg_specs(specs.clone())
                    .with_window_support(info.supports_window)
                    .with_required_secrets(metadata_secrets(info)),
                );
                register_aggregate_if_absent(ctx, &spec.alias, name.clone(), udaf, *nullary)
            }
            PreparedGlobalKind::Table(metadata) => {
                let named_arguments = named_arguments(&metadata.specs);
                let table: Arc<dyn TableFunctionImpl> = Arc::new(VgiTableFunction::new(
                    conn.clone(),
                    &spec.catalog,
                    &info.schema_name,
                    &info.name,
                    Some(metadata.clone()),
                ));
                register_table_if_absent(ctx, &spec.alias, name.clone(), table, named_arguments)
            }
        };
        if !registered {
            // Global publication is an ergonomic alias. The qualified worker
            // function remains available, so an earlier DataFusion registry
            // owner wins without making this catalog impossible to attach.
            continue;
        }
        published.push(name.clone());
    }
    published
}

const IMPLEMENTED_LOCAL_OPTIONS: &[&str] = &[
    "cache",
    "pool",
    "pool_max",
    "pool_timeout",
    "rpc_timeout",
    "worker_debug",
    "launcher_idle_timeout",
    "launcher_state_dir",
    "data_version_spec",
    "implementation_version",
    "bearer_token",
    "oauth_refresh_token",
    "attach_companions",
    "global_functions",
    "allow_local_format_paths",
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

fn register_macros(ctx: &SessionContext, spec: &AttachSpec, provider: &VgiCatalogProvider) {
    let namespace = sql_catalog_namespace(provider);
    let Ok(mut sessions) = macro_registry().lock() else {
        return;
    };
    let macros = sessions.entry(ctx.session_id()).or_default();
    let alias_prefix = format!("{}.", spec.alias.to_ascii_lowercase());
    macros.retain(|name, _| !name.starts_with(&alias_prefix));
    for info in provider.metadata_macros() {
        let (kind, mut body) = match sql_macro_kind(&info.macro_type.0) {
            Ok(kind) => (kind, parse_macro_body(kind, &info.definition)),
            Err(error) => (SqlMacroKind::Scalar, SqlMacroBody::Invalid(error)),
        };
        qualify_macro_body(&mut body, &spec.alias, &info.schema_name, &namespace);
        let defaults = match vgi_client::decode_macro_defaults(info) {
            Ok(defaults) => defaults
                .into_iter()
                .map(|(name, value)| (name.to_ascii_lowercase(), value))
                .collect(),
            Err(error) => {
                body = SqlMacroBody::Invalid(format!("invalid parameter defaults: {error}"));
                HashMap::new()
            }
        };
        let registered = Arc::new(SqlMacro {
            kind,
            parameters: info.parameters.clone(),
            defaults,
            body,
        });
        macros.insert(
            format!("{}.{}.{}", spec.alias, info.schema_name, info.name).to_ascii_lowercase(),
            Arc::clone(&registered),
        );
        if info
            .schema_name
            .eq_ignore_ascii_case(provider.default_schema())
        {
            macros.insert(
                format!("{}.{}", spec.alias, info.name).to_ascii_lowercase(),
                registered,
            );
        }
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
            let declared_named_arguments = metadata
                .as_ref()
                .map(|metadata| named_arguments(&metadata.specs))
                .unwrap_or_default();
            let make = || -> Arc<dyn TableFunctionImpl> {
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
                register_table_if_absent(
                    ctx,
                    &spec.alias,
                    name,
                    make(),
                    declared_named_arguments.clone(),
                );
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
                let udf = VgiScalarUdf::discovered_overloads_with_volatility(
                    conn.clone(),
                    &spec.catalog,
                    schema_name,
                    function,
                    &name,
                    overloads.clone(),
                    *volatility,
                );
                register_scalar_if_absent(
                    ctx,
                    &spec.alias,
                    name,
                    AsyncScalarUDF::new(Arc::new(udf)).into_scalar_udf(),
                );
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
        for (function, specs, volatility, supports_window, required_secrets) in schema.aggregates()
        {
            for name in publish_names(&spec.alias, schema_name, function) {
                let udaf = AggregateUDF::new_from_impl(
                    VgiAggregateUdf::new_with_volatility(
                        conn.clone(),
                        &spec.catalog,
                        schema_name,
                        function,
                        &name,
                        *volatility,
                    )
                    .with_arg_specs(specs.clone())
                    .with_window_support(*supports_window)
                    .with_required_secrets(required_secrets.clone()),
                );
                register_aggregate_if_absent(
                    ctx,
                    &spec.alias,
                    name,
                    udaf,
                    specs.minimum_positional_arity() == 0,
                );
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
async fn detach(ctx: &SessionContext, alias: &str) -> DFResult<()> {
    let lifecycle = session_lifecycle_lock(ctx);
    let _guard = lifecycle.lock().await;
    if ctx.catalog(alias).is_none() {
        return plan_err!("no catalog attached as {alias:?}");
    }
    deregister_alias_functions(ctx, alias);
    remove_default_schema(ctx, alias);
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
#[path = "session_macro_tests.rs"]
mod macro_qualification_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn install_test_macro(
        ctx: &SessionContext,
        names: &[&str],
        kind: SqlMacroKind,
        parameters: &[&str],
        defaults: &[(&str, i64)],
        definition: &str,
    ) {
        let registered = Arc::new(SqlMacro {
            kind,
            parameters: parameters.iter().map(|value| value.to_string()).collect(),
            defaults: defaults
                .iter()
                .map(|(name, value)| {
                    (
                        name.to_ascii_lowercase(),
                        Arc::new(datafusion::arrow::array::Int64Array::from(vec![*value]))
                            as ArrayRef,
                    )
                })
                .collect(),
            body: parse_macro_body(kind, definition),
        });
        let mut sessions = macro_registry().lock().unwrap();
        let macros = sessions.entry(ctx.session_id()).or_default();
        for name in names {
            macros.insert(name.to_ascii_lowercase(), Arc::clone(&registered));
        }
    }

    async fn query_i64(ctx: &SessionContext, query: &str) -> i64 {
        let batches = sql(ctx, query)
            .await
            .unwrap_or_else(|error| panic!("{query} failed: {error}"))
            .collect()
            .await
            .unwrap_or_else(|error| panic!("{query} did not collect: {error}"));
        let value =
            datafusion::common::ScalarValue::try_from_array(batches[0].column(0).as_ref(), 0)
                .unwrap();
        match value {
            datafusion::common::ScalarValue::Int64(Some(value)) => value,
            other => panic!("{query} returned {other:?}, expected one Int64"),
        }
    }

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
    fn remote_local_format_paths_require_an_explicit_opt_in() {
        let denied = spec("example?location=https://worker.example/vgi")
            .connection()
            .unwrap();
        assert!(!denied.allows_local_format_paths());

        let allowed =
            spec("example?location=https://worker.example/vgi&allow_local_format_paths=true")
                .connection()
                .unwrap();
        assert!(allowed.allows_local_format_paths());
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
    fn rpc_timeout_is_validated_and_overrides_the_session_default() {
        let connection = spec("example?location=tcp://127.0.0.1:1&rpc_timeout=7")
            .connection()
            .expect("positive timeout");
        assert_eq!(
            connection.rpc_timeout(),
            Some(std::time::Duration::from_secs(7))
        );

        let runtime = Arc::new(crate::VgiRuntime::new(crate::VgiSessionOptions {
            rpc_timeout: Some(std::time::Duration::from_secs(11)),
            ..Default::default()
        }));
        let connection = spec("example?location=tcp://127.0.0.1:1&rpc_timeout=7")
            .connection_with_runtime(Arc::clone(&runtime))
            .unwrap();
        assert_eq!(
            connection.rpc_timeout(),
            Some(std::time::Duration::from_secs(7)),
            "attachment timeout must win over the session default"
        );

        let inherited = spec("example?location=tcp://127.0.0.1:1")
            .connection_with_runtime(runtime)
            .unwrap();
        assert_eq!(
            inherited.rpc_timeout(),
            Some(std::time::Duration::from_secs(11))
        );

        let error = spec("example?location=tcp://127.0.0.1:1&rpc_timeout=0")
            .connection()
            .unwrap_err()
            .to_string();
        assert!(error.contains("must be greater than zero"), "{error}");
        let error = spec("example?location=tcp://127.0.0.1:1&rpc_timeout=-1")
            .connection()
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-negative integer"), "{error}");
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

    #[test]
    fn duckdb_equals_rewrites_only_declared_named_table_arguments() {
        let ctx = SessionContext::new();
        registration_registry()
            .lock()
            .unwrap()
            .entry(ctx.session_id())
            .or_default()
            .entry("example".to_string())
            .or_default()
            .table
            .insert(
                "example.sequence".to_string(),
                HashSet::from(["increment".to_string()]),
            );
        let parse_and_rewrite = |sql: &str| {
            let state = ctx.state();
            let dialect = state.config_options().sql_parser.dialect;
            let mut statement = state.sql_to_statement(sql, &dialect).expect("parses");
            rewrite_vgi_sql(&ctx, &mut statement).expect("rewrites");
            statement.to_string()
        };

        let named =
            parse_and_rewrite("SELECT * FROM example.sequence(5, increment=10, batch_size := 2)");
        assert!(
            named.contains(&format!("{NAMED_ARG_PREFIX}increment")),
            "declared `increment` was not carried as a named argument: {named}"
        );
        assert!(
            named.contains(&format!("{NAMED_ARG_PREFIX}batch_size")),
            "explicit named syntax regressed: {named}"
        );

        let equality = parse_and_rewrite("SELECT * FROM example.sequence(5, other=10)");
        assert!(
            !equality.contains(&format!("{NAMED_ARG_PREFIX}other")),
            "an undeclared equality was mistaken for a named argument: {equality}"
        );
        assert!(
            equality.contains("other = 10"),
            "equality was lost: {equality}"
        );
    }

    #[test]
    fn zero_argument_aggregate_gets_an_invocation_marked_row_witness() {
        let ctx = SessionContext::new();
        registration_registry()
            .lock()
            .unwrap()
            .entry(ctx.session_id())
            .or_default()
            .entry("example".to_string())
            .or_default()
            .aggregate
            .insert("example.main.defaulted_count".to_string(), true);
        let parse_and_rewrite = |sql: &str| {
            let state = ctx.state();
            let dialect = state.config_options().sql_parser.dialect;
            let mut statement = state.sql_to_statement(sql, &dialect).expect("parses");
            rewrite_vgi_sql(&ctx, &mut statement).expect("rewrites");
            statement.to_string()
        };

        let omitted = parse_and_rewrite("SELECT example.main.defaulted_count() FROM range(3)");
        assert!(
            omitted.contains(crate::aggregate::ROW_WITNESS_FIELD),
            "zero-argument invocation was not marked: {omitted}"
        );

        let supplied = parse_and_rewrite("SELECT example.main.defaulted_count(7) FROM range(3)");
        assert!(
            !supplied.contains(crate::aggregate::ROW_WITNESS_FIELD),
            "a real positional argument was mistaken for a witness: {supplied}"
        );
    }

    #[test]
    fn attached_catalog_two_part_tables_use_the_worker_default_schema() {
        let ctx = SessionContext::new();
        record_default_schema(&ctx, "example", "main");
        let state = ctx.state();
        let dialect = state.config_options().sql_parser.dialect;
        let mut statement = state
            .sql_to_statement(
                "SELECT * FROM example.first_ten JOIN local.items ON true",
                &dialect,
            )
            .expect("parses");
        rewrite_vgi_sql(&ctx, &mut statement).expect("rewrites");
        let rewritten = statement.to_string();
        assert!(
            rewritten.contains("example.main.first_ten"),
            "attached alias did not gain its default schema: {rewritten}"
        );
        assert!(
            rewritten.contains("local.items"),
            "unattached two-part name was changed: {rewritten}"
        );
    }

    #[tokio::test]
    async fn scalar_macros_expand_expressions_named_arguments_and_defaults() {
        let ctx = SessionContext::new();
        install_test_macro(
            &ctx,
            &["example.vgi_multiply", "example.main.vgi_multiply"],
            SqlMacroKind::Scalar,
            &["x", "y"],
            &[],
            "x * y",
        );
        install_test_macro(
            &ctx,
            &["example.vgi_clamp", "example.main.vgi_clamp"],
            SqlMacroKind::Scalar,
            &["val", "lo", "hi"],
            &[("lo", 0), ("hi", 100)],
            "GREATEST(lo, LEAST(hi, val))",
        );

        assert_eq!(
            query_i64(&ctx, "SELECT example.vgi_multiply(3, 4)").await,
            12
        );
        assert_eq!(
            query_i64(&ctx, "SELECT example.main.vgi_multiply(2 + 3, 4)").await,
            20
        );
        assert_eq!(query_i64(&ctx, "SELECT example.vgi_clamp(50)").await, 50);
        assert_eq!(
            query_i64(&ctx, "SELECT example.vgi_clamp(5, lo := 10)").await,
            10
        );
        assert_eq!(
            query_i64(&ctx, "SELECT example.vgi_clamp(50, hi := 25)").await,
            25
        );
    }

    #[tokio::test]
    async fn table_macros_expand_to_existing_datafusion_relations() {
        let ctx = SessionContext::new();
        install_test_macro(
            &ctx,
            &["example.vgi_range_table", "example.main.vgi_range_table"],
            SqlMacroKind::Table,
            &["n"],
            &[],
            "SELECT * FROM range(n)",
        );

        assert_eq!(
            query_i64(&ctx, "SELECT COUNT(*) FROM example.vgi_range_table(10)").await,
            10
        );
        assert_eq!(
            query_i64(
                &ctx,
                "SELECT MIN(value) FROM example.main.vgi_range_table(n := 5)",
            )
            .await,
            0
        );
        assert_eq!(
            query_i64(
                &ctx,
                "SELECT MAX(r.value) FROM example.vgi_range_table(5) AS r",
            )
            .await,
            4
        );
    }

    #[tokio::test]
    async fn macro_binding_rejects_ambiguous_or_incomplete_calls() {
        let ctx = SessionContext::new();
        install_test_macro(
            &ctx,
            &["example.pair"],
            SqlMacroKind::Scalar,
            &["left", "right"],
            &[],
            "left + right",
        );

        for (query, expected) in [
            (
                "SELECT example.pair(1)",
                "missing required argument(s): right",
            ),
            (
                "SELECT example.pair(1, other := 2)",
                "has no parameter named `other`",
            ),
            (
                "SELECT example.pair(1, right := 2, right := 3)",
                "received parameter `right` more than once",
            ),
            (
                "SELECT example.pair(right := 2, 1)",
                "does not accept positional arguments after named arguments",
            ),
        ] {
            let error = sql(&ctx, query).await.unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "unexpected error for {query}: {error}"
            );
        }
    }

    #[tokio::test]
    async fn scalar_macros_compose_and_recursive_macros_fail_cleanly() {
        let ctx = SessionContext::new();
        install_test_macro(
            &ctx,
            &["example.inc"],
            SqlMacroKind::Scalar,
            &["x"],
            &[],
            "x + 1",
        );
        install_test_macro(
            &ctx,
            &["example.twice"],
            SqlMacroKind::Scalar,
            &["x"],
            &[],
            "example.inc(x) * 2",
        );
        install_test_macro(
            &ctx,
            &["example.loop_scalar"],
            SqlMacroKind::Scalar,
            &["x"],
            &[],
            "example.loop_scalar(x)",
        );
        install_test_macro(
            &ctx,
            &["example.loop_table"],
            SqlMacroKind::Table,
            &["n"],
            &[],
            "SELECT * FROM example.loop_table(n)",
        );

        assert_eq!(query_i64(&ctx, "SELECT example.twice(4)").await, 10);
        let scalar_error = sql(&ctx, "SELECT example.loop_scalar(1)")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            scalar_error.contains("recursive VGI scalar macro"),
            "{scalar_error}"
        );
        let table_error = sql(&ctx, "SELECT * FROM example.loop_table(1)")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            table_error.contains("recursive VGI table macro"),
            "{table_error}"
        );
    }

    #[test]
    fn scalar_macro_parser_rejects_query_clauses_it_cannot_preserve() {
        for definition in ["x FROM secret_table", "x WHERE false", "x LIMIT 1"] {
            let SqlMacroBody::Invalid(error) = parse_macro_body(SqlMacroKind::Scalar, definition)
            else {
                panic!("accepted lossy scalar macro definition: {definition}");
            };
            assert!(error.contains("only one expression"), "{error}");
        }
        assert!(matches!(
            parse_macro_body(
                SqlMacroKind::Scalar,
                "x + (SELECT MAX(y) FROM values_table)"
            ),
            SqlMacroBody::Scalar(_)
        ));

        assert_eq!(sql_macro_kind("SCALAR"), Ok(SqlMacroKind::Scalar));
        assert_eq!(sql_macro_kind("table_macro"), Ok(SqlMacroKind::Table));
        assert!(sql_macro_kind("not_a_table").is_err());
    }
}
