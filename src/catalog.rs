// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Exposing a whole VGI catalog to DataFusion.
//!
//! # The half-async problem, and how this answers it
//!
//! DataFusion's [`SchemaProvider`] is half async: [`table`](SchemaProvider::table)
//! may await, but [`table_names`](SchemaProvider::table_names) and
//! [`table_exist`](SchemaProvider::table_exist) may not. So *something* has to
//! be resolved eagerly, and the only question is what.
//!
//! The obvious reading — resolve everything up front — is ruinous here. Naming
//! a table is one `catalog_schema_contents_functions` call for the whole schema;
//! producing a [`TableProvider`] means **binding** that function, because
//! `TableProvider::schema()` is synchronous and DataFusion needs the schema
//! during planning. Binding every table at attach time costs one bind per table
//! whether or not the query touches it. Against the reference fixture worker,
//! which defines a couple of hundred functions, that was the difference between
//! attaching in about a second and attaching in about thirteen minutes.
//!
//! So this splits the two: **names eagerly, binds lazily.** One RPC per schema
//! at attach; a bind on first use, memoised. That also matches how the DuckDB
//! extension behaves — its `VgiCatalogSet` lazy-loads entries and its table set
//! resolves single tables on demand rather than materialising the catalog.
//!
//! A bind that fails is remembered as "not a usable bare table" rather than
//! retried on every lookup — plenty of functions require arguments and will
//! never bind bare, and re-binding them on each plan would reintroduce the cost
//! this design exists to avoid.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::{CatalogProvider, SchemaProvider, Session, TableProvider};
use datafusion::common::{Constraints, DFSchema, DataFusionError, Result as DFResult, Statistics};
use datafusion::execution::SessionState;
use datafusion::logical_expr::utils::expr_to_columns;
use datafusion::logical_expr::{
    Expr, ExprSchemable, TableProviderFilterPushDown, TableType, Volatility,
};
use datafusion::physical_expr::expressions::Column as PhysicalColumn;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::ExecutionPlan;

use crate::{datafusion_constraints, to_df, VgiConnection, VgiTableProvider};

type CachedTable = Result<Arc<dyn TableProvider>, String>;

/// A catalog table whose schema is available from discovery and whose scan
/// function is bound only when DataFusion actually scans it.
///
/// DataFusion builds `information_schema.columns` and `views` by asking for
/// every listed provider, even when SQL filters name one table. Binding here
/// would therefore turn metadata lookup into one RPC per table and make a
/// valid argument-dependent/multi-branch table abort the whole query.
#[derive(Debug)]
struct VgiCatalogTableProvider {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    info: vgi_client::TableInfo,
    at: Option<vgi_client::At>,
    /// Complete catalog schema, including virtual generated columns.
    output_schema: SchemaRef,
    /// Columns the worker's nominated scan function physically emits.
    physical_schema: SchemaRef,
    constraints: Constraints,
    bound: tokio::sync::OnceCell<Arc<VgiTableProvider>>,
    statistics: tokio::sync::OnceCell<Option<Arc<Statistics>>>,
}

impl VgiCatalogTableProvider {
    fn new(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        info: vgi_client::TableInfo,
        at: Option<vgi_client::At>,
    ) -> DFResult<Arc<Self>> {
        let output_schema = vgi_protocol::ipc::read_schema(&info.columns.0).map_err(to_df)?;
        let physical_schema = Arc::new(Schema::new_with_metadata(
            output_schema
                .fields()
                .iter()
                .filter(|field| !field.metadata().contains_key("generated_expression"))
                .cloned()
                .collect::<Vec<_>>(),
            output_schema.metadata().clone(),
        ));
        let constraints = datafusion_constraints(
            info.primary_key_constraints.clone(),
            info.unique_constraints.clone(),
            output_schema.fields().len(),
        )?;
        Ok(Arc::new(Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            info,
            at,
            output_schema,
            physical_schema,
            constraints,
            bound: tokio::sync::OnceCell::new(),
            statistics: tokio::sync::OnceCell::new(),
        }))
    }

    async fn bound(&self) -> DFResult<&Arc<VgiTableProvider>> {
        let provider = self
            .bound
            .get_or_try_init(|| async {
                let provider = match &self.at {
                    Some(at) => {
                        VgiTableProvider::bind_catalog_table_at(
                            self.conn.clone(),
                            &self.catalog,
                            &self.schema_name,
                            self.info.clone(),
                            at.clone(),
                        )
                        .await
                    }
                    None => {
                        VgiTableProvider::bind_catalog_table(
                            self.conn.clone(),
                            &self.catalog,
                            &self.schema_name,
                            self.info.clone(),
                        )
                        .await
                    }
                }?;
                provider.with_declared_schema(Arc::clone(&self.physical_schema))
            })
            .await?;
        Ok(provider)
    }

    async fn column_statistics(&self) -> DFResult<Option<Arc<Statistics>>> {
        self.statistics
            .get_or_try_init(|| async {
                if !self.info.supports_column_statistics || self.at.is_some() {
                    return Ok(None);
                }
                let raw = if let Some(inline) = self
                    .info
                    .column_statistics
                    .as_ref()
                    .filter(|value| !value.0.is_empty())
                {
                    vgi_protocol::ipc::read_batch(&inline.0).map_err(to_df)?
                } else {
                    let connection = self.conn.clone();
                    let catalog = self.catalog.clone();
                    let schema = self.info.schema_name.clone();
                    let table = self.info.name.clone();
                    tokio::task::spawn_blocking(move || {
                        let mut client = connection.connect()?;
                        let attached = connection.attach(&mut client, &catalog)?;
                        client
                            .table_column_statistics(&attached, &schema, &table)
                            .map_err(to_df)
                    })
                    .await
                    .map_err(|error| DataFusionError::External(Box::new(error)))??
                };
                Ok(Some(Arc::new(crate::statistics_for_catalog_table(
                    &self.output_schema,
                    &raw,
                    self.info.cardinality_estimate.0,
                    self.info.cardinality_max.0,
                    None,
                ))))
            })
            .await
            .cloned()
    }

    async fn filters_prune_table(&self, state: &dyn Session, filters: &[Expr]) -> DFResult<bool> {
        let Some(statistics) = self.column_statistics().await? else {
            return Ok(false);
        };
        Ok(crate::filters_prune_statistics(
            state,
            &self.output_schema,
            statistics,
            filters,
        ))
    }

    fn has_generated_columns(&self) -> bool {
        self.physical_schema.fields().len() != self.output_schema.fields().len()
    }

    /// Build the catalog-visible projection above a physical VGI scan.
    ///
    /// Generated expressions are catalog metadata, while the backing function
    /// emits only stored columns. DataFusion already has the SQL-expression and
    /// physical-projection APIs needed to bridge those two schemas, so no
    /// engine-specific logical node is necessary here.
    fn generated_projection(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        input: Arc<dyn ExecutionPlan>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "VGI generated columns require DataFusion SessionState SQL expression support"
                        .to_string(),
                )
            })?;
        let physical_df_schema = DFSchema::try_from(self.physical_schema.as_ref().clone())?;
        let selected: Vec<usize> = projection
            .cloned()
            .unwrap_or_else(|| (0..self.output_schema.fields().len()).collect());
        let projected_schema = self.output_schema.project(&selected)?;
        let mut expressions = Vec::with_capacity(selected.len());

        for index in selected {
            let field = self.output_schema.field(index);
            let expr = if let Some(sql) = field.metadata().get("generated_expression") {
                let logical = session_state
                    .create_logical_expr(sql, &physical_df_schema)?
                    .cast_to(field.data_type(), &physical_df_schema)?;
                state.create_physical_expr(logical, &physical_df_schema)?
            } else {
                let physical_index = self.physical_schema.index_of(field.name())?;
                Arc::new(PhysicalColumn::new(field.name(), physical_index))
            };
            expressions.push(ProjectionExpr {
                expr,
                alias: field.name().clone(),
            });
        }

        Ok(Arc::new(ProjectionExec::try_new_with_schema_metadata(
            expressions,
            input,
            &projected_schema,
        )?))
    }

    /// A generated predicate cannot be sent to a function that has no such
    /// physical input column. Physical-only predicates remain useful worker
    /// hints; this provider reports every filter as `Inexact`, so DataFusion
    /// still evaluates the predicate above the generated projection.
    fn physical_filters(&self, filters: &[Expr]) -> Vec<Expr> {
        let physical_names = self
            .physical_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<HashSet<_>>();
        filters
            .iter()
            .filter(|filter| {
                let mut columns = HashSet::new();
                expr_to_columns(filter, &mut columns).is_ok()
                    && columns
                        .iter()
                        .all(|column| physical_names.contains(column.name.as_str()))
            })
            .cloned()
            .collect()
    }
}

#[async_trait]
impl TableProvider for VgiCatalogTableProvider {
    fn schema(&self) -> datafusion::arrow::datatypes::SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(&self.constraints)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // The scan bind discovers exactness later. Inexact is safe here: VGI
        // still receives supported predicates and DataFusion always rechecks.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if self.filters_prune_table(state, filters).await? {
            let schema = match projection {
                Some(indices) => Arc::new(self.output_schema.project(indices)?),
                None => Arc::clone(&self.output_schema),
            };
            return Ok(Arc::new(EmptyExec::new(schema)));
        }
        let bound = self.bound().await?;
        if !self.has_generated_columns() {
            return bound.scan(state, projection, filters, limit).await;
        }

        // Fetch every stored column because an arbitrary generated expression
        // may depend on any of them. Do not push LIMIT below the generated
        // projection: an outer predicate may remove rows after generation.
        let physical_filters = self.physical_filters(filters);
        let input = bound.scan(state, None, &physical_filters, None).await?;
        self.generated_projection(state, projection, input)
    }
}

/// Discovery metadata needed to choose the correct execution protocol for a
/// callable table function.
#[derive(Debug, Clone)]
pub(crate) struct TableFunctionMetadata {
    pub specs: vgi_client::ArgSpecs,
    pub buffered: bool,
    pub input_from_args: bool,
}

/// One VGI schema.
#[derive(Debug)]
pub struct VgiSchemaProvider {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    /// Per-function execution shape and argument declarations.
    table_functions: HashMap<String, TableFunctionMetadata>,
    /// Every name the schema advertises — catalog **tables** and table
    /// **functions**, resolved once at attach.
    ///
    /// Both live in one namespace because SQL has one: `ex.data.t` does not say
    /// which kind `t` is, and the worker guarantees the names are distinct
    /// within a schema.
    names: Vec<String>,
    /// Actual relations exposed through DataFusion metadata. Table functions
    /// remain resolvable through `table()` for the useful bare-call error, but
    /// are routines rather than rows in `SHOW TABLES`.
    relation_names: Vec<String>,
    /// Which of those names are catalog tables. They bind differently: a table
    /// is scanned through the function the worker nominates
    /// (`catalog_table_scan_function_get`), with the worker's own arguments.
    tables: HashMap<String, vgi_client::TableInfo>,
    /// Scalar functions in this schema, with what the worker declares about
    /// their parameters. They are not tables and never appear in
    /// [`Self::table_names`]; they are published into DataFusion's separate
    /// function registry at attach time.
    ///
    /// The specs travel with the name because a call cannot be built without
    /// them: a const parameter belongs in the bind, not the input batch.
    scalars: Vec<(String, Vec<vgi_client::ArgSpecs>, Volatility)>,
    /// Aggregate functions in this schema, published into DataFusion's
    /// aggregate registry at attach time.
    aggregates: Vec<(
        String,
        vgi_client::ArgSpecs,
        Volatility,
        bool,
        Vec<vgi_client::SecretLookupRequest>,
    )>,
    /// Scalar and table macro declarations retained for metadata inspection.
    metadata_macros: Vec<vgi_client::dtos::MacroInfo>,
    /// SQL views declared by the worker. Their definitions are planned after
    /// the catalog and its functions have been registered with DataFusion.
    views: HashMap<String, vgi_client::dtos::ViewInfo>,
    /// Complete worker declarations retained for metadata diagnostics. The
    /// execution-specific collections above are derived from these same rows.
    functions: Vec<vgi_client::dtos::FunctionInfo>,
    /// Bind results, memoised. An `Err` records a function that will not bind
    /// bare — most fixture functions take arguments — so it is not retried,
    /// and the worker's own reason is kept to report at plan time.
    bound: Mutex<HashMap<String, CachedTable>>,
    /// Historical binds are isolated from the current-table cache because a
    /// past version may expose a different schema.
    versioned: Mutex<HashMap<(String, String, String), CachedTable>>,
}

impl VgiSchemaProvider {
    /// List one schema's tables and table functions. Two RPCs; no binds.
    pub async fn discover(
        conn: VgiConnection,
        catalog: &str,
        schema_name: &str,
    ) -> DFResult<Arc<Self>> {
        let (c, cat, sch) = (conn.clone(), catalog.to_string(), schema_name.to_string());
        let (tables, table_functions, scalars, aggregates, metadata_macros, views, functions) =
            tokio::task::spawn_blocking(move || {
                let mut client = c.connect()?;
                let attached = c.attach(&mut client, &cat)?;
                let tables = client.tables(&attached, &sch).map_err(to_df)?;
                let table_infos = client
                    .functions(&attached, &sch, vgi_client::FunctionKind::Table)
                    .map_err(to_df)?;
                // `function_type` distinguishes the three table shapes that share
                // one listing filter; the buffered one needs the Sink+Source
                // protocol rather than a streaming exchange.
                //
                // The wire carries the enum's *member name* — `TABLE_BUFFERING`,
                // not the lowercase `table_buffering` value — the same convention
                // that governs `FunctionKind`. Matched case-insensitively so a
                // worker that sends either spelling is understood.
                let table_functions = table_infos
                    .iter()
                    .map(|f| {
                        let specs = vgi_client::ArgSpecs::parse(&f.arguments.0).map_err(to_df)?;
                        let metadata = TableFunctionMetadata {
                            buffered: f.function_type.0.eq_ignore_ascii_case("table_buffering"),
                            input_from_args: f.input_from_args,
                            specs,
                        };
                        Ok::<_, DataFusionError>((f.name.clone(), metadata))
                    })
                    .collect::<Result<HashMap<_, _>, _>>()?;
                let scalar_infos = client
                    .functions(&attached, &sch, vgi_client::FunctionKind::Scalar)
                    .map_err(to_df)?;
                let aggregate_infos = client
                    .functions(&attached, &sch, vgi_client::FunctionKind::Aggregate)
                    .map_err(to_df)?;
                let aggregates = aggregate_infos
                    .iter()
                    .map(|f| {
                        let specs = vgi_client::ArgSpecs::parse(&f.arguments.0).map_err(to_df)?;
                        let secrets = f
                            .required_secrets
                            .iter()
                            .map(|secret| vgi_client::SecretLookupRequest {
                                secret_type: secret.secret_type.clone(),
                                scope: secret.scope.clone(),
                                name: secret.secret_name.clone(),
                            })
                            .collect();
                        Ok::<_, DataFusionError>((
                            f.name.clone(),
                            specs,
                            volatility(f.stability.as_ref().map(|v| v.0.as_str())),
                            f.supports_window,
                            secrets,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let scalar_macro_infos = client
                    .macros(&attached, &sch, vgi_client::MacroKind::Scalar)
                    .map_err(to_df)?;
                let table_macro_infos = client
                    .macros(&attached, &sch, vgi_client::MacroKind::Table)
                    .map_err(to_df)?;
                let metadata_macros = scalar_macro_infos
                    .into_iter()
                    .chain(table_macro_infos)
                    .collect::<Vec<_>>();
                let views = client
                    .views(&attached, &sch)
                    .map_err(to_df)?
                    .into_iter()
                    .map(|info| (info.name.clone(), info))
                    .collect::<HashMap<_, _>>();
                // DataFusion registers one scalar UDF per SQL name, whereas
                // VGI advertises one FunctionInfo per overload. Preserve the
                // complete overload set so the UDF can choose the right const
                // layout for each call instead of whichever overload happened
                // to be registered first.
                let mut scalar_overloads: HashMap<String, (Vec<vgi_client::ArgSpecs>, Volatility)> =
                    HashMap::new();
                for f in &scalar_infos {
                    let specs = vgi_client::ArgSpecs::parse(&f.arguments.0).map_err(to_df)?;
                    let declared = volatility(f.stability.as_ref().map(|v| v.0.as_str()));
                    let entry = scalar_overloads
                        .entry(f.name.clone())
                        .or_insert_with(|| (Vec::new(), declared));
                    entry.0.push(specs);
                    entry.1 = most_volatile(entry.1, declared);
                }
                let scalars = scalar_overloads
                    .into_iter()
                    .map(|(name, (overloads, volatility))| (name, overloads, volatility))
                    .collect::<Vec<_>>();
                let functions = table_infos
                    .into_iter()
                    .chain(scalar_infos)
                    .chain(aggregate_infos)
                    .collect::<Vec<_>>();
                Ok::<_, DataFusionError>((
                    tables,
                    table_functions,
                    scalars,
                    aggregates,
                    metadata_macros,
                    views,
                    functions,
                ))
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let tables: HashMap<String, vgi_client::TableInfo> =
            tables.into_iter().map(|t| (t.name.clone(), t)).collect();
        let mut relation_names: Vec<String> = tables.keys().cloned().collect();
        relation_names.extend(
            views
                .keys()
                .filter(|name| !tables.contains_key(*name))
                .cloned(),
        );
        let mut names = relation_names.clone();
        names.extend(
            table_functions
                .keys()
                .filter(|name| !tables.contains_key(*name))
                .cloned(),
        );

        Ok(Arc::new(Self {
            conn,
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            names,
            relation_names,
            tables,
            scalars,
            aggregates,
            metadata_macros,
            views,
            functions,
            table_functions,
            bound: Mutex::new(HashMap::new()),
            versioned: Mutex::new(HashMap::new()),
        }))
    }

    /// Names that are catalog tables, not functions.
    pub fn table_names_only(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    /// Scalar functions this schema advertises, with their parameter specs.
    pub fn scalars(&self) -> &[(String, Vec<vgi_client::ArgSpecs>, Volatility)] {
        &self.scalars
    }

    /// Aggregate functions this schema advertises.
    pub fn aggregates(
        &self,
    ) -> &[(
        String,
        vgi_client::ArgSpecs,
        Volatility,
        bool,
        Vec<vgi_client::SecretLookupRequest>,
    )] {
        &self.aggregates
    }

    /// Views declared by this schema.
    pub(crate) fn views(&self) -> impl Iterator<Item = (&String, &vgi_client::dtos::ViewInfo)> {
        self.views.iter()
    }

    /// Complete function declarations used to publish SQL metadata views.
    pub(crate) fn functions(&self) -> &[vgi_client::dtos::FunctionInfo] {
        &self.functions
    }

    pub(crate) fn metadata_macros(&self) -> &[vgi_client::dtos::MacroInfo] {
        &self.metadata_macros
    }

    pub(crate) fn tables(&self) -> impl Iterator<Item = &vgi_client::dtos::TableInfo> {
        self.tables.values()
    }

    /// Install a planned view (or its durable planning error) in the same lazy
    /// table cache used for remote tables.
    pub(crate) fn install_view(&self, name: &str, table: CachedTable) {
        if let Ok(mut cache) = self.bound.lock() {
            cache.insert(name.to_string(), table);
        }
    }

    /// Discovery metadata for a callable table function.
    pub(crate) fn table_function_metadata(&self, name: &str) -> Option<&TableFunctionMetadata> {
        self.table_functions.get(name)
    }

    /// Names that are callable table functions (excluding tables and views).
    pub(crate) fn table_function_names(&self) -> impl Iterator<Item = &String> {
        self.table_functions.keys()
    }

    /// Look up a memoised bind without holding the lock across an await.
    fn cached(&self, name: &str) -> Option<CachedTable> {
        self.bound.lock().ok()?.get(name).cloned()
    }

    /// Bind a catalog table at an explicit VGI time-travel coordinate.
    pub(crate) async fn table_at(
        &self,
        name: &str,
        at: vgi_client::At,
    ) -> DFResult<Arc<dyn TableProvider>> {
        if !self.tables.contains_key(name) {
            return Err(DataFusionError::Plan(format!(
                "VGI time travel is only supported for catalog tables; `{name}` is not one"
            )));
        }
        let key = (name.to_string(), at.unit.clone(), at.value.clone());
        if let Some(hit) = self
            .versioned
            .lock()
            .ok()
            .and_then(|cache| cache.get(&key).cloned())
        {
            return hit.map_err(bind_failed(name));
        }

        // Discovery describes the current table. Ask for TableInfo again at
        // this coordinate so schema evolution is visible during planning.
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema = self.schema_name.clone();
        let table = name.to_string();
        let lookup_at = at.clone();
        let info = tokio::task::spawn_blocking(move || {
            let mut client = conn.connect()?;
            let attached = conn.attach(&mut client, &catalog)?;
            client
                .table_get(&attached, &schema, &table, Some(&lookup_at))
                .map_err(to_df)?
                .ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "VGI catalog table `{schema}.{table}` was not found at {} {}",
                        lookup_at.unit, lookup_at.value
                    ))
                })
        })
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))??;

        let bound = VgiCatalogTableProvider::new(
            self.conn.clone(),
            &self.catalog,
            &self.schema_name,
            info,
            Some(at),
        )
        .map(|provider| provider as Arc<dyn TableProvider>)
        .map_err(|error| error.to_string());
        if let Ok(mut cache) = self.versioned.lock() {
            cache.insert(key, bound.clone());
        }
        bound.map_err(bind_failed(name))
    }
}

fn volatility(value: Option<&str>) -> Volatility {
    match value {
        Some(value) if value.eq_ignore_ascii_case("VOLATILE") => Volatility::Volatile,
        Some(value) if value.eq_ignore_ascii_case("CONSISTENT_WITHIN_QUERY") => Volatility::Stable,
        _ => Volatility::Immutable,
    }
}

fn most_volatile(left: Volatility, right: Volatility) -> Volatility {
    match (left, right) {
        (Volatility::Volatile, _) | (_, Volatility::Volatile) => Volatility::Volatile,
        (Volatility::Stable, _) | (_, Volatility::Stable) => Volatility::Stable,
        _ => Volatility::Immutable,
    }
}

#[async_trait]
impl SchemaProvider for VgiSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.relation_names.clone()
    }

    async fn table_type(&self, name: &str) -> DFResult<Option<TableType>> {
        if self.tables.contains_key(name) {
            Ok(Some(TableType::Base))
        } else if self.views.contains_key(name) {
            Ok(Some(TableType::View))
        } else {
            Ok(None)
        }
    }

    async fn table(&self, name: &str) -> DFResult<Option<Arc<dyn TableProvider>>> {
        if !self.names.iter().any(|n| n == name) {
            return Ok(None);
        }
        if let Some(hit) = self.cached(name) {
            return hit.map(Some).map_err(bind_failed(name));
        }
        if self.views.contains_key(name) {
            return Err(DataFusionError::Plan(format!(
                "VGI view `{name}` was discovered but has not been planned"
            )));
        }

        // Two callers racing the same cold name both bind; the loser's result
        // is dropped. That is cheaper and simpler than holding a lock across
        // the await, and binds are idempotent.
        let bound = match self.tables.get(name) {
            Some(info) => VgiCatalogTableProvider::new(
                self.conn.clone(),
                &self.catalog,
                &self.schema_name,
                info.clone(),
                None,
            )
            .map(|provider| provider as Arc<dyn TableProvider>),
            None => {
                VgiTableProvider::bind(self.conn.clone(), &self.catalog, &self.schema_name, name)
                    .await
                    .map(|provider| provider as Arc<dyn TableProvider>)
            }
        }
        .map_err(|e| e.to_string());

        if let Ok(mut cache) = self.bound.lock() {
            cache.insert(name.to_string(), bound.clone());
        }
        bound.map(Some).map_err(bind_failed(name))
    }

    fn table_exist(&self, name: &str) -> bool {
        // Answers from the name list, so it stays synchronous and cheap — which
        // is the whole point of not binding eagerly. A function that is
        // advertised but needs arguments reports `true` here and then fails in
        // `table()` with the worker's own reason, which is a far better message
        // than "table not found" for something that plainly exists.
        self.names.iter().any(|n| n == name)
    }
}

/// Turn a bind failure into a plan error that says why.
fn bind_failed(name: &str) -> impl Fn(String) -> DataFusionError + '_ {
    move |reason| {
        DataFusionError::Plan(format!(
            "VGI function `{name}` is advertised by the worker but does not bind \
             as a bare table: {reason}"
        ))
    }
}

/// A whole VGI catalog.
#[derive(Debug)]
pub struct VgiCatalogProvider {
    schemas: HashMap<String, Arc<VgiSchemaProvider>>,
    comment: Option<String>,
    tags: Vec<(String, String)>,
    default_schema: String,
    resolved_data_version: Option<String>,
    resolved_implementation_version: Option<String>,
    schema_infos: Vec<vgi_client::dtos::SchemaInfo>,
    /// The prefix the worker asked for on globally-published functions
    /// (`global_function_prefix`), empty when it asked for none.
    ///
    /// A worker that already publishes globals to DuckDB has an opinion about
    /// what they should be called; honouring it means one worker gets the same
    /// spelling on both engines, rather than each client inventing its own.
    global_function_prefix: String,
    global_functions: Vec<vgi_client::dtos::FunctionInfo>,
    companion_catalogs: Vec<vgi_client::dtos::AttachCatalogInfo>,
}

impl VgiCatalogProvider {
    /// Attach a catalog and list its schemas.
    pub async fn discover(conn: VgiConnection, catalog: &str) -> DFResult<Arc<Self>> {
        let (c, cat) = (conn.clone(), catalog.to_string());
        let (
            schema_infos,
            comment,
            tags,
            default_schema,
            resolved_data_version,
            resolved_implementation_version,
            global_function_prefix,
            global_functions,
            companion_catalogs,
        ) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = c.attach(&mut client, &cat)?;
            let info = attached.info();
            let prefix = info.global_function_prefix.clone();
            let comment = info.comment.clone();
            let tags = info.tags.clone();
            let default_schema = info.default_schema.clone();
            let resolved_data_version = info.resolved_data_version.clone();
            let resolved_implementation_version = info.resolved_implementation_version.clone();
            let global_functions = attached.global_functions().map_err(to_df)?;
            let companion_catalogs = attached.companion_catalogs().map_err(to_df)?;
            let schema_infos = client.schemas(&attached).map_err(to_df)?;
            Ok::<_, DataFusionError>((
                schema_infos,
                comment,
                tags,
                default_schema,
                resolved_data_version,
                resolved_implementation_version,
                prefix,
                global_functions,
                companion_catalogs,
            ))
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let mut schemas: HashMap<String, Arc<VgiSchemaProvider>> = HashMap::new();
        for name in schema_infos.iter().map(|schema| schema.name.clone()) {
            let sp = VgiSchemaProvider::discover(conn.clone(), catalog, &name).await?;
            schemas.insert(name, sp);
        }
        Ok(Arc::new(Self {
            schemas,
            comment,
            tags,
            default_schema,
            resolved_data_version,
            resolved_implementation_version,
            schema_infos,
            global_function_prefix,
            global_functions,
            companion_catalogs,
        }))
    }
}

impl VgiCatalogProvider {
    pub(crate) fn catalog_comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    pub(crate) fn catalog_tags(&self) -> &[(String, String)] {
        &self.tags
    }

    pub(crate) fn default_schema(&self) -> &str {
        &self.default_schema
    }

    pub(crate) fn resolved_data_version(&self) -> Option<&str> {
        self.resolved_data_version.as_deref()
    }

    pub(crate) fn resolved_implementation_version(&self) -> Option<&str> {
        self.resolved_implementation_version.as_deref()
    }

    pub(crate) fn schema_infos(&self) -> &[vgi_client::dtos::SchemaInfo] {
        &self.schema_infos
    }

    /// The worker's requested prefix for globally-published functions.
    pub fn global_function_prefix(&self) -> &str {
        &self.global_function_prefix
    }

    /// Function descriptors explicitly nominated for global publication.
    pub fn global_functions(&self) -> &[vgi_client::dtos::FunctionInfo] {
        &self.global_functions
    }

    /// Companion catalogs requested by this attachment.
    pub fn companion_catalogs(&self) -> &[vgi_client::dtos::AttachCatalogInfo] {
        &self.companion_catalogs
    }

    /// This catalog's schemas, concretely — the registration paths need more
    /// than `SchemaProvider` exposes (scalar names are not tables).
    pub fn vgi_schemas(&self) -> impl Iterator<Item = (&String, &Arc<VgiSchemaProvider>)> {
        self.schemas.iter()
    }

    pub(crate) fn functions(&self) -> impl Iterator<Item = &vgi_client::dtos::FunctionInfo> {
        self.schemas.values().flat_map(|schema| schema.functions())
    }

    pub(crate) fn metadata_macros(&self) -> impl Iterator<Item = &vgi_client::dtos::MacroInfo> {
        self.schemas
            .values()
            .flat_map(|schema| schema.metadata_macros())
    }

    pub(crate) fn tables(&self) -> impl Iterator<Item = &vgi_client::dtos::TableInfo> {
        self.schemas.values().flat_map(|schema| schema.tables())
    }

    pub(crate) fn metadata_views(&self) -> Vec<(vgi_client::dtos::ViewInfo, Vec<String>)> {
        self.schemas
            .values()
            .flat_map(|schema| {
                schema.views().map(|(name, info)| {
                    let columns = schema
                        .cached(name)
                        .and_then(Result::ok)
                        .map(|provider| {
                            provider
                                .schema()
                                .fields()
                                .iter()
                                .map(|field| field.name().clone())
                                .collect()
                        })
                        .unwrap_or_default();
                    (info.clone(), columns)
                })
            })
            .collect()
    }

    /// Resolve one historical table through its concrete VGI schema.
    pub(crate) async fn table_at(
        &self,
        schema: &str,
        table: &str,
        at: vgi_client::At,
    ) -> DFResult<Arc<dyn TableProvider>> {
        let provider = self.schemas.get(schema).ok_or_else(|| {
            DataFusionError::Plan(format!("VGI schema `{schema}` does not exist"))
        })?;
        provider.table_at(table, at).await
    }
}

impl CatalogProvider for VgiCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas
            .get(name)
            .map(|s| Arc::clone(s) as Arc<dyn SchemaProvider>)
    }
}
