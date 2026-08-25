// Copyright 2025, 2026 Query Farm LLC - https://query.farm

#![warn(missing_docs)]

//! Query a remote VGI catalog from Apache DataFusion.
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//! use vgi_datafusion::{VgiConnection, VgiTableProvider};
//!
//! # async fn demo() -> datafusion::error::Result<()> {
//! let conn = VgiConnection::subprocess(["my-worker"]);
//! let ctx = SessionContext::new();
//! ctx.register_table(
//!     "remote_orders",
//!     VgiTableProvider::bind(conn, "my_catalog", "main", "orders").await?,
//! )?;
//! ctx.sql("SELECT count(*) FROM remote_orders").await?.show().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # The blocking bridge
//!
//! `vgi-client` is synchronous, as are the Python and Java VGI clients. Every
//! call therefore runs inside [`tokio::task::spawn_blocking`], and each scan
//! partition owns its own connection — which is not a workaround but how VGI
//! parallelism works: a worker advertises `max_workers` and expects that many
//! independent connections.
//!
//! This is why [`VgiConnection`] is a *factory* rather than a client: a
//! `VgiClient` cannot be shared across partitions, so each one makes its own.
//!
//! # What maps cleanly, and what does not
//!
//! Producer-mode table functions map onto `TableProvider` almost exactly:
//! projection, filters, limit and split claims all ride VGI's scan calls.
//! Exchange-mode and buffered functions use a scalar subquery as their TABLE
//! argument; DataFusion restricts that subquery to one column, so wider table
//! inputs need an upstream planner change.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, BooleanArray, Int64Array, StringArray, UnionArray};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::{
    ColumnStatistics, DataFusionError, Result as DFResult, ScalarValue, Statistics,
};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::{expressions::Column, EquivalenceProperties, PhysicalSortExpr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, StatisticsArgs,
};
use vgi_client::{
    Arguments, AttachOptions, AttachedCatalog, BindSpec, FunctionKind, PlanOptions, PooledClient,
    ScanOptions, ScanPlan, ScanSplitInfo, VgiClient, VgiLocation, WorkerPool,
};

mod aggregate;
mod catalog;
mod filters;
mod scalar;
mod session;
mod table_function;
mod table_input;

pub use aggregate::VgiAggregateUdf;
pub use catalog::{VgiCatalogProvider, VgiSchemaProvider};
pub use scalar::VgiScalarUdf;
pub use session::{sql, AttachSpec};
pub use table_function::VgiTableFunction;

/// How to reach a VGI worker.
///
/// A factory rather than a connection: each scan partition opens its own, which
/// is how a VGI scan fans out across the worker's advertised `max_workers`.
///
/// Connections come from a [`WorkerPool`], and that is load-bearing rather than
/// tuning. Worker startup is ~1.8 s for the Python reference worker, and this
/// adapter opens a connection per schema, per table bind, and per scan
/// partition — so without reuse, attaching a catalog costs one process start
/// per table.
#[derive(Clone)]
pub struct VgiConnection {
    pool: WorkerPool,
    location: VgiLocation,
    connection_options: vgi_client::ConnectionOptions,
    label: String,
    /// Shared authentication state for every HTTP client in this attachment.
    auth: Option<Arc<dyn vgi_client::auth::CatalogAuth>>,
    /// Per-catalog options used when the session handle is first established.
    attach_options: Arc<HashMap<String, AttachOptions>>,
    /// Attach handles, one per catalog, established once and reused.
    ///
    /// `attach_opaque_data` is the worker's **session token**, not a
    /// connection detail: a worker scopes per-attach state to it, so
    /// re-attaching for every call starts a fresh session each time and any
    /// state the caller accumulated is invisible to the next call. That is not
    /// a subtle inefficiency — it silently changes results. The `accumulate`
    /// fixture, which collects rows under a name across calls, returned only
    /// the current call's rows until this was cached, and answered "no
    /// accumulation named … in this session" for reads.
    ///
    /// Caching it is also what the extension does: it attaches once per
    /// catalog and hands the same bytes to every worker it later talks to,
    /// which is why the token is opaque rather than a live handle — it has to
    /// survive being carried to a *different* pooled worker.
    attached: Arc<Mutex<HashMap<String, vgi_client::AttachedCatalog>>>,
}

impl fmt::Debug for VgiConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VgiConnection")
            .field("label", &self.label)
            .field("pool", &self.pool.stats())
            .finish()
    }
}

impl VgiConnection {
    /// Reach a worker named the way the DuckDB extension names one.
    ///
    /// Accepts every `LOCATION` spelling `vgi_client` understands — a bare
    /// command, `http://`, `unix://`, `tcp://`, `launch:` — so one string drives
    /// both clients and a corpus written for one runs against the other.
    pub fn from_location(location: &str) -> DFResult<Self> {
        Ok(Self::pooled(
            VgiLocation::parse(location).map_err(to_df)?,
            WorkerPool::default(),
        ))
    }

    /// Reach a parsed location through a pool you supply.
    ///
    /// Share one pool across catalogs pointing at the same worker and they
    /// share its connections; pass [`vgi_client::PoolConfig::disabled`] to prove
    /// a test does not depend on reuse (the extension's `pool false`).
    pub fn pooled(location: VgiLocation, pool: WorkerPool) -> Self {
        let auth = matches!(location, VgiLocation::Http(_)).then(|| {
            Arc::new(vgi_client::auth::OAuthAuth::new(
                Box::new(vgi_client::auth::oauth::UreqTransport),
                Box::new(vgi_client::auth::StderrInteraction),
            )) as Arc<dyn vgi_client::auth::CatalogAuth>
        });
        Self {
            label: location.label(),
            location,
            pool,
            connection_options: vgi_client::ConnectionOptions::default(),
            auth,
            attach_options: Arc::new(HashMap::new()),
            attached: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Use explicit authentication state for this HTTP connection factory.
    ///
    /// Clones share the same state, so OAuth login and refresh are single-flight
    /// across catalog discovery and parallel scan partitions.
    pub fn with_auth(mut self, auth: Arc<dyn vgi_client::auth::CatalogAuth>) -> DFResult<Self> {
        if !matches!(self.location, VgiLocation::Http(_)) {
            return Err(DataFusionError::Plan(
                "VGI authentication requires an HTTP(S) LOCATION".to_string(),
            ));
        }
        self.auth = Some(auth);
        self.attached = Arc::new(Mutex::new(HashMap::new()));
        Ok(self)
    }

    /// Configure the options used to attach one catalog.
    #[must_use]
    pub fn with_catalog_attach_options(
        mut self,
        catalog: impl Into<String>,
        options: AttachOptions,
    ) -> Self {
        let mut configured = self.attach_options.as_ref().clone();
        configured.insert(catalog.into(), options);
        self.attach_options = Arc::new(configured);
        self.attached = Arc::new(Mutex::new(HashMap::new()));
        self
    }

    /// Configure local subprocess/launcher behavior for this attachment.
    #[must_use]
    pub fn with_connection_options(mut self, options: vgi_client::ConnectionOptions) -> Self {
        self.connection_options = options;
        self.attached = Arc::new(Mutex::new(HashMap::new()));
        self
    }

    /// Attach `catalog`, reusing the handle from a previous attach.
    ///
    /// Every call site should use this rather than `client.attach(...)`
    /// directly — see [`Self::attached`] for why a fresh attach per call is a
    /// correctness problem and not just an extra round trip.
    pub fn attach(
        &self,
        client: &mut VgiClient,
        catalog: &str,
    ) -> DFResult<vgi_client::AttachedCatalog> {
        if let Ok(cache) = self.attached.lock() {
            if let Some(handle) = cache.get(catalog) {
                return Ok(handle.clone());
            }
        }
        let options = self
            .attach_options
            .get(catalog)
            .cloned()
            .unwrap_or_default();
        let handle = client.attach(catalog, options).map_err(to_df)?;
        if let Ok(mut cache) = self.attached.lock() {
            cache.insert(catalog.to_string(), handle.clone());
        }
        Ok(handle)
    }

    /// Spawn a worker as a child process.
    pub fn subprocess<I, S>(cmd: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let argv: Vec<String> = cmd.into_iter().map(Into::into).collect();
        Self::pooled(VgiLocation::Subprocess(argv), WorkerPool::default())
    }

    /// Talk to a worker serving VGI over HTTP.
    pub fn http(base_url: impl Into<String>) -> Self {
        Self::pooled(VgiLocation::Http(base_url.into()), WorkerPool::default())
    }

    /// Check out a connection.
    ///
    /// The guard derefs to [`VgiClient`] and returns the connection to the pool
    /// when it drops, so callers keep writing `let mut client = conn.connect()?`
    /// and get reuse for free.
    pub fn connect(&self) -> DFResult<PooledClient> {
        match &self.auth {
            Some(auth) => self
                .pool
                .acquire_with_auth(&self.location, Arc::clone(auth))
                .map_err(to_df),
            None => self
                .pool
                .acquire_with_options(&self.location, self.connection_options.clone())
                .map_err(to_df),
        }
    }

    /// The pool behind this connection, for stats or an explicit flush.
    pub fn pool(&self) -> &WorkerPool {
        &self.pool
    }

    /// A short label, for plan display.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// The cheapest column to fetch when the caller wants only a row count.
///
/// # This is DuckDB's rule, relocated
///
/// The DuckDB extension never faces this decision, because DuckDB resolves it in
/// the optimizer: `remove_unused_columns.cpp` refuses to hand a table function an
/// empty column list and substitutes `LogicalGet::GetAnyColumn()` — a virtual
/// empty column if the scan advertises one, else rowid, else column 0. So by the
/// time `VgiTableFunctionInitGlobal` runs, `input.column_ids` already holds
/// exactly one entry and a `count(*)` costs one column on the wire.
///
/// DataFusion has no such rule: it passes `Some(vec![])` straight to
/// `TableProvider::scan` and expects a zero-column batch carrying the row count.
/// And forwarding `[]` to the worker is not an option — an empty projection list
/// already means **all columns** in the VGI protocol, so it would ask for the
/// opposite of what is wanted. Hence the same choice, made here instead.
///
/// The one divergence: DuckDB takes rowid-or-column-0, this takes the narrowest
/// fixed-width column. Both send exactly one column; this one is never wider.
/// `None` when the table has no columns at all.
fn narrowest_column(bound: &vgi_client::BoundFunction) -> Option<i64> {
    use datafusion::arrow::datatypes::DataType;
    let schema = bound.output_schema();
    if schema.fields().is_empty() {
        return None;
    }
    let width = |d: &DataType| -> u32 {
        match d {
            DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Float64 => 8,
            // Anything variable-width or nested is worse than any fixed-width
            // column, whatever its declared size.
            _ => u32::MAX,
        }
    };
    schema
        .fields()
        .iter()
        .enumerate()
        .min_by_key(|(_, f)| width(f.data_type()))
        .map(|(i, _)| i as i64)
}

/// Capabilities that affect how a table scan is planned.
#[derive(Debug, Clone, Copy, Default)]
struct FunctionCapabilities {
    projection_pushdown: bool,
    /// `None` means discovery could not identify the function, so planning is
    /// still attempted for compatibility with hidden catalog scan functions
    /// and older workers.
    supports_splits: Option<bool>,
}

/// Capabilities shared by every advertised overload of `function`.
///
/// A missing function is conservative rather than exceptional: catalog scan
/// helpers can be bindable without being advertised as user-callable functions.
/// Older workers may also omit function discovery entirely while still
/// supporting direct binds, in which case projection remains local and split
/// planning is probed directly.
fn function_capabilities(
    client: &mut PooledClient,
    attached: &AttachedCatalog,
    schema_name: &str,
    function: &str,
) -> DFResult<FunctionCapabilities> {
    let functions = match client.functions(attached, schema_name, FunctionKind::Table) {
        Ok(functions) => functions,
        Err(e) if e.error_type == "MethodNotImplementedError" => return Ok(Default::default()),
        Err(e) => return Err(to_df(e)),
    };
    let matches = functions
        .iter()
        .filter(|info| info.name == function)
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Ok(Default::default());
    }
    Ok(FunctionCapabilities {
        projection_pushdown: matches
            .iter()
            .all(|info| info.projection_pushdown == Some(true)),
        supports_splits: Some(matches.iter().all(|info| info.supports_splits)),
    })
}

/// Make a batch match the schema its plan declared.
///
/// Two cases matter. A zero-column schema is `count(*)`: build an empty batch
/// that still carries the row count, since dropping the count would lose the
/// answer. Otherwise select the declared columns by name, which also handles a
/// worker that correctly ignored projection IDs it never opted into receiving.
fn conform(
    batch: datafusion::arrow::array::RecordBatch,
    schema: &SchemaRef,
) -> DFResult<datafusion::arrow::array::RecordBatch> {
    use datafusion::arrow::array::{RecordBatch, RecordBatchOptions};

    if batch.schema().fields() == schema.fields() {
        return Ok(batch);
    }
    if schema.fields().is_empty() {
        // `count(*)`: the plan wants a row count and no columns, so whatever
        // the worker sent is discarded and only the cardinality survives.
        return Ok(RecordBatch::try_new_with_options(
            schema.clone(),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?);
    }

    // Match by NAME, never by position.
    //
    // Projection pushdown is advisory: a worker may honour it and return
    // exactly the requested columns, or ignore it and return all of them.
    // Taking the first N columns positionally is correct in the first case and
    // silently wrong in the second — `SELECT b` would receive column `a`'s
    // values under the label `b`. That is not a hypothetical: it is what
    // `cache/coverage.test` caught, and the shape of the result (right column
    // count, right names, wrong data) means nothing downstream can notice.
    let columns = schema
        .fields()
        .iter()
        .map(|want| {
            batch.column_by_name(want.name()).cloned().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "worker returned no column named `{}`; it emitted [{}]",
                    want.name(),
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
        })
        .collect::<DFResult<Vec<_>>>()?;

    Ok(RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// Turn a VGI error into a DataFusion one, keeping the message.
fn to_df(e: vgi_client::RpcError) -> DataFusionError {
    DataFusionError::External(Box::new(std::io::Error::other(e.to_string())))
}

/// Run a synchronous VGI planning call outside any Tokio runtime.
///
/// DataFusion's table/scalar/aggregate type hooks are synchronous even when the
/// surrounding planner is async. The HTTP transport uses reqwest's blocking
/// client, which must not create its private runtime from a Tokio worker thread.
pub(crate) fn run_blocking_planner_call<T, F>(work: F) -> DFResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> DFResult<T> + Send + 'static,
{
    std::thread::Builder::new()
        .name("vgi-planner-rpc".to_string())
        .spawn(work)
        .map_err(|error| DataFusionError::External(Box::new(error)))?
        .join()
        .map_err(|_| DataFusionError::Execution("VGI planner RPC thread panicked".to_string()))?
}

/// A VGI table function exposed as a DataFusion table.
#[derive(Debug, Clone)]
pub struct VgiTableProvider {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    /// The call's bind arguments. Empty for a function reached as a bare table;
    /// populated when reached through [`VgiTableFunction`], which is why the
    /// same type serves both. They must be re-sent at scan time: `init` echoes
    /// the bind call back, so a scan bound without them is a different call.
    arguments: Arguments,
    /// Worker-supplied argument bytes, for a catalog table. Takes precedence
    /// over `arguments` and is forwarded verbatim.
    raw_arguments: Option<vgi_client::Bytes>,
    output_schema: SchemaRef,
    /// The worker opted into receiving projection IDs. Functions that do not
    /// opt in return their full schema and are narrowed locally instead.
    projection_pushdown: bool,
    /// The discovery-time split declaration. `None` means the function was not
    /// discoverable, so `table_function_plan` is still probed for compatibility.
    supports_splits: Option<bool>,
    max_workers: usize,
}

impl VgiTableProvider {
    /// Bind a remote function, resolving its schema.
    ///
    /// The bind runs now rather than at scan time because `TableProvider::schema`
    /// is synchronous — DataFusion needs the schema during planning, and there is
    /// nowhere to await by then.
    pub async fn bind(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
    ) -> DFResult<Arc<Self>> {
        Self::bind_with_arguments(conn, catalog, schema_name, function, Arguments::new()).await
    }

    /// Bind a **catalog table**.
    ///
    /// A VGI catalog table is not storage this client reads. It is a function
    /// call the worker picked: `catalog_table_scan_function_get` (or an inlined
    /// `scan_function` on the table info, saving the round trip) names the
    /// function and supplies its arguments.
    ///
    /// Those arguments arrive in a **different encoding** from bind arguments —
    /// a flat batch whose columns are the arguments (`arg_<N>` positional,
    /// anything else named), with no `args` struct wrapper. They must therefore
    /// be decoded and re-encoded rather than forwarded; passing the bytes
    /// straight through fails on the worker with `Field "args" does not exist
    /// in schema`. The DuckDB extension decodes them the same way
    /// (`DecodeScanArguments`).
    ///
    /// The output schema still comes from the bind rather than from
    /// `TableInfo::columns`. The two should agree, but the bind is what the
    /// scan will actually produce, and a mismatch there would surface as
    /// corrupt data rather than an error — DataFusion trusts the declared
    /// schema.
    pub async fn bind_catalog_table(
        conn: VgiConnection,
        catalog: impl Into<String>,
        _schema_name: impl Into<String>,
        info: vgi_client::TableInfo,
    ) -> DFResult<Arc<Self>> {
        let catalog = catalog.into();
        let c = conn.clone();
        let cat2 = catalog.clone();

        let (function, function_schema, arguments, output_schema, capabilities) =
            tokio::task::spawn_blocking(move || {
                let mut client = c.connect()?;
                let attached = c.attach(&mut client, &cat2)?;
                let scan = client
                    .table_scan_function(&attached, &info, None)
                    .map_err(to_df)?;
                let arguments = Arguments::from_scan_arguments(&scan.arguments.0).map_err(to_df)?;

                // The scan function does not necessarily live in the table's
                // schema. A worker registers function names per schema and may
                // reuse one name across them, so the bind has to name the schema
                // the function was actually found in — the extension resolves this
                // the same way, and says so in `vgi_table_entry.cpp`. In the
                // reference worker, tables in `data` are scanned by functions in
                // `main`, which fails outright without this.
                let default_schema = attached.default_schema().to_string();
                let mut candidates = vec![info.schema_name.clone()];
                if default_schema != info.schema_name {
                    candidates.push(default_schema);
                }

                let mut last_err = None;
                for schema in &candidates {
                    let spec = BindSpec::table(&scan.function_name)
                        .in_schema(schema)
                        .with_arguments(arguments.clone());
                    match client.bind(&attached, &spec) {
                        Ok(bound) => {
                            let capabilities = function_capabilities(
                                &mut client,
                                &attached,
                                schema,
                                &scan.function_name,
                            )?;
                            return Ok::<_, DataFusionError>((
                                scan.function_name,
                                schema.clone(),
                                arguments,
                                bound.output_schema().clone(),
                                capabilities,
                            ));
                        }
                        Err(e) => last_err = Some(e),
                    }
                }
                Err(to_df(last_err.expect("at least one candidate schema")))
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;

        Ok(Arc::new(Self {
            conn,
            catalog,
            // The scan binds in the function's schema, which may differ from
            // the table's; the scan re-binds and must resolve identically.
            schema_name: function_schema,
            function,
            arguments,
            raw_arguments: None,
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            max_workers: 1,
        }))
    }

    /// Bind a call synchronously.
    ///
    /// [`TableFunctionImpl::call_with_args`] is not async, so a table function
    /// has nowhere to await. That is fine here rather than a compromise:
    /// `vgi_client` is a blocking client, so this is the *direct* path and the
    /// async wrappers are the ones paying for a `spawn_blocking` hop. It does
    /// occupy the calling thread for the duration of one bind RPC, during
    /// planning.
    ///
    /// [`TableFunctionImpl::call_with_args`]: datafusion::catalog::TableFunctionImpl::call_with_args
    pub fn bind_blocking(
        conn: VgiConnection,
        catalog: &str,
        schema_name: &str,
        function: &str,
        arguments: Arguments,
    ) -> DFResult<Arc<Self>> {
        let mut client = conn.connect()?;
        let attached = conn.attach(&mut client, catalog)?;
        let spec = BindSpec::table(function)
            .in_schema(schema_name)
            .with_arguments(arguments.clone());
        let bound = client.bind(&attached, &spec).map_err(to_df)?;
        let output_schema = bound.output_schema().clone();
        let capabilities = function_capabilities(&mut client, &attached, schema_name, function)?;
        drop(client);

        Ok(Arc::new(Self {
            conn,
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            function: function.to_string(),
            arguments,
            raw_arguments: None,
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            max_workers: 1,
        }))
    }

    /// Bind a call that carries arguments.
    pub async fn bind_with_arguments(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        arguments: Arguments,
    ) -> DFResult<Arc<Self>> {
        let catalog = catalog.into();
        let schema_name = schema_name.into();
        let function = function.into();
        let c = conn.clone();
        let (cat2, sch2, fn2) = (catalog.clone(), schema_name.clone(), function.clone());
        let args2 = arguments.clone();

        let (output_schema, capabilities, max_workers) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = c.attach(&mut client, &cat2)?;
            let spec = BindSpec::table(&fn2).in_schema(&sch2).with_arguments(args2);
            let bound = client.bind(&attached, &spec).map_err(to_df)?;
            let schema = bound.output_schema().clone();
            let capabilities = function_capabilities(&mut client, &attached, &sch2, &fn2)?;
            // One partition at BIND time, deliberately. The real count is
            // decided at `scan`, where splits are planned.
            //
            // `max_workers` is only readable from a scan's header, so learning
            // it here would mean opening a scan and abandoning it — which both
            // costs a scan per bind and leaves the connection mid-stream, so
            // the pool would hand the next caller a broken one. (It did: binds
            // failed with "empty IPC stream (no schema)" until this came out.)
            //
            // A split plan carries `max_workers` directly, so the number that
            // matters arrives at `scan` without a wasted scan here. Without a
            // plan, one partition is also the honest answer: joining an existing
            // execution needs its id shared across partitions, and that
            // rendezvous is exactly what splits remove.
            Ok::<_, DataFusionError>((schema, capabilities, 1usize))
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        Ok(Arc::new(Self {
            conn,
            catalog,
            schema_name,
            function,
            arguments,
            raw_arguments: None,
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            max_workers,
        }))
    }
}

/// Split planning and bin-packing.
impl VgiTableProvider {
    /// Divide the scan into splits and pack them into partitions.
    ///
    /// Returns `None` when this scan is not split-capable, which keeps the
    /// pre-splits behaviour intact rather than making every worker implement
    /// planning: a worker that has not opted in answers with a single
    /// empty-payload split, which means "the whole scan is one unit of work" and
    /// is treated here as no plan at all.
    async fn plan_splits(
        &self,
        projection: Option<Vec<i64>>,
        // The blob AND the columns it reads travel together — see
        // filters::Pushdown. Passing the blob alone is what let the two drift.
        pushdown: filters::Pushdown,
        limit: Option<usize>,
        target_partitions: usize,
    ) -> DFResult<Option<PlannedSplits>> {
        // An advertised function that did not opt in must retain the ordinary
        // scan path. Unknown functions are still probed because catalog scan
        // helpers may intentionally be bindable without being discoverable.
        if self.supports_splits == Some(false) {
            return Ok(None);
        }

        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arguments = self.arguments.clone();
        let raw_arguments = self.raw_arguments.clone();

        // The client is blocking, so planning runs on a blocking thread rather
        // than a tokio worker — a blocking call on the runtime would stall every
        // other task sharing it.
        let plan =
            tokio::task::spawn_blocking(move || -> DFResult<Option<vgi_client::ScanPlan>> {
                let mut client = conn.connect()?;
                let attached = conn.attach(&mut client, &catalog)?;
                let spec = BindSpec::table(&function).in_schema(&schema_name);
                let spec = match &raw_arguments {
                    Some(raw) => spec.with_raw_arguments(raw.clone()),
                    None => spec.with_arguments(arguments.clone()),
                };
                let bound = client.bind(&attached, &spec).map_err(to_df)?;

                let opts = PlanOptions {
                    projection,
                    pushdown_filters: pushdown.blob,
                    // Columns the filter reads but the projection may omit. The
                    // worker keys a pushed filter by its position in what it emits,
                    // so without these a filter on an unprojected column evaluates
                    // against whichever column lands in that slot instead — wrong
                    // rows, silently.
                    filter_columns: Some(pushdown.columns),
                    // The parallelism FLOOR: a small but expensive table still needs
                    // one reader per partition, which a byte target alone would
                    // never give it. No byte target is sent — this provider has no
                    // basis to invent one, and sizing belongs where the knowledge
                    // is, which is the worker.
                    min_splits: Some(target_partitions as i64),
                    // Push the FULL limit into every split. Over-production is legal
                    // and the engine re-applies the limit above the coalesce, while
                    // dividing by N would under-produce under skew.
                    row_limit: limit.map(|l| l as i64),
                    ..Default::default()
                };
                match client.plan(&bound, &opts) {
                    Ok(plan) => Ok(Some(plan)),
                    // ONLY "this worker has no such method" is the pre-splits path.
                    // Swallowing every error meant a transport failure, an auth
                    // failure, or the page-cap refusal (which VgiClient::plan raises
                    // precisely so a partial enumeration is never scanned) all
                    // silently became a serial full scan — with the diagnostic
                    // discarded, and for a split-only worker an error pointing at a
                    // setting this engine does not have.
                    Err(e) if e.error_type == "MethodNotImplementedError" => Ok(None),
                    Err(e) => {
                        // A connection that saw an error must not go back to the pool:
                        // recycling one that failed mid-protocol hands the next caller
                        // a worker in an unknown state. The swallow above is what made
                        // that reachable — before it, the error propagated and took the
                        // query with it.
                        client.poison();
                        Err(to_df(e))
                    }
                }
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let Some(plan) = plan else {
            return Ok(None);
        };

        // The framework default for a worker that never opted in is one split
        // carrying nothing at all. Treat that as no plan, so such a worker keeps
        // the exact behaviour it had.
        if plan.splits.len() == 1
            && plan.splits[0].token.is_empty()
            && plan.splits[0].estimated_rows.is_none()
            && plan.splits[0].estimated_bytes.is_none()
        {
            return Ok(None);
        }

        // An UNBOUNDED split is a shard read forever. A DataFusion task must
        // terminate, and the plan below declares Boundedness::Bounded — which
        // frees the optimizer to put a SortExec or a hash aggregate above the
        // scan, buffering a stream that never ends. So refuse, rather than plan
        // a query that hangs with no error and no memory ceiling.
        //
        // The predicate needs BOTH positions: `end_position: None` alone is also
        // the default for every ordinary batch split, so refusing on that would
        // reject every worker. A split that names where it starts and not where
        // it stops is the one making an unbounded claim.
        if let Some(unbounded) = plan
            .splits
            .iter()
            .position(|s| s.start_position.is_some() && s.end_position.is_none())
        {
            return Err(DataFusionError::NotImplemented(format!(
                "VGI function '{}' planned split {} as unbounded (it names a start position \
                 and no end). This engine's tasks must terminate, so an endless split cannot \
                 be scheduled.",
                self.function, unbounded
            )));
        }
        if plan.start_position.is_some() && plan.end_position.is_none() {
            return Err(DataFusionError::NotImplemented(format!(
                "VGI function '{}' returned an unbounded plan range (it names a start position \
                 and no resolved end frontier). This engine's tasks must terminate.",
                self.function
            )));
        }

        let estimated_total_rows = plan.estimated_total_rows;
        let estimated_total_bytes = plan.estimated_total_bytes;
        let mut planned = pack_splits(plan, target_partitions);
        // Plan-level totals are advisory and first-page-wins. Prefer the sum of
        // split facts when every split supplied one (which also works for a
        // paginated plan whose first page omitted its eventual total), then
        // fall back to the plan estimate.
        if planned.num_rows == Precision::Absent {
            planned.num_rows = estimated_total_rows
                .and_then(|n| usize::try_from(n).ok())
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent);
        }
        if planned.total_byte_size == Precision::Absent {
            planned.total_byte_size = estimated_total_bytes
                .and_then(|n| usize::try_from(n).ok())
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent);
        }
        Ok(Some(planned))
    }
}

/// One DataFusion partition's split claims and their combined cardinality.
#[derive(Debug, Clone)]
struct SplitGroup {
    split_indices: Vec<usize>,
    tokens: Vec<Vec<u8>>,
    num_rows: Precision<usize>,
    total_byte_size: Precision<usize>,
}

/// The complete planned scan after split claims have been packed.
#[derive(Debug, Clone)]
struct PlannedSplits {
    groups: Vec<SplitGroup>,
    num_rows: Precision<usize>,
    total_byte_size: Precision<usize>,
    /// The complete VGI plan is retained so every redemption echoes its
    /// execution context and optimizer metadata remains available.
    plan: ScanPlan,
}

/// Pack splits into partition-sized groups, weighted by byte estimate.
///
/// The packing is what an engine whose `partition_count()` IS its concurrency
/// has to do: it cannot claim greedily, because the count is fixed before any
/// reading starts. So it approximates with longest-processing-time-first, which
/// is the standard heuristic and needs only a per-split weight.
///
/// `max_workers` is NORMATIVE and enforced here rather than left to the worker
/// to refuse with a 429: over-fanning is structural for this engine, so the cap
/// belongs at the only point that decides the fan-out.
///
/// A missing byte estimate degrades this to round-robin by count — which is
/// correct, just skew-blind. That is the documented cost of not populating
/// `estimated_bytes`, and it is why the field is described as load-bearing for
/// packing engines specifically.
fn pack_splits(plan: ScanPlan, target_partitions: usize) -> PlannedSplits {
    let splits = &plan.splits;
    if splits.is_empty() {
        return PlannedSplits {
            groups: Vec::new(),
            // A zero-split plan is the protocol's definitive "no work" result.
            num_rows: Precision::Exact(0),
            total_byte_size: Precision::Exact(0),
            plan,
        };
    }
    let cap = match plan.max_workers {
        Some(m) if m > 0 => (m as usize).min(target_partitions),
        _ => target_partitions,
    };
    let n = cap.min(splits.len()).max(1);

    // Longest-processing-time-first: sort descending by weight, then hand each
    // split to whichever bin is currently lightest.
    let mut order: Vec<usize> = (0..splits.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(splits[i].estimated_bytes.unwrap_or(0)));

    let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut weights: Vec<i64> = vec![0; n];
    for i in order {
        let (lightest, _) = weights
            .iter()
            .enumerate()
            .min_by_key(|(_, w)| **w)
            .expect("at least one bin");
        group_indices[lightest].push(i);
        weights[lightest] += splits[i].estimated_bytes.unwrap_or(1).max(1);
    }

    let num_rows = split_row_count(splits.iter());
    let total_byte_size = split_byte_count(splits.iter());
    let groups = group_indices
        .into_iter()
        .map(|indices| SplitGroup {
            num_rows: split_row_count(indices.iter().map(|&i| &splits[i])),
            total_byte_size: split_byte_count(indices.iter().map(|&i| &splits[i])),
            tokens: indices
                .iter()
                .copied()
                .map(|i| splits[i].token.clone())
                .collect(),
            split_indices: indices,
        })
        .collect();
    PlannedSplits {
        groups,
        num_rows,
        total_byte_size,
        plan,
    }
}

/// Sum split row counts without turning partial or malformed estimates into a
/// false total. Exactness is preserved only when every contributing split says
/// its count is exact.
fn split_row_count<'a>(splits: impl Iterator<Item = &'a ScanSplitInfo>) -> Precision<usize> {
    let mut total = 0usize;
    let mut exact = true;
    for split in splits {
        let Some(rows) = split
            .estimated_rows
            .and_then(|rows| usize::try_from(rows).ok())
        else {
            return Precision::Absent;
        };
        let Some(sum) = total.checked_add(rows) else {
            return Precision::Absent;
        };
        total = sum;
        exact &= split.rows_exact;
    }
    if exact {
        Precision::Exact(total)
    } else {
        Precision::Inexact(total)
    }
}

fn split_byte_count<'a>(splits: impl Iterator<Item = &'a ScanSplitInfo>) -> Precision<usize> {
    let mut total = 0usize;
    for split in splits {
        let Some(bytes) = split
            .estimated_bytes
            .and_then(|bytes| usize::try_from(bytes).ok())
        else {
            return Precision::Absent;
        };
        let Some(sum) = total.checked_add(bytes) else {
            return Precision::Absent;
        };
        total = sum;
    }
    Precision::Inexact(total)
}

#[async_trait]
impl TableProvider for VgiTableProvider {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Every filter is reported `Inexact`, so DataFusion re-applies it above the
    /// scan and pushdown stays a pure optimisation — a worker that ignores or
    /// mis-applies the blob still yields correct rows. Claiming `Exact` would
    /// make every translation decision load-bearing for correctness in exchange
    /// for saving a local filter pass.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<datafusion::logical_expr::TableProviderFilterPushDown>> {
        use datafusion::logical_expr::TableProviderFilterPushDown;
        Ok(filters
            .iter()
            .map(|f| {
                if filters::is_pushable(f, &self.output_schema) {
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let projected = match projection {
            None => self.output_schema.clone(),
            Some(p) => Arc::new(self.output_schema.project(p)?),
        };
        let projection_ids: Option<Vec<i64>> =
            projection.map(|p| p.iter().map(|i| *i as i64).collect());
        let pushed_projection = if self.projection_pushdown {
            projection_ids.clone()
        } else {
            None
        };
        let pushdown = filters::serialize(filters, &self.output_schema)?;

        // Divide the scan into named splits, then bin-pack them into partitions.
        //
        // Unlike an engine that hands out a fixed reader count and lets readers
        // claim greedily, DataFusion's partition_count() IS its concurrency and
        // is fixed at planning time — so the packing has to happen here, and
        // max_workers has to be enforced here rather than relying on the worker
        // to push back with a 429.
        let target_partitions = state.config_options().execution.target_partitions.max(1);
        let split_groups = self
            .plan_splits(
                pushed_projection,
                pushdown.clone(),
                limit,
                target_partitions,
            )
            .await?;
        let partitions = match &split_groups {
            // No plan: today's behaviour, one partition doing the work.
            None => self.max_workers,
            // A plan with no splits is legal and means "no work" — but it must
            // clamp to ONE (empty) partition, because UnknownPartitioning(0)
            // makes CoalescePartitionsExec fail outright and partition
            // statistics assert on the index.
            Some(plan) => plan.groups.len().max(1),
        };

        Ok(Arc::new(VgiScanExec::new(
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
            self.arguments.clone(),
            self.raw_arguments.clone(),
            projection_ids,
            self.projection_pushdown,
            pushdown,
            limit,
            projected,
            partitions,
            split_groups,
        )))
    }
}

/// The physical scan.
#[derive(Debug)]
pub struct VgiScanExec {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    /// The bind arguments of the call this scan came from. `init` echoes the
    /// bind back, so re-binding without them would be a different call.
    arguments: Arguments,
    /// A catalog table's worker-supplied argument bytes, forwarded verbatim.
    raw_arguments: Option<vgi_client::Bytes>,
    projection: Option<Vec<i64>>,
    projection_pushdown: bool,
    pushdown: filters::Pushdown,
    limit: Option<usize>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    /// One group of split tokens per partition, or `None` when this scan is not
    /// split-capable and keeps the pre-splits single-reader path.
    ///
    /// The groups are decided at planning time because this engine's partition
    /// count IS its concurrency — there is no equivalent of a reader claiming
    /// its next unit of work mid-scan.
    split_groups: Option<PlannedSplits>,
    /// Statistics supplied by split planning, for the whole scan and for each
    /// physical partition after bin-packing.
    statistics: Arc<Statistics>,
    partition_statistics: Vec<Arc<Statistics>>,
}

impl VgiScanExec {
    #[allow(clippy::too_many_arguments)]
    fn new(
        conn: VgiConnection,
        catalog: String,
        schema_name: String,
        function: String,
        arguments: Arguments,
        raw_arguments: Option<vgi_client::Bytes>,
        projection: Option<Vec<i64>>,
        projection_pushdown: bool,
        pushdown: filters::Pushdown,
        limit: Option<usize>,
        schema: SchemaRef,
        partitions: usize,
        split_groups: Option<PlannedSplits>,
    ) -> Self {
        let equivalence = split_groups
            .as_ref()
            .and_then(|planned| split_ordering(&schema, planned))
            .map(|ordering| EquivalenceProperties::new_with_orderings(schema.clone(), [ordering]))
            .unwrap_or_else(|| EquivalenceProperties::new(schema.clone()));
        let properties = Arc::new(PlanProperties::new(
            equivalence,
            // One partition per connection the worker will accept. VGI hands
            // each connection a disjoint slice, so the partitions really are
            // independent.
            Partitioning::UnknownPartitioning(partitions.max(1)),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let (statistics, partition_statistics) = match &split_groups {
            Some(plan) => {
                let overall = Arc::new(statistics_for_splits(
                    &schema,
                    &plan.plan,
                    0..plan.plan.splits.len(),
                    plan.num_rows,
                    plan.total_byte_size,
                ));
                let partitions = if plan.groups.is_empty() {
                    vec![Arc::clone(&overall)]
                } else {
                    plan.groups
                        .iter()
                        .map(|group| {
                            Arc::new(statistics_for_splits(
                                &schema,
                                &plan.plan,
                                group.split_indices.iter().copied(),
                                group.num_rows,
                                group.total_byte_size,
                            ))
                        })
                        .collect()
                };
                (overall, partitions)
            }
            None => {
                let unknown = Arc::new(Statistics::new_unknown(&schema));
                (Arc::clone(&unknown), vec![unknown; partitions.max(1)])
            }
        };
        Self {
            conn,
            catalog,
            schema_name,
            function,
            arguments,
            raw_arguments,
            projection,
            projection_pushdown,
            pushdown,
            limit,
            schema,
            properties,
            split_groups,
            statistics,
            partition_statistics,
        }
    }
}

fn split_ordering(schema: &SchemaRef, planned: &PlannedSplits) -> Option<Vec<PhysicalSortExpr>> {
    // VGI's claim is within one split. Concatenating two independently sorted
    // runs does not make the DataFusion partition sorted, so bin-packing clears
    // the claim even when every source split names the same ordering.
    if planned.groups.is_empty()
        || planned
            .groups
            .iter()
            .any(|group| group.split_indices.len() != 1)
    {
        return None;
    }
    let fields = planned.plan.sort_order.as_ref()?;
    if fields.is_empty() {
        return None;
    }
    fields
        .iter()
        .map(|field| {
            let index = schema.index_of(&field.column).ok()?;
            let options = datafusion::arrow::compute::SortOptions {
                descending: field.direction.0 == "desc",
                nulls_first: field.nulls.0 == "nulls_first",
            };
            Some(PhysicalSortExpr::new(
                Arc::new(Column::new(&field.column, index)),
                options,
            ))
        })
        .collect()
}

fn statistics_with_estimates(
    schema: &SchemaRef,
    num_rows: Precision<usize>,
    total_byte_size: Precision<usize>,
) -> Statistics {
    let mut statistics = Statistics::new_unknown(schema);
    statistics.num_rows = num_rows;
    statistics.total_byte_size = total_byte_size;
    statistics
}

fn bool_at(
    batch: &datafusion::arrow::record_batch::RecordBatch,
    name: &str,
    row: usize,
) -> Option<bool> {
    let array = batch.column_by_name(name)?;
    let array = array.as_any().downcast_ref::<BooleanArray>()?;
    (!array.is_null(row)).then(|| array.value(row))
}

fn i64_at(
    batch: &datafusion::arrow::record_batch::RecordBatch,
    name: &str,
    row: usize,
) -> Option<i64> {
    let array = batch.column_by_name(name)?;
    let array = array.as_any().downcast_ref::<Int64Array>()?;
    (!array.is_null(row)).then(|| array.value(row))
}

fn statistic_value(
    batch: &datafusion::arrow::record_batch::RecordBatch,
    name: &str,
    row: usize,
    target: &DataType,
) -> Option<ScalarValue> {
    let union = batch
        .column_by_name(name)?
        .as_any()
        .downcast_ref::<UnionArray>()?;
    let type_id = union.type_id(row);
    let offset = union.value_offset(row);
    let child = union.child(type_id);
    if child.is_null(offset) {
        return None;
    }
    ScalarValue::try_from_array(child.as_ref(), offset)
        .ok()?
        .cast_to(target)
        .ok()
}

/// Convert one VGI split's Arrow optimizer metadata into DataFusion's shape.
///
/// Worker column statistics have no explicit exactness bit, so their values are
/// conservative `Inexact`. Partition bounds are the exact `(min, max)` range
/// named by the split and therefore replace those two fields with `Exact`.
fn statistics_for_split(schema: &SchemaRef, split: &ScanSplitInfo) -> Statistics {
    let mut statistics = statistics_with_estimates(
        schema,
        split_row_count(std::iter::once(split)),
        split_byte_count(std::iter::once(split)),
    );
    let mut columns: HashMap<String, ColumnStatistics> = HashMap::new();

    if let Some(batch) = &split.column_statistics {
        if let Some(names) = batch
            .column_by_name("column_name")
            .and_then(|array| array.as_any().downcast_ref::<StringArray>())
        {
            for row in 0..batch.num_rows() {
                if names.is_null(row) {
                    continue;
                }
                let name = names.value(row);
                let Some(field) = schema.field_with_name(name).ok() else {
                    continue;
                };
                let num_rows = statistics.num_rows;
                let null_count = match (
                    bool_at(batch, "has_null", row),
                    bool_at(batch, "has_not_null", row),
                ) {
                    (Some(false), _) => Precision::Exact(0),
                    (_, Some(false)) => num_rows,
                    _ => Precision::Absent,
                };
                let distinct_count = i64_at(batch, "distinct_count", row)
                    .and_then(|value| usize::try_from(value).ok())
                    .map(Precision::Inexact)
                    .unwrap_or(Precision::Absent);
                let min_value = statistic_value(batch, "min", row, field.data_type())
                    .map(Precision::Inexact)
                    .unwrap_or(Precision::Absent);
                let max_value = statistic_value(batch, "max", row, field.data_type())
                    .map(Precision::Inexact)
                    .unwrap_or(Precision::Absent);
                columns.insert(
                    name.to_string(),
                    ColumnStatistics::new_unknown()
                        .with_null_count(null_count)
                        .with_distinct_count(distinct_count)
                        .with_min_value(min_value)
                        .with_max_value(max_value),
                );
            }
        }
    }

    if let Some(bounds) = &split.partition_bounds {
        for field in bounds.schema().fields() {
            let Some(output) = schema.field_with_name(field.name()).ok() else {
                continue;
            };
            let Some(array) = bounds.column_by_name(field.name()) else {
                continue;
            };
            let min = ScalarValue::try_from_array(array.as_ref(), 0)
                .and_then(|value| value.cast_to(output.data_type()))
                .ok();
            let max = ScalarValue::try_from_array(array.as_ref(), 1)
                .and_then(|value| value.cast_to(output.data_type()))
                .ok();
            let column = columns
                .entry(field.name().clone())
                .or_insert_with(ColumnStatistics::new_unknown);
            column.min_value = min
                .clone()
                .map(Precision::Exact)
                .unwrap_or(Precision::Absent);
            column.max_value = max
                .clone()
                .map(Precision::Exact)
                .unwrap_or(Precision::Absent);
            if min.as_ref().is_some_and(|min| !min.is_null()) && min == max {
                column.distinct_count = Precision::Exact(1);
            }
        }
    }

    statistics.column_statistics = schema
        .fields()
        .iter()
        .map(|field| {
            columns
                .remove(field.name())
                .unwrap_or_else(ColumnStatistics::new_unknown)
        })
        .collect();
    statistics
}

fn statistics_for_splits(
    schema: &SchemaRef,
    plan: &ScanPlan,
    indices: impl IntoIterator<Item = usize>,
    num_rows: Precision<usize>,
    total_byte_size: Precision<usize>,
) -> Statistics {
    let split_statistics: Vec<_> = indices
        .into_iter()
        .filter_map(|index| plan.splits.get(index))
        .map(|split| statistics_for_split(schema, split))
        .collect();
    let mut statistics = Statistics::try_merge_iter(split_statistics.iter(), schema)
        .unwrap_or_else(|_| Statistics::new_unknown(schema));
    statistics.num_rows = num_rows;
    statistics.total_byte_size = total_byte_size;
    statistics
}

impl DisplayAs for VgiScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VgiScanExec: worker={}, function={}.{}",
            self.conn.label(),
            self.schema_name,
            self.function
        )?;
        if self.projection.is_some() {
            write!(f, ", projected")?;
        }
        if let Some(l) = self.limit {
            write!(f, ", limit={l}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for VgiScanExec {
    fn name(&self) -> &str {
        "VgiScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        ) -> DFResult<datafusion::common::tree_node::TreeNodeRecursion>,
    ) -> DFResult<datafusion::common::tree_node::TreeNodeRecursion> {
        // A leaf with no physical expressions of its own: pushdown is expressed
        // on the wire to the worker, not as expressions in the plan.
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    }

    fn statistics_from_inputs(
        &self,
        _input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> DFResult<Arc<Statistics>> {
        Ok(match args.partition() {
            Some(partition) => self
                .partition_statistics
                .get(partition)
                .cloned()
                .unwrap_or_else(|| Arc::new(Statistics::new_unknown(&self.schema))),
            None => Arc::clone(&self.statistics),
        })
    }

    fn execute(
        &self,
        partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arguments = self.arguments.clone();
        let raw_arguments = self.raw_arguments.clone();
        let projection = self.projection.clone();
        let projection_pushdown = self.projection_pushdown;
        let pushdown = self.pushdown.clone();
        let limit = self.limit;
        let out_schema = self.schema.clone();
        // The tokens this partition redeems. An empty group is legal — a plan
        // can pack fewer splits than partitions — and reads as no work.
        let split_tokens: Option<Vec<Vec<u8>>> = self.split_groups.as_ref().map(|plan| {
            plan.groups
                .get(partition)
                .map(|group| group.tokens.clone())
                .unwrap_or_default()
        });
        let split_plan = self
            .split_groups
            .as_ref()
            .map(|planned| planned.plan.clone());
        // One handle for the blocking scan, one for the stream adapter.
        let scan_schema = self.schema.clone();

        // A bounded channel so a slow consumer applies backpressure to the
        // worker rather than letting batches pile up in memory.
        let (tx, rx) =
            tokio::sync::mpsc::channel::<DFResult<datafusion::arrow::array::RecordBatch>>(2);

        tokio::task::spawn_blocking(move || {
            let run = || -> DFResult<()> {
                let mut client = conn.connect()?;
                let attached = conn.attach(&mut client, &catalog)?;
                // A catalog table carries the worker's own argument bytes;
                // anything else carries the call's typed arguments. Either way
                // the scan must re-send exactly what the plan bound with —
                // `init` echoes the bind call back.
                let spec = BindSpec::table(&function).in_schema(&schema_name);
                let spec = match &raw_arguments {
                    Some(raw) => spec.with_raw_arguments(raw.clone()),
                    None => spec.with_arguments(arguments.clone()),
                };
                let bound = client.bind(&attached, &spec).map_err(to_df)?;

                // An EMPTY projection is `count(*)`: DataFusion wants row counts
                // and no columns. `[]` cannot be sent as-is — "no columns" reads
                // like "unset" on the wire — but asking for *everything* and
                // discarding it would drag every column across for a query that
                // needs none. Ask for the single narrowest column instead, then
                // drop it locally: one column on the wire rather than all of
                // them, and the row count is preserved either way.
                let push = match (&projection, projection_pushdown) {
                    (Some(p), true) if p.is_empty() => narrowest_column(&bound).map(|i| vec![i]),
                    (other, true) => other.clone(),
                    (_, false) => None,
                };
                // A split scan is genuinely parallel: each partition redeems its
                // OWN tokens, so no partition has to learn another's execution
                // id. That rendezvous is exactly what splits remove — a token
                // names its work, so any process can redeem it independently.
                //
                // Without a plan we fall back to the pre-splits behaviour, where
                // partition 0 does the work and the rest are empty: correct, but
                // not parallel, because joining an existing execution WOULD need
                // the shared id.
                if let Some(tokens) = &split_tokens {
                    if tokens.is_empty() {
                        return Ok(());
                    }
                } else if partition > 0 {
                    return Ok(());
                }

                let opts = ScanOptions {
                    projection: push,
                    pushdown_filters: pushdown.blob.clone(),
                    // See PlanOptions above: the filter's columns must be
                    // requested even when the projection omits them, or the
                    // worker evaluates the predicate against the wrong column.
                    filter_columns: Some(pushdown.columns.clone()),
                    row_limit: limit.map(|l| l as i64),
                    ..Default::default()
                };
                let opts = match (&split_plan, split_tokens.clone()) {
                    (Some(plan), Some(tokens)) => plan.redemption_options(tokens, opts),
                    _ => opts,
                };
                let mut scan = client.scan(&bound, &opts).map_err(to_df)?;

                let mut emitted = 0usize;
                while let Some(batch) = scan.next_batch().map_err(to_df)? {
                    // DataFusion requires every batch to match the declared
                    // schema exactly, so conform rather than trusting the
                    // worker to have honoured the projection.
                    let batch = conform(batch, &scan_schema)?;
                    let batch = match limit {
                        Some(l) if emitted + batch.num_rows() > l => {
                            batch.slice(0, l.saturating_sub(emitted))
                        }
                        _ => batch,
                    };
                    emitted += batch.num_rows();
                    if tx.blocking_send(Ok(batch)).is_err() {
                        // Receiver dropped: the query was cancelled.
                        let _ = scan.cancel();
                        return Ok(());
                    }
                    if limit.is_some_and(|l| emitted >= l) {
                        let _ = scan.cancel();
                        break;
                    }
                }
                Ok(())
            };
            if let Err(e) = run() {
                let _ = tx.blocking_send(Err(e));
            }
        });

        let stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(out_schema, stream)))
    }
}
