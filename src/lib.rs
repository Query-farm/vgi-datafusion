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
//! projection, filters and limit all ride the same `scan` call VGI already
//! pushes down. Exchange-mode functions do not, because DataFusion resolves
//! table-function arguments against an empty schema and so cannot express a
//! table function that takes rows. See the feasibility study for the routes
//! around that.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use vgi_client::{
    Arguments, AttachOptions, BindSpec, PooledClient, ScanOptions, VgiClient, VgiLocation,
    WorkerPool,
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
    label: String,
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
        Self {
            label: location.label(),
            location,
            pool,
            attached: Arc::new(Mutex::new(HashMap::new())),
        }
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
        let handle = client
            .attach(catalog, AttachOptions::default())
            .map_err(to_df)?;
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
        self.pool.acquire(&self.location).map_err(to_df)
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

/// Make a batch match the schema its plan declared.
///
/// Two cases matter. A zero-column schema is `count(*)`: build an empty batch
/// that still carries the row count, since dropping the count would lose the
/// answer. Otherwise take the leading columns, which is what a positional
/// projection means.
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
        schema_name: impl Into<String>,
        info: vgi_client::TableInfo,
    ) -> DFResult<Arc<Self>> {
        let catalog = catalog.into();
        let schema_name = schema_name.into();
        let c = conn.clone();
        let cat2 = catalog.clone();

        let (function, function_schema, arguments, output_schema) =
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
                            return Ok::<_, DataFusionError>((
                                scan.function_name,
                                schema.clone(),
                                arguments,
                                bound.output_schema().clone(),
                            ))
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
        drop(client);

        Ok(Arc::new(Self {
            conn,
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            function: function.to_string(),
            arguments,
            raw_arguments: None,
            output_schema,
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

        let (output_schema, max_workers) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = c.attach(&mut client, &cat2)?;
            let spec = BindSpec::table(&fn2).in_schema(&sch2).with_arguments(args2);
            let bound = client.bind(&attached, &spec).map_err(to_df)?;
            let schema = bound.output_schema().clone();
            // One partition, deliberately.
            //
            // `max_workers` is only readable from a scan's header, so learning
            // it here would mean opening a scan and abandoning it — which both
            // costs a scan per bind and leaves the connection mid-stream, so
            // the pool would hand the next caller a broken one. (It did: binds
            // failed with "empty IPC stream (no schema)" until this came out.)
            //
            // And it would buy nothing: only partition 0 reads today, so
            // reporting the worker's `max_workers` would advertise a
            // parallelism this operator does not have. When the fan-out lands
            // it needs the execution id shared across partitions anyway, which
            // is a rendezvous at `scan` time rather than a number at bind time.
            Ok::<_, DataFusionError>((schema, 1usize))
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
            max_workers,
        }))
    }
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
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let projected = match projection {
            None => self.output_schema.clone(),
            Some(p) => Arc::new(self.output_schema.project(p)?),
        };
        Ok(Arc::new(VgiScanExec::new(
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
            self.arguments.clone(),
            self.raw_arguments.clone(),
            projection.map(|p| p.iter().map(|i| *i as i64).collect()),
            filters::serialize(filters, &self.output_schema)?,
            limit,
            projected,
            self.max_workers,
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
    pushdown_filters: Option<Vec<u8>>,
    limit: Option<usize>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
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
        pushdown_filters: Option<Vec<u8>>,
        limit: Option<usize>,
        schema: SchemaRef,
        partitions: usize,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            // One partition per connection the worker will accept. VGI hands
            // each connection a disjoint slice, so the partitions really are
            // independent.
            Partitioning::UnknownPartitioning(partitions.max(1)),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            conn,
            catalog,
            schema_name,
            function,
            arguments,
            raw_arguments,
            projection,
            pushdown_filters,
            limit,
            schema,
            properties,
        }
    }
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
        let pushdown_filters = self.pushdown_filters.clone();
        let limit = self.limit;
        let out_schema = self.schema.clone();
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
                let push = match &projection {
                    Some(p) if p.is_empty() => narrowest_column(&bound).map(|i| vec![i]),
                    other => other.clone(),
                };
                let opts = ScanOptions {
                    projection: push,
                    pushdown_filters: pushdown_filters.clone(),
                    ..Default::default()
                };
                let mut scan = client.scan(&bound, &opts).map_err(to_df)?;

                // Partition 0 opens the execution; later partitions would join
                // it with its execution id. Sharing that id across partitions
                // needs a rendezvous this adapter does not yet have, so for now
                // partition 0 does the work and the rest are empty — correct,
                // just not yet parallel. See the README.
                if partition > 0 {
                    return Ok(());
                }

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
