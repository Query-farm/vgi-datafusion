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

use std::fmt;
use std::sync::Arc;

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
use vgi_client::{AttachOptions, BindSpec, ScanOptions, VgiClient};

mod catalog;
mod filters;
mod scalar;
mod session;

pub use catalog::{VgiCatalogProvider, VgiSchemaProvider};
pub use scalar::VgiScalarUdf;
pub use session::{sql, AttachSpec};

/// How to reach a VGI worker.
///
/// A factory rather than a connection: each scan partition opens its own, which
/// is how a VGI scan fans out across the worker's advertised `max_workers`.
#[derive(Clone)]
pub struct VgiConnection {
    make: Arc<dyn Fn() -> vgi_client::Result<VgiClient> + Send + Sync>,
    label: String,
}

impl fmt::Debug for VgiConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VgiConnection")
            .field("label", &self.label)
            .finish()
    }
}

impl VgiConnection {
    /// Spawn a worker as a child process.
    pub fn subprocess<I, S>(cmd: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let argv: Vec<String> = cmd.into_iter().map(Into::into).collect();
        let label = argv.first().cloned().unwrap_or_else(|| "worker".into());
        Self {
            make: Arc::new(move || VgiClient::connect_subprocess(&argv)),
            label,
        }
    }

    /// Talk to a worker serving VGI over HTTP.
    pub fn http(base_url: impl Into<String>) -> Self {
        let url = base_url.into();
        let label = url.clone();
        Self {
            make: Arc::new(move || VgiClient::connect_http(&url)),
            label,
        }
    }

    /// Build a connection from any factory.
    pub fn from_fn(
        label: impl Into<String>,
        make: impl Fn() -> vgi_client::Result<VgiClient> + Send + Sync + 'static,
    ) -> Self {
        Self {
            make: Arc::new(make),
            label: label.into(),
        }
    }

    /// Open one connection.
    pub fn connect(&self) -> DFResult<VgiClient> {
        (self.make)().map_err(to_df)
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
        return Ok(RecordBatch::try_new_with_options(
            schema.clone(),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?);
    }
    let want = schema.fields().len();
    if batch.num_columns() < want {
        return Err(DataFusionError::Execution(format!(
            "worker emitted {} columns but the plan declared {want}",
            batch.num_columns()
        )));
    }
    Ok(RecordBatch::try_new(
        schema.clone(),
        batch.columns()[..want].to_vec(),
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
        let catalog = catalog.into();
        let schema_name = schema_name.into();
        let function = function.into();
        let c = conn.clone();
        let (cat2, sch2, fn2) = (catalog.clone(), schema_name.clone(), function.clone());

        let (output_schema, max_workers) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = client
                .attach(&cat2, AttachOptions::default())
                .map_err(to_df)?;
            let spec = BindSpec::table(&fn2).in_schema(&sch2);
            let bound = client.bind(&attached, &spec).map_err(to_df)?;
            let schema = bound.output_schema().clone();
            // `max_workers` is only known once a scan opens, so ask now and
            // throw the stream away — a cheap probe that lets `scan` report a
            // truthful partition count during planning.
            let workers = client
                .scan(&bound, &ScanOptions::default())
                .map(|s| s.max_workers().max(1) as usize)
                .unwrap_or(1);
            Ok::<_, DataFusionError>((schema, workers))
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        Ok(Arc::new(Self {
            conn,
            catalog,
            schema_name,
            function,
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
                let attached = client
                    .attach(&catalog, AttachOptions::default())
                    .map_err(to_df)?;
                let spec = BindSpec::table(&function).in_schema(&schema_name);
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
