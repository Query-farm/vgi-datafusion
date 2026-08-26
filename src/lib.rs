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

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use datafusion::arrow::array::{Array, BooleanArray, Int64Array, StringArray, UnionArray};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::pruning::PrunableStatistics;
use datafusion::common::stats::Precision;
use datafusion::common::{
    ColumnStatistics, Constraint, Constraints, DFSchema, DataFusionError, Result as DFResult,
    ScalarValue, Statistics,
};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::{
    expressions::{Column, DynamicFilterPhysicalExpr},
    EquivalenceProperties, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_optimizer::pruning::PruningPredicateBuilder;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricCategory, MetricsSet,
    RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, StatisticsArgs,
};
use futures::StreamExt;
use vgi_client::{
    Arguments, AttachOptions, AttachedCatalog, BindSpec, FunctionKind, PlanOptions, PooledClient,
    ScanOptions, ScanPlan, ScanSplitInfo, VgiClient, VgiLocation, WorkerPool,
};

mod aggregate;
mod catalog;
mod diagnostics;
mod filters;
mod runtime;
mod scalar;
mod session;
mod settings;
mod table_function;
mod table_input;

pub use aggregate::VgiAggregateUdf;
pub use catalog::{VgiCatalogProvider, VgiSchemaProvider};
pub use runtime::{
    ExchangeCacheStats, PlanCacheStats, VgiEvent, VgiEventSink, VgiLocalityHook, VgiResolvedSecret,
    VgiRuntime, VgiSecretResolver, VgiSessionOptions, VgiSplitLocality,
};
pub use scalar::VgiScalarUdf;
pub use session::{sql, AttachSpec};
pub use settings::VgiSettings;
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
    /// Session services shared by every clone and scan partition.
    runtime: Arc<VgiRuntime>,
    /// Attachment-level veto for worker-opted-in result caching.
    cache_enabled: bool,
    /// Explicit opt-in for remote workers to nominate client-local format paths.
    allow_local_format_paths: bool,
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
            .field("cache_enabled", &self.cache_enabled)
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
            runtime: Arc::new(VgiRuntime::default()),
            cache_enabled: true,
            allow_local_format_paths: false,
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

    /// Use services owned by the containing DataFusion session.
    #[must_use]
    pub fn with_runtime(mut self, runtime: Arc<VgiRuntime>) -> Self {
        self.connection_options.rpc_timeout = runtime.options().rpc_timeout;
        self.runtime = runtime;
        self
    }

    /// Enable or veto worker-controlled caching for this attachment.
    #[must_use]
    pub fn with_cache_enabled(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Permit a remote worker to nominate paths on DataFusion's local filesystem.
    ///
    /// Local subprocess, Unix-socket, launcher, and loopback network workers
    /// are trusted this way automatically. Other HTTP/TCP callers must opt in
    /// because a remote catalog could otherwise turn discovery into arbitrary
    /// local-file reads.
    #[must_use]
    pub fn with_local_format_paths(mut self, enabled: bool) -> Self {
        self.allow_local_format_paths = enabled;
        self
    }

    /// Session services used by this connection.
    pub fn runtime(&self) -> &Arc<VgiRuntime> {
        &self.runtime
    }

    /// Clone this attachment for metadata RPCs without retaining its owning
    /// session runtime through that runtime's metadata registry.
    ///
    /// Pool, authentication, attach handles, options, and timeout settings are
    /// still shared. Only the event/cache runtime is replaced, breaking the
    /// otherwise circular `runtime -> metadata -> connection -> runtime` link.
    pub(crate) fn metadata_connection(&self) -> Self {
        let mut connection = self.clone();
        connection.runtime = Arc::new(VgiRuntime::default());
        connection
    }

    fn cache_identity_scope(&self, catalog: &str) -> Option<String> {
        let identity = self
            .auth
            .as_ref()
            .map(|auth| auth.identity())
            .unwrap_or(vgi_client::auth::Identity::Anonymous);
        vgi_client::auth::identity_scope(catalog, &identity, b"vgi-datafusion-result-cache:v1")
    }

    fn cache_attach_context(&self, catalog: &str) -> Vec<u8> {
        let Some(options) = self.attach_options.get(catalog) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut push = |bytes: &[u8]| {
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        };
        push(
            options
                .options
                .as_ref()
                .map(|value| value.0.as_slice())
                .unwrap_or_default(),
        );
        push(
            options
                .data_version_spec
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
        push(
            options
                .implementation_version
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
        out
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

    /// Deadline inherited from the session or overridden by this attachment.
    ///
    /// Cache followers are part of the attachment's RPC path, so they must use
    /// the same effective deadline as the connection they are waiting behind.
    pub(crate) fn rpc_timeout(&self) -> Option<std::time::Duration> {
        self.connection_options.rpc_timeout
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

    /// Encode the session's configured `vgi.*` values using this worker's
    /// attach-time Arrow setting declarations.
    pub(crate) fn settings_for(
        &self,
        attached: &vgi_client::AttachedCatalog,
    ) -> DFResult<Option<vgi_client::Bytes>> {
        let declarations = vgi_client::decode_setting_specs(attached.info()).map_err(to_df)?;
        crate::settings::encode_settings(&self.runtime.session_settings(), &declarations)
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
                .acquire_with_auth(
                    &self.location,
                    Arc::clone(auth),
                    self.connection_options.clone(),
                )
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

    /// Whether a worker location is local enough to nominate host filesystem
    /// paths for native format branches.
    ///
    /// A non-loopback HTTP/TCP catalog may still nominate object-store URLs
    /// already configured in DataFusion's runtime, but it must not turn an
    /// attach into arbitrary reads from the client's local filesystem.
    pub(crate) fn allows_local_format_paths(&self) -> bool {
        self.allow_local_format_paths
            || matches!(
                &self.location,
                VgiLocation::Subprocess(_) | VgiLocation::Unix(_) | VgiLocation::Launch(_)
            )
            || match &self.location {
                VgiLocation::Http(url) => http_host(url).is_some_and(host_is_loopback),
                VgiLocation::Tcp { host, .. } => host_is_loopback(host),
                _ => false,
            }
    }
}

fn http_host(url: &str) -> Option<&str> {
    let (_, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?.rsplit('@').next()?;
    if let Some(bracketed) = authority.strip_prefix('[') {
        return bracketed.split_once(']').map(|(host, _)| host);
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if port.bytes().all(|byte| byte.is_ascii_digit()) => Some(host),
        _ => Some(authority),
    }
}

fn host_is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
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
    filters_exactly_applied: bool,
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
        filters_exactly_applied: matches.iter().all(|info| info.filters_exactly_applied),
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
    column_mapping: Option<&HashMap<String, String>>,
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
            let worker_name = column_mapping
                .and_then(|mapping| mapping.get(want.name()))
                .unwrap_or_else(|| want.name());
            batch.column_by_name(worker_name).cloned().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "worker returned no column named `{worker_name}` for catalog column `{}`; it emitted [{}]",
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

pub(crate) fn resolve_secret_batch(
    runtime: &Arc<VgiRuntime>,
    requests: Vec<vgi_client::SecretLookupRequest>,
) -> DFResult<Vec<u8>> {
    use datafusion::arrow::array::{ArrayRef, RecordBatch, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};

    let resolver = runtime.secret_resolver().cloned().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "VGI worker requested {} secret(s), but this session has no VgiSecretResolver",
            requests.len()
        ))
    })?;
    // Planner-facing DataFusion extension points are synchronous. Resolve on a
    // dedicated thread/runtime so an async host resolver never nests or blocks
    // the query runtime's executor.
    let resolved = std::thread::Builder::new()
        .name("vgi-secret-resolver".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            runtime.block_on(async move {
                let mut out = Vec::with_capacity(requests.len());
                for request in requests {
                    let secret = resolver
                        .resolve(
                            &request.secret_type,
                            request.scope.as_deref(),
                            request.name.as_deref(),
                        )
                        .await?
                        .ok_or_else(|| {
                            DataFusionError::Plan(format!(
                                "no host secret matched VGI type `{}`{}{}",
                                request.secret_type,
                                request
                                    .scope
                                    .as_deref()
                                    .map(|scope| format!(", scope `{scope}`"))
                                    .unwrap_or_default(),
                                request
                                    .name
                                    .as_deref()
                                    .map(|name| format!(", name `{name}`"))
                                    .unwrap_or_default()
                            ))
                        })?;
                    out.push(secret);
                }
                Ok::<_, DataFusionError>(out)
            })
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?
        .join()
        .map_err(|_| DataFusionError::Execution("VGI secret resolver thread panicked".into()))??;

    let mut names = std::collections::HashSet::new();
    let mut outer_fields = Vec::with_capacity(resolved.len());
    let mut outer_arrays = Vec::with_capacity(resolved.len());
    for secret in resolved {
        if secret.name.is_empty() || !names.insert(secret.name.clone()) {
            return Err(DataFusionError::Plan(
                "VGI secret resolver returned an empty or duplicate secret name".into(),
            ));
        }
        let mut inner_fields = Vec::with_capacity(secret.fields.len());
        let mut inner_arrays = Vec::with_capacity(secret.fields.len());
        for (name, value) in secret.fields {
            inner_fields.push(Arc::new(Field::new(
                name,
                value.data_type(),
                value.is_null(),
            )));
            inner_arrays.push(value.to_array_of_size(1)?);
        }
        let fields = Fields::from(inner_fields);
        let array = StructArray::new(fields.clone(), inner_arrays, None);
        outer_fields.push(Arc::new(Field::new(
            secret.name,
            DataType::Struct(fields),
            false,
        )));
        outer_arrays.push(Arc::new(array) as ArrayRef);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(outer_fields)), outer_arrays)?;
    vgi_protocol::ipc::write_batch(&batch).map_err(to_df)
}

pub(crate) fn bind_with_secrets(
    conn: &VgiConnection,
    client: &mut VgiClient,
    attached: &AttachedCatalog,
    spec: &BindSpec,
) -> DFResult<vgi_client::BoundFunction> {
    bind_with_secrets_status(conn, client, attached, spec).map(|(bound, _)| bound)
}

/// Bind a producer function and report whether resolved secrets were supplied.
/// The bit is retained by its provider to suppress plan and result caching
/// across resolver rotations without placing secret bytes in cache identity.
pub(crate) fn bind_with_secrets_status(
    conn: &VgiConnection,
    client: &mut VgiClient,
    attached: &AttachedCatalog,
    spec: &BindSpec,
) -> DFResult<(vgi_client::BoundFunction, bool)> {
    let mut spec = spec.clone();
    if spec.settings.is_none() {
        spec.settings = conn.settings_for(attached)?;
    }
    let first = client.bind(attached, &spec).map_err(to_df)?;
    let requests = first.required_secrets();
    if requests.is_empty() {
        return Ok((first, false));
    }
    let secrets = resolve_secret_batch(conn.runtime(), requests)?;
    let second = client
        .bind_with_resolved_secrets(attached, &spec, secrets)
        .map_err(to_df)?;
    if !second.required_secret_types().is_empty() {
        return Err(DataFusionError::Execution(
            "VGI worker requested secrets twice; only one resolved retry is allowed".into(),
        ));
    }
    Ok((second, true))
}

pub(crate) fn bind_with_input_secrets(
    conn: &VgiConnection,
    client: &mut VgiClient,
    attached: &AttachedCatalog,
    spec: &BindSpec,
    input_schema: &datafusion::arrow::datatypes::Schema,
) -> DFResult<vgi_client::BoundFunction> {
    bind_with_input_secrets_status(conn, client, attached, spec, input_schema)
        .map(|(bound, _)| bound)
}

/// Bind an exchange function and report whether this bind consumed resolved
/// secret material. Result caching is disabled for those binds: resolvers may
/// rotate a secret while the principal remains unchanged, and secret bytes
/// must never be copied into a cache key merely to distinguish generations.
pub(crate) fn bind_with_input_secrets_status(
    conn: &VgiConnection,
    client: &mut VgiClient,
    attached: &AttachedCatalog,
    spec: &BindSpec,
    input_schema: &datafusion::arrow::datatypes::Schema,
) -> DFResult<(vgi_client::BoundFunction, bool)> {
    let mut spec = spec.clone();
    if spec.settings.is_none() {
        spec.settings = conn.settings_for(attached)?;
    }
    let first = client
        .bind_with_input(attached, &spec, input_schema)
        .map_err(to_df)?;
    let requests = first.required_secrets();
    if requests.is_empty() {
        return Ok((first, false));
    }
    let secrets = resolve_secret_batch(conn.runtime(), requests)?;
    let second = client
        .bind_with_input_and_resolved_secrets(attached, &spec, input_schema, secrets)
        .map_err(to_df)?;
    if !second.required_secret_types().is_empty() {
        return Err(DataFusionError::Execution(
            "VGI worker requested secrets twice; only one resolved retry is allowed".into(),
        ));
    }
    Ok((second, true))
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
    /// The original bind is retained because function-level statistics are a
    /// per-bind RPC and depend on its exact arguments and opaque response.
    statistics_bind: vgi_client::BoundFunction,
    function_statistics: tokio::sync::OnceCell<Option<Arc<Statistics>>>,
    output_schema: SchemaRef,
    /// The worker opted into receiving projection IDs. Functions that do not
    /// opt in return their full schema and are narrowed locally instead.
    projection_pushdown: bool,
    /// The discovery-time split declaration. `None` means the function was not
    /// discoverable, so `table_function_plan` is still probed for compatibility.
    supports_splits: Option<bool>,
    /// Whether DataFusion may omit its local copy of a pushed filter.
    filters_exactly_applied: bool,
    /// Catalog version captured by the bind and included in cache identity.
    catalog_version: i64,
    /// Whether the planning bind supplied resolved secrets. Secret resolvers
    /// may rotate values without changing identity, so these calls bypass
    /// plan and result caches rather than putting secret bytes in their keys.
    uses_resolved_secrets: bool,
    /// Explicit historical coordinate for this catalog-table scan.
    at: Option<vgi_client::At>,
    /// Primary-key and unique constraints advertised for catalog tables.
    /// DataFusion has no native representation for VGI check, foreign-key, or
    /// standalone NOT NULL metadata.
    constraints: Option<Constraints>,
    /// Catalog column name -> backing function column name when the table
    /// deliberately renames a function's positional output.
    column_mapping: Option<Arc<HashMap<String, String>>>,
    max_workers: usize,
}

impl VgiTableProvider {
    /// Use a catalog-declared schema after binding its scan function.
    ///
    /// VGI table discovery is the SQL catalog contract. Bind schemas may carry
    /// different names, nullability, or field metadata. Types and positions
    /// must agree; the catalog names are authoritative and worker batches are
    /// renamed by the positional mapping retained here.
    pub(crate) fn with_declared_schema(
        mut self: Arc<Self>,
        declared: SchemaRef,
    ) -> DFResult<Arc<Self>> {
        let compatible = self.output_schema.fields().len() == declared.fields().len()
            && self
                .output_schema
                .fields()
                .iter()
                .zip(declared.fields())
                .all(|(bound, catalog)| bound.data_type() == catalog.data_type());
        if !compatible {
            return Err(DataFusionError::Plan(format!(
                "VGI catalog schema {:?} is incompatible with scan bind schema {:?}",
                declared, self.output_schema
            )));
        }
        let provider = Arc::get_mut(&mut self).expect("freshly bound VGI provider has one owner");
        provider.column_mapping = Some(Arc::new(
            declared
                .fields()
                .iter()
                .zip(provider.output_schema.fields())
                .map(|(catalog, bound)| (catalog.name().clone(), bound.name().clone()))
                .collect(),
        ));
        provider.output_schema = declared;
        Ok(self)
    }

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
        Self::bind_catalog_table_inner(conn, catalog, schema_name, info, None).await
    }

    /// Bind a catalog table at a historical VGI coordinate.
    pub async fn bind_catalog_table_at(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        info: vgi_client::TableInfo,
        at: vgi_client::At,
    ) -> DFResult<Arc<Self>> {
        Self::bind_catalog_table_inner(conn, catalog, schema_name, info, Some(at)).await
    }

    /// Bind one function-backed arm of a VGI catalog table.
    ///
    /// Multi-branch metadata carries the same flat scan-argument encoding as
    /// the legacy single-function response. The function may be declared in
    /// either the table's schema or the attached catalog's default schema, so
    /// resolve it with the same candidate order used by ordinary catalog
    /// tables. The outer catalog provider owns reconciliation and constraints.
    pub(crate) async fn bind_catalog_branch(
        conn: VgiConnection,
        catalog: impl Into<String>,
        table_schema: impl Into<String>,
        function: impl Into<String>,
        raw_arguments: vgi_client::Bytes,
    ) -> DFResult<Arc<Self>> {
        let catalog = catalog.into();
        let table_schema = table_schema.into();
        let function = function.into();
        let arguments = if raw_arguments.0.is_empty() {
            Arguments::new()
        } else {
            Arguments::from_scan_arguments(&raw_arguments.0).map_err(to_df)?
        };
        let c = conn.clone();
        let cat = catalog.clone();
        let table_schema_for_bind = table_schema.clone();
        let function_for_bind = function.clone();
        let arguments_for_bind = arguments.clone();
        let (
            function_schema,
            statistics_bind,
            capabilities,
            catalog_version,
            uses_resolved_secrets,
        ) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = c.attach(&mut client, &cat)?;
            let default_schema = attached.default_schema().to_string();
            let mut candidates = vec![table_schema_for_bind];
            if default_schema != candidates[0] {
                candidates.push(default_schema);
            }

            let mut last_error = None;
            for schema in candidates {
                let spec = BindSpec::table(&function_for_bind)
                    .in_schema(&schema)
                    .with_arguments(arguments_for_bind.clone());
                match bind_with_secrets_status(&c, &mut client, &attached, &spec) {
                    Ok((bound, uses_resolved_secrets)) => {
                        let capabilities = function_capabilities(
                            &mut client,
                            &attached,
                            &schema,
                            &function_for_bind,
                        )?;
                        return Ok::<_, DataFusionError>((
                            schema,
                            bound,
                            capabilities,
                            attached.info().catalog_version,
                            uses_resolved_secrets,
                        ));
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            Err(last_error.expect("at least one catalog branch schema candidate"))
        })
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))??;
        let output_schema = statistics_bind.output_schema().clone();

        Ok(Arc::new(Self {
            conn,
            catalog,
            schema_name: function_schema,
            function,
            arguments,
            raw_arguments: None,
            statistics_bind,
            function_statistics: tokio::sync::OnceCell::new(),
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            filters_exactly_applied: capabilities.filters_exactly_applied,
            catalog_version,
            uses_resolved_secrets,
            at: None,
            constraints: None,
            column_mapping: None,
            max_workers: 1,
        }))
    }

    async fn bind_catalog_table_inner(
        conn: VgiConnection,
        catalog: impl Into<String>,
        _schema_name: impl Into<String>,
        info: vgi_client::TableInfo,
        at: Option<vgi_client::At>,
    ) -> DFResult<Arc<Self>> {
        let catalog = catalog.into();
        let c = conn.clone();
        let cat2 = catalog.clone();
        let bind_at = at.clone();
        let primary_keys = info.primary_key_constraints.clone();
        let unique_keys = info.unique_constraints.clone();

        let (
            function,
            function_schema,
            arguments,
            statistics_bind,
            capabilities,
            catalog_version,
            uses_resolved_secrets,
        ) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = c.attach(&mut client, &cat2)?;
            let scan = client
                .table_scan_function(&attached, &info, bind_at.as_ref())
                .map_err(to_df)?;
            let arguments = if scan.arguments.0.is_empty() {
                Arguments::new()
            } else {
                Arguments::from_scan_arguments(&scan.arguments.0).map_err(to_df)?
            };

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
                let mut spec = BindSpec::table(&scan.function_name)
                    .in_schema(schema)
                    .with_arguments(arguments.clone());
                spec.at = bind_at.clone();
                match bind_with_secrets_status(&c, &mut client, &attached, &spec) {
                    Ok((bound, uses_resolved_secrets)) => {
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
                            bound,
                            capabilities,
                            attached.info().catalog_version,
                            uses_resolved_secrets,
                        ));
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            Err(last_err.expect("at least one candidate schema"))
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;
        let output_schema = statistics_bind.output_schema().clone();
        let constraints = Some(datafusion_constraints(
            primary_keys,
            unique_keys,
            output_schema.fields().len(),
        )?);

        Ok(Arc::new(Self {
            conn,
            catalog,
            // The scan binds in the function's schema, which may differ from
            // the table's; the scan re-binds and must resolve identically.
            schema_name: function_schema,
            function,
            arguments,
            raw_arguments: None,
            statistics_bind,
            function_statistics: tokio::sync::OnceCell::new(),
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            filters_exactly_applied: capabilities.filters_exactly_applied,
            catalog_version,
            uses_resolved_secrets,
            at,
            constraints,
            column_mapping: None,
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
        let catalog_version = attached.info().catalog_version;
        let spec = BindSpec::table(function)
            .in_schema(schema_name)
            .with_arguments(arguments.clone());
        let (bound, uses_resolved_secrets) =
            bind_with_secrets_status(&conn, &mut client, &attached, &spec)?;
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
            statistics_bind: bound,
            function_statistics: tokio::sync::OnceCell::new(),
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            filters_exactly_applied: capabilities.filters_exactly_applied,
            catalog_version,
            uses_resolved_secrets,
            at: None,
            constraints: None,
            column_mapping: None,
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

        let (statistics_bind, capabilities, max_workers, catalog_version, uses_resolved_secrets) =
            tokio::task::spawn_blocking(move || {
                let mut client = c.connect()?;
                let attached = c.attach(&mut client, &cat2)?;
                let spec = BindSpec::table(&fn2).in_schema(&sch2).with_arguments(args2);
                let (bound, uses_resolved_secrets) =
                    bind_with_secrets_status(&c, &mut client, &attached, &spec)?;
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
                Ok::<_, DataFusionError>((
                    bound,
                    capabilities,
                    1usize,
                    attached.info().catalog_version,
                    uses_resolved_secrets,
                ))
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;
        let output_schema = statistics_bind.output_schema().clone();

        Ok(Arc::new(Self {
            conn,
            catalog,
            schema_name,
            function,
            arguments,
            raw_arguments: None,
            statistics_bind,
            function_statistics: tokio::sync::OnceCell::new(),
            output_schema,
            projection_pushdown: capabilities.projection_pushdown,
            supports_splits: capabilities.supports_splits,
            filters_exactly_applied: capabilities.filters_exactly_applied,
            catalog_version,
            uses_resolved_secrets,
            at: None,
            constraints: None,
            column_mapping: None,
            max_workers,
        }))
    }

    async fn bound_function_statistics(&self) -> Option<Arc<Statistics>> {
        self.function_statistics
            .get_or_init(|| async {
                let connection = self.conn.clone();
                let catalog = self.catalog.clone();
                let bound = self.statistics_bind.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut client = connection.connect()?;
                    let _attached = connection.attach(&mut client, &catalog)?;
                    client.table_function_statistics(&bound).map_err(to_df)
                })
                .await;
                let raw = match result {
                    Ok(Ok(raw)) => raw,
                    Ok(Err(error)) => {
                        let mut event = VgiEvent::new("table_function.statistics_error");
                        event.catalog = Some(self.catalog.clone());
                        event.function = Some(format!("{}.{}", self.schema_name, self.function));
                        event.message = Some(error.to_string());
                        self.conn.runtime.emit(event);
                        return None;
                    }
                    Err(error) => {
                        let mut event = VgiEvent::new("table_function.statistics_error");
                        event.catalog = Some(self.catalog.clone());
                        event.function = Some(format!("{}.{}", self.schema_name, self.function));
                        event.message = Some(error.to_string());
                        self.conn.runtime.emit(event);
                        return None;
                    }
                };
                if raw.num_columns() == 0 {
                    return None;
                }
                Some(Arc::new(statistics_for_catalog_table(
                    &self.output_schema,
                    &raw,
                    None,
                    None,
                    self.column_mapping.as_deref(),
                )))
            })
            .await
            .clone()
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

        let plan_cache_key = (!self.uses_resolved_secrets).then(|| ()).and_then(|()| {
            self.conn
                .cache_identity_scope(&self.catalog)
                .and_then(|identity_scope| {
                    let arguments = self
                        .raw_arguments
                        .as_ref()
                        .map(|value| value.0.clone())
                        .or_else(|| self.arguments.to_ipc().ok().map(|value| value.0))?;
                    Some(crate::runtime::PlanCacheKey {
                        identity_scope,
                        worker_label: self.conn.label().to_string(),
                        function: format!("{}.{}", self.schema_name, self.function),
                        arguments,
                        projection: projection.clone(),
                        filters: pushdown.cache_identity(),
                        row_limit: limit.and_then(|value| i64::try_from(value).ok()),
                        target_partitions,
                        catalog_version: self.catalog_version,
                        at: self
                            .at
                            .as_ref()
                            .map(|at| (at.unit.clone(), at.value.clone())),
                        settings: self.conn.runtime.session_settings_identity(),
                        attach_options: self.conn.cache_attach_context(&self.catalog),
                    })
                })
        });
        let cached_plan = plan_cache_key
            .as_ref()
            .and_then(|key| self.conn.runtime.plan_get(key));

        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arguments = self.arguments.clone();
        let raw_arguments = self.raw_arguments.clone();
        let at = self.at.clone();

        // The client is blocking, so planning runs on a blocking thread rather
        // than a tokio worker — a blocking call on the runtime would stall every
        // other task sharing it.
        let plan = if let Some(plan) = cached_plan {
            let mut event = VgiEvent::new("plan_cache.hit");
            event.catalog = Some(self.catalog.clone());
            event.function = Some(format!("{}.{}", self.schema_name, self.function));
            self.conn.runtime.emit(event);
            Some(plan)
        } else {
            let mut event = VgiEvent::new("plan_cache.miss");
            event.catalog = Some(self.catalog.clone());
            event.function = Some(format!("{}.{}", self.schema_name, self.function));
            self.conn.runtime.emit(event);
            let plan_started = std::time::Instant::now();
            let plan =
                tokio::task::spawn_blocking(move || -> DFResult<Option<vgi_client::ScanPlan>> {
                    let mut client = conn.connect()?;
                    let attached = conn.attach(&mut client, &catalog)?;
                    let mut spec = BindSpec::table(&function).in_schema(&schema_name);
                    spec = match &raw_arguments {
                        Some(raw) => spec.with_raw_arguments(raw.clone()),
                        None => spec.with_arguments(arguments.clone()),
                    };
                    spec.at = at;
                    let bound = bind_with_secrets(&conn, &mut client, &attached, &spec)?;

                    let opts = PlanOptions {
                        projection,
                        pushdown_filters: pushdown.blob,
                        join_keys: (!pushdown.join_keys.is_empty()).then_some(pushdown.join_keys),
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
            let mut event = VgiEvent::new("split_plan.complete");
            event.catalog = Some(self.catalog.clone());
            event.function = Some(format!("{}.{}", self.schema_name, self.function));
            event.duration = Some(plan_started.elapsed());
            event.message = Some(format!(
                "{} split(s)",
                plan.as_ref().map(|plan| plan.splits.len()).unwrap_or(0)
            ));
            self.conn.runtime.emit(event);
            if let (Some(key), Some(plan)) = (plan_cache_key, plan.as_ref()) {
                if let Some(seconds) = plan
                    .cache_max_age_seconds
                    .and_then(|value| u64::try_from(value).ok())
                {
                    self.conn.runtime.plan_insert(
                        key,
                        plan.clone(),
                        std::time::Duration::from_secs(seconds),
                    );
                }
            }
            plan
        };

        let Some(plan) = plan else {
            return Ok(None);
        };

        if let (Some(hook), Some(locations)) =
            (self.conn.runtime.locality_hook(), plan.locations.as_ref())
        {
            let splits = plan
                .splits
                .iter()
                .enumerate()
                .map(|(split_index, split)| VgiSplitLocality {
                    split_index,
                    locations: split
                        .location_ids
                        .iter()
                        .flatten()
                        .filter_map(|id| usize::try_from(*id).ok())
                        .filter_map(|id| locations.get(id).cloned())
                        .collect(),
                })
                .collect::<Vec<_>>();
            hook.planned(&self.catalog, &self.function, &splits);
        }

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
    /// A start position without an end frontier is a live, unbounded stream.
    unbounded: bool,
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
    let unbounded = (plan.start_position.is_some() && plan.end_position.is_none())
        || plan
            .splits
            .iter()
            .any(|split| split.start_position.is_some() && split.end_position.is_none());
    let splits = &plan.splits;
    if splits.is_empty() {
        return PlannedSplits {
            groups: Vec::new(),
            // A zero-split plan is the protocol's definitive "no work" result.
            num_rows: Precision::Exact(0),
            total_byte_size: Precision::Exact(0),
            plan,
            unbounded,
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
        unbounded,
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

pub(crate) fn datafusion_constraints(
    primary_keys: Vec<Vec<i32>>,
    unique_keys: Vec<Vec<i32>>,
    column_count: usize,
) -> DFResult<Constraints> {
    fn indices(values: Vec<i32>, column_count: usize) -> DFResult<Vec<usize>> {
        values
            .into_iter()
            .map(|value| {
                let index = usize::try_from(value).map_err(|_| {
                    DataFusionError::Plan(format!(
                        "VGI table constraint contains negative column index {value}"
                    ))
                })?;
                if index >= column_count {
                    return Err(DataFusionError::Plan(format!(
                        "VGI table constraint column index {index} exceeds schema width {column_count}"
                    )));
                }
                Ok(index)
            })
            .collect()
    }

    let mut constraints = Vec::with_capacity(primary_keys.len() + unique_keys.len());
    for key in primary_keys {
        constraints.push(Constraint::PrimaryKey(indices(key, column_count)?));
    }
    for key in unique_keys {
        constraints.push(Constraint::Unique(indices(key, column_count)?));
    }
    Ok(Constraints::new_unverified(constraints))
}

#[async_trait]
impl TableProvider for VgiTableProvider {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.constraints.as_ref()
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
                    if self.filters_exactly_applied {
                        TableProviderFilterPushDown::Exact
                    } else {
                        TableProviderFilterPushDown::Inexact
                    }
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
        if !filters.is_empty() {
            if let Some(statistics) = self.bound_function_statistics().await {
                if filters_prune_statistics(state, &self.output_schema, statistics, filters) {
                    return Ok(Arc::new(EmptyExec::new(projected)));
                }
            }
        }
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
            self.catalog_version,
            self.at.clone(),
            self.column_mapping.clone(),
            !self.uses_resolved_secrets,
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
    /// Runtime filters linked by DataFusion's post-optimizer pushdown pass.
    /// Hash joins produce an `IN` list plus min/max bounds; Top-K produces
    /// tightening comparison bounds.
    dynamic_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
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
    metrics: ExecutionPlanMetricsSet,
    cache_metrics: VgiCacheMetrics,
    cache_key: Option<vgi_client::CacheKey>,
    cache_probe: OnceLock<Option<vgi_client::CachedEntry>>,
    cache_flight: Arc<OnceLock<crate::runtime::ResultFlightClaim>>,
    cache_flight_probe: Arc<OnceLock<Option<vgi_client::CachedEntry>>>,
    cache_capture: Option<Arc<ScanCacheCapture>>,
    at: Option<vgi_client::At>,
    column_mapping: Option<Arc<HashMap<String, String>>>,
}

#[derive(Clone, Debug)]
struct VgiCacheMetrics {
    hits: Count,
    misses: Count,
    stores: Count,
    waits: Count,
    coalesced_hits: Count,
    coalesced_retries: Count,
    coalesced_aborts: Count,
    revalidations: Count,
    stale_serves: Count,
    refusals: Count,
    capture_aborts: Count,
}

impl VgiCacheMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        let counter = |name| MetricBuilder::new(metrics).global_counter(name);
        Self {
            hits: counter("cache_hits"),
            misses: counter("cache_misses"),
            stores: counter("cache_stores"),
            waits: counter("cache_waits"),
            coalesced_hits: counter("cache_coalesced_hits"),
            coalesced_retries: counter("cache_coalesced_retries"),
            coalesced_aborts: counter("cache_coalesced_aborts"),
            revalidations: counter("cache_revalidations"),
            stale_serves: counter("cache_stale_serves"),
            refusals: counter("cache_refusals"),
            capture_aborts: counter("cache_capture_aborts"),
        }
    }
}

#[derive(Clone, Debug)]
struct VgiPartitionMetrics {
    baseline: BaselineMetrics,
    worker: VgiWorkerMetrics,
}

#[derive(Clone, Debug)]
struct VgiWorkerMetrics {
    worker_scans: Count,
    worker_batches: Count,
    worker_rows: Count,
    /// Decoded Arrow memory attributed to batches received from the worker.
    /// VGI does not currently report compressed transport-byte counts.
    worker_bytes: Count,
}

impl VgiPartitionMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            worker: VgiWorkerMetrics {
                worker_scans: MetricBuilder::new(metrics).counter("worker_scans", partition),
                worker_batches: MetricBuilder::new(metrics)
                    .with_category(MetricCategory::Rows)
                    .counter("worker_batches", partition),
                worker_rows: MetricBuilder::new(metrics)
                    .with_category(MetricCategory::Rows)
                    .counter("worker_rows", partition),
                worker_bytes: MetricBuilder::new(metrics)
                    .with_category(MetricCategory::Bytes)
                    .counter("worker_bytes", partition),
            },
        }
    }
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
        catalog_version: i64,
        at: Option<vgi_client::At>,
        column_mapping: Option<Arc<HashMap<String, String>>>,
        result_cache_eligible: bool,
    ) -> Self {
        let split_output_ordering = split_groups
            .as_ref()
            .and_then(|planned| split_ordering(&schema, planned));
        let cache_preserves_ordering = split_output_ordering.is_none();
        let equivalence = split_output_ordering
            .map(|ordering| EquivalenceProperties::new_with_orderings(schema.clone(), [ordering]))
            .unwrap_or_else(|| EquivalenceProperties::new(schema.clone()));
        let boundedness = if split_groups.as_ref().is_some_and(|plan| plan.unbounded) {
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            }
        } else {
            Boundedness::Bounded
        };
        let properties = Arc::new(PlanProperties::new(
            equivalence,
            // One partition per connection the worker will accept. VGI hands
            // each connection a disjoint slice, so the partitions really are
            // independent.
            Partitioning::UnknownPartitioning(partitions.max(1)),
            EmissionType::Incremental,
            boundedness,
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
        let cache_key = if result_cache_eligible
            && conn.cache_enabled
            && conn.runtime.options().cache_enabled
            && cache_preserves_ordering
        {
            conn.cache_identity_scope(&catalog)
                .and_then(|identity_scope| {
                    let arguments = raw_arguments
                        .as_ref()
                        .map(|value| value.0.clone())
                        .or_else(|| arguments.to_ipc().ok().map(|value| value.0))?;
                    Some(vgi_client::CacheKey {
                        catalog: catalog.clone(),
                        identity_scope,
                        worker_label: conn.label().to_string(),
                        function: format!("{schema_name}.{function}"),
                        arguments,
                        projection: projection.clone(),
                        filters: pushdown.cache_identity(),
                        catalog_version,
                        at: at.as_ref().map(|at| (at.unit.clone(), at.value.clone())),
                        settings: conn.runtime.session_settings_identity(),
                        attach_options: conn.cache_attach_context(&catalog),
                        row_limit: limit.and_then(|value| i64::try_from(value).ok()),
                        ordering: None,
                        sample: None,
                        plan: None,
                    })
                })
        } else {
            None
        };
        let metrics = ExecutionPlanMetricsSet::new();
        let cache_metrics = VgiCacheMetrics::new(&metrics);
        let cache_capture = cache_key.as_ref().map(|key| {
            Arc::new(ScanCacheCapture::new(
                Arc::clone(conn.runtime.result_cache()),
                Arc::clone(&conn.runtime),
                key.clone(),
                partitions.max(1),
                cache_metrics.clone(),
            ))
        });
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
            dynamic_filters: Vec::new(),
            limit,
            schema,
            properties,
            split_groups,
            statistics,
            partition_statistics,
            metrics,
            cache_metrics,
            cache_key,
            cache_probe: OnceLock::new(),
            cache_flight: Arc::new(OnceLock::new()),
            cache_flight_probe: Arc::new(OnceLock::new()),
            cache_capture,
            at,
            column_mapping,
        }
    }

    fn with_dynamic_filters(&self, dynamic_filters: Vec<Arc<DynamicFilterPhysicalExpr>>) -> Self {
        let metrics = ExecutionPlanMetricsSet::new();
        let cache_metrics = VgiCacheMetrics::new(&metrics);
        Self {
            conn: self.conn.clone(),
            catalog: self.catalog.clone(),
            schema_name: self.schema_name.clone(),
            function: self.function.clone(),
            arguments: self.arguments.clone(),
            raw_arguments: self.raw_arguments.clone(),
            projection: self.projection.clone(),
            projection_pushdown: self.projection_pushdown,
            pushdown: self.pushdown.clone(),
            dynamic_filters,
            limit: self.limit,
            schema: self.schema.clone(),
            properties: self.properties.clone(),
            split_groups: self.split_groups.clone(),
            statistics: self.statistics.clone(),
            partition_statistics: self.partition_statistics.clone(),
            metrics,
            cache_metrics,
            // A runtime predicate can change after lookup. A result keyed only
            // by its planning-time filter must never enter the VGI result cache.
            cache_key: None,
            cache_probe: OnceLock::new(),
            cache_flight: Arc::new(OnceLock::new()),
            cache_flight_probe: Arc::new(OnceLock::new()),
            cache_capture: None,
            at: self.at.clone(),
            column_mapping: self.column_mapping.clone(),
        }
    }
}

fn snapshot_dynamic_filters(
    filters: &[Arc<DynamicFilterPhysicalExpr>],
) -> DFResult<(u64, Vec<Arc<dyn PhysicalExpr>>)> {
    loop {
        let before = dynamic_filter_generation(filters);
        let snapshots = filters
            .iter()
            .map(|filter| filter.current())
            .collect::<DFResult<Vec<_>>>()?;
        let after = dynamic_filter_generation(filters);
        if before == after {
            return Ok((after, snapshots));
        }
    }
}

fn dynamic_filter_generation(filters: &[Arc<DynamicFilterPhysicalExpr>]) -> u64 {
    filters.iter().fold(0_u64, |generation, filter| {
        generation.wrapping_add(filter.snapshot_generation())
    })
}

struct ScanCacheCapture {
    cache: Arc<vgi_client::ResultCache>,
    runtime: Arc<VgiRuntime>,
    key: vgi_client::CacheKey,
    metrics: VgiCacheMetrics,
    flight: Mutex<Option<Arc<crate::runtime::ResultFlightProducer>>>,
    state: Mutex<ScanCacheCaptureState>,
}

struct ScanCacheCaptureState {
    partitions: Vec<Option<Vec<datafusion::arrow::array::RecordBatch>>>,
    controls: Vec<Option<vgi_client::CacheControl>>,
    started: Vec<bool>,
    aborted: bool,
    committed: bool,
}

impl fmt::Debug for ScanCacheCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanCacheCapture")
            .field("function", &self.key.function)
            .finish_non_exhaustive()
    }
}

impl ScanCacheCapture {
    fn new(
        cache: Arc<vgi_client::ResultCache>,
        runtime: Arc<VgiRuntime>,
        key: vgi_client::CacheKey,
        partitions: usize,
        metrics: VgiCacheMetrics,
    ) -> Self {
        Self {
            cache,
            runtime,
            key,
            metrics,
            flight: Mutex::new(None),
            state: Mutex::new(ScanCacheCaptureState {
                partitions: vec![None; partitions],
                controls: vec![None; partitions],
                started: vec![false; partitions],
                aborted: false,
                committed: false,
            }),
        }
    }

    fn abort(&self, reason: &'static str) {
        let mut state = self.state.lock().unwrap();
        if !state.aborted && !state.committed {
            state.aborted = true;
            state.partitions.iter_mut().for_each(|slot| *slot = None);
            self.cache.record_capture_abort();
            self.metrics.capture_aborts.add(1);
            let mut event = VgiEvent::new("cache.capture_aborted");
            event.catalog = Some(self.key.catalog.clone());
            event.function = Some(self.key.function.clone());
            event.message = Some(reason.to_string());
            self.runtime.emit(event);
            if let Some(flight) = self.flight.lock().unwrap().as_ref() {
                flight.abort(reason);
            }
        }
    }

    fn set_flight(&self, flight: Arc<crate::runtime::ResultFlightProducer>) {
        let mut active = self.flight.lock().unwrap();
        if active.is_none() {
            *active = Some(flight);
        }
    }

    fn revalidated(&self) {
        let mut state = self.state.lock().unwrap();
        if state.aborted || state.committed {
            return;
        }
        state.committed = true;
        if let Some(flight) = self.flight.lock().unwrap().as_ref() {
            flight.stored();
        }
    }

    fn start(&self, partition: usize) {
        let mut state = self.state.lock().unwrap();
        if let Some(started) = state.started.get_mut(partition) {
            *started = true;
        }
    }

    fn complete(
        &self,
        partition: usize,
        batches: Vec<datafusion::arrow::array::RecordBatch>,
        control: Option<vgi_client::CacheControl>,
    ) {
        let mut state = self.state.lock().unwrap();
        if state.aborted || state.committed || partition >= state.partitions.len() {
            return;
        }
        state.partitions[partition] = Some(batches);
        state.controls[partition] = control;
        if state.partitions.iter().any(Option::is_none) {
            return;
        }
        let Some(control) = state.controls.iter().flatten().next().cloned() else {
            state.aborted = true;
            self.cache.remove(&self.key);
            if let Some(flight) = self.flight.lock().unwrap().as_ref() {
                flight.abort("worker did not opt result into caching");
            }
            return;
        };
        if state
            .controls
            .iter()
            .flatten()
            .any(|candidate| candidate != &control)
        {
            state.aborted = true;
            self.cache.remove(&self.key);
            self.cache.record_capture_abort();
            self.metrics.capture_aborts.add(1);
            if let Some(flight) = self.flight.lock().unwrap().as_ref() {
                flight.abort("split cache controls disagreed");
            }
            return;
        }
        let batches = state
            .partitions
            .iter()
            .flatten()
            .flat_map(|batches| batches.iter().cloned())
            .collect::<Vec<_>>();
        let bytes = batches
            .iter()
            .flat_map(|batch| batch.columns())
            .map(|array| array.get_array_memory_size())
            .sum();
        match self
            .cache
            .eligibility(Some(&control), Some(&self.key.identity_scope), bytes)
        {
            Ok(ttl) => {
                self.cache
                    .insert(self.key.clone(), batches, ttl, Some(&control));
                state.committed = true;
                self.metrics.stores.add(1);
                let mut event = VgiEvent::new("cache.store");
                event.catalog = Some(self.key.catalog.clone());
                event.function = Some(self.key.function.clone());
                self.runtime.emit(event);
                if let Some(flight) = self.flight.lock().unwrap().as_ref() {
                    flight.stored();
                }
            }
            Err(reason) => {
                state.aborted = true;
                self.cache.remove(&self.key);
                self.metrics.refusals.add(1);
                let mut event = VgiEvent::new("cache.refused");
                event.catalog = Some(self.key.catalog.clone());
                event.function = Some(self.key.function.clone());
                event.message = Some(format!("{reason:?}"));
                self.runtime.emit(event);
                if let Some(flight) = self.flight.lock().unwrap().as_ref() {
                    flight.abort(format!("cache refused result: {reason:?}"));
                }
            }
        }
    }
}

impl Drop for ScanCacheCapture {
    fn drop(&mut self) {
        let Ok(state) = self.state.get_mut() else {
            return;
        };
        if !state.aborted && !state.committed && state.started.iter().any(|started| *started) {
            self.cache.record_capture_abort();
            self.metrics.capture_aborts.add(1);
            let mut event = VgiEvent::new("cache.capture_aborted");
            event.catalog = Some(self.key.catalog.clone());
            event.function = Some(self.key.function.clone());
            event.message = Some("execution ended before every partition completed".to_string());
            self.runtime.emit(event);
            if let Some(flight) = self
                .flight
                .get_mut()
                .ok()
                .and_then(|flight| flight.as_ref())
            {
                flight.abort("execution ended before every partition completed");
            }
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

/// Convert catalog-level VGI column statistics into DataFusion statistics.
///
/// These bounds are the worker's declared optimizer contract (the same values
/// DuckDB installs on its catalog table), so min/max are exact and may safely
/// drive DataFusion's existing pruning predicate. Cardinality remains inexact
/// unless the worker's estimate and maximum agree.
pub(crate) fn statistics_for_catalog_table(
    schema: &SchemaRef,
    batch: &datafusion::arrow::record_batch::RecordBatch,
    estimate: Option<i64>,
    maximum: Option<i64>,
    column_mapping: Option<&HashMap<String, String>>,
) -> Statistics {
    let num_rows = match (
        estimate.and_then(|value| usize::try_from(value).ok()),
        maximum.and_then(|value| usize::try_from(value).ok()),
    ) {
        (Some(estimate), Some(maximum)) if estimate == maximum => Precision::Exact(estimate),
        (Some(estimate), _) => Precision::Inexact(estimate),
        _ => Precision::Absent,
    };
    let mut statistics = statistics_with_estimates(schema, num_rows, Precision::Absent);
    let mut columns = HashMap::new();
    if let Some(names) = batch
        .column_by_name("column_name")
        .and_then(|array| array.as_any().downcast_ref::<StringArray>())
    {
        for row in 0..batch.num_rows() {
            if names.is_null(row) {
                continue;
            }
            let worker_name = names.value(row);
            let name = column_mapping
                .and_then(|mapping| {
                    mapping.iter().find_map(|(catalog, worker)| {
                        (worker == worker_name).then_some(catalog.as_str())
                    })
                })
                .unwrap_or(worker_name);
            let Ok(field) = schema.field_with_name(name) else {
                continue;
            };
            let null_count = match (
                bool_at(batch, "has_null", row),
                bool_at(batch, "has_not_null", row),
            ) {
                (Some(false), _) => Precision::Exact(0),
                (_, Some(false)) => statistics.num_rows,
                _ => Precision::Absent,
            };
            let distinct_count = i64_at(batch, "distinct_count", row)
                .and_then(|value| usize::try_from(value).ok())
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent);
            let mut min = statistic_value(batch, "min", row, field.data_type());
            let mut max = statistic_value(batch, "max", row, field.data_type());
            // A dictionary/ENUM can be exposed as Utf8 while retaining ordinal
            // bounds. Those are not lexicographic bounds, so they are unsafe for
            // DataFusion string pruning and must be ignored.
            if matches!((&min, &max), (Some(min), Some(max)) if min > max) {
                min = None;
                max = None;
            }
            columns.insert(
                name.to_string(),
                ColumnStatistics::new_unknown()
                    .with_null_count(null_count)
                    .with_distinct_count(distinct_count)
                    .with_min_value(min.map(Precision::Exact).unwrap_or(Precision::Absent))
                    .with_max_value(max.map(Precision::Exact).unwrap_or(Precision::Absent)),
            );
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

pub(crate) fn filters_prune_statistics(
    state: &dyn Session,
    schema: &SchemaRef,
    statistics: Arc<Statistics>,
    filters: &[Expr],
) -> bool {
    let Some(filter) = conjunction(filters.iter().cloned()) else {
        return false;
    };
    let Ok(df_schema) = DFSchema::try_from(schema.as_ref().clone()) else {
        return false;
    };
    let Ok(predicate) = state.create_physical_expr(filter, &df_schema) else {
        return false;
    };
    let Some(predicate) = PruningPredicateBuilder::new()
        .with_file_schema(Arc::clone(schema))
        .build(predicate)
    else {
        return false;
    };
    let statistics = PrunableStatistics::new(vec![statistics], Arc::clone(schema));
    predicate
        .prune(&statistics)
        .ok()
        .is_some_and(|keep| keep == [false])
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
        if !self.dynamic_filters.is_empty() {
            write!(f, ", dynamic_filters={}", self.dynamic_filters.len())?;
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
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
        f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        ) -> DFResult<datafusion::common::tree_node::TreeNodeRecursion>,
    ) -> DFResult<datafusion::common::tree_node::TreeNodeRecursion> {
        // Make the shared dynamic-expression ids visible to their producers.
        // HashJoinExec uses this traversal to avoid building a filter nobody
        // consumes.
        for filter in &self.dynamic_filters {
            let expression: Arc<dyn PhysicalExpr> = filter.clone();
            if matches!(
                f(&expression)?,
                datafusion::common::tree_node::TreeNodeRecursion::Stop
            ) {
                return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
            }
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &datafusion::common::config::ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let mut results = Vec::with_capacity(child_pushdown_result.parent_filters.len());
        let mut dynamic_filters = self.dynamic_filters.clone();
        for parent in child_pushdown_result.parent_filters {
            if phase == FilterPushdownPhase::Post {
                match Arc::downcast::<DynamicFilterPhysicalExpr>(parent.filter) {
                    Ok(filter) => {
                        if !dynamic_filters
                            .iter()
                            .any(|existing| existing.expression_id() == filter.expression_id())
                        {
                            dynamic_filters.push(filter);
                        }
                        // This is an optimizer hint. The hash join / Top-K
                        // remains the correctness boundary even if an older
                        // worker ignores the VGI continuation metadata.
                        results.push(PushedDown::Yes);
                    }
                    Err(_) => results.push(PushedDown::No),
                }
            } else {
                results.push(PushedDown::No);
            }
        }
        let updated_node = (dynamic_filters.len() != self.dynamic_filters.len()).then(|| {
            Arc::new(self.with_dynamic_filters(dynamic_filters)) as Arc<dyn ExecutionPlan>
        });
        Ok(FilterPushdownPropagation {
            filters: results,
            updated_node,
        })
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
        let partition_metrics = VgiPartitionMetrics::new(&self.metrics, partition);
        if let Some(key) = &self.cache_key {
            let first_probe = self.cache_probe.get().is_none();
            let cached = self
                .cache_probe
                .get_or_init(|| self.conn.runtime.result_cache().get(key))
                .clone();
            if first_probe {
                if cached.is_some() {
                    self.cache_metrics.hits.add(1);
                } else {
                    self.cache_metrics.misses.add(1);
                }
                let mut event = VgiEvent::new(if cached.is_some() {
                    "cache.hit"
                } else {
                    "cache.miss"
                });
                event.catalog = Some(self.catalog.clone());
                event.function = Some(format!("{}.{}", self.schema_name, self.function));
                self.conn.runtime.emit(event);
            }
            if let Some(entry) = cached {
                let out_schema = self.schema.clone();
                let batches = (partition == 0)
                    .then(|| entry.batches().to_vec())
                    .unwrap_or_default();
                let baseline = partition_metrics.baseline;
                let stream = futures::stream::iter(batches.into_iter().map(Ok))
                    .map(move |batch| batch.record_output(&baseline));
                return Ok(Box::pin(RecordBatchStreamAdapter::new(out_schema, stream)));
            }
        }
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arguments = self.arguments.clone();
        let raw_arguments = self.raw_arguments.clone();
        let at = self.at.clone();
        let projection = self.projection.clone();
        let projection_pushdown = self.projection_pushdown;
        let pushdown = self.pushdown.clone();
        let dynamic_filters = self.dynamic_filters.clone();
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
        let column_mapping = self.column_mapping.clone();
        let cache_capture = self.cache_capture.clone();
        let revalidation = if self.split_groups.is_none() {
            self.cache_key
                .as_ref()
                .and_then(|key| self.conn.runtime.result_cache().get_for_revalidation(key))
        } else {
            None
        };
        let cache_key = self.cache_key.clone();
        let cache = Arc::clone(self.conn.runtime.result_cache());
        let worker_metrics = partition_metrics.worker.clone();
        let cache_metrics = self.cache_metrics.clone();

        // A bounded channel so a slow consumer applies backpressure to the
        // worker rather than letting batches pile up in memory.
        let (tx, rx) =
            tokio::sync::mpsc::channel::<DFResult<datafusion::arrow::array::RecordBatch>>(2);

        let job = move || {
            if let Some(capture) = &cache_capture {
                capture.start(partition);
            }
            let started = std::time::Instant::now();
            let mut event = VgiEvent::new("scan.start");
            event.catalog = Some(catalog.clone());
            event.function = Some(format!("{schema_name}.{function}"));
            event.split = split_tokens
                .as_ref()
                .map(|tokens| format!("partition={partition}, tokens={}", tokens.len()));
            conn.runtime.emit(event);
            let run = || -> DFResult<()> {
                let mut client = conn.connect()?;
                let attached = conn.attach(&mut client, &catalog)?;
                // A catalog table carries the worker's own argument bytes;
                // anything else carries the call's typed arguments. Either way
                // the scan must re-send exactly what the plan bound with —
                // `init` echoes the bind call back.
                let mut spec = BindSpec::table(&function).in_schema(&schema_name);
                spec = match &raw_arguments {
                    Some(raw) => spec.with_raw_arguments(raw.clone()),
                    None => spec.with_arguments(arguments.clone()),
                };
                spec.at = at;
                let bound = bind_with_secrets(&conn, &mut client, &attached, &spec)?;

                // Snapshot after the stream is first polled. HashJoinExec builds
                // its left side before polling the probe stream, so this timing
                // lets the completed join-key set reach VGI init rather than
                // racing an eagerly-started remote scan.
                let (mut dynamic_generation, dynamic_snapshot) =
                    snapshot_dynamic_filters(&dynamic_filters)?;
                let initial_dynamic =
                    filters::serialize_physical(&dynamic_snapshot, &scan_schema, true)?;
                let initial_pushdown = filters::merge(&pushdown, &initial_dynamic)?;
                // Join-key side IPC is init-only. If either the SQL predicate
                // or the first runtime snapshot contains membership keys, keep
                // that complete initial predicate active rather than replacing
                // it with continuation metadata that cannot carry the keys.
                let can_refine_on_continuation = initial_pushdown.join_keys.is_empty();

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
                        if let Some(capture) = &cache_capture {
                            capture.complete(partition, Vec::new(), None);
                        }
                        return Ok(());
                    }
                } else if partition > 0 {
                    if let Some(capture) = &cache_capture {
                        capture.complete(partition, Vec::new(), None);
                    }
                    return Ok(());
                }

                let opts = ScanOptions {
                    projection: push,
                    pushdown_filters: initial_pushdown.blob.clone(),
                    join_keys: (!initial_pushdown.join_keys.is_empty())
                        .then_some(initial_pushdown.join_keys.clone()),
                    // See PlanOptions above: the filter's columns must be
                    // requested even when the projection omits them, or the
                    // worker evaluates the predicate against the wrong column.
                    filter_columns: Some(initial_pushdown.columns.clone()),
                    row_limit: limit.map(|l| l as i64),
                    if_none_match: revalidation.as_ref().and_then(|entry| entry.etag.clone()),
                    if_modified_since: revalidation
                        .as_ref()
                        .and_then(|entry| entry.last_modified.clone()),
                    ..Default::default()
                };
                let opts = match (&split_plan, split_tokens.clone()) {
                    (Some(plan), Some(tokens)) => plan.redemption_options(tokens, opts),
                    _ => opts,
                };
                let mut scan = match client.scan(&bound, &opts) {
                    Ok(scan) => scan,
                    Err(_error)
                        if revalidation.as_ref().is_some_and(|entry| {
                            entry.may_serve_on_error_at(std::time::Instant::now())
                        }) =>
                    {
                        let entry = revalidation.as_ref().expect("checked above");
                        for batch in entry.batches() {
                            if tx.blocking_send(Ok(batch.clone())).is_err() {
                                break;
                            }
                        }
                        cache.record_stale_serve();
                        cache_metrics.stale_serves.add(1);
                        if let Some(capture) = &cache_capture {
                            capture.revalidated();
                        }
                        return Ok(());
                    }
                    Err(error) => return Err(to_df(error)),
                };
                worker_metrics.worker_scans.add(1);

                let mut emitted = 0usize;
                let mut captured = Vec::new();
                let mut tick_pushdown: Option<filters::Pushdown> = None;
                loop {
                    let generation = dynamic_filter_generation(&dynamic_filters);
                    if generation != dynamic_generation {
                        let (stable_generation, snapshots) =
                            snapshot_dynamic_filters(&dynamic_filters)?;
                        if can_refine_on_continuation {
                            let dynamic =
                                filters::serialize_physical(&snapshots, &scan_schema, false)?;
                            tick_pushdown = Some(filters::merge(&pushdown, &dynamic)?);
                        } else {
                            tick_pushdown = None;
                        }
                        dynamic_generation = stable_generation;
                    }
                    let next = match scan.next_batch_with_pushdown_filters(
                        tick_pushdown
                            .as_ref()
                            .and_then(|filter| filter.blob.as_deref()),
                    ) {
                        Ok(next) => next,
                        Err(error) => {
                            if emitted == 0
                                && revalidation.as_ref().is_some_and(|entry| {
                                    entry.may_serve_on_error_at(std::time::Instant::now())
                                })
                            {
                                let entry = revalidation.as_ref().expect("checked above");
                                for batch in entry.batches() {
                                    if tx.blocking_send(Ok(batch.clone())).is_err() {
                                        break;
                                    }
                                }
                                cache.record_stale_serve();
                                cache_metrics.stale_serves.add(1);
                                if let Some(capture) = &cache_capture {
                                    capture.revalidated();
                                }
                                return Ok(());
                            }
                            if let Some(capture) = &cache_capture {
                                capture.abort("worker error");
                            }
                            return Err(to_df(error));
                        }
                    };
                    let Some(batch) = next else {
                        break;
                    };
                    worker_metrics.worker_batches.add(1);
                    worker_metrics.worker_rows.add(batch.num_rows());
                    worker_metrics.worker_bytes.add(
                        datafusion::common::utils::memory::get_record_batch_memory_size(&batch),
                    );
                    // DataFusion requires every batch to match the declared
                    // schema exactly, so conform rather than trusting the
                    // worker to have honoured the projection.
                    let batch = conform(batch, &scan_schema, column_mapping.as_deref())?;
                    let batch = match limit {
                        Some(l) if emitted + batch.num_rows() > l => {
                            batch.slice(0, l.saturating_sub(emitted))
                        }
                        _ => batch,
                    };
                    emitted += batch.num_rows();
                    if cache_capture.is_some() {
                        captured.push(batch.clone());
                    }
                    if tx.blocking_send(Ok(batch)).is_err() {
                        // Receiver dropped: the query was cancelled.
                        let _ = scan.cancel();
                        let mut event = VgiEvent::new("scan.cancelled");
                        event.catalog = Some(catalog.clone());
                        event.function = Some(format!("{schema_name}.{function}"));
                        event.message = Some("consumer dropped".to_string());
                        conn.runtime.emit(event);
                        if let Some(capture) = &cache_capture {
                            capture.abort("consumer dropped");
                        }
                        return Ok(());
                    }
                    if limit.is_some_and(|l| emitted >= l) {
                        let _ = scan.cancel();
                        let mut event = VgiEvent::new("scan.cancelled");
                        event.catalog = Some(catalog.clone());
                        event.function = Some(format!("{schema_name}.{function}"));
                        event.message = Some("limit satisfied".to_string());
                        conn.runtime.emit(event);
                        if let Some(capture) = &cache_capture {
                            capture.abort("limit abandoned scan");
                        }
                        break;
                    }
                }
                if !limit.is_some_and(|l| emitted >= l) {
                    let control = scan.cache_control().cloned();
                    if control.as_ref().is_some_and(|control| control.not_modified) {
                        let entry = revalidation.as_ref().ok_or_else(|| {
                            DataFusionError::Execution(format!(
                                "VGI function `{function}` returned not_modified without a conditional request"
                            ))
                        })?;
                        if emitted != 0 {
                            return Err(DataFusionError::Execution(format!(
                                "VGI function `{function}` returned rows and not_modified together"
                            )));
                        }
                        let key = cache_key.as_ref().ok_or_else(|| {
                            DataFusionError::Execution(format!(
                                "VGI function `{function}` returned not_modified without a cache key"
                            ))
                        })?;
                        let ttl = match cache.eligibility(
                            control.as_ref(),
                            Some(key.identity_scope.as_str()),
                            entry.bytes(),
                        ) {
                            Ok(ttl) => ttl,
                            Err(reason) => {
                                cache.remove(key);
                                return Err(DataFusionError::Execution(format!(
                                    "VGI function `{function}` returned not_modified with ineligible cache control: {reason:?}"
                                )));
                            }
                        };
                        cache.slide(key, ttl);
                        for batch in entry.batches() {
                            tx.blocking_send(Ok(batch.clone())).map_err(|_| {
                                DataFusionError::Execution(
                                    "cached revalidation consumer dropped".to_string(),
                                )
                            })?;
                        }
                        if let Some(capture) = &cache_capture {
                            capture.revalidated();
                        }
                        cache_metrics.revalidations.add(1);
                        let mut event = VgiEvent::new("cache.revalidated");
                        event.catalog = Some(catalog.clone());
                        event.function = Some(format!("{schema_name}.{function}"));
                        conn.runtime.emit(event);
                        return Ok(());
                    }
                    if let Some(capture) = &cache_capture {
                        capture.complete(partition, captured, control);
                    }
                }
                Ok(())
            };
            match run() {
                Ok(()) => {
                    let mut event = VgiEvent::new("scan.complete");
                    event.catalog = Some(catalog.clone());
                    event.function = Some(format!("{schema_name}.{function}"));
                    event.duration = Some(started.elapsed());
                    conn.runtime.emit(event);
                }
                Err(e) => {
                    if let Some(capture) = &cache_capture {
                        capture.abort("scan failed");
                    }
                    let mut event = VgiEvent::new("scan.error");
                    event.catalog = Some(catalog.clone());
                    event.function = Some(format!("{schema_name}.{function}"));
                    event.duration = Some(started.elapsed());
                    event.message = Some(e.to_string());
                    conn.runtime.emit(event);
                    let _ = tx.blocking_send(Err(e));
                }
            }
        };

        // Keep `execute()` lazy. In particular, HashJoinExec constructs the
        // probe stream before its build finishes but does not poll it until the
        // join-key dynamic filter is ready. Starting the blocking RPC here
        // would consume the remote scan before DataFusion had a chance to
        // publish that filter.
        let flight_context = self.cache_key.clone().map(|key| {
            (
                Arc::clone(&self.cache_flight),
                key,
                Arc::clone(self.conn.runtime.result_cache()),
                Arc::clone(&self.cache_flight_probe),
                Arc::clone(&self.conn.runtime),
                self.conn.rpc_timeout(),
                self.catalog.clone(),
                format!("{}.{}", self.schema_name, self.function),
                self.cache_metrics.clone(),
                self.cache_capture.clone(),
            )
        });
        let stream = futures::stream::unfold(
            (rx, Some(job), flight_context, VecDeque::new()),
            move |(mut rx, mut job, flight_context, mut replay)| async move {
                if let Some((
                    flight,
                    key,
                    cache,
                    probe,
                    runtime,
                    rpc_timeout,
                    catalog,
                    function,
                    cache_metrics,
                    capture,
                )) = flight_context
                {
                    match flight.get_or_init(|| runtime.acquire_result_flight(&key)) {
                        crate::runtime::ResultFlightClaim::Producer(producer) => {
                            if let Some(capture) = &capture {
                                capture.set_flight(Arc::clone(producer));
                            }
                            if let Some(job) = job.take() {
                                tokio::task::spawn_blocking(job);
                            }
                        }
                        crate::runtime::ResultFlightClaim::Follower(waiter) => {
                            cache_metrics.waits.add(1);
                            let mut event = VgiEvent::new("cache.wait");
                            event.catalog = Some(catalog.clone());
                            event.function = Some(function.clone());
                            runtime.emit(event);
                            match waiter.clone().wait_timeout(rpc_timeout).await {
                                crate::runtime::ResultFlightOutcome::Stored => {
                                    match probe
                                        .get_or_init(|| {
                                            cache
                                                .get(&key)
                                                .or_else(|| cache.get_for_revalidation(&key))
                                        })
                                        .clone()
                                    {
                                        Some(entry) => {
                                            // Dropping the unstarted fallback job also
                                            // drops its sender, so replay closes normally.
                                            drop(job.take());
                                            if partition == 0 {
                                                replay.extend(
                                                    entry.batches().iter().cloned().map(Ok),
                                                );
                                                cache_metrics.coalesced_hits.add(1);
                                                let mut event =
                                                    VgiEvent::new("cache.coalesced_hit");
                                                event.catalog = Some(catalog);
                                                event.function = Some(function);
                                                runtime.emit(event);
                                            }
                                        }
                                        None => {
                                            // A zero-TTL revalidatable entry, short
                                            // expiry, or intervening eviction can make
                                            // a successful fill unavailable by the
                                            // time this follower wakes. Cache policy
                                            // must never turn that race into a query
                                            // failure: execute this plan normally.
                                            cache_metrics.coalesced_retries.add(1);
                                            let mut event = VgiEvent::new("cache.coalesced_retry");
                                            event.catalog = Some(catalog);
                                            event.function = Some(function);
                                            runtime.emit(event);
                                            if let Some(job) = job.take() {
                                                tokio::task::spawn_blocking(job);
                                            }
                                        }
                                    }
                                }
                                crate::runtime::ResultFlightOutcome::Aborted(reason) => {
                                    cache_metrics.coalesced_aborts.add(1);
                                    let mut event = VgiEvent::new("cache.coalesced_abort");
                                    event.catalog = Some(catalog);
                                    event.function = Some(function);
                                    event.message = Some(reason);
                                    runtime.emit(event);
                                    if let Some(job) = job.take() {
                                        tokio::task::spawn_blocking(job);
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(job) = job.take() {
                    tokio::task::spawn_blocking(job);
                }
                if let Some(item) = replay.pop_front() {
                    return Some((item, (rx, job, None, replay)));
                }
                rx.recv().await.map(|item| (item, (rx, job, None, replay)))
            },
        );
        let baseline = partition_metrics.baseline;
        let stream = stream.map(move |batch| batch.record_output(&baseline));
        Ok(Box::pin(RecordBatchStreamAdapter::new(out_schema, stream)))
    }
}
