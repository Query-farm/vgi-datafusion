// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Session-scoped services shared by every VGI attachment.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use datafusion::common::ScalarValue;
use vgi_client::disk_cache::{DiskCache, DiskCacheOptions};
use vgi_client::{CacheLimits, ResultCache};

pub use vgi_client::DiskCacheCodec;

/// Stable, credential-free reasons why an adapter execution cannot use the
/// VGI result cache.
///
/// These strings are an observability contract shared by producer, scalar, and
/// table-input paths. Keep them about execution shape and host policy: arguments,
/// filter values, split tokens, identities, and secrets must never be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheIneligibleReason {
    DisabledGlobal,
    DisabledAttach,
    IdentityUnresolved,
    DynamicFilter,
    Unbounded,
    OrderedSplit,
    SecretDependent,
    VolatileFunction,
    PerValueDisabled,
    StatefulExchange,
    LiteralInput,
    PartialExchange,
    OrderedSink,
    EmptyInput,
    UnsupportedShape,
}

impl CacheIneligibleReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DisabledGlobal => "disabled_global",
            Self::DisabledAttach => "disabled_attach",
            Self::IdentityUnresolved => "identity_unresolved",
            Self::DynamicFilter => "dynamic_filter",
            Self::Unbounded => "unbounded",
            Self::OrderedSplit => "ordered_split",
            Self::SecretDependent => "secret_dependent",
            Self::VolatileFunction => "volatile_function",
            Self::PerValueDisabled => "per_value_disabled",
            Self::StatefulExchange => "stateful_exchange",
            Self::LiteralInput => "literal_input",
            Self::PartialExchange => "partial_exchange",
            Self::OrderedSink => "ordered_sink",
            Self::EmptyInput => "empty_input",
            Self::UnsupportedShape => "unsupported_shape",
        }
    }
}

fn disk_cache_error(error: vgi_client::DiskCacheError) -> datafusion::common::DataFusionError {
    datafusion::common::DataFusionError::Execution(format!(
        "VGI durable cache operation failed: {error}"
    ))
}

/// Host-owned configuration for the optional durable VGI result-cache tier.
///
/// This state is deliberately not exposed as a worker setting and never enters
/// a VGI result cache key. Multiple runtimes may point at the same root; the
/// client cache's atomic refs and leases coordinate those processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VgiDurableCacheOptions {
    /// Directory containing durable objects, refs, temporary captures, and
    /// cache bookkeeping.
    pub root: PathBuf,
    /// Admission bound for committed and replay-leased encoded Arrow payloads;
    /// metadata and active captures are excluded.
    pub max_bytes: u64,
    /// Admission bound for committed references plus leased orphan objects.
    pub max_entries: usize,
    /// Arrow IPC payload codec.
    pub codec: DiskCacheCodec,
}

impl VgiDurableCacheOptions {
    /// Create durable-cache options with the client's conservative byte,
    /// entry-count, and Zstandard defaults rooted at a host-owned directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let options = DiskCacheOptions::new(root);
        Self {
            root: options.root,
            max_bytes: options.max_bytes,
            max_entries: options.max_entries,
            codec: options.codec,
        }
    }
}

/// Discovery metadata retained for SQL inspection after a VGI catalog is
/// attached. Execution registration and diagnostics consume the same worker
/// declarations, so the metadata view cannot drift from what was published.
#[derive(Debug, Clone)]
pub(crate) struct VgiCatalogMetadata {
    pub connection: crate::VgiConnection,
    pub worker_catalog: String,
    pub comment: Option<String>,
    pub tags: Vec<(String, String)>,
    pub resolved_data_version: Option<String>,
    pub resolved_implementation_version: Option<String>,
    pub schemas: Vec<vgi_client::dtos::SchemaInfo>,
    pub tables: Vec<vgi_client::dtos::TableInfo>,
    /// Branch metadata resolved on demand by catalog scans. The key is the
    /// case-normalized `(schema, table)` pair within this attached alias.
    pub table_branches: HashMap<(String, String), vgi_client::CatalogScanBranches>,
    /// View declaration plus the columns of its successfully planned
    /// DataFusion ViewTable. VGI carries comments but not a separate output
    /// schema, so the latter is captured after view planning.
    pub views: Vec<(vgi_client::dtos::ViewInfo, Vec<String>)>,
    pub functions: Vec<vgi_client::dtos::FunctionInfo>,
    pub macros: Vec<vgi_client::dtos::MacroInfo>,
    pub settings: Vec<vgi_client::SettingSpec>,
    pub global_function_prefix: String,
    pub global_functions: Vec<vgi_client::dtos::FunctionInfo>,
    /// Fully published global names owned by this live attachment. A worker
    /// may nominate more functions than DataFusion publishes when the attach
    /// opts out or an earlier registration owns a collision.
    pub published_global_functions: Vec<String>,
}

/// Configuration for one DataFusion session's VGI runtime.
#[derive(Debug, Clone)]
pub struct VgiSessionOptions {
    /// Whether worker-opted-in results may be cached.
    pub cache_enabled: bool,
    /// Bounds for the in-memory result cache.
    pub cache_limits: CacheLimits,
    /// Optional host-owned durable result-cache tier.
    pub durable_cache: Option<VgiDurableCacheOptions>,
    /// Maximum number of structured events retained for SQL/API inspection.
    pub event_history_capacity: usize,
    /// Optional timeout applied to an individual blocking RPC.
    ///
    /// There is deliberately no timeout by default. DataFusion cancellation is
    /// still propagated independently of this setting.
    pub rpc_timeout: Option<Duration>,
}

impl Default for VgiSessionOptions {
    fn default() -> Self {
        Self {
            cache_enabled: true,
            cache_limits: CacheLimits::default(),
            durable_cache: None,
            event_history_capacity: 10_000,
            rpc_timeout: None,
        }
    }
}

/// A structured lifecycle, transport, split, or cache event.
#[derive(Debug, Clone)]
pub struct VgiEvent {
    /// Wall-clock time at which the event was emitted.
    pub timestamp: SystemTime,
    /// Stable event name, such as `cache.hit` or `scan.cancelled`.
    pub kind: String,
    /// Attached catalog, when known.
    pub catalog: Option<String>,
    /// Schema-qualified worker function, when known.
    pub function: Option<String>,
    /// Physical split/partition identifier, when known.
    pub split: Option<String>,
    /// Operation duration, when the event finishes an operation.
    pub duration: Option<Duration>,
    /// Human-readable detail. Adapter-constructed structural details never
    /// contain credentials. Worker-originated log, RPC, exception, and error
    /// text is untrusted and may be retained verbatim.
    pub message: Option<String>,
}

impl VgiEvent {
    pub(crate) fn new(kind: impl Into<String>) -> Self {
        Self {
            timestamp: SystemTime::now(),
            kind: kind.into(),
            catalog: None,
            function: None,
            split: None,
            duration: None,
            message: None,
        }
    }
}

/// Receives structured VGI events as they happen.
pub trait VgiEventSink: Send + Sync + 'static {
    /// Consume one event. Implementations must not block query execution.
    fn emit(&self, event: &VgiEvent);
}

/// Resolves a worker-declared secret without exposing the host environment.
#[async_trait]
pub trait VgiSecretResolver: Send + Sync + 'static {
    /// Return one named secret for the declared type, scope, and optional name.
    async fn resolve(
        &self,
        secret_type: &str,
        scope: Option<&str>,
        name: Option<&str>,
    ) -> datafusion::common::Result<Option<VgiResolvedSecret>>;
}

/// One host-resolved secret. Fields retain Arrow types on the VGI wire.
#[derive(Debug, Clone)]
pub struct VgiResolvedSecret {
    /// Unique secret name used as the outer VGI secrets-batch field.
    pub name: String,
    /// Secret properties. The adapter never renders credentials in its
    /// constructed structural details. Verbatim worker-originated log, RPC,
    /// exception, and error text is outside that guarantee.
    pub fields: std::collections::BTreeMap<String, ScalarValue>,
}

/// Locality facts for one VGI split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VgiSplitLocality {
    /// Index of the split in the VGI plan.
    pub split_index: usize,
    /// Worker-advertised location names, in preference order.
    pub locations: Vec<String>,
}

/// Allows a distributed DataFusion host to consume VGI locality metadata.
pub trait VgiLocalityHook: Send + Sync + 'static {
    /// Observe the locality choices for a newly planned scan.
    fn planned(&self, catalog: &str, function: &str, splits: &[VgiSplitLocality]);
}

/// Mutable services shared by all VGI catalogs in a DataFusion session.
pub struct VgiRuntime {
    options: VgiSessionOptions,
    cache: Arc<ResultCache>,
    disk_cache: Option<Arc<DiskCache>>,
    events: Mutex<VecDeque<VgiEvent>>,
    plan_cache: Mutex<HashMap<PlanCacheKey, CachedPlan>>,
    plan_cache_stats: Mutex<PlanCacheStats>,
    result_flights: Arc<ResultFlightRegistry>,
    exchange_cache_stats: Mutex<ExchangeCacheStats>,
    /// Function identities that have advertised `vgi.cache.per_value` on a
    /// live exchange. This avoids row-by-row IPC hashing for ordinary scalars
    /// that never opted into that cache tier.
    per_value_opt_ins: Mutex<HashSet<Vec<u8>>>,
    /// Snapshot of DataFusion's `vgi.*` configuration at the start of the
    /// current adapter SQL call. Bind requests encode this snapshot as typed
    /// IPC rather than consulting process-global state.
    session_settings: Mutex<crate::VgiSettings>,
    catalog_metadata: Mutex<HashMap<String, VgiCatalogMetadata>>,
    event_sink: Option<Arc<dyn VgiEventSink>>,
    secret_resolver: Option<Arc<dyn VgiSecretResolver>>,
    locality_hook: Option<Arc<dyn VgiLocalityHook>>,
}

impl fmt::Debug for VgiRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VgiRuntime")
            .field("options", &self.options)
            .field("cache_stats", &self.cache.stats())
            .field("disk_cache", &self.disk_cache.is_some())
            .field("event_sink", &self.event_sink.is_some())
            .field("secret_resolver", &self.secret_resolver.is_some())
            .field("locality_hook", &self.locality_hook.is_some())
            .finish()
    }
}

impl Default for VgiRuntime {
    fn default() -> Self {
        Self::new(VgiSessionOptions::default())
    }
}

impl VgiRuntime {
    /// Create a session runtime with bounded memory and no external callbacks.
    pub fn new(options: VgiSessionOptions) -> Self {
        Self::try_new(options).expect("invalid VGI durable-cache configuration")
    }

    /// Create a session runtime, returning durable-cache configuration and I/O
    /// failures to hosts that opt into the disk tier.
    pub fn try_new(options: VgiSessionOptions) -> datafusion::common::Result<Self> {
        let session_settings = crate::VgiSettings::with_cache_limits(options.cache_limits);
        let disk_cache = options
            .durable_cache
            .as_ref()
            .map(|options| {
                DiskCache::open(DiskCacheOptions {
                    root: options.root.clone(),
                    max_bytes: options.max_bytes,
                    max_entries: options.max_entries,
                    codec: options.codec,
                })
                .map(Arc::new)
                .map_err(|error| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "could not open VGI durable cache at {}: {error}",
                        options.root.display()
                    ))
                })
            })
            .transpose()?;
        Ok(Self {
            cache: Arc::new(ResultCache::new(options.cache_limits)),
            disk_cache,
            options,
            events: Mutex::new(VecDeque::new()),
            plan_cache: Mutex::new(HashMap::new()),
            plan_cache_stats: Mutex::new(PlanCacheStats::default()),
            result_flights: Arc::new(ResultFlightRegistry::default()),
            exchange_cache_stats: Mutex::new(ExchangeCacheStats::default()),
            per_value_opt_ins: Mutex::new(HashSet::new()),
            session_settings: Mutex::new(session_settings),
            catalog_metadata: Mutex::new(HashMap::new()),
            event_sink: None,
            secret_resolver: None,
            locality_hook: None,
        })
    }

    pub(crate) fn set_table_branches(
        &self,
        alias: &str,
        schema: &str,
        table: &str,
        branches: vgi_client::CatalogScanBranches,
    ) {
        if let Ok(mut catalogs) = self.catalog_metadata.lock() {
            if let Some(metadata) = catalogs.get_mut(alias) {
                metadata.table_branches.insert(
                    (schema.to_ascii_lowercase(), table.to_ascii_lowercase()),
                    branches,
                );
            }
        }
    }

    /// Install a non-blocking event sink.
    #[must_use]
    pub fn with_event_sink(mut self, sink: Arc<dyn VgiEventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }

    /// Install the host's explicit secret resolver.
    #[must_use]
    pub fn with_secret_resolver(mut self, resolver: Arc<dyn VgiSecretResolver>) -> Self {
        self.secret_resolver = Some(resolver);
        self
    }

    /// Install the host scheduler's locality observer.
    #[must_use]
    pub fn with_locality_hook(mut self, hook: Arc<dyn VgiLocalityHook>) -> Self {
        self.locality_hook = Some(hook);
        self
    }

    /// Runtime configuration.
    pub fn options(&self) -> &VgiSessionOptions {
        &self.options
    }

    /// The session's bounded memory cache.
    pub fn result_cache(&self) -> &Arc<ResultCache> {
        &self.cache
    }

    /// The optional durable result-cache tier.
    pub fn durable_result_cache(&self) -> Option<&Arc<DiskCache>> {
        self.disk_cache.as_ref()
    }

    /// Flush both result-cache tiers and return the number of removed entries.
    pub fn flush_result_cache(&self) -> datafusion::common::Result<usize> {
        let mut removed = self.cache.flush_all();
        if let Some(cache) = &self.disk_cache {
            removed = removed.saturating_add(cache.flush_all().map_err(disk_cache_error)?);
        }
        Ok(removed)
    }

    /// Reap expired or over-budget entries from both result-cache tiers.
    pub fn reap_result_cache(&self) -> datafusion::common::Result<usize> {
        let mut removed = self.cache.reap();
        if let Some(cache) = &self.disk_cache {
            removed = removed.saturating_add(cache.reap().map_err(disk_cache_error)?);
        }
        Ok(removed)
    }

    /// A snapshot of retained events, oldest first.
    pub fn events(&self) -> Vec<VgiEvent> {
        self.events.lock().unwrap().iter().cloned().collect()
    }

    /// Remove retained events and return the number removed.
    pub fn clear_events(&self) -> usize {
        let mut events = self.events.lock().unwrap();
        let count = events.len();
        events.clear();
        count
    }

    pub(crate) fn set_catalog_metadata(&self, alias: &str, metadata: VgiCatalogMetadata) {
        self.catalog_metadata
            .lock()
            .unwrap()
            .insert(alias.to_ascii_lowercase(), metadata);
    }

    pub(crate) fn remove_catalog_metadata(&self, alias: &str) {
        self.catalog_metadata
            .lock()
            .unwrap()
            .remove(&alias.to_ascii_lowercase());
    }

    pub(crate) fn catalog_metadata(&self) -> Vec<(String, VgiCatalogMetadata)> {
        self.catalog_metadata
            .lock()
            .unwrap()
            .iter()
            .map(|(alias, metadata)| (alias.clone(), metadata.clone()))
            .collect()
    }

    pub(crate) fn emit(&self, event: VgiEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(&event);
        }
        let capacity = self.options.event_history_capacity;
        if capacity == 0 {
            return;
        }
        let mut events = self.events.lock().unwrap();
        while events.len() >= capacity {
            events.pop_front();
        }
        events.push_back(event);
    }

    pub(crate) fn emit_cache_ineligible(
        &self,
        catalog: &str,
        function: &str,
        reason: CacheIneligibleReason,
    ) {
        let mut event = VgiEvent::new("cache.ineligible");
        event.catalog = Some(catalog.to_string());
        event.function = Some(function.to_string());
        event.message = Some(format!("reason={}", reason.as_str()));
        self.emit(event);
    }

    pub(crate) fn locality_hook(&self) -> Option<&Arc<dyn VgiLocalityHook>> {
        self.locality_hook.as_ref()
    }

    pub(crate) fn has_per_value_opt_in(&self, identity: &[u8]) -> bool {
        self.per_value_opt_ins.lock().unwrap().contains(identity)
    }

    pub(crate) fn note_per_value_opt_in(&self, identity: Vec<u8>) {
        self.per_value_opt_ins.lock().unwrap().insert(identity);
    }

    pub(crate) fn replace_session_settings(&self, settings: crate::VgiSettings) {
        let limits = settings
            .adapter_settings()
            .result_cache_limits(self.options.cache_limits.default_ttl);
        self.cache.set_limits(limits);
        *self.session_settings.lock().unwrap() = settings;
    }

    pub(crate) fn session_settings(&self) -> crate::VgiSettings {
        self.session_settings.lock().unwrap().clone()
    }

    pub(crate) fn adapter_settings(&self) -> crate::settings::VgiAdapterSettings {
        self.session_settings.lock().unwrap().adapter_settings()
    }

    pub(crate) fn session_settings_identity(&self, catalog: &str) -> Vec<u8> {
        let declared_adapter_settings = self
            .catalog_metadata
            .lock()
            .unwrap()
            .iter()
            .filter(|(alias, metadata)| {
                alias.eq_ignore_ascii_case(catalog)
                    || metadata.worker_catalog.eq_ignore_ascii_case(catalog)
            })
            .flat_map(|(_, metadata)| metadata.settings.iter())
            .filter(|setting| crate::settings::is_adapter_setting(&setting.name))
            .map(|setting| setting.name.to_ascii_lowercase())
            .collect::<HashSet<_>>();
        let settings = self.session_settings.lock().unwrap();
        let mut out = Vec::new();
        for (name, value) in settings.values() {
            // These settings control only local admission and execution shape.
            // They cannot change worker result bytes and therefore must not
            // fragment either cache tier. A worker may independently declare
            // the same name, however; in that catalog the value is forwarded
            // and remains part of result identity.
            if crate::settings::is_adapter_setting(name)
                && !declared_adapter_settings.contains(name)
            {
                continue;
            }
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        out
    }

    pub(crate) fn note_exchange_cache_hit(&self, bytes_served: usize) {
        let mut stats = self.exchange_cache_stats.lock().unwrap();
        stats.hits = stats.hits.saturating_add(1);
        stats.bytes_served = stats
            .bytes_served
            .saturating_add(u64::try_from(bytes_served).unwrap_or(u64::MAX));
    }

    pub(crate) fn note_exchange_cache_store(&self) {
        let mut stats = self.exchange_cache_stats.lock().unwrap();
        stats.stores = stats.stores.saturating_add(1);
    }

    /// Snapshot cache activity attributable to scalar/table-input exchanges.
    pub fn exchange_cache_stats(&self) -> ExchangeCacheStats {
        *self.exchange_cache_stats.lock().unwrap()
    }

    pub(crate) fn secret_resolver(&self) -> Option<&Arc<dyn VgiSecretResolver>> {
        self.secret_resolver.as_ref()
    }

    pub(crate) fn plan_get(&self, key: &PlanCacheKey) -> Option<vgi_client::ScanPlan> {
        let now = Instant::now();
        let mut plans = self.plan_cache.lock().unwrap();
        let hit = plans
            .get(key)
            .filter(|cached| now.duration_since(cached.stored_at) < cached.max_age)
            .map(|cached| cached.plan.clone());
        if hit.is_none() {
            plans.remove(key);
        }
        let mut stats = self.plan_cache_stats.lock().unwrap();
        if hit.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        stats.entries = plans.len();
        hit
    }

    pub(crate) fn plan_insert(
        &self,
        key: PlanCacheKey,
        plan: vgi_client::ScanPlan,
        max_age: Duration,
    ) {
        if max_age.is_zero() || plan.scope != vgi_protocol::cache_control::CACHE_SCOPE_CATALOG {
            return;
        }
        let mut plans = self.plan_cache.lock().unwrap();
        plans.insert(
            key,
            CachedPlan {
                plan,
                stored_at: Instant::now(),
                max_age,
            },
        );
        let mut stats = self.plan_cache_stats.lock().unwrap();
        stats.inserts += 1;
        stats.entries = plans.len();
    }

    /// Snapshot split-plan cache counters.
    pub fn plan_cache_stats(&self) -> PlanCacheStats {
        *self.plan_cache_stats.lock().unwrap()
    }

    /// Flush all reusable split plans and return the number removed.
    pub fn flush_plan_cache(&self) -> usize {
        let mut plans = self.plan_cache.lock().unwrap();
        let count = plans.len();
        plans.clear();
        self.plan_cache_stats.lock().unwrap().entries = 0;
        count
    }

    pub(crate) fn acquire_result_flight(&self, key: &vgi_client::CacheKey) -> ResultFlightClaim {
        self.result_flights.acquire(key)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResultFlightOutcome {
    Stored,
    Aborted(String),
}

#[derive(Debug)]
struct ResultFlight {
    outcome: tokio::sync::watch::Sender<Option<ResultFlightOutcome>>,
    blocking_outcome: Mutex<Option<ResultFlightOutcome>>,
    blocking_ready: std::sync::Condvar,
}

#[derive(Debug, Default)]
struct ResultFlightRegistry {
    flights: Mutex<HashMap<vgi_client::CacheKey, Arc<ResultFlight>>>,
}

impl ResultFlightRegistry {
    fn acquire(self: &Arc<Self>, key: &vgi_client::CacheKey) -> ResultFlightClaim {
        let mut flights = self.flights.lock().unwrap();
        if let Some(flight) = flights.get(key) {
            return ResultFlightClaim::Follower(ResultFlightWaiter {
                outcome: flight.outcome.subscribe(),
                flight: Arc::clone(flight),
            });
        }
        let (outcome, _) = tokio::sync::watch::channel(None);
        let flight = Arc::new(ResultFlight {
            outcome,
            blocking_outcome: Mutex::new(None),
            blocking_ready: std::sync::Condvar::new(),
        });
        flights.insert(key.clone(), Arc::clone(&flight));
        ResultFlightClaim::Producer(Arc::new(ResultFlightProducer {
            registry: Arc::downgrade(self),
            key: key.clone(),
            flight,
            finished: Mutex::new(false),
        }))
    }

    fn finish(
        &self,
        key: &vgi_client::CacheKey,
        flight: &Arc<ResultFlight>,
        outcome: ResultFlightOutcome,
    ) {
        let mut flights = self.flights.lock().unwrap();
        if flights
            .get(key)
            .is_some_and(|active| Arc::ptr_eq(active, flight))
        {
            flights.remove(key);
        }
        drop(flights);
        *flight.blocking_outcome.lock().unwrap() = Some(outcome.clone());
        flight.blocking_ready.notify_all();
        flight.outcome.send_replace(Some(outcome));
    }
}

#[derive(Debug)]
pub(crate) enum ResultFlightClaim {
    Producer(Arc<ResultFlightProducer>),
    Follower(ResultFlightWaiter),
}

#[derive(Debug, Clone)]
pub(crate) struct ResultFlightWaiter {
    outcome: tokio::sync::watch::Receiver<Option<ResultFlightOutcome>>,
    flight: Arc<ResultFlight>,
}

impl ResultFlightWaiter {
    pub(crate) async fn wait(mut self) -> ResultFlightOutcome {
        loop {
            if let Some(outcome) = self.outcome.borrow_and_update().clone() {
                return outcome;
            }
            if self.outcome.changed().await.is_err() {
                return ResultFlightOutcome::Aborted("cache producer disappeared".to_string());
            }
        }
    }

    pub(crate) async fn wait_timeout(self, timeout: Option<Duration>) -> ResultFlightOutcome {
        let Some(timeout) = timeout else {
            return self.wait().await;
        };
        match tokio::time::timeout(timeout, self.wait()).await {
            Ok(outcome) => outcome,
            Err(_) => {
                ResultFlightOutcome::Aborted("cache flight wait exceeded rpc_timeout".to_string())
            }
        }
    }

    /// Wait from the adapter's existing blocking RPC paths. Scalar and
    /// table-input exchanges are synchronous already, so this does not block a
    /// newly-async execution path; it only replaces duplicate worker calls.
    pub(crate) fn wait_blocking_timeout(&self, timeout: Option<Duration>) -> ResultFlightOutcome {
        let mut outcome = self.flight.blocking_outcome.lock().unwrap();
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        loop {
            if let Some(outcome) = outcome.clone() {
                return outcome;
            }
            if let Some(deadline) = deadline {
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    return ResultFlightOutcome::Aborted(
                        "cache flight wait exceeded rpc_timeout".to_string(),
                    );
                };
                let (guard, timed) = self
                    .flight
                    .blocking_ready
                    .wait_timeout(outcome, remaining)
                    .unwrap();
                outcome = guard;
                if timed.timed_out() && outcome.is_none() {
                    return ResultFlightOutcome::Aborted(
                        "cache flight wait exceeded rpc_timeout".to_string(),
                    );
                }
            } else {
                outcome = self.flight.blocking_ready.wait(outcome).unwrap();
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResultFlightProducer {
    registry: Weak<ResultFlightRegistry>,
    key: vgi_client::CacheKey,
    flight: Arc<ResultFlight>,
    finished: Mutex<bool>,
}

impl ResultFlightProducer {
    pub(crate) fn stored(&self) {
        self.finish(ResultFlightOutcome::Stored);
    }

    pub(crate) fn abort(&self, reason: impl Into<String>) {
        self.finish(ResultFlightOutcome::Aborted(reason.into()));
    }

    fn finish(&self, outcome: ResultFlightOutcome) {
        let mut finished = self.finished.lock().unwrap();
        if *finished {
            return;
        }
        *finished = true;
        if let Some(registry) = self.registry.upgrade() {
            registry.finish(&self.key, &self.flight, outcome);
        } else {
            *self.flight.blocking_outcome.lock().unwrap() = Some(outcome.clone());
            self.flight.blocking_ready.notify_all();
            self.flight.outcome.send_replace(Some(outcome));
        }
    }
}

impl Drop for ResultFlightProducer {
    fn drop(&mut self) {
        let finished = self.finished.get_mut().map(|value| *value).unwrap_or(true);
        if !finished {
            if let Some(registry) = self.registry.upgrade() {
                registry.finish(
                    &self.key,
                    &self.flight,
                    ResultFlightOutcome::Aborted("cache producer dropped".to_string()),
                );
            } else {
                let outcome = ResultFlightOutcome::Aborted("cache producer dropped".to_string());
                *self.flight.blocking_outcome.lock().unwrap() = Some(outcome.clone());
                self.flight.blocking_ready.notify_all();
                self.flight.outcome.send_replace(Some(outcome));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PlanCacheKey {
    pub(crate) identity_scope: String,
    pub(crate) worker_label: String,
    pub(crate) function: String,
    pub(crate) arguments: Vec<u8>,
    pub(crate) projection: Option<Vec<i64>>,
    pub(crate) filters: Option<Vec<u8>>,
    pub(crate) row_limit: Option<i64>,
    pub(crate) sample: Option<Vec<u8>>,
    pub(crate) target_partitions: usize,
    pub(crate) catalog_version: i64,
    pub(crate) at: Option<(String, String)>,
    pub(crate) settings: Vec<u8>,
    pub(crate) attach_options: Vec<u8>,
}

struct CachedPlan {
    plan: vgi_client::ScanPlan,
    stored_at: Instant,
    max_age: Duration,
}

/// Counters for reusable VGI split plans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanCacheStats {
    /// Reused plans.
    pub hits: u64,
    /// Lookups requiring fresh planning.
    pub misses: u64,
    /// Plans inserted.
    pub inserts: u64,
    /// Plans currently retained.
    pub entries: usize,
}

/// Session-lifetime counters for worker-opted-in exchange result caching.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExchangeCacheStats {
    /// Exchange input units served from cache.
    pub hits: u64,
    /// Exchange output units inserted into cache.
    pub stores: u64,
    /// Approximate retained Arrow bytes returned by exchange cache hits.
    pub bytes_served: u64,
}

#[cfg(test)]
mod result_flight_tests {
    use super::*;

    #[test]
    fn cache_ineligible_events_use_stable_sanitized_reasons() {
        let runtime = VgiRuntime::default();
        runtime.emit_cache_ineligible(
            "example",
            "main.cached",
            CacheIneligibleReason::IdentityUnresolved,
        );

        let events = runtime.events();
        let [event] = events.as_slice() else {
            panic!("expected exactly one event")
        };
        assert_eq!(event.kind, "cache.ineligible");
        assert_eq!(event.catalog.as_deref(), Some("example"));
        assert_eq!(event.function.as_deref(), Some("main.cached"));
        assert_eq!(event.message.as_deref(), Some("reason=identity_unresolved"));
        assert_eq!(event.split, None);
    }

    #[test]
    fn cache_ineligible_reason_vocabulary_is_stable() {
        let reasons = [
            (CacheIneligibleReason::DisabledGlobal, "disabled_global"),
            (CacheIneligibleReason::DisabledAttach, "disabled_attach"),
            (
                CacheIneligibleReason::IdentityUnresolved,
                "identity_unresolved",
            ),
            (CacheIneligibleReason::DynamicFilter, "dynamic_filter"),
            (CacheIneligibleReason::Unbounded, "unbounded"),
            (CacheIneligibleReason::OrderedSplit, "ordered_split"),
            (CacheIneligibleReason::SecretDependent, "secret_dependent"),
            (CacheIneligibleReason::VolatileFunction, "volatile_function"),
            (
                CacheIneligibleReason::PerValueDisabled,
                "per_value_disabled",
            ),
            (CacheIneligibleReason::StatefulExchange, "stateful_exchange"),
            (CacheIneligibleReason::LiteralInput, "literal_input"),
            (CacheIneligibleReason::PartialExchange, "partial_exchange"),
            (CacheIneligibleReason::OrderedSink, "ordered_sink"),
            (CacheIneligibleReason::EmptyInput, "empty_input"),
            (CacheIneligibleReason::UnsupportedShape, "unsupported_shape"),
        ];
        for (reason, expected) in reasons {
            assert_eq!(reason.as_str(), expected);
        }
    }

    #[test]
    fn durable_options_follow_client_defaults() {
        let options = VgiDurableCacheOptions::new("durable-cache");
        let client = DiskCacheOptions::new("durable-cache");
        assert_eq!(options.root, client.root);
        assert_eq!(options.max_bytes, client.max_bytes);
        assert_eq!(options.max_entries, client.max_entries);
        assert_eq!(options.codec, client.codec);
    }

    #[test]
    fn exchange_cache_counters_are_session_scoped_and_cumulative() {
        let runtime = VgiRuntime::default();
        runtime.note_exchange_cache_store();
        runtime.note_exchange_cache_store();
        runtime.note_exchange_cache_hit(40);
        runtime.note_exchange_cache_hit(2);
        assert_eq!(
            runtime.exchange_cache_stats(),
            ExchangeCacheStats {
                hits: 2,
                stores: 2,
                bytes_served: 42,
            }
        );
    }

    #[test]
    fn sql_limit_overrides_reset_to_runtime_constructor_limits() {
        let constructor_limits = CacheLimits {
            max_entry_bytes: 17,
            max_total_bytes: 31,
            max_entries: 2,
            default_ttl: Duration::from_secs(9),
        };
        let runtime = VgiRuntime::new(VgiSessionOptions {
            cache_limits: constructor_limits,
            ..VgiSessionOptions::default()
        });

        runtime.replace_session_settings(crate::VgiSettings::with_cache_limits(constructor_limits));
        assert_eq!(runtime.result_cache().limits(), constructor_limits);

        let mut settings = crate::VgiSettings::with_cache_limits(constructor_limits);
        settings
            .set_value(crate::settings::RESULT_CACHE_MAX_ENTRY_BYTES, "1")
            .unwrap();
        runtime.replace_session_settings(settings.clone());
        assert_eq!(runtime.result_cache().limits().max_entry_bytes, 1);
        assert_eq!(
            runtime.result_cache().limits().default_ttl,
            Duration::from_secs(9)
        );

        settings.reset_value(crate::settings::RESULT_CACHE_MAX_ENTRY_BYTES);
        runtime.replace_session_settings(settings);
        assert_eq!(runtime.result_cache().limits(), constructor_limits);
    }

    fn key(function: &str) -> vgi_client::CacheKey {
        vgi_client::CacheKey {
            catalog: "example".to_string(),
            identity_scope: "example:anonymous".to_string(),
            worker_label: "fixture".to_string(),
            function: function.to_string(),
            arguments: Vec::new(),
            projection: None,
            filters: None,
            catalog_version: 1,
            at: None,
            settings: Vec::new(),
            attach_options: Vec::new(),
            row_limit: None,
            ordering: None,
            sample: None,
            plan: None,
        }
    }

    #[tokio::test]
    async fn identical_misses_share_one_producer_until_it_stores() {
        let runtime = VgiRuntime::default();
        let key = key("main.cache_nonce");
        let producer = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Producer(producer) => producer,
            ResultFlightClaim::Follower(_) => panic!("first claim must produce"),
        };
        let follower = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Follower(follower) => follower,
            ResultFlightClaim::Producer(_) => panic!("second claim must follow"),
        };

        producer.stored();
        assert_eq!(follower.wait().await, ResultFlightOutcome::Stored);
        assert!(matches!(
            runtime.acquire_result_flight(&key),
            ResultFlightClaim::Producer(_)
        ));
    }

    #[tokio::test]
    async fn dropping_a_producer_wakes_followers_without_a_store() {
        let runtime = VgiRuntime::default();
        let key = key("main.cancelled");
        let producer = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Producer(producer) => producer,
            ResultFlightClaim::Follower(_) => panic!("first claim must produce"),
        };
        let follower = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Follower(follower) => follower,
            ResultFlightClaim::Producer(_) => panic!("second claim must follow"),
        };

        drop(producer);
        assert!(matches!(
            follower.wait().await,
            ResultFlightOutcome::Aborted(reason) if reason == "cache producer dropped"
        ));
        assert!(matches!(
            runtime.acquire_result_flight(&key),
            ResultFlightClaim::Producer(_)
        ));
    }

    #[test]
    fn blocking_exchange_follower_observes_the_producer_outcome() {
        let runtime = VgiRuntime::default();
        let key = key("main.blocking");
        let producer = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Producer(producer) => producer,
            ResultFlightClaim::Follower(_) => panic!("first claim must produce"),
        };
        let follower = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Follower(follower) => follower,
            ResultFlightClaim::Producer(_) => panic!("second claim must follow"),
        };
        producer.stored();
        assert_eq!(
            follower.wait_blocking_timeout(None),
            ResultFlightOutcome::Stored
        );
    }

    #[test]
    fn blocking_exchange_follower_honors_rpc_timeout() {
        let runtime = VgiRuntime::default();
        let key = key("main.blocking_timeout");
        let _producer = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Producer(producer) => producer,
            ResultFlightClaim::Follower(_) => panic!("first claim must produce"),
        };
        let follower = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Follower(follower) => follower,
            ResultFlightClaim::Producer(_) => panic!("second claim must follow"),
        };
        assert!(matches!(
            follower.wait_blocking_timeout(Some(Duration::from_millis(1))),
            ResultFlightOutcome::Aborted(reason) if reason.contains("rpc_timeout")
        ));
    }

    #[tokio::test]
    async fn async_scan_follower_honors_rpc_timeout() {
        let runtime = VgiRuntime::default();
        let key = key("main.async_timeout");
        let _producer = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Producer(producer) => producer,
            ResultFlightClaim::Follower(_) => panic!("first claim must produce"),
        };
        let follower = match runtime.acquire_result_flight(&key) {
            ResultFlightClaim::Follower(follower) => follower,
            ResultFlightClaim::Producer(_) => panic!("second claim must follow"),
        };
        assert!(matches!(
            follower.wait_timeout(Some(Duration::from_millis(1))).await,
            ResultFlightOutcome::Aborted(reason) if reason.contains("rpc_timeout")
        ));
    }
}
