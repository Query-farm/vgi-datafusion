// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Session-scoped services shared by every VGI attachment.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use datafusion::common::ScalarValue;
use vgi_client::{CacheLimits, ResultCache};

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
    /// View declaration plus the columns of its successfully planned
    /// DataFusion ViewTable. VGI carries comments but not a separate output
    /// schema, so the latter is captured after view planning.
    pub views: Vec<(vgi_client::dtos::ViewInfo, Vec<String>)>,
    pub functions: Vec<vgi_client::dtos::FunctionInfo>,
    pub macros: Vec<vgi_client::dtos::MacroInfo>,
    pub global_function_prefix: String,
    pub global_functions: Vec<vgi_client::dtos::FunctionInfo>,
}

/// Configuration for one DataFusion session's VGI runtime.
#[derive(Debug, Clone)]
pub struct VgiSessionOptions {
    /// Whether worker-opted-in results may be cached.
    pub cache_enabled: bool,
    /// Bounds for the in-memory result cache.
    pub cache_limits: CacheLimits,
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
    /// Human-readable detail that never contains credentials.
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
    /// Secret properties. Credentials are never copied to logs or diagnostics.
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
    events: Mutex<VecDeque<VgiEvent>>,
    plan_cache: Mutex<HashMap<PlanCacheKey, CachedPlan>>,
    plan_cache_stats: Mutex<PlanCacheStats>,
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
        Self {
            cache: Arc::new(ResultCache::new(options.cache_limits)),
            options,
            events: Mutex::new(VecDeque::new()),
            plan_cache: Mutex::new(HashMap::new()),
            plan_cache_stats: Mutex::new(PlanCacheStats::default()),
            catalog_metadata: Mutex::new(HashMap::new()),
            event_sink: None,
            secret_resolver: None,
            locality_hook: None,
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

    pub(crate) fn locality_hook(&self) -> Option<&Arc<dyn VgiLocalityHook>> {
        self.locality_hook.as_ref()
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
    pub(crate) target_partitions: usize,
    pub(crate) catalog_version: i64,
    pub(crate) at: Option<(String, String)>,
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
