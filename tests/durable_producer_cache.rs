// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end contract for the host-owned durable producer-scan cache.
//!
//! Run this test target once with a Unix worker and once with an HTTP worker:
//! `VGI_TEST_WORKER=unix:///...` and `VGI_TEST_WORKER=http://127.0.0.1:...`.
//! Every test uses a unique durable root while retaining one external worker,
//! so a stable `cache_nonce` proves replay did not reinvoke the producer.
//!
//! There is deliberately no zero-row adapter fixture here. Existing cacheable
//! producers advertise freshness on their first data batch, so their zero-row
//! form cannot genuinely opt in. The storage invariant is covered directly by
//! vgi-client's `commits_an_all_empty_result_from_its_declared_schema` test.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use datafusion::arrow::array::{Array, Int64Array};
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;
use vgi_client::disk_cache::DiskCacheCodec;
use vgi_client::CacheLimits;
use vgi_datafusion::{VgiDurableCacheOptions, VgiRuntime, VgiSessionOptions};

const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
const REVALIDATION_SYNC_TIMEOUT: Duration = Duration::from_secs(60);
static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestCacheRoot(PathBuf);

impl TestCacheRoot {
    fn new(label: &str) -> Self {
        let unique = ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock precedes Unix epoch")
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "vgi-datafusion-durable-{label}-{}-{timestamp}-{unique}",
            std::process::id()
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestCacheRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!(
                    "could not remove durable-cache test root {}: {error}",
                    self.0.display()
                );
            }
        }
    }
}

fn transport_location() -> Option<String> {
    let location = match std::env::var("VGI_TEST_WORKER") {
        Ok(location) if !location.trim().is_empty() => location,
        _ => {
            eprintln!("skipping: set VGI_TEST_WORKER to a Unix or HTTP example worker");
            return None;
        }
    };
    assert!(
        location.starts_with("unix://")
            || location.starts_with("http://")
            || location.starts_with("https://"),
        "durable-cache contract requires a persistent Unix or HTTP worker, got {location:?}"
    );
    Some(location)
}

fn sql_quote(value: &str) -> String {
    value.replace('\'', "''")
}

fn disk_only_options(root: &Path) -> VgiSessionOptions {
    // Keep per-entry eligibility large enough for disk capture while forcing L1
    // to evict immediately. A recreated runtime therefore cannot accidentally
    // satisfy the replay assertion from process-local Arrow batches.
    let cache_limits = CacheLimits {
        max_entry_bytes: 64 * 1024 * 1024,
        max_total_bytes: 0,
        max_entries: 0,
        ..CacheLimits::default()
    };
    VgiSessionOptions {
        cache_limits,
        durable_cache: Some(VgiDurableCacheOptions {
            root: root.to_path_buf(),
            max_bytes: 256 * 1024 * 1024,
            max_entries: 1_024,
            codec: DiskCacheCodec::default(),
        }),
        ..VgiSessionOptions::default()
    }
}

fn layered_options(root: &Path) -> VgiSessionOptions {
    VgiSessionOptions {
        durable_cache: Some(VgiDurableCacheOptions {
            root: root.to_path_buf(),
            max_bytes: 256 * 1024 * 1024,
            max_entries: 1_024,
            codec: DiskCacheCodec::default(),
        }),
        ..VgiSessionOptions::default()
    }
}

async fn attached(
    location: &str,
    root: &Path,
) -> datafusion::common::Result<(SessionContext, Arc<VgiRuntime>)> {
    attached_with_partitions(location, root, 1).await
}

async fn attached_with_partitions(
    location: &str,
    root: &Path,
    target_partitions: usize,
) -> datafusion::common::Result<(SessionContext, Arc<VgiRuntime>)> {
    attached_with_options(location, target_partitions, disk_only_options(root)).await
}

async fn attached_with_options(
    location: &str,
    target_partitions: usize,
    options: VgiSessionOptions,
) -> datafusion::common::Result<(SessionContext, Arc<VgiRuntime>)> {
    let runtime = Arc::new(VgiRuntime::try_new(options)?);
    let context = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(target_partitions)
            .with_batch_size(128)
            .with_extension(Arc::clone(&runtime)),
    );
    vgi_datafusion::sql(
        &context,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            sql_quote(location)
        ),
    )
    .await?;
    Ok((context, runtime))
}

async fn query_i64(context: &SessionContext, sql: &str) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(context, sql).await?.collect().await?;
    let values = batches
        .first()
        .expect("query returns one batch")
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("query returns Int64");
    assert_eq!(values.len(), 1, "query returns one row: {sql}");
    Ok(values.value(0))
}

async fn query_count_sum(
    context: &SessionContext,
    sql: &str,
) -> datafusion::common::Result<(i64, i64)> {
    let batches = vgi_datafusion::sql(context, sql).await?.collect().await?;
    let batch = batches.first().expect("aggregate query returns one batch");
    let value = |column| {
        batch
            .column(column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("aggregate query returns Int64")
            .value(0)
    };
    Ok((value(0), value(1)))
}

fn find_vgi_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.downcast_ref::<vgi_datafusion::VgiScanExec>().is_some() {
        return Some(Arc::clone(plan));
    }
    plan.children()
        .into_iter()
        .find_map(|child| find_vgi_scan(child))
}

fn scan_metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    plan.metrics()
        .expect("VgiScanExec exposes metrics")
        .iter()
        .filter(|metric| metric.value().name() == name)
        .map(|metric| metric.value().as_usize())
        .sum()
}

async fn query_count_sum_with_worker_scans(
    context: &SessionContext,
    sql: &str,
) -> datafusion::common::Result<((i64, i64), usize)> {
    let dataframe = vgi_datafusion::sql(context, sql).await?;
    let plan = dataframe.create_physical_plan().await?;
    let scan = find_vgi_scan(&plan).expect("query contains a VGI scan");
    let batches = collect(plan, context.task_ctx()).await?;
    let batch = batches.first().expect("aggregate query returns one batch");
    let value = |column| {
        batch
            .column(column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("aggregate query returns Int64")
            .value(0)
    };
    Ok(((value(0), value(1)), scan_metric(&scan, "worker_scans")))
}

async fn query_i64_with_worker_scans(
    context: &SessionContext,
    sql: &str,
) -> datafusion::common::Result<(i64, usize)> {
    let dataframe = vgi_datafusion::sql(context, sql).await?;
    let plan = dataframe.create_physical_plan().await?;
    let scan = find_vgi_scan(&plan).expect("query contains a VGI scan");
    let batches = collect(plan, context.task_ctx()).await?;
    let value = batches
        .first()
        .expect("query returns one batch")
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("query returns Int64")
        .value(0);
    Ok((value, scan_metric(&scan, "worker_scans")))
}

async fn wait_for_path(path: &Path) {
    // On a saturated debug-build host, opening and binding the conditional
    // scan can take substantially longer than filesystem cleanup. This is a
    // readiness bound, not the cache operation's latency contract.
    let deadline = tokio::time::Instant::now() + REVALIDATION_SYNC_TIMEOUT;
    while !path.exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn visit_tree(
    root: &Path,
    directory: &Path,
    entries: &mut BTreeMap<PathBuf, Option<u64>>,
) -> std::io::Result<()> {
    if !directory.exists() {
        return Ok(());
    }
    let mut children = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    children.sort_by_key(fs::DirEntry::file_name);
    for child in children {
        let path = child.path();
        let relative = path
            .strip_prefix(root)
            .expect("cache child remains below its root")
            .to_path_buf();
        let metadata = child.metadata()?;
        if metadata.is_dir() {
            entries.insert(relative, None);
            visit_tree(root, &path, entries)?;
        } else if metadata.is_file() {
            entries.insert(relative, Some(metadata.len()));
        }
    }
    Ok(())
}

fn tree_snapshot(root: &Path) -> std::io::Result<BTreeMap<PathBuf, Option<u64>>> {
    let mut entries = BTreeMap::new();
    visit_tree(root, root, &mut entries)?;
    Ok(entries)
}

fn regular_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    Ok(tree_snapshot(root)?
        .into_iter()
        .filter_map(|(relative, size)| size.map(|_| root.join(relative)))
        .collect())
}

async fn wait_for_tree(root: &Path, expected: &BTreeMap<PathBuf, Option<u64>>) {
    let deadline = tokio::time::Instant::now() + CLEANUP_TIMEOUT;
    loop {
        let actual = tree_snapshot(root).expect("inspect durable-cache root");
        if &actual == expected {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "durable capture left files behind\nexpected: {expected:#?}\nactual: {actual:#?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_capture_aborts(runtime: &VgiRuntime, expected: u64) {
    let deadline = tokio::time::Instant::now() + CLEANUP_TIMEOUT;
    loop {
        let actual = runtime
            .durable_result_cache()
            .expect("durable cache is configured")
            .stats()
            .expect("read durable cache stats")
            .capture_aborts;
        if actual == expected {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "expected {expected} durable capture aborts, observed {actual}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn host_root_replays_across_runtime_recreation_without_worker_reinvocation(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("recreate");

    let (cold_context, cold_runtime) = attached(&location, root.path()).await?;
    let cold = query_i64(&cold_context, "SELECT nonce FROM ex.data.cache_nonce").await?;
    assert!(
        !regular_files(root.path())?.is_empty(),
        "the host-configured durable root received no cache files"
    );
    assert_eq!(
        cold_runtime.result_cache().stats().entries,
        0,
        "the cold result must be disk-only, not retained by L1"
    );
    let cold_disk = cold_runtime
        .durable_result_cache()
        .expect("durable cache is configured")
        .stats()
        .expect("read durable cache stats");
    assert_eq!(cold_disk.entries, 1);
    assert_eq!(cold_disk.inserts, 1);
    drop(cold_context);
    drop(cold_runtime);

    let (warm_context, warm_runtime) = attached(&location, root.path()).await?;
    let warm = query_i64(&warm_context, "SELECT nonce FROM ex.data.cache_nonce").await?;
    assert_eq!(
        warm, cold,
        "a new VgiRuntime reinvoked cache_nonce instead of replaying the durable result"
    );
    assert_eq!(warm_runtime.result_cache().stats().entries, 0);
    let warm_disk = warm_runtime
        .durable_result_cache()
        .expect("durable cache is configured")
        .stats()
        .expect("read durable cache stats");
    assert_eq!(warm_disk.entries, 1);
    assert_eq!(warm_disk.hits, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn validator_survives_restart_and_not_modified_replays_disk_bytes(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("validator-restart");
    let sql = "SELECT nonce FROM ex.main.cache_revalidation_policy('not_modified')";

    let (cold_context, _) = attached(&location, root.path()).await?;
    let cold = query_i64(&cold_context, sql).await?;
    drop(cold_context);

    let (warm_context, warm_runtime) = attached(&location, root.path()).await?;
    assert_eq!(query_i64(&warm_context, sql).await?, cold);
    let disk = warm_runtime
        .durable_result_cache()
        .expect("durable cache configured")
        .stats()
        .expect("durable stats");
    assert_eq!(disk.revalidations, 1);
    assert_eq!(disk.entries, 1);
    assert!(warm_runtime.events().iter().any(|event| {
        event.kind == "cache.revalidated" && event.message.as_deref() == Some("tier=disk")
    }));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn durable_validator_revocation_evicts_stale_bytes() -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("validator-revoke");
    let sql = "SELECT nonce FROM ex.main.cache_revalidation_policy('revoke')";

    let (cold_context, _) = attached(&location, root.path()).await?;
    query_i64(&cold_context, sql).await?;
    drop(cold_context);

    let (warm_context, warm_runtime) = attached(&location, root.path()).await?;
    let error = query_i64(&warm_context, sql)
        .await
        .expect_err("no_store plus not_modified must revoke durable reuse");
    assert!(error.to_string().contains("ineligible cache control"));
    assert_eq!(
        warm_runtime
            .durable_result_cache()
            .expect("durable cache configured")
            .stats()
            .expect("durable stats")
            .entries,
        0
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn durable_validator_honors_stale_if_error_after_restart() -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("validator-stale-error");
    let sql = "SELECT nonce FROM ex.main.cache_revalidation_policy('stale_if_error')";

    let (cold_context, _) = attached(&location, root.path()).await?;
    let cold = query_i64(&cold_context, sql).await?;
    drop(cold_context);

    let (warm_context, warm_runtime) = attached(&location, root.path()).await?;
    assert_eq!(query_i64(&warm_context, sql).await?, cold);
    let disk = warm_runtime
        .durable_result_cache()
        .expect("durable cache configured")
        .stats()
        .expect("durable stats");
    assert_eq!(disk.stale_serves, 1);
    assert_eq!(disk.entries, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_revocation_removes_matching_durable_generation() -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("layered-validator-revoke");
    let policy = format!(
        "revoke_then_error:{}:{}",
        std::process::id(),
        ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let sql = format!(
        "SELECT nonce FROM ex.main.cache_revalidation_policy('{}')",
        sql_quote(&policy)
    );

    let (context, runtime) =
        attached_with_options(&location, 1, layered_options(root.path())).await?;
    query_i64(&context, &sql).await?;
    query_i64(&context, &sql)
        .await
        .expect_err("no_store plus not_modified must revoke both cache tiers");
    assert_eq!(runtime.result_cache().stats().entries, 0);
    assert_eq!(
        runtime
            .durable_result_cache()
            .expect("durable cache configured")
            .stats()
            .expect("durable stats")
            .entries,
        0
    );
    drop(context);
    drop(runtime);

    let (restarted, _) = attached(&location, root.path()).await?;
    let error = query_i64(&restarted, &sql)
        .await
        .expect_err("revoked durable bytes must not stale-serve after restart");
    assert!(
        error
            .to_string()
            .contains("injected post-revocation worker failure"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_rotation_updates_matching_durable_policy() -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("layered-validator-rotation");
    let policy = format!(
        "rotate_then_error:{}:{}",
        std::process::id(),
        ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let sql = format!(
        "SELECT nonce FROM ex.main.cache_revalidation_policy('{}')",
        sql_quote(&policy)
    );

    let (context, runtime) =
        attached_with_options(&location, 1, layered_options(root.path())).await?;
    let cold = query_i64(&context, &sql).await?;
    assert_eq!(query_i64(&context, &sql).await?, cold);
    assert_eq!(runtime.result_cache().stats().revalidations, 1);
    assert_eq!(
        runtime
            .durable_result_cache()
            .expect("durable cache configured")
            .stats()
            .expect("durable stats")
            .revalidations,
        1
    );
    drop(context);
    drop(runtime);

    let (restarted, restarted_runtime) = attached(&location, root.path()).await?;
    let error = query_i64(&restarted, &sql)
        .await
        .expect_err("withdrawn stale-if-error must not replay durable bytes");
    assert!(
        error
            .to_string()
            .contains("injected conditional v2 failure"),
        "durable validator was not rotated to v2: {error}"
    );
    assert_eq!(
        restarted_runtime
            .durable_result_cache()
            .expect("durable cache configured")
            .stats()
            .expect("durable stats")
            .stale_serves,
        0
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn in_flight_l1_limit_reduction_falls_back_to_matching_disk() -> datafusion::common::Result<()>
{
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("layered-validator-limit");
    let sync = TestCacheRoot::new("layered-validator-limit-sync");
    fs::create_dir_all(sync.path())?;
    let entered = sync.path().join("entered");
    let release = sync.path().join("release");
    let policy = format!(
        "blocked_not_modified|{}|{}",
        entered.display(),
        release.display()
    );
    let sql = format!(
        "SELECT nonce FROM ex.main.cache_revalidation_policy('{}')",
        sql_quote(&policy)
    );

    let (context, runtime) =
        attached_with_options(&location, 1, layered_options(root.path())).await?;
    let cold = query_i64(&context, &sql).await?;
    let query_context = context.clone();
    let query_sql = sql.clone();
    let revalidation = tokio::spawn(async move { query_i64(&query_context, &query_sql).await });
    wait_for_path(&entered).await;

    let mut limits = runtime.result_cache().limits();
    limits.max_entry_bytes = 0;
    runtime.result_cache().set_limits(limits);
    fs::write(&release, b"release")?;

    assert_eq!(
        revalidation
            .await
            .expect("revalidation task did not panic")?,
        cold
    );
    assert_eq!(runtime.result_cache().stats().entries, 0);
    let disk = runtime
        .durable_result_cache()
        .expect("durable cache configured")
        .stats()
        .expect("durable stats");
    assert_eq!(disk.entries, 1);
    assert_eq!(disk.revalidations, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_capture_replays_after_runtime_and_partition_count_change(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("split-recreate");
    const QUERY: &str = "SELECT count(*), sum(n) \
                         FROM ex.main.split_cacheable(n := 100, splits := 4)";

    let (cold_context, cold_runtime) = attached_with_partitions(&location, root.path(), 4).await?;
    assert_eq!(query_count_sum(&cold_context, QUERY).await?, (100, 4_950));
    let cold_cache = cold_runtime
        .durable_result_cache()
        .expect("durable cache is configured");
    let cold_stats = cold_cache.stats().expect("read durable cache stats");
    assert_eq!(
        cold_stats.entries,
        1,
        "split capture was not committed: stats={cold_stats:?}, events={:?}",
        cold_runtime.events()
    );
    assert_eq!(cold_stats.inserts, 1);
    let entries = cold_cache.entries().expect("inspect durable cache entries");
    let entry = entries
        .iter()
        .find(|entry| entry.function.ends_with("split_cacheable"))
        .expect("split_cacheable entry was durably published");
    assert_eq!(entry.rows, 100);
    assert!(
        entry.partitions > 1,
        "parallel cold capture collapsed to {} partition",
        entry.partitions
    );
    drop(cold_context);
    drop(cold_runtime);

    let (warm_context, warm_runtime) = attached_with_partitions(&location, root.path(), 1).await?;
    assert_eq!(query_count_sum(&warm_context, QUERY).await?, (100, 4_950));
    let warm_cache = warm_runtime
        .durable_result_cache()
        .expect("durable cache is configured");
    let warm_stats = warm_cache.stats().expect("read durable cache stats");
    assert_eq!(warm_stats.hits, 1, "warm scan did not use durable replay");
    assert_eq!(
        warm_stats.inserts, 0,
        "partition-count change refilled the worker result"
    );
    let warm_entry = warm_cache
        .entries()
        .expect("inspect durable cache entries")
        .into_iter()
        .find(|entry| entry.function.ends_with("split_cacheable"))
        .expect("split_cacheable entry disappeared during replay");
    assert!(warm_entry.partitions > 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_validator_requires_unanimous_executed_group_agreement(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("split-validator-unanimous");
    const QUERY: &str = "SELECT count(*), sum(n) \
                         FROM ex.main.split_cacheable( \
                           n := 100, splits := 2, policy := 'not_modified')";

    // The packer creates two execution partitions for the two split groups.
    let (cold_context, _) = attached_with_partitions(&location, root.path(), 6).await?;
    assert_eq!(query_count_sum(&cold_context, QUERY).await?, (100, 4_950));
    drop(cold_context);

    // A different target partition count must not affect the stored physical
    // substreams. Partition 0 validates both groups serially and replays the
    // flattened durable result exactly once.
    let (warm_context, warm_runtime) = attached_with_partitions(&location, root.path(), 4).await?;
    let (result, worker_scans) = query_count_sum_with_worker_scans(&warm_context, QUERY).await?;
    assert_eq!(result, (100, 4_950));
    assert_eq!(
        worker_scans, 2,
        "only the two executed split groups should vote"
    );
    let stats = warm_runtime
        .durable_result_cache()
        .expect("durable cache configured")
        .stats()
        .expect("durable stats");
    assert_eq!(stats.revalidations, 1);
    assert_eq!(stats.inserts, 0);
    assert_eq!(stats.entries, 1);
    assert!(warm_runtime.events().iter().any(|event| {
        event.kind == "cache.revalidated"
            && event.message.as_deref() == Some("split_validation=unanimous")
    }));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_split_entry_is_revoked_when_replanning_finds_zero_splits(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("split-validator-zero-transition");
    let policy = format!(
        "zero_after_cold:{}",
        root.path()
            .file_name()
            .expect("test root has a file name")
            .to_string_lossy()
    );
    let query = format!(
        "SELECT count(*) FROM ex.main.split_cacheable(\
         n := 100, splits := 4, policy := '{}')",
        sql_quote(&policy)
    );

    let (cold_context, _) = attached_with_partitions(&location, root.path(), 4).await?;
    assert_eq!(query_i64(&cold_context, &query).await?, 100);
    drop(cold_context);

    let (warm_context, warm_runtime) = attached_with_partitions(&location, root.path(), 2).await?;
    let (rows, worker_scans) = query_i64_with_worker_scans(&warm_context, &query).await?;
    assert_eq!(rows, 0, "the stale split payload was replayed");
    assert_eq!(worker_scans, 0, "a definitive zero-split plan has no work");
    let stats = warm_runtime
        .durable_result_cache()
        .expect("durable cache configured")
        .stats()
        .expect("durable stats");
    assert_eq!(stats.entries, 0, "the stale generation was not revoked");
    assert_eq!(stats.revalidations, 0);
    assert_eq!(stats.inserts, 0, "an empty no-work result was published");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn mixed_fresh_revoked_and_incompatible_split_votes_rerun_the_whole_result(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };

    for policy in ["mixed", "disagree", "revoke"] {
        let root = TestCacheRoot::new(&format!("split-validator-{policy}"));
        let query = format!(
            "SELECT count(*), sum(n) FROM ex.main.split_cacheable(\
             n := 100, splits := 4, policy := '{policy}')"
        );
        let (cold_context, _) = attached_with_partitions(&location, root.path(), 4).await?;
        assert_eq!(query_count_sum(&cold_context, &query).await?, (100, 4_950));
        drop(cold_context);

        let (warm_context, warm_runtime) =
            attached_with_partitions(&location, root.path(), 2).await?;
        let (result, worker_scans) =
            query_count_sum_with_worker_scans(&warm_context, &query).await?;
        assert_eq!(
            result,
            (100, 4_950),
            "validation rows leaked or the unconditional {policy} rerun was partial"
        );
        assert_eq!(
            worker_scans, 3,
            "{policy} should validate distinct groups before one unconditional full-token rerun"
        );
        let stats = warm_runtime
            .durable_result_cache()
            .expect("durable cache configured")
            .stats()
            .expect("durable stats");
        assert_eq!(stats.revalidations, 0, "{policy} was treated as unanimous");
        assert_eq!(stats.inserts, 1, "{policy} rerun was not durably published");
        assert_eq!(stats.entries, 1);
        assert!(!warm_runtime.events().iter().any(|event| {
            event.kind == "cache.revalidated"
                && event.message.as_deref() == Some("split_validation=unanimous")
        }));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_validation_error_fails_closed_without_stale_replay() -> datafusion::common::Result<()>
{
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("split-validator-error");
    const QUERY: &str = "SELECT count(*), sum(n) \
                         FROM ex.main.split_cacheable( \
                           n := 100, splits := 4, policy := 'error')";

    let (cold_context, _) = attached_with_partitions(&location, root.path(), 4).await?;
    assert_eq!(query_count_sum(&cold_context, QUERY).await?, (100, 4_950));
    drop(cold_context);

    let (warm_context, warm_runtime) = attached_with_partitions(&location, root.path(), 1).await?;
    let error = vgi_datafusion::sql(&warm_context, QUERY)
        .await?
        .collect()
        .await
        .expect_err("split validation error must not serve stale bytes");
    assert!(
        error
            .to_string()
            .contains("injected split validation failure"),
        "unexpected validation error: {error}"
    );
    let stats = warm_runtime
        .durable_result_cache()
        .expect("durable cache configured")
        .stats()
        .expect("durable stats");
    assert_eq!(stats.stale_serves, 0);
    assert_eq!(stats.revalidations, 0);
    assert_eq!(
        stats.entries, 1,
        "failed validation must not publish a partial"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_and_dropped_consumers_never_publish_partial_entries(
) -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("atomic");
    let (context, runtime) = attached(&location, root.path()).await?;

    // Seed one valid object so this assertion distinguishes targeted cleanup
    // from an implementation that simply removes the whole durable root.
    query_i64(&context, "SELECT nonce FROM ex.data.cache_nonce").await?;
    let baseline = tree_snapshot(root.path())?;
    let baseline_stats = runtime
        .durable_result_cache()
        .expect("durable cache is configured")
        .stats()
        .expect("read durable cache stats");

    let error = vgi_datafusion::sql(&context, "SELECT n FROM ex.data.cache_poison")
        .await?
        .collect()
        .await
        .expect_err("cache_poison must fail after its first cacheable batch");
    assert!(
        error.to_string().contains("intentional mid-stream failure"),
        "unexpected poison error: {error}"
    );
    wait_for_tree(root.path(), &baseline).await;
    wait_for_capture_aborts(&runtime, baseline_stats.capture_aborts + 1).await;
    let poison_stats = runtime
        .durable_result_cache()
        .expect("durable cache is configured")
        .stats()
        .expect("read durable cache stats");
    assert_eq!(poison_stats.entries, baseline_stats.entries);
    assert_eq!(poison_stats.inserts, baseline_stats.inserts);
    assert_eq!(
        poison_stats.capture_aborts,
        baseline_stats.capture_aborts + 1
    );

    let dataframe =
        vgi_datafusion::sql(&context, "SELECT n FROM ex.main.cache_big(rows := 250000)").await?;
    let plan = dataframe.create_physical_plan().await?;
    let mut stream = plan.execute(0, context.task_ctx())?;
    let first = tokio::time::timeout(CLEANUP_TIMEOUT, stream.next())
        .await
        .expect("cache_big did not produce its first batch")
        .expect("cache_big ended before its first batch")?;
    assert!(first.num_rows() > 0);
    drop(stream);
    wait_for_tree(root.path(), &baseline).await;
    wait_for_capture_aborts(&runtime, baseline_stats.capture_aborts + 2).await;
    let drop_stats = runtime
        .durable_result_cache()
        .expect("durable cache is configured")
        .stats()
        .expect("read durable cache stats");
    assert_eq!(drop_stats.entries, baseline_stats.entries);
    assert_eq!(drop_stats.inserts, baseline_stats.inserts);
    assert_eq!(drop_stats.capture_aborts, baseline_stats.capture_aborts + 2);

    // The previously committed entry remains usable after both aborted fills.
    query_i64(&context, "SELECT nonce FROM ex.data.cache_nonce").await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn corrupt_durable_objects_are_clean_misses_and_recompute() -> datafusion::common::Result<()>
{
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let root = TestCacheRoot::new("corrupt");

    let (cold_context, cold_runtime) = attached(&location, root.path()).await?;
    let cold = query_i64(&cold_context, "SELECT nonce FROM ex.data.cache_nonce").await?;
    drop(cold_context);
    drop(cold_runtime);

    let object_files = regular_files(root.path())?
        .into_iter()
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "arrow")
        })
        .collect::<Vec<_>>();
    assert!(
        !object_files.is_empty(),
        "durable fill published no Arrow object below {}",
        root.path().display()
    );
    let object = &object_files[0];
    let mut bytes = fs::read(object)?;
    let first = bytes
        .first_mut()
        .expect("published Arrow object must not be empty");
    *first ^= 0xff;
    fs::write(object, bytes)?;

    let (repair_context, repair_runtime) = attached(&location, root.path()).await?;
    let repaired = query_i64(&repair_context, "SELECT nonce FROM ex.data.cache_nonce").await?;
    assert_ne!(
        repaired, cold,
        "corrupt durable bytes were served instead of causing a clean miss"
    );
    let repair_stats = repair_runtime
        .durable_result_cache()
        .expect("durable cache is configured")
        .stats()
        .expect("read durable cache stats");
    assert_eq!(repair_stats.corruptions, 1);
    assert_eq!(repair_stats.inserts, 1);
    drop(repair_context);
    drop(repair_runtime);

    let (replay_context, replay_runtime) = attached(&location, root.path()).await?;
    assert_eq!(
        query_i64(&replay_context, "SELECT nonce FROM ex.data.cache_nonce").await?,
        repaired,
        "the clean recomputation was not durably published"
    );
    assert_eq!(
        replay_runtime
            .durable_result_cache()
            .expect("durable cache is configured")
            .stats()
            .expect("read durable cache stats")
            .hits,
        1
    );
    Ok(())
}
