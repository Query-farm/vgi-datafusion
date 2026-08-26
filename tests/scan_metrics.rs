// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Native DataFusion execution metrics for VGI scans and result-cache work.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::SessionContext;
use vgi_datafusion::{VgiConnection, VgiRuntime, VgiTableProvider};

fn example_worker() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("vgi-rust")
        .join("target");
    for profile in ["release", "debug"] {
        let executable = root.join(profile).join(if cfg!(windows) {
            "vgi-example-worker.exe"
        } else {
            "vgi-example-worker"
        });
        if executable.exists() {
            return Some(executable);
        }
    }
    None
}

fn find_vgi_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.downcast_ref::<vgi_datafusion::VgiScanExec>().is_some() {
        return Some(Arc::clone(plan));
    }
    plan.children()
        .into_iter()
        .find_map(|child| find_vgi_scan(child))
}

fn metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    plan.metrics()
        .expect("VgiScanExec exposes metrics")
        .iter()
        .filter(|metric| metric.value().name() == name)
        .map(|metric| metric.value().as_usize())
        .sum()
}

async fn run(ctx: &SessionContext) -> datafusion::error::Result<(Arc<dyn ExecutionPlan>, usize)> {
    let dataframe = ctx.sql("SELECT n FROM remote_cacheable").await?;
    let plan = dataframe.create_physical_plan().await?;
    let scan = find_vgi_scan(&plan).expect("query has a VGI scan");
    let batches = collect(plan, ctx.task_ctx()).await?;
    let rows = batches.iter().map(|batch| batch.num_rows()).sum();
    Ok((scan, rows))
}

#[tokio::test(flavor = "multi_thread")]
async fn scan_and_cache_metrics_are_attributed_to_each_plan() -> datafusion::error::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let runtime = Arc::new(VgiRuntime::default());
    let connection =
        VgiConnection::subprocess([worker.to_string_lossy().to_string()]).with_runtime(runtime);
    let ctx = SessionContext::new();
    ctx.register_table(
        "remote_cacheable",
        VgiTableProvider::bind_with_arguments(
            connection,
            "example",
            "main",
            "cacheable_numbers",
            vgi_client::Arguments::new().named("n", 10_i64),
        )
        .await?,
    )?;

    let (miss, rows) = run(&ctx).await?;
    assert_eq!(rows, 10);
    assert_eq!(metric(&miss, "cache_misses"), 1);
    assert_eq!(metric(&miss, "cache_hits"), 0);
    assert_eq!(metric(&miss, "cache_stores"), 1);
    assert_eq!(metric(&miss, "worker_scans"), 1);
    assert!(metric(&miss, "worker_batches") > 0);
    assert_eq!(metric(&miss, "worker_rows"), 10);
    assert!(metric(&miss, "worker_bytes") > 0);
    assert_eq!(metric(&miss, "output_rows"), 10);

    let (hit, rows) = run(&ctx).await?;
    assert_eq!(rows, 10);
    assert_eq!(metric(&hit, "cache_hits"), 1);
    assert_eq!(metric(&hit, "cache_misses"), 0);
    assert_eq!(metric(&hit, "cache_stores"), 0);
    assert_eq!(metric(&hit, "worker_scans"), 0);
    assert_eq!(metric(&hit, "worker_batches"), 0);
    assert_eq!(metric(&hit, "worker_rows"), 0);
    assert_eq!(metric(&hit, "worker_bytes"), 0);
    assert_eq!(metric(&hit, "output_rows"), 10);

    let explain = ctx
        .sql("EXPLAIN ANALYZE SELECT n FROM remote_cacheable")
        .await?
        .collect()
        .await?;
    let explain = datafusion::arrow::util::pretty::pretty_format_batches(&explain)?.to_string();
    assert!(explain.contains("VgiScanExec"), "{explain}");
    assert!(explain.contains("cache_hits=1"), "{explain}");
    assert!(explain.contains("worker_scans=0"), "{explain}");
    assert!(explain.contains("output_rows=10"), "{explain}");
    Ok(())
}
