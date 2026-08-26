// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Production-shaped cancellation and backpressure checks for real transports.
//!
//! Run once with a Unix `VGI_TEST_WORKER` and once with an HTTP one. The slow
//! producer emits one row per worker tick, which makes the adapter's bounded
//! queue measurable: after the consumer stops polling, at most the two queued
//! batches plus the one blocked send may have been fetched from the worker.

use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, Int64Array};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;
use vgi_datafusion::{VgiRuntime, VgiScanExec};

const EVENT_TIMEOUT: Duration = Duration::from_secs(5);
// These tests intentionally stall blocking worker calls. Running them against
// one small HTTP fixture concurrently would measure server thread-pool
// saturation rather than the adapter behavior each test names.
static TRANSPORT_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn attached() -> datafusion::common::Result<Option<(SessionContext, Arc<VgiRuntime>)>> {
    attached_with_options("").await
}

async fn attached_with_options(
    attach_options: &str,
) -> datafusion::common::Result<Option<(SessionContext, Arc<VgiRuntime>)>> {
    let Ok(worker) = std::env::var("VGI_TEST_WORKER") else {
        eprintln!("skipping: set VGI_TEST_WORKER to a Unix or HTTP example worker");
        return Ok(None);
    };
    assert!(
        worker.starts_with("unix://")
            || worker.starts_with("http://")
            || worker.starts_with("https://"),
        "hardening test requires a real Unix or HTTP transport, got {worker:?}"
    );

    let runtime = Arc::new(VgiRuntime::default());
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::clone(&runtime)),
    );
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}' {attach_options})",
            worker.replace('\'', "''"),
        ),
    )
    .await?;
    Ok(Some((ctx, runtime)))
}

#[tokio::test(flavor = "multi_thread")]
async fn stalled_rpc_times_out_and_transport_recovers() -> datafusion::common::Result<()> {
    let _guard = TRANSPORT_TEST_LOCK.lock().await;
    let Some((ctx, _runtime)) = attached_with_options(", rpc_timeout 1").await? else {
        return Ok(());
    };

    let started = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(4),
        vgi_datafusion::sql(
            &ctx,
            "SELECT n FROM ex.main.slow_cancellable(\
                 '', sleep_ms := 5000, count := 1\
             )",
        )
        .await?
        .collect(),
    )
    .await
    .expect("transport rpc_timeout did not interrupt the stalled worker RPC");
    assert!(result.is_err(), "stalled worker RPC unexpectedly succeeded");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "transport timeout exceeded its outer safety bound"
    );

    let sum = tokio::time::timeout(Duration::from_secs(3), healthy_sum(&ctx))
        .await
        .expect("timed-out connection blocked the next pool checkout")?;
    assert_eq!(sum, 45, "transport was unhealthy after RPC timeout");
    Ok(())
}

fn find_vgi_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.downcast_ref::<VgiScanExec>().is_some() {
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

async fn wait_for_metric(plan: &Arc<dyn ExecutionPlan>, name: &str, at_least: usize) -> usize {
    let deadline = Instant::now() + EVENT_TIMEOUT;
    loop {
        let value = metric(plan, name);
        if value >= at_least || Instant::now() >= deadline {
            return value;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_event(runtime: &VgiRuntime, message: &str) {
    let deadline = Instant::now() + EVENT_TIMEOUT;
    loop {
        if runtime.events().iter().any(|event| {
            event.kind == "scan.cancelled"
                && event.function.as_deref() == Some("main.slow_cancellable")
                && event.message.as_deref() == Some(message)
        }) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for scan.cancelled ({message}); events: {:?}",
            runtime.events()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn healthy_sum(ctx: &SessionContext) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT sum(n) FROM ex.main.sequence(10, batch_size := 2)",
    )
    .await?
    .collect()
    .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("sum is Int64")
        .value(0))
}

#[tokio::test(flavor = "multi_thread")]
async fn slow_consumer_is_bounded_and_drop_cancels_cleanly() -> datafusion::common::Result<()> {
    let _guard = TRANSPORT_TEST_LOCK.lock().await;
    let Some((ctx, runtime)) = attached().await? else {
        return Ok(());
    };
    runtime.clear_events();

    let dataframe = vgi_datafusion::sql(
        &ctx,
        "SELECT n FROM ex.main.slow_cancellable('', sleep_ms := 1, count := 1000000)",
    )
    .await?;
    let plan = dataframe.create_physical_plan().await?;
    let scan = find_vgi_scan(&plan).expect("query has a VgiScanExec");
    let mut stream = plan.execute(0, ctx.task_ctx())?;

    let first = tokio::time::timeout(EVENT_TIMEOUT, stream.next())
        .await
        .expect("slow producer did not emit its first batch")
        .expect("slow producer ended before its first batch")?;
    assert_eq!(first.num_rows(), 1);

    // The VGI scan's channel has capacity two. With the first batch consumed,
    // the worker can queue two more and fetch one batch whose send blocks: four
    // worker batches total, independent of how long the consumer stays idle.
    assert_eq!(wait_for_metric(&scan, "worker_batches", 4).await, 4);
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        metric(&scan, "worker_batches"),
        4,
        "a stalled consumer must backpressure the blocking worker scan"
    );

    drop(stream);
    wait_for_event(&runtime, "consumer dropped").await;
    assert_eq!(
        healthy_sum(&ctx).await?,
        45,
        "attachment poisoned by cancel"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn limit_cancels_early_and_expired_split_does_not_poison_transport(
) -> datafusion::common::Result<()> {
    let _guard = TRANSPORT_TEST_LOCK.lock().await;
    let Some((ctx, runtime)) = attached().await? else {
        return Ok(());
    };
    runtime.clear_events();

    let batches = tokio::time::timeout(
        EVENT_TIMEOUT,
        vgi_datafusion::sql(
            &ctx,
            "SELECT n FROM ex.main.slow_cancellable(\
                 '', sleep_ms := 5, count := 1000000\
             ) LIMIT 3",
        )
        .await?
        .collect(),
    )
    .await
    .expect("LIMIT must not drain the million-row producer")?;
    let values = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("n is Int64")
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec![0, 1, 2]);
    wait_for_event(&runtime, "limit satisfied").await;

    let error = match vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM ex.main.split_stale_plan(n := 20, splits := 2)",
    )
    .await
    {
        Ok(dataframe) => dataframe
            .collect()
            .await
            .expect_err("stale split token must fail redemption"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("SPLIT_SNAPSHOT_EXPIRED") && message.contains("re-run the query"),
        "expired split lost its actionable error kind: {message}"
    );
    assert_eq!(
        healthy_sum(&ctx).await?,
        45,
        "attachment poisoned by expired split"
    );
    Ok(())
}
