// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end boundaries for the streaming exchange result-cache tier.

mod common;

use std::sync::Arc;

use datafusion::arrow::array::{Array, UInt64Array};
use datafusion::prelude::{SessionConfig, SessionContext};

async fn attached() -> datafusion::common::Result<Option<SessionContext>> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(None);
    };
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_batch_size(1024),
    );
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            common::sql_quote(&worker.to_string_lossy())
        ),
    )
    .await?;
    Ok(Some(ctx))
}

async fn exchange_stats(ctx: &SessionContext) -> datafusion::common::Result<(u64, u64)> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT exchange_hits, exchange_stores FROM vgi_result_cache_stats()",
    )
    .await?
    .collect()
    .await?;
    let value = |index| {
        batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("cache statistic is UInt64")
            .value(0)
    };
    Ok((value(0), value(1)))
}

async fn cache_entries(ctx: &SessionContext) -> datafusion::common::Result<u64> {
    let batches = vgi_datafusion::sql(ctx, "SELECT entries FROM vgi_cache_stats()")
        .await?
        .collect()
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cache entries is UInt64")
        .value(0))
}

async fn cache_revalidations(ctx: &SessionContext) -> datafusion::common::Result<u64> {
    let batches = vgi_datafusion::sql(ctx, "SELECT revalidations FROM vgi_cache_stats()")
        .await?
        .collect()
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cache revalidations is UInt64")
        .value(0))
}

async fn cache_stale_serves(ctx: &SessionContext) -> datafusion::common::Result<u64> {
    let batches = vgi_datafusion::sql(ctx, "SELECT stale_serves FROM vgi_cache_stats()")
        .await?
        .collect()
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cache stale_serves is UInt64")
        .value(0))
}

async fn concurrent_queries(
    ctx: &SessionContext,
    query: &'static str,
    count: usize,
) -> datafusion::common::Result<()> {
    // Plan first so planner/discovery scheduling cannot turn one intended
    // concurrent execution wave into several zero-TTL revalidation waves.
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        frames.push(vgi_datafusion::sql(ctx, query).await?);
    }
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(count));
    let mut tasks = Vec::with_capacity(count);
    for frame in frames {
        let barrier = std::sync::Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            frame.collect().await
        }));
    }
    for task in tasks {
        task.await
            .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))??;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn stateful_and_literal_exchanges_never_enter_streaming_batch_cache(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    // Each fixture deliberately advertises cache-control. Eligibility—not a
    // missing worker opt-in—is therefore what keeps these calls out.
    for query in [
        "SELECT * FROM ex.main.cached_serial_echo((SELECT x FROM range(5) t(x)))",
        "SELECT * FROM ex.main.cached_finalizing_echo((SELECT x FROM range(5) t(x)))",
        "SELECT * FROM ex.main.cached_double(21)",
    ] {
        for _ in 0..2 {
            vgi_datafusion::sql(&ctx, query).await?.collect().await?;
        }
        assert_eq!(exchange_stats(&ctx).await?, (0, 0), "{query}");
        assert_eq!(cache_entries(&ctx).await?, 0, "{query}");
    }

    // A buffering function without finalize cache metadata remains uncached;
    // buffered caching is an explicit whole-lifecycle worker opt-in.
    let buffered = "SELECT * FROM ex.main.sum_all_columns(\
                    (SELECT x FROM range(5) t(x)), logging := false)";
    for _ in 0..2 {
        vgi_datafusion::sql(&ctx, buffered).await?.collect().await?;
    }
    assert_eq!(exchange_stats(&ctx).await?, (0, 0), "{buffered}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn buffered_cache_keys_and_commits_the_complete_input() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const FIVE: &str = "SELECT x FROM ex.main.cached_sum_all(\
                        (SELECT x FROM range(5) t(x)), logging := false)";
    for _ in 0..2 {
        let batches = vgi_datafusion::sql(&ctx, FIVE).await?.collect().await?;
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("buffered sum is Int64")
                .value(0),
            10
        );
    }
    assert_eq!(exchange_stats(&ctx).await?, (1, 1));
    assert_eq!(cache_entries(&ctx).await?, 1);

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT x FROM ex.main.cached_sum_all(\
         (SELECT x FROM range(6) t(x)), logging := false)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("buffered sum is Int64")
            .value(0),
        15
    );
    assert_eq!(exchange_stats(&ctx).await?, (1, 2));
    assert_eq!(cache_entries(&ctx).await?, 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn buffered_revalidation_is_conditional_and_single_flight() -> datafusion::common::Result<()>
{
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT x FROM ex.main.cached_reval_sum_all(\
                         (SELECT x FROM range(5) t(x)), logging := false)";
    vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1);
    assert_eq!(cache_revalidations(&ctx).await?, 0);

    concurrent_queries(&ctx, QUERY, 8).await?;
    assert_eq!(
        cache_revalidations(&ctx).await?,
        1,
        "one buffered lifecycle should validate an overlapping request wave"
    );
    let (hits, stores) = exchange_stats(&ctx).await?;
    assert!(hits >= 8);
    assert_eq!(stores, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn buffered_revalidation_honors_stale_if_error() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT x FROM ex.main.cached_reval_error_sum_all(\
                         (SELECT x FROM range(5) t(x)), logging := false)";
    for _ in 0..2 {
        let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("buffered sum is Int64")
                .value(0),
            10
        );
    }
    assert_eq!(cache_entries(&ctx).await?, 1);
    assert_eq!(cache_stale_serves(&ctx).await?, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn buffered_revalidation_revocation_evicts_stale_bytes() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT x FROM ex.main.cached_reval_no_store_sum_all(\
                         (SELECT x FROM range(5) t(x)), logging := false)";
    vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1, "cold buffered entry");
    assert!(
        vgi_datafusion::sql(&ctx, QUERY)
            .await?
            .collect()
            .await
            .is_err(),
        "no_store + not_modified must not replay stale buffered bytes"
    );
    assert_eq!(cache_entries(&ctx).await?, 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn fresh_buffered_revocation_serves_result_but_evicts_stale_bytes(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT x FROM ex.main.cached_reval_fresh_no_store_sum_all(\
                         (SELECT x FROM range(5) t(x)), logging := false)";
    let cold = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1, "cold buffered entry");
    let fresh = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    for batches in [cold, fresh] {
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("buffered sum is Int64")
                .value(0),
            10
        );
    }
    assert_eq!(cache_entries(&ctx).await?, 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_buffered_lifecycle_never_commits_a_partial_result() -> datafusion::common::Result<()>
{
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    let query = "SELECT * FROM ex.main.exception_finalize(\
                 (SELECT x FROM range(5) t(x)), logging := false)";
    assert!(vgi_datafusion::sql(&ctx, query)
        .await?
        .collect()
        .await
        .is_err());
    assert_eq!(cache_entries(&ctx).await?, 0);
    assert_eq!(exchange_stats(&ctx).await?, (0, 0));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn oversized_buffered_result_records_capture_abort_without_failing_query(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let mut options = vgi_datafusion::VgiSessionOptions::default();
    options.cache_limits.max_entry_bytes = 1;
    let runtime = Arc::new(vgi_datafusion::VgiRuntime::new(options));
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::clone(&runtime)),
    );
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            common::sql_quote(&worker.to_string_lossy())
        ),
    )
    .await?;

    const QUERY: &str = "SELECT x FROM ex.main.cached_sum_all(\
                         (SELECT x FROM range(5) t(x)), logging := false)";
    for expected_aborts in 1..=2 {
        let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("buffered sum is Int64")
                .value(0),
            10
        );
        assert_eq!(runtime.result_cache().stats().entries, 0);
        assert_eq!(
            runtime.result_cache().stats().capture_aborts,
            expected_aborts,
            "every over-cap whole-input result is returned but not cached"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_buffered_lifecycle_cannot_commit_after_its_future_is_dropped(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT x FROM ex.main.cached_slow_sum_all(\
                         (SELECT x FROM range(5) t(x)), logging := false)";
    let frame = vgi_datafusion::sql(&ctx, QUERY).await?;
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(25), frame.collect())
            .await
            .is_err(),
        "the fixture should still be inside its blocking buffered lifecycle"
    );
    tokio::time::sleep(std::time::Duration::from_millis(1_000)).await;
    assert_eq!(cache_entries(&ctx).await?, 0);
    assert_eq!(exchange_stats(&ctx).await?, (0, 0));

    vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1);
    assert_eq!(exchange_stats(&ctx).await?, (0, 1));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scalar_per_value_cache_caps_new_stores_per_call_at_256() -> datafusion::common::Result<()>
{
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT sum(ex.main.cached_double_scalar(x)) FROM range(300) t(x)",
    )
    .await?
    .collect()
    .await?;
    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("sum is Int64")
        .value(0);
    assert_eq!(sum, 89_700);
    assert_eq!(exchange_stats(&ctx).await?, (0, 256));
    assert_eq!(cache_entries(&ctx).await?, 256);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn streaming_revalidation_is_conditional_and_single_flight() -> datafusion::common::Result<()>
{
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT sum(x) FROM ex.main.cached_reval_echo(\
                         (SELECT x FROM range(5) t(x)))";
    vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1);
    assert_eq!(cache_revalidations(&ctx).await?, 0);

    concurrent_queries(&ctx, QUERY, 8).await?;
    assert_eq!(
        cache_revalidations(&ctx).await?,
        1,
        "overlapping stale reads must share one conditional worker exchange"
    );
    let (hits, stores) = exchange_stats(&ctx).await?;
    assert!(
        hits >= 8,
        "revalidation and its followers replay cached bytes"
    );
    assert_eq!(stores, 1, "not_modified must slide rather than replace");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn scalar_per_value_revalidation_is_conditional_and_single_flight(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT sum(ex.main.cached_reval_double_scalar(x + 21)) FROM range(1) t(x)";
    let cold = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(
        cold[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("cached scalar returns Int64")
            .value(0),
        42
    );
    assert_eq!(cache_entries(&ctx).await?, 1);

    concurrent_queries(&ctx, QUERY, 8).await?;
    assert_eq!(
        cache_revalidations(&ctx).await?,
        1,
        "overlapping stale scalar values must share one conditional exchange"
    );
    let (hits, stores) = exchange_stats(&ctx).await?;
    assert!(hits >= 8);
    assert_eq!(stores, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn table_input_revalidation_honors_stale_if_error() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT sum(x) FROM ex.main.cached_reval_policy(\
                         (SELECT x FROM range(5) t(x)), 'error')";
    for _ in 0..2 {
        let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("sum is Int64")
                .value(0),
            10
        );
    }
    assert_eq!(cache_entries(&ctx).await?, 1);
    assert_eq!(cache_stale_serves(&ctx).await?, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scalar_revalidation_honors_stale_if_error() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT sum(ex.main.cached_reval_policy_scalar(\
                         x + 21, 'error')) FROM range(1) t(x)";
    for _ in 0..2 {
        let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("sum is Int64")
                .value(0),
            42
        );
    }
    assert_eq!(cache_entries(&ctx).await?, 1);
    assert_eq!(cache_stale_serves(&ctx).await?, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn ineligible_table_input_revalidation_evicts_stale_bytes() -> datafusion::common::Result<()>
{
    for policy in ["not_modified_no_store", "transaction"] {
        let Some(ctx) = attached().await? else {
            return Ok(());
        };
        let query = format!(
            "SELECT * FROM ex.main.cached_reval_policy(\
             (SELECT x FROM range(5) t(x)), '{policy}')"
        );
        vgi_datafusion::sql(&ctx, &query).await?.collect().await?;
        assert_eq!(cache_entries(&ctx).await?, 1, "cold {policy}");
        assert!(
            vgi_datafusion::sql(&ctx, &query)
                .await?
                .collect()
                .await
                .is_err(),
            "an ineligible not_modified response must not replay stale bytes: {policy}"
        );
        assert_eq!(cache_entries(&ctx).await?, 0, "revoked {policy}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn fresh_scalar_revocation_evicts_stale_bytes() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT sum(ex.main.cached_reval_policy_scalar(\
                         x + 21, 'fresh_no_store')) FROM range(1) t(x)";
    let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1, "cold scalar entry");
    for batches in [
        batches,
        vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?,
    ] {
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("sum is Int64")
                .value(0),
            42
        );
    }
    assert_eq!(cache_entries(&ctx).await?, 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn fresh_table_input_revocation_evicts_stale_bytes() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    const QUERY: &str = "SELECT sum(x) FROM ex.main.cached_reval_policy(\
                         (SELECT x FROM range(5) t(x)), 'fresh_no_store')";
    let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    assert_eq!(cache_entries(&ctx).await?, 1, "cold table-input entry");
    for batches in [
        batches,
        vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?,
    ] {
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("sum is Int64")
                .value(0),
            10
        );
    }
    assert_eq!(cache_entries(&ctx).await?, 0);
    Ok(())
}
