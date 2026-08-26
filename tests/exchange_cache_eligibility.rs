// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end boundaries for the streaming exchange result-cache tier.

mod common;

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

    // A buffering function has its own whole-input/finalize lifecycle. Its
    // cache advertisement must never be mistaken for a streaming batch entry.
    let buffered = "SELECT * FROM ex.main.cached_sum_all(\
                    (SELECT x FROM range(5) t(x)), logging := false)";
    for _ in 0..2 {
        vgi_datafusion::sql(&ctx, buffered).await?.collect().await?;
    }
    assert_eq!(exchange_stats(&ctx).await?, (0, 0), "{buffered}");
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
