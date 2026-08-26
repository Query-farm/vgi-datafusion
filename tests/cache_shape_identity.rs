// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Result-cache identity follows worker capabilities, not local query shape.

mod common;

use datafusion::arrow::array::{Array, Int64Array, UInt64Array};
use datafusion::prelude::{SessionConfig, SessionContext};

async fn attached() -> datafusion::common::Result<Option<SessionContext>> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(None);
    };
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
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

async fn scalar_i64(ctx: &SessionContext, sql: &str) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(ctx, sql).await?.collect().await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("query returns Int64")
        .value(0))
}

async fn values_i64(ctx: &SessionContext, sql: &str) -> datafusion::common::Result<Vec<i64>> {
    let batches = vgi_datafusion::sql(ctx, sql).await?.collect().await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("query returns Int64")
                .clone();
            (0..values.len()).map(move |index| values.value(index))
        })
        .collect())
}

async fn entries_for(ctx: &SessionContext, function: &str) -> datafusion::common::Result<i64> {
    scalar_i64(
        ctx,
        &format!(
            "SELECT COUNT(*) FROM vgi_result_cache() WHERE function = '{}'",
            function.replace('\'', "''")
        ),
    )
    .await
}

async fn cache_hits(ctx: &SessionContext) -> datafusion::common::Result<u64> {
    let batches = vgi_datafusion::sql(ctx, "SELECT hits FROM vgi_result_cache_stats()")
        .await?
        .collect()
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cache hits are UInt64")
        .value(0))
}

async fn flush(ctx: &SessionContext) -> datafusion::common::Result<()> {
    vgi_datafusion::sql(ctx, "SELECT * FROM vgi_result_cache_flush()")
        .await?
        .collect()
        .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn local_projections_share_one_full_result_entry() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    flush(&ctx).await?;
    let hits_before = cache_hits(&ctx).await?;

    // The fixture does not advertise projection pushdown. Capture its full
    // three-column worker batch once, then conform every replay locally.
    assert_eq!(
        values_i64(&ctx, "SELECT a FROM ex.data.cache_multicol ORDER BY a").await?,
        [0, 1, 2, 3]
    );
    assert_eq!(
        values_i64(&ctx, "SELECT b FROM ex.data.cache_multicol ORDER BY b").await?,
        [0, 10, 20, 30]
    );
    assert_eq!(
        values_i64(&ctx, "SELECT c FROM ex.data.cache_multicol ORDER BY c").await?,
        [0, 100, 200, 300]
    );
    // A zero-column count(*) replay must retain the cached batch's row count.
    assert_eq!(
        scalar_i64(&ctx, "SELECT COUNT(*) FROM ex.data.cache_multicol").await?,
        4
    );
    assert_eq!(entries_for(&ctx, "cache_multicol").await?, 1);
    assert!(cache_hits(&ctx).await? >= hits_before + 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_identity_is_retained_only_when_the_worker_can_apply_it(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    flush(&ctx).await?;

    // cacheable_numbers is explicitly non-filter-capable. Both predicates are
    // evaluated by DataFusion over the same cached full worker result.
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT COUNT(*) FROM ex.data.cacheable_numbers WHERE n >= 5",
        )
        .await?,
        5
    );
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT COUNT(*) FROM ex.data.cacheable_numbers WHERE n >= 7",
        )
        .await?,
        3
    );
    assert_eq!(entries_for(&ctx, "cacheable_numbers").await?, 1);

    flush(&ctx).await?;
    // cache_multicol is deliberately hidden from function discovery even
    // though it backs a catalog table. Its filter capability is unknown, so
    // retain the optimistic wire filter and conservative filter-specific key.
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT COUNT(*) FROM ex.data.cache_multicol WHERE a >= 1",
        )
        .await?,
        3
    );
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT COUNT(*) FROM ex.data.cache_multicol WHERE a >= 2",
        )
        .await?,
        2
    );
    assert_eq!(entries_for(&ctx, "cache_multicol").await?, 2);

    flush(&ctx).await?;
    let hits_before = cache_hits(&ctx).await?;
    // cache_filtered advertises filter pushdown and applies it. Distinct
    // predicates must remain distinct cache identities; repeating one hits it.
    for (predicate, expected) in [("n >= 5", 5), ("n >= 7", 3), ("n >= 5", 5)] {
        assert_eq!(
            scalar_i64(
                &ctx,
                &format!("SELECT COUNT(*) FROM ex.cache_filtered(rows := 10) WHERE {predicate}"),
            )
            .await?,
            expected
        );
    }
    assert_eq!(entries_for(&ctx, "cache_filtered").await?, 2);
    assert!(cache_hits(&ctx).await? > hits_before);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn pushed_projections_keep_distinct_cache_identities() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    flush(&ctx).await?;
    let hits_before = cache_hits(&ctx).await?;

    assert_eq!(
        values_i64(&ctx, "SELECT a FROM ex.data.cache_projection ORDER BY a").await?,
        [1, 2, 3]
    );
    assert_eq!(
        values_i64(&ctx, "SELECT b FROM ex.data.cache_projection ORDER BY b").await?,
        [10, 20, 30]
    );
    assert_eq!(
        values_i64(&ctx, "SELECT a FROM ex.data.cache_projection ORDER BY a").await?,
        [1, 2, 3]
    );
    assert_eq!(entries_for(&ctx, "cache_projection").await?, 2);
    assert!(cache_hits(&ctx).await? > hits_before);
    Ok(())
}
