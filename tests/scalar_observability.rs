// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Structured scalar worker-write observability.

mod common;

use datafusion::arrow::array::{Array, Int64Array, StringArray};
use datafusion::prelude::SessionContext;

async fn attached(cache: bool) -> datafusion::common::Result<Option<SessionContext>> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(None);
    };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}', CACHE {cache})",
            common::sql_quote(&worker.to_string_lossy())
        ),
    )
    .await?;
    Ok(Some(ctx))
}

async fn sum_cached_double(ctx: &SessionContext, rows: usize) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(
        ctx,
        &format!("SELECT sum(ex.main.cached_double_scalar(x % 3)) FROM range({rows}) t(x)"),
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

async fn clear_logs(ctx: &SessionContext) -> datafusion::common::Result<()> {
    vgi_datafusion::sql(ctx, "SELECT vgi_logs_clear()")
        .await?
        .collect()
        .await?;
    Ok(())
}

async fn scalar_writes(ctx: &SessionContext) -> datafusion::common::Result<Vec<(String, String)>> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT function, message FROM vgi_logs() \
         WHERE event = 'scalar.write_input' ORDER BY timestamp_ms",
    )
    .await?
    .collect()
    .await?;
    let mut writes = Vec::new();
    for batch in batches {
        let functions = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("function is Utf8");
        let messages = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("message is Utf8");
        for row in 0..batch.num_rows() {
            assert!(!functions.is_null(row));
            assert!(!messages.is_null(row));
            writes.push((
                functions.value(row).to_string(),
                messages.value(row).to_string(),
            ));
        }
    }
    Ok(writes)
}

async fn cache_ineligible(
    ctx: &SessionContext,
) -> datafusion::common::Result<Vec<(String, String)>> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT function, message FROM vgi_logs() \
         WHERE event = 'cache.ineligible' ORDER BY timestamp_ms",
    )
    .await?
    .collect()
    .await?;
    let mut out = Vec::new();
    for batch in batches {
        let functions = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("function is Utf8");
        let messages = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("message is Utf8");
        for row in 0..batch.num_rows() {
            out.push((
                functions.value(row).to_string(),
                messages.value(row).to_string(),
            ));
        }
    }
    Ok(out)
}

#[tokio::test(flavor = "multi_thread")]
async fn per_value_dedup_reports_only_the_batch_sent_and_hits_stay_silent(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached(true).await? else {
        return Ok(());
    };

    // Stable input dedup is independent of cache opt-in, so the first response
    // both advertises per-value caching and sees only three distinct rows.
    assert_eq!(sum_cached_double(&ctx, 9).await?, 18);
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![(
            "main.cached_double_scalar".to_string(),
            "input_rows=3".to_string()
        )]
    );

    // Preserve the learned opt-in but empty its stored values. The next call
    // deduplicates nine rows to the three distinct misses actually sent.
    vgi_datafusion::sql(&ctx, "SELECT vgi_cache_flush()")
        .await?
        .collect()
        .await?;
    clear_logs(&ctx).await?;
    assert_eq!(sum_cached_double(&ctx, 9).await?, 18);
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![(
            "main.cached_double_scalar".to_string(),
            "input_rows=3".to_string()
        )]
    );

    // Every value is now cached, so this invocation must not claim a worker
    // write merely because the SQL scalar itself was evaluated.
    clear_logs(&ctx).await?;
    assert_eq!(sum_cached_double(&ctx, 6).await?, 12);
    assert!(scalar_writes(&ctx).await?.is_empty());

    // Disabling the per-value tier bypasses those existing entries. Stable
    // input dedup still applies, and the actual three-row worker write is
    // visible rather than being mistaken for a cache hit.
    vgi_datafusion::sql(&ctx, "SET vgi_result_cache_per_value = false")
        .await?
        .collect()
        .await?;
    clear_logs(&ctx).await?;
    assert_eq!(sum_cached_double(&ctx, 6).await?, 12);
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![(
            "main.cached_double_scalar".to_string(),
            "input_rows=3".to_string()
        )]
    );
    assert_eq!(
        cache_ineligible(&ctx).await?,
        vec![(
            "main.cached_double_scalar".to_string(),
            "reason=per_value_disabled".to_string(),
        )],
        "one scalar execution must emit one sanitized admission reason"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn cache_disabled_scalar_reports_every_worker_batch_without_values(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached(false).await? else {
        return Ok(());
    };

    assert_eq!(sum_cached_double(&ctx, 6).await?, 12);
    clear_logs(&ctx).await?;
    assert_eq!(sum_cached_double(&ctx, 6).await?, 12);
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![(
            "main.cached_double_scalar".to_string(),
            "input_rows=3".to_string()
        )]
    );
    assert_eq!(
        cache_ineligible(&ctx).await?,
        vec![(
            "main.cached_double_scalar".to_string(),
            "reason=disabled_attach".to_string(),
        )],
        "one scalar execution must emit one sanitized admission reason"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn stable_dedup_setting_and_volatile_inputs_report_actual_rpc_rows(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached(true).await? else {
        return Ok(());
    };

    clear_logs(&ctx).await?;
    let batches = vgi_datafusion::sql(&ctx, "SELECT sum(ex.main.double(x % 3)) FROM range(6) t(x)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("sum is Int64")
            .value(0),
        12
    );
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![("main.double".to_string(), "input_rows=3".to_string())]
    );

    vgi_datafusion::sql(&ctx, "SET vgi_exchange_input_dedup = false")
        .await?
        .collect()
        .await?;
    clear_logs(&ctx).await?;
    let batches = vgi_datafusion::sql(&ctx, "SELECT sum(ex.main.double(x % 3)) FROM range(6) t(x)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("sum is Int64")
            .value(0),
        12
    );
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![("main.double".to_string(), "input_rows=6".to_string())]
    );

    vgi_datafusion::sql(&ctx, "RESET vgi_exchange_input_dedup")
        .await?
        .collect()
        .await?;
    clear_logs(&ctx).await?;
    let batches = vgi_datafusion::sql(&ctx, "SELECT ex.main.random_int(1, 10) FROM range(6)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        6
    );
    assert_eq!(
        scalar_writes(&ctx).await?,
        vec![("main.random_int".to_string(), "input_rows=6".to_string())]
    );
    Ok(())
}
