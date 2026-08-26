// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Structured observability for actual streaming table-input writes.

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

async fn clear_logs(ctx: &SessionContext) -> datafusion::common::Result<()> {
    vgi_datafusion::sql(ctx, "SELECT vgi_logs_clear()")
        .await?
        .collect()
        .await?;
    Ok(())
}

async fn cached_echo(ctx: &SessionContext) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT sum(x) FROM ex.main.cached_echo((SELECT x FROM range(5) t(x)))",
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

async fn writes(ctx: &SessionContext) -> datafusion::common::Result<Vec<(String, String)>> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT function, message FROM vgi_logs() \
         WHERE event = 'table_in_out.write_input' ORDER BY timestamp_ms",
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
            assert!(!functions.is_null(row));
            assert!(!messages.is_null(row));
            out.push((
                functions.value(row).to_string(),
                messages.value(row).to_string(),
            ));
        }
    }
    Ok(out)
}

#[tokio::test(flavor = "multi_thread")]
async fn table_input_writes_report_actual_batches_and_cache_hits_stay_silent(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached(true).await? else {
        return Ok(());
    };

    clear_logs(&ctx).await?;
    assert_eq!(cached_echo(&ctx).await?, 10);
    assert_eq!(
        writes(&ctx).await?,
        vec![("main.cached_echo".to_string(), "input_rows=5".to_string())]
    );

    clear_logs(&ctx).await?;
    assert_eq!(cached_echo(&ctx).await?, 10);
    assert!(writes(&ctx).await?.is_empty(), "a full hit sends no input");

    let Some(no_cache) = attached(false).await? else {
        return Ok(());
    };
    clear_logs(&no_cache).await?;
    assert_eq!(cached_echo(&no_cache).await?, 10);
    assert_eq!(cached_echo(&no_cache).await?, 10);
    assert_eq!(
        writes(&no_cache).await?,
        vec![
            ("main.cached_echo".to_string(), "input_rows=5".to_string()),
            ("main.cached_echo".to_string(), "input_rows=5".to_string()),
        ]
    );
    Ok(())
}
