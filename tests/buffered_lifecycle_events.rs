// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Structured adapter events for the buffered sink/combine/source lifecycle.

use crate::common;

use datafusion::arrow::array::{Array, StringArray};
use datafusion::prelude::SessionContext;

async fn lifecycle_messages(ctx: &SessionContext) -> datafusion::common::Result<Vec<String>> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT message FROM duckdb_logs() \
         WHERE type = 'VGI' AND message LIKE 'table_buffering.%' \
         ORDER BY message",
    )
    .await?
    .collect()
    .await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("duckdb_logs message is Utf8")
                .iter()
                .flatten()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect())
}

#[tokio::test(flavor = "multi_thread")]
async fn successful_buffered_rpcs_emit_non_secret_lifecycle_events(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();
    let location = common::sql_quote(&worker.to_string_lossy());
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example' AS ex (TYPE vgi, LOCATION '{location}')"),
    )
    .await?;

    let query = "SELECT x AS total FROM ex.main.cached_sum_all((\
                     SELECT x FROM range(10) t(x)\
                 ))";
    vgi_datafusion::sql(&ctx, query).await?.collect().await?;

    let first = lifecycle_messages(&ctx).await?;
    assert_eq!(
        first
            .iter()
            .filter(|message| message.starts_with("table_buffering.combine "))
            .count(),
        1,
        "one actual combine RPC must produce one event: {first:?}"
    );
    let combine = first
        .iter()
        .find(|message| message.starts_with("table_buffering.combine "))
        .expect("combine event");
    for expected in [
        "catalog=example",
        "function=main.cached_sum_all",
        "input_batches=",
        "state_ids=",
        "finalize_ids=",
    ] {
        assert!(
            combine.contains(expected),
            "missing {expected:?}: {combine}"
        );
    }
    for forbidden in ["input_rows=", "arguments=", "secret", "execution_id="] {
        assert!(
            !combine.to_ascii_lowercase().contains(forbidden),
            "lifecycle event exposed forbidden detail {forbidden:?}: {combine}"
        );
    }
    assert!(
        first
            .iter()
            .any(|message| message.starts_with("table_buffering.begin ")),
        "successful begin RPC was not observable: {first:?}"
    );
    assert!(
        first
            .iter()
            .any(|message| message.starts_with("table_buffering.finalize ")),
        "successful finalize phase was not observable: {first:?}"
    );

    // The identical call is served by the buffered whole-input cache. It does
    // not execute combine and therefore must not synthesize another lifecycle
    // event merely because a result was replayed.
    vgi_datafusion::sql(&ctx, query).await?.collect().await?;
    let second = lifecycle_messages(&ctx).await?;
    assert_eq!(
        second
            .iter()
            .filter(|message| message.starts_with("table_buffering.combine "))
            .count(),
        1,
        "cache replay must not claim an RPC that did not run: {second:?}"
    );
    Ok(())
}
