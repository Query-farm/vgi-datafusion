// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! DataFusion adaptation of VGI's `order_pushdown.test` read-only records.

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use vgi_datafusion::VgiOrderPushdownSessionStateBuilderExt;

use crate::common;

fn context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_vgi_order_pushdown()
        .build();
    SessionContext::new_with_state(state)
}

async fn assert_diagnostics(
    ctx: &SessionContext,
    sql: &str,
    expected_n: &[i64],
    column: &str,
    direction: &str,
    null_order: &str,
    limit: i64,
) -> datafusion::error::Result<()> {
    let batches = vgi_datafusion::sql(ctx, sql).await?.collect().await?;
    let mut actual_n = Vec::new();
    for batch in &batches {
        let n = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let columns = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let directions = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let null_orders = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let limits = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            actual_n.push(n.value(row));
            assert_eq!(columns.value(row), column);
            assert_eq!(directions.value(row), direction);
            assert_eq!(null_orders.value(row), null_order);
            assert_eq!(limits.value(row), limit);
        }
    }
    assert_eq!(actual_n, expected_n);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn corpus_order_top_n_hints_are_safe_and_host_verified() -> datafusion::error::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = context();
    let location = common::sql_quote(&worker.to_string_lossy());
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={location}' AS ex")).await?;

    let columns = "n, order_column, order_direction, order_null_order, order_limit";

    // order_pushdown records 2-7a: direct column, direction/null policy,
    // and OFFSET's checked limit+offset early-stop bound.
    assert_diagnostics(
        &ctx,
        &format!(
            "SELECT {columns} FROM ex.main.order_echo(100) \
             ORDER BY n DESC NULLS LAST LIMIT 2"
        ),
        &[99, 98],
        "n",
        "desc",
        "nulls_last",
        2,
    )
    .await?;
    assert_diagnostics(
        &ctx,
        &format!(
            "SELECT {columns} FROM ex.main.order_echo(5) \
             ORDER BY s ASC NULLS FIRST LIMIT 2"
        ),
        &[0, 1],
        "s",
        "asc",
        "nulls_first",
        2,
    )
    .await?;
    assert_diagnostics(
        &ctx,
        &format!(
            "SELECT {columns} FROM ex.main.order_echo(100) \
             ORDER BY n LIMIT 3 OFFSET 2"
        ),
        &[2, 3, 4],
        "n",
        "asc",
        "nulls_last",
        5,
    )
    .await?;

    // Record 11's second key can affect which tied rows survive, so VGI may
    // receive the useful first-key order but not an early-stop limit.
    assert_diagnostics(
        &ctx,
        &format!(
            "SELECT {columns} FROM ex.main.order_echo(100) \
             ORDER BY n, s LIMIT 2"
        ),
        &[0, 1],
        "n",
        "asc",
        "nulls_last",
        -1,
    )
    .await?;

    // Record 10: ordering remains useful, but no worker early stop may happen
    // before pushed/residual filtering is known complete.
    assert_diagnostics(
        &ctx,
        &format!(
            "SELECT {columns} FROM ex.main.order_echo(100) WHERE n < 10 \
             ORDER BY n LIMIT 3"
        ),
        &[0, 1, 2],
        "n",
        "asc",
        "nulls_last",
        -1,
    )
    .await?;

    // Record 13: a projected subset remains pushable only because the sort
    // key maps through the projection to a direct child column.
    let projected = vgi_datafusion::sql(
        &ctx,
        "SELECT n, order_column, order_limit FROM ex.main.order_echo(100) \
         ORDER BY n LIMIT 3",
    )
    .await?
    .collect()
    .await?;
    let rendered = datafusion::arrow::util::pretty::pretty_format_batches(&projected)?.to_string();
    assert!(
        rendered.contains("| 0 | n"),
        "projected hint missing: {rendered}"
    );
    assert!(
        rendered.contains("| 2 | n"),
        "host Top-N missing: {rendered}"
    );

    // Record 12: a computed sort key is never represented as a VGI column.
    assert_diagnostics(
        &ctx,
        &format!(
            "SELECT {columns} FROM ex.main.order_echo(5) \
             ORDER BY n + 1 LIMIT 2"
        ),
        &[0, 1],
        "(none)",
        "(none)",
        "(none)",
        -1,
    )
    .await?;

    // The physical rule must leave DataFusion's host Top-K sort in place while
    // exposing the validated hint on the scan. DataFusion fuses LIMIT into
    // SortExec as `fetch`, so a separate GlobalLimitExec is not required.
    let explain = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM ex.main.order_echo(20) ORDER BY n LIMIT 2",
    )
    .await?
    .collect()
    .await?;
    let explain = datafusion::arrow::util::pretty::pretty_format_batches(&explain)?.to_string();
    assert!(explain.contains("SortExec"), "host sort missing: {explain}");
    assert!(
        explain.contains("TopK(fetch=2)"),
        "host Top-K bound missing: {explain}"
    );
    assert!(
        explain.contains("order_by=n"),
        "VGI hint missing: {explain}"
    );

    Ok(())
}
