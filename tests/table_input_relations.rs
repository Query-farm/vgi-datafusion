// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! RelationPlanner coverage for VGI functions that consume a table.

use crate::common;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::prelude::SessionContext;

async fn attached() -> datafusion::common::Result<Option<SessionContext>> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(None);
    };
    let ctx = SessionContext::new();
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

#[tokio::test(flavor = "multi_thread")]
async fn streaming_input_preserves_three_top_level_columns() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT e.id, e.label, e.weight
         FROM ex.main.echo((
             SELECT * FROM (VALUES
                 (1, 'one', 1.5),
                 (2, 'two', 2.5),
                 (3, 'three', 3.5)
             ) AS input(id, label, weight)
         )) AS e
         ORDER BY e.id",
    )
    .await?
    .collect()
    .await?;

    assert_eq!(
        batches[0]
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        vec!["id", "label", "weight"]
    );
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id remains Int64");
    let labels = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("label remains Utf8");
    assert_eq!(ids.values(), &[1, 2, 3]);
    assert_eq!(
        labels.iter().collect::<Vec<_>>(),
        vec![Some("one"), Some("two"), Some("three")]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn buffered_input_sums_multiple_columns_with_named_constants(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT a, b FROM ex.main.sum_all_columns((
             SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS input(a, b)
         ), logging := false)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    let value = |column| {
        batches[0]
            .column(column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("buffered sums are Int64")
            .value(0)
    };
    assert_eq!((value(0), value(1)), (6, 60));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn single_column_and_explain_remain_compatible() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT x FROM ex.main.echo((SELECT x FROM range(3) AS input(x))) ORDER BY x",
    )
    .await?
    .collect()
    .await?;
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("range output is Int64");
    assert_eq!(values.values(), &[0, 1, 2]);

    let explain = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM ex.main.echo((
             SELECT * FROM (VALUES (1, 2)) AS input(a, b)
         ))",
    )
    .await?
    .collect()
    .await?;
    assert!(!explain.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_and_partitioned_wide_inputs_preserve_their_relations(
) -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT a, b FROM ex.main.echo((
             SELECT 1 AS a, 'left' AS b
             UNION ALL
             SELECT 2 AS a, 'right' AS b
         )) ORDER BY a",
    )
    .await?
    .collect()
    .await?;
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("union output remains Int64");
    assert_eq!(ids.values(), &[1, 2]);

    let empty = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM ex.main.echo((
             SELECT CAST(NULL AS BIGINT) AS a, CAST(NULL AS VARCHAR) AS b
             WHERE false
         ))",
    )
    .await?
    .collect()
    .await?;
    let count = empty[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count remains Int64")
        .value(0);
    assert_eq!(count, 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn detach_removes_relation_planner_descriptors() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };
    vgi_datafusion::sql(&ctx, "DETACH ex").await?;
    let error = vgi_datafusion::sql(
        &ctx,
        "SELECT * FROM ex.main.echo((
             SELECT * FROM (VALUES (1, 2)) AS input(a, b)
         ))",
    )
    .await
    .expect_err("detached VGI functions must not remain in the planner registry");
    assert!(
        error
            .to_string()
            .contains("The subquery should only return one column"),
        "unexpected detach error: {error}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn match_recognize_accepts_a_wide_relation_and_list_constants(
) -> datafusion::common::Result<()> {
    let Ok(worker) = std::env::var("VGI_MATCHRECOGNIZE_WORKER") else {
        eprintln!("skipping: set VGI_MATCHRECOGNIZE_WORKER to the worker's HTTP URL");
        return Ok(());
    };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'mr' AS mr (TYPE vgi, LOCATION '{}')",
            common::sql_quote(&worker)
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT symbol, match_no, bottom_price, drawdown
         FROM mr.main.match_recognize((
             SELECT * FROM (VALUES
                 ('ACME', 1, 10), ('ACME', 2, 8), ('ACME', 3, 6),
                 ('ACME', 4, 9), ('ACME', 5, 11), ('ACME', 6, 7)
             ) AS input(symbol, ts, price)
         ),
             partition_by := ['symbol'],
             order_by := ['ts'],
             pattern := 'START DOWN+ UP+',
             define := '{\"DOWN\":\"price < PREV(price)\",\"UP\":\"price > PREV(price)\"}',
             measures := '{\"match_no\":\"MATCH_NUMBER()\",\"bottom_price\":\"LAST(DOWN.price)\",\"drawdown\":\"FIRST(START.price) - LAST(DOWN.price)\"}'
         )",
    )
    .await?
    .collect()
    .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    let symbol = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("partition key remains Utf8")
        .value(0);
    let integer = |column| {
        batches[0]
            .column(column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("match measures remain Int64")
            .value(0)
    };
    assert_eq!(
        (symbol, integer(1), integer(2), integer(3)),
        ("ACME", 1, 6, 4)
    );
    Ok(())
}
