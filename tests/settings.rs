// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end coverage for VGI settings through DataFusion's config extension.

mod common;

use datafusion::arrow::array::{Float64Array, Int64Array, StringArray};
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
async fn scalar_settings_support_unqualified_set_change_and_reset() -> datafusion::common::Result<()>
{
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    let metadata = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM duckdb_settings() WHERE name IN \
         ('vgi_verbose_mode', 'greeting', 'multiplier', 'threshold', 'config')",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        metadata[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        5
    );

    vgi_datafusion::sql(&ctx, "SET multiplier = 5")
        .await?
        .collect()
        .await?;
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.multiply_by_setting(v) FROM (VALUES (1), (2), (3)) t(v) ORDER BY v",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[5, 10, 15]
    );

    vgi_datafusion::sql(&ctx, "SET vgi.multiplier = 10")
        .await?
        .collect()
        .await?;
    let batches = vgi_datafusion::sql(&ctx, "SELECT ex.main.multiply_by_setting(2)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        20
    );

    vgi_datafusion::sql(&ctx, "RESET multiplier").await?;
    let batches = vgi_datafusion::sql(&ctx, "SELECT ex.main.multiply_by_setting(2)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        2
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn settings_reach_table_table_input_and_struct_consumers() -> datafusion::common::Result<()> {
    let Some(ctx) = attached().await? else {
        return Ok(());
    };

    vgi_datafusion::sql(&ctx, "SET greeting = 'Bonjour'")
        .await?
        .collect()
        .await?;
    vgi_datafusion::sql(&ctx, "SET vgi.scale_factor = 2.5")
        .await?
        .collect()
        .await?;
    let scaled = vgi_datafusion::sql(&ctx, "SELECT ex.main.scale_by_setting(4.0)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        scaled[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        10.0
    );

    let greeting = vgi_datafusion::sql(&ctx, "SELECT greeting FROM ex.main.settings_aware(1)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        greeting[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "Bonjour"
    );

    vgi_datafusion::sql(&ctx, "SET threshold = 3")
        .await?
        .collect()
        .await?;
    let filtered = vgi_datafusion::sql(
        &ctx,
        "SELECT value FROM ex.main.filter_by_setting(\
         (SELECT * FROM (VALUES (0), (1), (2), (3), (4)) t(value))) ORDER BY value",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        filtered[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[3, 4]
    );

    // DataFusion has no struct-literal syntax for SET values. A JSON string
    // keeps the setting in DataFusion's existing ConfigExtension API and is
    // cast to the worker's advertised Arrow Struct type at bind.
    vgi_datafusion::sql(
        &ctx,
        r#"SET vgi.config = '{"start":10,"step":5,"label":"item"}'"#,
    )
    .await?
    .collect()
    .await?;
    let configured = vgi_datafusion::sql(
        &ctx,
        "SELECT n, label FROM ex.main.struct_settings(3) ORDER BY n",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        configured[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[10, 15, 20]
    );
    assert_eq!(
        configured[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some("item_0"), Some("item_1"), Some("item_2")]
    );
    Ok(())
}
