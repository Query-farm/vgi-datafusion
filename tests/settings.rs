// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end coverage for VGI settings through DataFusion's config extension.

use crate::common;

use datafusion::arrow::array::{Float64Array, Int64Array, StringArray};
use std::sync::Arc;

use datafusion::execution::context::SessionConfig;
use datafusion::prelude::SessionContext;

fn configured(ctx: &SessionContext, name: &str) -> Option<String> {
    ctx.copied_config()
        .options()
        .extensions
        .get::<vgi_datafusion::VgiSettings>()
        .and_then(|settings| settings.get(name))
        .map(str::to_string)
}

#[tokio::test]
async fn adapter_tuning_supports_unqualified_set_defaults_and_reset(
) -> datafusion::common::Result<()> {
    let ctx = SessionContext::new();

    // Calling through the adapter installs its ConfigExtension even without
    // an attached worker. Both host-owned compatibility knobs default on.
    vgi_datafusion::sql(&ctx, "SELECT 1").await?;
    assert_eq!(
        configured(&ctx, "vgi_exchange_input_dedup").as_deref(),
        Some("true")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_per_value").as_deref(),
        Some("true")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entry_bytes").as_deref(),
        Some("67108864")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_bytes").as_deref(),
        Some("268435456")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entries").as_deref(),
        Some("131072")
    );

    vgi_datafusion::sql(&ctx, "SET vgi_exchange_input_dedup = false")
        .await?
        .collect()
        .await?;
    vgi_datafusion::sql(&ctx, "SET vgi.vgi_result_cache_per_value = false")
        .await?
        .collect()
        .await?;
    assert_eq!(
        configured(&ctx, "vgi_exchange_input_dedup").as_deref(),
        Some("false")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_per_value").as_deref(),
        Some("false")
    );

    vgi_datafusion::sql(&ctx, "RESET vgi_exchange_input_dedup").await?;
    vgi_datafusion::sql(&ctx, "RESET vgi.vgi_result_cache_per_value").await?;
    assert_eq!(
        configured(&ctx, "vgi_exchange_input_dedup").as_deref(),
        Some("true")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_per_value").as_deref(),
        Some("true")
    );
    Ok(())
}

#[tokio::test]
async fn result_cache_limits_apply_immediately_and_reset_to_runtime_defaults(
) -> datafusion::common::Result<()> {
    let constructor_limits = vgi_client::CacheLimits {
        max_entry_bytes: 17,
        max_total_bytes: 31,
        max_entries: 2,
        ..vgi_client::CacheLimits::default()
    };
    let runtime = Arc::new(vgi_datafusion::VgiRuntime::new(
        vgi_datafusion::VgiSessionOptions {
            cache_limits: constructor_limits,
            ..vgi_datafusion::VgiSessionOptions::default()
        },
    ));
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_extension(Arc::clone(&runtime)));
    vgi_datafusion::sql(&ctx, "SELECT 1").await?;
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entry_bytes").as_deref(),
        Some("17")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_bytes").as_deref(),
        Some("31")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entries").as_deref(),
        Some("2")
    );

    vgi_datafusion::sql(&ctx, "SET vgi_result_cache_max_entry_bytes = 1").await?;
    vgi_datafusion::sql(&ctx, "SET vgi_result_cache_max_bytes = 2").await?;
    vgi_datafusion::sql(&ctx, "SET vgi_result_cache_max_entries = 3").await?;
    let limits = runtime.result_cache().limits();
    assert_eq!(limits.max_entry_bytes, 1);
    assert_eq!(limits.max_total_bytes, 2);
    assert_eq!(limits.max_entries, 3);

    vgi_datafusion::sql(&ctx, "RESET vgi.vgi_result_cache_max_entry_bytes").await?;
    vgi_datafusion::sql(&ctx, "RESET vgi_result_cache_max_bytes").await?;
    vgi_datafusion::sql(&ctx, "RESET vgi_result_cache_max_entries").await?;
    assert_eq!(runtime.result_cache().limits(), constructor_limits);
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entry_bytes").as_deref(),
        Some("17")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_bytes").as_deref(),
        Some("31")
    );
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entries").as_deref(),
        Some("2")
    );
    Ok(())
}

#[tokio::test]
async fn preinstalled_vgi_settings_remain_authoritative() -> datafusion::common::Result<()> {
    let runtime = Arc::new(vgi_datafusion::VgiRuntime::default());
    let mut supplied = vgi_datafusion::VgiSettings::default();
    supplied.set_value("vgi_result_cache_max_entry_bytes", "99")?;
    supplied.set_value("host_marker", "kept")?;
    let config = SessionConfig::new()
        .with_extension(Arc::clone(&runtime))
        .with_option_extension(supplied);
    let ctx = SessionContext::new_with_config(config);

    vgi_datafusion::sql(&ctx, "SELECT 1").await?;
    assert_eq!(
        configured(&ctx, "vgi_result_cache_max_entry_bytes").as_deref(),
        Some("99")
    );
    assert_eq!(configured(&ctx, "host_marker").as_deref(), Some("kept"));
    assert_eq!(runtime.result_cache().limits().max_entry_bytes, 99);
    Ok(())
}

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
