// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! DataFusion-native lifecycle coverage for worker-nominated global aliases.

mod common;

use datafusion::arrow::array::{Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, TimeUnit};
use datafusion::logical_expr::{create_udf, ScalarUDF, Volatility};
use datafusion::prelude::SessionContext;
use std::sync::Arc;

async fn attach(
    ctx: &SessionContext,
    worker: &std::path::Path,
    alias: &str,
    publish: &str,
) -> datafusion::common::Result<()> {
    vgi_datafusion::sql(
        ctx,
        &format!(
            "ATTACH 'example' AS {alias} (TYPE vgi, LOCATION '{}', global_functions {publish})",
            common::sql_quote(&worker.to_string_lossy())
        ),
    )
    .await?;
    Ok(())
}

async fn count(ctx: &SessionContext, query: &str) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(ctx, query).await?.collect().await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0))
}

async fn string_value(ctx: &SessionContext, query: &str) -> datafusion::common::Result<String> {
    let batches = vgi_datafusion::sql(ctx, query).await?.collect().await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0)
        .to_string())
}

async fn assert_planning_error(ctx: &SessionContext, query: &str) {
    if let Ok(_) = vgi_datafusion::sql(ctx, query).await {
        panic!("expected planning to fail: {query}");
    }
}

fn native_global_scalar() -> ScalarUDF {
    create_udf(
        "vgi_example_global_scalar",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(|arguments| Ok(arguments[0].clone())),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_discovery_matches_the_canonical_schema() -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();
    let batches = vgi_datafusion::sql(
        &ctx,
        &format!(
            "SELECT * FROM vgi_catalogs('{}') WHERE catalog = 'accumulate'",
            common::sql_quote(&worker.to_string_lossy())
        ),
    )
    .await?
    .collect()
    .await?;
    let batch = &batches[0];
    let schema = batch.schema();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        vec![
            "catalog",
            "implementation_version",
            "data_version_spec",
            "attach_options",
            "releases",
            "source_url",
        ]
    );
    assert_eq!(
        batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "2.0.0"
    );
    let DataType::List(option_item) = schema.field(3).data_type() else {
        panic!("attach_options must be a list");
    };
    let DataType::Struct(option_fields) = option_item.data_type() else {
        panic!("attach_options items must be structs");
    };
    assert_eq!(
        option_fields
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        vec!["name", "description", "type", "default_value", "required"]
    );
    let DataType::List(release_item) = schema.field(4).data_type() else {
        panic!("releases must be a list");
    };
    let DataType::Struct(release_fields) = release_item.data_type() else {
        panic!("release items must be structs");
    };
    assert_eq!(
        release_fields
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        vec!["version", "released_at", "summary", "notes_url"]
    );
    assert!(matches!(
        release_fields[1].data_type(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) if timezone.as_ref() == "UTC"
    ));
    assert!(batch.column(5).is_null(0));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn opt_out_collision_ownership_detach_and_reattach_use_datafusion_registries(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();

    // SQL boolean spelling is case-insensitive, including local ATTACH policy.
    attach(&ctx, &worker, "quiet", "FALSE").await?;
    assert_eq!(
        count(&ctx, "SELECT count(*) FROM vgi_global_functions()").await?,
        0
    );
    assert_eq!(
        string_value(&ctx, "SELECT quiet.main.global_scalar(1)").await?,
        "global_scalar:1"
    );
    assert_planning_error(&ctx, "SELECT vgi_example_global_scalar(1)").await;
    vgi_datafusion::sql(&ctx, "DETACH quiet").await?;

    attach(&ctx, &worker, "owner", "TRUE").await?;
    attach(&ctx, &worker, "later", "true").await?;
    assert_eq!(
        count(&ctx, "SELECT count(*) FROM vgi_global_functions()").await?,
        4
    );
    let owners = vgi_datafusion::sql(
        &ctx,
        "SELECT DISTINCT catalog_name FROM vgi_global_functions()",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        owners[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "owner"
    );
    assert_eq!(
        string_value(&ctx, "SELECT vgi_example_global_scalar(7)").await?,
        "global_scalar:7"
    );
    assert_eq!(
        string_value(&ctx, "SELECT vgi_example_global_scalar('hello')").await?,
        "global_scalar_text:hello"
    );
    assert_eq!(
        count(&ctx, "SELECT count(*) FROM vgi_example_global_table()").await?,
        3
    );
    assert_eq!(
        count(
            &ctx,
            "SELECT vgi_example_global_agg(v) \
             FROM (VALUES (2::BIGINT), (3::BIGINT)) t(v)"
        )
        .await?,
        5
    );
    let worker_path = string_value(
        &ctx,
        "SELECT worker_path FROM vgi_global_functions() ORDER BY global_name LIMIT 1",
    )
    .await?;
    assert!(!worker_path.is_empty());

    // Invalid local policy is rejected before re-ATTACH can replace any of the
    // existing catalog's registrations or metadata.
    let invalid = format!(
        "ATTACH 'example' AS owner (TYPE vgi, LOCATION '{}', global_functions perhaps)",
        common::sql_quote(&worker.to_string_lossy())
    );
    assert!(vgi_datafusion::sql(&ctx, &invalid).await.is_err());
    assert_eq!(
        string_value(&ctx, "SELECT vgi_example_global_scalar(8)").await?,
        "global_scalar:8"
    );
    assert_eq!(
        string_value(&ctx, "SELECT owner.main.global_scalar(8)").await?,
        "global_scalar:8"
    );

    vgi_datafusion::sql(&ctx, "DETACH later").await?;
    assert_eq!(
        string_value(&ctx, "SELECT vgi_example_global_scalar(9)").await?,
        "global_scalar:9"
    );
    vgi_datafusion::sql(&ctx, "DETACH owner").await?;
    assert_eq!(
        count(&ctx, "SELECT count(*) FROM vgi_global_functions()").await?,
        0
    );
    assert_planning_error(&ctx, "SELECT vgi_example_global_scalar(1)").await;

    attach(&ctx, &worker, "owner", "true").await?;
    assert_eq!(
        count(
            &ctx,
            "SELECT count(*) FROM vgi_global_functions() WHERE live"
        )
        .await?,
        4
    );
    assert_eq!(
        string_value(&ctx, "SELECT vgi_example_global_scalar(9)").await?,
        "global_scalar:9"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn native_collision_and_replacement_survive_detach() -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };

    let collision = SessionContext::new();
    collision.register_udf(native_global_scalar());
    attach(&collision, &worker, "colliding", "true").await?;
    assert_eq!(
        count(&collision, "SELECT count(*) FROM vgi_global_functions()").await?,
        3,
        "the native scalar owns its collision while the other globals publish"
    );
    assert_eq!(
        count(&collision, "SELECT vgi_example_global_scalar(17)").await?,
        17
    );
    vgi_datafusion::sql(&collision, "DETACH colliding").await?;
    assert_eq!(
        count(&collision, "SELECT vgi_example_global_scalar(18)").await?,
        18,
        "DETACH must not remove the earlier native owner"
    );

    let replacement = SessionContext::new();
    attach(&replacement, &worker, "replaced", "true").await?;
    assert_eq!(
        string_value(&replacement, "SELECT vgi_example_global_scalar(19)").await?,
        "global_scalar:19"
    );
    replacement.register_udf(native_global_scalar());
    assert_eq!(
        count(&replacement, "SELECT vgi_example_global_scalar(20)").await?,
        20
    );
    vgi_datafusion::sql(&replacement, "DETACH replaced").await?;
    assert_eq!(
        count(&replacement, "SELECT vgi_example_global_scalar(21)").await?,
        21,
        "DETACH must not remove a newer native replacement"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_attaches_have_one_linearized_global_owner() -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();
    let (left, right) = tokio::join!(
        attach(&ctx, &worker, "left", "true"),
        attach(&ctx, &worker, "right", "true")
    );
    left?;
    right?;

    assert_eq!(
        count(&ctx, "SELECT count(*) FROM vgi_global_functions()").await?,
        4
    );
    let owner = string_value(
        &ctx,
        "SELECT DISTINCT catalog_name FROM vgi_global_functions()",
    )
    .await?;
    let loser = if owner == "left" { "right" } else { "left" };
    vgi_datafusion::sql(&ctx, &format!("DETACH {loser}")).await?;
    assert_eq!(
        string_value(&ctx, "SELECT vgi_example_global_scalar('concurrent')").await?,
        "global_scalar_text:concurrent"
    );
    vgi_datafusion::sql(&ctx, &format!("DETACH {owner}")).await?;
    assert_planning_error(&ctx, "SELECT vgi_example_global_scalar(1)").await;
    Ok(())
}
