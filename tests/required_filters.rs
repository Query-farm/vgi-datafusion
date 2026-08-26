// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end enforcement of catalog-table `required_filters` metadata.

use datafusion::prelude::SessionContext;

mod common;

async fn attach(ctx: &SessionContext, worker: &std::path::Path) -> datafusion::common::Result<()> {
    let location = common::sql_quote(&worker.to_string_lossy());
    vgi_datafusion::sql(ctx, &format!("ATTACH 'example?location={location}' AS ex")).await?;
    Ok(())
}

async fn planning_error(ctx: &SessionContext, query: &str) -> String {
    let dataframe = vgi_datafusion::sql(ctx, query)
        .await
        .expect("SQL should reach physical planning");
    dataframe
        .create_physical_plan()
        .await
        .expect_err("required filters should reject the scan")
        .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn required_filter_cnf_rejects_missing_groups_and_accepts_or_members(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();
    attach(&ctx, &worker).await?;

    let error = planning_error(&ctx, "SELECT * FROM ex.data.rff_simple").await;
    assert!(
        error.contains("requires WHERE filters on: a") && error.contains("Missing: a"),
        "unexpected required-filter error: {error}"
    );

    let error = planning_error(&ctx, "SELECT * FROM ex.data.rff_multi WHERE top = 100").await;
    assert!(
        error.contains("Missing: s.a"),
        "unexpected required-filter error: {error}"
    );

    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.data.rff_or WHERE b = 20")
        .await?
        .collect()
        .await?;
    assert_eq!(rows.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);

    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.data.rff_none")
        .await?
        .collect()
        .await?;
    assert_eq!(rows.iter().map(|batch| batch.num_rows()).sum::<usize>(), 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn struct_subfields_are_precise_and_parent_filters_satisfy_descendants(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();
    attach(&ctx, &worker).await?;

    let error = planning_error(&ctx, "SELECT * FROM ex.data.rff_struct WHERE s.a = 1").await;
    assert!(
        error.contains("Missing: s.b"),
        "a filter on s.a must not masquerade as a filter on all of s: {error}"
    );

    let rows = vgi_datafusion::sql(
        &ctx,
        "SELECT s.a, s.b FROM ex.data.rff_struct WHERE s.a = 1 AND s.b = 10",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(rows.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);

    // The protocol deliberately defines a predicate on the containing struct
    // as sufficient for every required child path.
    let dataframe = vgi_datafusion::sql(
        &ctx,
        "SELECT s.a FROM ex.data.rff_struct WHERE s IS NOT NULL",
    )
    .await?;
    dataframe.create_physical_plan().await?;
    Ok(())
}
