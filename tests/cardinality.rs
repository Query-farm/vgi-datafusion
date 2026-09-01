// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Catalog-inlined VGI cardinality reaches DataFusion's existing logical and
//! physical statistics APIs without a per-bind cardinality round trip.

use crate::common;

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::common::stats::Precision;
use datafusion::physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

fn vgi_scan_rows(plan: &dyn ExecutionPlan) -> Option<Precision<usize>> {
    if plan.name() == "VgiScanExec" {
        return Some(
            StatisticsContext::new()
                .compute(plan, &StatisticsArgs::new())
                .expect("VGI statistics compute")
                .num_rows,
        );
    }
    plan.children()
        .into_iter()
        .find_map(|child| vgi_scan_rows(child.as_ref()))
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_catalog_cardinality_reaches_datafusion_statistics() -> datafusion::error::Result<()>
{
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = common::sql_quote(&worker.to_string_lossy());
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example?location={worker}' AS example"),
    )
    .await?;

    let provider = ctx
        .catalog("example")
        .expect("attached catalog")
        .schema("data")
        .expect("data schema")
        .table("cardinality_inlined_table")
        .await?
        .expect("cardinality table");
    assert_eq!(
        provider.statistics().expect("inlined statistics").num_rows,
        Precision::Exact(10_000)
    );

    let dataframe =
        vgi_datafusion::sql(&ctx, "SELECT * FROM example.data.cardinality_inlined_table").await?;
    let plan: Arc<dyn ExecutionPlan> = dataframe.create_physical_plan().await?;
    assert_eq!(vgi_scan_rows(plan.as_ref()), Some(Precision::Exact(10_000)));

    let logs = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM vgi_logs()
         WHERE event = 'vgi.cardinality.inlined'
           AND function = 'data.cardinality_inlined_table'",
    )
    .await?
    .collect()
    .await?;
    let count = logs[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert!(count >= 1, "inlined-cardinality event was not exposed");
    Ok(())
}
