// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! `TABLESAMPLE SYSTEM` reaches VGI plan/init through DataFusion's relation
//! planner extension point, while DataFusion-owned sampling stays local.

use crate::common;

use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::prelude::SessionContext;

#[tokio::test(flavor = "multi_thread")]
async fn system_sample_reaches_worker_and_bernoulli_does_not() -> datafusion::error::Result<()> {
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

    let dataframe = vgi_datafusion::sql(
        &ctx,
        "SELECT sample_percentage, sample_seed
         FROM example.main.sample_echo(3)
         TABLESAMPLE SYSTEM(50 PERCENT) REPEATABLE(42)",
    )
    .await?;
    let plan = dataframe.create_physical_plan().await?;
    let rendered = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        rendered.contains("sample=50 percent") && rendered.contains("sample_seed=42"),
        "sample hint missing from VGI scan:\n{rendered}"
    );
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        3
    );
    for batch in &batches {
        let percentages = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("sample_percentage is Float64");
        let seeds = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("sample_seed is Int64");
        assert!(percentages.iter().flatten().all(|value| value == 50.0));
        assert!(seeds.iter().flatten().all(|value| value == 42));
    }

    let bernoulli = vgi_datafusion::sql(
        &ctx,
        "SELECT sample_percentage, sample_seed
         FROM example.main.sample_echo(3)
         TABLESAMPLE BERNOULLI(100 PERCENT)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        bernoulli
            .iter()
            .map(|batch| batch.num_rows())
            .sum::<usize>(),
        3
    );
    for batch in &bernoulli {
        let percentages = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("sample_percentage is Float64");
        let seeds = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("sample_seed is Int64");
        assert!(percentages.iter().flatten().all(|value| value == -1.0));
        assert!(seeds.iter().flatten().all(|value| value == -1));
    }
    Ok(())
}
