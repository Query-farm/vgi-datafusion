// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! A pushed LIMIT must stop a streaming TABLE-input exchange before its child
//! is materialized. Run with an HTTP `VGI_TEST_WORKER` to cover the transport
//! whose request-per-batch cost made the eager path reliably time out.

use std::time::Duration;

use datafusion::common::ScalarValue;
use datafusion::prelude::SessionContext;

#[tokio::test(flavor = "multi_thread")]
async fn large_table_input_limit_is_incremental() -> datafusion::common::Result<()> {
    let Ok(worker) = std::env::var("VGI_TEST_WORKER") else {
        eprintln!("skipping: set VGI_TEST_WORKER to an HTTP example worker");
        return Ok(());
    };
    if !worker.starts_with("http://") && !worker.starts_with("https://") {
        eprintln!("skipping: focused regression requires an HTTP VGI_TEST_WORKER");
        return Ok(());
    }

    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}', pool false)",
            worker.replace('\'', "''")
        ),
    )
    .await?;

    let dataframe = vgi_datafusion::sql(
        &ctx,
        "SELECT min(v), max(v), count(*) FROM (\
             SELECT * FROM ex.main.echo((\
                 SELECT i AS v FROM range(1, 100000001) t(i)\
             )) LIMIT 5\
         ) limited",
    )
    .await?;
    let plan = dataframe.create_physical_plan().await?;
    let rendered = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        rendered.contains("VgiLimitedTableInputExec")
            && rendered.contains("limit=5")
            && rendered.contains("cache=disabled(partial_exchange)"),
        "pushed limit did not select the incremental TABLE-input plan:\n{rendered}"
    );

    let batches = tokio::time::timeout(
        Duration::from_secs(5),
        datafusion::physical_plan::collect(plan, ctx.task_ctx()),
    )
    .await
    .expect("LIMIT 5 must not consume the 100-million-row child")?;
    let batch = batches.first().expect("aggregate result batch");
    let values = (0..3)
        .map(|column| ScalarValue::try_from_array(batch.column(column), 0))
        .collect::<datafusion::common::Result<Vec<_>>>()?;
    assert_eq!(
        values,
        vec![
            ScalarValue::Int64(Some(1)),
            ScalarValue::Int64(Some(5)),
            ScalarValue::Int64(Some(5)),
        ]
    );
    Ok(())
}
