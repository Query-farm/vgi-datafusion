// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! SQL control of the session-owned VGI result-cache bounds.

use crate::common;

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::execution::context::SessionConfig;
use datafusion::prelude::SessionContext;

#[tokio::test(flavor = "multi_thread")]
async fn sql_entry_limit_bounds_buffered_capture_and_reset_restores_caching(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let runtime = Arc::new(vgi_datafusion::VgiRuntime::default());
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::clone(&runtime)),
    );
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            common::sql_quote(&worker.to_string_lossy())
        ),
    )
    .await?;

    const QUERY: &str = "SELECT x FROM ex.main.cached_sum_all(\
                         (SELECT x FROM range(10) t(x)), logging := false)";
    vgi_datafusion::sql(&ctx, "SET vgi_result_cache_max_entry_bytes = 1").await?;
    for expected_aborts in 1..=2 {
        let batches = vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("buffered sum is Int64")
                .value(0),
            45
        );
        assert_eq!(runtime.result_cache().stats().entries, 0);
        assert_eq!(
            runtime.result_cache().stats().capture_aborts,
            expected_aborts
        );
    }

    vgi_datafusion::sql(&ctx, "RESET vgi_result_cache_max_entry_bytes").await?;
    vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    let hits_before = runtime.result_cache().stats().hits;
    vgi_datafusion::sql(&ctx, QUERY).await?.collect().await?;
    let stats = runtime.result_cache().stats();
    assert_eq!(stats.entries, 1);
    assert!(
        stats.hits > hits_before,
        "repeat is served from result cache"
    );
    Ok(())
}
