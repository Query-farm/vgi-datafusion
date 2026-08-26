// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! EXPLAIN ANALYZE is the only execution shape that asks a worker for its
//! post-execution `dynamic_to_string` diagnostics.

mod common;

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use vgi_client::Arguments;
use vgi_datafusion::{VgiConnection, VgiRuntime, VgiSessionStateBuilderExt, VgiTableProvider};

fn callback_events(runtime: &VgiRuntime) -> usize {
    runtime
        .events()
        .iter()
        .filter(|event| event.kind == "table_function.dynamic_to_string")
        .count()
}

async fn rendered(context: &SessionContext, sql: &str) -> datafusion::common::Result<String> {
    let batches = context.sql(sql).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn profiling_is_analyze_only_and_skips_limit_and_cache_replay(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };

    let runtime = Arc::new(VgiRuntime::default());
    let connection = VgiConnection::subprocess([worker.to_string_lossy().to_string()])
        .with_runtime(Arc::clone(&runtime));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_vgi_physical_optimizer()
        .build();
    let context = SessionContext::new_with_state(state);

    context.register_table(
        "remote_profile",
        VgiTableProvider::bind_with_arguments(
            connection.clone(),
            "example",
            "main",
            "profiling_demo",
            Arguments::new().positional(500_i64),
        )
        .await?,
    )?;

    runtime.clear_events();
    context
        .sql("SELECT sum(n) FROM remote_profile")
        .await?
        .collect()
        .await?;
    assert_eq!(callback_events(&runtime), 0, "ordinary query callback");

    runtime.clear_events();
    let plain = rendered(&context, "EXPLAIN SELECT * FROM remote_profile").await?;
    assert!(!plain.contains("rows_produced:"), "{plain}");
    assert!(!plain.contains("Batch Bytes:"), "{plain}");
    assert_eq!(callback_events(&runtime), 0, "plain EXPLAIN callback");

    runtime.clear_events();
    let analyzed = rendered(
        &context,
        "EXPLAIN ANALYZE SELECT count(*) FROM remote_profile",
    )
    .await?;
    assert!(analyzed.contains("Worker:"), "{analyzed}");
    assert!(analyzed.contains("Function: profiling_demo"), "{analyzed}");
    assert!(analyzed.contains("rows_produced: 500"), "{analyzed}");
    assert!(analyzed.contains("batches_emitted:"), "{analyzed}");
    assert!(analyzed.contains("elapsed_ms:"), "{analyzed}");
    assert!(analyzed.contains("Batches:"), "{analyzed}");
    assert!(analyzed.contains("Batch Bytes:"), "{analyzed}");
    assert!(analyzed.contains("rows: min"), "{analyzed}");
    assert_eq!(callback_events(&runtime), 1, "completed scan callback");

    context.register_table(
        "remote_profile_cacheable",
        VgiTableProvider::bind_with_arguments(
            connection.clone(),
            "example",
            "main",
            "profiling_demo",
            Arguments::new()
                .positional(32_i64)
                .named("cache_ttl", 300_i64),
        )
        .await?,
    )?;
    runtime.clear_events();
    let cacheable = rendered(
        &context,
        "EXPLAIN ANALYZE SELECT sum(n) FROM remote_profile_cacheable",
    )
    .await?;
    assert!(cacheable.contains("rows_produced: 32"), "{cacheable}");
    let events = runtime.events();
    let stored = events
        .iter()
        .position(|event| event.kind == "cache.store")
        .expect("completed profile result was cached");
    let profiled = events
        .iter()
        .position(|event| event.kind == "table_function.dynamic_to_string")
        .expect("completed profile callback was observed");
    assert!(
        stored < profiled,
        "cache publication must precede optional profiling callback: {events:?}"
    );

    runtime.clear_events();
    let cached_profile = rendered(
        &context,
        "EXPLAIN ANALYZE SELECT sum(n) FROM remote_profile_cacheable",
    )
    .await?;
    assert!(cached_profile.contains("cache_hits=1"), "{cached_profile}");
    assert_eq!(
        callback_events(&runtime),
        0,
        "profile cache replay callback"
    );

    context.register_table(
        "remote_profile_large",
        VgiTableProvider::bind_with_arguments(
            connection.clone(),
            "example",
            "main",
            "profiling_demo",
            Arguments::new().positional(1_000_000_i64),
        )
        .await?,
    )?;
    runtime.clear_events();
    let limited = rendered(
        &context,
        "EXPLAIN ANALYZE SELECT * FROM remote_profile_large LIMIT 10",
    )
    .await?;
    assert!(limited.contains("Function: profiling_demo"), "{limited}");
    assert!(!limited.contains("rows_produced:"), "{limited}");
    assert_eq!(callback_events(&runtime), 0, "LIMIT callback");

    context.register_table(
        "remote_error",
        VgiTableProvider::bind_with_arguments(
            connection.clone(),
            "example",
            "main",
            "cache_poison",
            Arguments::new(),
        )
        .await?,
    )?;
    runtime.clear_events();
    let error = context
        .sql("EXPLAIN ANALYZE SELECT * FROM remote_error")
        .await?
        .collect()
        .await
        .expect_err("fixture fails after its first batch");
    assert!(error.to_string().contains("intentional mid-stream failure"));
    assert_eq!(callback_events(&runtime), 0, "failed scan callback");

    context.register_table(
        "remote_cacheable",
        VgiTableProvider::bind_with_arguments(
            connection,
            "example",
            "main",
            "cacheable_numbers",
            Arguments::new().named("n", 16_i64),
        )
        .await?,
    )?;
    context
        .sql("SELECT sum(n) FROM remote_cacheable")
        .await?
        .collect()
        .await?;
    runtime.clear_events();
    let replay = rendered(
        &context,
        "EXPLAIN ANALYZE SELECT sum(n) FROM remote_cacheable",
    )
    .await?;
    assert!(replay.contains("cache_hits=1"), "{replay}");
    assert_eq!(callback_events(&runtime), 0, "cache replay callback");

    Ok(())
}
