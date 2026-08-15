// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! `ATTACH` driven entirely from SQL text.
//!
//! These tests exist to prove one thing the unit tests cannot: that a session
//! reaches a real worker's rows without a single Rust-side `register_table` or
//! `register_catalog` call. Every statement below is a string. That is the
//! precondition for replaying a `.test` corpus.

use std::path::PathBuf;

use datafusion::prelude::SessionContext;

/// Locate the worker built by the sibling `vgi-rust` workspace.
fn example_worker() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("vgi-rust")
        .join("target");
    for profile in ["debug", "release"] {
        let exe = root.join(profile).join(if cfg!(windows) {
            "vgi-example-worker.exe"
        } else {
            "vgi-example-worker"
        });
        if exe.exists() {
            return Some(exe);
        }
    }
    None
}

macro_rules! skip_without_worker {
    () => {
        match example_worker() {
            Some(p) => p.to_string_lossy().to_string(),
            None => {
                eprintln!("skipping: vgi-example-worker not built");
                return Ok(());
            }
        }
    };
}

#[tokio::test(flavor = "multi_thread")]
async fn attach_then_select_is_pure_sql() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();

    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={worker}' AS ex")).await?;

    let batches = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.ten_thousand")
        .await?
        .collect()
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(rows > 0, "attached scan produced no rows");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn ordinary_statements_pass_straight_through() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={worker}' AS ex")).await?;

    // A non-ATTACH statement is planned by DataFusion untouched — including one
    // that mixes a local relation with the remote catalog.
    vgi_datafusion::sql(&ctx, "CREATE TABLE local AS SELECT 1 AS x").await?;
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) AS n FROM ex.main.ten_thousand CROSS JOIN local",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_dialect_parses_the_same_attach() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();

    // Under the DuckDB dialect the statement lands in a different sqlparser
    // variant (`AttachDuckDBDatabase`), and the trailing `(TYPE VGI)` — the one
    // option sqlparser does model — parses and is ignored.
    vgi_datafusion::sql(&ctx, "SET datafusion.sql_parser.dialect = 'DuckDB'").await?;
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example?location={worker}' AS ex (TYPE VGI)"),
    )
    .await?;

    let batches = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.ten_thousand LIMIT 5")
        .await?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn detach_makes_the_tables_unreachable() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, "SET datafusion.sql_parser.dialect = 'DuckDB'").await?;
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={worker}' AS ex")).await?;
    vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.ten_thousand LIMIT 1")
        .await?
        .collect()
        .await?;

    vgi_datafusion::sql(&ctx, "DETACH ex").await?;
    let err = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.ten_thousand LIMIT 1")
        .await
        .expect_err("the table should be gone after DETACH");
    assert!(
        err.to_string().contains("ten_thousand"),
        "unhelpful post-DETACH error: {err}"
    );

    // Detaching something that was never attached is an error, not a no-op.
    assert!(vgi_datafusion::sql(&ctx, "DETACH nope").await.is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bad_attach_reports_the_fix() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    let err = vgi_datafusion::sql(&ctx, "ATTACH 'example' AS ex")
        .await
        .expect_err("no location given");
    assert!(err.to_string().contains("location=<worker>"), "{err}");
    Ok(())
}
