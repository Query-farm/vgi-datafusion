// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! `ATTACH` driven entirely from SQL text.
//!
//! These tests exist to prove one thing the unit tests cannot: that a session
//! reaches a real worker's rows without a single Rust-side `register_table` or
//! `register_catalog` call. Every statement below is a string. That is the
//! precondition for replaying a `.test` corpus.

use std::path::PathBuf;

use datafusion::arrow::array::StringArray;
use datafusion::common::Constraint;
use datafusion::prelude::{SessionConfig, SessionContext};

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
async fn vgi_time_travel_preserves_schema_data_and_pushdown() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example?location={worker}' AS example"),
    )
    .await?;

    let v1 = vgi_datafusion::sql(
        &ctx,
        "SELECT * FROM example.data.versioned_data AT (VERSION => 1) ORDER BY id",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(v1[0].schema().fields().len(), 1);
    assert_eq!(v1.iter().map(|batch| batch.num_rows()).sum::<usize>(), 3);

    let v2 = vgi_datafusion::sql(
        &ctx,
        "SELECT * FROM example.data.versioned_data AT (TIMESTAMP => TIMESTAMP '2021-06-15') ORDER BY id",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(v2[0].schema().fields().len(), 4);
    assert_eq!(v2.iter().map(|batch| batch.num_rows()).sum::<usize>(), 5);

    let pushed = vgi_datafusion::sql(
        &ctx,
        "SELECT id FROM example.data.tt_pushdown_fn AT (VERSION => 2) WHERE id >= 8 ORDER BY id",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        pushed.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        3
    );

    let explained = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM example.data.versioned_data AT (VERSION => 1)",
    )
    .await?
    .collect()
    .await?;
    assert!(
        !explained.is_empty(),
        "EXPLAIN should retain the historical provider"
    );

    let missing = vgi_datafusion::sql(
        &ctx,
        "SELECT score FROM example.data.versioned_data AT (VERSION => 1)",
    )
    .await
    .expect_err("version 1 has no score column");
    assert!(missing.to_string().contains("score"), "{missing}");

    let unsupported =
        vgi_datafusion::sql(&ctx, "SELECT * FROM example.data.numbers AT (VERSION => 1)")
            .await
            .expect_err("numbers does not support time travel");
    assert!(
        unsupported
            .to_string()
            .contains("does not support time travel"),
        "{unsupported}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn datafusion_metadata_lists_vgi_relations_without_binding_functions(
) -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example?location={worker}' AS example"),
    )
    .await?;

    let tables = vgi_datafusion::sql(&ctx, "SHOW TABLES")
        .await?
        .collect()
        .await?;
    let mut saw_versioned_data = false;
    let mut saw_table_function = false;
    for batch in &tables {
        let catalogs = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let schemas = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let names = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            if catalogs.value(row) == "example"
                && schemas.value(row) == "data"
                && names.value(row) == "versioned_data"
            {
                saw_versioned_data = true;
            }
            if catalogs.value(row) == "example" && names.value(row) == "make_pairs" {
                saw_table_function = true;
            }
        }
    }
    assert!(
        saw_versioned_data,
        "SHOW TABLES omitted the VGI catalog table"
    );
    assert!(
        !saw_table_function,
        "SHOW TABLES must not bind or list a table function"
    );

    let schemas = vgi_datafusion::sql(
        &ctx,
        "SELECT catalog_name, schema_name FROM information_schema.schemata \
         WHERE catalog_name = 'example' AND schema_name = 'data'",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        schemas.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );

    let functions = vgi_datafusion::sql(&ctx, "SHOW FUNCTIONS LIKE '%example_sequence%'")
        .await?
        .collect()
        .await?;
    assert!(
        functions
            .iter()
            .map(|batch| batch.num_rows())
            .sum::<usize>()
            > 0,
        "the VGI table function should appear as a routine: {functions:?}"
    );

    let columns = vgi_datafusion::sql(&ctx, "SHOW COLUMNS FROM example.data.versioned_data")
        .await?
        .collect()
        .await?;
    assert!(
        columns.iter().map(|batch| batch.num_rows()).sum::<usize>() >= 2,
        "SHOW COLUMNS should expose the current historical-table schema"
    );

    let views = vgi_datafusion::sql(
        &ctx,
        "SELECT table_name FROM information_schema.views \
         WHERE table_catalog = 'example' AND table_schema = 'main'",
    )
    .await?
    .collect()
    .await?;
    assert!(
        views.iter().map(|batch| batch.num_rows()).sum::<usize>() > 0,
        "information_schema.views should expose worker views"
    );

    let table = ctx
        .catalog("example")
        .unwrap()
        .schema("data")
        .unwrap()
        .table("versioned_constraints")
        .await?
        .unwrap();
    let constraints = table.constraints().expect("VGI table supports constraints");
    assert!(constraints.iter().any(
        |constraint| matches!(constraint, Constraint::PrimaryKey(columns) if columns == &[0])
    ));
    assert!(constraints
        .iter()
        .any(|constraint| matches!(constraint, Constraint::Unique(columns) if columns == &[2])));
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
    vgi_datafusion::sql(&ctx, "SELECT vgi_example_global_scalar(1)")
        .await?
        .collect()
        .await?;

    let events = vgi_datafusion::sql(&ctx, "SELECT sum(count) AS events FROM vgi_log_stats()")
        .await?
        .collect()
        .await?;
    let event_count = events[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::UInt64Array>()
        .expect("sum of UInt64 event counts")
        .value(0);
    assert!(event_count > 0, "the session should retain VGI events");

    vgi_datafusion::sql(&ctx, "DETACH ex").await?;
    let err = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.ten_thousand LIMIT 1")
        .await
        .expect_err("the table should be gone after DETACH");
    assert!(
        err.to_string().contains("ten_thousand"),
        "unhelpful post-DETACH error: {err}"
    );
    let global_err = vgi_datafusion::sql(&ctx, "SELECT vgi_example_global_scalar(1)")
        .await
        .expect_err("DETACH should deregister alias-owned global functions");
    assert!(
        global_err.to_string().contains("vgi_example_global_scalar"),
        "unhelpful global-function error after DETACH: {global_err}"
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
