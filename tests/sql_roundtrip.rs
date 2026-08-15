// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! SQL over a real VGI worker.
//!
//! The point of the whole adapter: DataFusion plans and executes ordinary SQL,
//! and the rows come from a worker process over Arrow IPC.

use std::path::PathBuf;

use datafusion::arrow::array::Array;
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use vgi_datafusion::{VgiConnection, VgiTableProvider};

/// Locate the worker built by the sibling `vgi-rust` workspace.
///
/// Anchored on `CARGO_MANIFEST_DIR` rather than counting `pop()`s up from
/// `current_exe` — an earlier version of this got that count wrong by one, and
/// every test in the file silently "passed" without ever reaching a worker.
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
            Some(p) => p,
            None => {
                eprintln!("skipping: vgi-example-worker not built");
                return Ok(());
            }
        }
    };
}

#[tokio::test(flavor = "multi_thread")]
async fn selects_from_a_remote_table_function() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    let table = VgiTableProvider::bind(conn, "example", "main", "ten_thousand").await?;
    ctx.register_table("remote", table)?;

    let batches = ctx.sql("SELECT * FROM remote").await?.collect().await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(rows > 0, "the remote scan produced no rows");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn datafusion_aggregates_over_remote_rows() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    ctx.register_table(
        "remote",
        VgiTableProvider::bind(conn, "example", "main", "ten_thousand").await?,
    )?;

    // The aggregate runs in DataFusion; only the scan is remote. `count(*)`
    // also exercises the empty-projection path: DataFusion asks for row counts
    // and zero columns.
    let batches = ctx
        .sql("SELECT count(*) AS n FROM remote")
        .await?
        .collect()
        .await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is int64")
        .value(0);
    assert_eq!(
        n, 10_000,
        "`ten_thousand` generates 10000 rows; a count of {n} means rows were lost or duplicated"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn projection_reaches_the_worker() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    let table = VgiTableProvider::bind(conn, "example", "main", "ten_thousand").await?;
    let width = table.schema().fields().len();
    ctx.register_table("remote", table)?;
    if width < 2 {
        return Ok(());
    }

    let first = ctx.table("remote").await?.schema().field(0).name().clone();
    let batches = ctx
        .sql(&format!("SELECT \"{first}\" FROM remote"))
        .await?
        .collect()
        .await?;
    for b in &batches {
        assert_eq!(b.num_columns(), 1, "projection should narrow the scan");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn the_plan_names_the_remote_scan() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    ctx.register_table(
        "remote",
        VgiTableProvider::bind(conn, "example", "main", "ten_thousand").await?,
    )?;

    let plan = ctx
        .sql("SELECT * FROM remote")
        .await?
        .create_physical_plan()
        .await?;
    let text = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        text.contains("VgiScanExec"),
        "the plan should show where the rows come from:\n{text}"
    );
    Ok(())
}
