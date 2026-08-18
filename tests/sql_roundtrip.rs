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

/// Find the schema that declares `filter_echo_table_scan`.
///
/// It lives in a non-default schema, so the test discovers it rather than
/// hardcoding a name that a catalog rearrangement would silently break.
async fn filter_echo_schema(conn: &VgiConnection) -> Option<String> {
    let c = conn.clone();
    tokio::task::spawn_blocking(move || {
        let mut client = c.connect().ok()?;
        let cat = client
            .attach("example", vgi_client::AttachOptions::default())
            .ok()?;
        for s in client.schemas(&cat).ok()? {
            let fns = client
                .functions(&cat, &s.name, vgi_client::FunctionKind::Table)
                .ok()?;
            if fns.iter().any(|f| f.name == "filter_echo_table_scan") {
                return Some(s.name);
            }
        }
        None
    })
    .await
    .ok()
    .flatten()
}

/// Read the single distinct value of the `pushed_filters` column.
fn pushed_filters(batches: &[datafusion::arrow::array::RecordBatch]) -> String {
    use datafusion::arrow::array::StringArray;
    for b in batches {
        let Some(idx) = b.schema().index_of("pushed_filters").ok() else {
            continue;
        };
        let col = b
            .column(idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        if col.len() > 0 && !col.is_null(0) {
            return col.value(0).to_string();
        }
    }
    String::new()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_predicate_reaches_the_worker_and_the_rows_are_right() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
    let Some(schema) = filter_echo_schema(&conn).await else {
        eprintln!("skipping: filter_echo_table_scan not in this catalog");
        return Ok(());
    };

    let ctx = SessionContext::new();
    ctx.register_table(
        "echo",
        VgiTableProvider::bind(conn, "example", &schema, "filter_echo_table_scan").await?,
    )?;

    // The fixture generates n = 0..99 and echoes whatever filters it was handed.
    let batches = ctx
        .sql("SELECT n, pushed_filters FROM echo WHERE n > 90")
        .await?
        .collect()
        .await?;

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 9, "n > 90 over 0..99 is nine rows");

    let echoed = pushed_filters(&batches);
    assert!(
        !echoed.is_empty(),
        "the worker reported no pushed filters — the predicate never reached it, \
         and DataFusion filtered locally instead"
    );
    assert!(
        echoed.contains('n') && echoed.contains("90"),
        "the worker should have received a predicate on n against 90; got {echoed:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unpushable_predicate_still_produces_correct_rows() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
    let Some(schema) = filter_echo_schema(&conn).await else {
        return Ok(());
    };

    let ctx = SessionContext::new();
    ctx.register_table(
        "echo",
        VgiTableProvider::bind(conn, "example", &schema, "filter_echo_table_scan").await?,
    )?;

    // A column-to-column comparison cannot be expressed on the wire, so nothing
    // is pushed — DataFusion must still return the right answer. `s` is
    // 'row_<n>', so this matches nothing.
    let batches = ctx
        .sql("SELECT n FROM echo WHERE s = CAST(n AS VARCHAR)")
        .await?
        .collect()
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 0, "'row_<n>' never equals '<n>'");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_conjunction_narrows_correctly() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
    let Some(schema) = filter_echo_schema(&conn).await else {
        return Ok(());
    };

    let ctx = SessionContext::new();
    ctx.register_table(
        "echo",
        VgiTableProvider::bind(conn, "example", &schema, "filter_echo_table_scan").await?,
    )?;

    let batches = ctx
        .sql("SELECT n FROM echo WHERE n >= 10 AND n < 20")
        .await?
        .collect()
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 10, "[10, 20) is ten rows");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_disjunction_widens_correctly() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
    let Some(schema) = filter_echo_schema(&conn).await else {
        return Ok(());
    };

    let ctx = SessionContext::new();
    ctx.register_table(
        "echo",
        VgiTableProvider::bind(conn, "example", &schema, "filter_echo_table_scan").await?,
    )?;

    // An OR is the case where pushing half the predicate would lose rows, so
    // this is the shape most likely to expose a translation mistake.
    let batches = ctx
        .sql("SELECT n FROM echo WHERE n < 5 OR n > 95")
        .await?
        .collect()
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 9, "0..4 plus 96..99");
    Ok(())
}

// --- splits ----------------------------------------------------------------
//
// A split scan is the reason this adapter can be parallel at all. Before it,
// partition 0 did the work and the rest were empty: correct, but serial, because
// joining an existing execution needs an execution id shared across partitions
// and there is no rendezvous for that at planning time. A split token names its
// own work, so each partition redeems independently and no rendezvous is needed.

/// The baseline every other split assertion rests on: a split scan must return
/// row-for-row what the same data returns unsplit. If this disagrees, nothing
/// else about splits is meaningful.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_scan_returns_every_row_exactly_once() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    ctx.register_table(
        "remote",
        VgiTableProvider::bind_with_arguments(
            conn,
            "example",
            "main",
            "split_sequence",
            vgi_client::Arguments::new()
                .named("n", 1000i64)
                .named("splits", 17i64),
        )
        .await?,
    )?;

    // count(*) and count(DISTINCT) together: a duplicated split shows up in the
    // first, a dropped one in both, and neither shows up in a sum alone.
    let batches = ctx
        .sql("SELECT count(*) AS n, count(DISTINCT n) AS d FROM remote")
        .await?
        .collect()
        .await?;
    let col = |i: usize| {
        batches[0]
            .column(i)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("count is int64")
            .value(0)
    };
    assert_eq!(col(0), 1000, "rows were lost or duplicated across splits");
    assert_eq!(col(1), 1000, "splits overlapped: {} distinct", col(1));
    Ok(())
}

/// Splitting must be invisible to the answer, so the split COUNT must not change
/// it — including counts that do not divide the row count evenly, which is where
/// an off-by-one in the range arithmetic shows up as a missing boundary row.
#[tokio::test(flavor = "multi_thread")]
async fn the_split_count_does_not_change_the_answer() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();

    for splits in [1i64, 3, 8, 37] {
        let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
        let ctx = SessionContext::new();
        ctx.register_table(
            "remote",
            VgiTableProvider::bind_with_arguments(
                conn,
                "example",
                "main",
                "split_sequence",
                vgi_client::Arguments::new()
                    .named("n", 250i64)
                    .named("splits", splits),
            )
            .await?,
        )?;

        let batches = ctx
            .sql("SELECT count(*) AS c, sum(n) AS s FROM remote")
            .await?
            .collect()
            .await?;
        let c = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("count is int64")
            .value(0);
        assert_eq!(c, 250, "{splits} splits produced {c} rows, expected 250");
    }
    Ok(())
}

/// Zero splits is legal and means "no work" — a fully-pruned scan reaches it.
///
/// It must produce an EMPTY RESULT rather than an error, and the plan must still
/// clamp to one partition: `UnknownPartitioning(0)` makes `CoalescePartitionsExec`
/// fail outright and partition statistics assert on the index, so a literal zero
/// would turn a legal empty answer into an internal error.
#[tokio::test(flavor = "multi_thread")]
async fn zero_splits_is_an_empty_result_not_an_error() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    ctx.register_table(
        "remote",
        VgiTableProvider::bind_with_arguments(
            conn,
            "example",
            "main",
            "split_zero",
            vgi_client::Arguments::new()
                .named("n", 10i64)
                .named("splits", 4i64),
        )
        .await?,
    )?;

    let batches = ctx.sql("SELECT * FROM remote").await?.collect().await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 0, "a zero-split plan produced {rows} rows");
    Ok(())
}

/// A split yielding NO ROWS must not truncate the scan.
///
/// Distinct from zero splits and far likelier in practice — a filter pruned one —
/// and it is the shape that silently drops every later split if an empty split is
/// mistaken for end-of-stream.
#[tokio::test(flavor = "multi_thread")]
async fn a_zero_row_split_does_not_end_the_scan() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    ctx.register_table(
        "remote",
        VgiTableProvider::bind_with_arguments(
            conn,
            "example",
            "main",
            "split_empty_ranges",
            vgi_client::Arguments::new()
                .named("n", 120i64)
                .named("splits", 6i64),
        )
        .await?,
    )?;

    let batches = ctx
        .sql("SELECT count(*) AS c FROM remote")
        .await?
        .collect()
        .await?;
    let c = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is int64")
        .value(0);
    assert_eq!(c, 120, "an empty split truncated the scan at {c} rows");
    Ok(())
}

/// Splits fan the scan across partitions, which is what they exist to enable
/// here — before them this provider reported N partitions but only partition 0
/// ever read.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_scan_reports_more_than_one_partition() -> datafusion::error::Result<()> {
    use datafusion::physical_plan::ExecutionPlan;

    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    let provider = VgiTableProvider::bind_with_arguments(
        conn,
        "example",
        "main",
        "split_sequence",
        vgi_client::Arguments::new()
            .named("n", 500i64)
            .named("splits", 12i64),
    )
    .await?;

    let plan = provider.scan(&ctx.state(), None, &[], None).await?;
    let partitions = plan.properties().output_partitioning().partition_count();
    assert!(
        partitions > 1,
        "a 12-split scan planned {partitions} partition(s); splits are supposed to fan out"
    );
    // ...but never more partitions than splits: an empty partition is legal, a
    // plan that invents them is just waste.
    assert!(
        partitions <= 12,
        "planned {partitions} partitions for 12 splits"
    );
    Ok(())
}

/// A worker that never opted into planning keeps its exact pre-splits behaviour.
///
/// The framework default is a single empty-payload split, which means "the whole
/// scan is one unit of work" — treating that as a real plan would quietly change
/// the execution shape of every existing worker.
#[tokio::test(flavor = "multi_thread")]
async fn a_non_split_worker_is_unaffected() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    ctx.register_table(
        "remote",
        VgiTableProvider::bind(conn, "example", "main", "ten_thousand").await?,
    )?;

    let batches = ctx
        .sql("SELECT count(*) AS c FROM remote")
        .await?
        .collect()
        .await?;
    let c = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is int64")
        .value(0);
    assert_eq!(c, 10_000);
    Ok(())
}
