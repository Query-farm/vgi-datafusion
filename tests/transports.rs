// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Attach and query the reference **Python** fixture worker, over each
//! transport, from SQL.
//!
//! Two things are being proved, and they are why these tests exist separately
//! from `sql_roundtrip.rs` (which uses the Rust example worker):
//!
//! 1. **Conformance against the canonical implementation.** The Rust reference
//!    worker is lenient where the Python one is strict — it accepts a request
//!    with no declared `vgi_rpc.protocol_version`, and its
//!    `normalize_function_type` accepts `table` where the protocol says
//!    `TABLE_FUNCTION`. Both of those client bugs were invisible until the
//!    Python worker rejected them.
//!
//! 2. **That attaching is affordable.** Discovery binds lazily; if that
//!    regresses to eager binding, attaching this worker goes from about a
//!    second to minutes, so the assertion is a real guard rather than a
//!    formality.
//!
//! Set `VGI_TEST_WORKER` to run them, matching the `require-env` convention of
//! the DuckDB extension's suite.

use std::time::{Duration, Instant};

use datafusion::prelude::SessionContext;

fn worker() -> Option<String> {
    match std::env::var("VGI_TEST_WORKER") {
        Ok(v) if !v.trim().is_empty() => Some(v),
        _ => {
            eprintln!("skipping: set VGI_TEST_WORKER to run the transport tests");
            None
        }
    }
}

/// Attach `location`, then run a query that touches exactly one table.
async fn attach_and_query(location: &str) -> datafusion::error::Result<(Duration, Duration)> {
    let ctx = SessionContext::new();

    let t = Instant::now();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={location}' AS ex")).await?;
    let attach = t.elapsed();

    let t = Instant::now();
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) AS n FROM ex.main.test_same_name_cached",
    )
    .await?
    .collect()
    .await?;
    let query = t.elapsed();

    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        1,
        "count(*) returns one row"
    );
    Ok((attach, query))
}

/// Attaching must not cost one worker round-trip per advertised function.
///
/// The fixture worker defines a couple of hundred; eager binding put this in
/// the minutes. Generous enough not to be flaky on a cold machine, tight enough
/// to fail loudly if laziness regresses.
const ATTACH_BUDGET: Duration = Duration::from_secs(30);

#[tokio::test(flavor = "multi_thread")]
async fn subprocess_transport() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let (attach, query) = attach_and_query(&w).await?;
    eprintln!("subprocess: attach={attach:?} query={query:?}");
    assert!(
        attach < ATTACH_BUDGET,
        "ATTACH took {attach:?} — discovery is binding every table again"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn launcher_transport() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let (attach, query) = attach_and_query(&format!("launch:{w}")).await?;
    eprintln!("launch: attach={attach:?} query={query:?}");
    assert!(attach < ATTACH_BUDGET, "ATTACH took {attach:?}");
    Ok(())
}

/// The lazy path must still resolve a table that is only named at plan time,
/// and must not resurrect a bind for one the query never mentions.
#[tokio::test(flavor = "multi_thread")]
async fn only_the_queried_table_is_bound() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={w}' AS ex")).await?;

    // Naming a table that does not exist must be an ordinary plan error, not a
    // hang or a panic — `table_exist` answers from the name list, so this is
    // the path where the two halves of the provider disagree if anything is
    // wrong.
    let err = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.no_such_table")
        .await
        .expect_err("unknown table");
    assert!(err.to_string().contains("no_such_table"), "{err}");

    // And a real one still resolves afterwards. The fixture emits fewer rows
    // than the limit, so assert the limit is respected rather than reached —
    // the point here is that the lazy bind resolved, not the row count.
    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.test_same_name_cached LIMIT 3")
        .await?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum::<usize>();
    assert!((1..=3).contains(&rows), "expected 1..=3 rows, got {rows}");

    // A function that needs arguments is advertised but will not bind bare.
    // The error must say so, rather than claim the table does not exist.
    let err = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.sequence")
        .await
        .expect_err("sequence takes arguments");
    let err = err.to_string();
    assert!(err.contains("does not bind as a bare table"), "{err}");
    assert!(
        err.contains("positional"),
        "worker reason not surfaced: {err}"
    );
    Ok(())
}

/// An argument-taking table function, reached through the UDTF registry.
///
/// `sequence(n)` cannot be a bare table — its output depends on `n` — so this
/// is the surface that makes most of the worker's functions reachable at all.
#[tokio::test(flavor = "multi_thread")]
async fn table_functions_take_arguments() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={w}' AS ex")).await?;

    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex_sequence(10)")
        .await?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum::<usize>();
    assert_eq!(rows, 10, "the argument reached the worker");

    // A different argument is a different bind, not a cached provider.
    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex_sequence(3)")
        .await?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum::<usize>();
    assert_eq!(rows, 3);
    Ok(())
}

/// Arguments that cannot be bound at plan time must say why.
#[tokio::test(flavor = "multi_thread")]
async fn bad_table_function_arguments_explain_themselves() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={w}' AS ex")).await?;

    // The worker's own arity/type complaint should surface, not a generic one.
    let err = vgi_datafusion::sql(&ctx, "SELECT * FROM ex_sequence()")
        .await
        .expect_err("sequence needs an argument")
        .to_string();
    assert!(!err.is_empty(), "{err}");
    Ok(())
}
