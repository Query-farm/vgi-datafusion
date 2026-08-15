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

use datafusion::arrow::array::Array;
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

    // The qualified spelling — what the DuckDB corpus writes.
    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.main.sequence(10)")
        .await?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum::<usize>();
    assert_eq!(rows, 10, "a qualified call reaches the right function");

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

/// Catalog tables — the other half of a schema.
///
/// A VGI catalog table is not a function call the user writes; the worker
/// nominates the scan function and supplies its arguments. The fixture worker
/// keeps 59 of these in schema `data`, and until they were exposed most of the
/// corpus had nothing to query.
#[tokio::test(flavor = "multi_thread")]
async fn catalog_tables_are_queryable() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={w}' AS ex")).await?;

    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.data.ten_thousand_table LIMIT 5")
        .await?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum::<usize>();
    assert_eq!(rows, 5, "a catalog table scans through its scan function");

    // count(*) over the same table exercises the narrowest-column path against
    // a worker-nominated scan function rather than a user-written call.
    let batches = vgi_datafusion::sql(&ctx, "SELECT count(*) AS n FROM ex.data.ten_thousand_table")
        .await?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

/// Qualified calls must reach the function they name, and qualified *table*
/// references must be left alone.
///
/// These are one test because they are the same guard seen from both sides: the
/// rewrite keys on whether the relation has arguments, and getting that wrong
/// breaks one or the other.
#[tokio::test(flavor = "multi_thread")]
async fn qualified_names_resolve_without_disturbing_tables() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={w}' AS ex")).await?;

    // A qualified table reference has no arguments and must not be collapsed —
    // it resolves through the catalog, not the function registry.
    let rows = vgi_datafusion::sql(&ctx, "SELECT * FROM ex.data.ten_thousand_table LIMIT 2")
        .await?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum::<usize>();
    assert_eq!(rows, 2, "a qualified table reference still works");

    // Nested in a subquery, so the walk is proven to recurse rather than only
    // inspecting the top-level FROM.
    let rows = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) AS n FROM (SELECT * FROM ex.main.sequence(4)) t",
    )
    .await?
    .collect()
    .await?
    .iter()
    .map(|b| b.num_rows())
    .sum::<usize>();
    assert_eq!(rows, 1);

    // The schema is part of the key, so a name published in two schemas stays
    // distinct — the case a bare prefix cannot express.
    for schema in ["main", "data"] {
        let q = format!("SELECT * FROM ex.{schema}.test_same_name_cached()");
        vgi_datafusion::sql(&ctx, &q)
            .await
            .unwrap_or_else(|e| panic!("{q} failed: {e}"))
            .collect()
            .await?;
    }
    Ok(())
}

/// Scalar functions, qualified and short.
///
/// The qualified spelling needs no rewrite — a scalar call already flattens its
/// whole path into the lookup key — so this is the surface the corpus uses most.
#[tokio::test(flavor = "multi_thread")]
async fn scalar_functions_resolve() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={w}' AS ex")).await?;

    for q in [
        "SELECT ex.main.double(21) AS v",
        "SELECT ex_double(21) AS v",
    ] {
        let batches = vgi_datafusion::sql(&ctx, q)
            .await
            .unwrap_or_else(|e| panic!("{q} failed to plan: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{q} failed to run: {e}"));
        let n: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(n, 1, "{q}");
        let v = batches[0].column(0);
        let v = v
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap_or_else(|| panic!("{q}: expected Int64, got {:?}", v.data_type()));
        assert_eq!(v.value(0), 42, "{q}");
    }
    Ok(())
}

/// Const parameters: a bind-time constant, not a column.
///
/// `cached_add_const(value, addend)` declares `addend` as a `ConstParam`, so its
/// value belongs in the bind's arguments — and the bind carries *only* the
/// constants, compacted. Sending a placeholder for the columnar parameter too
/// shifts the constant to the wrong slot, and the worker answers with a column
/// of NULLs rather than an error, which is how this stayed invisible.
#[tokio::test(flavor = "multi_thread")]
async fn const_parameters_reach_the_worker() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example' AS ex (TYPE vgi, LOCATION '{w}')"),
    )
    .await?;

    // Values taken from the DuckDB corpus (scalar/per_value_edge.test), so this
    // asserts agreement with the reference engine, not just non-nullness.
    for (q, want) in [
        (
            "SELECT sum(ex.cached_add_const(x % 3, 10)) AS v FROM range(6) t(x)",
            66i64,
        ),
        (
            "SELECT sum(ex.cached_add_const(x % 3, 20)) AS v FROM range(6) t(x)",
            126,
        ),
        (
            "SELECT sum(ex.cached_add_const(x % 3, 10)) AS v FROM range(9) t(x)",
            99,
        ),
    ] {
        let batches = vgi_datafusion::sql(&ctx, q).await?.collect().await?;
        let col = batches[0].column(0);
        let got = col
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap_or_else(|| panic!("{q}: expected Int64, got {:?}", col.data_type()));
        assert!(
            !got.is_null(0),
            "{q} returned NULL — the const never arrived"
        );
        assert_eq!(got.value(0), want, "{q}");
    }
    Ok(())
}

/// A narrowed projection must return the columns it asked for.
///
/// Projection pushdown is advisory — a worker may honour it or return every
/// column — so the scan cannot take the first N columns positionally. Doing so
/// produced a result with the right shape, the right column names and the
/// wrong data, which nothing downstream can detect. Caught by
/// `cache/coverage.test`.
#[tokio::test(flavor = "multi_thread")]
async fn a_narrowed_projection_returns_the_right_columns() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example' AS ex (TYPE vgi, LOCATION '{w}')"),
    )
    .await?;

    // The fixture's columns are deliberately distinguishable: a=i, b=i*10,
    // c=i*100. Reading the wrong column is therefore visible in the values,
    // which a same-typed table would hide.
    for (q, want) in [
        (
            "SELECT b FROM ex.data.cache_multicol ORDER BY b",
            vec![0i64, 10, 20, 30],
        ),
        (
            "SELECT c FROM ex.data.cache_multicol ORDER BY c",
            vec![0, 100, 200, 300],
        ),
    ] {
        let batches = vgi_datafusion::sql(&ctx, q).await?.collect().await?;
        let got: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Int64Array>()
                    .expect("Int64")
                    .clone();
                (0..c.len()).map(move |i| c.value(i))
            })
            .collect();
        assert_eq!(got, want, "{q}");
    }
    Ok(())
}

/// A catalog table's scan function may live in another schema.
///
/// A worker registers function names per schema and may reuse a name across
/// them, so a table in `data` can be scanned by a function in `main`. Binding
/// in the table's schema alone fails outright — the reference worker answers
/// "Function 'products_scan' is not registered in schema 'data'. It is
/// available in: ['main']" — so the table's schema is tried first and the
/// catalog's default schema second, which is how the extension resolves it too.
#[tokio::test(flavor = "multi_thread")]
async fn a_scan_function_outside_the_tables_schema_still_binds() -> datafusion::error::Result<()> {
    let Some(w) = worker() else { return Ok(()) };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example' AS ex (TYPE vgi, LOCATION '{w}')"),
    )
    .await?;

    for q in [
        "SELECT count(*) FROM ex.data.products",
        "SELECT count(*) FROM ex.data.departments",
    ] {
        let batches = vgi_datafusion::sql(&ctx, q)
            .await
            .unwrap_or_else(|e| panic!("{q} failed to plan: {e}"))
            .collect()
            .await?;
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            1,
            "{q}"
        );
    }
    Ok(())
}
