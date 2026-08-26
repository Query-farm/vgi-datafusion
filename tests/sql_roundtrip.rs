// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! SQL over a real VGI worker.
//!
//! The point of the whole adapter: DataFusion plans and executes ordinary SQL,
//! and the rows come from a worker process over Arrow IPC.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::Array;
use datafusion::catalog::TableProvider;
use datafusion::common::stats::Precision;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::{StatisticsArgs, StatisticsContext};
use datafusion::prelude::{SessionConfig, SessionContext};
use vgi_datafusion::{VgiConnection, VgiTableProvider};
use vgi_datafusion::{VgiResolvedSecret, VgiRuntime, VgiSecretResolver, VgiSessionOptions};

#[derive(Debug)]
struct ExampleSecretResolver;

#[async_trait::async_trait]
impl VgiSecretResolver for ExampleSecretResolver {
    async fn resolve(
        &self,
        secret_type: &str,
        _scope: Option<&str>,
        _name: Option<&str>,
    ) -> datafusion::common::Result<Option<VgiResolvedSecret>> {
        if secret_type != "vgi_example" {
            return Ok(None);
        }
        Ok(Some(VgiResolvedSecret {
            name: "datafusion_test_secret".to_string(),
            fields: std::collections::BTreeMap::from([
                (
                    "type".to_string(),
                    ScalarValue::Utf8(Some(secret_type.to_string())),
                ),
                (
                    "secret_string".to_string(),
                    ScalarValue::Utf8(Some("from-datafusion".to_string())),
                ),
                ("port".to_string(), ScalarValue::Int64(Some(5432))),
                ("use_ssl".to_string(), ScalarValue::Boolean(Some(true))),
            ]),
        }))
    }
}

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
async fn worker_function_metadata_is_queryable_and_detaches() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    let location = worker.to_string_lossy();
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={location}' AS ex")).await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT function_name, parameter_types, return_type, description \
         FROM duckdb_functions() \
         WHERE database_name = 'ex' AND schema_name = 'main' \
         AND function_name = 'double'",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM vgi_function_arguments() \
         WHERE catalog_name = 'ex' AND function_name = 'multiply' \
         AND arg_name = 'factor' AND is_const AND arg_description IS NOT NULL",
    )
    .await?
    .collect()
    .await?;
    let documented_const = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(documented_const, 1);

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT column_name, column_type, min, max, distinct_count \
         FROM vgi_table_statistics('ex', 'data', 'departments') \
         ORDER BY column_name",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        3
    );
    let rendered = datafusion::arrow::util::pretty::pretty_format_batches(&batches)?.to_string();
    for expected in [
        "budget", "DOUBLE", "50000.0", "500000.0", "id", "BIGINT", "Accounti", "Sales", "VARCHAR",
    ] {
        assert!(
            rendered.contains(expected),
            "missing {expected} from {rendered}"
        );
    }

    let explain = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM ex.data.departments WHERE id > 100",
    )
    .await?
    .collect()
    .await?;
    let explain = datafusion::arrow::util::pretty::pretty_format_batches(&explain)?.to_string();
    assert!(explain.contains("EmptyExec"), "unexpected plan: {explain}");

    let explain = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM ex.data.departments WHERE id >= 1",
    )
    .await?
    .collect()
    .await?;
    let explain = datafusion::arrow::util::pretty::pretty_format_batches(&explain)?.to_string();
    assert!(
        explain.contains("VgiScanExec"),
        "unexpected plan: {explain}"
    );

    let explain = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM ex.main.sequence(10) WHERE n = 10",
    )
    .await?
    .collect()
    .await?;
    let explain = datafusion::arrow::util::pretty::pretty_format_batches(&explain)?.to_string();
    assert!(explain.contains("EmptyExec"), "unexpected plan: {explain}");

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*), min(value), max(value) FROM ex.data.numbers",
    )
    .await?
    .collect()
    .await?;
    let numbers = datafusion::arrow::util::pretty::pretty_format_batches(&batches)?.to_string();
    for expected in ["100", "0", "99"] {
        assert!(
            numbers.contains(expected),
            "missing {expected} from {numbers}"
        );
    }

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM vgi_table_statistics('ex', 'data', 'versioned_data')",
    )
    .await?
    .collect()
    .await?;
    let no_statistics = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(no_statistics, 0);

    vgi_datafusion::sql(&ctx, "DETACH ex").await?;
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM duckdb_functions() WHERE database_name = 'ex'",
    )
    .await?
    .collect()
    .await?;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(count, 0);
    Ok(())
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
async fn worker_opted_results_are_cached_per_session() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let runtime = std::sync::Arc::new(VgiRuntime::new(VgiSessionOptions::default()));
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()])
        .with_runtime(runtime.clone());
    let ctx = SessionContext::new();
    ctx.register_table(
        "remote_cache_nonce",
        VgiTableProvider::bind(conn, "example", "main", "cache_nonce").await?,
    )?;

    let read = || async {
        let batches = ctx
            .sql("SELECT nonce FROM remote_cache_nonce")
            .await?
            .collect()
            .await?;
        let values = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("nonce is Int64");
        Ok::<i64, datafusion::error::DataFusionError>(values.value(0))
    };
    let first = read().await?;
    let second = read().await?;
    assert_eq!(first, second, "a cache hit must replay the stored nonce");
    let stats = runtime.result_cache().stats();
    assert_eq!(stats.inserts, 1, "events: {:?}", runtime.events());
    assert_eq!(stats.hits, 1, "events: {:?}", runtime.events());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_results_commit_only_after_every_partition() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let runtime = Arc::new(VgiRuntime::new(VgiSessionOptions::default()));
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_extension(runtime.clone()));
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    for _ in 0..2 {
        let batches = vgi_datafusion::sql(
            &ctx,
            "SELECT n FROM ex.main.split_cacheable(n := 100, splits := 4)",
        )
        .await?
        .collect()
        .await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            100
        );
    }
    let stats = runtime.result_cache().stats();
    assert_eq!(stats.inserts, 1, "events: {:?}", runtime.events());
    assert_eq!(stats.hits, 1, "events: {:?}", runtime.events());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_plans_are_reused_only_when_the_worker_allows_it() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let runtime = Arc::new(VgiRuntime::new(VgiSessionOptions::default()));
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_extension(runtime.clone()));
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    for _ in 0..2 {
        vgi_datafusion::sql(
            &ctx,
            "SELECT count(*) FROM ex.main.split_partitioned(rows_per_country := 1)",
        )
        .await?
        .collect()
        .await?;
    }
    let stats = runtime.plan_cache_stats();
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.hits, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn advertised_aggregates_support_sliding_window_frames() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.vgi_window_sum(x) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s \
         FROM (VALUES (1::BIGINT), (2::BIGINT), (3::BIGINT)) t(x) ORDER BY x",
    )
    .await?
    .collect()
    .await?;
    let got = batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("window sum is Int64")
                .clone();
            (0..values.len()).map(move |index| values.value(index))
        })
        .collect::<Vec<_>>();
    assert_eq!(got, vec![1, 3, 5]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn aggregate_const_parameters_reach_bind_and_finalize() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.vgi_percentile(x::DOUBLE, 0.0) AS lo, \
                ex.main.vgi_percentile(x::DOUBLE, 0.9) AS hi \
         FROM range(10) t(x)",
    )
    .await?
    .collect()
    .await?;
    let lo = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .expect("percentile returns Float64")
        .value(0);
    let hi = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .expect("percentile returns Float64")
        .value(0);
    assert_eq!((lo, hi), (0.0, 9.0));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_declared_views_use_datafusion_view_tables() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT (SELECT count(*) FROM ex.main.first_ten) AS first_ten, \
                (SELECT count(*) FROM ex.main.even_numbers) AS evens",
    )
    .await?
    .collect()
    .await?;
    let first_ten = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    let evens = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!((first_ten, evens), (10, 50));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_nominated_global_functions_are_published() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let scalar = vgi_datafusion::sql(&ctx, "SELECT vgi_example_global_scalar(7)")
        .await?
        .collect()
        .await?;
    let value = scalar[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringArray>()
        .expect("global scalar returns Utf8")
        .value(0);
    assert_eq!(value, "global_scalar:7");

    let table = vgi_datafusion::sql(&ctx, "SELECT count(*) FROM vgi_example_global_table()")
        .await?
        .collect()
        .await?;
    assert_eq!(
        table[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("count is Int64")
            .value(0),
        3
    );

    let aggregate = vgi_datafusion::sql(
        &ctx,
        "SELECT vgi_example_global_agg(x) FROM (VALUES (2::BIGINT), (3::BIGINT)) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        aggregate[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("global aggregate returns Int64")
            .value(0),
        5
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_secret_lookups_use_the_session_resolver() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let runtime = Arc::new(
        VgiRuntime::new(VgiSessionOptions::default())
            .with_secret_resolver(Arc::new(ExampleSecretResolver)),
    );
    let config = SessionConfig::new().with_extension(runtime);
    let ctx = SessionContext::new_with_config(config);
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT value FROM ex.main.secret_demo() WHERE key = 'secret_string'",
    )
    .await?
    .collect()
    .await?;
    let value = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringArray>()
        .expect("secret value is Utf8")
        .value(0);
    assert_eq!(value, "from-datafusion");

    let aggregate = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.secret_typed_sum(x) FROM (VALUES (2::BIGINT), (3::BIGINT)) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        aggregate[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .expect("secret selected Float64 output")
            .value(0),
        5.0
    );
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
async fn blended_literal_table_function_runs_as_a_one_row_exchange() -> datafusion::error::Result<()>
{
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT geohash FROM ex.main.geo_encode(52.52, 13.41, precision := 2)",
    )
    .await?
    .collect()
    .await?;
    let values = batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StringArray>()
                .expect("geohash is Utf8")
                .clone();
            (0..values.len()).map(move |index| values.value(index).to_string())
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec!["52.52:13.41"]);

    let batches = vgi_datafusion::sql(&ctx, "SELECT ex.main.vgi_multiply(6, 7) AS answer")
        .await?
        .collect()
        .await?;
    let answer = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("macro result is Int64")
        .value(0);
    assert_eq!(answer, 42, "catalog scalar macros expand locally");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_multi_branch_scans_union_and_enforce_branch_filters(
) -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(&ctx, "SELECT count(*) FROM ex.data.multi_branch_numbers")
        .await?
        .collect()
        .await?;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(count, 100, "both sequence(50) branches must contribute");

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*), min(n), max(n) \
         FROM ex.data.multi_branch_filtered_numbers WHERE n > 75",
    )
    .await?
    .collect()
    .await?;
    let values = batches[0]
        .columns()
        .iter()
        .map(|column| {
            column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("aggregate is Int64")
                .value(0)
        })
        .collect::<Vec<_>>();
    assert_eq!(values, [24, 76, 99]);
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

/// A function that does not advertise projection pushdown emits its full
/// schema. The adapter must leave projection IDs off the wire and narrow the
/// result locally by name; otherwise the strict client rejects the batch (or,
/// worse, `SELECT b` can return column `a`).
#[tokio::test(flavor = "multi_thread")]
async fn projection_stays_local_when_the_worker_does_not_opt_in() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    let table = VgiTableProvider::bind(conn, "example", "main", "cache_multicol").await?;
    ctx.register_table("remote", table)?;

    let batches = ctx
        .sql("SELECT b FROM remote ORDER BY b")
        .await?
        .collect()
        .await?;
    let values = batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("b is int64")
                .clone();
            (0..column.len()).map(move |i| column.value(i))
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec![0, 10, 20, 30]);
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

/// A filter on a column the projection does NOT include still filters on that
/// column.
///
/// This is the case that fails silently rather than loudly. The worker keys a
/// pushed filter by the column's position in what it EMITS, so if the scan asks
/// only for `n` while the filter names `s`, the predicate is evaluated against
/// whichever column occupies that slot — `n` — and the query returns a
/// confidently wrong answer with no error anywhere. The fix is that the scan
/// requests the UNION of projected and filter-referenced columns and trims back
/// afterwards, so the reported schema is unchanged.
///
/// `s` is 'row_<n>', so this matches exactly one row.
///
/// NOT a discriminating test of the fix, and it is worth saying so rather than
/// leaving someone to assume otherwise: with the union removed this still
/// passes. Two things mask it here — the example worker resolves a pushed
/// filter by column NAME before falling back to the index
/// (`vgi/src/pushdown.rs`), and this provider reports every filter `Inexact`,
/// so DataFusion re-applies it above the scan. The mis-keying only bites a
/// worker that resolves by index. What this test does hold is the answer, which
/// is worth a regression either way; the discriminating coverage is
/// `filters::tests::reports_the_columns_its_specs_read` and the stub-transport
/// tests in vgi-client that assert the requested column order on the wire.
#[tokio::test(flavor = "multi_thread")]
async fn a_filter_on_an_unprojected_column_still_filters_that_column(
) -> datafusion::error::Result<()> {
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
        .sql("SELECT n FROM echo WHERE s = 'row_7'")
        .await?
        .collect()
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1, "exactly the row whose s is 'row_7'");

    let col = batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .map(|b| b.column(0).clone())
        .expect("one row");
    let ns = col
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("n is int64");
    assert_eq!(
        ns.value(0),
        7,
        "and it is row 7, not whatever landed in slot 0"
    );
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

/// The canonical VGI split corpus binds `n` and `splits` by name. DataFusion's
/// table planner rejects named FunctionArgs before the UDTF hook, so the SQL
/// wrapper carries them through a private expression and restores their names.
/// Reversing them here proves they were not merely treated positionally.
#[tokio::test(flavor = "multi_thread")]
async fn split_arguments_bind_by_name_from_sql() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) AS n, count(DISTINCT n) AS d \
         FROM ex.main.split_sequence(splits := 7, n := 123)",
    )
    .await?
    .collect()
    .await?;
    for column in 0..2 {
        let value = batches[0]
            .column(column)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("count is int64")
            .value(0);
        assert_eq!(value, 123);
    }
    Ok(())
}

/// DataFusion wraps an explained statement in its own AST node. The VGI SQL
/// pre-pass must recurse through that wrapper or DataFusion sees the raw named
/// arguments and rejects them before the table function can bind.
#[tokio::test(flavor = "multi_thread")]
async fn explain_rewrites_named_table_function_arguments() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "EXPLAIN SELECT * FROM ex.main.split_sequence(splits := 7, n := 123)",
    )
    .await?
    .collect()
    .await?;
    assert!(
        batches.iter().any(|batch| batch.num_rows() > 0),
        "EXPLAIN should return a plan"
    );
    Ok(())
}

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

/// Pagination and statistics are the two client-facing pieces added after the
/// initial split implementation. A total advertised only on the final page is
/// intentionally not a plan-level fact (first-page-wins), so the adapter must
/// derive the exact cardinality from the complete split enumeration.
#[tokio::test(flavor = "multi_thread")]
async fn paginated_splits_compose_and_publish_exact_statistics() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    let provider = VgiTableProvider::bind_with_arguments(
        conn,
        "example",
        "main",
        "split_paginated",
        vgi_client::Arguments::new()
            .named("n", 137i64)
            .named("splits", 13i64),
    )
    .await?;

    let plan = provider.scan(&ctx.state(), None, &[], None).await?;
    let statistics = StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new())?;
    assert_eq!(statistics.num_rows, Precision::Exact(137));

    let partition_count = plan.properties().output_partitioning().partition_count();
    let mut partition_rows = 0usize;
    for partition in 0..partition_count {
        let statistics = StatisticsContext::new().compute(
            plan.as_ref(),
            &StatisticsArgs::new().with_partition(Some(partition)),
        )?;
        let Precision::Exact(rows) = statistics.num_rows else {
            panic!("partition {partition} did not retain exact split row counts");
        };
        partition_rows += rows;
    }
    assert_eq!(partition_rows, 137);

    ctx.register_table("remote", provider)?;
    let batches = ctx
        .sql("SELECT count(*) AS n, count(DISTINCT n) AS d FROM remote")
        .await?
        .collect()
        .await?;
    let count = |column: usize| {
        batches[0]
            .column(column)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("count is int64")
            .value(0)
    };
    assert_eq!(count(0), 137, "a plan page was dropped or repeated");
    assert_eq!(count(1), 137, "paginated split ranges overlap");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_optimizer_metadata_reaches_datafusion() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    let provider = VgiTableProvider::bind_with_arguments(
        conn,
        "example",
        "main",
        "split_partitioned",
        vgi_client::Arguments::new()
            .named("rows_per_country", 3i64)
            .named("require_plan_context", true),
    )
    .await?;

    let plan = provider.scan(&ctx.state(), None, &[], None).await?;
    let statistics = StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new())?;
    assert_eq!(statistics.num_rows, Precision::Exact(12));
    assert_eq!(statistics.total_byte_size, Precision::Inexact(192));
    assert!(
        plan.properties().output_ordering().is_some(),
        "one VGI split per DataFusion partition preserves within-split ordering"
    );

    let first = StatisticsContext::new().compute(
        plan.as_ref(),
        &StatisticsArgs::new().with_partition(Some(0)),
    )?;
    assert_eq!(first.num_rows, Precision::Exact(3));
    assert!(matches!(
        (&first.column_statistics[0].min_value, &first.column_statistics[0].max_value),
        (
            Precision::Exact(ScalarValue::Utf8(Some(min))),
            Precision::Exact(ScalarValue::Utf8(Some(max)))
        ) if min == max
    ));

    // Execution is part of this test: the fixture refuses a split init unless
    // the plan's execution_id and init_opaque_data are echoed unchanged.
    ctx.register_table("remote", provider)?;
    let batches = ctx.sql("SELECT * FROM remote").await?.collect().await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        12
    );
    Ok(())
}

/// More claims than DataFusion partitions forces each connection to redeem a
/// token group sequentially, the execution shape used after bin-packing.
#[tokio::test(flavor = "multi_thread")]
async fn many_splits_are_redeemed_through_fewer_partitions() -> datafusion::error::Result<()> {
    let worker = skip_without_worker!();
    let conn = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);

    let ctx = SessionContext::new();
    let provider = VgiTableProvider::bind_with_arguments(
        conn,
        "example",
        "main",
        "split_many",
        vgi_client::Arguments::new()
            .named("n", 4096i64)
            .named("splits", 257i64),
    )
    .await?;
    let plan = provider.scan(&ctx.state(), None, &[], None).await?;
    assert!(
        plan.properties().output_partitioning().partition_count() < 257,
        "the test must exercise grouped claims rather than one partition per split"
    );

    ctx.register_table("remote", provider)?;
    let batches = ctx
        .sql("SELECT count(*) AS n, count(DISTINCT n) AS d FROM remote")
        .await?
        .collect()
        .await?;
    for column in 0..2 {
        let value = batches[0]
            .column(column)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("count is int64")
            .value(0);
        assert_eq!(value, 4096);
    }
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
