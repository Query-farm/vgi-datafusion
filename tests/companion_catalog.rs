// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end coverage for VGI catalog-table branches backed by an
//! automatically attached VGI companion catalog.

use std::path::PathBuf;

use datafusion::arrow::array::{Array, Int64Array, StringArray};
use datafusion::prelude::SessionContext;

const ROOT_CATALOG: &str = "datafusion_companion";
const SOURCE_CATALOG: &str = "datafusion_source";
const SOURCE_ALIAS: &str = "datafusion_source_alias";

fn example_worker() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("vgi-rust")
        .join("target");
    for profile in ["debug", "release"] {
        let executable = root.join(profile).join(if cfg!(windows) {
            "vgi-example-worker.exe"
        } else {
            "vgi-example-worker"
        });
        if executable.exists() {
            return Some(executable);
        }
    }
    None
}

/// Quote one argv token for VGI's POSIX command-line parser.
fn shell_argument(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Quote a value as a SQL single-quoted literal.
fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn fixture_location() -> Option<String> {
    if cfg!(windows) {
        return None;
    }
    let worker = example_worker()?;
    Some(format!(
        "/usr/bin/env VGI_WORKER_CATALOG_NAME={ROOT_CATALOG} {}",
        shell_argument(worker.to_string_lossy().as_ref())
    ))
}

macro_rules! skip_without_fixture {
    () => {
        match fixture_location() {
            Some(location) => location,
            None => {
                eprintln!("skipping: companion fixture worker not built or unsupported platform");
                return Ok(());
            }
        }
    };
}

async fn attach_root(
    context: &SessionContext,
    location: &str,
    companions: bool,
) -> datafusion::common::Result<()> {
    let location = sql_string(location);
    vgi_datafusion::sql(
        context,
        &format!(
            "ATTACH '{ROOT_CATALOG}' AS fed (TYPE vgi, LOCATION {location}, attach_companions {companions})"
        ),
    )
    .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn companion_source_arms_scan_filter_project_and_limit() -> datafusion::common::Result<()> {
    let location = skip_without_fixture!();
    let context = SessionContext::new();
    attach_root(&context, &location, true).await?;

    assert!(
        context.catalog(SOURCE_ALIAS).is_some(),
        "the required companion should be attached under its declared alias"
    );

    let batches = vgi_datafusion::sql(
        &context,
        "SELECT id, label, weight FROM fed.main.hot_cold ORDER BY id",
    )
    .await?
    .collect()
    .await?;
    let ids = batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64")
                .clone();
            (0..values.len()).map(move |index| values.value(index))
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, [1, 2, 3, 100, 200]);

    // Projection and limit are implemented above the reconciled union, while
    // the WHERE predicate is also offered to each companion provider as a safe
    // pruning hint and rechecked by DataFusion because the outer scan is Inexact.
    let batches = vgi_datafusion::sql(
        &context,
        "SELECT label FROM fed.main.hot_cold WHERE id >= 2 ORDER BY id LIMIT 2",
    )
    .await?
    .collect()
    .await?;
    let labels = batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("label is Utf8")
                .clone();
            (0..values.len()).map(move |index| values.value(index).to_string())
        })
        .collect::<Vec<_>>();
    assert_eq!(labels, ["cold-b", "cold-c"]);

    let batches = vgi_datafusion::sql(&context, "SELECT weight FROM fed.main.hot_cold LIMIT 2")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        2,
        "the catalog-table union must honor a pushed scan limit"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn companion_opt_out_and_ambiguous_worker_catalog_fail_clearly(
) -> datafusion::common::Result<()> {
    let location = skip_without_fixture!();
    let context = SessionContext::new();
    attach_root(&context, &location, false).await?;

    let error = vgi_datafusion::sql(&context, "SELECT * FROM fed.main.hot_cold")
        .await?
        .collect()
        .await
        .expect_err("the source catalog was deliberately not attached");
    assert!(
        error.to_string().contains("is not attached"),
        "unexpected missing-companion error: {error}"
    );

    for alias in ["source_a", "source_b"] {
        let location = sql_string(&location);
        vgi_datafusion::sql(
            &context,
            &format!("ATTACH '{SOURCE_CATALOG}' AS {alias} (TYPE vgi, LOCATION {location})"),
        )
        .await?;
    }
    let error = vgi_datafusion::sql(&context, "SELECT * FROM fed.main.hot_cold")
        .await?
        .collect()
        .await
        .expect_err("a worker catalog mounted twice has no safe implicit alias");
    let message = error.to_string();
    assert!(
        message.contains("maps to multiple attached aliases"),
        "unexpected ambiguity error: {message}"
    );
    assert!(message.contains("source_a") && message.contains("source_b"));
    Ok(())
}

#[test]
fn fixture_location_quoting_handles_spaces_and_quotes() {
    assert_eq!(shell_argument("/tmp/a b'c"), "'/tmp/a b'\\''c'");
    assert_eq!(sql_string("worker 'quoted'"), "'worker ''quoted'''");
}

#[tokio::test(flavor = "multi_thread")]
async fn companion_source_arm_cycle_reports_the_path() -> datafusion::common::Result<()> {
    let location = skip_without_fixture!();
    let context = SessionContext::new();
    attach_root(&context, &location, true).await?;

    let error = vgi_datafusion::sql(&context, "SELECT * FROM fed.main.cycle_entry")
        .await?
        .collect()
        .await
        .expect_err("the fixture contains an intentional indirect cycle");
    let message = error.to_string();
    assert!(
        message.contains("catalog-table scan branch cycle detected"),
        "unexpected cycle error: {message}"
    );
    for identity in [
        "fed.main.cycle_entry",
        "datafusion_source_alias.main.cycle_back",
    ] {
        assert!(
            message.contains(identity),
            "missing {identity} from {message}"
        );
    }
    Ok(())
}
