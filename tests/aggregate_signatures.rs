// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! VGI aggregate signature shapes that need adapter handling beyond an
//! ordinary fixed, typed DataFusion UDAF.

use std::path::PathBuf;

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::logical_expr::{create_udf, Volatility};
use datafusion::prelude::{SessionConfig, SessionContext};

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

macro_rules! skip_without_worker {
    () => {
        match example_worker() {
            Some(path) => path,
            None => {
                eprintln!("skipping: vgi-example-worker not built");
                return Ok(());
            }
        }
    };
}

async fn attached(worker: &std::path::Path) -> datafusion::common::Result<SessionContext> {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;
    Ok(ctx)
}

#[tokio::test(flavor = "multi_thread")]
async fn nullary_aggregate_preserves_row_counts() -> datafusion::common::Result<()> {
    let worker = skip_without_worker!();
    let ctx = attached(&worker).await?;

    for (sql, expected) in [
        ("SELECT ex.main.vgi_count() FROM range(100)", 100),
        ("SELECT ex.main.vgi_count() FROM range(0)", 0),
        ("SELECT ex.main.vgi_count()", 1),
    ] {
        let batches = vgi_datafusion::sql(&ctx, sql).await?.collect().await?;
        let value = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("vgi_count returns Int64")
            .value(0);
        assert_eq!(value, expected, "{sql}");
    }

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT x % 3 AS g, ex.main.vgi_count() AS n \
         FROM range(9) t(x) GROUP BY g ORDER BY g",
    )
    .await?
    .collect()
    .await?;
    let counts = batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("vgi_count returns Int64")
                .clone();
            (0..values.len()).map(move |index| values.value(index))
        })
        .collect::<Vec<_>>();
    assert_eq!(counts, vec![3, 3, 3]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn nullary_aggregate_supports_sliding_frames() -> datafusion::common::Result<()> {
    let worker = skip_without_worker!();
    let ctx = attached(&worker).await?;
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.vgi_count() OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
         FROM range(4) t(x) ORDER BY x",
    )
    .await?
    .collect()
    .await?;
    let counts = batches
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("vgi_count returns Int64")
                .clone();
            (0..values.len()).map(move |index| values.value(index))
        })
        .collect::<Vec<_>>();
    assert_eq!(counts, vec![1, 2, 2, 2]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn any_and_variadic_aggregate_signatures_reach_vgi_bind() -> datafusion::common::Result<()> {
    let worker = skip_without_worker!();
    let ctx = attached(&worker).await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.vgi_generic_sum(x::INTEGER) FROM range(4) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("ANY output follows INTEGER input")
            .value(0),
        6
    );
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.vgi_generic_sum(x::DOUBLE) FROM range(4) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("ANY output follows DOUBLE input")
            .value(0),
        6.0
    );
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT ex.main.vgi_sum_all(x::DOUBLE, (x * 2)::DOUBLE) FROM range(4) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("variadic aggregate returns Float64")
            .value(0),
        18.0
    );

    let error = vgi_datafusion::sql(&ctx, "SELECT ex.main.vgi_sum_all() FROM range(5)")
        .await
        .expect_err("the worker rejects an empty variadic tail");
    assert!(
        error.to_string().contains("requires at least 1 value"),
        "worker validation was not preserved: {error}"
    );

    let error = vgi_datafusion::sql(&ctx, "SELECT ex.main.vgi_generic_sum(1, 2)")
        .await
        .expect_err("a fixed ANY signature has one argument");
    assert!(
        error.to_string().contains("expects 1")
            || error.to_string().contains("1 argument")
            || error
                .to_string()
                .contains("does not support zero arguments"),
        "unexpected fixed-arity diagnostic: {error}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scalar_collision_is_not_rewritten_as_a_nullary_aggregate() -> datafusion::common::Result<()>
{
    let worker = skip_without_worker!();
    let ctx = attached(&worker).await?;

    // DataFusion deliberately gives a scalar UDF precedence over a UDAF with
    // the same ordinary-call name. If the nullary compatibility pass injected
    // its row witness here, this one-argument scalar would incorrectly succeed
    // and return 1 instead of preserving DataFusion's namespace rule.
    ctx.register_udf(create_udf(
        "ex.main.vgi_count",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(|arguments| Ok(arguments[0].clone())),
    ));
    for sql in [
        "SELECT ex.main.vgi_count()",
        "SELECT ex.main.vgi_count() OVER ()",
    ] {
        let error = vgi_datafusion::sql(&ctx, sql)
            .await
            .expect_err("the zero-argument call resolves to the colliding scalar");
        assert!(
            error.to_string().contains("expects 1")
                || error.to_string().contains("1 argument")
                || error
                    .to_string()
                    .contains("does not support zero arguments"),
            "unexpected scalar arity diagnostic for {sql}: {error}"
        );
    }

    Ok(())
}
