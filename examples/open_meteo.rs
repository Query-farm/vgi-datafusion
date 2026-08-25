// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end Open-Meteo smoke harness with a hard timeout per SQL interaction.
//!
//! ```shell
//! cargo run --example open_meteo
//! ```
//!
//! Override the deployed worker or timeout with `VGI_OPEN_METEO_LOCATION` and
//! `VGI_QUERY_TIMEOUT_SECS`.

use std::error::Error;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, Float64Array, RecordBatch};
use datafusion::arrow::util::pretty::print_batches;
use datafusion::prelude::SessionContext;

type HarnessResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[tokio::main]
async fn main() -> HarnessResult<()> {
    let location = std::env::var("VGI_OPEN_METEO_LOCATION")
        .unwrap_or_else(|_| "https://vgi-open-meteo.rusty-bb6.workers.dev".to_string());
    let timeout = std::env::var("VGI_QUERY_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30));
    let escaped_location = location.replace('\'', "''");
    let ctx = SessionContext::new();

    run(
        &ctx,
        timeout,
        "attach",
        &format!("ATTACH 'open_meteo' AS m (TYPE vgi, LOCATION '{escaped_location}');"),
    )
    .await?;

    let geocoded = run(
        &ctx,
        timeout,
        "geocoding",
        "SELECT *
         FROM m.main.geocoding(
           'Glen Allen, VA',
           count := 1,
           country_code := 'US'
         );",
    )
    .await?;
    let (latitude, longitude) = coordinates(&geocoded)?;

    run(
        &ctx,
        timeout,
        "catalog SQL macros",
        "SELECT m.main.weather_code_emoji(0) AS icon,
                m.main.weather_code_text(0) AS conditions;",
    )
    .await?;

    run(
        &ctx,
        timeout,
        "hourly forecast",
        &format!(
            "SELECT w.time,
                    round(w.temperature_2m, 1) AS temp_f,
                    m.main.weather_code_emoji(w.weather_code) AS icon,
                    m.main.weather_code_text(w.weather_code) AS conditions
             FROM m.main.forecast_hourly(
                    {latitude}, {longitude},
                    forecast_days := 2,
                    temperature_unit := 'fahrenheit'
             ) AS w
             WHERE w.time >= now()
             ORDER BY w.time
             LIMIT 6;"
        ),
    )
    .await?;

    run(&ctx, timeout, "detach", "DETACH m;").await?;
    println!("\nOpen-Meteo harness passed.");
    Ok(())
}

async fn run(
    ctx: &SessionContext,
    timeout: Duration,
    label: &str,
    query: &str,
) -> HarnessResult<Vec<RecordBatch>> {
    println!("\n[{label}]\n{query}");
    let started = Instant::now();
    let work = async {
        let frame = vgi_datafusion::sql(ctx, query).await?;
        frame.collect().await
    };
    let batches = match tokio::time::timeout(timeout, work).await {
        Ok(result) => result?,
        Err(_) => {
            eprintln!(
                "\nTIMEOUT: `{label}` did not finish within {:.1}s",
                timeout.as_secs_f64()
            );
            // VGI's HTTP client is blocking underneath. Exiting here is
            // deliberate: dropping the Tokio runtime would otherwise wait for
            // the blocked worker thread and turn a diagnostic timeout into a
            // second hang.
            std::process::exit(124);
        }
    };
    println!("completed in {:.2?}", started.elapsed());
    if batches
        .iter()
        .any(|batch| batch.num_rows() > 0 && batch.num_columns() > 0)
    {
        print_batches(&batches)?;
    }
    Ok(batches)
}

fn coordinates(batches: &[RecordBatch]) -> HarnessResult<(f64, f64)> {
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let latitude = batch
            .column_by_name("latitude")
            .and_then(|array| array.as_any().downcast_ref::<Float64Array>())
            .ok_or("geocoding returned no Float64 `latitude` column")?;
        let longitude = batch
            .column_by_name("longitude")
            .and_then(|array| array.as_any().downcast_ref::<Float64Array>())
            .ok_or("geocoding returned no Float64 `longitude` column")?;
        if latitude.is_valid(0) && longitude.is_valid(0) {
            return Ok((latitude.value(0), longitude.value(0)));
        }
    }
    Err("geocoding returned no coordinate row".into())
}
