// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! DataFusion-native spellings used by sparse scalar corpus overlays.

mod common;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, StringArray, UInt16Array};
use datafusion::prelude::SessionContext;

#[tokio::test(flavor = "multi_thread")]
async fn arrow_unsigned_and_named_struct_spellings_reach_scalar_overloads(
) -> datafusion::common::Result<()> {
    let Some(worker) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let ctx = SessionContext::new();
    let location = worker.to_string_lossy().replace('\'', "''");
    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'example?location={location}' AS example"),
    )
    .await?;

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT example.double(arrow_cast(10, 'UInt8')), \
                example.add_values(2::INTEGER, arrow_cast(3, 'UInt32')), \
                example.type_info(arrow_cast(42, 'UInt64'))",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt16Array>()
            .expect("UInt8 double promotes to UInt16")
            .value(0),
        20
    );
    assert_eq!(
        batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("mixed signed/unsigned addition promotes to Int64")
            .value(0),
        5
    );
    assert_eq!(
        batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("type_info returns Utf8")
            .value(0),
        "uint64"
    );

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT example.geo_distance_struct( \
             named_struct('lat', arrow_cast(0.0, 'Float64'), 'lon', arrow_cast(0.0, 'Float64')), \
             named_struct('lat', arrow_cast(3.0, 'Float64'), 'lon', arrow_cast(4.0, 'Float64')) \
         )",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("geo distance returns Float64")
            .value(0),
        5.0
    );

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT array_length(example.main.unnest_tensor( \
             {tensor: [10, 20, 30], axes: {i: [0, 1, 2]}} \
         ))",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::UInt64Array>()
            .expect("array_length returns UInt64")
            .value(0),
        3
    );

    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT example.main.unnest_tensor( \
             arrow_cast(NULL, 'Struct(\"tensor\": List(Int32), \"axes\": Struct(\"i\": List(Int32)))') \
         ) IS NULL",
    )
    .await?
    .collect()
    .await?;
    assert!(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::BooleanArray>()
        .expect("IS NULL returns Boolean")
        .value(0));

    // A bare NULL in an AnyArrow vararg group adopts the unambiguous type of
    // its concrete peers before binding and remains null through the RPC.
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT example.sum_values(NULL, 2, 3), \
                example.sum_values(1, NULL, 3)",
    )
    .await?
    .collect()
    .await?;
    assert!(batches[0].column(0).is_null(0));
    assert!(batches[0].column(1).is_null(0));
    Ok(())
}
