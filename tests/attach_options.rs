// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Typed VGI catalog ATTACH options through SQL.

use std::path::PathBuf;

use datafusion::prelude::SessionContext;

fn worker() -> Option<String> {
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
            return Some(format!(
                "env VGI_WORKER_CATALOG_NAME=attach_options {}",
                exe.display()
            ));
        }
    }
    None
}

#[tokio::test(flavor = "multi_thread")]
async fn all_declared_types_round_trip() -> datafusion::error::Result<()> {
    let Some(worker) = worker() else {
        return Ok(());
    };
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            r#"ATTACH 'attach_options' AS ao (
                TYPE vgi,
                LOCATION '{worker}',
                opt_bool false,
                opt_int8 -1,
                opt_int16 -1000,
                opt_int32 7,
                opt_int64 9999999999,
                opt_uint8 200,
                opt_uint16 60000,
                opt_uint32 4000000000,
                opt_uint64 18000000000,
                opt_float32 1.25,
                opt_float64 3.25,
                opt_string 'world',
                opt_blob '\xDE\xAD\xBE\xEF'::BLOB,
                opt_date DATE '2025-01-02',
                opt_time TIME '12:34:56',
                opt_timestamp TIMESTAMP '2025-01-02 03:04:05',
                opt_timestamp_tz TIMESTAMPTZ '2025-01-02 03:04:05Z',
                opt_decimal 12.3400::DECIMAL(18,4),
                opt_list [10, 20, 30],
                opt_struct {{'a': 42, 'b': 'hi'}}
            )"#
        ),
    )
    .await?;

    let batches = vgi_datafusion::sql(&ctx, "SELECT * FROM ao.main.echo_attach_options()")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
    assert_eq!(batches[0].num_columns(), 20);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn required_unknown_and_unsupported_options_are_clear() -> datafusion::error::Result<()> {
    let Some(worker) = worker() else {
        return Ok(());
    };
    let ctx = SessionContext::new();

    let missing = vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'attach_options_required' (TYPE vgi, LOCATION '{worker}', region 'west')"),
    )
    .await
    .expect_err("api_key is required")
    .to_string();
    assert!(missing.contains("api_key"), "{missing}");

    let unknown = vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'attach_options' (TYPE vgi, LOCATION '{worker}', nope 1)"),
    )
    .await
    .expect_err("unknown worker option")
    .to_string();
    assert!(
        unknown.contains("unknown ATTACH option `nope`"),
        "{unknown}"
    );

    vgi_datafusion::sql(
        &ctx,
        &format!("ATTACH 'attach_options' (TYPE vgi, LOCATION '{worker}', cache true)"),
    )
    .await
    .expect("cache is a supported local option");
    let stats = vgi_datafusion::sql(&ctx, "SELECT entries FROM vgi_cache_stats()")
        .await?
        .collect()
        .await?;
    assert_eq!(stats.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);
    Ok(())
}
