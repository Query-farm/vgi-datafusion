// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! In-band worker logs reach both DataFusion diagnostic surfaces.
//!
//! Run this target once with the existing Unix fixture and once with HTTP:
//! `VGI_TEST_WORKER=unix:///...` or `VGI_TEST_WORKER=http://127.0.0.1:...`.

use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Array, StringArray};
use datafusion::prelude::{SessionConfig, SessionContext};
use vgi_datafusion::{VgiEvent, VgiEventSink, VgiRuntime};

#[derive(Default)]
struct CapturingSink(Mutex<Vec<VgiEvent>>);

impl CapturingSink {
    fn clear(&self) {
        self.0.lock().unwrap().clear();
    }

    fn worker_logs(&self) -> Vec<VgiEvent> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.kind.starts_with("worker.log."))
            .cloned()
            .collect()
    }
}

impl VgiEventSink for CapturingSink {
    fn emit(&self, event: &VgiEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

fn transport_location() -> Option<String> {
    let location = match std::env::var("VGI_TEST_WORKER") {
        Ok(location) if !location.trim().is_empty() => location,
        _ => {
            eprintln!("skipping: set VGI_TEST_WORKER to a Unix or HTTP example worker");
            return None;
        }
    };
    assert!(
        location.starts_with("unix://")
            || location.starts_with("http://")
            || location.starts_with("https://"),
        "worker-log contract requires a Unix or HTTP worker, got {location:?}"
    );
    Some(location)
}

fn quote(value: &str) -> String {
    value.replace('\'', "''")
}

async fn string_rows(
    context: &SessionContext,
    sql: &str,
) -> datafusion::common::Result<Vec<Vec<String>>> {
    let batches = vgi_datafusion::sql(context, sql).await?.collect().await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            let columns = (0..batch.num_columns())
                .map(|column| {
                    batch
                        .column(column)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .expect("diagnostic columns are Utf8")
                })
                .collect::<Vec<_>>();
            (0..batch.num_rows())
                .map(|row| {
                    columns
                        .iter()
                        .map(|column| {
                            assert!(!column.is_null(row));
                            column.value(row).to_string()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_logs_reach_history_duckdb_shape_and_sink_once() -> datafusion::common::Result<()> {
    let Some(location) = transport_location() else {
        return Ok(());
    };
    let sink = Arc::new(CapturingSink::default());
    let runtime = Arc::new(VgiRuntime::default().with_event_sink(sink.clone()));
    let context = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::clone(&runtime)),
    );
    vgi_datafusion::sql(
        &context,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            quote(&location)
        ),
    )
    .await?;
    runtime.clear_events();
    sink.clear();

    let batches = vgi_datafusion::sql(&context, "SELECT * FROM ex.main.logging_generator(2)")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        2
    );

    let expected = vec![
        vec![
            "worker.log.info".to_string(),
            "Generation complete".to_string(),
        ],
        vec![
            "worker.log.info".to_string(),
            "Starting generation of 2 values".to_string(),
        ],
    ];
    assert_eq!(
        string_rows(
            &context,
            "SELECT event, message FROM vgi_logs() \
             WHERE event LIKE 'worker.log.%' ORDER BY message",
        )
        .await?,
        expected
    );

    assert_eq!(
        string_rows(
            &context,
            "SELECT type, message FROM duckdb_logs() \
             WHERE message LIKE 'worker.log.%' ORDER BY message",
        )
        .await?,
        vec![
            vec![
                "VGI".to_string(),
                "worker.log.info Generation complete".to_string()
            ],
            vec![
                "VGI".to_string(),
                "worker.log.info Starting generation of 2 values".to_string(),
            ],
        ]
    );

    let delivered = sink.worker_logs();
    assert_eq!(delivered.len(), 2, "each in-band log reaches the sink once");
    assert_eq!(delivered[0].kind, "worker.log.info");
    assert_eq!(
        delivered[0].message.as_deref(),
        Some("Starting generation of 2 values")
    );
    assert_eq!(delivered[1].message.as_deref(), Some("Generation complete"));
    Ok(())
}
