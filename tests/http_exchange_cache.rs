// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Conditional exchange-cache behavior over a real HTTP transport.
//!
//! The broader cache suite also exercises subprocess and Unix workers. These
//! focused tests keep HTTP continuation metadata honest: validators,
//! `not_modified`, stale fallback, and revocation must survive
//! HTTP request/response boundaries for both streaming table input and stable
//! scalar per-value exchanges.

mod common;

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, Int64Array, UInt64Array};
use datafusion::prelude::{SessionConfig, SessionContext};

struct HttpWorker {
    child: Child,
    url: String,
}

impl HttpWorker {
    const START_TIMEOUT: Duration = Duration::from_secs(10);

    fn startup_failed(mut child: Child, message: String) -> ! {
        let _ = child.kill();
        let _ = child.wait();
        panic!("HTTP worker failed to start: {message}");
    }

    fn start(executable: &Path) -> Self {
        let mut child = Command::new(executable)
            .arg("--http")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn HTTP worker");
        let stdout = child.stdout.take().expect("worker stdout");
        let (lines, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match stdout.read_line(&mut line) {
                    Ok(0) => {
                        let _ = lines.send(Err("stdout closed before PORT announcement".into()));
                        break;
                    }
                    Ok(_) => {
                        if lines.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = lines.send(Err(format!("failed reading stdout: {error}")));
                        break;
                    }
                }
            }
        });

        let deadline = Instant::now() + Self::START_TIMEOUT;
        let port = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match receiver.recv_timeout(remaining.min(Duration::from_millis(50))) {
                Ok(Ok(line)) => {
                    if let Some(port) = line.trim().strip_prefix("PORT:") {
                        match port.parse::<u16>() {
                            Ok(port) => break port,
                            Err(error) => Self::startup_failed(
                                child,
                                format!("invalid PORT announcement {port:?}: {error}"),
                            ),
                        }
                    }
                }
                Ok(Err(error)) => Self::startup_failed(child, error),
                Err(RecvTimeoutError::Disconnected) => Self::startup_failed(
                    child,
                    "stdout reader stopped before PORT announcement".into(),
                ),
                Err(RecvTimeoutError::Timeout) => {}
            }

            match child.try_wait() {
                Ok(Some(status)) => Self::startup_failed(
                    child,
                    format!("child exited before PORT announcement with {status}"),
                ),
                Ok(None) => {}
                Err(error) => {
                    Self::startup_failed(child, format!("could not inspect child status: {error}"))
                }
            }
            if Instant::now() >= deadline {
                Self::startup_failed(
                    child,
                    format!(
                        "timed out after {:.1}s waiting for PORT announcement",
                        Self::START_TIMEOUT.as_secs_f64()
                    ),
                );
            }
        };

        Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
        }
    }
}

impl Drop for HttpWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Copy, Debug)]
struct CacheStats {
    entries: u64,
    revalidations: u64,
    stale_serves: u64,
    exchange_hits: u64,
    exchange_stores: u64,
}

async fn attached(url: &str) -> datafusion::common::Result<SessionContext> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_batch_size(1024),
    );
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            common::sql_quote(url)
        ),
    )
    .await?;
    Ok(ctx)
}

async fn stats(ctx: &SessionContext) -> datafusion::common::Result<CacheStats> {
    let batches = vgi_datafusion::sql(
        ctx,
        "SELECT entries, revalidations, stale_serves, exchange_hits, exchange_stores \
         FROM vgi_cache_stats()",
    )
    .await?
    .collect()
    .await?;
    let value = |index| {
        batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("cache statistic is UInt64")
            .value(0)
    };
    Ok(CacheStats {
        entries: value(0),
        revalidations: value(1),
        stale_serves: value(2),
        exchange_hits: value(3),
        exchange_stores: value(4),
    })
}

fn sum_value(batches: &[datafusion::arrow::record_batch::RecordBatch]) -> i64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("sum is Int64")
        .value(0)
}

async fn query_sum(ctx: &SessionContext, query: &str) -> datafusion::common::Result<i64> {
    let batches = vgi_datafusion::sql(ctx, query).await?.collect().await?;
    Ok(sum_value(&batches))
}

#[tokio::test(flavor = "multi_thread")]
async fn http_not_modified_replays_streaming_and_scalar_exchanges() -> datafusion::common::Result<()>
{
    let Some(executable) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = HttpWorker::start(&executable);

    const STREAMING: &str = "SELECT sum(x) FROM ex.main.cached_reval_echo(\
                             (SELECT x FROM range(5) t(x)))";
    let streaming = attached(&worker.url).await?;
    assert_eq!(query_sum(&streaming, STREAMING).await?, 10);
    let cold = stats(&streaming).await?;
    assert_eq!(cold.entries, 1);
    assert_eq!(cold.revalidations, 0);
    assert_eq!(cold.exchange_stores, 1);
    assert_eq!(query_sum(&streaming, STREAMING).await?, 10);
    let warm = stats(&streaming).await?;
    assert_eq!(
        warm.revalidations,
        cold.revalidations + 1,
        "the warm HTTP table-input request must carry its validator and accept not_modified"
    );
    assert_eq!(warm.exchange_hits, cold.exchange_hits + 1);
    assert_eq!(warm.exchange_stores, cold.exchange_stores);

    const SCALAR: &str =
        "SELECT sum(ex.main.cached_reval_double_scalar(x + 21)) FROM range(1) t(x)";
    let scalar = attached(&worker.url).await?;
    assert_eq!(query_sum(&scalar, SCALAR).await?, 42);
    let cold = stats(&scalar).await?;
    assert_eq!(cold.entries, 1);
    assert_eq!(cold.revalidations, 0);
    assert_eq!(cold.exchange_stores, 1);
    assert_eq!(query_sum(&scalar, SCALAR).await?, 42);
    let warm = stats(&scalar).await?;
    assert_eq!(
        warm.revalidations,
        cold.revalidations + 1,
        "the warm HTTP scalar-value request must carry its validator and accept not_modified"
    );
    assert_eq!(warm.exchange_hits, cold.exchange_hits + 1);
    assert_eq!(warm.exchange_stores, cold.exchange_stores);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn http_buffered_whole_input_revalidation_policies_are_preserved(
) -> datafusion::common::Result<()> {
    let Some(executable) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = HttpWorker::start(&executable);

    const NOT_MODIFIED: &str = "SELECT x FROM ex.main.cached_reval_sum_all(\
                                (SELECT x FROM range(5) t(x)), logging := false)";
    let not_modified = attached(&worker.url).await?;
    assert_eq!(query_sum(&not_modified, NOT_MODIFIED).await?, 10);
    let cold = stats(&not_modified).await?;
    assert_eq!(cold.entries, 1);
    assert_eq!(cold.revalidations, 0);
    assert_eq!(cold.exchange_stores, 1);
    assert_eq!(query_sum(&not_modified, NOT_MODIFIED).await?, 10);
    let warm = stats(&not_modified).await?;
    assert_eq!(warm.entries, 1);
    assert_eq!(warm.revalidations, cold.revalidations + 1);
    assert_eq!(warm.exchange_hits, cold.exchange_hits + 1);
    assert_eq!(warm.exchange_stores, cold.exchange_stores);

    const STALE_IF_ERROR: &str = "SELECT x FROM ex.main.cached_reval_error_sum_all(\
                                  (SELECT x FROM range(5) t(x)), logging := false)";
    let stale_if_error = attached(&worker.url).await?;
    assert_eq!(query_sum(&stale_if_error, STALE_IF_ERROR).await?, 10);
    let cold = stats(&stale_if_error).await?;
    assert_eq!(cold.entries, 1);
    assert_eq!(cold.stale_serves, 0);
    assert_eq!(cold.exchange_stores, 1);
    assert_eq!(query_sum(&stale_if_error, STALE_IF_ERROR).await?, 10);
    let warm = stats(&stale_if_error).await?;
    assert_eq!(warm.entries, 1);
    assert_eq!(warm.stale_serves, cold.stale_serves + 1);
    assert_eq!(warm.exchange_hits, cold.exchange_hits + 1);
    assert_eq!(warm.exchange_stores, cold.exchange_stores);

    const NOT_MODIFIED_NO_STORE: &str = "SELECT x FROM ex.main.cached_reval_no_store_sum_all(\
                                        (SELECT x FROM range(5) t(x)), logging := false)";
    let not_modified_no_store = attached(&worker.url).await?;
    assert_eq!(
        query_sum(&not_modified_no_store, NOT_MODIFIED_NO_STORE).await?,
        10
    );
    let cold = stats(&not_modified_no_store).await?;
    assert_eq!(cold.entries, 1);
    assert_eq!(cold.exchange_stores, 1);
    assert!(
        query_sum(&not_modified_no_store, NOT_MODIFIED_NO_STORE)
            .await
            .is_err(),
        "not_modified plus no_store must not replay stale buffered bytes"
    );
    let rejected = stats(&not_modified_no_store).await?;
    assert_eq!(rejected.entries, 0);
    assert_eq!(rejected.stale_serves, cold.stale_serves);
    assert_eq!(rejected.exchange_stores, cold.exchange_stores);

    const FRESH_NO_STORE: &str = "SELECT x FROM ex.main.cached_reval_fresh_no_store_sum_all(\
                                  (SELECT x FROM range(5) t(x)), logging := false)";
    let fresh_no_store = attached(&worker.url).await?;
    assert_eq!(query_sum(&fresh_no_store, FRESH_NO_STORE).await?, 10);
    let cold = stats(&fresh_no_store).await?;
    assert_eq!(cold.entries, 1);
    assert_eq!(cold.exchange_stores, 1);
    assert_eq!(query_sum(&fresh_no_store, FRESH_NO_STORE).await?, 10);
    let fresh = stats(&fresh_no_store).await?;
    assert_eq!(fresh.entries, 0);
    assert_eq!(fresh.stale_serves, cold.stale_serves);
    assert_eq!(fresh.exchange_stores, cold.exchange_stores);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn http_stale_if_error_and_revocation_cover_both_exchange_shapes(
) -> datafusion::common::Result<()> {
    let Some(executable) = common::example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = HttpWorker::start(&executable);

    const STREAMING_ERROR: &str = "SELECT sum(x) FROM ex.main.cached_reval_policy(\
                                   (SELECT x FROM range(5) t(x)), 'error')";
    let streaming_error = attached(&worker.url).await?;
    assert_eq!(query_sum(&streaming_error, STREAMING_ERROR).await?, 10);
    assert_eq!(query_sum(&streaming_error, STREAMING_ERROR).await?, 10);
    let after = stats(&streaming_error).await?;
    assert_eq!(after.entries, 1);
    assert_eq!(after.stale_serves, 1);

    const SCALAR_ERROR: &str = "SELECT sum(ex.main.cached_reval_policy_scalar(\
                                x + 21, 'error')) FROM range(1) t(x)";
    let scalar_error = attached(&worker.url).await?;
    assert_eq!(query_sum(&scalar_error, SCALAR_ERROR).await?, 42);
    assert_eq!(query_sum(&scalar_error, SCALAR_ERROR).await?, 42);
    let after = stats(&scalar_error).await?;
    assert_eq!(after.entries, 1);
    assert_eq!(after.stale_serves, 1);

    const STREAMING_REVOKE: &str = "SELECT sum(x) FROM ex.main.cached_reval_policy(\
                                    (SELECT x FROM range(5) t(x)), 'fresh_no_store')";
    let streaming_revoke = attached(&worker.url).await?;
    assert_eq!(query_sum(&streaming_revoke, STREAMING_REVOKE).await?, 10);
    assert_eq!(stats(&streaming_revoke).await?.entries, 1);
    assert_eq!(query_sum(&streaming_revoke, STREAMING_REVOKE).await?, 10);
    assert_eq!(stats(&streaming_revoke).await?.entries, 0);

    const SCALAR_REVOKE: &str = "SELECT sum(ex.main.cached_reval_policy_scalar(\
                                 x + 21, 'fresh_no_store')) FROM range(1) t(x)";
    let scalar_revoke = attached(&worker.url).await?;
    assert_eq!(query_sum(&scalar_revoke, SCALAR_REVOKE).await?, 42);
    assert_eq!(stats(&scalar_revoke).await?.entries, 1);
    assert_eq!(query_sum(&scalar_revoke, SCALAR_REVOKE).await?, 42);
    assert_eq!(stats(&scalar_revoke).await?.entries, 0);
    Ok(())
}
