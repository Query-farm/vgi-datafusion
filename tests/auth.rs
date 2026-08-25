// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Authentication through SQL ATTACH against a real protected VGI worker.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use datafusion::prelude::SessionContext;

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

struct ProtectedWorker {
    child: Child,
    url: String,
}

impl ProtectedWorker {
    fn start(exe: &PathBuf) -> Self {
        let mut child = Command::new(exe)
            .arg("--http")
            .env("VGI_BEARER_TOKENS", "sentinel-token=test-principal")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn protected worker");
        let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let mut line = String::new();
        let port = loop {
            line.clear();
            assert!(stdout.read_line(&mut line).expect("read port") > 0);
            if let Some(port) = line.trim().strip_prefix("PORT:") {
                break port.parse::<u16>().expect("port");
            }
        };
        Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
        }
    }
}

impl Drop for ProtectedWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bearer_auth_reaches_catalog_and_parallel_scans() -> datafusion::error::Result<()> {
    let Some(exe) = example_worker() else {
        return Ok(());
    };
    let worker = ProtectedWorker::start(&exe);
    let ctx = SessionContext::new();
    vgi_datafusion::sql(
        &ctx,
        &format!(
            "ATTACH 'example' (TYPE vgi, LOCATION '{}', bearer_token 'sentinel-token')",
            worker.url
        ),
    )
    .await?;
    let batches = vgi_datafusion::sql(
        &ctx,
        "SELECT count(*) FROM example.main.partitioned_sequence(total_rows := 100, partitions := 4)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );

    // Aggregate finalization is a synchronous DataFusion callback. Keep this
    // on the HTTP lane: constructing reqwest's blocking runtime directly from
    // that async callback used to panic even though scans correctly used
    // spawn_blocking.
    let aggregate = vgi_datafusion::sql(
        &ctx,
        "SELECT example.main.vgi_percentile(x::DOUBLE, 0.9) FROM range(10) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        aggregate[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .expect("percentile returns Float64")
            .value(0),
        9.0
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn auth_failures_are_actionable_and_redacted() -> datafusion::error::Result<()> {
    let Some(exe) = example_worker() else {
        return Ok(());
    };
    let worker = ProtectedWorker::start(&exe);

    let wrong = vgi_datafusion::sql(
        &SessionContext::new(),
        &format!(
            "ATTACH 'example' (TYPE vgi, LOCATION '{}', bearer_token 'wrong-sentinel')",
            worker.url
        ),
    )
    .await
    .expect_err("wrong bearer must fail")
    .to_string();
    assert!(wrong.contains("bearer token was rejected"), "{wrong}");
    assert!(!wrong.contains("wrong-sentinel"), "{wrong}");

    let missing = vgi_datafusion::sql(
        &SessionContext::new(),
        &format!("ATTACH 'example' (TYPE vgi, LOCATION '{}')", worker.url),
    )
    .await
    .expect_err("protected worker needs auth")
    .to_string();
    assert!(missing.contains("no OAuth challenge"), "{missing}");

    let both = vgi_datafusion::sql(
        &SessionContext::new(),
        &format!(
            "ATTACH 'example' (TYPE vgi, LOCATION '{}', bearer_token 'a', oauth_refresh_token 'b')",
            worker.url
        ),
    )
    .await
    .expect_err("credentials are exclusive")
    .to_string();
    assert!(both.contains("cannot specify both"), "{both}");
    assert!(!both.contains("bearer_token 'a'"), "{both}");
    Ok(())
}
