// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! DataFusion runtime join filters crossing the VGI scan boundary.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use datafusion::arrow::array::{Array, StringArray};
use datafusion::prelude::{SessionConfig, SessionContext};
use vgi_datafusion::{VgiConnection, VgiTableProvider};

fn example_worker() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("vgi-rust")
        .join("target");
    ["debug", "release"].into_iter().find_map(|profile| {
        let path = root.join(profile).join(if cfg!(windows) {
            "vgi-example-worker.exe"
        } else {
            "vgi-example-worker"
        });
        path.exists().then_some(path)
    })
}

struct HttpWorker {
    child: Child,
    port: u16,
}

impl HttpWorker {
    fn start(exe: &PathBuf) -> Self {
        let mut child = Command::new(exe)
            .arg("--http")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn HTTP worker");
        let stdout = child.stdout.take().expect("worker stdout");
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let port = loop {
            line.clear();
            assert!(
                reader.read_line(&mut line).expect("read worker port") > 0,
                "HTTP worker exited before announcing its port"
            );
            if let Some(port) = line.trim().strip_prefix("PORT:") {
                break port.parse().expect("valid worker port");
            }
        };
        Self { child, port }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for HttpWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn filter_echo_schema(connection: &VgiConnection) -> Option<String> {
    let connection = connection.clone();
    tokio::task::spawn_blocking(move || {
        let mut client = connection.connect().ok()?;
        let catalog = client
            .attach("example", vgi_client::AttachOptions::default())
            .ok()?;
        for schema in client.schemas(&catalog).ok()? {
            let functions = client
                .functions(&catalog, &schema.name, vgi_client::FunctionKind::Table)
                .ok()?;
            if functions
                .iter()
                .any(|function| function.name == "filter_echo_table_scan")
            {
                return Some(schema.name);
            }
        }
        None
    })
    .await
    .ok()
    .flatten()
}

#[tokio::test(flavor = "multi_thread")]
async fn hash_join_keys_reach_vgi_init_and_results_stay_exact() -> datafusion::error::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let connection = VgiConnection::subprocess([worker.to_string_lossy().to_string()]);
    let Some(schema) = filter_echo_schema(&connection).await else {
        eprintln!("skipping: filter_echo_table_scan not in this catalog");
        return Ok(());
    };

    let mut config = SessionConfig::new().with_target_partitions(1);
    config
        .options_mut()
        .optimizer
        .enable_join_dynamic_filter_pushdown = true;
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table(
        "echo",
        VgiTableProvider::bind(connection, "example", &schema, "filter_echo_table_scan").await?,
    )?;

    let dataframe = ctx
        .sql(
            "SELECT e.n, e.pushed_filters
             FROM (VALUES (1), (3), (5)) AS keys(id)
             JOIN echo e ON keys.id = e.n
             ORDER BY e.n",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let rendered = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        rendered.contains("VgiScanExec") && rendered.contains("dynamic_filters=1"),
        "DataFusion did not link its join filter to the VGI scan:\n{rendered}"
    );

    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        3
    );
    let pushed = batches
        .iter()
        .find_map(|batch| {
            let index = batch.schema().index_of("pushed_filters").ok()?;
            let strings = batch.column(index).as_any().downcast_ref::<StringArray>()?;
            (!strings.is_empty() && !strings.is_null(0)).then(|| strings.value(0).to_string())
        })
        .unwrap_or_default();
    assert!(
        pushed.contains("n IN (1, 3, 5)"),
        "worker did not receive the join-key set: {pushed:?}"
    );
    Ok(())
}

async fn assert_topk_refinements(location: &str) -> datafusion::error::Result<()> {
    let mut config = SessionConfig::new().with_target_partitions(1);
    config
        .options_mut()
        .optimizer
        .enable_topk_dynamic_filter_pushdown = true;
    let ctx = SessionContext::new_with_config(config);
    let location = location.replace('\'', "''");
    vgi_datafusion::sql(&ctx, &format!("ATTACH 'example?location={location}' AS ex")).await?;

    let dataframe = vgi_datafusion::sql(
        &ctx,
        "SELECT DISTINCT pushed_filters
         FROM (
           SELECT *
           FROM ex.main.dynamic_filter_echo(10000, batch_size := 100)
           ORDER BY n LIMIT 500
         )",
    )
    .await?;
    let plan = dataframe.create_physical_plan().await?;
    let rendered = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        rendered.contains("VgiScanExec") && rendered.contains("dynamic_filters=1"),
        "DataFusion did not link its Top-K filter to the VGI scan:\n{rendered}"
    );
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    let generations = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("pushed_filters is Utf8")
                .iter()
                .flatten()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(
        generations.len() > 1,
        "worker observed only these filter generations; continuation metadata did not tighten: {generations:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn topk_refinements_reach_vgi_continuation_ticks() -> datafusion::error::Result<()> {
    let location = match std::env::var("VGI_DYNAMIC_FILTER_LOCATION") {
        Ok(location) if !location.trim().is_empty() => location,
        _ => {
            let Some(worker) = example_worker() else {
                eprintln!("skipping: vgi-example-worker not built");
                return Ok(());
            };
            worker.to_string_lossy().to_string()
        }
    };
    assert_topk_refinements(&location).await
}

#[tokio::test(flavor = "multi_thread")]
async fn topk_refinements_reach_http_continuation_ticks() -> datafusion::error::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = HttpWorker::start(&worker);
    assert_topk_refinements(&worker.url()).await
}
