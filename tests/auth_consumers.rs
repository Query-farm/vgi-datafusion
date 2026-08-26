// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! End-to-end authentication coverage across every read-only VGI function
//! shape DataFusion can execute today.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, Float64Array, Int64Array, StringArray};
use datafusion::common::ScalarValue;
use datafusion::prelude::{SessionConfig, SessionContext};
use vgi_datafusion::{
    VgiConnection, VgiResolvedSecret, VgiRuntime, VgiSecretResolver, VgiSessionOptions,
    VgiTableProvider,
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct Lookup {
    secret_type: String,
    scope: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Default)]
struct DeterministicResolver {
    lookups: Mutex<Vec<Lookup>>,
}

impl DeterministicResolver {
    fn lookups(&self) -> Vec<Lookup> {
        self.lookups.lock().expect("lookup mutex poisoned").clone()
    }
}

#[async_trait::async_trait]
impl VgiSecretResolver for DeterministicResolver {
    async fn resolve(
        &self,
        secret_type: &str,
        scope: Option<&str>,
        name: Option<&str>,
    ) -> datafusion::common::Result<Option<VgiResolvedSecret>> {
        self.lookups
            .lock()
            .expect("lookup mutex poisoned")
            .push(Lookup {
                secret_type: secret_type.to_string(),
                scope: scope.map(str::to_string),
                name: name.map(str::to_string),
            });
        if secret_type != "vgi_example" || scope.is_some() || name.is_some() {
            return Ok(None);
        }
        Ok(Some(VgiResolvedSecret {
            name: "deterministic_test_secret".to_string(),
            fields: BTreeMap::from([
                (
                    "type".to_string(),
                    ScalarValue::Utf8(Some(secret_type.to_string())),
                ),
                (
                    "secret_string".to_string(),
                    ScalarValue::Utf8(Some("coverage-secret".to_string())),
                ),
                (
                    "api_key".to_string(),
                    ScalarValue::Utf8(Some("k-test".to_string())),
                ),
                ("port".to_string(), ScalarValue::Int32(Some(4242))),
                ("use_ssl".to_string(), ScalarValue::Boolean(Some(true))),
                ("timeout".to_string(), ScalarValue::Float64(Some(2.5))),
            ]),
        }))
    }
}

async fn attached_with_resolver(
    worker: &Path,
    resolver: Arc<DeterministicResolver>,
) -> datafusion::common::Result<SessionContext> {
    let runtime =
        Arc::new(VgiRuntime::new(VgiSessionOptions::default()).with_secret_resolver(resolver));
    let context = SessionContext::new_with_config(SessionConfig::new().with_extension(runtime));
    vgi_datafusion::sql(
        &context,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;
    Ok(context)
}

async fn query_string(context: &SessionContext, query: &str) -> datafusion::common::Result<String> {
    let batches = vgi_datafusion::sql(context, query).await?.collect().await?;
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("query returns Utf8");
    Ok(values.value(0).to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn deterministic_secret_reaches_nullary_scalars() -> datafusion::common::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let resolver = Arc::new(DeterministicResolver::default());
    let context = attached_with_resolver(&worker, Arc::clone(&resolver)).await?;

    // Nullary scalar: two-phase bind and exchange both retain the resolved
    // secret. The accessor-oriented scalar independently verifies typed fields.
    let json = query_string(&context, "SELECT ex.main.return_secret_value()").await?;
    assert!(
        json.contains("coverage-secret"),
        "unexpected secret JSON: {json}"
    );
    assert_eq!(
        query_string(&context, "SELECT ex.main.secret_field()").await?,
        "port=4242;name=coverage-secret"
    );
    let cache_rows = vgi_datafusion::sql(
        &context,
        "SELECT CAST(count(*) AS BIGINT) FROM vgi_cache_entries()",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        cache_rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is Int64")
            .value(0),
        0,
        "secret-dependent scalars must ignore worker cache opt-in"
    );
    // DataFusion may ask for a dynamic return type more than once while
    // planning. Secret-dependent type answers are deliberately not memoized
    // across resolver rotations, so require both phases without asserting a
    // planner implementation detail.
    assert!(
        resolver.lookups().len() >= 4,
        "planning and execution must both resolve the host secret"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn deterministic_secret_reaches_table_aggregate_and_exchange_shapes(
) -> datafusion::common::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let resolver = Arc::new(DeterministicResolver::default());
    let context = attached_with_resolver(&worker, Arc::clone(&resolver)).await?;

    // Direct table function.
    let batches = vgi_datafusion::sql(
        &context,
        "SELECT value, arrow_type FROM ex.main.secret_demo() WHERE key = 'port'",
    )
    .await?
    .collect()
    .await?;
    let value = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("value is Utf8")
        .value(0);
    let arrow_type = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("arrow_type is Utf8")
        .value(0);
    assert_eq!((value, arrow_type), ("4242", "int32"));

    // Function-backed catalog table: its lazy bare-table bind must perform the
    // same two-phase secret retry as a directly called table function.
    assert_eq!(
        query_string(
            &context,
            "SELECT value FROM ex.data.secret_demo_table WHERE key = 'secret_string'",
        )
        .await?,
        "coverage-secret"
    );
    assert_eq!(
        query_string(
            &context,
            "SELECT value FROM ex.data.secret_demo_table WHERE key = 'secret_string'",
        )
        .await?,
        "coverage-secret"
    );
    let cache_rows = vgi_datafusion::sql(
        &context,
        "SELECT CAST(count(*) AS BIGINT) FROM vgi_cache_entries()",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        cache_rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is Int64")
            .value(0),
        0,
        "secret-dependent producer results must ignore worker cache opt-in"
    );

    // Aggregate metadata requests the secret before the input schema is known;
    // the resolved Boolean deliberately selects its Float64 return arm.
    let batches = vgi_datafusion::sql(
        &context,
        "SELECT ex.main.secret_typed_sum(x) FROM (VALUES (2::BIGINT), (3::BIGINT)) t(x)",
    )
    .await?
    .collect()
    .await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("use_ssl=true selects Float64")
            .value(0),
        5.0
    );

    // Streaming table-in/out: bind keeps both the scalar arguments and input
    // schema across its resolved-secret retry, then process receives the same
    // secret for every row.
    let batches = vgi_datafusion::sql(
        &context,
        "SELECT x, secret_string FROM ex.main.secret_in_out(\
            (SELECT * FROM (VALUES (11::BIGINT), (22::BIGINT)) AS input(x))) ORDER BY x",
    )
    .await?
    .collect()
    .await?;
    let xs = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("x is Int64");
    let secrets = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("secret_string is Utf8");
    assert_eq!((xs.value(0), xs.value(1)), (11, 22));
    assert_eq!(
        (secrets.value(0), secrets.value(1)),
        ("coverage-secret", "coverage-secret")
    );

    let lookups = resolver.lookups();
    assert!(
        lookups.len() >= 4,
        "every bind shape should resolve independently: {lookups:?}"
    );
    assert!(lookups.iter().all(|lookup| {
        lookup.secret_type == "vgi_example" && lookup.scope.is_none() && lookup.name.is_none()
    }));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_host_secret_resolver_fails_before_execution() -> datafusion::common::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let context = SessionContext::new();
    vgi_datafusion::sql(
        &context,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}')",
            worker.to_string_lossy()
        ),
    )
    .await?;

    let error = vgi_datafusion::sql(&context, "SELECT * FROM ex.main.secret_demo()")
        .await
        .expect_err("the CLI-like session has no host secret resolver")
        .to_string();
    assert!(error.contains("no VgiSecretResolver"), "{error}");
    assert!(!error.contains("coverage-secret") && !error.contains("k-test"));
    Ok(())
}

struct ProtectedWorker {
    child: Child,
    url: String,
}

impl ProtectedWorker {
    const START_TIMEOUT: Duration = Duration::from_secs(10);

    fn startup_failed(mut child: Child, message: String) -> ! {
        let _ = child.kill();
        let _ = child.wait();
        panic!("protected worker failed to start: {message}");
    }

    fn start(executable: &Path) -> Self {
        let mut child = Command::new(executable)
            .arg("--http")
            .env(
                "VGI_BEARER_TOKENS",
                "identity-alpha=alice,identity-beta=bob",
            )
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn protected worker");
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
        match child.try_wait() {
            Ok(Some(status)) => Self::startup_failed(
                child,
                format!("child exited immediately after PORT announcement with {status}"),
            ),
            Ok(None) => {}
            Err(error) => Self::startup_failed(
                child,
                format!("could not inspect child status after PORT announcement: {error}"),
            ),
        }
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
async fn bearer_principals_cannot_cross_serve_cached_results() -> datafusion::common::Result<()> {
    let Some(executable) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = ProtectedWorker::start(&executable);
    let runtime = Arc::new(VgiRuntime::new(VgiSessionOptions::default()));
    let context =
        SessionContext::new_with_config(SessionConfig::new().with_extension(Arc::clone(&runtime)));

    vgi_datafusion::sql(
        &context,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}', bearer_token 'identity-alpha')",
            worker.url
        ),
    )
    .await?;
    assert_eq!(
        query_string(&context, "SELECT who FROM ex.data.cache_whoami").await?,
        "alice"
    );
    assert_eq!(
        query_string(&context, "SELECT who FROM ex.data.cache_whoami").await?,
        "alice",
        "the warm read should reuse alice's entry"
    );
    vgi_datafusion::sql(&context, "DETACH ex").await?;

    vgi_datafusion::sql(
        &context,
        &format!(
            "ATTACH 'example' AS ex (TYPE vgi, LOCATION '{}', bearer_token 'identity-beta')",
            worker.url
        ),
    )
    .await?;
    assert_eq!(
        query_string(&context, "SELECT who FROM ex.data.cache_whoami").await?,
        "bob",
        "bob must never receive alice's cached bytes"
    );

    let stats = runtime.result_cache().stats();
    assert_eq!(stats.entries, 2, "one cache entry per bearer identity");
    assert!(stats.hits >= 1, "alice's warm read should hit the cache");
    Ok(())
}

/// A locally resolved OAuth identity. The fixture still authenticates the HTTP
/// request with its deterministic bearer value; overriding `identity()` is what
/// isolates this test to the OAuth issuer/subject cache-key branch without an
/// external discovery endpoint, token exchange, or credentials.
struct ResolvedOAuthAuth {
    bearer: String,
    issuer: String,
    subject: String,
}

impl vgi_client::auth::CatalogAuth for ResolvedOAuthAuth {
    fn bearer_token(&self) -> Option<String> {
        Some(self.bearer.clone())
    }

    fn handle_unauthorized(
        &self,
        _challenge: Option<&vgi_client::auth::OAuthChallenge>,
    ) -> vgi_client::Result<String> {
        Err(vgi_client::RpcError::new(
            "AuthError",
            "the deterministic OAuth fixture bearer was rejected",
        ))
    }

    fn is_explicitly_configured(&self) -> bool {
        true
    }

    fn identity(&self) -> vgi_client::auth::Identity {
        vgi_client::auth::Identity::OAuth {
            issuer: self.issuer.clone(),
            subject: self.subject.clone(),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn resolved_oauth_subjects_cannot_cross_serve_cached_results(
) -> datafusion::common::Result<()> {
    let Some(executable) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let worker = ProtectedWorker::start(&executable);
    let runtime = Arc::new(VgiRuntime::new(VgiSessionOptions::default()));
    let context = SessionContext::new();

    for (table, bearer, subject) in [
        ("oauth_alice", "identity-alpha", "alice-subject"),
        ("oauth_bob", "identity-beta", "bob-subject"),
    ] {
        let auth = Arc::new(ResolvedOAuthAuth {
            bearer: bearer.to_string(),
            issuer: "https://issuer.fixture.example".to_string(),
            subject: subject.to_string(),
        });
        let connection = VgiConnection::http(&worker.url)
            .with_auth(auth)?
            .with_runtime(Arc::clone(&runtime));
        context.register_table(
            table,
            VgiTableProvider::bind(connection, "example", "main", "cache_whoami").await?,
        )?;
    }

    assert_eq!(
        query_string(&context, "SELECT who FROM oauth_alice").await?,
        "alice"
    );
    assert_eq!(
        query_string(&context, "SELECT who FROM oauth_alice").await?,
        "alice"
    );
    assert_eq!(
        query_string(&context, "SELECT who FROM oauth_bob").await?,
        "bob",
        "OAuth subject bob must never receive alice's cached bytes"
    );

    let stats = runtime.result_cache().stats();
    assert_eq!(stats.entries, 2, "one cache entry per OAuth subject");
    assert!(stats.hits >= 1);
    Ok(())
}
