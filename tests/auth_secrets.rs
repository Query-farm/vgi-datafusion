// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Deterministic host-secret resolution against the example VGI worker.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::StringArray;
use datafusion::common::ScalarValue;
use datafusion::prelude::{SessionConfig, SessionContext};
use vgi_datafusion::{VgiResolvedSecret, VgiRuntime, VgiSecretResolver, VgiSessionOptions};

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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Lookup {
    secret_type: String,
    scope: Option<String>,
    name: Option<String>,
}

struct ScopedSecretResolver {
    duplicate_names: bool,
    lookups: Mutex<Vec<Lookup>>,
}

impl ScopedSecretResolver {
    fn new(duplicate_names: bool) -> Self {
        Self {
            duplicate_names,
            lookups: Mutex::new(Vec::new()),
        }
    }

    fn lookups(&self) -> Vec<Lookup> {
        self.lookups.lock().expect("lookup mutex poisoned").clone()
    }
}

#[async_trait::async_trait]
impl VgiSecretResolver for ScopedSecretResolver {
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

        let (resolved_name, api_key) = match (secret_type, scope) {
            ("vgi_example", Some("s3://bucket-a/")) => ("scoped_a", "ka"),
            ("vgi_example", Some("s3://bucket-b/")) => ("scoped_b", "kb"),
            _ => return Ok(None),
        };
        Ok(Some(VgiResolvedSecret {
            name: if self.duplicate_names {
                "duplicate".to_string()
            } else {
                resolved_name.to_string()
            },
            fields: BTreeMap::from([
                (
                    "type".to_string(),
                    ScalarValue::Utf8(Some(secret_type.to_string())),
                ),
                (
                    "scope".to_string(),
                    ScalarValue::Utf8(scope.map(str::to_string)),
                ),
                (
                    "api_key".to_string(),
                    ScalarValue::Utf8(Some(api_key.to_string())),
                ),
            ]),
        }))
    }
}

async fn attach_with_resolver(
    worker: &std::path::Path,
    resolver: Arc<ScopedSecretResolver>,
) -> datafusion::common::Result<SessionContext> {
    let runtime =
        Arc::new(VgiRuntime::new(VgiSessionOptions::default()).with_secret_resolver(resolver));
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_extension(runtime));
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

async fn api_key(ctx: &SessionContext, path: &str) -> datafusion::common::Result<String> {
    let batches = vgi_datafusion::sql(
        ctx,
        &format!("SELECT api_key FROM ex.main.multi_secret_demo('{path}')"),
    )
    .await?
    .collect()
    .await?;
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("api_key is Utf8");
    Ok(values.value(0).to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn same_type_multi_scope_secrets_survive_one_bind() -> datafusion::common::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let resolver = Arc::new(ScopedSecretResolver::new(false));
    let ctx = attach_with_resolver(&worker, Arc::clone(&resolver)).await?;

    assert_eq!(api_key(&ctx, "s3://bucket-a/data.parquet").await?, "ka");
    assert_eq!(api_key(&ctx, "s3://bucket-b/other/file.csv").await?, "kb");

    let requested = resolver.lookups().into_iter().collect::<BTreeSet<_>>();
    assert_eq!(
        requested,
        BTreeSet::from([
            Lookup {
                secret_type: "vgi_example".to_string(),
                scope: Some("s3://bucket-a/".to_string()),
                name: None,
            },
            Lookup {
                secret_type: "vgi_example".to_string(),
                scope: Some("s3://bucket-b/".to_string()),
                name: None,
            },
        ])
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn same_type_multi_scope_secrets_require_unique_names() -> datafusion::common::Result<()> {
    let Some(worker) = example_worker() else {
        eprintln!("skipping: vgi-example-worker not built");
        return Ok(());
    };
    let resolver = Arc::new(ScopedSecretResolver::new(true));
    let ctx = attach_with_resolver(&worker, resolver).await?;

    let error = vgi_datafusion::sql(
        &ctx,
        "SELECT api_key FROM ex.main.multi_secret_demo('s3://bucket-a/data.parquet')",
    )
    .await
    .expect_err("same-name secrets must not overwrite one another")
    .to_string();
    assert!(error.contains("duplicate secret name"), "{error}");
    assert!(!error.contains("ka") && !error.contains("kb"), "{error}");
    Ok(())
}
