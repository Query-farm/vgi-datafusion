// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Exposing a whole VGI catalog to DataFusion.
//!
//! # The half-async problem, and how this answers it
//!
//! DataFusion's [`SchemaProvider`] is half async: [`table`](SchemaProvider::table)
//! may await, but [`table_names`](SchemaProvider::table_names) and
//! [`table_exist`](SchemaProvider::table_exist) may not. So *something* has to
//! be resolved eagerly, and the only question is what.
//!
//! The obvious reading — resolve everything up front — is ruinous here. Naming
//! a table is one `catalog_schema_contents_functions` call for the whole schema;
//! producing a [`TableProvider`] means **binding** that function, because
//! `TableProvider::schema()` is synchronous and DataFusion needs the schema
//! during planning. Binding every table at attach time costs one bind per table
//! whether or not the query touches it. Against the reference fixture worker,
//! which defines a couple of hundred functions, that was the difference between
//! attaching in about a second and attaching in about thirteen minutes.
//!
//! So this splits the two: **names eagerly, binds lazily.** One RPC per schema
//! at attach; a bind on first use, memoised. That also matches how the DuckDB
//! extension behaves — its `VgiCatalogSet` lazy-loads entries and its table set
//! resolves single tables on demand rather than materialising the catalog.
//!
//! A bind that fails is remembered as "not a usable bare table" rather than
//! retried on every lookup — plenty of functions require arguments and will
//! never bind bare, and re-binding them on each plan would reintroduce the cost
//! this design exists to avoid.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, SchemaProvider, TableProvider};
use datafusion::common::{DataFusionError, Result as DFResult};

use crate::{to_df, VgiConnection, VgiTableProvider};

/// One VGI schema.
#[derive(Debug)]
pub struct VgiSchemaProvider {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    /// Every function name the schema advertises, resolved once at attach.
    names: Vec<String>,
    /// Bind results, memoised. An `Err` records a function that will not bind
    /// bare — most fixture functions take arguments — so it is not retried,
    /// and the worker's own reason is kept to report at plan time.
    bound: Mutex<HashMap<String, Result<Arc<dyn TableProvider>, String>>>,
}

impl VgiSchemaProvider {
    /// List one schema's table functions. One RPC; no binds.
    pub async fn discover(
        conn: VgiConnection,
        catalog: &str,
        schema_name: &str,
    ) -> DFResult<Arc<Self>> {
        let (c, cat, sch) = (conn.clone(), catalog.to_string(), schema_name.to_string());
        let names = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = client
                .attach(&cat, vgi_client::AttachOptions::default())
                .map_err(to_df)?;
            let fns = client
                .functions(&attached, &sch, vgi_client::FunctionKind::Table)
                .map_err(to_df)?;
            Ok::<_, DataFusionError>(fns.into_iter().map(|f| f.name).collect::<Vec<_>>())
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        Ok(Arc::new(Self {
            conn,
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            names,
            bound: Mutex::new(HashMap::new()),
        }))
    }

    /// Look up a memoised bind without holding the lock across an await.
    fn cached(&self, name: &str) -> Option<Result<Arc<dyn TableProvider>, String>> {
        self.bound.lock().ok()?.get(name).cloned()
    }
}

#[async_trait]
impl SchemaProvider for VgiSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.names.clone()
    }

    async fn table(&self, name: &str) -> DFResult<Option<Arc<dyn TableProvider>>> {
        if !self.names.iter().any(|n| n == name) {
            return Ok(None);
        }
        if let Some(hit) = self.cached(name) {
            return hit.map(Some).map_err(bind_failed(name));
        }

        // Two callers racing the same cold name both bind; the loser's result
        // is dropped. That is cheaper and simpler than holding a lock across
        // the await, and binds are idempotent.
        let bound =
            VgiTableProvider::bind(self.conn.clone(), &self.catalog, &self.schema_name, name)
                .await
                .map(|p| p as Arc<dyn TableProvider>)
                .map_err(|e| e.to_string());

        if let Ok(mut cache) = self.bound.lock() {
            cache.insert(name.to_string(), bound.clone());
        }
        bound.map(Some).map_err(bind_failed(name))
    }

    fn table_exist(&self, name: &str) -> bool {
        // Answers from the name list, so it stays synchronous and cheap — which
        // is the whole point of not binding eagerly. A function that is
        // advertised but needs arguments reports `true` here and then fails in
        // `table()` with the worker's own reason, which is a far better message
        // than "table not found" for something that plainly exists.
        self.names.iter().any(|n| n == name)
    }
}

/// Turn a bind failure into a plan error that says why.
fn bind_failed(name: &str) -> impl Fn(String) -> DataFusionError + '_ {
    move |reason| {
        DataFusionError::Plan(format!(
            "VGI function `{name}` is advertised by the worker but does not bind \
             as a bare table: {reason}"
        ))
    }
}

/// A whole VGI catalog.
#[derive(Debug)]
pub struct VgiCatalogProvider {
    schemas: HashMap<String, Arc<dyn SchemaProvider>>,
}

impl VgiCatalogProvider {
    /// Attach a catalog and list its schemas.
    pub async fn discover(conn: VgiConnection, catalog: &str) -> DFResult<Arc<Self>> {
        let (c, cat) = (conn.clone(), catalog.to_string());
        let names = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = client
                .attach(&cat, vgi_client::AttachOptions::default())
                .map_err(to_df)?;
            let s = client.schemas(&attached).map_err(to_df)?;
            Ok::<_, DataFusionError>(s.into_iter().map(|s| s.name).collect::<Vec<_>>())
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let mut schemas: HashMap<String, Arc<dyn SchemaProvider>> = HashMap::new();
        for name in names {
            let sp = VgiSchemaProvider::discover(conn.clone(), catalog, &name).await?;
            schemas.insert(name, sp as Arc<dyn SchemaProvider>);
        }
        Ok(Arc::new(Self { schemas }))
    }
}

impl CatalogProvider for VgiCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).cloned()
    }
}
