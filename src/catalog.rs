// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Exposing a whole VGI catalog to DataFusion.
//!
//! DataFusion's `SchemaProvider` is half async: `table()` may await, but
//! `table_names()` and `table_exist()` may not. So discovery runs once, eagerly,
//! and the result is served synchronously afterwards — which is also what
//! DataFusion's own `AsyncSchemaProvider::resolve` does for remote catalogs.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, SchemaProvider, TableProvider};
use datafusion::common::{DataFusionError, Result as DFResult};

use crate::{to_df, VgiConnection, VgiTableProvider};

/// One VGI schema.
#[derive(Debug)]
pub struct VgiSchemaProvider {
    tables: HashMap<String, Arc<dyn TableProvider>>,
}

impl VgiSchemaProvider {
    /// Discover every function-backed table in one schema.
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

        let mut tables: HashMap<String, Arc<dyn TableProvider>> = HashMap::new();
        for name in names {
            // A function that will not bind without arguments is not usable as a
            // bare table; skip it rather than failing the whole discovery.
            if let Ok(p) = VgiTableProvider::bind(conn.clone(), catalog, schema_name, &name).await {
                tables.insert(name, p as Arc<dyn TableProvider>);
            }
        }
        Ok(Arc::new(Self { tables }))
    }
}

#[async_trait]
impl SchemaProvider for VgiSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(&self, name: &str) -> DFResult<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.get(name).cloned())
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

/// A whole VGI catalog.
#[derive(Debug)]
pub struct VgiCatalogProvider {
    schemas: HashMap<String, Arc<dyn SchemaProvider>>,
}

impl VgiCatalogProvider {
    /// Attach a catalog and discover every schema in it.
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
