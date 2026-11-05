// Copyright 2026 Query Farm LLC - https://query.farm

//! Small VGI 3.0 catalog-shape adapters.

use datafusion::common::{DataFusionError, Result as DFResult};

/// Return the one schema component DataFusion's catalog SPI can represent.
///
/// VGI 3.0 preserves arbitrary raw schema paths. DataFusion's catalog provider
/// has one schema-name slot below a catalog, so accepting a nested path by
/// joining components would corrupt identifiers that themselves contain dots.
pub(crate) fn require_flat_schema_path(path: &[String], object: &str) -> DFResult<String> {
    match path {
        [name] if !name.is_empty() => Ok(name.clone()),
        [] => Err(DataFusionError::NotImplemented(format!(
            "VGI 3.0 {object} has an empty schema path"
        ))),
        _ => Err(DataFusionError::NotImplemented(format!(
            "VGI 3.0 {object} uses nested schema path {}; DataFusion's catalog SPI exposes only one schema level",
            display_schema_path(path)
        ))),
    }
}

/// The leaf component used after catalog discovery has validated that paths
/// are flat. Empty paths remain an empty string for diagnostics of malformed
/// metadata rather than panicking.
pub(crate) fn schema_leaf(path: &[String]) -> &str {
    path.last().map(String::as_str).unwrap_or("")
}

/// Unambiguous human-readable raw path.
pub(crate) fn display_schema_path(path: &[String]) -> String {
    path.iter()
        .map(|part| format!("{part:?}"))
        .collect::<Vec<_>>()
        .join("/")
}

/// DuckDB-compatible display for a protocol 3 column path. Preserve the
/// ordinary top-level name exactly; use the unambiguous raw-path rendering
/// only when the protocol path really is nested.
pub(crate) fn display_column_path(path: &[String]) -> String {
    match path {
        [name] => name.clone(),
        _ => display_schema_path(path),
    }
}
