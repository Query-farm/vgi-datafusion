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

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::array::{
    Array, FixedSizeListArray, LargeListArray, LargeStringArray, ListArray, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::catalog::{CatalogProvider, SchemaProvider, Session, TableProvider};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{
    Constraints, DFSchema, DataFusionError, Result as DFResult, ScalarValue, Statistics,
};
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::execution::SessionState;
use datafusion::logical_expr::utils::expr_to_columns;
use datafusion::logical_expr::{
    Expr, ExprSchemable, Operator, TableProviderFilterPushDown, TableType, Volatility,
};
use datafusion::physical_expr::expressions::{
    binary as physical_binary, case as physical_case, cast as physical_cast,
    Column as PhysicalColumn, Literal as PhysicalLiteral,
};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::{datafusion_constraints, to_df, VgiConnection, VgiTableProvider};

type CachedTable = Result<Arc<dyn TableProvider>, String>;

#[derive(Debug)]
struct BoundCatalogBranch {
    info: vgi_client::ScanBranch,
    provider: Arc<dyn TableProvider>,
    /// DuckDB's exact CSV null marker. DataFusion 55 applies `null_regex`
    /// during inference but not execution, so string columns need a local
    /// projection until that existing option is honored by CsvSource.
    null_string: Option<String>,
}

#[derive(Debug)]
struct NativeFormatSpec {
    format: String,
    locations: Vec<String>,
    options: Vec<(String, vgi_client::ArgValue)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CatalogTableSpec {
    catalog: String,
    schema: String,
    table: String,
}

/// Collect the catalog column paths constrained by predicates offered to a
/// table provider. DataFusion lowers `s.a` and deeper struct access to the
/// variadic `get_field(s, 'a', ...)` UDF. Record that as the precise dotted
/// path and skip its children: also recording the base `s` would incorrectly
/// satisfy every required descendant of the struct.
fn filtered_catalog_paths(filters: &[Expr]) -> DFResult<HashSet<String>> {
    let mut present = HashSet::new();
    for filter in filters {
        filter.apply(|expr| {
            if let Expr::ScalarFunction(function) = expr {
                if function.name() == "get_field" {
                    if let Some(path) = static_get_field_path(expr) {
                        present.insert(path);
                        return Ok(TreeNodeRecursion::Jump);
                    }
                }
            }
            if let Expr::Column(column) = expr {
                present.insert(column.name.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
    }
    Ok(present)
}

fn static_get_field_path(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name.clone()),
        Expr::ScalarFunction(function)
            if function.name() == "get_field" && function.args.len() >= 2 =>
        {
            let mut path = static_get_field_path(&function.args[0])?;
            for field in &function.args[1..] {
                let name = match field {
                    Expr::Literal(ScalarValue::Utf8(Some(name)), _)
                    | Expr::Literal(ScalarValue::Utf8View(Some(name)), _)
                    | Expr::Literal(ScalarValue::LargeUtf8(Some(name)), _) => name,
                    _ => return None,
                };
                path.push('.');
                path.push_str(name);
            }
            Some(path)
        }
        _ => None,
    }
}

fn required_path_is_present(required: &str, present: &HashSet<String>) -> bool {
    let mut candidate = required;
    loop {
        if present.contains(candidate) {
            return true;
        }
        let Some((parent, _)) = candidate.rsplit_once('.') else {
            return false;
        };
        candidate = parent;
    }
}

fn render_required_filter_group(group: &[String]) -> String {
    if group.len() == 1 {
        group[0].clone()
    } else {
        format!("one of ({})", group.join(", "))
    }
}

fn render_required_filter_groups<'a>(groups: impl IntoIterator<Item = &'a Vec<String>>) -> String {
    groups
        .into_iter()
        .map(|group| render_required_filter_group(group))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CatalogTableIdentity {
    catalog: String,
    schema: String,
    table: String,
}

impl CatalogTableIdentity {
    fn new(catalog: &str, schema: &str, table: &str) -> Self {
        Self {
            catalog: catalog.to_ascii_lowercase(),
            schema: schema.to_ascii_lowercase(),
            table: table.to_ascii_lowercase(),
        }
    }

    fn display(&self) -> String {
        format!("{}.{}.{}", self.catalog, self.schema, self.table)
    }
}

fn catalog_table_spec(branch: &vgi_client::ScanBranch) -> DFResult<Option<CatalogTableSpec>> {
    let Some(table) = branch
        .source_table
        .as_deref()
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let catalog = branch
        .source_catalog
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "VGI catalog-table branch for source table {table:?} has no source_catalog"
            ))
        })?;
    let schema = branch
        .source_schema
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "VGI catalog-table branch for source table {table:?} has no source_schema"
            ))
        })?;
    Ok(Some(CatalogTableSpec {
        catalog: catalog.to_string(),
        schema: schema.to_string(),
        table: table.to_string(),
    }))
}

fn select_catalog_alias(
    requested: &str,
    catalog_names: &[String],
    worker_catalogs: &[(String, String)],
) -> DFResult<String> {
    // The protocol normally names the companion's ATTACH alias. Preserve that
    // qualification whenever it exists rather than guessing from its target.
    if catalog_names.iter().any(|name| name == requested) {
        return Ok(requested.to_string());
    }

    let mut aliases = catalog_names
        .iter()
        .filter(|name| name.eq_ignore_ascii_case(requested))
        .cloned()
        .collect::<Vec<_>>();
    aliases.sort();
    aliases.dedup();
    match aliases.as_slice() {
        [alias] => return Ok(alias.clone()),
        [] => {}
        _ => {
            return Err(DataFusionError::Plan(format!(
                "VGI catalog-table source catalog {requested:?} ambiguously matches attached aliases {}",
                aliases.join(", ")
            )));
        }
    }

    // A worker can name the catalog target rather than the host alias. Only
    // accept that spelling when retained VGI attachment metadata identifies a
    // unique mounted alias; silently choosing one of two mounts would scan the
    // wrong catalog.
    let available = catalog_names.iter().collect::<HashSet<_>>();
    aliases = worker_catalogs
        .iter()
        .filter(|(alias, worker)| {
            worker.eq_ignore_ascii_case(requested) && available.contains(alias)
        })
        .map(|(alias, _)| alias.clone())
        .collect();
    aliases.sort();
    aliases.dedup();
    match aliases.as_slice() {
        [alias] => Ok(alias.clone()),
        [] => Err(DataFusionError::Plan(format!(
            "VGI catalog-table source catalog {requested:?} is not attached"
        ))),
        _ => Err(DataFusionError::Plan(format!(
            "VGI catalog-table source catalog {requested:?} maps to multiple attached aliases {}; use the companion attachment alias",
            aliases.join(", ")
        ))),
    }
}

fn native_reader_format(function: &str) -> Option<&'static str> {
    match function.to_ascii_lowercase().as_str() {
        "read_csv" | "read_csv_auto" => Some("csv"),
        "read_parquet" | "parquet_scan" => Some("parquet"),
        "read_json" | "read_json_auto" | "read_ndjson" => Some("json"),
        "read_arrow" => Some("arrow"),
        _ => None,
    }
}

fn strings_from_arrow(array: &dyn Array, label: &str) -> DFResult<Vec<String>> {
    if array.is_empty() || array.is_null(0) {
        return Err(DataFusionError::Plan(format!(
            "VGI native format {label} must not be NULL"
        )));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(vec![values.value(0).to_string()]);
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(vec![values.value(0).to_string()]);
    }
    let nested = if let Some(values) = array.as_any().downcast_ref::<ListArray>() {
        Some(values.value(0))
    } else if let Some(values) = array.as_any().downcast_ref::<LargeListArray>() {
        Some(values.value(0))
    } else {
        array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .map(|values| values.value(0))
    };
    if let Some(values) = nested {
        if let Some(strings) = values.as_any().downcast_ref::<StringArray>() {
            return (0..strings.len())
                .map(|index| {
                    (!strings.is_null(index))
                        .then(|| strings.value(index).to_string())
                        .ok_or_else(|| {
                            DataFusionError::Plan(format!(
                                "VGI native format {label} contains a NULL location"
                            ))
                        })
                })
                .collect();
        }
        if let Some(strings) = values.as_any().downcast_ref::<LargeStringArray>() {
            return (0..strings.len())
                .map(|index| {
                    (!strings.is_null(index))
                        .then(|| strings.value(index).to_string())
                        .ok_or_else(|| {
                            DataFusionError::Plan(format!(
                                "VGI native format {label} contains a NULL location"
                            ))
                        })
                })
                .collect();
        }
    }
    Err(DataFusionError::Plan(format!(
        "VGI native format {label} must be a string or list of strings, found {:?}",
        array.data_type()
    )))
}

fn locations_from_argument(value: &vgi_client::ArgValue) -> DFResult<Vec<String>> {
    match value {
        vgi_client::ArgValue::Text(value) => Ok(vec![value.clone()]),
        vgi_client::ArgValue::Arrow(array) => strings_from_arrow(array.as_ref(), "location"),
        vgi_client::ArgValue::Null(_) | vgi_client::ArgValue::Placeholder(_) => Err(
            DataFusionError::Plan("VGI native format location must not be NULL".to_string()),
        ),
        other => Err(DataFusionError::Plan(format!(
            "VGI native format location must be a string or list of strings, found {other:?}"
        ))),
    }
}

fn option_value(value: &vgi_client::ArgValue, name: &str) -> DFResult<String> {
    match value {
        vgi_client::ArgValue::Int(value) => Ok(value.to_string()),
        vgi_client::ArgValue::Float(value) => Ok(value.to_string()),
        vgi_client::ArgValue::Text(value) => Ok(value.clone()),
        vgi_client::ArgValue::Bool(value) => Ok(value.to_string()),
        vgi_client::ArgValue::Arrow(array) => {
            if array.is_empty() || array.is_null(0) {
                return Err(DataFusionError::Plan(format!(
                    "VGI native format option {name:?} must not be NULL"
                )));
            }
            let formatter = ArrayFormatter::try_new(array.as_ref(), &FormatOptions::default())?;
            Ok(formatter.value(0).to_string())
        }
        vgi_client::ArgValue::Null(_) | vgi_client::ArgValue::Placeholder(_) => {
            Err(DataFusionError::Plan(format!(
                "VGI native format option {name:?} must not be NULL"
            )))
        }
    }
}

fn exact_regex(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 4);
    escaped.push('^');
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped.push('$');
    escaped
}

fn datafusion_format_options(
    options: &[(String, vgi_client::ArgValue)],
) -> DFResult<HashMap<String, String>> {
    let mut translated = HashMap::new();
    for (name, value) in options {
        let normalized = name.to_ascii_lowercase();
        let exact_null = matches!(normalized.as_str(), "nullstr" | "null_string");
        let key = match normalized.as_str() {
            "delim" | "delimiter" => "format.delimiter".to_string(),
            "header" | "has_header" => "format.has_header".to_string(),
            "nullstr" | "null_string" => "format.null_regex".to_string(),
            _ if normalized.starts_with("format.") => normalized,
            _ => format!("format.{normalized}"),
        };
        let mut value = option_value(value, name)?;
        if exact_null {
            value = exact_regex(&value);
        }
        if translated.insert(key.clone(), value).is_some() {
            return Err(DataFusionError::Plan(format!(
                "VGI native format supplied duplicate option {key:?} through aliases"
            )));
        }
    }
    Ok(translated)
}

fn exact_null_string(options: &[(String, vgi_client::ArgValue)]) -> DFResult<Option<String>> {
    options
        .iter()
        .find(|(name, _)| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "nullstr" | "null_string"
            )
        })
        .map(|(name, value)| option_value(value, name))
        .transpose()
}

fn native_format_spec(branch: &vgi_client::ScanBranch) -> DFResult<Option<NativeFormatSpec>> {
    if let Some(format) = branch
        .format_name
        .as_deref()
        .filter(|format| !format.is_empty())
    {
        let arguments = match branch.format_options.as_ref() {
            Some(options) => {
                vgi_client::Arguments::from_scan_arguments(&options.0).map_err(to_df)?
            }
            None => vgi_client::Arguments::new(),
        };
        if !arguments.positional_values().is_empty() {
            return Err(DataFusionError::Plan(
                "VGI native format options must all be named".to_string(),
            ));
        }
        return Ok(Some(NativeFormatSpec {
            format: format.to_ascii_lowercase(),
            locations: branch.format_locations.clone().unwrap_or_default(),
            options: arguments.named_values().to_vec(),
        }));
    }

    let Some(format) = native_reader_format(&branch.function_name) else {
        return Ok(None);
    };
    let arguments =
        vgi_client::Arguments::from_scan_arguments(&branch.arguments.0).map_err(to_df)?;
    let [location] = arguments.positional_values() else {
        return Err(DataFusionError::Plan(format!(
            "VGI native reader {:?} requires exactly one positional location argument",
            branch.function_name
        )));
    };
    let locations = locations_from_argument(location)?;
    Ok(Some(NativeFormatSpec {
        format: format.to_string(),
        locations,
        options: arguments.named_values().to_vec(),
    }))
}

#[cfg(test)]
mod required_filter_tests {
    use super::*;
    use datafusion::functions::core::expr_fn::get_field_path;
    use datafusion::prelude::{col, lit};

    #[test]
    fn dotted_get_field_paths_do_not_collapse_to_the_parent_struct() {
        let filters = vec![
            col("top").eq(lit(1_i64)),
            get_field_path(col("s"), vec![lit("a")]).eq(lit(2_i64)),
            get_field_path(col("wrapper"), vec![lit("mid"), lit("leaf")]).is_not_null(),
        ];
        let present = filtered_catalog_paths(&filters).unwrap();
        assert!(present.contains("top"));
        assert!(present.contains("s.a"));
        assert!(present.contains("wrapper.mid.leaf"));
        assert!(!present.contains("s"));
        assert!(!present.contains("wrapper"));
    }

    #[test]
    fn only_present_parent_paths_satisfy_required_descendants() {
        let present = HashSet::from(["bbox".to_string(), "ticker.symbol".to_string()]);
        assert!(required_path_is_present("bbox.xmin", &present));
        assert!(required_path_is_present("ticker.symbol", &present));
        assert!(!required_path_is_present("ticker", &present));
        assert!(!required_path_is_present("bbox2.xmin", &present));
    }

    #[test]
    fn cnf_groups_render_with_or_members() {
        let required = vec![vec!["ticker".into(), "cik".into()], vec!["date".into()]];
        assert_eq!(
            render_required_filter_groups(&required),
            "one of (ticker, cik), date"
        );
    }
}

#[cfg(test)]
mod native_format_tests {
    use super::*;
    use datafusion::arrow::array::Int32Array;
    use datafusion::arrow::datatypes::Field;
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;

    fn catalog_provider_named(mount_alias: &str, name: &str) -> Arc<VgiCatalogTableProvider> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let info = vgi_client::TableInfo {
            comment: None,
            tags: Vec::new(),
            name: name.into(),
            schema_name: "main".into(),
            columns: vgi_protocol::ipc::write_schema(&schema).unwrap().into(),
            not_null_constraints: Vec::new(),
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
            primary_key_constraints: Vec::new(),
            foreign_key_constraints: Vec::new(),
            supports_insert: false,
            supports_update: false,
            supports_delete: false,
            supports_returning: false,
            supports_column_statistics: false,
            scan_function: None,
            insert_function: None,
            update_function: None,
            delete_function: None,
            cardinality_estimate: None.into(),
            cardinality_max: None.into(),
            column_statistics: None,
            bind_result: None,
            required_filters: Vec::new(),
        };
        VgiCatalogTableProvider::new(
            VgiConnection::subprocess(["/bin/false"]),
            mount_alias,
            "example",
            "main",
            info,
            None,
        )
        .unwrap()
    }

    fn catalog_provider_for_test() -> Arc<VgiCatalogTableProvider> {
        catalog_provider_named("root", "events")
    }

    fn source_branch(provider: Arc<dyn TableProvider>) -> BoundCatalogBranch {
        BoundCatalogBranch {
            info: vgi_client::ScanBranch {
                function_name: String::new(),
                arguments: Vec::<u8>::new().into(),
                branch_filter: None,
                writable: false,
                source_catalog: Some("source".into()),
                source_schema: Some("main".into()),
                source_table: Some("events".into()),
                format_name: None,
                format_locations: None,
                format_options: None,
            },
            provider,
            null_string: None,
        }
    }

    #[test]
    fn translates_duckdb_csv_options_without_losing_semantics() {
        let options = vec![
            ("delim".to_string(), vgi_client::ArgValue::Text("|".into())),
            ("header".to_string(), vgi_client::ArgValue::Bool(true)),
            (
                "nullstr".to_string(),
                vgi_client::ArgValue::Text("row_2".into()),
            ),
        ];
        let translated = datafusion_format_options(&options).unwrap();
        assert_eq!(translated["format.delimiter"], "|");
        assert_eq!(translated["format.has_header"], "true");
        assert_eq!(translated["format.null_regex"], "^row_2$");
    }

    #[test]
    fn null_string_becomes_an_exact_regex() {
        assert_eq!(exact_regex(r"a.b[0]\\end"), r"^a\.b\[0\]\\\\end$");
    }

    #[test]
    fn reads_scalar_arrow_string_locations() {
        let value =
            vgi_client::ArgValue::Arrow(Arc::new(StringArray::from(vec!["/tmp/branch.parquet"])));
        assert_eq!(
            locations_from_argument(&value).unwrap(),
            ["/tmp/branch.parquet"]
        );
    }

    #[test]
    fn only_local_worker_transports_may_nominate_local_files() {
        assert!(VgiConnection::from_location("/opt/vgi/worker")
            .unwrap()
            .allows_local_format_paths());
        assert!(VgiConnection::from_location("unix:///tmp/vgi.sock")
            .unwrap()
            .allows_local_format_paths());
        assert!(VgiConnection::from_location("http://127.0.0.1:8080")
            .unwrap()
            .allows_local_format_paths());
        assert!(VgiConnection::from_location("tcp://[::1]:9000")
            .unwrap()
            .allows_local_format_paths());
        assert!(!VgiConnection::from_location("https://worker.example/vgi")
            .unwrap()
            .allows_local_format_paths());
        assert!(!VgiConnection::from_location("tcp://worker.example:9000")
            .unwrap()
            .allows_local_format_paths());
        assert!(VgiConnection::from_location("https://worker.example/vgi")
            .unwrap()
            .with_local_format_paths(true)
            .allows_local_format_paths());
    }

    #[test]
    fn catalog_table_branches_require_fully_qualified_sources() {
        let branch = vgi_client::ScanBranch {
            function_name: String::new(),
            arguments: Vec::<u8>::new().into(),
            branch_filter: None,
            writable: false,
            source_catalog: Some("lake".into()),
            source_schema: Some("main".into()),
            source_table: Some("events".into()),
            format_name: None,
            format_locations: None,
            format_options: None,
        };
        assert_eq!(
            catalog_table_spec(&branch).unwrap(),
            Some(CatalogTableSpec {
                catalog: "lake".into(),
                schema: "main".into(),
                table: "events".into(),
            })
        );

        let mut missing_schema = branch;
        missing_schema.source_schema = None;
        let error = catalog_table_spec(&missing_schema).unwrap_err();
        assert!(error.to_string().contains("no source_schema"));
    }

    #[test]
    fn source_catalog_prefers_alias_then_unique_worker_catalog() {
        let catalogs = vec!["main_mount".to_string(), "lake_alias".to_string()];
        let workers = vec![
            ("main_mount".to_string(), "example".to_string()),
            ("lake_alias".to_string(), "acme_lake".to_string()),
        ];
        assert_eq!(
            select_catalog_alias("lake_alias", &catalogs, &workers).unwrap(),
            "lake_alias"
        );
        assert_eq!(
            select_catalog_alias("acme_lake", &catalogs, &workers).unwrap(),
            "lake_alias"
        );
        assert!(select_catalog_alias("missing", &catalogs, &workers)
            .unwrap_err()
            .to_string()
            .contains("is not attached"));
    }

    #[test]
    fn source_catalog_rejects_ambiguous_worker_catalog_mounts() {
        let catalogs = vec!["lake_one".to_string(), "lake_two".to_string()];
        let workers = vec![
            ("lake_one".to_string(), "acme_lake".to_string()),
            ("lake_two".to_string(), "acme_lake".to_string()),
        ];
        let error = select_catalog_alias("acme_lake", &catalogs, &workers).unwrap_err();
        assert!(error.to_string().contains("multiple attached aliases"));
        assert!(error.to_string().contains("lake_one, lake_two"));
    }

    #[test]
    fn reports_the_indirect_catalog_branch_cycle_path() {
        let a = catalog_provider_named("root", "events");
        let b = catalog_provider_named("lake", "events_archive");
        let c = catalog_provider_named("cold", "events_history");
        let b_provider: Arc<dyn TableProvider> = b.clone();
        a.branches
            .set(Some(vec![source_branch(b_provider)]))
            .unwrap();
        let c_provider: Arc<dyn TableProvider> = c.clone();
        b.branches
            .set(Some(vec![source_branch(c_provider)]))
            .unwrap();
        let a_provider: Arc<dyn TableProvider> = a;
        assert_eq!(
            c.catalog_source_cycle(&a_provider).unwrap(),
            [
                "cold.main.events_history",
                "root.main.events",
                "lake.main.events_archive",
                "cold.main.events_history"
            ]
        );
    }

    #[tokio::test]
    async fn resolves_catalog_table_branch_through_datafusion_catalogs() {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        ctx.register_table(
            "source_events",
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap()),
        )
        .unwrap();
        let owner = catalog_provider_for_test();
        let state = ctx.state();
        let resolved = owner
            .bind_catalog_table_source(
                &state,
                CatalogTableSpec {
                    catalog: "datafusion".into(),
                    schema: "public".into(),
                    table: "source_events".into(),
                },
                0,
            )
            .await
            .unwrap();
        assert_eq!(resolved.schema(), schema);
    }

    #[tokio::test]
    async fn rejects_direct_catalog_table_branch_recursion() {
        let ctx = SessionContext::new();
        let owner = catalog_provider_for_test();
        ctx.register_table("events", owner.clone()).unwrap();
        let state = ctx.state();
        let error = owner
            .bind_catalog_table_source(
                &state,
                CatalogTableSpec {
                    catalog: "datafusion".into(),
                    schema: "public".into(),
                    table: "events".into(),
                },
                0,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("directly references itself"));
    }

    #[tokio::test]
    async fn catalog_table_branches_share_reconciliation_and_branch_filters() {
        let ctx = SessionContext::new();
        let raw_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&raw_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .unwrap();
        let raw_provider =
            Arc::new(MemTable::try_new(raw_schema, vec![vec![batch]]).unwrap()) as Arc<_>;
        let branch = BoundCatalogBranch {
            info: vgi_client::ScanBranch {
                function_name: String::new(),
                arguments: Vec::<u8>::new().into(),
                branch_filter: Some("id >= 2".into()),
                writable: false,
                source_catalog: Some("datafusion".into()),
                source_schema: Some("public".into()),
                source_table: Some("source_events".into()),
                format_name: None,
                format_locations: None,
                format_options: None,
            },
            provider: raw_provider,
            null_string: None,
        };
        let owner = catalog_provider_for_test();
        let state = ctx.state();
        let plan = owner
            .scan_multi_branches(&state, &[branch], None, &[], None)
            .await
            .unwrap();
        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap();
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[2]);
    }
}

/// A catalog table whose schema is available from discovery and whose scan
/// function is bound only when DataFusion actually scans it.
///
/// DataFusion builds `information_schema.columns` and `views` by asking for
/// every listed provider, even when SQL filters name one table. Binding here
/// would therefore turn metadata lookup into one RPC per table and make a
/// valid argument-dependent/multi-branch table abort the whole query.
#[derive(Debug)]
struct VgiCatalogTableProvider {
    conn: VgiConnection,
    mount_alias: String,
    catalog: String,
    schema_name: String,
    info: vgi_client::TableInfo,
    at: Option<vgi_client::At>,
    /// Complete catalog schema, including virtual generated columns.
    output_schema: SchemaRef,
    /// Columns the worker's nominated scan function physically emits.
    physical_schema: SchemaRef,
    constraints: Constraints,
    bound: tokio::sync::OnceCell<Arc<VgiTableProvider>>,
    /// Function-backed multi-branch sources. `None` means the branches RPC
    /// resolved to the ordinary legacy-compatible one-function shape.
    branches: tokio::sync::OnceCell<Option<Vec<BoundCatalogBranch>>>,
    statistics: tokio::sync::OnceCell<Option<Arc<Statistics>>>,
}

impl VgiCatalogTableProvider {
    fn new(
        conn: VgiConnection,
        mount_alias: impl Into<String>,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        info: vgi_client::TableInfo,
        at: Option<vgi_client::At>,
    ) -> DFResult<Arc<Self>> {
        let output_schema = vgi_protocol::ipc::read_schema(&info.columns.0).map_err(to_df)?;
        let physical_schema = Arc::new(Schema::new_with_metadata(
            output_schema
                .fields()
                .iter()
                .filter(|field| !field.metadata().contains_key("generated_expression"))
                .cloned()
                .collect::<Vec<_>>(),
            output_schema.metadata().clone(),
        ));
        let constraints = datafusion_constraints(
            info.primary_key_constraints.clone(),
            info.unique_constraints.clone(),
            output_schema.fields().len(),
        )?;
        Ok(Arc::new(Self {
            conn,
            mount_alias: mount_alias.into(),
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            info,
            at,
            output_schema,
            physical_schema,
            constraints,
            bound: tokio::sync::OnceCell::new(),
            branches: tokio::sync::OnceCell::new(),
            statistics: tokio::sync::OnceCell::new(),
        }))
    }

    fn enforce_required_filters(&self, filters: &[Expr]) -> DFResult<()> {
        let required = &self.info.required_filters;
        if required.is_empty() {
            return Ok(());
        }
        let present = filtered_catalog_paths(filters)?;
        let missing = required
            .iter()
            .filter(|group| {
                !group
                    .iter()
                    .any(|path| required_path_is_present(path, &present))
            })
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        Err(DataFusionError::Plan(format!(
            "VGI table '{}.{}.{}' requires WHERE filters on: {}. Missing: {}. \
             Add predicates targeting those columns (or a filter on a parent struct) \
             to avoid scanning the entire table.",
            self.mount_alias,
            self.schema_name,
            self.info.name,
            render_required_filter_groups(required),
            render_required_filter_groups(missing),
        )))
    }

    async fn bound(&self) -> DFResult<&Arc<VgiTableProvider>> {
        let provider = self
            .bound
            .get_or_try_init(|| async {
                let provider = match &self.at {
                    Some(at) => {
                        VgiTableProvider::bind_catalog_table_at(
                            self.conn.clone(),
                            &self.catalog,
                            &self.schema_name,
                            self.info.clone(),
                            at.clone(),
                        )
                        .await
                    }
                    None => {
                        VgiTableProvider::bind_catalog_table(
                            self.conn.clone(),
                            &self.catalog,
                            &self.schema_name,
                            self.info.clone(),
                        )
                        .await
                    }
                }?;
                provider.with_declared_schema(Arc::clone(&self.physical_schema))
            })
            .await?;
        Ok(provider)
    }

    async fn bind_native_format(
        &self,
        state: &dyn Session,
        spec: NativeFormatSpec,
        branch_index: usize,
    ) -> DFResult<Arc<dyn TableProvider>> {
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "VGI native format branches require DataFusion SessionState file-format support"
                        .to_string(),
                )
            })?;
        if spec.locations.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "VGI native format branch {branch_index} for `{}.{}` names no locations",
                self.schema_name, self.info.name
            )));
        }
        let paths = spec
            .locations
            .iter()
            .map(|location| {
                let path = ListingTableUrl::parse(location)?;
                if path.get_url().scheme() == "file" && !self.conn.allows_local_format_paths() {
                    return Err(DataFusionError::Plan(format!(
                        "remote VGI catalog {:?} may not nominate local format location {location:?}",
                        self.mount_alias
                    )));
                }
                Ok(path)
            })
            .collect::<DFResult<Vec<_>>>()?;
        let factory = session_state
            .get_file_format_factory(&spec.format)
            .ok_or_else(|| {
                DataFusionError::NotImplemented(format!(
                    "VGI native format branch {branch_index} for `{}.{}` requires DataFusion file format {:?}, but no such format is registered",
                    self.schema_name, self.info.name, spec.format
                ))
            })?;
        let options = datafusion_format_options(&spec.options)?;
        let format = factory.create(session_state, &options)?;
        // The worker explicitly declared the format, so do not exclude a file
        // merely because its location lacks that format's conventional suffix.
        let listing_options = ListingOptions::new(format).with_file_extension("");
        let config = ListingTableConfig::new_with_multi_paths(paths)
            .with_listing_options(listing_options)
            .infer_schema(session_state)
            .await?;
        Ok(Arc::new(ListingTable::try_new(config)?))
    }

    fn identity(&self) -> CatalogTableIdentity {
        CatalogTableIdentity::new(&self.mount_alias, &self.schema_name, &self.info.name)
    }

    fn catalog_source_cycle(&self, provider: &Arc<dyn TableProvider>) -> Option<Vec<String>> {
        fn reaches(
            provider: &Arc<dyn TableProvider>,
            target: *const VgiCatalogTableProvider,
            visited: &mut HashSet<usize>,
            path: &mut Vec<CatalogTableIdentity>,
        ) -> bool {
            let Some(source) = provider.downcast_ref::<VgiCatalogTableProvider>() else {
                return false;
            };
            let pointer = source as *const VgiCatalogTableProvider;
            if !visited.insert(pointer as usize) {
                return false;
            }
            path.push(source.identity());
            if pointer == target {
                return true;
            }
            if let Some(branches) = source.branches.get().and_then(Option::as_ref) {
                for branch in branches {
                    if reaches(&branch.provider, target, visited, path) {
                        return true;
                    }
                }
            }
            path.pop();
            false
        }

        let mut path = vec![self.identity()];
        reaches(
            provider,
            self as *const VgiCatalogTableProvider,
            &mut HashSet::new(),
            &mut path,
        )
        .then(|| path.into_iter().map(|item| item.display()).collect())
    }

    async fn bind_catalog_table_source(
        &self,
        state: &dyn Session,
        spec: CatalogTableSpec,
        branch_index: usize,
    ) -> DFResult<Arc<dyn TableProvider>> {
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "VGI catalog-table branches require DataFusion SessionState catalog support"
                        .to_string(),
                )
            })?;
        let catalog_names = session_state.catalog_list().catalog_names();
        let worker_catalogs = self
            .conn
            .runtime()
            .catalog_metadata()
            .into_iter()
            .map(|(alias, metadata)| (alias, metadata.worker_catalog))
            .collect::<Vec<_>>();
        let alias = select_catalog_alias(&spec.catalog, &catalog_names, &worker_catalogs)?;
        let catalog = session_state
            .catalog_list()
            .catalog(&alias)
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "VGI catalog-table scan branch {branch_index} resolved source catalog {:?} to alias {alias:?}, but that alias is no longer attached",
                    spec.catalog
                ))
            })?;
        let schema = catalog.schema(&spec.schema).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "VGI catalog-table scan branch {branch_index} references missing schema {alias}.{}",
                spec.schema
            ))
        })?;
        let provider = schema.table(&spec.table).await?.ok_or_else(|| {
            DataFusionError::Plan(format!(
                "VGI catalog-table scan branch {branch_index} references missing table {alias}.{}.{}",
                spec.schema, spec.table
            ))
        })?;

        if provider
            .downcast_ref::<VgiCatalogTableProvider>()
            .is_some_and(|source| std::ptr::eq(source, self))
        {
            return Err(DataFusionError::Plan(format!(
                "VGI catalog-table scan branch {branch_index} for `{}.{}` directly references itself as {alias}.{}.{}",
                self.schema_name, self.info.name, spec.schema, spec.table
            )));
        }
        if let Some(path) = self.catalog_source_cycle(&provider) {
            return Err(DataFusionError::Plan(format!(
                "VGI catalog-table scan branch cycle detected: {}",
                path.join(" -> ")
            )));
        }
        Ok(provider)
    }

    async fn multi_branches(
        &self,
        state: &dyn Session,
    ) -> DFResult<Option<&Vec<BoundCatalogBranch>>> {
        let branches = self
            .branches
            .get_or_try_init(|| async {
                let connection = self.conn.clone();
                let catalog = self.catalog.clone();
                let table = self.info.clone();
                let at = self.at.clone();
                let resolved = tokio::task::spawn_blocking(move || {
                    let mut client = connection.connect()?;
                    let attached = connection.attach(&mut client, &catalog)?;
                    client
                        .table_scan_branches(&attached, &table, at.as_ref())
                        .map_err(to_df)
                })
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))??;
                let (kind, message) = match resolved.resolution {
                    vgi_client::ScanBranchesResolution::BranchesRpc => (
                        "catalog.rpc.scan_branches",
                        "method=catalog_table_scan_branches_get",
                    ),
                    vgi_client::ScanBranchesResolution::LegacyFallbackAfterProbe => (
                        "catalog.rpc.scan_branches.fallback",
                        "catalog_table_scan_branches_get unsupported; method=catalog_table_scan_function_get",
                    ),
                    vgi_client::ScanBranchesResolution::LegacyCached => (
                        "catalog.rpc.scan_branches.legacy",
                        "method=catalog_table_scan_function_get",
                    ),
                };
                let mut event = crate::VgiEvent::new(kind);
                event.catalog = Some(self.mount_alias.clone());
                event.function = Some(format!("{}.{}", self.schema_name, self.info.name));
                event.message = Some(message.to_string());
                self.conn.runtime().emit(event);
                self.conn.runtime().set_table_branches(
                    &self.mount_alias,
                    &self.schema_name,
                    &self.info.name,
                    resolved.clone(),
                );

                let needs_union = resolved.branches.len() > 1
                    || resolved.branches.iter().any(|branch| {
                        branch.branch_filter.is_some()
                            || branch.writable
                            || branch.source_table.is_some()
                            || branch.format_name.is_some()
                    });
                if !needs_union {
                    return Ok(None);
                }
                if self.at.is_some() {
                    return Err(DataFusionError::Plan(format!(
                        "AT (...) clauses are not supported on multi-branch VGI table `{}.{}`",
                        self.schema_name, self.info.name
                    )));
                }
                // `required_extensions` names DuckDB host extensions. It is
                // retained in diagnostics, but is not itself a DataFusion
                // capability check: catalog-table arms prove support by
                // resolving their attached provider, and format arms prove it
                // through `get_file_format_factory` in `bind_native_format`.
                let mut bound = Vec::with_capacity(resolved.branches.len());
                for (index, branch) in resolved.branches.into_iter().enumerate() {
                    let (provider, null_string): (Arc<dyn TableProvider>, Option<String>) =
                        if let Some(spec) = catalog_table_spec(&branch)? {
                            (
                                self.bind_catalog_table_source(state, spec, index).await?,
                                None,
                            )
                        } else if let Some(spec) = native_format_spec(&branch)? {
                            let null_string = exact_null_string(&spec.options)?;
                            (
                                self.bind_native_format(state, spec, index).await?,
                                null_string,
                            )
                        } else if branch.function_name.is_empty() {
                            return Err(DataFusionError::NotImplemented(format!(
                                "VGI scan branch {index} for `{}.{}` names no supported provider",
                                self.schema_name, self.info.name
                            )));
                        } else {
                            (
                                VgiTableProvider::bind_catalog_branch(
                                    self.conn.clone(),
                                    &self.catalog,
                                    &self.schema_name,
                                    &branch.function_name,
                                    branch.arguments.clone(),
                                )
                                .await?,
                                None,
                            )
                        };
                    bound.push(BoundCatalogBranch {
                        info: branch,
                        provider,
                        null_string,
                    });
                }
                Ok(Some(bound))
            })
            .await?;
        Ok(branches.as_ref())
    }

    fn reconcile_branch(&self, input: Arc<dyn ExecutionPlan>) -> DFResult<Arc<dyn ExecutionPlan>> {
        let raw_schema = input.schema();
        for raw in raw_schema.fields() {
            if !self
                .physical_schema
                .fields()
                .iter()
                .any(|canonical| canonical.name().eq_ignore_ascii_case(raw.name()))
            {
                return Err(DataFusionError::Plan(format!(
                    "VGI branch returned column {:?} which is not in table `{}.{}`",
                    raw.name(),
                    self.schema_name,
                    self.info.name
                )));
            }
        }

        let mut expressions = Vec::with_capacity(self.physical_schema.fields().len());
        for canonical in self.physical_schema.fields() {
            let matches = raw_schema
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, raw)| raw.name().eq_ignore_ascii_case(canonical.name()))
                .collect::<Vec<_>>();
            let expression = match matches.as_slice() {
                [] => Arc::new(PhysicalLiteral::new(ScalarValue::try_new_null(
                    canonical.data_type(),
                )?)) as Arc<dyn datafusion::physical_expr::PhysicalExpr>,
                [(index, raw)] => {
                    let column = Arc::new(PhysicalColumn::new(raw.name(), *index))
                        as Arc<dyn datafusion::physical_expr::PhysicalExpr>;
                    physical_cast(column, raw_schema.as_ref(), canonical.data_type().clone())?
                }
                _ => {
                    return Err(DataFusionError::Plan(format!(
                        "VGI branch returned duplicate case-insensitive column {:?}",
                        canonical.name()
                    )));
                }
            };
            expressions.push(ProjectionExpr {
                expr: expression,
                alias: canonical.name().clone(),
            });
        }
        Ok(Arc::new(ProjectionExec::try_new_with_schema_metadata(
            expressions,
            input,
            &self.physical_schema,
        )?))
    }

    fn apply_native_null_string(
        &self,
        branch: &BoundCatalogBranch,
        input: Arc<dyn ExecutionPlan>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let Some(null_string) = branch.null_string.as_ref() else {
            return Ok(input);
        };
        let schema = input.schema();
        let mut expressions = Vec::with_capacity(schema.fields().len());
        for (index, field) in schema.fields().iter().enumerate() {
            let column = Arc::new(PhysicalColumn::new(field.name(), index))
                as Arc<dyn datafusion::physical_expr::PhysicalExpr>;
            let marker = match field.data_type() {
                DataType::Utf8 => Some(ScalarValue::Utf8(Some(null_string.clone()))),
                DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(Some(null_string.clone()))),
                DataType::Utf8View => Some(ScalarValue::Utf8View(Some(null_string.clone()))),
                _ => None,
            };
            let expr = if let Some(marker) = marker {
                let equals = physical_binary(
                    Arc::clone(&column),
                    Operator::Eq,
                    Arc::new(PhysicalLiteral::new(marker)),
                    schema.as_ref(),
                )?;
                physical_case(
                    None,
                    vec![(
                        equals,
                        Arc::new(PhysicalLiteral::new(ScalarValue::try_new_null(
                            field.data_type(),
                        )?)),
                    )],
                    Some(column),
                )?
            } else {
                column
            };
            expressions.push(ProjectionExpr {
                expr,
                alias: field.name().clone(),
            });
        }
        Ok(Arc::new(ProjectionExec::try_new_with_schema_metadata(
            expressions,
            input,
            &schema,
        )?))
    }

    fn apply_branch_filter(
        &self,
        state: &dyn Session,
        branch: &BoundCatalogBranch,
        input: Arc<dyn ExecutionPlan>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let Some(sql) = branch
            .info
            .branch_filter
            .as_deref()
            .filter(|sql| !sql.trim().is_empty())
        else {
            return Ok(input);
        };
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "VGI branch filters require DataFusion SessionState SQL expression support"
                        .to_string(),
                )
            })?;
        let schema = DFSchema::try_from(self.physical_schema.as_ref().clone())?;
        let logical = session_state.create_logical_expr(sql, &schema)?;

        // A branch filter is a source-scope contract. Do not let a missing
        // column become a NULL-filled canonical and silently change it into an
        // always-false predicate.
        let mut columns = HashSet::new();
        expr_to_columns(&logical, &mut columns)?;
        for column in columns {
            if !branch
                .provider
                .schema()
                .fields()
                .iter()
                .any(|field| field.name().eq_ignore_ascii_case(&column.name))
            {
                let source = branch
                    .info
                    .source_table
                    .as_deref()
                    .map(|table| {
                        format!(
                            "catalog table {}.{}.{}",
                            branch.info.source_catalog.as_deref().unwrap_or("?"),
                            branch.info.source_schema.as_deref().unwrap_or("?"),
                            table
                        )
                    })
                    .unwrap_or_else(|| format!("function {:?}", branch.info.function_name));
                return Err(DataFusionError::Plan(format!(
                    "VGI branch filter references column {:?} not exposed by branch {source}",
                    column.name
                )));
            }
        }
        let predicate = state.create_physical_expr(logical, &schema)?;
        Ok(Arc::new(FilterExec::try_new(predicate, input)?))
    }

    async fn scan_multi_branches(
        &self,
        state: &dyn Session,
        branches: &[BoundCatalogBranch],
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let mut inputs = Vec::with_capacity(branches.len());
        for branch in branches {
            // Re-check after every provider's branch cache has had a chance to
            // initialize. This catches cycles whose arms were first resolved
            // concurrently, without treating independent concurrent scans as
            // recursion.
            if let Some(path) = self.catalog_source_cycle(&branch.provider) {
                return Err(DataFusionError::Plan(format!(
                    "VGI catalog-table scan branch cycle detected: {}",
                    path.join(" -> ")
                )));
            }
            // Send only predicates whose columns exist on this raw branch.
            // The outer provider is Inexact, so DataFusion still rechecks
            // every query predicate after reconciliation and union. Native
            // Parquet providers can nevertheless use these hints for row-group
            // pruning, while a branch missing a canonical column never sees an
            // invalid predicate.
            let raw_names = branch
                .provider
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().to_ascii_lowercase())
                .collect::<HashSet<_>>();
            let branch_filters = filters
                .iter()
                .filter(|filter| {
                    let mut columns = HashSet::new();
                    expr_to_columns(filter, &mut columns).is_ok()
                        && columns
                            .iter()
                            .all(|column| raw_names.contains(&column.name.to_ascii_lowercase()))
                })
                .cloned()
                .collect::<Vec<_>>();
            let raw = branch
                .provider
                .scan(state, None, &branch_filters, None)
                .await?;
            let raw = self.apply_native_null_string(branch, raw)?;
            let reconciled = self.reconcile_branch(raw)?;
            inputs.push(self.apply_branch_filter(state, branch, reconciled)?);
        }
        let mut plan = UnionExec::try_new(inputs)?;

        if self.has_generated_columns() {
            return self.generated_projection(state, projection, plan);
        }
        if let Some(indices) = projection {
            let projected_schema = self.output_schema.project(indices)?;
            let expressions: Vec<ProjectionExpr> = indices
                .iter()
                .map(|index| ProjectionExpr {
                    expr: Arc::new(PhysicalColumn::new(
                        self.output_schema.field(*index).name(),
                        *index,
                    )),
                    alias: self.output_schema.field(*index).name().clone(),
                })
                .collect();
            plan = Arc::new(ProjectionExec::try_new_with_schema_metadata(
                expressions,
                plan,
                &projected_schema,
            )?);
        }
        if let Some(limit) = limit {
            plan = Arc::new(GlobalLimitExec::new(plan, 0, Some(limit)));
        }
        Ok(plan)
    }

    async fn column_statistics(&self) -> DFResult<Option<Arc<Statistics>>> {
        self.statistics
            .get_or_try_init(|| async {
                if !self.info.supports_column_statistics || self.at.is_some() {
                    return Ok(None);
                }
                let raw = if let Some(inline) = self
                    .info
                    .column_statistics
                    .as_ref()
                    .filter(|value| !value.0.is_empty())
                {
                    vgi_protocol::ipc::read_batch(&inline.0).map_err(to_df)?
                } else {
                    let connection = self.conn.clone();
                    let catalog = self.catalog.clone();
                    let schema = self.info.schema_name.clone();
                    let table = self.info.name.clone();
                    tokio::task::spawn_blocking(move || {
                        let mut client = connection.connect()?;
                        let attached = connection.attach(&mut client, &catalog)?;
                        client
                            .table_column_statistics(&attached, &schema, &table)
                            .map_err(to_df)
                    })
                    .await
                    .map_err(|error| DataFusionError::External(Box::new(error)))??
                };
                Ok(Some(Arc::new(crate::statistics_for_catalog_table(
                    &self.output_schema,
                    &raw,
                    self.info.cardinality_estimate.0,
                    self.info.cardinality_max.0,
                    None,
                ))))
            })
            .await
            .cloned()
    }

    async fn filters_prune_table(&self, state: &dyn Session, filters: &[Expr]) -> DFResult<bool> {
        let Some(statistics) = self.column_statistics().await? else {
            return Ok(false);
        };
        Ok(crate::filters_prune_statistics(
            state,
            &self.output_schema,
            statistics,
            filters,
        ))
    }

    fn has_generated_columns(&self) -> bool {
        self.physical_schema.fields().len() != self.output_schema.fields().len()
    }

    /// Build the catalog-visible projection above a physical VGI scan.
    ///
    /// Generated expressions are catalog metadata, while the backing function
    /// emits only stored columns. DataFusion already has the SQL-expression and
    /// physical-projection APIs needed to bridge those two schemas, so no
    /// engine-specific logical node is necessary here.
    fn generated_projection(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        input: Arc<dyn ExecutionPlan>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "VGI generated columns require DataFusion SessionState SQL expression support"
                        .to_string(),
                )
            })?;
        let physical_df_schema = DFSchema::try_from(self.physical_schema.as_ref().clone())?;
        let selected: Vec<usize> = projection
            .cloned()
            .unwrap_or_else(|| (0..self.output_schema.fields().len()).collect());
        let projected_schema = self.output_schema.project(&selected)?;
        let mut expressions = Vec::with_capacity(selected.len());

        for index in selected {
            let field = self.output_schema.field(index);
            let expr = if let Some(sql) = field.metadata().get("generated_expression") {
                let logical = session_state
                    .create_logical_expr(sql, &physical_df_schema)?
                    .cast_to(field.data_type(), &physical_df_schema)?;
                state.create_physical_expr(logical, &physical_df_schema)?
            } else {
                let physical_index = self.physical_schema.index_of(field.name())?;
                Arc::new(PhysicalColumn::new(field.name(), physical_index))
            };
            expressions.push(ProjectionExpr {
                expr,
                alias: field.name().clone(),
            });
        }

        Ok(Arc::new(ProjectionExec::try_new_with_schema_metadata(
            expressions,
            input,
            &projected_schema,
        )?))
    }

    /// A generated predicate cannot be sent to a function that has no such
    /// physical input column. Physical-only predicates remain useful worker
    /// hints; this provider reports every filter as `Inexact`, so DataFusion
    /// still evaluates the predicate above the generated projection.
    fn physical_filters(&self, filters: &[Expr]) -> Vec<Expr> {
        let physical_names = self
            .physical_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<HashSet<_>>();
        filters
            .iter()
            .filter(|filter| {
                let mut columns = HashSet::new();
                expr_to_columns(filter, &mut columns).is_ok()
                    && columns
                        .iter()
                        .all(|column| physical_names.contains(column.name.as_str()))
            })
            .cloned()
            .collect()
    }
}

#[async_trait]
impl TableProvider for VgiCatalogTableProvider {
    fn schema(&self) -> datafusion::arrow::datatypes::SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(&self.constraints)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // The scan bind discovers exactness later. Inexact is safe here: VGI
        // still receives supported predicates and DataFusion always rechecks.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        self.enforce_required_filters(filters)?;
        if self.filters_prune_table(state, filters).await? {
            let schema = match projection {
                Some(indices) => Arc::new(self.output_schema.project(indices)?),
                None => Arc::clone(&self.output_schema),
            };
            return Ok(Arc::new(EmptyExec::new(schema)));
        }
        if let Some(branches) = self.multi_branches(state).await? {
            return self
                .scan_multi_branches(state, branches, projection, filters, limit)
                .await;
        }
        let bound = self.bound().await?;
        if !self.has_generated_columns() {
            return bound.scan(state, projection, filters, limit).await;
        }

        // Fetch every stored column because an arbitrary generated expression
        // may depend on any of them. Do not push LIMIT below the generated
        // projection: an outer predicate may remove rows after generation.
        let physical_filters = self.physical_filters(filters);
        let input = bound.scan(state, None, &physical_filters, None).await?;
        self.generated_projection(state, projection, input)
    }
}

/// Discovery metadata needed to choose the correct execution protocol for a
/// callable table function.
#[derive(Debug, Clone)]
pub(crate) struct TableFunctionMetadata {
    pub specs: vgi_client::ArgSpecs,
    pub buffered: bool,
    /// Buffered whole-input caching is valid only when the worker declares
    /// that its result depends on the input multiset rather than row order.
    pub sink_order_dependent: bool,
    pub input_from_args: bool,
    /// Per-input-batch memoization is valid only for the parallel streaming
    /// map shape. A serial worker or FINALIZE callback may carry state across
    /// batches, so identical batches are not independent cache units.
    pub stream_cache_eligible: bool,
}

/// One VGI schema.
#[derive(Debug)]
pub struct VgiSchemaProvider {
    conn: VgiConnection,
    mount_alias: String,
    catalog: String,
    schema_name: String,
    /// Per-function execution shape and argument declarations.
    table_functions: HashMap<String, TableFunctionMetadata>,
    /// Every name the schema advertises — catalog **tables** and table
    /// **functions**, resolved once at attach.
    ///
    /// Both live in one namespace because SQL has one: `ex.data.t` does not say
    /// which kind `t` is, and the worker guarantees the names are distinct
    /// within a schema.
    names: Vec<String>,
    /// Actual relations exposed through DataFusion metadata. Table functions
    /// remain resolvable through `table()` for the useful bare-call error, but
    /// are routines rather than rows in `SHOW TABLES`.
    relation_names: Vec<String>,
    /// Which of those names are catalog tables. They bind differently: a table
    /// is scanned through the function the worker nominates
    /// (`catalog_table_scan_function_get`), with the worker's own arguments.
    tables: HashMap<String, vgi_client::TableInfo>,
    /// Scalar functions in this schema, with what the worker declares about
    /// their parameters. They are not tables and never appear in
    /// [`Self::table_names`]; they are published into DataFusion's separate
    /// function registry at attach time.
    ///
    /// The specs travel with the name because a call cannot be built without
    /// them: a const parameter belongs in the bind, not the input batch.
    scalars: Vec<(String, Vec<vgi_client::ArgSpecs>, Volatility)>,
    /// Aggregate functions in this schema, published into DataFusion's
    /// aggregate registry at attach time.
    aggregates: Vec<(
        String,
        vgi_client::ArgSpecs,
        Volatility,
        bool,
        Vec<vgi_client::SecretLookupRequest>,
    )>,
    /// Scalar and table macro declarations retained for metadata inspection.
    metadata_macros: Vec<vgi_client::dtos::MacroInfo>,
    /// SQL views declared by the worker. Their definitions are planned after
    /// the catalog and its functions have been registered with DataFusion.
    views: HashMap<String, vgi_client::dtos::ViewInfo>,
    /// Complete worker declarations retained for metadata diagnostics. The
    /// execution-specific collections above are derived from these same rows.
    functions: Vec<vgi_client::dtos::FunctionInfo>,
    /// Bind results, memoised. An `Err` records a function that will not bind
    /// bare — most fixture functions take arguments — so it is not retried,
    /// and the worker's own reason is kept to report at plan time.
    bound: Mutex<HashMap<String, CachedTable>>,
    /// Historical binds are isolated from the current-table cache because a
    /// past version may expose a different schema.
    versioned: Mutex<HashMap<(String, String, String), CachedTable>>,
}

impl VgiSchemaProvider {
    /// List one schema's tables and table functions. Two RPCs; no binds.
    pub async fn discover(
        conn: VgiConnection,
        mount_alias: &str,
        catalog: &str,
        schema_name: &str,
    ) -> DFResult<Arc<Self>> {
        let (c, cat, sch) = (conn.clone(), catalog.to_string(), schema_name.to_string());
        let (tables, table_functions, scalars, aggregates, metadata_macros, views, functions) =
            tokio::task::spawn_blocking(move || {
                let mut client = c.connect()?;
                let attached = c.attach(&mut client, &cat)?;
                let tables = client.tables(&attached, &sch).map_err(to_df)?;
                let table_infos = client
                    .functions(&attached, &sch, vgi_client::FunctionKind::Table)
                    .map_err(to_df)?;
                // `function_type` distinguishes the three table shapes that share
                // one listing filter; the buffered one needs the Sink+Source
                // protocol rather than a streaming exchange.
                //
                // The wire carries the enum's *member name* — `TABLE_BUFFERING`,
                // not the lowercase `table_buffering` value — the same convention
                // that governs `FunctionKind`. Matched case-insensitively so a
                // worker that sends either spelling is understood.
                let table_functions = table_infos
                    .iter()
                    .map(|f| {
                        let specs = vgi_client::ArgSpecs::parse(&f.arguments.0).map_err(to_df)?;
                        let metadata = TableFunctionMetadata {
                            buffered: f.function_type.0.eq_ignore_ascii_case("table_buffering"),
                            sink_order_dependent: f.sink_order_dependent,
                            input_from_args: f.input_from_args,
                            stream_cache_eligible: f.max_workers != Some(1) && !f.has_finalize,
                            specs,
                        };
                        Ok::<_, DataFusionError>((f.name.clone(), metadata))
                    })
                    .collect::<Result<HashMap<_, _>, _>>()?;
                let scalar_infos = client
                    .functions(&attached, &sch, vgi_client::FunctionKind::Scalar)
                    .map_err(to_df)?;
                let aggregate_infos = client
                    .functions(&attached, &sch, vgi_client::FunctionKind::Aggregate)
                    .map_err(to_df)?;
                let aggregates = aggregate_infos
                    .iter()
                    .map(|f| {
                        let specs = vgi_client::ArgSpecs::parse(&f.arguments.0).map_err(to_df)?;
                        let secrets = f
                            .required_secrets
                            .iter()
                            .map(|secret| vgi_client::SecretLookupRequest {
                                secret_type: secret.secret_type.clone(),
                                scope: secret.scope.clone(),
                                name: secret.secret_name.clone(),
                            })
                            .collect();
                        Ok::<_, DataFusionError>((
                            f.name.clone(),
                            specs,
                            volatility(f.stability.as_ref().map(|v| v.0.as_str())),
                            f.supports_window,
                            secrets,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let scalar_macro_infos = client
                    .macros(&attached, &sch, vgi_client::MacroKind::Scalar)
                    .map_err(to_df)?;
                let table_macro_infos = client
                    .macros(&attached, &sch, vgi_client::MacroKind::Table)
                    .map_err(to_df)?;
                let metadata_macros = scalar_macro_infos
                    .into_iter()
                    .chain(table_macro_infos)
                    .collect::<Vec<_>>();
                let views = client
                    .views(&attached, &sch)
                    .map_err(to_df)?
                    .into_iter()
                    .map(|info| (info.name.clone(), info))
                    .collect::<HashMap<_, _>>();
                // DataFusion registers one scalar UDF per SQL name, whereas
                // VGI advertises one FunctionInfo per overload. Preserve the
                // complete overload set so the UDF can choose the right const
                // layout for each call instead of whichever overload happened
                // to be registered first.
                let mut scalar_overloads: HashMap<String, (Vec<vgi_client::ArgSpecs>, Volatility)> =
                    HashMap::new();
                for f in &scalar_infos {
                    let specs = vgi_client::ArgSpecs::parse(&f.arguments.0).map_err(to_df)?;
                    let declared = volatility(f.stability.as_ref().map(|v| v.0.as_str()));
                    let entry = scalar_overloads
                        .entry(f.name.clone())
                        .or_insert_with(|| (Vec::new(), declared));
                    entry.0.push(specs);
                    entry.1 = most_volatile(entry.1, declared);
                }
                let scalars = scalar_overloads
                    .into_iter()
                    .map(|(name, (overloads, volatility))| (name, overloads, volatility))
                    .collect::<Vec<_>>();
                let functions = table_infos
                    .into_iter()
                    .chain(scalar_infos)
                    .chain(aggregate_infos)
                    .collect::<Vec<_>>();
                Ok::<_, DataFusionError>((
                    tables,
                    table_functions,
                    scalars,
                    aggregates,
                    metadata_macros,
                    views,
                    functions,
                ))
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let tables: HashMap<String, vgi_client::TableInfo> =
            tables.into_iter().map(|t| (t.name.clone(), t)).collect();
        let mut relation_names: Vec<String> = tables.keys().cloned().collect();
        relation_names.extend(
            views
                .keys()
                .filter(|name| !tables.contains_key(*name))
                .cloned(),
        );
        let mut names = relation_names.clone();
        names.extend(
            table_functions
                .keys()
                .filter(|name| !tables.contains_key(*name))
                .cloned(),
        );

        Ok(Arc::new(Self {
            conn,
            mount_alias: mount_alias.to_string(),
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            names,
            relation_names,
            tables,
            scalars,
            aggregates,
            metadata_macros,
            views,
            functions,
            table_functions,
            bound: Mutex::new(HashMap::new()),
            versioned: Mutex::new(HashMap::new()),
        }))
    }

    /// Names that are catalog tables, not functions.
    pub fn table_names_only(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    /// Scalar functions this schema advertises, with their parameter specs.
    pub fn scalars(&self) -> &[(String, Vec<vgi_client::ArgSpecs>, Volatility)] {
        &self.scalars
    }

    /// Aggregate functions this schema advertises.
    pub fn aggregates(
        &self,
    ) -> &[(
        String,
        vgi_client::ArgSpecs,
        Volatility,
        bool,
        Vec<vgi_client::SecretLookupRequest>,
    )] {
        &self.aggregates
    }

    /// Views declared by this schema.
    pub(crate) fn views(&self) -> impl Iterator<Item = (&String, &vgi_client::dtos::ViewInfo)> {
        self.views.iter()
    }

    /// Complete function declarations used to publish SQL metadata views.
    pub(crate) fn functions(&self) -> &[vgi_client::dtos::FunctionInfo] {
        &self.functions
    }

    pub(crate) fn metadata_macros(&self) -> &[vgi_client::dtos::MacroInfo] {
        &self.metadata_macros
    }

    pub(crate) fn tables(&self) -> impl Iterator<Item = &vgi_client::dtos::TableInfo> {
        self.tables.values()
    }

    /// Install a planned view (or its durable planning error) in the same lazy
    /// table cache used for remote tables.
    pub(crate) fn install_view(&self, name: &str, table: CachedTable) {
        if let Ok(mut cache) = self.bound.lock() {
            cache.insert(name.to_string(), table);
        }
    }

    /// Discovery metadata for a callable table function.
    pub(crate) fn table_function_metadata(&self, name: &str) -> Option<&TableFunctionMetadata> {
        self.table_functions.get(name)
    }

    /// Names that are callable table functions (excluding tables and views).
    pub(crate) fn table_function_names(&self) -> impl Iterator<Item = &String> {
        self.table_functions.keys()
    }

    /// Look up a memoised bind without holding the lock across an await.
    fn cached(&self, name: &str) -> Option<CachedTable> {
        self.bound.lock().ok()?.get(name).cloned()
    }

    /// Bind a catalog table at an explicit VGI time-travel coordinate.
    pub(crate) async fn table_at(
        &self,
        name: &str,
        at: vgi_client::At,
    ) -> DFResult<Arc<dyn TableProvider>> {
        if !self.tables.contains_key(name) {
            return Err(DataFusionError::Plan(format!(
                "VGI time travel is only supported for catalog tables; `{name}` is not one"
            )));
        }
        let key = (name.to_string(), at.unit.clone(), at.value.clone());
        if let Some(hit) = self
            .versioned
            .lock()
            .ok()
            .and_then(|cache| cache.get(&key).cloned())
        {
            return hit.map_err(bind_failed(name));
        }

        // Discovery describes the current table. Ask for TableInfo again at
        // this coordinate so schema evolution is visible during planning.
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema = self.schema_name.clone();
        let table = name.to_string();
        let lookup_at = at.clone();
        let info = tokio::task::spawn_blocking(move || {
            let mut client = conn.connect()?;
            let attached = conn.attach(&mut client, &catalog)?;
            client
                .table_get(&attached, &schema, &table, Some(&lookup_at))
                .map_err(to_df)?
                .ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "VGI catalog table `{schema}.{table}` was not found at {} {}",
                        lookup_at.unit, lookup_at.value
                    ))
                })
        })
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))??;

        let bound = VgiCatalogTableProvider::new(
            self.conn.clone(),
            &self.mount_alias,
            &self.catalog,
            &self.schema_name,
            info,
            Some(at),
        )
        .map(|provider| provider as Arc<dyn TableProvider>)
        .map_err(|error| error.to_string());
        if let Ok(mut cache) = self.versioned.lock() {
            cache.insert(key, bound.clone());
        }
        bound.map_err(bind_failed(name))
    }
}

fn volatility(value: Option<&str>) -> Volatility {
    match value {
        Some(value) if value.eq_ignore_ascii_case("VOLATILE") => Volatility::Volatile,
        Some(value) if value.eq_ignore_ascii_case("CONSISTENT_WITHIN_QUERY") => Volatility::Stable,
        _ => Volatility::Immutable,
    }
}

fn most_volatile(left: Volatility, right: Volatility) -> Volatility {
    match (left, right) {
        (Volatility::Volatile, _) | (_, Volatility::Volatile) => Volatility::Volatile,
        (Volatility::Stable, _) | (_, Volatility::Stable) => Volatility::Stable,
        _ => Volatility::Immutable,
    }
}

#[async_trait]
impl SchemaProvider for VgiSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.relation_names.clone()
    }

    async fn table_type(&self, name: &str) -> DFResult<Option<TableType>> {
        if self.tables.contains_key(name) {
            Ok(Some(TableType::Base))
        } else if self.views.contains_key(name) {
            Ok(Some(TableType::View))
        } else {
            Ok(None)
        }
    }

    async fn table(&self, name: &str) -> DFResult<Option<Arc<dyn TableProvider>>> {
        if !self.names.iter().any(|n| n == name) {
            return Ok(None);
        }
        if let Some(hit) = self.cached(name) {
            return hit.map(Some).map_err(bind_failed(name));
        }
        if self.views.contains_key(name) {
            return Err(DataFusionError::Plan(format!(
                "VGI view `{name}` was discovered but has not been planned"
            )));
        }

        // Two callers racing the same cold name both bind; the loser's result
        // is dropped. That is cheaper and simpler than holding a lock across
        // the await, and binds are idempotent.
        let bound = match self.tables.get(name) {
            Some(info) => VgiCatalogTableProvider::new(
                self.conn.clone(),
                &self.mount_alias,
                &self.catalog,
                &self.schema_name,
                info.clone(),
                None,
            )
            .map(|provider| provider as Arc<dyn TableProvider>),
            None => {
                VgiTableProvider::bind(self.conn.clone(), &self.catalog, &self.schema_name, name)
                    .await
                    .map(|provider| provider as Arc<dyn TableProvider>)
            }
        }
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
    schemas: HashMap<String, Arc<VgiSchemaProvider>>,
    comment: Option<String>,
    tags: Vec<(String, String)>,
    default_schema: String,
    resolved_data_version: Option<String>,
    resolved_implementation_version: Option<String>,
    schema_infos: Vec<vgi_client::dtos::SchemaInfo>,
    /// The prefix the worker asked for on globally-published functions
    /// (`global_function_prefix`), empty when it asked for none.
    ///
    /// A worker that already publishes globals to DuckDB has an opinion about
    /// what they should be called; honouring it means one worker gets the same
    /// spelling on both engines, rather than each client inventing its own.
    global_function_prefix: String,
    global_functions: Vec<vgi_client::dtos::FunctionInfo>,
    settings: Vec<vgi_client::SettingSpec>,
    companion_catalogs: Vec<vgi_client::dtos::AttachCatalogInfo>,
}

impl VgiCatalogProvider {
    /// Attach a catalog and list its schemas.
    pub async fn discover(conn: VgiConnection, catalog: &str) -> DFResult<Arc<Self>> {
        Self::discover_as(conn, catalog, catalog).await
    }

    /// Attach a catalog under an explicit DataFusion alias.
    pub(crate) async fn discover_as(
        conn: VgiConnection,
        catalog: &str,
        mount_alias: &str,
    ) -> DFResult<Arc<Self>> {
        let (c, cat) = (conn.clone(), catalog.to_string());
        let (
            schema_infos,
            comment,
            tags,
            default_schema,
            resolved_data_version,
            resolved_implementation_version,
            global_function_prefix,
            global_functions,
            settings,
            companion_catalogs,
        ) = tokio::task::spawn_blocking(move || {
            let mut client = c.connect()?;
            let attached = c.attach(&mut client, &cat)?;
            let info = attached.info();
            let prefix = info.global_function_prefix.clone();
            let comment = info.comment.clone();
            let tags = info.tags.clone();
            let default_schema = info.default_schema.clone();
            let resolved_data_version = info.resolved_data_version.clone();
            let resolved_implementation_version = info.resolved_implementation_version.clone();
            let global_functions = attached.global_functions().map_err(to_df)?;
            let settings = vgi_client::decode_setting_specs(info).map_err(to_df)?;
            let companion_catalogs = attached.companion_catalogs().map_err(to_df)?;
            let schema_infos = client.schemas(&attached).map_err(to_df)?;
            Ok::<_, DataFusionError>((
                schema_infos,
                comment,
                tags,
                default_schema,
                resolved_data_version,
                resolved_implementation_version,
                prefix,
                global_functions,
                settings,
                companion_catalogs,
            ))
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let mut schemas: HashMap<String, Arc<VgiSchemaProvider>> = HashMap::new();
        for name in schema_infos.iter().map(|schema| schema.name.clone()) {
            let sp = VgiSchemaProvider::discover(conn.clone(), mount_alias, catalog, &name).await?;
            schemas.insert(name, sp);
        }
        Ok(Arc::new(Self {
            schemas,
            comment,
            tags,
            default_schema,
            resolved_data_version,
            resolved_implementation_version,
            schema_infos,
            global_function_prefix,
            global_functions,
            settings,
            companion_catalogs,
        }))
    }
}

impl VgiCatalogProvider {
    pub(crate) fn catalog_comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    pub(crate) fn catalog_tags(&self) -> &[(String, String)] {
        &self.tags
    }

    pub(crate) fn default_schema(&self) -> &str {
        &self.default_schema
    }

    pub(crate) fn resolved_data_version(&self) -> Option<&str> {
        self.resolved_data_version.as_deref()
    }

    pub(crate) fn resolved_implementation_version(&self) -> Option<&str> {
        self.resolved_implementation_version.as_deref()
    }

    pub(crate) fn schema_infos(&self) -> &[vgi_client::dtos::SchemaInfo] {
        &self.schema_infos
    }

    /// The worker's requested prefix for globally-published functions.
    pub fn global_function_prefix(&self) -> &str {
        &self.global_function_prefix
    }

    /// Function descriptors explicitly nominated for global publication.
    pub fn global_functions(&self) -> &[vgi_client::dtos::FunctionInfo] {
        &self.global_functions
    }

    /// Typed session settings declared by this attachment.
    pub fn settings(&self) -> &[vgi_client::SettingSpec] {
        &self.settings
    }

    /// Companion catalogs requested by this attachment.
    pub fn companion_catalogs(&self) -> &[vgi_client::dtos::AttachCatalogInfo] {
        &self.companion_catalogs
    }

    /// This catalog's schemas, concretely — the registration paths need more
    /// than `SchemaProvider` exposes (scalar names are not tables).
    pub fn vgi_schemas(&self) -> impl Iterator<Item = (&String, &Arc<VgiSchemaProvider>)> {
        self.schemas.iter()
    }

    pub(crate) fn functions(&self) -> impl Iterator<Item = &vgi_client::dtos::FunctionInfo> {
        self.schemas.values().flat_map(|schema| schema.functions())
    }

    pub(crate) fn metadata_macros(&self) -> impl Iterator<Item = &vgi_client::dtos::MacroInfo> {
        self.schemas
            .values()
            .flat_map(|schema| schema.metadata_macros())
    }

    pub(crate) fn tables(&self) -> impl Iterator<Item = &vgi_client::dtos::TableInfo> {
        self.schemas.values().flat_map(|schema| schema.tables())
    }

    pub(crate) fn metadata_views(&self) -> Vec<(vgi_client::dtos::ViewInfo, Vec<String>)> {
        self.schemas
            .values()
            .flat_map(|schema| {
                schema.views().map(|(name, info)| {
                    let columns = schema
                        .cached(name)
                        .and_then(Result::ok)
                        .map(|provider| {
                            provider
                                .schema()
                                .fields()
                                .iter()
                                .map(|field| field.name().clone())
                                .collect()
                        })
                        .unwrap_or_default();
                    (info.clone(), columns)
                })
            })
            .collect()
    }

    /// Resolve one historical table through its concrete VGI schema.
    pub(crate) async fn table_at(
        &self,
        schema: &str,
        table: &str,
        at: vgi_client::At,
    ) -> DFResult<Arc<dyn TableProvider>> {
        let provider = self.schemas.get(schema).ok_or_else(|| {
            DataFusionError::Plan(format!("VGI schema `{schema}` does not exist"))
        })?;
        provider.table_at(table, at).await
    }
}

impl CatalogProvider for VgiCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas
            .get(name)
            .map(|s| Arc::clone(s) as Arc<dyn SchemaProvider>)
    }
}
