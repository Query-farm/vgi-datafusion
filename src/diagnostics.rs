// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! SQL surfaces for session-scoped VGI cache and event diagnostics.

use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Int64Array, ListBuilder, MapBuilder, StringArray, StringBuilder,
    UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemTable, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::common::{plan_err, Result as DFResult, ScalarValue};
use datafusion::logical_expr::{
    create_udf, ColumnarValue, ScalarFunctionArgs, ScalarFunctionImplementation, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;

use crate::VgiRuntime;

#[derive(Debug)]
struct DiagnosticsTable {
    runtime: Arc<VgiRuntime>,
    kind: DiagnosticsKind,
}

#[derive(Debug, Clone, Copy)]
enum DiagnosticsKind {
    CacheStats,
    PlanCacheStats,
    CacheEntries,
    DuckDbCacheEntries,
    Logs,
    DuckDbLogs,
    DuckDbFunctions,
    VgiFunctionArguments,
    LogStats,
    CacheFlush,
    CacheReap,
}

impl TableFunctionImpl for DiagnosticsTable {
    fn call_with_args(&self, args: TableFunctionArgs) -> DFResult<Arc<dyn TableProvider>> {
        if !args.exprs().is_empty() {
            return plan_err!("VGI diagnostic table functions take no arguments");
        }
        let batch = match self.kind {
            DiagnosticsKind::CacheStats => cache_stats(&self.runtime)?,
            DiagnosticsKind::PlanCacheStats => plan_cache_stats(&self.runtime)?,
            DiagnosticsKind::CacheEntries => cache_entries(&self.runtime)?,
            DiagnosticsKind::DuckDbCacheEntries => duckdb_cache_entries(&self.runtime)?,
            DiagnosticsKind::Logs => logs(&self.runtime)?,
            DiagnosticsKind::DuckDbLogs => duckdb_logs(&self.runtime)?,
            DiagnosticsKind::DuckDbFunctions => duckdb_functions(&self.runtime)?,
            DiagnosticsKind::VgiFunctionArguments => vgi_function_arguments(&self.runtime)?,
            DiagnosticsKind::LogStats => log_stats(&self.runtime)?,
            DiagnosticsKind::CacheFlush => operation_result(
                "flushed",
                (self.runtime.result_cache().flush_all() + self.runtime.flush_plan_cache()) as u64,
            )?,
            DiagnosticsKind::CacheReap => {
                operation_result("removed", self.runtime.result_cache().reap() as u64)?
            }
        };
        Ok(Arc::new(MemTable::try_new(
            batch.schema(),
            vec![vec![batch]],
        )?))
    }
}

fn operation_result(name: &str, value: u64) -> DFResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::UInt64, false)]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(UInt64Array::from(vec![value]))],
    )?)
}

/// DuckDB spells physical types differently from Arrow. The shared VGI corpus
/// uses `typeof` to assert overload selection, so expose that harmless dialect
/// compatibility through DataFusion's ordinary scalar-UDF API.
#[derive(Debug, PartialEq, Eq, Hash)]
struct DuckDbTypeOf {
    signature: Signature,
}

impl DuckDbTypeOf {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for DuckDbTypeOf {
    fn name(&self) -> &str {
        "typeof"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let Some(argument) = args.args.first() else {
            return plan_err!("typeof expects one argument");
        };
        Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
            duckdb_type_name(&argument.data_type()),
        ))))
    }
}

pub(crate) fn duckdb_type_name(data_type: &DataType) -> String {
    match data_type {
        DataType::Null => "NULL".to_string(),
        DataType::Boolean => "BOOLEAN".to_string(),
        DataType::Int8 => "TINYINT".to_string(),
        DataType::Int16 => "SMALLINT".to_string(),
        DataType::Int32 => "INTEGER".to_string(),
        DataType::Int64 => "BIGINT".to_string(),
        DataType::UInt8 => "UTINYINT".to_string(),
        DataType::UInt16 => "USMALLINT".to_string(),
        DataType::UInt32 => "UINTEGER".to_string(),
        DataType::UInt64 => "UBIGINT".to_string(),
        DataType::Float16 | DataType::Float32 => "FLOAT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "VARCHAR".to_string(),
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => "BLOB".to_string(),
        DataType::Date32 | DataType::Date64 => "DATE".to_string(),
        DataType::Time32(_) | DataType::Time64(_) => "TIME".to_string(),
        DataType::Timestamp(_, timezone) if timezone.is_some() => {
            "TIMESTAMP WITH TIME ZONE".to_string()
        }
        DataType::Timestamp(_, _) => "TIMESTAMP".to_string(),
        DataType::Duration(_) | DataType::Interval(_) => "INTERVAL".to_string(),
        DataType::Decimal32(precision, scale)
        | DataType::Decimal64(precision, scale)
        | DataType::Decimal128(precision, scale)
        | DataType::Decimal256(precision, scale) => format!("DECIMAL({precision},{scale})"),
        DataType::List(field) | DataType::LargeList(field) | DataType::ListView(field) => {
            format!("{}[]", duckdb_type_name(field.data_type()))
        }
        DataType::LargeListView(field) => format!("{}[]", duckdb_type_name(field.data_type())),
        DataType::FixedSizeList(field, size) => {
            format!("{}[{size}]", duckdb_type_name(field.data_type()))
        }
        DataType::Struct(fields) => format!(
            "STRUCT({})",
            fields
                .iter()
                .map(|field| format!("{} {}", field.name(), duckdb_type_name(field.data_type())))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataType::Map(field, _) => match field.data_type() {
            DataType::Struct(fields) if fields.len() == 2 => format!(
                "MAP({}, {})",
                duckdb_type_name(fields[0].data_type()),
                duckdb_type_name(fields[1].data_type())
            ),
            _ => format!("MAP({})", duckdb_type_name(field.data_type())),
        },
        DataType::Union(fields, _) => format!(
            "UNION({})",
            fields
                .iter()
                .map(|(_, field)| format!(
                    "{} {}",
                    field.name(),
                    duckdb_type_name(field.data_type())
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataType::Dictionary(_, value) => duckdb_type_name(value),
        other => format!("{other}"),
    }
}

fn plan_cache_stats(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let stats = runtime.plan_cache_stats();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hits", DataType::UInt64, false),
        Field::new("misses", DataType::UInt64, false),
        Field::new("inserts", DataType::UInt64, false),
        Field::new("entries", DataType::UInt64, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        [
            stats.hits,
            stats.misses,
            stats.inserts,
            stats.entries as u64,
        ]
        .into_iter()
        .map(|value| Arc::new(UInt64Array::from(vec![value])) as ArrayRef)
        .collect(),
    )?)
}

fn cache_stats(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let stats = runtime.result_cache().stats();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hits", DataType::UInt64, false),
        Field::new("misses", DataType::UInt64, false),
        Field::new("inserts", DataType::UInt64, false),
        Field::new("evictions_lru", DataType::UInt64, false),
        Field::new("evictions_ttl", DataType::UInt64, false),
        Field::new("refusals", DataType::UInt64, false),
        Field::new("capture_aborts", DataType::UInt64, false),
        Field::new("revalidations", DataType::UInt64, false),
        Field::new("stale_serves", DataType::UInt64, false),
        Field::new("entries", DataType::UInt64, false),
        Field::new("total_bytes", DataType::UInt64, false),
    ]));
    let values = [
        stats.hits,
        stats.misses,
        stats.inserts,
        stats.evictions_lru,
        stats.evictions_ttl,
        stats.refusals,
        stats.capture_aborts,
        stats.revalidations,
        stats.stale_serves,
        stats.entries as u64,
        stats.total_bytes as u64,
    ];
    Ok(RecordBatch::try_new(
        schema,
        values
            .into_iter()
            .map(|value| Arc::new(UInt64Array::from(vec![value])) as ArrayRef)
            .collect(),
    )?)
}

fn cache_entries(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let entries = runtime.result_cache().entries();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("catalog", DataType::Utf8, false),
        Field::new("function", DataType::Utf8, false),
        Field::new("rows", DataType::UInt64, false),
        Field::new("bytes", DataType::UInt64, false),
        Field::new("age_ms", DataType::UInt64, false),
        Field::new("stale", DataType::Boolean, false),
        Field::new("hits", DataType::UInt64, false),
        Field::new("revalidatable", DataType::Boolean, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                entries.iter().map(|entry| entry.key_fingerprint.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                entries.iter().map(|entry| entry.catalog.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                entries.iter().map(|entry| entry.function.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.rows as u64),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.bytes as u64),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.age.as_millis() as u64),
            )),
            Arc::new(BooleanArray::from(
                entries.iter().map(|entry| entry.stale).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.hits),
            )),
            Arc::new(BooleanArray::from(
                entries
                    .iter()
                    .map(|entry| entry.revalidatable)
                    .collect::<Vec<_>>(),
            )),
        ],
    )?)
}

/// Compatibility shape for the DuckDB extension's `vgi_result_cache()`.
///
/// Keep the DataFusion-native diagnostic above stable and expose only fields
/// this cache actually owns. In particular, disk tier and per-substream fields
/// are not invented here: queries asking for them must continue to identify a
/// real incomplete feature.
fn duckdb_cache_entries(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let entries = runtime.result_cache().entries();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key_hash", DataType::Utf8, false),
        Field::new("catalog", DataType::Utf8, false),
        Field::new("function", DataType::Utf8, false),
        Field::new("num_rows", DataType::UInt64, false),
        Field::new("bytes", DataType::UInt64, false),
        Field::new("age_ms", DataType::UInt64, false),
        Field::new("stale", DataType::Boolean, false),
        Field::new("hits", DataType::UInt64, false),
        Field::new("revalidatable", DataType::Boolean, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                entries.iter().map(|entry| entry.key_fingerprint.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                entries.iter().map(|entry| entry.catalog.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(entries.iter().map(|entry| {
                entry
                    .function
                    .rsplit_once('.')
                    .map(|(_, name)| name)
                    .unwrap_or(&entry.function)
            }))),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.rows as u64),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.bytes as u64),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.age.as_millis() as u64),
            )),
            Arc::new(BooleanArray::from(
                entries.iter().map(|entry| entry.stale).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from_iter_values(
                entries.iter().map(|entry| entry.hits),
            )),
            Arc::new(BooleanArray::from(
                entries
                    .iter()
                    .map(|entry| entry.revalidatable)
                    .collect::<Vec<_>>(),
            )),
        ],
    )?)
}

fn logs(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let events = runtime.events();
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp_ms", DataType::Int64, false),
        Field::new("event", DataType::Utf8, false),
        Field::new("catalog", DataType::Utf8, true),
        Field::new("function", DataType::Utf8, true),
        Field::new("split", DataType::Utf8, true),
        Field::new("duration_ms", DataType::UInt64, true),
        Field::new("message", DataType::Utf8, true),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(events.iter().map(|event| {
                event
                    .timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64
            }))),
            Arc::new(StringArray::from_iter_values(
                events.iter().map(|event| event.kind.as_str()),
            )),
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.catalog.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.function.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.split.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                events
                    .iter()
                    .map(|event| event.duration.map(|duration| duration.as_millis() as u64))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.message.as_deref())
                    .collect::<Vec<_>>(),
            )),
        ],
    )?)
}

/// Compatibility shape for DuckDB's `duckdb_logs()` table function.
///
/// These rows are backed by the adapter's real structured event history. The
/// message serializer keeps the event name and its useful dimensions in the
/// text column used by the shared SQL corpus. It does not synthesize worker
/// log records or cache tiers that this integration did not observe.
fn duckdb_logs(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let events = runtime.events();
    let schema = Arc::new(Schema::new(vec![
        Field::new("type", DataType::Utf8, false),
        Field::new("message", DataType::Utf8, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(events.iter().map(|_| "VGI"))),
            Arc::new(StringArray::from_iter_values(
                events.iter().map(duckdb_log_message),
            )),
        ],
    )?)
}

fn duckdb_log_message(event: &crate::VgiEvent) -> String {
    let kind = match event.kind.as_str() {
        "cache.hit" => "result_cache.hit",
        "cache.miss" => "result_cache.miss",
        "cache.store" => "result_cache.store",
        "cache.refused" | "cache.capture_aborted" => "result_cache.abort",
        "cache.revalidated" => "result_cache.revalidate",
        other => other,
    };
    let mut fields = vec![kind.to_string()];
    if event.kind == "cache.hit" || event.kind == "cache.store" {
        fields.push("tier=memory".to_string());
    }
    if event.kind == "cache.revalidated" {
        fields.push("outcome=not_modified".to_string());
    }
    if let Some(catalog) = &event.catalog {
        fields.push(format!("catalog={catalog}"));
    }
    if let Some(function) = &event.function {
        fields.push(format!("function={function}"));
    }
    if let Some(split) = &event.split {
        fields.push(format!("split={split}"));
    }
    if let Some(duration) = event.duration {
        fields.push(format!("duration_ms={}", duration.as_millis()));
    }
    if let Some(message) = &event.message {
        if event.kind == "cache.refused" || event.kind == "cache.capture_aborted" {
            fields.push(format!("reason={message}"));
        } else {
            fields.push(message.clone());
        }
    }
    fields.join(" ")
}

#[derive(Debug)]
struct DuckDbFunctionRow {
    database_name: String,
    schema_name: String,
    function_name: String,
    function_type: String,
    parameters: Vec<String>,
    parameter_types: Vec<String>,
    return_type: Option<String>,
    stability: Option<String>,
    description: String,
    varargs: Option<String>,
    categories: Vec<String>,
    tags: Vec<(String, String)>,
}

fn function_row(
    database_name: String,
    function_name: String,
    info: &vgi_client::dtos::FunctionInfo,
) -> DFResult<DuckDbFunctionRow> {
    let arguments = if info.arguments.0.is_empty() {
        Arc::new(Schema::empty())
    } else {
        vgi_protocol::ipc::read_schema(&info.arguments.0).map_err(crate::to_df)?
    };
    let parameter_type = |field: &Field| match field.metadata().get("vgi_type") {
        Some(value) if value.eq_ignore_ascii_case("any") => "ANY".to_string(),
        Some(value) if value.eq_ignore_ascii_case("table") => "TABLE".to_string(),
        _ => duckdb_type_name(field.data_type()),
    };
    let fixed_arguments = arguments.fields().iter().filter(|field| {
        !field
            .metadata()
            .get("vgi_varargs")
            .is_some_and(|value| value == "true")
    });
    let parameters = fixed_arguments
        .clone()
        .map(|field| {
            field
                .name()
                .strip_prefix("named_")
                .unwrap_or(field.name())
                .to_string()
        })
        .collect::<Vec<_>>();
    let parameter_types = fixed_arguments
        .map(|field| parameter_type(field))
        .collect::<Vec<_>>();
    let varargs = arguments
        .fields()
        .iter()
        .find(|field| {
            field
                .metadata()
                .get("vgi_varargs")
                .is_some_and(|value| value == "true")
        })
        .map(|field| parameter_type(field));
    let kind = info.function_type.0.to_ascii_lowercase();
    let function_type = if kind == "scalar" {
        "scalar"
    } else if kind == "aggregate" {
        "aggregate"
    } else {
        "table"
    }
    .to_string();
    let return_type = if function_type == "table" || info.output_schema.0.is_empty() {
        None
    } else {
        let output = vgi_protocol::ipc::read_schema(&info.output_schema.0).map_err(crate::to_df)?;
        output.fields().first().map(|field| {
            if field
                .metadata()
                .get("vgi:any")
                .is_some_and(|value| value == "true")
            {
                "ANY".to_string()
            } else {
                duckdb_type_name(field.data_type())
            }
        })
    };
    Ok(DuckDbFunctionRow {
        database_name,
        schema_name: info.schema_name.clone(),
        function_name,
        function_type,
        parameters,
        parameter_types,
        return_type,
        stability: info.stability.as_ref().map(|value| value.0.clone()),
        description: info.description.clone(),
        varargs,
        categories: info.categories.clone(),
        tags: info.tags.clone(),
    })
}

fn string_list_array(rows: &[Vec<String>]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for value in row {
            builder.values().append_value(value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

/// Compatibility projection of attached VGI routine declarations through the
/// columns the shared corpus reads from DuckDB's `duckdb_functions()`.
fn duckdb_functions(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let mut rows = Vec::new();
    for (alias, metadata) in runtime.catalog_metadata() {
        for info in &metadata.functions {
            rows.push(function_row(alias.clone(), info.name.clone(), info)?);
        }
        for info in &metadata.macros {
            let is_table = info.macro_type.0.eq_ignore_ascii_case("table_macro")
                || info.macro_type.0.eq_ignore_ascii_case("table");
            rows.push(DuckDbFunctionRow {
                database_name: alias.clone(),
                schema_name: info.schema_name.clone(),
                function_name: info.name.clone(),
                function_type: if is_table { "table_macro" } else { "macro" }.to_string(),
                parameters: info.parameters.clone(),
                parameter_types: Vec::new(),
                return_type: None,
                stability: None,
                description: info.comment.clone().unwrap_or_default(),
                varargs: None,
                categories: Vec::new(),
                tags: info.tags.clone(),
            });
        }
        for info in &metadata.global_functions {
            let name = if metadata.global_function_prefix.is_empty() {
                info.name.to_ascii_lowercase()
            } else {
                format!("{}_{}", metadata.global_function_prefix, info.name).to_ascii_lowercase()
            };
            rows.push(function_row("system".to_string(), name, info)?);
        }
    }

    let parameters = string_list_array(
        &rows
            .iter()
            .map(|row| row.parameters.clone())
            .collect::<Vec<_>>(),
    );
    let parameter_types = string_list_array(
        &rows
            .iter()
            .map(|row| row.parameter_types.clone())
            .collect::<Vec<_>>(),
    );
    let categories = string_list_array(
        &rows
            .iter()
            .map(|row| row.categories.clone())
            .collect::<Vec<_>>(),
    );
    let mut tags = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for row in &rows {
        for (key, value) in &row.tags {
            tags.keys().append_value(key);
            tags.values().append_value(value);
        }
        tags.append(true)?;
    }
    let tags: ArrayRef = Arc::new(tags.finish());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.database_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.schema_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.function_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.function_type.as_str()),
        )),
        parameters,
        parameter_types,
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.return_type.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.stability.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.description.as_str()),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.varargs.as_deref())
                .collect::<Vec<_>>(),
        )),
        categories,
        tags,
    ];
    let names = [
        "database_name",
        "schema_name",
        "function_name",
        "function_type",
        "parameters",
        "parameter_types",
        "return_type",
        "stability",
        "description",
        "varargs",
        "categories",
        "tags",
    ];
    let schema = Arc::new(Schema::new(
        names
            .into_iter()
            .zip(&arrays)
            .map(|(name, array)| Field::new(name, array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    Ok(RecordBatch::try_new(schema, arrays)?)
}

#[derive(Debug)]
struct FunctionArgumentRow {
    catalog_name: String,
    schema_name: String,
    function_name: String,
    function_type: String,
    field_index: u64,
    arg_position: Option<i64>,
    arg_name: String,
    arg_type: String,
    is_positional: bool,
    is_named: bool,
    is_const: bool,
    is_varargs: bool,
    is_any_type: bool,
    arg_description: Option<String>,
    arg_default: Option<String>,
    arg_choices: Option<String>,
    arg_range: Option<String>,
    arg_pattern: Option<String>,
}

fn append_argument_rows(
    rows: &mut Vec<FunctionArgumentRow>,
    catalog: &str,
    schema_name: &str,
    function_name: &str,
    function_type: &str,
    schema: &Schema,
) {
    let mut positional = 0_i64;
    for (field_index, field) in schema.fields().iter().enumerate() {
        let metadata = field.metadata();
        let is_named = metadata
            .get("vgi_arg")
            .is_some_and(|value| value.eq_ignore_ascii_case("named"))
            || field.name().starts_with("named_");
        let is_positional = !is_named;
        let arg_position = is_positional.then_some(positional);
        if is_positional {
            positional += 1;
        }
        let is_any_type = metadata
            .get("vgi_type")
            .is_some_and(|value| value.eq_ignore_ascii_case("any"));
        let arg_type = if is_any_type {
            "ANY".to_string()
        } else if metadata
            .get("vgi_type")
            .is_some_and(|value| value.eq_ignore_ascii_case("table"))
        {
            "TABLE".to_string()
        } else {
            duckdb_type_name(field.data_type())
        };
        rows.push(FunctionArgumentRow {
            catalog_name: catalog.to_string(),
            schema_name: schema_name.to_string(),
            function_name: function_name.to_string(),
            function_type: function_type.to_string(),
            field_index: field_index as u64,
            arg_position,
            arg_name: field
                .name()
                .strip_prefix("named_")
                .unwrap_or(field.name())
                .to_string(),
            arg_type,
            is_positional,
            is_named,
            is_const: metadata
                .get("vgi_const")
                .is_some_and(|value| value == "true"),
            is_varargs: metadata
                .get("vgi_varargs")
                .is_some_and(|value| value == "true"),
            is_any_type,
            arg_description: metadata.get("vgi_doc").cloned(),
            arg_default: metadata.get("vgi_default").cloned(),
            arg_choices: metadata.get("vgi_choices").cloned(),
            arg_range: metadata.get("vgi_range").cloned(),
            arg_pattern: metadata.get("vgi_pattern").cloned(),
        });
    }
}

fn vgi_function_arguments(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let mut rows = Vec::new();
    for (alias, metadata) in runtime.catalog_metadata() {
        for info in &metadata.functions {
            let schema = if info.arguments.0.is_empty() {
                Arc::new(Schema::empty())
            } else {
                vgi_protocol::ipc::read_schema(&info.arguments.0).map_err(crate::to_df)?
            };
            append_argument_rows(
                &mut rows,
                &alias,
                &info.schema_name,
                &info.name,
                &info.function_type.0.to_ascii_lowercase(),
                &schema,
            );
        }
        for info in &metadata.macros {
            let schema = match &info.arguments_schema {
                Some(arguments) if !arguments.0.is_empty() => {
                    vgi_protocol::ipc::read_schema(&arguments.0).map_err(crate::to_df)?
                }
                _ => Arc::new(Schema::new(
                    info.parameters
                        .iter()
                        .map(|name| Field::new(name, DataType::Null, true))
                        .collect::<Vec<_>>(),
                )),
            };
            append_argument_rows(
                &mut rows,
                &alias,
                &info.schema_name,
                &info.name,
                &info.macro_type.0.to_ascii_lowercase(),
                &schema,
            );
        }
    }

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.catalog_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.schema_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.function_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.function_type.as_str()),
        )),
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|row| row.field_index),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.arg_position).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.arg_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.arg_type.as_str()),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.is_positional).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.is_named).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.is_const).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.is_varargs).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.is_any_type).collect::<Vec<_>>(),
        )),
        optional_strings(&rows, |row| row.arg_description.as_deref()),
        optional_strings(&rows, |row| row.arg_default.as_deref()),
        optional_strings(&rows, |row| row.arg_choices.as_deref()),
        optional_strings(&rows, |row| row.arg_range.as_deref()),
        optional_strings(&rows, |row| row.arg_pattern.as_deref()),
    ];
    let names = [
        "catalog_name",
        "schema_name",
        "function_name",
        "function_type",
        "field_index",
        "arg_position",
        "arg_name",
        "arg_type",
        "is_positional",
        "is_named",
        "is_const",
        "is_varargs",
        "is_any_type",
        "arg_description",
        "arg_default",
        "arg_choices",
        "arg_range",
        "arg_pattern",
    ];
    let schema = Arc::new(Schema::new(
        names
            .into_iter()
            .zip(&arrays)
            .map(|(name, array)| Field::new(name, array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    Ok(RecordBatch::try_new(schema, arrays)?)
}

fn optional_strings<'a, T>(rows: &'a [T], value: impl Fn(&'a T) -> Option<&'a str>) -> ArrayRef {
    Arc::new(StringArray::from(
        rows.iter().map(value).collect::<Vec<_>>(),
    ))
}

fn log_stats(runtime: &VgiRuntime) -> DFResult<RecordBatch> {
    let mut grouped = std::collections::BTreeMap::<String, (u64, i64)>::new();
    for event in runtime.events() {
        let timestamp = event
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let entry = grouped.entry(event.kind).or_default();
        entry.0 += 1;
        entry.1 = entry.1.max(timestamp);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("event", DataType::Utf8, false),
        Field::new("count", DataType::UInt64, false),
        Field::new("last_timestamp_ms", DataType::Int64, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                grouped.keys().map(String::as_str),
            )),
            Arc::new(UInt64Array::from_iter_values(
                grouped.values().map(|(count, _)| *count),
            )),
            Arc::new(Int64Array::from_iter_values(
                grouped.values().map(|(_, timestamp)| *timestamp),
            )),
        ],
    )?)
}

fn zero_arg_u64(
    name: &str,
    operation: impl Fn() -> u64 + Send + Sync + 'static,
) -> datafusion::logical_expr::ScalarUDF {
    let operation = Arc::new(operation);
    let fun: ScalarFunctionImplementation = Arc::new(move |_args| {
        Ok(ColumnarValue::Scalar(ScalarValue::UInt64(
            Some(operation()),
        )))
    });
    create_udf(name, vec![], DataType::UInt64, Volatility::Volatile, fun)
}

pub(crate) fn register(ctx: &SessionContext, runtime: Arc<VgiRuntime>) {
    let state = ctx.state();
    for (name, kind) in [
        ("vgi_cache_stats", DiagnosticsKind::CacheStats),
        ("vgi_plan_cache_stats", DiagnosticsKind::PlanCacheStats),
        ("vgi_cache_entries", DiagnosticsKind::CacheEntries),
        ("vgi_logs", DiagnosticsKind::Logs),
        ("vgi_log_stats", DiagnosticsKind::LogStats),
        // DuckDB-extension compatibility aliases backed by the same
        // session-scoped DataFusion cache. These deliberately do not emulate
        // disk/exchange cache fields the adapter has not implemented.
        ("vgi_result_cache_stats", DiagnosticsKind::CacheStats),
        ("vgi_result_cache", DiagnosticsKind::DuckDbCacheEntries),
        ("vgi_result_cache_flush", DiagnosticsKind::CacheFlush),
        ("vgi_result_cache_reap", DiagnosticsKind::CacheReap),
        ("duckdb_logs", DiagnosticsKind::DuckDbLogs),
        ("duckdb_functions", DiagnosticsKind::DuckDbFunctions),
        (
            "vgi_function_arguments",
            DiagnosticsKind::VgiFunctionArguments,
        ),
    ] {
        if !state.table_functions().contains_key(name) {
            ctx.register_udtf(
                name,
                Arc::new(DiagnosticsTable {
                    runtime: Arc::clone(&runtime),
                    kind,
                }),
            );
        }
    }

    if !state.scalar_functions().contains_key("typeof") {
        ctx.register_udf(ScalarUDF::new_from_impl(DuckDbTypeOf::new()));
    }

    if !state.scalar_functions().contains_key("vgi_cache_flush") {
        let cache = Arc::clone(runtime.result_cache());
        let runtime_for_flush = Arc::clone(&runtime);
        ctx.register_udf(zero_arg_u64("vgi_cache_flush", move || {
            (cache.flush_all() + runtime_for_flush.flush_plan_cache()) as u64
        }));
    }
    if !state.scalar_functions().contains_key("vgi_cache_reap") {
        let cache = Arc::clone(runtime.result_cache());
        ctx.register_udf(zero_arg_u64("vgi_cache_reap", move || cache.reap() as u64));
    }
    if !state.scalar_functions().contains_key("vgi_logs_clear") {
        ctx.register_udf(zero_arg_u64("vgi_logs_clear", move || {
            runtime.clear_events() as u64
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::StringArray;

    #[tokio::test]
    async fn duckdb_typeof_names_use_the_existing_scalar_udf_surface() -> DFResult<()> {
        let ctx = SessionContext::new();
        register(&ctx, Arc::new(VgiRuntime::default()));

        let batches = ctx
            .sql(
                "SELECT typeof(CAST(1 AS TINYINT)) AS t1, \
                 typeof(CAST(1 AS INTEGER)) AS t2, typeof(CAST(1 AS BIGINT)) AS t3, \
                 typeof(CAST(1 AS DOUBLE)) AS t4, typeof('x') AS t5",
            )
            .await?
            .collect()
            .await?;
        let batch = &batches[0];
        let values = batch
            .columns()
            .iter()
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("typeof returns Utf8")
                    .value(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            ["TINYINT", "INTEGER", "BIGINT", "DOUBLE", "VARCHAR"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn duckdb_cache_names_alias_the_session_cache() -> DFResult<()> {
        let ctx = SessionContext::new();
        register(&ctx, Arc::new(VgiRuntime::default()));

        let flush = ctx
            .sql("SELECT flushed FROM vgi_result_cache_flush()")
            .await?
            .collect()
            .await?;
        assert_eq!(flush[0].num_rows(), 1);

        let stats = ctx
            .sql("SELECT entries, total_bytes FROM vgi_result_cache_stats()")
            .await?
            .collect()
            .await?;
        assert_eq!(stats[0].num_rows(), 1);

        let entries = ctx
            .sql("SELECT key_hash, function, num_rows FROM vgi_result_cache()")
            .await?
            .collect()
            .await?;
        assert_eq!(entries.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn duckdb_logs_serializes_real_adapter_events() -> DFResult<()> {
        let ctx = SessionContext::new();
        let runtime = Arc::new(VgiRuntime::default());
        register(&ctx, Arc::clone(&runtime));

        let mut event = crate::VgiEvent::new("cache.hit");
        event.catalog = Some("weather".to_string());
        event.function = Some("main.forecast".to_string());
        runtime.emit(event);

        let batches = ctx
            .sql(
                "SELECT type, message FROM duckdb_logs() \
                 WHERE type = 'VGI' AND message LIKE '%result_cache.hit%' \
                 AND message LIKE '%tier=memory%'",
            )
            .await?
            .collect()
            .await?;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        Ok(())
    }
}
