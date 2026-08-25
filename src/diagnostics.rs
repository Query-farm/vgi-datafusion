// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! SQL surfaces for session-scoped VGI cache and event diagnostics.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemTable, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::common::{plan_err, Result as DFResult, ScalarValue};
use datafusion::logical_expr::{
    create_udf, ColumnarValue, ScalarFunctionImplementation, Volatility,
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
    Logs,
    LogStats,
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
            DiagnosticsKind::Logs => logs(&self.runtime)?,
            DiagnosticsKind::LogStats => log_stats(&self.runtime)?,
        };
        Ok(Arc::new(MemTable::try_new(
            batch.schema(),
            vec![vec![batch]],
        )?))
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
