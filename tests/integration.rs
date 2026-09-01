// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Consolidated integration-test harness.
//!
//! Keeping the test areas in modules preserves focused libtest filtering while
//! ensuring the DataFusion dependency graph is linked into one executable.

#[path = "common/mod.rs"]
mod common;

#[path = "aggregate_signatures.rs"]
mod aggregate_signatures;
#[path = "attach_options.rs"]
mod attach_options;
#[path = "attach_sql.rs"]
mod attach_sql;
#[path = "auth.rs"]
mod auth;
#[path = "auth_consumers.rs"]
mod auth_consumers;
#[path = "auth_secrets.rs"]
mod auth_secrets;
#[path = "buffered_lifecycle_events.rs"]
mod buffered_lifecycle_events;
#[path = "cache_limits.rs"]
mod cache_limits;
#[path = "cache_shape_identity.rs"]
mod cache_shape_identity;
#[path = "cardinality.rs"]
mod cardinality;
#[path = "companion_catalog.rs"]
mod companion_catalog;
#[path = "dynamic_filters.rs"]
mod dynamic_filters;
#[path = "dynamic_profiling.rs"]
mod dynamic_profiling;
#[path = "exchange_cache_eligibility.rs"]
mod exchange_cache_eligibility;
#[path = "global_functions.rs"]
mod global_functions;
#[path = "http_exchange_cache.rs"]
mod http_exchange_cache;
#[path = "order_pushdown.rs"]
mod order_pushdown;
#[path = "release_fixture.rs"]
mod release_fixture;
#[path = "required_filters.rs"]
mod required_filters;
#[path = "sampling.rs"]
mod sampling;
#[path = "scalar_corpus_compatibility.rs"]
mod scalar_corpus_compatibility;
#[path = "scalar_observability.rs"]
mod scalar_observability;
#[path = "scan_metrics.rs"]
mod scan_metrics;
#[path = "settings.rs"]
mod settings;
#[path = "sql_roundtrip.rs"]
mod sql_roundtrip;
#[path = "table_input_limit.rs"]
mod table_input_limit;
#[path = "table_input_observability.rs"]
mod table_input_observability;
#[path = "table_input_relations.rs"]
mod table_input_relations;
#[path = "transport_stream_hardening.rs"]
mod transport_stream_hardening;
#[path = "transports.rs"]
mod transports;

// CI selects both persistent-worker modules with the `persistent_worker_`
// libtest filter while still reusing this single linked executable.
#[path = "durable_producer_cache.rs"]
mod persistent_worker_durable_producer_cache;
#[path = "worker_log_forwarding.rs"]
mod persistent_worker_log_forwarding;
