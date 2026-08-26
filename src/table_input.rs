// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Table-valued arguments — `f('name', (SELECT … ))`.
//!
//! # What a TABLE argument is on the wire
//!
//! It is not a third kind of argument. A VGI function that declares a
//! `TableInput` parameter is an **exchange-mode** function: the table becomes
//! the call's *input stream* and the scalar arguments stay bind arguments. The
//! DuckDB extension builds it the same way — the TABLE parameter turns into the
//! bind's `input_schema` and the rows flow through the ordinary exchange.
//!
//! So this reuses the machinery that already carries scalar calls:
//! `bind_with_input` to resolve the output schema, then `open_exchange` to push
//! batches and read answers.
//!
//! # Only single-column tables, and why that is not my choice
//!
//! DataFusion plans a table-function argument as an ordinary expression, and a
//! subquery in expression position is a **scalar** subquery — which by
//! definition yields one column. A wider one is refused during planning, before
//! any of this code runs:
//!
//! ```text
//! SELECT * FROM f('x', (SELECT * FROM (VALUES (1,2)) AS t(a,b)));
//! Error during planning: Too many columns! The subquery should only return
//! one column: t.a, t.b
//! ```
//!
//! There is no hook to intercept that — the rejection happens in the expression
//! planner, and `TableFunctionArgs` only ever sees `Expr`s. Widening it needs a
//! change in DataFusion itself, so a multi-column table argument is out of
//! reach here rather than merely unimplemented.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;

use crate::{to_df, VgiConnection, VgiEvent};

/// Exchange tiers do not yet send conditional validators back to a worker.
/// Refuse immediately-stale policies here instead of retaining bytes that a
/// plain lookup can never reuse. Producer scans have a separate revalidation
/// path and continue to accept this policy through `ResultCache` directly.
pub(crate) fn exchange_cache_ttl(
    cache: &vgi_client::ResultCache,
    control: Option<&vgi_client::CacheControl>,
    identity_scope: &str,
    bytes: usize,
) -> Result<std::time::Duration, vgi_client::cache::Ineligible> {
    let ttl = cache.eligibility(control, Some(identity_scope), bytes)?;
    if ttl.is_zero() {
        return Err(vgi_client::cache::Ineligible::NoFreshness);
    }
    Ok(ttl)
}

/// The subquery behind a TABLE argument, plus where it sat in the call.
#[derive(Debug, Clone)]
pub(crate) struct TableArgument {
    /// Position among the call's arguments — the scalar arguments keep their
    /// relative order once this one is removed.
    pub index: usize,
    /// The subquery's plan, executed at scan time rather than at bind.
    pub plan: Arc<LogicalPlan>,
}

impl TableArgument {
    /// Find the single TABLE argument in a call, if there is one.
    ///
    /// More than one is refused rather than silently taking the first: a
    /// function with two table inputs is a different shape, and guessing which
    /// one is the input stream would bind the wrong thing.
    pub fn find(exprs: &[Expr]) -> DFResult<Option<Self>> {
        let mut found: Option<Self> = None;
        for (index, expr) in exprs.iter().enumerate() {
            if let Expr::ScalarSubquery(sub) = expr {
                if found.is_some() {
                    return Err(DataFusionError::Plan(
                        "a VGI call may take at most one table argument".into(),
                    ));
                }
                found = Some(Self {
                    index,
                    plan: Arc::clone(&sub.subquery),
                });
            }
        }
        Ok(found)
    }

    /// The schema the worker should expect for the input stream.
    ///
    /// The subquery's **own column names** are carried through, not `col_<i>`.
    /// That is what the extension does — it builds the bind's input schema from
    /// `input_table_names`, the names of the TABLE argument's columns — and a
    /// worker echoes them into its output, so renaming here would surface as
    /// `col_0` where the caller wrote `x`.
    ///
    /// Note this differs from a *scalar* call, where the arguments are
    /// positional and really are named `col_<i>`.
    pub fn input_schema(&self) -> SchemaRef {
        Arc::new(Schema::new(
            self.plan
                .schema()
                .fields()
                .iter()
                .map(|f| Field::new(f.name(), f.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ))
    }

    /// Plan the subquery for execution.
    pub async fn physical_plan(&self, session: &dyn Session) -> DFResult<Arc<dyn ExecutionPlan>> {
        session.create_physical_plan(&self.plan).await
    }
}

/// Run an exchange-mode call: push the table argument's rows, collect answers.
///
/// Blocking, like the rest of the client, and therefore called from
/// `spawn_blocking`. Batches are pushed one at a time and every answer is kept;
/// a worker may answer a batch with nothing (a filter) or with more rows than it
/// received (a generator), so the output is not positionally tied to the input.
pub(crate) fn run_exchange(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arguments: vgi_client::Arguments,
    input_schema: &Schema,
    inputs: Vec<datafusion::arrow::array::RecordBatch>,
    stream_cache_eligible: bool,
) -> DFResult<Vec<datafusion::arrow::array::RecordBatch>> {
    use vgi_client::{BindSpec, ScanOptions};

    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let catalog_version = attached.info().catalog_version;
    let argument_bytes = arguments.to_ipc().map_err(to_df)?.0;
    let spec = BindSpec::table(function)
        .in_schema(schema_name)
        .with_arguments(arguments);
    let (bound, used_resolved_secrets) =
        crate::bind_with_input_secrets_status(conn, &mut client, &attached, &spec, input_schema)?;
    let stream_cache_eligible = stream_cache_eligible && !used_resolved_secrets;
    let cache_key_template = if stream_cache_eligible {
        exchange_cache_key_template(
            conn,
            catalog,
            schema_name,
            function,
            &argument_bytes,
            catalog_version,
            bound.output_schema().as_ref(),
            b"table_in_out_batch_v2",
        )?
    } else {
        None
    };

    // Probe every input unit before borrowing the client for an exchange. A
    // fully warm call must not initialize a worker conversation, and doing the
    // classification up front also keeps one clean mutable borrow for all
    // misses that remain.
    let mut units = Vec::with_capacity(inputs.len());
    let mut has_miss = false;
    for batch in inputs {
        let cache_key = cache_key_template
            .as_ref()
            .map(|template| template.key_for_input(&batch))
            .transpose()?;
        let cached = cache_key
            .as_ref()
            .and_then(|key| conn.runtime.result_cache().get(key))
            .map(|entry| {
                conn.runtime.note_exchange_cache_hit(entry.bytes());
                entry.batches().to_vec()
            });
        if cached.is_some() {
            emit_exchange_cache_event(conn, catalog, schema_name, function, "cache.hit", None);
        } else if cache_key.is_some() {
            emit_exchange_cache_event(conn, catalog, schema_name, function, "cache.miss", None);
            has_miss = true;
        } else {
            has_miss = true;
        }
        units.push((batch, cache_key, cached));
    }

    let mut out = Vec::new();
    if !has_miss {
        for (_, _, cached) in units {
            out.extend(cached.expect("all units were cache hits"));
        }
        return Ok(out);
    }

    let mut exchange = client
        .open_exchange(&bound, &ScanOptions::default())
        .map_err(to_df)?;
    for (batch, cache_key, cached) in units {
        if let Some(cached) = cached {
            out.extend(cached);
            continue;
        }
        if let Some(answer) = exchange.send(&batch).map_err(to_df)? {
            if let Some(key) = cache_key {
                let control = exchange.cache_control().cloned();
                let bytes = answer
                    .columns()
                    .iter()
                    .map(|array| array.get_array_memory_size())
                    .sum();
                match exchange_cache_ttl(
                    conn.runtime.result_cache(),
                    control.as_ref(),
                    &key.identity_scope,
                    bytes,
                ) {
                    Ok(ttl) => {
                        conn.runtime.result_cache().insert(
                            key,
                            vec![answer.clone()],
                            ttl,
                            control.as_ref(),
                        );
                        conn.runtime.note_exchange_cache_store();
                        emit_exchange_cache_event(
                            conn,
                            catalog,
                            schema_name,
                            function,
                            "cache.store",
                            None,
                        );
                    }
                    Err(reason) => emit_exchange_cache_event(
                        conn,
                        catalog,
                        schema_name,
                        function,
                        "cache.refused",
                        Some(format!("exchange batch: {reason:?}")),
                    ),
                }
            }
            out.push(answer);
        }
    }
    // Input EOS. A function with a FINALIZE phase would answer only after
    // this, through `finalize_table_in_out`; that shape is not wired up yet, so
    // a buffered function reached this way returns nothing rather than wrong
    // rows.
    exchange.close().map_err(to_df)?;
    Ok(out)
}

/// Static portion of an exchange cache key.
///
/// Arguments, schemas, and attachment context can be much larger than one
/// cached result. Hash them once per call rather than cloning them into every
/// per-input key (where key memory is outside the result-cache byte accounting).
pub(crate) struct ExchangeCacheKeyTemplate {
    catalog: String,
    identity_scope: String,
    worker_label: String,
    function: String,
    catalog_version: i64,
    static_digest: [u8; 32],
}

impl ExchangeCacheKeyTemplate {
    pub(crate) fn input_digest(
        &self,
        input: &datafusion::arrow::array::RecordBatch,
    ) -> DFResult<[u8; 32]> {
        use sha2::{Digest, Sha256};

        let input = vgi_protocol::ipc::write_batch(input).map_err(to_df)?;
        let mut digest = Sha256::new();
        hash_exchange_field(&mut digest, b"vgi_exchange_input_v2");
        hash_exchange_field(&mut digest, &self.static_digest);
        hash_exchange_field(&mut digest, &input);
        Ok(digest.finalize().into())
    }

    pub(crate) fn key_for_digest(&self, input_digest: [u8; 32]) -> vgi_client::CacheKey {
        vgi_client::CacheKey {
            catalog: self.catalog.clone(),
            identity_scope: self.identity_scope.clone(),
            worker_label: self.worker_label.clone(),
            function: self.function.clone(),
            // Keep both variable-sized regions compact. The static digest
            // covers the canonical arguments, output schema, and attach
            // context; plan is the domain-separated digest of this input row.
            arguments: self.static_digest.to_vec(),
            projection: None,
            filters: None,
            catalog_version: self.catalog_version,
            at: None,
            settings: Vec::new(),
            attach_options: Vec::new(),
            row_limit: None,
            ordering: None,
            sample: None,
            plan: Some(input_digest.to_vec()),
        }
    }

    pub(crate) fn key_for_input(
        &self,
        input: &datafusion::arrow::array::RecordBatch,
    ) -> DFResult<vgi_client::CacheKey> {
        self.input_digest(input)
            .map(|digest| self.key_for_digest(digest))
    }
}

fn hash_exchange_field(digest: &mut sha2::Sha256, field: &[u8]) {
    use sha2::Digest;

    digest.update((field.len() as u64).to_le_bytes());
    digest.update(field);
}

/// Build the compact static portion of exchange cache keys once per call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exchange_cache_key_template(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arguments: &[u8],
    catalog_version: i64,
    output_schema: &Schema,
    kind: &[u8],
) -> DFResult<Option<ExchangeCacheKeyTemplate>> {
    if !conn.cache_enabled || !conn.runtime.options().cache_enabled {
        return Ok(None);
    }
    let Some(identity_scope) = conn.cache_identity_scope(catalog) else {
        return Ok(None);
    };
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    hash_exchange_field(&mut digest, b"vgi_exchange_static_v2");
    hash_exchange_field(&mut digest, kind);
    hash_exchange_field(&mut digest, catalog.as_bytes());
    hash_exchange_field(&mut digest, identity_scope.as_bytes());
    hash_exchange_field(&mut digest, conn.label().as_bytes());
    hash_exchange_field(&mut digest, format!("{schema_name}.{function}").as_bytes());
    hash_exchange_field(&mut digest, arguments);
    hash_exchange_field(&mut digest, &catalog_version.to_le_bytes());
    hash_exchange_field(&mut digest, &conn.cache_attach_context(catalog));
    let output_schema = vgi_protocol::ipc::write_schema(output_schema).map_err(to_df)?;
    hash_exchange_field(&mut digest, &output_schema);
    Ok(Some(ExchangeCacheKeyTemplate {
        catalog: catalog.to_string(),
        identity_scope,
        worker_label: conn.label().to_string(),
        function: format!("{schema_name}.{function}"),
        catalog_version,
        static_digest: digest.finalize().into(),
    }))
}

pub(crate) fn emit_exchange_cache_event(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    kind: &str,
    message: Option<String>,
) {
    let mut event = VgiEvent::new(kind);
    event.catalog = Some(catalog.to_string());
    event.function = Some(format!("{schema_name}.{function}"));
    event.message = message;
    conn.runtime.emit(event);
}

/// A childless blended table-in/out call whose literal arguments form one
/// input row.
#[derive(Debug)]
pub(crate) struct VgiLiteralInputProvider {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    arguments: vgi_client::Arguments,
    input_schema: SchemaRef,
    input: datafusion::arrow::array::RecordBatch,
    output_schema: SchemaRef,
}

impl VgiLiteralInputProvider {
    pub(crate) fn bind_blocking(
        conn: VgiConnection,
        catalog: &str,
        schema_name: &str,
        function: &str,
        arguments: vgi_client::Arguments,
        input_schema: SchemaRef,
        input: datafusion::arrow::array::RecordBatch,
    ) -> DFResult<Arc<Self>> {
        use vgi_client::BindSpec;

        let mut client = conn.connect()?;
        let attached = conn.attach(&mut client, catalog)?;
        let spec = BindSpec::table(function)
            .in_schema(schema_name)
            .with_arguments(arguments.clone());
        let bound = crate::bind_with_input_secrets(
            &conn,
            &mut client,
            &attached,
            &spec,
            input_schema.as_ref(),
        )?;
        let output_schema = bound.output_schema().clone();

        Ok(Arc::new(Self {
            conn,
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            function: function.to_string(),
            arguments,
            input_schema,
            input,
            output_schema,
        }))
    }
}

#[async_trait::async_trait]
impl datafusion::catalog::TableProvider for VgiLiteralInputProvider {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> datafusion::logical_expr::TableType {
        datafusion::logical_expr::TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        use datafusion::datasource::memory::MemorySourceConfig;

        let (conn, catalog, schema_name, function, arguments) = (
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
            self.arguments.clone(),
        );
        let input_schema = self.input_schema.clone();
        let input = self.input.clone();
        let output_schema = self.output_schema.clone();
        let batches = tokio::task::spawn_blocking(move || {
            run_exchange(
                &conn,
                &catalog,
                &schema_name,
                &function,
                arguments,
                input_schema.as_ref(),
                vec![input],
                false,
            )
        })
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))??;

        let batches = batches
            .into_iter()
            .map(|batch| crate::conform(batch, &output_schema, None))
            .collect::<DFResult<Vec<_>>>()?;
        Ok(MemorySourceConfig::try_new_exec(
            &[batches],
            output_schema,
            projection.cloned(),
        )?)
    }
}

/// Run a **buffered** call: every row is ingested before any output exists.
///
/// A `TableBufferingFunction` is a different protocol, not a variation on the
/// streaming one — sending it an `INPUT`-phase exchange fails outright
/// ("Unsupported init phase for TableBufferingFunction"). The shape is
/// Sink-then-Source:
///
/// 1. `buffering_begin` mints the execution id everything else is scoped to;
/// 2. `buffering_process` ingests each chunk and returns an opaque state id;
/// 3. `buffering_combine` collapses those ids into the ones to drain;
/// 4. `buffering_finalize` drains each in producer mode.
///
/// The state ids are the worker's own — this only round-trips them.
pub(crate) fn run_buffered(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arguments: vgi_client::Arguments,
    input_schema: &Schema,
    inputs: Vec<datafusion::arrow::array::RecordBatch>,
) -> DFResult<Vec<datafusion::arrow::array::RecordBatch>> {
    use vgi_client::BindSpec;

    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let spec = BindSpec::table(function)
        .in_schema(schema_name)
        .with_arguments(arguments);
    let bound = crate::bind_with_input_secrets(conn, &mut client, &attached, &spec, input_schema)?;

    let execution_id = client.buffering_begin(&bound).map_err(to_df)?;

    let mut state_ids = Vec::with_capacity(inputs.len());
    for (i, batch) in inputs.iter().enumerate() {
        // The batch index is what lets a worker reconstruct source order from
        // chunks it may process out of order.
        let id = client
            .buffering_process(&attached, &spec, &execution_id, batch, Some(i as i64))
            .map_err(to_df)?;
        state_ids.push(id);
    }

    let finalize_ids = client
        .buffering_combine(&attached, &spec, &execution_id, state_ids)
        .map_err(to_df)?;

    let mut out = Vec::new();
    for id in &finalize_ids {
        let mut scan = client
            .buffering_finalize(&bound, &execution_id, id)
            .map_err(to_df)?;
        while let Some(batch) = scan.next_batch().map_err(to_df)? {
            out.push(batch);
        }
    }
    Ok(out)
}

/// A call with a TABLE argument, exposed as a DataFusion table.
///
/// The subquery is *not* executed at bind time — only its schema is needed
/// then. Execution is deferred to `scan`, which is async and has a `Session`,
/// so the rows are produced once the plan actually runs rather than during
/// planning.
#[derive(Debug)]
pub struct VgiTableInputProvider {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    arguments: vgi_client::Arguments,
    table_arg: TableArgument,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    /// Whether the worker declared this a `TableBufferingFunction`, which is a
    /// different protocol rather than a variation on the streaming one.
    buffered: bool,
    stream_cache_eligible: bool,
}

impl VgiTableInputProvider {
    /// Bind the call, resolving its output schema from the table's schema.
    pub(crate) fn bind_blocking(
        conn: VgiConnection,
        catalog: &str,
        schema_name: &str,
        function: &str,
        arguments: vgi_client::Arguments,
        table_arg: TableArgument,
        buffered: bool,
        stream_cache_eligible: bool,
    ) -> DFResult<Arc<Self>> {
        use vgi_client::BindSpec;

        let input_schema = table_arg.input_schema();
        let mut client = conn.connect()?;
        let attached = conn.attach(&mut client, catalog)?;
        let spec = BindSpec::table(function)
            .in_schema(schema_name)
            .with_arguments(arguments.clone());
        let bound =
            crate::bind_with_input_secrets(&conn, &mut client, &attached, &spec, &input_schema)?;
        let output_schema = bound.output_schema().clone();

        Ok(Arc::new(Self {
            conn,
            catalog: catalog.to_string(),
            schema_name: schema_name.to_string(),
            function: function.to_string(),
            arguments,
            table_arg,
            input_schema,
            output_schema,
            buffered,
            stream_cache_eligible,
        }))
    }
}

#[async_trait::async_trait]
impl datafusion::catalog::TableProvider for VgiTableInputProvider {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> datafusion::logical_expr::TableType {
        datafusion::logical_expr::TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::datasource::memory::MemorySourceConfig;
        use futures::TryStreamExt;

        // Run the subquery now: its rows are this call's input stream.
        let plan = self.table_arg.physical_plan(state).await?;
        let task_ctx = state.task_ctx();
        let mut inputs: Vec<RecordBatch> = Vec::new();
        for partition in 0..plan.properties().output_partitioning().partition_count() {
            let stream = plan.execute(partition, Arc::clone(&task_ctx))?;
            let mut batches: Vec<RecordBatch> = stream.try_collect().await?;
            inputs.append(&mut batches);
        }

        // Rebuild each batch against the declared input schema. The columns and
        // types already match — it was derived from this very plan — but the
        // nullability flags may differ, and the worker validates against what
        // the bind declared.
        let input_schema = self.input_schema.clone();
        let inputs: Vec<RecordBatch> = inputs
            .into_iter()
            .map(|b| {
                RecordBatch::try_new(input_schema.clone(), b.columns().to_vec())
                    .map_err(DataFusionError::from)
            })
            .collect::<DFResult<_>>()?;

        let (conn, cat, sch, func, args) = (
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
            self.arguments.clone(),
        );
        let schema = self.output_schema.clone();
        let for_worker = input_schema.clone();

        let buffered = self.buffered;
        let stream_cache_eligible = self.stream_cache_eligible;
        let out = tokio::task::spawn_blocking(move || {
            if buffered {
                run_buffered(&conn, &cat, &sch, &func, args, &for_worker, inputs)
            } else {
                run_exchange(
                    &conn,
                    &cat,
                    &sch,
                    &func,
                    args,
                    &for_worker,
                    inputs,
                    stream_cache_eligible,
                )
            }
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let out: Vec<RecordBatch> = out
            .into_iter()
            .map(|b| crate::conform(b, &schema, None))
            .collect::<DFResult<_>>()?;

        // One partition: the exchange is a single conversation with one
        // worker, so the answers arrive as one ordered sequence.
        Ok(MemorySourceConfig::try_new_exec(
            &[out],
            schema,
            projection.cloned(),
        )?)
    }
}

#[cfg(test)]
mod cache_key_tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn one_row(schema: &Arc<Schema>, value: i64) -> RecordBatch {
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![value]))],
        )
        .unwrap()
    }

    #[test]
    fn exchange_keys_hash_static_context_once_and_input_per_unit() {
        let conn = VgiConnection::subprocess(["unused"]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let large_arguments = vec![0x5a; 1024 * 1024];
        let template = exchange_cache_key_template(
            &conn,
            "example",
            "main",
            "echo",
            &large_arguments,
            7,
            schema.as_ref(),
            b"scalar_per_value_v2",
        )
        .unwrap()
        .unwrap();

        let first = template.key_for_input(&one_row(&schema, 1)).unwrap();
        let first_again = template.key_for_input(&one_row(&schema, 1)).unwrap();
        let second = template.key_for_input(&one_row(&schema, 2)).unwrap();

        assert_eq!(first, first_again);
        assert_ne!(first, second);
        assert_eq!(first.arguments.len(), 32);
        assert_eq!(first.plan.as_ref().unwrap().len(), 32);
        assert!(first.attach_options.is_empty());
        assert_ne!(first.arguments.as_slice(), large_arguments.as_slice());

        let different_static = exchange_cache_key_template(
            &conn,
            "example",
            "main",
            "echo",
            b"different arguments",
            7,
            schema.as_ref(),
            b"scalar_per_value_v2",
        )
        .unwrap()
        .unwrap()
        .key_for_input(&one_row(&schema, 1))
        .unwrap();
        assert_ne!(first, different_static);
    }
}
