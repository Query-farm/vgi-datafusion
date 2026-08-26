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

use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;

use crate::{to_df, VgiConnection, VgiEvent};

/// Apply the shared result-cache eligibility rules to an exchange result.
/// Zero-TTL entries are retained only when the worker supplied a conditional
/// validator; the exchange paths revalidate those bytes before serving them.
pub(crate) fn exchange_cache_ttl(
    cache: &vgi_client::ResultCache,
    control: Option<&vgi_client::CacheControl>,
    identity_scope: &str,
    bytes: usize,
) -> Result<std::time::Duration, vgi_client::cache::Ineligible> {
    cache.eligibility(control, Some(identity_scope), bytes)
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
#[allow(clippy::too_many_arguments)] // Mirrors the protocol call boundary.
pub(crate) fn run_exchange(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arguments: vgi_client::Arguments,
    input_schema: &Schema,
    inputs: Vec<datafusion::arrow::array::RecordBatch>,
    shape_ineligible: Option<crate::runtime::CacheIneligibleReason>,
) -> DFResult<Vec<datafusion::arrow::array::RecordBatch>> {
    use vgi_client::{BindSpec, ScanOptions};

    enum UnitState {
        Hit(Vec<datafusion::arrow::array::RecordBatch>),
        Producer(Arc<crate::runtime::ResultFlightProducer>),
        Follower(crate::runtime::ResultFlightWaiter),
        Uncached,
    }

    struct Unit {
        input: datafusion::arrow::array::RecordBatch,
        key: Option<vgi_client::CacheKey>,
        state: UnitState,
        output: Option<Vec<datafusion::arrow::array::RecordBatch>>,
    }

    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let catalog_version = attached.info().catalog_version;
    let argument_bytes = arguments.to_ipc().map_err(to_df)?.0;
    let spec = BindSpec::table(function)
        .in_schema(schema_name)
        .with_arguments(arguments);
    let (bound, secret_dependent) = crate::bind_with_input_secrets_dependency(
        conn,
        &mut client,
        &attached,
        &spec,
        input_schema,
    )?;
    let cache_ineligible = conn
        .cache_environment_ineligible_reason(catalog)
        .or(shape_ineligible)
        .or(secret_dependent.then_some(crate::runtime::CacheIneligibleReason::SecretDependent));
    if let Some(reason) = cache_ineligible {
        conn.runtime
            .emit_cache_ineligible(catalog, &format!("{schema_name}.{function}"), reason);
    }
    let cache_key_template = if cache_ineligible.is_none() {
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

    // Probe and claim every key before doing work. Claims are retained until
    // their bytes have been stored (or the attempt aborts), so concurrent
    // identical exchanges cannot stampede the worker.
    let mut units = Vec::with_capacity(inputs.len());
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
        let state = if let Some(cached) = cached {
            emit_exchange_cache_event(conn, catalog, schema_name, function, "cache.hit", None);
            UnitState::Hit(cached)
        } else if let Some(cache_key) = cache_key.as_ref() {
            emit_exchange_cache_event(conn, catalog, schema_name, function, "cache.miss", None);
            match conn.runtime.acquire_result_flight(cache_key) {
                crate::runtime::ResultFlightClaim::Producer(producer) => {
                    UnitState::Producer(producer)
                }
                crate::runtime::ResultFlightClaim::Follower(waiter) => UnitState::Follower(waiter),
            }
        } else {
            UnitState::Uncached
        };
        units.push(Unit {
            input: batch,
            key: cache_key,
            state,
            output: None,
        });
    }

    // Revalidations need one init per key because the validator is init
    // metadata. Ordinary cold producers and uncached inputs continue to share
    // one exchange, preserving the existing streaming behavior.
    let revalidations = units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| match (&unit.state, &unit.key) {
            (UnitState::Producer(_), Some(key)) => conn
                .runtime
                .result_cache()
                .get_for_revalidation(key)
                .map(|entry| (index, entry)),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (index, stale) in revalidations {
        let key = units[index].key.as_ref().expect("revalidation has key");
        let opts = ScanOptions {
            if_none_match: stale.etag.clone(),
            if_modified_since: stale.last_modified.clone(),
            ..Default::default()
        };
        let attempt = (|| -> DFResult<_> {
            let mut exchange = client.open_exchange(&bound, &opts).map_err(to_df)?;
            let answer = exchange.send(&units[index].input).map_err(to_df)?;
            emit_table_input_write(
                conn,
                catalog,
                schema_name,
                function,
                units[index].input.num_rows(),
            );
            let control = exchange.cache_control().cloned();
            exchange.close().map_err(to_df)?;
            let answer = answer.ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "{function} ended a conditional exchange without an answer"
                ))
            })?;
            Ok((answer, control))
        })();
        let (answer, control) = match attempt {
            Ok(answer) => answer,
            Err(error) if stale.may_serve_on_error_at(std::time::Instant::now()) => {
                conn.runtime.result_cache().record_stale_serve();
                conn.runtime.note_exchange_cache_hit(stale.bytes());
                units[index].output = Some(stale.batches().to_vec());
                if let UnitState::Producer(producer) = &units[index].state {
                    producer.stored();
                }
                emit_exchange_cache_event(
                    conn,
                    catalog,
                    schema_name,
                    function,
                    "cache.stale_if_error",
                    Some(error.to_string()),
                );
                continue;
            }
            Err(error) => return Err(error),
        };
        if control.as_ref().is_some_and(|control| control.not_modified) {
            if answer.num_rows() != 0 {
                return Err(DataFusionError::Execution(format!(
                    "VGI function `{function}` returned rows and not_modified together"
                )));
            }
            let ttl = match exchange_cache_ttl(
                conn.runtime.result_cache(),
                control.as_ref(),
                &key.identity_scope,
                stale.bytes(),
            ) {
                Ok(ttl) => ttl,
                Err(reason) => {
                    conn.runtime.result_cache().remove(key);
                    if let UnitState::Producer(producer) = &units[index].state {
                        producer.abort(format!("revalidation revoked cache: {reason:?}"));
                    }
                    return Err(DataFusionError::Execution(format!(
                        "VGI function `{function}` returned not_modified with ineligible cache control: {reason:?}"
                    )));
                }
            };
            conn.runtime.result_cache().slide(key, ttl);
            conn.runtime.note_exchange_cache_hit(stale.bytes());
            units[index].output = Some(stale.batches().to_vec());
            if let UnitState::Producer(producer) = &units[index].state {
                producer.stored();
            }
            emit_exchange_cache_event(
                conn,
                catalog,
                schema_name,
                function,
                "cache.revalidated",
                None,
            );
        } else {
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
                        key.clone(),
                        vec![answer.clone()],
                        ttl,
                        control.as_ref(),
                    );
                    conn.runtime.note_exchange_cache_store();
                    if let UnitState::Producer(producer) = &units[index].state {
                        producer.stored();
                    }
                    units[index].output = Some(vec![answer]);
                }
                Err(reason) => {
                    conn.runtime.result_cache().remove(key);
                    if let UnitState::Producer(producer) = &units[index].state {
                        producer.abort(format!("cache refused result: {reason:?}"));
                    }
                    units[index].output = Some(vec![answer]);
                }
            }
        }
    }

    let direct = units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| {
            (unit.output.is_none()
                && matches!(&unit.state, UnitState::Producer(_) | UnitState::Uncached))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    if !direct.is_empty() {
        let mut exchange = client
            .open_exchange(&bound, &ScanOptions::default())
            .map_err(to_df)?;
        for index in direct {
            let answer = exchange.send(&units[index].input).map_err(to_df)?;
            emit_table_input_write(
                conn,
                catalog,
                schema_name,
                function,
                units[index].input.num_rows(),
            );
            let control = exchange.cache_control().cloned();
            if let Some(answer) = answer {
                if let Some(key) = units[index].key.as_ref() {
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
                                key.clone(),
                                vec![answer.clone()],
                                ttl,
                                control.as_ref(),
                            );
                            conn.runtime.note_exchange_cache_store();
                            if let UnitState::Producer(producer) = &units[index].state {
                                producer.stored();
                            }
                            emit_exchange_cache_event(
                                conn,
                                catalog,
                                schema_name,
                                function,
                                "cache.store",
                                None,
                            );
                        }
                        Err(reason) => {
                            if let UnitState::Producer(producer) = &units[index].state {
                                producer.abort(format!("cache refused result: {reason:?}"));
                            }
                            emit_exchange_cache_event(
                                conn,
                                catalog,
                                schema_name,
                                function,
                                "cache.refused",
                                Some(format!("exchange batch: {reason:?}")),
                            );
                        }
                    }
                }
                units[index].output = Some(vec![answer]);
            } else {
                if let UnitState::Producer(producer) = &units[index].state {
                    producer.abort("worker returned no cacheable answer");
                }
                units[index].output = Some(Vec::new());
            }
        }
        exchange.close().map_err(to_df)?;
    }

    // Followers wake only after the producer has stored or aborted. On abort,
    // run once without another cache claim so a worker refusing cache remains
    // correct and cannot form a retry loop.
    for unit in &mut units {
        let UnitState::Follower(waiter) = &unit.state else {
            continue;
        };
        let key = unit.key.as_ref().expect("follower has key");
        if matches!(
            waiter.wait_blocking_timeout(conn.rpc_timeout()),
            crate::runtime::ResultFlightOutcome::Stored
        ) {
            if let Some(entry) = conn
                .runtime
                .result_cache()
                .get(key)
                .or_else(|| conn.runtime.result_cache().get_for_revalidation(key))
            {
                conn.runtime.note_exchange_cache_hit(entry.bytes());
                unit.output = Some(entry.batches().to_vec());
                continue;
            }
        }
        let mut exchange = client
            .open_exchange(&bound, &ScanOptions::default())
            .map_err(to_df)?;
        let answer = exchange.send(&unit.input).map_err(to_df)?;
        emit_table_input_write(conn, catalog, schema_name, function, unit.input.num_rows());
        unit.output = Some(answer.into_iter().collect());
        exchange.close().map_err(to_df)?;
    }

    let mut out = Vec::new();
    for unit in units {
        match unit.state {
            UnitState::Hit(cached) => out.extend(cached),
            _ => out.extend(unit.output.unwrap_or_default()),
        }
    }
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
    settings: Vec<u8>,
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
            settings: self.settings.clone(),
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

    /// Hash a complete unordered input multiset as one cache unit.
    ///
    /// Buffered caching is offered only for workers that declare
    /// `sink_order_dependent=false`. Canonical per-row IPC digests are sorted,
    /// preserving duplicates while making both row order and physical batch
    /// boundaries irrelevant. The input schema is included explicitly so an
    /// empty input cannot cross-serve a differently typed bind.
    pub(crate) fn key_for_unordered_inputs(
        &self,
        inputs: &[datafusion::arrow::array::RecordBatch],
        input_schema: &Schema,
    ) -> DFResult<vgi_client::CacheKey> {
        use sha2::{Digest, Sha256};

        let mut rows = Vec::<[u8; 32]>::new();
        for input in inputs {
            for row in 0..input.num_rows() {
                let encoded =
                    vgi_protocol::ipc::write_batch(&input.slice(row, 1)).map_err(to_df)?;
                rows.push(Sha256::digest(encoded).into());
            }
        }
        rows.sort_unstable();

        let mut digest = Sha256::new();
        hash_exchange_field(&mut digest, b"vgi_buffered_whole_input_multiset_v2");
        hash_exchange_field(&mut digest, &self.static_digest);
        let input_schema = vgi_protocol::ipc::write_schema(input_schema).map_err(to_df)?;
        hash_exchange_field(&mut digest, &input_schema);
        hash_exchange_field(&mut digest, &(rows.len() as u64).to_le_bytes());
        for row in rows {
            hash_exchange_field(&mut digest, &row);
        }
        Ok(self.key_for_digest(digest.finalize().into()))
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
    let settings = conn.runtime().session_settings_identity(catalog);
    hash_exchange_field(&mut digest, &settings);
    let output_schema = vgi_protocol::ipc::write_schema(output_schema).map_err(to_df)?;
    hash_exchange_field(&mut digest, &output_schema);
    Ok(Some(ExchangeCacheKeyTemplate {
        catalog: catalog.to_string(),
        identity_scope,
        worker_label: conn.cache_worker_identity(),
        function: format!("{schema_name}.{function}"),
        catalog_version,
        settings,
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

/// Record one table-input batch only after the worker accepted the send.
///
/// Cardinality and function identity are useful for exchange diagnostics;
/// input values, bind arguments, execution ids, and secrets are deliberately
/// excluded from the event.
pub(crate) fn emit_table_input_write(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    input_rows: usize,
) {
    let mut event = VgiEvent::new("table_in_out.write_input");
    event.catalog = Some(catalog.to_string());
    event.function = Some(format!("{schema_name}.{function}"));
    event.message = Some(format!("input_rows={input_rows}"));
    conn.runtime.emit(event);
}

fn emit_table_buffering_event(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    kind: &str,
    duration: std::time::Duration,
    message: Option<String>,
) {
    let mut event = VgiEvent::new(kind);
    event.catalog = Some(catalog.to_string());
    event.function = Some(format!("{schema_name}.{function}"));
    event.duration = Some(duration);
    event.message = message;
    conn.runtime.emit(event);
}

#[derive(Debug)]
enum BufferedCacheControl {
    Consistent(Option<vgi_client::CacheControl>),
    Refused,
}

fn normalize_buffered_cache_control(
    function: &str,
    controls: Vec<Option<vgi_client::CacheControl>>,
) -> DFResult<BufferedCacheControl> {
    let Some(first) = controls.first().cloned() else {
        return Ok(BufferedCacheControl::Consistent(None));
    };
    if controls.iter().all(|control| control == &first) {
        return Ok(BufferedCacheControl::Consistent(first));
    }
    if controls
        .iter()
        .flatten()
        .any(|control| control.not_modified)
    {
        return Err(DataFusionError::Execution(format!(
            "VGI buffered function `{function}` mixed not_modified with incompatible finalize cache control"
        )));
    }
    // Cache metadata is an optimization. Fresh rows remain valid even when
    // independent finalize streams disagree or only some opt in.
    Ok(BufferedCacheControl::Refused)
}

/// Marks a detached blocking buffered lifecycle ineligible to commit if its
/// async DataFusion scan future is dropped before the lifecycle completes.
struct BufferedCommitGuard {
    allowed: Arc<Mutex<bool>>,
    completed: bool,
}

impl BufferedCommitGuard {
    fn new() -> Self {
        Self {
            allowed: Arc::new(Mutex::new(true)),
            completed: false,
        }
    }

    fn token(&self) -> Arc<Mutex<bool>> {
        Arc::clone(&self.allowed)
    }

    fn complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for BufferedCommitGuard {
    fn drop(&mut self) {
        if !self.completed {
            *self.allowed.lock().unwrap() = false;
        }
    }
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
                Some(crate::runtime::CacheIneligibleReason::LiteralInput),
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
#[allow(clippy::too_many_arguments)] // Mirrors the buffering protocol boundary.
pub(crate) fn run_buffered(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arguments: vgi_client::Arguments,
    input_schema: &Schema,
    inputs: Vec<datafusion::arrow::array::RecordBatch>,
    sink_order_dependent: bool,
    commit_allowed: Arc<Mutex<bool>>,
) -> DFResult<Vec<datafusion::arrow::array::RecordBatch>> {
    use vgi_client::{BindSpec, ScanOptions};

    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let catalog_version = attached.info().catalog_version;
    let argument_bytes = arguments.to_ipc().map_err(to_df)?.0;
    let spec = BindSpec::table(function)
        .in_schema(schema_name)
        .with_arguments(arguments);
    let (bound, secret_dependent) = crate::bind_with_input_secrets_dependency(
        conn,
        &mut client,
        &attached,
        &spec,
        input_schema,
    )?;

    let run_lifecycle = |client: &mut vgi_client::VgiClient,
                         options: &ScanOptions|
     -> DFResult<(
        Vec<datafusion::arrow::array::RecordBatch>,
        Vec<Option<vgi_client::CacheControl>>,
    )> {
        let begin_started = std::time::Instant::now();
        let execution_id = client.buffering_begin(&bound).map_err(to_df)?;
        emit_table_buffering_event(
            conn,
            catalog,
            schema_name,
            function,
            "table_buffering.begin",
            begin_started.elapsed(),
            None,
        );

        let mut state_ids = Vec::with_capacity(inputs.len());
        for (index, batch) in inputs.iter().enumerate() {
            // The worker lifecycle still receives the physical batch index.
            // Unordered cache identity below is deliberately independent of
            // those boundaries; ordered sinks are ineligible for that cache.
            let id = client
                .buffering_process(&attached, &spec, &execution_id, batch, Some(index as i64))
                .map_err(to_df)?;
            state_ids.push(id);
        }

        let state_id_count = state_ids.len();
        let combine_started = std::time::Instant::now();
        let finalize_ids = client
            .buffering_combine(&attached, &spec, &execution_id, state_ids)
            .map_err(to_df)?;
        emit_table_buffering_event(
            conn,
            catalog,
            schema_name,
            function,
            "table_buffering.combine",
            combine_started.elapsed(),
            Some(format!(
                "input_batches={} state_ids={state_id_count} finalize_ids={}",
                inputs.len(),
                finalize_ids.len()
            )),
        );

        let mut out = Vec::new();
        let mut controls = Vec::with_capacity(finalize_ids.len());
        let finalize_started = std::time::Instant::now();
        for id in &finalize_ids {
            let mut scan = client
                .buffering_finalize_with_options(&bound, &execution_id, id, options)
                .map_err(to_df)?;
            while let Some(batch) = scan.next_batch().map_err(to_df)? {
                out.push(batch);
            }
            controls.push(scan.cache_control().cloned());
        }
        if !finalize_ids.is_empty() {
            emit_table_buffering_event(
                conn,
                catalog,
                schema_name,
                function,
                "table_buffering.finalize",
                finalize_started.elapsed(),
                Some(format!(
                    "finalize_streams={} output_batches={}",
                    finalize_ids.len(),
                    out.len()
                )),
            );
        }
        Ok((out, controls))
    };

    // VGI's buffered-cache contract is a reduction over the complete input
    // multiset. An ordered sink can observe row order, so it is deliberately
    // ineligible instead of being cached under a weaker identity.
    let cache_ineligible = conn
        .cache_environment_ineligible_reason(catalog)
        .or(secret_dependent.then_some(crate::runtime::CacheIneligibleReason::SecretDependent))
        .or(sink_order_dependent.then_some(crate::runtime::CacheIneligibleReason::OrderedSink));
    if let Some(reason) = cache_ineligible {
        conn.runtime
            .emit_cache_ineligible(catalog, &format!("{schema_name}.{function}"), reason);
    }
    let template = if cache_ineligible.is_some() {
        None
    } else {
        exchange_cache_key_template(
            conn,
            catalog,
            schema_name,
            function,
            &argument_bytes,
            catalog_version,
            bound.output_schema().as_ref(),
            b"table_buffering_whole_input_v1",
        )?
    };
    let Some(key) = template
        .as_ref()
        .map(|template| template.key_for_unordered_inputs(&inputs, input_schema))
        .transpose()?
    else {
        return run_lifecycle(&mut client, &ScanOptions::default()).map(|(batches, _)| batches);
    };

    let cache = conn.runtime.result_cache();
    if let Some(entry) = cache.get(&key) {
        conn.runtime.note_exchange_cache_hit(entry.bytes());
        emit_exchange_cache_event(
            conn,
            catalog,
            schema_name,
            function,
            "cache.hit",
            Some("tier=buffered_whole_input".to_string()),
        );
        return Ok(entry.batches().to_vec());
    }
    emit_exchange_cache_event(
        conn,
        catalog,
        schema_name,
        function,
        "cache.miss",
        Some("tier=buffered_whole_input".to_string()),
    );

    match conn.runtime.acquire_result_flight(&key) {
        crate::runtime::ResultFlightClaim::Follower(waiter) => {
            if matches!(
                waiter.wait_blocking_timeout(conn.rpc_timeout()),
                crate::runtime::ResultFlightOutcome::Stored
            ) {
                if let Some(entry) = cache.get(&key).or_else(|| cache.get_for_revalidation(&key)) {
                    conn.runtime.note_exchange_cache_hit(entry.bytes());
                    emit_exchange_cache_event(
                        conn,
                        catalog,
                        schema_name,
                        function,
                        "cache.coalesced_hit",
                        Some("tier=buffered_whole_input".to_string()),
                    );
                    return Ok(entry.batches().to_vec());
                }
            }
            // A refusal, timeout, or intervening eviction cannot make the query
            // fail. Execute without taking another flight, and do not store.
            run_lifecycle(&mut client, &ScanOptions::default()).map(|(batches, _)| batches)
        }
        crate::runtime::ResultFlightClaim::Producer(flight) => {
            let stale = cache.get_for_revalidation(&key);
            let options = stale
                .as_ref()
                .map(|entry| ScanOptions {
                    if_none_match: entry.etag.clone(),
                    if_modified_since: entry.last_modified.clone(),
                    ..Default::default()
                })
                .unwrap_or_default();
            let (batches, controls) = match run_lifecycle(&mut client, &options) {
                Ok(result) => result,
                Err(error) => {
                    let commit = commit_allowed.lock().unwrap();
                    if !*commit {
                        flight.abort("buffered consumer cancelled");
                        return Err(error);
                    }
                    if !stale
                        .as_ref()
                        .is_some_and(|entry| entry.may_serve_on_error_at(std::time::Instant::now()))
                    {
                        flight.abort("buffered lifecycle failed");
                        return Err(error);
                    }
                    let stale = stale.as_ref().expect("checked above");
                    cache.record_stale_serve();
                    conn.runtime.note_exchange_cache_hit(stale.bytes());
                    flight.stored();
                    emit_exchange_cache_event(
                        conn,
                        catalog,
                        schema_name,
                        function,
                        "cache.stale_if_error",
                        Some(format!("tier=buffered_whole_input {error}")),
                    );
                    return Ok(stale.batches().to_vec());
                }
            };
            let commit = commit_allowed.lock().unwrap();
            if !*commit {
                flight.abort("buffered consumer cancelled");
                return Ok(batches);
            }
            let control = match normalize_buffered_cache_control(function, controls)? {
                BufferedCacheControl::Consistent(control) => control,
                BufferedCacheControl::Refused => {
                    cache.remove(&key);
                    flight.abort("buffered finalize cache controls disagreed");
                    emit_exchange_cache_event(
                        conn,
                        catalog,
                        schema_name,
                        function,
                        "cache.refused",
                        Some("tier=buffered_whole_input finalize controls disagreed".to_string()),
                    );
                    return Ok(batches);
                }
            };

            if control.as_ref().is_some_and(|control| control.not_modified) {
                let Some(stale) = stale.as_ref() else {
                    flight.abort("not_modified without a conditional request");
                    return Err(DataFusionError::Execution(format!(
                        "VGI buffered function `{function}` returned not_modified without a conditional request"
                    )));
                };
                if batches.iter().any(|batch| batch.num_rows() != 0) {
                    flight.abort("not_modified returned rows");
                    return Err(DataFusionError::Execution(format!(
                        "VGI buffered function `{function}` returned rows and not_modified together"
                    )));
                }
                let ttl = match exchange_cache_ttl(
                    cache,
                    control.as_ref(),
                    &key.identity_scope,
                    stale.bytes(),
                ) {
                    Ok(ttl) => ttl,
                    Err(reason) => {
                        cache.remove(&key);
                        flight.abort(format!("revalidation revoked cache: {reason:?}"));
                        return Err(DataFusionError::Execution(format!(
                            "VGI buffered function `{function}` returned not_modified with ineligible cache control: {reason:?}"
                        )));
                    }
                };
                cache.slide(&key, ttl);
                conn.runtime.note_exchange_cache_hit(stale.bytes());
                flight.stored();
                emit_exchange_cache_event(
                    conn,
                    catalog,
                    schema_name,
                    function,
                    "cache.revalidated",
                    Some("tier=buffered_whole_input".to_string()),
                );
                return Ok(stale.batches().to_vec());
            }

            let bytes = batches
                .iter()
                .flat_map(|batch| batch.columns())
                .map(|array| array.get_array_memory_size())
                .sum();
            match exchange_cache_ttl(cache, control.as_ref(), &key.identity_scope, bytes) {
                Ok(ttl) => {
                    cache.insert(key, batches.clone(), ttl, control.as_ref());
                    conn.runtime.note_exchange_cache_store();
                    flight.stored();
                    emit_exchange_cache_event(
                        conn,
                        catalog,
                        schema_name,
                        function,
                        "cache.store",
                        Some("tier=buffered_whole_input".to_string()),
                    );
                }
                Err(reason) => {
                    if reason == vgi_client::cache::Ineligible::EntryTooLarge {
                        // Buffered output is necessarily materialized before
                        // finalize completes. Record that the bounded cache
                        // capture was abandoned even though the query result
                        // itself remains valid and is returned normally.
                        cache.record_capture_abort();
                    }
                    cache.remove(&key);
                    flight.abort(format!("cache refused buffered result: {reason:?}"));
                    emit_exchange_cache_event(
                        conn,
                        catalog,
                        schema_name,
                        function,
                        "cache.refused",
                        Some(format!("tier=buffered_whole_input {reason:?}")),
                    );
                }
            }
            Ok(batches)
        }
    }
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
    sink_order_dependent: bool,
    stream_cache_eligible: bool,
}

impl VgiTableInputProvider {
    /// Bind the call, resolving its output schema from the table's schema.
    #[allow(clippy::too_many_arguments)] // Keeps discovery facts explicit at the bind boundary.
    pub(crate) fn bind_blocking(
        conn: VgiConnection,
        catalog: &str,
        schema_name: &str,
        function: &str,
        arguments: vgi_client::Arguments,
        table_arg: TableArgument,
        buffered: bool,
        sink_order_dependent: bool,
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
            sink_order_dependent,
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
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::datasource::memory::MemorySourceConfig;
        use futures::TryStreamExt;

        // Plan the subquery now: its rows are this call's input stream.
        let plan = self.table_arg.physical_plan(state).await?;

        // A pushed limit is the one case where eager collection is observably
        // wrong for a streaming exchange: DataFusion cannot cancel work hidden
        // inside TableProvider::scan, so LIMIT 5 over a large child would first
        // materialize and send the entire child. Keep the exchange in the
        // physical plan instead. The buffered protocol still needs all input
        // before it can produce any output and therefore retains the whole-
        // input path below. This limited path intentionally bypasses result
        // caching: cancellation leaves an incomplete child exchange, which
        // must never be committed under the full per-batch cache identity.
        if !self.buffered {
            if let Some(limit) = limit {
                return Ok(Arc::new(
                    crate::table_input_stream::VgiLimitedTableInputExec::try_new(
                        self.conn.clone(),
                        self.catalog.clone(),
                        self.schema_name.clone(),
                        self.function.clone(),
                        self.arguments.clone(),
                        plan,
                        Arc::clone(&self.input_schema),
                        Arc::clone(&self.output_schema),
                        projection.cloned(),
                        limit,
                    )?,
                ));
            }
        }

        // No finite limit: run and retain the complete exchange result as the
        // existing cache-aware path expects.
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
        let sink_order_dependent = self.sink_order_dependent;
        let stream_cache_ineligible = (!self.stream_cache_eligible)
            .then_some(crate::runtime::CacheIneligibleReason::StatefulExchange);
        let mut buffered_commit = buffered.then(BufferedCommitGuard::new);
        let commit_allowed = buffered_commit.as_ref().map(BufferedCommitGuard::token);
        let out = tokio::task::spawn_blocking(move || {
            if buffered {
                run_buffered(
                    &conn,
                    &cat,
                    &sch,
                    &func,
                    args,
                    &for_worker,
                    inputs,
                    sink_order_dependent,
                    commit_allowed.expect("buffered execution has commit guard"),
                )
            } else {
                run_exchange(
                    &conn,
                    &cat,
                    &sch,
                    &func,
                    args,
                    &for_worker,
                    inputs,
                    stream_cache_ineligible,
                )
            }
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        if let Some(commit) = &mut buffered_commit {
            commit.complete();
        }
        let out = out?;

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

    fn rows(schema: &Arc<Schema>, values: Vec<i64>) -> RecordBatch {
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap()
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

    #[test]
    fn exchange_keys_are_isolated_by_session_settings() {
        let conn = VgiConnection::subprocess(["unused"]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let key = || {
            exchange_cache_key_template(
                &conn,
                "example",
                "main",
                "settings_aware",
                &[],
                7,
                schema.as_ref(),
                b"scalar_per_value_v2",
            )
            .unwrap()
            .unwrap()
            .key_for_input(&one_row(&schema, 1))
            .unwrap()
        };
        let initial = key();
        let mut settings = crate::VgiSettings::default();
        settings.set_value("multiplier", "5").unwrap();
        conn.runtime().replace_session_settings(settings);
        let configured = key();
        assert_ne!(initial, configured);
        assert_ne!(initial.settings, configured.settings);
    }

    #[test]
    fn buffered_keys_are_unordered_multisets_and_keep_protocol_identity() {
        let conn = VgiConnection::subprocess(["unused"]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let template = exchange_cache_key_template(
            &conn,
            "example",
            "main",
            "sum_all",
            &[],
            7,
            schema.as_ref(),
            b"table_buffering_whole_input_v1",
        )
        .unwrap()
        .unwrap();
        let one_batch = template
            .key_for_unordered_inputs(&[rows(&schema, vec![1, 2])], schema.as_ref())
            .unwrap();
        let same = template
            .key_for_unordered_inputs(&[rows(&schema, vec![1, 2])], schema.as_ref())
            .unwrap();
        let split = template
            .key_for_unordered_inputs(&[one_row(&schema, 1), one_row(&schema, 2)], schema.as_ref())
            .unwrap();
        let reversed = template
            .key_for_unordered_inputs(&[one_row(&schema, 2), one_row(&schema, 1)], schema.as_ref())
            .unwrap();
        let duplicate = template
            .key_for_unordered_inputs(
                &[
                    one_row(&schema, 1),
                    one_row(&schema, 1),
                    one_row(&schema, 2),
                ],
                schema.as_ref(),
            )
            .unwrap();

        assert_eq!(one_batch, same);
        assert_eq!(
            one_batch, split,
            "batch boundaries do not change a multiset"
        );
        assert_eq!(split, reversed, "unordered buffered input has one identity");
        assert_ne!(
            split, duplicate,
            "duplicate multiplicity remains key material"
        );

        let streaming = exchange_cache_key_template(
            &conn,
            "example",
            "main",
            "sum_all",
            &[],
            7,
            schema.as_ref(),
            b"table_in_out_batch_v2",
        )
        .unwrap()
        .unwrap()
        .key_for_unordered_inputs(&[rows(&schema, vec![1, 2])], schema.as_ref())
        .unwrap();
        assert_ne!(one_batch, streaming, "protocol modes cannot cross-serve");
    }

    #[test]
    fn fresh_finalize_control_mismatch_refuses_cache_but_partial_304_is_an_error() {
        let ttl = vgi_client::CacheControl::ttl(60);
        assert!(matches!(
            normalize_buffered_cache_control("sum_all", vec![Some(ttl.clone()), None]).unwrap(),
            BufferedCacheControl::Refused
        ));
        assert!(matches!(
            normalize_buffered_cache_control(
                "sum_all",
                vec![Some(ttl.clone()), Some(ttl.clone())]
            )
            .unwrap(),
            BufferedCacheControl::Consistent(Some(control)) if control == ttl
        ));

        let partial_304 = vgi_client::CacheControl::ttl(0)
            .with_etag("v1")
            .with_revalidatable()
            .with_not_modified();
        let error = normalize_buffered_cache_control(
            "sum_all",
            vec![Some(partial_304), Some(vgi_client::CacheControl::ttl(60))],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("mixed not_modified"), "{error}");
    }
}
