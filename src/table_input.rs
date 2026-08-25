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

use crate::{to_df, VgiConnection};

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
) -> DFResult<Vec<datafusion::arrow::array::RecordBatch>> {
    use vgi_client::{BindSpec, ScanOptions};

    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let spec = BindSpec::table(function)
        .in_schema(schema_name)
        .with_arguments(arguments);
    let bound = client
        .bind_with_input(&attached, &spec, input_schema)
        .map_err(to_df)?;

    let mut ex = client
        .open_exchange(&bound, &ScanOptions::default())
        .map_err(to_df)?;
    let mut out = Vec::new();
    for batch in inputs {
        if let Some(answer) = ex.send(&batch).map_err(to_df)? {
            out.push(answer);
        }
    }
    // Input EOS. A function with a FINALIZE phase would answer only after
    // this, through `finalize_table_in_out`; that shape is not wired up yet, so
    // a buffered function reached this way returns nothing rather than wrong
    // rows.
    ex.close().map_err(to_df)?;
    Ok(out)
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
        let bound = client
            .bind_with_input(&attached, &spec, input_schema.as_ref())
            .map_err(to_df)?;
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
            )
        })
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))??;

        let batches = batches
            .into_iter()
            .map(|batch| crate::conform(batch, &output_schema))
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
    let bound = client
        .bind_with_input(&attached, &spec, input_schema)
        .map_err(to_df)?;

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
    ) -> DFResult<Arc<Self>> {
        use vgi_client::BindSpec;

        let input_schema = table_arg.input_schema();
        let mut client = conn.connect()?;
        let attached = conn.attach(&mut client, catalog)?;
        let spec = BindSpec::table(function)
            .in_schema(schema_name)
            .with_arguments(arguments.clone());
        let bound = client
            .bind_with_input(&attached, &spec, &input_schema)
            .map_err(to_df)?;
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
        let out = tokio::task::spawn_blocking(move || {
            if buffered {
                run_buffered(&conn, &cat, &sch, &func, args, &for_worker, inputs)
            } else {
                run_exchange(&conn, &cat, &sch, &func, args, &for_worker, inputs)
            }
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        let out: Vec<RecordBatch> = out
            .into_iter()
            .map(|b| crate::conform(b, &schema))
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
