// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Incremental execution for limited streaming TABLE-input calls.
//!
//! DataFusion passes a pushed output limit to
//! [`datafusion::catalog::TableProvider::scan`]. A
//! streaming VGI exchange must honor that limit while it is still consuming
//! its child: materializing the complete child and worker result first makes a
//! `LIMIT 5` over a large input do all of the remote work before the limit can
//! cancel it. This node keeps the child and worker exchange incremental and
//! uses bounded channels in both directions for backpressure.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::StreamExt;
use vgi_client::{BindSpec, ScanOptions};

use crate::{to_df, VgiConnection};

/// A single-output-partition streaming exchange over a physical TABLE child.
#[derive(Debug)]
pub(crate) struct VgiLimitedTableInputExec {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    arguments: vgi_client::Arguments,
    input: Arc<dyn ExecutionPlan>,
    input_schema: SchemaRef,
    worker_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: usize,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl VgiLimitedTableInputExec {
    /// Build the limited exchange around the already-planned TABLE child.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        conn: VgiConnection,
        catalog: String,
        schema_name: String,
        function: String,
        arguments: vgi_client::Arguments,
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
        worker_schema: SchemaRef,
        projection: Option<Vec<usize>>,
        limit: usize,
    ) -> DFResult<Self> {
        let schema = match &projection {
            Some(projection) => Arc::new(worker_schema.project(projection)?),
            None => Arc::clone(&worker_schema),
        };
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            // This node is only selected when DataFusion supplies a finite
            // output limit, so even an unbounded child has bounded output.
            Boundedness::Bounded,
        ));
        Ok(Self {
            conn,
            catalog,
            schema_name,
            function,
            arguments,
            input,
            input_schema,
            worker_schema,
            projection,
            limit,
            schema,
            properties,
        })
    }

    fn with_input(&self, input: Arc<dyn ExecutionPlan>) -> DFResult<Self> {
        Self::try_new(
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
            self.arguments.clone(),
            input,
            Arc::clone(&self.input_schema),
            Arc::clone(&self.worker_schema),
            self.projection.clone(),
            self.limit,
        )
    }
}

impl DisplayAs for VgiLimitedTableInputExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VgiLimitedTableInputExec: worker={}, function={}.{}, limit={}, cache=disabled(partial_exchange)",
            self.conn.label(),
            self.schema_name,
            self.function,
            self.limit
        )
    }
}

impl ExecutionPlan for VgiLimitedTableInputExec {
    fn name(&self) -> &str {
        "VgiLimitedTableInputExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let [input] = children.as_slice() else {
            return Err(DataFusionError::Internal(format!(
                "{} expected one child, got {}",
                self.name(),
                children.len()
            )));
        };
        Ok(Arc::new(self.with_input(Arc::clone(input))?))
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        ) -> DFResult<datafusion::common::tree_node::TreeNodeRecursion>,
    ) -> DFResult<datafusion::common::tree_node::TreeNodeRecursion> {
        // This wrapper owns no physical expressions. Execution-plan traversal
        // visits its child independently.
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    }

    fn execute(
        &self,
        partition: usize,
        task_ctx: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "{} has one output partition, requested {partition}",
                self.name()
            )));
        }

        self.conn.runtime().emit_cache_ineligible(
            &self.catalog,
            &format!("{}.{}", self.schema_name, self.function),
            crate::runtime::CacheIneligibleReason::PartialExchange,
        );

        let output_schema = Arc::clone(&self.schema);
        if self.limit == 0 {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                output_schema,
                futures::stream::empty::<DFResult<RecordBatch>>(),
            )));
        }

        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arguments = self.arguments.clone();
        let input = Arc::clone(&self.input);
        let input_schema = Arc::clone(&self.input_schema);
        let worker_schema = Arc::clone(&self.worker_schema);
        let projection = self.projection.clone();
        let limit = self.limit;
        let worker_limit = i64::try_from(limit).ok();

        // The worker blocks on RPC and the child is asynchronous. A small
        // channel between them prevents either side from materializing the
        // complete input, while the equally small output channel lets a
        // downstream LIMIT stop the worker promptly.
        let (output_tx, output_rx) = tokio::sync::mpsc::channel::<DFResult<RecordBatch>>(2);
        let start = move || {
            let (input_tx, mut input_rx) = tokio::sync::mpsc::channel::<DFResult<RecordBatch>>(2);

            tokio::task::spawn_blocking(move || {
                let run = || -> DFResult<()> {
                    let mut client = conn.connect()?;
                    let attached = conn.attach(&mut client, &catalog)?;
                    let spec = BindSpec::table(&function)
                        .in_schema(&schema_name)
                        .with_arguments(arguments);
                    let bound = crate::bind_with_input_secrets(
                        &conn,
                        &mut client,
                        &attached,
                        &spec,
                        input_schema.as_ref(),
                    )?;
                    let mut exchange = client
                        .open_exchange(
                            &bound,
                            &ScanOptions {
                                // This is a worker optimization, not the
                                // correctness boundary: older workers may
                                // ignore it, so output is still truncated and
                                // the exchange cancelled locally below.
                                row_limit: worker_limit,
                                ..Default::default()
                            },
                        )
                        .map_err(to_df)?;
                    let mut emitted = 0usize;

                    while let Some(input) = input_rx.blocking_recv() {
                        let input = input?;
                        let input = RecordBatch::try_new(
                            Arc::clone(&input_schema),
                            input.columns().to_vec(),
                        )?;
                        let answer = exchange.send(&input).map_err(to_df)?;
                        crate::table_input::emit_table_input_write(
                            &conn,
                            &catalog,
                            &schema_name,
                            &function,
                            input.num_rows(),
                        );
                        let Some(batch) = answer else {
                            continue;
                        };
                        let batch = crate::conform(batch, &worker_schema, None)?;
                        let batch = match &projection {
                            Some(projection) => batch.project(projection)?,
                            None => batch,
                        };
                        let remaining = limit.saturating_sub(emitted);
                        let batch = if batch.num_rows() > remaining {
                            batch.slice(0, remaining)
                        } else {
                            batch
                        };
                        emitted += batch.num_rows();
                        if output_tx.blocking_send(Ok(batch)).is_err() {
                            // The downstream plan was cancelled or satisfied
                            // its own limit. Drop performs best-effort cancel.
                            return Ok(());
                        }
                        if emitted >= limit {
                            // The result is already complete. Cancellation is
                            // cleanup, and vgi-client poisons the owning pooled
                            // connection itself if that cleanup fails.
                            let _ = exchange.cancel();
                            return Ok(());
                        }
                    }

                    exchange.close().map_err(to_df)
                };
                if let Err(error) = run() {
                    let _ = output_tx.blocking_send(Err(error));
                }
            });

            tokio::spawn(async move {
                for child_partition in 0..input.properties().output_partitioning().partition_count()
                {
                    let mut stream = match input.execute(child_partition, Arc::clone(&task_ctx)) {
                        Ok(stream) => stream,
                        Err(error) => {
                            let _ = input_tx.send(Err(error)).await;
                            return;
                        }
                    };
                    while let Some(batch) = stream.next().await {
                        let failed = batch.is_err();
                        if input_tx.send(batch).await.is_err() || failed {
                            return;
                        }
                    }
                }
            });
        };

        // Do not connect or execute the child until the output is first
        // polled. Plans are frequently constructed speculatively by joins and
        // optimizer nodes, and `execute` itself is not a start-work signal.
        let stream = futures::stream::unfold(
            (output_rx, Some(start)),
            |(mut output_rx, mut start)| async move {
                if let Some(start) = start.take() {
                    start();
                }
                output_rx
                    .recv()
                    .await
                    .map(|batch| (batch, (output_rx, start)))
            },
        );
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }
}
