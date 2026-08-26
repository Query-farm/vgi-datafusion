// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! VGI `TABLESAMPLE SYSTEM` pushdown through DataFusion's relation planner API.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use datafusion::arrow::datatypes::Float64Type;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{plan_err, DataFusionError, Result as DFResult};
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::logical_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::simplify_expressions::simplify_literal::parse_literal;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::ast::{self, TableFactor, TableSampleMethod, TableSampleUnit};
use vgi_client::Sample;

use crate::{VgiRuntime, VgiTableProvider};

#[derive(Debug)]
struct VgiSamplePlanner;

impl RelationPlanner for VgiSamplePlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DFResult<RelationPlanning> {
        let original = relation.clone();
        let TableFactor::Table {
            sample: Some(sample),
            alias,
            name,
            args,
            with_hints,
            version,
            with_ordinality,
            partitions,
            json_path,
            index_hints,
        } = relation
        else {
            return Ok(RelationPlanning::Original(Box::new(original)));
        };

        let sample = match sample {
            ast::TableSampleKind::BeforeTableAlias(sample)
            | ast::TableSampleKind::AfterTableAlias(sample) => sample,
        };
        let base_relation = TableFactor::Table {
            sample: None,
            alias: alias.clone(),
            name,
            args,
            with_hints,
            version,
            with_ordinality,
            partitions,
            json_path,
            index_hints,
        };
        let input = context.plan(base_relation)?;

        // A relation planner is global to the SessionContext, so leave ordinary
        // DataFusion relations and any other extension planner completely alone.
        let (input, vgi_scans) = rewrite_vgi_scans(input, None)?;
        if vgi_scans == 0 {
            return Ok(RelationPlanning::Original(Box::new(original)));
        }
        if vgi_scans != 1 {
            return Err(DataFusionError::NotImplemented(
                "TABLESAMPLE on a relation expanding to multiple VGI scans is not supported"
                    .to_string(),
            ));
        }
        if sample.offset.is_some() || sample.bucket.is_some() {
            return Err(DataFusionError::NotImplemented(
                "VGI TABLESAMPLE does not support OFFSET or BUCKET sampling".to_string(),
            ));
        }

        let Some(quantity) = sample.quantity.as_ref() else {
            return plan_err!("VGI TABLESAMPLE requires a percentage");
        };
        if quantity.unit != Some(TableSampleUnit::Percent) {
            return Err(DataFusionError::NotImplemented(
                "VGI TABLESAMPLE supports percentage sampling only".to_string(),
            ));
        }
        let quantity_expr = context.sql_to_expr(quantity.value.clone(), input.schema())?;
        let percentage = parse_literal::<Float64Type>(&quantity_expr)?;
        if !percentage.is_finite() || !(0.0..=100.0).contains(&percentage) {
            return plan_err!(
                "VGI TABLESAMPLE percentage must be between 0 and 100, got {percentage}"
            );
        }
        let seed = sample
            .seed
            .as_ref()
            .map(|seed| {
                seed.value.to_string().parse::<i64>().map_err(|_| {
                    DataFusionError::Plan("TABLESAMPLE seed must be an integer".to_string())
                })
            })
            .transpose()?;

        let plan = match sample.name {
            Some(TableSampleMethod::System) => {
                rewrite_vgi_scans(input, Some(Sample { percentage, seed }))?.0
            }
            // VGI has no Bernoulli protocol hint. At 100%, DataFusion's local
            // operation is mathematically the identity, so retain the base scan
            // and deliberately send no sample metadata. This is the portable
            // corpus case and mirrors DuckDB's division of responsibilities.
            Some(TableSampleMethod::Bernoulli) if percentage == 100.0 => input,
            Some(method) => {
                return Err(DataFusionError::NotImplemented(format!(
                    "VGI TABLESAMPLE {method} is not supported; only SYSTEM percentage sampling and BERNOULLI(100 PERCENT) are available"
                )));
            }
            None => {
                return Err(DataFusionError::NotImplemented(
                    "VGI TABLESAMPLE requires the SYSTEM method".to_string(),
                ));
            }
        };

        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(
            plan, alias,
        ))))
    }
}

/// Find the VGI scan produced for one table factor and optionally replace its
/// immutable provider with a sampled clone.
fn rewrite_vgi_scans(plan: LogicalPlan, sample: Option<Sample>) -> DFResult<(LogicalPlan, usize)> {
    let mut found = 0usize;
    let plan = plan
        .transform_up(|plan| {
            let LogicalPlan::TableScan(mut scan) = plan else {
                return Ok(Transformed::no(plan));
            };
            let Ok(provider) = source_as_provider(&scan.source) else {
                return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
            };
            let Some(vgi) = provider.downcast_ref::<VgiTableProvider>() else {
                return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
            };
            found += 1;
            let Some(sample) = sample else {
                return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
            };
            scan.source = provider_as_source(Arc::new(vgi.with_sample_hint(sample)?));
            Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
        })?
        .data;
    Ok((plan, found))
}

/// Install the planner once for this session. The weak runtime entry lets a
/// dropped SessionContext/session id be registered again rather than leaving a
/// process-global tombstone.
pub(crate) fn ensure_registered(ctx: &SessionContext, runtime: &Arc<VgiRuntime>) -> DFResult<()> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Weak<VgiRuntime>>>> = OnceLock::new();
    let sessions = SESSIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sessions = sessions.lock().unwrap();
    sessions.retain(|_, runtime| runtime.strong_count() > 0);
    let session_id = ctx.session_id();
    if sessions.get(&session_id).and_then(Weak::upgrade).is_some() {
        return Ok(());
    }
    ctx.register_relation_planner(Arc::new(VgiSamplePlanner))?;
    sessions.insert(session_id, Arc::downgrade(runtime));
    Ok(())
}
