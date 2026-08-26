// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! EXPLAIN ANALYZE-only VGI execution profiling.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::analyze::AnalyzeExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

use crate::VgiScanExec;

/// Adds the VGI physical optimizer rules to a DataFusion session-state builder.
///
/// The profiling rule marks VGI scans only when they are descendants of
/// `EXPLAIN ANALYZE`. This is what keeps ordinary queries and plain `EXPLAIN`
/// from issuing the post-execution worker callback.
pub trait VgiSessionStateBuilderExt {
    /// Install the VGI rule with DataFusion's default query planner.
    ///
    /// A host that already has a custom query planner should use
    /// [`Self::with_vgi_physical_optimizer_and_query_planner`] instead.
    #[must_use]
    fn with_vgi_physical_optimizer(self) -> Self;

    /// Install the VGI rule while preserving a host-supplied query planner.
    #[must_use]
    fn with_vgi_physical_optimizer_and_query_planner(
        self,
        query_planner: Arc<dyn QueryPlanner + Send + Sync>,
    ) -> Self;
}

impl VgiSessionStateBuilderExt for SessionStateBuilder {
    fn with_vgi_physical_optimizer(self) -> Self {
        self.with_physical_optimizer_rule(Arc::new(VgiExplainAnalyzeProfiling))
            .with_query_planner(Arc::new(VgiQueryPlanner { inner: None }))
    }

    fn with_vgi_physical_optimizer_and_query_planner(
        self,
        query_planner: Arc<dyn QueryPlanner + Send + Sync>,
    ) -> Self {
        self.with_physical_optimizer_rule(Arc::new(VgiExplainAnalyzeProfiling))
            .with_query_planner(Arc::new(VgiQueryPlanner {
                inner: Some(query_planner),
            }))
    }
}

/// DataFusion builds an `AnalyzeExec` only after it has optimized the analyzed
/// input, so ordinary physical optimizer rules never see that wrapper. This
/// query-planner bridge delegates all planning to DataFusion, then applies the
/// same VGI rule once to the completed outer plan.
#[derive(Debug)]
struct VgiQueryPlanner {
    inner: Option<Arc<dyn QueryPlanner + Send + Sync>>,
}

#[async_trait]
impl QueryPlanner for VgiQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = match &self.inner {
            Some(inner) => inner.create_physical_plan(logical_plan, session).await?,
            None => {
                DefaultPhysicalPlanner::default()
                    .create_physical_plan(logical_plan, session)
                    .await?
            }
        };
        VgiExplainAnalyzeProfiling.optimize(plan, session.config_options())
    }
}

#[derive(Debug)]
struct VgiExplainAnalyzeProfiling;

impl PhysicalOptimizerRule for VgiExplainAnalyzeProfiling {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if plan.downcast_ref::<AnalyzeExec>().is_none() {
            return Ok(plan);
        }

        plan.transform_down(|node| {
            let Some(scan) = node.downcast_ref::<VgiScanExec>() else {
                return Ok(Transformed::no(node));
            };
            Ok(Transformed::yes(
                Arc::new(scan.with_dynamic_profile()) as Arc<dyn ExecutionPlan>
            ))
        })
        .map(|transformed| transformed.data)
    }

    fn name(&self) -> &str {
        "vgi_explain_analyze_profiling"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{DynamicProfile, DynamicProfilePartition};

    #[test]
    fn partition_profiles_merge_in_partition_order() {
        let profile = DynamicProfile::default();
        profile.record(
            1,
            DynamicProfilePartition {
                values: vec![
                    ("shared".to_string(), "partition-1".to_string()),
                    ("z".to_string(), "last".to_string()),
                ],
                batches: 2,
                rows: 7,
                bytes: 70,
                min_rows: Some(3),
                max_rows: Some(4),
                min_bytes: Some(30),
                max_bytes: Some(40),
            },
        );
        profile.record(
            0,
            DynamicProfilePartition {
                values: vec![
                    ("a".to_string(), "first".to_string()),
                    ("shared".to_string(), "partition-0".to_string()),
                ],
                batches: 1,
                rows: 2,
                bytes: 20,
                min_rows: Some(2),
                max_rows: Some(2),
                min_bytes: Some(20),
                max_bytes: Some(20),
            },
        );

        let snapshot = profile.snapshot();
        assert_eq!(snapshot.batches, 3);
        assert_eq!(snapshot.rows, 9);
        assert_eq!(snapshot.bytes, 90);
        assert_eq!(snapshot.min_rows, Some(2));
        assert_eq!(snapshot.max_rows, Some(4));
        assert_eq!(snapshot.min_bytes, Some(20));
        assert_eq!(snapshot.max_bytes, Some(40));
        assert_eq!(
            snapshot.values.into_iter().collect::<Vec<_>>(),
            [
                ("a".to_string(), "first".to_string()),
                ("shared".to_string(), "partition-1".to_string()),
                ("z".to_string(), "last".to_string()),
            ]
        );
    }
}
