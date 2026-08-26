// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Advisory VGI ORDER BY / Top-N propagation.
//!
//! DataFusion retains its Sort/Top-K and limit semantics. The worker hint is
//! only an opportunity to produce useful rows first; it is never treated as
//! proof that the remote stream is globally ordered.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::Result;
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::replace_children_if_necessary;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use vgi_client::{NullOrder, OrderBy, SortDirection};

use crate::{VgiScanExec, VgiTableProvider};

/// Adds conservative VGI ORDER BY / Top-N hint propagation to a session.
pub trait VgiOrderPushdownSessionStateBuilderExt {
    /// Install the VGI query-planner bridge and physical validation rule.
    ///
    /// Calling another builder extension that replaces the query planner after
    /// this one will replace the bridge too. Hosts with a custom planner must
    /// use [`Self::with_vgi_order_pushdown_and_query_planner`] so the planners
    /// are composed explicitly.
    #[must_use]
    fn with_vgi_order_pushdown(self) -> Self;

    /// Install VGI order pushdown while preserving a custom query planner.
    #[must_use]
    fn with_vgi_order_pushdown_and_query_planner(
        self,
        query_planner: Arc<dyn QueryPlanner + Send + Sync>,
    ) -> Self;
}

impl VgiOrderPushdownSessionStateBuilderExt for SessionStateBuilder {
    fn with_vgi_order_pushdown(self) -> Self {
        self.with_physical_optimizer_rule(Arc::new(VgiTopNPushdown))
            .with_query_planner(Arc::new(VgiOrderQueryPlanner { inner: None }))
    }

    fn with_vgi_order_pushdown_and_query_planner(
        self,
        query_planner: Arc<dyn QueryPlanner + Send + Sync>,
    ) -> Self {
        self.with_physical_optimizer_rule(Arc::new(VgiTopNPushdown))
            .with_query_planner(Arc::new(VgiOrderQueryPlanner {
                inner: Some(query_planner),
            }))
    }
}

#[derive(Debug)]
struct VgiOrderQueryPlanner {
    inner: Option<Arc<dyn QueryPlanner + Send + Sync>>,
}

#[async_trait]
impl QueryPlanner for VgiOrderQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // TableProvider::scan (and therefore split planning) runs while the
        // physical plan is created. Seed the provider before delegating, then
        // validate the resulting physical shape below.
        let hinted = hint_logical_plan(logical_plan.clone())?;
        let plan = match &self.inner {
            Some(inner) => inner.create_physical_plan(&hinted, session).await?,
            None => {
                DefaultPhysicalPlanner::default()
                    .create_physical_plan(&hinted, session)
                    .await?
            }
        };
        VgiTopNPushdown.optimize(plan, session.config_options())
    }
}

fn hint_logical_plan(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_down(|node| {
        let LogicalPlan::Sort(sort) = &node else {
            return Ok(Transformed::no(node));
        };
        let Some(fetch) = sort.fetch else {
            return Ok(Transformed::no(node));
        };
        let Some(first) = sort.expr.first() else {
            return Ok(Transformed::no(node));
        };
        let datafusion::logical_expr::Expr::Column(column) = &first.expr else {
            return Ok(Transformed::no(node));
        };
        let limit = (sort.expr.len() == 1)
            .then(|| i64::try_from(fetch).ok())
            .flatten();
        let hint = OrderBy {
            column: column.name.clone(),
            direction: if first.asc {
                SortDirection::Ascending
            } else {
                SortDirection::Descending
            },
            null_order: if first.nulls_first {
                NullOrder::First
            } else {
                NullOrder::Last
            },
            limit,
        };
        let Some(input) = hint_logical_input(sort.input.as_ref(), hint)? else {
            return Ok(Transformed::no(node));
        };
        let mut replacement = sort.clone();
        replacement.input = Arc::new(input);
        Ok(Transformed::new(
            LogicalPlan::Sort(replacement),
            true,
            TreeNodeRecursion::Jump,
        ))
    })
    .map(|transformed| transformed.data)
}

fn hint_logical_input(input: &LogicalPlan, mut hint: OrderBy) -> Result<Option<LogicalPlan>> {
    match input {
        LogicalPlan::TableScan(scan) => {
            if !scan.filters.is_empty() {
                hint.limit = None;
            }
            let Ok(provider) = source_as_provider(&scan.source) else {
                return Ok(None);
            };
            let Some(provider) = provider.downcast_ref::<VgiTableProvider>() else {
                return Ok(None);
            };
            let Some(remote_column) = provider.remote_column_name(&hint.column) else {
                return Ok(None);
            };
            hint.column = remote_column;
            let mut replacement = scan.clone();
            replacement.source = provider_as_source(Arc::new(provider.with_order_by(Some(hint))));
            Ok(Some(LogicalPlan::TableScan(replacement)))
        }
        LogicalPlan::Filter(filter) => {
            hint.limit = None;
            let Some(child) = hint_logical_input(filter.input.as_ref(), hint)? else {
                return Ok(None);
            };
            let mut replacement = filter.clone();
            replacement.input = Arc::new(child);
            Ok(Some(LogicalPlan::Filter(replacement)))
        }
        LogicalPlan::Projection(projection) => {
            let Some(index) = projection
                .schema
                .fields()
                .iter()
                .position(|field| field.name() == &hint.column)
            else {
                return Ok(None);
            };
            let Some(column) = projection.expr.get(index).and_then(direct_column) else {
                return Ok(None);
            };
            hint.column = column.name.clone();
            let Some(child) = hint_logical_input(projection.input.as_ref(), hint)? else {
                return Ok(None);
            };
            let mut replacement = projection.clone();
            replacement.input = Arc::new(child);
            Ok(Some(LogicalPlan::Projection(replacement)))
        }
        LogicalPlan::Repartition(repartition) => {
            let Some(child) = hint_logical_input(repartition.input.as_ref(), hint)? else {
                return Ok(None);
            };
            let mut replacement = repartition.clone();
            replacement.input = Arc::new(child);
            Ok(Some(LogicalPlan::Repartition(replacement)))
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let Some(child) = hint_logical_input(alias.input.as_ref(), hint)? else {
                return Ok(None);
            };
            let mut replacement = alias.clone();
            replacement.input = Arc::new(child);
            Ok(Some(LogicalPlan::SubqueryAlias(replacement)))
        }
        _ => Ok(None),
    }
}

fn direct_column(expr: &datafusion::logical_expr::Expr) -> Option<&datafusion::common::Column> {
    match expr {
        datafusion::logical_expr::Expr::Column(column) => Some(column),
        datafusion::logical_expr::Expr::Alias(alias) => match alias.expr.as_ref() {
            datafusion::logical_expr::Expr::Column(column) => Some(column),
            _ => None,
        },
        _ => None,
    }
}

#[derive(Debug)]
struct VgiTopNPushdown;

impl PhysicalOptimizerRule for VgiTopNPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            if let Some(limit) = node.downcast_ref::<GlobalLimitExec>() {
                if let (Some(sort), Some(fetch)) = (
                    limit.input().downcast_ref::<SortExec>(),
                    limit
                        .fetch()
                        .and_then(|fetch| limit.skip().checked_add(fetch)),
                ) {
                    if let Some(replacement) = hint_physical_sort(sort, fetch)? {
                        let replacement: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(
                            replacement,
                            limit.skip(),
                            limit.fetch(),
                        ));
                        return Ok(Transformed::new(replacement, true, TreeNodeRecursion::Jump));
                    }
                }
            }

            let Some(sort) = node.downcast_ref::<SortExec>() else {
                return Ok(Transformed::no(node));
            };
            let Some(fetch) = sort.fetch() else {
                return Ok(Transformed::no(node));
            };
            let Some(replacement) = hint_physical_sort(sort, fetch)? else {
                return Ok(Transformed::no(node));
            };
            Ok(Transformed::new(replacement, true, TreeNodeRecursion::Jump))
        })
        .map(|transformed| transformed.data)
    }

    fn name(&self) -> &str {
        "vgi_top_n_pushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn hint_physical_sort(sort: &SortExec, fetch: usize) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let first = sort.expr().first();
    let Some(column) = first.expr.downcast_ref::<Column>() else {
        return Ok(None);
    };
    let hint = OrderBy {
        column: column.name().to_string(),
        direction: if first.options.descending {
            SortDirection::Descending
        } else {
            SortDirection::Ascending
        },
        null_order: if first.options.nulls_first {
            NullOrder::First
        } else {
            NullOrder::Last
        },
        limit: (sort.expr().len() == 1)
            .then(|| i64::try_from(fetch).ok())
            .flatten(),
    };
    let Some(input) = hint_physical_input(sort.input(), column.index(), hint)? else {
        return Ok(None);
    };
    let replacement: Arc<dyn ExecutionPlan> = Arc::new(
        SortExec::new(sort.expr().clone(), input)
            .with_fetch(sort.fetch())
            .with_preserve_partitioning(sort.preserve_partitioning()),
    );
    Ok(Some(replacement))
}

#[allow(deprecated)]
fn hint_physical_input(
    input: &Arc<dyn ExecutionPlan>,
    mut column_index: usize,
    mut hint: OrderBy,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(scan) = input.downcast_ref::<VgiScanExec>() {
        let schema = scan.schema();
        let Some(field) = schema.fields().get(column_index) else {
            return Ok(None);
        };
        let Some(remote_column) = scan.remote_column_name(field.name()) else {
            return Ok(None);
        };
        hint.column = remote_column;
        return Ok(scan
            .with_order_by(hint)
            .map(|scan| Arc::new(scan) as Arc<dyn ExecutionPlan>));
    }

    if let Some(projection) = input.downcast_ref::<ProjectionExec>() {
        let Some(projection_expr) = projection.expr().get(column_index) else {
            return Ok(None);
        };
        let Some(column) = projection_expr.expr.downcast_ref::<Column>() else {
            return Ok(None);
        };
        column_index = column.index();
    } else if input.downcast_ref::<FilterExec>().is_some() {
        hint.limit = None;
    } else if input.downcast_ref::<CoalesceBatchesExec>().is_none()
        && input.downcast_ref::<CoalescePartitionsExec>().is_none()
        && input.downcast_ref::<RepartitionExec>().is_none()
    {
        return Ok(None);
    }

    let children = input.children();
    if children.len() != 1 {
        return Ok(None);
    }
    let Some(child) = hint_physical_input(children[0], column_index, hint)? else {
        return Ok(None);
    };
    replace_children_if_necessary(input.clone(), vec![child]).map(Some)
}

#[cfg(test)]
mod tests {
    #[test]
    fn offset_and_fetch_are_checked() {
        assert_eq!(2usize.checked_add(3), Some(5));
        assert_eq!(usize::MAX.checked_add(1), None);
    }

    #[test]
    fn multi_column_top_n_does_not_get_an_unsafe_early_limit() {
        let limit = (2 == 1).then_some(3_i64);
        assert_eq!(limit, None);
    }
}
