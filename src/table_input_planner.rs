// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Relation-valued arguments for VGI table functions.
//!
//! DataFusion's ordinary table-function hook receives a slice of logical
//! expressions. Consequently `(SELECT a, b)` in an argument position is
//! planned as a scalar subquery and rejected for returning more than one
//! column before [`crate::VgiTableFunction`] can see it. DataFusion 55's
//! [`RelationPlanner`] is the intended extension point for custom `FROM`
//! relations, so intercept known VGI calls while their SQL subquery is still a
//! relation and feed the complete plan into the existing table-input provider.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use datafusion::common::{plan_err, DFSchema, Result as DFResult, TableReference};
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::ast::{
    Expr as SQLExpr, FunctionArg, FunctionArgExpr, ObjectName, ObjectNamePart, TableFactor,
};

use crate::{table_input::TableArgument, VgiRuntime};

#[derive(Debug)]
struct VgiTableInputPlanner {
    session_id: String,
}

impl RelationPlanner for VgiTableInputPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DFResult<RelationPlanning> {
        let original = relation.clone();
        let TableFactor::Table {
            name,
            alias,
            args: Some(arguments),
            sample: None,
            ..
        } = relation
        else {
            return Ok(RelationPlanning::Original(Box::new(original)));
        };

        let function_name = object_name(&name);
        let Some(function) =
            crate::session::registered_vgi_table_function(&self.session_id, &function_name)
        else {
            return Ok(RelationPlanning::Original(Box::new(original)));
        };

        // Find the relation before converting any other argument. This keeps
        // ordinary producer calls on DataFusion's native UDTF path and avoids
        // changing the behavior of non-VGI table functions.
        let table_positions = arguments
            .args
            .iter()
            .enumerate()
            .filter_map(|(index, argument)| match argument {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::Subquery(query))) => {
                    Some((index, query.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if table_positions.is_empty() {
            return Ok(RelationPlanning::Original(Box::new(original)));
        }
        if table_positions.len() > 1 {
            return plan_err!("a VGI call may take at most one table argument");
        }
        let (table_index, query) = table_positions
            .into_iter()
            .next()
            .expect("checked one table argument");

        // Planning a Derived table preserves every output field. In contrast,
        // sending this Query through sql_to_expr would invoke scalar-subquery
        // validation and discard the relation semantics we need.
        let input = context.plan(TableFactor::Derived {
            lateral: false,
            subquery: query,
            alias: None,
            sample: None,
        })?;
        let table_arg = TableArgument {
            index: table_index,
            plan: Arc::new(input),
        };

        let mut scalar_exprs = Vec::with_capacity(arguments.args.len().saturating_sub(1));
        for (index, argument) in arguments.args.into_iter().enumerate() {
            if index == table_index {
                continue;
            }
            let FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) = argument else {
                return plan_err!("Unsupported function argument type: {argument}");
            };
            scalar_exprs.push(context.sql_expr_to_logical_expr(expr, &DFSchema::empty())?);
        }

        let provider = function.bind_table_argument(table_arg, &scalar_exprs)?;
        let plan = LogicalPlanBuilder::scan(
            TableReference::Bare {
                table: format!("{function_name}()").into(),
            },
            provider_as_source(provider),
            None,
        )?
        .build()?;
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(
            plan, alias,
        ))))
    }
}

fn object_name(name: &ObjectName) -> String {
    name.0
        .iter()
        .filter_map(|part| match part {
            ObjectNamePart::Identifier(identifier) => Some(identifier.value.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Install the planner once per live session.
pub(crate) fn ensure_registered(ctx: &SessionContext, runtime: &Arc<VgiRuntime>) -> DFResult<()> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Weak<VgiRuntime>>>> = OnceLock::new();
    let sessions = SESSIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sessions = sessions.lock().unwrap();
    sessions.retain(|_, runtime| runtime.strong_count() > 0);
    let session_id = ctx.session_id();
    if sessions.get(&session_id).and_then(Weak::upgrade).is_some() {
        return Ok(());
    }
    ctx.register_relation_planner(Arc::new(VgiTableInputPlanner {
        session_id: session_id.clone(),
    }))?;
    sessions.insert(session_id, Arc::downgrade(runtime));
    Ok(())
}
