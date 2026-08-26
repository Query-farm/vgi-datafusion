// Copyright 2025, 2026 Query Farm LLC - https://query.farm

use super::*;

fn namespace(
    expressions: &[&str],
    relation_functions: &[&str],
    relations: &[&str],
) -> SqlCatalogNamespace {
    let schema = SqlMacroNamespace {
        expression_functions: expressions
            .iter()
            .map(|name| name.to_ascii_lowercase())
            .collect(),
        relation_functions: relation_functions
            .iter()
            .map(|name| name.to_ascii_lowercase())
            .collect(),
        relations: relations
            .iter()
            .map(|name| name.to_ascii_lowercase())
            .collect(),
    };
    SqlCatalogNamespace {
        schemas: [("main".to_string(), schema)].into_iter().collect(),
    }
}

fn cross_schema_namespace() -> SqlCatalogNamespace {
    SqlCatalogNamespace {
        schemas: [
            (
                "main".to_string(),
                SqlMacroNamespace {
                    expression_functions: ["local_scalar".to_string()].into_iter().collect(),
                    relation_functions: ["local_rows".to_string()].into_iter().collect(),
                    relations: ["local_view".to_string()].into_iter().collect(),
                },
            ),
            (
                "analytics".to_string(),
                SqlMacroNamespace {
                    expression_functions: ["score".to_string()].into_iter().collect(),
                    relation_functions: ["events_for".to_string()].into_iter().collect(),
                    relations: ["report_view".to_string(), "facts".to_string()]
                        .into_iter()
                        .collect(),
                },
            ),
        ]
        .into_iter()
        .collect(),
    }
}

fn rendered(body: &SqlMacroBody) -> String {
    match body {
        SqlMacroBody::Scalar(expression) => expression.to_string(),
        SqlMacroBody::Table(query) => query.to_string(),
        SqlMacroBody::Invalid(error) => panic!("macro body did not parse: {error}"),
    }
}

#[test]
fn scalar_macro_qualifies_only_same_schema_worker_functions() {
    let mut body = parse_macro_body(
        SqlMacroKind::Scalar,
        "same_name(x) + abs(x) + other.main.same_name(x)",
    );
    qualify_macro_body(
        &mut body,
        "attached",
        "main",
        &namespace(&["same_name"], &[], &[]),
    );

    let sql = rendered(&body);
    assert!(sql.contains("attached.main.same_name(x)"), "{sql}");
    assert!(sql.contains("abs(x)"), "built-in was rewritten: {sql}");
    assert!(
        sql.contains("other.main.same_name(x)"),
        "explicit qualification changed: {sql}"
    );
    assert!(!sql.contains("attached.main.abs"), "{sql}");
}

#[test]
fn scalar_macro_qualifies_recognized_cross_schema_functions() {
    let mut body = parse_macro_body(
        SqlMacroKind::Scalar,
        "local_scalar(x) + analytics.score(x) + analytics.abs(x) + other.analytics.score(x)",
    );
    qualify_macro_body(&mut body, "attached", "main", &cross_schema_namespace());

    let sql = rendered(&body);
    assert!(sql.contains("attached.main.local_scalar(x)"), "{sql}");
    assert!(sql.contains("attached.analytics.score(x)"), "{sql}");
    assert!(
        sql.contains("analytics.abs(x)"),
        "unrecognized two-part builtin changed: {sql}"
    );
    assert!(
        sql.contains("other.analytics.score(x)"),
        "already three-part function changed: {sql}"
    );
}

#[test]
fn table_macro_preserves_ctes_and_builtins_but_qualifies_worker_calls() {
    let mut body = parse_macro_body(
        SqlMacroKind::Table,
        "WITH events AS (SELECT * FROM range(n)) \
         SELECT same_name(range) FROM events \
         UNION ALL SELECT same_name(n) FROM sequence(0)",
    );
    qualify_macro_body(
        &mut body,
        "attached",
        "main",
        &namespace(&["same_name"], &["sequence"], &["events"]),
    );

    let sql = rendered(&body);
    assert!(sql.contains("FROM range(n)"), "built-in changed: {sql}");
    assert!(sql.contains("FROM events"), "CTE changed: {sql}");
    assert!(
        sql.contains("FROM attached.main.sequence(0)"),
        "worker table function was not qualified: {sql}"
    );
    assert_eq!(sql.matches("attached.main.same_name").count(), 2, "{sql}");
    assert!(
        !sql.contains("attached.main.events"),
        "CTE was rewritten: {sql}"
    );
}

#[test]
fn cte_scope_does_not_hide_an_outer_worker_relation() {
    let mut body = parse_macro_body(
        SqlMacroKind::Table,
        "SELECT * FROM \
         (WITH events AS (SELECT * FROM range(1)) SELECT * FROM events) nested \
         CROSS JOIN events",
    );
    qualify_macro_body(
        &mut body,
        "attached",
        "main",
        &namespace(&[], &[], &["events"]),
    );

    let sql = rendered(&body);
    assert!(
        sql.contains("SELECT * FROM events"),
        "inner CTE changed: {sql}"
    );
    assert!(
        sql.contains("CROSS JOIN attached.main.events"),
        "outer worker relation was hidden by nested CTE scope: {sql}"
    );
}

#[test]
fn non_recursive_cte_visibility_is_definition_ordered() {
    let mut self_named = parse_macro_body(
        SqlMacroKind::Table,
        "WITH events AS (SELECT * FROM events) SELECT * FROM events",
    );
    qualify_macro_body(
        &mut self_named,
        "attached",
        "main",
        &namespace(&[], &[], &["events"]),
    );
    let sql = rendered(&self_named);
    assert!(
        sql.contains("events AS (SELECT * FROM attached.main.events)"),
        "a non-recursive CTE incorrectly hid the worker relation in its own definition: {sql}"
    );
    assert!(
        sql.ends_with("SELECT * FROM events"),
        "the CTE was not visible to the outer query: {sql}"
    );

    let mut ordered = parse_macro_body(
        SqlMacroKind::Table,
        "WITH first AS (SELECT * FROM later), \
              later AS (SELECT * FROM range(1)), \
              third AS (SELECT * FROM later) \
         SELECT * FROM third",
    );
    qualify_macro_body(
        &mut ordered,
        "attached",
        "main",
        &namespace(&[], &[], &["later"]),
    );
    let sql = rendered(&ordered);
    assert!(
        sql.contains("first AS (SELECT * FROM attached.main.later)"),
        "a later CTE was visible too early: {sql}"
    );
    assert!(
        sql.contains("third AS (SELECT * FROM later)"),
        "a preceding CTE was not visible: {sql}"
    );
}

#[test]
fn catalog_owned_view_qualifies_cross_schema_tables_functions_and_views() {
    let mut statements = DFParser::parse_sql(
        "WITH report_view AS (SELECT * FROM range(1)) \
         SELECT analytics.score(r.n) \
         FROM analytics.report_view r \
         CROSS JOIN analytics.events_for(3) e \
         CROSS JOIN analytics.facts f \
         CROSS JOIN report_view c \
         CROSS JOIN other.analytics.report_view kept",
    )
    .unwrap();
    let DFStatement::Statement(mut statement) = statements.pop_front().unwrap() else {
        panic!("view definition did not parse as a SQL statement");
    };
    qualify_catalog_sql(
        statement.as_mut(),
        "attached",
        "main",
        &cross_schema_namespace(),
    );

    let sql = statement.to_string();
    assert!(sql.contains("attached.analytics.score(r.n)"), "{sql}");
    assert!(
        sql.contains("FROM attached.analytics.report_view r"),
        "cross-schema view was not qualified: {sql}"
    );
    assert!(
        sql.contains("CROSS JOIN attached.analytics.events_for(3) e"),
        "cross-schema table function was not qualified: {sql}"
    );
    assert!(
        sql.contains("CROSS JOIN attached.analytics.facts f"),
        "cross-schema table was not qualified: {sql}"
    );
    assert!(
        sql.contains("CROSS JOIN report_view c"),
        "CTE was rewritten: {sql}"
    );
    assert!(
        sql.contains("other.analytics.report_view kept"),
        "already three-part view changed: {sql}"
    );
    assert!(sql.contains("FROM range(1)"), "builtin changed: {sql}");
}
