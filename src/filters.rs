// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Translating DataFusion predicates into VGI's pushdown-filter blob.
//!
//! # The wire shape
//!
//! An Arrow IPC batch where column 0 is a one-row string holding a JSON array of
//! filter specs — its field carries `vgi_filter_version` metadata — and columns
//! 1.. carry the constants, so a spec's `value_ref: N` names column `N + 1` and
//! reads row 0. IN sets travel as VGI v2 side join-key batches, referenced by a
//! `join_keys` spec. Top-level specs combine with AND.
//!
//! # Why every filter is reported `Inexact`
//!
//! DataFusion re-applies an `Inexact` filter above the scan, so the worker's
//! answer is treated as a *superset* and the rows are filtered again locally.
//! That makes pushdown a pure optimisation: a worker that ignores the blob,
//! applies it partially, or gets it subtly wrong still yields correct results.
//!
//! This is a real advantage over the DuckDB extension, which has no such
//! setting — DuckDB does not re-apply a pushed predicate, so a filtered scan
//! there must be row-exact and every pushdown decision is load-bearing for
//! correctness. Here it is load-bearing only for speed. Claiming `Exact` would
//! throw that away for no gain we can currently measure.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::physical_expr::expressions::{
    BinaryExpr as PhysicalBinaryExpr, CastExpr, Column as PhysicalColumn, InListExpr,
    IsNotNullExpr, IsNullExpr, Literal,
};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};

/// A constant hoisted out of a predicate, to become its own typed Arrow column.
///
/// VGI's wire format carries Arrow arrays, so normalizing all integers to
/// `Int64`, floats to `Float64`, and strings to `Utf8` needlessly lost unsigned
/// ranges, decimal scale, timestamp time zones, and dictionary identity. Keep
/// the DataFusion scalar intact for every type VGI's Arrow comparison path can
/// evaluate losslessly.
struct Constant(ScalarValue);

impl Constant {
    fn from_scalar(v: &ScalarValue) -> Option<Self> {
        // `col = NULL` is not `col IS NULL`, and pushing it as an equality
        // would ask the worker for the wrong rows.
        if v.is_null() || !supported_constant_type(&v.data_type()) {
            return None;
        }
        Some(Self(v.clone()))
    }

    fn to_array(&self) -> DFResult<ArrayRef> {
        self.0.to_array_of_size(1)
    }
}

/// Types handled losslessly by both Arrow IPC and VGI's Arrow comparison
/// evaluator. Nested list/struct/map/union values remain local because Arrow's
/// scalar comparison kernel deliberately has no nested null semantics.
fn supported_constant_type(data_type: &DataType) -> bool {
    use DataType::*;

    match data_type {
        Boolean
        | Float16
        | Float32
        | Float64
        | Decimal32(_, _)
        | Decimal64(_, _)
        | Decimal128(_, _)
        | Decimal256(_, _)
        | Int8
        | Int16
        | Int32
        | Int64
        | UInt8
        | UInt16
        | UInt32
        | UInt64
        | Utf8
        | Utf8View
        | LargeUtf8
        | Binary
        | BinaryView
        | FixedSizeBinary(_)
        | LargeBinary
        | Date32
        | Date64
        | Time32(_)
        | Time64(_)
        | Timestamp(_, _)
        | Interval(_)
        | Duration(_) => true,
        Dictionary(key, value) => {
            matches!(
                key.as_ref(),
                Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
            ) && supported_dictionary_value_type(value)
        }
        _ => false,
    }
}

fn supported_dictionary_value_type(data_type: &DataType) -> bool {
    !matches!(data_type, DataType::Dictionary(_, _)) && supported_constant_type(data_type)
}

/// Accumulates specs and their constants while walking the expression tree.
struct Builder<'a> {
    schema: &'a SchemaRef,
    constants: Vec<Constant>,
    /// Bind-schema positions of every column a spec referenced, in first-seen
    /// order.
    ///
    /// The caller needs these because DataFusion may push a filter on a column
    /// the projection does NOT include. The worker keys a pushed filter by its
    /// position in what it emits, so unless those columns are requested too the
    /// filter lands on whichever column happens to occupy that slot — silently
    /// filtering on the wrong data rather than failing.
    referenced_columns: Vec<i64>,
    /// Side IPC batches for physical `IN` expressions produced by joins.
    join_keys: Vec<Vec<u8>>,
}

impl<'a> Builder<'a> {
    fn add_constant(&mut self, c: Constant) -> usize {
        self.constants.push(c);
        self.constants.len() - 1
    }

    fn checkpoint(&self) -> (usize, usize, usize) {
        (
            self.constants.len(),
            self.referenced_columns.len(),
            self.join_keys.len(),
        )
    }

    fn rollback(&mut self, checkpoint: (usize, usize, usize)) {
        self.constants.truncate(checkpoint.0);
        self.referenced_columns.truncate(checkpoint.1);
        self.join_keys.truncate(checkpoint.2);
    }

    /// Serialize a non-empty, single-column membership set as a VGI v2 side
    /// join-key batch. Validation and IPC construction happen before mutating
    /// the builder, so an unsupported value cannot leak a projected column or
    /// a partial side batch into a different pushable predicate.
    fn membership(&mut self, name: String, values: Vec<ScalarValue>) -> Option<serde_json::Value> {
        if values.is_empty() || values.iter().any(|v| Constant::from_scalar(v).is_none()) {
            return None;
        }
        let keys = join_keys_ipc(&name, values).ok()?;
        let index = self.column_index(&name)?;
        self.join_keys.push(keys);
        Some(serde_json::json!({
            "type": "join_keys",
            "column_name": name,
            "column_index": index,
            "keys_column": name,
        }))
    }

    /// The column's position in the table's bind-time schema.
    ///
    /// Reported alongside the name because the wire spec carries both; the
    /// worker matches on the name, and the index is relative to the *unprojected*
    /// schema the worker itself bound.
    fn column_index(&mut self, name: &str) -> Option<i64> {
        let idx = self
            .schema
            .index_of(name)
            .ok()
            .map(|i| i64::try_from(i).unwrap_or(0))?;
        // Recorded on the way through, so a column can only be referenced by a
        // spec if it also reaches `filter_columns`. Deriving the two separately
        // is what lets them drift.
        if !self.referenced_columns.contains(&idx) {
            self.referenced_columns.push(idx);
        }
        Some(idx)
    }

    /// Translate one predicate, or `None` if it is not expressible.
    fn build(&mut self, expr: &Expr) -> Option<serde_json::Value> {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => self.binary(left, *op, right),
            Expr::IsNull(inner) => {
                let name = column_name(inner)?;
                Some(serde_json::json!({
                    "type": "is_null",
                    "column_name": name,
                    "column_index": self.column_index(&name)?,
                }))
            }
            Expr::IsNotNull(inner) => {
                let name = column_name(inner)?;
                Some(serde_json::json!({
                    "type": "is_not_null",
                    "column_name": name,
                    "column_index": self.column_index(&name)?,
                }))
            }
            Expr::InList(list) => {
                // A negated IN would need the worker to invert the set; leave it
                // to DataFusion rather than guess at the semantics.
                if list.negated {
                    return None;
                }
                let name = column_name(&list.expr)?;
                let mut values = Vec::with_capacity(list.list.len());
                for item in &list.list {
                    values.push(literal(item)?);
                }
                self.membership(name, values)
            }
            _ => None,
        }
    }

    /// Translate a physical predicate snapshot. DataFusion's post-optimizer
    /// dynamic-filter hook uses physical expressions, whereas ordinary
    /// `TableProvider` pushdown arrives as logical [`Expr`]s.
    fn build_physical(
        &mut self,
        expr: &dyn PhysicalExpr,
        side_join_keys: bool,
    ) -> Option<serde_json::Value> {
        if let Some(binary) = expr.downcast_ref::<PhysicalBinaryExpr>() {
            return self.physical_binary(binary, side_join_keys);
        }
        if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
            let name = physical_column_name(is_null.arg().as_ref())?;
            return Some(serde_json::json!({
                "type": "is_null",
                "column_name": name,
                "column_index": self.column_index(&name)?,
            }));
        }
        if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
            let name = physical_column_name(is_not_null.arg().as_ref())?;
            return Some(serde_json::json!({
                "type": "is_not_null",
                "column_name": name,
                "column_index": self.column_index(&name)?,
            }));
        }
        if let Some(list) = expr.downcast_ref::<InListExpr>() {
            if list.negated() || list.is_empty() {
                return None;
            }
            let name = physical_column_name(list.expr().as_ref())?;
            let values = list
                .list()
                .iter()
                .map(|value| physical_literal(value.as_ref()))
                .collect::<Option<Vec<_>>>()?;
            if side_join_keys {
                return self.membership(name, values);
            }
            // Join-key side batches cannot ride continuation metadata. The
            // initial snapshot uses the side batch above; later snapshots keep
            // any accompanying range predicates but omit membership rather
            // than emitting the obsolete inline IN shape.
            return None;
        }
        None
    }

    /// Top-level predicates are implicitly conjoined by the VGI wire format,
    /// so flatten physical AND trees. Besides matching DataFusion's generated
    /// min/max + membership filters, this also supports multi-column join keys
    /// without inventing a cross-column conjunction node (which VGI does not
    /// have).
    fn build_physical_top(
        &mut self,
        expr: &dyn PhysicalExpr,
        side_join_keys: bool,
        out: &mut Vec<serde_json::Value>,
    ) {
        if let Some(binary) = expr.downcast_ref::<PhysicalBinaryExpr>() {
            if *binary.op() == Operator::And {
                self.build_physical_top(binary.left().as_ref(), side_join_keys, out);
                self.build_physical_top(binary.right().as_ref(), side_join_keys, out);
                return;
            }
        }
        // DataFusion 55 represents a small multi-column hash-join filter as
        // `struct(col_a, col_b, ...) IN ((a1, b1, ...), ...)`. VGI's existing
        // join-key channel carries scalar columns rather than tuple values.
        // Sending each tuple field as an independent top-level membership set
        // is nevertheless safe: it admits the Cartesian product of the field
        // sets, a superset of the original tuples. The hash join still tests
        // the correlated keys locally, so it restores the exact tuple result.
        // This gives the worker useful pruning without inventing a new
        // protocol expression.
        if side_join_keys {
            if let Some(list) = expr.downcast_ref::<InListExpr>() {
                if let Some(specs) = self.struct_memberships(list) {
                    out.extend(specs);
                    return;
                }
            }
        }
        if let Some(spec) = self.build_physical(expr, side_join_keys) {
            out.push(spec);
        }
    }

    /// Decompose DataFusion's multi-column struct membership into independent
    /// VGI join-key sets. This is only called for init-time side batches.
    ///
    /// The operation is atomic: any unsupported/null tuple field rolls back
    /// every side batch and referenced column produced so far.
    fn struct_memberships(&mut self, list: &InListExpr) -> Option<Vec<serde_json::Value>> {
        if list.negated() || list.is_empty() {
            return None;
        }
        let columns = physical_struct_columns(list.expr().as_ref())?;
        if columns.len() < 2
            || columns
                .iter()
                .enumerate()
                .any(|(index, name)| columns[..index].contains(name))
        {
            return None;
        }

        let mut field_values = vec![Vec::with_capacity(list.len()); columns.len()];
        for value in list.list() {
            let ScalarValue::Struct(tuple) = physical_literal(value.as_ref())? else {
                return None;
            };
            if tuple.len() != 1 || tuple.is_null(0) || tuple.num_columns() != columns.len() {
                return None;
            }
            for (index, values) in field_values.iter_mut().enumerate() {
                values.push(ScalarValue::try_from_array(tuple.column(index).as_ref(), 0).ok()?);
            }
        }

        let checkpoint = self.checkpoint();
        let mut specs = Vec::with_capacity(columns.len());
        for (name, values) in columns.into_iter().zip(field_values) {
            let Some(spec) = self.membership(name, values) else {
                self.rollback(checkpoint);
                return None;
            };
            specs.push(spec);
        }
        Some(specs)
    }

    fn physical_binary(
        &mut self,
        binary: &PhysicalBinaryExpr,
        side_join_keys: bool,
    ) -> Option<serde_json::Value> {
        let op = *binary.op();
        if op == Operator::Or && side_join_keys {
            if let Some((name, values)) = physical_or_equalities(binary) {
                return self.membership(name, values);
            }
        }
        if op == Operator::And || op == Operator::Or {
            let checkpoint = self.checkpoint();
            let left = self.build_physical(binary.left().as_ref(), side_join_keys);
            let right = self.build_physical(binary.right().as_ref(), side_join_keys);
            let combined = match op {
                Operator::And => match (left, right) {
                    (Some(left), Some(right)) => self.conjunction("and", left, right),
                    (Some(one), None) | (None, Some(one)) => Some(one),
                    (None, None) => None,
                },
                Operator::Or => match (left, right) {
                    (Some(left), Some(right)) => self.conjunction("or", left, right),
                    _ => None,
                },
                _ => unreachable!(),
            };
            if combined.is_none()
                || (op == Operator::Or && combined.as_ref().is_some_and(contains_join_keys))
            {
                self.rollback(checkpoint);
                return None;
            }
            return combined;
        }
        let (name, value, op) = match (
            physical_column_name(binary.left().as_ref()),
            physical_literal(binary.right().as_ref()),
        ) {
            (Some(name), Some(value)) => (name, value, op),
            _ => match (
                physical_literal(binary.left().as_ref()),
                physical_column_name(binary.right().as_ref()),
            ) {
                (Some(value), Some(name)) => (name, value, flip(op)?),
                _ => return None,
            },
        };
        let index = self.column_index(&name)?;
        let token = op_token(op)?;
        let value_ref = self.add_constant(Constant::from_scalar(&value)?);
        Some(serde_json::json!({
            "type": "constant",
            "column_name": name,
            "column_index": index,
            "op": token,
            "value_ref": value_ref,
        }))
    }

    /// Build an `and`/`or` node over two already-translated children.
    ///
    /// # Every node carries a column, conjunctions included
    ///
    /// The worker reads `column_name` and `column_index` off *every* spec
    /// before it looks at the type (`table_filter_pushdown.py`), so a
    /// conjunction that omits them is a hard parse failure — a `KeyError`
    /// surfacing as `FilterDeserializationError: 'column_name'`, which is what
    /// `WHERE n > 2 OR n < 1` produced.
    ///
    /// The value to carry is the children's own column, because the protocol
    /// models a conjunction as a filter *on one column* — DuckDB's
    /// `CONJUNCTION_OR` is exactly that, and the extension passes the parent's
    /// column through to its children.
    ///
    /// So a conjunction spanning two different columns has no representation
    /// here and is not pushed down. For `AND` that costs nothing: the caller
    /// already keeps whichever side translates, and a superset is safe under
    /// `Inexact`. For `OR` it must be all-or-nothing, so a cross-column `OR`
    /// stays with DataFusion.
    fn conjunction(
        &mut self,
        kind: &str,
        left: serde_json::Value,
        right: serde_json::Value,
    ) -> Option<serde_json::Value> {
        let name = left.get("column_name")?.as_str()?.to_string();
        if right.get("column_name")?.as_str()? != name {
            return None;
        }
        Some(serde_json::json!({
            "type": kind,
            "column_name": name,
            "column_index": self.column_index(&name)?,
            "children": [left, right],
        }))
    }

    fn binary(&mut self, left: &Expr, op: Operator, right: &Expr) -> Option<serde_json::Value> {
        // AND keeps whichever side is expressible: dropping one conjunct still
        // yields a superset, which `Inexact` makes safe.
        if op == Operator::And {
            let l = self.build(left);
            let r = self.build(right);
            return match (l, r) {
                (Some(a), Some(b)) => Some(self.conjunction("and", a, b)?),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            };
        }
        // OR must keep BOTH sides or none. Dropping one disjunct narrows the
        // result — the worker would omit rows the predicate accepts, and no
        // amount of re-filtering above the scan brings them back.
        if op == Operator::Or {
            if let Some((name, values)) = logical_or_equalities(left, right) {
                return self.membership(name, values);
            }
            let checkpoint = self.checkpoint();
            let combined = self
                .build(left)
                .zip(self.build(right))
                .and_then(|(left, right)| self.conjunction("or", left, right));
            if combined.is_none() || combined.as_ref().is_some_and(contains_join_keys) {
                self.rollback(checkpoint);
                return None;
            }
            return combined;
        }

        // `col <op> literal`, or `literal <op> col` with the operator flipped.
        let (name, value, op) = match (column_name(left), literal(right)) {
            (Some(n), Some(v)) => (n, v, op),
            _ => match (literal(left), column_name(right)) {
                (Some(v), Some(n)) => (n, v, flip(op)?),
                _ => return None,
            },
        };
        let index = self.column_index(&name)?;
        let token = op_token(op)?;
        let value_ref = self.add_constant(Constant::from_scalar(&value)?);
        Some(serde_json::json!({
            "type": "constant",
            "column_name": name,
            "column_index": index,
            "op": token,
            "value_ref": value_ref,
        }))
    }
}

fn contains_join_keys(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => {
            object.get("type").and_then(serde_json::Value::as_str) == Some("join_keys")
                || object.values().any(contains_join_keys)
        }
        serde_json::Value::Array(values) => values.iter().any(contains_join_keys),
        _ => false,
    }
}

fn column_name(e: &Expr) -> Option<String> {
    match e {
        Expr::Column(c) => Some(c.name.clone()),
        // A cast wrapping a column still names that column; the worker compares
        // against the constant we send, and `Inexact` covers any coercion
        // difference.
        Expr::Cast(c) => column_name(&c.expr),
        _ => None,
    }
}

fn literal(e: &Expr) -> Option<ScalarValue> {
    match e {
        Expr::Literal(v, _) => Some(v.clone()),
        Expr::Cast(c) => literal(&c.expr).and_then(|v| v.cast_to(c.field.data_type()).ok()),
        _ => None,
    }
}

fn physical_column_name(expr: &dyn PhysicalExpr) -> Option<String> {
    if let Some(column) = expr.downcast_ref::<PhysicalColumn>() {
        return Some(column.name().to_string());
    }
    expr.downcast_ref::<CastExpr>()
        .and_then(|cast| physical_column_name(cast.expr.as_ref()))
}

fn physical_literal(expr: &dyn PhysicalExpr) -> Option<ScalarValue> {
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Some(literal.value().clone());
    }
    expr.downcast_ref::<CastExpr>().and_then(|cast| {
        physical_literal(cast.expr.as_ref()).and_then(|v| v.cast_to(cast.cast_type()).ok())
    })
}

/// Return the scan columns assembled by DataFusion's physical `struct`
/// function, if every struct field is a direct (optionally cast) column.
fn physical_struct_columns(expr: &dyn PhysicalExpr) -> Option<Vec<String>> {
    let function = expr.downcast_ref::<ScalarFunctionExpr>()?;
    if function.name() != "struct" {
        return None;
    }
    function
        .args()
        .iter()
        .map(|argument| physical_column_name(argument.as_ref()))
        .collect()
}

/// Flatten `column = literal OR ...` on one column into a typed membership
/// set. This is the only disjunction shape that VGI's existing `join_keys`
/// representation can express without inventing tuple or expression-filter
/// protocol semantics.
fn logical_or_equalities(left: &Expr, right: &Expr) -> Option<(String, Vec<ScalarValue>)> {
    let mut terms = Vec::new();
    collect_logical_or_equalities(left, &mut terms)?;
    collect_logical_or_equalities(right, &mut terms)?;
    same_column_membership(terms)
}

fn collect_logical_or_equalities(expr: &Expr, out: &mut Vec<(String, ScalarValue)>) -> Option<()> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };
    if binary.op == Operator::Or {
        collect_logical_or_equalities(&binary.left, out)?;
        collect_logical_or_equalities(&binary.right, out)?;
        return Some(());
    }
    if binary.op != Operator::Eq {
        return None;
    }
    let term = match (column_name(&binary.left), literal(&binary.right)) {
        (Some(name), Some(value)) => (name, value),
        _ => match (literal(&binary.left), column_name(&binary.right)) {
            (Some(value), Some(name)) => (name, value),
            _ => return None,
        },
    };
    out.push(term);
    Some(())
}

fn physical_or_equalities(binary: &PhysicalBinaryExpr) -> Option<(String, Vec<ScalarValue>)> {
    let mut terms = Vec::new();
    collect_physical_or_equalities(binary.left().as_ref(), &mut terms)?;
    collect_physical_or_equalities(binary.right().as_ref(), &mut terms)?;
    same_column_membership(terms)
}

fn collect_physical_or_equalities(
    expr: &dyn PhysicalExpr,
    out: &mut Vec<(String, ScalarValue)>,
) -> Option<()> {
    let binary = expr.downcast_ref::<PhysicalBinaryExpr>()?;
    if *binary.op() == Operator::Or {
        collect_physical_or_equalities(binary.left().as_ref(), out)?;
        collect_physical_or_equalities(binary.right().as_ref(), out)?;
        return Some(());
    }
    if *binary.op() != Operator::Eq {
        return None;
    }
    let term = match (
        physical_column_name(binary.left().as_ref()),
        physical_literal(binary.right().as_ref()),
    ) {
        (Some(name), Some(value)) => (name, value),
        _ => match (
            physical_literal(binary.left().as_ref()),
            physical_column_name(binary.right().as_ref()),
        ) {
            (Some(value), Some(name)) => (name, value),
            _ => return None,
        },
    };
    out.push(term);
    Some(())
}

fn same_column_membership(terms: Vec<(String, ScalarValue)>) -> Option<(String, Vec<ScalarValue>)> {
    let name = terms.first()?.0.clone();
    if terms.iter().any(|(term_name, _)| term_name != &name) {
        return None;
    }
    Some((name, terms.into_iter().map(|(_, value)| value).collect()))
}

/// The VGI op token for a comparison, or `None` for anything else.
fn op_token(op: Operator) -> Option<&'static str> {
    Some(match op {
        Operator::Eq => "eq",
        Operator::NotEq => "ne",
        Operator::Lt => "lt",
        Operator::LtEq => "le",
        Operator::Gt => "gt",
        Operator::GtEq => "ge",
        _ => return None,
    })
}

/// Mirror a comparison so `5 < x` becomes `x > 5`.
fn flip(op: Operator) -> Option<Operator> {
    Some(match op {
        Operator::Eq => Operator::Eq,
        Operator::NotEq => Operator::NotEq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        _ => return None,
    })
}

/// Whether a predicate can be expressed on the wire at all.
pub(crate) fn is_pushable(expr: &Expr, schema: &SchemaRef) -> bool {
    let mut b = Builder {
        schema,
        constants: Vec::new(),
        referenced_columns: Vec::new(),
        join_keys: Vec::new(),
    };
    b.build(expr).is_some()
}

/// Serialize the expressible predicates into the pushdown blob.
///
/// Returns `None` when nothing could be expressed, which the caller should treat
/// as "send no filters" rather than "send an empty filter set".
/// A serialized pushdown, plus the bind-schema columns it reads.
///
/// The two travel together on purpose. A caller holding only the blob cannot
/// tell which columns the worker will need in order to evaluate it, and a
/// filter on a column outside the projection then evaluates against whatever
/// column occupies that position instead — wrong rows, no error. See
/// `ScanOptions::filter_columns`.
#[derive(Debug, Clone, Default)]
pub(crate) struct Pushdown {
    /// The filter blob, or `None` when nothing was expressible.
    pub blob: Option<Vec<u8>>,
    /// Bind-schema positions the specs referenced, first-seen order.
    pub columns: Vec<i64>,
    /// VGI v2 single-column IPC batches referenced by `join_keys` specs.
    pub join_keys: Vec<Vec<u8>>,
}

impl Pushdown {
    /// Stable cache identity for both the filter AST and its out-of-line IN
    /// values. The wire blob alone no longer identifies a scan once the same
    /// `join_keys` spec can name different side batches.
    pub(crate) fn cache_identity(&self) -> Option<Vec<u8>> {
        let blob = self.blob.as_ref()?;
        if self.join_keys.is_empty() {
            return Some(blob.clone());
        }
        let mut identity = b"vgi-pushdown-cache-v2".to_vec();
        append_identity_part(&mut identity, blob);
        for keys in &self.join_keys {
            append_identity_part(&mut identity, keys);
        }
        Some(identity)
    }
}

fn append_identity_part(identity: &mut Vec<u8>, part: &[u8]) {
    identity.extend_from_slice(&(part.len() as u64).to_le_bytes());
    identity.extend_from_slice(part);
}

pub(crate) fn serialize(exprs: &[Expr], schema: &SchemaRef) -> DFResult<Pushdown> {
    let mut b = Builder {
        schema,
        constants: Vec::new(),
        referenced_columns: Vec::new(),
        join_keys: Vec::new(),
    };
    let specs: Vec<serde_json::Value> = exprs.iter().filter_map(|e| b.build(e)).collect();
    if specs.is_empty() {
        return Ok(Pushdown::default());
    }
    let mut pushdown = finish_pushdown(specs, b.constants, b.referenced_columns)?;
    pushdown.join_keys = b.join_keys;
    Ok(pushdown)
}

/// Serialize snapshots of DataFusion runtime filters.
///
/// With `side_join_keys`, physical `IN` lists use VGI's `join_keys` side IPC
/// batches (schema metadata `vgi_join_keys_version=2`). Continuation ticks do
/// not carry those side batches, so callers use `false` there to retain only
/// the independently expressible constant/range parts of a snapshot.
pub(crate) fn serialize_physical(
    exprs: &[Arc<dyn PhysicalExpr>],
    schema: &SchemaRef,
    side_join_keys: bool,
) -> DFResult<Pushdown> {
    let mut b = Builder {
        schema,
        constants: Vec::new(),
        referenced_columns: Vec::new(),
        join_keys: Vec::new(),
    };
    let mut specs = Vec::new();
    for expr in exprs {
        b.build_physical_top(expr.as_ref(), side_join_keys, &mut specs);
    }
    let mut pushdown = finish_pushdown(specs, b.constants, b.referenced_columns)?;
    pushdown.join_keys = b.join_keys;
    Ok(pushdown)
}

fn finish_pushdown(
    specs: Vec<serde_json::Value>,
    constants: Vec<Constant>,
    referenced_columns: Vec<i64>,
) -> DFResult<Pushdown> {
    if specs.is_empty() {
        return Ok(Pushdown::default());
    }
    let json = serde_json::Value::Array(specs).to_string();

    // Column 0 is the spec JSON, and its field carries the version metadata the
    // worker looks for. Columns 1.. are the constants in `value_ref` order.
    let mut fields = vec![
        Field::new("filter_spec", DataType::Utf8, false).with_metadata(
            [("vgi_filter_version".to_string(), "1".to_string())]
                .into_iter()
                .collect(),
        ),
    ];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![json]))];
    for (i, c) in constants.iter().enumerate() {
        let arr = c.to_array()?;
        fields.push(Field::new(
            format!("value_{i}"),
            arr.data_type().clone(),
            true,
        ));
        columns.push(arr);
    }

    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w =
            datafusion::arrow::ipc::writer::StreamWriter::try_new(&mut buf, &batch.schema())?;
        w.write(&batch)?;
        w.finish()?;
    }
    Ok(Pushdown {
        blob: Some(buf),
        columns: referenced_columns,
        join_keys: Vec::new(),
    })
}

fn join_keys_ipc(column: &str, values: Vec<ScalarValue>) -> DFResult<Vec<u8>> {
    let array = ScalarValue::iter_to_array(values)?;
    let schema = Arc::new(
        Schema::new(vec![Field::new(column, array.data_type().clone(), true)]).with_metadata(
            [("vgi_join_keys_version".to_string(), "2".to_string())]
                .into_iter()
                .collect(),
        ),
    );
    let batch = RecordBatch::try_new(schema, vec![array])?;
    let mut out = Vec::new();
    {
        let mut writer =
            datafusion::arrow::ipc::writer::StreamWriter::try_new(&mut out, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(out)
}

/// Conjoin already-serialized static and runtime pushdowns.
///
/// Both wire blobs are one-row IPC batches. Appending the runtime specs and
/// constant columns requires shifting their `value_ref` indices by the number
/// of constants already present in the static batch.
pub(crate) fn merge(left: &Pushdown, right: &Pushdown) -> DFResult<Pushdown> {
    match (&left.blob, &right.blob) {
        (None, None) => return Ok(Pushdown::default()),
        (Some(_), None) => return Ok(left.clone()),
        (None, Some(_)) => return Ok(right.clone()),
        (Some(_), Some(_)) => {}
    }
    let left_batch = read_pushdown(left.blob.as_deref().expect("checked"))?;
    let right_batch = read_pushdown(right.blob.as_deref().expect("checked"))?;
    let left_constants = left_batch.num_columns() - 1;
    let mut merged_specs = parse_specs(&left_batch)?;
    let mut right_specs = parse_specs(&right_batch)?;
    for spec in &mut right_specs {
        shift_value_refs(spec, left_constants);
    }
    merged_specs.extend(right_specs);

    let mut fields = vec![left_batch.schema().field(0).clone()];
    let mut columns = vec![left_batch.column(0).clone()];
    for batch in [&left_batch, &right_batch] {
        for index in 1..batch.num_columns() {
            fields.push(Field::new(
                format!("value_{}", fields.len() - 1),
                batch.column(index).data_type().clone(),
                true,
            ));
            columns.push(batch.column(index).clone());
        }
    }
    columns[0] = Arc::new(StringArray::from(vec![serde_json::Value::Array(
        merged_specs,
    )
    .to_string()]));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    let mut out = Vec::new();
    {
        let mut writer =
            datafusion::arrow::ipc::writer::StreamWriter::try_new(&mut out, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    let mut referenced_columns = left.columns.clone();
    for column in &right.columns {
        if !referenced_columns.contains(column) {
            referenced_columns.push(*column);
        }
    }
    Ok(Pushdown {
        blob: Some(out),
        columns: referenced_columns,
        join_keys: left
            .join_keys
            .iter()
            .chain(&right.join_keys)
            .cloned()
            .collect(),
    })
}

fn read_pushdown(bytes: &[u8]) -> DFResult<RecordBatch> {
    let mut reader =
        datafusion::arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)?;
    reader.next().transpose()?.ok_or_else(|| {
        datafusion::common::DataFusionError::Execution("empty VGI filter IPC".into())
    })
}

fn parse_specs(batch: &RecordBatch) -> DFResult<Vec<serde_json::Value>> {
    let json = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::common::DataFusionError::Execution("VGI filter spec is not UTF-8".into())
        })?;
    serde_json::from_str(json.value(0)).map_err(|error| {
        datafusion::common::DataFusionError::Execution(format!("invalid VGI filter JSON: {error}"))
    })
}

fn shift_value_refs(value: &mut serde_json::Value, offset: usize) {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(reference) = object.get_mut("value_ref") {
                if let Some(index) = reference.as_u64() {
                    *reference = serde_json::json!(index + offset as u64);
                }
            }
            if let Some(serde_json::Value::Array(references)) = object.get_mut("value_refs") {
                for reference in references {
                    if let Some(index) = reference.as_u64() {
                        *reference = serde_json::json!(index + offset as u64);
                    }
                }
            }
            for child in object.values_mut() {
                shift_value_refs(child, offset);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                shift_value_refs(child, offset);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Array, Int64Array};
    use datafusion::common::config::ConfigOptions;
    use datafusion::common::scalar::ScalarStructBuilder;
    use datafusion::logical_expr::{col, lit};
    use datafusion::physical_expr::expressions::{
        col as physical_col, lit as physical_lit, BinaryExpr as PhysicalBinaryExpr, InListExpr,
    };

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn wide_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("u", DataType::UInt64, true),
            Field::new("d", DataType::Date32, true),
            Field::new(
                "ts",
                DataType::Timestamp(
                    datafusion::arrow::datatypes::TimeUnit::Nanosecond,
                    Some("America/New_York".into()),
                ),
                true,
            ),
            Field::new("amount", DataType::Decimal128(24, 6), true),
            Field::new("bytes", DataType::Binary, true),
            Field::new(
                "dict",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
        ]))
    }

    /// Decode a serialized blob back into (specs JSON, constant columns).
    fn decode(bytes: &[u8]) -> (serde_json::Value, RecordBatch) {
        let mut r = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(bytes),
            None,
        )
        .unwrap();
        let batch = r.next().unwrap().unwrap();
        let json = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
            .to_string();
        (serde_json::from_str(&json).unwrap(), batch)
    }

    #[test]
    fn a_simple_comparison_becomes_a_constant_filter() {
        let e = col("n").gt(lit(5i64));
        let bytes = serialize(&[e], &schema()).unwrap().blob.expect("pushable");
        let (specs, batch) = decode(&bytes);

        assert_eq!(specs[0]["type"], "constant");
        assert_eq!(specs[0]["column_name"], "n");
        assert_eq!(specs[0]["op"], "gt");
        assert_eq!(specs[0]["column_index"], 0);

        // value_ref 0 -> column 1, scalar at row 0.
        let vr = specs[0]["value_ref"].as_u64().unwrap() as usize;
        let v = batch.column(vr + 1);
        assert_eq!(v.len(), 1);
        assert_eq!(v.as_any().downcast_ref::<Int64Array>().unwrap().value(0), 5);
    }

    #[test]
    fn typed_constants_retain_their_arrow_types_on_the_wire() {
        let timezone: Arc<str> = "America/New_York".into();
        let dict = ScalarValue::Dictionary(
            Box::new(DataType::Int8),
            Box::new(ScalarValue::Utf8(Some("alpha".to_string()))),
        );
        let expressions = vec![
            col("u").eq(lit(ScalarValue::UInt64(Some(u64::MAX)))),
            col("d").eq(lit(ScalarValue::Date32(Some(20_000)))),
            col("ts").eq(lit(ScalarValue::TimestampNanosecond(
                Some(1_700_000_000_123_456_789),
                Some(timezone.clone()),
            ))),
            col("amount").eq(lit(ScalarValue::Decimal128(Some(12_345_678), 24, 6))),
            col("bytes").eq(lit(ScalarValue::Binary(Some(vec![0, 1, 255])))),
            col("dict").eq(lit(dict)),
        ];
        let pushdown = serialize(&expressions, &wide_schema()).unwrap();
        let (_, batch) = decode(pushdown.blob.as_deref().unwrap());
        assert_eq!(batch.column(1).data_type(), &DataType::UInt64);
        assert_eq!(batch.column(2).data_type(), &DataType::Date32);
        assert_eq!(
            batch.column(3).data_type(),
            &DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Nanosecond,
                Some(timezone)
            )
        );
        assert_eq!(batch.column(4).data_type(), &DataType::Decimal128(24, 6));
        assert_eq!(batch.column(5).data_type(), &DataType::Binary);
        assert_eq!(
            batch.column(6).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
        );
    }

    #[test]
    fn literal_casts_are_materialized_before_serialization() {
        let decimal_schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(18, 2),
            true,
        )]));
        let cast = Expr::Cast(datafusion::logical_expr::expr::Cast::new(
            Box::new(lit(1_250_i64)),
            DataType::Decimal128(18, 2),
        ));
        let pushdown = serialize(&[col("amount").eq(cast)], &decimal_schema).unwrap();
        let (_, batch) = decode(pushdown.blob.as_deref().unwrap());
        assert_eq!(batch.column(1).data_type(), &DataType::Decimal128(18, 2));

        let cast = Arc::new(CastExpr::new(
            physical_lit(1_250_i64),
            DataType::Decimal128(18, 2),
            None,
        )) as Arc<dyn PhysicalExpr>;
        let expression = Arc::new(PhysicalBinaryExpr::new(
            physical_col("amount", &decimal_schema).unwrap(),
            Operator::Eq,
            cast,
        )) as Arc<dyn PhysicalExpr>;
        let pushdown = serialize_physical(&[expression], &decimal_schema, false).unwrap();
        let (_, batch) = decode(pushdown.blob.as_deref().unwrap());
        assert_eq!(batch.column(1).data_type(), &DataType::Decimal128(18, 2));
    }

    #[test]
    fn nested_constants_remain_local() {
        let list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::Int64(Some(1))],
            &DataType::Int64,
        ));
        let nested_schema = Arc::new(Schema::new(vec![Field::new(
            "items",
            list.data_type(),
            true,
        )]));
        assert!(serialize(&[col("items").eq(lit(list))], &nested_schema)
            .unwrap()
            .blob
            .is_none());
    }

    #[test]
    fn the_version_metadata_rides_on_field_zero() {
        // The worker looks for this to know it can read the blob at all.
        let bytes = serialize(&[col("n").eq(lit(1i64))], &schema())
            .unwrap()
            .blob
            .unwrap();
        let (_, batch) = decode(&bytes);
        assert_eq!(
            batch.schema().field(0).metadata().get("vgi_filter_version"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn a_reversed_comparison_is_flipped_not_dropped() {
        // `5 < n` means the same as `n > 5`; reading it left-to-right would
        // push the wrong operator.
        let bytes = serialize(&[lit(5i64).lt(col("n"))], &schema())
            .unwrap()
            .blob
            .unwrap();
        let (specs, _) = decode(&bytes);
        assert_eq!(specs[0]["column_name"], "n");
        assert_eq!(specs[0]["op"], "gt");
    }

    #[test]
    fn every_comparison_operator_maps() {
        for (e, want) in [
            (col("n").eq(lit(1i64)), "eq"),
            (col("n").not_eq(lit(1i64)), "ne"),
            (col("n").lt(lit(1i64)), "lt"),
            (col("n").lt_eq(lit(1i64)), "le"),
            (col("n").gt(lit(1i64)), "gt"),
            (col("n").gt_eq(lit(1i64)), "ge"),
        ] {
            let bytes = serialize(&[e], &schema()).unwrap().blob.unwrap();
            let (specs, _) = decode(&bytes);
            assert_eq!(specs[0]["op"], want);
        }
    }

    #[test]
    fn null_checks_carry_no_constant() {
        let bytes = serialize(&[col("n").is_null()], &schema())
            .unwrap()
            .blob
            .unwrap();
        let (specs, batch) = decode(&bytes);
        assert_eq!(specs[0]["type"], "is_null");
        assert_eq!(batch.num_columns(), 1, "no value columns for a null check");

        let bytes = serialize(&[col("n").is_not_null()], &schema())
            .unwrap()
            .blob
            .unwrap();
        let (specs, _) = decode(&bytes);
        assert_eq!(specs[0]["type"], "is_not_null");
    }

    #[test]
    fn a_static_in_list_uses_v2_join_key_side_ipc() {
        let e = col("n").in_list(vec![lit(1i64), lit(2i64), lit(3i64)], false);
        let pushdown = serialize(&[e], &schema()).unwrap();
        let bytes = pushdown.blob.unwrap();
        let (specs, batch) = decode(&bytes);
        assert_eq!(specs[0]["type"], "join_keys");
        assert_eq!(specs[0]["keys_column"], "n");
        assert!(specs[0].get("value_ref").is_none());
        assert_eq!(batch.num_columns(), 1, "values ride in the side batch");
        assert_eq!(pushdown.join_keys.len(), 1);
        let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&pushdown.join_keys[0]),
            None,
        )
        .unwrap();
        let keys = reader.next().unwrap().unwrap();
        assert_eq!(keys.num_rows(), 3);
        assert_eq!(
            keys.schema().metadata().get("vgi_join_keys_version"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn typed_in_values_retain_decimal_precision_and_scale() {
        let values = vec![
            lit(ScalarValue::Decimal128(Some(1_250_000), 24, 6)),
            lit(ScalarValue::Decimal128(Some(2_500_000), 24, 6)),
        ];
        let pushdown = serialize(&[col("amount").in_list(values, false)], &wide_schema()).unwrap();
        let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&pushdown.join_keys[0]),
            None,
        )
        .unwrap();
        let keys = reader.next().unwrap().unwrap();
        assert_eq!(keys.column(0).data_type(), &DataType::Decimal128(24, 6));
        assert_eq!(keys.num_rows(), 2);
    }

    #[test]
    fn same_column_equality_or_becomes_one_membership_batch() {
        let expression = col("n")
            .eq(lit(1_i64))
            .or(lit(3_i64).eq(col("n")))
            .or(col("n").eq(lit(5_i64)));
        let pushdown = serialize(&[expression], &schema()).unwrap();
        let (specs, batch) = decode(pushdown.blob.as_deref().unwrap());
        assert_eq!(specs[0]["type"], "join_keys");
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(pushdown.join_keys.len(), 1);

        let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&pushdown.join_keys[0]),
            None,
        )
        .unwrap();
        let keys = reader.next().unwrap().unwrap();
        let values = keys
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.values(), &[1, 3, 5]);
    }

    #[test]
    fn equality_or_on_different_columns_is_not_tuple_membership() {
        let expression = col("n").eq(lit(1_i64)).or(col("name").eq(lit("one")));
        let pushdown = serialize(&[expression], &schema()).unwrap();
        assert!(pushdown.blob.is_none());
        assert!(pushdown.join_keys.is_empty());
    }

    #[test]
    fn a_negated_in_list_is_left_to_datafusion() {
        let e = col("n").in_list(vec![lit(1i64)], true);
        assert!(serialize(&[e], &schema()).unwrap().blob.is_none());
    }

    #[test]
    fn an_or_containing_join_keys_is_left_to_datafusion() {
        let left = col("n").in_list(vec![lit(1_i64), lit(2_i64)], false);
        let right = col("n").eq(lit(9_i64));
        let pushdown = serialize(&[left.or(right)], &schema()).unwrap();
        assert!(pushdown.blob.is_none());
        assert!(pushdown.columns.is_empty());
        assert!(pushdown.join_keys.is_empty());
    }

    #[test]
    fn join_key_values_participate_in_cache_identity() {
        let one = serialize(
            &[col("n").in_list(vec![lit(1_i64), lit(2_i64)], false)],
            &schema(),
        )
        .unwrap();
        let two = serialize(
            &[col("n").in_list(vec![lit(3_i64), lit(4_i64)], false)],
            &schema(),
        )
        .unwrap();
        assert_eq!(one.blob, two.blob, "the filter AST names the side batch");
        assert_ne!(one.cache_identity(), two.cache_identity());
    }

    #[test]
    fn and_keeps_whichever_side_is_expressible() {
        // Dropping a conjunct still yields a superset, which Inexact re-filters.
        let unsupported = col("n").gt(col("name")); // column-to-column
        let e = col("n").gt(lit(5i64)).and(unsupported);
        let bytes = serialize(&[e], &schema())
            .unwrap()
            .blob
            .expect("the pushable half survives");
        let (specs, _) = decode(&bytes);
        assert_eq!(specs[0]["type"], "constant");
        assert_eq!(specs[0]["op"], "gt");
    }

    #[test]
    fn or_is_dropped_entirely_unless_both_sides_translate() {
        // This is the asymmetry that matters. Keeping half a disjunction would
        // NARROW the result — the worker would omit rows the predicate accepts,
        // and re-filtering above the scan cannot bring them back.
        let unsupported = col("n").gt(col("name"));
        let e = col("n").gt(lit(5i64)).or(unsupported);
        assert!(
            serialize(&[e], &schema()).unwrap().blob.is_none(),
            "half an OR must not be pushed"
        );

        // Both sides expressible: pushed as an or.
        let both = col("n").gt(lit(5i64)).or(col("n").lt(lit(0i64)));
        let (specs, _) = decode(&serialize(&[both], &schema()).unwrap().blob.unwrap());
        assert_eq!(specs[0]["type"], "or");
        assert_eq!(specs[0]["children"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn a_null_literal_is_not_pushed_as_equality() {
        // `col = NULL` is not `col IS NULL`; pushing it would ask for the wrong
        // rows, and under three-valued logic it matches nothing.
        let e = col("n").eq(lit(ScalarValue::Int64(None)));
        assert!(serialize(&[e], &schema()).unwrap().blob.is_none());
    }

    #[test]
    fn a_column_to_column_comparison_is_not_pushable() {
        assert!(!is_pushable(&col("n").gt(col("name")), &schema()));
    }

    #[test]
    fn a_predicate_on_an_unknown_column_is_not_pushable() {
        // Can happen when a filter references something the scan does not
        // provide; sending an index we cannot resolve would be worse than
        // sending nothing.
        assert!(!is_pushable(&col("nope").eq(lit(1i64)), &schema()));
    }

    #[test]
    fn several_predicates_serialize_as_a_top_level_conjunction() {
        let bytes = serialize(
            &[col("n").gt(lit(1i64)), col("name").eq(lit("x"))],
            &schema(),
        )
        .unwrap()
        .blob
        .unwrap();
        let (specs, batch) = decode(&bytes);
        assert_eq!(
            specs.as_array().unwrap().len(),
            2,
            "top level ANDs on the worker"
        );
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(specs[1]["column_index"], 1, "name is the second column");
    }

    #[test]
    fn nothing_pushable_serializes_to_nothing() {
        assert!(serialize(&[col("n").gt(col("name"))], &schema())
            .unwrap()
            .blob
            .is_none());
        assert!(serialize(&[], &schema()).unwrap().blob.is_none());
    }

    /// The columns a pushdown reads come back with it, so a caller cannot
    /// request the blob without learning what it needs.
    ///
    /// This is the whole reason `Pushdown` is a struct rather than a bare blob.
    /// The worker keys a pushed filter by the column's position in what it
    /// EMITS, so if the projection omits a filtered column the predicate is
    /// evaluated against whichever column takes that slot — the scan returns
    /// wrong rows and nothing errors. A caller that only has the blob has no way
    /// to know which columns to add.
    #[test]
    fn reports_the_columns_its_specs_read() {
        // `name` is at bind index 1; a projection of [0] would leave it out.
        let pd = serialize(&[col("name").eq(lit("x"))], &schema()).unwrap();
        assert!(pd.blob.is_some());
        assert_eq!(pd.columns, vec![1]);
    }

    /// Several predicates over the same column report it once, in first-seen
    /// order — the union is a projection, and a repeat would request a
    /// duplicate column.
    #[test]
    fn dedups_and_orders_reported_columns() {
        let pd = serialize(
            &[
                col("name").eq(lit("x")),
                col("n").gt(lit(1i64)),
                col("name").not_eq(lit("y")),
            ],
            &schema(),
        )
        .unwrap();
        assert_eq!(pd.columns, vec![1, 0]);
    }

    /// A predicate that is not expressible reports no columns either. Reporting
    /// one here would widen the scan for a filter that never reached the worker.
    #[test]
    fn an_unpushable_predicate_reports_no_columns() {
        let pd = serialize(&[col("n").eq(col("name"))], &schema()).unwrap();
        assert!(pd.blob.is_none());
        assert!(pd.columns.is_empty());
    }

    #[test]
    fn physical_runtime_filter_serializes_with_join_key_side_ipc() {
        let needle = physical_col("n", &schema()).unwrap();
        let list = InListExpr::try_new(
            needle,
            vec![
                physical_lit(1_i64),
                physical_lit(3_i64),
                physical_lit(5_i64),
            ],
            false,
            &schema(),
        )
        .unwrap();
        let dynamic =
            serialize_physical(&[Arc::new(list) as Arc<dyn PhysicalExpr>], &schema(), true)
                .unwrap();
        let (specs, batch) = decode(dynamic.blob.as_deref().unwrap());
        assert_eq!(specs[0]["type"], "join_keys");
        assert_eq!(specs[0]["keys_column"], "n");
        assert_eq!(batch.num_columns(), 1, "values ride in the side batch");
        assert_eq!(dynamic.join_keys.len(), 1);

        let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&dynamic.join_keys[0]),
            None,
        )
        .unwrap();
        let keys = reader.next().unwrap().unwrap();
        assert_eq!(keys.num_rows(), 3);
        assert_eq!(
            keys.schema().metadata().get("vgi_join_keys_version"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn physical_struct_membership_becomes_safe_per_column_join_keys() {
        let fields = vec![
            Field::new("c0", DataType::Int64, true),
            Field::new("c1", DataType::Utf8, true),
        ]
        .into();
        let tuple_expr = Arc::new(ScalarFunctionExpr::new(
            "struct",
            datafusion::functions::core::r#struct(),
            vec![
                physical_col("n", &schema()).unwrap(),
                physical_col("name", &schema()).unwrap(),
            ],
            Arc::new(Field::new("struct", DataType::Struct(fields), true)),
            Arc::new(ConfigOptions::default()),
        )) as Arc<dyn PhysicalExpr>;
        let tuple = |n, name| {
            ScalarStructBuilder::new()
                .with_scalar(
                    Field::new("c0", DataType::Int64, true),
                    ScalarValue::from(n),
                )
                .with_scalar(
                    Field::new("c1", DataType::Utf8, true),
                    ScalarValue::from(name),
                )
                .build()
                .unwrap()
        };
        let list = InListExpr::try_new(
            tuple_expr,
            vec![
                physical_lit(tuple(1_i64, "one")),
                physical_lit(tuple(3_i64, "three")),
            ],
            false,
            &schema(),
        )
        .unwrap();

        let dynamic =
            serialize_physical(&[Arc::new(list) as Arc<dyn PhysicalExpr>], &schema(), true)
                .unwrap();
        let (specs, batch) = decode(dynamic.blob.as_deref().unwrap());
        assert_eq!(specs.as_array().unwrap().len(), 2);
        assert_eq!(specs[0]["type"], "join_keys");
        assert_eq!(specs[0]["column_name"], "n");
        assert_eq!(specs[1]["type"], "join_keys");
        assert_eq!(specs[1]["column_name"], "name");
        assert_eq!(batch.num_columns(), 1, "all values use side batches");
        assert_eq!(dynamic.columns, vec![0, 1]);
        assert_eq!(dynamic.join_keys.len(), 2);

        let mut n_reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&dynamic.join_keys[0]),
            None,
        )
        .unwrap();
        let n_keys = n_reader.next().unwrap().unwrap();
        assert_eq!(
            n_keys
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1, 3]
        );

        let mut name_reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&dynamic.join_keys[1]),
            None,
        )
        .unwrap();
        let name_keys = name_reader.next().unwrap().unwrap();
        let names = name_keys
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "one");
        assert_eq!(names.value(1), "three");
    }

    #[test]
    fn physical_struct_membership_rolls_back_after_a_late_null_field() {
        let fields = vec![
            Field::new("c0", DataType::Int64, true),
            Field::new("c1", DataType::Utf8, true),
        ]
        .into();
        let tuple_expr = Arc::new(ScalarFunctionExpr::new(
            "struct",
            datafusion::functions::core::r#struct(),
            vec![
                physical_col("n", &schema()).unwrap(),
                physical_col("name", &schema()).unwrap(),
            ],
            Arc::new(Field::new("struct", DataType::Struct(fields), true)),
            Arc::new(ConfigOptions::default()),
        )) as Arc<dyn PhysicalExpr>;
        let tuple = ScalarStructBuilder::new()
            .with_scalar(
                Field::new("c0", DataType::Int64, true),
                ScalarValue::from(1_i64),
            )
            .with_scalar(
                Field::new("c1", DataType::Utf8, true),
                ScalarValue::Utf8(None),
            )
            .build()
            .unwrap();
        let list =
            InListExpr::try_new(tuple_expr, vec![physical_lit(tuple)], false, &schema()).unwrap();

        let dynamic =
            serialize_physical(&[Arc::new(list) as Arc<dyn PhysicalExpr>], &schema(), true)
                .unwrap();
        assert!(dynamic.blob.is_none());
        assert!(dynamic.columns.is_empty());
        assert!(dynamic.join_keys.is_empty());
    }

    #[test]
    fn physical_equality_or_uses_the_existing_join_key_channel() {
        let equality = |value| {
            Arc::new(PhysicalBinaryExpr::new(
                physical_col("n", &schema()).unwrap(),
                Operator::Eq,
                physical_lit(value),
            )) as Arc<dyn PhysicalExpr>
        };
        let expression = Arc::new(PhysicalBinaryExpr::new(
            Arc::new(PhysicalBinaryExpr::new(
                equality(2_i64),
                Operator::Or,
                equality(4_i64),
            )),
            Operator::Or,
            equality(6_i64),
        )) as Arc<dyn PhysicalExpr>;
        let dynamic = serialize_physical(&[expression], &schema(), true).unwrap();
        let (specs, _) = decode(dynamic.blob.as_deref().unwrap());
        assert_eq!(specs[0]["type"], "join_keys");
        assert_eq!(dynamic.join_keys.len(), 1);

        let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(&dynamic.join_keys[0]),
            None,
        )
        .unwrap();
        let keys = reader.next().unwrap().unwrap();
        let values = keys
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.values(), &[2, 4, 6]);
    }

    #[test]
    fn physical_runtime_range_filter_can_ride_continuation_ticks() {
        let expression = Arc::new(PhysicalBinaryExpr::new(
            physical_col("n", &schema()).unwrap(),
            Operator::Gt,
            physical_lit(10_i64),
        )) as Arc<dyn PhysicalExpr>;
        let dynamic = serialize_physical(&[expression], &schema(), false).unwrap();
        let (specs, batch) = decode(dynamic.blob.as_deref().unwrap());
        assert_eq!(specs[0]["type"], "constant");
        assert_eq!(specs[0]["op"], "gt");
        assert_eq!(batch.num_columns(), 2);
        assert!(dynamic.join_keys.is_empty());
    }

    #[test]
    fn physical_in_filter_is_omitted_from_continuation_metadata() {
        let needle = physical_col("n", &schema()).unwrap();
        let list = InListExpr::try_new(
            needle,
            vec![physical_lit(2_i64), physical_lit(4_i64)],
            false,
            &schema(),
        )
        .unwrap();
        let dynamic =
            serialize_physical(&[Arc::new(list) as Arc<dyn PhysicalExpr>], &schema(), false)
                .unwrap();
        assert!(dynamic.blob.is_none());
        assert!(dynamic.columns.is_empty());
        assert!(dynamic.join_keys.is_empty());
    }

    #[test]
    fn merging_static_and_runtime_filters_rebases_value_references() {
        let static_filter = serialize(&[col("n").gt(lit(1_i64))], &schema()).unwrap();
        let expression = Arc::new(PhysicalBinaryExpr::new(
            physical_col("n", &schema()).unwrap(),
            Operator::Lt,
            physical_lit(10_i64),
        )) as Arc<dyn PhysicalExpr>;
        let runtime = serialize_physical(&[expression], &schema(), false).unwrap();
        let merged = merge(&static_filter, &runtime).unwrap();
        let (specs, batch) = decode(merged.blob.as_deref().unwrap());
        assert_eq!(specs.as_array().unwrap().len(), 2);
        assert_eq!(specs[0]["value_ref"], 0);
        assert_eq!(specs[1]["value_ref"], 1);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn merging_pushdowns_preserves_join_key_side_batches() {
        let membership = serialize(
            &[col("n").in_list(vec![lit(1_i64), lit(3_i64)], false)],
            &schema(),
        )
        .unwrap();
        let bound = serialize(&[col("n").lt(lit(10_i64))], &schema()).unwrap();
        let merged = merge(&membership, &bound).unwrap();
        assert_eq!(merged.join_keys.len(), 1);
        let (specs, _) = decode(merged.blob.as_deref().unwrap());
        assert_eq!(specs[0]["type"], "join_keys");
        assert_eq!(specs[1]["type"], "constant");
    }
}
