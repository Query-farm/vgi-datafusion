// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Translating DataFusion predicates into VGI's pushdown-filter blob.
//!
//! # The wire shape
//!
//! An Arrow IPC batch where column 0 is a one-row string holding a JSON array of
//! filter specs — its field carries `vgi_filter_version` metadata — and columns
//! 1.. carry the constants, so a spec's `value_ref: N` names column `N + 1` and
//! reads row 0. Top-level specs combine with AND.
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

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};

/// A constant hoisted out of a predicate, to become its own column.
enum Constant {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl Constant {
    fn from_scalar(v: &ScalarValue) -> Option<Self> {
        Some(match v {
            ScalarValue::Int8(Some(x)) => Self::Int(i64::from(*x)),
            ScalarValue::Int16(Some(x)) => Self::Int(i64::from(*x)),
            ScalarValue::Int32(Some(x)) => Self::Int(i64::from(*x)),
            ScalarValue::Int64(Some(x)) => Self::Int(*x),
            ScalarValue::UInt8(Some(x)) => Self::Int(i64::from(*x)),
            ScalarValue::UInt16(Some(x)) => Self::Int(i64::from(*x)),
            ScalarValue::UInt32(Some(x)) => Self::Int(i64::from(*x)),
            ScalarValue::Float32(Some(x)) => Self::Float(f64::from(*x)),
            ScalarValue::Float64(Some(x)) => Self::Float(*x),
            ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Self::Text(s.clone()),
            ScalarValue::Boolean(Some(b)) => Self::Bool(*b),
            // A NULL constant is deliberately unsupported: `col = NULL` is not
            // `col IS NULL`, and pushing it as an equality would ask the worker
            // for the wrong rows.
            _ => return None,
        })
    }

    fn to_array(&self) -> ArrayRef {
        match self {
            Self::Int(v) => Arc::new(Int64Array::from(vec![*v])),
            Self::Float(v) => Arc::new(Float64Array::from(vec![*v])),
            Self::Text(v) => Arc::new(StringArray::from(vec![v.clone()])),
            Self::Bool(v) => Arc::new(BooleanArray::from(vec![*v])),
        }
    }
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
}

impl<'a> Builder<'a> {
    fn add_constant(&mut self, c: Constant) -> usize {
        self.constants.push(c);
        self.constants.len() - 1
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
                let index = self.column_index(&name)?;
                // Every element becomes its own constant column, mirroring how
                // the C++ extension emits an IN set.
                let mut refs = Vec::with_capacity(list.list.len());
                for item in &list.list {
                    let Expr::Literal(v, _) = item else {
                        return None;
                    };
                    refs.push(self.add_constant(Constant::from_scalar(v)?));
                }
                if refs.is_empty() {
                    return None;
                }
                Some(serde_json::json!({
                    "type": "in",
                    "column_name": name,
                    "column_index": index,
                    "value_refs": refs,
                }))
            }
            _ => None,
        }
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
            let l = self.build(left)?;
            let r = self.build(right)?;
            return self.conjunction("or", l, r);
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
        Expr::Cast(c) => literal(&c.expr),
        _ => None,
    }
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
}

pub(crate) fn serialize(exprs: &[Expr], schema: &SchemaRef) -> DFResult<Pushdown> {
    let mut b = Builder {
        schema,
        constants: Vec::new(),
        referenced_columns: Vec::new(),
    };
    let specs: Vec<serde_json::Value> = exprs.iter().filter_map(|e| b.build(e)).collect();
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
    for (i, c) in b.constants.iter().enumerate() {
        let arr = c.to_array();
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
        columns: b.referenced_columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Array;
    use datafusion::logical_expr::{col, lit};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
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
    fn an_in_list_hoists_every_element() {
        let e = col("n").in_list(vec![lit(1i64), lit(2i64), lit(3i64)], false);
        let bytes = serialize(&[e], &schema()).unwrap().blob.unwrap();
        let (specs, batch) = decode(&bytes);
        assert_eq!(specs[0]["type"], "in");
        assert_eq!(specs[0]["value_refs"].as_array().unwrap().len(), 3);
        assert_eq!(batch.num_columns(), 4, "spec column plus three constants");
    }

    #[test]
    fn a_negated_in_list_is_left_to_datafusion() {
        let e = col("n").in_list(vec![lit(1i64)], true);
        assert!(serialize(&[e], &schema()).unwrap().blob.is_none());
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
}
