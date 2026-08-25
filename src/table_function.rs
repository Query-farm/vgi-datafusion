// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Table functions that take arguments — `SELECT * FROM sequence(10)`.
//!
//! # Why this is separate from [`VgiTableProvider`]
//!
//! A [`TableProvider`] is a *table*: it has a schema and no call syntax. Most
//! VGI table functions are not that — they take arguments, and their output
//! schema depends on them. `sequence(10)` and `sequence(10, 2)` are the same
//! function with different binds.
//!
//! DataFusion models exactly this with [`TableFunctionImpl`]: the planner hands
//! over the call's argument expressions and gets back a `TableProvider` bound to
//! *that* call. So the split here mirrors the worker's own: zero-argument
//! functions are reachable as bare tables through the schema provider, and
//! argument-taking ones are reachable as table functions through this.
//!
//! # Named arguments
//!
//! VGI functions take named arguments too — `sequence(10, batch_size := 5)` is
//! ordinary in the DuckDB extension. DataFusion's `TableFactor::Table` planner
//! rejects that form before [`TableFunctionImpl`] can see it —
//!
//! ```text
//! SELECT * FROM generate_series(1, 3, step => 1);
//! Error during planning: Unsupported function argument type: step => 1
//! ```
//!
//! [`crate::sql`] therefore rewrites each named argument to a private one-field
//! struct literal before planning. [`TableFunctionArgs`] carries that ordinary
//! expression through, and this module unwraps it into [`Arguments::named`].
//! Calls made directly through `SessionContext::sql` do not pass through that
//! compatibility rewrite and retain DataFusion's limitation.
//!
//! # Arguments must be constants
//!
//! A bind happens during planning, before any row exists, so an argument has to
//! be foldable to a literal then. A column reference is refused by name.

use std::sync::Arc;

use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::common::{plan_err, Result as DFResult, ScalarValue};
use datafusion::logical_expr::Expr;
use vgi_client::{ArgValue, Arguments};

use crate::{catalog::TableFunctionMetadata, VgiConnection, VgiTableProvider};

/// One VGI table function, callable with arguments.
#[derive(Debug)]
pub struct VgiTableFunction {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    metadata: Option<TableFunctionMetadata>,
}

impl VgiTableFunction {
    /// Wrap a worker function as a DataFusion table function.
    pub(crate) fn new(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        metadata: Option<TableFunctionMetadata>,
    ) -> Self {
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            metadata,
        }
    }

    /// The name this function is registered under.
    pub fn function(&self) -> &str {
        &self.function
    }
}

impl TableFunctionImpl for VgiTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> DFResult<Arc<dyn TableProvider>> {
        // A TABLE argument makes this an exchange-mode call: the subquery
        // becomes the input stream and the remaining arguments stay bind
        // arguments, which is how the extension models it too.
        if let Some(table_arg) = crate::table_input::TableArgument::find(args.exprs())? {
            let scalars: Vec<Expr> = args
                .exprs()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != table_arg.index)
                .map(|(_, e)| e.clone())
                .collect();
            let arguments = to_arguments(&self.function, &scalars)?;
            let conn = self.conn.clone();
            let catalog = self.catalog.clone();
            let schema_name = self.schema_name.clone();
            let function = self.function.clone();
            let buffered = self.metadata.as_ref().is_some_and(|m| m.buffered);
            return crate::run_blocking_planner_call(move || {
                crate::table_input::VgiTableInputProvider::bind_blocking(
                    conn,
                    &catalog,
                    &schema_name,
                    &function,
                    arguments,
                    table_arg,
                    buffered,
                )
            })
            .map(|p| p as Arc<dyn TableProvider>);
        }

        // A blended table-in/out function exposes its positional arguments as
        // per-row input columns. A literal call is therefore a one-row exchange,
        // not a producer scan. Binding it as a producer gives the worker no
        // input and can continuation-loop forever while waiting for a row.
        if let Some(metadata) = self.metadata.as_ref().filter(|m| m.input_from_args) {
            let (arguments, input_schema, input) =
                blended_literal_input(&self.function, &metadata.specs, args.exprs())?;
            let conn = self.conn.clone();
            let catalog = self.catalog.clone();
            let schema_name = self.schema_name.clone();
            let function = self.function.clone();
            return crate::run_blocking_planner_call(move || {
                crate::table_input::VgiLiteralInputProvider::bind_blocking(
                    conn,
                    &catalog,
                    &schema_name,
                    &function,
                    arguments,
                    input_schema,
                    input,
                )
            })
            .map(|p| p as Arc<dyn TableProvider>);
        }

        let arguments = to_arguments(&self.function, args.exprs())?;
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        crate::run_blocking_planner_call(move || {
            VgiTableProvider::bind_blocking(conn, &catalog, &schema_name, &function, arguments)
        })
        .map(|p| p as Arc<dyn TableProvider>)
    }
}

/// Split a childless blended call into bind arguments and one input row.
fn blended_literal_input(
    function: &str,
    specs: &vgi_client::ArgSpecs,
    exprs: &[Expr],
) -> DFResult<(
    Arguments,
    datafusion::arrow::datatypes::SchemaRef,
    datafusion::arrow::array::RecordBatch,
)> {
    use datafusion::arrow::array::RecordBatchOptions;
    use datafusion::arrow::datatypes::{Field, Schema};

    let positional_specs = specs.positional().collect::<Vec<_>>();
    let mut positional_index = 0usize;
    let mut arguments = Arguments::new();
    let mut fields = Vec::new();
    let mut columns = Vec::new();

    for (argument_index, expr) in exprs.iter().enumerate() {
        if let Some((name, value)) = named_arg(function, argument_index, expr)? {
            arguments = arguments.named(name, value);
            continue;
        }

        let spec = positional_specs.get(positional_index).ok_or_else(|| {
            datafusion::common::plan_datafusion_err!(
                "VGI function `{function}` received more positional arguments than it declares"
            )
        })?;
        let value = match expr {
            Expr::Literal(value, _) => value.clone(),
            Expr::Column(column) => {
                return plan_err!(
                    "VGI function `{function}` argument {argument_index} refers to column \
                     `{column}`; correlated LATERAL table functions are not yet representable \
                     in DataFusion"
                )
            }
            other => {
                return plan_err!(
                    "VGI function `{function}` argument {argument_index} is not a constant: {other}"
                )
            }
        };
        let value = value.cast_to(&spec.data_type)?;
        fields.push(Field::new(&spec.name, spec.data_type.clone(), true));
        columns.push(value.to_array_of_size(1)?);
        positional_index += 1;
    }

    let schema = Arc::new(Schema::new(fields));
    let input = datafusion::arrow::array::RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )?;
    Ok((arguments, schema, input))
}

/// Fold a call's argument expressions into [`Arguments`].
fn to_arguments(function: &str, exprs: &[Expr]) -> DFResult<Arguments> {
    let mut out = Arguments::new();
    for (i, e) in exprs.iter().enumerate() {
        if let Some((name, value)) = named_arg(function, i, e)? {
            out = out.named(name, value);
        } else {
            out = out.positional(to_arg_value(function, i, e)?);
        }
    }
    Ok(out)
}

/// Unwrap the private struct literal inserted by `session::rewrite_table_functions`.
/// DataFusion may constant-fold it to a Struct scalar before the table-function
/// hook, so both the folded and expression forms are accepted.
fn named_arg(
    function_name: &str,
    index: usize,
    expr: &Expr,
) -> DFResult<Option<(String, ArgValue)>> {
    if let Expr::Literal(ScalarValue::Struct(array), _) = expr {
        if array.num_columns() == 1 {
            if let Some(name) = array.fields()[0]
                .name()
                .strip_prefix(crate::session::NAMED_ARG_PREFIX)
            {
                let value = ScalarValue::try_from_array(array.column(0), 0)?;
                return scalar_to_arg(function_name, index, &value)
                    .map(|value| Some((name.to_string(), value)));
            }
        }
    }

    if let Expr::ScalarFunction(function) = expr {
        if function.name() == "named_struct" && function.args.len() == 2 {
            if let Expr::Literal(ScalarValue::Utf8(Some(field)), _) = &function.args[0] {
                if let Some(name) = field.strip_prefix(crate::session::NAMED_ARG_PREFIX) {
                    return to_arg_value(function_name, index, &function.args[1])
                        .map(|value| Some((name.to_string(), value)));
                }
            }
        }
    }
    Ok(None)
}

fn to_arg_value(function: &str, i: usize, expr: &Expr) -> DFResult<ArgValue> {
    match expr {
        Expr::Literal(v, _) => scalar_to_arg(function, i, v),
        // A SQL named argument is carried by `named_arg` above. An unrelated
        // alias can still arrive other ways, and dropping it would silently
        // change which argument the value binds to.
        Expr::Alias(a) => plan_err!(
            "VGI function `{function}` argument {i} is named (`{}`), and DataFusion \
             table functions take positional arguments only; pass it positionally",
            a.name
        ),
        Expr::Column(c) => plan_err!(
            "VGI function `{function}` argument {i} refers to column `{c}`, but a \
             table function's arguments are bound during planning, before any row \
             exists; pass a constant"
        ),
        other => plan_err!("VGI function `{function}` argument {i} is not a constant: {other}"),
    }
}

pub(crate) fn scalar_to_arg(function: &str, i: usize, v: &ScalarValue) -> DFResult<ArgValue> {
    use ScalarValue as S;
    Ok(match v {
        S::Int8(Some(n)) => ArgValue::Int(*n as i64),
        S::Int16(Some(n)) => ArgValue::Int(*n as i64),
        S::Int32(Some(n)) => ArgValue::Int(*n as i64),
        S::Int64(Some(n)) => ArgValue::Int(*n),
        S::UInt8(Some(n)) => ArgValue::Int(*n as i64),
        S::UInt16(Some(n)) => ArgValue::Int(*n as i64),
        S::UInt32(Some(n)) => ArgValue::Int(*n as i64),
        // A u64 above i64::MAX would wrap silently, which is worse than a
        // refusal — the worker would bind a negative count.
        S::UInt64(Some(n)) => match i64::try_from(*n) {
            Ok(n) => ArgValue::Int(n),
            Err(_) => {
                return plan_err!(
                    "VGI function `{function}` argument {i} ({n}) exceeds the \
                     signed 64-bit range the protocol carries"
                )
            }
        },
        S::Float32(Some(f)) => ArgValue::Float(*f as f64),
        S::Float64(Some(f)) => ArgValue::Float(*f),
        S::Utf8(Some(s)) | S::LargeUtf8(Some(s)) | S::Utf8View(Some(s)) => {
            ArgValue::Text(s.clone())
        }
        S::Boolean(Some(b)) => ArgValue::Bool(*b),
        // A NULL argument is meaningful — the protocol carries a *typed* null,
        // so keep the type the planner inferred rather than flattening it.
        null if null.is_null() => ArgValue::Null(null.data_type()),
        other => {
            return plan_err!(
                "VGI function `{function}` argument {i} has type {} which this \
                 adapter does not carry as a bind argument",
                other.data_type()
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::DataType;

    fn lit(v: ScalarValue) -> Expr {
        Expr::Literal(v, None)
    }

    fn one(v: ScalarValue) -> DFResult<ArgValue> {
        to_arg_value("f", 0, &lit(v))
    }

    #[test]
    fn integer_widths_all_narrow_to_the_protocols_i64() {
        for v in [
            ScalarValue::Int8(Some(1)),
            ScalarValue::Int16(Some(1)),
            ScalarValue::Int32(Some(1)),
            ScalarValue::Int64(Some(1)),
            ScalarValue::UInt8(Some(1)),
            ScalarValue::UInt32(Some(1)),
        ] {
            assert!(matches!(one(v).unwrap(), ArgValue::Int(1)));
        }
    }

    #[test]
    fn a_u64_beyond_i64_is_refused_rather_than_wrapped() {
        let err = one(ScalarValue::UInt64(Some(u64::MAX)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("64-bit range"), "{err}");
    }

    #[test]
    fn strings_floats_and_bools_map_across() {
        assert!(matches!(
            one(ScalarValue::Utf8(Some("x".into()))).unwrap(),
            ArgValue::Text(s) if s == "x"
        ));
        assert!(matches!(
            one(ScalarValue::Float64(Some(1.5))).unwrap(),
            ArgValue::Float(f) if f == 1.5
        ));
        assert!(matches!(
            one(ScalarValue::Boolean(Some(true))).unwrap(),
            ArgValue::Bool(true)
        ));
    }

    #[test]
    fn a_null_keeps_the_type_the_planner_inferred() {
        match one(ScalarValue::Int64(None)).unwrap() {
            ArgValue::Null(t) => assert_eq!(t, DataType::Int64),
            other => panic!("expected a typed null, got {other:?}"),
        }
        match one(ScalarValue::Utf8(None)).unwrap() {
            ArgValue::Null(t) => assert_eq!(t, DataType::Utf8),
            other => panic!("expected a typed null, got {other:?}"),
        }
    }

    #[test]
    fn a_column_argument_says_why_it_cannot_work() {
        let err = to_arg_value("f", 1, &Expr::Column("c".into()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("before any row exists"), "{err}");
    }

    #[test]
    fn positional_order_is_preserved() {
        let args = to_arguments(
            "f",
            &[
                lit(ScalarValue::Int64(Some(10))),
                lit(ScalarValue::Utf8(Some("b".into()))),
            ],
        )
        .unwrap();
        // Round-trip through the wire encoding: the field names are what the
        // worker matches on, so their order is the contract.
        let bytes = args.to_ipc().unwrap().0;
        let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(bytes),
            None,
        )
        .expect("valid IPC stream");
        let batch = reader.next().expect("one batch").expect("readable");
        let fields = match batch.schema().field(0).data_type() {
            DataType::Struct(f) => f.clone(),
            other => panic!("args should be a struct, got {other:?}"),
        };
        assert_eq!(fields[0].name(), "positional_0");
        assert_eq!(fields[1].name(), "positional_1");
    }
}
