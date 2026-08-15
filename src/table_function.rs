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
//! # Positional arguments only
//!
//! VGI functions take named arguments too — `sequence(10, batch_size := 5)` is
//! ordinary in the DuckDB extension. DataFusion cannot express that: both
//! spellings are refused during planning, not parsing —
//!
//! ```text
//! SELECT * FROM generate_series(1, 3, step => 1);
//! Error during planning: Unsupported function argument type: step => 1
//! ```
//!
//! — so [`TableFunctionArgs`] carries a positional `&[Expr]` and there is
//! nowhere to put a name. Named arguments therefore need a DataFusion change,
//! and until then a caller must pass positionally or the worker must accept it.
//! [`VgiTableFunction::call_with_args`] rejects an aliased argument with that
//! explanation rather than silently dropping the name.
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

use crate::{VgiConnection, VgiTableProvider};

/// One VGI table function, callable with arguments.
#[derive(Debug)]
pub struct VgiTableFunction {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    /// Whether the worker declared this a `TableBufferingFunction`. Only
    /// consulted for a call that carries a table argument, since that is the
    /// only shape where the two protocols diverge here.
    buffered: bool,
}

impl VgiTableFunction {
    /// Wrap a worker function as a DataFusion table function.
    pub fn new(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        buffered: bool,
    ) -> Self {
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            buffered,
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
            return crate::table_input::VgiTableInputProvider::bind_blocking(
                self.conn.clone(),
                &self.catalog,
                &self.schema_name,
                &self.function,
                arguments,
                table_arg,
                self.buffered,
            )
            .map(|p| p as Arc<dyn TableProvider>);
        }

        let arguments = to_arguments(&self.function, args.exprs())?;
        VgiTableProvider::bind_blocking(
            self.conn.clone(),
            &self.catalog,
            &self.schema_name,
            &self.function,
            arguments,
        )
        .map(|p| p as Arc<dyn TableProvider>)
    }
}

/// Fold a call's argument expressions into [`Arguments`].
fn to_arguments(function: &str, exprs: &[Expr]) -> DFResult<Arguments> {
    let mut out = Arguments::new();
    for (i, e) in exprs.iter().enumerate() {
        out = out.positional(to_arg_value(function, i, e)?);
    }
    Ok(out)
}

fn to_arg_value(function: &str, i: usize, expr: &Expr) -> DFResult<ArgValue> {
    match expr {
        Expr::Literal(v, _) => scalar_to_arg(function, i, v),
        // `f(x := 1)` never reaches here today — DataFusion refuses it during
        // planning — but an alias can arrive other ways, and dropping the name
        // would silently change which argument the value binds to.
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
