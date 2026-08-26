// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! A VGI scalar function as a DataFusion async UDF.
//!
//! `AsyncScalarUDFImpl` is DataFusion's purpose-built seam for remote
//! functions — its own docs say so — and it is what makes a per-batch RPC
//! respectable rather than a blocked runtime thread. `AsyncFuncExec` hoists the
//! calls out of the projection and batches them at `ideal_batch_size`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::{Field, FieldRef};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use vgi_client::{ArgSpecs, ArgValue, Arguments};

use crate::{to_df, VgiConnection};

/// A remote scalar function.
#[derive(Debug, Clone)]
pub struct VgiScalarUdf {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    /// The function's name **on the worker** — what gets bound.
    function: String,
    /// The name DataFusion knows it by, which is the registry key.
    ///
    /// Distinct from [`Self::function`] because the flat, session-wide registry
    /// has no notion of catalog or schema, so one worker function is published
    /// under several keys (the qualified path, and a short alias). All of them
    /// dispatch to the same `function`.
    registered_name: String,
    signature: Signature,
    volatility: Volatility,
    /// A fixed return type, when the caller knows it.
    ///
    /// `None` means "ask the worker", which is the normal case: VGI resolves a
    /// scalar's return type at **bind** time from the argument types, so there
    /// is nothing static to declare. See [`Self::resolve_return_type`].
    return_type: Option<DataType>,
    /// What the worker declares about its parameters.
    ///
    /// Load-bearing for [`ArgSpec::is_const`]: a const parameter is a bind-time
    /// constant, so its value belongs in the bind's `Arguments` and must not be
    /// shipped as a column. A worker that finds no value for it answers with
    /// NULLs rather than an error.
    /// Every overload advertised under this SQL name. DataFusion owns one UDF
    /// registry entry per name, while VGI resolves overloads at bind time.
    /// Keeping the set is essential because overloads may put const parameters
    /// in different positions.
    overloads: Vec<ArgSpecs>,
    /// Memoised bind-time return types, keyed on the argument types **and** any
    /// const values, since a worker may resolve its output type from them.
    resolved: Arc<Mutex<HashMap<String, DataType>>>,
    batch_size: Option<usize>,
}

impl VgiScalarUdf {
    /// Describe a remote scalar function.
    ///
    /// The signature and return type are supplied rather than discovered
    /// because DataFusion needs both during planning, where there is nowhere to
    /// await; read them from `catalog.functions(...)` first.
    pub fn new(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        name: impl Into<String>,
        signature: Signature,
        return_type: DataType,
    ) -> Self {
        let name = name.into();
        let volatility = signature.volatility;
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: name.clone(),
            registered_name: name,
            signature,
            volatility,
            return_type: Some(return_type),
            overloads: vec![ArgSpecs::default()],
            resolved: Arc::new(Mutex::new(HashMap::new())),
            batch_size: None,
        }
    }

    /// Describe a function discovered from a catalog, resolving its return type
    /// against the worker on demand.
    ///
    /// # Why the signature is permissive and the return type is not declared
    ///
    /// VGI resolves both arity and result type at **bind**, from the actual
    /// argument types — that is the point of the protocol, and it is why
    /// `catalog.functions(...)` cannot hand over a DataFusion `Signature` and a
    /// `DataType` up front. So the signature accepts anything and the worker
    /// stays the authority: a wrong arity or type comes back as the worker's own
    /// bind error, which is a better message than one this adapter could invent.
    ///
    /// `registered_name` is the key DataFusion will use, which differs from the
    /// worker's `function` whenever a function is published under more than one
    /// name.
    pub fn discovered(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        registered_name: impl Into<String>,
        specs: ArgSpecs,
    ) -> Self {
        Self::discovered_with_volatility(
            conn,
            catalog,
            schema_name,
            function,
            registered_name,
            specs,
            Volatility::Volatile,
        )
    }

    /// Describe a discovered function with worker-declared volatility.
    pub fn discovered_with_volatility(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        registered_name: impl Into<String>,
        specs: ArgSpecs,
        volatility: Volatility,
    ) -> Self {
        Self::discovered_overloads_with_volatility(
            conn,
            catalog,
            schema_name,
            function,
            registered_name,
            vec![specs],
            volatility,
        )
    }

    /// Describe all VGI overloads published under one DataFusion UDF name.
    pub fn discovered_overloads_with_volatility(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        registered_name: impl Into<String>,
        overloads: Vec<ArgSpecs>,
        volatility: Volatility,
    ) -> Self {
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            registered_name: registered_name.into(),
            signature: Signature::variadic_any(volatility),
            volatility,
            return_type: None,
            overloads: if overloads.is_empty() {
                vec![ArgSpecs::default()]
            } else {
                overloads
            },
            resolved: Arc::new(Mutex::new(HashMap::new())),
            batch_size: None,
        }
    }

    /// Split a call's arguments into bind constants and per-row columns.
    ///
    /// The protocol's central distinction for scalars: a `ConstParam` is
    /// resolved once at bind and never appears in the input batch, while every
    /// other parameter is a column. Sending a const as a column leaves the
    /// worker with no value for it — and it answers with NULLs, not an error,
    /// so this is the difference between right answers and quietly wrong ones.
    ///
    /// The bind's `Arguments` carry **only the constants**, compacted in
    /// declared order — a columnar parameter contributes nothing there, not
    /// even a placeholder. That is the extension's contract, stated in
    /// `vgi_scalar_function_impl.cpp`: *"Pass only the extracted const values —
    /// non-const params come from input batch columns."* Sending placeholders
    /// alongside shifts every constant to the wrong position, and the worker
    /// reads a null where it expected a value, answering with a column of
    /// NULLs rather than an error.
    ///
    /// `values` supplies the constants: `Some` where the planner knows the
    /// argument is a literal. A const parameter whose value is not known yet
    /// (return-type resolution before constant folding) becomes a typed null,
    /// which is enough for the worker to resolve an output type.
    fn split_arguments(
        &self,
        specs: &ArgSpecs,
        types: &[DataType],
        values: &[Option<ArgValue>],
        coerce_to_declared_types: bool,
    ) -> (Arguments, Vec<Field>) {
        let mut arguments = Arguments::new();
        let mut columns = Vec::new();
        for (i, ty) in types.iter().enumerate() {
            if specs.positional_is_const(i) {
                let declared_type = coerce_to_declared_types
                    .then(|| Self::positional_spec(specs, i))
                    .flatten()
                    .map(|spec| spec.data_type.clone())
                    .filter(|ty| *ty != DataType::Null)
                    .unwrap_or_else(|| ty.clone());
                let value = values
                    .get(i)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| ArgValue::Null(declared_type));
                arguments = arguments.positional(value);
            } else {
                let column_type = coerce_to_declared_types
                    .then(|| Self::positional_spec(specs, i))
                    .flatten()
                    .map(|spec| spec.data_type.clone())
                    .filter(|declared| *declared != DataType::Null)
                    .unwrap_or_else(|| ty.clone());
                columns.push(Field::new(
                    format!("col_{}", columns.len()),
                    column_type,
                    true,
                ));
            }
        }
        (arguments, columns)
    }

    /// Pick the advertised overload whose arity and declared types best match
    /// this call. The worker remains the final overload authority at bind; this
    /// selection only decides which arguments are constants and therefore must
    /// be removed from the input batch.
    fn select_specs(&self, types: &[DataType]) -> (&ArgSpecs, bool) {
        let mut best: Option<(&ArgSpecs, i64)> = None;
        let mut arity_fallback: Option<&ArgSpecs> = None;
        for specs in &self.overloads {
            let positional: Vec<_> = specs.positional().collect();
            let varargs = positional.iter().position(|spec| spec.is_varargs);
            let arity_matches = match varargs {
                Some(index) => types.len() >= index,
                None => types.len() == positional.len(),
            };
            if !arity_matches {
                continue;
            }
            arity_fallback.get_or_insert(specs);

            let mut score = 0i64;
            let mut compatible = true;
            for (index, actual) in types.iter().enumerate() {
                let Some(spec) = positional
                    .get(index)
                    .copied()
                    .or_else(|| varargs.and_then(|i| positional.get(i).copied()))
                else {
                    continue;
                };
                match scalar_type_score(actual, &spec.data_type) {
                    Some(value) => score += value,
                    // DataFusion permits useful literal coercions for bind-time
                    // constants (for example `'5'` to an Int64 ConstParam).
                    // Keep such a candidate in play; the value conversion will
                    // still fail with a precise planning error if Arrow cannot
                    // perform the cast.
                    None if spec.is_const => score -= 1,
                    None => {
                        compatible = false;
                        break;
                    }
                }
            }
            if compatible && best.is_none_or(|(_, current)| score > current) {
                best = Some((specs, score));
            }
        }
        best.map(|(specs, _)| (specs, true))
            // Preserve an overload's const/column layout so the worker can
            // produce its authoritative bind error, but do not cast the call
            // into an overload that did not actually accept its column types.
            .unwrap_or_else(|| (arity_fallback.unwrap_or_else(|| &self.overloads[0]), false))
    }

    fn positional_spec(specs: &ArgSpecs, index: usize) -> Option<&vgi_client::ArgSpec> {
        let positional: Vec<_> = specs.positional().collect();
        positional
            .get(index)
            .copied()
            .or_else(|| positional.iter().find(|spec| spec.is_varargs).copied())
    }

    fn declared_const_type(specs: &ArgSpecs, index: usize) -> Option<&DataType> {
        Self::positional_spec(specs, index)
            .filter(|spec| spec.is_const)
            .map(|spec| &spec.data_type)
    }

    /// Coerce a literal to the ConstParam's declared type before encoding it.
    /// VGI overload resolution and the function implementation both see the
    /// bind value, so forwarding DataFusion's literal type unchanged can pick
    /// the wrong overload or quietly produce a default value (`'5'` for an
    /// integer ConstParam was the regression that exposed this).
    fn const_scalar_to_arg(
        &self,
        specs: &ArgSpecs,
        index: usize,
        scalar: &datafusion::common::ScalarValue,
    ) -> DFResult<ArgValue> {
        let declared = Self::declared_const_type(specs, index).unwrap_or(&DataType::Null);
        let coerced;
        let scalar = if *declared == DataType::Null {
            scalar
        } else {
            coerced = scalar.cast_to(declared).map_err(|error| {
                DataFusionError::Plan(format!(
                    "VGI scalar `{}` const argument {index} cannot be cast from {} to {declared}: {error}",
                    self.function,
                    scalar.data_type()
                ))
            })?;
            &coerced
        };
        crate::table_function::scalar_to_arg(&self.function, index, scalar)
    }

    fn const_array_to_arg(
        &self,
        specs: &ArgSpecs,
        index: usize,
        array: &dyn datafusion::arrow::array::Array,
    ) -> DFResult<ArgValue> {
        let declared = Self::declared_const_type(specs, index).unwrap_or(&DataType::Null);
        if *declared == DataType::Null || array.data_type() == declared {
            return ArgValue::from_array_row0(array, &self.function).map_err(to_df);
        }
        let coerced = datafusion::arrow::compute::cast(array, declared).map_err(|error| {
            DataFusionError::Plan(format!(
                "VGI scalar `{}` const argument {index} cannot be cast from {} to {declared}: {error}",
                self.function,
                array.data_type()
            ))
        })?;
        ArgValue::from_array_row0(coerced.as_ref(), &self.function).map_err(to_df)
    }

    /// Ask the worker what this call returns, and remember the answer.
    ///
    /// `ScalarUDFImpl::return_type` is synchronous and runs during planning, so
    /// this blocks on one bind RPC. That is the direct path rather than a
    /// compromise — `vgi_client` is a blocking client — and the result is
    /// memoised per argument-type list, so a query with a thousand calls to the
    /// same function pays once.
    fn resolve_return_type(
        &self,
        arg_types: &[DataType],
        values: &[Option<ArgValue>],
    ) -> DFResult<DataType> {
        if let Some(t) = &self.return_type {
            return Ok(t.clone());
        }
        let key = format!("{arg_types:?}|{values:?}");
        if let Ok(cache) = self.resolved.lock() {
            if let Some(t) = cache.get(&key) {
                return Ok(t.clone());
            }
        }

        use datafusion::arrow::datatypes::Schema;
        use vgi_client::{BindSpec, FunctionType};

        let (specs, compatible) = self.select_specs(arg_types);
        let (arguments, columns) = self.split_arguments(specs, arg_types, values, compatible);
        let input_schema = Schema::new(columns);

        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let out = crate::run_blocking_planner_call(move || {
            let mut client = conn.connect()?;
            let attached = conn.attach(&mut client, &catalog)?;
            let mut spec = BindSpec::table(&function).in_schema(&schema_name);
            spec.function_type = FunctionType::Scalar;
            spec.arguments = arguments;

            let bound = crate::bind_with_input_secrets(
                &conn,
                &mut client,
                &attached,
                &spec,
                &input_schema,
            )?;
            bound
                .output_schema()
                .fields()
                .first()
                .map(|field| field.data_type().clone())
                .ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "VGI scalar `{function}` bound to an output schema with no columns"
                    ))
                })
        })?;

        if let Ok(mut cache) = self.resolved.lock() {
            cache.insert(key, out.clone());
        }
        Ok(out)
    }

    /// Ask DataFusion to hand over this many rows per call.
    #[must_use]
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = Some(n);
        self
    }
}

impl std::hash::Hash for VgiScalarUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.registered_name.hash(state);
        self.catalog.hash(state);
        self.schema_name.hash(state);
    }
}

impl PartialEq for VgiScalarUdf {
    fn eq(&self, other: &Self) -> bool {
        self.registered_name == other.registered_name
            && self.catalog == other.catalog
            && self.schema_name == other.schema_name
    }
}

impl Eq for VgiScalarUdf {}

fn scalar_type_score(actual: &DataType, expected: &DataType) -> Option<i64> {
    use DataType::*;
    if actual == expected {
        return Some(4);
    }
    // A null actual has no contradictory type information. A null declaration
    // is VGI's Arrow representation of AnyArrow and accepts every actual type.
    if *actual == Null || *expected == Null {
        return Some(0);
    }
    let integer = |ty: &DataType| {
        matches!(
            ty,
            Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
        )
    };
    let numeric = |ty: &DataType| {
        integer(ty)
            || matches!(
                ty,
                Float16 | Float32 | Float64 | Decimal128(_, _) | Decimal256(_, _)
            )
    };
    if integer(actual) && integer(expected) {
        Some(3)
    } else if numeric(actual) && numeric(expected) && !integer(actual) && !integer(expected) {
        // VGI groups floating-point and decimal values together, but does not
        // consider an integer column compatible with a floating-point column.
        Some(2)
    } else if matches!(actual, Utf8 | LargeUtf8 | Utf8View)
        && matches!(expected, Utf8 | LargeUtf8 | Utf8View)
    {
        Some(3)
    } else if matches!(actual, Binary | LargeBinary | BinaryView)
        && matches!(expected, Binary | LargeBinary | BinaryView)
    {
        Some(3)
    } else {
        None
    }
}

impl ScalarUDFImpl for VgiScalarUdf {
    fn name(&self) -> &str {
        &self.registered_name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, args: &[DataType]) -> DFResult<DataType> {
        // Reached only when the planner has no literal information; a const
        // parameter then resolves as a typed null, which is enough to learn an
        // output type. `return_field_from_args` is the richer path.
        self.resolve_return_type(args, &[])
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> DFResult<FieldRef> {
        // The hook that makes const parameters work: it carries the *values* of
        // the literal arguments, which `return_type` cannot see, so the bind is
        // issued with the constant the caller actually wrote.
        let types: Vec<DataType> = args
            .arg_fields
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        let (specs, compatible) = self.select_specs(&types);
        let values: Vec<Option<ArgValue>> = args
            .scalar_arguments
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if specs.positional_is_const(i) {
                    v.map(|scalar| {
                        if compatible {
                            self.const_scalar_to_arg(specs, i, scalar)
                        } else {
                            crate::table_function::scalar_to_arg(&self.function, i, scalar)
                        }
                    })
                    .transpose()
                } else {
                    Ok(None)
                }
            })
            .collect::<DFResult<_>>()?;
        let out = self.resolve_return_type(&types, &values)?;
        Ok(Arc::new(Field::new(self.name(), out, true)))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        // Reached only if something bypassed AsyncFuncExec; a remote call has
        // no synchronous form.
        Err(DataFusionError::Internal(format!(
            "{} is a remote function and must be invoked asynchronously",
            self.registered_name
        )))
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for VgiScalarUdf {
    fn ideal_batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        use datafusion::arrow::array::{RecordBatch, RecordBatchOptions};
        use datafusion::arrow::datatypes::{Field, Schema};

        let rows = args.number_rows;

        // Const parameters are resolved at bind and must not be shipped as
        // columns; everything else is a column. See `split_arguments`.
        let types: Vec<DataType> = args.args.iter().map(|a| a.data_type()).collect();
        let (specs, compatible) = self.select_specs(&types);
        let values: Vec<Option<ArgValue>> = args
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                ColumnarValue::Scalar(s) if specs.positional_is_const(i) => {
                    if compatible {
                        self.const_scalar_to_arg(specs, i, s).map(Some)
                    } else {
                        crate::table_function::scalar_to_arg(&self.function, i, s).map(Some)
                    }
                }
                ColumnarValue::Scalar(_) => Ok(None),
                // A literal may arrive already expanded across the batch — the
                // engine is free to materialise it — and then every row holds
                // the same constant, so row 0 is the value the bind needs.
                // Without this a const parameter silently becomes a typed null
                // and the worker answers with NULLs.
                ColumnarValue::Array(arr) if specs.positional_is_const(i) => {
                    if compatible {
                        self.const_array_to_arg(specs, i, arr.as_ref()).map(Some)
                    } else {
                        ArgValue::from_array_row0(arr.as_ref(), &self.function)
                            .map(Some)
                            .map_err(to_df)
                    }
                }
                ColumnarValue::Array(_) => Ok(None),
            })
            .collect::<DFResult<_>>()?;
        let (arguments, _) = self.split_arguments(specs, &types, &values, compatible);

        let columns: Vec<datafusion::arrow::array::ArrayRef> = args
            .args
            .iter()
            .enumerate()
            .filter(|(i, _)| !specs.positional_is_const(*i))
            .map(|(i, a)| {
                let array = a.to_array(rows)?;
                let declared = Self::positional_spec(specs, i).map(|spec| &spec.data_type);
                match declared.filter(|ty| compatible && **ty != DataType::Null) {
                    Some(declared) if array.data_type() != declared => {
                        datafusion::arrow::compute::cast(array.as_ref(), declared).map_err(|error| {
                            DataFusionError::Execution(format!(
                                "VGI scalar `{}` argument {i} cannot be cast from {} to {declared}: {error}",
                                self.function,
                                array.data_type()
                            ))
                        })
                    }
                    _ => Ok(array),
                }
            })
            .collect::<DFResult<_>>()?;
        // DataFusion does not currently constant-fold async UDFs. Preserve the
        // ordinary SQL semantics for an immutable/stable all-const call by
        // evaluating one worker row and broadcasting it, while volatile calls
        // still receive every row and may produce a different value each time.
        let rpc_rows = if rows > 0 && columns.is_empty() && self.volatility != Volatility::Volatile
        {
            1
        } else {
            rows
        };
        // `col_<i>`, not any other spelling: this is the wire convention the
        // DuckDB extension uses (`vgi_scalar_function_impl.cpp` builds
        // `input_names` as `"col_" + i`), and a worker that reads its arguments
        // **by name** finds nothing under another one — returning a column of
        // NULLs rather than an error. A single-argument function that reads
        // positionally works either way, which is what made this look like it
        // worked at all.
        let fields: Vec<Field> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| Field::new(format!("col_{i}"), c.data_type().clone(), true))
            .collect();
        let input = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields.clone())),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rpc_rows)),
        )?;

        let (conn, cat, sch, name) = (
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
        );
        let input_schema = Schema::new(fields);

        let out = tokio::task::spawn_blocking(move || {
            use vgi_client::{BindSpec, FunctionType, ScanOptions};
            let mut client = conn.connect()?;
            let attached = conn.attach(&mut client, &cat)?;
            let mut spec = BindSpec::table(&name).in_schema(&sch);
            spec.function_type = FunctionType::Scalar;
            spec.arguments = arguments;

            let bound = crate::bind_with_input_secrets(
                &conn,
                &mut client,
                &attached,
                &spec,
                &input_schema,
            )?;
            let mut ex = client
                .open_exchange(&bound, &ScanOptions::default())
                .map_err(to_df)?;
            let answer = ex.send(&input).map_err(to_df)?;
            let _ = ex.close();
            answer.ok_or_else(|| {
                DataFusionError::Execution(format!("{name} returned no answer for its input"))
            })
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        if out.num_rows() != rpc_rows {
            return Err(DataFusionError::Execution(format!(
                "{} answered {} rows for {} input rows; a scalar function must be 1:1",
                self.registered_name,
                out.num_rows(),
                rpc_rows
            )));
        }
        if out.num_columns() != 1 {
            return Err(DataFusionError::Execution(format!(
                "{} answered {} columns; a scalar function must return exactly one",
                self.registered_name,
                out.num_columns()
            )));
        }
        let output = if rpc_rows == 1 && rows > 1 {
            datafusion::common::ScalarValue::try_from_array(out.column(0), 0)?
                .to_array_of_size(rows)?
        } else {
            out.column(0).clone()
        };
        Ok(ColumnarValue::Array(output))
    }
}

#[cfg(test)]
mod overload_tests {
    use super::*;
    use vgi_client::ArgSpec;

    fn positional(data_type: DataType) -> ArgSpec {
        ArgSpec {
            name: "value".to_string(),
            data_type,
            is_const: false,
            is_named: false,
            is_varargs: false,
            doc: None,
        }
    }

    fn varargs(data_type: DataType) -> ArgSpecs {
        let mut spec = positional(data_type);
        spec.is_varargs = true;
        ArgSpecs(vec![spec])
    }

    fn udf(overloads: Vec<ArgSpecs>) -> VgiScalarUdf {
        VgiScalarUdf::discovered_overloads_with_volatility(
            VgiConnection::http("http://127.0.0.1:1"),
            "example",
            "main",
            "overloaded",
            "overloaded",
            overloads,
            Volatility::Immutable,
        )
    }

    #[test]
    fn any_varargs_beats_an_incompatible_typed_arm() {
        let typed = varargs(DataType::Int64);
        let any = varargs(DataType::Null);
        let udf = udf(vec![typed, any.clone()]);

        let (selected, compatible) = udf.select_specs(&[DataType::Boolean, DataType::Boolean]);
        assert!(compatible);
        assert_eq!(selected, &any);
    }

    #[test]
    fn integer_columns_do_not_match_float_overloads() {
        let float = ArgSpecs(vec![positional(DataType::Float64)]);
        let any = ArgSpecs(vec![positional(DataType::Null)]);
        let udf = udf(vec![float, any.clone()]);

        let (selected, compatible) = udf.select_specs(&[DataType::Int64]);
        assert!(compatible);
        assert_eq!(selected, &any);
    }

    #[test]
    fn no_compatible_arm_preserves_types_for_the_worker_error() {
        let only = varargs(DataType::Int64);
        let udf = udf(vec![only.clone()]);

        let (selected, compatible) = udf.select_specs(&[DataType::Boolean]);
        assert!(!compatible);
        assert_eq!(selected, &only);

        let (_, fields) = udf.split_arguments(selected, &[DataType::Boolean], &[None], compatible);
        assert_eq!(fields[0].data_type(), &DataType::Boolean);
    }

    #[test]
    fn incompatible_fallback_still_uses_the_matching_arity() {
        let one_arg = ArgSpecs(vec![positional(DataType::Int64)]);
        let two_args = ArgSpecs(vec![
            positional(DataType::Int64),
            positional(DataType::Int64),
        ]);
        let udf = udf(vec![one_arg, two_args.clone()]);

        let (selected, compatible) = udf.select_specs(&[DataType::Boolean, DataType::Boolean]);
        assert!(!compatible);
        assert_eq!(selected, &two_args);
    }

    #[test]
    fn const_arguments_remain_eligible_for_literal_coercion() {
        let mut spec = positional(DataType::Int64);
        spec.is_const = true;
        let specs = ArgSpecs(vec![spec]);
        let udf = udf(vec![specs.clone()]);

        let (selected, compatible) = udf.select_specs(&[DataType::Utf8]);
        assert!(compatible);
        assert_eq!(selected, &specs);
    }
}
