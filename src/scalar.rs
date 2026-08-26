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
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
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
        let accepts_zero_arguments = overloads.is_empty()
            || overloads
                .iter()
                .any(|specs| specs.minimum_positional_arity() == 0);
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            registered_name: registered_name.into(),
            signature: if accepts_zero_arguments {
                // VariadicAny has a minimum arity of one in DataFusion. Admit
                // zero only when a worker overload explicitly declares it;
                // every other arity still reaches the worker through the
                // permissive variadic arm for authoritative validation.
                Signature::one_of(
                    vec![TypeSignature::Nullary, TypeSignature::VariadicAny],
                    volatility,
                )
            } else {
                Signature::variadic_any(volatility)
            },
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
            let minimum = specs.minimum_positional_arity();
            let arity_matches = match varargs {
                Some(_) => types.len() >= minimum,
                None => (minimum..=positional.len()).contains(&types.len()),
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

    /// Give untyped NULLs in an AnyArrow vararg group the unambiguous type of
    /// their concrete peers.
    ///
    /// DataFusion's permissive `VariadicAny` signature deliberately does not
    /// coerce arguments to a common type. That is normally what VGI wants, but
    /// a bare SQL NULL therefore reaches the worker as Arrow `Null`, even for
    /// calls such as `sum_values(NULL, 2, 3)`. DuckDB types that NULL as BIGINT
    /// from the surrounding varargs, and the VGI function quite reasonably
    /// rejects a physical `Null` input as non-addable.
    ///
    /// Only infer when every non-null member of the vararg group has exactly
    /// the same type. Mixed concrete types remain untouched so the worker stays
    /// the overload/coercion authority rather than this adapter guessing a
    /// common supertype.
    fn resolve_untyped_null_varargs(specs: &ArgSpecs, types: &[DataType]) -> Vec<DataType> {
        let positional: Vec<_> = specs.positional().collect();
        let Some((varargs_index, varargs)) = positional
            .iter()
            .enumerate()
            .find(|(_, spec)| spec.is_varargs)
        else {
            return types.to_vec();
        };
        if varargs.data_type != DataType::Null || varargs_index >= types.len() {
            return types.to_vec();
        }

        let mut concrete = types[varargs_index..]
            .iter()
            .filter(|ty| **ty != DataType::Null);
        let Some(inferred) = concrete.next().cloned() else {
            return types.to_vec();
        };
        if concrete.any(|ty| ty != &inferred) {
            return types.to_vec();
        }

        types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                if index >= varargs_index && *ty == DataType::Null {
                    inferred.clone()
                } else {
                    ty.clone()
                }
            })
            .collect()
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
        // A setting may change a worker's resolved output type (the example
        // worker's verbose-mode function deliberately does). Never reuse a
        // bind-derived type across different session-setting snapshots.
        let key = format!(
            "{arg_types:?}|{values:?}|{:?}",
            self.conn.runtime().session_settings_identity()
        );
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

            let (bound, used_resolved_secrets) = crate::bind_with_input_secrets_status(
                &conn,
                &mut client,
                &attached,
                &spec,
                &input_schema,
            )?;
            let output_type = bound
                .output_schema()
                .fields()
                .first()
                .map(|field| field.data_type().clone())
                .ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "VGI scalar `{function}` bound to an output schema with no columns"
                    ))
                })?;
            Ok((output_type, used_resolved_secrets))
        })?;

        if !out.1 {
            if let Ok(mut cache) = self.resolved.lock() {
                cache.insert(key, out.0.clone());
            }
        }
        Ok(out.0)
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
        let (specs, _) = self.select_specs(args);
        let resolved_types = Self::resolve_untyped_null_varargs(specs, args);
        self.resolve_return_type(&resolved_types, &[])
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
        let resolved_types = Self::resolve_untyped_null_varargs(specs, &types);
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
        let out = self.resolve_return_type(&resolved_types, &values)?;
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
        let resolved_types = Self::resolve_untyped_null_varargs(specs, &types);
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
        let (arguments, _) = self.split_arguments(specs, &resolved_types, &values, compatible);

        let columns: Vec<datafusion::arrow::array::ArrayRef> = args
            .args
            .iter()
            .enumerate()
            .filter(|(i, _)| !specs.positional_is_const(*i))
            .map(|(i, a)| {
                let array = a.to_array(rows)?;
                let declared = Self::positional_spec(specs, i).map(|spec| &spec.data_type);
                let resolved_null = (array.data_type() == &DataType::Null
                    && resolved_types[i] != DataType::Null)
                    .then_some(&resolved_types[i]);
                let target = declared
                    .filter(|ty| compatible && **ty != DataType::Null)
                    .or(resolved_null);
                match target {
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
        let cacheable = self.volatility != Volatility::Volatile;
        let adapter_settings = self.conn.runtime().adapter_settings();
        let dedup_enabled = cacheable && adapter_settings.exchange_input_dedup();
        let per_value_enabled = adapter_settings.result_cache_per_value();

        let (conn, cat, sch, name) = (
            self.conn.clone(),
            self.catalog.clone(),
            self.schema_name.clone(),
            self.function.clone(),
        );
        let input_schema = Schema::new(fields);

        let (out, gather, sent_rows) = tokio::task::spawn_blocking(move || {
            let (input, gather) = if dedup_enabled {
                deduplicate_scalar_input(input)?
            } else {
                (input, None)
            };
            let sent_rows = input.num_rows();
            let out = run_scalar_exchange(
                &conn,
                &cat,
                &sch,
                &name,
                arguments,
                &input_schema,
                input,
                cacheable,
                per_value_enabled,
            )?;
            Ok::<_, DataFusionError>((out, gather, sent_rows))
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;

        if out.num_rows() != sent_rows {
            return Err(DataFusionError::Execution(format!(
                "{} answered {} rows for {} input rows; a scalar function must be 1:1",
                self.registered_name,
                out.num_rows(),
                sent_rows
            )));
        }
        if out.num_columns() != 1 {
            return Err(DataFusionError::Execution(format!(
                "{} answered {} columns; a scalar function must return exactly one",
                self.registered_name,
                out.num_columns()
            )));
        }
        let output = if let Some(gather) = gather {
            datafusion::arrow::compute::take(out.column(0).as_ref(), &gather, None)?
        } else if rpc_rows == 1 && rows > 1 {
            datafusion::common::ScalarValue::try_from_array(out.column(0), 0)?
                .to_array_of_size(rows)?
        } else {
            out.column(0).clone()
        };
        Ok(ColumnarValue::Array(output))
    }
}

/// Bounded function-level identity for a live `per_value` advertisement.
/// Arguments and principals deliberately stay out: they remain dimensions of
/// every result key, while including them here would grow this small capability
/// registry without bound.
fn per_value_opt_in_identity(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    catalog_version: i64,
) -> Vec<u8> {
    use sha2::{Digest, Sha256};

    let attach = conn.cache_attach_context(catalog);
    let mut digest = Sha256::new();
    for field in [
        b"scalar_per_value_opt_in_v1".as_slice(),
        conn.label().as_bytes(),
        catalog.as_bytes(),
        schema_name.as_bytes(),
        function.as_bytes(),
        attach.as_slice(),
    ] {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field);
    }
    digest.update(catalog_version.to_le_bytes());
    digest.finalize().to_vec()
}

// Matches DuckDB's `vgi_result_cache_per_value_max_stores_per_chunk` default.
// Bound only new stores; hits and result assembly remain complete.
const PER_VALUE_MAX_NEW_STORES_PER_CALL: usize = 256;

/// Copy a single output row into independently owned Arrow buffers before it
/// enters the cache. `RecordBatch::slice` is zero-copy and would otherwise keep
/// the entire RPC response alive for each one-row cache entry.
fn materialize_output_row(
    batch: &datafusion::arrow::array::RecordBatch,
    row: usize,
) -> DFResult<datafusion::arrow::array::RecordBatch> {
    use datafusion::arrow::array::UInt32Array;
    use datafusion::arrow::compute::take;

    let row = u32::try_from(row).map_err(|_| {
        DataFusionError::Internal("scalar output row index exceeds Arrow's u32 limit".to_string())
    })?;
    let indices = UInt32Array::from(vec![row]);
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None).map_err(DataFusionError::from))
        .collect::<DFResult<Vec<_>>>()?;
    Ok(datafusion::arrow::array::RecordBatch::try_new(
        batch.schema(),
        columns,
    )?)
}

/// Collapse byte-identical stable scalar input rows and retain their original
/// row-to-unique mapping for result assembly.
///
/// Canonical Arrow IPC handles every VGI/Arrow input type, including nested
/// values, without inventing a second equality implementation. Treating
/// bitwise-distinct but SQL-equal values as separate is conservative: it may
/// miss a dedup opportunity but can never combine unequal worker inputs.
fn deduplicate_scalar_input(
    input: datafusion::arrow::array::RecordBatch,
) -> DFResult<(
    datafusion::arrow::array::RecordBatch,
    Option<datafusion::arrow::array::UInt32Array>,
)> {
    use datafusion::arrow::compute::concat_batches;

    if input.num_rows() <= 1 {
        return Ok((input, None));
    }
    let mut unique_by_ipc = HashMap::<Vec<u8>, u32>::new();
    let mut uniques = Vec::new();
    let mut row_to_unique = Vec::with_capacity(input.num_rows());
    for row in 0..input.num_rows() {
        let input_row = input.slice(row, 1);
        let ipc = vgi_protocol::ipc::write_batch(&input_row).map_err(to_df)?;
        let unique = match unique_by_ipc.get(&ipc).copied() {
            Some(unique) => unique,
            None => {
                let unique = u32::try_from(uniques.len()).map_err(|_| {
                    DataFusionError::Execution(
                        "stable scalar batch has more than u32::MAX distinct inputs".to_string(),
                    )
                })?;
                unique_by_ipc.insert(ipc, unique);
                uniques.push(input_row);
                unique
            }
        };
        row_to_unique.push(unique);
    }
    if uniques.len() == input.num_rows() {
        return Ok((input, None));
    }
    let unique_input = concat_batches(&input.schema(), uniques.iter())?;
    Ok((
        unique_input,
        Some(datafusion::arrow::array::UInt32Array::from(row_to_unique)),
    ))
}

/// Record one scalar input batch after the exchange accepted it.
///
/// The event deliberately contains only cardinality and function identity.
/// Scalar values, bind arguments, and resolved secrets must never enter the
/// session log history.
fn emit_scalar_write_input(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    input_rows: usize,
) {
    let mut event = crate::VgiEvent::new("scalar.write_input");
    event.catalog = Some(catalog.to_string());
    event.function = Some(format!("{schema_name}.{function}"));
    event.message = Some(format!("input_rows={input_rows}"));
    conn.runtime().emit(event);
}

/// Execute a scalar exchange with worker-opted-in per-value memoization.
///
/// A stable scalar is a 1:1 map. Distinct input tuples are therefore safe to
/// send once, cache independently, and gather back into the caller's original
/// row order. Volatile functions bypass this path completely: equal inputs may
/// intentionally produce different answers.
#[allow(clippy::too_many_arguments)]
fn run_scalar_exchange(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arguments: Arguments,
    input_schema: &datafusion::arrow::datatypes::Schema,
    input: datafusion::arrow::array::RecordBatch,
    cacheable: bool,
    per_value_enabled: bool,
) -> DFResult<datafusion::arrow::array::RecordBatch> {
    use datafusion::arrow::array::{RecordBatch, UInt32Array};
    use datafusion::arrow::compute::{concat_batches, take};
    use vgi_client::{BindSpec, FunctionType, ScanOptions};

    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let catalog_version = attached.info().catalog_version;
    let argument_bytes = arguments.to_ipc().map_err(to_df)?.0;
    let mut spec = BindSpec::table(function).in_schema(schema_name);
    spec.function_type = FunctionType::Scalar;
    spec.arguments = arguments;
    let (bound, used_resolved_secrets) =
        crate::bind_with_input_secrets_status(conn, &mut client, &attached, &spec, input_schema)?;

    let direct = |client: &mut vgi_client::VgiClient,
                  input: &RecordBatch,
                  options: &ScanOptions|
     -> DFResult<(RecordBatch, Option<vgi_client::CacheControl>)> {
        let mut exchange = client.open_exchange(&bound, options).map_err(to_df)?;
        let answer = exchange.send(input).map_err(to_df)?;
        emit_scalar_write_input(conn, catalog, schema_name, function, input.num_rows());
        let control = exchange.cache_control().cloned();
        exchange.close().map_err(to_df)?;
        answer.map(|answer| (answer, control)).ok_or_else(|| {
            DataFusionError::Execution(format!("{function} returned no answer for its input"))
        })
    };

    if !cacheable || used_resolved_secrets || input.num_rows() == 0 || !per_value_enabled {
        return direct(&mut client, &input, &ScanOptions::default()).map(|(answer, _)| answer);
    }

    if !conn.cache_enabled
        || !conn.runtime.options().cache_enabled
        || conn.cache_identity_scope(catalog).is_none()
    {
        return direct(&mut client, &input, &ScanOptions::default()).map(|(answer, _)| answer);
    }
    let opt_in_identity =
        per_value_opt_in_identity(conn, catalog, schema_name, function, catalog_version);

    // Cache capability is advertised on an output batch, not during catalog
    // discovery. Until this exact function identity opts in, make the ordinary
    // one-batch RPC: non-cacheable scalars must not pay row-by-row IPC hashing
    // forever. If the response opts in, populate its distinct rows after the
    // fact so the very first call still warms the per-value tier.
    if !conn.runtime.has_per_value_opt_in(&opt_in_identity) {
        let (answer, control) = direct(&mut client, &input, &ScanOptions::default())?;
        let per_value = control.as_ref().is_some_and(|control| control.per_value);
        let shape_is_scalar = answer.num_rows() == input.num_rows() && answer.num_columns() == 1;
        let mut stored = 0usize;
        if per_value && shape_is_scalar {
            let template = crate::table_input::exchange_cache_key_template(
                conn,
                catalog,
                schema_name,
                function,
                &argument_bytes,
                catalog_version,
                bound.output_schema().as_ref(),
                b"scalar_per_value_v2",
            )?
            .expect("cache availability was checked above");
            let mut seen = std::collections::HashSet::<[u8; 32]>::new();
            for row in 0..input.num_rows() {
                if seen.len() >= PER_VALUE_MAX_NEW_STORES_PER_CALL {
                    break;
                }
                let input_row = input.slice(row, 1);
                let input_digest = template.input_digest(&input_row)?;
                if !seen.insert(input_digest) {
                    continue;
                }
                let key = template.key_for_digest(input_digest);
                let output = materialize_output_row(&answer, row)?;
                let bytes = output
                    .columns()
                    .iter()
                    .map(|array| array.get_array_memory_size())
                    .sum();
                if let Ok(ttl) = crate::table_input::exchange_cache_ttl(
                    conn.runtime.result_cache(),
                    control.as_ref(),
                    &key.identity_scope,
                    bytes,
                ) {
                    conn.runtime
                        .result_cache()
                        .insert(key, vec![output], ttl, control.as_ref());
                    conn.runtime.note_exchange_cache_store();
                    stored += 1;
                }
            }
        }
        if stored > 0 {
            conn.runtime.note_per_value_opt_in(opt_in_identity);
            crate::table_input::emit_exchange_cache_event(
                conn,
                catalog,
                schema_name,
                function,
                "cache.store",
                Some(format!("tier=per_value tuples={stored}")),
            );
        }
        return Ok(answer);
    }

    // Build one cache unit per distinct input tuple. If the attachment cannot
    // be cached (disabled or unresolved identity), retain the old one-exchange
    // path and avoid doing dedup/gather work solely for a cache that is off.
    let template = crate::table_input::exchange_cache_key_template(
        conn,
        catalog,
        schema_name,
        function,
        &argument_bytes,
        catalog_version,
        bound.output_schema().as_ref(),
        b"scalar_per_value_v2",
    )?
    .expect("cache availability was checked above");
    let mut unique_by_digest = HashMap::<[u8; 32], usize>::new();
    let mut uniques = Vec::<(vgi_client::CacheKey, RecordBatch)>::new();
    let mut row_to_unique = Vec::<usize>::with_capacity(input.num_rows());
    for row in 0..input.num_rows() {
        let row_batch = input.slice(row, 1);
        let input_digest = template.input_digest(&row_batch)?;
        let unique = match unique_by_digest.get(&input_digest).copied() {
            Some(index) => index,
            None => {
                let index = uniques.len();
                unique_by_digest.insert(input_digest, index);
                uniques.push((template.key_for_digest(input_digest), row_batch));
                index
            }
        };
        row_to_unique.push(unique);
    }

    let cache = conn.runtime.result_cache();
    let mut outputs = vec![None::<RecordBatch>; uniques.len()];
    let mut misses = Vec::<usize>::new();
    let mut hit_count = 0usize;
    for (index, (key, _)) in uniques.iter().enumerate() {
        let hit = cache.get(key).and_then(|entry| {
            let [batch] = entry.batches() else {
                return None;
            };
            (batch.num_rows() == 1
                && batch.num_columns() == 1
                && batch.schema().as_ref() == bound.output_schema().as_ref())
            .then(|| (batch.clone(), entry.bytes()))
        });
        match hit {
            Some((batch, bytes)) => {
                conn.runtime.note_exchange_cache_hit(bytes);
                outputs[index] = Some(batch);
                hit_count += 1;
            }
            None => misses.push(index),
        }
    }
    if hit_count > 0 {
        crate::table_input::emit_exchange_cache_event(
            conn,
            catalog,
            schema_name,
            function,
            if misses.is_empty() {
                "cache.hit"
            } else {
                "cache.partial_hit"
            },
            Some(format!(
                "tier=per_value reused_tuples={hit_count} computed_tuples={}",
                misses.len()
            )),
        );
    } else {
        crate::table_input::emit_exchange_cache_event(
            conn,
            catalog,
            schema_name,
            function,
            "cache.miss",
            Some(format!("tier=per_value distinct_tuples={}", uniques.len())),
        );
    }

    enum MissClaim {
        Producer {
            flight: Arc<crate::runtime::ResultFlightProducer>,
            stale: Option<vgi_client::CachedEntry>,
        },
        Follower(crate::runtime::ResultFlightWaiter),
    }
    let mut claims = (0..uniques.len())
        .map(|_| None)
        .collect::<Vec<Option<MissClaim>>>();
    for &index in &misses {
        let key = &uniques[index].0;
        claims[index] = Some(match conn.runtime.acquire_result_flight(key) {
            crate::runtime::ResultFlightClaim::Producer(flight) => MissClaim::Producer {
                flight,
                stale: cache.get_for_revalidation(key),
            },
            crate::runtime::ResultFlightClaim::Follower(waiter) => MissClaim::Follower(waiter),
        });
    }

    // A conditional validator applies to one exchange init, hence one scalar
    // cache key. Revalidate producer claims individually; ordinary cold claims
    // remain batched below.
    let revalidations = misses
        .iter()
        .copied()
        .filter(|index| {
            matches!(
                claims[*index].as_ref(),
                Some(MissClaim::Producer { stale: Some(_), .. })
            )
        })
        .collect::<Vec<_>>();
    for index in revalidations {
        let Some(MissClaim::Producer {
            flight,
            stale: Some(stale),
        }) = claims[index].as_ref()
        else {
            unreachable!("revalidation classification changed")
        };
        let key = &uniques[index].0;
        let options = ScanOptions {
            if_none_match: stale.etag.clone(),
            if_modified_since: stale.last_modified.clone(),
            ..Default::default()
        };
        let (answer, control) = match direct(&mut client, &uniques[index].1, &options) {
            Ok(answer) => answer,
            Err(error) if stale.may_serve_on_error_at(std::time::Instant::now()) => {
                cache.record_stale_serve();
                conn.runtime.note_exchange_cache_hit(stale.bytes());
                let [cached] = stale.batches() else {
                    flight.abort("stale-if-error entry was not one batch");
                    return Err(error);
                };
                outputs[index] = Some(cached.clone());
                flight.stored();
                crate::table_input::emit_exchange_cache_event(
                    conn,
                    catalog,
                    schema_name,
                    function,
                    "cache.stale_if_error",
                    Some(format!("tier=per_value {error}")),
                );
                continue;
            }
            Err(error) => return Err(error),
        };
        if control.as_ref().is_some_and(|control| control.not_modified) {
            if answer.num_rows() != 0 {
                return Err(DataFusionError::Execution(format!(
                    "VGI scalar `{function}` returned rows and not_modified together"
                )));
            }
            let [cached] = stale.batches() else {
                flight.abort("revalidation entry was not one batch");
                continue;
            };
            let ttl = match crate::table_input::exchange_cache_ttl(
                cache,
                control.as_ref(),
                &key.identity_scope,
                stale.bytes(),
            ) {
                Ok(ttl) => ttl,
                Err(reason) => {
                    cache.remove(key);
                    flight.abort(format!("revalidation revoked cache: {reason:?}"));
                    return Err(DataFusionError::Execution(format!(
                        "VGI scalar `{function}` returned not_modified with ineligible cache control: {reason:?}"
                    )));
                }
            };
            cache.slide(key, ttl);
            conn.runtime.note_exchange_cache_hit(stale.bytes());
            outputs[index] = Some(cached.clone());
            flight.stored();
            crate::table_input::emit_exchange_cache_event(
                conn,
                catalog,
                schema_name,
                function,
                "cache.revalidated",
                Some("tier=per_value".to_string()),
            );
        } else if answer.num_rows() == 1 && answer.num_columns() == 1 {
            let output = materialize_output_row(&answer, 0)?;
            let bytes = output
                .columns()
                .iter()
                .map(|array| array.get_array_memory_size())
                .sum();
            match crate::table_input::exchange_cache_ttl(
                cache,
                control.as_ref(),
                &key.identity_scope,
                bytes,
            ) {
                Ok(ttl) if control.as_ref().is_some_and(|control| control.per_value) => {
                    cache.insert(key.clone(), vec![output.clone()], ttl, control.as_ref());
                    conn.runtime.note_exchange_cache_store();
                    outputs[index] = Some(output);
                    flight.stored();
                }
                Ok(_) => {
                    cache.remove(key);
                    outputs[index] = Some(output);
                    flight.abort("worker withdrew per-value cache opt-in");
                }
                Err(reason) => {
                    cache.remove(key);
                    outputs[index] = Some(output);
                    flight.abort(format!("cache refused revalidated result: {reason:?}"));
                }
            }
        } else {
            return Err(DataFusionError::Execution(format!(
                "{function} answered {} rows and {} columns for one conditional scalar input; expected a 1:1, one-column result",
                answer.num_rows(),
                answer.num_columns()
            )));
        }
    }

    let producers = misses
        .iter()
        .copied()
        .filter(|index| {
            matches!(
                claims[*index].as_ref(),
                Some(MissClaim::Producer { stale: None, .. })
            )
        })
        .collect::<Vec<_>>();
    if !producers.is_empty() {
        let miss_batches = producers
            .iter()
            .map(|index| &uniques[*index].1)
            .collect::<Vec<_>>();
        let miss_input = concat_batches(&input.schema(), miss_batches)?;
        let mut exchange = client
            .open_exchange(&bound, &ScanOptions::default())
            .map_err(to_df)?;
        let fresh = exchange.send(&miss_input).map_err(to_df)?;
        emit_scalar_write_input(conn, catalog, schema_name, function, miss_input.num_rows());
        let fresh = fresh.ok_or_else(|| {
            DataFusionError::Execution(format!("{function} returned no answer for its input"))
        })?;
        let control = exchange.cache_control().cloned();
        exchange.close().map_err(to_df)?;
        if fresh.num_rows() != producers.len() || fresh.num_columns() != 1 {
            return Err(DataFusionError::Execution(format!(
                "{function} answered {} rows and {} columns for {} distinct scalar inputs; expected a 1:1, one-column result",
                fresh.num_rows(),
                fresh.num_columns(),
                producers.len()
            )));
        }

        let per_value = control.as_ref().is_some_and(|control| control.per_value);
        let mut stored = 0usize;
        let mut store_attempts = 0usize;
        for (fresh_row, unique_index) in producers.into_iter().enumerate() {
            let output = fresh.slice(fresh_row, 1);
            let Some(MissClaim::Producer { flight, .. }) = claims[unique_index].as_ref() else {
                unreachable!("fresh scalar claim changed")
            };
            if per_value && store_attempts < PER_VALUE_MAX_NEW_STORES_PER_CALL {
                store_attempts += 1;
                let cached_output = materialize_output_row(&fresh, fresh_row)?;
                let bytes = cached_output
                    .columns()
                    .iter()
                    .map(|array| array.get_array_memory_size())
                    .sum();
                let key = &uniques[unique_index].0;
                match crate::table_input::exchange_cache_ttl(
                    cache,
                    control.as_ref(),
                    &key.identity_scope,
                    bytes,
                ) {
                    Ok(ttl) => {
                        cache.insert(key.clone(), vec![cached_output], ttl, control.as_ref());
                        conn.runtime.note_exchange_cache_store();
                        flight.stored();
                        stored += 1;
                    }
                    Err(reason) => {
                        flight.abort(format!("cache refused result: {reason:?}"));
                        crate::table_input::emit_exchange_cache_event(
                            conn,
                            catalog,
                            schema_name,
                            function,
                            "cache.refused",
                            Some(format!("tier=per_value {reason:?}")),
                        );
                    }
                }
            } else {
                flight.abort(if per_value {
                    "per-value new-store cap reached"
                } else {
                    "worker withdrew per-value cache opt-in"
                });
            }
            outputs[unique_index] = Some(output);
        }
        if stored > 0 {
            crate::table_input::emit_exchange_cache_event(
                conn,
                catalog,
                schema_name,
                function,
                "cache.store",
                Some(format!("tier=per_value tuples={stored}")),
            );
        }
    }

    // Followers consume the producer's stored row. If the producer aborted or
    // a zero-TTL entry is no longer available as a fresh hit, execute this row
    // directly without claiming another flight; refusal must not form a loop.
    for index in misses {
        let Some(MissClaim::Follower(waiter)) = claims[index].as_ref() else {
            continue;
        };
        let key = &uniques[index].0;
        if matches!(
            waiter.wait_blocking_timeout(conn.rpc_timeout()),
            crate::runtime::ResultFlightOutcome::Stored
        ) {
            let entry = cache.get(key).or_else(|| cache.get_for_revalidation(key));
            if let Some(entry) = entry {
                if let [batch] = entry.batches() {
                    if batch.num_rows() == 1 && batch.num_columns() == 1 {
                        conn.runtime.note_exchange_cache_hit(entry.bytes());
                        outputs[index] = Some(batch.clone());
                        continue;
                    }
                }
            }
        }
        outputs[index] = Some(direct(&mut client, &uniques[index].1, &ScanOptions::default())?.0);
    }

    let unique_outputs = outputs
        .into_iter()
        .map(|batch| {
            batch.ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "{function} per-value cache assembly left an input unanswered"
                ))
            })
        })
        .collect::<DFResult<Vec<_>>>()?;
    let distinct = concat_batches(bound.output_schema(), &unique_outputs)?;
    let indices = UInt32Array::from(
        row_to_unique
            .into_iter()
            .map(|index| index as u32)
            .collect::<Vec<_>>(),
    );
    let output = take(distinct.column(0).as_ref(), &indices, None)?;
    Ok(RecordBatch::try_new(
        bound.output_schema().clone(),
        vec![output],
    )?)
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
            default: None,
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
    fn any_varargs_infer_untyped_nulls_from_identical_peers() {
        let specs = varargs(DataType::Null);
        assert_eq!(
            VgiScalarUdf::resolve_untyped_null_varargs(
                &specs,
                &[DataType::Null, DataType::Int64, DataType::Int64],
            ),
            vec![DataType::Int64, DataType::Int64, DataType::Int64]
        );
    }

    #[test]
    fn any_varargs_do_not_guess_for_mixed_or_all_null_inputs() {
        let specs = varargs(DataType::Null);
        let mixed = [DataType::Null, DataType::Int64, DataType::Float64];
        assert_eq!(
            VgiScalarUdf::resolve_untyped_null_varargs(&specs, &mixed),
            mixed
        );
        let nulls = [DataType::Null, DataType::Null];
        assert_eq!(
            VgiScalarUdf::resolve_untyped_null_varargs(&specs, &nulls),
            nulls
        );
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
    fn omitted_trailing_defaults_match_the_advertised_overload() {
        let required = positional(DataType::Int64);
        let mut optional = positional(DataType::Utf8);
        optional.default = Some(r#""fallback""#.to_string());
        let defaulted = ArgSpecs(vec![required, optional]);
        let other = ArgSpecs(vec![
            positional(DataType::Boolean),
            positional(DataType::Boolean),
        ]);
        let selected_udf = udf(vec![other, defaulted.clone()]);

        let (selected, compatible) = selected_udf.select_specs(&[DataType::Int64]);
        assert!(compatible);
        assert_eq!(selected, &defaulted);

        let mut only = positional(DataType::Int64);
        only.default = Some("0".to_string());
        let all_defaulted = udf(vec![ArgSpecs(vec![only])]);
        assert!(matches!(
            &all_defaulted.signature().type_signature,
            TypeSignature::OneOf(variants) if variants.contains(&TypeSignature::Nullary)
        ));
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

    #[test]
    fn nullary_named_only_and_varargs_overloads_reach_zero_argument_bind() {
        let nullary = udf(vec![ArgSpecs::default()]);
        assert!(matches!(
            &nullary.signature().type_signature,
            TypeSignature::OneOf(variants)
                if variants.contains(&TypeSignature::Nullary)
                    && variants.contains(&TypeSignature::VariadicAny)
        ));

        let mut named = positional(DataType::Utf8);
        named.is_named = true;
        let named_only = udf(vec![ArgSpecs(vec![named])]);
        assert!(matches!(
            &named_only.signature().type_signature,
            TypeSignature::OneOf(variants) if variants.contains(&TypeSignature::Nullary)
        ));

        let varargs = udf(vec![varargs(DataType::Int64)]);
        assert!(matches!(
            &varargs.signature().type_signature,
            TypeSignature::OneOf(variants) if variants.contains(&TypeSignature::Nullary)
        ));

        let unary = udf(vec![ArgSpecs(vec![positional(DataType::Int64)])]);
        assert_eq!(unary.signature().type_signature, TypeSignature::VariadicAny);
    }

    #[test]
    fn cached_scalar_row_owns_only_its_materialized_buffers() {
        use std::sync::Arc;

        use datafusion::arrow::array::{Array, RecordBatch, StringArray};
        use datafusion::arrow::datatypes::{Field, Schema};

        let values = (0..1024)
            .map(|index| format!("row-{index:04}-{}", "x".repeat(128)))
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(values.clone()))],
        )
        .unwrap();

        let row = materialize_output_row(&batch, 777).unwrap();
        let value = row
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(value.value(0), values[777]);
        assert_eq!(row.num_rows(), 1);
        assert!(
            row.column(0).get_array_memory_size() * 100 < batch.column(0).get_array_memory_size(),
            "one cached row should not retain the full RPC response"
        );
        assert_eq!(PER_VALUE_MAX_NEW_STORES_PER_CALL, 256);
    }
}
