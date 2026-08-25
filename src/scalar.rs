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
    specs: ArgSpecs,
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
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: name.clone(),
            registered_name: name,
            signature,
            return_type: Some(return_type),
            specs: ArgSpecs::default(),
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
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            registered_name: registered_name.into(),
            signature: Signature::variadic_any(Volatility::Volatile),
            return_type: None,
            specs,
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
        types: &[DataType],
        values: &[Option<ArgValue>],
    ) -> (Arguments, Vec<Field>) {
        let mut arguments = Arguments::new();
        let mut columns = Vec::new();
        for (i, ty) in types.iter().enumerate() {
            if self.specs.positional_is_const(i) {
                let value = values
                    .get(i)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| ArgValue::Null(ty.clone()));
                arguments = arguments.positional(value);
            } else {
                columns.push(Field::new(
                    format!("col_{}", columns.len()),
                    ty.clone(),
                    true,
                ));
            }
        }
        (arguments, columns)
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

        let (arguments, columns) = self.split_arguments(arg_types, values);
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

            let bound = client
                .bind_with_input(&attached, &spec, &input_schema)
                .map_err(to_df)?;
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
        let values: Vec<Option<ArgValue>> = args
            .scalar_arguments
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.and_then(|s| crate::table_function::scalar_to_arg(&self.function, i, s).ok())
            })
            .collect();
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
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::{Field, Schema};

        let rows = args.number_rows;

        // Const parameters are resolved at bind and must not be shipped as
        // columns; everything else is a column. See `split_arguments`.
        let types: Vec<DataType> = args.args.iter().map(|a| a.data_type()).collect();
        let values: Vec<Option<ArgValue>> = args
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                ColumnarValue::Scalar(s) => {
                    crate::table_function::scalar_to_arg(&self.function, i, s).ok()
                }
                // A literal may arrive already expanded across the batch — the
                // engine is free to materialise it — and then every row holds
                // the same constant, so row 0 is the value the bind needs.
                // Without this a const parameter silently becomes a typed null
                // and the worker answers with NULLs.
                ColumnarValue::Array(arr) if self.specs.positional_is_const(i) => {
                    ArgValue::from_array_row0(arr.as_ref(), &self.function).ok()
                }
                ColumnarValue::Array(_) => None,
            })
            .collect();
        let (arguments, _) = self.split_arguments(&types, &values);

        let columns: Vec<datafusion::arrow::array::ArrayRef> = args
            .args
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.specs.positional_is_const(*i))
            .map(|(_, a)| a.to_array(rows))
            .collect::<DFResult<_>>()?;
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
        let input = RecordBatch::try_new(Arc::new(Schema::new(fields.clone())), columns)?;

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

            let bound = client
                .bind_with_input(&attached, &spec, &input_schema)
                .map_err(to_df)?;
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

        if out.num_rows() != rows {
            return Err(DataFusionError::Execution(format!(
                "{} answered {} rows for {} input rows; a scalar function must be 1:1",
                self.registered_name,
                out.num_rows(),
                rows
            )));
        }
        Ok(ColumnarValue::Array(out.column(0).clone()))
    }
}
