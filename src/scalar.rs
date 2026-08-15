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
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};

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
    /// Memoised bind-time return types, keyed on the argument types.
    resolved: Arc<Mutex<HashMap<Vec<DataType>, DataType>>>,
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
    ) -> Self {
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            registered_name: registered_name.into(),
            signature: Signature::variadic_any(Volatility::Volatile),
            return_type: None,
            resolved: Arc::new(Mutex::new(HashMap::new())),
            batch_size: None,
        }
    }

    /// Ask the worker what this call returns, and remember the answer.
    ///
    /// `ScalarUDFImpl::return_type` is synchronous and runs during planning, so
    /// this blocks on one bind RPC. That is the direct path rather than a
    /// compromise — `vgi_client` is a blocking client — and the result is
    /// memoised per argument-type list, so a query with a thousand calls to the
    /// same function pays once.
    fn resolve_return_type(&self, arg_types: &[DataType]) -> DFResult<DataType> {
        if let Some(t) = &self.return_type {
            return Ok(t.clone());
        }
        if let Ok(cache) = self.resolved.lock() {
            if let Some(t) = cache.get(arg_types) {
                return Ok(t.clone());
            }
        }

        use datafusion::arrow::datatypes::{Field, Schema};
        use vgi_client::{ArgValue, Arguments, AttachOptions, BindSpec, FunctionType};

        let fields: Vec<Field> = arg_types
            .iter()
            .enumerate()
            .map(|(i, t)| Field::new(format!("arg{i}"), t.clone(), true))
            .collect();
        let input_schema = Schema::new(fields);

        let mut client = self.conn.connect()?;
        let attached = client
            .attach(&self.catalog, AttachOptions::default())
            .map_err(to_df)?;
        let mut spec = BindSpec::table(&self.function).in_schema(&self.schema_name);
        spec.function_type = FunctionType::Scalar;
        let mut args = Arguments::new();
        for t in arg_types {
            args = args.positional(ArgValue::Placeholder(t.clone()));
        }
        spec.arguments = args;

        let bound = client
            .bind_with_input(&attached, &spec, &input_schema)
            .map_err(to_df)?;
        let schema = bound.output_schema();
        let out = schema
            .fields()
            .first()
            .map(|f| f.data_type().clone())
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "VGI scalar `{}` bound to an output schema with no columns",
                    self.function
                ))
            })?;

        if let Ok(mut cache) = self.resolved.lock() {
            cache.insert(arg_types.to_vec(), out.clone());
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
        self.resolve_return_type(args)
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
        let columns: Vec<datafusion::arrow::array::ArrayRef> = args
            .args
            .iter()
            .map(|a| a.to_array(rows))
            .collect::<DFResult<_>>()?;
        let fields: Vec<Field> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| Field::new(format!("arg{i}"), c.data_type().clone(), true))
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
            use vgi_client::{
                ArgValue, Arguments, AttachOptions, BindSpec, FunctionType, ScanOptions,
            };
            let mut client = conn.connect()?;
            let attached = client
                .attach(&cat, AttachOptions::default())
                .map_err(to_df)?;
            let mut spec = BindSpec::table(&name).in_schema(&sch);
            spec.function_type = FunctionType::Scalar;
            // Column arguments are typed placeholders; the values ride the
            // input batch.
            let mut args_builder = Arguments::new();
            for f in input_schema.fields() {
                args_builder =
                    args_builder.positional(ArgValue::Placeholder(f.data_type().clone()));
            }
            spec.arguments = args_builder;

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
