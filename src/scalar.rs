// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! A VGI scalar function as a DataFusion async UDF.
//!
//! `AsyncScalarUDFImpl` is DataFusion's purpose-built seam for remote
//! functions — its own docs say so — and it is what makes a per-batch RPC
//! respectable rather than a blocked runtime thread. `AsyncFuncExec` hoists the
//! calls out of the projection and batches them at `ideal_batch_size`.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature};

use crate::{to_df, VgiConnection};

/// A remote scalar function.
#[derive(Debug, Clone)]
pub struct VgiScalarUdf {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    name: String,
    signature: Signature,
    return_type: DataType,
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
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            name: name.into(),
            signature,
            return_type,
            batch_size: None,
        }
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
        self.name.hash(state);
        self.catalog.hash(state);
        self.schema_name.hash(state);
    }
}

impl PartialEq for VgiScalarUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.catalog == other.catalog
            && self.schema_name == other.schema_name
    }
}

impl Eq for VgiScalarUdf {}

impl ScalarUDFImpl for VgiScalarUdf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> DFResult<DataType> {
        Ok(self.return_type.clone())
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        // Reached only if something bypassed AsyncFuncExec; a remote call has
        // no synchronous form.
        Err(DataFusionError::Internal(format!(
            "{} is a remote function and must be invoked asynchronously",
            self.name
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
            self.name.clone(),
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
                self.name,
                out.num_rows(),
                rows
            )));
        }
        Ok(ColumnarValue::Array(out.column(0).clone()))
    }
}
