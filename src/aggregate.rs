// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! A VGI aggregate as a DataFusion `AggregateUDF`.
//!
//! # Why the partial state is the input itself
//!
//! DataFusion splits an aggregate into a partial pass and a final pass, and
//! asks each accumulator to hand over an intermediate state that another one
//! can merge. VGI has no way to produce that: its state lives **in the worker**,
//! keyed by `(execution_id, group_id)`, and there is no RPC that serialises it
//! out. `aggregate_combine` takes a merge batch that maps group ids *within one
//! execution* — it cannot merge two separate executions, which is exactly what
//! a partial/final split would need.
//!
//! Nor can the state be guessed at. A VGI aggregate is an arbitrary worker
//! function; a running sum, an ML model, a t-digest — the framework has no idea
//! which. The only intermediate state that is universally correct for an
//! unknown aggregate is **the input rows themselves**, so that is what this
//! carries: rows accumulate locally, merge by concatenation, and the whole
//! exchange with the worker happens once, at `evaluate`.
//!
//! That trades memory for correctness, and the trade is deliberate. An
//! aggregate that is wrong under parallelism is worse than one that holds its
//! input, and this way *every* worker aggregate composes correctly with
//! DataFusion's plans rather than only the associative ones.
//!
//! # Consequences worth knowing
//!
//! * Memory is O(input rows per group). A high-cardinality `GROUP BY` over a
//!   large input will hold it all.
//! * `size()` is reported honestly so DataFusion's memory accounting can see
//!   it, rather than under-reporting and being surprised.
//! * The worker is contacted once per group, at finalize — not per batch.

use std::sync::Arc;

use datafusion::arrow::array::{new_empty_array, Array, ArrayRef};
use datafusion::arrow::compute::concat;
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{DataFusionError, Result as DFResult, ScalarValue};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{Accumulator, AggregateUDFImpl, Signature, Volatility};

use crate::{to_df, VgiConnection};

/// A remote aggregate function.
#[derive(Debug, Clone)]
pub struct VgiAggregateUdf {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    /// The aggregate's name on the worker — what gets bound.
    function: String,
    /// The name DataFusion knows it by, which is the registry key.
    registered_name: String,
    signature: Signature,
}

impl VgiAggregateUdf {
    /// Describe an aggregate discovered from a catalog.
    ///
    /// The signature is permissive for the same reason a scalar's is: VGI
    /// resolves arity and types at bind, so the worker stays the authority and
    /// a bad call reports the worker's own error.
    pub fn new(
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
        }
    }
}

impl std::hash::Hash for VgiAggregateUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.registered_name.hash(state);
        self.catalog.hash(state);
        self.schema_name.hash(state);
    }
}

impl PartialEq for VgiAggregateUdf {
    fn eq(&self, other: &Self) -> bool {
        self.registered_name == other.registered_name
            && self.catalog == other.catalog
            && self.schema_name == other.schema_name
    }
}

impl Eq for VgiAggregateUdf {}

impl AggregateUDFImpl for VgiAggregateUdf {
    fn name(&self) -> &str {
        &self.registered_name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DFResult<DataType> {
        // The worker resolves the output type at bind, from the argument
        // types — there is nothing static to declare, so ask it.
        bind_output_type(
            &self.conn,
            &self.catalog,
            &self.schema_name,
            &self.function,
            arg_types,
        )
    }

    /// The intermediate state: one list per argument, holding the input rows.
    ///
    /// See the module docs — a remote aggregate has no serialisable state of
    /// its own, so the input *is* the state.
    fn state_fields(&self, args: StateFieldsArgs) -> DFResult<Vec<FieldRef>> {
        Ok(args
            .input_fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                Arc::new(Field::new(
                    format!("{}_input_{i}", self.registered_name),
                    DataType::List(Arc::new(Field::new("item", f.data_type().clone(), true))),
                    false,
                ))
            })
            .collect())
    }

    fn accumulator(&self, args: AccumulatorArgs) -> DFResult<Box<dyn Accumulator>> {
        Ok(Box::new(VgiAccumulator {
            conn: self.conn.clone(),
            catalog: self.catalog.clone(),
            schema_name: self.schema_name.clone(),
            function: self.function.clone(),
            arg_types: args
                .exprs
                .iter()
                .map(|e| e.data_type(args.schema))
                .collect::<DFResult<Vec<_>>>()?,
            buffered: Vec::new(),
        }))
    }
}

/// Bind the aggregate to learn its output type.
fn bind_output_type(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arg_types: &[DataType],
) -> DFResult<DataType> {
    use datafusion::arrow::datatypes::Schema;
    use vgi_client::{AttachOptions, BindSpec, FunctionType};

    let input_schema = Schema::new(input_fields(arg_types));
    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let mut spec = BindSpec::table(function).in_schema(schema_name);
    spec.function_type = FunctionType::Aggregate;

    let bound = client
        .aggregate_bind(&attached, &spec, &input_schema)
        .map_err(to_df)?;
    let out = bound
        .output_schema()
        .fields()
        .first()
        .map(|f| f.data_type().clone())
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "VGI aggregate `{function}` bound to an output schema with no columns"
            ))
        })?;
    // The bind allocated worker state; release it rather than leaving it for
    // the idle sweep.
    let _ = client.aggregate_destroy(&attached, &bound, &[0]);
    Ok(out)
}

/// The worker's expected input column names, matching the extension's `col_<i>`.
fn input_fields(arg_types: &[DataType]) -> Vec<Field> {
    arg_types
        .iter()
        .enumerate()
        .map(|(i, t)| Field::new(format!("col_{i}"), t.clone(), true))
        .collect()
}

/// Accumulates input rows, then runs the whole aggregate at `evaluate`.
#[derive(Debug)]
struct VgiAccumulator {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    arg_types: Vec<DataType>,
    /// One buffer per argument; `buffered[i]` are the chunks seen for argument
    /// `i`, concatenated only when needed.
    buffered: Vec<Vec<ArrayRef>>,
}

impl VgiAccumulator {
    fn push(&mut self, values: &[ArrayRef]) {
        if self.buffered.is_empty() {
            self.buffered = vec![Vec::new(); values.len()];
        }
        for (i, v) in values.iter().enumerate() {
            if let Some(slot) = self.buffered.get_mut(i) {
                slot.push(Arc::clone(v));
            }
        }
    }

    /// One array per **declared argument**, everything seen so far.
    ///
    /// Keyed on `arg_types` rather than on the buffer: an accumulator that
    /// never saw a row has an empty buffer, and iterating that would yield a
    /// batch with no columns at all — which fails schema validation ("number of
    /// columns(0) must match number of fields(1)") instead of aggregating over
    /// nothing. An empty group is a legitimate outcome of a GROUP BY.
    fn collected(&self) -> DFResult<Vec<ArrayRef>> {
        self.arg_types
            .iter()
            .enumerate()
            .map(|(i, ty)| match self.buffered.get(i) {
                Some(chunks) if !chunks.is_empty() => {
                    let refs: Vec<&dyn Array> = chunks.iter().map(|c| c.as_ref()).collect();
                    concat(&refs).map_err(DataFusionError::from)
                }
                _ => Ok(new_empty_array(ty)),
            })
            .collect()
    }
}

impl Accumulator for VgiAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DFResult<()> {
        self.push(values);
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DFResult<()> {
        // Each state is a list column: one list per partial accumulator, whose
        // items are that accumulator's rows. Flattening them back gives the
        // same multiset of inputs, which is what makes this correct for an
        // aggregate whose semantics are unknown.
        use datafusion::arrow::array::ListArray;

        let mut flattened = Vec::with_capacity(states.len());
        for state in states {
            let list = state.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "VGI aggregate `{}` state is not a list",
                    self.function
                ))
            })?;
            let mut parts: Vec<ArrayRef> = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                if list.is_valid(i) {
                    parts.push(list.value(i));
                }
            }
            let refs: Vec<&dyn Array> = parts.iter().map(|p| p.as_ref()).collect();
            flattened.push(if refs.is_empty() {
                new_empty_array(&list.value_type())
            } else {
                concat(&refs).map_err(DataFusionError::from)?
            });
        }
        self.push(&flattened);
        Ok(())
    }

    fn state(&mut self) -> DFResult<Vec<ScalarValue>> {
        // One list per argument, each holding every row this accumulator saw.
        self.collected()?
            .into_iter()
            .map(|a| {
                let items = (0..a.len())
                    .map(|i| ScalarValue::try_from_array(&a, i))
                    .collect::<DFResult<Vec<_>>>()?;
                Ok(ScalarValue::List(ScalarValue::new_list_nullable(
                    &items,
                    a.data_type(),
                )))
            })
            .collect()
    }

    fn evaluate(&mut self) -> DFResult<ScalarValue> {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::Schema;
        use vgi_client::{with_group_ids, AttachOptions, BindSpec, FunctionType};

        let columns = self.collected()?;
        let schema = Arc::new(Schema::new(input_fields(&self.arg_types)));
        let batch = RecordBatch::try_new(schema.clone(), columns)?;

        let mut client = self.conn.connect()?;
        let attached = self.conn.attach(&mut client, &self.catalog)?;
        let mut spec = BindSpec::table(&self.function).in_schema(&self.schema_name);
        spec.function_type = FunctionType::Aggregate;
        let bound = client
            .aggregate_bind(&attached, &spec, &schema)
            .map_err(to_df)?;

        // A single group: this accumulator *is* one group, and DataFusion has
        // already partitioned the rows for us.
        let group_ids = vec![0i64; batch.num_rows()];
        let with_ids = with_group_ids(&group_ids, &batch).map_err(to_df)?;
        client
            .aggregate_update(&attached, &bound, &with_ids)
            .map_err(to_df)?;

        let out = client
            .aggregate_finalize(&attached, &bound, &[0])
            .map_err(to_df)?;
        let _ = client.aggregate_destroy(&attached, &bound, &[0]);

        if out.num_rows() != 1 || out.num_columns() == 0 {
            return Err(DataFusionError::Execution(format!(
                "VGI aggregate `{}` finalized to {} rows / {} columns; expected one value",
                self.function,
                out.num_rows(),
                out.num_columns()
            )));
        }
        ScalarValue::try_from_array(out.column(0), 0)
    }

    fn size(&self) -> usize {
        // Reported honestly: this accumulator really does hold its input, and
        // under-reporting would hide that from DataFusion's memory accounting.
        std::mem::size_of_val(self)
            + self
                .buffered
                .iter()
                .flat_map(|chunks| chunks.iter())
                .map(|a| a.get_array_memory_size())
                .sum::<usize>()
    }
}
