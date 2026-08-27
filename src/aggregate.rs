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
use datafusion::logical_expr::{
    Accumulator, AggregateUDFImpl, ColumnarValue, Signature, TypeSignature, Volatility,
};
use vgi_client::{ArgSpecs, ArgValue, Arguments};

use crate::{to_df, VgiConnection};

/// Reserved struct field carried by the SQL compatibility pass when a
/// zero-argument aggregate needs a private row-count witness.
pub(crate) const ROW_WITNESS_FIELD: &str = "__vgi_datafusion_row_witness__";

fn is_invocation_row_witness(arg_types: &[DataType]) -> bool {
    matches!(
        arg_types,
        [DataType::Struct(fields)]
            if fields.len() == 1
                && fields[0].name() == ROW_WITNESS_FIELD
                && fields[0].data_type() == &DataType::Int64
    )
}

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
    /// Parameter roles declared by the worker. Const parameters are bind-time
    /// arguments, even though DataFusion evaluates every aggregate expression
    /// into an array for `update_batch`.
    specs: ArgSpecs,
    /// A strictly nullary worker aggregate is exposed to DataFusion with one
    /// private row-witness argument. DataFusion's accumulator API supplies
    /// only `&[]` for a genuine nullary call, which loses the input batch's row
    /// count. The SQL compatibility pass injects a literal `1`; this flag makes
    /// sure it is retained as local state but never sent to the worker.
    row_witness: bool,
    /// Whether the worker provides the dedicated VGI window callback.
    ///
    /// DataFusion asks for a sliding accumulator separately from an ordinary
    /// grouped accumulator. Keeping this bit here lets the former use the
    /// worker's window RPC without changing plain GROUP BY semantics.
    supports_window: bool,
    required_secrets: Vec<vgi_client::SecretLookupRequest>,
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
        Self::new_with_volatility(
            conn,
            catalog,
            schema_name,
            function,
            registered_name,
            Volatility::Volatile,
        )
    }

    /// Describe an aggregate with worker-declared volatility.
    pub fn new_with_volatility(
        conn: VgiConnection,
        catalog: impl Into<String>,
        schema_name: impl Into<String>,
        function: impl Into<String>,
        registered_name: impl Into<String>,
        volatility: Volatility,
    ) -> Self {
        Self {
            conn,
            catalog: catalog.into(),
            schema_name: schema_name.into(),
            function: function.into(),
            registered_name: registered_name.into(),
            signature: Signature::variadic_any(volatility),
            specs: ArgSpecs::default(),
            row_witness: false,
            supports_window: false,
            required_secrets: Vec::new(),
        }
    }

    pub(crate) fn with_arg_specs(mut self, specs: ArgSpecs) -> Self {
        let positional = specs.positional().collect::<Vec<_>>();
        self.row_witness = specs.positional().next().is_none();
        // A declared positional parameter distinguishes a real one-argument
        // call from the private row witness used by a truly nullary aggregate.
        // Until that witness can carry an invocation-arity marker, an
        // all-default positional aggregate must still receive at least one
        // argument; admitting zero would silently aggregate zero rows.
        let minimum_arity = specs.minimum_positional_arity().max(1);
        self.signature = if self.row_witness {
            // SQL rewrites the worker's nullary call to one private literal so
            // every accumulator update carries the number of input rows.
            Signature::any(1, self.signature.volatility)
        } else if positional.iter().any(|spec| spec.is_varargs) {
            // VariadicAny alone rejects zero arguments before VGI can return
            // its authoritative minimum-arity diagnostic.
            Signature::one_of(
                vec![TypeSignature::Nullary, TypeSignature::VariadicAny],
                self.signature.volatility,
            )
        } else if minimum_arity == positional.len() {
            // VGI AnyArrow means any concrete Arrow type, not arbitrary arity.
            Signature::any(positional.len(), self.signature.volatility)
        } else {
            // A VGI default is applied by the worker at bind. DataFusion must
            // admit every arity that can omit a trailing default, but should
            // not synthesize or serialize that default itself.
            let signatures = (minimum_arity..=positional.len())
                .map(|arity| {
                    if arity == 0 {
                        TypeSignature::Nullary
                    } else {
                        TypeSignature::Any(arity)
                    }
                })
                .collect();
            Signature::one_of(signatures, self.signature.volatility)
        };
        self.specs = specs;
        self
    }

    pub(crate) fn with_window_support(mut self, supports_window: bool) -> Self {
        self.supports_window = supports_window;
        self
    }

    pub(crate) fn with_required_secrets(
        mut self,
        required_secrets: Vec<vgi_client::SecretLookupRequest>,
    ) -> Self {
        self.required_secrets = required_secrets;
        self
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
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arg_types = if self.row_witness || is_invocation_row_witness(arg_types) {
            Vec::new()
        } else {
            arg_types.to_vec()
        };
        let specs = self.specs.clone();
        let required_secrets = self.required_secrets.clone();
        crate::run_blocking_planner_call(move || {
            bind_output_type(
                &conn,
                &catalog,
                &schema_name,
                &function,
                &arg_types,
                &specs,
                &required_secrets,
            )
        })
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
        self.make_accumulator(args, false)
    }

    fn create_sliding_accumulator(&self, args: AccumulatorArgs) -> DFResult<Box<dyn Accumulator>> {
        self.make_accumulator(args, self.supports_window)
    }
}

impl VgiAggregateUdf {
    fn make_accumulator(
        &self,
        args: AccumulatorArgs,
        use_window_callback: bool,
    ) -> DFResult<Box<dyn Accumulator>> {
        let arg_types = args
            .expr_fields
            .iter()
            .map(|field| field.data_type().clone())
            .collect::<Vec<_>>();
        let row_witness = self.row_witness || is_invocation_row_witness(&arg_types);
        let arguments = if row_witness {
            Arguments::new()
        } else {
            aggregate_arguments(&self.function, &self.specs, args.exprs)?
        };
        Ok(Box::new(VgiAccumulator {
            conn: self.conn.clone(),
            catalog: self.catalog.clone(),
            schema_name: self.schema_name.clone(),
            function: self.function.clone(),
            arg_types,
            specs: self.specs.clone(),
            arguments,
            required_secrets: self.required_secrets.clone(),
            use_window_callback,
            row_witness,
            buffered: Vec::new(),
            window_worker: None,
        }))
    }
}

/// Bind the aggregate to learn its output type.
pub(crate) fn bind_output_type(
    conn: &VgiConnection,
    catalog: &str,
    schema_name: &str,
    function: &str,
    arg_types: &[DataType],
    specs: &ArgSpecs,
    required_secrets: &[vgi_client::SecretLookupRequest],
) -> DFResult<DataType> {
    use datafusion::arrow::datatypes::Schema;
    use vgi_client::{BindSpec, FunctionType};

    let (arguments, fields) = typed_null_arguments(arg_types, specs);
    let input_schema = Schema::new(fields);
    let mut client = conn.connect()?;
    let attached = conn.attach(&mut client, catalog)?;
    let mut spec = BindSpec::table(function).in_schema(schema_name);
    spec.function_type = FunctionType::Aggregate;
    spec.arguments = arguments;

    let bound = aggregate_bind(
        conn,
        &mut client,
        &attached,
        &spec,
        &input_schema,
        required_secrets,
    )?;
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

fn column_input_fields(arg_types: &[DataType], specs: &ArgSpecs) -> Vec<Field> {
    arg_types
        .iter()
        .enumerate()
        .filter(|(i, _)| !specs.positional_is_const(*i))
        .enumerate()
        .map(|(column, (_, ty))| Field::new(format!("col_{column}"), ty.clone(), true))
        .collect()
}

fn typed_null_arguments(arg_types: &[DataType], specs: &ArgSpecs) -> (Arguments, Vec<Field>) {
    let mut arguments = Arguments::new();
    for (i, ty) in arg_types.iter().enumerate() {
        if specs.positional_is_const(i) {
            arguments = arguments.positional(ArgValue::Null(ty.clone()));
        }
    }
    (arguments, column_input_fields(arg_types, specs))
}

/// Extract aggregate ConstParams from DataFusion's physical expressions.
///
/// Evaluating against an empty batch handles literals wrapped in casts as well
/// as bare `Literal` expressions. A real column cannot evaluate there and is
/// rejected: taking row zero would silently give a row-varying argument
/// bind-time semantics.
fn aggregate_arguments(
    function: &str,
    specs: &ArgSpecs,
    exprs: &[Arc<dyn datafusion::physical_expr::PhysicalExpr>],
) -> DFResult<Arguments> {
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::datatypes::Schema;

    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let mut arguments = Arguments::new();
    for (i, expr) in exprs.iter().enumerate() {
        if !specs.positional_is_const(i) {
            continue;
        }
        let value = match expr.evaluate(&empty) {
            Ok(ColumnarValue::Scalar(value)) => value,
            _ => {
                return Err(DataFusionError::Plan(format!(
                    "VGI aggregate `{function}` argument {i} is a ConstParam and must be a constant"
                )))
            }
        };
        arguments =
            arguments.positional(crate::table_function::scalar_to_arg(function, i, &value)?);
    }
    Ok(arguments)
}

/// Accumulates input rows, then runs the whole aggregate at `evaluate`.
#[derive(Debug)]
struct VgiAccumulator {
    conn: VgiConnection,
    catalog: String,
    schema_name: String,
    function: String,
    arg_types: Vec<DataType>,
    specs: ArgSpecs,
    arguments: Arguments,
    required_secrets: Vec<vgi_client::SecretLookupRequest>,
    /// Evaluate the current sliding frame with the worker's dedicated window
    /// callback instead of its ordinary aggregate finalize implementation.
    use_window_callback: bool,
    /// Whether argument zero exists only to preserve DataFusion's input row
    /// count for a nullary worker aggregate.
    row_witness: bool,
    /// One buffer per argument; `buffered[i]` are the chunks seen for argument
    /// `i`, concatenated only when needed.
    buffered: Vec<Vec<ArrayRef>>,
    /// A dedicated blocking thread retains one HTTP/stream connection and one
    /// aggregate bind across every frame in this accumulator. DataFusion's
    /// sliding-accumulator API invokes `evaluate` once per output row; opening
    /// the complete VGI lifecycle for each invocation made a 5,000-row window
    /// perform tens of thousands of avoidable HTTP requests.
    window_worker: Option<WindowWorker>,
}

enum WindowCommand {
    Evaluate {
        batch: datafusion::arrow::record_batch::RecordBatch,
        reply: std::sync::mpsc::SyncSender<DFResult<ScalarValue>>,
    },
    Stop,
}

struct WindowWorker {
    commands: std::sync::mpsc::Sender<WindowCommand>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for WindowWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowWorker").finish_non_exhaustive()
    }
}

impl WindowWorker {
    #[allow(clippy::too_many_arguments)]
    fn start(
        conn: VgiConnection,
        catalog: String,
        schema_name: String,
        function: String,
        arguments: Arguments,
        required_secrets: Vec<vgi_client::SecretLookupRequest>,
        input_schema: Arc<datafusion::arrow::datatypes::Schema>,
    ) -> DFResult<Self> {
        use vgi_client::{BindSpec, FunctionType};

        let (commands, receiver) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
        let thread = std::thread::Builder::new()
            .name("vgi-window-rpc".to_string())
            .spawn(move || {
                let initialized = (|| {
                    let mut client = conn.connect()?;
                    let attached = conn.attach(&mut client, &catalog)?;
                    let mut spec = BindSpec::table(&function).in_schema(&schema_name);
                    spec.function_type = FunctionType::Aggregate;
                    spec.arguments = arguments;
                    let bound = aggregate_bind(
                        &conn,
                        &mut client,
                        &attached,
                        &spec,
                        &input_schema,
                        &required_secrets,
                    )?;
                    Ok::<_, DataFusionError>((client, attached, bound))
                })();
                let (mut client, attached, bound) = match initialized {
                    Ok(session) => {
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        session
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };

                let mut reusable = true;
                while let Ok(command) = receiver.recv() {
                    match command {
                        WindowCommand::Evaluate { batch, reply } => {
                            let result =
                                evaluate_window_frame(&mut client, &bound, &function, &batch);
                            reusable = result.is_ok();
                            let failed = result.is_err();
                            if failed {
                                client.poison();
                            }
                            if reply.send(result).is_err() || failed {
                                break;
                            }
                        }
                        WindowCommand::Stop => break,
                    }
                }
                if reusable {
                    let _ = client.with(|client| client.aggregate_destroy(&attached, &bound, &[0]));
                }
            })
            .map_err(|error| DataFusionError::External(Box::new(error)))?;

        ready_rx.recv().map_err(|_| {
            DataFusionError::Execution(
                "VGI window worker exited before initialization completed".to_string(),
            )
        })??;
        Ok(Self {
            commands,
            thread: Some(thread),
        })
    }

    fn evaluate(
        &self,
        batch: datafusion::arrow::record_batch::RecordBatch,
    ) -> DFResult<ScalarValue> {
        let (reply, result) = std::sync::mpsc::sync_channel(0);
        self.commands
            .send(WindowCommand::Evaluate { batch, reply })
            .map_err(|_| {
                DataFusionError::Execution("VGI window worker is no longer running".to_string())
            })?;
        result.recv().map_err(|_| {
            DataFusionError::Execution(
                "VGI window worker exited without returning a result".to_string(),
            )
        })?
    }
}

impl Drop for WindowWorker {
    fn drop(&mut self) {
        let _ = self.commands.send(WindowCommand::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn evaluate_window_frame(
    client: &mut vgi_client::PooledClient,
    bound: &vgi_client::BoundAggregate,
    function: &str,
    batch: &datafusion::arrow::record_batch::RecordBatch,
) -> DFResult<ScalarValue> {
    let rows = batch.num_rows() as i64;
    let partition = client
        .with(|client| client.window_init(bound, 0, batch))
        .map_err(to_df)?;
    let evaluated = client.with(|client| client.window_evaluate(&partition, 0, &[(0, rows)]));
    let out = match evaluated {
        Ok(out) => out,
        Err(error) => return Err(to_df(error)),
    };
    client
        .with(|client| client.window_destroy(&partition))
        .map_err(to_df)?;

    if out.num_rows() != 1 || out.num_columns() == 0 {
        return Err(DataFusionError::Execution(format!(
            "VGI window aggregate `{function}` evaluated to {} rows / {} columns; expected one value",
            out.num_rows(),
            out.num_columns()
        )));
    }
    ScalarValue::try_from_array(out.column(0), 0)
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

    fn retract_batch(&mut self, values: &[ArrayRef]) -> DFResult<()> {
        let rows = values.first().map(|array| array.len()).unwrap_or(0);
        if values.iter().any(|array| array.len() != rows) {
            return Err(DataFusionError::Internal(format!(
                "VGI aggregate `{}` received retract columns with different lengths",
                self.function
            )));
        }
        for chunks in &mut self.buffered {
            let mut remaining = rows;
            while remaining > 0 {
                let Some(first) = chunks.first() else {
                    return Err(DataFusionError::Internal(format!(
                        "VGI aggregate `{}` retracted more rows than it holds",
                        self.function
                    )));
                };
                if first.len() <= remaining {
                    remaining -= first.len();
                    chunks.remove(0);
                } else {
                    chunks[0] = first.slice(remaining, first.len() - remaining);
                    remaining = 0;
                }
            }
        }
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
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

        let collected = self.collected()?;
        let row_count = collected.first().map(|column| column.len()).unwrap_or(0);
        let columns = collected
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !self.row_witness && !self.specs.positional_is_const(*i))
            .map(|(_, column)| column)
            .collect();
        let schema = Arc::new(Schema::new(if self.row_witness {
            Vec::new()
        } else {
            column_input_fields(&self.arg_types, &self.specs)
        }));
        let options = datafusion::arrow::record_batch::RecordBatchOptions::new()
            .with_row_count(Some(row_count));
        let batch = RecordBatch::try_new_with_options(schema.clone(), columns, &options)?;
        if self.use_window_callback {
            if self.window_worker.is_none() {
                self.window_worker = Some(WindowWorker::start(
                    self.conn.clone(),
                    self.catalog.clone(),
                    self.schema_name.clone(),
                    self.function.clone(),
                    self.arguments.clone(),
                    self.required_secrets.clone(),
                    schema,
                )?);
            }
            return self
                .window_worker
                .as_ref()
                .expect("window worker was initialized")
                .evaluate(batch);
        }
        let conn = self.conn.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let function = self.function.clone();
        let arguments = self.arguments.clone();
        let required_secrets = self.required_secrets.clone();

        // DataFusion invokes `Accumulator::evaluate` synchronously while
        // polling its async execution streams. HTTP uses reqwest::blocking,
        // which panics if it creates or destroys its private runtime there, so
        // the whole aggregate RPC lifecycle belongs on a blocking OS thread.
        crate::run_blocking_planner_call(move || {
            use vgi_client::{with_group_ids, BindSpec, FunctionType};

            let mut client = conn.connect()?;
            let attached = conn.attach(&mut client, &catalog)?;
            let mut spec = BindSpec::table(&function).in_schema(&schema_name);
            spec.function_type = FunctionType::Aggregate;
            spec.arguments = arguments;
            let bound = aggregate_bind(
                &conn,
                &mut client,
                &attached,
                &spec,
                &schema,
                &required_secrets,
            )?;

            // A single group: this accumulator *is* one group, and DataFusion
            // has already partitioned the rows for us.
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
                    "VGI aggregate `{function}` finalized to {} rows / {} columns; expected one value",
                    out.num_rows(),
                    out.num_columns()
                )));
            }
            ScalarValue::try_from_array(out.column(0), 0)
        })
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

fn aggregate_bind(
    conn: &VgiConnection,
    client: &mut vgi_client::VgiClient,
    attached: &vgi_client::AttachedCatalog,
    spec: &vgi_client::BindSpec,
    input_schema: &datafusion::arrow::datatypes::Schema,
    required_secrets: &[vgi_client::SecretLookupRequest],
) -> DFResult<vgi_client::BoundAggregate> {
    if required_secrets.is_empty() {
        return client
            .aggregate_bind(attached, spec, input_schema)
            .map_err(to_df);
    }
    let secrets = crate::resolve_secret_batch(conn.runtime(), required_secrets.to_vec())?;
    client
        .aggregate_bind_with_resolved_secrets(attached, spec, input_schema, secrets)
        .map_err(to_df)
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::DataType;
    use vgi_client::{ArgSpec, ArgSpecs};

    use super::*;

    fn argument(name: &str, is_named: bool, is_const: bool) -> ArgSpec {
        ArgSpec {
            name: name.to_string(),
            data_type: DataType::Int64,
            is_const,
            is_named,
            is_varargs: false,
            default: None,
            doc: None,
        }
    }

    #[test]
    // Regression: an all-named declaration still has no row-bearing input,
    // while an all-ConstParam positional declaration retains its old arity.
    fn named_only_signature_is_nullary_but_positional_const_is_not() {
        let aggregate = VgiAggregateUdf::new(
            VgiConnection::subprocess(["unused"]),
            "example",
            "main",
            "named_default",
            "named_default",
        )
        .with_arg_specs(ArgSpecs(vec![argument("option", true, true)]));
        assert!(aggregate.row_witness);

        let aggregate = VgiAggregateUdf::new(
            VgiConnection::subprocess(["unused"]),
            "example",
            "main",
            "positional_const",
            "positional_const",
        )
        .with_arg_specs(ArgSpecs(vec![argument("option", false, true)]));
        assert!(!aggregate.row_witness);
    }

    #[test]
    fn trailing_defaults_expand_the_accepted_aggregate_arities() {
        let required = argument("value", false, false);
        let mut first_default = argument("precision", false, true);
        first_default.default = Some("2".to_string());
        let mut second_default = argument("ignore_nulls", false, true);
        second_default.default = Some("true".to_string());

        let aggregate = VgiAggregateUdf::new(
            VgiConnection::subprocess(["unused"]),
            "example",
            "main",
            "defaulted",
            "defaulted",
        )
        .with_arg_specs(ArgSpecs(vec![required, first_default, second_default]));

        assert_eq!(
            aggregate.signature.type_signature,
            TypeSignature::OneOf(vec![
                TypeSignature::Any(1),
                TypeSignature::Any(2),
                TypeSignature::Any(3),
            ])
        );
        assert!(!aggregate.row_witness);
    }

    #[test]
    fn reserved_struct_type_marks_only_the_zero_argument_invocation() {
        let marker = DataType::Struct(
            vec![Arc::new(Field::new(
                ROW_WITNESS_FIELD,
                DataType::Int64,
                false,
            ))]
            .into(),
        );
        assert!(is_invocation_row_witness(&[marker]));
        assert!(!is_invocation_row_witness(&[DataType::Int64]));

        let mut defaulted = argument("value", false, true);
        defaulted.default = Some("1".to_string());
        let aggregate = VgiAggregateUdf::new(
            VgiConnection::subprocess(["unused"]),
            "example",
            "main",
            "defaulted",
            "defaulted",
        )
        .with_arg_specs(ArgSpecs(vec![defaulted]));
        assert_eq!(aggregate.signature.type_signature, TypeSignature::Any(1));
        assert!(!aggregate.row_witness, "the marker is per invocation");
    }
}
