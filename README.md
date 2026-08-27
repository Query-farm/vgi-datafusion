<div align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-rust/main/docs/vgi-logo.png" alt="Vector Gateway Interface" width="320">
</div>

# vgi-datafusion

Query a remote **VGI** catalog from Apache DataFusion with ordinary SQL.

```rust
let conn = VgiConnection::subprocess(["my-worker"]);
let ctx = SessionContext::new();
ctx.register_table(
    "orders",
    VgiTableProvider::bind(conn, "my_catalog", "main", "orders").await?,
)?;
ctx.sql("SELECT count(*) FROM orders").await?.show().await?;
```

## What maps, and what does not

| VGI shape | DataFusion seam | Status |
|---|---|:-:|
| Table function (producer) | `TableProvider` + custom `ExecutionPlan` | ✅ |
| Catalog / schema discovery | `CatalogProvider` / `SchemaProvider` | ✅ |
| SQL metadata | `SHOW` + `information_schema` | ✅ tables, views, columns, schemata, routines |
| Table time travel | version-specific `TableProvider` + VGI `At` | ✅ |
| Views and prefixed global functions | `ViewTable` + UDF/UDTF registries | ✅ |
| Scalar and table SQL macros | sqlparser AST expansion into existing expressions/relations | ✅ |
| Set operations, CTEs, semi/anti joins | existing DataFusion logical/physical plans + SQL spelling normalization | ✅ |
| Scalar function | `AsyncScalarUDFImpl` | ✅ |
| Aggregate and sliding window-frame use | `AggregateUDFImpl` with retract | ◐ |
| Projection & LIMIT pushdown | `scan(projection, limit)` | ✅ |
| Filter pushdown | `supports_filters_pushdown` | ✅, exact for directly bound functions |
| Dynamic filters and join keys | physical filter pushdown + continuation metadata | ◐ |
| Split planning | physical scan partitions + plan cache | ✅ |
| Worker-opted result cache | bounded memory + opt-in durable producer tier | ◐ |
| Typed session settings | DataFusion `ConfigExtension` + `SET` / `RESET` | ✅ |
| Worker secrets | host `VgiSecretResolver` | ✅ |
| Structured diagnostics and worker logs | SQL functions + host event sink | ✅ |
| Table-in-out / buffered | table-valued subquery argument | ✅, one input column |

An exchange-mode VGI function is reached with a scalar subquery as its TABLE
argument. DataFusion constrains a scalar subquery to one column, so wider table
inputs remain unavailable without an upstream DataFusion planner change.

## VGI-enabled DataFusion CLI

Build the CLI from the sibling DataFusion checkout after adding
`vgi-datafusion` to `datafusion-cli`:

```shell
cd ../datafusion
cargo build -p datafusion-cli
./target/debug/datafusion-cli
```

Attach a worker once per CLI session. Both the DuckDB spelling and the compact
query-string spelling are accepted:

```sql
ATTACH 'open_meteo' AS m (
  TYPE vgi,
  LOCATION 'https://vgi-open-meteo.rusty-bb6.workers.dev'
);

-- Equivalent:
ATTACH 'open_meteo?location=https://vgi-open-meteo.rusty-bb6.workers.dev' AS m;
```

The alias becomes a DataFusion catalog, so remote objects are addressed as
`m.<schema>.<function-or-table>`. `DETACH m;` removes it from the session.

Catalog tables that advertise VGI time travel accept the same fully-qualified
DuckDB syntax. Historical schema evolution is reflected during DataFusion
planning, and version/timestamp coordinates are isolated in both result and
split-plan cache keys:

```sql
SELECT *
FROM m.data.versioned_table AT (VERSION => 2);

SELECT *
FROM m.data.versioned_table
  AT (TIMESTAMP => TIMESTAMP '2024-01-01');
```

The coordinate must be a non-NULL integer or timestamp literal. Transactions
and catalog-wide snapshot pinning remain separate, unsupported features.

The CLI enables DataFusion's `information_schema`, and VGI discovery metadata
is exposed without binding or scanning every remote function:

```sql
SHOW TABLES;
SHOW COLUMNS FROM m.data.versioned_table;
SHOW FUNCTIONS LIKE '%m_%';

SELECT * FROM information_schema.views
WHERE table_catalog = 'm';
```

These are DataFusion's native metadata surfaces. The adapter also supplies the
small `duckdb_*` compatibility projections used by the shared corpus, backed by
the same retained catalog state. VGI primary-key and unique constraints feed
DataFusion's native optimizer constraint API, while check, foreign-key, and
standalone NOT NULL metadata have no matching DataFusion constraint type.

Catalog discovery is also available before `ATTACH`. The canonical VGI
`vgi_catalogs` schema—including typed attach-option and release lists—is exposed
directly from a worker LOCATION:

```sql
SELECT * FROM vgi_catalogs('https://worker.example/');
```

Worker-declared settings are installed in DataFusion's dynamic `vgi.*`
configuration namespace and encoded to their advertised Arrow types at every
bind. Once a worker is attached, its unqualified DuckDB spelling is accepted
too:

```sql
SET multiplier = 5;              -- rewritten to vgi.multiplier
SET vgi.scale_factor = 2.5;      -- native DataFusion spelling
SET vgi.config = '{"start":10,"step":5,"label":"item"}';
RESET multiplier;

SELECT * FROM duckdb_settings() WHERE name = 'multiplier';
```

JSON strings provide DataFusion's spelling for Arrow Struct settings. Setting
values are included in split-plan, producer-result, exchange, and scalar-type
cache identity, so changing a setting cannot cross-serve cached state. The
adapter also owns five host settings. Stable scalar input dedup is controlled by
`vgi_exchange_input_dedup`, worker-opted scalar per-value memoization can be
vetoed with `vgi_result_cache_per_value`, and
`vgi_result_cache_max_entry_bytes`, `vgi_result_cache_max_bytes`, and
`vgi_result_cache_max_entries` bound session cache memory. All support
unqualified `SET`/`RESET`; adapter settings reach a worker only if it
independently declares a setting with the same name.

HTTP attachments automatically follow an RFC 9728 OAuth challenge. The CLI
prints a device URL and code, then retries the attach after approval; access and
refresh tokens remain in the shared attachment state and are not persisted by
the adapter. A static bearer or previously-issued refresh token can also be
seeded with DuckDB-compatible options:

```sql
-- Interactive OAuth is automatic:
ATTACH 'volcanos' (
  TYPE vgi,
  LOCATION 'https://vgi-volcanos.fly.dev/'
);

-- Explicit credentials (do not put these in checked-in SQL):
ATTACH 'private' AS p (
  TYPE vgi,
  LOCATION 'https://worker.example/',
  bearer_token '...'
);
```

`oauth_refresh_token` is mutually exclusive with `bearer_token`. Credential
values are redacted from adapter/client debug output, and authenticated pool
entries are isolated by their shared auth state.

VGI worker-declared attach options are also accepted directly in the option
list. The adapter discovers their Arrow types, evaluates constant SQL values,
casts them to the declared schema, enforces required/unknown options, and sends
one typed row to `catalog_attach`. Local options currently implemented are
`pool`, `pool_max`, `pool_timeout`, `worker_debug`,
`rpc_timeout`, `global_functions`,
`launcher_idle_timeout`, `launcher_state_dir`, `data_version_spec`,
`implementation_version`, `cache`, `attach_companions`, and
`allow_local_format_paths`. Launcher options are
rejected outside a `launch:` LOCATION. Worker-declared global functions are
registered automatically; `global_functions false` keeps their qualified
catalog forms but suppresses global aliases. Existing native/global registry
owners win collisions, concurrent attaches choose one owner, and `DETACH` or
replacement removes only aliases still owned by that attachment. Raw `secrets`
and `attach_companion_secrets` options fail explicitly; embedders instead
install a scoped `VgiSecretResolver` on the session's `VgiRuntime`.

Catalog scan branches may nominate CSV, Parquet, JSON/NDJSON, or Arrow files;
the adapter reads them with DataFusion's registered native format factories.
Subprocess, launcher, Unix, and loopback network workers may nominate local
paths. Non-loopback HTTP/TCP workers may nominate configured object-store URLs,
but local paths require the explicit client-side
`allow_local_format_paths true` trust opt-in.

The worker must opt a result into caching. Attach with `cache false` to veto it
for a catalog. Inspect the session-owned cache and structured event history from
SQL:

```sql
SELECT * FROM vgi_cache_stats();
SELECT * FROM vgi_cache_entries();
SELECT * FROM vgi_plan_cache_stats();
SELECT * FROM vgi_logs();
SELECT vgi_cache_flush(), vgi_logs_clear();
```

DataFusion routes in-band worker logs from built-in subprocess, Unix, TCP, and
HTTP connections as `worker.log.*` events to the same SQL history and host event
sink; structured extras are retained as JSON text. (`vgi-client` uses the Rust
`log` facade when no embedding sink is installed.) The current wire message has
no request or function identifier, so these events cannot yet be correlated to
one call. Worker access logs and subprocess stderr are separate operator-facing
channels and are not forwarded into DataFusion diagnostics. In-band message text
and extras are untrusted, worker-controlled payloads and are retained verbatim;
the adapter cannot guarantee that a worker will not log sensitive values.
Worker-originated RPC, exception, and error text may likewise be retained in
error diagnostics and must be treated as untrusted.

For an embedded application, put a configured `VgiRuntime` into the
`SessionConfig` extensions before constructing the `SessionContext`. The same
runtime can carry a secret resolver, non-blocking event sink, and split-locality
hook. If no extension is supplied, the SQL adapter creates one shared runtime
for all VGI attachments in that session.

Enable the optional durable result-cache tier with a host-owned directory. The
constructor supplies conservative byte/count bounds and Zstandard compression;
its public fields can be overridden before creating the runtime.

```rust
use vgi_datafusion::{VgiDurableCacheOptions, VgiRuntime, VgiSessionOptions};

let mut options = VgiSessionOptions::default();
options.durable_cache = Some(VgiDurableCacheOptions::new("/var/cache/my-app/vgi"));
let runtime = VgiRuntime::try_new(options).expect("open durable VGI cache");
```

Run the complete Open-Meteo example with a labelled 30-second timeout around
each interaction:

```shell
cargo run --example open_meteo
```

Set `VGI_OPEN_METEO_LOCATION` to test another deployment and
`VGI_QUERY_TIMEOUT_SECS` to change the per-query deadline. The harness geocodes
Glen Allen, feeds the returned coordinates into `forecast_hourly`, expands the
worker's weather-code SQL macros, and prints six forecast rows.

## Two constraints worth knowing before you build

**DataFusion 55 is the supported release.** It uses Arrow 59.2, matching
`vgi-protocol` and the vgi-rust workspace. The manifest includes both version
`55.0.0` and a sibling-checkout path: published consumers use the released
crate, while local `datafusion-cli` development uses one shared copy of
DataFusion's types.

**Worker RPCs are blocking.** `vgi-client` is synchronous, like the Python and
Java VGI clients, so every call runs inside `spawn_blocking`. `VgiConnection` is
a *factory* rather than a client: physical partitions use independent pooled
connections. `rpc_timeout <positive-seconds>` on `ATTACH` overrides the session
default for that attachment on HTTP, Unix, and TCP. Dropping an unfinished scan
or satisfying a LIMIT sends protocol cancellation; failed open/header/read or
cancel cleanup poisons that connection so the pool cannot reuse a desynchronized
stream. Subprocess pipe I/O cannot yet enforce the timeout.

## Known gaps

- **Streaming positions.** A split that names a start position without an end
  now declares `Boundedness::Unbounded`. Checkpoint/resume policy, reconnect,
  backpressure, and long-running soak coverage still need production hardening.
- **Partitioning and locality.** Catalog-inlined cardinality, partition bounds,
  and column statistics feed DataFusion's provider and physical statistics APIs,
  and within-split ordering is advertised when a physical partition contains
  exactly one split. VGI partition transforms are not mapped
  blindly to DataFusion hash partitioning because their hash and partition-number
  semantics differ. A host locality hook receives worker location hints, but the
  standalone DataFusion CLI has no distributed scheduler-affinity implementation.
- **Cache breadth.** Complete, worker-opted producer and split results have a
  safe bounded memory tier with conditional revalidation and per-key
  single-flight for concurrent misses. Stateless streaming table-in/out calls
  additionally memoize complete input batches, and stable scalar functions may
  opt into bounded per-value memoization. Both exchange tiers use validators for
  immediate-stale conditional revalidation, coalesce identical misses, and honor
  worker-authorized stale-if-error. A worker that withdraws caching with
  `no_store` or another ineligible policy evicts the stale bytes instead of
  replaying them. A call is secret-dependent as soon as its first bind declares
  a secret requirement, even when the host resolver returns no matching rows;
  those calls bypass these caches. Native
  DataFusion metrics expose producer cache/worker activity in `EXPLAIN ANALYZE`.
  Cache diagnostics distinguish exchange hits, stores, and bytes served.
  Producer cache vetoes emit one credential-free `cache.ineligible` event per
  standard DataFrame/`execute_stream_partitioned` run, deduplicated across
  partitions sharing its `TaskContext`. Custom physical-plan callers determine
  that boundary through their own task-context reuse. Scalar and table-input
  vetoes emit the same stable `reason=...` vocabulary per UDF worker batch or
  exchange invocation; DataFusion's async scalar API exposes no SQL-query
  execution identity on which to deduplicate multiple batches.
  `vgi_result_cache_max_entry_bytes`, `vgi_result_cache_max_bytes`, and
  `vgi_result_cache_max_entries` apply live bounded-memory limits through SQL;
  lowering a limit evicts entries that no longer fit and `RESET` restores the
  session constructor values.
  Buffered functions additionally cache the complete input multiset after a
  successful finalize. Their keys preserve duplicate multiplicity while being
  independent of row order and physical batch boundaries; ordered sinks,
  secrets, cancellation, failed lifecycle phases, inconsistent policy, and
  over-cap results never commit. An optional host-owned durable tier persists
  complete bounded producer and split results as Arrow IPC using the configured
  codec (Zstandard by default) and shares them safely between local processes.
  Non-split producers persist ETag/Last-Modified policy, conditionally
  revalidate after restart, honor stale-if-error, and remove only the observed
  durable generation when the worker revokes reuse. Stale split results are
  validated atomically: partition zero serially asks every nonempty split group,
  and replays only after unanimous compatible `not_modified` responses. Any
  fresh, mixed, or revoked response removes the selected generation and reruns
  every split without conditions; validation errors fail closed rather than
  serving stale bytes. Bounded non-split producer entries in memory honor a
  worker-advertised stale-while-revalidate window when the attachment has an
  RPC timeout: callers receive stale rows immediately while one runtime-local
  background flight conditionally refreshes or revokes the entry. The durable
  tier is not used for exchange, scalar, correlated
  1:N, dynamic-filtered, unbounded,
  ordered-split, secret-dependent, or non-catalog-scoped calls. Its recency
  ordering is approximate per process, and its crash-safety contract requires
  a local Unix filesystem with advisory locks, atomic rename, and directory
  `fsync`. Constructor limits govern committed Arrow payload admission rather
  than forming a hard filesystem quota, are separate from live SQL memory
  limits, and should agree across processes sharing one root. Storage locks do
  not coalesce worker calls across processes. Durable, split, scalar, and
  exchange stale-while-revalidate remain unimplemented.
- **Dynamic filters and join keys.** DataFusion 55 hash-join filters are linked
  to VGI scans. Completed single-column `IN` sets use `join_keys` side IPC at
  init (`vgi_join_keys_version=2`), while later range/constant refinements ride
  continuations as standard-base64 `vgi_pushdown_filters` metadata. Subprocess,
  Unix/TCP byte streams, and plain or authenticated HTTP all preserve those
  between-tick updates. The join remains DataFusion's correctness boundary, so
  older workers may ignore the hint safely, and dynamically filtered scans
  bypass the result cache.
  Static and initial runtime predicates preserve supported Arrow scalar types,
  and same-column equality `OR` trees use the same join-key side batch as `IN`.
  DataFusion's hash-table/Bloom lookup expression for very large joins and
  multi-column struct membership are not serializable to VGI yet; dynamic
  filters also prune within already-planned splits rather than changing the
  split set.
- **Correlated LATERAL table functions.** DataFusion binds a table function
  while planning, before an outer row exists. A call such as
  `LATERAL m.main.forecast_hourly(g.latitude, g.longitude)` is therefore not yet
  representable; run the geocoder first and pass its coordinates as constants
  in a second statement.
- **Planner-only hints.** VGI `TABLESAMPLE SYSTEM`
  percentage/seed hints use DataFusion 55's relation-planner extension and are
  included in split planning, scan initialization, and cache identity.
  `BERNOULLI(100 PERCENT)` remains a host-owned identity operation and sends no
  VGI hint. A session-builder extension conservatively maps direct-column
  `ORDER BY ... LIMIT/OFFSET` into VGI planning and scan hints while retaining
  DataFusion's Sort/Top-K and limit semantics as the correctness boundary. Computed sort keys
  are not pushed, and worker early-stop limits are withheld for filters and
  multi-key ordering. Hosts that install another custom QueryPlanner must use
  the composable builder method rather than replacing this bridge afterward.
  Transactions, mutations, and custom COPY formats do not
  yet have an end-to-end adapter path. Table-level
  `AT (VERSION|TIMESTAMP => literal)` time travel is supported independently.
  See `docs/implementation-inventory.md` for the API boundaries and recommended
  sequence.

## SQLLogicTest compatibility

The shared VGI corpus is tracked by capability, group, and individual file.
See [`docs/corpus-compatibility.md`](docs/corpus-compatibility.md) for the
completion rules, current EC2 baseline, adaptation policy, prioritized gaps,
and regression command. [`corpus/compatibility.json`](corpus/compatibility.json)
is the machine-readable status manifest; the runner rejects newly added corpus
groups until they are assigned and reviewed.

Reviewed scalar overlays use DataFusion's native `arrow_cast`, `named_struct`,
`array_length`, and binary encoding functions for DuckDB-only unsigned,
typed-struct, list-length, and binary-length spellings. They adapt SQL syntax,
not VGI execution semantics.

## License

See `LICENSE` at the repository root.
