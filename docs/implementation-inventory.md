# VGI × DataFusion implementation inventory

Reviewed against the local DataFusion 55, `vgi-rust`, and `vgi-rpc` checkouts on
26 August 2026. This inventory favors VGI capabilities that map onto existing
DataFusion APIs. It does not propose recreating DuckDB operators inside
DataFusion where there is no matching planning or execution seam.

## Implemented in this change set

### Result and plan caching

- Each DataFusion session owns a bounded in-memory VGI result cache. Defaults
  are 64 MiB per entry, 256 MiB total, and 131,072 entries.
- Embedders may additionally configure a host-owned durable producer-result
  tier. It stores complete multipart results as Arrow IPC with a configured
  codec (Zstandard by default), survives runtime and process restarts, and
  coordinates concurrent local processes with atomic refs plus capture/replay
  leases. Durable admission counts committed Arrow payloads and leased orphan
  generations, so an active reader cannot hide payload capacity from a new
  commit. It is not a hard filesystem quota: ref/manifest/lease metadata and
  active temporary captures are excluded, and each concurrent capture may grow
  to the configured per-capture byte limit. Processes sharing a root should use
  the same constructor limits because policy values are not persisted. This
  tier is opt-in; the host must give it application-state storage rather than a
  temporary spill path.
- Results are cached only when both the worker opts in and the attachment has
  not disabled caching. Complete producer scans and complete split scans are
  captured; an error, cancellation, limit, or incomplete partition aborts the
  candidate atomically.
- Secret dependency is fail-closed: a first bind that declares any secret
  requirement bypasses plan and result caches even when the host resolver
  returns no matching rows.
- Keys isolate catalog identity and version, authenticated identity, worker and
  function, arguments, attach options, typed session settings, remotely applied
  query shape, and limit. Projection and filter identity is capability-aware:
  advertised pushdown stays in the key; explicitly local filters do not; and
  unknown legacy filter behavior remains conservatively keyed.
  A worker that does not advertise projection pushdown stores one full result
  and DataFusion conforms each hit to the requested local projection, including
  a zero-column `count(*)` projection.
  Split results with advertised ordering are not cached because flattened
  replay could violate DataFusion plan properties.
- Producer results, stateless table-input batches, stable scalar per-value
  entries, and buffered whole-input results support ETag/Last-Modified
  conditional revalidation, including immediate-stale validator policies, and
  worker-authorized stale-if-error fallback.
- Bounded non-split producer entries in memory honor worker-advertised
  stale-while-revalidate when the effective attachment has an RPC timeout. The
  query replays stale bytes immediately, while one runtime-local result flight
  refreshes in the background; fresh rows are captured atomically, validator
  responses slide the entry, and errors or revocation affect later reads rather
  than the stale response already authorized by the worker. Split, durable,
  scalar, and table-input SWR remain excluded.
- The durable tier persists validators for complete, bounded, catalog-scoped
  producer and split scans. Non-split producers can conditionally revalidate
  after runtime restart, atomically slide the observed generation, honor
  worker-authorized stale-if-error, and conditionally remove that generation
  when reuse is revoked. Split scans validate every nonempty planned group
  serially on partition zero and replay only after unanimous compatible
  `not_modified` responses. A fresh, mixed, incompatible, or revoked response
  discards the validation wave, conditionally removes the selected durable
  generation, and reruns every split without validators; validation errors fail
  closed rather than replaying stale bytes.
- Concurrent identical misses are coalesced per complete cache key. One query
  fills the entry while followers replay the atomic result; cancellation,
  refusal, expiry, and `no_store` wake followers to execute normally rather
  than making cache policy a correctness dependency.
- Catalog-scoped split plans are reused only for their advertised lifetime.
  Version-scoped and transaction-scoped plans are not reused.
- Stateless parallel streaming table-in/out calls can memoize each complete
  input batch after the worker advertises cache control. Stateful FINALIZE,
  serial, literal-single-row, and secret-dependent calls are excluded.
- Unordered buffering functions can memoize their complete input multiset.
  Canonical row digests make the key independent of input order and DataFusion
  batch boundaries while preserving duplicates; `sink_order_dependent`
  functions are excluded. A result commits only after the complete lifecycle
  and every finalize stream succeed, and cancellation or an over-cap capture
  returns the query result without storing partial state.
- Stable scalar functions can opt into per-value memoization. Distinct tuples
  are sent once, partial hits send only misses, and outputs are gathered back in
  caller order. Table-input and scalar exchange misses use the same per-key
  single-flight discipline as producers. Volatile and secret-dependent calls
  take the direct path. Adapter-owned `vgi_exchange_input_dedup` and
  `vgi_result_cache_per_value` settings default on and support `SET`/`RESET`.
- Conditional exchange failures may serve stale bytes only within the worker's
  advertised `stale_if_error` window. A `not_modified` or fresh response that
  withdraws eligibility (`no_store`, transaction scope, or per-value opt-in)
  evicts the old entry instead of silently extending or retaining it.
- SQL diagnostics expose cache statistics, entries, flushing, reaping, and plan
  cache statistics. Entry inspection reports real batch and producer-substream
  counts, time-travel coordinates, the memory/disk tier, tier-specific bytes,
  and an empty partition label for whole-result entries. Memory bytes measure
  retained Arrow memory and its substreams are executed producer streams; disk
  bytes measure encoded Arrow payloads and its substream count is the stored
  partition-file count. Disk rows report durable revalidation capability,
  while per-entry hit counts remain unavailable. Aggregate durable
  revalidation and stale-serve counters are exposed. Flush and reap operate on
  both configured tiers. Producer scan/cache counters also
  appear as native DataFusion execution metrics and in `EXPLAIN ANALYZE`;
  result-cache statistics distinguish exchange hits, stores, and bytes served.
- Cache admission vetoes use one credential-free reason vocabulary across
  producer, scalar, streaming exchange, and buffered exchange execution.
  Producer reporting emits one `cache.ineligible` event per standard
  DataFrame/`execute_stream_partitioned` run, deduplicated across partitions
  sharing its `TaskContext`; custom physical-plan callers determine that
  boundary through task-context reuse. Scalar and table-input reporting is per
  UDF worker batch or exchange invocation because DataFusion's async scalar API
  exposes no SQL-query execution identity. Values, filter contents, split
  tokens, identities, and secrets are never included in these adapter-generated
  veto events.
- SQL-owned `vgi_result_cache_max_entry_bytes`,
  `vgi_result_cache_max_bytes`, and `vgi_result_cache_max_entries` settings
  update the live session cache immediately. Reductions evict entries that no
  longer fit, constructor-supplied defaults remain visible, and `RESET`
  restores those constructor values.

The durable tier is intentionally narrower than the memory tier: it currently
covers bounded producer and split scans, not scalar values, streaming or
buffered table-input exchanges, correlated 1:N calls, dynamic-filtered scans,
unbounded scans, or secret-dependent calls. Stale-while-revalidate remains
memory-only for bounded non-split producers. Eviction recency is approximate
and process-local, although bounds
and publication are coordinated across processes. Crash-durability relies on a
local Unix filesystem with advisory locks, atomic same-filesystem rename, and
directory `fsync`; network filesystems with weaker semantics are unsupported.
Cross-process locks protect storage publication, integrity, and bounds; result
single-flight remains runtime-local, so independent cold processes may both
invoke the worker before one committed generation wins publication.
Disk event counters and LRU touches are instance-local and nonpersistent; only
entry and occupancy snapshots rescan the shared root. Durable diagnostics,
flush, and reap perform blocking filesystem work when invoked through SQL.
Flushing removes lookup refs immediately, while an object leased by an active
replay remains physically present until release and a later reap.

### Catalog and function fidelity

- Worker views are qualified to their attachment and registered as DataFusion
  `ViewTable`s after catalog and function discovery.
- Catalog-owned macro and view SQL qualifies recognized one-part and
  schema-qualified two-part worker objects to the attachment while preserving
  builtins, CTEs, and already fully-qualified names.
- `vgi_catalogs(location)` is registered before any attachment and returns the
  canonical VGI discovery schema, including typed attach-option and release
  lists.
- Worker-nominated global scalar, aggregate, table, and buffered functions use
  the declared VGI prefix. `global_functions false` opts one attachment out;
  existing registry owners win collisions, concurrent publication is
  linearized, and `DETACH` or replacement removes only alias-owned entries.
- VGI-only companion catalogs attach recursively with depth and cycle guards;
  required unsupported companions fail rather than silently disappearing.
- Multi-branch catalog tables execute worker-nominated CSV, Parquet, JSON/NDJSON,
  and Arrow file arms through DataFusion's registered file-format factories and
  `ListingTable`. Typed paths and options are preserved, schemas reconcile by
  column name, and eligible filters reach the underlying native provider.
- Catalog-table arms resolve already attached DataFusion providers by alias or
  unique VGI worker-catalog identity. They share reconciliation and pushdown,
  and direct or indirect source cycles fail with an explicit path.
- Function stability maps to DataFusion volatility. Directly bound table
  functions map exact worker filter application to
  `TableProviderFilterPushDown::Exact`; lazy catalog tables recheck locally.
- Aggregate functions preserve bind-time ConstParams. DataFusion sliding
  accumulators use the worker's dedicated VGI window callback when advertised;
  ordinary `GROUP BY` continues through aggregate update/finalize.
- Scalar overload selection rejects incompatible typed arms without losing
  ConstParam literal coercion, allowing `ANY` arms or the worker's authoritative
  bind error instead of silently casting into the wrong overload.
- Reviewed scalar corpus overlays express DuckDB-only unsigned casts, typed
  struct constructors, list length, and binary length through DataFusion's
  native `arrow_cast`, `named_struct`, `array_length`, and encoding functions.
- Worker setting declarations retain their Arrow types and defaults. A dynamic
  DataFusion `ConfigExtension` accepts native `SET vgi.name`, while the adapter
  also qualifies attached workers' unprefixed `SET name` spelling and handles
  `RESET`. Scalar, table, and table-in/out binds receive typed one-row IPC;
  Struct values use JSON strings. `duckdb_settings()` projects the same live
  declarations and values for corpus/CLI inspection.
- Catalog-table `required_filters` are enforced as the protocol's AND-of-OR
  groups. Nested struct paths remain precise, parent predicates satisfy child
  requirements, and errors identify the missing groups before a scan starts.

### Secrets, runtime hooks, and observability

- An embedder can install an asynchronous `VgiSecretResolver`. Producer,
  table, scalar, and input binds resolve typed, scoped secret requests and
  retry the bind once. Aggregate declarations resolve their static secrets.
- The adapter never copies resolved secret values into SQL, cache keys, or its
  constructed structural event fields. Secret-dependent plan, producer,
  exchange, and per-value results bypass caches so resolver rotation cannot
  reuse stale credential-derived output. Worker-originated log, RPC, exception,
  and error text is untrusted and may be retained verbatim in diagnostics; a
  worker is responsible for not rendering secrets there. SQL `CREATE SECRET`
  and raw secret values in `ATTACH` remain unsupported.
- Deterministic coverage verifies multiple same-type, differently scoped
  secrets in one bind and rejects duplicate resolved names without exposing
  credential values.
- `VgiEventSink` receives structured catalog, plan, scan, cache, error, and
  cancellation events. The session also retains a bounded event history for
  `vgi_logs()` and `vgi_log_stats()`. Successful buffered begin/combine/finalize
  calls, scalar input writes, and streaming table-input sends report structural
  counts without arguments, row values, secrets, or synthetic events on cache
  replay.
- DataFusion routes in-band worker logs on built-in subprocess, Unix, TCP, and
  HTTP connections as severity-specific `worker.log.*` events to that same sink
  and SQL history, with structured extras retained as JSON text. A standalone
  `vgi-client` uses the Rust `log` facade when no embedding sink is installed.
  These worker-controlled payloads are untrusted and may contain anything the
  worker chooses to log, including sensitive values; the adapter does not redact
  them. Worker-originated RPC, exception, and error text carried into error
  diagnostics has the same trust boundary.
  The protocol log message does not carry request/function correlation; worker
  access logs and subprocess stderr remain separate operator-facing channels.
- `VgiLocalityHook` exposes planned split locations to an embedding scheduler.
  The adapter does not invent DataFusion hash partitioning from VGI transforms.
- `VgiSessionOptions` configures cache bounds, event history, and an optional
  RPC timeout. HTTP, Unix, and TCP clients honor that timeout; `None` leaves
  long-lived calls unbounded. Subprocess pipe deadlines remain unavailable.
- A positive `rpc_timeout` ATTACH option overrides that session default for one
  attachment. Blocking exchange followers use the same deadline.
- Dropping an unfinished producer scan sends one protocol cancellation. Open,
  header/decode, read, or cancellation failures poison the owning client so a
  pooled connection is discarded; natural end-of-stream remains reusable.
- A non-buffered table-input call with a pushed finite limit stays inside the
  physical plan and streams through bounded channels. DataFusion backpressure,
  local truncation, and cancellation stop the child and worker without
  materializing the complete input; partial exchanges bypass result caching.
- The single table-input column may carry nested Arrow lists, fixed-size lists,
  maps, structs, dictionaries, temporal values, decimals, and binary values.
  Reviewed DataFusion-native corpus overlays preserve those wire shapes; Arrow
  Union and DuckDB BIT constructors remain unavailable in DataFusion SQL.
- Start-only VGI splits now declare `Boundedness::Unbounded` instead of being
  rejected or misrepresented as bounded.

## SQL/runtime capability map

| Capability | Status | Notes |
|---|:---:|---|
| ATTACH / DETACH | Supported | DuckDB-style syntax, typed worker options, OAuth, cache/global-function policy, per-attach RPC timeout, and companion discovery |
| Catalogs, schemas, tables, views | Supported | Pre-attach `vgi_catalogs(location)` uses the canonical schema; views are registered after dependencies and qualified to their alias |
| Native format scan branches | Supported | CSV, Parquet, JSON/NDJSON, and Arrow arms use registered DataFusion formats; custom formats require a host registration |
| Scalar and SQL macro functions | Supported | Async scalar UDFs plus scalar/table macro expansion with typed defaults and named arguments |
| Typed session settings | Supported | DataFusion `ConfigExtension`, `SET`/`RESET`, Arrow scalar and Struct encoding, metadata view, cache isolation |
| Aggregate/window-frame use | Partial | ConstParams, retract, and advertised sliding-window callbacks work; DataFusion lacks EXCLUDE and aggregate-local ORDER BY window forms |
| Table and buffered functions | Supported | One input column can carry nested Arrow values; multiple top-level input columns remain constrained by scalar-subquery planning |
| Global functions | Supported | Default-on/opt-out policy, collision ownership, concurrent attach linearization, replacement, and DETACH cleanup included |
| Projection, static filters, LIMIT | Supported | Direct functions preserve exactness; lazy catalog tables recheck filters |
| Cardinality metadata | Supported | Catalog-inlined estimate/max values feed `TableProvider::statistics()` and `VgiScanExec` through DataFusion 55's `StatisticsContext`; equal estimate/max is exact, other estimates remain inexact, and historical scans do not reuse current-table cardinality |
| Set operations and joins | Supported | VGI scans compose with UNION/INTERSECT/EXCEPT, IN/EXISTS, CTEs, and DataFusion semi/anti plans; the SQL adapter maps DuckDB's unqualified `SEMI JOIN`/`ANTI JOIN` spelling to the equivalent left-directed plans, including in nested SELECTs |
| Split planning | Supported | Parallel partitions, ordering properties, plan cache, unbounded metadata |
| Session result cache | Partial | The bounded memory tier covers producer/split scans, streaming per-batch, stable scalar per-value, and unordered buffered whole-input results with conditional revalidation, runtime-local single-flight, stale-if-error, and revocation eviction. Bounded non-split producer entries additionally honor worker SWR under an effective RPC timeout, serving immediately while one background flight refreshes. An opt-in Arrow IPC tier (Zstandard by default) durably shares complete bounded producer/split results across local processes with leases, atomic publication, restart recovery, and combined SQL diagnostics/maintenance. Non-split producer validators survive restart with conditional refresh, stale-if-error, and exact-generation revocation. Split validators require unanimous serial all-group agreement before replay; fresh, mixed, or revoked votes trigger an unconditional whole-result rerun, while validation errors fail closed. Durable bounds are constructor-owned; live SQL limits govern memory. Durable exchange/scalar/correlated entries and durable/split/exchange/scalar SWR remain unwired |
| Logs and diagnostics | Supported | SQL tables/scalars plus an embedder event sink; built-in transports forward in-band worker logs, without request/function correlation, access logs, or stderr |
| Worker-requested secrets | Supported | Host resolver API; no SQL secret store |
| Locality | Partial | Host callback exists; DataFusion CLI has no distributed scheduler |
| Correlated LATERAL table calls | Not wired | DataFusion binds table functions before an outer row is available |
| Correlated scalar subqueries with VGI aggregates | Not wired | DataFusion 55 leaves a `ScalarSubquery` expression that its physical planner cannot lower; uncorrelated scalar aggregates and correlated IN/EXISTS paths work |
| Dynamic join filters | Partial | Single-column `IN` and same-column equality-OR sets use VGI v2 side IPC at init, with exact scalar types preserved. Small multi-column hash-join tuple sets are safely decomposed into per-column marginal sets for worker pruning while DataFusion retains the exact tuple join locally. Later constant/range generations use `vgi_pushdown_filters` over subprocess, Unix/TCP, and plain or authenticated HTTP continuations; large hash/Bloom state and tuple-correlated protocol expressions remain local |
| TABLESAMPLE hints | Supported | DataFusion's relation-planner API maps VGI `SYSTEM` percentage/seed sampling into split planning and scan initialization; plan/result cache identities include the exact hint, while host-owned Bernoulli sampling is not misrepresented as VGI SYSTEM sampling |
| ORDER BY / Top-N hints | Supported | An opt-in session-builder extension recognizes direct-column DataFusion Sort/Top-K/GlobalLimit plans, sends advisory ordering through split planning and scan initialization, and keys plan/result caches by the hint. DataFusion retains host Sort/Top-K and limit semantics; computed expressions are rejected and early-stop limits are withheld for filters and multi-key ordering |
| Table time travel | Supported | Fully-qualified VGI tables accept literal `AT (VERSION => …)` and `AT (TIMESTAMP => …)`; historical schemas and cache identities are isolated |
| Catalog metadata | Supported | `SHOW TABLES`, `SHOW COLUMNS`, `SHOW FUNCTIONS`, and DataFusion `information_schema` expose VGI tables, views, columns, schemata, and routines without eager scan-function binds |
| Table constraints | Partial | Primary-key and unique metadata feed `TableProvider::constraints()`; DataFusion has no check, foreign-key, standalone NOT NULL, or `duckdb_constraints()` metadata surface |
| Transactions / snapshot pinning | Not wired | Requires SQL lifecycle interception and catalog-scoped transaction semantics |
| DML, DDL, and custom COPY | Not wired | Requires vgi-rust RPC wrappers and DataFusion sink/provider integration |

## Existing DataFusion APIs worth using next

1. **Cache breadth with matching semantics.** Add correlated 1:N entries only
   where a complete deterministic key and cancellation boundary can be proved.
   Keep durable storage opt-in, evaluate compact scalar/exchange packing, and
   extend the bounded non-split producer SWR contract only where equivalent
   cancellation and atomic replacement semantics exist.
2. **Unbounded execution hardening.** Gate resume on worker advertisement and
   add checkpoint/reconnect, cancellation, backpressure, and soak coverage.
3. **Catalog and function breadth.** Broaden view translation and keep
   host-native SQL spelling differences in sparse reviewed overlays. Scalar and
   aggregate applicable corpus coverage is complete; future additions should
   continue using the worker as the coercion authority.
4. **Dynamic-filter breadth.** Add a VGI representation for DataFusion's large
   hash/Bloom lookup and tuple-correlated multi-column membership expressions;
   the current per-column marginal decomposition is safe but intentionally
   inexact. Decide whether completed filters may safely refine split
   enumeration before init.
5. **Metadata and documentation surfaces.** Preserve worker docs, examples,
   tags, settings requirements, and late-materialization declarations for host
   inspection before building custom operators around them.

## Deliberately deferred boundaries

Correlated LATERAL table functions, multi-column streaming table inputs, and
late-materialization rewrites do not map
cleanly onto the current `TableProvider` callback. They should remain explicit
gaps unless DataFusion exposes a suitable API or the project chooses to own a
custom logical/physical extension. That keeps the integration focused on
wiring VGI into DataFusion rather than developing a parallel query engine.

## Production verification gate

The current promoted 327-file shared VGI SQLLogicTest baseline completes
against both HTTP and an explicitly managed Unix-socket worker. Each transport
executes 3,554 records with zero timeouts and produces 2,278 exact, 128
rendering-equivalent, and 169 genuinely different results. The promoted reports
are byte-identical.
The remaining failures are tracked capability gaps, primarily DuckDB-only SQL
and diagnostics, correlated table calls, wide table input, writes, and secret
host configuration. Publishing, stalled subprocess-RPC cancellation,
long-lived unbounded-stream tests, and consuming a released `vgi-rpc` with the
HTTP blocking-dispatch fix remain release gates.

The focused 10-file multi-branch slice additionally executes 75/75 positive
records with all 53 comparable results exact over subprocess, Unix sockets, and
loopback HTTP. This includes native CSV/Parquet arms, typed format options,
schema reconciliation, branch filtering, and split redemption.

Post-baseline focused verification also covers capability-aware scan cache
identity and full-result replay across local projections and filters; truthful
cache entry layout, time-travel, tier, and byte diagnostics; concurrent producer
and split cache fills; producer, table-input, scalar per-value, and buffered
whole-input immediate-stale revalidation; exchange single-flight and stale-if-error;
revocation eviction; unordered buffered multiset identity and bounded capture;
truthful scalar/buffered/table-input lifecycle events; incremental finite-LIMIT exchange;
catalog-source provider resolution; pre-attach canonical catalog discovery;
global-function opt-out/collision/lifecycle; scan Drop cancellation and pool
poisoning; bounded Unix/HTTP transport behavior; callback-only window median;
overload incompatibility; native scalar and nested table-input overlays; safe
multi-column dynamic join-key marginals; and multi-scope secret binds. The
reviewed aggregate window slice now executes 15/15 applicable records with all
14 results agreeing.

The last promoted pre-durable release-mode adapter suite passed 249/249 tests
plus two doctests. This
includes the focused cache-identity, multi-column dynamic-filter, and transport
hardening contracts in addition to cache-limit, table-input observability,
scalar-null, deterministic single-flight, and the existing transport and
integration coverage.
The durable-cache change separately passes the full vgi-client all-feature
suite (215 unit tests, 39 integration tests, and one doctest), including a true
multi-process initialization/reopen test, and the four-test DataFusion durable
producer contract over both Unix sockets and patched-transport HTTP. The final
staged release corpus also passes over both transports with byte-identical reports:
3,554 executed records, 580 classified compatibility gaps, zero blocked cases,
zero timeouts, and no regressions against either promoted baseline.
The typed static/dynamic filter-pushdown area executes 185/185 applicable
records with every comparable result agreeing; its remaining records are
reviewed DataFusion SQL/planning boundaries rather than failed VGI filters.
The settings slice executes 42/42 records with all 14 results exact, and the
required-filter slice executes 45/45 applicable records with all 25 results
exact, over both Unix and HTTP. A 13-file
aggregate/macro/filter/cache promotion slice is identical over Unix and HTTP:
both execute 145/173 records and all 113 comparable queries agree. Aggregate
executes 42/42; macro/catalog executes 24/24; cache executes 30/38. Separately,
the completed seven-file focused cache slice executes 79/79 applicable records
with zero failures over both transports; its two genuine diagnostic
differences record the cache-ineligibility reason logs that were missing in that
snapshot. Sanitized reasons are implemented now; this historical result is not
relabeled until the focused corpus is regenerated.
