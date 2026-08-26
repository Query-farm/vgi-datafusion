# VGI × DataFusion implementation inventory

Reviewed against the local DataFusion 55, `vgi-rust`, and `vgi-rpc` checkouts on
26 August 2026. This inventory favors VGI capabilities that map onto existing
DataFusion APIs. It does not propose recreating DuckDB operators inside
DataFusion where there is no matching planning or execution seam.

## Implemented in this change set

### Result and plan caching

- Each DataFusion session owns a bounded in-memory VGI result cache. Defaults
  are 64 MiB per entry, 256 MiB total, and 131,072 entries.
- Results are cached only when both the worker opts in and the attachment has
  not disabled caching. Complete producer scans and complete split scans are
  captured; an error, cancellation, limit, or incomplete partition aborts the
  candidate atomically.
- Keys isolate catalog identity and version, authenticated identity, worker and
  function, arguments, attach options, typed session settings, projection,
  static filters, and limit.
  Split results with advertised ordering are not cached because flattened
  replay could violate DataFusion plan properties.
- Large stale entries support ETag/Last-Modified conditional revalidation and
  worker-authorized stale-if-error fallback.
- Concurrent identical misses are coalesced per complete cache key. One query
  fills the entry while followers replay the atomic result; cancellation,
  refusal, expiry, and `no_store` wake followers to execute normally rather
  than making cache policy a correctness dependency.
- Catalog-scoped split plans are reused only for their advertised lifetime.
  Version-scoped and transaction-scoped plans are not reused.
- Stateless parallel streaming table-in/out calls can memoize each complete
  input batch after the worker advertises cache control. Stateful FINALIZE,
  serial, literal-single-row, and secret-consuming calls are excluded.
- Stable scalar functions can opt into per-value memoization. Distinct tuples
  are sent once, partial hits send only misses, and outputs are gathered back in
  caller order. Volatile, secret-consuming, and immediately-stale policies take
  the direct path.
- SQL diagnostics expose cache statistics, entries, flushing, reaping, and plan
  cache statistics. Producer scan/cache counters also appear as native
  DataFusion execution metrics and in `EXPLAIN ANALYZE`; result-cache statistics
  distinguish exchange hits, stores, and bytes served.

This is intentionally narrower than the DuckDB cache. Buffered whole-input
results, correlated 1:N per-value calls, exchange conditional revalidation and
single-flight, persistent disk storage, stale-while-revalidate, and compression
remain deferred.

### Catalog and function fidelity

- Worker views are qualified to their attachment and registered as DataFusion
  `ViewTable`s after catalog and function discovery.
- Catalog-owned macro and view SQL qualifies recognized one-part and
  schema-qualified two-part worker objects to the attachment while preserving
  builtins, CTEs, and already fully-qualified names.
- Worker-nominated global scalar, aggregate, table, and buffered functions use
  the declared VGI prefix and are removed on `DETACH` or replacement.
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
- Secret values stay out of SQL, cache keys, events, and diagnostic tables.
  Secret-dependent plan, producer, exchange, and per-value results bypass
  caches so resolver rotation cannot reuse stale credential-derived output.
  SQL `CREATE SECRET` and raw secret values in `ATTACH` remain unsupported.
- Deterministic coverage verifies multiple same-type, differently scoped
  secrets in one bind and rejects duplicate resolved names without exposing
  credential values.
- `VgiEventSink` receives structured catalog, plan, scan, cache, error, and
  cancellation events. The session also retains a bounded event history for
  `vgi_logs()` and `vgi_log_stats()`.
- `VgiLocalityHook` exposes planned split locations to an embedding scheduler.
  The adapter does not invent DataFusion hash partitioning from VGI transforms.
- `VgiSessionOptions` configures cache bounds, event history, and an optional
  RPC timeout. HTTP, Unix, and TCP clients honor that timeout; `None` leaves
  long-lived calls unbounded. Subprocess pipe deadlines remain unavailable.
- Start-only VGI splits now declare `Boundedness::Unbounded` instead of being
  rejected or misrepresented as bounded.

## SQL/runtime capability map

| Capability | Status | Notes |
|---|:---:|---|
| ATTACH / DETACH | Supported | DuckDB-style syntax, typed worker options, OAuth, cache veto, and companion discovery |
| Catalogs, schemas, tables, views | Supported | Views are registered after dependencies and qualified to their alias |
| Native format scan branches | Supported | CSV, Parquet, JSON/NDJSON, and Arrow arms use registered DataFusion formats; custom formats require a host registration |
| Scalar and SQL macro functions | Supported | Async scalar UDFs plus scalar/table macro expansion with typed defaults and named arguments |
| Typed session settings | Supported | DataFusion `ConfigExtension`, `SET`/`RESET`, Arrow scalar and Struct encoding, metadata view, cache isolation |
| Aggregate/window-frame use | Partial | ConstParams, retract, and advertised sliding-window callbacks work; DataFusion lacks EXCLUDE and aggregate-local ORDER BY window forms |
| Table and buffered functions | Supported | Table input is currently constrained to one column by scalar-subquery planning |
| Global functions | Supported | Prefix/collision rules and DETACH cleanup included |
| Projection, static filters, LIMIT | Supported | Direct functions preserve exactness; lazy catalog tables recheck filters |
| Split planning | Supported | Parallel partitions, ordering properties, plan cache, unbounded metadata |
| Session result cache | Partial | Producer/split caching with revalidation and single-flight, streaming per-batch exchange caching, stable scalar per-value caching, and native scan metrics; no buffered/1:N exchange cache, disk, or SWR |
| Logs and diagnostics | Supported | SQL tables/scalars plus an embedder event sink |
| Worker-requested secrets | Supported | Host resolver API; no SQL secret store |
| Locality | Partial | Host callback exists; DataFusion CLI has no distributed scheduler |
| Correlated LATERAL table calls | Not wired | DataFusion binds table functions before an outer row is available |
| Dynamic join filters | Partial | Single-column `IN` and same-column equality-OR sets use VGI v2 side IPC at init, with exact scalar types preserved; later constant/range generations use `vgi_pushdown_filters` over subprocess, Unix/TCP, and plain or authenticated HTTP continuations. Large hash/Bloom lookups and multi-column struct expressions remain local |
| ORDER BY / TABLESAMPLE hints | Not wired | Current provider scan callback does not carry these VGI hints |
| Table time travel | Supported | Fully-qualified VGI tables accept literal `AT (VERSION => …)` and `AT (TIMESTAMP => …)`; historical schemas and cache identities are isolated |
| Catalog metadata | Supported | `SHOW TABLES`, `SHOW COLUMNS`, `SHOW FUNCTIONS`, and DataFusion `information_schema` expose VGI tables, views, columns, schemata, and routines without eager scan-function binds |
| Table constraints | Partial | Primary-key and unique metadata feed `TableProvider::constraints()`; DataFusion has no check, foreign-key, standalone NOT NULL, or `duckdb_constraints()` metadata surface |
| Transactions / snapshot pinning | Not wired | Requires SQL lifecycle interception and catalog-scoped transaction semantics |
| DML, DDL, and custom COPY | Not wired | Requires vgi-rust RPC wrappers and DataFusion sink/provider integration |

## Existing DataFusion APIs worth using next

1. **Cache breadth with matching semantics.** Add buffered whole-input and
   correlated 1:N tiers only where a complete deterministic key and
   cancellation boundary can be proved. Add conditional revalidation and
   single-flight to the exchange tiers; keep disk persistence optional.
2. **Unbounded execution hardening.** Gate resume on worker advertisement and
   add checkpoint/reconnect, cancellation, backpressure, and soak coverage.
3. **Catalog and scalar breadth.** Broaden view translation and continue
   signature coverage for nested, unsigned, and null-coercion forms.
4. **Dynamic-filter breadth.** Add a VGI representation for DataFusion's large
   hash/Bloom lookup and multi-column struct membership expressions, and decide
   whether completed filters may safely refine split enumeration before init.
5. **Metadata and documentation surfaces.** Preserve worker docs, examples,
   tags, settings requirements, and late-materialization declarations for host
   inspection before building custom operators around them.

## Deliberately deferred boundaries

Correlated LATERAL table functions, multi-column streaming table inputs,
late-materialization rewrites, VGI ordering hints, and sampling do not map
cleanly onto the current `TableProvider` callback. They should remain explicit
gaps unless DataFusion exposes a suitable API or the project chooses to own a
custom logical/physical extension. That keeps the integration focused on
wiring VGI into DataFusion rather than developing a parallel query engine.

## Production verification gate

The 327-file shared VGI SQLLogicTest corpus now completes against both HTTP and
an explicitly managed Unix-socket worker. Unix executes 3,273 records and HTTP
3,271; both produce 2,122 exact, 106 rendering-equivalent, and 187 genuinely
different results. The only transport delta is two additional HTTP timeouts in
`table_in_out/parallel_fanout.test`; every completed result classification is
otherwise identical.
The remaining failures are tracked capability gaps, primarily DuckDB-only SQL
and diagnostics, correlated table calls, wide table input, writes, and secret
host configuration. Publishing, stalled subprocess-RPC cancellation, and
long-lived unbounded-stream tests remain release gates.

The focused 10-file multi-branch slice additionally executes 75/75 positive
records with all 53 comparable results exact over subprocess, Unix sockets, and
loopback HTTP. This includes native CSV/Parquet arms, typed format options,
schema reconciliation, branch filtering, and split redemption.

Post-baseline focused verification also covers concurrent producer and split
cache fills, immediate-stale revalidation, `no_store` fallback, catalog-source
provider resolution, callback-only window median, overload incompatibility,
and multi-scope secret binds. The reviewed aggregate window slice now executes
15/15 applicable records with all 14 results agreeing.

The current release-mode adapter suite passes 194/194 tests plus two doctests.
The settings slice executes 42/42 records with all 14 results exact, and the
required-filter slice executes 45/45 applicable records with all 25 results
exact, over both Unix and HTTP. A 13-file
aggregate/macro/filter/cache promotion slice is identical over Unix and HTTP:
both execute 145/173 records and all 113 comparable queries agree. Aggregate
executes 42/42; macro/catalog executes 24/24; cache executes 30/38.
