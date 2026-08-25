# VGI × DataFusion implementation inventory

Reviewed against the local DataFusion 55, `vgi-rust`, and `vgi-rpc` checkouts on
25 August 2026. This inventory favors VGI capabilities that map onto existing
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
  function, arguments, attach options, projection, static filters, and limit.
  Split results with advertised ordering are not cached because flattened
  replay could violate DataFusion plan properties.
- Large stale entries support ETag/Last-Modified conditional revalidation and
  worker-authorized stale-if-error fallback.
- Catalog-scoped split plans are reused only for their advertised lifetime.
  Version-scoped and transaction-scoped plans are not reused.
- SQL diagnostics expose cache statistics, entries, flushing, reaping, and plan
  cache statistics.

This is intentionally narrower than the DuckDB cache. Exchange/per-value
results, persistent disk storage, stale-while-revalidate, compression, and
cross-query single-flight remain deferred.

### Catalog and function fidelity

- Worker views are qualified to their attachment and registered as DataFusion
  `ViewTable`s after catalog and function discovery.
- Worker-nominated global scalar, aggregate, table, and buffered functions use
  the declared VGI prefix and are removed on `DETACH` or replacement.
- VGI-only companion catalogs attach recursively with depth and cycle guards;
  required unsupported companions fail rather than silently disappearing.
- Function stability maps to DataFusion volatility. Directly bound table
  functions map exact worker filter application to
  `TableProviderFilterPushDown::Exact`; lazy catalog tables recheck locally.
- Aggregate functions preserve bind-time ConstParams and support DataFusion
  sliding window frames through retractable aggregate state. Dedicated VGI window RPC execution is not
  currently selectable through DataFusion's same-name aggregate/window SQL
  resolution.

### Secrets, runtime hooks, and observability

- An embedder can install an asynchronous `VgiSecretResolver`. Producer,
  table, scalar, and input binds resolve typed, scoped secret requests and
  retry the bind once. Aggregate declarations resolve their static secrets.
- Secret values stay out of SQL, cache keys, events, and diagnostic tables.
  SQL `CREATE SECRET` and raw secret values in `ATTACH` remain unsupported.
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
| Scalar and SQL macro functions | Supported | Async scalar UDFs and scalar macro expansion |
| Aggregate/window-frame use | Partial | ConstParams and retract work; dedicated VGI window RPC deferred |
| Table and buffered functions | Supported | Table input is currently constrained to one column by scalar-subquery planning |
| Global functions | Supported | Prefix/collision rules and DETACH cleanup included |
| Projection, static filters, LIMIT | Supported | Direct functions preserve exactness; lazy catalog tables recheck filters |
| Split planning | Supported | Parallel partitions, ordering properties, plan cache, unbounded metadata |
| Session result cache | Partial | Safe memory tier and revalidation; no exchange/disk/SWR/single-flight |
| Logs and diagnostics | Supported | SQL tables/scalars plus an embedder event sink |
| Worker-requested secrets | Supported | Host resolver API; no SQL secret store |
| Locality | Partial | Host callback exists; DataFusion CLI has no distributed scheduler |
| Correlated LATERAL table calls | Not wired | DataFusion binds table functions before an outer row is available |
| Dynamic join filters | Partial | Single-column join-key sets use VGI v2 side IPC at init; later constant/range generations use `vgi_pushdown_filters` over byte-stream continuations. HTTP currently loses that metadata before the worker; large hash-lookup and multi-column struct expressions remain local |
| ORDER BY / TABLESAMPLE hints | Not wired | Current provider scan callback does not carry these VGI hints |
| Table time travel | Supported | Fully-qualified VGI tables accept literal `AT (VERSION => …)` and `AT (TIMESTAMP => …)`; historical schemas and cache identities are isolated |
| Catalog metadata | Supported | `SHOW TABLES`, `SHOW COLUMNS`, `SHOW FUNCTIONS`, and DataFusion `information_schema` expose VGI tables, views, columns, schemata, and routines without eager scan-function binds |
| Table constraints | Partial | Primary-key and unique metadata feed `TableProvider::constraints()`; DataFusion has no check, foreign-key, standalone NOT NULL, or `duckdb_constraints()` metadata surface |
| Transactions / snapshot pinning | Not wired | Requires SQL lifecycle interception and catalog-scoped transaction semantics |
| DML, DDL, and custom COPY | Not wired | Requires vgi-rust RPC wrappers and DataFusion sink/provider integration |

## Existing DataFusion APIs worth using next

1. **Cache breadth with matching semantics.** Add per-key single-flight first,
   then exchange or single-value entries only where a complete deterministic
   key and cancellation boundary can be proved. Keep disk persistence optional.
2. **Unbounded execution hardening.** Gate resume on worker advertisement and
   add checkpoint/reconnect, cancellation, backpressure, and soak coverage.
3. **Transactions and writes.** DataFusion 55 exposes mutation provider hooks,
   while the SQL adapter can intercept transaction statements. First add the
   missing mutation RPC wrappers to `vgi-client`, then define catalog-scoped
   transaction and cache invalidation semantics.
4. **Dynamic-filter breadth.** Add a VGI representation for DataFusion's large
   hash-table lookup and multi-column struct membership expressions, and decide
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
Unix-socket launcher transports. HTTP executed 2,429 records and 91.3% of
comparable query results agreed; the latest Unix run executed 2,446 and 91.2%
agreed after enabling the same metadata surface as the CLI.
The remaining failures are tracked capability gaps, primarily DuckDB-only SQL
and diagnostics, correlated table calls, wide table input, writes, and secret
host configuration. Publishing, stalled subprocess-RPC cancellation, and
long-lived unbounded-stream tests remain release gates.
