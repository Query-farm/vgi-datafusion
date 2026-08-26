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
| Scalar function | `AsyncScalarUDFImpl` | ✅ |
| Aggregate and sliding window-frame use | `AggregateUDFImpl` with retract | ◐ |
| Projection & LIMIT pushdown | `scan(projection, limit)` | ✅ |
| Filter pushdown | `supports_filters_pushdown` | ✅, exact for directly bound functions |
| Dynamic filters and join keys | physical filter pushdown + continuation metadata | ◐ |
| Split planning | physical scan partitions + plan cache | ✅ |
| Worker-opted result cache | bounded session memory + revalidation | ◐ |
| Worker secrets | host `VgiSecretResolver` | ✅ |
| Structured diagnostics | SQL functions + host event sink | ✅ |
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

These are DataFusion's metadata surfaces; DuckDB-specific names such as
`duckdb_tables()` are not aliases. VGI primary-key and unique constraints feed
DataFusion's native optimizer constraint API, while check, foreign-key, and
standalone NOT NULL metadata have no matching DataFusion constraint type.

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
`launcher_idle_timeout`, `launcher_state_dir`, `data_version_spec`, and
`implementation_version`, `cache`, and `attach_companions`. Launcher options are
rejected outside a `launch:` LOCATION. Worker-declared global functions are
registered automatically. Raw `secrets` and `attach_companion_secrets` options
fail explicitly; embedders instead install a scoped `VgiSecretResolver` on the
session's `VgiRuntime`.

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

For an embedded application, put a configured `VgiRuntime` into the
`SessionConfig` extensions before constructing the `SessionContext`. The same
runtime can carry a secret resolver, non-blocking event sink, and split-locality
hook. If no extension is supplied, the SQL adapter creates one shared runtime
for all VGI attachments in that session.

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
connections. DataFusion cancellation still propagates to active scans, and an
embedder may configure an RPC timeout; subprocess pipe I/O cannot yet enforce
that timeout.

## Known gaps

- **Streaming positions.** A split that names a start position without an end
  now declares `Boundedness::Unbounded`. Checkpoint/resume policy, reconnect,
  backpressure, and long-running soak coverage still need production hardening.
- **Partitioning and locality.** Partition bounds and column statistics feed
  DataFusion statistics, and within-split ordering is advertised when a physical
  partition contains exactly one split. VGI partition transforms are not mapped
  blindly to DataFusion hash partitioning because their hash and partition-number
  semantics differ. A host locality hook receives worker location hints, but the
  standalone DataFusion CLI has no distributed scheduler-affinity implementation.
- **Cache breadth.** Complete, worker-opted producer and split results have a
  safe bounded memory tier with conditional revalidation. Exchange/per-value
  results, disk persistence, stale-while-revalidate, compression, and
  cross-query single-flight are not implemented.
- **Dynamic filters and join keys.** DataFusion 55 hash-join filters are linked
  to VGI scans. Completed single-column `IN` sets use `join_keys` side IPC at
  init (`vgi_join_keys_version=2`), while later range/constant refinements ride
  continuations as standard-base64 `vgi_pushdown_filters` metadata. Subprocess,
  Unix/TCP byte streams, and plain or authenticated HTTP all preserve those
  between-tick updates. The join remains DataFusion's correctness boundary, so
  older workers may ignore the hint safely, and dynamically filtered scans
  bypass the result cache.
  DataFusion's hash-table lookup expression for very large joins and
  multi-column struct membership are not serializable to VGI yet; dynamic
  filters also prune within already-planned splits rather than changing the
  split set.
- **Correlated LATERAL table functions.** DataFusion binds a table function
  while planning, before an outer row exists. A call such as
  `LATERAL m.main.forecast_hourly(g.latitude, g.longitude)` is therefore not yet
  representable; run the geocoder first and pass its coordinates as constants
  in a second statement.
- **Transactions, writes, and planner-only hints.** VGI transactions, mutations,
  custom COPY formats, order hints, and sampling do not yet have an end-to-end
  adapter path. Table-level `AT (VERSION|TIMESTAMP => literal)` time travel is
  supported independently. See `docs/implementation-inventory.md` for the API
  boundaries and recommended sequence.

## SQLLogicTest compatibility

The shared VGI corpus is tracked by capability, group, and individual file.
See [`docs/corpus-compatibility.md`](docs/corpus-compatibility.md) for the
completion rules, current EC2 baseline, adaptation policy, prioritized gaps,
and regression command. [`corpus/compatibility.json`](corpus/compatibility.json)
is the machine-readable status manifest; the runner rejects newly added corpus
groups until they are assigned and reviewed.

## License

See `LICENSE` at the repository root.
