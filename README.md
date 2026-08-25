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
| Scalar function | `AsyncScalarUDFImpl` | ✅ |
| Projection & LIMIT pushdown | `scan(projection, limit)` | ✅ |
| Filter pushdown | `supports_filters_pushdown` | ✅, locally rechecked |
| Split planning | physical scan partitions | ✅ |
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
`implementation_version`. Launcher options are rejected outside a `launch:`
LOCATION. Cache, secret, companion, and global-function options fail explicitly
until those integrations exist.

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

**Everything blocks.** `vgi-client` is synchronous, like the Python and Java VGI
clients, so every call runs inside `spawn_blocking`. `VgiConnection` is a
*factory* rather than a client for that reason: a connection cannot be shared
across partitions, and VGI parallelism is defined as N independent connections
anyway.

## Known gaps

- **Streaming positions.** A split that names a start position without an end
  is unbounded, while this provider declares bounded DataFusion tasks. Such a
  plan is refused rather than allowed to hang a blocking operator.
- **Partitioning and locality.** Partition bounds and column statistics feed
  DataFusion statistics, and within-split ordering is advertised when a physical
  partition contains exactly one split. VGI partition transforms are not mapped
  blindly to DataFusion hash partitioning because their hash and partition-number
  semantics differ. Location hints and cache age are retained in the scan plan,
  but local DataFusion has no scheduler-affinity or result-plan cache hook for them.
- **Dynamic filters and join keys.** DataFusion has no adapter bridge from its
  runtime join filters to VGI continuation ticks yet. Ordinary static filters
  are pushed and conservatively re-applied locally.
- **Correlated LATERAL table functions.** DataFusion binds a table function
  while planning, before an outer row exists. A call such as
  `LATERAL m.main.forecast_hourly(g.latitude, g.longitude)` is therefore not yet
  representable; run the geocoder first and pass its coordinates as constants
  in a second statement.

## License

See `LICENSE` at the repository root.
