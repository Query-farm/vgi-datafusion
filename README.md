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

**DataFusion is a path dependency, deliberately.** Released DataFusion 54.1.0
is built on arrow 58; `vgi-protocol` and the vgi-rust workspace are on arrow 59.
Those are different `ArrayRef` types and cannot meet. DataFusion's main branch
has moved to 59.2.0, so this builds against a checkout until a release carrying
it exists.

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
