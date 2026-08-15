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
| Filter pushdown | `supports_filters_pushdown` | not wired |
| Table-in-out / buffered | — | **no host construct** |

DataFusion resolves table-function arguments against an empty schema, so it
cannot express a table function that takes rows. Exchange-mode VGI functions
therefore have no direct SQL surface; the routes around it (an async scalar UDF
returning `List<Struct>` plus `unnest`, or a UDTF taking a table *name*) are
described in the feasibility study.

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

- **Parallel scan partitions.** The plan reports the worker's `max_workers` as
  its partition count, but only partition 0 currently reads; the rest return
  empty. Correct, not yet parallel — the missing piece is sharing the primary's
  `execution_id` across partitions, which needs a rendezvous the adapter does
  not have. `vgi-client` itself does this correctly and has a test for it.
- **Filter pushdown** is not wired to `supports_filters_pushdown`, so filters
  are applied above the scan. VGI can take them; the translation from
  DataFusion `Expr` to the wire encoding is about 250 lines and is not written.

## License

See `LICENSE` at the repository root.
