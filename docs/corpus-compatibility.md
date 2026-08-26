# VGI SQLLogicTest compatibility program

The target is every applicable test under `../vgi/test/sql/integration`. The
upstream files remain unchanged. DuckDB syntax may be adapted only when
DataFusion has equivalent semantics; an adaptation must never conceal a missing
VGI protocol feature or a wrong result.

The machine-readable source of truth is
[`corpus/compatibility.json`](../corpus/compatibility.json). It assigns every
corpus group to exactly one capability area and records that area's declared
status, remaining work, and SQL policy. The corpus runner refuses to start when
a newly added upstream group is not assigned, so corpus growth cannot silently
fall outside the inventory.

## Status and completion rules

An area is `complete` only when:

1. Every applicable positive record executes.
2. Every comparable result is exact or differs only by a reviewed rendering
   convention.
3. Every skipped, not-applicable, and adapted record has been reviewed.
4. Transport-sensitive behavior passes through both Unix and HTTP.

`partial` means useful paths pass but applicable work remains. `not_started`
means there is no integration yet. `blocked` requires an external fixture such
as credentials, GitHub, or a container. `out_of_scope` is reserved for an
assertion about DuckDB or the DuckDB VGI extension itself, not as a generic way
to suppress an inconvenient failure.

## Reproducible reports

Use the optimized runner for a full corpus pass:

```bash
VGI_TEST_WORKER=../vgi-rust/target/release/vgi-example-worker \
  cargo run --release --bin corpus -- \
  --jobs 4 \
  --json corpus/baselines/subprocess.json \
  ../vgi/test/sql/integration
```

During implementation, pass only the affected files or a group directory. For
example, the complete function-metadata slice runs in a few seconds:

```bash
target/release/corpus --jobs 4 \
  --json target/corpus-focused.json \
  --compare-selected corpus/baselines/subprocess.json \
  ../vgi/test/sql/integration/catalog/function_arguments.test \
  ../vgi/test/sql/integration/scalar/function_registration.test \
  ../vgi/test/sql/integration/table/function_registration.test
```

`--compare-selected` checks only the files included in the current invocation
against their entries in the full committed baseline. Use ordinary `--compare`
for promotion runs: it additionally rejects any baseline file missing from the
new full report.

## DataFusion SQL overlays

The upstream `.test` file is the only copy of record order and expected rows.
When DuckDB SQL has equivalent DataFusion semantics but cannot run unchanged,
put a sparse sidecar at:

```text
corpus/overlays/<upstream-relative-path>.datafusion.json
```

For example, an adaptation for `scalar/example.test` lives at
`corpus/overlays/scalar/example.test.datafusion.json`. Each entry identifies a
1-based record, includes the exact original SQL as a drift guard, and records a
reviewed replacement or an `out_of_scope`/`blocked` classification with its
reason:

```json
{
  "schema_version": 1,
  "records": [
    {
      "record": 3,
      "original_sql": "SELECT duckdb_specific_syntax();",
      "replacement_sql": "SELECT datafusion_equivalent_syntax();",
      "kind": "equivalent_sql",
      "reason": "Both expressions have the same documented semantics."
    }
  ]
}
```

The runner aborts when an entry is malformed, duplicated, targets a missing
record, or its `original_sql` no longer matches upstream. Adaptations are listed
per file and counted in the JSON report. Use `--no-overlays` to measure the
generic harness adapters without reviewed per-record substitutions, or
`--overlays DIR` to review a proposed overlay tree.

Do not create full `.datafusion.test` copies for isolated syntax differences.
They duplicate expected output and unchanged SQL, drift independently, and can
be picked up by broad `*.test` discovery. In particular, do not put them beside
the originals under `../vgi/test/sql/integration`: both the VGI DuckDB runner
and this runner recursively discover `*.test`, so both variants would run. A
separate DataFusion-native fixture is appropriate only when nearly the entire
setup or assertion contract differs; keep it in this repository under `tests/`
or a dedicated, explicitly selected `corpus/native/` tree.

Files use independent DataFusion sessions, so `--jobs N` runs them concurrently
while merging results back in source order. Keep `N` bounded on shared hosts;
four workers is the EC2 default for corpus runs. A full run is reserved for
baseline promotion rather than the inner development loop.

Parallelize implementation by capability (adapter, overlays, classification,
and verification), but serialize release linking and tests that share the same
Cargo target directory. On the current EC2 host, increasing corpus concurrency
from four to eight workers did not materially improve the table slice because a
few large files dominate its wall time. Focused file lists are the fastest inner
loop.

For normal development, write a temporary report and compare it with the
committed baseline. The command exits non-zero if an existing file executes
fewer records, gains failures/timeouts/value mismatches, or loses agreeing value
checks:

```bash
VGI_TEST_WORKER=../vgi-rust/target/release/vgi-example-worker \
  target/release/corpus \
  --jobs 4 \
  --json target/corpus-current.json \
  --compare corpus/baselines/subprocess.json \
  ../vgi/test/sql/integration
```

The JSON report contains totals plus outcomes by capability area, upstream
group, and individual file. Improvements are accepted by reviewing the diff and
replacing the baseline deliberately. Unix and HTTP baselines should remain
separate because transport equivalence is itself part of completion.

## Current baseline — 2026-08-25

The canonical EC2 subprocess run of the 327-file normal corpus contains 4,114
measured positive records. This baseline includes the DataFusion-native
`typeof`, result-cache diagnostic aliases, `duckdb_logs()`,
`duckdb_functions()`, `vgi_function_arguments()`, and
`vgi_table_statistics()` compatibility views backed by adapter state and the
worker's retained discovery metadata. Catalog and bound-function column
statistics also feed DataFusion's existing pruning API.

| Metric | Initial | Current |
|---|---:|---:|
| Files run / skipped by missing environment | 278 / 49 | 278 / 49 |
| Records executed | 2,473 / 4,114 (60.1%) | 2,999 / 4,114 (72.9%) |
| Comparable results agreeing | 1,604 / 1,753 (91.5%) | 1,945 / 2,162 (90.0%) |
| Exact results | 1,567 | 1,882 |
| Rendering-equivalent results | 37 | 63 |
| Genuine value differences | 149 | 217 |
| DuckDB/extension configuration records reported separately | 607 | 607 |
| Timeouts | 0 | 0 |

The larger mismatch count is expected evidence, not a regression: 409 more
queries now reach result comparison instead of failing during planning. The
regression gate separately guarantees that no previously agreeing check was
lost. `cache/basic.test` is now fully executable with all 14 value checks
exact. The two catalog/function statistics files are also complete: all 91
positive records execute and all 89 query results agree (63 exact and 26
engine-plan rendering equivalents).

The deferred writable corpus under `../vgi/test_deferred/writable` is tracked
as a future input, not mixed into this baseline. Six `.test_slow` files are also
excluded from the normal 327-file count and should be separate soak/performance
gates.

Post-baseline focused verification now completes the macro slice. Scalar and
table macro definitions are parsed once, typed defaults and positional/named
arguments are bound locally, nested scalar macros compose, and recursive macro
definitions fail deterministically. Without overlays, 22/24 records execute and
all 19 comparable results are exact. Two reviewed overlays account only for
DataFusion's `range.value` versus DuckDB's `range.range` column spelling; with
them, `macro/macros.test` and `catalog/function_arguments_macros.test` execute
24/24 records with all 21 query results exact.

Function-backed multi-branch catalog scans are also complete in the focused
slice. The client prefers `catalog_table_scan_branches_get`, narrowly caches the
legacy fallback, and validates branch shape and writability. DataFusion binds
each function arm, reconciles columns by name, enforces branch filters, and
unions the plans through its existing physical operators. Across the seven
catalog/split files that do not require external branch fixtures, 47/47 positive
records execute and all 34 comparable results are exact. One reviewed overlay
rephrases DuckDB's unsupported `DESCRIBE`-as-derived-table syntax while querying
the same diagnostic column and type.

Native format branches now complete the wider 10-file multi-branch slice.
Worker-declared CSV, Parquet, JSON/NDJSON, and Arrow locations use DataFusion's
registered file-format factories; typed options, exact CSV null markers, schema
reconciliation, and eligible filter pushdown are retained. Three sparse
overlays translate only DuckDB's `COPY ... (FORMAT ...)` fixture syntax and its
`range.range` column spelling. Subprocess, Unix, and loopback HTTP runs each
execute 75/75 records, with all 53 comparable query results exact.

The focused aggregate window slice now executes all 15 applicable records and
all 14 results agree after routing advertised sliding accumulators through the
VGI window callback. Four SQL forms unsupported by DataFusion 55—three frame
`EXCLUDE` variants and aggregate-local `ORDER BY` in a window—are retained as
reviewed `out_of_scope` records rather than adapter failures.

The 2026-08-26 promotion slice covers 13 aggregate, macro/catalog, typed-filter,
and cache files over both Unix sockets and HTTP. Both transports execute
145/173 measured records, and all 113 comparable queries agree (109 exact,
four rendering-only differences). Aggregate is 42/42, macro/catalog is 24/24,
and the cache slice is 30/38. The remaining 28 records are classified
DataFusion SQL/type or table-expression boundaries; neither transport has a
unique failure.

## What remains

The following order maximizes useful coverage while keeping new DataFusion
engine work to a minimum:

1. **Corpus adaptation and classification.** DuckDB/extension-surface failures
   fell from 678 to 70; 118 parser failures remain. Continue with reviewable
   equivalents for metadata queries, struct extraction, and harmless dialect
   differences. Keep extension settings and DuckDB storage internals
   `out_of_scope` where DataFusion has no semantic equivalent.
2. **Advertised function publication and binding.** The largest error shape is
   now 138 instances of `table function ... not found`, down from 655. Resolve
   the remainder by capability area, prioritizing functions that map to an
   existing DataFusion table-provider or UDTF API.
3. **Result cache breadth.** Memory producer/split caching, conditional
   revalidation, per-key single-flight, streaming per-input-batch caching,
   stable scalar per-value caching, diagnostics, flush/reap, compatible event
   inspection, and native DataFusion scan metrics work. Buffered whole-input
   caching, correlated 1:N per-value entries, exchange revalidation/single-flight,
   disk persistence/spilling, compression, stale-while-revalidate, and
   worker-log forwarding remain. Cache execution improved from 297/710 to
   533/710 records before this focused promotion.
4. **Table-in/out engine boundary.** Correlated LATERAL calls account for 53
   explicit failures. Wide table subqueries account for at least another 70.
   The adapter accepts the single-column expression shape DataFusion exposes
   naturally, plus literal one-row exchange. These two gaps are tracked but are
   not candidates for new DataFusion engine work in this project.
5. **Catalog objects.** The focused scalar/table macro slice and the complete
   function/native-format multi-branch slice now pass. Catalog-table source
   arms now have a real companion-worker fixture and resolve existing DataFusion
   providers with cycle/ambiguity checks. Catalog-owned macro/view SQL qualifies
   worker objects inside the attachment. Custom formats without a registered
   DataFusion factory, broader views, and database metadata adaptations remain.
   Function inventory,
   overloads, argument docs/constraints, tags, categories, and global
   nominations use retained worker metadata. The committed full baseline still
   records 64/90 for catalog, 4/19 for macros, and 5/14 for views until the next
   promotion run.
6. **Secrets and authenticated fixtures.** Deterministic read-only consumer
   tests cover table, lazy table, scalar, aggregate, and streaming binds; two
   same-type scopes; duplicate-name rejection; Bearer/OAuth identity isolation;
   and cache bypass for secret-derived results. External OAuth
   discovery/device/refresh lifecycle and CLI secret mapping remain in the
   blocked/host-policy lane.
7. **Aggregate correctness.** Zero-argument, named-only, ANY, and variadic
   signatures now reach authoritative worker validation, including grouped and
   window execution over empty/nonempty inputs. Every comparable result in the
   focused ordinary aggregate slice agrees. Nested tensor/struct and correlated
   inputs remain, while window `EXCLUDE` and aggregate-local `ORDER BY` are
   classified DataFusion boundaries.
8. **COPY and mutations.** Some shared COPY records can be adapted to
   DataFusion-native host file reads/writes, but the adapter does not implement
   VGI COPY RPCs. Writable tables (`INSERT`, `UPDATE`, `DELETE`, `RETURNING`)
   likewise remain outside this read/query-focused effort.
9. **Remaining optimizer hints.** Catalog and bound-function min/max pruning
   now use DataFusion's existing pruning builder. Static SQL membership and the
   initial DataFusion runtime membership snapshot use VGI v2 `join_keys`, with
   the side IPC included in split planning, scan initialization, and cache
   identity. Exact supported Arrow scalar types and same-column equality `OR`
   membership are also covered. ORDER BY, TABLESAMPLE, late materialization,
   continued refinement after an init-time membership set, multi-column
   membership, and very large/Bloom join-key state remain explicit partial
   features.

Focused regression contracts for projection pushdown, narrow-bind mismatch,
and unary error propagation are already marked `complete`; they establish the
first no-regression floor while the broader areas move from `partial` toward
`complete`.
