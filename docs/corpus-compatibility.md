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

Files use independent DataFusion sessions, so `--jobs N` runs them concurrently
while merging results back in source order. Keep `N` bounded on shared hosts;
four workers is the EC2 default for corpus runs. A full run is reserved for
baseline promotion rather than the inner development loop.

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
   revalidation, diagnostics, flush/reap, and compatible event inspection work.
   Exchange/per-value caching, disk persistence/spilling, compression,
   single-flight, worker-log forwarding, and DataFusion-native EXPLAIN/metrics
   remain. Cache execution improved from 297/710 to 533/710 records.
4. **Table-in/out engine boundary.** Correlated LATERAL calls account for 53
   explicit failures. Wide table subqueries account for at least another 70.
   The adapter accepts the single-column expression shape DataFusion exposes
   naturally, plus literal one-row exchange. These two gaps are tracked but are
   not candidates for new DataFusion engine work in this project.
5. **Catalog objects.** Macros, broader views, and multi-branch catalogs need
   publication or a documented SQL adaptation. Function inventory, overloads,
   argument docs/constraints, tags, categories, and global nominations now use
   retained worker metadata. Current execution is 64/90 for catalog, 4/19 for
   macros, and 5/14 for views.
6. **Secrets and authenticated fixtures.** Twenty-four failures explicitly lack
   a `VgiSecretResolver`. Add deterministic corpus resolvers before judging the
   worker behavior; keep real OAuth and external-service tests in the blocked
   fixture lane.
7. **Aggregate correctness.** Aggregate execution is broad, but the baseline
   exposes a real window result mismatch (expected moving values, received
   NULLs) and remaining zero-argument/ANY/varargs signatures. These are bugs or
   adapter gaps, not dialect exclusions.
8. **COPY and mutations.** COPY FROM/TO partially executes after syntax
   translation. Writable tables (`INSERT`, `UPDATE`, `DELETE`, `RETURNING`) need
   mutation RPC wrappers and transaction/cache invalidation semantics and are
   currently `not_started`.
9. **Remaining optimizer hints.** Catalog and bound-function min/max pruning
   now use DataFusion's existing pruning builder. ORDER BY, TABLESAMPLE, late
   materialization, large hash-lookup filters, and multi-column dynamic
   membership remain explicit partial features.

Focused regression contracts for projection pushdown, narrow-bind mismatch,
and unary error propagation are already marked `complete`; they establish the
first no-regression floor while the broader areas move from `partial` toward
`complete`.
