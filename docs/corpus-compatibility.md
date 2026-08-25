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
  --json corpus/baselines/subprocess.json \
  ../vgi/test/sql/integration
```

For normal development, write a temporary report and compare it with the
committed baseline. The command exits non-zero if an existing file executes
fewer records, gains failures/timeouts/value mismatches, or loses agreeing value
checks:

```bash
VGI_TEST_WORKER=../vgi-rust/target/release/vgi-example-worker \
  target/release/corpus \
  --json target/corpus-current.json \
  --compare corpus/baselines/subprocess.json \
  ../vgi/test/sql/integration
```

The JSON report contains totals plus outcomes by capability area, upstream
group, and individual file. Improvements are accepted by reviewing the diff and
replacing the baseline deliberately. Unix and HTTP baselines should remain
separate because transport equivalence is itself part of completion.

## Initial baseline — 2026-08-25

The canonical EC2 subprocess run of the 327-file normal corpus contains 4,114
measured positive records:

| Metric | Result |
|---|---:|
| Files run / skipped by missing environment | 278 / 49 |
| Records executed | 2,473 / 4,114 (60.1%) |
| Comparable results agreeing | 1,604 / 1,753 (91.5%) |
| Exact results | 1,567 |
| Rendering-equivalent results | 37 |
| Genuine value differences | 149 |
| DuckDB/extension configuration records reported separately | 607 |
| Timeouts | 0 |

The deferred writable corpus under `../vgi/test_deferred/writable` is tracked
as a future input, not mixed into this baseline. Six `.test_slow` files are also
excluded from the normal 327-file count and should be separate soak/performance
gates.

## What remains

The following order maximizes useful coverage while keeping new DataFusion
engine work to a minimum:

1. **Corpus adaptation and classification.** There are 678 DuckDB-only failures
   and 118 parser failures. Add explicit, reviewable equivalents for `typeof`,
   metadata queries, struct extraction, and harmless dialect differences. Mark
   extension settings, `duckdb_logs`, and DuckDB storage diagnostics
   `out_of_scope` when no DataFusion semantic equivalent exists.
2. **Advertised function publication and binding.** The largest error shape is
   662 instances of `table function ... not found`. Resolve these by capability
   area: many cache, catalog, COPY, and table-in/out functions are advertised by
   the worker but are not currently published in a form DataFusion can call.
3. **Table-in/out breadth.** Correlated LATERAL calls account for at least 53
   explicit failures. Wide table subqueries account for at least another 70;
   the adapter currently accepts only the single-column expression shape
   DataFusion exposes naturally. Literal one-row exchange is already wired.
4. **Result cache breadth.** Memory producer/split caching and conditional
   revalidation work. Exchange/per-value caching, disk persistence/spilling,
   compression, single-flight, and DataFusion-native EXPLAIN/metrics remain.
   Cache is the largest single incomplete group: 297/710 records execute.
5. **Catalog objects.** Macros, broader views, and multi-branch catalogs need
   publication or a documented SQL adaptation. Current execution is 26/90 for
   catalog, 2/19 for macros, and 2/14 for views.
6. **Secrets and authenticated fixtures.** Twenty-two failures explicitly lack
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
9. **Remaining optimizer hints.** ORDER BY, TABLESAMPLE, late materialization,
   large hash-lookup filters, and multi-column dynamic membership remain
   explicit partial features.

Focused regression contracts for projection pushdown, narrow-bind mismatch,
and unary error propagation are already marked `complete`; they establish the
first no-regression floor while the broader areas move from `partial` toward
`complete`.
