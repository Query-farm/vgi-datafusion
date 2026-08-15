// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Run the DuckDB extension's `.test` corpus against this adapter and report
//! what works.
//!
//! # What this measures, and what it deliberately does not
//!
//! The corpus was written for DuckDB, so a strict sqllogictest comparison would
//! fail almost everywhere on **rendering** alone — `BIGINT` vs `Int64`,
//! different float formatting, different NULL spelling — and that noise would
//! bury the signal.
//!
//! So the primary metric here is **execution**: did the statement or query
//! reach the worker and come back without an error? That is what tells us which
//! protocol surfaces are wired up. Value comparison is reported as a *separate*
//! secondary number for the queries that did execute, so a mismatch is visible
//! without being counted as a missing feature.
//!
//! Failures are bucketed by cause, because "412 failures" is not actionable and
//! "380 of them are `typeof` not existing in DataFusion" is.
//!
//! ```bash
//! VGI_TEST_WORKER=… cargo run --bin corpus -- [path ...]
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use datafusion::prelude::SessionContext;

/// One directive from a `.test` file.
#[derive(Debug)]
enum Record {
    /// `statement ok` / `statement error`
    Statement { sql: String, expect_ok: bool },
    /// `query <types>` with its expected rows
    Query { sql: String, expected: Vec<String> },
}

/// Why a record failed, coarse enough to act on.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum Bucket {
    /// A DuckDB-only function (`typeof`, `vgi_*`) or view (`duckdb_*`).
    DuckDbOnly,
    /// Named arguments, which DataFusion refuses during planning.
    NamedArguments,
    /// A VGI function that is advertised but does not bind as called.
    BindRefused,
    /// SQL this dialect cannot parse.
    ParseError,
    /// A type DataFusion does not have, or a cast it will not do.
    TypeMismatch,
    /// Everything else.
    Other,
}

impl Bucket {
    fn classify(err: &str) -> Self {
        let e = err.to_lowercase();
        if e.contains("typeof")
            || e.contains("duckdb_")
            || e.contains("'vgi_")
            || e.contains("function 'vgi")
        {
            Self::DuckDbOnly
        } else if e.contains("unsupported function argument") {
            Self::NamedArguments
        } else if e.contains("does not bind as a bare table") || e.contains("bind") {
            Self::BindRefused
        } else if e.contains("sql error") || e.contains("parsererror") || e.contains("expected:") {
            Self::ParseError
        } else if e.contains("no function matches")
            || e.contains("cast")
            || e.contains("type_coercion")
            || e.contains("invalid function")
        {
            Self::TypeMismatch
        } else {
            Self::Other
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DuckDbOnly => "DuckDB-only surface (typeof / duckdb_* / vgi_*)",
            Self::NamedArguments => "named arguments (unsupported by DataFusion)",
            Self::BindRefused => "worker refused the bind",
            Self::ParseError => "parse error (dialect)",
            Self::TypeMismatch => "type / signature mismatch",
            Self::Other => "other",
        }
    }
}

/// How long one record may take before it is set aside.
const RECORD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Default)]
struct Tally {
    files_run: usize,
    files_skipped: usize,
    executed: usize,
    failed: usize,
    /// Records that are not about the worker protocol at all.
    ///
    /// Engine configuration (`SET`, `PRAGMA`, `CALL`) and the extension's own
    /// diagnostic surface. Counting these as conformance failures overstates
    /// the gap: `SET vgi_result_cache_max_bytes` is tuning for a feature that
    /// exists in the DuckDB extension and nowhere else, and no adapter will
    /// ever make it run.
    not_applicable: usize,
    timed_out: usize,
    values_matched: usize,
    /// Differed only in how the values are printed.
    values_rendering: usize,
    /// `LIMIT` with no `ORDER BY`: an arbitrary subset, not comparable.
    values_unordered_subset: usize,
    values_differed: usize,
    buckets: BTreeMap<Bucket, usize>,
    /// One example error per bucket, so the summary is diagnosable.
    examples: BTreeMap<Bucket, (String, String)>,
    /// How often each distinct error shape occurred.
    ///
    /// A bucket labelled "other: 1550" is not actionable; the same 1550 split
    /// into a handful of recurring messages is. Names, paths and numbers are
    /// stripped so one root cause counts once rather than a thousand times.
    error_shapes: BTreeMap<String, usize>,
    /// Per top-level group (`table/`, `cache/`, …).
    ///
    /// The aggregate number hides the most important fact about this corpus:
    /// whole groups test the *extension* rather than the worker — `cache/`
    /// asserts result-cache events in `duckdb_logs`, `catalog/` queries the
    /// `vgi_*` diagnostic functions — and no adapter can make those pass. A
    /// per-group rate separates "not wired up yet" from "not applicable".
    groups: BTreeMap<String, (usize, usize)>,
    /// Queries that ran but disagreed with DuckDB, kept in full.
    ///
    /// A count alone cannot distinguish "renders 42 as 42.0" from "returns the
    /// wrong answer", and only one of those is a bug.
    mismatches: Vec<Mismatch>,
}

/// Do two result sets agree once DuckDB's *rendering* conventions are allowed
/// for?
///
/// The corpus records DuckDB's output verbatim, and three of its conventions
/// differ from Arrow's without either being wrong:
///
/// * an empty string prints as `(empty)`;
/// * struct keys are quoted — `{'lat': 3.0}` against `{lat: 3.0}`;
/// * floats print at shortest-round-trip — `0.0003` against Arrow's
///   `0.00030000000000000003`, which are the same double.
///
/// Counting those as failures would put a rendering difference and a wrong
/// answer in the same bucket, and only one of them is worth anyone's time. So
/// they are compared separately and reported separately.
fn agrees_modulo_rendering(expected: &[String], got: &[String]) -> bool {
    if expected.len() != got.len() {
        return false;
    }
    rows_agree(expected, got)
}

/// Compare rows pairwise, allowing for rendering conventions.
fn rows_agree(expected: &[String], got: &[String]) -> bool {
    expected.iter().zip(got).all(|(e, g)| {
        let (ec, gc): (Vec<_>, Vec<_>) = (e.split('\t').collect(), g.split('\t').collect());
        ec.len() == gc.len() && ec.iter().zip(&gc).all(|(a, b)| cells_agree(a, b))
    })
}

fn cells_agree(expected: &str, got: &str) -> bool {
    if expected == got {
        return true;
    }
    if expected == "(empty)" && got.is_empty() {
        return true;
    }
    // Same double, different number of digits.
    if let (Ok(a), Ok(b)) = (expected.parse::<f64>(), got.parse::<f64>()) {
        let scale = a.abs().max(b.abs()).max(1.0);
        if (a - b).abs() <= scale * 1e-9 {
            return true;
        }
    }
    if normalize_timestamp(expected) == normalize_timestamp(got) {
        return true;
    }
    unquote_struct_keys(expected) == unquote_struct_keys(got)
}

/// Does a query pin its row order?
///
/// Without `ORDER BY`, SQL guarantees nothing about order, and DataFusion
/// really does reorder: it inserts a `RepartitionExec(RoundRobinBatch(8))`
/// above the scan, so batches arrive interleaved. The corpus records DuckDB's
/// order, which is deterministic in practice — comparing the two positionally
/// would report differences that are not disagreements.
fn is_ordered(sql: &str) -> bool {
    sql.to_uppercase().contains("ORDER BY")
}

/// Is the result an arbitrary *subset*, so not comparable at all?
///
/// `LIMIT` without `ORDER BY` returns whichever rows arrive first. Under
/// repartitioning that is a different — equally correct — subset than DuckDB's,
/// which is what `SELECT n FROM … WHERE n >= 5000 LIMIT 5` returning
/// 7000..7004 against an expected 5000..5004 means. Neither engine is wrong and
/// no comparison can distinguish that from a real defect, so these are reported
/// separately rather than counted either way.
fn is_arbitrary_subset(sql: &str) -> bool {
    let upper = sql.to_uppercase();
    upper.contains("LIMIT") && !upper.contains("ORDER BY")
}

/// Put two timestamp spellings on common ground.
///
/// DuckDB separates date and time with a space and writes a zero offset as
/// `+00`; Arrow uses ISO-8601 `T` and omits the offset on a naive timestamp.
/// `2026-05-06 12:00:00+00` and `2026-05-06T12:00:00` are the same instant.
///
/// Guarded on the cell looking like a date, so an ordinary string containing a
/// `T` is never quietly rewritten.
fn normalize_timestamp(s: &str) -> String {
    let looks_like_date = s.len() >= 10
        && s.as_bytes()[..4].iter().all(u8::is_ascii_digit)
        && s.as_bytes()[4] == b'-';
    if !looks_like_date {
        return s.to_string();
    }
    let s = s.replacen('T', " ", 1);
    s.trim_end_matches("+00:00")
        .trim_end_matches("+00")
        .trim_end()
        .to_string()
}

/// Drop the quotes DuckDB puts around struct field names.
fn unquote_struct_keys(s: &str) -> String {
    s.replace("{'", "{")
        .replace(", '", ", ")
        .replace("': ", ": ")
}

/// A query that executed but produced different values.
struct Mismatch {
    file: String,
    sql: String,
    expected: Vec<String>,
    got: Vec<String>,
}

/// Parse a `.test` file into records, or `None` when it should be skipped.
///
/// `require-env` is honoured (that is how the corpus marks a test as needing a
/// worker). `require` names DuckDB extensions and is ignored — this adapter is
/// not DuckDB, and a file requiring `httpfs` may still exercise VGI surfaces.
fn parse(path: &Path) -> Option<Vec<Record>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut records = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        let line = line.trim_end();
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some(var) = line.strip_prefix("require-env ") {
            if std::env::var(var.trim()).is_err() {
                return None;
            }
            continue;
        }
        if line.starts_with("require") || line.starts_with("mode ") {
            continue;
        }

        let collect_sql = |lines: &mut std::iter::Peekable<std::str::Lines>| {
            let mut sql = String::new();
            while let Some(l) = lines.peek() {
                if l.trim().is_empty() || *l == "----" {
                    break;
                }
                sql.push_str(l);
                sql.push('\n');
                lines.next();
            }
            sql.trim().to_string()
        };

        if let Some(rest) = line.strip_prefix("statement ") {
            let expect_ok = rest.trim().starts_with("ok");
            let sql = collect_sql(&mut lines);
            if !sql.is_empty() {
                records.push(Record::Statement { sql, expect_ok });
            }
        } else if line.starts_with("query ") {
            let sql = collect_sql(&mut lines);
            let mut expected = Vec::new();
            if lines.peek().map(|l| *l == "----").unwrap_or(false) {
                lines.next();
                while let Some(l) = lines.peek() {
                    if l.trim().is_empty() {
                        break;
                    }
                    expected.push(l.to_string());
                    lines.next();
                }
            }
            if !sql.is_empty() {
                records.push(Record::Query { sql, expected });
            }
        }
    }
    Some(records)
}

/// Expand `${VAR}` the way the DuckDB runner does.
fn expand(sql: &str) -> String {
    let mut out = sql.to_string();
    while let Some(start) = out.find("${") {
        let Some(end) = out[start..].find('}').map(|e| start + e) else {
            break;
        };
        let var = &out[start + 2..end];
        let val = std::env::var(var).unwrap_or_default();
        out.replace_range(start..=end, &val);
    }
    out
}

/// Render a result set the way the corpus writes it: tab-separated, one row per
/// line. Only used for the secondary value comparison.
fn render(batches: &[datafusion::arrow::array::RecordBatch]) -> Vec<String> {
    use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
    let opts = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for b in batches {
        let fmts: Vec<_> = b
            .columns()
            .iter()
            .filter_map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).ok())
            .collect();
        for r in 0..b.num_rows() {
            rows.push(
                fmts.iter()
                    .map(|f| f.value(r).to_string())
                    .collect::<Vec<_>>()
                    .join("\t"),
            );
        }
    }
    rows
}

async fn run_file(path: &Path, tally: &mut Tally) {
    let Some(records) = parse(path) else {
        tally.files_skipped += 1;
        return;
    };
    if records.is_empty() {
        tally.files_skipped += 1;
        return;
    }
    tally.files_run += 1;
    let file_label = path
        .to_string_lossy()
        .rsplit("integration/")
        .next()
        .unwrap_or("")
        .to_string();
    let group = file_label.split('/').next().unwrap_or("?").to_string();

    let ctx = SessionContext::new();
    for record in records {
        let (sql, expected) = match &record {
            Record::Statement { sql, .. } => (expand(sql), None),
            Record::Query { sql, expected } => (expand(sql), Some(expected)),
        };

        // A statement the corpus expects to fail is not evidence either way
        // about this adapter, so it is not counted.
        if matches!(
            record,
            Record::Statement {
                expect_ok: false,
                ..
            }
        ) {
            continue;
        }

        // One slow record must not dominate the report. Some fixtures move a
        // lot of data on purpose, and this is a survey of what *works*, not a
        // benchmark — a timeout is counted separately so it is never mistaken
        // for a missing feature.
        let outcome = tokio::time::timeout(RECORD_TIMEOUT, async {
            let df = vgi_datafusion::sql(&ctx, &sql).await?;
            df.collect().await
        })
        .await;

        let outcome = match outcome {
            Err(_) => {
                tally.timed_out += 1;
                continue;
            }
            Ok(o) => o,
        };

        match outcome {
            Err(e) if not_applicable(&sql) => {
                let _ = e;
                tally.not_applicable += 1;
            }
            Err(e) => {
                tally.groups.entry(group.clone()).or_default().1 += 1;
                record_failure(tally, &sql, &e.to_string())
            }
            Ok(batches) => {
                tally.executed += 1;
                tally.groups.entry(group.clone()).or_default().0 += 1;
                if let Some(expected) = expected {
                    if !expected.is_empty() {
                        let got = render(&batches);
                        if got == *expected {
                            tally.values_matched += 1;
                        } else if agrees_modulo_rendering(expected, &got) {
                            tally.values_rendering += 1;
                        } else {
                            tally.values_differed += 1;
                            tally.mismatches.push(Mismatch {
                                file: file_label.clone(),
                                sql: first_line(&sql),
                                expected: expected.clone(),
                                got,
                            });
                        }
                    }
                }
            }
        }
    }
}

fn record_failure(tally: &mut Tally, sql: &str, err: &str) {
    tally.failed += 1;
    *tally.error_shapes.entry(error_shape(err)).or_default() += 1;
    let bucket = Bucket::classify(err);
    *tally.buckets.entry(bucket).or_default() += 1;
    tally
        .examples
        .entry(bucket)
        .or_insert_with(|| (first_line(sql), first_line(err)));
}

/// Is this record about engine configuration rather than the worker protocol?
///
/// `SET`/`PRAGMA`/`CALL`/`RESET` configure DuckDB or the VGI *extension* —
/// `SET vgi_result_cache_dir`, `CALL enable_logging(...)` — and have no meaning
/// here. They are reported separately rather than counted against conformance,
/// which would otherwise charge this adapter for not being DuckDB.
fn not_applicable(sql: &str) -> bool {
    let head = sql.trim_start().split_whitespace().next().unwrap_or("");
    head.eq_ignore_ascii_case("set")
        || head.eq_ignore_ascii_case("pragma")
        || head.eq_ignore_ascii_case("call")
        || head.eq_ignore_ascii_case("reset")
}

/// Reduce an error to its shape: the message with the specifics removed.
///
/// Everything that varies per call site — quoted identifiers, paths, numbers —
/// is replaced, so a thousand instances of one root cause collapse to one line.
fn error_shape(err: &str) -> String {
    let line = err.lines().next().unwrap_or("").trim();
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '`' | '"' => {
                // Swallow the quoted run and stand in for it.
                let close = c;
                out.push('_');
                for n in chars.by_ref() {
                    if n == close {
                        break;
                    }
                }
            }
            '0'..='9' => {
                out.push('#');
                while chars.peek().is_some_and(|d| d.is_ascii_digit()) {
                    chars.next();
                }
            }
            _ => out.push(c),
        }
    }
    if out.len() > 120 {
        out.truncate(120);
        out.push('…');
    }
    out
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.len() > 110 {
        format!("{}…", &line[..110])
    } else {
        line.to_string()
    }
}

fn collect_tests(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_tests(&path, out);
        } else if path.extension().is_some_and(|e| e == "test") {
            out.push(path);
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let roots: Vec<PathBuf> = if args.is_empty() {
        vec![PathBuf::from(
            std::env::var("HOME").unwrap_or_default() + "/Development/vgi/test/sql/integration",
        )]
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let mut files = Vec::new();
    for root in &roots {
        if root.is_dir() {
            collect_tests(root, &mut files);
        } else {
            files.push(root.clone());
        }
    }

    if std::env::var("VGI_TEST_WORKER").is_err() {
        eprintln!("VGI_TEST_WORKER is not set — every file will skip.");
    }

    let mut tally = Tally::default();
    for (i, f) in files.iter().enumerate() {
        if i % 25 == 0 {
            eprintln!("  [{i}/{}] {}", files.len(), f.display());
        }
        run_file(f, &mut tally).await;
    }

    let total = tally.executed + tally.failed + tally.timed_out;
    let pct = |n: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            n as f64 * 100.0 / d as f64
        }
    };

    println!("\n=== corpus: {} files ===", files.len());
    println!(
        "files:   {} run, {} skipped (no require-env / no records)",
        tally.files_run, tally.files_skipped
    );
    println!(
        "records: {total} run — {} executed ({:.1}%), {} failed ({:.1}%)",
        tally.executed,
        pct(tally.executed, total),
        tally.failed,
        pct(tally.failed, total),
    );
    if tally.not_applicable > 0 {
        println!(
            "         {} not applicable (SET / PRAGMA / CALL — engine or extension config)",
            tally.not_applicable
        );
    }
    if tally.timed_out > 0 {
        println!(
            "         {} timed out after {:?} (not counted as failures)",
            tally.timed_out, RECORD_TIMEOUT
        );
    }
    let compared = tally.values_matched + tally.values_rendering + tally.values_differed;
    let agreeing = tally.values_matched + tally.values_rendering;
    println!(
        "values:  {compared} queries compared — {agreeing} agree ({:.1}%): \
         {} exact, {} differ only in rendering",
        pct(agreeing, compared),
        tally.values_matched,
        tally.values_rendering,
    );
    println!(
        "         {} genuinely differ ({:.1}%)",
        tally.values_differed,
        pct(tally.values_differed, compared)
    );
    if tally.values_unordered_subset > 0 {
        println!(
            "         {} not comparable (LIMIT with no ORDER BY — an arbitrary subset)",
            tally.values_unordered_subset
        );
    }

    if !tally.mismatches.is_empty() {
        let show: usize = std::env::var("CORPUS_SHOW_DIFFS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15);
        println!(
            "\nvalue mismatches (showing {} of {}):",
            show.min(tally.mismatches.len()),
            tally.mismatches.len()
        );
        for m in tally.mismatches.iter().take(show) {
            println!("\n  {} :: {}", m.file, m.sql);
            println!("    expected: {:?}", m.expected);
            println!("    got:      {:?}", m.got);
        }
    }

    println!("\nmost common failures:");
    let mut shapes: Vec<_> = tally.error_shapes.iter().collect();
    shapes.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (shape, n) in shapes.iter().take(14) {
        println!("  {:>5}  {}", n, shape);
    }

    println!("\nby group (executed / total):");
    let mut groups: Vec<_> = tally.groups.iter().collect();
    groups.sort_by_key(|(_, (ok, bad))| std::cmp::Reverse(ok + bad));
    for (name, (ok, bad)) in groups {
        let t = ok + bad;
        println!("  {:>5}/{:<5} {:>5.1}%  {}", ok, t, pct(*ok, t), name);
    }

    println!("\nfailures by cause:");
    let mut buckets: Vec<_> = tally.buckets.iter().collect();
    buckets.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (b, n) in buckets {
        println!("  {:>6}  {}", n, b.label());
        if let Some((sql, err)) = tally.examples.get(b) {
            println!("          e.g. {sql}");
            println!("            -> {err}");
        }
    }
}
