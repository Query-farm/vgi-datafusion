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

#[derive(Default)]
struct Tally {
    files_run: usize,
    files_skipped: usize,
    executed: usize,
    failed: usize,
    values_matched: usize,
    values_differed: usize,
    buckets: BTreeMap<Bucket, usize>,
    /// One example error per bucket, so the summary is diagnosable.
    examples: BTreeMap<Bucket, (String, String)>,
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

        match vgi_datafusion::sql(&ctx, &sql).await {
            Err(e) => record_failure(tally, &sql, &e.to_string()),
            Ok(df) => match df.collect().await {
                Err(e) => record_failure(tally, &sql, &e.to_string()),
                Ok(batches) => {
                    tally.executed += 1;
                    if let Some(expected) = expected {
                        if !expected.is_empty() {
                            let got = render(&batches);
                            if got == *expected {
                                tally.values_matched += 1;
                            } else {
                                tally.values_differed += 1;
                            }
                        }
                    }
                }
            },
        }
    }
}

fn record_failure(tally: &mut Tally, sql: &str, err: &str) {
    tally.failed += 1;
    let bucket = Bucket::classify(err);
    *tally.buckets.entry(bucket).or_default() += 1;
    tally
        .examples
        .entry(bucket)
        .or_insert_with(|| (first_line(sql), first_line(err)));
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

    let total = tally.executed + tally.failed;
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
    let compared = tally.values_matched + tally.values_differed;
    println!(
        "values:  {compared} queries compared — {} matched ({:.1}%), {} differed",
        tally.values_matched,
        pct(tally.values_matched, compared),
        tally.values_differed,
    );

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
