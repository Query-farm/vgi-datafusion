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
use std::sync::Arc;

use datafusion::prelude::{SessionConfig, SessionContext};
use futures::stream::{self, StreamExt};

/// One directive from a `.test` file.
#[derive(Debug)]
enum Record {
    /// `statement ok` / `statement error`
    Statement { sql: String, expect_ok: bool },
    /// `query <types>` with its expected rows
    Query { sql: String, expected: Vec<String> },
}

impl Record {
    fn sql(&self) -> &str {
        match self {
            Self::Statement { sql, .. } | Self::Query { sql, .. } => sql,
        }
    }
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
    /// Machine-readable outcomes by corpus-relative file.
    files: BTreeMap<String, FileTally>,
    /// Expected-error statements are syntax/error-contract tests, not positive
    /// capability evidence, so they are recorded but excluded from rates.
    expected_errors_ignored: usize,
    adapted_records: usize,
    blocked: usize,
}

impl Tally {
    fn merge(&mut self, other: Self) {
        self.files_run += other.files_run;
        self.files_skipped += other.files_skipped;
        self.executed += other.executed;
        self.failed += other.failed;
        self.not_applicable += other.not_applicable;
        self.timed_out += other.timed_out;
        self.values_matched += other.values_matched;
        self.values_rendering += other.values_rendering;
        self.values_unordered_subset += other.values_unordered_subset;
        self.values_differed += other.values_differed;
        self.expected_errors_ignored += other.expected_errors_ignored;
        self.adapted_records += other.adapted_records;
        self.blocked += other.blocked;
        for (bucket, count) in other.buckets {
            *self.buckets.entry(bucket).or_default() += count;
        }
        for (shape, count) in other.error_shapes {
            *self.error_shapes.entry(shape).or_default() += count;
        }
        for (group, (executed, failed)) in other.groups {
            let entry = self.groups.entry(group).or_default();
            entry.0 += executed;
            entry.1 += failed;
        }
        for (bucket, example) in other.examples {
            self.examples.entry(bucket).or_insert(example);
        }
        self.mismatches.extend(other.mismatches);
        self.files.extend(other.files);
    }
}

#[derive(Default)]
struct FileTally {
    skipped: bool,
    records: usize,
    expected_errors_ignored: usize,
    adapted_records: usize,
    blocked: usize,
    executed: usize,
    failed: usize,
    not_applicable: usize,
    timed_out: usize,
    values_matched: usize,
    values_rendering: usize,
    values_unordered_subset: usize,
    values_differed: usize,
    buckets: BTreeMap<Bucket, usize>,
    adaptations: Vec<AppliedAdaptation>,
}

#[derive(Clone)]
struct Area {
    id: String,
    title: String,
    status: String,
    groups: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverlayKind {
    EquivalentSql,
    OutOfScope,
    Blocked,
}

impl OverlayKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "equivalent_sql" => Ok(Self::EquivalentSql),
            "out_of_scope" => Ok(Self::OutOfScope),
            "blocked" => Ok(Self::Blocked),
            other => Err(format!(
                "overlay kind must be equivalent_sql, out_of_scope, or blocked; found {other:?}"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::EquivalentSql => "equivalent_sql",
            Self::OutOfScope => "out_of_scope",
            Self::Blocked => "blocked",
        }
    }
}

#[derive(Debug, Clone)]
struct RecordOverlay {
    record: usize,
    original_sql: String,
    replacement_sql: Option<String>,
    kind: OverlayKind,
    reason: String,
    issue: Option<String>,
    overlay_path: String,
}

#[derive(Debug, Clone)]
struct AppliedAdaptation {
    record: usize,
    mechanism: String,
    kind: String,
    reason: String,
    issue: Option<String>,
    original_sql: String,
    replacement_sql: Option<String>,
}

type OverlayMap = BTreeMap<String, Vec<RecordOverlay>>;

/// Do two result sets agree once DuckDB's *rendering* conventions are allowed
/// for?
///
/// The corpus records DuckDB's output verbatim, and three of its conventions
/// differ from Arrow's without either being wrong:
///
/// * an empty string prints as `(empty)`;
/// * struct keys are quoted — `{'lat': 3.0}` against `{lat: 3.0}`;
/// * map entries use `=` — `{a=1}` against Arrow's `{a: 1}`;
/// * BLOB bytes use mixed `\\xHH`/printable text against Arrow's hexadecimal;
/// * floats print at shortest-round-trip — `0.0003` against Arrow's
///   `0.00030000000000000003`, which are the same double.
///
/// Counting those as failures would put a rendering difference and a wrong
/// answer in the same bucket, and only one of them is worth anyone's time. So
/// they are compared separately and reported separately.
fn agrees_modulo_rendering(expected: &[String], got: &[String]) -> bool {
    if explain_rows_agree(expected, got) {
        return true;
    }
    if expected.len() != got.len() {
        return false;
    }
    rows_agree(expected, got)
}

/// Match DuckDB EXPLAIN assertions to the equivalent DataFusion node names.
///
/// DuckDB emits only its physical row and calls the nodes `EMPTY_RESULT` and
/// `VGI_TABLE_SCAN`; DataFusion emits logical plus physical rows and calls them
/// `EmptyExec` and `VgiScanExec`. This intentionally recognizes only the exact
/// corpus assertions below, so a scan where an empty result was expected still
/// remains a genuine mismatch.
fn explain_rows_agree(expected: &[String], got: &[String]) -> bool {
    let [expected] = expected else {
        return false;
    };
    let physical = got
        .iter()
        .filter_map(|row| row.strip_prefix("physical_plan\t"))
        .collect::<Vec<_>>();
    if physical.is_empty() {
        return false;
    }
    match expected.as_str() {
        "physical_plan\t<REGEX>:.*EMPTY_RESULT.*" => {
            physical.iter().any(|plan| plan.contains("EmptyExec"))
        }
        "physical_plan\t<!REGEX>:.*EMPTY_RESULT.*" => {
            physical.iter().all(|plan| !plan.contains("EmptyExec"))
        }
        "physical_plan\t<REGEX>:.*VGI_TABLE_SCAN.*" => {
            physical.iter().any(|plan| plan.contains("VgiScanExec"))
        }
        _ => false,
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)] // Rendering tests stay beside their adapter seam.
mod rendering_tests {
    use super::{adapt_duckdb_binary_sql, agrees_modulo_rendering, cells_agree};

    #[test]
    fn matches_equivalent_datafusion_plan_nodes_without_masking_missing_pruning() {
        let empty = vec![
            "logical_plan\tFilter".into(),
            "physical_plan\tFilterExec\n  EmptyExec".into(),
        ];
        let scan = vec![
            "logical_plan\tFilter".into(),
            "physical_plan\tFilterExec\n  VgiScanExec".into(),
        ];
        assert!(agrees_modulo_rendering(
            &["physical_plan\t<REGEX>:.*EMPTY_RESULT.*".into()],
            &empty,
        ));
        assert!(agrees_modulo_rendering(
            &["physical_plan\t<REGEX>:.*VGI_TABLE_SCAN.*".into()],
            &scan,
        ));
        assert!(agrees_modulo_rendering(
            &["physical_plan\t<!REGEX>:.*EMPTY_RESULT.*".into()],
            &scan,
        ));
        assert!(!agrees_modulo_rendering(
            &["physical_plan\t<REGEX>:.*EMPTY_RESULT.*".into()],
            &scan,
        ));
    }

    #[test]
    fn matches_maps_and_non_finite_floats_without_masking_value_changes() {
        assert!(cells_agree("{a=1, b=[2, 3]}", "{a: 1, b: [2, 3]}"));
        assert!(cells_agree(
            "{outer={inner=42}, labels=[x=y, z]}",
            "{outer: {inner: 42}, labels: [x=y, z]}"
        ));
        assert!(cells_agree(
            "{'2024-01-01 00:00:00'=a}",
            "{2024-01-01T00:00:00: a}"
        ));
        assert!(cells_agree("nan", "NaN"));
        assert!(!cells_agree("{a=1}", "{a: 2}"));
        assert!(!cells_agree("nan", "1"));
    }

    #[test]
    fn matches_blob_bytes_without_masking_different_bytes() {
        assert!(cells_agree(r"\xFF\xEE\xDD", "ffeedd"));
        // DuckDB escapes the non-printable DE byte and prints the remaining
        // ASCII bytes literally; Arrow prints every byte as hexadecimal.
        assert!(cells_agree(r"\xDEADBEEF", "de414442454546"));
        assert!(cells_agree(r"{\xDE\xAD=x}", "{dead: x}"));
        assert!(!cells_agree(r"\xDE\xAD", "deaf"));
        assert!(!cells_agree(r"{\xDE\xAD=x}", "{beef: x}"));
    }

    #[test]
    fn adapts_duckdb_binary_syntax_only_in_the_corpus_harness() {
        let adapted = adapt_duckdb_binary_sql(r"SELECT hex('\xCA\xFE'::BLOB)")
            .expect("valid adaptation")
            .expect("binary syntax changed");
        assert!(
            adapted.contains("upper(encode(X'cafe', 'hex'))"),
            "{adapted}"
        );

        let dynamic = adapt_duckdb_binary_sql("SELECT payload::BLOB FROM packets")
            .expect("valid adaptation")
            .expect("binary type changed");
        assert!(dynamic.contains("payload::BYTEA"), "{dynamic}");

        assert!(
            adapt_duckdb_binary_sql("SELECT custom.hex(payload) FROM packets")
                .expect("valid SQL")
                .is_none(),
            "qualified application function must not be rewritten"
        );
    }
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
        if a == b || (a.is_nan() && b.is_nan()) {
            return true;
        }
        let scale = a.abs().max(b.abs()).max(1.0);
        if (a - b).abs() <= scale * 1e-9 {
            return true;
        }
    }
    if normalize_timestamp(expected) == normalize_timestamp(got) {
        return true;
    }
    normalize_composite(expected) == normalize_composite(got)
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

/// Put DuckDB and Arrow composite-value renderings on common ground.
fn normalize_composite(s: &str) -> String {
    let normalized = normalize_blob_tokens(&normalize_map_separators(s))
        .replace("{'", "{")
        .replace(", '", ", ")
        .replace("': ", ": ");
    normalize_embedded_timestamps(&normalized)
}

/// Normalize DuckDB's BLOB display to Arrow's lowercase hexadecimal display.
///
/// DuckDB escapes non-printable bytes as `\\xHH` but leaves printable bytes
/// alone, so `DE 41 44` appears as `\\xDEAD`. Arrow displays the same bytes as
/// `de4144`. Only tokens whose first non-space characters are a valid escape
/// are treated as binary; ordinary words and hexadecimal-looking strings are
/// left untouched.
fn normalize_blob_tokens(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0usize;
    let mut token_start = true;
    while i < bytes.len() {
        if token_start && bytes[i].is_ascii_whitespace() {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        if token_start
            && i + 3 < bytes.len()
            && bytes[i] == b'\\'
            && bytes[i + 1] == b'x'
            && bytes[i + 2].is_ascii_hexdigit()
            && bytes[i + 3].is_ascii_hexdigit()
        {
            let start = i;
            while i < bytes.len() && !matches!(bytes[i], b',' | b':' | b'=' | b'}' | b']') {
                i += 1;
            }
            let token = &bytes[start..i];
            let mut decoded = Vec::with_capacity(token.len());
            let mut j = 0usize;
            while j < token.len() {
                if j + 3 < token.len()
                    && token[j] == b'\\'
                    && token[j + 1] == b'x'
                    && token[j + 2].is_ascii_hexdigit()
                    && token[j + 3].is_ascii_hexdigit()
                {
                    let pair =
                        std::str::from_utf8(&token[j + 2..j + 4]).expect("ASCII hex pair checked");
                    decoded.push(u8::from_str_radix(pair, 16).expect("ASCII hex pair checked"));
                    j += 4;
                } else {
                    decoded.push(token[j]);
                    j += 1;
                }
            }
            for byte in decoded {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                out.push(HEX[(byte >> 4) as usize]);
                out.push(HEX[(byte & 0x0f) as usize]);
            }
            token_start = false;
            continue;
        }

        let byte = bytes[i];
        out.push(byte);
        token_start = matches!(byte, b'{' | b'[' | b',' | b':' | b'=');
        i += 1;
    }
    String::from_utf8(out).expect("normalization preserves UTF-8 outside ASCII BLOB tokens")
}

fn normalize_embedded_timestamps(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    for (i, ch) in s.char_indices() {
        let embedded_timestamp = ch == 'T'
            && i >= 10
            && i + 2 < bytes.len()
            && bytes[i - 10..i - 6].iter().all(u8::is_ascii_digit)
            && bytes[i - 6] == b'-'
            && bytes[i - 5..i - 3].iter().all(u8::is_ascii_digit)
            && bytes[i - 3] == b'-'
            && bytes[i - 2..i].iter().all(u8::is_ascii_digit)
            && bytes[i + 1..i + 3].iter().all(u8::is_ascii_digit);
        out.push(if embedded_timestamp { ' ' } else { ch });
    }
    out
}

/// Change DuckDB's map-entry `=` to Arrow's `:` without touching equals signs
/// inside quoted keys/values or later in an unquoted value.
///
/// Each `{...}` level tracks whether it is still waiting for that entry's
/// first separator. This also handles nested maps and structs while ensuring a
/// genuine value difference remains a difference.
fn normalize_map_separators(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    // (waiting for this map/struct entry's separator, square-bracket depth at
    // which the `{` opened). A comma inside a list value must not begin a new
    // map entry.
    let mut frames: Vec<(bool, usize)> = Vec::new();
    let mut square_depth = 0usize;
    let mut quoted = false;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            out.push(ch);
            if quoted && chars.peek() == Some(&'\'') {
                out.push(chars.next().expect("peeked escaped quote"));
            } else {
                quoted = !quoted;
            }
            continue;
        }
        if !quoted {
            match ch {
                '{' => frames.push((true, square_depth)),
                '}' => {
                    frames.pop();
                }
                ',' => {
                    if let Some((waiting, entry_depth)) = frames.last_mut() {
                        if square_depth == *entry_depth {
                            *waiting = true;
                        }
                    }
                }
                ':' => {
                    if let Some((waiting, _)) = frames.last_mut() {
                        *waiting = false;
                    }
                }
                '=' if frames.last().is_some_and(|(waiting, entry_depth)| {
                    *waiting && square_depth == *entry_depth
                }) =>
                {
                    out.push(':');
                    if chars.peek().is_none_or(|next| !next.is_whitespace()) {
                        out.push(' ');
                    }
                    if let Some((waiting, _)) = frames.last_mut() {
                        *waiting = false;
                    }
                    continue;
                }
                '[' => square_depth += 1,
                ']' => square_depth = square_depth.saturating_sub(1),
                _ => {}
            }
        }
        out.push(ch);
    }
    out
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
    parse_text(&text, true)
}

fn parse_text(text: &str, honor_require_env: bool) -> Option<Vec<Record>> {
    let mut records = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        let line = line.trim_end();
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some(var) = line.strip_prefix("require-env ") {
            if honor_require_env && std::env::var(var.trim()).is_err() {
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

/// Apply a generic, harness-only DuckDB binary dialect adaptation.
///
/// This deliberately does not live in `vgi_datafusion::sql`: DataFusion users
/// can write Arrow Binary as `BYTEA`/hex literals and format it with
/// `encode(value, 'hex')`. The shared protocol corpus uses DuckDB's BLOB and
/// unary `hex()` spellings, so translate those only while measuring the corpus.
fn adapt_duckdb_binary_sql(sql: &str) -> Result<Option<String>, String> {
    use datafusion::sql::parser::{DFParser, Statement as DFStatement};
    use datafusion::sql::sqlparser::ast::{
        DataType as SQLDataType, Expr as SQLExpr, Function as SQLFunction, FunctionArg,
        FunctionArgExpr, FunctionArgumentList, FunctionArguments, Ident, ObjectName,
        ObjectNamePart, Value, VisitMut, VisitorMut,
    };
    use std::ops::ControlFlow;

    struct Rewrite {
        changed: bool,
    }

    impl VisitorMut for Rewrite {
        type Break = String;

        fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<Self::Break> {
            if let SQLExpr::Cast {
                expr: value,
                data_type: data_type @ SQLDataType::Blob(_),
                ..
            } = expr
            {
                if let SQLExpr::Value(value) = value.as_ref() {
                    if let Value::SingleQuotedString(text) = &value.value {
                        let bytes = match decode_duckdb_blob_text(text) {
                            Ok(bytes) => bytes,
                            Err(error) => return ControlFlow::Break(error),
                        };
                        let hex = bytes
                            .into_iter()
                            .map(|byte| format!("{byte:02x}"))
                            .collect::<String>();
                        *expr = SQLExpr::value(Value::HexStringLiteral(hex));
                        self.changed = true;
                        return ControlFlow::Continue(());
                    }
                }
                *data_type = SQLDataType::Bytea;
                self.changed = true;
                return ControlFlow::Continue(());
            }

            let SQLExpr::Function(function) = expr else {
                return ControlFlow::Continue(());
            };
            let [name] = function.name.0.as_slice() else {
                return ControlFlow::Continue(());
            };
            if !name
                .as_ident()
                .is_some_and(|name| name.value.eq_ignore_ascii_case("hex"))
                || !matches!(function.parameters, FunctionArguments::None)
                || function.filter.is_some()
                || function.null_treatment.is_some()
                || function.over.is_some()
                || !function.within_group.is_empty()
            {
                return ControlFlow::Continue(());
            }
            let FunctionArguments::List(arguments) = &mut function.args else {
                return ControlFlow::Continue(());
            };
            if arguments.args.len() != 1
                || arguments.duplicate_treatment.is_some()
                || !arguments.clauses.is_empty()
            {
                return ControlFlow::Continue(());
            }

            function.name = ObjectName(vec![ObjectNamePart::Identifier(Ident::new("encode"))]);
            arguments
                .args
                .push(FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::value(
                    Value::SingleQuotedString("hex".to_string()),
                ))));
            let encode = expr.clone();
            *expr = SQLExpr::Function(SQLFunction {
                name: ObjectName(vec![ObjectNamePart::Identifier(Ident::new("upper"))]),
                uses_odbc_syntax: false,
                parameters: FunctionArguments::None,
                args: FunctionArguments::List(FunctionArgumentList {
                    duplicate_treatment: None,
                    args: vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(encode))],
                    clauses: vec![],
                }),
                filter: None,
                null_treatment: None,
                over: None,
                within_group: vec![],
            });
            self.changed = true;
            ControlFlow::Continue(())
        }
    }

    let mut statements = match DFParser::parse_sql(sql) {
        Ok(statements) => statements,
        // ATTACH's extended option list and deliberately unsupported DuckDB
        // syntax still belong to the normal adapter/parser path.
        Err(_) => return Ok(None),
    };
    if statements.len() != 1 {
        return Ok(None);
    }
    let mut statement = statements.pop_front().expect("length checked");
    let DFStatement::Statement(inner) = &mut statement else {
        return Ok(None);
    };
    let mut rewrite = Rewrite { changed: false };
    if let ControlFlow::Break(error) = inner.as_mut().visit(&mut rewrite) {
        return Err(error);
    }
    Ok(rewrite.changed.then(|| statement.to_string()))
}

fn decode_duckdb_blob_text(value: &str) -> Result<Vec<u8>, String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if i + 3 < bytes.len() && bytes[i] == b'\\' && bytes[i + 1] == b'x' {
            let hex = std::str::from_utf8(&bytes[i + 2..i + 4])
                .map_err(|_| "invalid DuckDB BLOB escape".to_string())?;
            out.push(
                u8::from_str_radix(hex, 16)
                    .map_err(|_| format!("invalid DuckDB BLOB escape `\\x{hex}`"))?,
            );
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Ok(out)
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

async fn run_file(path: &Path, tally: &mut Tally, overlays: Option<&[RecordOverlay]>) {
    let file_label = corpus_relative(path);
    let group = corpus_group(&file_label).to_string();
    let Some(records) = parse(path) else {
        tally.files_skipped += 1;
        tally.files.insert(
            file_label,
            FileTally {
                skipped: true,
                ..Default::default()
            },
        );
        return;
    };
    if records.is_empty() {
        tally.files_skipped += 1;
        tally.files.insert(
            file_label,
            FileTally {
                skipped: true,
                ..Default::default()
            },
        );
        return;
    }
    tally.files_run += 1;
    let mut file = FileTally::default();

    // Match datafusion-cli: SHOW and standards-based metadata are part of the
    // user-facing SQL surface, and the CLI enables information_schema by
    // default. Leaving it off here misclassifies supported VGI metadata as an
    // adapter failure.
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    for (record_offset, record) in records.into_iter().enumerate() {
        let record_number = record_offset + 1;
        file.records += 1;

        // A statement the corpus expects to fail is not evidence either way
        // about this adapter, so it is not counted.
        if matches!(
            record,
            Record::Statement {
                expect_ok: false,
                ..
            }
        ) {
            tally.expected_errors_ignored += 1;
            file.expected_errors_ignored += 1;
            continue;
        }

        let overlay = overlays
            .unwrap_or_default()
            .iter()
            .find(|overlay| overlay.record == record_number);
        if let Some(overlay) = overlay {
            file.adaptations.push(AppliedAdaptation {
                record: record_number,
                mechanism: overlay.overlay_path.clone(),
                kind: overlay.kind.label().to_string(),
                reason: overlay.reason.clone(),
                issue: overlay.issue.clone(),
                original_sql: overlay.original_sql.clone(),
                replacement_sql: overlay.replacement_sql.clone(),
            });
            match overlay.kind {
                OverlayKind::OutOfScope => {
                    tally.not_applicable += 1;
                    file.not_applicable += 1;
                    continue;
                }
                OverlayKind::Blocked => {
                    tally.blocked += 1;
                    file.blocked += 1;
                    continue;
                }
                OverlayKind::EquivalentSql => {}
            }
        }

        let original_sql = record.sql();
        let selected_sql = overlay
            .and_then(|overlay| overlay.replacement_sql.as_deref())
            .unwrap_or(original_sql);
        let mut sql = expand(selected_sql);
        let mut adapted = overlay.is_some();
        if let Ok(Some(binary_sql)) = adapt_duckdb_binary_sql(&sql) {
            file.adaptations.push(AppliedAdaptation {
                record: record_number,
                mechanism: "builtin:duckdb_binary".to_string(),
                kind: OverlayKind::EquivalentSql.label().to_string(),
                reason: "DuckDB BLOB/hex syntax mapped to DataFusion Binary/encode semantics"
                    .to_string(),
                issue: None,
                original_sql: sql.clone(),
                replacement_sql: Some(binary_sql.clone()),
            });
            sql = binary_sql;
            adapted = true;
        }
        if adapted {
            tally.adapted_records += 1;
            file.adapted_records += 1;
        }
        let expected = match &record {
            Record::Statement { .. } => None,
            Record::Query { expected, .. } => Some(expected),
        };

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
                file.timed_out += 1;
                continue;
            }
            Ok(o) => o,
        };

        match outcome {
            Err(e) if not_applicable(&sql) => {
                let _ = e;
                tally.not_applicable += 1;
                file.not_applicable += 1;
            }
            Err(e) => {
                tally.groups.entry(group.clone()).or_default().1 += 1;
                file.failed += 1;
                let bucket = record_failure(tally, &sql, &e.to_string());
                *file.buckets.entry(bucket).or_default() += 1;
            }
            Ok(batches) => {
                tally.executed += 1;
                file.executed += 1;
                tally.groups.entry(group.clone()).or_default().0 += 1;
                if let Some(expected) = expected {
                    if !expected.is_empty() {
                        // `LIMIT` with no `ORDER BY` returns whichever rows
                        // arrive first, so under repartitioning it is a
                        // different — equally correct — subset than DuckDB's.
                        // No comparison can tell that apart from a defect.
                        if is_arbitrary_subset(&sql) {
                            tally.values_unordered_subset += 1;
                            file.values_unordered_subset += 1;
                            continue;
                        }
                        let mut got = render(&batches);
                        let mut expected = expected.clone();
                        if !is_ordered(&sql) {
                            // sqllogictest's `rowsort`: with no ORDER BY the
                            // rows are a multiset, not a list.
                            got.sort();
                            expected.sort();
                        }
                        let expected = &expected;
                        if got == *expected {
                            tally.values_matched += 1;
                            file.values_matched += 1;
                        } else if agrees_modulo_rendering(expected, &got) {
                            tally.values_rendering += 1;
                            file.values_rendering += 1;
                        } else {
                            tally.values_differed += 1;
                            file.values_differed += 1;
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
    tally.files.insert(file_label, file);
}

fn record_failure(tally: &mut Tally, sql: &str, err: &str) -> Bucket {
    tally.failed += 1;
    *tally.error_shapes.entry(error_shape(err)).or_default() += 1;
    let bucket = Bucket::classify(err);
    *tally.buckets.entry(bucket).or_default() += 1;
    tally
        .examples
        .entry(bucket)
        .or_insert_with(|| (first_line(sql), first_line(err)));
    bucket
}

fn corpus_relative(path: &Path) -> String {
    path.to_string_lossy()
        .rsplit("integration/")
        .next()
        .unwrap_or("")
        .to_string()
}

fn corpus_group(file: &str) -> &str {
    file.split('/').next().unwrap_or("?")
}

/// Is this record about engine configuration rather than the worker protocol?
///
/// `SET`/`PRAGMA`/`CALL`/`RESET` configure DuckDB or the VGI *extension* —
/// `SET vgi_result_cache_dir`, `CALL enable_logging(...)` — and have no meaning
/// here. They are reported separately rather than counted against conformance,
/// which would otherwise charge this adapter for not being DuckDB.
fn not_applicable(sql: &str) -> bool {
    let head = sql.split_whitespace().next().unwrap_or("");
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

fn overlay_path(root: &Path, source: &Path) -> Option<PathBuf> {
    let relative = corpus_relative(source);
    let relative = Path::new(&relative);
    if relative.is_absolute()
        || relative.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        })
    {
        return None;
    }
    Some(root.join(format!("{}.datafusion.json", relative.display())))
}

/// Load sparse DataFusion substitutions while keeping the upstream test as the
/// sole source of record order and expected rows.
///
/// `original_sql` is deliberately exact: an upstream edit makes the overlay
/// stale and aborts the selected run instead of silently applying a replacement
/// to a different assertion.
fn load_overlays(files: &[PathBuf], root: Option<&Path>) -> Result<OverlayMap, String> {
    let Some(root) = root else {
        return Ok(BTreeMap::new());
    };
    let mut overlays = BTreeMap::new();
    for source in files {
        let Some(path) = overlay_path(root, source) else {
            continue;
        };
        if !path.exists() {
            continue;
        }
        let source_text = std::fs::read_to_string(source)
            .map_err(|error| format!("could not read {}: {error}", source.display()))?;
        let records = parse_text(&source_text, false)
            .ok_or_else(|| format!("could not parse records from {}", source.display()))?;
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("could not read overlay {}: {error}", path.display()))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid overlay {}: {error}", path.display()))?;
        if value.get("schema_version").and_then(|value| value.as_u64()) != Some(1) {
            return Err(format!(
                "overlay {} must declare schema_version 1",
                path.display()
            ));
        }
        let entries = value
            .get("records")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("overlay {} must contain a records array", path.display()))?;
        let mut by_record = BTreeMap::new();
        for entry in entries {
            let record = entry
                .get("record")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    format!(
                        "overlay {} has a record without a positive 1-based index",
                        path.display()
                    )
                })?;
            let string = |name: &str| {
                entry
                    .get(name)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        format!(
                            "overlay {} record {record} is missing string {name:?}",
                            path.display()
                        )
                    })
            };
            let original_sql = string("original_sql")?;
            let kind = OverlayKind::parse(&string("kind")?)?;
            let reason = string("reason")?;
            if reason.trim().is_empty() {
                return Err(format!(
                    "overlay {} record {record} has an empty reason",
                    path.display()
                ));
            }
            let replacement_sql = entry
                .get("replacement_sql")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            if kind == OverlayKind::EquivalentSql
                && replacement_sql.as_deref().is_none_or(str::is_empty)
            {
                return Err(format!(
                    "overlay {} record {record} equivalent_sql requires replacement_sql",
                    path.display()
                ));
            }
            if kind != OverlayKind::EquivalentSql && replacement_sql.is_some() {
                return Err(format!(
                    "overlay {} record {record} {} must not replace SQL",
                    path.display(),
                    kind.label()
                ));
            }
            let source_record = records.get(record - 1).ok_or_else(|| {
                format!(
                    "overlay {} targets record {record}, but {} has only {} records",
                    path.display(),
                    source.display(),
                    records.len()
                )
            })?;
            if source_record.sql() != original_sql {
                return Err(format!(
                    "stale overlay {} record {record}: original_sql no longer matches {}\n  overlay: {}\n  source:  {}",
                    path.display(),
                    source.display(),
                    first_line(&original_sql),
                    first_line(source_record.sql())
                ));
            }
            let overlay = RecordOverlay {
                record,
                original_sql,
                replacement_sql,
                kind,
                reason,
                issue: entry
                    .get("issue")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                overlay_path: path.display().to_string(),
            };
            if by_record.insert(record, overlay).is_some() {
                return Err(format!(
                    "overlay {} contains duplicate record {record}",
                    path.display()
                ));
            }
        }
        overlays.insert(corpus_relative(source), by_record.into_values().collect());
    }
    Ok(overlays)
}

fn load_manifest(path: &Path) -> Result<Vec<Area>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "could not read compatibility manifest {}: {e}",
            path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid compatibility manifest {}: {e}", path.display()))?;
    let definitions = value
        .get("areas")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "compatibility manifest must contain an `areas` array".to_string())?;
    definitions
        .iter()
        .map(|area| {
            let string = |key: &str| {
                area.get(key)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| format!("compatibility area is missing string `{key}`"))
            };
            let groups = area
                .get("groups")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| "compatibility area is missing array `groups`".to_string())?
                .iter()
                .map(|group| {
                    group
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| "compatibility group must be a string".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Area {
                id: string("id")?,
                title: string("title")?,
                status: string("status")?,
                groups,
            })
        })
        .collect()
}

fn area_map(areas: &[Area]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for area in areas {
        for group in &area.groups {
            if let Some(previous) = out.insert(group.clone(), area.id.clone()) {
                return Err(format!(
                    "corpus group `{group}` is assigned to both `{previous}` and `{}`",
                    area.id
                ));
            }
        }
    }
    Ok(out)
}

fn validate_manifest_coverage(
    files: &[PathBuf],
    groups: &BTreeMap<String, String>,
) -> Result<(), String> {
    let mut missing = Vec::new();
    for file in files {
        let relative = corpus_relative(file);
        let group = corpus_group(&relative);
        if !groups.contains_key(group) && !missing.iter().any(|item| item == group) {
            missing.push(group.to_string());
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "compatibility manifest does not assign corpus group(s): {}",
            missing.join(", ")
        ))
    }
}

fn write_json_report(
    path: &Path,
    roots: &[PathBuf],
    source_files: usize,
    tally: &Tally,
    areas: &[Area],
    group_areas: &BTreeMap<String, String>,
) -> Result<(), String> {
    use serde_json::json;

    let files = tally
        .files
        .iter()
        .map(|(name, file)| {
            let buckets = file
                .buckets
                .iter()
                .map(|(bucket, count)| (bucket.label().to_string(), json!(count)))
                .collect::<serde_json::Map<_, _>>();
            let adaptations = file
                .adaptations
                .iter()
                .map(|adaptation| {
                    json!({
                        "record": adaptation.record,
                        "mechanism": adaptation.mechanism,
                        "kind": adaptation.kind,
                        "reason": adaptation.reason,
                        "issue": adaptation.issue,
                        "original_sql": adaptation.original_sql,
                        "replacement_sql": adaptation.replacement_sql,
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "path": name,
                "area": group_areas.get(corpus_group(name)),
                "skipped": file.skipped,
                "records": file.records,
                "expected_errors_ignored": file.expected_errors_ignored,
                "adapted_records": file.adapted_records,
                "executed": file.executed,
                "failed": file.failed,
                "not_applicable": file.not_applicable,
                "blocked": file.blocked,
                "timed_out": file.timed_out,
                "values": {
                    "exact": file.values_matched,
                    "rendering_equivalent": file.values_rendering,
                    "different": file.values_differed,
                    "unordered_subset": file.values_unordered_subset,
                },
                "failure_buckets": buckets,
                "adaptations": adaptations,
            })
        })
        .collect::<Vec<_>>();

    let groups = tally
        .groups
        .iter()
        .map(|(group, (executed, failed))| {
            json!({
                "group": group,
                "area": group_areas.get(group),
                "executed": executed,
                "failed": failed,
            })
        })
        .collect::<Vec<_>>();

    let area_summaries = areas
        .iter()
        .map(|area| {
            let mut file_count = 0usize;
            let mut files_skipped = 0usize;
            let mut executed = 0usize;
            let mut failed = 0usize;
            let mut not_applicable = 0usize;
            let mut adapted_records = 0usize;
            let mut blocked = 0usize;
            let mut timed_out = 0usize;
            for (name, file) in &tally.files {
                if group_areas.get(corpus_group(name)) == Some(&area.id) {
                    file_count += 1;
                    files_skipped += usize::from(file.skipped);
                    executed += file.executed;
                    failed += file.failed;
                    not_applicable += file.not_applicable;
                    adapted_records += file.adapted_records;
                    blocked += file.blocked;
                    timed_out += file.timed_out;
                }
            }
            json!({
                "id": area.id,
                "title": area.title,
                "declared_status": area.status,
                "groups": area.groups,
                "files": file_count,
                "files_skipped": files_skipped,
                "executed": executed,
                "failed": failed,
                "not_applicable": not_applicable,
                "adapted_records": adapted_records,
                "blocked": blocked,
                "timed_out": timed_out,
            })
        })
        .collect::<Vec<_>>();

    let failure_buckets = tally
        .buckets
        .iter()
        .map(|(bucket, count)| (bucket.label().to_string(), json!(count)))
        .collect::<serde_json::Map<_, _>>();
    let measured = tally.executed + tally.failed + tally.timed_out;
    let report = json!({
        "schema_version": 2,
        "source_roots": roots.iter().map(|root| root.display().to_string()).collect::<Vec<_>>(),
        "source_files": source_files,
        "totals": {
            "files_run": tally.files_run,
            "files_skipped": tally.files_skipped,
            "measured_records": measured,
            "executed": tally.executed,
            "failed": tally.failed,
            "not_applicable": tally.not_applicable,
            "adapted_records": tally.adapted_records,
            "blocked": tally.blocked,
            "timed_out": tally.timed_out,
            "expected_errors_ignored": tally.expected_errors_ignored,
            "values": {
                "exact": tally.values_matched,
                "rendering_equivalent": tally.values_rendering,
                "different": tally.values_differed,
                "unordered_subset": tally.values_unordered_subset,
            },
        },
        "failure_buckets": failure_buckets,
        "areas": area_summaries,
        "groups": groups,
        "files": files,
    });
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| format!("could not serialize corpus report: {e}"))?;
    std::fs::write(path, bytes)
        .map_err(|e| format!("could not write corpus report {}: {e}", path.display()))
}

fn compare_reports(
    baseline_path: &Path,
    current_path: &Path,
    selected_files_only: bool,
) -> Result<Vec<String>, String> {
    let read = |path: &Path| -> Result<serde_json::Value, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("could not read corpus report {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| format!("invalid corpus report {}: {e}", path.display()))
    };
    let baseline = read(baseline_path)?;
    let current = read(current_path)?;
    let index = |report: &serde_json::Value| {
        report
            .get("files")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|file| Some((file.get("path")?.as_str()?.to_string(), file.clone())))
            .collect::<BTreeMap<_, _>>()
    };
    let baseline_files = index(&baseline);
    let current_files = index(&current);
    let number = |file: &serde_json::Value, pointer: &str| {
        file.pointer(pointer)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    };
    let mut regressions = Vec::new();
    let paths = if selected_files_only {
        current_files
            .keys()
            .filter(|path| baseline_files.contains_key(*path))
            .cloned()
            .collect::<Vec<_>>()
    } else {
        baseline_files.keys().cloned().collect::<Vec<_>>()
    };
    for path in paths {
        let before = &baseline_files[&path];
        let Some(after) = current_files.get(&path) else {
            regressions.push(format!("{path}: missing from current report"));
            continue;
        };
        for (pointer, label, higher_is_bad) in [
            ("/executed", "executed records", false),
            ("/failed", "failed records", true),
            ("/timed_out", "timeouts", true),
        ] {
            let old = number(before, pointer);
            let new = number(after, pointer);
            if (higher_is_bad && new > old) || (!higher_is_bad && new < old) {
                regressions.push(format!("{path}: {label} changed from {old} to {new}"));
            }
        }
        let old_agree =
            number(before, "/values/exact") + number(before, "/values/rendering_equivalent");
        let new_agree =
            number(after, "/values/exact") + number(after, "/values/rendering_equivalent");
        if new_agree < old_agree {
            regressions.push(format!(
                "{path}: agreeing value checks changed from {old_agree} to {new_agree}"
            ));
        }
        let old_different = number(before, "/values/different");
        let new_different = number(after, "/values/different");
        let old_compared = old_agree + old_different;
        let new_compared = new_agree + new_different;
        // A formerly failing query that starts executing can reveal a value
        // mismatch for the first time. That is new evidence to triage, not a
        // regression. Only flag additional mismatches when the comparison set
        // did not grow; lost agreements are independently guarded above.
        if new_compared <= old_compared && new_different > old_different {
            regressions.push(format!(
                "{path}: value mismatches changed from {old_different} to {new_different}"
            ));
        }
    }
    Ok(regressions)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut roots = Vec::new();
    let mut json_path = std::env::var_os("CORPUS_JSON").map(PathBuf::from);
    let mut compare_path: Option<PathBuf> = None;
    let mut compare_selected = false;
    let mut jobs = std::env::var("CORPUS_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let mut manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("corpus")
        .join("compatibility.json");
    let mut overlay_root = Some(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("corpus")
            .join("overlays"),
    );
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => {
                let Some(path) = args.next() else {
                    eprintln!("--json requires a path");
                    std::process::exit(2);
                };
                json_path = Some(PathBuf::from(path));
            }
            "--manifest" => {
                let Some(path) = args.next() else {
                    eprintln!("--manifest requires a path");
                    std::process::exit(2);
                };
                manifest_path = PathBuf::from(path);
            }
            "--overlays" => {
                let Some(path) = args.next() else {
                    eprintln!("--overlays requires a directory");
                    std::process::exit(2);
                };
                overlay_root = Some(PathBuf::from(path));
            }
            "--no-overlays" => overlay_root = None,
            "--compare" | "--compare-selected" => {
                let Some(path) = args.next() else {
                    eprintln!("{arg} requires a baseline report path");
                    std::process::exit(2);
                };
                compare_path = Some(PathBuf::from(path));
                compare_selected = arg == "--compare-selected";
            }
            "--jobs" | "-j" => {
                let Some(value) = args.next() else {
                    eprintln!("--jobs requires a positive integer");
                    std::process::exit(2);
                };
                jobs = match value.parse::<usize>() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("--jobs requires a positive integer, found {value:?}");
                        std::process::exit(2);
                    }
                };
            }
            "--help" | "-h" => {
                println!(
                    "Usage: corpus [--jobs N] [--json REPORT.json] \
                     [--compare BASELINE.json | --compare-selected BASELINE.json] \
                     [--manifest compatibility.json] [--overlays DIR | --no-overlays] [PATH ...]"
                );
                return;
            }
            _ if arg.starts_with('-') => {
                eprintln!("unknown option: {arg}");
                std::process::exit(2);
            }
            _ => roots.push(PathBuf::from(arg)),
        }
    }
    let roots: Vec<PathBuf> = if roots.is_empty() {
        vec![PathBuf::from(
            std::env::var("HOME").unwrap_or_default() + "/Development/vgi/test/sql/integration",
        )]
    } else {
        roots
    };

    let mut files = Vec::new();
    for root in &roots {
        if root.is_dir() {
            collect_tests(root, &mut files);
        } else {
            files.push(root.clone());
        }
    }

    let areas = match load_manifest(&manifest_path) {
        Ok(areas) => areas,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let group_areas = match area_map(&areas)
        .and_then(|groups| validate_manifest_coverage(&files, &groups).map(|_| groups))
    {
        Ok(groups) => groups,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let overlays = match load_overlays(&files, overlay_root.as_deref()) {
        Ok(overlays) => Arc::new(overlays),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };

    if std::env::var("VGI_TEST_WORKER").is_err() {
        eprintln!("VGI_TEST_WORKER is not set — every file will skip.");
    }

    let file_count = files.len();
    let mut partials = stream::iter(files.into_iter().enumerate())
        .map(|(i, file)| {
            let overlays = Arc::clone(&overlays);
            async move {
                if i % 25 == 0 {
                    eprintln!("  [{i}/{file_count}] {}", file.display());
                }
                let mut tally = Tally::default();
                let label = corpus_relative(&file);
                run_file(&file, &mut tally, overlays.get(&label).map(Vec::as_slice)).await;
                (i, tally)
            }
        })
        .buffer_unordered(jobs)
        .collect::<Vec<_>>()
        .await;
    // Merge in source order so examples and mismatch presentation remain
    // deterministic even when files finish out of order.
    partials.sort_by_key(|(i, _)| *i);
    let mut tally = Tally::default();
    for (_, partial) in partials {
        tally.merge(partial);
    }

    let total = tally.executed + tally.failed + tally.timed_out;
    let pct = |n: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            n as f64 * 100.0 / d as f64
        }
    };

    println!("\n=== corpus: {file_count} files ===");
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
            "         {} not applicable (engine, dialect, or extension surface)",
            tally.not_applicable
        );
    }
    if tally.blocked > 0 {
        println!(
            "         {} blocked by reviewed external fixtures",
            tally.blocked
        );
    }
    if tally.adapted_records > 0 {
        println!(
            "         {} records used reviewed DataFusion-equivalent SQL",
            tally.adapted_records
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

    println!("\nby compatibility area:");
    for area in &areas {
        let (mut executed, mut failed) = (0usize, 0usize);
        for group in &area.groups {
            if let Some((ok, bad)) = tally.groups.get(group) {
                executed += ok;
                failed += bad;
            }
        }
        let total = executed + failed;
        println!(
            "  {:>5}/{:<5} {:>5.1}%  {:<14} {}",
            executed,
            total,
            pct(executed, total),
            area.status,
            area.title,
        );
    }

    if let Some(path) = json_path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if let Err(error) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "could not create report directory {}: {error}",
                    parent.display()
                );
                std::process::exit(2);
            }
        }
        if let Err(error) =
            write_json_report(&path, &roots, file_count, &tally, &areas, &group_areas)
        {
            eprintln!("{error}");
            std::process::exit(2);
        }
        println!("\nJSON report: {}", path.display());
        if let Some(baseline) = compare_path {
            match compare_reports(&baseline, &path, compare_selected) {
                Ok(regressions) if regressions.is_empty() => {
                    let scope = if compare_selected {
                        "selected-file baseline comparison"
                    } else {
                        "baseline comparison"
                    };
                    println!("{scope}: no regressions");
                }
                Ok(regressions) => {
                    eprintln!(
                        "baseline comparison found {} regression(s):",
                        regressions.len()
                    );
                    for regression in regressions {
                        eprintln!("  {regression}");
                    }
                    std::process::exit(1);
                }
                Err(error) => {
                    eprintln!("{error}");
                    std::process::exit(2);
                }
            }
        }
    } else if compare_path.is_some() {
        eprintln!("--compare requires --json so there is a current report to compare");
        std::process::exit(2);
    }
}
