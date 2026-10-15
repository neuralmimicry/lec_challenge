//! Published input domains are checked before any oracle uses typed indexing.
use super::*;
use anyhow::ensure;

pub(super) fn representation(value: &Value, array_limit: usize) -> Result<()> {
    match value {
        Value::Number(n) => ensure!(
            n.as_i64().is_some(),
            "only signed 64-bit integer JSON numbers are permitted"
        ),
        Value::String(s) => ensure!(
            s.is_ascii() && s.len() <= 256,
            "strings must be ASCII and at most 256 bytes"
        ),
        Value::Array(a) => {
            ensure!(a.len() <= array_limit, "array exceeds published limit");
            for v in a {
                representation(v, array_limit)?;
            }
        }
        Value::Object(m) => {
            for (k, v) in m {
                ensure!(k.is_ascii() && k.len() <= 256, "invalid JSON object key");
                representation(v, array_limit)?;
            }
        }
        Value::Bool(_) | Value::Null => {}
    }
    Ok(())
}
fn array(v: &Value) -> Result<&[Value]> {
    v.as_array().map(Vec::as_slice).context("array required")
}
fn integer(v: &Value) -> Result<i64> {
    v.as_i64().context("signed integer required")
}
fn text(v: &Value) -> Result<&str> {
    v.as_str().context("string required")
}
fn integers(v: &Value) -> Result<Vec<i64>> {
    array(v)?.iter().map(integer).collect()
}
fn tuple(v: &Value, n: usize) -> Result<Vec<i64>> {
    let a = integers(v)?;
    ensure!(a.len() == n, "expected {n} integers");
    Ok(a)
}
fn matrix(v: &Value, width: usize) -> Result<Vec<Vec<i64>>> {
    array(v)?.iter().map(|r| tuple(r, width)).collect()
}
fn pair_arrays(v: &Value) -> Result<()> {
    ensure!(array(v)?.len() == 2, "two arrays required");
    integers(&v[0])?;
    integers(&v[1])?;
    Ok(())
}
fn nonnegative(n: i64) -> Result<()> {
    ensure!(n >= 0, "nonnegative integer required");
    Ok(())
}
fn graph(v: &Value, id: &str) -> Result<()> {
    let n = integer(&v["n"])?;
    ensure!((1..=64).contains(&n), "graph requires 1..64 vertices");
    let edges = matrix(&v["edges"], 2)?;
    let mut seen = BTreeSet::new();
    for e in edges {
        ensure!(
            (0..n).contains(&e[0]) && (0..n).contains(&e[1]),
            "graph vertex outside 0..n"
        );
        ensure!(seen.insert(e), "duplicate graph edge");
    }
    if matches!(id, "graph.reachable" | "graph.distance") {
        ensure!(
            (0..n).contains(&integer(&v["start"])?),
            "invalid graph start"
        );
    }
    if id == "graph.distance" {
        ensure!(
            (0..n).contains(&integer(&v["target"])?),
            "invalid graph target"
        );
    }
    Ok(())
}
fn rectangular(v: &Value, square: bool) -> Result<()> {
    let a = array(v)?;
    ensure!(!a.is_empty(), "matrix needs at least one row");
    let width = array(&a[0])?.len();
    ensure!(width > 0, "matrix needs at least one column");
    matrix(v, width)?;
    ensure!(!square || a.len() == width, "square matrix required");
    Ok(())
}

pub(super) fn validate_input(id: &str, v: &Value) -> Result<()> {
    representation(v, 64)?;
    match id.split('.').next().context("task source missing")? {
        "ledger" => {
            let a = integers(v)?;
            if matches!(id, "ledger.sum" | "ledger.prefix") {
                ensure!(
                    a.iter().all(|n| n.unsigned_abs() <= 10000),
                    "ledger integer magnitude exceeds 10000"
                );
            }
            if id == "ledger.differences" {
                for p in a.windows(2) {
                    p[1].checked_sub(p[0])
                        .context("adjacent difference exceeds i64")?;
                }
            }
        }
        "stock" => {
            for s in array(v)? {
                text(s)?;
            }
        }
        "text" | "validation" => {
            text(v)?;
        }
        "intervals" => {
            for r in matrix(v, 2)? {
                ensure!(r[0] < r[1], "half-open interval must have positive length");
                r[1].checked_sub(r[0])
                    .context("interval arithmetic exceeds i64")?;
            }
        }
        "graph" => graph(v, id)?,
        "matrices" => rectangular(v, id == "matrices.diagonals")?,
        "search" => match id {
            "search.occurrences" => {
                text(&v["text"])?;
                ensure!(!text(&v["pattern"])?.is_empty(), "pattern must be nonempty");
            }
            "search.common_prefix" => {
                for s in array(v)? {
                    text(s)?;
                }
            }
            "search.runs" => {
                text(v)?;
            }
            "search.anagram" => {
                ensure!(array(v)?.len() == 2, "anagram requires two strings");
                text(&v[0])?;
                text(&v[1])?;
            }
            _ => bail!("unknown search task"),
        },
        "paths" => {
            let s = text(v)?;
            ensure!(!s.contains('\0'), "POSIX path cannot contain NUL");
            if id == "paths.extension" {
                ensure!(!s.contains('/'), "filename must not contain slash");
            } else {
                ensure!(s.starts_with('/'), "absolute path required");
                if id == "paths.ancestors" {
                    ensure!(
                        s == "/"
                            || (!s.ends_with('/')
                                && !s.contains("//")
                                && s.split('/').skip(1).all(|p| p != "." && p != "..")),
                        "canonical path required"
                    );
                }
            }
        }
        "sequences" => match id {
            "sequences.interleave" | "sequences.zip_longest" => pair_arrays(v)?,
            "sequences.chunks" => {
                integers(&v["values"])?;
                ensure!(integer(&v["size"])? >= 1, "positive chunk size required");
            }
            _ => {
                integers(v)?;
            }
        },
        "calendar" => match id {
            "calendar.leap" => ensure!((1..=9999).contains(&integer(v)?), "year out of range"),
            "calendar.month_days" => {
                let a = tuple(v, 2)?;
                ensure!(
                    (1..=9999).contains(&a[0]) && (1..=12).contains(&a[1]),
                    "year/month out of range"
                );
            }
            "calendar.ordinal" => {
                let a = tuple(v, 3)?;
                ensure!(
                    (1..=9999).contains(&a[0])
                        && (1..=12).contains(&a[1])
                        && a[2] >= 1
                        && a[2] <= days(a[0], a[1]),
                    "ordinal requires a valid date"
                );
            }
            _ => {
                tuple(v, 3)?;
            }
        },
        "bits" => {
            let a = if id == "bits.hamming" {
                tuple(v, 2)?
            } else {
                vec![integer(v)?]
            };
            ensure!(
                a.iter().all(|n| (0..=u32::MAX as i64).contains(n)),
                "unsigned 32-bit values required"
            );
        }
        "money" => match id {
            "money.parse" => {
                text(v)?;
            }
            "money.format" => ensure!(
                integer(v)?.unsigned_abs() <= 100000000,
                "monetary format magnitude exceeds 100000000 cents"
            ),
            "money.split" => {
                let a = tuple(v, 2)?;
                nonnegative(a[0])?;
                ensure!(
                    a[1] >= 1 && a[1] <= 64,
                    "positive share count within the array bound required"
                );
            }
            _ => {
                for r in matrix(v, 2)? {
                    nonnegative(r[0])?;
                    nonnegative(r[1])?;
                    r[0].checked_mul(r[1])
                        .context("basket multiplication exceeds i64")?;
                }
            }
        },
        "dispatch" => match id {
            "dispatch.round_robin" => {
                for n in integers(&v["jobs"])? {
                    nonnegative(n)?;
                }
                ensure!(integer(&v["quantum"])? >= 1, "positive quantum required");
            }
            "dispatch.priority" => {
                let mut seen = BTreeSet::new();
                for r in matrix(v, 2)? {
                    ensure!(seen.insert(r[1]), "priority job IDs must be unique");
                }
            }
            _ => {
                for r in matrix(v, 2)? {
                    nonnegative(r[0])?;
                    nonnegative(r[1])?;
                }
            }
        },
        "encoding" => match id {
            "encoding.hex_encode" => ensure!(
                integers(v)?.iter().all(|n| (0..=255).contains(n)),
                "byte values required"
            ),
            "encoding.caesar" => {
                text(&v["text"])?;
                integer(&v["shift"])?;
            }
            _ => {
                text(v)?;
            }
        },
        "tables" => {
            if id == "tables.inner_join" {
                ensure!(array(v)?.len() == 2, "join requires two tables");
                for table in array(v)? {
                    let mut keys = BTreeSet::new();
                    for r in matrix(table, 2)? {
                        ensure!(keys.insert(r[0]), "join keys must be unique on each side");
                    }
                }
            } else {
                matrix(v, 2)?;
            }
        }
        "geometry" => {
            let a = matrix(v, 2)?;
            if id == "geometry.area2" {
                ensure!(a.len() >= 3, "polygon requires at least three vertices");
            }
        }
        "statistics" => {
            if id == "statistics.quantile" {
                ensure!(
                    !integers(&v["values"])?.is_empty(),
                    "quantile requires nonempty values"
                );
                ensure!(
                    (1..=100).contains(&integer(&v["p"])?),
                    "percentile out of range"
                );
            } else {
                let a = integers(v)?;
                ensure!(
                    id == "statistics.ranks" || !a.is_empty(),
                    "nonempty statistical input required"
                );
            }
        }
        "sets" => pair_arrays(v)?,
        "rational" => {
            let fractions = if matches!(id, "rational.add" | "rational.compare") {
                ensure!(array(v)?.len() == 2, "two fractions required");
                vec![tuple(&v[0], 2)?, tuple(&v[1], 2)?]
            } else {
                vec![tuple(v, 2)?]
            };
            let limit = if matches!(id, "rational.add" | "rational.compare") {
                1000
            } else {
                10000
            };
            for f in fractions {
                ensure!(
                    f[1] != 0 && f.iter().all(|n| n.unsigned_abs() <= limit),
                    "fraction denominator/magnitude outside contract"
                );
            }
        }
        "messages" => match id {
            "messages.headers" => {
                for s in array(v)? {
                    ensure!(text(s)?.contains(':'), "header needs a colon");
                }
            }
            "messages.address" => {
                let s = text(v)?;
                let parts: Vec<_> = s.split('@').collect();
                ensure!(
                    parts.len() == 2 && parts.iter().all(|p| !p.is_empty()),
                    "address requires exactly one @ and two nonempty sides"
                );
            }
            "messages.wrap" => {
                let width = integer(&v["width"])?;
                ensure!(width > 0, "positive wrap width required");
                ensure!(
                    text(&v["text"])?
                        .split_ascii_whitespace()
                        .all(|w| w.len() as i64 <= width),
                    "word exceeds wrap width"
                );
            }
            _ => {
                text(v)?;
            }
        },
        "automata" => {
            let s = text(v)?;
            match id {
                "automata.divisible3" => ensure!(
                    !s.is_empty() && s.bytes().all(|b| b == b'0' || b == b'1'),
                    "nonempty binary input required"
                ),
                "automata.transitions" => ensure!(
                    s.bytes().all(|b| b == b'0' || b == b'1'),
                    "binary input required"
                ),
                "automata.balanced_prefix" => ensure!(
                    s.bytes().all(|b| b == b'(' || b == b')'),
                    "parenthesis string required"
                ),
                _ => {}
            }
        }
        "transactions" => {
            if id == "transactions.settle" {
                for r in matrix(v, 3)? {
                    nonnegative(r[2])?;
                }
            } else {
                let initial = integer(&v["initial"])?;
                let deltas = integers(&v["deltas"])?;
                if id == "transactions.first_overdraft" {
                    nonnegative(initial)?;
                }
                let mut running = initial;
                for d in deltas {
                    running = running
                        .checked_add(d)
                        .context("transaction arithmetic exceeds i64")?;
                }
            }
        }
        "ranges" => match id {
            "ranges.expand" => {
                let mut n = 0i64;
                for r in matrix(v, 2)? {
                    ensure!(r[0] <= r[1], "inclusive range endpoints reversed");
                    n = n
                        .checked_add(
                            r[1].checked_sub(r[0])
                                .and_then(|d| d.checked_add(1))
                                .context("range size exceeds i64")?,
                        )
                        .context("expanded total exceeds i64")?;
                }
                ensure!(n <= 256, "expanded range total exceeds 256");
            }
            "ranges.compress" => {
                integers(v)?;
            }
            "ranges.missing" => {
                integers(&v["values"])?;
                let lo = integer(&v["low"])?;
                let hi = integer(&v["high"])?;
                ensure!(
                    hi.checked_sub(lo).is_some_and(|n| (0..=256).contains(&n)),
                    "missing range bounds outside contract"
                );
            }
            _ => {
                integers(&v["values"])?;
                ensure!(integer(&v["k"])? >= 1, "positive one-based index required");
            }
        },
        _ => bail!("unknown task source"),
    }
    Ok(())
}

pub(super) fn validate_output(id: &str, v: &Value) -> Result<()> {
    let limit = match id {
        "search.occurrences" | "search.runs" | "ranges.expand" => 256,
        "paths.ancestors"
        | "sequences.interleave"
        | "encoding.hex_decode"
        | "sets.symmetric"
        | "messages.hashtags"
        | "messages.wrap"
        | "transactions.settle" => 128,
        "ranges.missing" => 257,
        _ => 64,
    };
    representation(v, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_shape_and_published_bounds_are_rejected_before_oracle_indexing() {
        for (id, input) in [
            ("money.format", json!(-100000001)),
            ("matrices.transpose", json!([[1, 2], [3]])),
            (
                "graph.reachable",
                json!({"n":2,"start":0,"edges":[[0,1],[0,1]]}),
            ),
            ("rational.add", json!([[1, 0], [2, 3]])),
            ("messages.wrap", json!({"text":"longword","width":2})),
            ("search.occurrences", json!({"text":"abc","pattern":""})),
            ("ledger.sum", json!([10001])),
        ] {
            assert!(validate_input(id, &input).is_err(), "{id}");
        }
    }
}
