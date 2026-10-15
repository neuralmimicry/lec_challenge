//! Private, locally authored Rust contracts. Source packs are lineage groups, not
//! a claim of independent human authorship or unseen underlying algorithms.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::Path,
};
mod domain;

pub const SHARED_DOMAIN: &str = "Shared domain: Numbers are signed 64-bit integers; all required intermediate and final arithmetic fits i64. Arrays contain at most 64 items and strings are ASCII with at most 256 bytes, unless an explicit task exception says otherwise. Whitespace means space, TAB, LF, FF or CR. JSON booleans and null retain their normal types.";
/// Shared protocol text carries no source identity and must not inflate the
/// lexical overlap used to screen independent task contracts.
pub fn substantive_specification(text: &str) -> &str {
    text.split_once("\nShared domain:")
        .map(|(spec, _)| spec)
        .unwrap_or(text)
}
fn output_exception(id: &str) -> &str {
    match id {
        x if x.starts_with("graph.") => " Graph vertex counts are at most 64.",
        "money.split" => " The number of shares is at most 64.",
        "search.occurrences" | "search.runs" => " The returned array may contain up to 256 items.",
        "paths.ancestors"
        | "sequences.interleave"
        | "encoding.hex_decode"
        | "sets.symmetric"
        | "messages.hashtags"
        | "messages.wrap"
        | "transactions.settle" => " The returned array may contain up to 128 items.",
        "ranges.expand" => " The returned array may contain up to 256 integers.",
        "ranges.missing" => " The returned array may contain up to 257 integers.",
        _ => "",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Development,
    HeldOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: Value,
    pub expected: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub source_id: String,
    pub lineage_id: String,
    pub split: Split,
    pub specification: String,
    pub specification_sha256: String,
    pub oracle_sha256: String,
    pub public_examples: Vec<Example>,
    pub provenance: String,
    pub licence: String,
}
impl Task {
    /// Only this public material may enter a provider prompt.
    pub fn prompt(&self) -> String {
        format!(
            "Write Rust 2024 code defining `pub fn solve(input: serde_json::Value) -> serde_json::Value`. Use only std and serde_json. Return only Rust source, without a main function or Markdown. A trusted wrapper reads one JSON line and writes the returned JSON value. Do not read files, use the network, spawn processes or access tests. Inputs satisfy the stated contract. Contract: {}\nPublic examples: {}",
            self.specification,
            serde_json::to_string(&self.public_examples).expect("examples serialise")
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub input: Value,
    pub expected: Value,
    pub label: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusSummary {
    pub tasks: usize,
    pub train: usize,
    pub development: usize,
    pub held_out: usize,
    pub sources: usize,
    pub corpus_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeldoutEntry {
    pub task_id: String,
    pub source_id: String,
    pub lineage_id: String,
    pub specification: String,
    pub specification_sha256: String,
}

pub fn hash(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

// Specifications and the public expected values below were written separately
// from the executable oracle. Tests compare all 96 pairs and private properties.
pub fn corpus() -> Vec<Task> {
    type Definition<'a> = (&'a str, &'a str, Value, Value);
    type SourcePack<'a> = (&'a str, [Definition<'a>; 4]);
    let packs: &[SourcePack<'_>] = &[
        (
            "ledger",
            [
                (
                    "sum",
                    "Input is an array of integers in [-10000,10000], length <=64. Return its sum; the empty sum is zero.",
                    json!([2, -4, 8]),
                    json!(6),
                ),
                (
                    "prefix",
                    "Input is an integer array. Return inclusive prefix sums; empty input returns []. Bounds: length <=64 and absolute values <=10000.",
                    json!([2, -4, 8]),
                    json!([2, -2, 6]),
                ),
                (
                    "differences",
                    "Input is an integer array. Return each next value minus the previous value; lengths zero and one return [].",
                    json!([2, -4, 8]),
                    json!([-6, 12]),
                ),
                (
                    "bounds",
                    "Input is an integer array. Return [minimum,maximum], or null for an empty array.",
                    json!([2, -4, 8]),
                    json!([-4, 8]),
                ),
            ],
        ),
        (
            "stock",
            [
                (
                    "counts",
                    "Input is an array of ASCII strings. Return an object mapping every distinct string to its frequency.",
                    json!(["b", "a", "b"]),
                    json!({"a":1,"b":2}),
                ),
                (
                    "stable_unique",
                    "Input is an array of ASCII strings. Return first occurrences only, preserving their original order.",
                    json!(["b", "a", "b"]),
                    json!(["b", "a"]),
                ),
                (
                    "duplicates",
                    "Input is an array of ASCII strings. Return distinct strings occurring at least twice, in lexicographic order.",
                    json!(["b", "a", "b"]),
                    json!(["b"]),
                ),
                (
                    "mode",
                    "Input is an array of ASCII strings. Return the most frequent string, lexicographically smallest on ties; return null if empty.",
                    json!(["b", "a", "b"]),
                    json!("b"),
                ),
            ],
        ),
        (
            "text",
            [
                (
                    "lower_trim",
                    "Input is an ASCII string. Trim leading/trailing ASCII whitespace, then lowercase ASCII letters, preserving internal whitespace.",
                    json!("  Hi THERE \t"),
                    json!("hi there"),
                ),
                (
                    "squash",
                    "Input is an ASCII string. Split on ASCII whitespace and join words with one space, removing edge whitespace.",
                    json!("  Hi\t THERE \n"),
                    json!("Hi THERE"),
                ),
                (
                    "slug",
                    "Input is an ASCII string. Lowercase letters; replace each maximal run of non-alphanumeric characters by one hyphen; remove edge hyphens.",
                    json!(" Hello, WORLD! "),
                    json!("hello-world"),
                ),
                (
                    "reverse_words",
                    "Input is an ASCII string. Reverse the sequence of ASCII-whitespace-delimited words and join with one space.",
                    json!(" one  two\tthree "),
                    json!("three two one"),
                ),
            ],
        ),
        (
            "intervals",
            [
                (
                    "merge",
                    "Input is an array of [start,end] integer half-open intervals with start<end. Sort and merge overlapping or touching intervals; return sorted disjoint intervals.",
                    json!([[3, 5], [1, 3], [8, 9]]),
                    json!([[1, 5], [8, 9]]),
                ),
                (
                    "coverage",
                    "Input is an array of integer half-open intervals [start,end], start<end. Return total length of their union.",
                    json!([[3, 5], [1, 3], [8, 9]]),
                    json!(5),
                ),
                (
                    "largest_gap",
                    "Input is an array of integer half-open intervals, start<end. Merge touching/overlapping intervals and return the largest internal uncovered gap, or zero with fewer than two merged intervals.",
                    json!([[3, 5], [1, 3], [8, 9]]),
                    json!(3),
                ),
                (
                    "peak",
                    "Input is an array of integer half-open intervals, start<end. Return maximum simultaneous overlap; an end at t does not overlap a start at t.",
                    json!([[1, 3], [2, 4], [3, 5]]),
                    json!(2),
                ),
            ],
        ),
        (
            "graph",
            [
                (
                    "reachable",
                    "Input {n,start,edges} describes a directed graph on 0..n-1, n>=1, edges unique [from,to]. Return all nodes reachable from start, including start, sorted ascending.",
                    json!({"n":4,"start":0,"edges":[[0,1],[1,2]]}),
                    json!([0, 1, 2]),
                ),
                (
                    "distance",
                    "Input {n,start,target,edges} is a directed unweighted graph, n>=1. Return shortest edge distance start->target, or -1 if unreachable.",
                    json!({"n":4,"start":0,"target":2,"edges":[[0,1],[1,2]]}),
                    json!(2),
                ),
                (
                    "indegrees",
                    "Input {n,edges} is a directed graph with unique edges on 0..n-1. Return indegree of each node in index order, including self-loops.",
                    json!({"n":4,"edges":[[0,1],[1,2]]}),
                    json!([0, 1, 1, 0]),
                ),
                (
                    "topological",
                    "Input {n,edges} is a directed graph. Repeatedly remove the smallest currently zero-indegree node. Return that topological order, or null if cyclic.",
                    json!({"n":4,"edges":[[0,1],[1,2]]}),
                    json!([0, 1, 2, 3]),
                ),
            ],
        ),
        (
            "validation",
            [
                (
                    "luhn",
                    "Input is an ASCII string. Return true iff it contains at least two decimal digits, no other characters, and passes the Luhn checksum (rightmost digit undoubled).",
                    json!("79927398713"),
                    json!(true),
                ),
                (
                    "ipv4",
                    "Input is an ASCII string. Return true iff exactly four decimal components separated by dots each have value 0..255 and no leading zero unless the component is exactly 0.",
                    json!("192.168.0.1"),
                    json!(true),
                ),
                (
                    "brackets",
                    "Input is an ASCII string. Return whether (), [] and {} are properly nested, ignoring every other character.",
                    json!("a{b[c](d)}"),
                    json!(true),
                ),
                (
                    "identifier",
                    "Input is an ASCII string. Return whether it matches [A-Za-z_][A-Za-z0-9_]*. Do not reject language keywords.",
                    json!("_name2"),
                    json!(true),
                ),
            ],
        ),
        (
            "matrices",
            [
                (
                    "transpose",
                    "Input is a rectangular integer matrix with at least one row and one column. Return its transpose.",
                    json!([[1, 2, 3], [4, 5, 6]]),
                    json!([[1, 4], [2, 5], [3, 6]]),
                ),
                (
                    "row_sums",
                    "Input is a rectangular integer matrix with at least one row and column. Return each row's sum.",
                    json!([[1, 2, 3], [4, 5, 6]]),
                    json!([6, 15]),
                ),
                (
                    "rotate",
                    "Input is a rectangular integer matrix with at least one row and column. Return the matrix rotated clockwise by 90 degrees.",
                    json!([[1, 2, 3], [4, 5, 6]]),
                    json!([[4, 1], [5, 2], [6, 3]]),
                ),
                (
                    "diagonals",
                    "Input is a nonempty square integer matrix. Return [main diagonal sum,anti-diagonal sum]. A centre cell contributes once to each sum.",
                    json!([[1, 2], [3, 4]]),
                    json!([5, 5]),
                ),
            ],
        ),
        (
            "search",
            [
                (
                    "occurrences",
                    "Input {text,pattern} contains ASCII strings with nonempty pattern. Return zero-based starting indices of every occurrence, including overlaps.",
                    json!({"text":"ababa","pattern":"aba"}),
                    json!([0, 2]),
                ),
                (
                    "common_prefix",
                    "Input is an array of ASCII strings. Return their longest common prefix, or the empty string for an empty array.",
                    json!(["flower", "flow", "flight"]),
                    json!("fl"),
                ),
                (
                    "runs",
                    "Input is an ASCII string. Return consecutive character runs as [[one-character string,count],...]. Empty input returns [].",
                    json!("aaabbcaa"),
                    json!([["a", 3], ["b", 2], ["c", 1], ["a", 2]]),
                ),
                (
                    "anagram",
                    "Input is [a,b], two ASCII strings. Ignore ASCII whitespace and letter case, keep punctuation; return whether their remaining character multisets agree.",
                    json!(["Dormitory", "Dirty room"]),
                    json!(true),
                ),
            ],
        ),
        (
            "paths",
            [
                (
                    "normalise",
                    "Input is an absolute POSIX path string. Collapse repeated / and . components; .. removes a preceding component but cannot go above root. Return canonical absolute path without trailing / except root. No filesystem access.",
                    json!("/a//b/../c/./"),
                    json!("/a/c"),
                ),
                (
                    "basename",
                    "Input is an absolute POSIX path. Ignore trailing slashes; return the last nonempty raw component, or / for root. Do not resolve . or ...",
                    json!("/a/b///"),
                    json!("b"),
                ),
                (
                    "extension",
                    "Input is a filename without /. Return lowercase ASCII suffix after the final dot, except a sole leading dot or a trailing dot yields empty string.",
                    json!("archive.TAR.GZ"),
                    json!("gz"),
                ),
                (
                    "ancestors",
                    "Input is a canonical absolute POSIX path (no . or ..). Return proper ancestor paths starting at / and ending at its parent; root has no ancestors.",
                    json!("/a/b/c"),
                    json!(["/", "/a", "/a/b"]),
                ),
            ],
        ),
        (
            "sequences",
            [
                (
                    "interleave",
                    "Input is [a,b], integer arrays. Alternate elements starting with a; append the remaining tail when one array ends.",
                    json!([[1, 2, 3], [8, 9]]),
                    json!([1, 8, 2, 9, 3]),
                ),
                (
                    "zip_longest",
                    "Input is [a,b], integer arrays. Return pairs by position, using null for a missing side, up to the longer length.",
                    json!([[1, 2], [9]]),
                    json!([[1, 9], [2, null]]),
                ),
                (
                    "chunks",
                    "Input {values,size} has an integer array and positive integer size. Return consecutive chunks of at most size elements, preserving order.",
                    json!({"values":[1,2,3,4,5],"size":2}),
                    json!([[1, 2], [3, 4], [5]]),
                ),
                (
                    "move_zeroes",
                    "Input is an integer array. Move all zeros to the end, preserving the order of nonzero elements and the length.",
                    json!([0, 1, 0, -2, 3]),
                    json!([1, -2, 3, 0, 0]),
                ),
            ],
        ),
        (
            "calendar",
            [
                (
                    "leap",
                    "Input is a Gregorian year integer in 1..9999. Return whether it is a leap year (divisible by four except centuries not divisible by 400).",
                    json!(2000),
                    json!(true),
                ),
                (
                    "month_days",
                    "Input is [year,month] with year 1..9999 and month 1..12. Return Gregorian days in that month.",
                    json!([2024, 2]),
                    json!(29),
                ),
                (
                    "ordinal",
                    "Input is [year,month,day], a valid Gregorian date in years 1..9999. Return its one-based day of year.",
                    json!([2024, 3, 1]),
                    json!(61),
                ),
                (
                    "valid_date",
                    "Input is [year,month,day] integers. Return whether it is a valid Gregorian date, restricted to years 1..9999.",
                    json!([2023, 2, 29]),
                    json!(false),
                ),
            ],
        ),
        (
            "bits",
            [
                (
                    "popcount",
                    "Input is an integer in 0..4294967295. Return the number of set bits in its unsigned 32-bit representation.",
                    json!(13),
                    json!(3),
                ),
                (
                    "hamming",
                    "Input is [a,b], unsigned 32-bit integers. Return the number of differing bit positions.",
                    json!([9, 3]),
                    json!(2),
                ),
                (
                    "parity",
                    "Input is an unsigned 32-bit integer. Return 0 for an even number of set bits and 1 for an odd number.",
                    json!(13),
                    json!(1),
                ),
                (
                    "lowest",
                    "Input is an unsigned 32-bit integer. Return the zero-based position of its lowest set bit, or -1 for zero.",
                    json!(40),
                    json!(3),
                ),
            ],
        ),
        (
            "money",
            [
                (
                    "parse",
                    "Input is an ASCII decimal monetary string. Accept optional leading minus, one or more digits, and optional dot followed by exactly two digits. Return signed cents, or null for invalid syntax; magnitudes <=1000000 pounds.",
                    json!("-12.30"),
                    json!(-1230),
                ),
                (
                    "format",
                    "Input is signed cents, magnitude <=100000000. Return pounds with optional minus and exactly two fractional digits.",
                    json!(-1230),
                    json!("-12.30"),
                ),
                (
                    "split",
                    "Input [cents,n] has nonnegative cents and n>=1. Split into n integer shares differing by at most one, with larger shares first.",
                    json!([10, 3]),
                    json!([4, 3, 3]),
                ),
                (
                    "basket",
                    "Input is an array of [unit_price_cents,quantity], both nonnegative integers. Return total cents without tax.",
                    json!([[120, 2], [99, 3]]),
                    json!(537),
                ),
            ],
        ),
        (
            "dispatch",
            [
                (
                    "finish_order",
                    "Input is an array of [arrival,duration], nonnegative integers. A single server processes input order, starting each job at max(previous finish,arrival). Return finish times.",
                    json!([[0, 3], [1, 2], [8, 1]]),
                    json!([3, 5, 9]),
                ),
                (
                    "waiting",
                    "Input is an array of [arrival,duration] processed in input order by one server, initially free at zero. Return total time jobs wait before service.",
                    json!([[0, 3], [1, 2], [8, 1]]),
                    json!(2),
                ),
                (
                    "round_robin",
                    "Input {jobs,quantum} has nonnegative integer work amounts and positive quantum. Cycle job indices, giving up to quantum work each turn; return indices in order of completion, initially zero-work jobs first in index order.",
                    json!({"jobs":[3,1,2],"quantum":2}),
                    json!([1, 2, 0]),
                ),
                (
                    "priority",
                    "Input is an array of [priority,id] integer pairs with unique ids. Return ids sorted by decreasing priority, with increasing id on ties.",
                    json!([[2, 7], [3, 8], [2, 1]]),
                    json!([8, 1, 7]),
                ),
            ],
        ),
        (
            "encoding",
            [
                (
                    "hex_encode",
                    "Input is an array of bytes (integers 0..255). Return concatenated two-digit lowercase hexadecimal.",
                    json!([0, 15, 255]),
                    json!("000fff"),
                ),
                (
                    "hex_decode",
                    "Input is an ASCII string. Decode even-length hexadecimal accepting upper/lowercase; return byte array, or null for invalid digits or odd length.",
                    json!("00aF"),
                    json!([0, 175]),
                ),
                (
                    "caesar",
                    "Input {text,shift} has ASCII text and integer shift. Rotate lowercase a..z only modulo 26; preserve all other characters.",
                    json!({"text":"xyz A!","shift":3}),
                    json!("abc A!"),
                ),
                (
                    "rot13",
                    "Input is an ASCII string. Apply ROT13 to upper/lowercase letters and preserve other characters.",
                    json!("Hello, Z!"),
                    json!("Uryyb, M!"),
                ),
            ],
        ),
        (
            "tables",
            [
                (
                    "sort_rows",
                    "Input is an array of [key,value] integer rows. Sort ascending by key and then value; keep duplicates.",
                    json!([[2, 9], [1, 4], [2, 3]]),
                    json!([[1, 4], [2, 3], [2, 9]]),
                ),
                (
                    "last_by_key",
                    "Input is an array of [key,value] integer rows. Keep the last value per key and return [key,value] rows sorted by key.",
                    json!([[2, 9], [1, 4], [2, 3]]),
                    json!([[1, 4], [2, 3]]),
                ),
                (
                    "group_sums",
                    "Input is an array of [key,value] integer rows. Sum values per key and return rows sorted by key.",
                    json!([[2, 9], [1, 4], [2, 3]]),
                    json!([[1, 4], [2, 12]]),
                ),
                (
                    "inner_join",
                    "Input [left,right] has arrays of [key,value] integer rows, each side with unique keys. Return [key,left_value,right_value] for common keys in ascending key order.",
                    json!([[[2, 9], [1, 4]], [[2, 3], [3, 8]]]),
                    json!([[2, 9, 3]]),
                ),
            ],
        ),
        (
            "geometry",
            [
                (
                    "box",
                    "Input is an array of integer [x,y] points. Return [min_x,min_y,max_x,max_y], or null when empty.",
                    json!([[2, 3], [-1, 7], [4, 0]]),
                    json!([-1, 0, 4, 7]),
                ),
                (
                    "manhattan",
                    "Input is an array of integer [x,y] points. Return each point's Manhattan distance from the origin in input order.",
                    json!([[2, 3], [-1, 7], [4, 0]]),
                    json!([5, 8, 4]),
                ),
                (
                    "area2",
                    "Input is an array of integer polygon vertices in boundary order, at least three vertices. Return signed twice-area by shoelace; counter-clockwise is positive.",
                    json!([[0, 0], [3, 0], [0, 2]]),
                    json!(6),
                ),
                (
                    "quadrants",
                    "Input is an array of integer [x,y] points. Return counts [QI,QII,QIII,QIV,axis], treating either coordinate zero as axis.",
                    json!([[1, 2], [-1, 2], [-1, -2], [1, -2], [0, 3]]),
                    json!([1, 1, 1, 1, 1]),
                ),
            ],
        ),
        (
            "statistics",
            [
                (
                    "median2",
                    "Input is a nonempty integer array. Return twice the mathematical median; for even length use the sum of the two central sorted values.",
                    json!([7, 1, 4, 2]),
                    json!(6),
                ),
                (
                    "mode",
                    "Input is a nonempty integer array. Return its most frequent value, choosing the smallest on ties.",
                    json!([7, 1, 7, 1]),
                    json!(1),
                ),
                (
                    "ranks",
                    "Input is an integer array. Return dense ascending ranks starting at one for each original element; equal values share a rank.",
                    json!([40, 10, 40, 20]),
                    json!([3, 1, 3, 2]),
                ),
                (
                    "quantile",
                    "Input {values,p} has nonempty integer values and integer percentile p in 1..100. Return the nearest-rank quantile at sorted index ceil(p*n/100)-1.",
                    json!({"values":[4,1,3,2],"p":75}),
                    json!(3),
                ),
            ],
        ),
        (
            "sets",
            [
                (
                    "intersection",
                    "Input [a,b] has integer arrays representing sets (duplicates permitted). Return their intersection sorted ascending with no duplicates.",
                    json!([[3, 1, 1, 2], [2, 3, 4]]),
                    json!([2, 3]),
                ),
                (
                    "symmetric",
                    "Input [a,b] has integer arrays representing sets. Return values in exactly one set, sorted ascending without duplicates.",
                    json!([[3, 1, 1, 2], [2, 3, 4]]),
                    json!([1, 4]),
                ),
                (
                    "jaccard",
                    "Input [a,b] has integer arrays representing sets. Return reduced integer fraction [intersection_size,union_size]; two empty sets have similarity [1,1].",
                    json!([[1, 2], [2, 3]]),
                    json!([1, 3]),
                ),
                (
                    "subset",
                    "Input [a,b] has integer arrays representing sets. Return whether every distinct value in a occurs in b.",
                    json!([[1, 1], [1, 2]]),
                    json!(true),
                ),
            ],
        ),
        (
            "rational",
            [
                (
                    "reduce",
                    "Input [numerator,denominator] has denominator nonzero. Return reduced fraction with positive denominator; zero is [0,1]. Magnitudes <=10000.",
                    json!([6, -8]),
                    json!([-3, 4]),
                ),
                (
                    "add",
                    "Input [[a,b],[c,d]] contains two fractions with nonzero denominators. Return their sum reduced with positive denominator. Magnitudes <=1000.",
                    json!([[1, 2], [1, 3]]),
                    json!([5, 6]),
                ),
                (
                    "compare",
                    "Input [[a,b],[c,d]] contains fractions with nonzero denominators. Return -1,0,1 according as the first is less/equal/greater. Magnitudes <=1000.",
                    json!([[1, -2], [-2, 3]]),
                    json!(1),
                ),
                (
                    "reciprocal",
                    "Input [numerator,denominator] with denominator nonzero. Return the reciprocal in reduced form with positive denominator, or null for zero numerator.",
                    json!([6, -8]),
                    json!([-4, 3]),
                ),
            ],
        ),
        (
            "messages",
            [
                (
                    "headers",
                    "Input is an array of ASCII header lines. Every line has a colon. Split at the first colon; trim ASCII whitespace; lowercase name; return object of arrays of trimmed values in input order per name.",
                    json!(["X-ID: a", "x-id:b", "Y: c:d"]),
                    json!({"x-id":["a","b"],"y":["c:d"]}),
                ),
                (
                    "address",
                    "Input is an ASCII string with exactly one @ and nonempty sides. Return [unchanged local part,lowercase domain].",
                    json!("Paul@EXAMPLE.COM"),
                    json!(["Paul", "example.com"]),
                ),
                (
                    "hashtags",
                    "Input is ASCII text. A hashtag is # followed by one or more ASCII letters/digits/underscores. Return lowercase hashtag bodies in first-seen order, without duplicates.",
                    json!("Hi #Rust #AI #rust #!"),
                    json!(["rust", "ai"]),
                ),
                (
                    "wrap",
                    "Input {text,width} has ASCII-whitespace-separated words each no longer than positive width. Greedily fit words on lines with single spaces. Return an array of lines; blank text yields [].",
                    json!({"text":"one two three four","width":7}),
                    json!(["one two", "three", "four"]),
                ),
            ],
        ),
        (
            "automata",
            [
                (
                    "divisible3",
                    "Input is a nonempty binary digit string (leading zeros allowed). Return whether its nonnegative value is divisible by three, without fixed-width overflow.",
                    json!("110"),
                    json!(true),
                ),
                (
                    "balanced_prefix",
                    "Input is a string containing only ( and ). Return the length of the longest prefix that is well-formed parentheses, or zero; once a prefix has negative balance no longer prefix is valid.",
                    json!("()(())(()"),
                    json!(6),
                ),
                (
                    "classes",
                    "Input is an ASCII string. Count [uppercase,lowercase,digits,ASCII_whitespace,other]. Every character contributes to exactly one category.",
                    json!("Aa1 !"),
                    json!([1, 1, 1, 1, 1]),
                ),
                (
                    "transitions",
                    "Input is a binary string. Return [count_00,count_01,count_10,count_11] across adjacent pairs, including overlaps.",
                    json!("00110"),
                    json!([1, 1, 1, 1]),
                ),
            ],
        ),
        (
            "transactions",
            [
                (
                    "balance",
                    "Input {initial,deltas} has integer initial balance and integer transaction deltas. Return final balance.",
                    json!({"initial":10,"deltas":[-3,5,-8]}),
                    json!(4),
                ),
                (
                    "running_min",
                    "Input {initial,deltas} contains an initial balance and deltas. Return minimum balance observed, including the initial state.",
                    json!({"initial":10,"deltas":[-3,5,-8]}),
                    json!(4),
                ),
                (
                    "first_overdraft",
                    "Input {initial,deltas} has nonnegative initial and integer deltas. Return zero-based index of first transaction after which balance is negative, or -1.",
                    json!({"initial":2,"deltas":[-1,-2,5]}),
                    json!(1),
                ),
                (
                    "settle",
                    "Input is an array of [from,to,amount] with integer account IDs and nonnegative amount. Return sorted [account,net_change] including every mentioned account; subtract from sender and add to recipient.",
                    json!([[1, 2, 5], [2, 3, 2]]),
                    json!([[1, -5], [2, 3], [3, 2]]),
                ),
            ],
        ),
        (
            "ranges",
            [
                (
                    "expand",
                    "Input is an array of inclusive integer [start,end] ranges, start<=end and total expansion <=256. Return sorted union of represented integers without duplicates.",
                    json!([[3, 5], [1, 3]]),
                    json!([1, 2, 3, 4, 5]),
                ),
                (
                    "compress",
                    "Input is an integer array. Sort/deduplicate and return maximal inclusive consecutive ranges [start,end].",
                    json!([4, 2, 3, 3, 8]),
                    json!([[2, 4], [8, 8]]),
                ),
                (
                    "missing",
                    "Input {values,low,high} has integers with low<=high and high-low<=256. Return every integer in inclusive bounds absent from values, ascending.",
                    json!({"values":[2,4,9],"low":1,"high":5}),
                    json!([1, 3, 5]),
                ),
                (
                    "kth",
                    "Input {values,k} has an integer array and positive one-based k. Return the kth smallest distinct value, or null when too few distinct values.",
                    json!({"values":[7,2,2,5],"k":2}),
                    json!(5),
                ),
            ],
        ),
    ];
    let mut oracle_digest = Sha256::new();
    oracle_digest.update(include_bytes!("tasks.rs"));
    oracle_digest.update(include_bytes!("tasks/domain.rs"));
    let oracle_hash = format!("{:x}", oracle_digest.finalize());
    packs.iter().enumerate().flat_map(|(i,(source, definitions))| {
        let oracle_hash = oracle_hash.clone();
        definitions.iter().map(move |(name, spec, input, expected)| {let id=format!("{source}.{name}");let specification=format!("{spec}{}\n{SHARED_DOMAIN}",output_exception(&id));Task {
            id,source_id:format!("private-{source}-v1"),lineage_id:format!("local-contract-pack-{i:02}"),
            split:if i<12 {Split::Train} else if i<16 {Split::Development} else {Split::HeldOut},
            specification_sha256:hash(specification.as_bytes()),specification,oracle_sha256:oracle_hash.clone(),
            public_examples:vec![Example {input:input.clone(),expected:expected.clone()}],
            provenance:"Locally AI-authored synthetic contract, 15 October 2026; shared author; novel specification lineage, not proof of algorithm novelty or independent human authorship".into(),licence:"Apache-2.0".into(),
        }})
    }).collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("validated array")
        .iter()
        .map(|n| n.as_i64().expect("validated integer"))
        .collect()
}
fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("validated array")
        .iter()
        .map(|n| n.as_str().expect("validated string").to_string())
        .collect()
}
fn rows(v: &Value) -> Vec<Vec<i64>> {
    v.as_array()
        .expect("validated matrix")
        .iter()
        .map(ints)
        .collect()
}
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
fn fraction(mut a: i64, mut b: i64) -> Value {
    if a == 0 {
        return json!([0, 1]);
    }
    if b < 0 {
        a = -a;
        b = -b;
    }
    let g = gcd(a, b);
    json!([a / g, b / g])
}
fn merged(v: &Value) -> Vec<Vec<i64>> {
    let mut a = rows(v);
    a.sort();
    let mut out: Vec<Vec<i64>> = vec![];
    for p in a {
        if let Some(last) = out.last_mut().filter(|last| p[0] <= last[1]) {
            last[1] = last[1].max(p[1]);
            continue;
        }
        out.push(p);
    }
    out
}
fn leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}
fn days(y: i64, m: i64) -> i64 {
    match m {
        2 => {
            if leap(y) {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    }
}

/// Host-only oracle. It is never copied into a candidate container or prompt.
pub fn oracle(id: &str, v: &Value) -> Result<Value> {
    if crate::teacher_search::qualification::is_task(id) {
        return crate::teacher_search::qualification::oracle(id, v);
    }
    domain::validate_input(id, v)?;
    let out = match id {
        "ledger.sum" => json!(ints(v).iter().sum::<i64>()),
        "ledger.prefix" => {
            let mut s = 0;
            json!(
                ints(v)
                    .iter()
                    .map(|x| {
                        s += x;
                        s
                    })
                    .collect::<Vec<_>>()
            )
        }
        "ledger.differences" => json!(ints(v).windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>()),
        "ledger.bounds" => {
            let a = ints(v);
            if a.is_empty() {
                Value::Null
            } else {
                json!([a.iter().min(), a.iter().max()])
            }
        }
        "stock.counts" | "stock.duplicates" | "stock.mode" => {
            let mut m: BTreeMap<String, usize> = BTreeMap::new();
            for s in strs(v) {
                *m.entry(s).or_default() += 1;
            }
            match id {
                "stock.counts" => json!(m),
                "stock.duplicates" => json!(
                    m.into_iter()
                        .filter_map(|(s, n)| (n > 1).then_some(s))
                        .collect::<Vec<_>>()
                ),
                _ => {
                    let max = m.values().max().copied().unwrap_or(0);
                    m.iter()
                        .find_map(|(s, n)| (*n == max).then(|| json!(s)))
                        .unwrap_or(Value::Null)
                }
            }
        }
        "stock.stable_unique" => {
            let mut seen = BTreeSet::new();
            json!(
                strs(v)
                    .into_iter()
                    .filter(|s| seen.insert(s.clone()))
                    .collect::<Vec<_>>()
            )
        }
        "text.lower_trim" => json!(
            v.as_str()
                .unwrap()
                .trim_matches(|c: char| c.is_ascii_whitespace())
                .to_ascii_lowercase()
        ),
        "text.squash" => json!(
            v.as_str()
                .unwrap()
                .split_ascii_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        ),
        "text.reverse_words" => json!(
            v.as_str()
                .unwrap()
                .split_ascii_whitespace()
                .rev()
                .collect::<Vec<_>>()
                .join(" ")
        ),
        "text.slug" => {
            let mut s = String::new();
            let mut pending = false;
            for c in v.as_str().unwrap().chars() {
                if c.is_ascii_alphanumeric() {
                    if pending && !s.is_empty() {
                        s.push('-');
                    }
                    pending = false;
                    s.push(c.to_ascii_lowercase());
                } else {
                    pending = true;
                }
            }
            json!(s)
        }
        "intervals.merge" => json!(merged(v)),
        "intervals.coverage" => json!(merged(v).iter().map(|r| r[1] - r[0]).sum::<i64>()),
        "intervals.largest_gap" => json!(
            merged(v)
                .windows(2)
                .map(|w| w[1][0] - w[0][1])
                .max()
                .unwrap_or(0)
        ),
        "intervals.peak" => {
            let mut events = vec![];
            for r in rows(v) {
                events.push((r[0], 1));
                events.push((r[1], -1));
            }
            events.sort();
            let (mut n, mut peak) = (0, 0);
            for (_, d) in events {
                n += d;
                peak = peak.max(n);
            }
            json!(peak)
        }
        x if x.starts_with("graph.") => {
            let n = v["n"].as_u64().unwrap() as usize;
            let mut adj = vec![vec![]; n];
            let mut deg = vec![0usize; n];
            for e in rows(&v["edges"]) {
                adj[e[0] as usize].push(e[1] as usize);
                deg[e[1] as usize] += 1;
            }
            match x {
                "graph.indegrees" => json!(deg),
                "graph.topological" => {
                    let mut ready: BTreeSet<usize> = (0..n).filter(|i| deg[*i] == 0).collect();
                    let mut order = vec![];
                    while let Some(i) = ready.pop_first() {
                        order.push(i);
                        for j in &adj[i] {
                            deg[*j] -= 1;
                            if deg[*j] == 0 {
                                ready.insert(*j);
                            }
                        }
                    }
                    if order.len() == n {
                        json!(order)
                    } else {
                        Value::Null
                    }
                }
                _ => {
                    let start = v["start"].as_u64().unwrap() as usize;
                    let mut dist = vec![-1i64; n];
                    dist[start] = 0;
                    let mut q = VecDeque::from([start]);
                    while let Some(i) = q.pop_front() {
                        for j in &adj[i] {
                            if dist[*j] < 0 {
                                dist[*j] = dist[i] + 1;
                                q.push_back(*j);
                            }
                        }
                    }
                    if x == "graph.reachable" {
                        json!((0..n).filter(|i| dist[*i] >= 0).collect::<Vec<_>>())
                    } else {
                        json!(dist[v["target"].as_u64().unwrap() as usize])
                    }
                }
            }
        }
        "validation.luhn" => {
            let s = v.as_str().unwrap();
            let valid = s.len() >= 2 && s.bytes().all(|b| b.is_ascii_digit());
            let sum = s
                .bytes()
                .rev()
                .enumerate()
                .map(|(i, b)| {
                    let mut n = (b as i64) - 48;
                    if i % 2 == 1 {
                        n *= 2;
                        if n > 9 {
                            n -= 9;
                        }
                    }
                    n
                })
                .sum::<i64>();
            json!(valid && sum % 10 == 0)
        }
        "validation.ipv4" => {
            let p: Vec<_> = v.as_str().unwrap().split('.').collect();
            json!(
                p.len() == 4
                    && p.iter().all(|s| !s.is_empty()
                        && s.bytes().all(|b| b.is_ascii_digit())
                        && (s.len() == 1 || !s.starts_with('0'))
                        && s.parse::<u16>().map(|n| n <= 255).unwrap_or(false))
            )
        }
        "validation.brackets" => {
            let mut stack = vec![];
            let mut good = true;
            for c in v.as_str().unwrap().chars() {
                match c {
                    '(' | '[' | '{' => stack.push(c),
                    ')' | ']' | '}'
                        if stack.pop()
                            != Some(match c {
                                ')' => '(',
                                ']' => '[',
                                _ => '{',
                            }) =>
                    {
                        good = false;
                        break;
                    }
                    _ => {}
                }
            }
            json!(good && stack.is_empty())
        }
        "validation.identifier" => {
            let s = v.as_str().unwrap().as_bytes();
            json!(
                !s.is_empty()
                    && (s[0].is_ascii_alphabetic() || s[0] == b'_')
                    && s.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_')
            )
        }
        x if x.starts_with("matrices.") => {
            let a = rows(v);
            let (n, m) = (a.len(), a[0].len());
            match x {
                "matrices.transpose" => json!(
                    (0..m)
                        .map(|j| (0..n).map(|i| a[i][j]).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                ),
                "matrices.row_sums" => {
                    json!(a.iter().map(|r| r.iter().sum::<i64>()).collect::<Vec<_>>())
                }
                "matrices.rotate" => json!(
                    (0..m)
                        .map(|j| (0..n).rev().map(|i| a[i][j]).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                ),
                _ => json!([
                    (0..n).map(|i| a[i][i]).sum::<i64>(),
                    (0..n).map(|i| a[i][n - 1 - i]).sum::<i64>()
                ]),
            }
        }
        "search.occurrences" => {
            let t = v["text"].as_str().unwrap();
            let p = v["pattern"].as_str().unwrap();
            json!(
                (0..=t.len())
                    .filter(|i| t[*i..].starts_with(p))
                    .collect::<Vec<_>>()
            )
        }
        "search.common_prefix" => {
            let a = strs(v);
            let mut p = a.first().cloned().unwrap_or_default();
            for s in a.iter().skip(1) {
                while !s.starts_with(&p) {
                    p.pop();
                }
            }
            json!(p)
        }
        "search.runs" => {
            let mut out: Vec<(String, usize)> = vec![];
            for c in v.as_str().unwrap().chars() {
                let s = c.to_string();
                if let Some((_, n)) = out.last_mut().filter(|(last, _)| *last == s) {
                    *n += 1;
                    continue;
                }
                out.push((s, 1));
            }
            json!(out)
        }
        "search.anagram" => {
            let norm = |s: &str| {
                let mut b = s
                    .bytes()
                    .filter(|b| !b.is_ascii_whitespace())
                    .map(|b| b.to_ascii_lowercase())
                    .collect::<Vec<_>>();
                b.sort();
                b
            };
            json!(norm(v[0].as_str().unwrap()) == norm(v[1].as_str().unwrap()))
        }
        "paths.normalise" => {
            let mut a = vec![];
            for p in v.as_str().unwrap().split('/') {
                match p {
                    "" | "." => {}
                    ".." => {
                        a.pop();
                    }
                    _ => a.push(p),
                }
            }
            json!(format!("/{}", a.join("/")))
        }
        "paths.basename" => json!(
            v.as_str()
                .unwrap()
                .split('/')
                .rfind(|s| !s.is_empty())
                .unwrap_or("/")
        ),
        "paths.extension" => {
            let s = v.as_str().unwrap();
            json!(
                s.rfind('.')
                    .filter(|i| *i > 0 && *i + 1 < s.len())
                    .map(|i| s[i + 1..].to_ascii_lowercase())
                    .unwrap_or_default()
            )
        }
        "paths.ancestors" => {
            let s = v.as_str().unwrap();
            let mut a = vec![];
            for (i, c) in s.char_indices() {
                if c == '/' && (i > 0 || s != "/") {
                    a.push(if i == 0 {
                        "/".to_string()
                    } else {
                        s[..i].to_string()
                    });
                }
            }
            json!(a)
        }
        "sequences.interleave" | "sequences.zip_longest" => {
            let a = v[0].as_array().unwrap();
            let b = v[1].as_array().unwrap();
            let mut o = vec![];
            for i in 0..a.len().max(b.len()) {
                if id == "sequences.zip_longest" {
                    o.push(json!([
                        a.get(i).unwrap_or(&Value::Null),
                        b.get(i).unwrap_or(&Value::Null)
                    ]));
                } else {
                    if let Some(x) = a.get(i) {
                        o.push(x.clone());
                    }
                    if let Some(x) = b.get(i) {
                        o.push(x.clone());
                    }
                }
            }
            json!(o)
        }
        "sequences.chunks" => json!(
            v["values"]
                .as_array()
                .unwrap()
                .chunks(v["size"].as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        ),
        "sequences.move_zeroes" => {
            let a = ints(v);
            let mut o = a.iter().filter(|x| **x != 0).copied().collect::<Vec<_>>();
            o.resize(a.len(), 0);
            json!(o)
        }
        "calendar.leap" => json!(leap(v.as_i64().unwrap())),
        "calendar.month_days" => json!(days(v[0].as_i64().unwrap(), v[1].as_i64().unwrap())),
        "calendar.ordinal" => {
            let a = ints(v);
            json!((1..a[1]).map(|m| days(a[0], m)).sum::<i64>() + a[2])
        }
        "calendar.valid_date" => {
            let a = ints(v);
            json!(
                (1..=9999).contains(&a[0])
                    && (1..=12).contains(&a[1])
                    && a[2] >= 1
                    && a[2] <= days(a[0], a[1])
            )
        }
        "bits.popcount" => json!((v.as_u64().unwrap() as u32).count_ones()),
        "bits.hamming" => {
            json!(((v[0].as_u64().unwrap() as u32) ^ (v[1].as_u64().unwrap() as u32)).count_ones())
        }
        "bits.parity" => json!((v.as_u64().unwrap() as u32).count_ones() % 2),
        "bits.lowest" => {
            let n = v.as_u64().unwrap() as u32;
            json!(if n == 0 {
                -1
            } else {
                n.trailing_zeros() as i64
            })
        }
        "money.parse" => {
            let s = v.as_str().unwrap();
            let (neg, t) = if let Some(t) = s.strip_prefix('-') {
                (true, t)
            } else {
                (false, s)
            };
            let p: Vec<_> = t.split('.').collect();
            let good = p.len() <= 2
                && !p[0].is_empty()
                && p[0].bytes().all(|b| b.is_ascii_digit())
                && (p.len() == 1 || (p[1].len() == 2 && p[1].bytes().all(|b| b.is_ascii_digit())));
            if good {
                let n = p[0].parse::<i64>()? * 100
                    + if p.len() == 2 {
                        p[1].parse::<i64>()?
                    } else {
                        0
                    };
                json!(if neg { -n } else { n })
            } else {
                Value::Null
            }
        }
        "money.format" => {
            let n = v.as_i64().unwrap();
            json!(format!(
                "{}{}.{:02}",
                if n < 0 { "-" } else { "" },
                n.abs() / 100,
                n.abs() % 100
            ))
        }
        "money.split" => {
            let a = ints(v);
            json!(
                (0..a[1])
                    .map(|i| a[0] / a[1] + i64::from(i < a[0] % a[1]))
                    .collect::<Vec<_>>()
            )
        }
        "money.basket" => json!(rows(v).iter().map(|r| r[0] * r[1]).sum::<i64>()),
        "dispatch.finish_order" | "dispatch.waiting" => {
            let (mut finish, mut wait) = (0, 0);
            let mut out = vec![];
            for r in rows(v) {
                let start = finish.max(r[0]);
                wait += start - r[0];
                finish = start + r[1];
                out.push(finish);
            }
            if id == "dispatch.waiting" {
                json!(wait)
            } else {
                json!(out)
            }
        }
        "dispatch.round_robin" => {
            let mut a = ints(&v["jobs"]);
            let q = v["quantum"].as_i64().unwrap();
            let mut o = (0..a.len()).filter(|i| a[*i] == 0).collect::<Vec<_>>();
            while o.len() < a.len() {
                for (i, n) in a.iter_mut().enumerate() {
                    if *n > 0 {
                        *n = (*n - q).max(0);
                        if *n == 0 {
                            o.push(i);
                        }
                    }
                }
            }
            json!(o)
        }
        "dispatch.priority" => {
            let mut a = rows(v);
            a.sort_by_key(|r| (-r[0], r[1]));
            json!(a.iter().map(|r| r[1]).collect::<Vec<_>>())
        }
        "encoding.hex_encode" => json!(
            ints(v)
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect::<String>()
        ),
        "encoding.hex_decode" => {
            let s = v.as_str().unwrap();
            // Two hex digits encode one byte; the low length bit detects an
            // incomplete pair without requiring a newer standard-library API.
            if s.len() & 1 != 0 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
                Value::Null
            } else {
                json!(
                    (0..s.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                        .collect::<Vec<_>>()
                )
            }
        }
        "encoding.caesar" | "encoding.rot13" => {
            let (s, k) = if id == "encoding.rot13" {
                (v.as_str().unwrap(), 13)
            } else {
                (
                    v["text"].as_str().unwrap(),
                    v["shift"].as_i64().unwrap().rem_euclid(26) as u8,
                )
            };
            json!(
                s.bytes()
                    .map(|b| {
                        let base = if b.is_ascii_lowercase() {
                            Some(b'a')
                        } else if id == "encoding.rot13" && b.is_ascii_uppercase() {
                            Some(b'A')
                        } else {
                            None
                        };
                        base.map(|a| ((b - a + k) % 26 + a) as char)
                            .unwrap_or(b as char)
                    })
                    .collect::<String>()
            )
        }
        "tables.sort_rows" => {
            let mut a = rows(v);
            a.sort();
            json!(a)
        }
        "tables.last_by_key" | "tables.group_sums" => {
            let mut m = BTreeMap::new();
            for r in rows(v) {
                if id == "tables.group_sums" {
                    *m.entry(r[0]).or_insert(0) += r[1];
                } else {
                    m.insert(r[0], r[1]);
                }
            }
            json!(m.into_iter().map(|(k, v)| [k, v]).collect::<Vec<_>>())
        }
        "tables.inner_join" => {
            let b: BTreeMap<_, _> = rows(&v[1]).iter().map(|r| (r[0], r[1])).collect();
            let mut a = rows(&v[0]);
            a.sort();
            json!(
                a.iter()
                    .filter_map(|r| b.get(&r[0]).map(|b| [r[0], r[1], *b]))
                    .collect::<Vec<_>>()
            )
        }
        "geometry.box" => {
            let a = rows(v);
            if a.is_empty() {
                Value::Null
            } else {
                json!([
                    a.iter().map(|r| r[0]).min(),
                    a.iter().map(|r| r[1]).min(),
                    a.iter().map(|r| r[0]).max(),
                    a.iter().map(|r| r[1]).max()
                ])
            }
        }
        "geometry.manhattan" => json!(
            rows(v)
                .iter()
                .map(|r| r[0].abs() + r[1].abs())
                .collect::<Vec<_>>()
        ),
        "geometry.area2" => {
            let a = rows(v);
            json!(
                (0..a.len())
                    .map(|i| {
                        let j = (i + 1) % a.len();
                        a[i][0] * a[j][1] - a[j][0] * a[i][1]
                    })
                    .sum::<i64>()
            )
        }
        "geometry.quadrants" => {
            let mut n = [0; 5];
            for p in rows(v) {
                let i = match (p[0].signum(), p[1].signum()) {
                    (1, 1) => 0,
                    (-1, 1) => 1,
                    (-1, -1) => 2,
                    (1, -1) => 3,
                    _ => 4,
                };
                n[i] += 1;
            }
            json!(n)
        }
        "statistics.median2" => {
            let mut a = ints(v);
            a.sort();
            json!(a[(a.len() - 1) / 2] + a[a.len() / 2])
        }
        "statistics.mode" => {
            let mut m = BTreeMap::new();
            for n in ints(v) {
                *m.entry(n).or_insert(0) += 1;
            }
            let max = *m.values().max().unwrap();
            json!(m.into_iter().find(|(_, n)| *n == max).unwrap().0)
        }
        "statistics.ranks" => {
            let a = ints(v);
            let mut b = a.clone();
            b.sort();
            b.dedup();
            json!(
                a.iter()
                    .map(|n| b.binary_search(n).unwrap() + 1)
                    .collect::<Vec<_>>()
            )
        }
        "statistics.quantile" => {
            let mut a = ints(&v["values"]);
            a.sort();
            let p = v["p"].as_u64().unwrap() as usize;
            json!(a[(p * a.len()).div_ceil(100) - 1])
        }
        x if x.starts_with("sets.") => {
            let a: BTreeSet<_> = ints(&v[0]).into_iter().collect();
            let b: BTreeSet<_> = ints(&v[1]).into_iter().collect();
            match x {
                "sets.intersection" => json!(a.intersection(&b).collect::<Vec<_>>()),
                "sets.symmetric" => json!(a.symmetric_difference(&b).collect::<Vec<_>>()),
                "sets.subset" => json!(a.is_subset(&b)),
                _ => {
                    let u = a.union(&b).count();
                    if u == 0 {
                        json!([1, 1])
                    } else {
                        fraction(a.intersection(&b).count() as i64, u as i64)
                    }
                }
            }
        }
        "rational.reduce" => fraction(v[0].as_i64().unwrap(), v[1].as_i64().unwrap()),
        "rational.reciprocal" => {
            let a = ints(v);
            if a[0] == 0 {
                Value::Null
            } else {
                fraction(a[1], a[0])
            }
        }
        "rational.add" | "rational.compare" => {
            let a = ints(&v[0]);
            let b = ints(&v[1]);
            if id == "rational.add" {
                fraction(a[0] * b[1] + b[0] * a[1], a[1] * b[1])
            } else {
                json!(((a[0] * b[1] - b[0] * a[1]) * (a[1] * b[1]).signum()).signum())
            }
        }
        "messages.headers" => {
            let mut m: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for s in strs(v) {
                let (a, b) = s.split_once(':').unwrap();
                m.entry(
                    a.trim_matches(|c: char| c.is_ascii_whitespace())
                        .to_ascii_lowercase(),
                )
                .or_default()
                .push(b.trim_matches(|c: char| c.is_ascii_whitespace()).into());
            }
            json!(m)
        }
        "messages.address" => {
            let (a, b) = v.as_str().unwrap().split_once('@').unwrap();
            json!([a, b.to_ascii_lowercase().as_str()])
        }
        "messages.hashtags" => {
            let s = v.as_str().unwrap().as_bytes();
            let (mut i, mut seen, mut out) = (0, BTreeSet::new(), vec![]);
            while i < s.len() {
                if s[i] == b'#' {
                    let start = i + 1;
                    i = start;
                    while i < s.len() && (s[i].is_ascii_alphanumeric() || s[i] == b'_') {
                        i += 1;
                    }
                    if i > start {
                        let tag = std::str::from_utf8(&s[start..i])
                            .unwrap()
                            .to_ascii_lowercase();
                        if seen.insert(tag.clone()) {
                            out.push(tag);
                        }
                    }
                } else {
                    i += 1;
                }
            }
            json!(out)
        }
        "messages.wrap" => {
            let w = v["width"].as_u64().unwrap() as usize;
            let mut lines: Vec<String> = vec![];
            for word in v["text"].as_str().unwrap().split_ascii_whitespace() {
                if let Some(last) = lines
                    .last_mut()
                    .filter(|last| last.len() + 1 + word.len() <= w)
                {
                    last.push(' ');
                    last.push_str(word);
                    continue;
                }
                lines.push(word.into());
            }
            json!(lines)
        }
        "automata.divisible3" => json!(
            v.as_str()
                .unwrap()
                .bytes()
                .fold(0, |r, b| (r * 2 + (b - b'0') as u32) % 3)
                == 0
        ),
        "automata.balanced_prefix" => {
            let (mut n, mut best) = (0i32, 0);
            for (i, c) in v.as_str().unwrap().bytes().enumerate() {
                n += if c == b'(' { 1 } else { -1 };
                if n < 0 {
                    break;
                }
                if n == 0 {
                    best = i + 1;
                }
            }
            json!(best)
        }
        "automata.classes" => {
            let mut a = [0; 5];
            for b in v.as_str().unwrap().bytes() {
                let i = if b.is_ascii_uppercase() {
                    0
                } else if b.is_ascii_lowercase() {
                    1
                } else if b.is_ascii_digit() {
                    2
                } else if b.is_ascii_whitespace() {
                    3
                } else {
                    4
                };
                a[i] += 1;
            }
            json!(a)
        }
        "automata.transitions" => {
            let mut a = [0; 4];
            for w in v.as_str().unwrap().as_bytes().windows(2) {
                a[((w[0] - b'0') * 2 + w[1] - b'0') as usize] += 1;
            }
            json!(a)
        }
        "transactions.balance" | "transactions.running_min" | "transactions.first_overdraft" => {
            let mut n = v["initial"].as_i64().unwrap();
            let mut min = n;
            let mut first = -1;
            for (i, d) in ints(&v["deltas"]).iter().enumerate() {
                n += d;
                min = min.min(n);
                if n < 0 && first < 0 {
                    first = i as i64;
                }
            }
            json!(match id {
                "transactions.balance" => n,
                "transactions.running_min" => min,
                _ => first,
            })
        }
        "transactions.settle" => {
            let mut m = BTreeMap::new();
            for r in rows(v) {
                *m.entry(r[0]).or_insert(0) -= r[2];
                *m.entry(r[1]).or_insert(0) += r[2];
            }
            json!(m.into_iter().map(|(k, v)| [k, v]).collect::<Vec<_>>())
        }
        "ranges.expand" => {
            let mut s = BTreeSet::new();
            for r in rows(v) {
                s.extend(r[0]..=r[1]);
            }
            json!(s)
        }
        "ranges.compress" => {
            let s: BTreeSet<_> = ints(v).into_iter().collect();
            let mut a: Vec<[i64; 2]> = vec![];
            for n in s {
                if let Some(last) = a.last_mut().filter(|last| n == last[1] + 1) {
                    last[1] = n;
                    continue;
                }
                a.push([n, n]);
            }
            json!(a)
        }
        "ranges.missing" => {
            let s: BTreeSet<_> = ints(&v["values"]).into_iter().collect();
            json!(
                (v["low"].as_i64().unwrap()..=v["high"].as_i64().unwrap())
                    .filter(|n| !s.contains(n))
                    .collect::<Vec<_>>()
            )
        }
        "ranges.kth" => {
            let s: BTreeSet<_> = ints(&v["values"]).into_iter().collect();
            s.into_iter()
                .nth(v["k"].as_u64().unwrap() as usize - 1)
                .map(|n| json!(n))
                .unwrap_or(Value::Null)
        }
        _ => bail!("unknown task ID {id}"),
    };
    domain::validate_output(id, &out)?;
    Ok(out)
}

/// Private edge cases are separate from the serialised public task catalogue.
pub fn private_cases(task: &Task) -> Result<Vec<Case>> {
    if crate::teacher_search::qualification::is_task(&task.id) {
        return crate::teacher_search::qualification::private_cases(task);
    }
    let id = task.id.as_str();
    let inputs: Vec<Value> = match id {
        x if x.starts_with("ledger.") => vec![
            json!([]),
            json!([0]),
            json!([-7, -2, -19]),
            json!([10000, -10000, 3, 3, 0]),
            json!((0..64).map(|i| i - 32).collect::<Vec<_>>()),
        ],
        x if x.starts_with("stock.") => vec![
            json!([]),
            json!(["z"]),
            json!(["", "a", "", "b", "a"]),
            json!(["Z", "a", "Z", "a"]),
            json!(["x", "x", "x", "y", "z", "y"]),
        ],
        x if x.starts_with("text.") => vec![
            json!(""),
            json!(" \t\n\r"),
            json!("A--B__C 123"),
            json!("UPPER lower"),
            json!("a\u{000b}b\u{000c}c"),
        ],
        x if x.starts_with("intervals.") => vec![
            json!([]),
            json!([[-5, -1]]),
            json!([[0, 2], [2, 4], [1, 3]]),
            json!([[-8, -5], [2, 3], [10, 11]]),
            json!([[0, 10], [1, 2], [3, 4], [0, 10]]),
        ],
        x if x.starts_with("graph.") => vec![
            json!({"n":1,"start":0,"target":0,"edges":[]}),
            json!({"n":3,"start":2,"target":0,"edges":[[0,1]]}),
            json!({"n":3,"start":0,"target":2,"edges":[[0,1],[1,2],[2,0]]}),
            json!({"n":4,"start":0,"target":3,"edges":[[0,2],[0,1],[1,3],[2,3]]}),
            json!({"n":2,"start":0,"target":1,"edges":[[0,0],[0,1]]}),
        ],
        "validation.luhn" => vec![
            json!(""),
            json!("0"),
            json!("00"),
            json!("79927398714"),
            json!("4532015112830366"),
            json!("12 3"),
            json!("ab"),
        ],
        "validation.ipv4" => vec![
            json!("0.0.0.0"),
            json!("255.255.255.255"),
            json!("01.2.3.4"),
            json!("1.2.3.256"),
            json!("1.2.3"),
            json!("1..3.4"),
            json!("+1.2.3.4"),
        ],
        "validation.brackets" => vec![
            json!(""),
            json!("([{}])"),
            json!("([)]"),
            json!("abc"),
            json!("]"),
            json!("(("),
        ],
        "validation.identifier" => vec![
            json!(""),
            json!("_"),
            json!("1a"),
            json!("a-b"),
            json!("fn"),
            json!("Z_09"),
        ],
        "matrices.diagonals" => vec![
            json!([[7]]),
            json!([[1, 0], [0, -2]]),
            json!([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            json!([[0, 0], [0, 0]]),
        ],
        x if x.starts_with("matrices.") => vec![
            json!([[7]]),
            json!([[1, 2, 3, 4]]),
            json!([[1], [2], [3]]),
            json!([[-1, 0], [8, -5]]),
        ],
        "search.occurrences" => vec![
            json!({"text":"","pattern":"a"}),
            json!({"text":"aaaaa","pattern":"aa"}),
            json!({"text":"ab","pattern":"abc"}),
            json!({"text":"xxxyxx","pattern":"x"}),
            json!({"text":"abc","pattern":"abc"}),
        ],
        "search.common_prefix" => vec![
            json!([]),
            json!([""]),
            json!(["abc"]),
            json!(["abc", "ab", ""]),
            json!(["same", "same"]),
            json!(["car", "cart", "carbon"]),
        ],
        "search.runs" => vec![
            json!(""),
            json!("a"),
            json!("abab"),
            json!("   !!!"),
            json!("ZZZZZ"),
        ],
        "search.anagram" => vec![
            json!(["", ""]),
            json!(["a!", "A!"]),
            json!(["a!", "a"]),
            json!(["aab", "abb"]),
            json!([" A\tB ", "ba"]),
        ],
        "paths.normalise" => vec![
            json!("/"),
            json!("/../../x"),
            json!("///a//b"),
            json!("/a/./b/../../"),
            json!("/x/.../y"),
        ],
        "paths.basename" => vec![
            json!("/"),
            json!("////"),
            json!("/a"),
            json!("/a/.."),
            json!("/x/.//"),
        ],
        "paths.extension" => vec![
            json!("file"),
            json!(".bashrc"),
            json!("a."),
            json!(".config.JSON"),
            json!("x.TXT"),
        ],
        "paths.ancestors" => vec![
            json!("/"),
            json!("/a"),
            json!("/a/b"),
            json!("/one/two/three/four"),
        ],
        "sequences.interleave" | "sequences.zip_longest" => vec![
            json!([[], []]),
            json!([[1], []]),
            json!([[], [3, 4]]),
            json!([[0, -1], [2, 3, 4]]),
        ],
        "sequences.chunks" => vec![
            json!({"values":[],"size":3}),
            json!({"values":[1,2],"size":1}),
            json!({"values":[0,-1,2],"size":9}),
            json!({"values":[1,2,3,4],"size":2}),
        ],
        "sequences.move_zeroes" => vec![
            json!([]),
            json!([0, 0]),
            json!([1, -1]),
            json!([0, 0, 2, 0, -3, 0]),
        ],
        "calendar.leap" => vec![
            json!(1),
            json!(1900),
            json!(2004),
            json!(2100),
            json!(2400),
            json!(9999),
        ],
        "calendar.month_days" => vec![
            json!([1900, 2]),
            json!([2000, 2]),
            json!([2023, 4]),
            json!([2023, 12]),
            json!([1, 1]),
        ],
        "calendar.ordinal" => vec![
            json!([1900, 3, 1]),
            json!([2000, 12, 31]),
            json!([2023, 1, 1]),
            json!([2023, 12, 31]),
            json!([2004, 2, 29]),
        ],
        "calendar.valid_date" => vec![
            json!([0, 1, 1]),
            json!([10000, 1, 1]),
            json!([2000, 2, 29]),
            json!([1900, 2, 29]),
            json!([2023, 0, 2]),
            json!([2023, 4, 31]),
            json!([2023, 12, 0]),
        ],
        "bits.hamming" => vec![
            json!([0, 0]),
            json!([0, 4294967295u64]),
            json!([2147483648u64, 0]),
            json!([12345, 54321]),
            json!([7, 7]),
        ],
        x if x.starts_with("bits.") => vec![
            json!(0),
            json!(1),
            json!(2147483648u64),
            json!(4294967295u64),
            json!(1024),
            json!(2863311530u64),
        ],
        "money.parse" => vec![
            json!("0"),
            json!("-0.01"),
            json!("001.20"),
            json!("1.2"),
            json!("+2.00"),
            json!(".12"),
            json!("2.000"),
            json!(" 2"),
        ],
        "money.format" => vec![json!(0), json!(1), json!(-1), json!(100), json!(-100000000)],
        "money.split" => vec![json!([0, 4]), json!([3, 5]), json!([12, 4]), json!([1, 1])],
        "money.basket" => vec![
            json!([]),
            json!([[0, 5]]),
            json!([[8, 0], [7, 4]]),
            json!([[999, 2], [1, 3]]),
        ],
        "dispatch.finish_order" | "dispatch.waiting" => vec![
            json!([]),
            json!([[5, 0]]),
            json!([[0, 2], [0, 3], [1, 0]]),
            json!([[10, 2], [2, 3], [20, 1]]),
        ],
        "dispatch.round_robin" => vec![
            json!({"jobs":[],"quantum":1}),
            json!({"jobs":[0,1,0],"quantum":2}),
            json!({"jobs":[5,5,1],"quantum":1}),
            json!({"jobs":[2,2],"quantum":8}),
        ],
        "dispatch.priority" => vec![
            json!([]),
            json!([[0, 9]]),
            json!([[-1, 3], [0, 2], [-1, 1]]),
            json!([[2, 3], [2, 2], [2, 1]]),
        ],
        "encoding.hex_encode" => vec![
            json!([]),
            json!([1, 16, 128]),
            json!([255, 254, 253]),
            json!([0, 0, 0]),
        ],
        "encoding.hex_decode" => vec![
            json!(""),
            json!("fF01"),
            json!("a"),
            json!("xz"),
            json!("01 2"),
            json!("abcdef"),
        ],
        "encoding.caesar" => vec![
            json!({"text":"","shift":1}),
            json!({"text":"azAZ!","shift":-1}),
            json!({"text":"hello","shift":52}),
            json!({"text":"abc xyz","shift":-27}),
        ],
        "encoding.rot13" => vec![
            json!(""),
            json!("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            json!("abcdefghijklmnopqrstuvwxyz"),
            json!("123 !"),
        ],
        "tables.inner_join" => vec![
            json!([[], []]),
            json!([[[1, 2]], []]),
            json!([[[3, 8], [1, 2]], [[1, 7], [3, 9]]]),
            json!([[[1, 2]], [[2, 3]]]),
        ],
        x if x.starts_with("tables.") => vec![
            json!([]),
            json!([[0, 0]]),
            json!([[2, -3], [1, 9], [2, 4], [1, -9]]),
            json!([[1, 2], [1, 2], [1, 1]]),
        ],
        "geometry.area2" => vec![
            json!([[0, 0], [0, 2], [3, 0]]),
            json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
            json!([[0, 0], [1, 1], [2, 2]]),
            json!([[2, 3], [5, 3], [2, 5]]),
        ],
        x if x.starts_with("geometry.") => vec![
            json!([]),
            json!([[0, 0]]),
            json!([[-5, -3], [2, -8], [-1, 9]]),
            json!([[1, 0], [0, -2], [-1, 0], [0, 2]]),
        ],
        "statistics.quantile" => vec![
            json!({"values":[7],"p":1}),
            json!({"values":[4,3,2,1],"p":1}),
            json!({"values":[4,3,2,1],"p":100}),
            json!({"values":[1,2,3],"p":50}),
            json!({"values":[-5,-2,0,0,9],"p":80}),
        ],
        "statistics.ranks" => vec![
            json!([]),
            json!([9]),
            json!([-5, -2, -5, 0]),
            json!([3, 3, 3]),
            json!([4, 3, 2, 1]),
        ],
        x if x.starts_with("statistics.") => vec![
            json!([9]),
            json!([-5, -2, -5, 0]),
            json!([3, 3, 3]),
            json!([4, 3, 2, 1]),
            json!([-9, -1]),
        ],
        x if x.starts_with("sets.") => vec![
            json!([[], []]),
            json!([[1], []]),
            json!([[1, 1], [1, 1]]),
            json!([[-3, 0, 2], [0, 4, 4]]),
            json!([[1, 2, 3], [1, 2, 3, 4]]),
        ],
        "rational.add" | "rational.compare" => vec![
            json!([[0, 3], [0, -2]]),
            json!([[1, -2], [-1, 2]]),
            json!([[-2, -3], [1, 6]]),
            json!([[3, 7], [-2, 5]]),
        ],
        x if x.starts_with("rational.") => vec![
            json!([0, -9]),
            json!([-6, -8]),
            json!([1, -1]),
            json!([999, 27]),
            json!([-7, 3]),
        ],
        "messages.headers" => vec![
            json!([]),
            json!(["A:"]),
            json!([" a : x ", "A:y", "B:z:z"]),
            json!(["X:  \t "]),
        ],
        "messages.address" => vec![
            json!("a@B"),
            json!("UPPER@lower.test"),
            json!("a.b+tag@EXAMPLE.ORG"),
            json!("X@Y.Z"),
        ],
        "messages.hashtags" => vec![
            json!(""),
            json!("# #! ##one"),
            json!("#A #a #A_1 #a_1"),
            json!("x#tag,#Other-#3"),
        ],
        "messages.wrap" => vec![
            json!({"text":"","width":1}),
            json!({"text":" a  b c ","width":1}),
            json!({"text":"aa bb c","width":5}),
            json!({"text":"longword x","width":8}),
        ],
        "automata.divisible3" => vec![
            json!("0"),
            json!("1"),
            json!("11"),
            json!("0001001"),
            json!("1".repeat(120)),
            json!("10".repeat(61)),
        ],
        "automata.balanced_prefix" => vec![
            json!(""),
            json!(")()"),
            json!("(("),
            json!("()()"),
            json!("(()))()"),
        ],
        "automata.classes" => vec![json!(""), json!("AZaz09\t\n!?"), json!("   "), json!("a_A")],
        "automata.transitions" => vec![
            json!(""),
            json!("1"),
            json!("0000"),
            json!("1111"),
            json!("101010"),
        ],
        "transactions.settle" => vec![
            json!([]),
            json!([[1, 1, 8]]),
            json!([[2, 3, 0], [3, 2, 5]]),
            json!([[-1, 0, 7], [0, 2, 7], [2, -1, 7]]),
        ],
        "transactions.first_overdraft" => vec![
            json!({"initial":0,"deltas":[]}),
            json!({"initial":0,"deltas":[-1,2]}),
            json!({"initial":5,"deltas":[-5,0,1]}),
            json!({"initial":3,"deltas":[1,-8,8,-8]}),
        ],
        x if x.starts_with("transactions.") => vec![
            json!({"initial":0,"deltas":[]}),
            json!({"initial":-5,"deltas":[3,-8]}),
            json!({"initial":5,"deltas":[-5,0,1]}),
            json!({"initial":3,"deltas":[1,-8,8,-8]}),
        ],
        "ranges.expand" => vec![
            json!([]),
            json!([[2, 2]]),
            json!([[-3, -1], [-1, 2], [5, 6]]),
            json!([[1, 4], [2, 3]]),
        ],
        "ranges.compress" => vec![
            json!([]),
            json!([2]),
            json!([-3, -2, 0, 1, 3, 3]),
            json!([4, 3, 2, 1]),
        ],
        "ranges.missing" => vec![
            json!({"values":[],"low":-2,"high":2}),
            json!({"values":[3],"low":3,"high":3}),
            json!({"values":[-9,0,0,8],"low":-1,"high":1}),
            json!({"values":[1,2,3],"low":1,"high":3}),
        ],
        "ranges.kth" => vec![
            json!({"values":[],"k":1}),
            json!({"values":[2,2],"k":2}),
            json!({"values":[4,-3,4,0],"k":1}),
            json!({"values":[4,-3,4,0],"k":3}),
        ],
        _ => bail!("no private cases for {id}"),
    };
    let mut out = Vec::new();
    for (i, input) in inputs.into_iter().enumerate() {
        out.push(Case {
            expected: oracle(id, &input)
                .with_context(|| format!("oracle {id} private case {i}"))?,
            input,
            label: format!("private-edge-{i}"),
        });
    }
    // Fresh deterministic inputs extend the integer contracts beyond public
    // constants. They inherit the original source; seeds never create sources.
    if id.starts_with("ledger.") {
        for seed in 1..=8 {
            let input = json!(
                (0..(seed + 2))
                    .map(|i| ((i * 37 + seed * 19) % 97) - 48)
                    .collect::<Vec<_>>()
            );
            out.push(Case {
                expected: oracle(id, &input)?,
                input,
                label: format!("generated-ledger-{seed}"),
            });
        }
    }
    Ok(out)
}

pub fn validate_corpus() -> Result<CorpusSummary> {
    let tasks = corpus();
    let mut ids = BTreeSet::new();
    let mut sources: BTreeMap<String, Split> = BTreeMap::new();
    for t in &tasks {
        if !ids.insert(&t.id) {
            bail!("duplicate task {}", t.id);
        }
        if sources
            .insert(t.source_id.clone(), t.split)
            .filter(|split| *split != t.split)
            .is_some()
        {
            bail!("source leakage");
        }
        for e in &t.public_examples {
            if oracle(&t.id, &e.input)? != e.expected {
                bail!("public oracle mismatch {}", t.id);
            }
        }
        if private_cases(t)?.len() < 4 {
            bail!("too few private cases {}", t.id);
        }
    }
    let summary = CorpusSummary {
        tasks: tasks.len(),
        train: tasks.iter().filter(|t| t.split == Split::Train).count(),
        development: tasks
            .iter()
            .filter(|t| t.split == Split::Development)
            .count(),
        held_out: tasks.iter().filter(|t| t.split == Split::HeldOut).count(),
        sources: sources.len(),
        corpus_sha256: hash(serde_json::to_vec(&tasks)?),
    };
    if (
        summary.tasks,
        summary.train,
        summary.development,
        summary.held_out,
        summary.sources,
    ) != (96, 48, 16, 32, 24)
    {
        bail!("invalid corpus split");
    }
    Ok(summary)
}

pub fn write_corpus(root: &Path) -> Result<CorpusSummary> {
    let summary = validate_corpus()?;
    fs::create_dir_all(root.join("tasks"))?;
    fs::create_dir_all(root.join("grader_private"))?;
    let tasks = corpus();
    fs::write(
        root.join("tasks/catalogue.json"),
        serde_json::to_vec_pretty(&tasks)?,
    )?;
    let index: Vec<_> = tasks
        .iter()
        .filter(|t| t.split == Split::HeldOut)
        .map(|t| HeldoutEntry {
            task_id: t.id.clone(),
            source_id: t.source_id.clone(),
            lineage_id: t.lineage_id.clone(),
            specification: t.specification.clone(),
            specification_sha256: t.specification_sha256.clone(),
        })
        .collect();
    fs::write(
        root.join("grader_private/heldout-index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    for t in &tasks {
        fs::write(
            root.join("grader_private")
                .join(format!("{}.cases.json", t.id)),
            serde_json::to_vec_pretty(&private_cases(t)?)?,
        )?;
    }
    fs::write(
        root.join("tasks/manifest.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn monetary_format_grading_stays_within_the_declared_input_contract() {
        let task = corpus()
            .into_iter()
            .find(|t| t.id == "money.format")
            .unwrap();
        for case in private_cases(&task).unwrap() {
            assert!(case.input.as_i64().unwrap().unsigned_abs() <= 100_000_000);
        }
    }
    #[test]
    fn all_public_examples_and_source_split_are_checked() {
        let s = validate_corpus().unwrap();
        assert_eq!(s.sources, 24);
    }
    #[test]
    fn private_inputs_break_public_answer_hardcoding() {
        for t in corpus() {
            let cases = private_cases(&t).unwrap();
            assert!(
                cases
                    .iter()
                    .any(|c| c.expected != t.public_examples[0].expected),
                "{}",
                t.id
            );
        }
    }
    #[test]
    fn independent_interval_length_property() {
        for seed in 0..40 {
            let input = json!([[-4, seed % 5 + 1], [seed % 3, 8], [9, 11]]);
            let direct = (-8..16)
                .filter(|x| rows(&input).iter().any(|r| *x >= r[0] && *x < r[1]))
                .count();
            assert_eq!(oracle("intervals.coverage", &input).unwrap(), json!(direct));
        }
    }
    #[test]
    fn fractions_obey_sign_and_reduction() {
        for a in -20..=20 {
            for b in -20..=20 {
                if b == 0 {
                    continue;
                }
                let r = oracle("rational.reduce", &json!([a, b])).unwrap();
                let n = r[0].as_i64().unwrap();
                let d = r[1].as_i64().unwrap();
                assert!(d > 0);
                assert_eq!(n * b, a * d);
                assert_eq!(gcd(n, d), 1);
            }
        }
    }
    #[test]
    fn serialised_public_catalogue_excludes_private_inputs() {
        let encoded = serde_json::to_string(&corpus()).unwrap();
        assert!(!encoded.contains("private-edge-"));
        assert!(!encoded.contains("generated-ledger-"));
    }
}
