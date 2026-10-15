//! Fresh, locally authored development qualification; never promotion evidence.
//! Public contracts, private expectations and reference programmes freeze before
//! candidate results. Source groups are lineage units, not independent authors.
use crate::tasks::{self, Case, Example, Split, Task};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
};

pub const PACK_ID: &str = "fresh-development-qualification-v1";
const AUTHOR: &str = "Locally authored in Rust by the assessment's Codex engineering agent on 15 October 2026; authored before this pack's candidate outcomes. Four new lineage groups; no claim that familiar algorithms were unseen by model pretraining. Model selection only, never held-out promotion or proof of learning.";

pub fn is_task(id: &str) -> bool {
    id.starts_with("qualification.")
}

pub fn corpus() -> Vec<Task> {
    let definitions = vec![
        (
            "clock.shift",
            "Input {clock,delta}: clock is valid 24-hour HH:MM with exactly two digits per field; delta is minutes in [-10080,10080]. Add delta to minutes since midnight. Return {clock:HH:MM,day_offset:integer}; use mathematical floor division for the signed day offset, so 00:00 minus one minute is 23:59 on day -1.",
            json!({"clock":"23:50","delta":25}),
            json!({"clock":"00:15","day_offset":1}),
        ),
        (
            "clock.window",
            "Input {start,end,query}, each a valid exactly padded HH:MM. Return whether query is in the daily cyclic half-open interval [start,end). Crossing midnight wraps; equal start and end means an empty interval, not a whole day.",
            json!({"start":"22:00","end":"02:00","query":"00:30"}),
            json!(true),
        ),
        (
            "clock.duration",
            "Input is an ASCII duration string. Accept exactly [-]H:MM:SS: optional minus only, hours one to three decimal digits without leading zero unless exactly 0, minutes and seconds exactly two digits in 00..59. Return signed total seconds, or null for invalid syntax. Negative zero is allowed and returns 0. No whitespace or plus sign is allowed.",
            json!("-2:03:04"),
            json!(-7384),
        ),
        (
            "clock.round_up",
            "Input {clock,step}: valid exactly padded HH:MM and integer step in 1..1440. Round minutes since midnight upwards to the smallest multiple of step greater than or equal to the input. Return {clock:HH:MM,day_offset:integer} using the rounded value; rounding is on the minute axis before wrapping at midnight.",
            json!({"clock":"23:59","step":60}),
            json!({"clock":"00:00","day_offset":1}),
        ),
        (
            "telemetry.counter",
            "Input is an array of nonnegative integer counter readings, each <=1000000. The first is a baseline and contributes zero. For each later reading add current-previous if current>=previous; otherwise treat it as a reset and add current. Return the total increase; empty and singleton inputs return 0.",
            json!([10, 14, 2, 5]),
            json!(9),
        ),
        (
            "telemetry.excursions",
            "Input {values,low,high}, integer low<=high and integer values, each magnitude <=1000000. Return maximal consecutive index ranges [start,end) whose values lie outside the closed acceptable band [low,high], in ascending order. Values equal to either bound are acceptable.",
            json!({"values":[0,8,9,3,-1],"low":0,"high":5}),
            json!([[1, 3], [4, 5]]),
        ),
        (
            "telemetry.buckets",
            "Input {width,events}: width is an integer in 1..10000; events is an array of [time,value] integers of magnitude <=1000000. Bucket start is floor(time/width)*width using mathematical floor for negatives. Return [bucket_start,count,sum_of_values] rows sorted by bucket_start. Count duplicate events separately; empty events return [].",
            json!({"width":10,"events":[[-1,2],[0,3],[-10,4],[10,-1]]}),
            json!([[-10, 2, 6], [0, 1, 3], [10, 1, -1]]),
        ),
        (
            "telemetry.integral",
            "Input {until,readings}: until is an integer in 0..1000000; readings is an array of [time,value], strictly increasing times in 0..until and values of magnitude <=1000000. Gauge value is 0 before the first reading, then stays at the most recent reading. Return its integer time integral over [0,until). A reading at until contributes no area.",
            json!({"until":10,"readings":[[2,3],[5,-1]]}),
            json!(4),
        ),
        (
            "address.endpoint",
            "Input is an ASCII host:port string. Return {host:lowercase,port:integer}, or null. Require exactly one colon. Host is 1..253 bytes of nonempty dot-separated labels, each 1..63 ASCII alphanumeric or hyphen characters with alphanumeric first/last character. No trailing dot. Port is canonical decimal in 1..65535, with no leading zero, sign or whitespace. IPv6 is outside this contract.",
            json!("API.Example:443"),
            json!({"host":"api.example","port":443}),
        ),
        (
            "address.cidr",
            "Input is an ASCII IPv4/prefix string. Each IPv4 octet is canonical decimal 0..255 with no leading zero unless 0. Prefix is canonical decimal 0..32. Return {network:dotted_ipv4,broadcast:dotted_ipv4,size:integer}, where host bits are cleared/set respectively and size=2^(32-prefix). Invalid strings return null.",
            json!("192.168.1.99/24"),
            json!({"network":"192.168.1.0","broadcast":"192.168.1.255","size":256}),
        ),
        (
            "address.mac",
            "Input is an ASCII MAC address string. Accept exactly six two-digit hexadecimal octets separated consistently by either ':' or '-'. Return six lowercase two-digit octets separated by ':'. Reject mixed separators, whitespace and all other syntax with null.",
            json!("AA-bB-00-12-34-FF"),
            json!("aa:bb:00:12:34:ff"),
        ),
        (
            "address.percent_path",
            "Input is an ASCII path string. Validate every '%' followed by exactly two hexadecimal digits. Decode a triplet only if its byte is an ASCII letter, digit, '-', '.', '_', or '~'; otherwise retain that triplet with uppercase hexadecimal digits. Preserve every raw non-'%' character exactly. Return null if any escape is malformed. '+' is a literal plus.",
            json!("/a%7eb%2fc+%41"),
            json!("/a~b%2Fc+A"),
        ),
        (
            "path.search_list",
            "Input is an ASCII colon-separated Unix search-list string. Return entries in first-occurrence order, replace each empty entry with '.', and remove subsequent duplicate entries after that replacement. Compare literally: do not normalise '/', '.', '..' or case. Empty input returns ['.']. Never inspect the filesystem.",
            json!("/bin::/bin:.:/usr/bin:"),
            json!(["/bin", ".", "/usr/bin"]),
        ),
        (
            "path.glob",
            "Input {pattern,name}: ASCII strings of at most 128 bytes, neither contains '/'. Return whole-string glob matching: '?' matches exactly one arbitrary byte and '*' matches zero or more bytes. All other bytes, including '[' and backslash, match literally; no escaping or character classes. Empty strings are allowed.",
            json!({"pattern":"a*?c","name":"abbbc"}),
            json!(true),
        ),
        (
            "path.relative",
            "Input {base,target}: canonical absolute ASCII POSIX paths. '/' is valid; other paths have no repeated or trailing slash and no '.', '..' or empty components. Return a lexical relative path from base directory to target: one '..' per unmatched base component, then unmatched target components, joined by '/'; identical paths return '.'. Never inspect the filesystem.",
            json!({"base":"/a/b","target":"/a/c/d"}),
            json!("../c/d"),
        ),
        (
            "path.rotations",
            "Input {stem,files}: stem is a nonempty ASCII filename without slash; files is an array of ASCII filenames. Return {active:boolean,rotations:array}: active iff stem occurs exactly; rotations are unique ascending integers N for names exactly stem+'.'+N where N is canonical decimal in 1..999 (no leading zero/sign). Ignore every other name. Never inspect the filesystem.",
            json!({"stem":"app.log","files":["app.log.2","app.log","app.log.01","app.log.2","app.log.1"]}),
            json!({"active":true,"rotations":[1,2]}),
        ),
    ];
    let source_hash = tasks::hash(include_bytes!("qualification.rs"));
    definitions
        .into_iter()
        .map(|(id, specification, input, expected)| {
            let domain = id.split('.').next().expect("fixed task domain");
            let specification = format!("{specification}\n{}", tasks::SHARED_DOMAIN);
            Task {
                id: format!("qualification.{id}"),
                source_id: format!("qualification-{domain}-v1"),
                lineage_id: format!("qualification-{domain}-v1"),
                split: Split::Development,
                specification_sha256: tasks::hash(specification.as_bytes()),
                specification,
                oracle_sha256: source_hash.clone(),
                public_examples: vec![Example { input, expected }],
                provenance: AUTHOR.into(),
                licence: "Apache-2.0".into(),
            }
        })
        .collect()
}

fn text(v: &Value) -> Result<&str> {
    let s = v.as_str().context("ASCII string required")?;
    ensure!(
        s.is_ascii() && s.len() <= 256,
        "string exceeds published domain"
    );
    Ok(s)
}
fn integer(v: &Value, lo: i64, hi: i64) -> Result<i64> {
    let n = v.as_i64().context("signed integer required")?;
    ensure!((lo..=hi).contains(&n), "integer outside published domain");
    Ok(n)
}
fn array(v: &Value) -> Result<&[Value]> {
    let a = v.as_array().context("array required")?;
    ensure!(a.len() <= 64, "array exceeds published domain");
    Ok(a)
}
fn clock(v: &Value) -> Result<i64> {
    let s = text(v)?;
    let b = s.as_bytes();
    ensure!(
        b.len() == 5 && b[2] == b':' && [b[0], b[1], b[3], b[4]].iter().all(u8::is_ascii_digit),
        "valid padded HH:MM required"
    );
    let hour = s[..2].parse::<i64>()?;
    let minute = s[3..].parse::<i64>()?;
    ensure!(hour < 24 && minute < 60, "invalid HH:MM");
    Ok(hour * 60 + minute)
}
fn clock_result(n: i64) -> Value {
    json!({"clock":format!("{:02}:{:02}",n.rem_euclid(1440)/60,n.rem_euclid(60)),"day_offset":n.div_euclid(1440)})
}
fn canonical_decimal(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0'))
    {
        return None;
    }
    s.parse().ok()
}
fn ipv4(s: &str) -> Option<u32> {
    let p = s.split('.').collect::<Vec<_>>();
    if p.len() != 4 {
        return None;
    }
    p.into_iter().try_fold(0_u32, |n, p| {
        let byte = canonical_decimal(p)?;
        (byte <= 255).then_some((n << 8) | byte)
    })
}
fn dotted(n: u32) -> String {
    std::net::Ipv4Addr::from(n).to_string()
}

pub fn oracle(id: &str, v: &Value) -> Result<Value> {
    Ok(match id {
        "qualification.clock.shift" => {
            clock_result(clock(&v["clock"])? + integer(&v["delta"], -10080, 10080)?)
        }
        "qualification.clock.window" => {
            let (a, b, q) = (clock(&v["start"])?, clock(&v["end"])?, clock(&v["query"])?);
            json!(if a < b {
                a <= q && q < b
            } else if a > b {
                q >= a || q < b
            } else {
                false
            })
        }
        "qualification.clock.duration" => {
            let s = text(v)?;
            let (negative, s) = s.strip_prefix('-').map(|s| (true, s)).unwrap_or((false, s));
            let p = s.split(':').collect::<Vec<_>>();
            let parsed = (|| {
                if p.len() != 3
                    || !(1..=3).contains(&p[0].len())
                    || p[1].len() != 2
                    || p[2].len() != 2
                {
                    return None;
                }
                let h = canonical_decimal(p[0])?;
                if !p[1].bytes().chain(p[2].bytes()).all(|b| b.is_ascii_digit()) {
                    return None;
                }
                let m = p[1].parse::<i64>().ok()?;
                let sec = p[2].parse::<i64>().ok()?;
                (m < 60 && sec < 60)
                    .then_some((h as i64 * 3600 + m * 60 + sec) * if negative { -1 } else { 1 })
            })();
            json!(parsed)
        }
        "qualification.clock.round_up" => {
            let n = clock(&v["clock"])?;
            let step = integer(&v["step"], 1, 1440)?;
            clock_result(((n + step - 1) / step) * step)
        }
        "qualification.telemetry.counter" => {
            let a = array(v)?
                .iter()
                .map(|n| integer(n, 0, 1000000))
                .collect::<Result<Vec<_>>>()?;
            json!(
                a.windows(2)
                    .map(|p| if p[1] >= p[0] { p[1] - p[0] } else { p[1] })
                    .sum::<i64>()
            )
        }
        "qualification.telemetry.excursions" => {
            let low = integer(&v["low"], -1000000, 1000000)?;
            let high = integer(&v["high"], low, 1000000)?;
            let a = array(&v["values"])?;
            let mut out = Vec::new();
            let mut start = None;
            for (i, n) in a.iter().enumerate() {
                let n = integer(n, -1000000, 1000000)?;
                if n < low || n > high {
                    start.get_or_insert(i);
                } else if let Some(s) = start.take() {
                    out.push([s, i]);
                }
            }
            if let Some(s) = start {
                out.push([s, a.len()]);
            }
            json!(out)
        }
        "qualification.telemetry.buckets" => {
            let w = integer(&v["width"], 1, 10000)?;
            let mut rows = BTreeMap::<i64, (i64, i64)>::new();
            for e in array(&v["events"])? {
                let p = array(e)?;
                ensure!(p.len() == 2, "time/value pair required");
                let t = integer(&p[0], -1000000, 1000000)?;
                let value = integer(&p[1], -1000000, 1000000)?;
                let row = rows.entry(t.div_euclid(w) * w).or_default();
                row.0 += 1;
                row.1 += value;
            }
            json!(
                rows.into_iter()
                    .map(|(t, (n, s))| [t, n, s])
                    .collect::<Vec<_>>()
            )
        }
        "qualification.telemetry.integral" => {
            let until = integer(&v["until"], 0, 1000000)?;
            let (mut last, mut gauge, mut area) = (0, 0, 0);
            let mut seen = false;
            for e in array(&v["readings"])? {
                let p = array(e)?;
                ensure!(p.len() == 2, "time/value pair required");
                let t = integer(&p[0], 0, until)?;
                ensure!(!seen || t > last, "strictly increasing times required");
                area += (t - last) * gauge;
                last = t;
                gauge = integer(&p[1], -1000000, 1000000)?;
                seen = true;
            }
            json!(area + (until - last) * gauge)
        }
        "qualification.address.endpoint" => {
            let s = text(v)?;
            let p = s.split(':').collect::<Vec<_>>();
            let parsed = (|| {
                if p.len() != 2 {
                    return None;
                }
                let h = p[0];
                let port = canonical_decimal(p[1])?;
                if h.is_empty()
                    || h.len() > 253
                    || !(1..=65535).contains(&port)
                    || !h.split('.').all(|l| {
                        !l.is_empty()
                            && l.len() <= 63
                            && l.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                            && l.as_bytes()[0].is_ascii_alphanumeric()
                            && l.as_bytes()[l.len() - 1].is_ascii_alphanumeric()
                    })
                {
                    return None;
                }
                Some(json!({"host":h.to_ascii_lowercase(),"port":port}))
            })();
            parsed.unwrap_or(Value::Null)
        }
        "qualification.address.cidr" => {
            let s = text(v)?;
            let parsed = (|| {
                let (a, p) = s.split_once('/')?;
                let n = ipv4(a)?;
                let p = canonical_decimal(p)?;
                if p > 32 {
                    return None;
                }
                let mask = if p == 0 { 0 } else { u32::MAX << (32 - p) };
                Some(
                    json!({"network":dotted(n&mask),"broadcast":dotted(n|!mask),"size":1_u64<<(32-p)}),
                )
            })();
            parsed.unwrap_or(Value::Null)
        }
        "qualification.address.mac" => {
            let s = text(v)?;
            let b = s.as_bytes();
            if b.len() != 17
                || !matches!(b[2], b':' | b'-')
                || !(0..17).all(|i| {
                    if i % 3 == 2 {
                        b[i] == b[2]
                    } else {
                        b[i].is_ascii_hexdigit()
                    }
                })
            {
                Value::Null
            } else {
                json!(s.to_ascii_lowercase().replace('-', ":"))
            }
        }
        "qualification.address.percent_path" => {
            let s = text(v)?;
            let b = s.as_bytes();
            let (mut out, mut i) = (String::new(), 0);
            let mut valid = true;
            while i < b.len() {
                if b[i] != b'%' {
                    out.push(b[i] as char);
                    i += 1;
                    continue;
                }
                if i + 2 >= b.len()
                    || !b[i + 1].is_ascii_hexdigit()
                    || !b[i + 2].is_ascii_hexdigit()
                {
                    valid = false;
                    break;
                }
                let byte = u8::from_str_radix(&s[i + 1..i + 3], 16)?;
                if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
                    out.push(byte as char);
                } else {
                    out.push_str(&format!("%{byte:02X}"));
                }
                i += 3;
            }
            if valid { json!(out) } else { Value::Null }
        }
        "qualification.path.search_list" => {
            let mut seen = BTreeSet::new();
            json!(
                text(v)?
                    .split(':')
                    .map(|p| if p.is_empty() { "." } else { p })
                    .filter(|p| seen.insert((*p).to_owned()))
                    .collect::<Vec<_>>()
            )
        }
        "qualification.path.glob" => {
            let p = text(&v["pattern"])?;
            let s = text(&v["name"])?;
            ensure!(
                p.len() <= 128 && s.len() <= 128 && !p.contains('/') && !s.contains('/'),
                "glob domain violated"
            );
            let mut dp = vec![false; s.len() + 1];
            dp[0] = true;
            for c in p.bytes() {
                let mut next = vec![false; s.len() + 1];
                if c == b'*' {
                    next[0] = dp[0];
                    for j in 1..=s.len() {
                        next[j] = dp[j] || next[j - 1];
                    }
                } else {
                    for j in 1..=s.len() {
                        next[j] = dp[j - 1] && (c == b'?' || c == s.as_bytes()[j - 1]);
                    }
                }
                dp = next;
            }
            json!(dp[s.len()])
        }
        "qualification.path.relative" => {
            let parse = |v: &Value| -> Result<Vec<String>> {
                let s = text(v)?;
                ensure!(
                    s.starts_with('/')
                        && (s == "/"
                            || (!s.ends_with('/')
                                && !s.contains("//")
                                && s.split('/').skip(1).all(|p| p != "." && p != ".."))),
                    "canonical absolute path required"
                );
                Ok(s.split('/')
                    .filter(|p| !p.is_empty())
                    .map(str::to_owned)
                    .collect())
            };
            let a = parse(&v["base"])?;
            let b = parse(&v["target"])?;
            let k = a.iter().zip(&b).take_while(|(a, b)| a == b).count();
            let parts = std::iter::repeat_n("..", a.len() - k)
                .chain(b[k..].iter().map(String::as_str))
                .collect::<Vec<_>>();
            json!(if parts.is_empty() {
                ".".into()
            } else {
                parts.join("/")
            })
        }
        "qualification.path.rotations" => {
            let stem = text(&v["stem"])?;
            ensure!(
                !stem.is_empty() && !stem.contains('/'),
                "filename stem required"
            );
            let prefix = format!("{stem}.");
            let mut active = false;
            let mut rotations = BTreeSet::new();
            for f in array(&v["files"])? {
                let f = text(f)?;
                active |= f == stem;
                if let Some(n) = f
                    .strip_prefix(&prefix)
                    .and_then(canonical_decimal)
                    .filter(|n| (1..=999).contains(n))
                {
                    rotations.insert(n);
                }
            }
            json!({"active":active,"rotations":rotations})
        }
        _ => bail!("unknown qualification task {id}"),
    })
}

/// Independently written expected constants catch oracle mistakes as well as
/// candidate bugs. None of these private inputs enters the provider prompt.
pub fn private_cases(task: &Task) -> Result<Vec<Case>> {
    let pairs: Vec<(Value, Value)> = match task.id.as_str() {
        "qualification.clock.shift" => vec![
            (
                json!({"clock":"00:00","delta":-1}),
                json!({"clock":"23:59","day_offset":-1}),
            ),
            (
                json!({"clock":"12:30","delta":-10080}),
                json!({"clock":"12:30","day_offset":-7}),
            ),
            (
                json!({"clock":"23:59","delta":1}),
                json!({"clock":"00:00","day_offset":1}),
            ),
            (
                json!({"clock":"00:00","delta":0}),
                json!({"clock":"00:00","day_offset":0}),
            ),
        ],
        "qualification.clock.window" => vec![
            (
                json!({"start":"08:00","end":"09:00","query":"09:00"}),
                json!(false),
            ),
            (
                json!({"start":"08:00","end":"09:00","query":"08:00"}),
                json!(true),
            ),
            (
                json!({"start":"22:00","end":"02:00","query":"02:00"}),
                json!(false),
            ),
            (
                json!({"start":"00:00","end":"00:00","query":"12:00"}),
                json!(false),
            ),
            (
                json!({"start":"22:00","end":"02:00","query":"21:59"}),
                json!(false),
            ),
        ],
        "qualification.clock.duration" => vec![
            (json!("0:00:00"), json!(0)),
            (json!("-0:00:00"), json!(0)),
            (json!("999:59:59"), json!(3599999)),
            (json!("01:00:00"), Value::Null),
            (json!("1:60:00"), Value::Null),
            (json!("+1:00:00"), Value::Null),
            (json!("1:00:0"), Value::Null),
            (json!(""), Value::Null),
        ],
        "qualification.clock.round_up" => vec![
            (
                json!({"clock":"00:00","step":1440}),
                json!({"clock":"00:00","day_offset":0}),
            ),
            (
                json!({"clock":"00:01","step":1440}),
                json!({"clock":"00:00","day_offset":1}),
            ),
            (
                json!({"clock":"23:59","step":1000}),
                json!({"clock":"09:20","day_offset":1}),
            ),
            (
                json!({"clock":"12:00","step":15}),
                json!({"clock":"12:00","day_offset":0}),
            ),
        ],
        "qualification.telemetry.counter" => vec![
            (json!([]), json!(0)),
            (json!([999]), json!(0)),
            (json!([5, 5, 0, 0, 7]), json!(7)),
            (json!([10, 3, 1, 4]), json!(7)),
            (json!([0, 1000000, 0, 1000000]), json!(2000000)),
        ],
        "qualification.telemetry.excursions" => vec![
            (json!({"values":[],"low":0,"high":0}), json!([])),
            (json!({"values":[1,2,3],"low":1,"high":3}), json!([])),
            (
                json!({"values":[-1,0,1,-1],"low":0,"high":0}),
                json!([[0, 1], [2, 4]]),
            ),
            (json!({"values":[8,9],"low":-1,"high":1}), json!([[0, 2]])),
        ],
        "qualification.telemetry.buckets" => vec![
            (json!({"width":5,"events":[]}), json!([])),
            (
                json!({"width":3,"events":[[-4,1],[-3,2],[-2,3],[-1,4],[0,5]]}),
                json!([[-6, 1, 1], [-3, 3, 9], [0, 1, 5]]),
            ),
            (
                json!({"width":1,"events":[[2,7],[2,-7],[1,4]]}),
                json!([[1, 1, 4], [2, 2, 0]]),
            ),
        ],
        "qualification.telemetry.integral" => vec![
            (json!({"until":9,"readings":[]}), json!(0)),
            (json!({"until":0,"readings":[[0,7]]}), json!(0)),
            (json!({"until":10,"readings":[[10,100]]}), json!(0)),
            (
                json!({"until":8,"readings":[[0,-2],[3,5],[7,0]]}),
                json!(14),
            ),
            (json!({"until":5,"readings":[[2,4]]}), json!(12)),
        ],
        "qualification.address.endpoint" => vec![
            (json!("x:1"), json!({"host":"x","port":1})),
            (json!("A-B.C:65535"), json!({"host":"a-b.c","port":65535})),
            (json!("a:01"), Value::Null),
            (json!("a.:80"), Value::Null),
            (json!("-a:80"), Value::Null),
            (json!("a_b:80"), Value::Null),
            (json!("a:65536"), Value::Null),
            (json!("[::1]:80"), Value::Null),
        ],
        "qualification.address.cidr" => vec![
            (
                json!("255.255.255.255/0"),
                json!({"network":"0.0.0.0","broadcast":"255.255.255.255","size":4294967296_i64}),
            ),
            (
                json!("1.2.3.4/32"),
                json!({"network":"1.2.3.4","broadcast":"1.2.3.4","size":1}),
            ),
            (
                json!("10.0.0.3/31"),
                json!({"network":"10.0.0.2","broadcast":"10.0.0.3","size":2}),
            ),
            (json!("01.2.3.4/24"), Value::Null),
            (json!("1.2.3.4/33"), Value::Null),
            (json!("1.2.3.4/00"), Value::Null),
        ],
        "qualification.address.mac" => vec![
            (json!("00:00:00:00:00:00"), json!("00:00:00:00:00:00")),
            (json!("FF:AB:CD:EF:01:23"), json!("ff:ab:cd:ef:01:23")),
            (json!("aa:bb-cc:dd:ee:ff"), Value::Null),
            (json!("a:bb:cc:dd:ee:ff"), Value::Null),
            (json!("gg:bb:cc:dd:ee:ff"), Value::Null),
            (json!(""), Value::Null),
        ],
        "qualification.address.percent_path" => vec![
            (json!(""), json!("")),
            (json!("%2f%00%FF%20"), json!("%2F%00%FF%20")),
            (json!("%2D%2e%5f%7E%30%7a"), json!("-._~0z")),
            (json!("a+b"), json!("a+b")),
            (json!("abc%"), Value::Null),
            (json!("%0x"), Value::Null),
            (json!("%252F"), json!("%252F")),
        ],
        "qualification.path.search_list" => vec![
            (json!(""), json!(["."])),
            (json!(":::"), json!(["."])),
            (json!("a:A:a"), json!(["a", "A"])),
            (json!("/a:/a/:/a/../a"), json!(["/a", "/a/", "/a/../a"])),
            (json!(".:x::x:y"), json!([".", "x", "y"])),
        ],
        "qualification.path.glob" => vec![
            (json!({"pattern":"","name":""}), json!(true)),
            (json!({"pattern":"*","name":""}), json!(true)),
            (json!({"pattern":"?","name":""}), json!(false)),
            (json!({"pattern":"a**b","name":"ab"}), json!(true)),
            (json!({"pattern":"[a]","name":"a"}), json!(false)),
            (json!({"pattern":"*a?b","name":"aaab"}), json!(true)),
            (json!({"pattern":"a*b","name":"ac"}), json!(false)),
        ],
        "qualification.path.relative" => vec![
            (json!({"base":"/","target":"/"}), json!(".")),
            (json!({"base":"/a/b","target":"/a/b"}), json!(".")),
            (json!({"base":"/a/b","target":"/"}), json!("../..")),
            (json!({"base":"/","target":"/x/y"}), json!("x/y")),
            (json!({"base":"/a","target":"/ab"}), json!("../ab")),
        ],
        "qualification.path.rotations" => vec![
            (
                json!({"stem":"x","files":[]}),
                json!({"active":false,"rotations":[]}),
            ),
            (
                json!({"stem":"x","files":["x.0","x.01","x.-1","x.1000","x.1.tmp","yx.2"]}),
                json!({"active":false,"rotations":[]}),
            ),
            (
                json!({"stem":"a.b","files":["a.b.999","a.b.2","a.b.2","a.b"]}),
                json!({"active":true,"rotations":[2,999]}),
            ),
        ],
        _ => bail!("unknown qualification task {}", task.id),
    };
    Ok(pairs
        .into_iter()
        .enumerate()
        .map(|(i, (input, expected))| Case {
            input,
            expected,
            label: format!("qualification-private-{i}"),
        })
        .collect())
}

/// Trusted reference programmes contain algorithms only, never private inputs.
/// They are compiled/executed solely by the ordinary isolated sandbox QA.
pub fn reference_source(id: &str) -> Result<String> {
    let body = match id {
        "qualification.clock.shift" => {
            r#"let n=minute(&input["clock"])+input["delta"].as_i64().unwrap(); clock_value(n)"#
        }
        "qualification.clock.window" => {
            r#"let a=minute(&input["start"]);let b=minute(&input["end"]);let q=minute(&input["query"]);json!((q-a).rem_euclid(1440)<(b-a).rem_euclid(1440))"#
        }
        "qualification.clock.duration" => {
            r#"let raw=input.as_str().unwrap();let neg=raw.starts_with('-');let s=if neg{&raw[1..]}else{raw};let p=s.split(':').collect::<Vec<_>>();if p.len()!=3||p[0].is_empty()||p[0].len()>3||p[1].len()!=2||p[2].len()!=2{return Value::Null;}let Some(h)=decimal(p[0])else{return Value::Null;};if !p[1].bytes().chain(p[2].bytes()).all(|b|b.is_ascii_digit()){return Value::Null;}let m=p[1].parse::<i64>().unwrap();let s=p[2].parse::<i64>().unwrap();if m>=60||s>=60{return Value::Null;}json!((h as i64*3600+m*60+s)*if neg{-1}else{1})"#
        }
        "qualification.clock.round_up" => {
            r#"let n=minute(&input["clock"]);let d=input["step"].as_i64().unwrap();clock_value(n+(-n).rem_euclid(d))"#
        }
        "qualification.telemetry.counter" => {
            r#"let a=input.as_array().unwrap();let mut sum=0;for w in a.windows(2){let p=w[0].as_i64().unwrap();let n=w[1].as_i64().unwrap();sum+=if n<p{n}else{n-p};}json!(sum)"#
        }
        "qualification.telemetry.excursions" => {
            r#"let a=input["values"].as_array().unwrap();let lo=input["low"].as_i64().unwrap();let hi=input["high"].as_i64().unwrap();let mut out=Vec::new();let mut i=0;while i<a.len(){let x=a[i].as_i64().unwrap();if (lo..=hi).contains(&x){i+=1;continue;}let start=i;i+=1;while i<a.len()&&!(lo..=hi).contains(&a[i].as_i64().unwrap()){i+=1;}out.push([start,i]);}json!(out)"#
        }
        "qualification.telemetry.buckets" => {
            r#"let w=input["width"].as_i64().unwrap();let mut out=std::collections::BTreeMap::<i64,[i64;2]>::new();for e in input["events"].as_array().unwrap(){let t=e[0].as_i64().unwrap();let start=t-t.rem_euclid(w);let x=out.entry(start).or_insert([0,0]);x[0]+=1;x[1]+=e[1].as_i64().unwrap();}json!(out.into_iter().map(|(t,[n,s])|[t,n,s]).collect::<Vec<_>>())"#
        }
        "qualification.telemetry.integral" => {
            r#"let r=input["readings"].as_array().unwrap();let until=input["until"].as_i64().unwrap();let mut total=0;for(i,p)in r.iter().enumerate(){let end=if i+1<r.len(){r[i+1][0].as_i64().unwrap()}else{until};total+=(end-p[0].as_i64().unwrap())*p[1].as_i64().unwrap();}json!(total)"#
        }
        "qualification.address.endpoint" => {
            r#"let s=input.as_str().unwrap();let p=s.split(':').collect::<Vec<_>>();if p.len()!=2{return Value::Null;}let h=p[0];let Some(n)=decimal(p[1])else{return Value::Null;};if h.is_empty()||h.len()>253||n==0||n>65535{return Value::Null;}for l in h.split('.'){let b=l.as_bytes();if b.is_empty()||b.len()>63||!b[0].is_ascii_alphanumeric()||!b[b.len()-1].is_ascii_alphanumeric()||!b.iter().all(|c|c.is_ascii_alphanumeric()||*c==b'-'){return Value::Null;}}json!({"host":h.to_ascii_lowercase(),"port":n})"#
        }
        "qualification.address.cidr" => {
            r#"let s=input.as_str().unwrap();let Some((a,p))=s.split_once('/')else{return Value::Null;};let Some(prefix)=decimal(p).filter(|n|*n<=32)else{return Value::Null;};let parts=a.split('.').collect::<Vec<_>>();if parts.len()!=4{return Value::Null;}let mut ip=0_u32;for p in parts{let Some(n)=decimal(p).filter(|n|*n<=255)else{return Value::Null;};ip=(ip<<8)|n;}let size=1_u64<<(32-prefix);let network=(ip as u64/size)*size;let show=|n:u64|std::net::Ipv4Addr::from(n as u32).to_string();json!({"network":show(network),"broadcast":show(network+size-1),"size":size})"#
        }
        "qualification.address.mac" => {
            r#"let s=input.as_str().unwrap();let b=s.as_bytes();if b.len()!=17{return Value::Null;}let sep=b[2];if sep!=b':'&&sep!=b'-'{return Value::Null;}let p=s.split(sep as char).collect::<Vec<_>>();if p.len()!=6||!p.iter().all(|p|p.len()==2&&p.bytes().all(|b|b.is_ascii_hexdigit())){return Value::Null;}json!(p.join(":").to_ascii_lowercase())"#
        }
        "qualification.address.percent_path" => {
            r#"let s=input.as_str().unwrap();let mut chars=s.chars();let mut out=String::new();while let Some(c)=chars.next(){if c!='%'{out.push(c);continue;}let Some(a)=chars.next()else{return Value::Null;};let Some(b)=chars.next()else{return Value::Null;};let(Some(x),Some(y))=(a.to_digit(16),b.to_digit(16))else{return Value::Null;};let n=(x*16+y)as u8;if n.is_ascii_alphanumeric()||b"-._~".contains(&n){out.push(n as char);}else{out.push_str(&format!("%{n:02X}"));}}json!(out)"#
        }
        "qualification.path.search_list" => {
            r#"let mut out=Vec::<String>::new();for p in input.as_str().unwrap().split(':'){let p=if p.is_empty(){"."}else{p};if !out.iter().any(|x|x==p){out.push(p.to_owned());}}json!(out)"#
        }
        "qualification.path.glob" => {
            r#"let p=input["pattern"].as_str().unwrap().as_bytes();let s=input["name"].as_str().unwrap().as_bytes();let mut table=vec![vec![false;s.len()+1];p.len()+1];table[0][0]=true;for i in 1..=p.len(){for j in 0..=s.len(){table[i][j]=if p[i-1]==b'*'{table[i-1][j]||(j>0&&table[i][j-1])}else{j>0&&table[i-1][j-1]&&(p[i-1]==b'?'||p[i-1]==s[j-1])};}}json!(table[p.len()][s.len()])"#
        }
        "qualification.path.relative" => {
            r#"let a=input["base"].as_str().unwrap().split('/').filter(|x|!x.is_empty()).collect::<Vec<_>>();let b=input["target"].as_str().unwrap().split('/').filter(|x|!x.is_empty()).collect::<Vec<_>>();let mut same=0;while same<a.len()&&same<b.len()&&a[same]==b[same]{same+=1;}let mut out=vec!["..";a.len()-same];out.extend_from_slice(&b[same..]);json!(if out.is_empty(){".".to_owned()}else{out.join("/")})"#
        }
        "qualification.path.rotations" => {
            r#"let stem=input["stem"].as_str().unwrap();let prefix=format!("{stem}.");let mut active=false;let mut values=Vec::new();for f in input["files"].as_array().unwrap(){let f=f.as_str().unwrap();if f==stem{active=true;}if let Some(s)=f.strip_prefix(&prefix){if let Some(n)=decimal(s){if n>=1&&n<=999&&!values.contains(&n){values.push(n);}}}}values.sort();json!({"active":active,"rotations":values})"#
        }
        _ => bail!("no qualification reference for {id}"),
    };
    Ok(format!(
        r#"use serde_json::{{json, Value}};
fn decimal(s:&str)->Option<u32>{{if s.is_empty()||!s.bytes().all(|b|b.is_ascii_digit())||(s.len()>1&&s.starts_with('0')){{None}}else{{s.parse().ok()}}}}
fn minute(v:&Value)->i64{{let s=v.as_str().unwrap();s[..2].parse::<i64>().unwrap()*60+s[3..].parse::<i64>().unwrap()}}
fn clock_value(n:i64)->Value{{json!({{"clock":format!("{{:02}}:{{:02}}",n.rem_euclid(1440)/60,n.rem_euclid(60)),"day_offset":n.div_euclid(1440)}})}}
pub fn solve(input:Value)->Value{{{body}}}
"#
    ))
}

/// Persist the complete pack before any candidate qualification outcome exists.
/// Existing identical freezes resume; a changed pack requires a fresh directory.
pub fn freeze(path: &Path) -> Result<String> {
    let tasks = corpus();
    let mut private = BTreeMap::new();
    let mut reference = BTreeMap::new();
    for task in &tasks {
        for e in &task.public_examples {
            ensure!(
                oracle(&task.id, &e.input)? == e.expected,
                "qualification public oracle mismatch: {}",
                task.id
            );
        }
        let cases = private_cases(task)?;
        for e in &cases {
            ensure!(
                oracle(&task.id, &e.input)? == e.expected,
                "qualification private oracle mismatch: {} {}",
                task.id,
                e.label
            );
        }
        reference.insert(task.id.clone(), tasks::hash(reference_source(&task.id)?));
        private.insert(task.id.clone(), cases);
    }
    let manifest = json!({"schema_version":1,"pack_id":PACK_ID,"purpose":"exploratory_model_selection_only","author_and_lineage":AUTHOR,"source_sha256":tasks::hash(include_bytes!("qualification.rs")),"tasks":tasks,"private_cases":private,"reference_source_sha256":reference,"promotable":false});
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut file) => {
            file.write_all(&bytes)?;
            file.sync_all()?;
            if let Some(parent) = path.parent() {
                fs::File::open(parent)?.sync_all()?;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => ensure!(
            fs::read(path)? == bytes,
            "qualification freeze differs; start a fresh search output"
        ),
        Err(e) => return Err(e.into()),
    }
    Ok(tasks::hash(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_and_private_expectations_and_references_cover_four_disjoint_sources() {
        let tasks = corpus();
        assert_eq!(tasks.len(), 16);
        let sources = tasks.iter().map(|t| &t.source_id).collect::<BTreeSet<_>>();
        assert_eq!(sources.len(), 4);
        let original = tasks::corpus()
            .into_iter()
            .map(|t| t.source_id)
            .collect::<BTreeSet<_>>();
        for source in sources {
            assert!(!original.contains(source));
        }
        for task in tasks {
            assert_eq!(task.split, Split::Development);
            for e in &task.public_examples {
                assert_eq!(
                    oracle(&task.id, &e.input).unwrap(),
                    e.expected,
                    "{} public",
                    task.id
                );
            }
            let cases = private_cases(&task).unwrap();
            assert!(cases.len() >= 3);
            for e in cases {
                assert_eq!(
                    oracle(&task.id, &e.input).unwrap(),
                    e.expected,
                    "{} {}",
                    task.id,
                    e.label
                );
            }
            syn::parse_file(&reference_source(&task.id).unwrap()).unwrap();
        }
    }
    #[test]
    fn freeze_is_immutable_and_reusable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pack.json");
        let first = freeze(&path).unwrap();
        assert_eq!(freeze(&path).unwrap(), first);
        fs::write(&path, b"changed").unwrap();
        assert!(freeze(&path).is_err());
    }
    #[test]
    fn invalid_structural_domains_fail_before_typed_indexing() {
        for (id, input) in [
            (
                "qualification.clock.shift",
                json!({"clock":"99:00","delta":0}),
            ),
            (
                "qualification.telemetry.buckets",
                json!({"width":0,"events":[]}),
            ),
            (
                "qualification.telemetry.integral",
                json!({"until":4,"readings":[[3,1],[2,2]]}),
            ),
            ("qualification.path.glob", json!({"pattern":"/","name":"x"})),
            (
                "qualification.path.relative",
                json!({"base":"/a/../b","target":"/"}),
            ),
        ] {
            assert!(oracle(id, &input).is_err(), "{id}");
        }
    }
}
