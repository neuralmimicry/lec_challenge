# Fresh development qualification pack

The original 16 development tasks are already visible to the teacher-selection
process. The separate `fresh-development-qualification-v1` pack adds 16 contracts
from four newly authored lineage groups. It supplements the original selection
pack and never enters the sealed held-out evaluation or promotion gate.

| Source group | Four contracts |
| --- | --- |
| Clock arithmetic | Signed clock shifts, cyclic half-open windows, strict signed duration parsing, upward clock rounding |
| Telemetry | Counter resets, threshold excursions, negative-time buckets, piecewise gauge integration |
| Address strings | Strict host/port parsing, IPv4 CIDR boundaries, MAC canonicalisation, selective percent decoding |
| Path strings | Stable search-list deduplication, whole-string glob matching, lexical relative paths, numbered file rotations |

All address and path operations process strings only; they perform no network or
filesystem access. The task protocol accepts and returns JSON through
`pub fn solve(input: serde_json::Value) -> serde_json::Value`, using Rust `std`
and `serde_json`. Original source IDs and the original 96-task catalogue are
unchanged. The new tasks explicitly use `Split::Development` with distinct source
and lineage IDs. Source grouping records derivation within this pack; a single
engineering agent authored all four groups, and familiar algorithms may already
be represented in model pretraining.

`teacher_search::qualification::freeze(path)` writes an immutable manifest before
any candidate's new primary development result. It contains the complete public
contracts, author/lineage statements, explicit private inputs and independently
written expected values, Rust oracle-source hash and trusted-reference source
hashes. An identical existing freeze resumes; different bytes fail closed. The
manifest is host-owned. `Task::prompt()` receives public examples only, and the
candidate container does not receive hidden expectations or oracle code.

Qualification requires complete observations, zero infrastructure/generation
errors and at least 80% on **each** 16-task pack: at least 13/16 primary and
13/16 fresh tasks. A candidate below the primary threshold does not consume a
fresh trial. Empty, truncated and errored outputs remain failures in their
planned denominators. These thresholds are exploratory selection rules, with
no statistical claim of superiority and no authority to ship a student.

The pack's unit checks verify all hand-authored public/private expectations
against the typed oracle, source separation, domain rejection, syntactically
valid reference source and immutable freeze behaviour:

```sh
cargo test --locked --offline --lib teacher_search::qualification
```

Run all trusted references in fresh isolated containers with the native Rust
example; it uses two bounded graders, checks the real sandbox controls, saves
every observation and creates a fresh UUID report directory by default:

```sh
cargo run --locked --offline --example qualification_qa
```

The [recorded container QA](../reports/comparative/search/qualification-qa/2bf2d6ec-c5e6-4233-87e0-88a58525a287/report.json)
passed all 16 reference programmes and 101 public/private cases in approximately
124 seconds. Every execution had complete trusted traces, and none was
quarantined. The exact pack SHA-256 is
`ae71a3cd82a34a94b2ce761ad5c223a246b46ed8bad8ce4f4ccfd6d64ffb75c1`.
This verifies the grading contracts and reference implementations; it reports
neither model quality nor training improvement.
