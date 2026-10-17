# Read-only review of teacher-search round 002

The complete source evidence verifies. The bounded search correctly reports
`cached_only_shortfall`, with no qualified models against its two-model target.
Exit 1 is the explicit unmet-quality-target outcome, not a successful search.

| Original source | Complete development result | Generation/infrastructure errors | Interpretation |
| --- | ---: | ---: | --- |
| Qwen2.5-Coder-7B, Q4, imported from round 001 | 7/16 | 0 | Strongest measured retained fallback; below 80% |
| DeepSeek-Coder-6.7B, Q4 | 0/16 | 1 | Inconclusive error-bearing comparison; retained |
| Qwen3-4B-Instruct-2507, Q4 | 3/16 | 0 | Verified below threshold; owned inactive weights pruned |

`verification.json` reconstructs all three complete comparative reports using
the current public Rust verification API, including their immutable generation,
task, grader and sandbox audit dependencies. All 48 planned observations remain
in their denominators. DeepSeek's one error is `tables.group_sums`: generation
reached the declared 512-token cap with `finish_reason: length` and
`complete: false`. It is not a positive sample. Qwen3 passed `money.basket`,
`dispatch.finish_order` and `dispatch.waiting`.

The independent fresh 16-task qualification pack exists and is frozen. No trial
qualified for its dispatch: `src/teacher_search/selection.rs` requires an
error-free original score of at least 80% (13/16). Therefore all fresh-pack
scores are absent, rather than zero or silently successful. No assessment
held-out tasks were opened, no checkpoint was promoted and no SHIP is claimed.

The pruning API independently verifies Qwen3's owned deletion authority, pinned
publisher revision, original asset manifest and current weight absence. Its
2,497,280,736 logical weight bytes were removed, with 2,497,286,144 allocated
bytes recorded by the deletion event. These are file-accounting observations,
not a measured change in filesystem free space. Licences and metadata remain.
The earlier failed Qwen3 attempt remains explicitly linked in the search event
history and is inventoried in the verification output.

The search used explicit cached-only mode with zero download reservation.
It did not make paid provider calls; local compute, storage, bandwidth and
energy cash costs remain unmeasured. Locally hosted original model families
do not satisfy the assessment's three independent real-provider businesses.

The review used a standalone Rust helper linked against existing reviewed
libraries with `serde_json/float_roundtrip`, limited to CPUs 0 and 1. It did not
run Cargo, inference, GPU work, containers, downloads, grading or pruning.
Original reports, policies, freezes and binaries were not changed. The helper
source and executable are retained alongside its output; source artefacts are
inventoried with hashes and lengths in `verification.json`.
