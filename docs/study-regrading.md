# Regrading retained local study generations

`study-regrade` imports complete original train/development evaluations and
grades their retained outputs under a separately frozen host policy. It does
not load a model, generate another answer, access the sealed held-out split or
make a promotion decision. This is useful when a host-controller defect makes
original grading unhealthy while the original generation records remain valid.

The command is implemented in Rust. The shell launcher only selects the pinned
Rust executable, as justified in [deviations.md](deviations.md).

For the declared base/SFT/composed cohort, supply all six original directories:

```sh
./run.sh study-regrade --offline \
  --original reports/local-study/ORIGINAL/base-development \
  --original reports/local-study/ORIGINAL/sft-development \
  --original reports/local-study/ORIGINAL/composed-development \
  --original reports/local-study/ORIGINAL/base-train \
  --original reports/local-study/ORIGINAL/sft-train \
  --original reports/local-study/ORIGINAL/composed-train \
  --expected-generations 192 \
  --maximum-concurrent-graders 4 \
  --output reports/local-study/regraded-NEW
```

Replace the example paths with actual original directories. No measured result
is implied by this example. A new grading protocol needs an explicitly new
output. Resume by repeating the exact command; the launcher resolves the frozen
**grading** executable from `batch-plan.json`, independently of the original
generation executables. `--fresh` creates a new sibling output while retaining
the previous evidence.

Before launching a container, the importer checks the complete declared
denominator, original engine, checkpoint/base/tokeniser bytes, runtime receipts,
decoding request and every raw output. It verifies tensor headers and extents
without allocating tensor storage. Train and development remain distinct. An
original output retains its original `plan_sha256`; derived wrappers separately
bind the new engine, controller, policy, image, preflight and code extractor.
The new grading image may differ from the original image. Every wrapper in the
batch uses the same new grading identity.

Original errors, empty answers, token-limit truncations and timeouts are durable
zero rows. They are never compiled or replaced. Complete nonempty answers are
submitted to bounded concurrent containers. Concurrency is capped by the
declared ceiling, available CPU threads and guarded RAM. Blocking evidence
reads, hashing and receipt validation run outside asynchronous executor threads.

A prepared receipt is fsynced before launch and binds a unique run ID, ownership
nonce and deterministic attempt key to the exact task, code and configuration.
Recovery adopts that exact receipt. A complete report is reusable only after
independent verification of its checksum-bound audit, lifecycle, ownership and
output drains. A launch journal without a complete verified terminal report is
an interrupted infrastructure failure; resumption never launches that identity
again. A final observation is published atomically and is reverified on reuse.

A trusted compiler failure counts as a candidate zero only when the complete
controller evidence verifies. Missing or invalid audit/lifecycle evidence remains
an infrastructure error. Original files and verdicts are never overwritten, and
output/input overlap, including existing symlink ancestors, is rejected before
creating the regrade directory.

To reconstruct the complete batch without launching a container:

```sh
./run.sh study-regrade --offline --verify-only \
  --output reports/local-study/regraded-NEW
```

`batch-report.json` is published only after every original denominator entry has
a verified derived observation. A batch retaining an original generation error
or a new infrastructure error saves the complete evidence and exits with failure.
The pinned Rust executable also provides read-only `study-verify` and
`study-compare` commands. Invoke the grading executable recorded in
`batch-plan.json` directly, rather than relying on an older bundled application:

```sh
state/engines/GRADING_ENGINE_SHA/lec-distil study-verify \
  --evaluation reports/local-study/regraded-NEW/evaluation-000-development-CHECKPOINT_PREFIX
state/engines/GRADING_ENGINE_SHA/lec-distil study-compare \
  --baseline reports/local-study/regraded-NEW/evaluation-000-development-BASE_PREFIX \
  --candidate reports/local-study/regraded-NEW/evaluation-001-development-SFT_PREFIX
```

Replace the uppercase path components with the frozen executable hash and actual
child names. These commands print their reports and return a nonzero status for
an unhealthy evaluation or invalid quality comparison. They understand derived
child directories and verify the entire completed batch before returning a view.
Comparisons require matching
original generation settings and identical effective grading identities; original
and replacement grading cannot be mixed silently. A healthy development cohort
can support a descriptive comparison even when retained train generation errors
keep the full batch unhealthy. Every report states zero new model calls/tokens
and labels the original inference timings as reused measurements.

Unit QA covers complete denominator import, duplicate/order/protocol rejection,
immutable raw/request/plan bindings, retained failure rows, output overlap,
prepared-receipt adoption, interrupted launch refusal and exact launcher routing.
These synthetic tests do not establish real container enforcement or model
quality; real controller QA and a separately verified full regrade are required.
