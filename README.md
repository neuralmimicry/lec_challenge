# Coding-fleet distillation in Rust

A native Rust implementation of the LEC Principal AI Engineer assessment: configurable teacher acquisition, durable budgets, isolated executable grading, source-separated curation, real differentiable Qwen LoRA, checkpoint evaluation and conservative promotion.

**The assessment is not yet complete.** No teacher API credentials, spending authority or account-specific API output-training decisions were supplied. Hugging Face access was subsequently authorised for local comparative weights, without saving credentials. There are no real teacher API recordings or provider-trained assessment checkpoints. The offline demonstration fails explicitly when these required artifacts are absent. Local transport fixtures, random small-model tests and full-size synthetic tensor training are labelled QA evidence; they do not establish learning from teachers. See [STATUS.md](STATUS.md) and [limitations](docs/limitations.md).

## Run

The required assessment entry point is:

```sh
./run.sh demo --offline
```

The additional local research workflow is `./run.sh` (or `./run.sh explore
--offline` for cached-only execution). It resumes teacher search, data collection
and bounded method/student studies, recording gaps and a nonzero result whenever
the declared quality target or required evidence is unmet. Its default maximum
wall time is 24 hours; use `--maximum-run-seconds` to set a shorter invocation.
Interrupted work remains available to the next invocation.

It prints phase results, saves partial evidence and exits nonzero while required recordings/checkpoints are missing. `./run.sh drill --offline` executes independent assertions and reports every required drill, including incomplete actual-checkpoint cases. An honest NO SHIP from a complete valid evaluation is a completed decision; missing evidence is a failure.

The supported machine is Linux x86-64 with enforced cgroup v2 CPU/memory/swap/PID controls and native Podman or a verified Docker-compatible runtime. The measured workstation has 16 logical CPUs, about 60 GiB RAM and an 8 GiB RTX 2080. The assessment path defaults to CPU float32; the additional comparative engine has verified CUDA float16 execution. CUDA is not required for the assessment commands. Allow several GB for model/runtime assets and additional space for Cargo builds. See [reproduction](docs/reproduction.md) for exact prerequisites and measured timings.

For a source checkout, install the pinned Rust toolchain, perform an explicit initial online dependency build, then use the offline launcher:

```sh
cargo build --locked --release
./run.sh bootstrap-online
./run.sh tasks
./run.sh sandbox-preflight
cargo test --locked --offline
```

`bootstrap-online` downloads the checksum-pinned Apache-2.0 Qwen base and builds the sandbox image. It is separate from offline replay. A complete disconnected delivery also needs the binary, base/tokeniser, saved OCI image and real recordings; the source checkout alone is not that bundle.

## Review the implementation

| Boundary | Rust modules | Important invariant |
| --- | --- | --- |
| Teacher acquisition | `fleet`, `ledger` | Reserve worst-case nanodollar liabilities atomically before every call; unknown acceptance retains the reservation |
| Corpus and grading | `tasks`, `sandbox` | 96 contracts, 48/16/32 split across complete source groups; host owns expected answers and complete denominators |
| Candidate audit | `sandbox/audit.rs` | Parent-owned syscall trace; denied attempts quarantine even when candidate catches the error |
| Training data | `curate` | Immutable held-out lineage/index and real provider/controller evidence verified before opening optimiser |
| Model adaptation | `model`, `train` | Frozen 0.5B base; differentiable q/v LoRA, assistant-only targets, finite gradients and resumable AdamW |
| Decision | `evaluate`, `gate` | Complete checkpoint/request-bound outcomes; source t and cluster bootstrap lower bounds both exceed +2 points |
| Serving | `serve` | Bounded dedicated model worker; immutable checkpoint IDs; verified gate before production canary; atomic rollback |
| Orchestration | `main`, `drills`, `artefact` | Live progress, durable reports, explicit failures, fixtures separated from required real evidence |

The frozen [three-page design](docs/design.pdf), [editable design](docs/design.md), [deviations](docs/deviations.md), [acceptance checklist](docs/acceptance.md) and [agent plan](agents/PLAN.md) explain the choices. The first assessment commit is `74eba150f766a0d93c2a8ad18066daac562314cf`; the repository's earlier initial README commit is preserved.

## Commands and verified examples

```sh
# Real pinned base inference, independent of absent teacher credentials.
./run.sh infer --prompt 'Write a Rust function adding two i64 values.' --max-tokens 128

# Actual base or adapter behind a bounded localhost endpoint.
./run.sh serve

# Collect only after config/model/price/terms review and explicit spend authority.
./run.sh collect --live --authorised-usd 5.00 --round round-001
./run.sh collect --offline --round replay-001
./run.sh curate
./run.sh train
./run.sh train --config config/training-seed-control.toml --output checkpoints/seed-control

# Full checkpoint evaluations, or complete verified cached evaluation manifests.
./run.sh gate --incumbent base --candidate checkpoints/candidate
./run.sh power --quick
./run.sh costs
```

The live command shown is an example, not authorisation to spend. The default roster is disabled and unpriced until the applicable provider/model/account terms and price schedule are reviewed. Read [provider decisions](docs/rights.md), [fleet operations](docs/fleet.md), [sandbox](docs/sandbox.md), [training](docs/training.md) and [statistics](docs/statistics.md).

Teachers may be beaten on this narrow distribution by combining complementary verified solutions; this implementation makes no claim of general frontier superiority. The small and jointly authored held-out source collection substantially limits what an improvement claim can establish.

## Delivery evidence

Generated records are in `reports/`: drill JSON/Markdown, evaluation, costs, host/runtime probes, real base inference and full-size gradient QA. Raw artifacts, request hashes and tensor hashes connect reports to their evidence. [Reproduction](docs/reproduction.md) distinguishes local verification, a genuine provider round and the required fresh disconnected rehearsal. [Video notes](docs/video.md) describe the evidence walkthrough and remaining submission steps.

Local verification passed **81 tests**, formatting and strict all-target Clippy; see [captured verification](reports/verification.json). The [fresh-archive offline QA](reports/disconnected-qa.json) ran in a private network/mount namespace with the original checkout hidden, no provider keys, no outbound routes and fresh application state/cache. It served the real base and passed all four independent required drills in 88.55 seconds. All separate invariant assertions passed. The assessment itself correctly returned `1` / `INCOMPLETE`. The host Podman image cache was preloaded; this is local offline QA, not a completed provider-trained rehearsal.

`./run.sh package` produces ZIP and tar.gz deliveries with the prebuilt executable, pinned base/tokeniser, saved sandbox image, original notices, source and reports. The generated archives are about 1.4 GB each; their exact hashes are in `deliverables/SHA256SUMS`. The offline QA records the initially tested archive hash. Final packaging adds its evidence and these notes; its runtime inventory must still match `verified_input_asset_inventory_sha256` in the QA report.

Application and candidate logic, model adaptation, statistics and tests are Rust. The shell launcher and OCI/CI build recipes invoke native tools; structured configuration and documentation are not alternative application implementations. No Python trainer or grader is required.

## Local GPU teachers

The optional native CUDA engine now runs on this laptop's 8 GiB RTX 2080 with explicit float16 execution and no CPU fallback. The [pinned comparative roster](config/comparative-models.toml) adds Qwen2.5-Coder-1.5B, DeepSeek-Coder-1.3B and an optional SmolLM2-1.7B general instruction control. The [GPU guide](docs/gpu.md) explains the measured native-toolchain compatibility fix, one-model-at-a-time hosting and actual evidence. The [Hub downloader](docs/hub.md) reads tokens only from ephemeral stdin/environment, verifies publisher hashes and stores no credentials.

All three initial alternatives were downloaded and publisher-verified: 9,203,214,224 weight
bytes in total. The adaptive search subsequently pruned those weaker weights,
preserving their provenance and measured evidence. Their [actual serial GPU endpoint checks](reports/comparative/gpu-serving-teachers-fixed/report.json)
passed. The first Qwen 1.5B failure and its float32 attention-score fix remain
documented. Post-fix verification passed **106 tests**, formatting and strict
all-target Clippy; the [verification report](reports/comparative/verification-attention-fix/verification.json)
separately retains the unresolved optional build-tool notice gap.

The [completed development comparison](docs/comparative-models.md#completed-development-experiment--15-october-2026)
contains all 64 planned observations: Qwen 0.5B **0/16**, Qwen 1.5B **2/16**,
DeepSeek 1.3B **0/16**, SmolLM2 1.7B **0/16**. All generations completed; compile
and contract failures remain failures. The [resource table](reports/comparative/resources/resources.md)
and its JSON retain actual per-task times/tokens and evidence hashes. These are
descriptive original-model results, with no learning or promotion claim.

The adaptive [search roster](config/teacher-search.toml) now starts with a verified
7B Q4 teacher and visits further immutable coding alternatives. It retains the
strongest measured model, requires a separate fresh development confirmation for
qualification, and prunes owned inactive poor weights while preserving evidence.
The 7B model currently passes **7/16 development tasks**; no qualifying fleet or
learning gain is claimed. The original small-model roster remains a historical
experiment whose pruned assets require explicit online rehydration before new inference.

```sh
# Resume original engine/data; live download, verification and grading progress.
./run.sh select-teachers

# Cached-only mode: no Hub metadata requests or downloads.
./run.sh select-teachers --offline

# Detect affinity/cgroup CPU, available RAM, GPU and VRAM.
./run.sh hardware
```

Interrupted runs resume validated cached work by default. Teacher/model/data or
training-policy changes require an explicit new protocol; `--fresh` preserves
the previous directory and prints a new output path. The
[local-study skill](.agents/skills/rust-distillation-study/SKILL.md) records the
agent execution contracts. The conventional and genuinely ternary study tracks
are additional local research; see [current status](STATUS.md) for measured
checkpoints, comparisons and unfinished work. `bitnet.cpp` is the documented
native C++ inference exception; training and orchestration remain Rust.

The [actual SFT development comparison](reports/local-study/sft-development-comparison-c7c.md)
retains both 0/16 scores and explicitly marks the quality comparison unavailable:
seven adapter grades lack complete infrastructure evidence. Real operator
rounds and fresh-process reloads establish executable SFT, KL, hidden-feature
and attention objectives; they do not establish learning. Official-base BitNet
rounds at three and 48 steps changed no effective trits. The separate
[64-step round](reports/local-study/native-bitnet-64-result.md) changed 230,999
effective trits and passed independent reload; native numerical comparisons still
fail their fixed tolerance. These claims remain separate from coding quality.
The [latest completed verification](reports/local-study/verification-006/verification.json)
passed all eight software checks and 254 ordinary tests while retaining the
optional dependency-notice gap.

Comparisons use train/development tasks and preserve the sealed held-out set. Original-model scores measure no learning and cannot authorise promotion. DeepSeek output-trained derivatives inherit its licence/use obligations. The earlier delivery archives remain the verified CPU snapshot; rebuild and verify a new package to include this extension. The [full dependency inventory](docs/dependency-notices.md) reports one evidenced upstream notice gap in the optional CUDA build tool, separately from passing software checks.
