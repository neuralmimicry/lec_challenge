# Coding-fleet distillation in Rust

A native Rust implementation of the LEC Principal AI Engineer assessment: configurable teacher acquisition, durable budgets, isolated executable grading, source-separated curation, real differentiable Qwen LoRA, checkpoint evaluation and conservative promotion.

**The assessment is not yet complete.** No teacher API credentials, spending authority or account-specific API output-training decisions were supplied. Hugging Face access was subsequently authorised for local comparative weights, without saving credentials. There are no real teacher API recordings or provider-trained assessment checkpoints. The offline demonstration fails explicitly when these required artifacts are absent. Local transport fixtures, random small-model tests and full-size synthetic tensor training are labelled QA evidence; they do not establish learning from teachers. See [STATUS.md](STATUS.md) and [limitations](docs/limitations.md).

## Run

The required assessment entry point is:

```sh
./run.sh demo --offline
```

It prints phase results, saves partial evidence and exits nonzero while required recordings/checkpoints are missing. `./run.sh drill --offline` executes independent assertions and reports every required drill, including incomplete actual-checkpoint cases. An honest NO SHIP from a complete valid evaluation is a completed decision; missing evidence is a failure.

The supported machine is Linux x86-64 with enforced cgroup v2 CPU/memory/swap/PID controls and native Podman or a verified Docker-compatible runtime. The measured workstation has 16 logical CPUs, about 60 GiB RAM and an 8 GiB RTX 2080; the implemented and verified model path uses CPU float32. CUDA is not required. Allow several GB for model/runtime assets and additional space for Cargo builds. See [reproduction](docs/reproduction.md) for exact prerequisites and measured timings.

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
| Serving | `serve` | Bounded CPU worker; immutable checkpoint IDs; verified gate before production canary; atomic rollback |
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

```sh
./run-gpu.sh serve --comparative --device cuda --dtype f16 \
  --base offline_bundle/comparative/qwen-coder-1_5b \
  --config config/serving-teacher.toml

./run-gpu.sh compare-models --device cuda --dtype f16
```

Comparisons use train/development tasks and preserve the sealed held-out set. Original-model scores measure no learning and cannot authorise promotion. DeepSeek output-trained derivatives inherit its licence/use obligations. The earlier delivery archives remain the verified CPU snapshot; rebuild and verify a new package to include this extension. The [full dependency inventory](docs/dependency-notices.md) reports one evidenced upstream notice gap in the optional CUDA build tool, separately from passing software checks.
