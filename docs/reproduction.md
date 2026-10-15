# Reproduction and machine contract

## Supported environment and prerequisites

The supported implementation is Linux x86-64 with cgroup v2 CPU, memory, swap and PID controllers. Native Podman is the tested runtime. A Docker executable on its own is insufficient: this workstation's Docker client points at a rootless Podman service and reports several controls incorrectly through its compatibility API. The native executable and kernel probes establish enforcement.

Rust 1.98.1 is pinned for source builds. The binary uses CPU float32 and has no CUDA requirement. The observed host is Ubuntu 26.04.1, Intel Core i9-9880H, 16 logical CPUs, about 60 GiB RAM, an 8 GiB RTX 2080 and more than 300 GiB free disk at initial inspection. Host facts are captured by `preflight`; the GPU is an observation, not a tested training dependency.

Model weights occupy 988,097,824 bytes on disk; the CPU float32 base alone is about 1.98 GB. An actual 512-token synthetic forward/backward/Adam update, supervising 256 tokens, peaked at 7,967,212 KiB RSS (about 7.60 GiB) under `/usr/bin/time -v`. Its full process took 27.89 seconds, with 20.918 seconds inside the measured update. Allocate at least 16 GiB for the model workload and headroom; this single-step QA is not a measured multi-hour assessment round. The frozen compiler/supervisor OCI archive is 606,179,328 bytes. Build cache and extraction need further space. The prebuilt binary was linked on glibc 2.43; use Ubuntu 26.04 x86-64 or rebuild from source on the target system rather than assuming it runs on older glibc releases.

The launcher is a minimal shell `exec` wrapper. Application, training, tests, candidates, statistical calculations and evidence preparation are Rust. OCI build recipes invoke Cargo/rustc; ffmpeg is used only for standard video encoding. The reasons are recorded in deviations.md.

## Source bootstrap: explicitly online

```sh
cargo build --locked --release
./run.sh bootstrap-online
./run.sh tasks
./run.sh preflight
./run.sh sandbox-preflight
```

Install Rust and Podman through your workstation's normal package/toolchain process before these commands. This project does not silently install a daemon, alter system cgroups or assume privileged installation rights. The bootstrap command downloads the original public Qwen assets at an immutable revision and builds the compiler/supervisor image. Candidate runs use no network and never fetch crates. Provider collection is a separate explicit action requiring credentials, model/price snapshots, rights decisions and an authorised dollar cap.

## Local tests and actual model QA

```sh
cargo test --locked --offline
cargo test --locked --offline original_tokenizer_masks_prompt_and_preserves_assistant_eos -- --ignored
cargo run --locked --offline --release --example model_smoke -- offline_bundle/base
cargo run --locked --offline --release --example cpu_gradient_smoke -- offline_bundle/base
cargo run --locked --offline --release --example cpu_gradient_smoke -- offline_bundle/base 512
cargo run --locked --offline --release --example serving_smoke -- offline_bundle/base
./run.sh drill --offline
```

The first three model examples are reusable executable QA, with genuine base inference, genuine full-size forward/backward/update and real endpoint traffic. The gradient smoke uses explicitly synthetic token targets and publishes no assessment adapter. The serving smoke uses the base and an evaluation-only same-base alias; it demonstrates bounded serving/routing mechanics and cannot prove distinct-candidate quality or a production canary.

The drill exits nonzero until all required real evidence exists. Its JSON separates software-assurance fixtures from real candidate execution and actual-checkpoint controls. Read `reports/drill.md` rather than treating an expected assertion as a completed assessment round.

## A genuine provider round

1. Supply the three provider key environment variables locally. Never put secrets in config, Git, candidate containers, reports or the video.
2. Review `docs/rights.md` and the selected account/model terms. Pin exact model IDs/revision policy and applicable prices in `config/fleet.toml`. Record document hashes and a dated decision. Enable only rights-cleared, priced providers whose output cap covers every billable token.
3. Run `./run.sh collect --live --authorised-usd <explicit-cap> --round round-001`. This is the spending action. Cache/resume uses durable request keys; a killed process with an uncertain remote response retains its reservation and blocks reconciliation rather than assuming another call is free.
4. Run `./run.sh curate`, then primary/seed/control training using the predeclared config files. No passing winner means a repair queue, not an invented target.
5. Freeze candidate/configuration before opening sealed model outcomes. Use `./run.sh gate --incumbent base --candidate checkpoints/candidate`. For completed cached evaluation manifests, the same command verifies every referenced request, generation, audit and checkpoint artifact before making its decision.
6. Run the complete offline demo only after assets and real recordings exist. Do not treat repeated held-out evaluations as a tuning dashboard; the implementation refuses to overwrite an existing preregistration.

The full command orchestrates genuine training. On fresh application state it allocates `checkpoints/retrained/<execution-UUID>/<label>` even if completed adapters are already bundled. Outputs remain included evidence; only the resumable allocation pointer and process lease live under excluded `state/demo-training/`. Restart reuses that installation's validated completed work or resumes its durable optimiser state. Clearing `state/` allocates another genuine training execution without changing previous weights or data. `--use-recorded-checkpoint` explicitly skips optimisation and validates the once-published `checkpoints/demo-published.json` map, falling back to the four canonical bundled paths only when no map exists. A skipped phase is labelled in the report. Neither mode can turn absent recordings into a successful replay.

Evaluation freezes are keyed to actual tensor/data/configuration identity, not execution UUIDs. Exact repeated tensors may reuse verified complete cached evaluations while preserving their original evidence paths. Different tensors stop before any new development or held-out generation against the existing freeze; fresh independent sources and a new planned confirmation are required. A new full run therefore proves optimisation work, but does not imply permission to reopen the sealed assessment set.

## Disconnected delivery and rehearsal

The intended entry point is `./run.sh demo --offline`. A complete delivery needs the prebuilt binary or vendored Cargo dependencies, original base/tokeniser/licences, saved immutable OCI image, actual adapters, real provider recordings, dataset/checkpoint manifests and all evidence. The archive contains large assets separately from Git as appropriate; checksums identify each asset. The base is Apache-2.0; runtime and transitive notices must travel with their distributions.

Extract into a fresh directory, clear the application state/cache, unset provider credentials, load the OCI archive explicitly and run with outbound access disabled. The candidate image uses the frozen image ID. The executable resolves project resources from the new runtime directory, rather than a compiled-in original checkout path. A network namespace without an accessible runtime socket cannot operate a rootless runtime; a disconnected rehearsal needs a native local runtime connection that remains reachable inside the chosen network isolation mechanism. The rehearsal must record how outbound access was blocked and what runtime socket was used.

`cargo run --locked --offline --release --example disconnected_qa -- deliverables/lec-distil-delivery.tar.gz` extracts a delivery and verifies its complete runtime inventory before executing the bundled binary. Changed, missing, extra or symlinked assets fail this check. It starts a private user/network/mount namespace, enables only its isolated loopback, hides the original checkout and unsets provider credentials. It reaches the existing local Podman service through a Unix socket so the runtime remains available without outward network routes. The runtime image cache is preloaded rather than cleared. A fresh empty application state/cache is used. The helper preserves the observed nonzero assessment exit when real evidence is missing and labels that result local offline QA. A completed rehearsal can certify a later package only when its source commit, binary and complete runtime inventory still match.

An inventory or a successful individual smoke is not proof of the complete fresh-machine run. The final report must retain any missing phase and nonzero exit. No full disconnected provider-trained rehearsal is claimed while recordings and adapters are absent. `./run.sh package` creates checksum-indexed ZIP and tar.gz deliveries, including the binary, original notices, base and saved image. Its manifest remains incomplete until a checksum-bound successful full rehearsal exists.
