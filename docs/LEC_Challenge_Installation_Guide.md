# LEC Challenge installation guide

Ubuntu source installation, CUDA configuration and reliable sandbox operation

Prepared for Paul B. Isaac’s and NeuralMimicry Ltd. Version 3, 22 October 2026.

Install the Rust application, establish persistent rootless Podman storage, verify model assets and sandbox enforcement, then run local inference and bounded research. This guide covers the implemented bounded repair and labels historical sm00 observations separately.

The repair starts at commit `d78d127fc3c9da546fe4b9bee3ef4e3cf0e9076d` and was developed on local branch `repair/training-20261022`, with local `main` as the user-requested integration target. That starting commit alone does not contain the repaired commands. Check your actual revision, working-tree changes and command help. Implemented contracts, measured checks and remaining limitations are recorded in [the repair report](repair-20261022.md) and section 15.

Source changes are made on the RTX 2080 laptop with 8 GiB VRAM. Historical reports initially reside at `pbisaacs@192.168.1.66:/home/pbisaacs/Developer/lec_challenge` on sm00, with an RTX 3060 and 12 GiB VRAM. Remote repair validation used a separate clone at `/home/pbisaacs/Developer/lec-repair-20261022-validation`; do not overwrite the original evidence or its pinned engines.

## 1 Installation routes and completion criteria

For the existing sm00 installation, retain the dedicated store and run `source state/lec-sm00-env.sh` in every new terminal. Start with section 4 to inspect that configuration, then sections 6 to 10 as needed. Do not recreate manifests or historical engines merely to repeat installation.

For a fresh workstation, follow sections 2 to 10 in order. CUDA and larger teacher/BitNet models are optional additions. A source clone does not contain all ignored model weights, OCI archives, pinned engines or historical reports.

Installation is operational when the locked build succeeds, base assets verify, sandbox preflight reports `probes_passed: true`, the reference grade passes and a real inference request completes. A successful download is not a sandbox test. Successful training or normal generation is not proof of correct code or learning.

| Item | Historical sm00 observations supplied for the repair |
|---|---|
| Host and GPU | Linux x86_64; 32 available CPU threads; RTX 3060 with 12 GiB VRAM. Free RAM and VRAM change with other workloads. |
| Rust engine | CPU source build and a pinned CUDA engine both ran. These are historical executions, not repaired-engine CUDA validation. |
| Sandbox | Dedicated persistent rootless Podman store and crun passed the complete V3 enforcement preflight. |
| CUDA | Toolkit 12.9 and g++-14 worked. Toolkit 13.3 was rejected by the locked cudarc build. |
| SFT study | Both seeds completed 48 updates and healthy CUDA evaluations. Baseline and both adapters scored 0/16; no measured coding improvement. |
| Feature study | Earlier feature-only adapters produced repetitive, length-capped generations. The cause remains under investigation. |

The repaired source passed CPU checks on both machines with Rust 1.98.1. Its bounded train-only CPU smoke lowered assistant NLL but both base and adapter scored 0/1 with compilation failures. CUDA 12.9/g++-14 passed a minimal compile probe; full repaired-engine CUDA inference/backward execution remains outstanding. See section 15 for the measured scope.

## 2 Host packages and source checkout

Use a normal login user for Cargo, Podman and application commands. Use sudo only for system package administration. Mixing `sudo podman` and ordinary `podman` selects different stores and can make images appear missing.

```bash
sudo apt update
sudo apt install -y \
  build-essential pkg-config git curl ca-certificates \
  podman conmon crun uidmap slirp4netns fuse-overlayfs \
  dbus-user-session ripgrep
```

The asset/store doctors and selected-training diagnostic are native Rust commands; Python is not required for this workflow. Historical external recovery helpers may have their own optional dependencies. `ripgrep` is an inspection convenience. The Rust application uses bundled SQLite and Rustls, so a PostgreSQL service, Python virtual environment, PyTorch installation, system SQLite development package and OpenSSL development package are not required by the reviewed core manifest.

For optional native BitNet builds, install CMake and a native C/C++ toolchain. CUDA builds additionally require a working NVIDIA driver, compatible toolkit and compatible host compiler.

```bash
sudo apt install -y cmake

# Only for a new destination that does not already exist.
mkdir -p "$HOME/Developer"
cd "$HOME/Developer"
git clone https://github.com/neuralmimicry/lec_challenge.git
cd lec_challenge
git rev-parse HEAD
```

For an existing checkout, use this instead of cloning or resetting it:

```bash
cd "$HOME/Developer/lec_challenge"
git status --short
git rev-parse HEAD
```

Preserve local edits. Read `AGENTS.md`, `docs/reproduction.md`, `docs/gpu.md` and the current acceptance/design documents. Do not apply a destructive reset or unreviewed pull to a running study.

### Capacity and architecture

Use Linux with enforceable cgroup v2 CPU, memory, swap and PID controls. The primary verified installation is x86_64. CPU and hardware discovery include other platforms, but that does not certify all training, native BitNet or sandbox paths on Centriq ARM servers.

Allow at least 16 GiB RAM for the original Qwen training workload and additional host headroom. The recorded original base weight file is about 0.99 GB; decoded F32 weights, gradients and optimiser state require much more memory. Larger teachers, BitNet training, trace caches, multiple build targets and OCI layers require additional RAM and disk. Inspect actual capacity rather than treating sm00's resources as minimum requirements.

```bash
free -h
df -h "$PWD" "$HOME/.local/share"
stat -fc %T /sys/fs/cgroup
cat /sys/fs/cgroup/cgroup.controllers
```

Expect `cgroup2fs`. Use a real user login session with an owned runtime directory and appropriate subordinate UID/GID mappings. Inspect `/etc/subuid` and `/etc/subgid` if rootless setup fails. Fix delegation and mapping at host level; do not bypass resource controls or run candidates privileged.

## 3 Rust and reproducible Cargo commands

The reviewed repository pins Rust 1.98.1 through `rust-toolchain.toml`; `Cargo.toml` declares edition 2024 and Rust 1.98 or later. Use the checkout's actual pinned toolchain after source changes.

```bash
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf \
    https://sh.rustup.rs -o /tmp/lec-rustup-init.sh
  sh /tmp/lec-rustup-init.sh -y
fi
source "$HOME/.cargo/env"
rustup toolchain install 1.98.1 --profile minimal \
  --component rustfmt --component clippy
rustc --version
cargo --version
```

Perform the online dependency preparation explicitly, keeping the lockfile:

```bash
cd "$HOME/Developer/lec_challenge"
cargo --config net.offline=false fetch --locked
cargo --config net.offline=false build --locked --release
cargo run --locked --offline --release -- --help
```

After a successful build, maintenance commands can execute the actual binary directly. This avoids repeated Cargo resolution when diagnosing a container or asset issue:

```bash
lec_engine="$PWD/target/release/lec-distil"
test -x "$lec_engine"
"$lec_engine" --help
```

If a custom `CARGO_TARGET_DIR` or build target is active, locate the binary from the build output; do not assume the default path. Rebuild after source changes before using this binary to validate the new code. Do not replace an old content-addressed engine snapshot with it.

| Command | Purpose and installation policy |
|---|---|
| `cargo fetch --locked` | Downloads dependencies recorded by the lockfile. Use an explicit online configuration during preparation. |
| `cargo build --locked --release` | Compiles the application with the recorded dependency resolution. |
| `cargo run --locked --offline --release -- ...` | Builds if necessary from cached dependencies, then runs the command. Cargo offline does not control application asset downloads. |
| `cargo update` | Changes dependency resolution in Cargo.lock. Reserve for planned maintenance, not a cache-miss repair. |
| `cargo clean` | Removes build products in the selected target directory. It is optional and usually unnecessary for installation. |

Both launchers default to locked, offline Cargo builds. `bootstrap-online` permits an online Cargo build; `LEC_CARGO_ONLINE=1` explicitly permits online builds for other launcher commands. An explicit application `--offline` takes precedence and forces Cargo offline. `CARGO_NET_OFFLINE=false` alone does not override an explicit Cargo `--offline`. These policies concern dependency acquisition; application asset acquisition remains a separate decision.

```bash
# Explicit online preparation; bootstrap also acquires assets and builds the image.
./run.sh bootstrap-online
# Permit dependency acquisition for a maintenance command without acquiring models.
LEC_CARGO_ONLINE=1 ./run.sh sandbox-doctor
# Cache-only build and asset inspection, even when online builds are enabled.
LEC_CARGO_ONLINE=1 ./run.sh assets-doctor --base offline_bundle/base --offline
```

If a locked dependency is unavailable offline, run `cargo --config net.offline=false fetch --locked` during online preparation in the same user, registry/mirror and Cargo environment. Offline launch failures provide this hint without retrying online. The supplied historical `libc 0.2.191` cache miss belongs to a different build context; the current unchanged lockfile resolves `libc 0.2.190`. Investigate the registry error if fetching still fails; do not downgrade the lockfile to conceal the issue.

### Optional maintenance and cleaning

Review dependency changes on a separate maintenance branch. A dry run previews the change; an intentional update then needs diff review and the affected CPU/CUDA checks:

```bash
cargo --config net.offline=false update libc --dry-run
# Only when the dependency change is deliberately intended:
cargo --config net.offline=false update libc
git diff -- Cargo.lock
```

For suspected project build corruption, a scoped clean is preferable to deleting every target:

```bash
cargo clean --release -p lec-distil
cargo build --locked --offline --release
```

To remove stale Candle kernels after switching toolkits, set `CARGO_TARGET_DIR` to the exact affected GPU build directory and use `cargo clean --release -p candle-kernels`. The launcher may use nested target paths; `target/cuda` is not universal. Preserve `state/engines`, checkpoints, reports, model caches and OCI archives.

## 4 Persistent rootless Podman configuration

On an already repaired sm00 installation, load its existing configuration and inspect the actual store:

```bash
cd "$HOME/Developer/lec_challenge"
source state/lec-sm00-env.sh
podman info --format \
  'runtime={{.Host.OCIRuntime.Path}} graphRoot={{.Store.GraphRoot}} runRoot={{.Store.RunRoot}}'
```

Expected sm00 values are `/usr/bin/crun`, graph root `/home/pbisaacs/.local/share/lec-challenge/podman/storage` and run root `/run/user/1000/lec-challenge/storage`. The graph root holds durable images; the run root is session/runtime state on tmpfs. A temporary AARNN graph root under `/var/tmp/aarnn-ps-...` repeatedly lost the required sandbox image and should not be reused for LEC.

### Configure a fresh dedicated store

Use this only for initial setup of a dedicated LEC store. It preserves files already present; inspect any existing configuration that differs before proceeding. Selecting a new store does not migrate images from the old store. Restore verified archives into the selected store afterwards.

```bash
cd "$HOME/Developer/lec_challenge"
(
  set -euo pipefail
  umask 077
  lec_data="$HOME/.local/share/lec-challenge/podman"
  lec_session="${XDG_RUNTIME_DIR:?Use a normal user login session}"
  test -d "$lec_session"
  test -O "$lec_session"
  test "$(stat -fc %T "$lec_session")" = tmpfs
  lec_runtime="$lec_session/lec-challenge"
  mkdir -p state "$lec_data" "$lec_runtime"
  findmnt -n -o FSTYPE --target "$lec_data"

  if [ ! -e state/lec-sm00-storage.conf ]; then
    cat > state/lec-sm00-storage.conf <<EOF
[storage]
driver = "overlay"
graphroot = "$lec_data/storage"
runroot = "$lec_runtime/storage"
transient_store = false
EOF
  fi

  if [ ! -e state/lec-sm00-containers.conf ]; then
    cat > state/lec-sm00-containers.conf <<EOF
[engine]
runtime = "crun"
static_dir = "$lec_data/storage/libpod"
tmp_dir = "$lec_runtime/libpod/tmp"
volume_path = "$lec_data/storage/volumes"
[engine.runtimes]
crun = ["/usr/bin/crun"]
EOF
  fi
)
```

The graph root must be on a supported local filesystem, not NFS/CIFS or an unrelated transient store. This recipe retains the host's cgroup-manager setting; the effective enforcement preflight is mandatory. Do not switch cgroup managers for existing containers blindly.

For a new CPU installation, create the environment file only if it is absent:

```bash
if [ ! -e state/lec-sm00-env.sh ]; then
  cat > state/lec-sm00-env.sh <<'EOF'
export CONTAINERS_STORAGE_CONF="$PWD/state/lec-sm00-storage.conf"
export CONTAINERS_CONF_OVERRIDE="$PWD/state/lec-sm00-containers.conf"
unset LEC_DEVICE
EOF
fi
source state/lec-sm00-env.sh
crun --version
podman --version
podman info --format \
  'runtime={{.Host.OCIRuntime.Path}} graphRoot={{.Store.GraphRoot}} runRoot={{.Store.RunRoot}}'
```

Always enter the checkout before sourcing this fresh recipe. Existing sm00 environment files use their recorded absolute paths and include CUDA settings. Never overwrite them with the CPU example. A copied checkout or changed username/runtime directory needs configuration review before use.

Inspect the selected store with the native doctor:

```bash
"$PWD/target/release/lec-distil" sandbox-doctor
```

Inspect `rootless`, runtime errors, graph/run roots, effective storage configuration, image identity and warnings. Temporary or foreign-owned graph roots are flagged. The doctor observes the active store; it does not migrate it or authorise grading. Inspect any `CONTAINERS_CONF_OVERRIDE` profile as well as the explicit configuration paths reported by the doctor.

Do not use `podman system reset`, global image pruning, `--privileged`, `--security-opt seccomp=unconfined` or host execution of generated candidates as installation fixes.

## 5 Parent sandbox image and offline archives

The V3 builder needs a verified cached V2 parent. A manifest alone does not put the image into the current Podman store. The images directory is excluded from Git.

For an existing archive, verify its checksum against its retained manifest/checksum file before loading it. These values identify the original sm00 V2 archive only:

```bash
(
  set -euo pipefail
  lec_parent_archive=offline_bundle/images/policy-2/x86_64/sandbox.tar
  lec_parent_sha=baae8930452f34b013686f11435afebe7a3d1e2f426df9ff67578c86e628d768
  lec_parent_id=15f5149a54c8dda4e7c37a935322f6c0824670c3d7fc1ad63c82bc10bc1667b1
  printf '%s  %s\n' "$lec_parent_sha" "$lec_parent_archive" | \
    sha256sum --check -
  podman load --input "$lec_parent_archive"
  podman image inspect --format '{{.Id}}' "$lec_parent_id"
)
```

For a newly created archive, use its own recorded hash and ID, not sm00's values. If no parent archive or manifest exists, an explicit online parent build can prepare one:

```bash
(
  set -euo pipefail
  lec_arch=$(uname -m)
  lec_parent_tag="localhost/lec-rust-sandbox:2-$lec_arch"
  lec_parent_dir="offline_bundle/images/policy-2/$lec_arch"
  # Never overwrite an existing partial archive or manifest.
  test ! -e "$lec_parent_dir"
  mkdir -p "$lec_parent_dir"
  podman build --pull=missing -f sandbox/Containerfile.v2 \
    -t "$lec_parent_tag" .
  podman save --format oci-archive \
    --output "$lec_parent_dir/sandbox.tar" "$lec_parent_tag"
  lec_parent_id=$(podman image inspect \
    --format '{{.Id}}' "$lec_parent_tag")
  cargo run --locked --offline --release \
    --example package_sandbox -- \
    "$lec_parent_dir/sandbox.tar" "$lec_parent_id" \
    --portable --policy 2
)
```

This online build fetches the Containerfile's pinned Rust base and sandbox dependencies. The example records the actual archive hash, image ID, architecture and dependency provenance. If the directory already exists or the build fails, inspect and preserve it; restore a valid archive or quarantine only the incomplete preparation before retrying. Do not replace a historical manifest with a newly built image ID.

V2 is the build parent. Candidate execution uses the verified V3 policy. V3 bootstrap uses the cached parent without network/pulls; a fresh V3 build can have a different immutable image ID from an earlier build.

For a packaged V3 archive, use the native verifier against the current sandbox policy and architecture:

```bash
"$PWD/target/release/lec-distil" sandbox-doctor \
  --archive-directory offline_bundle/images
```

This selects `policy-3/<architecture>/manifest.json`, or a `manifest.json` directly in the specified directory. It checks the declared archive path, size, SHA-256, policy, architecture and manifest image identity. Inspect `archive.verified`, the reported active/archive IDs and runtime warnings; a JSON report alone is not a passing preflight. The CLI does not accept an old study plan: compare its reported image ID with that plan's frozen ID yourself before restoring for a resume. Explicit restoration adds `--restore-archive` (see section 13). An operator backup containing only `SHA256SUMS` and `image-id.txt` is not this manifest format.

## 6 Model paths and bootstrap

The original sm00 model failure was a directory called `model.safetensors`. Model payload paths must be regular files. Inspect pinned payloads before acquisition:

```bash
"$PWD/target/release/lec-distil" assets-doctor \
  --base offline_bundle/base --offline
```

The doctor distinguishes valid regular files, missing files, corrupt files, directories and other non-regular conflicts, including dangling symlinks. It performs no downloads. Review its per-file states; exit zero alone does not mean every payload is valid. To explicitly preserve conflicts by renaming them alongside the original paths:

```bash
"$PWD/target/release/lec-distil" assets-doctor \
  --base offline_bundle/base --quarantine-conflicts
```

Quarantine is an explicit mutation and cannot be combined with `--offline`; it does not acquire replacements. The doctor covers pinned payload paths, not `assets.json` or partial-download paths. Inspect metadata/partial-file conflicts separately and preserve them by an explicit rename if needed. Never recursively delete a conflicting directory or replace otherwise valid provenance.

Bootstrap with the source-built CPU binary in the same configured shell:

```bash
source state/lec-sm00-env.sh
lec_engine="$PWD/target/release/lec-distil"
"$lec_engine" bootstrap-online
```

Both `model` and `sandbox_image` must report `Ok`. A passing model phase does not imply a passing image phase. Keep valid assets and retry only the failed preparation after resolving its cause.

The base is `Qwen/Qwen2.5-Coder-0.5B-Instruct`, revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, with Apache-2.0 provenance. Its weight hash is:

```text
f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44
```

The manifest is `offline_bundle/base/assets.json`. Validate complete manifests and licences using the application's native validation path; payload inspection alone is insufficient. Bootstrap restores absent verified payloads after conflicts are resolved and preserves valid payloads and valid provenance manifests. Do not overwrite valid metadata during a missing-payload repair: changed provenance timestamps can invalidate bound historical evidence.

### Preserve the resulting V3 image

Archive the actual image by immutable ID in a new directory. This is an operator backup; a complete portable distribution also needs the project's build receipt, manifest, policy and execution closure.

```bash
(
  set -euo pipefail
  lec_image_id=$(podman image inspect --format '{{.Id}}' \
    "localhost/lec-rust-sandbox:3-$(uname -m)")
  mkdir -p offline_bundle/images/local-v3
  lec_archive_dir=$(mktemp -d \
    "$PWD/offline_bundle/images/local-v3/backup.XXXXXX")
  podman save --format oci-archive \
    --output "$lec_archive_dir/sandbox.tar" "$lec_image_id"
  printf '%s\n' "$lec_image_id" > "$lec_archive_dir/image-id.txt"
  (cd "$lec_archive_dir"; sha256sum sandbox.tar > SHA256SUMS)
  printf 'V3 backup saved in %s\n' "$lec_archive_dir"
)
```

The historically recovered sm00 V3 ID is `dce2b9a2130c9a2ddf43ab3201d8b9a58d5a87046974c0b2becea99cb0e1031d`. Its retained archive directory is `offline_bundle/images/local-v3/<that-id>/`. The later laptop repair smoke independently verified image `f86951873ef3392af722d4c1adc94ed64bd5badbef681fd0a4f9ff6aff1a3939`. These are different identities. Preserve each archive and binding; retagging a different image does not restore an earlier plan's original identity.

## 7 Sandbox and software verification

Run the actual enforcement probes before grading or research. These probes deliberately exercise memory/PID pressure and termination. Do not run a competing LEC workload while preflight is using the sandbox.

```bash
cd "$HOME/Developer/lec_challenge"
source state/lec-sm00-env.sh
lec_engine="$PWD/target/release/lec-distil"
"$lec_engine" assets-doctor --base offline_bundle/base --offline
"$lec_engine" sandbox-doctor --archive-directory offline_bundle/images
"$lec_engine" preflight
"$lec_engine" tasks
"$lec_engine" sandbox-preflight
"$lec_engine" grade --task ledger.sum \
  --code sandbox/examples/sum.rs
```

Expected task inventory is 96 contracts: 48 train, 16 development and 32 held-out, from 24 sources. Inventory is not permission to tune against held-out evaluation. The reference candidate must pass with complete tracing.

The sm00 passing V3 preflight measured UID 65532, a read-only root, no network, `memory_max` 536870912 bytes, swap limit zero, PID limit 64 and `cpu_max` 100000/100000. It also verified pressure handling and exact owned-cgroup deadline termination. Require `probes_passed: true`; the same tag or a working `podman run` is insufficient.

Verify source invariants after changes:

```bash
cargo fmt --all -- --check
cargo test --locked --offline --lib
cargo clippy --locked --offline --all-targets -- -D warnings
cargo build --locked --offline --release
cargo run --locked --offline --release -- --help
```

Run focused regression/integration checks as appropriate; the exact repair checks are in section 15 and the repair report. An unfiltered `cargo test` also includes integration tests: inspect their split usage before running them and keep sealed held-out sources closed during repair. Tests needing model assets, GPU hardware, native engines or a real sandbox must be reported separately from ordinary unit tests. Do not count skipped tests as passing. On a new host, run the repository's documented model/serving smoke examples when their dependencies are available.

## 8 Optional CUDA installation and selection

CPU mode remains supported. Historical sm00 executions used an RTX 3060, CUDA toolkit 12.9, g++-14, local compatibility headers and the historical CUDA engine. The repaired launcher has passed toolkit/compiler selection and a minimal CUDA 12.9/g++-14 compile probe on the laptop; the repaired full engine has not been executed on CUDA. Neither result claims NVIDIA officially qualifies Ubuntu 26.04 for CUDA 12.9.

The `CUDA Version` shown by `nvidia-smi` describes driver capability, not the installed toolkit selected for compilation. The locked cudarc 0.16.6 rejected toolkit 13.3 with `Unsupported cuda toolkit version: 13030` in the historical build. The repaired launcher checks the locked dependency's supported toolkit versions before expensive native compilation, checks the driver ceiling separately, then validates the host compiler with a real minimal CUDA compilation. Auto mode explains rejected candidates and can choose a compatible installed toolkit. An invalid explicit choice fails directly. Do not spoof `CUDARC_CUDA_VERSION`.

```bash
nvidia-smi
test -x /usr/local/cuda-12.9/bin/nvcc
/usr/local/cuda-12.9/bin/nvcc --version
test -x /usr/bin/g++-14
/usr/bin/g++-14 --version
```

If the compiler is missing, check its availability in the configured Ubuntu repositories and install the compatible package. Do not change the system-wide compiler alternatives just for this project:

```bash
apt-cache policy gcc-14 g++-14
sudo apt install -y gcc-14 g++-14
```

For a missing toolkit/driver, use NVIDIA's official installation guidance for the actual distribution and architecture, selecting a supported project/toolkit combination. Avoid blindly installing the generic newest `cuda` metapackage or replacing an already working driver. Toolkit 12.9's official distribution matrix and compiler constraints must be checked; the local overlay does not establish vendor certification.

### sm00 GPU environment

The recovered `state/lec-sm00-env.sh` already exports these settings. These paths are a tested local example, not a universal compiler/toolkit requirement. Valid explicit `LEC_CUDA_TOOLKIT`, CUDA root variables and `LEC_CUDA_HOST_COMPILER`/`NVCC_CCBIN` choices are honoured. For a fresh environment, add settings deliberately after confirming the paths; do not overwrite an existing profile:

```bash
export LEC_CUDA_TOOLKIT=/usr/local/cuda-12.9
export LEC_CUDA_HOST_COMPILER=/usr/bin/g++-14
export NVCC_CCBIN="$LEC_CUDA_HOST_COMPILER"
export LD_LIBRARY_PATH="$LEC_CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset LEC_DEVICE
./run.sh hardware
```

Use the resource-aware launcher for a new CUDA build and smoke request:

```bash
./run.sh infer --device cuda --gpu 0 --dtype f32 \
  --prompt 'Write a Rust function adding two i64 values.' \
  --max-tokens 128
```

Confirm CUDA execution, GPU ordinal and dtype in actual runtime metadata. Successful feature compilation or GPU discovery alone does not prove inference ran on the GPU. Keep F32 when reproducing the local studies; changing dtype creates a different numerical protocol.

The launcher can prepare a local CUDA 12.9 header overlay for the known glibc 2.43 declaration conflict. Its build receipt binds toolkit/compiler identity, compute capability, overlay hashes and compile-check provenance. Follow [the GPU guide](gpu.md) if manual preparation is needed. Never patch system headers or copy another machine's SM75 settings onto sm00's SM86 GPU.

### VRAM and competing workloads

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv
nvidia-smi
```

Earlier sm00 trials had only about 2.65 GiB free because other models occupied VRAM. Automatic CPU selection was therefore expected even with a compiled CUDA backend. For 1024-token Qwen 7B trials, the recorded conservative estimates were about 6.54 GB for Q4 and 8.11 GB for Q6; larger contexts need more headroom. These estimates are not fixed guarantees or partial-offload support. Stop only workloads you own and intend to stop; the installer must not kill unrelated AARNN, Webots or server processes.

## 9 Local inference and HTTP serving

Run a CPU inference smoke without involving the GPU launcher:

```bash
"$PWD/target/release/lec-distil" infer --device cpu --dtype f32 \
  --prompt 'Write a Rust function adding two i64 values.' \
  --max-tokens 128
```

Generated code is text until graded in the sandbox. Do not execute it on the host.

Start the original base server in a terminal with the environment sourced:

```bash
"$PWD/target/release/lec-distil" serve --device cpu --dtype f32
```

In a second terminal:

```bash
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/metadata
curl --fail-with-body http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a Rust function adding two i64 values.","max_new_tokens":128}'
```

The reviewed `config/serving.toml` binds to loopback port 8080, with queue capacity four, maximum output 256 tokens and a 120-second request timeout. CLI uses `--max-tokens`; HTTP uses `max_new_tokens`. This is a local API, not a browser UI. Stop the server with Ctrl+C.

For remote access, keep the loopback bind and use SSH forwarding from the client:

```bash
ssh -N -L 18080:127.0.0.1:8080 pbisaacs@sm00
```

The client's `http://127.0.0.1:18080` then reaches the server. Local-study adapters require the appropriate evaluation-only serving mode; consult `serve --help` rather than treating a research adapter as production-approved.

## 10 Larger teacher assets and optional BitNet

Base bootstrap does not provide the comparative teacher roster or the BitNet student. Audit actual manifests and payloads before starting an offline search. A prune marker, existing directory or manifest is not proof that the tensor file still exists and verifies.

The recorded sm00 search used Qwen Coder 7B Q4/Q6, Qwen Coder 14B Q2, Qwen general 7B Q4, DeepSeek Coder 6.7B Q4 and Magicoder 6.7B Q4. All require their independently pinned manifests, publisher checksums and applicable licences. Different quantisations of one original model do not become distinct teachers or independent provider businesses.

For new explicit online Hub acquisition, inspect the supported native command:

```bash
"$PWD/target/release/lec-distil" hub-download --help
```

Use the roster's exact model ID, immutable revision and destination. Credentials belong only in the supported ephemeral environment/stdin path. Never put tokens in commands, reports or Git. For a missing payload inside a valid existing cache, use a targeted verified restore; a general downloader may regenerate provenance and invalidate historical bindings.

The optional original BitNet student is `microsoft/bitnet-b1.58-2B-4T`, revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`. On sm00 its missing weight payload was restored without replacing valid metadata:

```text
offline_bundle/students/bitnet-2b-4t/model.safetensors
Bytes: 1178623988
SHA256:
8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b
```

For an entirely new BitNet cache, the repository provides an explicit acquisition example:

```bash
cargo run --locked --offline --release --example hub_bitnet_assets
```

This command's application can download models despite Cargo `--offline`; do not run it as a supposedly disconnected check or blindly rerun it over historical metadata. Read `docs/bitnet.md` first. Native C++ I2_S inference/export requires its pinned upstream source, CMake build and supported CPU instruction set. The reviewed x86 path expects AVX/AVX2, BMI2, F16C, FMA and SSE4.2; do not assume it runs unchanged on Centriq ARM.

Actual ternary training must change effective trits, export/reload correctly and meet the required checkpoint-bound native parity checks. Changed latent floats alone are insufficient. Keep failed parity and unchanged-trit observations visible.

## 11 Research workflow and frozen evidence

Plain `./run.sh` defaults to `explore` and the reviewed default wall budget is 24 hours. Choose the command, budget, output paths and runtime deliberately. A successful install does not require a full search or study.

Before research, source the same container/GPU profile, verify the exact sandbox image, inspect free resources and pass sandbox preflight. For a new bounded exploratory run:

```bash
cd "$HOME/Developer/lec_challenge"
source state/lec-sm00-env.sh
"$PWD/target/release/lec-distil" sandbox-preflight
lec_run_id="sm00-$(date +%Y%m%d-%H%M%S)-$RANDOM"
./run.sh explore \
  --output "reports/local-study/explore/$lec_run_id" \
  --search-output "reports/comparative/search/$lec_run_id" \
  --maximum-run-seconds 1800 --maximum-parallel-jobs 1
```

The 1800-second limit bounds that invocation; it does not guarantee the entire research programme completes. Default exploration can acquire public assets and run configured numerical work. For cache-only operation add application `--offline` after verifying the execution closure. Preserve inactive models and cache ownership/prune evidence when offline.

Fresh output paths prevent accidental resume of incompatible historical work. A missing original engine snapshot requires restoring that exact snapshot for a resume; copying the newest binary into its hash-named directory is invalid. When deliberately changing code, image, context, data or numerical settings, start a new protocol/output and keep the old run.

### Staged studies

For controlled work, prefer explicit stages: verified teacher comparison; train-only evidence collection; exact eligibility; selected-training diagnostics; small control; checkpoint export/reload; matched development evaluation; descriptive comparison; only then a justified larger experiment.

New plans freeze three phase runtimes independently:

| Phase | `study-plan` options |
|---|---|
| Teacher target capture | `--device auto\|cpu\|cuda`, `--gpu`, `--dtype f32\|f16` |
| Student optimisation | `--student-device cpu\|cuda`; F32 |
| Evaluation | `--evaluation-device cpu\|cuda`, `--evaluation-dtype f32\|f16`, `--evaluation-cpu-threads` |

Student and evaluation devices default to CPU; evaluation dtype defaults to F32. CPU/F16 and CUDA requests from CPU-only binaries fail before optimisation. Explicit CUDA still requires an available device and sufficient free VRAM. Phases run serially within a worker; reservations cover the largest phase's requirement. Inference receives the evaluation CPU reservation minus one (minimum one), leaving grading capacity. A compiled CUDA backend is insufficient evidence of actual GPU execution. BitNet remains CPU/F32.

Use current `--help` for each stage. Commands include `study-plan`, `study-prepare-arm`, `study-diagnose-training`, `study-run`, `study-select`, `study-evaluate`, `study-verify` and `study-compare`. Planner dispatch, target prewarm and the worker use the same executable eligibility contract. Preparing an arm writes its exact verified selection and immutable request without training (section 12). Native failed trials remain failed; a restart resumes eligible queued/interrupted progress, not an automatic retry of every failure.

For an explicit development re-evaluation of a verified completed checkpoint, choose its actual path and a fresh directory:

```bash
lec_checkpoint=/path/to/the/verified/checkpoint
test -f "$lec_checkpoint/manifest.json"
lec_eval="reports/local-study/evaluation/$(date +%Y%m%d-%H%M%S)-$RANDOM"
lec_engine="$PWD/target/release/lec-distil"
"$lec_engine" study-evaluate --base offline_bundle/base --student qwen \
  --checkpoint "$lec_checkpoint" --output "$lec_eval" \
  --split development --device cpu --dtype f32 --offline
"$lec_engine" study-verify --evaluation "$lec_eval"
```

Substitute the real checkpoint path; this is not a complete runnable study recipe without that input. This example uses CPU/F32. For CUDA, use a verified CUDA engine and explicit CUDA runtime in a fresh attempt. Either recovery remains bound to the original training provenance and leaves the original failed worker outcome intact.

Compare only against a compatible verified baseline. Runtime, dtype, engine, image, policy, task/decoding identities and applicable thread/resource fields must match, including actual inference thread identity. Keep the inference thread environment identical across processes; generate a new matched baseline when an old one is incompatible. A newer CPU verifier is not a substitute for historical inference provenance. Use the same intended verified engine for evaluation and verification whenever possible. After setting the actual matched baseline directory and adapter SHA-256:

```bash
"$lec_engine" study-verify --evaluation "$lec_eval" \
  --expected-split development --expected-checkpoint "$lec_adapter_sha"
"$lec_engine" study-compare \
  --baseline "$lec_matched_baseline" --candidate "$lec_eval"
```

The general `evaluate` command defaults to held-out in the reviewed source. Do not use it as an installation smoke or tuning dashboard. `study-evaluate` restricts this local workflow to train/development.

## 12 Token limits, selected-training diagnostics and honest outcomes

Collection context, generated output, student training capacity, privileged teacher trace capacity and evaluation output limits are separate constraints. Raising one does not raise the others. The current worker applies the existing frozen training cap to both actual student and teacher tokenisations, measuring each with its original prompt/tokeniser before loading weights. It does not introduce a separate larger teacher cap. All effective settings belong in the frozen request/cache identity.

Shared supported training bounds are **1–64 updates, 4–1024 sequence tokens including EOS, rank 1–32 and learning rate 1e-7–0.01**. Attention/trace capacity is checked separately: full teacher attention and resident trace payloads have 512 MiB checks, with the existing 8 GiB target-cache budget retained. Zero eligible examples makes an arm inapplicable with evidence. Cache-only selection rejects absent or incompatible tensors without acquiring them. Increasing collection context to 2048 does not expand a frozen 512-token training cap.

In the supplied historical sm00 reports, the original 1024-token content context was associated with generation failures. The separate 2048-token follow-up completed 127 available requests and added 59 passing samples, bringing its inherited bundle to 81. Seventeen requested variants remained unavailable, so strict complete-coverage status stayed false. These reports were not rerun as part of the repair. Valid partial research evidence is not full coverage.

The SFT-only control used only eligible complete-sequence examples, not every one of the 81 mixed-content records. Longer context variants can still exceed a 512-token teacher or student training limit. Never silently clip a sample, strip privileged teacher context or independently window preference sides to force eligibility.

| Observation | Meaning and next action |
|---|---|
| `stop` and zero infrastructure errors | Generation completed. Inspect sandbox correctness separately. |
| `length`, timeout or incomplete generation | Retain as failure in the planned denominator. More tokens are a new experiment, not a correction to the old score. |
| 48 steps and a verified adapter manifest | Training/export evidence exists. It does not establish useful generation or improvement. |
| Healthy baseline and candidate both 0/16 | A valid zero-delta result. Do not present a latency tie-break as a quality winner. |
| Wrong checkpoint/split or invalid evidence | Reject the result; never relax the binding to obtain a comparison. |

The observed 48-update SFT results were healthy for seeds 1729 and 2718, both 0/16 against the 0/16 baseline. The feature-only adapters instead repeated invalid text until the cap. These outcomes motivate diagnostics for the exact training examples, assistant masking, label shift/EOS, gradients, adapter application and feature alignment. The root cause of collapse is not established by these observations alone.

### Native diagnostic workflow

`study-diagnose-training` accepts exactly one verified `--request` or completed `--checkpoint`. Its default operation is read-only inspection: exact realised sample order, unique samples, repeated exposures, per-task/source coverage, exclusions, conditioning and token counts. It reconstructs the optimisation set rather than describing the entire train corpus as selected data.

Start with an existing checkpoint, substituting its real directory:

```bash
lec_engine="$PWD/target/release/lec-distil"
lec_checkpoint=/path/to/the/verified/checkpoint
"$lec_engine" study-diagnose-training --checkpoint "$lec_checkpoint" --offline
```

For a new frozen study, prepare one arm without reserving numerical work or starting optimisation. Use the same pinned engine as planning, its actual arm ID and a fresh output directory:

```bash
"$lec_engine" study-prepare-arm --study "$lec_study" --arm "$lec_arm" \
  --profile quick --output "$lec_fresh_request" --offline
"$lec_engine" study-diagnose-training \
  --request "$lec_fresh_request/request.json" --offline
```

Numerical operations require a fresh output directory. `--nll` measures teacher-forced assistant/EOS NLL, token accuracy and EOS likelihood before adaptation and after adapter reload, recording masks, prediction positions and identities. `--generate` separately compares base and reloaded adapter on the same selected training prompts with fixed decoding/runtime and real sandbox grading. They can be requested independently or together:

```bash
lec_diagnostic="reports/local-study/diagnostics/$(date +%Y%m%d-%H%M%S)-$RANDOM"
"$lec_engine" study-diagnose-training --checkpoint "$lec_checkpoint" \
  --nll --generate --maximum-examples 1 --max-new-tokens 512 \
  --maximum-seconds 900 --cpu-threads 4 --device cpu --dtype f32 --offline \
  --output "$lec_diagnostic"
```

The example measures one selected example; it does not claim to measure every optimisation example. The bound is 1–16 unique examples in realised order, at most 512 generated tokens per output and a 900-second total deadline. Base and adapter use the same prompts, seed 1729, temperature 0.2, top-p 0.95, runtime and threads. The diagnostic defaults to CPU/F32; choose explicit `cpu` or `cuda` for a reproducible numerical request. `auto` is rejected. Complete in-capacity sequences are required for generation; no arbitrary clipping or replacement of privileged teacher conditioning is allowed. Progress is streamed and partial results survive deadline expiry. Generated candidates run only inside the sandbox after full preflight. These commands do not download assets.

No updates occur by default. The explicit `--train-updates 1` option (maximum 3) additionally requires `--nll` and a fresh verified request frozen for exactly that number of updates, with exactly one selected complete-sequence SFT example. It rejects other objectives, assistant windows, changed budgets and existing checkpoint outputs. Use a separate fresh diagnostic output; do not apply this option to the checkpoint command above. The bounded repair smoke's exact request and command are in [the repair report](repair-20261022.md); its existing output paths must not be reused.

Interpret the outputs separately: export/reload, finite gradients and lower NLL establish numerical behaviour; healthy generation establishes completion; sandbox grades establish code correctness. An NLL-only run leaves coding scores `null`. Training-set gains are memorisation diagnostics and cannot establish development improvement or promotion.

## 13 Troubleshooting and recovery

### Missing sandbox tag or image ID

Source the same environment and inspect graph root/run root and the immutable expected ID. Do not infer permanent loss from a missing tag alone. For a packaged archive with the native manifest described in section 5, inspect it first, then explicitly restore the verified exact image into the selected store:

```bash
lec_archive_dir=offline_bundle/images
"$PWD/target/release/lec-distil" sandbox-doctor \
  --archive-directory "$lec_archive_dir"
# Explicit mutation: load only the verified archive into the active store.
"$PWD/target/release/lec-distil" sandbox-doctor \
  --archive-directory "$lec_archive_dir" --restore-archive
```

For the older operator-backup format containing `SHA256SUMS` and `image-id.txt` instead of `manifest.json`, verify and load that retained archive explicitly:

```bash
# Set this to the actual retained V3 backup directory.
lec_archive_dir=/path/to/verified/v3-backup
(
  set -euo pipefail
  cd "$lec_archive_dir"
  sha256sum --check SHA256SUMS
  podman load --input sandbox.tar
  lec_saved_id=$(cat image-id.txt)
  podman image inspect --format '{{.Id}}' "$lec_saved_id"
)
```

Run sandbox preflight afterwards. If the exact historical image cannot be restored, a verified rebuild supports new research with a new image binding; it does not make old plans resumable under a different ID.

### Missing attach socket or runtime startup failure

A missing socket is a symptom, not a diagnosis. Inspect the exact failed container's state/logs and current runtime. The sm00 runc/seccomp startup problem was resolved using crun without weakening V3 controls. Preserve diagnostic evidence; remove only the identified diagnostic container when it is no longer needed. Never run a blanket container/image reset.

### No compatible CUDA C++ host compiler

Inspect the launcher's rejected candidates and direct error. Confirm that the selected toolkit is supported by the locked dependency and driver, then choose a compatible installed compiler using `LEC_CUDA_HOST_COMPILER` and `NVCC_CCBIN` in the launching shell. CUDA 12.9 with `/usr/bin/g++-14` is the measured local combination. A system default compiler that is too new is not fixed by choosing a toolkit alone. Do not override the toolkit version or edit system headers to bypass validation.

### CUDA build works but trials select CPU

Inspect `resources.json`: actual free VRAM, estimated residency, context, compiled backend and selection reason. Background model allocations may prevent GPU selection. Verify `LD_LIBRARY_PATH` for the selected toolkit and actual runtime metadata. Do not equate compilation with GPU use.

### Cannot resume the original engine

Restore the exact engine and required dependency closure from backup, or start a new run with new output paths. Preserve historical reports. The pinned sm00 engine checksum is:

```text
9223f87d30f0ba75666619174c8939f2cec1998999b3801e946e00914b00aaa5
```

### Misleading split error or failed full study

Historical scheduler output used `worker evaluation uses an undeclared or held-out split` for split, checkpoint and generation-health failures. Do not infer held-out access from that old message alone. The repaired scheduler emits structured reasons distinguishing split/held-out access, checkpoint mismatch, invalid evidence, incomplete generation, inference timeout, grading infrastructure and other unhealthy completion, reconstructed from bound evidence. Inspect `evaluation-failure.json`, process evidence and retained paths. `worker-progress.json` preserves training/export/reload even when evaluation fails; exit zero never overrides an unhealthy report. A completed checkpoint can be evaluated in a fresh compatible protocol without rewriting the original failed outcome (section 11).

### Frozen token cap or preference conditioning failure

Inspect both actual teacher and student prompt lengths, EOS and assistant spans, chosen/rejected IDs and conditioning. Reject unsupported samples before a long run. A new cap or window policy requires a new configuration, resource estimate and trace identity. Do not rewrite a frozen request or substitute a different objective.

### Historical downloaded helpers

Browsers may save duplicates as `name (1).sh`; a command using `name.sh` still runs the old file. Preserve such helpers with their historical reports and verify their checksum/version when reading old evidence. Use the native preparation, diagnostic and evaluation commands for new repaired-engine work. Do not rerun historical helpers merely to reproduce already saved results.

## 14 Daily operation and offline backup

For ordinary maintenance on sm00:

```bash
cd "$HOME/Developer/lec_challenge"
source state/lec-sm00-env.sh
"$PWD/target/release/lec-distil" assets-doctor --base offline_bundle/base --offline
"$PWD/target/release/lec-distil" sandbox-doctor --archive-directory offline_bundle/images
"$PWD/target/release/lec-distil" --help
```

Before grading after a runtime/kernel/image/policy change, rerun sandbox preflight. Before a research run, also check disk, RAM and free VRAM. Explicitly choose CPU or CUDA and keep the same intended environment in child processes.

A reproducible offline installation needs the lockfile/toolchain and build closure or compatible executable; base/tokeniser/licences; every required teacher/BitNet payload and verified metadata; exact image archives/policies; reports/checkpoints; content-addressed engines and native backend dependencies. It also needs the dedicated Podman configuration. A Git clone and `assets.json` files alone are insufficient.

Back up `state/engines` deliberately even though state is excluded from Git. Recreate session run-root state after login; preserve graph-root image storage and verified archives. Do not prune models when disconnected or reinterpret old prune markers. Never include provider credentials in a bundle.

## 15 Implemented repair, compatibility and measured limits

The bounded repair is implemented in the source developed on `repair/training-20261022` for local integration into `main`. [The repair report](repair-20261022.md) records source causes, regression tests, exact commands and scientific hypotheses that remain unproved.

| Change | Installation or workflow consequence |
|---|---|
| Structured evaluation failures and durable progress | Distinguish evidence violations from generation/grading failures; retain completed checkpoints after evaluation failure. |
| Shared executable eligibility | Planner dispatch, prewarm and worker reconstruct actual teacher/student encodings, exact sample IDs/order, spans, content origins and exclusions. Preference chosen IDs resolve to their real content and common conditioning. |
| Native selected-training diagnostic | Inspect exact optimisation data; separately request NLL/accuracy/EOS measurements and paired sandbox-graded generation. |
| Feature safeguards and diagnostics | Alignment/gradient tests, loss/update records and bounded train-only health observations are implemented; the historical collapse cause remains unproved. Feature-only objectives remain explicit research methods. |
| Separate phase runtimes | Freeze capture, optimisation and evaluation runtimes/resources independently; reject unsupported requests before optimisation. Adapter inference avoids loading two base models simultaneously. |
| Honest automatic selection | Zero-score and incomplete-seed groups do not become automatic quality winners. Exploratory continuation needs a declared new budget/protocol; latency alone does not establish learning. |
| Launch/CUDA policy | Keep locked offline defaults, permit explicit online preparation, and validate toolkit/compiler compatibility before expensive compilation. |
| Native doctors and explicit restore | Inspect assets/store/image/archive identities, rename conflicts only on request and restore only verified exact archives. Full sandbox preflight remains mandatory. |

New study plan/spec/freeze/checkpoint-manifest schemas are **2**; historical schema-1 readers retain their original semantics. The prepared selection receipt is schema 2 and automatic study selection is schema 3. New diagnostic/rejection schemas are 1; partial worker progress is 2. The success-only worker outcome and frozen evaluation schemas remain 1. These are different documents, not one global schema version. Existing freezes, scores, failures and tensor caches are never upgraded in place. Changes to engine, limits, conditioning, objectives, image or runtime require new compatible bindings and fresh outputs.

The measured dependency/toolchain configuration was Rust/Cargo **1.98.1**, Candle **0.9.1**, cudarc **0.16.6**, tokenizers **0.23.2** and libc **0.2.190**, with no lockfile change. Both CPU machines had their locked dependencies cached, so no fetch was needed. The remote system default Rust 1.93.1 was rejected; validation used already-installed 1.98.1 through explicit toolchain paths without a global change.

| Repair check | Observed result |
|---|---|
| `cargo fmt --all -- --check` | Passed |
| `cargo test --locked --offline --lib` | 320 passed, 2 ignored, on laptop and isolated remote clone |
| `cargo clippy --locked --offline --all-targets -- -D warnings` | Passed |
| `cargo test --locked --offline --test run_launcher --example platform_launch` | 4 shell launcher tests and 12 native launcher tests passed |
| `cargo build --locked --offline --release` and release `--help` | Passed |
| Remote `cargo check --locked --offline --all-targets` | Passed with Rust 1.98.1 |
| Original-tokeniser QA, gate reporting and train-only sandbox adversarial check | Explicitly executed and passed; exact filters are in the repair report |
| Native asset/store doctors and full sandbox preflight | Passed for the actual repair assets/store; exact archive restoration was not needed or exercised |
| CUDA 12.9/g++-14 minimal compile probe | Passed; toolkit 13.3 rejected early. Full repaired-engine CUDA/backward execution and ARM hardware validation remain outstanding |

The real bounded smoke used one selected `matrices.diagonals` train example, seed 1729, one SFT update, rank 2, one job and CPU/F32/four-thread execution. Its complete sequence was 420 tokens including EOS, with 182 supervised tokens; no clipping was applied. Under identical decoding capped at 512 new tokens, base and adapter both stopped at 245 tokens and both scored **0/1** with Rust compilation failures. Assistant NLL fell **0.30092682864839876 → 0.2844318939546998** while token accuracy stayed **166/182**. Export/reload and unchanged base identity were verified within the 15-minute total budget. This establishes plumbing and an objective change, not coding improvement, generalisation or promotion.

The smoke engine is `61645f9cf2872284da85d954f187d709da0c020de3437cc366575300d4284558`; final delivery engine `6d3e8eba5ef0af2b8f680c2507c37f6bc6891da1891b2a3587c56db80a0db95d` independently reproduced NLL with zero new updates or generation. Their distinct protocols and reports remain intact. Tiny numerical overfit and fresh-process fixed-logit reload tests establish numerical invariants, not real assessment evidence. New preference/BitNet optimisation and the cause of historical feature collapse remain unverified. No held-out evaluation, broad retraining or new teacher search was run for this repair.

For daily use: enter the actual checkout and source its existing environment if present; build with the lockfile (fetch online first only for a missing dependency cache); inspect assets/store/archive and pass sandbox preflight; inspect exact request/checkpoint eligibility; run bounded selected-training diagnostics in fresh directories; compare compatible development evidence only when justified; then freeze any new experiment with complete seed groups and fixed budgets.

No new Python training stack, PostgreSQL deployment, Kubernetes, AARNN integration or paid provider access is needed for these repairs. The Rust application dependencies remain authoritative in Cargo.toml/Cargo.lock. Original provider-based assessment completion separately requires cleared output-training rights, exact model/price snapshots, actual authorised calls and recorded spending; local open-weight teachers do not satisfy that requirement.

## 16 Sources and verification record

Repository source and documentation: repaired `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `run.sh`, `run-gpu.sh`, `src/main.rs`, `examples/platform_launch.rs`, `examples/package_sandbox.rs`, `src/distillation/diagnose.rs`, `src/assets.rs`, `src/sandbox_doctor.rs`, `src/sandbox_archive.rs`, `sandbox/Containerfile.v2`, [reproduction guidance](reproduction.md), [GPU guidance](gpu.md), [BitNet guidance](bitnet.md) and [the repair report](repair-20261022.md). The starting commit alone does not describe the repair.

Observed sm00 results: user-supplied bootstrap, sandbox, CUDA, teacher-cache, context-follow-up and SFT logs from 19 to 22 October 2026, including `Pasted text(20261022-112442).txt`. The current instructions supersede version 1's unverified crun status and temporary-store advice.

Official references retained from the earlier installation review (not revalidated during this documentation-only update):

- Cargo fetch: https://doc.rust-lang.org/cargo/commands/cargo-fetch.html
- Cargo update: https://doc.rust-lang.org/cargo/commands/cargo-update.html
- Cargo clean: https://doc.rust-lang.org/cargo/commands/cargo-clean.html
- Podman configuration and rootless operation: https://docs.podman.io/en/stable/markdown/podman.1.html
- NVIDIA CUDA 12.9 Linux installation and supported environments: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-linux/index.html
- Rust installation: https://www.rust-lang.org/tools/install

Command names/options were checked against actual release-binary/Cargo help and their source contracts during this documentation revision. `git diff --check` and syntax-only `bash -n` validation of the Bash snippets passed; the installation snippets were not executed. Historical sm00 observations retain their original scope. Repair verification logs and actual numerical reports are under `reports/repair-20261022/`; host installation, package changes, downloads, archive restoration and new training runs were not repeated for this documentation-only update.
