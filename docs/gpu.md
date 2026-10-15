# Explicit native CUDA inference

The optional `cuda` Cargo feature enables native Candle GPU inference. The ordinary assessment loader remains pinned to the original Qwen 0.5B revision, tokeniser and hashes; its default execution is CPU float32. `RuntimeOptions::cuda(0)` requests GPU zero and float16 explicitly. An unavailable feature, driver, allocation or kernel is an error, with no CPU fallback. Actual device and precision appear in runtime metadata and endpoint health.

Comparative models use independently verified Hugging Face assets, immutable revisions and publisher hashes. The runtime accepts the selected Qwen2 and common Llama architectures, checking dimensions, tensor shapes, SiLU, attention biases and linear rotary scaling. DeepSeek's factor-four rotary scaling and lack of QKV biases are preserved. Original tokeniser chat templates run in Rust MiniJinja with the original block-whitespace settings, Python-compatible string methods, one million execution units, recursion limit 64 and a 262,144-byte output writer. EOS IDs come from the original generation/model configuration. Sharded safetensors are supported. No downloaded executable code is run.

Comparative inference has a 2,048-token total context cap; the assessment runtime retains 4,096. HTTP work uses the existing bounded dedicated worker. Model computation is cooperatively cancelled at token boundaries, with another deadline check after synchronised logits; a hard interruption of an executing CUDA kernel requires process supervision. Fuel and the writer bound template execution/output, while the selected original templates remain trusted inputs: arbitrary intermediate template allocations do not have a separate process memory limit.

Weights, projections, values and the KV cache use the requested precision. Queries
and keys are promoted to float32 **before** their attention-score matrix product;
scaling, masking and softmax also use float32. Qwen 1.5B's original float16 score
product overflowed before a later cast could repair it. The original failed
endpoint log is retained under
`reports/comparative/gpu-serving-coding-teachers/`. A regression test reproduces
the overflow with finite inputs and verifies finite corrected probabilities.
The CPU float32 reference also produced complete code. This fix preserves the
differentiable training graph and keeps the weights/cache in float16 on CUDA.

Existing assessment checkpoint evidence fixes CPU float32. A trained assessment adapter served with any other device or precision remains evaluation-only, even if its tensor hash matches a CPU-gated checkpoint. Comparative teachers also remain evaluation-only. Device or precision changes require matching quality evidence before production eligibility.

## This workstation's compatibility preparation

The observed GPU is an RTX 2080, compute capability 7.5, with 8 GiB VRAM and driver 595.91.07. CUDA 12.9 and 13.3 are installed. The driver rejected CUDA 13.3's PTX 9.3 as unsupported. CUDA 12.9 emits compatible PTX 8.8, but its original headers conflict with glibc 2.43's non-throwing `sinpi`, `cospi` and `rsqrt` declarations.

The Rust `cuda_compat` example copies locally installed 12.9 headers into excluded application state, adds matching `noexcept(true)` to precisely six original declarations, and actually compiles a header-only check. Its manifest records original/patched header hashes, exact rules and the observed compiler. System files and numerical kernels are unchanged. Copied NVIDIA headers are not redistributed. Candle kernels are pinned to 0.9.1 to avoid the later patch's unsupported SM75 MoE specialisations. CUDA dependency kernels and the header-only compilation probe necessarily use NVIDIA's native compiler; model/application logic and compatibility preparation are Rust.

```sh
cargo run --locked --offline --example cuda_compat -- /usr/local/cuda-12.9 state/cuda-compat
export CUDA_ROOT="$PWD/state/cuda-compat"
export CUDA_PATH="$CUDA_ROOT"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
export CUDARC_CUDA_VERSION=12090
export CUDA_COMPUTE_CAP=75
export NVCC_CCBIN=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS="-I$CUDA_ROOT/include"
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64
export CARGO_TARGET_DIR=target/cuda
export RAYON_NUM_THREADS=4
cargo build --locked --offline --features cuda --release --bin lec-distil --example gpu_smoke
target/cuda/release/examples/gpu_smoke offline_bundle/base --cpu-baseline
```

When switching from previously compiled CUDA 13.3 kernels, run `cargo clean --release -p candle-kernels` in this target directory before rebuilding. The upstream kernel builder's file-time cache otherwise retains old PTX despite changed toolkit environment variables. A fresh CUDA target directory has no such stale work. This compatibility procedure is specific to the observed installation; other machines should use their supported toolkit/compiler combination.

## Actual QA evidence

The [first CUDA smoke](../reports/comparative/gpu-smoke/c92b5eef-f65d-44dc-ab44-692bd5e12ea1.json) loaded the actual pinned base in 12.647 seconds and generated a complete 29-token Rust `add` function. `nvidia-smi` observed the Rust process allocating 1,148 MiB; metadata records GPU zero, float16 and no fallback. First-generation time was 14.889 seconds, including initial kernel loading/JIT, so this measurement establishes working GPU execution rather than a speed advantage.

The reusable smoke records cold and warmed generations, parses their outputs as Rust and optionally runs the identical prompt on CPU float32. It never executes generated code on the host, opens a task split, trains an adapter or establishes assessment improvement. Each invocation writes a new UUID-named report under `reports/comparative/gpu-smoke/`, preserving earlier and user-run evidence.

The [matched warm/CPU smoke](../reports/comparative/gpu-smoke/47fa9803-6a1f-4766-9b2f-12b36fb237d5.json) used the same 57-input/29-output-token prompt and produced identical complete Rust source: 1.309 seconds for the GPU process's first generation, 1.046 seconds warm, and 5.653 seconds for the CPU reference. The GPU allocation observed for that process was 1,180 MiB. This is a single smoke observation on one laptop, rather than a controlled throughput benchmark or a quality improvement result.

After the precision fix, the [three teacher endpoint checks](../reports/comparative/gpu-serving-teachers-fixed/report.json)
all returned complete code with actual CUDA/F16 metadata and matching driver
process allocations. Qwen 1.5B generated 29 tokens in 1.949 seconds with
3,598 MiB observed; DeepSeek 1.3B generated 77 tokens in 1.785 seconds with
3,022 MiB observed; SmolLM2 1.7B generated 31 tokens in 0.771 seconds with
3,694 MiB observed. Each used its original chat template. Their different output
lengths/tokenisers make these smoke observations unsuitable as a throughput
ranking. Allocations are snapshots after generation, rather than peak VRAM.

## Local teacher commands

The short `run-gpu.sh` launcher delegates hardware discovery, compatible toolkit
selection and executable pinning to Rust. It honours process affinity and
cgroup CPU/RAM ceilings, checks measured VRAM and queries the native CUDA driver
when `nvidia-smi` is absent, including Linux ARM64. It selects a toolkit supported
by the installed driver and GPU architecture. An explicit CUDA request cannot
silently fall back to CPU. Automatic CPU decisions include their reason. AMD
ROCm and Apple Metal execution are not currently implemented; such hosts use
the explicit CPU path. These are capability statements, not ARM64 hardware QA.

`config/comparative-models.toml` records the initial dense models. The adaptive
search records additional immutable registries under each trial. The three
weaker dense alternatives have now been pruned through the owned-cache policy;
their historical commands below require exact rehydration. A retained Qwen 7B
teacher can instead be started with:

```sh
./run.sh serve --comparative --device cuda --dtype f32 \
  --base offline_bundle/comparative/qwen-coder-7b-q4_k_m \
  --config config/serving-teacher.toml
```

The launcher never reads a stored credential. See `docs/hub.md` for authenticated
downloads, exact range recovery and offline operation. The following commands
document the earlier dense-model smoke checks:

```sh
./run-gpu.sh serve --comparative --device cuda --dtype f16 \
  --base offline_bundle/comparative/qwen-coder-1_5b \
  --config config/serving-teacher.toml

./run-gpu.sh infer --comparative --device cuda --dtype f16 \
  --base offline_bundle/comparative/deepseek-coder-1_3b \
  --prompt 'Write a Rust function adding two i64 values.' --max-tokens 128

# Stop other local models first: the comparison keeps one GPU model resident.
./run-gpu.sh compare-models --device cuda --dtype f16
```

The teacher endpoint binds only to localhost:8081 and exposes `/health`,
`/metadata` and bounded `/v1/generate`. Load the coding teachers serially on this
8 GiB card. SmolLM2 is a general instruction control, not a coding-specialised
teacher. Local model developers remain distinct from the assessment's three
independent API businesses.

After building, a transient user service can keep the teacher running beyond the
terminal session. It needs no Hugging Face credential once the files are local:

```sh
systemd-run --user --unit=lec-comparative-teacher --collect \
  --property=WorkingDirectory="$PWD" \
  --property=Restart=on-failure --property=RestartSec=5s \
  --setenv=LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  --setenv=RAYON_NUM_THREADS=4 \
  "$PWD/target/cuda/release/lec-distil" serve \
  --comparative --device cuda --dtype f16 \
  --base offline_bundle/comparative/qwen-coder-1_5b \
  --config config/serving-teacher.toml

systemctl --user status lec-comparative-teacher
lec_teacher_pid=$(systemctl --user show --property=MainPID --value lec-comparative-teacher)

# This Rust client checks actual output and the supervisor's driver PID.
cargo run --locked --offline --example check_teacher_endpoint -- \
  --process-id "$lec_teacher_pid" --output reports/comparative/active-teacher

systemctl --user stop lec-comparative-teacher
```

The native `systemd-run`/`systemctl` utilities operate the host process supervisor;
all model and verification logic remains Rust. Stop the service before a new
comparison or before hosting DeepSeek/SmolLM2. To switch teachers, rerun the
service command with the relevant registered model directory, and pass the
matching `--model` ID to `check_teacher_endpoint`. Select a fresh evidence output
directory for each verification. A transient service is not an installation that
starts automatically after reboot.

## Comparative evidence boundaries

`compare-models` defaults to all 16 development tasks and a fixed decoding seed;
it has no held-out option. The roster, model assets, contracts, decoding settings,
engine binary and sandbox policy are frozen before generation. Raw generations
are published before grading, and verified saved work is reused after a crash.
CPU/RAM limits bound the graders that overlap one GPU generator. Every candidate runs in a
fresh isolated container with host-owned oracles and trusted audit evidence.
Incomplete, empty and errored generations remain failures in the planned
denominator. Missing coverage and generation/infrastructure errors return
nonzero status.

Reports under `reports/comparative/` retain original code, trusted grader
artifacts, descriptive scores and actual GPU metadata. These original-model
development comparisons do not establish learning, general superiority or a
shipping decision. They do not automatically become assessment training rows.
Output-derived DeepSeek students must retain its inherited licence and use
obligations; the ordinary Apache-only label would be insufficient.

Derive a reusable resource table from completed comparison groups with the
native Rust example below. It verifies linked raw/grader artifacts and original
summary counts, and retains per-task inference timings/token counts. Zero remote
inference charges are separate from unmeasured electricity, depreciation,
downloads and labour; cost per point of learning is N/A without a training gain.

```sh
cargo run --locked --offline --example comparative_resources -- \
  reports/comparative/resources \
  reports/comparative/coding-development \
  reports/comparative/control-development
```

The existing delivery ZIP/tar.gz describe the earlier CPU assessment snapshot.
New GPU source and comparative assets require a newly packaged and verified
delivery before claiming that an archive includes these capabilities.
