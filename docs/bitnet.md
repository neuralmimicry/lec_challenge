# Native ternary BitNet student

The optional student uses the official Microsoft BitNet b1.58 model. It is not
a post-quantised Qwen model renamed as BitNet. Rust owns asset acquisition,
model adaptation, losses, provenance, export, orchestration and verification.
The explicitly authorised C++ exception is the pinned upstream BitNet.cpp
inference engine and its native kernels.

## Immutable assets and licences

| Asset | Immutable identity |
| --- | --- |
| Original model | `microsoft/bitnet-b1.58-2B-4T` |
| Original revision | `04c3b9ad9361b824064a1f25ea60a8be9599b127` |
| Packed tensor file | `model.safetensors`, 1,178,623,988 bytes |
| Tensor SHA-256 | `8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b` |
| Engine | `microsoft/BitNet` at `0b341e582afbf9e1011f24744b554c96a3477eb5` |
| Engine submodule | `isHuangXin/llama.cpp` at `390c307752ab78fd8189f359d6954c9ba1be74af` |

The original model and engine declare MIT. Original notices are preserved under
`docs/bitnet/provenance`, including llama.cpp, cpp-httplib and json.hpp notices.
The gguf-py notice is also preserved because it exists in the source tree; no
Python converter or model code is executed.

The Rust Hub downloader completed and publisher-verified the original model and
its tokeniser/configuration/licence assets under
`offline_bundle/students/bitnet-2b-4t`. That protected root is excluded from poor
teacher pruning. Its dedicated validator permits only the reviewed BitNet data
configuration and exact tensor pin. The stricter assessment/Qwen loader still
rejects remote-code metadata. Retained `auto_map` strings are data; they are
never imported or executed.

## Architecture and actual quantisation-aware training

The model has 30 decoder blocks, hidden width 2,560, 20 query heads, five KV
heads, intermediate width 6,912, RoPE base 500,000, context 4,096 and 128,256
tied embedding tokens. Its activation is squared ReLU. Besides ordinary input,
post-attention and final RMS normalisation, it has attention and feed-forward
subnormalisers before the output/down projections.

HF U8 weights pack four output rows per byte along the first axis. For packed
row `r`, group `g` and column `c`, the trit is
`((packed[r,c] >> (2*g)) & 3) - 1`; output row is
`g * packed_rows + r`. Reserved code three is rejected. Official offline
AutoBitLinear multiplies the ternary matrix by its stored weight scale. The
native decoder preserves that multiplier.

For q/v adaptation, low-rank latent updates enter the weight quantiser during
every training forward: `Q(W + alpha/rank * B*A)`. The quantiser uses the original
fixed weight scale, ternary rounding/clamping and an identity straight-through
derivative. This preserves the initial official ternary lattice. It differs
from online pretraining's changing mean-absolute scale and is recorded as such.
Activations use symmetric per-token int8 quantisation with round-to-even and
an identity straight-through derivative. Floating output-side LoRA is not used.

The base is frozen. Constant projection backward computes input gradients while
omitting base-weight gradients. `effective_trit_identity` hashes the actual q/v
ternary values and counts changes from the frozen base. Changed floating latent
parameters alone cannot establish an adapted ternary student. The worker must
retain actual finite losses, nonzero gradients, optimiser steps, changed
effective trits, saved tensors and reload evidence.

The full differentiable implementation uses explicit CPU F32. Its approximately
ten-GB decoded base and additional QAT graph reserve exceed this host's eight-GB
GPU. A portable resource guard runs before loading; the scheduler must also
account for longer-sequence activations and simultaneous jobs. There is no silent
CUDA-to-CPU fallback.

## Export and native inference

Rust writes GGUF directly; neither the upstream Python converter nor the native
quantiser CLI is required. HF first-axis packing is converted to I2_S's separate
128-value/four-by-32 interleaving. Each matrix retains its original scalar scale.
Norms and embeddings are written as exact F32 conversions of original BF16.
For a real adapter, Rust applies the same latent update and ternary quantiser
before writing I2_S, verifies its training manifest, and requires changed trits.

Each completed tensor fragment is hashed and durable. Assembly records complete
fragment boundaries, verifies retained prefixes and resumes after interruption.
An unpublished final rename is recovered rather than silently replaced. Existing
corruption, path traversal, symlinks, hard links and conflicting export inputs
fail closed. The baseline export is explicitly not a trained checkpoint.

The pinned engine's inherited BitNet graph selected SiLU, whereas the official
model configuration requires squared ReLU. The one-line correction is preserved
in `docs/bitnet/engine-relu2.patch`. The engine receipt binds that exact patch,
both source pins, CMake options, compiled executables/shared libraries and
licence evidence. An upstream converter's F32 path also divides the offline
scale, unlike HF AutoBitLinear's multiplication; the Rust direct I2_S path avoids
that inconsistent conversion. Native I2_S encoding is independently tested.

`BitNetRuntime` reloads the actual QAT adapter and original tokeniser in Rust.
It provides a correctness path with full-prefix recomputation. The faster
`BitNetCppRuntime` retains a headless CPU server and KV cache. Rust supervises its
loopback-only endpoint, clears inherited credentials from the child environment,
limits context/output/time/body size and threads, records affinity, and kills the
server on request failure. Original HF token IDs must match the native engine's
tokeniser probes before inference. Rust sends explicitly encoded BOS/chat/EOT
tokens rather than relying on a different engine chat renderer.

These blocking model backends must be constructed and called on the scheduler's
blocking worker. Their endpoint is evaluation-only until separately justified
quality and promotion evidence exists. An engine build or export alone does not
establish working inference, parity, quality or a successful training round.

## Commands and evidence boundaries

```sh
# Explicit public acquisition; no Python or executable model repository code.
cargo run --release --locked --offline --example hub_bitnet_assets

# Small synthetic numerical QA: actual STE/updates/trit changes and exact reload.
cargo run --locked --offline --example bitnet_operator_qa

# Official full-size synthetic forward/backward and measured CPU/RAM smoke.
cargo run --release --locked --offline --example bitnet_base_smoke -- --tokens 16

# Export a baseline, or add --checkpoint <verified-real-worker-output>.
cargo run --release --locked --offline --example bitnet_export -- \
  --output offline_bundle/students/bitnet-2b-4t/model.i2_s.gguf

# Pinned headless engine receipt, tokeniser check and genuine baseline generation.
cargo run --release --locked --offline --example bitnet_cpp_smoke
```

`bitnet::cpp::bootstrap_engine(offline, jobs)` is the explicit Rust source/build
entry point. With `offline=true`, missing cached sources, submodules or binaries
are reported unavailable; no network request is attempted. An interrupted
CMake build reuses its existing objects. Normal model/runtime loading never
invokes an online bootstrap implicitly.

Measured micro-QAT evidence is in
`reports/comparative/bitnet/operator-qa.json`: eight normalised-SGD steps,
57 changed effective trits out of 96, finite nonzero gradients and exact reload.
It is plainly labelled synthetic, excludes official pretrained weights and does
not establish coding quality or provider-derived learning. Full-size resource,
training, native inference and parity claims require their own completed reports.

The completed baseline native-engine smoke is retained in
`reports/comparative/bitnet/cpp-smoke.json`: the exact exported original model
generated a complete ten-token response in 7.876 seconds using CPU/F32 and four
threads, after original-tokeniser ID parity probes. Lifecycle QA is in
`reports/comparative/bitnet/supervisor-qa.json`: the owner thread survived its
creator worker's exit, normal drop killed and reaped the child, and killing an
isolated holder caused the kernel to kill its supervised controlled process.
Those controlled-process tests do not claim coding quality. The crash-safe
C++ lifetime implementation currently requires Linux; unsupported platforms
must use an independently supported native Rust path or report unavailability.
The dedicated HTTP driver also owns its own persistent runtime, so moving a
model between async workers cannot drop a Tokio runtime inside another runtime.
It disables idle HTTP connection pooling: a long separate model load can outlive
the server's keep-alive timeout while the driver's runtime is parked. Fresh
loopback connections avoid reusing that stale socket without automatically
replaying generation requests. Shared engine bootstrap holds an exclusive,
cancellable-wait file lease across receipt checking, checkout, build and receipt
publication, so concurrent first-time evaluations cannot race the same files.

The native server explicitly requests F32 key and value caches, matching the
Rust reference dtype; upstream's default caches were F16. Execution metadata
records `explicit_original_tokens_cpu_f32_kv_v2` and both cache dtypes. This
precision correction did not establish numerical parity. The original completed
comparison in `reports/comparative/bitnet/parity.json` failed its predeclared
maximum absolute log-probability error of 0.10. The fresh F32-cache comparison
in `parity-f32-kv.json` also failed: its three maximum errors were 0.661798,
0.322420 and 0.278463. All three most likely tokens matched. Both failed reports
and raw probes remain retained, and the criterion has not been relaxed. The
engine remains evaluation-only; matched top predictions and a working smoke
response cannot substitute for validated numerical equivalence or task learning.

```sh
# Fresh output is required; preserve earlier numerical failures.
RAYON_NUM_THREADS=4 cargo run --release --locked --offline \
  --example bitnet_parity -- \
  --output reports/comparative/bitnet/parity-new-attempt.json
```

The independently reconstructed schema-2 diagnostic pins and re-executes its
Rust binary, archives the compiled numerical source identities and freezes its
checkpoint/base/tokeniser/GGUF/export/engine/cache protocol before loading the
backends. It retains all original native responses and the complete Rust
vocabulary log probabilities. The three prompts, native top 16, equal top-token
requirement and 0.10 maximum error remain fixed. A provisional aggregate is not
published as a verified report until the independent reader reconstructs every
observation. Validated completed probes resume under the same frozen plan;
another executable or numerical protocol requires a fresh output.

Read-only diagnosis also found that the pinned native activation quantiser uses
`roundf`, which rounds ties away from zero, and a zero-only scale guard. Rust and
the original HF semantics use round-to-even with a `1e-5` minimum absolute scale.
The explicit numerical engine V2 combines the original ReLU² correction with
`nearbyintf` and that minimum. Its source and build live separately under
`state/engines-src/bitnet-quant-v2` and `state/engines/bitnet-quant-v2-build`;
the reviewed patch is `docs/bitnet/engine-quantisation-v2.patch`. Engine schema 1
and its original receipt still verify, while schema 2 requires the new exact
patch and declares `explicit_original_tokens_cpu_f32_kv_quant_v3`. This is a
specific numerical correction, not evidence that the remaining mismatch has
been resolved. A fresh actual report is required.

The completed, independently reconstructed V2 attempt is retained in
`reports/comparative/bitnet/parity-quant-v2.json`. It failed the unchanged 0.10
criterion: the three maximum absolute log-probability errors were 0.661798,
0.471329 and 0.291747. All three top predictions agreed. Its full Rust
probability masses were approximately 1.00000161, 1.00000271 and 1.00002191;
the verifier checks the complete vocabulary rather than renormalising a
retained top-k subset. This is a failed numerical observation, and establishes
neither numerical equivalence nor useful training. Its original plan, source
archive, full vectors, raw native probes and failed report remain intact.
Subsequent orchestration/source changes make that implementation receipt
historical; they cannot silently qualify today's backend. Further diagnosis
requires per-layer oracle evidence before another numerical correction.
The subsequent bounded [per-layer findings](bitnet/per-layer-findings.md)
verify exact original embedding rows, identify an initial RMS arithmetic
difference and compare native integer projection arithmetic with actual
selected-row Candle operations. Operand aliasing and intervention limits
remain explicit; the probe does not explain the complete failed comparison
or qualify the backend.

`nearbyintf` follows the executing thread's floating-point rounding mode. This
protocol relies on Linux `exec` initialising the default `FE_TONEAREST`
environment and the pinned engine preserving it when creating model workers.
Inspection of the pinned GGML/BitNet source found no `fesetround` or other
rounding-mode mutation; its SIMD rounding operations specify nearest rounding
without changing the thread mode. The engine is closed, pinned inference code,
with no loaded model-repository code or arbitrary plug-ins. This is a concrete
protocol assumption, not a claim about an arbitrary modified engine. A future
mode-changing path or unsafe floating-point compiler flags requires fresh
validation and must not inherit these numerical observations.

```sh
# Explicit cached-only Rust bootstrap and separately versioned engine probe.
# Fresh export preserves the original export implementation receipt as well.
RAYON_NUM_THREADS=4 cargo run --release --locked --offline \
  --example bitnet_parity -- --quantisation-v2 \
  --gguf reports/comparative/bitnet/schema2/model.i2_s.gguf \
  --output reports/comparative/bitnet/parity-quant-v2.json

# Add one or more explicit, exact-checkpoint evaluation attachments.
# A valid failed numerical report can be attached and remains unqualified.
RAYON_NUM_THREADS=4 cargo run --release --locked --offline \
  --example bitnet_parity -- --quantisation-v2 \
  --gguf reports/comparative/bitnet/schema2/model.i2_s.gguf \
  --output reports/comparative/bitnet/parity-quant-v2.json \
  --evaluation /path/to/frozen/evaluation
```

## Reusable backend selection and export cache

`bitnet::parity_run::{run, run_async, ParityRunOptions}` provides the same frozen
producer to the public command and the thin example wrapper. The asynchronous
entry point owns a dedicated native thread, so tensor work and the native
driver do not block Tokio's executor or inherit an entered runtime. The
executable must already be its verified immutable snapshot. A completed failed
comparison returns its explicit report as data; command wrappers return
non-success. Infrastructure failures retain their failure sidecar.

`BitNetBackendOptions` selects an optional engine manifest, GGUF and numerical
report. For a new evaluation, `prepare_backend` validates the requested
checkpoint/base/tokeniser before any bootstrap or export. A supplied numerical
report is independently reconstructed and selects its exact recorded
engine/GGUF/export tuple; matching explicit paths are allowed and conflicting
paths fail before mutation. A valid failed numerical report can accompany
descriptive measurements, but cannot qualify them for ranking. An existing
explicit GGUF must have the correct original source and adapter binding, or
valid resumable export authority, before the default engine is bootstrapped.

New default exports share
`state/bitnet-exports/<checkpoint-tensor-sha256>/<exporter-sha256>/model.i2_s.gguf`.
Train and development evaluations refer to this one checksum-bound export.
The existing exporter retains its exclusive lock, fragment checksums and
resumable assembly; simultaneous owners fail explicitly. This cache is outside
teacher-search pruning and currently has no automated pruner. Historical and
explicit export paths remain unchanged.

Saved evaluation plans take precedence over new defaults. The read-only
`resolve_frozen_backend` verifies their exact recorded tuple and rejects
conflicting overrides without acquiring assets or exporting again. Each large
artefact is hashed once within that resolution; no file-stat cache substitutes
for verification. Historical Rust-only plans return no native backend and keep
their original Rust runtime. Native host/ISA suitability is checked before
execution, while historical receipt inspection remains platform-independent.
