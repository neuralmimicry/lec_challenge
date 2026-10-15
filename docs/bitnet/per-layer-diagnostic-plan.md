# BitNet per-layer numerical diagnostic plan

Status: proposed investigation, based on read-only source and artefact inspection.
No tensor capture or new numerical experiment has been executed for this plan.
The existing failed measurements, executables, exports and tolerances remain
unchanged. This work is numerical diagnosis on the three fixed synthetic
prompts; it does not use assessment tasks or establish learning.

## Evidence and question

The completed schema-2 comparison in
[`parity-quant-v2.json`](../../reports/comparative/bitnet/parity-quant-v2.json)
records the official base checkpoint, F32 key/value caches and the separately
reviewed quantisation correction. Its maximum absolute top-16 log-probability
errors were **0.661798477, 0.471328735 and 0.291747093**. All three top token IDs
agreed. These observations fail the fixed **0.10** limit. They do not locate the
fault or show that floating-point accumulation is the cause.

The diagnostic must answer two different questions:

1. Where do the executions first cease to have identical corresponding values?
2. Which operation introduces or amplifies the discrepancy when supplied with
   the **same incoming values**, especially at an int8 rounding boundary?

Small upstream differences and a faulty downstream operation are different
findings. Comparing later layers on already different inputs cannot distinguish
them. Report both the first bitwise difference and the first discrete activation
code difference, without creating an intermediate tolerance that can qualify
the model. The existing end-to-end criterion remains authoritative.

## Inspected computation

The native engine is pinned to BitNet commit
`0b341e582afbf9e1011f24744b554c96a3477eb5` and llama.cpp commit
`390c307752ab78fd8189f359d6954c9ba1be74af`, with
[`engine-quantisation-v2.patch`](engine-quantisation-v2.patch). Its saved manifest
is `state/engines/bitnet-quant-v2-build/engine-manifest.json`. The inspected native
paths below are relative to its `3rdparty/llama.cpp` directory.

| Boundary | Rust reference | Pinned native path and observation |
| --- | --- | --- |
| Embedding and weights | `src/bitnet/codec.rs` decodes official first-axis packing; unquantised BF16 tensors become F32. | `src/models/bitnet.cpp` loads `token_embd.weight` and per-layer projections. `src/bitnet/export.rs` writes unquantised tensors as F32 and ternary matrices as I2_S with their original scalar scale. |
| RMS normalisation | `BitNetStudent::inner` uses `candle_nn::LayerNorm::rms_norm`; Candle 0.9.1 squares, reduces, divides by width, adds epsilon, takes a square root, divides and multiplies by the norm weight. | `llama-graph.cpp::build_norm` calls `ggml_rms_norm` then multiplies by the weight. `ggml-cpu/ops.cpp::ggml_compute_forward_rms_norm_f32` accumulates F32 squares into `ggml_float` (`double` in `ggml-cpu/vec.h`), then uses F32 mean and reciprocal square root. The multiply may be fused. |
| Projection input | `src/bitnet/quant.rs` computes `s = 127 / max(max(abs(x)), 1e-5)`, ties-to-even int8 codes, then F32 `code / s`. | Patched `ggml-cpu/quants.c::quantize_row_i8_s` computes the same declared scale and codes with `nearbyintf`, saving `s` and the sum of codes in private work memory. The native rounding environment remains an explicit assumption. |
| Projection reduction | `Projection::forward` multiplies dequantised F32 activations by F32 effective ternary weights using Candle matmul. | I2_S kernels accumulate biased two-bit codes times int8 activations. The caller subtracts the activation-code sum and applies `weight_scale / s` once. `ggml-cpu/repack.cpp::tensor_traits_i2s` intercepts I2_S operations when enabled; `ggml-cpu.c` also has a path. Both call `ggml-cpu-i2s.c` GEMM/GEMV routines. |
| RoPE | `src/bitnet.rs` constructs F32 frequencies from `theta=500000` and uses Candle `rope_slow`, rotating the two half-vectors. | `src/models/bitnet.cpp` calls `ggml_rope_ext`; `llama-model.cpp` selects NEOX RoPE for BitNet. Compare actual operation parameters and the computed positions, rather than assuming the defaults agree. |
| Attention | Rust repeats five K/V heads into twenty heads, computes QK products, divides by `sqrt(128)`, applies the causal mask, softmax and weighted V sum. | `llama-graph.cpp::build_attn` writes and reads F32 KV caches; `build_attn_mha` uses `kq`, `kq_soft_max`, `kqv` and `kqv_out`. Scaling and masking occur inside native softmax. Flash attention is explicitly off in the existing launcher. |
| Feed-forward and residuals | Input norm; gate and up projections; `relu(gate)^2 * up`; intermediate-width RMS subnormalisation; down projection; residual add. | `src/models/bitnet.cpp` and `build_ffn` expose `ffn_norm`, `ffn_up`, `ffn_gate`, `ffn_relu`, `ffn_sqr(relu)`, `ffn_gate_par`, `ffn_sub_out`, `ffn_sub_norm`, `ffn_down`, `l_out`. The reviewed patch selects ReLU squared. |
| Output | Final RMS norm; F32 tied-embedding matmul, without BitLinear activation quantisation; stable full-vocabulary log probabilities. | `result_norm`, then tied `tok_embd` matmul and `result_output`. Capture raw logits before sampler processing and use the same full-vocabulary log-sum-exp calculation. |

The original configuration has 30 layers, hidden width 2,560, intermediate width
6,912, 20 query heads, five KV heads, head width 128, vocabulary 128,256 and RMS
epsilon `1e-5`. The I2_S execution contract is Linux x86_64 with 128-value blocks.
Check the actual tensor metadata, embedded scales, absence of additional
external projection scales and absence of output-side native LoRA branches.
An adapted student is already merged through the ternary quantiser at export;
loading an additional native LoRA branch would be another computation.

## Freeze a separate diagnostic protocol

Use a fresh output directory, for example
`reports/comparative/bitnet/layer-diagnostic/round-001`. Before loading a model,
write an immutable `plan.json` binding:

- The failed comparison and plan hashes; base/checkpoint, tokeniser, config,
  GGUF, export and native manifest hashes; every loaded shared-library hash.
- The exact Rust diagnostic executable, CPU build receipt, Cargo.lock and
  archived implementation/header sources. Record Candle's resolved crate
  source identity as well as its lockfile version.
- The three **original token arrays** from the saved plan, original positions
  `0..S-1`, one sequence, its output-row mask, all native context parameters,
  CPU threads/affinity, SIMD requirements, F32 caches and flash-attention off.
- A named capture manifest, layout transforms, binary formats, size limits,
  deadlines and comparison metrics. No sampling or prompt re-tokenisation is
  needed. The saved arrays have lengths 14, 16 and 21; a 32-token bound is a
  guard, not permission to truncate or add tokens.
- `production_eligible=false`, `quality_evaluation=false`, and the distinct
  protocol name. A diagnostic receipt is never a numerical-equivalence
  attachment or a promotion report.

Keep the primary native run's final-layer output-row selection identical to
the failed run. Native `bitnet.cpp` selects `inp_out_ids` before the last
feed-forward block; Rust currently computes all rows. Compare the selected
last row at corresponding downstream cuts. Requesting logits for every token
changes final-layer matrix shapes and can select a different kernel. If useful,
run that as an explicitly separate shape experiment after the primary capture.

## Capture through existing interfaces

Implement orchestration, reference taps, decoding, storage and comparison in
Rust. Native inference remains the existing, justified upstream C++/C
interoperability boundary; this plan requires no new kernel or graph patch.

The pinned `llama.h` exposes `llama_context_params.cb_eval` and
`cb_eval_user_data`. Its `ggml-backend.h` defines the scheduler callback:
`ask=true` selects nodes; `ask=false` observes a completed node and returning
false cancels computation. A Rust diagnostic can use reviewed bindings to the
**pinned** C ABI, link the exact verified native shared libraries, pass the saved
token IDs directly in a `llama_batch` with explicit positions and sequence ID,
and capture data with `ggml_backend_tensor_get`. Pin the bindings' struct layout
and enum values to these headers; do not reuse another llama.cpp ABI. Record
the actual loaded library paths and bytes, not only the intended link paths.

The upstream `examples/eval-callback` and `common/debug.cpp` demonstrate the
callback, but their text prompt tokenisation and abbreviated tensor printing
are insufficient for this diagnostic. Capture complete selected tensors as raw
little-endian F32/I8/I32 buffers with named shape, dtype, byte strides and
canonical logical axes. Preserve physical source bytes when a view is
non-contiguous; calculate and verify each logical element address against the
buffer bound before canonicalisation. Do not hash padding as tensor values.

Use diagnostic-only Rust taps at the actual operations in `BitNetStudent`;
the existing `forward_trace` returns only a selected layer output and
head-averaged attention, which cannot locate intra-layer or per-head errors.
Taps should copy values after the existing operation without replacing its
arithmetic. No production-source changes are part of the present plan.

Native names are **not unique keys**. Qcur/Kcur are named at projection,
reshape and RoPE boundaries; `norm` appears several times per layer, and
`l_out` is named again after the control-vector path. Identify nodes by
observation ordinal, layer, op, shape, source-node identities and an occurrence
index. Pair `MUL_MAT`, view/reshape and `ROPE` separately. Verify the semantic
mapping from recorded parents; fail on missing or ambiguous boundaries.

Callbacks and synchronisation can change graph splitting or fusion. Retain an
unobserved native decode and an observed decode with identical parameters.
Compare their raw logits and report any differences before attributing a
trace to the original path. If observing a node changes its computation, label
that capture as an instrumentation intervention and do not silently substitute
it for the original execution. Repeat the same check for Rust taps.

## Staged captures and bounds

Run the two model backends serially under the existing resource lease. Avoid
loading the roughly ten-GiB Rust reference alongside native inference. Schedule
this work after active teacher, training and sandbox pressure work completes.

| Stage | Captures | Hard artefact bound |
| --- | --- | --- |
| A: locate a layer | All three original prompts; embedding, each of 30 layer residual outputs, final norm and last-token full-vocabulary logits, both backends. Capture every available token row before final-layer pruning. | 64 MiB tensor payload in total. With 51 prompt tokens, layer outputs alone are about 30 MiB across both backends. |
| B: isolate the operation | Prompt 0 first, because it has the largest observed final error; its first differing layer and preceding layer, at most 64 named tensors per backend. Capture complete relevant token rows, not only the final token. | 128 MiB tensor payload. Other prompts receive separate frozen follow-up plans if needed. |
| C: same-input projection/normalisation oracle | At most three selected projections, three token rows each, and eight output-weight rows per projection, aligned to native four-row boundaries. Keep corresponding original packed bytes, trits and scales. | 8 MiB tensor/oracle payload; 4 MiB JSON metadata per stage. |

Stage B cuts, in execution order, are: layer input; pre-weight and post-weight
attention RMS norm; q/k/v projection input and output; q/k before and after
RoPE; logical F32 cache rows; unscaled QK; scaled and masked QK; per-head
probabilities; weighted-V output before and after head concatenation; attention
subnorm; output projection; attention residual; feed-forward norm; raw gate
and up outputs; ReLU; square; gate/up product; feed-forward subnorm; down
projection; final residual. Capture final RMS and output logits if the first
discrepancy occurs there. Do not compare native raw `kq` directly with Rust's
already scaled/masked scores. Compare only live cache positions and verify
that every additional padded cache position is masked out.

Allow at most 16 MiB of queued capture buffers; write through a bounded Rust
worker and atomically publish checksum-bound files. Copy callback values while
their native allocation is valid. A full queue, absent tensor, callback error,
ABI mismatch, deadline, non-finite value or byte-cap breach cancels the run and
writes an explicit incomplete report; never drop a tensor and report success.
Use owned child/process cleanup and a 30-minute total wall deadline with
periodic progress. These are proposed execution bounds, not measured resource
results. No unrestricted whole-model or whole-vocabulary-per-layer dumps.

## Compare the same incoming values

First verify token IDs, positions, configuration, embedding rows, normalisation
weights, selected ternary rows and scalar scale bits. HF first-axis packing and
I2_S interleaving are different codecs; compare decoded trits in logical
`[output_feature, input_feature]` order. Reserved code 3 or a scale mismatch is
a codec/artefact failure, before an arithmetic investigation.

For each F32 pair report shape, finite counts, exact bit-equality count, maximum
absolute error, RMS error, relative L2 error, maximum-error coordinate and
bounded example pairs. Include signed-zero and ULP diagnostics with an explicit
definition. These are descriptive metrics; no invented per-layer pass epsilon.
Compare masks, indices, trits, int8 codes and integer sums exactly.

At the earliest suspect operation, evaluate both implementations with one
captured input row, then with the other captured input row. Preserve these as
**input interventions**, separate from the two natural forward traces:

1. **RMS normalisation:** retain input, norm weights and epsilon. Reconstruct the
   native F32-square/double-sum/F32-reciprocal path and the actual Candle path;
   retain means, reciprocal scales and outputs. Separate reduction/order
   effects from weight application or fusion. Oracle reconstructions are
   labelled reconstructions unless an actual native operation produced them.
2. **Activation quantisation:** retain maximum magnitude, its coordinate,
   minimum-scale branch, F32 scale bits, scaled inputs, int8 codes, integer
   code sum and dequantised outputs. Save nearest half-integer distance and
   the values around every code disagreement, bounded to 64 examples. Tiny
   F32 input changes can cross a rounding boundary even after ties-to-even is
   aligned. Compare scale and codes before blaming the projection reduction.
3. **Projection:** for code `c_j`, scale `s`, ternary value `t_ij` and weight
   scale `w`, compute a checked Rust integer oracle
   `I_i = sum_j(t_ij * c_j)`. Also retain the biased-code sum
   `B_i = sum_j((t_ij + 1) * c_j)` and `A = sum_j(c_j)`; verify
   `I_i == B_i - A` exactly. Reconstruct native F32 postscaling
   `(float(B_i) - A) * (w / s)` and compare with Rust's actual F32 matmul of
   `c_j / s` and `t_ij * w`. At width 6,912 these integer sums remain within
   int32 and the exact-integer range of F32, but the scaled products and their
   reduction need not match bitwise. Use F64 only as a separately labelled
   analytical oracle, never as a replacement production path.
4. **Kernel dispatch:** native batches of four token columns use GEMM;
   remaining columns use GEMV. Original lengths 14, 16 and 21 exercise
   different tails, and final output pruning introduces a one-row case.
   Record projection dimensions, row slices and actual backend/buffer
   metadata. Inspect both `repack.cpp` and generic paths; a source branch is
   not proof that it executed. A bounded standalone GGML operation can test
   the same captured rows in groups of one and four without editing a
   kernel, but label its changed shapes and dispatch as an intervention.
5. **RoPE and attention:** if q/k/v inputs agree, compare frequencies,
   positions, half-vector pairing, cos/sin and post-RoPE vectors before
   cache writes. Verify the five-to-twenty head mapping, live cache rows,
   causal mask, QK scale, per-head normalisation and V aggregation. Native
   softmax uses a double sum with its pinned vector exponential routine;
   Rust and native F32 labels alone do not establish identical accumulation
   or exponential arithmetic. Retain each stage rather than head averages.
6. **Non-linearity and output:** if inputs agree, isolate ReLU, square,
   elementwise product, intermediate-width norm and residual additions.
   Finally separate raw logit differences from full-vocabulary normalisation
   differences. Record native top-16 token IDs and compare their normalised
   values using the existing end-to-end calculation.

The native quantiser's private `wdata` is not exposed by the graph callback.
Capture its actual F32 input and output projection, and distinguish Rust
reconstruction of its codes from observation of native scratch memory. If the
already pinned library exports callable quantiser/kernel symbols, a reviewed
Rust FFI subprobe may invoke those exact bytes on copied inputs. First verify
symbol availability and ABI; do not assume internal symbols are public. If
unavailable, report that limitation and use the public GGML operation plus the
labelled oracle. Do not add a scratch-memory hook or kernel patch to claim
unavailable evidence.

## Artefacts, checks and completion

Proposed files are `plan.json`, per-backend `execution.json`, an append-only
capture journal, binary tensor payloads with `index.json`, `comparison.json`,
`comparison.md` and any explicit `failure.json`. Every capture entry binds
protocol hash, backend/checkpoint, prompt index, token positions, semantic cut,
native occurrence/op/parents, logical and physical layout, byte length and
SHA-256. A separate Rust reader verifies all hashes, reconstructs all metrics
and checks the declared capture set before marking the diagnostic complete.
Completed captures resume only under the same frozen executable and plan.

Before real execution, focused tests should cover strided views and axis
transposes; duplicate graph names; pruned output rows; I2_S 128-value packing;
quantiser ties/minimum-scale cases; biased-code correction; GEMM/GEMV shape
labels; padded masks; non-finite rejection; truncated/tampered buffers; queue
and byte caps; and interrupted/resumed capture journals. They verify capture
correctness and fail-closed reporting, not model quality.

The useful completion is a reproducible first-divergence report with same-input
evidence and an explicitly supported or unresolved cause. It may conclude that
several numerically different operations amplify one another. Only then propose
a minimal numerical change as a separately reviewed, freshly pinned protocol.
Re-run the unchanged three-prompt parity criterion after any such change. Until
it passes independently, BitNet remains numerically unqualified and excluded
from comparative ranking and assessment promotion.
