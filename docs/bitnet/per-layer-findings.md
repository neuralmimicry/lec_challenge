# Bounded native BitNet numerical findings

This is measured diagnostic evidence from the original base model, original
synthetic prompt zero and the first two native layers. It does not establish
coding quality, learning, assessment promotion or numerical qualification.
The failed schema-2 end-to-end errors and unchanged 0.10 criterion remain
authoritative. The first diagnostic changed no production source. A later,
separately tested RMS correction is described below; its bounded results do not
replace the end-to-end comparison.

## Retained execution

The native observer used the original schema-2 engine/GGUF/token tuple and
stopped explicitly at `l_out-1`. Its
`state/diagnostics/bitnet-layer-probe/round-001/index.json` retains 17,217,384
tensor bytes, original token positions, node/operand identities, complete
first-two-layer captures and separately labelled native-symbol interventions.
It records partial execution and instrumentation intervention, without a full
decode, full logits or an unobserved-logit control. Native ownership ended
before the separate Rust oracle ran.

The completed oracle report is
`state/diagnostics/bitnet-layer-probe/round-001-oracle-attempt-003-threads4/report.json`:
232,611 bytes, SHA-256
`ad232a7803aa88ef84502a351bc56743432b324cbe790f9b2dace3793328e93d`.
The frozen Rust oracle source identity is
`96653b4e6f30c609dff205e1e25b3aa4688e88bd7d4d0097ceec2c8ac1a46156`;
its source, executable, six passing focused tests and dependency/compiler
receipt remain under `state/diagnostics/bitnet-layer-probe/oracle-source/versions/`
with that identity. All referenced capture payloads were independently checked
against their recorded lengths and SHA-256 values.

Attempt 001 failed because the prototype expected the embedding name
`inp_embd`; the native builder had renamed it `embd`. Its failure report and
logs remain intact. A fresh oracle version resolves the explicit first RMS
operand link and verifies its `GET_ROWS` provenance. Attempt 002 then completed
using the ambient thread environment. Attempt 003 used explicit
`RAYON_NUM_THREADS=4`, `LEC_STUDY_CPU_THREADS=5`, `OMP_NUM_THREADS=4` and
`MKL_NUM_THREADS=4`; the two completed reports are byte-identical. Their output
directories and logs remain separate. No capture was regenerated to obtain
these findings.

## What the observations establish

All 35,840 embedding values for the fourteen original tokens match the original
BF16 embedding rows bit for bit after exact conversion to F32. The oracle
stream-verified the full original packed-tensor checksum and decoded only
71,680 embedding bytes, without allocating a full Rust model.

The first unweighted RMS operation provides a valid comparison on identical
original embedding inputs. A Rust reconstruction of the native F32-square,
double-sum, F32-mean and reciprocal-multiply path matches all 35,840 observed
native outputs exactly. Actual Candle 0.9.1 `LayerNorm::rms_norm` on those same
inputs differs by at most **0.0000123977661**, or **39 ULP**. No output is bitwise
equal in this particular comparison. This locates a concrete initial
normalisation arithmetic difference; it does not quantify its contribution to
the final failed log probabilities.

Reconstructing int8 quantisation immediately after those **unweighted** RMS
outputs produces two code disagreements. This is an explicit input
intervention: the natural model applies its affine norm weights before the
projection quantiser. It is not evidence that the actual weighted projection
inputs already have those two code disagreements.

For the first q/k/v projections, native-symbol subprobes on copied inputs match
the reconstructed activation codes, F32 scale bits and integer code sums
exactly. Their selected eight-row native GEMV outputs match both the integer
oracle and corresponding natural native outputs exactly: 24 of 24 values for
each projection. Across all fourteen captured token rows, the integer
postscaling reconstruction matches all 112 selected natural outputs per
projection exactly.

The actual selected-row Candle F32 matrix multiplication on the same captured
inputs differs slightly from the observed natural native projections:

| Projection | Maximum absolute F32 error, fourteen tokens × eight outputs |
| --- | ---: |
| q | 0.0000011920929 |
| k | 0.0000019073486 |
| v | 0.0000014603138 |

The Candle operation uses eight selected output rows, while the original native
projection has its full output width. This is a declared shape intervention;
its microkernel dispatch cannot be assumed identical to a full Rust model.
The measurement supports a real difference between F32 dequantised-product
reduction and native integer-dot/postscale arithmetic on these bounded inputs.
It does not prove that this explains the end-to-end 0.66 error. Three of
fourteen recorded projections were analysed; eleven remain explicitly deferred.

## Instrumentation limits and next diagnosis

Post-operation operand snapshots can contain overwritten in-place values. In
the first affine norm operation, operand capture 3 already has the same bytes
as weighted output capture 2, while original RMS output capture 0 differs.
The original capture 0 is therefore the valid pre-affine observation;
capture 3 must not be interpreted as the original unweighted input. Later RMS
operand snapshots also have unresolved reconstruction mismatches. Their raw
metrics are retained, but cannot yet establish native arithmetic faults or
valid comparisons on identical incoming values.

A subsequent diagnostic should capture or retain operands before the operation
can overwrite them, resolve physical aliases explicitly, and independently
verify affine norm weights and output pairing. It should then retain an
unobserved native-logit control and a natural Rust trace under the matched
thread/build contract, serialising the full models. Only that evidence can
connect the identified early arithmetic differences to later discrete code
changes and the final log-probability failure. No further numerical correction
or tolerance change follows from the present bounded probe.

These prototype artefacts live in excluded local state; they need deliberate
packaging if included with a submission. This account records their actual
findings and limits without replacing the original numerical evidence.

## Producer-bound follow-up and integrated RMS correction

The separate `alias-v2/round-001` capture retains completed producer outputs
before consumers can overwrite them. Its independent oracle report is
`state/diagnostics/bitnet-layer-probe/alias-v2/round-001-oracle-attempt-001/report.json`,
SHA-256 `4bfce9ccaf27a1f47613b142068ad8b80b40d29a1d38f605a2114cda36e01d2a`.
The oracle verifies 38 producer/constant edges, original norm weights and
quantisation scales, and 58 overlapping captured outputs. No post-operation
fallback supplies any of the eight RMS inputs. One actual weighted
quantisation-code disagreement occurs at layer one's feed-forward norm.
This resolves the measured operand aliases; it does not identify the first
natural full-model divergence.

The integrated [Rust RMS operation](../../src/bitnet/native_rms.rs) follows the
observed native reduction: F32 products, sequential F64 accumulation, then
native F32 mean, reciprocal and product. It implements the complete first-order
vector-Jacobian product, including affine-weight gradients and non-contiguous
inputs. The implementation explicitly supports CPU/F32, matching the BitNet
training protocol; unsupported execution requests fail clearly.

`state/diagnostics/bitnet-rms-prototype/integrated-cuts-attempt-001/report.json`
records the actual production operation on all eight producer-bound inputs.
All eight RMS outputs and all eight affine-weighted outputs match bit for bit:
**817,152 F32 values**, with no full model allocated. Separate finite-difference,
structural and gradient tests passed. The comparison binds the production
source, diagnostic executable and original capture/oracle hashes. These are
bounded arithmetic and derivative checks, rather than a coding evaluation.

The four normalisation sites in each block and the final model norm now use
this operation, and the parity implementation fingerprint includes its exact source. Native
integer-dot/postscale versus floating projection differences remain. A new
immutable executable has now run the original full-vocabulary protocol again,
at the unchanged **0.10** tolerance. All earlier failures remain intact.

The actual [new receipt](../../reports/comparative/bitnet/parity-native-rms-001.json)
completed all three prompts, retained full Rust vocabulary log probabilities
and the original native top-16 observations, and independently reconstructed
the fixed criterion. Its maximum absolute errors are **0.4428310394**,
**0.4030475616** and **0.1987771988**. All three top predictions agree, but all
three prompts still fail tolerance. The command correctly exited nonzero.
This is an observed numerical shortfall; agreeing top tokens cannot qualify a
validated ranking. It does not establish coding improvement or a trained
checkpoint comparison.

The corrected RMS cuts therefore do not resolve the full-model mismatch.
The next investigation remains producer-bound projection arithmetic, including
floating reductions and the native integer-dot/postscale path. Native GEMV may
use wrapping intermediate lanes where GEMM uses wider accumulation; a single
unchecked scalar-integer substitute would not be a justified correction.
Preserve this receipt, its source snapshots and both backend observations
before introducing any separate kernel/protocol version.
