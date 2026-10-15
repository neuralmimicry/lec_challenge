# Native Rust GGUF comparative inference

The GGUF backend lets a larger, publisher-verified coding teacher retain quantised
matrix weights on the selected CPU or CUDA device. It reuses the existing Rust
transformer, attention, rotary-position and generation code. It does not train an
adapter or replace the pinned assessment base, provider recordings or promotion
evidence.

`ModelRuntime::load_comparative_with_options` detects a verified
`gguf-assets.json`. The explicit alternative is
`ModelRuntime::load_quantised_with_context(base, options, context_tokens)`.
The default GGUF cap is 1,024 total tokens; an explicit cap may be 1–4,096 and must
fit both the original source and quantised publisher's declared context. Prompt
tokens plus the requested output allowance must fit that cap before generation
starts.

```rust
use lec_distil::model::{ExecutionDevice, InferenceDtype, ModelRuntime, RuntimeOptions};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let options = RuntimeOptions {
        device: ExecutionDevice::Cuda { ordinal: 0 },
        dtype: InferenceDtype::F32,
    };
    let runtime = ModelRuntime::load_quantised_with_context(
        Path::new("offline_bundle/comparative/qwen-coder-7b-q4_k_m"),
        options,
        1024,
    )?;
    println!("{}", runtime.execution_metadata());
    Ok(())
}
```

The example describes the public API; it requires complete local assets and a
CUDA-enabled binary. The executable QA command after those prerequisites are
available is:

```sh
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  target/cuda/release/examples/gpu_smoke \
  offline_bundle/comparative/qwen-coder-7b-q4_k_m --gguf
```

Use the machine-specific build preparation in [gpu.md](gpu.md) for this
workstation. A CPU request uses `RuntimeOptions::default()`. An explicit CUDA
request never silently falls back to CPU, and GGUF rejects float16 activations
because its native matrix kernels require float32 inputs. Compressed matrix
weights, the expanded embedding, activations and KV cache are distinct:

| Allocation | CPU | CUDA |
| --- | --- | --- |
| Transformer/output matrices | Native quantised `QTensor` | Native quantised `QTensor` |
| Embedding | Float32 | Float16 |
| Logical activations, attention scores and KV cache | Float32 | Float32 |
| Model offload | None | None |

The backend supports the implemented Qwen2 and common Llama contracts, with
validated dimensions, tensor shapes, RMS epsilon, rotary base/scaling and SiLU
semantics. GGUF Llama Q/K rows use interleaved rotary positions; Qwen2 retains
split-half positions. Original source tokeniser, chat template, BOS and EOS
remain independently pinned. Every original vocabulary string/ID must match its
GGUF entry. Unsupported architectures or incompatible converter semantics fail
explicitly.

The loader constructs `QMatMul::QTensor` directly for compressed matrices, so
`CANDLE_DEQUANTIZE_ALL` cannot silently expand the model. Publisher-provided
float32/float16 auxiliary matrices are loaded with their actual representation;
the metadata reports the complete GGUF tensor-type counts. Runtime health also
reports original and quantised publisher revisions, GGUF SHA-256, actual device,
logical precision, context cap and the absence of hybrid CPU offload.

`estimate_quantised_memory` estimates the complete file, expanded embedding,
float32 KV cache, quadratic attention scores, activation workspace and a runtime
reserve. This is conservative planning arithmetic rather than observed peak
allocation. Driver free-memory discovery and actual loading remain necessary.
The native CUDA Driver API probe works in a CPU-only binary on Linux amd64 and
arm64, restores the calling thread context and releases its temporary primary
context. It needs the installed driver library, rather than `nvcc` or
`nvidia-smi`.

Comparative models acquire a nonblocking shared lease on the root bundle's
`.hub-download.lock` before validation and loading. The lease belongs to the
private model object, so moving `runtime.model` retains protection. Model tensor
fields drop before the lease synchronises its retained device and closes the
descriptor. Downloader and pruner exclusive leases consequently refuse active
inference assets; process exit or a failed load releases the descriptor. The
assessment base remains separately protected by the managed-pruning policy.

The focused tests exercise a genuine synthetic Q4 GGUF round-trip, full versus
cached transformer logits, metadata/vocabulary tampering, context and memory
limits, shared/exclusive lease contention and failed-load release. Explicit
ignored hardware tests perform native CUDA Q4 computation and compare the
resulting logits against CPU, and verify CUDA Driver context restoration without
the Cargo CUDA feature. Synthetic implementation QA is labelled separately from
real teacher, checkpoint and learning evidence. The GPU smoke uses an ordinary
code-generation prompt, parses completed Rust outputs and records the process's
driver-visible GPU allocation; it never executes generated code on the host or
opens held-out tasks.

The [actual official Qwen 7B Q4 smoke](../reports/comparative/gpu-smoke/7f17a90c-6770-40fa-8781-dc9b26a3eea1.json)
completed on this workstation's RTX 2080. The model retained Q4K/Q6K matrices on
CUDA zero, with float32 activations/KV and a float16 embedding. The observed Rust
process allocated 5,340 MiB. Both generations returned the same complete
29-token Rust function: 967 ms first generation and 930 ms warm. Loading took
43.0 seconds including a complete publisher-checksum pass over the
4,683,073,536-byte file. The conservative 1,024-token memory plan was
6,538,528,768 bytes. These observations used an explicit four-thread CPU limit;
the launcher now selects available resources. They establish functioning native
quantised GPU inference and are not controlled throughput or quality rankings.
