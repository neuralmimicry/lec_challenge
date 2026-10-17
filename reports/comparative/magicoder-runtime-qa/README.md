# Magicoder runtime diagnosis

This is a synthetic runtime investigation. It does not reassess the frozen
development tasks, evaluate held-out sources, establish learning or authorise
promotion. Original search plans, generations, grades and reports remain intact.

The original Magicoder comparison retains 0/16 passes and 16 generation errors,
mostly incomplete, repetitive output. Its original pinned search executable
labelled this `verified_below_threshold`; that historical label is retained in
the evidence. With generation failures, the result is inconclusive about model
quality. Future search executables distinguish
`complete_with_errors_inconclusive`. The errors already prevent qualification
and pruning; the weights were retained.

The original model is `ise-uiuc/Magicoder-S-DS-6.7B`, represented by the
publisher-verified `TheBloke/Magicoder-S-DS-6.7B-GGUF` Q4_K_M asset. Rust source
configuration, original tokeniser and original `@@ Instruction`/`@@ Response`
template were inspected alongside the native GGUF implementation. The Llama
dimensions, RMS epsilon, rotary base 100,000, linear rotary factor four,
BOS 32,013 and EOS 32,014 agree. The native GGUF interleaved q/k rotary operation
matches the row permutation in the C++ reference converter. This inspection
alone cannot prove numerical equivalence.

The independently compiled Rust diagnostic retains its exact source, commands,
environment, prompt, logs and results under each attempt. It clears unrelated
environment variables, assigns four CPU threads and disables GPU use. Model
loads are sequential. Immediately before each load it measures Linux
`MemAvailable` and every accessible cgroup ancestor's remaining memory, then
requires at least 10 GiB after a ten per cent and 512 MiB reserve. Sandbox and
training agents coordinate larger allocations separately.

The C++ reference is the existing pinned llama.cpp build. Its use here is a
diagnostic interoperability exception: Rust owns the runner, input verification,
parsing and reports. No Python is used. The older publisher GGUF calls its
tokeniser `deepseek_coder`; the current C++ runtime requires the equivalent
explicit metadata spellings `tokenizer.ggml.model=gpt2` and
`tokenizer.ggml.pre=deepseek-coder`. The original five pre-tokenisation stages
were compared with the pinned C++ implementation. Automatic BOS insertion is
disabled because the original rendered prompt already contains its BOS token.
The GGUF file is unchanged. A transparent C++ Jinja template passes through the
already rendered original prompt rather than applying a second model template.

- `round-001` failed before loading transformer weights: the unmodified legacy
  tokeniser name is unsupported by the current C++ runtime. No output was
  accepted as a successful diagnostic.
- `round-002` used the explicit tokeniser metadata compatibility overrides. Its
  C++ 64-token greedy invocation completed. The original pinned Rust CPU runtime
  loaded with verified assets, CPU/F32 execution and four Candle threads, but
  the 64-token request exceeded its 120-second endpoint deadline and returned
  HTTP 504. No native completion was available for comparison. Subsequently,
  inspection found that the C++ `--file` option removes a trailing LF. Thus this
  attempt also lacked exact prompt equivalence; its C++ output cannot be used
  to infer a backend mismatch or a model-quality result.
- `tokeniser-parity` uses a temporary Rust FFI probe against the pinned
  `llama.h` ABI. It loads vocabulary metadata only, with no transformer tensors
  or inference context. All 62 original prompt IDs, including BOS, match the
  original Hugging Face tokeniser exactly. Both C++ `add_special` settings under
  the stated explicit BOS override produce those same IDs. This establishes
  tokeniser compatibility for this prompt, not general numerical parity.
- `round-003` failed its prompt-denominator guard before native Rust loading.
  Even `--binary-file` is followed by another trailing-LF removal in the C++
  interactive frontend. Its 61-token result was retained as a failed input
  validation, with no runtime-equivalence claim.
- `round-004` restored that single LF in the transparent template and passed the
  exact 62-token C++ input guard. The 16-token C++ generation completed. The
  original native Rust CPU endpoint again exceeded its 120-second generation
  deadline. Its HTTP 504 response is retained. Scalar CPU timeout does not
  establish model quality or a numerical mismatch.
- `round-005-cuda` reused the retained exact-input C++ reference and explicitly
  loaded the original native engine on CUDA/F32, after GPU coordination and
  actual memory checks. Its 16-token request reached the length limit, but the
  original serving endpoint correctly returned HTTP 422 and withheld the
  incomplete completion. That response is retained. The endpoint route cannot
  expose bounded partial-prefix evidence, so no text comparison was possible.

The new Rust `quantised_prefix_qa` example calls `ModelRuntime::generate`
directly, pins its diagnostic executable, freezes the exact original prompt
and token IDs before inference, records actual finite logits and retains a
length-truncated generation strictly as prefix evidence. Its execution remains
pending. Public serving semantics and original frozen trials are unchanged.
All diagnostic model processes from the attempts above have exited; their CPU,
RAM and GPU allocations were released.
