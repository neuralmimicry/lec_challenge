# Additional native teacher architectures

The first replacement catalogue exhausted its six candidates without a
qualified teacher. The separate follow-up configuration is
[teacher-search-qwen3.toml](../config/teacher-search-qwen3.toml). It preserves the
original search and its historical executable, retains the measured Qwen Coder
7B fallback for current studies, and declares two new trials: the existing
DeepSeek 6.7B asset with verified legacy metadata support, and Qwen3 4B Instruct
2507. These are pending trials until their actual reports exist. Metadata and
downloaded weights are not quality results.

Qwen3 uses an explicit attention head width. For this pinned model the residual
width is 2,560, query width is 32 × 128 = 4,096, and key/value width is
8 × 128 = 1,024. Reusing residual-width query/output matrices would implement a
different network. The Rust loader checks those exact shapes, the original
128-value per-head query/key RMS weights, their position before split-half
RoPE, the original tokeniser/template and both EOS identifiers. It rejects
missing norms, projection biases, mismatched GGUF key/value widths and
unsupported rotary semantics. The plain-user chat template injects no default
system instruction or thinking preamble.

Five independent numerical tests compare a tiny asymmetric network against a
separate scalar transformer, check cached decoding, verify actual LoRA
derivatives through head normalisation by finite differences, and exercise
invalid architecture/template cases. Additional GGUF tests compare exact
F32 projection layouts while retaining a genuinely quantised embedding and an
equal F32 output matrix. The existing Q4 activation/cache test remains separate:
native Q4 matrix multiplication quantises activations, so a dequantised dense
matmul is not its exact numerical oracle. The two earlier strict-fixture
failures remain in the original logs; the corrected focused run passed all
25 model tests. This technical QA does not establish publisher-wide numerical
parity or coding quality.

The failed DeepSeek recheck independently found its actual original GGUF key
`llama.rope.scale_linear = 4.0`. This older spelling represents the same linear
factor required by its pinned original model. The loader now accepts that key,
requires modern and legacy factors to agree when both exist, and still checks
the independently verified source configuration. Conflicts, non-finite factors
and unsupported scaling types fail closed. Original failed observations remain
unchanged; a new executable and protocol are required for another trial.

Qwen3 acquisition pins original source commit
`cdbee75f17c01a7cc42f958dc650907174af0554` and quantisation commit
`ae44f08e1392f39c0e474af10c3ff8355c8b6688`. The selected Q4 weight has expected
length 2,497,280,736 bytes and publisher LFS SHA-256
`2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e`.
The native Rust downloader checks actual bytes and retains original licence,
card, tokeniser and manifests. Public API metadata snapshots are under
`docs/comparative-models/provenance/qwen3-4b-instruct-2507/`; progress and the
eventual acquisition receipt belong to `reports/comparative/qwen3-acquisition-001/`.
No token is stored. Live download progress, not this prepared documentation,
determines whether the model is available offline.
