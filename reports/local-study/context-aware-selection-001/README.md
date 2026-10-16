# Context-aware device selection QA

The original-assessment code audit found that expanded content/correction
contexts reached the GGUF loader but automatic selection still estimated
1,024 tokens. `src/main.rs` now passes the requested context into both CPU and
GPU estimates. Explicit CUDA remains fail-closed, no requested limit is cropped,
and saved runtimes remain frozen.

The regression uses synthetic amd64/ARM64 capacity profiles and a real retained
Qwen configuration. A capacity between the 1,024- and 1,536-token estimates
selects CUDA for the former and CPU RAM for the latter; an explicit oversized
CUDA request fails. Invalid/native/source context limits fail before selection.
This is metadata dispatch QA, not an ARM64 execution or GPU allocation result.

All commands completed with exit zero:

| Command | Observed result |
| --- | --- |
| `cargo test --locked --offline --bin lec-distil` | 16 passed; zero failed. |
| `cargo test --locked --offline` | 279 library + 16 CLI tests passed; five explicitly ignored asset/container tests. |
| `cargo clippy --locked --offline --all-targets -- -D warnings` | Passed. |
| `cargo fmt --all -- --check` | Passed. |
| `git diff --check` | Passed. |

The actual container/runtime implementation was unchanged by this fix; its
separate [verification 009](../verification-009/verification.json) passed all
three actual container tests. That earlier coordinated report remains bound to
its own unchanged source snapshot and retains the optional dependency-notice
gap. The current experiments use their original pinned executables, not this
later CLI source.

| Input / captured log | SHA-256 |
| --- | --- |
| `src/main.rs` | `5409266b3ae6eaba73e807326a6adfe56b99ccf7fc98cf96d36b93d08fc9efa6` |
| [CLI tests](cli-tests.log) | `cc8cc8866c11179e3bcf0b100d2889321d9d056e4118e8cceafdc3ca78b7d222` |
| [Full tests](full-tests.log) | `a5de84fe1b68b43c74799903706b8eaccd709b449533c59c34bde210d981b65e` |
| [Strict Clippy](clippy.log) | `622f7b95cad241b7b208cfe2f2aa2e8d3e5405a1d22dacdc8d50b3b5cd5e7305` |

A filename-only scan for `hf_` followed by at least 30 alphanumeric characters
found no matches in Rust sources, Markdown, TOML or JSON under `src`, `examples`,
`docs`, `config`, `.agents`, `agents`, `offline_bundle`, README and STATUS.
It is a bounded pattern check, not proof that every possible secret format has
been excluded. No credential values were printed or retained by that check.
