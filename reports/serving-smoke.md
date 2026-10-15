# Actual local base endpoint QA

The reusable Rust example launched a real immutable Qwen base server on a temporary loopback port, exercised HTTP requests and routing, and stopped the child process. The run finished in 17.304 seconds. This is base-model operational evidence, **not** a trained-candidate canary or proof of learning.

| Observation | Measured result |
| --- | --- |
| `/health` | Healthy; all asset checksums validated; real one-token smoke output `Certainly`; 294 ms inference |
| `/v1/generate` | Complete 29-token Rust `add` function; 58 input tokens; 4,560 ms generation |
| Oversized HTTP body | 413 |
| Excessive output-token request | 400 |
| Twelve concurrent requests / queue capacity one | Ten 503 responses; two 504 deadline responses |
| Initial routing | Immutable real base, generation zero |
| Explicit evaluation routing | Same base tensor alias, generation one, zero production canary traffic |
| Evaluation-route inference | Same complete 29-token function; 4,446 ms generation |
| Atomic rollback | Generation two, original incumbent retained, candidate history retained, zero canary traffic |

All model responses and routes identify tensor SHA-256 `f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44`. Raw observations are in `serving-smoke.json`; server startup output is in `serving-smoke-process.log`.

Reproduce:

```sh
cargo build --release --bin lec-distil --example serving_smoke
target/release/examples/serving_smoke offline_bundle/base
```

No distinct trained checkpoint, authorised five-per-cent canary or measured coding-quality improvement was demonstrated. The evaluation alias intentionally names the same base process. Model cancellation occurs at token boundaries and cannot interrupt a matrix multiplication already executing. Generated code was not executed on the host.
