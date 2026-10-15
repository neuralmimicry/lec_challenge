# Native model QA — measured 15 October 2026

These checks use the actual pinned public Qwen2.5-Coder-0.5B-Instruct weights on CPU. They are implementation verification, **not** the assessment's provider-output training round, a submitted trained checkpoint or evidence of improved coding quality. The original recordings and rights-cleared training dataset are still required.

| Check | Measured result |
| --- | --- |
| Public base weights | 988,097,824 bytes; SHA-256 `f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44` |
| Native base inference | Complete 29-token Rust `add` function; 58 input tokens; 7.244 seconds generation |
| Base loading/checksums | 12.66 seconds in this run |
| Full-base synthetic gradient QA | 64 input / 32 supervised token tensors; 540,672 trainable adapter parameters, 96 variables |
| Actual forward/backward/Adam update | Loss 9.763630867; gradient norm 7.316696563; parameter-change L2 0.088307612 |
| Frozen base / adapter serialisation | Base sentinel unchanged; safetensors values round-tripped exactly |
| Synthetic-step timing | 4.172731 seconds after asset checksum validation; no assessment checkpoint saved |
| Focused unit verification | Four model, three optimiser/control and two serving invariants passed |
| Original-tokenizer mask verification | Explicit asset-dependent test passed: only assistant code plus EOS become labels |

Base inference and the gradient smoke ran concurrently, so these timings are observed workload timings rather than isolated throughput benchmarks. The small random model test exercises the complete differentiable architecture; it does not substitute for the real-base smoke or actual assessment training.

Commands:

```sh
cargo test --lib model::tests
cargo test --lib train::tests
cargo test --lib serve::tests
cargo test --lib train::tests::original_tokenizer_masks_prompt_and_preserves_assistant_eos -- --ignored
cargo run --release --example model_smoke -- offline_bundle/base
cargo run --release --example cpu_gradient_smoke -- offline_bundle/base 64
```

Raw measured outputs: `base-inference-smoke.json` and `native-gradient-smoke.json` in this directory. The model-generated function was inspected as output; it has not been executed outside the prescribed candidate sandbox.
