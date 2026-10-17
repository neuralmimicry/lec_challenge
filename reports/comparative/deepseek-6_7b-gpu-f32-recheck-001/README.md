# Fresh DeepSeek CUDA/F32 development attempt

The application exited **1** after recording all **16/16** planned development
observations. Every observation reports the same loading failure:
`model loading failed: GGUF/source rotary scaling differs`. There were no actual
model generations or coding verdicts. The displayed 0/16 is a retained failure
denominator; it does not measure this model's coding quality.

The immutable request used application `f5d2f2f5…`, CUDA device 0, F32
activations, the original Q4_K_M weights, four assigned CPUs, seed 1729,
temperature 0.2, top-p 0.95 and a 512-token output cap. The application froze
the original sixteen development tasks and three maximum isolated graders.
The actual loader failed before inference, so actual execution metadata is null.
No sealed held-out task was opened. This exploratory attempt cannot authorise
promotion or meet the independent API-provider requirement.

The standalone Rust verifier checked the request hash, task content and order,
runtime, application identity, decoding settings, raw-generation hashes,
per-task verdicts, complete report and execution events. Its bounded metadata
inspection found the older GGUF field `llama.rope.scale_linear = 4.0`.
The pinned source configuration specifies linear scaling with factor 4.0 too.
The application reads `llama.rope.scaling.factor`, defaulting its absence to
1.0. The rejection therefore identifies an unsupported legacy metadata key.
The original model and source metadata have been preserved without correction.

Artefacts:

- [Frozen request](requested-protocol.json), SHA-256
  `20a7ae3bf104737663c7771742bbdfc7942b54fcc785a63e9386248e7a979a4d`.
- [Actual plan](evaluation/plan.json), SHA-256
  `c0eb6b9a6ac62bd4f1c35803db2f748b8cdf1ac7607088924c2c80518e92289b`.
- [Full denominator report](evaluation/report.json) and
  [execution events](evaluation/execution.json).
- [Independent verification](independent-verification.json), SHA-256
  `8456fc8ad4dc8f1b849303c65ecccfa38d13d9db46f4da9211ede331463c4d61`.
- Rust verifier and four passing parser tests:
  `state/diagnostics/verify-deepseek-gpu-recheck.rs` and its adjacent build/test
  logs. The verifier reads metadata only; it does not allocate a model, decode
  tensor payloads or recompute the complete GGUF checksum. The application
  verified the complete original GGUF before its loader rejected the metadata.

After process exit, a separate `nvidia-smi` inspection showed **131 MiB / 8192
MiB** in use and GPU ownership was released. No retry has been performed under
this request. The historical interrupted CPU attempt remains separately
classified as incomplete; this new loader rejection does not rewrite it.
