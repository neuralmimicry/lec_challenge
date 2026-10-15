---
name: rust-distillation-evidence
description: Implement or verify native Rust Qwen LoRA distillation with provider-run provenance, assistant-only gradients and source-cluster promotion evidence in the LEC assessment repository.
---

# Rust distillation evidence

Use this capability when changing this repository's model adaptation or assessing whether its artifacts establish learning. Read the repository AGENTS.md and docs/acceptance.md first; provider collection and spending remain subject to the current user's authorisation.

For Qwen adaptation, use the differentiable paths in src/model.rs: RoPE, normalisation and softmax must retain the graph. An inference kernel can generate correct text yet have no backward implementation. Require a finite loss, nonzero adapter gradients, actual optimiser steps, changed adapter tensor hash, frozen base hash and fresh-process reload. Assistant loss positions come from the original chat-tokeniser offsets, excluding prompt and boundary-straddling tokens. Reject overlength examples rather than silently removing all learnable targets.

Before constructing an optimiser, validate the dataset against its manifest, source lineage, held-out specification index, passing sandbox reports and rights-cleared real attempt records. A reference solution, a local random tensor test and a deterministic provider transport fixture are useful QA evidence but cannot become provider-trained assessment samples.

For promotion, use the public gate command and src/evaluate.rs artifact verifier. An aggregate accuracy JSON, an adapter filename or a synthetic SHIP fixture cannot authorise serving. Require every expected source/task/seed observation; failures remain in the denominator. Sources are the independent units, with task/seed averages inside each source. Report both interval bounds, the practical margin and sensitivity to dependence. Do not tune on held-out outcomes or relabel a seed-control result.

Run `cargo test --locked --offline` after changes and the targeted real base smoke command when model code changes. If credentials, rights, container enforcement or real weights are absent, complete independent tests and emit an explicit failing/incomplete evidence report. Follow docs/reproduction.md for the current command contracts. Do not replace missing observations with plausible values.

For local comparative teachers, use the commit-pinned Rust Hub downloader and preserve its publisher-verified asset/provenance manifests. Read credentials only from ephemeral stdin/environment; never place them in arguments, configuration, reports or caches. Asset retrieval is separate from output-training eligibility. In particular, DeepSeek open-weight output distillation creates derivative-licence obligations; the API terms cannot replace the weight licence.

For GPU work on this laptop, read the repository's docs/gpu.md and use its verified native toolchain. A successful CUDA build is insufficient: require actual CUDA/F16 runtime metadata, complete generated code and a matching nvidia-smi process allocation. A newer compiler can emit PTX unsupported by the installed driver. Compatibility headers belong in excluded local state; never modify the system toolkit or redistribute copied NVIDIA headers. Keep a CPU float32 path and reject silent CPU fallback.

Exploratory comparisons use only train/development sources, with the plan frozen before generation and raw output persisted before grading. Locally hosted weights do not count as three independent provider businesses, and their descriptive scores cannot authorise promotion. A trained adapter served with device/precision different from its frozen evaluation remains evaluation-only until matching quality evidence exists.
