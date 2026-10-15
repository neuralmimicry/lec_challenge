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
