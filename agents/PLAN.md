# Specialist agent execution plan

Created before implementation. Agents execute tasks in parallel after the design freeze, under one integration owner and shared evidence rules. All code and code comments use UK English.

| Agent | Owned modules | Responsibilities and completion evidence |
| --- | --- | --- |
| Fleet and ledger | src/fleet.rs, src/ledger.rs | Real configurable provider adapters, transactional money/rate limits, exact replay, reservations/UNKNOWN recovery, failover, cost export, invariant tests |
| Sandbox, corpus and curation | src/tasks.rs, src/sandbox.rs, src/curate.rs | Source-separated Rust contracts/oracles, fresh isolated execution, trusted traces, attack probes, traceable training data and contamination rejection |
| Model, training and serving | src/model.rs, src/train.rs, src/serve.rs | Native differentiable Qwen LoRA, asset pinning, assistant-only labels, genuine steps/resume/reload, seeds/controls, bounded real inference and routing |
| Integration, statistics and QA | src/main.rs, src/gate.rs, src/evaluate.rs, src/drills.rs | CLI contracts, freeze, paired source gate and power, complete manifests, end-to-end drills, reproducible offline packaging and documentation |

Each owner reads the complete brief, reports feasibility gaps early, builds missing Rust support rather than substituting a stub, and runs focused tests. A model implementation needs differentiable kernels, not an inference-only graph. A secure grader needs observed kernel enforcement, not candidate self-reports. The integration owner reviews module contracts, runs cross-module tests and retains truthful partial reports where external evidence is missing.

Dependency order: design freeze → corpus/sandbox and ledger/model in parallel → real providers → grading/curation → genuine training → frozen evaluation/gate → serving/drills → disconnected rehearsal/package/video. Work independent of unavailable credentials continues to completion.

## Comparative GPU follow-up — 15 October 2026

The user authorised authenticated Hugging Face downloads without saving credentials and requested local GPU hosting of teacher models. The same specialists are executing this follow-up: fleet owns model/licence provenance and the comparative roster; sandbox owns the authenticated Rust Hub downloader; training owns CUDA execution, compatible model architectures and bounded serving; integration owns comparative development evaluation, CLI wiring and final QA. Credentials travel only through ephemeral process input/environment and are excluded from configuration, reports, caches and source. Existing user-run reports are preserved; new observations live under `reports/comparative/`. Local open-weight teachers are recorded separately from the assessment's three independent provider businesses. The sealed held-out set is not opened for this exploratory comparison.
