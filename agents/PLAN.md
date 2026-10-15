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
