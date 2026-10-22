# Specialist agent execution plan

## Bounded training repair — 22 October 2026

The single integration owner owns this repair on branch
`repair/training-20261022`, starting at
`d78d127fc3c9da546fe4b9bee3ef4e3cf0e9076d`. No additional agents are used.
Scope: distillation eligibility/numerical diagnostics, study evaluation failure
reporting/runtime contracts, launch compatibility and non-destructive asset/
sandbox diagnosis, regression tests and documentation. Existing report edits
and untracked exploration evidence are user work and remain untouched. Source
is edited on the RTX 2080 laptop; remote evidence is read from
`pbisaacs@192.168.1.66:/home/pbisaacs/Developer/lec_challenge`.
Historical engines, freezes, caches and failed observations are immutable.
The same owner maintains `docs/LEC_Challenge_Installation_Guide.md` against
the implemented launcher/CLI contracts and measured repair evidence.
The user has authorised committing this repair and merging it into local `main`.
Keep unrelated exploration report edits outside the commit; do not push.

## Provider setup repair — 16 October 2026

The integration owner owns the Rust `provider-init` command, local readiness
checks, path-aware configuration errors and matching README/provider guidance.
Initialisation must create a valid, priced, bounded inference-only configuration
without credentials, network access, spending or overwriting an existing file.
Tests use labelled fixtures; real account access remains a separate observation.
Existing runs, ledgers, unrelated edits and offline cache/model assets are retained.

## Latest-log assessment audit — 16 October 2026

The integration owner owns a read-only assessment evidence audit, its CLI and
documentation, and the exploration failure-reporting correction in `explore`.
The audit must compare original required evidence with the latest observed logs,
preserve child failure causes and distinguish local research from assessment
completion. Existing pinned runs, failed observations, held-out boundaries and
cache/model assets remain intact. No paid calls or new sealed evaluations are
authorised by this audit.

## Provider connectivity follow-up — 16 October 2026

The integration owner is extending `fleet`, `ledger`, provider CLI commands,
configuration and documentation for native OpenAI Responses/Codex, Gemini
(Developer API and GCP Vertex AI) and Anthropic Messages. This work includes
credential-free checks, bounded retries/failover, durable money/token limits and
offline asset preservation. No live spending or output-training clearance is
implied. Existing unrelated working-tree changes and historical evidence remain
intact; validation uses labelled fixtures without paid API calls.

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

## Adaptive studies and ternary students — 15 October 2026

The user extended the work to autonomous teacher replacement/pruning, offline cache operation, portable resource discovery, composable distillation methods and genuine BitNet student training. The same agents continue executing, with these explicit ownership boundaries:

| Agent | Current modules and work | Required evidence |
| --- | --- | --- |
| Fleet | `teacher_search`, `experiments`, `evaluate` resumption, `study_evaluate` | Immutable finite catalogue, capability/seed/content coverage, CPU/RAM/VRAM reservations, durable worker recovery, independently verified development comparisons |
| Sandbox and corpus | `hub`, `teacher_evidence`, `bitnet` backend/export/native inference integration | Publisher checksums/licences, cache-only operation, measured safe pruning, actual train-only content/feedback, exact ternary codec and engine parity |
| Training | `distillation`, differentiable model capture and optimiser surfaces | Real composed objectives, aligned teacher targets, assistant-only windows, durable Adam updates, conventional and ternary export/reload, changed effective trits |
| Integration | `platform`, `engine`, launchers, `study`, CLI, leaderboard and delivery documentation | Resource detection, immutable executable pinning, resumable command contracts, measured cross-method selection, secret scan and end-to-end QA |

The repository skill `.agents/skills/rust-distillation-study/SKILL.md` captures the non-obvious execution contracts. Training examples, native target tensors, a model checkpoint, development improvement and assessment promotion are separate claims and each needs its own observed evidence. Failed or unsupported methods remain visible. Full powerset enumeration does not imply that every arm has run.

## Assessment crosscheck — 16 October 2026

The user's explicit audit reminder is tracked in `docs/assessment-audit.md`.
Integration checks every original design/build/drill/delivery requirement,
corrects current-source launcher and statistical-reporting gaps, and finishes
immutable-output regrading. Fleet independently reconstructs the completed
replacement search and audits portable execution dependencies without inference.
Sandbox moves content publication off the async executor while preserving
ownership, durable ordering and cancellation safety, with isolated slow-writer
QA. Training investigates producer-bound BitNet projection arithmetic in excluded
diagnostic state only; failed full-model parity and its fixed tolerance remain
unchanged. Stale earlier task drafts are superseded by these bounded assignments.
The external provider, assessment evaluation/control, cost and delivery gaps
remain explicit; these agents cannot create absent credentials or billing
authority, turn a local teacher into an independent API business, or invent a
successful result.
