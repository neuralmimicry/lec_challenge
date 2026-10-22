# Design decisions and departures

## 22 October 2026 — bounded correctness and diagnostic repair

New local study plans and worker specifications use schema 2. Their executable
selection checks both original teacher conditioning and student conditioning
before a numerical reservation; the same bounded validator governs study
configuration and optimisation. The existing shared sequence cap is retained,
including EOS; collection and evaluation limits do not expand it. No old freeze,
trace, image identity, score or executable is rewritten. Schema-1 readers retain
historical preference-content and sequence selection semantics.

The native `study-diagnose-training` command inspects the actual selected update
order and optionally measures base versus reloaded-adapter assistant NLL and
paired sandbox generation on a bounded subset of those training examples.
This is training-set memorisation evidence under a separate runtime protocol.
Tiny differentiable Qwen fixtures and independent-process logit reloads are
explicit numerical QA, not assessment outputs.

New feature experiments record declared train-only generation-health probes,
normalised objective weights and per-tensor update norms. Optional early stopping
is frozen before training; failure and interruption cannot silently bypass it.
Feature-only objectives remain available without automatic SFT additions.
The actual collapse mechanism remains unproved.

New evaluation runtime/thread allocations are independent of optimisation and
teacher capture. Serial phase reservations use measured free VRAM. Existing
evaluation-only recovery and strict matched-baseline comparison remain required
when a runtime changes. Selection schema 3 refuses zero-score or incomplete-seed
groups as automatic quality choices; earlier selection schemas remain readable.

Cargo acquisition is explicit and separate from application offline policy.
Toolkit discovery respects locked cudarc 0.16.6 support and the driver ceiling,
validates the host compiler and freezes local overlay provenance in the build
receipt. No system headers, drivers or toolchains are edited. Native asset/store
diagnostics share exact-archive verification with restoration; quarantine is an
explicit rename. The frozen V3 controller and grading checks remain unchanged.
See `docs/repair-20261022.md` for observed checks and remaining limitations.

## 16 October 2026 — repeatable original-assessment log audit

The user requested a check of the newest logs against the original assessment.
The new Rust `assessment-audit` command inspects the corpus, strict provider /
dataset / checkpoint / gate evidence and historical drill/cost/delivery reports,
retaining hashes and separating independent verification from observations.
It cannot certify missing external evidence or invent an assessment score.
A bounded parent stage-report reader preserves a verified child's original
failure when no downstream report exists. This is a logging correction, not a
new successful PID-control drill or permission to repeat a failed model output.
Neither change alters the frozen design, source split, gate thresholds or
historical experiments.

## 16 October 2026 — additional native API connections

The user requested OpenAI Codex, GCP Gemini and Anthropic Claude connectivity,
missing-credential/failure handling and explicit token budgets. Native Rust
Responses, generateContent (Developer API and Vertex) and Messages adapters now
share the durable fleet transport/ledger. A separate general-inference command
does not execute outputs or clear output-training rights. The original
assessment's three independent rights-cleared acquisition businesses and
promotion requirements remain unchanged. Optional cumulative token limits are
additive to the original monetary reservations; legacy unknown token liabilities
fail closed when a cap is introduced. Offline/cache preservation is now enforced
before automatic pruning, including a fresh connectivity check per deletion.

The user's subsequent terminal report exposed a missing local configuration
step and ambiguous credential-only checks. Rust `provider-init` now publishes
an explicitly chosen, priced, bounded inference profile without overwriting an
existing file; readiness checks show disabled routes, absent prices/credentials
and zero allowances. This setup does not change the frozen assessment roster,
training rights or evaluation rules. The README uses an explicitly named
Responses coding model and labels conservative configured-rate costs; provider
aliases and account availability remain limitations. No new language exception
or automatic model migration is introduced.

## 15 October 2026 — user-directed Rust implementation

The assessment email allows any stack. The subsequent user instruction explicitly requires Rust throughout unless absolutely necessary and justified. This takes precedence over the prebuild pack's Python recommendation. The supervisor, provider clients, ledger, task language, trusted grader, curator, tensor training, statistics and HTTP serving are therefore Rust. Shell is limited to launching the compiled command and operating-system/container build instructions; TOML, JSON, Markdown and OCI recipes are data/documentation formats.

Rust coding tasks replace the proposed Python functions. Qwen2.5-Coder remains the proposed base, with native Candle differentiable LoRA and CPU support in place of PyTorch/PEFT. These changes increase implementation and verification effort, but avoid a hidden Python training dependency. Model revisions, changed weights and real gradients must still be verified.

The original dated design and PDF remain in assessment-prebuild. The assessment design is frozen at the current real time, without backdating. Existing repository history is preserved.

## Evidence constraints at initial preflight

No DeepSeek, Together or Mistral credentials were present among environment-variable names, and no paid-provider authority or account-specific training permission was supplied. Live collection and actual provider-trained assessment checkpoints remain blocked until these inputs exist. Local engineering and tests proceed; no synthetic recordings replace provider evidence.

The Docker command currently reaches a rootless Podman service; Docker compatibility output reports several resource controls unsupported, while native cgroup information needs direct enforcement tests. Merely finding the Docker executable does not pass sandbox preflight. The NVIDIA GPU is an 8 GiB RTX 2080, rather than the hypothetical 12 GiB device. CPU support is required and CUDA is not assumed.

## Native tooling and final verification

The video evidence selection, captions, timing and manifest are produced by Rust. Installed ffmpeg/ffprobe encode and inspect the standard H.264 MP4 container: implementing a video codec would not improve the assessment's Rust application or its evidence. Native tar/gzip/ZIP utilities encode delivery archives. The minimal launcher, OCI recipes and CI commands orchestrate the Rust executable; none replaces training, grading or statistics with another language.

Direct native Podman probes subsequently proved CPU/memory/swap/PID limits despite the Docker compatibility flags. A stricter timing probe separated active workload termination from runtime cleanup and led to an inspected per-container cgroup-v2 kill path. Actual kernel acknowledgement was tens of milliseconds; cleanup took several seconds and is reported separately. The trusted ptrace supervisor runs inside the container outside the candidate process, while expected answers and case selection stay on the host. Its structured audit is verified by the host rather than accepted from generated code.

Corpus-domain review before any teacher collection corrected an out-of-contract private money-format boundary and clarified shared integer/size constraints. These are pre-collection correctness changes, not post-hoc tuning of teacher or student outcomes. The statistical simulation also has an invariant checking that its source sharing achieves the stated pairwise correlation, and an optimised decision path checked against the complete public gate.

## 15 October 2026 — authenticated local comparative teachers

The user subsequently authorised Hugging Face model downloads, required that supplied tokens remain outside the codebase, and requested GPU hosting of teacher models on this laptop. New Rust modules provide authenticated pinned downloads and source-restricted comparative evaluation; the native engine adds optional CUDA float16 and compatible Llama-family inference. The assessment's original CPU float32 trainer and strict base provenance remain available. These local weights do not replace three independent real provider businesses, and exploratory development results cannot authorise the primary held-out gate.

The native CUDA dependency requires NVIDIA's compiler and original GPU kernels. A local glibc/CUDA header compatibility overlay is generated by Rust under excluded application state and compile-tested with a minimal emitted CUDA include; no Python or alternative application implementation is used. NVIDIA headers and installed system files are not modified or redistributed. The short GPU launcher selects the verified native toolchain. Detailed compatibility evidence and reproduction belong in docs/gpu.md and reports/comparative/.

## 15 October 2026 — user-directed adaptive method and student comparison

The user subsequently requested adaptive teacher replacement/pruning, portable resource discovery, offline reuse, resumable interrupted runs, composable distillation families and a genuinely trained BitNet student where applicable. These are an exploratory extension to the frozen assessment design. They use training/development sources and retain the sealed assessment set. The original design's preference for a small SFT-only build remains documented; the extension does not retrospectively change that freeze or satisfy missing independent provider evidence.

Rust implements probability, feature, relation, preference and sequence objectives, native teacher traces, experiment scheduling and BitNet adaptation. The optional pinned `microsoft/BitNet` C++ inference engine is an interoperability exception required to measure the user's specifically requested `bitnet.cpp` track. Reimplementing that engine in another language would not demonstrate compatibility with it. Rust acquires/verifies assets, interprets packed tensors, performs differentiable ternary adaptation, exports the native inference artefact and supervises the engine. No Python training or conversion pipeline is introduced. Exact engine changes and runtime parity evidence belong in `docs/bitnet.md`; a successful build alone is insufficient.

Interrupted work reuses its original executable and validated durable data by default. Source, model, data or training-policy changes require an explicitly new protocol; automatic retry cannot relabel an earlier run. Exploratory best-tested selection, operator QA and changed tensors remain distinct from a statistically justified promotion.

Publisher model cards, licences and reviewed native patch files are retained
byte-for-byte as provenance. Examples in an upstream model card are external
evidence and are not executed by this Rust implementation. Whitespace checks
exclude these exact source copies because normalising them would invalidate
their recorded hashes or change a patch's context; authored project sources
retain the ordinary checks.

## 16 October 2026 — explicit sandbox timer origins

Original controller-V2 observations exposed a real lifecycle defect: an attached
runtime-client deadline mixed startup, execution and removal, sometimes losing
otherwise completed audit output. Verification 007 also failed its strict
acknowledged-kill assertion. Keeping a native 20-second timer from container
start while starting another 20-second timer after setup cannot guarantee the
declared candidate allowance. The failures and original policies remain intact.

Policy/controller V3 therefore adds a trusted Rust PID 1 watchdog armed at EOF
before compilation, cached exact cgroup kill/event descriptors, independently
persisted bounded output and separate lifecycle budgets. Its outer native
startup/crash ceiling becomes 80 seconds; candidate execution remains capped
at 20 seconds, with independent namespace termination if the host disappears.
These are distinct timers, not a retrospective relaxation of previous results.
Actual V3 enforcement requires fresh container and crash QA. The original
generation artefacts are regradable under a separately frozen new policy,
without replacing model outputs or changing their original inference identity.
