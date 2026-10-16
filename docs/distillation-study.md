# Finite distillation study

This supplemental local-model study enumerates configured experiments and runs
real Rust adaptation workers within declared compute budgets. It does not repair
the assessment's missing three independent real API providers. A quick arm is
an operator smoke experiment, never evidence that a student learned or that the
best configuration has been found. Actual results live in the generated plan,
per-arm training/evaluation artefacts and execution report.

The [resumable exploration command](explore.md) connects teacher selection,
train-only evidence/content acquisition and quick/full study stages under one
explicit invocation budget, preserving the original engines and cached work.

The operator graph separates questions that are often conflated:

| Axis | Configured operators | Required evidence |
| --- | --- | --- |
| Training signal | Sequence SFT; full-vocabulary temperature KL; hidden features; attention/token relations | Verified teacher run or native tensor capture, aligned teacher/student positions and assistant-only target masks |
| Content | Process/plan; tool trajectory; contextual supervision; preferred/rejected answers with DPO | Actual recorded material and trusted outcomes; a solution is not relabelled as a reasoning trace |
| Data policy | Offline; on-policy | Fixed recorded examples, or genuine current-checkpoint student rollouts with linked teacher feedback |
| Organisation | Single/multiple teachers; ordered teacher curricula; self-distillation | Exact teacher identities/order, real intermediate adapters, or a frozen student-generated teacher source |
| Student | Conventional Qwen 0.5B LoRA; conditional ternary BitNet | Actual trainable weights, export and successful reload for that architecture |

All nonempty subsets of configured signal operators and every subset of content
operators are crossed with configured data policies, organiser flags, seeds and
student architectures. Teacher subsets are unordered for a simultaneous
ensemble; progressive stages enumerate every teacher order. Quantisations of
one original model cannot manufacture an independent multi-teacher ensemble.
Every configured arm receives a stable identity and either an applicable status
or explicit unmet capability reasons. The finite product must fit the configured
catalogue cap; it is never silently truncated.

On-policy and self-distillation applicability is specific to the actual initial
student checkpoint, architecture, base and content rows. Offline process/tool/
context capabilities cannot supply missing checkpoint-bound rollout examples.
Combining preference training with a content stratum requires an actual chosen
rollout sample in that same stratum. Selected teachers must share one verified
initial student identity. An existing adapter also fixes its rank and alpha:
an incompatible quick/full fidelity is labelled unavailable at that fidelity,
without silently expanding or replacing the adapter. Base-weight rollouts have
no inherited adapter rank or alpha constraint.

Hard-label/response/sequence distillation are aliases within sequence SFT;
logit/soft-target distillation belongs to the soft-token signal. Feature matching
and relational/attention transfer are separate signals. Process supervision,
tool-use traces, contextual transfer and preference optimisation concern content
and objectives that can be composed with signals. Offline/on-policy, ensembles,
ordered curricula and self-distillation organise data or stages. Listing an
alias twice would not create an additional implemented family. Reverse KL,
Jensen–Shannon objectives, reinforcement learning, specialist compression and
other algorithms are outside this explicitly configured grid until separately
implemented, tested and preregistered.

The historical `progressive` configuration field means ordered teacher stages
for one fixed student architecture. Each stage continues a real adapter, but
the next teacher remains its original model. This implements a curriculum. A
teacher-assistant architecture bridge would instead train an intermediate
model and use that adapted model to teach a smaller student; that separate
topology has not been implemented or measured. Curriculum coverage cannot be
reported as evidence of that bridge.

The fixed conventional student is the pinned
`Qwen/Qwen2.5-Coder-0.5B-Instruct`. KL requires verified complete vocabulary and
token-ID equivalence; different tokeniser file hashes can still encode the same
mapping. Feature/relation transfer instead uses independently checked exact
assistant UTF-8 byte anchors across tokenisers, with declared projection/head
pooling. Positions are never aligned merely by their token index. The
teacher's original model/revision, rights decisions, run/raw output, grader
audits and all training-source identities remain linked through the verified
teacher-evidence bundle. No development or sealed held-out sample becomes a
training target. Capability claims require actual artefacts, not a model name.

BitNet is conditional. A bitnet.cpp inference engine, downloaded ternary weights
or a quantised conventional checkpoint alone does not provide a train/export/
reload training path. Until that path is genuinely available, ternary arms stay
`not_applicable` with the precise missing capability. The user's explicit
exception permits the native C++ engine when necessary; the study controller,
collector, scheduler, reporting and conventional training remain Rust.

Matched local evaluation freezes the original checkpoint, base, tokeniser,
Rust executable, runtime request and sandbox policy before inference. If the
native C++ backend is selected, its additional frozen artefacts include the
engine manifest, exact GGUF and Rust export receipt. Independent verification
checks the pinned C++ binaries/libraries and reviewed graph patch, the original
base and adapted tensor identities, changed effective trits, exporter identity
and actual loaded engine/GGUF metadata. Tokeniser parity probes must have passed.
These receipts establish execution provenance, not improved model quality.
BitNet rankings additionally require independently reconstructed, checkpoint-bound
[Rust/C++ numerical equivalence](numerical-equivalence.md). Missing, stale or
failed receipts retain descriptive scores with explicit exclusion reasons.
The first load receipt and every subsequent load attempt retain their log links;
incidental ports, logs and CPU affinity do not change the frozen evaluation.

The quick profile prioritises isolated family coverage and then fills remaining
applicable operator gaps. Its three optimiser updates per arm are a **total**
budget, including progressive stages. The full profile permits eight
development-selected arms with 48 total updates each. Both use seeds frozen
before outcomes. The default catalogue includes seeds 1729 and 2718, so later
replication need not choose a favourable seed after viewing results. Selection
and refinement must use matched train/development evaluations and retain all
failures. The separately preregistered sealed gate confirms at most one chosen
winner; it is not an exploration score.

Every requested teacher and non-preference content stratum must receive an
actual update. Thus a three-teacher arm with two such content strata needs at
least six updates and is inapplicable at the three-update quick fidelity. It
remains in the full catalogue with the precise fidelity limitation; no requested
teacher/content axis is silently dropped to fit the screening budget.

The conventional learning rate remains 0.0002. The config preregisters 0.005 for
both ternary fidelities because infinitesimal latent movements around official
ternary centres can leave every effective trit unchanged. An arm still fails if
its effective exported/reloaded weights do not change. Equal update counts do
not imply equal compute; measured timing and resource evidence remain separate.

Quick windows retain real assistant tokens and at least one genuine preceding
context token within 64 tokens. Original byte offsets and removed spans remain
recorded; truncated completions do not acquire invented EOS targets. Teacher
tensor capture uses the exact same declared window before alignment checks.
The full profile rejects overlength examples instead. Windowed operator checks
and complete-context learning experiments must not be conflated.

The Tokio controller runs independent native worker subprocesses. It reserves
CPU threads, RAM and VRAM together before starting each worker, using the
process affinity limit and conservative measured available memory supplied by
the resource planner. Reservations cover the aggregate of active study jobs;
they are not independently applied per job. A GPU ordinal is explicit, and
disjoint cards are not added together to pretend that one model fits. Child
thread pools receive their assigned thread limits. The controller retains
stdout/stderr, numerical training evidence, actual exported tensors, reload
proof, timings, evaluation artefacts and measured costs. Local energy, hardware
depreciation and money remain unknown when they were not measured.

Before every first arm launch, the training-step budget and immutable pinned
worker executable are checked. Failed or missing output is a failed arm. JSON
metadata is not a checkpoint. Successful completion requires real adapter
tensors, finite loss/gradient evidence, a changed adapter and successful reload.
The controller does not issue HTTP; an offline worker uses only verified cached
evidence and model assets, including any locally generated rollouts.

Interruption resumes the existing frozen plan, worker requests and validated
completed outcomes. An in-progress arm restarts its native worker against the
same output directory, where durable optimiser progress determines the
remaining updates. Incidental hardware availability does not generate a new
plan or arm identity. A changed experiment needs an explicit fresh, versioned
output directory. Queued configurations remain `queued` when the budget ends;
a bounded profile completion is never described as exhausting the empirical
catalogue or proving a global optimum.

Both the strict assessment evaluator and the separate local-study evaluator
resume validated atomic generation and grading records by default. A complete
verified report returns without starting a model or container. A generation is
durable before grading starts, and a trusted outcome is durable before aggregate
reporting, so interruption between those operations does not regenerate already
finished output. Changed checkpoint/runtime/backend/corpus identities require an
explicit fresh output. The local-study API exposes train and development only;
it cannot select the sealed held-out corpus or authorise promotion.

Use the same local evaluator for an actual unadapted baseline and its trained
adapter. Runtime, seed, decoding and corpus must match for their descriptive
difference to be meaningful:

```sh
./run.sh study-evaluate --offline --device cpu --dtype f32 \
  --output reports/local-study/baseline-development
./run.sh study-evaluate --offline --device cpu --dtype f32 \
  --checkpoint reports/local-study/native-sft-resume-qa/interrupted \
  --output reports/local-study/sft-development
```

The second example evaluates the genuine three-step operator-screening adapter,
not a complete learning experiment. `--student bitnet` selects the official
cached ternary base and the pinned C++ runtime; adding `--checkpoint` evaluates
its genuinely adapted export. The evaluator independently validates base,
adapter, tokeniser, backend and raw/container evidence. A complete denominator
with generation or infrastructure errors is retained and returns failure.

The actual C7C development [comparison table](../reports/local-study/sft-development-comparison-c7c.md)
and [independently reconstructed JSON](../reports/local-study/sft-development-comparison-c7c.json)
retain baseline **0/16** and SFT adapter **0/16**. The adapter has seven grading
infrastructure errors, although all 16 generations completed. The quality
comparison is invalid and its improvement delta is unavailable. The separate
[retained-evidence diagnosis](../reports/local-study/sft-grading-diagnosis-c7c.json)
finds seven sandbox wall-deadline reports without retained audits: five
missing-audit rejections and two cleanup-inspection timeouts. It preserves
original verdicts and distinguishes later cleanup from missing execution proof.
The retained runtime event files do not cover those failed container names, so
they do not establish a more specific runtime or host-pressure cause.

`examples/study_comparison.rs` wraps `study_evaluate::compare_many` for reusable
read-only reconstruction, accepting `--baseline`, repeated `--candidate` and
`--output`. Optional `--diagnosis-output` and repeated `--runtime-events` write a
separate infrastructure diagnosis from retained files. Invalid comparisons are
written explicitly and return a nonzero exit; no inference or containers run.
The actual standalone build's compiler, original rlibs and pinned executable
are recorded in `state/diagnostics/study-comparison/build/receipt.json`.

## Serving an exploratory student

The local service accepts a separately validated exploratory checkpoint:

```sh
./run.sh serve --local-study --device cpu --dtype f32 \
  --base offline_bundle/base \
  --checkpoint reports/local-study/native-full-sft-seed1729 \
  --config config/serving-teacher.toml
```

The original base, adapter, manifest and complete training evidence are checked
before loading. Health includes the actual checkpoint, device and precision;
startup requires a finite, nonempty numerical probe. Inference uses the bounded
worker and request queue. This mode always reports `production_eligible=false`,
and production routing rejects it even if its tensor identity matches a useful
research result. It does not convert a development comparison into a promotion.

`examples/study_serving_smoke.rs` accepts `--engine`, `--checkpoint`, `--base`
and a fresh `--output`. It checks the real HTTP identity, saves a complete
generation, grades a deliberately selected training example in a fresh
container and proves production routing rejects the endpoint. Its report is
operational QA, not an unbiased coding score. An empty response, HTTP error,
failed execution or incomplete audit returns failure with retained evidence.
