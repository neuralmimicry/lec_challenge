# Native comparative distillation

`distillation::run_worker` performs real adapter optimisation using the locally
recorded teacher-evidence protocol. This is separate from the assessment's
provider-trained checkpoint contract. Neither successful numerical updates nor a
local teacher's development score establish learning or authorise promotion.

The worker first reconstructs and verifies the evidence bundle: training-source
identity, original model outputs, trusted passing sandbox reports and audits,
immutable assets and output-training rights. It checks the entire requested
method and content combination before opening an optimiser. Missing process,
tool-trajectory, context, preference, on-policy or self-distillation evidence
produces an explicit failure or an inapplicable catalogue entry. Existing answer
text is never silently relabelled as one of those signals.

The conventional student retains the exact pinned Qwen 0.5B base and assistant
boundary masking. The optional BitNet student uses the original verified packed
weights, original tokeniser/BOS/EOS, ReLU² and additional subnormalisers. Its
low-rank latent updates enter an actual ternary straight-through quantiser during
each forward pass. It supports CPU float32 explicitly. A BitNet arm must change
the effective ternary lattice as well as its latent adapter; otherwise the
worker writes `blocked-no-discrete-update.json`, preserves completed progress
and returns failure. Exporting a float adapter and applying quantisation later
would not satisfy that condition.

| Objective | Implemented training signal |
| --- | --- |
| Sequence SFT | Mean negative log probability on actual assistant tokens; prompt and boundary-straddling targets are excluded |
| Temperature KL | Teacher-to-student categorical KL multiplied by temperature squared; both full heads normalise before any categorical coarsening |
| Hidden features | Exact assistant byte-boundary anchors, fixed contiguous pooling to at most 64 common channels, row normalisation and mean squared error |
| Attention and relations | Mean squared error on head-averaged attention restricted and renormalised to shared anchors, averaged with hidden-state cosine-relation error |
| Preference DPO | Reference-corrected chosen/rejected **summed** assistant log probabilities, using the frozen initial student and a stable log-sigmoid |

Every weighted objective contributes to the same backward pass; weights are
normalised by their sum. Teacher and reference tensors are explicitly detached.
Intermediate-layer objectives can leave later adapters outside the backward
graph. Their parameters and optimiser moments remain unchanged. The original
assessment optimiser still rejects unexpectedly missing gradients.

DPO freezes the exact pair identity, one original task prompt and independently
encoded chosen/rejected assistant sequences before constructing the optimiser.
Both frozen-reference likelihoods and both trainable-policy likelihoods use
those same sequences. Richer correction-generation prompts remain separately
bound in provenance. Composed SFT and tensor losses retain their own declared
conditioning; a pure DPO arm avoids their redundant forward pass. A separately
verified oracle-only failure can be a negative even though its original sandbox
quarantine remains true. Compiler failures and security/runtime failures cannot
be DPO negatives.
Windowed DPO additionally requires identical retained pre-assistant token IDs
and original prompt positions for both alternatives. Unequal-length responses
whose independent windows retain different prompt tails are inapplicable and
fail before optimisation; sharing a textual prompt does not waive this check.

Context examples retain the richer original teacher prompt for provenance and
native target capture, while the student receives only `task.prompt()`. Frozen
transforms bind both exact byte strings, task position and omitted context;
assistant byte spans align across the different prefixes. A context label with
no genuine extra teacher context is rejected. Tool trajectories retain their
actual trusted-grader/revision prompt and establish that fixed interaction,
without claiming arbitrary tool selection. Ordered progressive teachers form
a curriculum, rather than an unexecuted teacher-assistant architecture bridge.

Different Qwen model sizes can have identical defined token IDs and different
padded output-head widths. The KL objective retains each original semantic ID
and puts **all** remaining probabilities into one residual category per model.
This preserves probability mass without pretending that padding IDs have shared
meaning. Each update records the mapping hash, semantic/padding counts, both
residual probabilities at its actual temperature, and zero discarded mass.
Unrelated token-ID mappings, including Qwen versus BitNet, do not support this
KL alignment. Feature and attention objectives instead align exact original
completion byte boundaries; their records retain matched and available counts.

Teacher capture finishes before student allocation. Native models produce only
the fields and samples required by the frozen optimiser order, then synchronise
and drop their model and shared asset lease. Targets are CPU safetensors with
checksum-bound provenance in `offline_bundle/distillation-traces`. Cache keys
include source assets/output, exact student windows and byte spans, requested
signals, precision/device and the pinned engine; an arm seed does not invalidate
otherwise identical targets. `prepare_teacher_targets` supports a separate GPU
prewarm phase. `TeacherTargetPolicy::CacheOnly` subsequently forbids a CPU worker
from allocating a teacher unexpectedly when a target is missing or invalid.

One target payload is bounded to 512 MiB; only the current update's payload is
loaded. Selected targets and the shared disk cache are bounded to 8 GiB, with a
measured 512 MiB disk floor before capture. Existing supersets of requested
signals can satisfy smaller requests without another capture. A failed budget check
preserves earlier evidence and performs no unaccounted capture. Larger studies
need an explicit budget change and a new versioned study plan. Memory planning
uses the maximum of the separate teacher and student phases. Earlier actual
Qwen 512-token CPU gradient QA reached approximately 7.75 GiB RSS; a full DPO arm
can retain two backward graphs and therefore needs additional headroom.

Quick operator arms may explicitly select `SequencePolicy::AssistantWindow`.
This retains real assistant tokens and at least one preceding original context
token, records all omitted context/targets and never invents EOS when a response
is truncated. It tests genuine operators and updates, not whole-task learning.
Full arms reject overlength examples. Every selected teacher receives actual
updates, and each teacher's allotted updates cycle all declared content kinds.
Insufficient coverage budgets fail before optimisation. Progressive teachers
share one total update allowance, distributed by quotient/remainder in their
declared order; the allowance is never multiplied by the teacher count.
Full-context freezes additionally retain the exact excluded sample IDs and
overlength reasons; the checkpoint verifier reconstructs those exclusions.
Other alignment or evidence failures cannot silently become length exclusions.

Before a costly full-context round, `distillation::preflight_eligibility` verifies
the original evidence and invokes that same worker selection without allocating
model weights, acquiring teacher targets or creating an optimiser. The native
helper exposes it as `distillation_qa eligibility REQUEST.json REPORT.json`.
The immutable report binds the exact request, bundle, base, tokeniser and helper;
it lists every in-capacity sample, complete token counts for excluded examples,
and the actual repeated optimiser order. Its `selection_sha256` covers those
rows and that order, allowing matched SFT and composed requests to demonstrate
identical data selection despite different objective names. Rejected examples
are measured with the shared complete-sequence encoder for diagnosis only and
remain excluded from the actual round. This preflight establishes eligibility,
without establishing successful target capture, training or quality.

`freeze.json` binds the complete worker configuration, base and evidence hashes,
engine binary, preceding checkpoint, exact example order and encoded sequences.
It contains no dynamic free-RAM measurement. Each update fsyncs adapter and Adam
state into a numbered step directory before atomically publishing progress.
Restart validates those files, the completed order prefix, seed, objectives,
runtime and inputs, then performs only the remaining updates. The kernel releases
the worker's advisory lease after process death. A new semantic configuration
requires an explicitly fresh output; ordinary interruption resumes the pinned
worker and validated cache.

The final experimental manifest retains initial and final adapter identities,
base hashes before/after, optimiser state, per-step/component losses, gradient
norms, realised teacher/content counts, alignment diagnostics, target artefacts,
selected-teacher rights and actual execution metadata. It requires nonzero finite
gradients, changed adapter tensors, exact safetensor reload and a fresh model
instance producing finite logits. External study evaluation reloads the adapter
in its own process and runs the declared train/development tasks. The assessment
loader continues to reject this manifest kind; conventional exploratory loading
uses `distillation::load_runtime`, and BitNet uses its native `BitNetRuntime`.
The verifier independently reconstructs the selected rows from their original
evidence, checks every objective component and weighted total, recomputes the
realised teacher/content counts, and requires complete per-step KL diagnostics
and target-trace coverage. On-policy and self-distillation additionally require
the recorded initial adapter or validated base-weight identity to match every
student rollout.

`examples/distillation_qa.rs` provides the same native worker and target-prewarm
boundaries without the study scheduler. Its `resume` mode runs an uninterrupted
control, sends an actual SIGKILL after observing a durable optimiser step,
resumes the same request, compares every final adapter and Adam tensor exactly,
and launches a separate process to verify checkpoint reload and finite logits.
The synthetic two-token reload probe establishes numerical reload only. The
requests used by the workers still require genuine passing teacher evidence.
Use `pin` to retain the executable before running durable work:

```sh
cargo build --release --locked --offline --example distillation_qa
target/release/examples/distillation_qa pin
# Use the printed immutable state/engines/<sha>/lec-distil path below.
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  state/engines/<sha>/lec-distil resume reports/local-study/native-sft-request.json reports/local-study/native-sft-resume-qa-new
```

Direct new-helper phases need the same explicit environment. Set the variable
below to the newly built immutable helper printed by the build/pin command;
use fresh request/output protocols for new work. The prepared full-context
requests in these examples have not been executed:

```sh
LEC_DISTILLATION_HELPER=state/engines/REPLACE_WITH_NEW_HELPER_SHA/lec-distil
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" prewarm reports/local-study/native-full-composed-prewarm-request.json
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" worker reports/local-study/native-full-composed-request.json
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" verify reports/local-study/native-full-composed-request.json reports/local-study/native-full-composed-reload.json
```

The allocation is five available CPUs, with four numerical threads; check the
actual affinity/cgroup allocation before launching. Keep all four environment
values and the exact executable unchanged between prewarm, worker, interruption
recovery and fresh-process verification. `Acquire` and `CacheOnly` remain
separate immutable phase requests. The original `af675…` helper and its recorded
historical commands remain the recovery route for their existing freezes;
these new launch requirements do not amend that earlier evidence.

New helper launches retain `helper-launch.json` before numerical work. The common
receipt binds the exact executable, presence and values of the four thread
settings above, effective Candle and shared Rayon pool sizes, and Linux CPU
affinity when available. Prewarm, worker and reload phases retain separate raw
request bindings against that one contract; `Acquire` and `CacheOnly` requests
therefore remain distinct. A changed engine or thread environment cannot resume
the same output silently. An earlier worker freeze without this receipt is
rejected before any receipt or lock file is written; its original pinned helper
remains the route for replaying historical work. Existing evidence is never
retrospectively labelled with thread settings it did not record.

The underlying teacher-target cache fingerprint is unchanged and does not
include numerical thread fields. A same-executable cache entry acquired under
another output may therefore carry different actual `TraceRecord` thread
metadata, even when this output has a valid common launch receipt. The fresh
full-context protocols use one new executable and the fixed 4/5/4/4 environment
throughout, which bounds that ambiguity for those runs. Broader scheduler cache
reuse must independently check the original trace's execution thread metadata;
a new receipt cannot relabel legacy captures or establish their missing
thread settings.

Each QA output must be new; interrupted training outputs within it are resumed
by the exact pinned child. A different objective or fidelity needs a new request
and output directory. This separate local-study QA cannot satisfy the original
assessment's three independent provider businesses or authorise promotion.

`collect-student-rollouts` acquires the organiser evidence. It first generates
the declared batch from immutable base weights or an explicitly named adapted
checkpoint, saves each original output before container grading, then releases
that model before loading corrective teachers. On-policy collection uses those
actual student attempts and aggregate trusted tool feedback. Self-distillation
loads the same frozen student checkpoint again with the original attempt,
feedback and an available verified example from a different training task.
It retains the adapter identity separately from its original Hub base identity;
neither generation nor later target capture can silently load just the base.

Every corrective output is saved and independently regraded. Empty, failed,
truncated or quarantined corrections remain failed observations. Original
quarantine verdicts are immutable. A separate content-role classifier permits
complete audited oracle-only failures as negatives and trusted compiler errors
as bounded, explicitly untrusted correction context only. Missing audits,
prohibited operations, malformed responses, runtime failures, timeouts and
incomplete generations remain excluded from feedback. Capabilities become available only
after a complete corrective answer passes. Zero positives leave the capability
unavailable and retain the complete failed collection. This protocol performs
frozen-current-student on-policy batch collection followed by offline updates;
it does not query a teacher at each optimiser step.

`--reuse-student-attempts-from <original-output>` creates an immutable continuation
under a new output directory. It imports exact original student plan/raw/grade
artefacts and the original generation-engine identity, without resampling or
rewriting those observations. The new verification/feedback engine and its
preflight remain separate. Original grades are checked against the original
controller configuration; new feedback uses the new enforced configuration.
An unverifiable historical grade remains a traceable failed observation and
cannot supply trusted feedback, even when other imported attempts can proceed.

`rollout-report.json` describes only the current frozen batch. It records new
record coverage, new passing correction samples, absent/error/empty/truncated
generations and failed/unverifiable grading observations. Inherited passing
samples cannot turn a failed batch into success. The CLI returns failure unless
`healthy_complete` is true; valid partial evidence remains usable by explicitly
applicable independent methods with the evidence gap retained.

The exploration workflow freezes a `cohort-catalogue.json`, then keeps base
student/mode evidence and studies under separate cohort identities. On-policy
and self-distillation share the original student attempts through immutable
imports, while each corrective method has its own evidence and study. Derived
configs preserve source seeds, fidelity budgets, progressive flags and loss
families, restricting the student/organiser and content to actual linked rows.
Their quick/development-selection/full stages share the invocation's remaining
wall budget and use the same resource scheduler. Offline-family errors remain
non-successes while independent cohorts can continue. BitNet cohorts require
the public strict numerical-equivalence verifier bound to the exact baseline,
plus training/reload and resource capability; missing or failed receipts remain
explicit exclusions. Combined external/self-teacher cohorts stay queued until
a verified same-origin record-composition boundary exists.

```sh
./run.sh collect-student-rollouts --offline --device cpu --dtype f32 \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --student-checkpoint reports/local-study/native-sft-resume-qa/interrupted \
  --mode on_policy --teachers qwen-coder-7b-q4_k_m \
  --maximum-tasks 4 --output reports/local-study/on-policy
./run.sh collect-student-rollouts --offline --device cpu --dtype f32 \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --student-checkpoint reports/local-study/native-sft-resume-qa/interrupted \
  --mode self_distillation --maximum-tasks 4 \
  --output reports/local-study/self-distillation
```

The exact initial student, base identity and architecture must match the worker.
An existing adapter additionally requires the same rank and alpha as the
requested fidelity. A quick rank-two checkpoint cannot silently seed a rank-eight
full arm. True base rollouts permit a newly initialised adapter, but their
provenance must name the validated base-weight checksum. All these observations
remain train-only; development and sealed assessment tasks cannot enter them.

Focused invariant checks are executable with:

```sh
cargo test --locked --offline --lib distillation::
cargo test --locked --offline --lib selected_layer_update_preserves_inactive_tensors
cargo test --locked --offline --lib full_qwen_architecture_backpropagates_and_preserves_frozen_base
```

These include analytic KL with different padding widths and nonzero tail mass,
teacher/reference gradient isolation, DPO gradient direction, different hidden
widths, byte/token alignment, explicit assistant-window accounting and selected
layer gradients. Synthetic tensor QA remains labelled separately from genuine
teacher-trained checkpoint and measured quality evidence.

Actual local-teacher operator evidence is retained under
`reports/local-study/native-sft-resume-qa/`. The complete 48-task Qwen 7B training
comparison supplied 18 independently passing outputs and no original same-task
preference pairs. The three-step CPU/F32 SFT control and interrupted run used
declared 64-token assistant windows, rank two, alpha four and seed 1701. A real
SIGKILL after durable step one was followed by completion to step three; all
96 adapter tensors and 192 Adam tensors matched the uninterrupted control bit
for bit, as did losses, gradients and sample order. The committed step-one files
remained unchanged. A separate process reloaded adapter
`b8b51c10874c1609e36c1b15ab8f0c0709d645f397d1979f7e17b4aea472ba5a`
and produced 303,872 finite logits. The full evidence is in `report.json` and
`fresh-process-reload.json`, bound to pinned engine
`af675af8e54beb9ddc672d13bb860aec4175d7071b43fe9345af8c4ba9bbf182`.
These are genuine teacher-trained updates and recovery/reload observations;
short assistant windows and synthetic reload tokens establish no task gain.

The corresponding three-step composed arm also completed and reloaded in a
separate process (`native-composed-operator/manifest.json` and
`native-composed-reload.json`). It exercised SFT, KL, hidden features and
attention/relations together. Its measured temperature-two residual masses were
nonzero for both output heads, so padding was included in the actual loss.
Separate pure hidden-feature and pure attention/relation arms also completed
real optimiser updates (`native-feature-operator` and
`native-attention-operator`). Per-component losses and gradients are retained;
decreasing losses across different selected samples are not a quality estimate.
The pure KL arm also completed three real updates (`native-kl-operator`).
All three pure objectives subsequently passed independent process reloads,
retained in `native-kl-reload.json`, `native-feature-reload.json` and
`native-attention-reload.json`; each produced 303,872 finite logits under the
original pinned operator engine. These synthetic two-token probes establish
reload/finiteness rather than coding quality or improvement.

The first native BitNet round used the genuine official base, three 32-token
assistant-window updates, rank two, alpha four and declared learning rate 0.005.
It produced finite gradients and changed latent tensors, but crossed **zero**
effective ternary thresholds among 245,760,000 adapted q/v weights. The worker
correctly returned failure and retained `native-bitnet-operator/` with
`blocked-no-discrete-update.json` and all three step journals. It published no
successful checkpoint manifest. A separately declared longer protocol must keep
this failed observation and cannot rewrite its frozen update budget.

The separately declared 48-step BitNet protocol also finished all genuine
32-token updates and correctly failed its discrete-update gate. Its effective
lattice remained unchanged across 245,760,000 adapted q/v values; all journals
and `native-bitnet-48-operator/blocked-no-discrete-update.json` are retained.
The read-only `native-bitnet-48-movement.json` diagnostic reconstructs every q/v
low-rank update from saved latent tensors without loading the full model.
Its largest absolute update divided by the original scale was 0.3269575, below
the half-bin threshold of 0.5. The nearest switch requires 1.52925 times the
saved update. Original scales ranged from 0.8359375 to 4.59375; 498,994 latent
values changed and 997,940 Adam-moment values were nonzero. The optimiser was
reachable, but this frozen update budget did not cross a quantisation boundary.
Labelled gain-scaling calculations are counterfactual diagnostics, not trained
checkpoints. The diagnostic executable, exact Rust build/source/dependency
receipt and input hashes are linked by the report. Any further protocol needs
a fresh request/output and its own actual discrete-update check.

The separately declared 64-step BitNet round completed genuine updates and
crossed 230,999 effective ternary decisions out of 245,760,000 adapted q/v
values. Its retained `native-bitnet-64-operator/manifest.json` binds adapter
`a7bd00bfe048d1d31dfe402558017c8e3c5e73eee1f8bab6720673e832cc19dd`.
The worker and a separate process using the original `af675…` helper both
exited successfully; `native-bitnet-64-reload.json` records 256,512 finite logits
from the synthetic two-token reload probe. The read-only
`native-bitnet-64-common-prefix.md` comparison establishes that its saved
48-step adapter and Adam tensors equal the original 48-step protocol exactly.
The earlier three-step and 48-step failures remain intact. This establishes a
real adapted ternary checkpoint and independent numerical reload, without
establishing coding improvement, native inference equivalence or promotion.
