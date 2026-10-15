# Resumable local exploration

The workflow below is implemented. A complete end-to-end invocation has not yet
been observed; individual numerical, recovery and inference evidence is recorded
separately. The command's integration status must not be read as a measured
teacher qualification or student learning result.

New CPU protocols use a verified explicit [CPU SIMD build profile](cpu-builds.md).
Resumption always retains the original executable and frozen ISA requirements.

`./run.sh explore` coordinates the supplemental local-model workflow. The
default invocation has an explicit 86,400-second wall budget, printed at startup.
Use `./run.sh explore --offline` for cached assets only. Each invocation resumes
the same output under `reports/local-study/explore/round-001`; changing a data,
teacher, decoding or numerical setting requires a fresh output. The CLI's
explicit fresh option creates a sibling directory and preserves earlier links.

The command performs these stages:

1. Wait for an existing teacher-search writer, then resume its exact original
   pinned executable. Preserve its previous state and report before retrying.
2. Freeze every qualified retained distinct original teacher source. If none
   qualifies, freeze only the strongest measured fallback and retain the
   qualification shortfall. Revisions, asset manifests and both development
   comparisons remain independently verifiable.
3. Collect all 48 original training tasks with actual teacher generations and
   trusted container grades. A matching previous Qwen 7B training comparison
   is reused, including failed and truncated outputs. A partial matching
   comparison resumes its original engine and immutable raw records.
4. Acquire actual process, contextual and feedback/revision variants in
   `content-v1`, retaining their rejected outcomes and availability reasons.
   The completed bundle is frozen before the study plan is constructed.
5. Create the finite capability-checked study, run its bounded quick profile,
   freeze a decision using independently verified development outcomes, and
   execute the selected full-fidelity groups with their preregistered seeds.

Before training, each applicable student architecture receives an actual
unadapted-base development evaluation through the same pinned engine and fixed
task/decoding/runtime protocol. Its child receives the same declared CPU thread
budget as a quick worker. Complete cached baselines are independently reverified
without new inference. A baseline containing generation/infrastructure errors
is retained, but cannot establish a quality improvement.

Each profile writes `baseline-comparisons.json` alongside its report. The table
contains actual source-macro success, baseline differences, generation timings,
actual inference threads, declared CPU/RAM/VRAM reservations and recorded worker
attempt times. Full-fidelity resource allocations may differ from quick
allocations, so a thread mismatch prevents a matched-speed interpretation.
Peak resident memory, energy and cash cost remain explicitly unmeasured. These
are descriptive local results, with no significance, promotion or global
optimality claim.

The frozen orchestration protocol pins a stage executable separately from the
older search executable. Rebuilding the launcher cannot replace either while
resuming a stage. The search reader verifies historical trials against the
search freeze's engine identity rather than today's executable. An offline
invocation cannot perform metadata requests, asset downloads or native engine
bootstrap downloads. Acquisition permission and the invocation's wall budget
can change without retuning the frozen experiment.

Shared model-asset leases cover collection, content generation, planning, target
acquisition and all worker lifetimes. Pruning cannot delete an active teacher.
When a later invocation finds a frozen teacher intentionally pruned, online
operation may rehydrate only the same owned, publisher-bound identity. Offline
operation reports it unavailable. A present corrupt asset or an unclaimed
missing asset fails verification; it cannot become an implicit cache repair or
a different selected teacher.

Every subprocess stage records a request, stdout, stderr and result receipt
under `attempts/<stage>/<attempt>/`. Heartbeats identify the active stage, its
retained log and remaining wall budget. The controller terminates owned Unix
process groups on timeout or cancellation. Linux parent-death signals cover
the worker, separate checkpoint evaluation process and the native server's
dedicated persistent supervisor. Atomic generation and grading records remain
available for the next invocation.

After an interruption, run the same command with the same output directory.
The launcher resolves the original content-addressed executable before considering
a new build. Verified completed downloads, generations, container verdicts and
optimiser steps are reused; incomplete work resumes at its last durable boundary.
Recorded model errors and incomplete answers remain observations in the original
denominator. A missing or corrupt executable, model or evidence artefact produces
an explicit failure rather than an automatic fresh experiment. An intentional
restart uses `--fresh`, which creates a sibling output directory and preserves
the earlier evidence.

Stage stderr is streamed live and retained through an asynchronous fixed-size
tee, including download bytes, speed and ETA. Stdout stays in its file so a
large catalogue does not flood the terminal. Log flushing and synchronisation
finish before the attempt receipt is published; a failed drain prevents success.

`report.json` contains the frozen protocol, actual stage receipts, selected
teacher identities, capability gaps, errors and measured invocation time. Its
`success` field remains false for incomplete stages, generation/infrastructure
evidence gaps, budget stops or a teacher qualification target shortfall. A
bounded fallback study may still complete and produce useful checkpoints under
`completed_with_teacher_target_shortfall`; this status is not a successful
qualification result. A profile with no healthy complete arm is an error.

The workflow exposes training and development sources only. The original
assessment's three independent real providers, sealed promotion gate, measured
improvement and deployment authorisation remain separate evidence requirements.
No new paid provider calls are made. Electricity, hardware amortisation and
cost per improvement point remain unknown unless actually measured. Unit
fixtures establish protocol and cancellation invariants, not model quality.
