# Actual evaluation interruption and recovery QA

`examples/study_evaluation_resume_qa.rs` controls its own evaluation subprocess.
It never signals an existing training process, never opens sealed held-out data
and never builds an executable. Its default checkpoint is the completed
`native-composed-operator`: three real operator-screening updates with SFT,
token KL, hidden features and attention relations. This is recovery evidence,
not evidence that those updates improved coding quality.

Build the example through the normal coordinated software QA, then supply an
explicit immutable application executable selected for the run:

```sh
target/debug/examples/study_evaluation_resume_qa \
  --engine state/engines/<exact-application-sha256>/lec-distil \
  --device cuda --threads 4 --wall-timeout-seconds 1800 \
  --output reports/local-study/study-evaluation-resume-qa/round-001
```

The placeholder SHA must be replaced with the actual pinned engine. Select a new
output for a new QA attempt; existing evidence is retained. CUDA requires an
explicit CUDA-capable engine and available capacity. `--device cpu` is supported
with its own explicit run budget. No device fallback is introduced by the QA.

Before starting the model, the diagnostic pins and executes its own immutable
copy, records the checkpoint and exact child arguments/thread environment, and
publishes the evaluation's launch receipt. The child freezes the complete
original 16-task development denominator and the evaluator's fixed decoding
protocol. The supervisor waits for an actual complete nonempty generation and
independently checked container grade/audit. A trusted compiler-failure verdict
can anchor completed failed work only when the existing strict classifier
independently verifies it as `CompilationFeedbackOnly`. The anchor records this
explicitly; the original failure/quarantine remains unchanged and no positive
or DPO-negative training role is inferred. Thus recovery does not depend on
candidate accuracy. Missing/invalid audits, model errors, missing outputs and
truncation cannot trigger the claimed completed-work boundary.

While the evaluator is still active and before a complete report exists, the
supervisor SIGKILLs only its retained child process group and reaps it. It saves
the actual signal and every published generation, observation, grade, audit,
first-load receipt and frozen plan checksum. It restarts the same immutable
engine with exactly the same arguments, environment and output directory.

A separate immutable verifier process reconstructs the completed report from
all original raw outputs and trusted grades. Every pre-interruption cached
artefact must still match its original checksum. Overall QA passes only when
the interruption occurred, those outputs stayed unchanged, the resumed command
completed successfully and the complete evaluation is healthy. Candidate
solutions can still fail coding tests; those zero scores remain in the report.
Generation or infrastructure errors cause an explicit QA failure.

The wall deadline covers both evaluation attempts and independent verification.
The supervisor kills and reaps its own process group on deadline/error, and
Linux parent-death handling protects the model child if the diagnostic dies.
Container ceilings and durable sandbox recovery remain separate enforced
controls. Failed attempts retain a JSON report, the interruption snapshot when
available, child stdout/stderr and any partial evaluation evidence. No successful
execution of this new diagnostic is claimed until an actual report exists.
