# Latest logs against the original assessment — 16 October 2026

The latest retained logs do **not** establish completion of the original
assessment. Passing protocol/unit tests and genuine local training checkpoints
do not supply absent real provider calls, clean held-out comparisons,
checkpoint-control drills or attributable learning costs.

## Repeatable read-only check

```sh
./run.sh assessment-audit
# Inspect another research report and choose a new output directory if needed.
./run.sh assessment-audit \
  --exploration reports/local-study/explore/round-001/report.json \
  --output reports/assessment-audit/review-002
```

The command writes `audit.json` and `audit.md` in a new directory and exits
nonzero while original evidence is missing, invalid or not independently
verified. The default directory has a new UUID; explicit existing output
directories are refused. It does not call providers, execute models/candidates,
generate held-out results, train, prune caches/models, publish or alter a frozen
experiment. File/hash-heavy checks run on a blocking worker.

The audit distinguishes **VERIFIED**, **OBSERVED**, **INCOMPLETE** and **INVALID**.
A historical passing report is an observation until its required proof has been
reconstructed. A reported development gain is not a sealed assessment gain.
The command cannot certify design quality or a fresh execution from filenames
and log labels; it assigns no speculative score out of 100. The longer
[assessment crosscheck](assessment-audit.md) maps each original design/build/
drill/delivery requirement to its implementation and remaining boundary.

## What the latest evidence says

The newest research summary is
`reports/local-study/explore/round-001/report.json`, last modified at
14:56:22 UK time on 16 October. Its `success` is `false` and its status is
`completed_with_teacher_target_shortfall`. The later provider API QA at around
15:00 verifies software behaviour, not successful real provider acquisition.

| Original requirement | Observed evidence | Assessment consequence |
| --- | --- | --- |
| At least forty source-separated tasks | 96 Rust contracts; train/development/held-out split 48/16/32 by whole source. | Exceeds the numerical minimum; shared AI authorship remains a source-independence limitation. |
| Three real independent providers | The original roster remains disabled/unpriced; API checks report absent credentials, and real API recordings are absent. New Codex/Gemini/Claude adapters are general inference until output-training permission is independently cleared. | Not complete. Local open-weight teachers and synthetic transport tests cannot substitute. |
| Real training set and checkpoint | Genuine local-teacher datasets and changed/reloaded adapters exist. The strict provider-derived assessment dataset/checkpoint is absent. | The trainer is real; the required end-to-end provider lineage is missing. |
| Proof of learning | One full-fidelity SFT seed reports 1/16 development passes versus the base's 0/16; the other SFT seed and the measured composed-loss arm report 0/16. | The observed +6.25 percentage points is descriptive on development data. It is not significant held-out improvement or proof of generally beating the teachers. |
| Two-checkpoint gate | The original `reports/eval.json` decision is `INVALID_EVALUATION`. | A valid SHIP/NO SHIP decision with complete actual checkpoint-bound held-out observations remains missing. |
| All nine drills | The preserved original report has four PASSED and five INCOMPLETE cases. | Real recorded-fleet crash/failover and the actual worse/overfit/seed checkpoint controls remain incomplete. |
| Serving and real costs | Local base/student endpoint QA exists; the original cost summary has zero real businesses, `acquisition_round_complete=false` and no round total or cost per improvement point. | Serving mechanics are observed. No provider-trained assessment endpoint/canary or complete attributable learning cost is established. |
| Clean disconnected delivery | Historical disconnected QA passed while the full assessment correctly failed. A video and archives describe an earlier snapshot. | An old binary/archive or successful handling of missing inputs cannot certify the current full submission. |

The latest on-policy rollout batch records 48 attempted records, zero newly
verified rollout samples and 48 failed records. The content stage reports 127
records, including 122 generation errors, with only four passing current
samples. Inherited samples cannot make that new batch healthy. The BitNet
baseline retains sixteen generation/infrastructure errors; its
`verified_complete` stage label denotes completed report coverage, not sixteen
successful model generations. The audit prints the separate `healthy` field.

## Concrete logging correction

The most recent self-distillation cohort's child ended with:

```text
FAILURE: host deadline terminated probe pids
```

Its durable receipt records exit 1 and 57.479108858 seconds. The parent then tried
to read an absent `rollout-report.json` and reduced the error to `No such file or
directory`. That hid the operational failure which prevented collection.

The parent now reads required stage reports through a bounded helper on a
blocking worker. If the report is missing, it verifies the original receipt and
stderr hash and retains the child's failure, exit code, receipt path and log
path. It never substitutes an empty success report or repeats inference.
Changed logs are rejected rather than trusted. The audit reconstructs this
cause from the existing receipt without rerunning the failed batch. Original
logs, protocols and observations remain unchanged.

This fixes loss of diagnostic evidence. It does not turn the failed PID-control
probe into passing sandbox enforcement. A later independently recorded probe
must pass before that workload can run; no timeout, tolerance, denominator or
held-out seed has been changed to repair the historical result.

## What is still needed

1. Configure three genuinely training-eligible provider businesses, exact
   models/revision policies, reviewed pricing and local credentials, and provide
   an explicit dollar ceiling. The [provider guide](api-providers.md) documents
   the credentials and token/money limits. No new paid call is authorised by
   requesting this log audit.
2. Collect and record real completions, grade in verified containers, curate and
   independently validate their full provenance, then train/reload the declared
   primary and control adapters. Preserve failures and original reservations.
3. Freeze the candidate/configuration, calibrate detectability on development,
   run complete sealed comparisons and interpret all real controls honestly.
   NO SHIP is an acceptable decision; invalid/missing evidence is incomplete.
4. Reconcile actual task/model/round costs, serve the matching checkpoint and
   verify canary/rollback as applicable. Package the exact current dependency
   closure, run a fresh disconnected rehearsal and refresh the video.

These missing observations cannot be manufactured by extending an adapter,
renaming a local checkpoint or printing a successful log label.

## Captured verification

The [fresh audit](../reports/assessment-audit/latest-log-review-002/audit.md)
returned exit 1 and `original_request_satisfied=false`. It identified thirteen
incomplete requirements, retained seven historical observations separately and
verified the task corpus. It reconstructed both failed cohort receipts,
including the original PID-probe deadline failure. No original report was
rewritten. All seven original/additional API credential-presence checks were
absent, with secret values never printed.

The final full `cargo test --locked --offline` run passed **327 tests**, with
five intentionally ignored; formatting and strict all-target Clippy passed.
[Verification metadata and logs](../reports/assessment-audit/verification-001/verification.json)
record the commands and scope. This is software/evidence-inspection QA, not a
new successful provider-trained assessment round.
