# Matched complete-context coding measurements

Status: protocol preparation; no quality result is claimed here. The frozen
JSON request controls execution. Each checkpoint receives a separate identity
receipt after genuine completion and fresh-process reload, before its first
generation. The composed round may finish after this protocol is written; its
acceptance criteria are fixed now and its eventual tensor identity cannot be
chosen from quality outcomes.

Evaluate the unchanged Qwen2.5-Coder-0.5B-Instruct base and the actual 48-step
SFT and composed adapters on the original 16 development tasks, then all 48
original training tasks. Run every evaluation with the same immutable
application `f5d2f2f5…`, CUDA device 0, F32, seed 1729, temperature 0.2, top-p
0.95, output limit 512 tokens and 120-second generation deadline. Assign four
CPUs and four numerical threads; isolated graders share the same frozen
controller version 2 and image policy. CUDA must be released by the current
owner before dispatch. Keep one model resident at a time.

The development baseline is shared by both comparisons, as is the train
baseline. No historical observations from other engines are reused. Each new
output uses durable per-task generation/grading artefacts and its original
pinned executable on resume. Preserve errors and partial outputs. A run with
generation or infrastructure errors cannot establish quality improvement.

Compare full denominator pass counts and source-macro scores only after the
independent verifier confirms every artefact. Report train and development
separately. These repeated local exploratory measurements use one seed and
jointly authored source groups; they cannot establish statistical learning,
general teacher superiority or assessment promotion. Sealed held-out tasks
are excluded. API-provider evidence, energy charges and cash expenditure
remain separate requirements.
