# Explicit supplemental teacher contexts

The original [round 001 report](round-001/content-report.json) remains unchanged:
12 completed current records, one passing process sample and 11 generation
errors. It reports `healthy_complete=false`; its 18 inherited positives cannot
turn the new batch into a success. The errors retain their original 1,024-token
runtime limit and cannot be repaired by rewriting those observations.

The new Rust capability adds an explicit total teacher context:

```sh
# Use a fresh, preregistered output and the newly verified pinned executable.
lec-distil collect-teacher-content \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --output reports/local-study/content-follow-up/round-002 \
  --context-tokens 1536 --max-tokens 768 --maximum-tasks 4 \
  --content process,context,tool_trajectory --offline \
  --device cuda --gpu 0 --dtype f32
```

This is a command contract, not an executed round-002 result. GPU ownership,
source verification and the fresh collection protocol must be settled before
running it. The corresponding rollout option is `--teacher-context-tokens`:
it applies only to original GGUF corrective teachers in on-policy collection.
It does not alter student rollouts, self-distillation runtimes or native tensor
capture. Dense, adapted and BitNet teacher overrides fail explicitly.

Omitting either option retains the original loader and its existing cap. The
optional fields are absent from historical serialised plans and policies. A new
explicit value changes the immutable plan identity; resuming a directory with
a different value fails instead of regenerating finished work.

For an explicit GGUF context, checks precede model allocation:

- The requested limit is within the implemented 1–4,096 range, the original
  source maximum and the GGUF publisher's declared context length.
- The original tokeniser and chat template count the exact augmented prompt.
  The entire declared maximum output is included; exceeding the total is an
  explicit generation failure, with no truncation.
- Context-dependent conservative CPU/GPU memory estimates must fit the current
  platform's guarded RAM/VRAM capacity. An explicit CUDA request cannot fall
  back to CPU. Availability may change after preflight; an allocation failure
  remains a failure, rather than a promise of an atomic resource reservation.

Actual runtime metadata already records `context_cap` and the quantised memory
plan. New evidence verification binds that cap to the explicit content or
corrective-feedback plan and checks its input/output token bound. Revalidating
saved observations consults neither transient hardware availability nor a
newly loaded teacher.

The actual [prompt and resource preflight](context-preflight-002.json) counted
all 12 immutable round-001 plans with the original publisher tokeniser/template.
The largest input is 548 tokens; 1,316 total tokens retain the full 768-token
output allowance. All 12 fit the proposed 1,536 cap. The helper allocated no
model, requested no GPU allocation and exited zero in 1.34 seconds with
71,024 KiB peak RSS. An earlier diagnostic path-capture failure remains in
`context-preflight-001.log`; it produced no successful report.

| Total context | Conservative GPU bytes | Minimum free VRAM for the existing guard |
| --- | ---: | ---: |
| 1,536 | 6,875,121,664 | 7,788,154,879 |
| 1,792 | 7,065,438,208 | 7,999,617,706 |
| 2,048 | 7,270,434,816 | 8,227,391,715 |

The observed free VRAM was 5,450,498,048 bytes while an authorised evaluation
occupied the GPU. None of these configurations fitted that live snapshot.
The proposed 1,536 cap therefore requires GPU release and a fresh guard before
collection. The estimates are arithmetic, not a measured future allocation.

The new unit checks cover source/publisher/runtime limits, token-count overflow,
the exact token-budget boundary, mismatched actual context metadata, historical
plan-field omission and CLI defaults. Repository-wide verification is a separate
record; implementing these checks does not establish a successful new generation,
training sample, quality gain or assessment promotion.
