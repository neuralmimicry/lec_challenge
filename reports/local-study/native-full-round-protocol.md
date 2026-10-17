# Prepared conventional full-context round

Status: requests prepared, not executed. These are fresh protocols; the existing
short-window checkpoints, failed BitNet observations and their freezes remain
unchanged. This local study cannot authorise assessment SHIP.

Both arms use the actual 18-positive Qwen teacher bundle from all 48 original
training tasks. The worker independently verifies provenance, rights, container
verdicts, original task/source identity and held-out exclusion before opening an
optimiser. It rejects examples exceeding 512 tokens rather than truncating their
context or assistant response. The exact accepted order and excluded sample IDs
are recorded in each new freeze. With identical seed, teacher, content and
sequence policy, the arms must select the same eligible rows/order; inspect the
actual freezes before making a paired comparison.

| Setting | Both arms |
| --- | --- |
| Base | Pinned Qwen2.5-Coder-0.5B-Instruct |
| Steps | 48 |
| Sequence policy | Complete example; reject over 512 tokens |
| Seed | 1729 |
| Adapter | Rank 8, alpha 16 |
| Learning rate | 0.0002 |
| Student | Native Rust CPU/F32 |
| Teacher targets | Actual Qwen 7B CUDA/F32 capture; detached safetensors |
| SFT arm | Sequence SFT, weight 1 |
| Composed arm | SFT, temperature-two KL, hidden features and attention/relations, weight 1 each |

Use a newly built, verified and pinned CUDA `distillation_qa` example executable.
The old `af675…` example is CPU-only and cannot perform the declared GPU capture.
All new direct helper phases must use the exact same launch contract:
`RAYON_NUM_THREADS=4`, `LEC_STUDY_CPU_THREADS=5`, `OMP_NUM_THREADS=4` and
`MKL_NUM_THREADS=4`. Reserve five available CPUs per helper and retain its
actual affinity. The helper freezes these values, the effective Candle/Rayon
pool sizes and its exact executable before numerical work. Changing a value,
its presence or the executable requires a fresh protocol; it cannot resume an
existing output or consume another executable's target cache.

Run the following with the printed new immutable CUDA helper path substituted
for the placeholder. These are prepared command contracts, not measured runs:

```sh
LEC_DISTILLATION_HELPER=state/engines/REPLACE_WITH_NEW_HELPER_SHA/lec-distil
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" prewarm reports/local-study/native-full-composed-prewarm-request.json
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" worker reports/local-study/native-full-sft-request.json
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" worker reports/local-study/native-full-composed-request.json
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" verify reports/local-study/native-full-sft-request.json reports/local-study/native-full-sft-reload.json
env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_DISTILLATION_HELPER" verify reports/local-study/native-full-composed-request.json reports/local-study/native-full-composed-reload.json
```

Prewarm needs the GPU exclusively available; release the teacher before CPU
training. Its request has
`Acquire`; both actual training requests use `CacheOnly`. Their target-runtime
identity is identical, so a missing/corrupt target fails before optimisation and
cannot trigger an unexpected teacher allocation. SFT requires no tensor targets.

Run each training request through the pinned example's `worker` command. The
actual worker binds its executable, request, accepted order, evidence, initial
adapter and durable Adam state. Do not change a request after a freeze exists.
Interruption resumes its original executable and progress. Finish with separate
`verify` processes and matched train/development task evaluations; no held-out
outcomes may tune this study.

The earlier real 512-token CPU gradient smoke took 40.06 seconds and peaked at
8,124,212 KiB RSS, approximately 7.75 GiB. This is a measured smoke, not a
full-round timing or composed-loss peak. Reserve 16 GiB per arm plus controller
and operating-system headroom. Start two workers only with at least 34 GiB
available and suitable CPU allocations; otherwise run sequentially. A provisional
40–120 minutes per arm is an estimate until these rounds are actually measured.
The declared full-profile maximum is 14,400 seconds. GPU capture and the current
BitNet round must finish before guarded dispatch; no successful checkpoint,
quality gain, energy charge or cost per improvement point is claimed here.
