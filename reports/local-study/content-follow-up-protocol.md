# Additional train-only teaching signals

Prepared before these new generations. This is a bounded acquisition protocol,
not an assertion that the records exist or the methods improve task quality.
The original content, failed on-policy batch and frozen worker outputs remain
unchanged. No assessment held-out source is opened.

Use application executable
`state/engines/f5d2f2f520f0c7a69186163c33bd28207c37414fb3cdae0499c169957a846944/lec-distil`.
Its CUDA/SIMD build receipt is in `build-cuda-simd-001/result.json`. Set
`LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64`, `RAYON_NUM_THREADS=4`,
`LEC_STUDY_CPU_THREADS=4`, `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` for these
collection commands. These application collections have their own frozen
policies; they are separate from the numerical helper's 4/5/4/4 contract.

The first acquisition uses the verified 18-positive Qwen bundle and the first
four original training contracts, in their existing corpus order. Request all
three supported content variants: generated plans as Rust comments, extra
same-source context and a genuine trusted-grader feedback/revision trajectory.
Use the retained Qwen Coder 7B Q4 teacher on CUDA/F32, seed 1729, temperature
0.2, top-p 0.95 and at most 768 new tokens. The command freezes the exact prompts
and records unavailable variants rather than inventing their dependencies.
Only independently passing complete programmes enter positive examples.
Preferences additionally require independently classified benign failures and a
common conditioning prompt; a quarantined security or infrastructure failure
cannot become a negative preference.

```sh
LEC_COLLECTION_ENGINE=state/engines/f5d2f2f520f0c7a69186163c33bd28207c37414fb3cdae0499c169957a846944/lec-distil
env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_COLLECTION_ENGINE" collect-teacher-content --offline \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --output reports/local-study/content-follow-up/round-001 \
  --content process,context,tool_trajectory --maximum-tasks 4 --max-tokens 768 \
  --device cuda --dtype f32
```

After GPU release, two further independent acquisitions reuse the original four
student attempts in `on-policy-qa`; no original attempt is regenerated or
regraded. Both retain the exact original conventional three-step student,
CPU/F32 runtime, seed 1729 and 256-token cap. The continuation classifier keeps
missing or invalid audit evidence as failures and excludes it from trusted
feedback. The original output is never edited to make the batch healthier.
An independently generated teacher correction still needs its own complete
container verdict before it can become a training example.

```sh
env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_COLLECTION_ENGINE" collect-student-rollouts --offline \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --student-checkpoint reports/local-study/native-sft-resume-qa/interrupted \
  --reuse-student-attempts-from reports/local-study/on-policy-qa \
  --output reports/local-study/on-policy-follow-up/round-001 \
  --mode on_policy --teachers qwen-coder-7b-q4_k_m \
  --maximum-tasks 4 --max-tokens 256 --seed 1729 --student-device cpu \
  --device cuda --dtype f32

env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$LEC_COLLECTION_ENGINE" collect-student-rollouts --offline \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --student-checkpoint reports/local-study/native-sft-resume-qa/interrupted \
  --reuse-student-attempts-from reports/local-study/on-policy-qa \
  --output reports/local-study/self-distillation-follow-up/round-001 \
  --mode self_distillation --maximum-tasks 4 --max-tokens 256 --seed 1729 \
  --student-device cpu --device cuda --dtype f32
```

These collections run serially with respect to the laptop's sole GPU and trusted
sandbox pressure preflights. CPU training may overlap only with adequate measured
RAM and reserved CPU cores. Preserve nonzero exits, raw outputs, health reports
and all original hashes. An available teaching signal is not evidence of useful
training: later composed arms require separate frozen requests, actual updates,
reloads and matched development comparisons. A fixed feedback/revision trace
does not establish arbitrary agentic tool selection. These bounded observations
cannot establish the best combination across the entire configured catalogue.
