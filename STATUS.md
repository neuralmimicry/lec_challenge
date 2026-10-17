# Assessment status

## Provider setup repair — 16 October 2026

The reported failure came from an absent `state/api-providers.local.toml`;
the unqualified check was inspecting the disabled, zero-budget shipped template.
Rust `provider-init` now creates a valid bounded inference profile atomically
without overwriting existing files/symlinks. It requires explicit model/revision,
prices, source/date and budget choices. Checks identify the file, selected model,
credential presence, readiness and zero-budget/disabled/unpriced blockers;
`--require-ready` can enforce those local checks without HTTP. Missing-file errors
include the path and setup action. The request default is 2,048 output tokens,
including reasoning; explicit smaller caps remain enforced.

The actual README initialisation command passed through `./run.sh` and created
the ignored, secret-free local file. It enables only OpenAI `gpt-6.1-sol`, with
USD 0.25 and 20,000-token ceilings in each of the four scopes. Its maximum call
reservation is USD 0.06144. The input pricing ceiling deliberately covers cache
writes; reported profile costs are conservative estimates, not exact invoices.
The dated official model/pricing sources and alias limitation are documented.
Output training remains forbidden; this profile cannot supply an assessment
training set without a separate applicable permission decision.

All four actual agent-environment credential checks were absent. The user's
terminal reported an OpenAI key in that shell; it is not inherited here. The
local OpenAI check has no configuration error and only the absent key blocker.
No live provider access or account entitlement is claimed, and no paid API call
was made. Labelled missing-key CLI injection failed before reservation with an
empty cost report; missing-config, empty-recording, require-ready and duplicate
initialisation probes all exited nonzero as required. Existing historical runs,
the user's original failure artefact and caches/models were preserved.

Verification passed **333 tests**, with five intentional ignores, strict
all-target Clippy, formatting and whitespace checks. The optimised release build
passed in 6m 20s; the subsequent launcher initialisation/check reused it in
1.77s / 0.25s of Cargo build time. Logs, source hashes, CLI outcomes and evidence
boundaries are retained in
[reports/provider-setup-qa/verification.json](reports/provider-setup-qa/verification.json).
README and `docs/api-providers.md` contain creation, credentials, four-provider
configuration, budgets, reasoning caps, replay and error-specific remedies.
The original assessment's real-provider/training/evaluation gaps remain open.

## Latest-log assessment review — 16 October 2026

The latest retained exploration still reports `success=false`; one local full
SFT seed reports 1/16 development passes versus the base's 0/16, without sealed
held-out proof. The original assessment still has zero real API-provider
businesses/recordings, no provider-derived assessment dataset/checkpoint, an
INVALID EVALUATION, five incomplete drills and no attributable round cost or
cost per improvement point. Local genuine checkpoints remain separate evidence.
Fresh credential-presence checks found all three original-roster and four
additional API environment credentials absent; no values were printed.

`assessment-audit` now creates a new read-only, checksum-linked JSON/Markdown
review and exits nonzero for missing, invalid or unverified original evidence.
It does not call providers, run models, change frozen observations or prune
caches/models. The [latest-log review](docs/assessment-latest-logs.md) describes
the measured gaps and closure order.

The latest rollout's original `host deadline terminated probe pids` failure was
masked by a parent missing-file error. Current source verifies the child receipt
and log before retaining its cause/exit/path; a missing report remains failure
even after exit zero. The read-only audit reconstructed that actual old failure
without rerunning it. Failed sandbox enforcement remains failed; no thresholds,
timeouts or sealed evaluations were changed.

Final verification passed **327 tests**, with five intentionally ignored, plus
formatting and strict all-target Clippy. The fresh
[audit](reports/assessment-audit/latest-log-review-002/audit.md) exited 1 with
`original_request_satisfied=false`, identified thirteen incomplete requirements
and reconstructed two failed child receipts. Seven historical observations
remain distinct from independent verification. Test logs, source hashes and
the exact verification boundary are retained in
`reports/assessment-audit/verification-001/`. No paid calls, new training or
sealed generations were performed during this review.

The design was frozen before assessment code in commit `74eba150f766a0d93c2a8ad18066daac562314cf` on 15 October 2026. Original history and the prebuild pack are preserved. Four specialist agents have implemented and executed their local work in parallel; responsibilities are recorded in agents/PLAN.md.

Operational constraint added on 16 October 2026: preserve caches and models when
working with `--offline` or when no internet connection can be detected. The
teacher-search workflow now skips offline pruning and requires confirmed Hub
connectivity before each deletion. Labelled fixture tests verify that inactive
weight bytes survive offline operation and failed connectivity detection.

## API connectivity follow-up — 16 October 2026

- Native Rust adapters now cover OpenAI Responses/Codex, Gemini Developer API,
  Gemini on GCP Vertex AI and Anthropic Messages. `provider-check` performs local
  checks without HTTP; `provider-request` supports explicitly authorised live
  inference or exact offline replay. Setup is in `docs/api-providers.md` and the
  README. The new template is disabled, unpriced and has zero budgets.
- Credentials use environment variables and provider-specific sensitive headers.
  Missing credentials and definite connection failures can use configured
  fallbacks. Permanent authentication/model/quota failures disable that route for
  the runner; transient errors honour bounded retries, circuits and Retry-After.
  Possible remote acceptance remains UNKNOWN and blocks automatic repurchase.
- Transactional round/task/provider/model token limits count input and billable
  output alongside the existing dollar budgets. Failed observations, unresolved
  liabilities and historical unknown token usage remain visible. General
  inference cannot clear output-training permission or enter assessment
  collection through a fallback.
- Verification: **321 tests passed, five intentionally ignored**, formatting and
  strict all-target Clippy passed. CLI QA observed exit 1 without spending
  authority, exit 1 with missing offline recordings and exit 2 for incompatible
  offline/spending flags. Earlier fixture-path and CLI-validation failures are
  retained in `reports/provider-api-qa/verification.json` and the intermediate
  log. No paid API calls, real provider outputs or live connectivity are claimed;
  all four credential checks reported absent. Existing provider/training evidence
  gaps remain unchanged.

## Verified local evidence

- Native Rust implementation across supervision, provider clients, SQLite ledger, tasks/oracles, grading, curation, Qwen adaptation, statistics and serving. No Python trainer or grader.
- 96 executable Rust function contracts from 24 declared local source packs, split 48 train / 16 development / 32 held out by complete source. All public expected examples and split checks pass. Shared AI authorship and oracle-review limits are explicit.
- Transactional integer-money budgets, caching, UNKNOWN acceptance handling, exact checksum-addressed replay, bounded retries/rates/circuits and independent-provider failover. Local fixture QA includes actual child-process SIGKILL and restart without another transport call. These are not real provider recordings.
- Real native Podman enforcement: non-root UID65532, read-only mount returning EROFS on a world-writable root file, no network namespace routes, CPU quota, 512 MiB memory, zero swap and 64 PIDs. Actual OOM137, excess child denial, running workload deadline/removal and complete ptrace/reference execution observed.
- Real adversarial candidate drills pass: skipped work/fake success/public-answer hardcoding fail and quarantine. Network/host-only file/actual hidden-test path/root-write attempts are denied and recorded by the trusted supervisor. Contaminated temporary datasets fail through the actual training entry point before optimiser use.
- Actual pinned Qwen2.5-Coder-0.5B base inference produced complete Rust code. Actual full-size 512-token synthetic forward/backward/AdamW QA supervised 256 tokens, changed q/v adapter variables, preserved the frozen-base sentinel and passed a safetensors round-trip. Its measured update took 20.918 seconds and the process peaked at about 7.60 GiB RSS. This is implementation QA, not a provider-trained assessment checkpoint.
- Actual localhost serving generated complete code, rejected oversized bodies/token caps, bounded queue pressure and deadlines, and exercised evaluation-only routing and rollback with the real base.
- The final local demo passed four required drills: grader gaming, escape attempts, planted held-out data through the actual trainer entry point and a budget stop before dispatch. All separate transport/statistical/infrastructure QA assertions passed. Five provider/checkpoint-dependent required drills remain incomplete.
- Formatting, strict all-target Clippy, standard tests, the original-tokeniser asset test and three real container integration tests passed. Exact commands, counts and captured output are retained in `reports/verification.json` and its linked logs.
- A captioned 120-second evidence video presents real reports with input hashes and discloses unfinished assessment work.
- Fresh-archive offline QA passed in 88.55 seconds: all bundled inputs verified before execution, original checkout hidden, no provider keys or outbound routes, fresh application state/cache, real base endpoint and four independent required drills. All invariant assertions passed. The assessment retained exit `1` / `INCOMPLETE`. The host runtime image cache was preloaded; this does not establish a complete provider-trained rehearsal. Evidence is in `reports/disconnected-qa.json` and `reports/disconnected-evidence/`.

## Local adaptive extension in progress

The user's later scope adds portable hardware discovery, adaptive teacher selection, owned-cache pruning, resumable composable distillation studies and a genuine ternary BitNet student. This extension retains the original sealed held-out boundary and does not replace independent API-provider evidence.

- The pinned native CUDA engine measured Qwen2.5-Coder-7B Q4_K_M at **7/16 development passes**, with complete generations and no infrastructure errors. It remains below the provisional 80% qualification threshold. Strongest fallback retention is explicit; no high-quality teacher fleet is claimed yet.
- The actual 48-task train collection produced **18 passing solutions**, with **six generation/infrastructure failures** retained in the complete denominator. Its command correctly exited nonzero. Only valid, complete, rights-cleared passing rows may enter a local-study bundle.
- The completed original replacement search exhausted its six declared candidates without qualifying a teacher. Qwen Coder 7B Q4 remains the strongest healthy result at **7/16**; general Qwen 7B scored **5/16**, and Qwen Coder 7B Q6 scored **6/16**. Qwen Coder 14B Q2 retained **6/16 with three errors**. All 16 Magicoder generations reached the output limit; these are incomplete observations rather than a valid quality ranking. DeepSeek 6.7B stopped during the sandbox memory preflight and remains unevaluated. The original executable's historical `verified_below_threshold` labels for errored trials remain unchanged; current code classifies those trials as inconclusive. See `reports/comparative/search/round-001/report.json`.
- Owned pruning actually removed **20,140,487,248 logical weight bytes** across the three weaker dense alternatives, general Qwen 7B Q4 and Qwen Coder 7B Q6. Publisher manifests, deletion authority and measured reports remain intact. The assessment base, retained 7B teacher and incomplete trials remain protected; active model leases prevent deletion during new work. The historical search reserved 25,209,217,184 asset bytes, a download allocation rather than measured network traffic or cash spending.
- Native BitNet numerical QA performed eight actual ternary-aware updates on a tiny synthetic model, changed **57/96 effective trits**, and verified exact adapter/packed-trit reload. This is an operator capability test, not an official model checkpoint or task improvement. The official 2B base also passed a real CPU/F32 16-token forward/backward check: finite loss and nonzero gradient, **no optimiser steps**, measured peak RSS about **15.09 GiB**. These distinctions are retained in `reports/comparative/bitnet/`.
- Actual local-teacher Rust SFT completed three updates from the 18 passing rows. A real SIGKILL after durable step one resumed to the same **96 adapter tensors and 192 Adam tensors**, loss/gradient histories and sample order as an uninterrupted control. Previously committed step files remained unchanged. A separate process reloaded the actual adapter and produced 303,872 finite logits. This is short-window operator/recovery evidence, not an observed task-quality gain; see `reports/local-study/native-sft-resume-qa/report.json`.
- The official full-size BitNet export loaded in the pinned C++ engine, passed original-tokeniser parity checks and generated a complete ten-token greeting in 7.876 seconds. Its supervised process shut down normally. This establishes real inference compatibility, not coding quality or successful adaptation; see `reports/comparative/bitnet/cpp-smoke.json`.
- Three real updates combining SFT, token KL, hidden features and attention relations completed and passed a fresh-process reload. All four objective components were finite, student gradients were nonzero, teacher padding probability was preserved and the adapter changed. These 64-token screening steps establish executable composed objectives, not task improvement; see `reports/local-study/native-composed-operator/manifest.json` and `reports/local-study/native-composed-reload.json`.
- The separate pure KL, hidden-feature and attention/relation operators also passed fresh-process reloads, each producing 303,872 finite logits. Their three-step manifests and `native-*-reload.json` receipts remain under `reports/local-study/`; this is numerical operator evidence.
- A matched CUDA/F32 baseline completed all 16 development tasks with **0/16 passes and zero generation/infrastructure errors**. The corresponding three-step SFT adapter also completed all 16 generations with **0/16 passes**, but retained **seven grading infrastructure failures** and correctly exited nonzero. Five lacked trusted audits and two reached the owned-container inspection deadline; all seven retained sandbox reports reached the frozen 20-second ceiling. These incomplete grades cannot establish candidate quality or improvement. Original outputs, preflight failures and engine `a429e2731fded954a0bdd7d97e11a5842902fe2f7d8d0700adaeef2cceb279db` remain unchanged under `reports/local-study/sft-{baseline,adapter}-development-c7c/`. Neither evaluation opens the assessment held-out split.
- The official-base BitNet SFT attempts completed **three and 48 actual 32-token updates**, with finite gradients and changed latent adapters, but both changed **zero of 245,760,000 effective q/v trits**. Both correctly blocked successful adapted manifests; their independent protocols and failed journals remain under `reports/local-study/native-bitnet-operator/` and `native-bitnet-48-operator/`. A read-only diagnostic examined all adapted packed values and the saved optimiser: the largest update was **0.32695754 scale units**, below the half-bin threshold, and the minimum saved-update gain for a first switch was **1.52925057**. Counterfactual gain calculations are analytical, not checkpoints. This measured shortfall motivates a separate bounded 64-step protocol; it does not justify changing either original run.
- The separate **64-step official BitNet round completed and passed independent reload**. It changed **230,999 of 245,760,000 effective q/v trits**, retained its packed identity, and produced **256,512 finite logits** in another process. Its first 48 adapter and Adam tensors exactly match the preserved 48-step round. Training took **38 minutes 45.52 seconds**, peaking at **16,330,624 KiB RSS**; external reload took **3 minutes 57.43 seconds**. The actual adapter and evidence are under `reports/local-study/native-bitnet-64-operator/` and `native-bitnet-64-reload.json`. This establishes genuine discrete adaptation and reload, not coding quality, native numerical equivalence or promotion.
- New Rust modules implement sequence, padded-vocabulary mass-preserving KL, hidden-feature, attention/relation and preference losses; genuine content and checkpoint-bound rollout collectors; finite capability-checked experiment catalogues; per-student CPU/RAM/VRAM scheduling; per-step optimiser recovery; and separate-process development evaluation. Method comparisons, full numerical parity, current whole-project QA and refreshed packaging are still being verified.
- All three numerical Rust/C++ BitNet comparisons failed the fixed 0.10 top-16 log-probability tolerance. The latest explicit F32-KV measurement used a separate reviewed native rounding/minimum-scale correction and retained maximum errors **0.661798477, 0.471328735 and 0.291747093**, despite agreeing top predictions. Its failed schema-2 report and original raw/full-vocabulary outputs remain in `reports/comparative/bitnet/parity-quant-v2.json` and `schema2/`. The correction did not establish equivalence. Selection requires independently reconstructed checkpoint/engine/export-bound evidence; further numerical changes require a per-layer diagnosis and a separate protocol.
- Actual controller-version-2 crash QA **passed** in `reports/sandbox-crash-qa/round-004/report.json`: an infinite candidate was observed, its owner was SIGKILLed, the independent runtime ceiling and durable journal recovery worked, a subsequent reference passed and an unrelated sentinel survived and was correctly removed. Earlier failures remain unchanged, including round003's final sentinel-cleanup deadline failure. Runtime events distinguish candidate death from later container removal.
- Portable sandbox V2 QA **passed** on this amd64 host in `reports/sandbox-portable-qa/report.json`: actual OOM137, child-limit denial, a one-second execution deadline, all four escape probes and all three grader-gaming candidates. Measured cgroup termination took 101 ms and removal 2,722 ms. ARM execution remains unverified.
- The latest completed coordinated verification, `reports/local-study/verification-006/verification.json`, passed **all eight software checks**: formatting, strict all-target Clippy, **254 ordinary tests**, 11 launcher tests, diagnostic lifecycle, two numerical-helper launch tests, original-tokeniser QA and all three real container integration tests. Exact source inputs remained unchanged throughout. The command correctly retained exit one because one original-notice gap remains in optional `bindgen_cuda-0.1.6`. Failed earlier verification passes remain intact. Source-level CPU SIMD receipts and arm64 compiler-cfg checks are implemented.
- The CUDA/SIMD build completed and pinned four role-specific executables in `reports/local-study/build-cuda-simd-001/result.json`, binding the actual compiler and explicit AVX2/FMA/F16C flags. Both prepared full-context arms passed eligibility using the new numerical helper: **17 accepted examples from 11 training sources**, one 541-token example rejected, identical selection/order SHA `8090729b420b520fe2707be87e9d41dbc7b09e7f6f6ad9c3d9baf61bb0f28b00`. GPU target prewarming precedes the actual 48-step SFT and composed rounds; eligibility alone is not a training result.
- Both full-context conventional rounds subsequently completed **48 genuine updates and separate-process reloads**. Each changed all **540,672 finite adapter values across 96 tensors**, while the actual base stayed unchanged. SFT took **11:48.33**, peaking at **7,676,504 KiB RSS**; composed SFT/KL/features/relations took **16:22.66**, peaking at **10,204,732 KiB RSS**. All four composed objectives were exercised at every update. Exact manifests, reloads, launch contracts and a separate Rust tensor audit are linked from `reports/local-study/native-full-round-result.md`. This is completed adaptation, not a measured coding gain.
- Actual evaluation SIGKILL/resume QA **passed** under the new pinned application: complete original generations/grades were unchanged, the resumed command and independent verifier exited zero, and all 16 development observations were healthy. The three-step composed checkpoint still scored **0/16**. See `reports/local-study/study-evaluation-resume-qa/round-001/report.json`.
- All **192 original full-round observations** are now complete under the unchanged executable/protocol. Development retained **0/16 for base, SFT and composed**, with respectively **one, three and nine grading infrastructure failures**. Train retained **0/48 base**, **11/48 SFT** and **6/48 composed**; SFT was healthy with no generation or grading errors, while base retained three errors and composed one. Two base errors are actual length-limited generations. Independent protocol/hash audits passed, but the original paired quality comparisons remain invalid. Regrading cannot repair model-generation failures. All original outputs remain in `reports/local-study/native-full-quality-001/`.
- Both second-seed **2718** rounds completed **48 full-context updates** using the original numerical helper and same 17 eligible examples. SFT took **14:55.76**, peaking at **7,680,300 KiB RSS**; its separate **20.35-second** CPU/F32 reload produced **303,872 finite logits**. The combined-loss round took **14:05.26**, peaking at **10,313,480 KiB RSS**, and passed a separate **29.73-second** reload with the same finite-logit count. Both independent `native-full-*-seed-audit-001.json` audits passed. These whole-training seed variants change initialisation and shuffled sample exposure, disclose CPU placement, preserve both original rounds and do not establish coding-quality or seed-equivalence results.
- Independent reconstruction found a specific JSON parsing defect: the original encoded SFT score `0.22916666666666666` (11/48) was parsed one ULP lower by the default parser. All counts, source grades, tokens and timings still matched. The new build enables correctly rounded JSON float parsing and tests exact round trips for every success fraction through the 192-observation denominator; original report bytes and strict equality remain unchanged. The diagnosis is retained in `reports/local-study/native-full-quality-001/report-reconstruction-diagnosis.json`.
- The first bounded content follow-up retained **12 records, one passing process example and 11 context-cap generation errors**. It correctly failed rather than treating inherited positives as successful new collection. New explicit context controls preserve historical defaults and bind actual limits. A model-free Rust preflight counted every original prompt: the largest needs **548 input + 768 output = 1,316 tokens**. A separately frozen **1,536-token** follow-up fits those requests and is subject to the live GPU guard; 2,048 would exceed this card's conservative memory estimate. See `reports/local-study/content-follow-up/`.
- A new Qwen3 4B Q4 teacher was downloaded and publisher-verified: **2,497,280,736 weight bytes**, SHA `2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e`. Native Rust Qwen3 support and the actual DeepSeek legacy linear-RoPE metadata spelling are implemented and tested. Actual Qwen3 inference/qualification remains pending a new pinned build. The original failed DeepSeek recheck remains a compatibility observation, not a poor-model score. See `docs/qwen3.md`.
- Verification **007** preserved unchanged source inputs and passed formatting, strict all-target Clippy, **270 ordinary tests**, launcher/lifecycle/helper/tokeniser checks and **two of three** real container tests. The real deadline/alias test failed because bounded SIGKILL/removal was not proved. Independent native events also show that slow runtime startup/removal can discard completed audit output when the old CLI wait is cancelled. A separately identified controller fix and immutable-generation regrading are in progress; the failed observations and verification report remain unchanged.
- A bounded Rust C-ABI diagnostic captured the original native engine's first two layers without loading a second full model. Original embeddings match exactly; same-input checks measured RMS and floating-matrix arithmetic differences. Their contribution to final log-probability drift remains unresolved; later operand aliasing limits are explicit. A separately tested, unintegrated RMS prototype matches all **35,840** first-cut native outputs exactly and passes independent first-order gradient checks. This does not establish an end-to-end fix. See `docs/bitnet/per-layer-findings.md`.
- Producer-bound follow-up resolved the eight measured RMS operand aliases. The actual integrated Rust RMS correction matches **817,152 F32 values** across all eight normalisations and eight weighted cuts bit for bit, and passes finite-difference/structural gradient checks. It is included in the new parity implementation identity. The original failed full-model comparisons remain; a fresh immutable-build comparison at the unchanged **0.10** tolerance is still required.
- Actual V3 portable sandbox QA passed in `reports/sandbox-portable-qa/v3-round-001/report.json`, including kernel resource enforcement, deadline, reference, escape and gaming checks. Separate V3 crash QA passed in `reports/sandbox-crash-qa/v3-round-001/report.json`: after the actual orchestrator SIGKILL, the observed exact cgroup became empty after **20.134 seconds**, the stopped container retained exit **124** and its trusted watchdog envelope, and subsequent dispatch recovered/removed that exact object. A reference passed and an unrelated sentinel survived before verified cleanup. No competing host kill was supplied during the independent-watchdog observation. This is local sandbox evidence, not a provider replay drill.
- Coordinated verification **009** passed all **nine software checks**, **293 ordinary tests** and all **three actual container tests**, with source inputs unchanged. Its overall exit remains **1** solely for the documented optional dependency original-notice gap. The new four-role CUDA/SIMD build is frozen in `reports/local-study/build-cuda-simd-002/`; existing experiments retain their original engines. The original assessment's full requirement/drill/delivery crosscheck is in `docs/assessment-audit.md`.
- The actual full-context SFT adapter is served through a local CPU/F32 endpoint in `reports/local-study/student-serving-001/report.json`. Health binds adapter `e2bcd135527659042304d936fa56c7240f6f96a43b9f81f2a888c38b7762a949`; HTTP generation was complete, the deliberately selected `ledger.prefix` training example passed V3 grading and production routing rejected the evaluation-only endpoint. This is operational QA, not a quality estimate, learning proof or assessment canary.
- The assessment code audit corrected automatic GPU/CPU estimates for explicitly expanded teacher contexts. The later CLI change passed **295 ordinary tests**, strict all-target Clippy, formatting and whitespace checks; source/log hashes and the exact evidence boundary are in `reports/local-study/context-aware-selection-001/`. The earlier coordinated verification remains unchanged. A bounded filename-only Hugging Face credential-pattern scan found no matches in the scanned source, documentation, configuration or asset metadata.
- Teacher follow-up round 002 retained DeepSeek **0/16 with one length-limited generation**; it remains inconclusive, not a healthy quality ranking. Qwen3 stopped before inference when a sandbox preflight settings check exceeded its host deadline. The first attempt's report/state/freeze are preserved in `reports/comparative/search/round-002-attempt-001/`; no Qwen3 coding score exists yet. The 192-row V3 regrading batch is still running and has retained one cleanup-verification failure; passing targeted QA has not removed that operational limit. See the assessment audit for exact lifecycle timings.
- The separately declared **1,408-token GPU content follow-up completed healthily**: 12/12 complete generations and grades, seven new passing examples, five rejected variants, zero generation/grading/infrastructure errors. A separate process independently verified the resulting **25-sample** bundle, including its 18 inherited positives. Original round 001 and the interrupted CPU round's four timeout records remain preserved. Evidence: `reports/local-study/content-follow-up/round-003-gpu-1408/` and its protocol/resource receipt.
- Fresh full-model BitNet parity with the integrated RMS correction **completed and failed** the unchanged **0.10** tolerance. Maximum errors were **0.4428310394 / 0.4030475616 / 0.1987771988**; all three top predictions agreed. The raw numerical observations, independently reconstructed result, exact source snapshots and nonzero exit remain in `reports/comparative/bitnet/parity-native-rms-001*`. No numerical-equivalence, coding gain or validated BitNet winner is claimed.

Existing delivery archives and their disconnected rehearsal describe the earlier CPU snapshot. They do not yet package or certify this extension. Linux amd64 is the measured platform; ARM64/CPU and supported CUDA resource discovery paths need their own hardware execution evidence. Other GPU backends currently use an explicit CPU decision.

## External evidence still unavailable

The agent still lacks provider credentials and applicable account/model
output-training decisions. The user's terminal reports a locally exported
OpenAI key; the new general-inference profile is bounded at USD 0.25, but its
account access is unverified and it carries no assessment training clearance.
The default assessment roster remains disabled/unpriced. There are no real
teacher recordings, approved curated provider dataset,
primary/seed/degraded/overfit assessment adapters or complete real paired
evaluations. No improvement, SHIP decision, real round cost per point or complete
provider-trained disconnected rehearsal is claimed.

`./run.sh demo --offline` and `./run.sh drill --offline` must currently exit nonzero and identify these gaps. The actual worse/train-overfit/seed controls remain incomplete; synthetic decision fixtures remain separate. Publication and email submission have not been authorised or attempted.

## Remaining sequence

1. Configure keys locally, approve a spending ceiling and resolve model/account terms, model IDs and prices.
2. Collect all three businesses, curate passing traceable winners and train all four predeclared adapters with their original evidence.
3. Freeze the primary candidate, run development calibration and one held-out evaluation, report actual controls without searching for favourable patterns.
4. Interpret the conservative gate, measure complete round costs, finish actual production canary mechanics if SHIP, and perform the complete fresh disconnected replay.

Detailed outcomes and exact missing paths are in reports/drill.json, reports/demo.json, reports/eval.json and docs/limitations.md. Formatting, lint and tests are recorded in reports/verification.json. Packaging records belong in reports/bundle-manifest.json and deliverables/SHA256SUMS; fresh-directory execution belongs in reports/disconnected-qa.json when generated. A successful local QA report does not complete the missing provider-trained assessment.
