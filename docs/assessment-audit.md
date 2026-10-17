# Original assessment crosscheck

The [latest-log review](assessment-latest-logs.md) records the newer exploration
and API QA boundary. Run `./run.sh assessment-audit` for a new read-only,
checksum-linked JSON/Markdown inspection; it does not rewrite this historical
crosscheck or run the missing provider/training/held-out work. The latest parent
rollout error hid a failed PID-control preflight; current source preserves that
verified child cause instead of reducing it to a missing-file message.

Audit date: 16 October 2026. Scope: the original LEC email, the implementation
brief, the Rust implementation and retained execution evidence. The later local
teacher/method/BitNet work is additional scope. It cannot replace missing
original-assessment evidence. This is a traceability review, not an independent
security certification or an estimated assessment mark.

**The original assessment remains incomplete.** The largest gaps are real calls
to three independent provider businesses, their rights-cleared training data,
complete frozen held-out/control evaluations, attributable round costs and a
refreshed delivery containing the extension's execution dependencies. Genuine
local-teacher training and sandbox QA exist; no verified generalisation gain or
SHIP decision exists.

Statuses distinguish implemented behaviour from an observed end-to-end result.
Historical reports certify their own executable, policy and inputs. A passing
unit fixture cannot fill an empty real-evidence cell.

## Design: every requested topic

The design was frozen in commit
`74eba150f766a0d93c2a8ad18066daac562314cf` before implementation. The retained
[freeze](design-freeze.txt), [hashes](DESIGN_SHA256SUMS),
[Markdown](design.md) and [three-page A4 PDF](design.pdf) agree with that
history. The later Rust/GPU/study changes belong in [deviations](deviations.md).

| Original requirement | Location and finding | Remaining qualification |
| --- | --- | --- |
| Loop and capability ceiling | Design §1 follows task → fleet → verifier → dataset → adapter → gate → canary → new tasks. It permits narrow complementary learning beyond an individual teacher; it does not promise general frontier superiority. | A real matched teacher/fleet/student comparison is still required to measure that hypothesis. Changed weights and train memorisation do not establish it. |
| A week's task supply and replenishment | §2 specifies private consenting repositories, independently authored contracts, oracle review, lineage inheritance and source diversity. There are 96 executable local tasks. | No continuing external private-repository supply agreement or measured week of ingestion exists. AI authorship and shared tooling weaken independence; the design discloses this. |
| Cheap-first grading, gaming and defence costs | §3 specifies syntax/schema checks before container execution, public/hidden/property oracles and trusted traces. Real gaming/escape QA exists. | All 96 contracts do not have separately human-audited second oracles. Finite tests cannot rule out every hardcoded solution. Grading time exists in run evidence, but full-round cash allocation is absent. |
| Interaction benefit, price and equal-cost experiment | §4 keeps independent best-of-three as the baseline and proposes a matched-dollar critique/revision experiment including context, retries and grading. | The optional equal-cost experiment was not run. Local content/repair collection is not this experiment. Best-of-N is permitted by the assessment, so critique interaction is not required for the baseline. |
| Adaptation method, durable criteria and provider terms | §4 chooses LoRA SFT, assistant-only labels, open weights, fit, memory and reproducible tooling. Terms are retained separately from output ownership. | Selected live account/model/revision prices and output-training decisions remain unresolved; the disabled fleet cannot be represented as an acquired roster. |
| Clean hold-out, gate, detectability and rollback | §5 specifies complete source splits, common decoding seeds, source-macro paired intervals, +2-point margin and sensitivity simulations; §6 retains the incumbent on failure. | Local novel wording is not proof of unseen algorithms. The eight test sources are weak evidence; real development calibration and frozen assessment observations are absent. |
| Canary and rollback | §6 describes immutable local endpoints, 5% gated traffic, health/error checks and atomic rollback. | Base/evaluation routing mechanics ran. A genuinely promoted trained-candidate canary has not run. |
| Cost table, dominant line and cost per point | §7 has explicit illustrative arithmetic, teacher-call dominance under those assumptions and the percentage-point denominator. | The table is a capacity estimate, not a bill. Actual learning-round cost and dominant measured line remain unavailable. |
| Exclusions and weakest part | The final two design sections identify bounded scope and low statistical power/source diversity. | Later exploratory additions expanded scope; the original frozen design remains unchanged and deviations must remain visible. |

## Build: every numbered requirement

| Requirement | Implementation / observed evidence | Status and closure |
| --- | --- | --- |
| One clean-machine command, no credentials, recorded offline replay | `run.sh`, `src/main.rs`, fleet replay and `src/package.rs`; the earlier [disconnected rehearsal](../reports/disconnected-qa.json) used a fresh extraction with no outbound routes. | **Partial.** The rehearsal honestly returned INCOMPLETE with no provider recordings. It certifies the earlier CPU archive, not the current extension. Complete real acquisition, package the exact execution closure and rerun offline in a fresh directory. |
| ≥40 tasks, executable graders, hold-out by source | `src/tasks.rs`, task/oracle submodules, corpus tests and source-lineage validation implement 96 tasks / 24 packs, split 48 train / 16 development / 32 held out. | **Observed locally.** Source-disjointness is checked. Shared authorship and incomplete independent oracle review remain limitations, rather than missing sample count. |
| Three real providers; configurable roster | `config/fleet.toml` and `src/fleet/`; business IDs, endpoints, revisions, prices and permission evidence are configuration. | **Blocked on external evidence.** Zero verified API businesses acquired. Local open-weight teachers do not meet this requirement. Enable only after selected account/model terms, keys, prices and an explicit spending ceiling exist; acquire and retain real responses from all three. |
| Rates, budget before every call, failover, cache and attribution | `src/fleet/`, `src/ledger.rs`, integer-nanodollar transactions, canonical request/sample keys, retained UNKNOWN reservations and `src/cost.rs`. Actual child-SIGKILL fixture QA exercises recovery. | **Implemented / fixture-tested.** Real provider acceptance, billing, retry/failover and exact recorded replay still need end-to-end acquisition evidence. Exactly-once remote charging is explicitly not guaranteed. |
| Container per run, non-root, read-only, no network, time/memory limits | `src/sandbox/`, frozen policy/controller V3 and actual [portable QA](../reports/sandbox-portable-qa/v3-round-001/report.json). [Crash QA](../reports/sandbox-crash-qa/v3-round-001/report.json) observes kernel death before recovery. | **Observed on Linux amd64.** Exact image/control identities are recorded. ARM64 code is implemented but hardware execution remains unverified; shared-kernel isolation has the disclosed limits. |
| Real deduplicated/decontaminated traceable training set | `src/curate.rs` independently checks run/attempt/rights provenance, exact/syntax similarity and held-out lineage before training. `src/teacher_evidence/` independently validates actual local train-only runs. | **Partial.** A real local-teacher set exists, but the strict API-provider dataset does not. Acquire eligible outputs, curate winners and independently validate the final dataset before opening the assessment optimiser. |
| Genuine adaptation and real checkpoint | Native differentiable `src/model/` and `src/train.rs`; actual local full-context SFT/composed rounds, changed tensors, Adam state, original-helper bindings and separate-process reloads. Both second-seed provenance audits passed. | **Genuine local checkpoint observed; assessment lineage missing.** This is not a stub. Repeat the assessment trainer on its required curated provider set. Local teacher learning cannot repair the three-provider gap. |
| One command, two checkpoints, decision, interval and MDE | `src/gate.rs`, `src/evaluate.rs`, `gate` CLI, artifact verification and exact-rule power simulation. Negative/null/missing/tamper fixtures exercise fail-closed decisions. | **Implemented / fixture-tested; real assessment evaluation absent.** `reports/eval.json` remains INVALID EVALUATION. Freeze one candidate/configuration, calibrate on development and evaluate complete common-seed held-out pairs once. |
| Checkpoint behind a local endpoint | `src/serve.rs`: bounded queue/body/output/deadline controls, immutable health identity, checkpoint loading and production gate checks. The [real student HTTP smoke](../reports/local-study/student-serving-001/report.json) passed for the exact SFT adapter and one selected training example, with independent sandbox grading and production-route rejection. | **Observed exploratory endpoint; assessment deployment incomplete.** This selected example is operational QA, not a quality estimate. A SHIP-trained assessment candidate endpoint/canary is absent. Research endpoints always remain evaluation-only. |
| Real task/round/point costs | `src/cost.rs` and [accounting contract](costs.md) retain usage, retries, uncertainty, cash/credits and measured compute separately. Current [cost report](../reports/costs.md) reports zero acquisition attempts and unavailable round/gain figures. | **Incomplete.** Collect real usage/prices and cash/compute allocation evidence. With no positive verified gain, cost per point is N/A; zero replay API spend is not a free learning round. |

## Drill: all nine required cases

`./run.sh drill --offline` dispatches all nine cases and writes explicit
PASSED/FAILED/INCOMPLETE outcomes. The preserved [original drill report](../reports/drill.json)
has four required cases passed and five incomplete. New sandbox QA is additional
evidence; it does not rewrite that report or complete provider-dependent cases.

| Case | Actual evidence boundary | What remains |
| --- | --- | --- |
| Kill orchestrator; resume without paying again | Real process SIGKILL/restart works with labelled transport fixtures. Actual V3 sandbox SIGKILL proves independent candidate death and owned recovery. | Run the recorded-fleet drill against genuine acquired responses; preserve every finished key and its original charge. UNKNOWN acceptance retains its reservation. |
| Provider error then rate limit; finish elsewhere | Production retry/rate/budget/failover path is tested with explicitly injected faults. | Exercise the same path with genuine response recordings and labelled injected errors/429s; no invented provider errors or usages. |
| Skip tests/hardcode; zero and quarantine | Actual fake-success/empty/skipping/public-answer-hardcoding candidates fail trusted oracles and quarantine. | Completed local evidence; retain original case/audit/image bindings. Finite-test residual risk remains. |
| Network, outside filesystem, hidden tests; denied and logged | Actual denied operations have trusted syscall evidence; candidate messages do not grant success. | Completed local probes on the measured platform; do not extrapolate to all kernel exploits. |
| Plant held-out task; catch before training | Temporary exact and lineage/reworded contamination is rejected through the actual training entry point before an optimiser. | Completed local evidence. Preserve canonical datasets and do not turn the injected row into a normal training sample. |
| Deliberately worse checkpoint blocked | Decision fixtures reject a negative held-out delta. | A real wrong-target adapter with a complete measured negative held-out result is still missing. Train and evaluate the declared control; if it does not exhibit the intended pattern, report that outcome. |
| Better on train, worse held out blocked | Fixtures demonstrate that train gain cannot rescue held-out regression. | A real separately trained overfit control and complete train/held-out observations are missing. Do not search sealed data for a convenient seed/pattern. |
| Seed-only checkpoints; no false win | Two genuine seeds now exist for both full-context local objectives, with independently verified equal settings/data and disclosed shuffled exposure. Null fixtures do not call identical outcomes a win. | Complete matched real quality measurements for the pair; a reload/provenance audit is not a no-significant-difference result. An occasional significant seed result must be reported honestly. |
| Budget exceeded; stop and save | Actual reservation below the next maximum prevents transport, retains progress and reports the stop. | Completed local mechanism evidence. Genuine acquisition still needs real-price budget coverage, including UNKNOWN attempts. |

Every empty, errored, truncated, missing or unverified observation remains a
failure. Original full-context train/development outputs are immutable. Derived
regrading can repair grading infrastructure evidence; it cannot repair the two
original length-limited base generations. A batch with retained generation
failures must still exit nonzero.

## Submission artefacts

| Requested item | Existing material | Remaining delivery work |
| --- | --- | --- |
| Repository or ZIP, README, one command | Local Git history, README, `run.sh`, earlier checksum-indexed archives. | No publication was authorised. Refresh the archive with current exact binaries, model dependencies, original notices and evidence; verify it after extraction. |
| Design document | Frozen three-page PDF and Markdown, with original prebuild retained. | Complete. Later deviations remain separate. |
| Drill, evaluation and costs | Truthful incomplete original reports, local-study and current sandbox reports. | Missing real provider/control/held-out/cost evidence must remain explicit, even if unfinished at submission. |
| Video ≤5 minutes | Earlier captioned 120-second evidence video and [outline](video.md). | It describes the earlier snapshot. Refresh only after final evidence/package stabilises, retaining truthful failures and limitations. |

## Closure order and audit cautions

The code audit found and corrected a concrete hardware-selection gap in
`src/main.rs`: explicit content/correction context sizes were passed to the
loader, while automatic device selection still estimated the historical
1,024-token context. Selection now estimates the requested context, so an
automatic run can select CPU RAM when the larger context cannot fit VRAM.
Explicit CUDA remains fail-closed, explicit limits are never silently cropped,
and saved runs retain their original runtime. The boundary tests use synthetic
amd64/ARM64 capacity profiles; they do not establish ARM64 hardware execution.
Current running experiments keep their already pinned binaries and protocols.
The fix passed 295 ordinary tests, strict all-target Clippy, formatting and
whitespace checks; [captured logs and source hashes](../reports/local-study/context-aware-selection-001/README.md)
retain the exact verification boundary.

The subsequent V3 batch also exposed an operational limit under overlapping
disk work. Derived `evaluation-001-development-e2bcd1355276` / `money.basket`
retains an infrastructure failure: run
`93c57dd9-2776-4bd7-9970-fc2cf0668a4f` had a real compiler error and a stopped
container, but terminal inspection took 11,998 ms and acknowledged removal
12,176 ms. The separate 30-second cleanup budget expired before verified
absence. Its lifecycle therefore says `container_removed=false`; the strict
verifier rejects it. These observations do not establish a candidate escape or
an isolated throughput cause. They do establish that passing portable/crash QA
does not guarantee a healthy large round on a busy host. Preserve the original
failure; investigate storage isolation and bounded housekeeping before any
separately versioned retry. Increasing a budget afterwards cannot repair this
observation or authorise a quality claim.

Another performance gap is repeated whole-weight verification across phase
boundaries. The teacher search's acquisition, comparison planning and model
loading each rehash the same multi-gigabyte file. Retained logs show checksum
throughput falling under concurrent I/O; this is measured overhead, not model
inference time. A future optimisation should reuse a bounded verification
receipt only within a retained immutable asset lease and validated open-file
identity, with mutation/replacement tests. Independent delivery verification
and cold resumption must still reconstruct content hashes. Filename-only or
unchecked persistent memoisation would weaken the evidence contract and is not
an acceptable performance fix. This optimisation is not implemented here.

The non-blocking requirement remains partially satisfied. The follow-up fixes
the identified content-collection path: policy/request preparation, raw
generation publication, grade links, record verification and final bundle
publication now use at most two admitted blocking workers per process. Workers
retain their directory lease and admission permit through fsync even if the
awaiter is cancelled. Completed generation and its raw write share one worker;
the outer failure report retains the same ownership. The new isolated tests
exercise executor responsiveness during a controlled slow writer, cancellation
without premature unlock/admission, exact failed-generation reuse and a changed
request's rejection without resampling. Schemas and durable dependency ordering
remain unchanged; existing frozen experiments keep their original binaries.
This closes that filesystem/hash publication gap, not every synchronous
operation in the application. Other CLI/collector paths and native allocation
destruction on cancellation still need their own latency/ownership review.

1. The immutable four-role CUDA/SIMD build, all nine software verification
   checks, real container verification and local-student endpoint smoke have
   completed. See `reports/local-study/build-cuda-simd-002/`,
   `reports/local-study/verification-009/verification.json` and
   `reports/local-study/student-serving-001/report.json`. Finish cached-model
   follow-ups and retain their actual outcomes.
2. Independently regrade retained full-context outputs under V3; reconstruct
   train/development comparisons while retaining generation failures. Do not
   open assessment held-out sources for teacher/method selection.
3. For the original assessment, supply the missing external provider inputs;
   complete acquisition, audited curation and primary/seed/degraded/overfit
   rounds using their declared settings. The supplied Hugging Face access does
   not authorise paid calls to unrelated APIs.
4. Freeze candidate/decoding/gate settings, calibrate detectability from
   development data, run complete held-out pairs and interpret real controls.
   NO SHIP is an acceptable result. Do not adjust a tolerance, threshold or seed
   after an unfavourable observation to manufacture a win.
5. Reconcile costs, verify serving/rollback as applicable, package the exact
   dependency closure, run the disconnected rehearsal and refresh the video.

The optional CUDA build dependency's original-notice gap is retained in the
[dependency inventory](dependency-notices.md); passing software QA does not
resolve a licence-provenance gap. The historical `progressive` study option is
an ordered teacher curriculum, not a teacher-assistant architecture bridge.
Method catalogue coverage, executed arms, useful generalisation, a best tested
combination and assessment promotion are separate claims. Neither a long list
of methods nor a genuine ternary checkpoint establishes all of them.

The follow-up audit also found stale executable dispatch: assessment commands
in a source checkout preferred a historical bundled CPU binary after a version
probe. The launcher now uses Cargo's current locked offline release build when
`.git` exists, with an explicit source-build override for exported source trees.
Extracted deliveries retain their prebuilt route. This fixes checkout dispatch,
not packaging: copying an old `target/release` file is not proof of current-source
build identity, and the study/native-engine dependency closure still needs a
fresh inventory and relocation rehearsal.

Gate reporting had two configuration gaps: fixed 95% confidence labels and an
80% power label even when alpha/power target differed. New reports retain and
render the actual settings, historical reports lacking confidence say so, and
empty sensitivity grids fail. The default frozen rule and thresholds are
unchanged. Complete gate artefact reconstruction now uses a blocking worker
rather than occupying the async executor. These are software corrections, not
new measured held-out or learning evidence.

## Later user requirements: additional gaps retained

| Additional requirement | Implemented / observed | Remaining boundary |
| --- | --- | --- |
| Rust and specialist agents | Responsibilities are in `agents/PLAN.md`; application, numerical training, evidence and QA are Rust. | Upstream BitNet C++ inference, NVIDIA compilation and ordinary operational/media tools are documented interoperability exceptions. |
| Hugging Face access without storing credentials | Native checksum-pinned downloader accepts ephemeral input/environment; metadata and licences are retained. | The bounded source/metadata pattern scan passed; it is not a universal secret detector. Supplied access is not a paid API budget. |
| Local GPU teacher hosting | Actual CUDA/F32 quantised teacher evaluation and CUDA/F16 dense endpoint checks exist, including driver allocations. | Explicit GPU failure never silently becomes CPU execution. Display allocations and free VRAM can change between guard and load. |
| Replace low scores with alternatives and retain the best | Original six-candidate search and the next two declared trials have real retained results, cache checks and qualification packs. Qwen3 completed at 3/16 with zero generation/grading errors; Qwen Coder 7B remains the healthy 7/16 fallback. | No teacher met the declared 80% rule on both packs. DeepSeek's 0/16 retains one length-limited generation and remains inconclusive. An exhausted finite catalogue is not a high-performing fleet. |
| Delete stale/unused/poor weights automatically | Earlier owned-cache pruning reclaimed 20,140,487,248 logical bytes; the completed Qwen3 trial reclaimed another 2,497,280,736. Provenance/deletion authority remain, and active leases, bases and selected checkpoints are protected. | Uncertain or interrupted trials are protected rather than silently classified as poor. General host/Hugging Face caches and checkpoint dependencies are not indiscriminately deleted. |
| amd64/ARM64; variable GPU, RAM and CPU | CPU affinity/cgroup discovery, explicit SIMD profiles, CUDA driver fallback and memory guards are implemented; expanded-context dispatch was fixed and tested. | Actual hardware QA is Linux amd64/NVIDIA. ARM64 execution, Apple Metal and AMD ROCm execution are not verified/implemented; unsupported accelerators currently receive an explicit CPU decision. |
| Disconnected operation and progress | Explicit offline performs no HTTP, including metadata; missing models are reported, not fabricated. Checksum/download/ETA and worker heartbeat output exist. | Pruned weights need exact authorised online rehydration for fresh inference. Existing observations remain readable offline. Disk-bound checksum/export/housekeeping work can dominate this host's progress. |
| Distillation types/combinations and efficient parallel trials | Genuine SFT, token KL, hidden-feature and attention operators, composed updates, content/feedback eligibility, native optimiser recovery and a bounded scheduler exist. Actual full-context SFT/composed and second-seed rounds ran. | The complete catalogue has not been empirically exhausted. Content/trajectory/on-policy capabilities need complete real evidence; teacher-assistant architecture transfer is not implemented by the historical curriculum option. No globally best combination or useful generalisation gain is established. |
| Genuine ternary BitNet comparison | The official-base 64-step round changed 230,999 effective trits and independently reloaded; native C++ inference ran. | Fixed-tolerance numerical equivalence and matched coding-quality ranking remain unresolved. A changed ternary checkpoint is not a validated winner. |
| Resume interrupted work | Immutable engines/plans, raw generations, prepared run identities, Adam state, launch journals and validated cache reuse are implemented and exercised. | UNKNOWN remote acceptance blocks rather than buying again. Changed protocols use new outputs; interrupted CPU content evidence remains separate from the explicit GPU follow-up. Full portable resumption needs the still-unpackaged execution closure. |
