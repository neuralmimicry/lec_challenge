# JetBrains Codex implementation brief

Build the assessment in a new, small repository. Treat `design.md` as the proposed pre-build design, review it with Paul, and freeze it in a design-only first commit before implementation. Do not introduce AARNN, Gail, Kubernetes or the existing HPC platform as dependencies. Python is the default here because it gives the shortest auditable path from provider calls to PyTorch/PEFT and statistics.

This brief is an implementation specification. None of its commands or components exists yet in this pack. Never present fixtures, assumptions or example logs as completed real work.

## 0. Non-negotiable evidence rules

1. Three independent provider businesses must actually be called during development. Three model aliases behind one provider account do not satisfy that condition. At least one real completion from each is mandatory; normal task collection should exercise the complete approved roster.
2. Do not invent API responses, provider request IDs, token counts, bills, weights, elapsed times, improvements or recordings. Label deterministic fault injections and synthetic gate fixtures clearly.
3. Training must run real forward/backward/optimiser steps on curated provider outputs and produce reloadable model weights. A loss curve, configuration file or renamed base checkpoint is insufficient.
4. Every execution error, empty response, truncated generation, timeout, missing case, malformed result and missing expected artifact is failure. Expected faults count as successful *drill handling* only when a trusted assertion observes the correct failure and the report explains it.
5. A statistically inconclusive real candidate means NO SHIP. Never tune thresholds or choose seeds after seeing held-out results to obtain a win.
6. Do not print credentials. Do not send communications or publish the repository. If live provider credentials, rights clearance or spending authority is missing, finish the independent local work and report the exact blocker. No fake replacement calls.

## 1. Preflight and freeze

Inspect the real workstation and report OS/architecture, CPU/RAM, disk space, Docker availability/version, cgroup mode and resource-controller support. Inspect GPU/driver capabilities if present; a 12 GB NVIDIA card can be considered only after a working local smoke test. Keep a genuine CPU training/inference mode. Avoid dependencies that assume CUDA or a particular architecture.

Read the full assessment and local `AGENTS.md` if any. Commit `docs/design.md`, a three-page rendered PDF, and the acceptance checklist before assessment code. Record timestamp, Git commit and SHA-256. Do not backdate the design. Any later design change goes in `docs/deviations.md` with reason and impact.

Resolve three output-training-eligible providers. Initial candidates are DeepSeek, Together hosting a suitably licensed coding model, and Mistral text API. Mistral's public commercial/additional terms have been reviewed; final model/account-specific clearance is still required. Pin live model IDs, model revisions when available, price schedules and token semantics. Preserve a dated rights decision with URLs/document hashes; unresolved permission means disabled for training. Do not silently swap in Anthropic/Gemini or assume a gateway removes restrictions. Read `sources.md` for verified evidence and gaps.

## 2. Small architecture and data contracts

Use a host-side trusted Python supervisor, SQLite with WAL and durable transactions, content-addressed artifact files, provider adapters using HTTP clients, a Docker sandbox backend, PyTorch/Transformers/PEFT, a small statistics module, and a local HTTP inference service. A large agent framework is unnecessary.

Suggested layout:

```text
docs/                 design, deviations, terms decisions, reproduction notes
config/               fleet, budgets, task split, training, gate, serving
src/distil/           cli, ledger, fleet, sandbox, curate, train, evaluate, gate, serve, drills
tasks/                private-source manifests, specifications, public examples
grader_private/       controller-side oracles and hidden case generation
recordings/           sanitised real request/response/usage records
checkpoints/          real adapters and manifests; base-model reference
offline_bundle/       base/tokenizer/runtime assets and checksum manifest
tests/                focused invariants and integration tests
reports/              generated drill, evaluation, costs and limitations
run.sh                one-command entry point
README.md             prerequisites, commands, evidence, gaps
```

Core records must include:

| Record | Minimum fields |
| --- | --- |
| Task | task ID, source/lineage ID, split, specification hash, oracle/grader hash, creation/provenance, licence |
| Request | canonical input hash, provider, model/revision, parameters, sample slot, worst-case reservation, price snapshot |
| Attempt | attempt ID, request ID, timestamps, HTTP/result state, failover reason, usage, cost or uncertainty interval |
| Run | response artifact hash, sandbox policy/image digest, case manifest, verdict, trace hashes, controller reason |
| Training row | task/source IDs, split, chosen code, original run/attempt IDs, grader version, permission decision, dedup result |
| Checkpoint | base/tokenizer revisions, data hash, configuration, seed, steps, tensor hashes, reload evidence |
| Evaluation | checkpoint IDs, task/source IDs, seed, complete outcomes, costs, inference parameters, artifact hashes |

Keep external calls, candidate execution, model training and verdict computation as separate trust boundaries. The candidate never controls a database write, model-selection decision or pass/fail report.

## 3. Task corpus and oracle quality

Create 96 short Python function tasks: 48 train, 16 development, 32 sealed held-out. Use 24 independent private source/specification packs with four tasks each; assign 12/4/8 packs respectively. The number of task packs is deliberate and limits inference power. Independently authored sources should cover different practical contracts, not renamed examples from one template.

Suitable small functions include data normalisation, validation, interval operations, structured transformations and small graph/string routines. Avoid requiring third-party packages inside candidate code. These are a narrow coding distribution; do not claim general software-engineering competence.

For every task provide a precise specification, public examples, an independently written reference solution, private edge cases, generated properties and explicit constraints. Differential-check the oracle and use hand-inspected expected cases. Keep difficult/ambiguous contracts out until resolved. Store source identity before generation; descendants and mutations inherit it.

Do not derive train and held-out by changing seeds, variable names, constants or public-benchmark samples. Inspect ancestor lineage across sources. Do not send oracle source, hidden tests or expected answers to teachers. Teacher evaluation on held-out inputs is allowed after model/config freezing, but held-out solutions never enter training or prompt optimisation. Keep provider retention settings and model evaluation dates in provenance.

## 4. Durable fleet runner

Roster configuration includes provider/business ID, endpoint, API-key environment variable name, exact model ID, available revision, timeout, RPM/TPM/concurrency limits, maximum input/output/reasoning tokens, price snapshot, budget scopes, retry limits and output-training permission. Fail closed on missing pricing or uncapped billable output. Reject unknown expensive tool/search features.

Before *every* attempted API call, atomically reserve the maximum billable cost under all relevant task, model/provider and round limits. Include already reserved in-flight calls. Use decimal/integer money arithmetic. Reserve the whole billable output cap, not only expected visible code. Use conservative tokenisation where a matching tokenizer is unavailable. Set provider-side account caps when supported. Reconcile reported usage; if unexplained billing exceeds the reservation, stop rather than hiding the discrepancy.

Cache by canonical full prompt, provider, model/revision, generation parameters and intended sample slot. Different intended samples must not collapse into the same cached run. Finished calls are reused on restart. Unique database constraints and transactional claims stop concurrent workers buying the same planned completion twice.

Publish/fsync a content-addressed response blob before marking completion; recovery reconciles blobs and database state. A remote acceptance with a missing response is UNKNOWN, never automatically 'free' or complete. Keep its budget reservation, use supported idempotency/request-status mechanisms, or stop for reconciliation. Exactly-once local work is achievable; exactly-once remote billing is not generally guaranteed.

Handle 429 with Retry-After plus bounded jitter/backoff; handle transient errors separately from permanent authentication/configuration errors. Enforce rate limits on retries too. Use a bounded circuit breaker and configured failover to another eligible provider. Record every attempted call, not just the successful result. Do not silently change provider/model identity in provenance.

Record real raw responses with secret headers removed, plus normalised completions, token usage, request IDs, pricing references and hashes. Replay transport matches exact request keys, has no network fallback, and fails loudly on a missing recording. Replayed faults must be labelled recorded real faults or deterministic injected faults. Normal production and replay share orchestration code.

## 5. Sandbox and grading boundary

Create a fresh container per candidate run. Baseline controls: non-root UID/GID, read-only root filesystem, `--network none`, all capabilities dropped, no-new-privileges, seccomp, PID limit, memory and swap limits, CPU limit, bounded tmpfs, private PID namespace and a host-enforced wall-clock timeout that kills the whole container. Mount only candidate source read-only and bounded scratch. Mount no host repository, credentials, hidden-test directory, home directory or Docker socket into that container.

Docker defaults alone are insufficient. Verify effective cgroups and settings, then prove memory pressure, runaway children and timeout terminate correctly. Unsupported control enforcement is a preflight failure. Runtime/model downloads occur outside candidate execution and only in an explicit bootstrap mode.

The trusted host controller keeps test implementation and expected answers. It sends bounded JSON inputs and validates typed responses against its own oracle. Do not import the test suite into the candidate's interpreter, mount hidden tests read-only into its container, accept candidate-authored JUnit reports, or treat exit zero as a pass. Read-only files are still readable. Hidden test *inputs* must be visible to solve the task; test code, oracle answers and control messages remain outside.

Supervise denied operations from the trusted host, using a verified syscall-tracing/audit path. Deny candidate networking at syscall level in addition to network isolation; no-network namespaces alone still contain loopback. Associate trace events with the run and record syscall/target/result without secrets. If required tracing cannot run on the actual platform, report the limitation and do not mark the escape drill complete. Candidate self-reports of failed attempts are insufficient.

Test network probes (including loopback), a known host-only sentinel path and the actual hidden-test location, plus root-filesystem writes. Check that all attempts fail and appear in trusted logs. A caught EPERM/ENOENT must not hide the attempt from quarantine. Test fake success JSON, `sys.exit(0)`, empty output, known-example hardcoding, test skipping, oversized output and hung execution. A finite test set cannot prove the absence of all hardcoding; show the specified attack fails and explain the residual limitation.

## 6. Curation

Run the curator only over training-source eligible runs. Require a fully passing controller verdict, a complete trace, approved usage rights, valid response schema and no quarantine reason. Select the best eligible candidate per task; keep rejected runs for audit. If nobody passes, keep the task in the repair queue without inventing a positive training example.

Check exact prompt/code hashes, canonicalised AST/token similarity, source lineage and near-duplicate candidates against the complete held-out specification index. Exact hashes alone miss reworded contamination. A similarity hit pauses the row for review; keep false-positive decisions auditable. No metadata field or dataset filename can override an identified held-out origin.

Before opening an optimiser, validate the complete dataset against its manifest and immutable held-out index. A contaminated row aborts training, not merely a warning after training. Save JSONL with real task-to-run provenance and a signed/hash manifest; no hidden expected answers in the SFT text. Verify loss masks cover assistant code only and include nonzero learnable tokens.

## 7. Real training round

Use `Qwen/Qwen2.5-Coder-0.5B-Instruct` at an actual resolved revision with its original tokenizer and chat template. Implement LoRA SFT with a frozen base, rank 8 as the initial development setting, small batches, capped sequences and explicit assistant-only labels. Pin dependency releases that work in the observed environment. Keep CPU float32 support; choose GPU dtype after a real hardware check.

Train on the curated winners, using development outcomes for any schedule selection. Record effective example count after filtering; 48 train tasks do not guarantee 48 passing samples. Assert finite loss, nonzero gradients, optimiser steps and a changed adapter hash. Save adapter safetensors/configuration, optimiser/resume state, seed, data/base hashes, logs and resource measurements. Reload a fresh process and generate actual code. Measure pre/post behaviour; changed weights alone are not proof of learning.

Run a second real adapter with the same data/hyperparameters and a different training seed for the seed-control comparison. Fix both training seeds before test access. Keep the base checkpoint as the incumbent. Do not suppress the real gate result if both adapters fail to improve.

## 8. Evaluation, gate and power

Implement a command of the form `./run.sh gate --incumbent <manifest> --candidate <manifest>` that evaluates both actual checkpoints or consumes verified complete cached evaluations keyed to their tensor hashes. It must never accept an unverified aggregate accuracy file as proof.

Freeze candidate choice, source weighting, evaluation seeds, primary metric, confidence method, alpha, +2 percentage-point practical margin, non-regression slices and resource ceilings before held-out access. Use three common decoding seeds with identical parameters for both checkpoints. Preserve all failed generations in the denominator. A task with 100 assertions is still one task; three seeds are repeats, not three independent tasks.

For each source, average paired per-task seed success differences, then average sources equally. Produce the predeclared two-sided 95% source-level t interval and a source-cluster bootstrap sensitivity interval. The small eight-source sample makes both approximations fragile; state assumptions and show sensitivity. Require both lower bounds greater than +2 points, complete data and passing security/operational checks. Any invalid evaluation blocks. A train gain cannot rescue a held-out loss.

The report must distinguish SHIP, NO SHIP / REGRESSION, NO SHIP / INCONCLUSIVE and INVALID EVALUATION. Operationally all except SHIP keep the incumbent. Interval overlap with zero is insufficient evidence of superiority; it does not prove equality.

Estimate MDE at 80% power with simulations of the *actual* gate, preserving the number of source clusters, shared task/seed structure and plausible discordance. Calibrate assumptions from development data before test access; report a sensitivity grid. State simulation RNG seed, trials, Monte Carlo error and whether any tested effect reached 80% power. If not, report 'MDE above tested range', never a fabricated scalar. The 27.1-point normal approximation in the design is only a sanity check for 32 independent task pairs with discordance 0.30; it omits clustering and the shipping margin.

Use focused statistical fixtures to verify direction, boundary cases, identical scores, no discordance, missing outputs and deterministic bootstrap reproducibility. Separately evaluate the real base/candidate/seed-control weights. Produce an actual deliberately degraded checkpoint and a train-overfit control, for example through incorrect-target training and a separately overtrained adapter. Develop controls using train/development only and freeze them before test access. Report their true measured outcomes; do not assume that either manipulation guarantees a particular regression. A score fixture proves decision logic, not actual checkpoint behaviour. If a real control does not exhibit the required pattern, mark that drill incomplete, retain the fixture evidence and explain the limitation. Never search held-out control variants until a desired outcome appears.

A seed-only checkpoint pair is not guaranteed to be statistically indistinguishable: seed variation can cause a real difference and false positives occur at the chosen alpha. Pre-register the seeds, show the actual result and include a deterministic null fixture for the required no-win behaviour. Never search held-out seeds until a desired conclusion appears. If the real pair is significant, disclose that and investigate using fresh data rather than relabelling it.

Do not repeatedly reuse the sealed set as a tuning dashboard. Freeze one primary candidate; any later confirmatory promotion requires fresh independent sources or a documented predeclared sequential/multiple-comparison design.

## 9. Local serving and rollback

Expose a localhost HTTP endpoint with `/health`, checkpoint metadata and a bounded inference route. Return real model-generated code. Load pinned base/tokenizer plus the selected adapter; health includes a real inference smoke check and checksum validation. Bound request size, queue length, concurrency, output tokens and timeout.

Keep incumbent and candidate as immutable versioned processes. The routing manifest controls new requests atomically; checkpoint IDs appear in every response/log. Stage and health-check before a 5% eligible-traffic canary. A NO SHIP adapter is available only on a clearly identified evaluation route; the production pointer stays on the incumbent.

Demonstrate promotion/rollback with local controlled traffic and operational fault injection. Roll back on a security event or predeclared error threshold; drain existing requests. Low-volume canary latency/errors demonstrate mechanics, not statistically established coding quality. Retain previous weights/configuration and a repeatable rollback command.

## 10. Offline packaging and the one command

Required final entry point: `./run.sh demo --offline`.

It must check prerequisites and checksums, load bundled runtime/container images, run the real recorded fleet through replay, regrade in fresh candidate containers, curate the resulting data, exercise the real CPU-capable training path, reload/evaluate checkpoints, run the gate, start/smoke-check local serving, execute every drill and emit human-readable plus JSON reports. Keep output live and bounded with phase progress and durations. Save partial state on interruption. No silent skipped phases; missing recordings/weights/images is failure.

Do a real CPU timing test early. If the complete retraining demo is too slow for review, provide a documented `--use-recorded-checkpoint` fast mode plus the full training command and original evidence. Label that mode's skipped training explicitly; never describe it as a training run. The default/full command must still execute genuine training if claiming that behaviour.

Define the machine contract honestly: clean supported Linux architecture with Docker Engine and enforced resource controllers, plus measured RAM/disk/time requirements. If 'clean machine' includes no Docker, provide/document an explicit prerequisite bootstrap rather than assuming installation rights. Separate `bootstrap-online` from offline replay. Bundle OCI image archives, base weights, tokenizer, adapters and all dependency/runtime assets for genuine network-disconnected execution; a first-run Hub or package download is not offline. Record asset licences and checksums. Expect a multi-GB bundle and choose repository plus release asset or zip accordingly.

Run a fresh rehearsal with provider keys absent and outbound access disabled after assets are present. Validate the entire process from a newly unpacked directory and an empty application cache, not only on the development workspace. Keep the candidate containers disconnected in both live and replay modes.

## 11. Drill contract

One command, `./run.sh drill --offline`, runs all cases, prints a concise outcome per case and writes JSON plus Markdown. A caught exception is not enough: each case must assert observable state/evidence and return failure if the asserted condition is missing. Make the drill resumable and deterministic where applicable.

| Drill | Required evidence |
| --- | --- |
| Kill orchestrator mid-round | Real child orchestrator receives SIGKILL after known responses are durable; restart resumes; finished request keys have no new transport calls/charges. Separately show UNKNOWN handling for the acceptance-before-persistence window. |
| Provider error then rate limit | Inject 500 then 429/Retry-After in a labelled transport wrapper; production retry/rate-limit/failover code completes remaining work through another eligible provider's real recordings. Log provider identities and budget effects. |
| Grader gaming | Real candidate programs skip work, fake success and hardcode public answers; controller scores zero and quarantine is persisted with reason. |
| Escape attempts | Real candidate tries network, host-only file and hidden-test path; all fail; trusted audit traces show the attempts and quarantine. Merely checking strings in generated code fails this drill. |
| Planted held-out task | Add a known held-out-origin row and a reworded/lineage duplicate to a temporary training artifact; curator/training preflight rejects them before any optimiser step. Do not contaminate the canonical dataset. |
| Worse checkpoint | The actual deliberately degraded checkpoint has a negative measured held-out delta and is blocked. A separate score fixture validates decision logic; it cannot alone complete this checkpoint drill. |
| Train gain, held-out loss | The actual overfit control improves on train, loses on held-out and is blocked. Report the observed pattern honestly; a matching fixture alone does not complete this drill. |
| Seed-only pair | Predeclared real seed-control adapters evaluated honestly and reported as no significant difference when supported. A separate deterministic null fixture demonstrates no-win logic; it cannot replace the real pair. Report the seed-only statistical caveat. |
| Budget exceeded | Set remaining budget below the next worst-case reservation; assert no transport call occurs, progress is durable and a clear budget-stop result is emitted. |

Also verify a deliberately unhandled infrastructure exception gives a nonzero command exit, a failure report and no promoted checkpoint. Expected drill budget stops can have a successful top-level *drill assertion* result; a normal incomplete live round cannot masquerade as completed work.

## 12. Measured costs and final reports

Generate costs from the attempt ledger, not README constants. For each task/model/round show input, cached input where applicable, visible/billable reasoning output, retries, historical USD, credits/cash treatment and any unknown-cost interval. Attribute evaluations, failed generations, GPU/CPU hours, electricity/rental assumptions, storage and serving separately. Free GPU time is zero cash, not zero compute; show both.

Report API historical acquisition cost separately from replay's zero new API spend and its real current compute. Include training and gate evaluation in the round total. Compute descriptive cost per point as `round_total / (100 * (candidate_rate - incumbent_rate))`; the denominator is percentage points, not a fractional rate. Non-positive gain is N/A; inconclusive positive gain is descriptive only, with proven cost per point unavailable. Give amortised cost per verified served task and the break-even volume if quality comparability is established.

Required output artifacts: `drill.json/md`, `eval.json/md`, `costs.csv/md`, dataset and checkpoint manifests, original training logs, limitations and reproduction notes. Do not fill absent measurements with the design's assumed rates. Keep the raw evidence traceable from summary to run, request and response.

## 13. Five-minute video and cold walkthrough

Suggested video script (maximum five minutes):

| Time | Show |
| --- | --- |
| 0:00-0:40 | Design ceiling, what the student is expected to learn, and limits of the claim |
| 0:40-1:25 | One command, real-recording provenance, configured provider identities and task split |
| 1:25-2:25 | Crash/resume, 429 failover, sandbox probes and contamination rejection |
| 2:25-3:25 | Actual training log, changed/reloaded adapter and the real held-out gate result |
| 3:25-4:20 | Negative/null gate controls, interval, MDE assumptions and honest NO SHIP if applicable |
| 4:20-5:00 | Real local endpoint, rollback, measured cost and unfinished items |

Paul must be able to explain: the budget transaction and unknown billing window; why source lineage matters; why hidden tests cannot share the candidate interpreter; the actual labels used by SFT; the gate's unit of independence; why more decoding seeds do not enlarge the independent test corpus; the difference between a fixture and real checkpoint evidence; and the limits of 'better than the teachers'.

Do not create an implementation Paul cannot explain. Keep modules small and comments tied to invariants. Prefer a completed auditable vertical slice over an unfinished platform.

## 14. Ordered milestones

1. Freeze design, machine/provider preflight and rights decisions. Resolve any blocker early.
2. Implement and verify sandbox/controller, task/oracle corpus and source partition.
3. Implement transactional fleet/replay, collect actual calls, and prove restart/failover/budgets.
4. Curate and audit the real training set; train/reload two adapters.
5. Freeze candidate/config, evaluate held-out once and publish the honest gate/MDE report.
6. Serve, canary/rollback, complete the drill and disconnected fresh-directory rehearsal.
7. Record the short video; package a repository/zip with exact reproduction instructions and explicit gaps.

At each milestone update `STATUS.md` with completed evidence, measured facts, failures and next work. If time becomes constrained, keep the real checkpoint and honest gate; reduce optional interaction/UI work. Never replace a required real component with an undisclosed stub.
