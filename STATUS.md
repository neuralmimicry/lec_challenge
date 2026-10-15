# Assessment status

The design was frozen before assessment code in commit `74eba150f766a0d93c2a8ad18066daac562314cf` on 15 October 2026. Original history and the prebuild pack are preserved. Four specialist agents have implemented and executed their local work in parallel; responsibilities are recorded in agents/PLAN.md.

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

## External evidence still unavailable

No provider credentials, explicit spending ceiling or applicable account/model output-training decisions were supplied. The default roster remains disabled/unpriced. There are no real teacher recordings, approved curated provider dataset, primary/seed/degraded/overfit assessment adapters or complete real paired evaluations. No improvement, SHIP decision, real round cost per point or complete provider-trained disconnected rehearsal is claimed.

`./run.sh demo --offline` and `./run.sh drill --offline` must currently exit nonzero and identify these gaps. The actual worse/train-overfit/seed controls remain incomplete; synthetic decision fixtures remain separate. Publication and email submission have not been authorised or attempted.

## Remaining sequence

1. Configure keys locally, approve a spending ceiling and resolve model/account terms, model IDs and prices.
2. Collect all three businesses, curate passing traceable winners and train all four predeclared adapters with their original evidence.
3. Freeze the primary candidate, run development calibration and one held-out evaluation, report actual controls without searching for favourable patterns.
4. Interpret the conservative gate, measure complete round costs, finish actual production canary mechanics if SHIP, and perform the complete fresh disconnected replay.

Detailed outcomes and exact missing paths are in reports/drill.json, reports/demo.json, reports/eval.json and docs/limitations.md. Formatting, lint and tests are recorded in reports/verification.json. Packaging records belong in reports/bundle-manifest.json and deliverables/SHA256SUMS; fresh-directory execution belongs in reports/disconnected-qa.json when generated. A successful local QA report does not complete the missing provider-trained assessment.
