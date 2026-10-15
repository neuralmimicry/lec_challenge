# Candidate isolation, grading and corpus evidence

Candidates are Rust functions with the signature `pub fn solve(input: serde_json::Value) -> serde_json::Value`. They use only the standard library and `serde_json`. The controller adds a fixed JSON-lines wrapper. Each input requires one JSON value; an exit code, a self-authored test report or a printed success message cannot award a pass. Failed, missing, malformed, timed-out and excessive responses remain failures. The task score is binary: partial case counts are diagnostic, and any prohibited operation makes the score zero.

The trusted host retains the oracle, case generation, expected values and complete denominator. It mounts one temporary directory containing candidate source, read-only. It never mounts the repository, credentials, home, container socket or `grader_private`. Private **inputs** go down the candidate's stdin; the oracle and expected answers do not. Rust compilation also happens inside the container, so `include!` and the linker do not run against the host filesystem. Cargo and package downloads are restricted to explicit online image bootstrap.

Every candidate gets a fresh container, UID/GID 65532, read-only root, private PID/IPC/cgroup namespaces, no network, no capabilities, no-new-privileges, an explicit native seccomp allowlist, bounded temporary storage, PID/memory/swap/CPU limits and a host deadline. The base OCI digest and dependency lockfile are pinned. The immutable built image ID is frozen before evaluation; a changed image or security policy invalidates cached evidence. Original assessment evidence uses Linux x86_64 policy V1. Portable V2 adds an explicit Linux AArch64 implementation, whose hardware enforcement is unverified until its native probes pass on an ARM machine. The workstation's Docker CLI reaches rootless Podman; the application uses native `podman`, and checks effective cgroup files rather than accepting Docker compatibility feature flags as proof.

`sandbox/audit.rs` is a Rust supervisor built into the image from trusted repository source. It starts the generated binary under `ptrace` before its first instruction. The child has dedicated stdin/stdout/stderr pipes, and never inherits the supervisor's report channel. The supervisor makes itself non-dumpable, rejects process signalling/tracing/descriptor theft, refuses descendants and asynchronous syscall submission, blocks path aliases and records prohibited network/filesystem attempts before allowing the candidate to resume. Its report becomes a host-owned, hashed audit artifact. This is a trusted controller inside the container, rather than an external kernel-wide audit service; that implementation distinction matters. A malicious candidate killing the supervisor produces an infrastructure failure, never a valid pass.

The network namespace is supplemented by seccomp and tracing, so loopback socket creation is denied too. The filesystem policy permits only the dynamic-loader files and a few runtime paths needed by a pure function. `..`, symbolic links, hard links, process-memory access and alternative syscall ABIs cannot bypass the path rules. The container root and mount policy provide a second boundary. These controls are defence in depth on a shared kernel, not a claim of protection against every kernel exploit.

Preflight reads the actual UID, `memory.max`, `memory.swap.max`, `cpu.max` and `pids.max`, checks an EROFS response on a world-writable image file and inspects the route table, then performs finite memory-pressure and child-pressure probes. The deadline probe first waits for a trusted started marker, then kills and removes the running container. Where delegated cgroup v2 control is available, the host writes the exact inspected container's `cgroup.kill` and observes the group becoming empty. Paths must end in that full container ID; a root or ancestor group is rejected. The fallback sends explicit runtime SIGKILL, with bounded acknowledgement, before removal. One actual hung-programme run observed its 20,000 ms deadline, acknowledged the whole-group kill in 42 ms and completed removal bookkeeping in a further 5,299 ms. These are separate measurements: cleanup time is not candidate execution time. Finally a real sum candidate must compile, execute against public/private cases and finish under the trusted trace. A missing controller or unsupported control is a preflight failure.

The gaming drill executes early-exit, fake-success and public-answer-hardcoding candidates. The escape drill executes loopback networking, a known host-only sentinel, the actual held-out index path and a root write. Assertions require trusted denied-syscall events and persisted quarantine. A caught exception inside generated code is insufficient. Raw audit output is replay-verified against the host oracle; an altered aggregate pass count is rejected.

The corpus contains 96 precise contracts, four per each of 24 local source/lineage packs, split 48/16/32 by complete pack. These are synthetic contracts written during this assessment by a shared AI author. They do not constitute 24 independently human-authored sources; statistical independence and novelty are weaker than the ideal in the brief. Public expected examples are explicit literals and are checked against executable host oracles. Private cases cover boundaries, and the integer ledger adds deterministic generated cases. Independent interval-coverage and fraction-invariant tests provide additional oracle checks. The other oracles still need independent human review and more broad property testing before a strong quality claim. No private reference or local example is silently re-labelled as a provider recording.

Every specification publishes shared defaults: signed 64-bit integers with representable arithmetic, arrays of at most 64 items and ASCII strings of at most 256 bytes. Transformations that expand their output have explicit 128/256/257-item exceptions. A host-side domain validator checks all public/private inputs before oracle indexing, including graph edge uniqueness, matrix shape, valid-date preconditions, positive sizes, monetary limits and rational denominators; expected values are checked against the shared representation too. The final audit corrected one private monetary input that exceeded its stated bound by one cent. Full shared text participates in specification, prompt and cache hashes, while lexical contamination screening removes that common protocol paragraph so it cannot manufacture apparent source similarity.

Curation looks up immutable task/source/lineage metadata, checks exact and near specification matches against the complete held-out index, parses candidate Rust and compares canonical AST token streams, and quarantines near duplicates for review. Similarity screening is conservative and can reject useful samples; it cannot prove semantic non-equivalence. Real provider identity, canonical request keys, model revision, rights decision and the exact returned code must agree with recording artifacts. A portable attempt proof exported from the actual ledger binds the completed attempt ID, request/source/provider/model, complete body, response artifact, usage, original price and settled cost. It regrades complete trusted audit evidence, copies all provenance into the dataset, and hashes the manifest. Eligible winners are chosen deterministically by measured cost, code length and stable request identifiers, then ordered by source/task. Immediately before optimisation, it rechecks every byte, source, prompt, answer, rights decision, attempt proof, raw audit and the immutable held-out index. A held-out origin cannot be excused by renaming fields or moving a file.

Use `./run.sh sandbox-preflight`, `./run.sh grade --task ledger.sum --code sandbox/examples/sum.rs`, and `./run.sh drill --offline`. A failed assessment drill command can still contain passing local security assertions when required real-provider or checkpoint evidence is absent; consult the structured report rather than treating a partial run as the completed assessment.

## Portable policy V2 and retained V1 evidence

The original controller, supervisor, cgroup controller, seccomp policy and
Containerfile are retained byte-for-byte under `sandbox/policies/v1/`, with
their hashes in `manifest.json`. The original default policy hash remains
`0bf995b5f69865792503015f6af52b457c861b679877956e939c2137ddb4a562`.
An original report without version/architecture fields still deserialises as
V1; those fields are omitted when serialising V1 configurations. Existing
reports and the original image are not regenerated by adding portable support.

The host controller has a separate version from the in-container syscall policy.
The current default uses controller V2, including on the original x86-64 V1
image; deserialising an old configuration without that field still selects
historical controller V1. Historical controller V1 is verification-only. Its
immutable bytes remain in `sandbox/policies/v1/`; new executions require the
current supervision. Controller V2's exact source, lease, operation owner and
recovery bytes are frozen under `sandbox/policies/controller-v2/`. The portable
policy's image recipe, dependency lock, trusted probe and ABI source bytes are
separately frozen under `sandbox/policies/v2/`. New builds reject mutable inputs
that differ from those snapshots, and a source-equality test catches accidental
host-controller drift. Behavioural changes require a new explicit version;
historical reports continue to use their own frozen verifier inputs.

`SandboxConfig::portable()` explicitly selects V2 on Linux x86-64; it is the
default configuration on Linux AArch64. V2 binds its native ABI, decoder,
register mutator, supervisor, controller, Containerfile and seccomp bytes to
its own policy hash. Its supervisor requires `PTRACE_GET_SYSCALL_INFO` (Linux
5.3+) and checks the kernel-reported native audit architecture at every stop.
Syscall arguments and pathname positions are decoded separately for each ABI;
both rename operands are checked. x32/compatibility ABIs fail closed. x86-64
denials replace `orig_rax` and return `rax=-EPERM`. AArch64 denials use
`NT_ARM_SYSTEM_CALL=-1`, then `NT_PRSTATUS` to return `x0=-EPERM`; changing x8
alone would not suppress dispatch. A trace decoding or register update error
cannot produce a successful grade.

V2's pinned official Rust OCI index is
`sha256:114c7a4425406451c2866b6aafe69fe29b1b298832db1277d411ac73c82d04d6`.
Exact publisher bytes are in `sandbox/rust-v2-index.json`; its x86-64 manifest
is the original V1 base, and its ARM64 manifest is
`sha256:e393e07b3c3c39f238de676b0eb60c03dc7aefe5d6e37be6d29c686bf4d7e256`.
The image is built natively; no emulated execution is counted as native ARM
verification. Each platform needs its own built image ID and archived image.
Offline bundle restoration rejects a mismatched policy/architecture before
invoking the runtime. Legacy archives without these fields mean V1/x86-64.

```sh
# Explicit online public-image bootstrap plus native enforcement QA.
cargo run --locked --offline --example sandbox_portable_qa -- --build

# Reuse the already-built native V2 image without downloading anything.
cargo run --locked --offline --example sandbox_portable_qa

# Compile/lint the actual AArch64 supervisor without claiming hardware tests.
rustup target add aarch64-unknown-linux-gnu
cargo clippy --locked --offline \
  --manifest-path sandbox/portable-check/Cargo.toml \
  --target aarch64-unknown-linux-gnu -- -D warnings

# Package a previously exported native V2 OCI image with its platform binding.
cargo run --release --locked --offline --example package_sandbox -- \
  /path/to/native-sandbox.tar IMMUTABLE_IMAGE_ID --portable
```

The native QA writes `reports/sandbox-portable-qa/report.json`, including the
actual architecture, enforcement preflight, security and gaming runs. An x86-64
result always records `arm_hardware_verified=false`. These are container QA
results, not checkpoint-quality or provider-assessment evidence.

Kernel pressure and deadline preflight deliberately stress the runtime and must
not overlap other candidate containers. The controller coordinates native
processes through private UID-scoped runtime lock files under `/tmp`; executable
aliases resolve to the same canonical identity. Ordinary grades take shared
leases and remain parallel. Preflight, image building and image restoration take
exclusive leases, with an arrival turnstile so new grades cannot starve a waiting
preflight. This turnstile is not a strict FIFO guarantee. Filesystem preparation
runs on a blocking worker; contested locks use
non-blocking `flock`, cancellable Tokio waits, a ten-minute bound and progress
messages. File closure releases leases on process exit. Earlier pinned
executables without this implementation must still be serialised manually.

Once dispatch starts, an owned asynchronous task retains its runtime lease
through bounded execution and cleanup even when its requesting future is
cancelled. Every candidate and trusted probe is planned in a private, fsynced
ownership journal before runtime dispatch. Records bind the canonical runtime,
workspace, UID, exact name/image, random labels, PID start time and boot identity.
Recovery skips live owners, verifies all recorded labels and targets only the
immutable container ID. An arbitrary inspection error never proves absence.
Each recovery claim is non-blocking and scoped to one record, so concurrent
ordinary grades retain parallelism. Completed records remain durable; journal
archival is a future performance improvement.

Task ownership cannot survive an orchestrator SIGKILL. Native Podman therefore
also enforces an independent `--timeout` ceiling, in addition to the host's
deadline and explicit kill/cleanup. Unsupported runtime semantics fail rather
than silently discarding this guarantee. `sandbox_crash_qa` deliberately kills
an actual holder after observing its infinite candidate, requires runtime
removal before invoking recovery, then checks a real passing grade and the
durable acknowledgement. A separately labelled sentinel must survive unchanged.
This is a local sandbox crash test, not evidence of paid-provider recovery.

```sh
cargo run --locked --offline --example sandbox_crash_qa -- \
  --output reports/sandbox-crash-qa/new-round --wall-timeout-secs 30
```

The crash QA accepts an explicit 1–60-second ceiling, defaulting to 20 seconds;
the actual declared value binds its configuration and policy hash. It does not
change the frozen controller or relabel earlier observations. The initial two
rounds used a stricter 15-second ceiling and failed before the SIGKILL phase.
The first trusted PID probe exited with code zero after 1.707 seconds of native
execution, but removal took a further 14.931 seconds; the host's attached-client
deadline expired while Podman retained its pipes. The second round's trusted
reference also exited with code zero in runtime events but exceeded its host
deadline during startup/execution/removal. Both remain failures, with zero
accepted cases and no complete audit. Raw lifecycle observations are retained
in `reports/sandbox-crash-qa/runtime-events.jsonl`. A subsequent explicitly
declared 30-second QA run must establish its own independent runtime ceiling
and complete evidence; successful runtime exit alone cannot repair old reports.

Round 003 declared 30 seconds and reached the actual SIGKILL exercise. Its core
report records all isolation/recovery assertions passing, including a real
post-crash reference and an unchanged unrelated sentinel. The overall command
then failed because final trusted-sentinel removal exceeded its ten-second
client deadline. `round-003/command-result.json` explicitly records this overall
failure; final `report.json` records that failed command too, while the exact
original successful core payload is preserved in `exercise-report.json`.
The corrected QA protocol separately retains `exercise-report.json`,
sends explicit SIGKILL to its owned trusted fixture, bounds removal bookkeeping
separately and publishes final success only after independently confirming
absence. It requires a fresh run and does not alter controller V2's frozen bytes.

The first V2 full preflight attempt overlapped other real container checks and
timed out on its reference candidate; its failed report and run remain retained
under `reports/sandbox-portable-qa/`. A subsequent direct same-image/same-policy
sum run exited successfully with 63 native traced syscalls and no prohibited
events. This established that the decoder worked, while leaving the failed
concurrent preflight as failed evidence.

The completed native x86-64 V2 QA now passes in
`reports/sandbox-portable-qa/report.json`, binding policy
`a476342070dbeab34857f25dcd6e4321d0b2cc9017db49a30c7904c1ad750e44`
and image `edff3af23b630f90eb02e4f1830fd2eca68ce88301d96776c65d3696a76af8ce`.
It verifies UID 65532, read-only root, no network, exact resource controllers,
OOM termination, 63 children followed by PID denial, an acknowledged whole-group
deadline kill, the passing reference, trusted denied-syscall escape events and
zero final scores/quarantine for all three gaming candidates. Its deadline probe
waited 1,000 ms, acknowledged cgroup kill in 101 ms and recorded a further
2,722 ms of removal bookkeeping. ARM hardware remains unverified.

The corrected complete SIGKILL command passes in
`reports/sandbox-crash-qa/round-004/report.json`, with the actual holder killed
by signal 9, independent runtime removal before recovery, durable dead-owner
journal acknowledgement, a passing post-crash reference, unrelated sentinel
survival and verified final sentinel cleanup. Its declared runtime ceiling is
30 seconds; observed SIGKILL-to-absence latency is 35.284 seconds, including
runtime removal housekeeping. Raw runtime lifecycle events are retained beside
the report. Neither completed local QA result is a paid-provider recovery drill
or checkpoint-learning claim, and the earlier failed rounds remain unchanged.
