# Verified local teacher evidence

This study is separate from the assessment's three independent API businesses.
Locally hosted open weights are genuine teacher models, but do not satisfy that
provider requirement. No local-study artifact authorises checkpoint promotion.

`src/teacher_evidence.rs` defines the typed boundary used by the study planner
and native loss worker. Its collector accepts exactly the original 48 training
tasks, representing 12 source groups. Development, qualification and held-out
inputs are rejected before their grading responses are inspected. Canonical
task, specification, oracle, source and lineage identities must all match.

For every teacher it retains the immutable original model pin, publisher-verified
asset manifest, original licence evidence and an explicit local output-training
decision. Apache-2.0 and MIT obligations remain applicable. DeepSeek output
adaptation retains the original model's derivative restrictions; API terms do
not replace those obligations. Unreviewed licences fail closed.

Every positive sequence binds its frozen comparison plan, original raw
generation, observation, independently reconstructed sandbox verdict and trusted
audit. The raw evidence must explicitly identify an actual local model
generation. References, transport fixtures, absent outputs, truncated completions,
grading errors and quarantined programmes cannot become positive examples.

Complete sequences are normalised for code deduplication within a teacher/task.
Provenance for different genuine teachers is retained rather than silently
discarded. Consumers should select teacher mixture weights explicitly rather
than treating repeated identical answers as independent evidence.

Capabilities distinguish `supported` from `available`, with a reason and
artifact links. Text may be available while logits, hidden states and attention
remain unrecorded. Verified Qwen2/Llama dense and GGUF runtimes support native
teacher-forced capture. A worker must actually acquire and bind those tensors.
Vocabulary alignment is a separate constraint for full-vocabulary KL; a generic
claim that two models both use BPE is insufficient.

The additional content collector makes new real generations:

- Process asks the teacher to write a short plan as Rust comments before its
  implementation. Both must be present and the final code must pass. This is
  observable generated process text, not proof of faithful internal reasoning.
- Context supplies a verified different task/solution from the same training
  source. The supporting example is pinned and no development or held-out
  content can be supplied.
- Tool trajectory supplies the original genuine candidate and actual trusted
  grader feedback, then asks for a revision and independently grades it. The
  feedback contains aggregate results and a run identifier, never private inputs
  or expected answers.
- Preferences contrast verified passing answers with genuine benign failing
  candidates for the same task. A separate, reconstructed eligibility proof
  permits a complete programme that exits successfully, obeys the JSON protocol,
  has no prohibited operations and fails only the host oracle to be a negative.
  Its original zero score and quarantine record remain unchanged. Compilation
  failures may supply revision feedback but cannot become preference negatives;
  security, runtime, schema, timeout and truncated failures are excluded from
  both roles. Each preference declares its common conditioning prompt and any
  transform from the actual chosen/rejected prompts, so an augmented chosen
  answer does not silently inherit a different context.

On-policy and self-distillation remain unavailable unless checkpoint-bound
student rollouts, teacher feedback and regrading are actually present. Ordinary
offline rows cannot be relabelled to enable those study arms.

Requests, original responses and grader links are durable before a final bundle
is published. Restarts reuse validated raw generations and completed records.
A changed policy requires a new explicit output directory. Reusing a completed
bundle preserves its exact bytes and original timestamp. `load_verified`
reconstructs the bundle from its original source observations, including content
and preference records; editing a convenience row or capability cannot bypass
those original bindings. Blocking model loading and hashing run outside the
asynchronous executor.

Historical bundle verification rechecks publisher metadata, original licences,
tokeniser/template identities, recording manifests and every provenance link
without rereading multi-gigabyte tensor contents. A tensor pruned by the owned
teacher-search workflow additionally requires its durable pruning proof.
Generation and teacher-forced tensor capture still use the full asset loader,
which checks all actual tensor bytes against the publisher before execution.
The distinction allows old evidence to remain inspectable without weakening
the checks on a new model run.

Example commands (select a cached registry and reserve its execution device):

```sh
cargo run --release --locked --offline --example teacher_evidence -- collect \
  --registry reports/comparative/search/round-001/trials/qwen-coder-7b-q4_k_m/registry.toml \
  --output reports/local-study/teacher-evidence --model qwen-coder-7b-q4_k_m \
  --cuda --dtype f32 --max-tokens 512 --seed 1729

cargo run --release --locked --offline --example teacher_evidence -- content \
  --bundle reports/local-study/teacher-evidence \
  --output reports/local-study/teacher-content --cuda --dtype f32 \
  --content process,context,tool_trajectory --maximum-tasks 48

cargo run --release --locked --offline --example teacher_evidence -- verify \
  reports/local-study/teacher-content
```

These are runnable command contracts, not a claim that all additional content
has already been generated. Current measured runs and coverage belong in their
actual generated manifests and reports.

Every content collection writes `content-report.json` for its current output.
The report counts requested and completed records, unavailable prompts, actual
generation/grading/infrastructure failures, empty or truncated completions,
current passing samples and current preference pairs separately from inherited
positives. Validated resumed records count towards the same batch. A batch is
healthy only when every requested slot completes without those failures and at
least one newly collected variant passes; an inherited collection of successful
answers cannot turn an empty or failed new collection into success. The report
does not infer learning from an accepted content sample.
The reusable collector returns the valid partial bundle as data; callers must
also inspect this report before declaring the collection successful. Both the
public command and the `teacher_evidence` example print the current health
report and return non-success when `healthy_complete` is false. An empty or
errored current batch therefore cannot report success because inherited
samples remain available.

## Composition boundary

Process/context/tool variants load a teacher's actual pinned adapter when one
is declared; they cannot silently substitute its base. The frozen request plan
also binds the optional adapter manifest, and replay checks its actual checkpoint
hash. Native BitNet execution must explicitly be CPU/F32. Further content variants
after student/self rollout records are currently refused: replay does not yet
implement an ordered dependency graph interleaving content and rollout records.
This is an explicit composability limit, not a base-only fallback or invented
derived sample. Independently supported non-rollout content records may compose
in their verified order, with the complete input bundle frozen by hash.
