# Complete-context checkpoint coding measurements

Status: all six evaluations are complete, with all **192/192** original
observations retained. This is a local exploratory study. No sealed
held-out task is used and no checkpoint is eligible for assessment promotion.

The original Qwen2.5-Coder-0.5B-Instruct base is compared with the actual
48-step SFT and composed adapters. Both adapters trained on the same seventeen
complete, verified examples, excluding the same overlength example, in the
same frozen order. All 540,672 finite adapter values changed in each arm and
both checkpoints passed separate-process reload checks. Those facts establish
real optimisation and reload, not better coding.

The [frozen protocol](requested-protocol.json) declares all 192 observations
before generation: sixteen development and forty-eight train tasks for each
of the three checkpoints. All evaluations use the exact same application
`f5d2f2f5…`, CUDA device 0, F32, four assigned CPUs/numerical threads, seed 1729,
temperature 0.2, top-p 0.95, a 512-token output cap, a 120-second generation
deadline and three effective concurrent isolated graders. The checkpoint
identities were separately bound before their first generation:

| Checkpoint | Actual tensor SHA-256 |
| --- | --- |
| Unadapted base | `f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44` |
| SFT, 48 steps | `e2bcd135527659042304d936fa56c7240f6f96a43b9f81f2a888c38b7762a949` |
| Composed, 48 steps | `2217a994f8b74ded1d52f30ef7add2cb190578e3d108168a85d1409160ad88c3` |

Development results:

| Checkpoint | Observations | Recorded passes | Generation errors | Grading infrastructure errors | Healthy complete run |
| --- | ---: | ---: | ---: | ---: | --- |
| Base | 16/16 | 0 | 0 | 1 | No |
| SFT | 16/16 | 0 | 0 | 3 | No |
| Composed | 16/16 | 0 | 0 | 9 | No |

The [independently reconstructed comparison](development-comparison.json)
correctly reports **`invalid_quality_comparison`** and leaves both improvement
estimates null. These observations cannot establish the models' complete
coding quality or a quality gain. Every original failure remains in its
denominator and each application exited 1. Complete generations remain
available for a separately frozen regrade without regenerating model output.

Training-split results:

| Checkpoint | Observations | Recorded passes | Generation errors | Grading infrastructure errors | Healthy complete run |
| --- | ---: | ---: | ---: | ---: | --- |
| Base | 48/48 | 0 | 2 | 1 | No |
| SFT | 48/48 | 11 | 0 | 0 | Yes |
| Composed | 48/48 | 6 | 0 | 1 | No |

SFT's eleven passes are actual executable training-task grades. They do not
establish a development gain or generalisation. The base's two incomplete
512-token outputs (`text.lower_trim`, `paths.extension`) remain failures even
after any repaired grading. The old train-comparison helper also exposed a
one-ULP JSON parsing discrepancy for the exact `11/48` rate. The independent
[arithmetic diagnosis](report-reconstruction-diagnosis.json) confirms that all
counts, token totals, generation times and source results agree with the
original artefacts; only that reparsed floating-point value differs. See the
[strict reproducer](report-reconstruction-diagnosis.md). A corrected parser
must verify the original bytes; neither tolerance relaxation nor rewriting
the original report is permitted.

The base timeout investigation found that container startup/removal consumed
the host wall-clock allowance after the generated program had already exited;
cancellation discarded the unconsumed audit stream. Later composed grades
also encountered image/owned-container inspection timeouts. The original
controller version 2 reports, raw generations, audits, exceptions and timing
logs are preserved. A repaired grading protocol must have its own identity
and cannot silently replace these observations.

Each `*-protocol-verification.json` separately verifies the actual task order,
checkpoint/raw-generation identities, application, runtime, decoding,
sandbox image/policy, recorded result and process exit against the preregistered
request. Passing this identity/hash audit does not make an unhealthy grading
run healthy. The current-build Rust comparison helper separately reconstructs
trusted grader evidence and report arithmetic. Its exact source, prebuilt
library, dependencies and executable are recorded under
`state/diagnostics/native-full-quality-comparison/build-002/receipt.json`.

GNU time, stdout and stderr for each process are retained as
`state/diagnostics/native-full-quality-<arm>-<split>.*`. These are actual local
wall/RSS/CPU observations, not energy bills or provider expenditure. Cash and
energy costs are unmeasured; cost per improvement point is unavailable because
there is no valid measured improvement. Local model developers do not count
as three independent real API-provider businesses.

| Process | Actual wall time | Process exit |
| --- | ---: | ---: |
| Base development | 3:25.13 | 1 |
| SFT development | 4:12.26 | 1 |
| Composed development | 6:05.52 | 1 |
| Base train | 8:44.78 | 1 |
| SFT train | 7:36.79 | 0 |
| Composed train | 10:06.89 | 1 |

All processes were reaped after the final evaluation. A separate GPU query
showed 132 MiB / 8192 MiB in use, and GPU ownership was explicitly released.
The separately frozen cached-cgroup termination fixture subsequently passed;
its own result is `state/diagnostics/cgroup-gated-002/result.json`. It does not
repair these original coding grades or validate a new production controller.
