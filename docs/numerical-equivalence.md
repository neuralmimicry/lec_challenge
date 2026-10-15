# Numerical qualification of the ternary student

Task scores and Rust/C++ numerical equivalence are separate evidence. A complete
BitNet development evaluation remains readable when equivalence is unavailable,
but it cannot enter a method ranking or support a quality improvement claim.
Healthy conventional Rust evaluations do not require this interoperability check.

The schema-2 diagnostic freezes an immutable plan before measurements. It binds
the exact checkpoint tensor and manifest, original base and tokeniser, exported
GGUF and export receipt, native engine manifest, explicit token protocol, F32 KV
cache, immutable diagnostic executable and archived Rust implementation sources.
An implementation change makes the old qualification stale; it never rewrites
the historical measurement. Engine schema 1 retains the original F32-cache
protocol; the separately reviewed schema-2 engine uses its own quantisation
protocol and build directory.

The three fixed prompts are “Say hello in one short sentence.”, “What is two plus
three? Answer briefly.” and “Complete the Rust expression: 3_i64 + 4_i64”. Each
uses the original formatted prompt and original token IDs. The raw native
pre-sampling top 16 and the entire Rust next-token log-probability vector are
retained as separate checksum-bound artefacts. The verifier reconstructs the
highest-probability token and every compared error. All three prompts require
equal highest-probability tokens and a maximum absolute log-probability error
of at most **0.10**. A matching top token alone cannot qualify the implementation.

Verification rejects missing prompts, changed conditioning/decoding, duplicate or
out-of-vocabulary native IDs, missing vocabulary entries, non-finite or
unnormalised probabilities, changed executables/exports and stale implementation
identities. A recorded `passed: true` is checked against the raw observations.
No transformer is loaded merely to compare the retained vectors; source assets
and checkpoint manifests are independently verified.

The diagnostic accepts an explicit output, preserving older failures. Its
optional repeatable `--evaluation <DIR>` attaches that specific report to an
existing evaluation through `numerical-equivalence.json`. This attachment freezes
the evaluation plan and report checksums. It is immutable and never inferred
from an unrelated default report. A valid failed report can be attached as
failure evidence.

The library boundary is
`study_evaluate::verify_numerical_equivalence(evaluation_dir)`. It returns explicit
required/qualified status, reasons, exact receipt links and reconstructed
observations. Schema-2 selections retain excluded descriptive rows and their
reasons; an all-unqualified selection has no selected arms. The CLI reports that
unavailability as failure, and full-fidelity execution rejects the empty list.
Frozen conventional schema-1 selections retain their original reconstruction.
Legacy BitNet selections cannot bypass the new qualification boundary.

The original `reports/comparative/bitnet/parity.json` and
`parity-f32-kv.json` both failed the declared criterion and remain failed.
Schema-1 reports do not provide the complete schema-2 evidence. Three synthetic
numerical prompts, even if successful, establish neither coding quality nor a
universal equivalence theorem. Training, development quality and sealed
assessment promotion still need their separate evidence.
