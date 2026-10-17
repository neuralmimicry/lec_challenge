# Strict JSON round-trip discrepancy

The original SFT training report records eleven successes from forty-eight
tasks. Its exact saved pass-rate text is `0.22916666666666666`. The default
`serde_json` parser used by the old helper reparses that text as
`0.22916666666666663`, one representable F64 value lower. Strict reconstruction
therefore rejects the report even though its underlying counts are correct.

Minimal Rust reproducer against the old default parser:

```rust
let expected = 11.0_f64 / 48.0;
let bytes = serde_json::to_vec(&expected)?;
let actual: serde_json::Value = serde_json::from_slice(&bytes)?;
assert_eq!(expected.to_bits(), actual.as_f64().unwrap().to_bits());
```

Expected bits: `4597424619607381333`. Old parser result:
`4597424619607381332`. Enabling `serde_json`'s `float_roundtrip` feature makes
strict reconstruction appropriate; numerical tolerance must not conceal a
report discrepancy. The regression test should keep the exact assertion.

The [independent diagnosis](report-reconstruction-diagnosis.json) recomputes
all three training summaries from every original observation and generation.
It finds no differences for base or composed, and only SFT's root `pass_rate`
differs. Passed/error counts, token totals, generation times and all per-source
grades match. Its Rust source is
`state/diagnostics/full-quality-report-diff.rs`, with adjacent compilation
and run logs. It allocates no model and changes no original report.

The relevant original evidence is [SFT train](sft-train/report.json),
[base train](base-train/report.json) and
[composed train](composed-train/report.json). Their encoded bytes remain
immutable. A future corrected verifier must read those same artefacts.
