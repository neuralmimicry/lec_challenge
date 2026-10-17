# Verified 64-step native BitNet operator result

The separately frozen fresh 64-step protocol completed and returned exit zero.
An independent process then reloaded its original official BitNet base and
saved adapter with the same pinned Rust helper, returning exit zero and
256,512 finite logits. The prior three-step and 48-step unchanged-lattice
failures remain intact.

| Observation | Measured result |
| --- | --- |
| Genuine optimiser updates | 64 |
| Changed effective ternary q/v values | 230,999 of 245,760,000 adapted values |
| Initial effective lattice SHA-256 | `c4c7d893f70b848b82253c2336ebdf074c2a8ec17a8c56b4d31422a1cfea578a` |
| Final effective lattice SHA-256 | `46e8b97e496415c1962423e2e058c3f816451209f528a1157d9f8ed3a54021d4` |
| Adapter SHA-256 | `a7bd00bfe048d1d31dfe402558017c8e3c5e73eee1f8bab6720673e832cc19dd` |
| Matched pinned helper SHA-256 | `af675af8e54beb9ddc672d13bb860aec4175d7071b43fe9345af8c4ba9bbf182` |
| Worker wall time | 38 minutes 45.52 seconds |
| Worker user / system CPU time | 2,796.33 / 107.32 seconds |
| Worker peak RSS | 16,330,624 KiB, approximately 15.57 GiB |
| External reload wall time | 3 minutes 57.43 seconds |
| External reload peak RSS | 10,675,996 KiB, approximately 10.18 GiB |
| External reload process | 881820 |
| External synthetic probe | 256,512 finite logits |

The fresh run's saved adapter and Adam tensors at update 48 are byte-identical
to the earlier failed round, as recorded in
`native-bitnet-64-common-prefix.md`. It starts from the original base and never
imports the failed round's latent state.

Primary evidence is `native-bitnet-64-operator/manifest.json`, its immutable
request, freeze and step journals, and `native-bitnet-64-reload.json`.
The complete worker/reload logs and GNU time observations are retained alongside
them. The original helper predates the newer helper launch-receipt protocol;
its original executable and declared four-Rayon-thread operator protocol remain
the authoritative historical execution record. No new launch receipt is added
retrospectively.

These are local-teacher numerical operator and checkpoint observations. The
32-token assistant windows and two-token external reload probe establish no
whole-task coding improvement, independent API-provider provenance, native
Rust/C++ numerical equivalence or assessment promotion. No price or cost per
improvement point is inferred from CPU time alone.
