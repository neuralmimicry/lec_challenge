# Completed conventional full-context rounds

Both preregistered native Rust rounds completed 48 genuine optimiser updates,
saved changed adapters and Adam state, and passed independent-process CPU/F32
reloads. This establishes completed adaptation and usable checkpoint artifacts.
Coding improvement remains a separate matched evaluation; these local-teacher
artifacts cannot authorise assessment promotion.

The [original protocol](native-full-round-protocol.md), both immutable requests,
failed earlier observations and pinned executables are retained. The actual
[matched selection audit](native-full-matched-selection.json) verifies identical
complete encoded examples, assistant masks, order, exclusions and launch
contracts. Of 18 passing original Qwen 7B examples, 17 were accepted across 11
training source groups. `calendar.ordinal` required 541 tokens including EOS and
was excluded under the fixed 512-token limit. No example was truncated.

Both arms used the original Qwen2.5-Coder-0.5B-Instruct base, seed 1729, rank 8,
alpha 16 and learning rate 0.0002. SFT used sequence supervision alone. The
composed arm used SFT, temperature-two token KL, hidden-feature transfer and
attention relations, each with declared weight one. All four components were
observed at every composed update. Component objective magnitudes are not
comparable measures of coding quality.

| Measured observation | SFT | Composed |
| --- | ---: | ---: |
| Completed updates | 48 | 48 |
| Finite adapter tensors | 96 | 96 |
| Changed F32 adapter values | 540,672 / 540,672 | 540,672 / 540,672 |
| Minimum observed gradient norm | 0.932394 | 0.534449 |
| Worker wall time | 11:48.33 | 16:22.66 |
| Worker peak RSS, KiB | 7,676,504 | 10,204,732 |
| Fresh-process reload wall time | 0:20.42 | 0:28.34 |
| Reload peak RSS, KiB | 4,308,888 | 4,347,144 |
| Finite reload logits | 303,872 | 303,872 |
| Worker and reload exit codes | 0 / 0 | 0 / 0 |

The final checkpoint hashes are:

- SFT: `e2bcd135527659042304d936fa56c7240f6f96a43b9f81f2a888c38b7762a949`;
  [manifest](native-full-sft-seed1729/manifest.json),
  [independent reload](native-full-sft-reload.json).
- Composed: `2217a994f8b74ded1d52f30ef7add2cb190578e3d108168a85d1409160ad88c3`;
  [manifest](native-full-composed-seed1729/manifest.json),
  [independent reload](native-full-composed-reload.json).

Both workers and both reloads used the same pinned CUDA-capable helper SHA
`5db09c632f6ffccc636bdfec2f693af102bf5327ce7510bc593ebe7089c9c126`,
actual CPU/F32 student execution, four Candle/Rayon threads, five declared CPU
slots and unchanged affinity `0-15`. The exact environment was
`RAYON_NUM_THREADS=4`, `LEC_STUDY_CPU_THREADS=5`, `OMP_NUM_THREADS=4` and
`MKL_NUM_THREADS=4`. The two CPU workers overlapped; their wall times must not be
added to represent elapsed round duration.

All 17 actual CUDA/F32 teacher targets were captured before training in
1:20.34, with 849,396 KiB peak host RSS and an observed 5,676 MiB GPU allocation.
The teacher process was reaped before the CPU workers. Both worker requests used
`CacheOnly`, so a missing or invalid trace could not silently allocate a teacher
or regenerate evidence. Prewarming and training used the exact same executable
and numerical launch contract.

The independent, model-free Rust
[final audit](native-full-round-results.json) verifies real file hashes, finite
F32 adapter contents and tensor movement, unchanged actual base weights, all 48
loss/gradient/component observations, exact realised update order, original
requests, successful GNU-time exits and separate reload bindings. It exited zero
in 14.12 seconds. Its source and executable identities are recorded in the
report; the diagnostic reads model files to hash them but allocates no model.

Recorded compute is actual local execution, without a metered cash charge.
The workers consumed 2,694.65 aggregate user-plus-system CPU seconds, excluding
teacher capture, reloads, sandbox evaluation and compilation. Electricity,
hardware amortisation and currency cost require independently supplied rates
and measurements. Cost per point of coding improvement remains unavailable
until a complete matched quality comparison establishes a non-zero improvement.

The teacher evidence comes from locally hosted original open weights, rather
than three independent API-provider businesses. No held-out task was explored.
The reload probes use two synthetic token IDs, and loss/gradient/tensor evidence
does not establish coding quality, general teacher superiority or SHIP. The
separately frozen matched GPU/F32 quality evaluation has its own runtime and
inference executable; its timings cannot be presented as CPU training timings.
