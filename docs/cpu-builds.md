# Reproducible CPU SIMD builds

The Rust launcher automatically selects a safe explicit CPU profile for a new
CPU protocol, including `--device cpu` and automatic mode without a usable CUDA
toolkit. On amd64 it requires AVX2, FMA, F16C and OS-enabled AVX together. On arm64
it selects NEON when detected. Otherwise it uses the portable generic target.
New CUDA builds use the same reviewed profile for CPU fallback and CPU target
capture. Historical and packaged engines retain their original generic builds.

```sh
# New explicit CPU work: use the detected safe profile.
./run.sh study-evaluate --device cpu --offline --output reports/local-study/cpu-new

# A new portable protocol, suitable for machines without optional x86 features.
LEC_CPU_PROFILE=portable ./run.sh study-evaluate --device cpu --offline \
  --output reports/local-study/cpu-portable-new
```

`LEC_CPU_PROFILE` accepts `auto` (the default) or `portable`. Resumption always
resolves the original executable and its saved requirements before considering
new compilation settings. The portable override therefore requires a new output
or explicit fresh protocol to change an existing experiment.

Every new CPU build uses `target-cpu=generic` plus its exact reviewed feature
list. It never uses `target-cpu=native`. Rust verifies the actual compiler cfg,
records `rustc -vV`, target triple, complete cfg, exact argument list and Cargo
encoded flags, and hashes Cargo.lock. Nonempty inherited Rust flags, compiler
wrappers or alternative target settings are rejected as ambiguous. The encoded
flag value overrides Cargo configuration. Each configuration has a separate
`target/cpu/<architecture>/<profile>/<configuration-hash>` build directory.
CUDA plus SIMD uses a separate `target/cuda-simd/...` subtree and explicitly
freezes `cargo_features=["cuda"]` in the same CPU receipt. The CPU-only empty
feature list remains absent from older JSON. Prewarming targets and a cache-only
worker can therefore use one exact CUDA executable, including its CPU paths,
without swapping a frozen engine or sharing incompatible Cargo caches.
Neither a dependency nor a numerical kernel is patched.

For a new build, inspection resolves the selected toolchain's actual
`sysroot/bin/rustc`, records its absolute path, SHA-256 and byte length, and
checks its reported version again before compilation. Both the launcher and
the build-only helper pass that exact compiler through `RUSTC` and
`CARGO_BUILD_RUSTC`, clear both Cargo compiler-wrapper aliases, and apply the
frozen encoded flags. A mutable rustup dispatcher therefore cannot silently
select another compiler between inspection and compilation. Historical receipts
without this optional compiler identity remain valid for executable replay;
rebuilding them requires a new build protocol.

`examples/build_study_engine.rs` is the native build-only counterpart for local
study execution. It builds the application, `distillation_qa`, the evaluation
recovery diagnostic and `quantised_prefix_qa` with one reviewed configuration, then pins all four with
their CPU receipts. Its frozen request hashes Rust sources, Cargo inputs and
the explicitly selected CUDA build settings. Restarting the same command with
unchanged inputs reuses Cargo's completed work; changed sources or settings
require a new output. It emits a progress message every 30 seconds during
compilation. `--cuda` requires the already verified toolkit environment from
`docs/gpu.md`; it does not infer a toolkit from an unverified installation.
The completed training executable must perform both teacher-target prewarming
and cache-only training. A later build cannot consume the earlier target cache
as though its executable identity were unchanged.

Direct `distillation_qa` `prewarm`, `worker` and `verify` phases additionally
freeze their numerical thread environment before model allocation. For the
documented four-thread launch, prefix every phase with
`env RAYON_NUM_THREADS=4 LEC_STUDY_CPU_THREADS=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`
and reserve five available CPUs. Preserve the exact executable, affinity and
all four values across the phases and interruption recovery; see the complete
[direct-helper examples](distillation.md). Existing historical freezes use their
original pinned helper and original launch contract. A new executable cannot
retrofit a missing `helper-launch.json` onto them.

After selecting the verified CUDA environment documented in `gpu.md`, run:

```sh
cargo run --locked --offline --example build_study_engine -- \
  --cuda --output reports/local-study/build-cuda-simd-001
```

The result names each executable's purpose and binds it to the complete request
hash. The two-hour compilation deadline is separate from study/training budgets.
Failures retain Cargo's cache and their request rather than publishing a completed
executable set. This is a build command, not evidence of model execution.

The resulting `PinnedEngine.cpu_build` receipt binds that configuration to the
exact executable SHA-256 and byte length. Pinning an already saved engine keeps
its receipt. A missing optional field in historical generic-engine JSON remains
absent when serialised, preserving the old evidence shape. Launcher receipts
and typed workflow protocols retain the CPU requirements; legacy hash-only
protocols also verify the retained engine metadata.

The launcher resolves `--fresh` once, passes the actual output directory to the
application and writes its complete launch receipt before startup. The application
does not create a second fresh directory. This also binds CPU requirements if
interruption occurs before a search or content collector writes its protocol.

Before execution or recovery, runtime feature detection checks the frozen host
requirements and verifies executable/receipt identity. This path does not invoke
rustc, Cargo, nvcc or a C++ compiler. An incompatible machine must retain the
original evidence and start a separately identified portable experiment.

The source change enables Candle's existing compile-time SIMD paths. It is not
a measured speed-up or a model-quality result. A fresh optimised executable
still needs the ordinary numerical/prefix checks and matched resource evidence
before any speed or quality comparison. Historical scalar observations remain
historical observations and are never relabelled as SIMD measurements.
