# Original dependency notices

The current inventory at `offline_bundle/notices/manifest.json` covers **350 resolved Cargo packages across all features, with one explicitly unresolved original-notice gap: bindgen_cuda 0.1.6**. Original notices are retained for the other 349 packages. The default CPU dependency graph contains 345 packages and has zero notice-file gaps; five further packages enter through CUDA. The complete inventory includes target-specific dependencies and does not assert that every resolved package is linked into the Linux binary. Existing notice bytes remain unchanged.

The inventory is bound to Cargo.lock SHA-256 `bd62f0019341d1a6432ba9153fefdfda757ac26856dc5da297995462176742bb`. The earlier 341-package, zero-gap inventory and its supplement are preserved verbatim under `offline_bundle/notices/history/`, bound to the earlier lock hash `1f6da15c332cf3d3da327343ac0568cb086493d3aa7d322878e5a6880c2403eb`. These historical records do not describe the current CUDA graph.

Eight packages omit conventionally named root licence files but have original upstream grants at the exact commit recorded in each package's `.cargo_vcs_info.json`:

| Package | Original upstream commit | Retained original notice files |
| --- | --- | --- |
| candle-nn 0.9.1 | `cd96fa80da255e34f7b16b4ff98b6a31d557201b` | `LICENSE-APACHE`, `LICENSE-MIT` |
| candle-kernels 0.9.1 | `cd96fa80da255e34f7b16b4ff98b6a31d557201b` | `LICENSE-APACHE`, `LICENSE-MIT` |
| r-efi 5.3.0 | `97b55bed1c2c91dcbf787674849f05337ff80b33` | `AUTHORS`, containing the original complete MIT permission grant and copyright attribution |
| r-efi 6.0.0 | `7e1b0322d31d625f81a5656096330934f9cd835d` | `AUTHORS`, containing the original complete MIT permission grant and copyright attribution |
| ug 0.4.0 | `854edf4f9d97c86723a48ab89362fc1339675180` | `LICENSE-APACHE`, `LICENSE-MIT` |
| ug-cuda 0.4.0 | `854edf4f9d97c86723a48ab89362fc1339675180` | `LICENSE-APACHE`, `LICENSE-MIT` |
| valuable 0.1.1 | `9efc29b6e58cef28f6566a47aa7e142a55fead77` | `LICENSE` |
| wasip2 1.0.1+wasi-0.2.4 | `3da562c06214feafc14d37bf290671636caa6718` | `LICENSE-APACHE`, `LICENSE-Apache-2.0_WITH_LLVM-exception`, `LICENSE-MIT` |

No generic licence template was substituted for a missing original. The r-efi files already contained their actual grant; the initial filename-only scanner had overlooked them. The retained originals preserve the upstream notices and their wording.

The new packaged notice files for `cudarc 0.16.6`, `glob 0.3.4`, `memo-map 0.3.4`, `minijinja 2.24.0`, `minijinja-contrib 2.24.0` and `sha1 0.10.7` were copied byte-for-byte from their published crate directories. `zeroize 1.9.1` was already a resolved transitive dependency and its original notices remain present. NVIDIA's installed SDK/driver are external toolchain requirements; their headers and libraries are not redistributed by this notice refresh.

`bindgen_cuda 0.1.6` is a CUDA build dependency whose original Cargo manifest declares MIT and names its author. Its complete upstream tree at commit `775cc76b607480819eb14b3658df8a96f6b21dab` contains only `.gitignore`, Cargo.toml, README.md and src/lib.rs: no original licence grant or copyright notice is present. The current official default branch resolves to the same commit; public repository metadata reports no detected licence, and the official LICENSE-path history response is empty. These observations do not negate the publisher's MIT declaration, but they leave the requested original full notice unavailable. No later applicable publisher licence was found and no copyright attribution was invented. Obtain that original notice from the publisher before claiming a complete CUDA source distribution.

The original packaged VCS/Cargo proof, complete pinned upstream tree/workspace manifest and current official research responses are preserved in `offline_bundle/notices/bindgen_cuda-0.1.6/`. The inventory binds their paths, sizes and SHA-256 hashes separately from actual licence files. The failed acquisition report is retained in `history/acquisition-bd62f0019341d1a6432ba9153fefdfda757ac26856dc5da297995462176742bb.json`.

`offline_bundle/notices/refresh.json` records the lockfile migration, observed package counts and nonzero verification result. `bindgen_cuda-0.1.6/SOURCE-RESEARCH.json` records the exact unauthenticated public API URLs and response hashes; it distinguishes observed absence from a claim about every possible licence filename or permission source.

`offline_bundle/notices/supplemental.json` records the package/version/repository, exact upstream commit/path, original crate VCS proof and Cargo manifest, original workspace Cargo manifest, full upstream Git-tree evidence, source URLs/blob identities, and SHA-256/size/path for every downloaded file. The files live under their package's notice directory, with original notices in its `upstream/` subtree. The inventory and supplements are bound to the current `Cargo.lock` hash. Offline regeneration verifies both source identity and every retained artifact hash before accepting a supplement.

The Rust generator is `examples/dependency_notices.rs`, with provenance handling in `examples/dependency_notices/upstream.rs`. The explicit acquisition command used was:

```sh
cargo run --locked --offline --example dependency_notices -- --fetch-missing
```

`--fetch-missing` explicitly enables public upstream HTTP acquisition. Cargo's `--offline` applies to dependency resolution; it does not disable those deliberate HTTP requests. Acquisition used no provider credentials and made no model calls.

The generator uses `cargo metadata --locked --offline --all-features` so optional CUDA dependencies cannot be omitted. It continues after an acquisition error, preserves the original evidence, writes the complete package/gap inventory and then returns nonzero if any notice or provenance verification remains incomplete. Acquisition observed 350 packages, one gap and one failed original-grant acquisition. The subsequent verification command makes no HTTP acquisitions and retains the same unresolved gap:

```sh
cargo run --locked --offline --example dependency_notices -- \
  --supplemental offline_bundle/notices/supplemental.json
```

These software notices accompany distributed binary and source assets. The historical CPU delivery archive remains unchanged by this local CUDA refresh. Software notice evidence is separate from contractual permission to train on third-party model output, documented in `docs/rights.md`.
