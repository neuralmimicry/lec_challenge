# Original dependency notices

The generated inventory at `offline_bundle/notices/manifest.json` now retains original notices for **341 resolved Cargo packages, with zero remaining notice-file gaps**. This scope includes target-specific packages; it does not assert that every resolved package is linked into the Linux binary. Existing packaged notices remain unchanged.

Six crates omitted conventionally named root licence files from their published package. Their original upstream notices were retrieved at the exact commit recorded in each package's `.cargo_vcs_info.json`:

| Package | Original upstream commit | Retained original notice files |
| --- | --- | --- |
| candle-nn 0.9.1 | `cd96fa80da255e34f7b16b4ff98b6a31d557201b` | `LICENSE-APACHE`, `LICENSE-MIT` |
| r-efi 5.3.0 | `97b55bed1c2c91dcbf787674849f05337ff80b33` | `AUTHORS`, containing the original complete MIT permission grant and copyright attribution |
| r-efi 6.0.0 | `7e1b0322d31d625f81a5656096330934f9cd835d` | `AUTHORS`, containing the original complete MIT permission grant and copyright attribution |
| ug 0.4.0 | `854edf4f9d97c86723a48ab89362fc1339675180` | `LICENSE-APACHE`, `LICENSE-MIT` |
| valuable 0.1.1 | `9efc29b6e58cef28f6566a47aa7e142a55fead77` | `LICENSE` |
| wasip2 1.0.1+wasi-0.2.4 | `3da562c06214feafc14d37bf290671636caa6718` | `LICENSE-APACHE`, `LICENSE-Apache-2.0_WITH_LLVM-exception`, `LICENSE-MIT` |

No generic licence template was substituted for a missing original. The r-efi files already contained their actual grant; the initial filename-only scanner had overlooked them. The retained originals preserve the upstream notices and their wording.

`offline_bundle/notices/supplemental.json` records the package/version/repository, exact upstream commit/path, original crate VCS proof and Cargo manifest, original workspace Cargo manifest, full upstream Git-tree evidence, source URLs/blob identities, and SHA-256/size/path for every downloaded file. The files live under their package's notice directory, with original notices in its `upstream/` subtree. The inventory and supplements are bound to the current `Cargo.lock` hash. Offline regeneration verifies both source identity and every retained artifact hash before accepting a supplement.

The Rust generator is `examples/dependency_notices.rs`, with provenance handling in `examples/dependency_notices/upstream.rs`. The explicit acquisition command used was:

```sh
cargo run --locked --offline --example dependency_notices -- --fetch-missing
```

`--fetch-missing` explicitly enables public upstream HTTP acquisition. Cargo's `--offline` applies to dependency resolution; it does not disable those deliberate HTTP requests. Acquisition used no provider credentials and made no model calls.

The following subsequent verification ran without an HTTP acquisition flag and again reported 341 packages and zero gaps:

```sh
cargo run --locked --offline --example dependency_notices -- \
  --supplemental offline_bundle/notices/supplemental.json
```

These software notices accompany the distributed binary and source assets. They are separate from the unresolved contractual permission to train on third-party model output, documented in `docs/rights.md`.
