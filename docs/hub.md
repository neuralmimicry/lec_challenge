# Pinned model assets

`src/hub.rs` downloads data and documentation from the Hugging Face Hub for the
native model loader. This explicit online operation is separate from offline
model loading and assessment evaluation. It does not evaluate held-out tasks or
make a quality claim about a comparator.

Build the optimised Rust helper with `cargo build --locked --offline --release --example hub_assets`.
Use a release build for multi-gigabyte files: unoptimised checksum verification
can take substantially longer, even after all network bytes have arrived.
For a public repository, run:

```sh
target/release/examples/hub_assets \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --destination offline_bundle/comparative/qwen-coder-1_5b \
  --concurrency 2
```

An explicitly requested ref such as `main` is resolved once through the API,
then the metadata is fetched again at the immutable commit. All file URLs use
that commit. Subsequent invocations use
the same saved plan; a changed repository, commit, size or publisher identity
requires a new destination. Use the recorded forty-character revision for exact
reproduction.

Optional authentication comes from `HF_TOKEN` or a single bounded line on stdin
with `--token-stdin`. The application never writes the token, prints it or puts it
in provenance. Stdin input buffers are zeroised on drop. The token is read with
terminal echo automatically disabled and restored on normal return or an error;
piped stdin is unchanged. The bearer header is
marked sensitive; HTTP client redirects are disabled and followed explicitly.
Authentication is applied only to the initial HTTPS `huggingface.co` request and
is removed permanently after every redirect, including a redirect back to the
original origin. Signed CDN locations and underlying transport errors are not
logged. Do not supply a credential in a command-line argument.

The allowlist includes original model/tokeniser configuration, tokeniser data,
safetensors weights and their shard index, README and licence documents. It
excludes pickle weights, Python files, scripts and unknown repository files.
The loader rejects executable `auto_map`/`custom_pipelines` configuration; there
is no remote-code execution path. Paths with traversal or symlinks are rejected.

Downloads stream through bounded asynchronous concurrency. A fifteen-second
receipt heartbeat reports byte counts; checksum verification remains pending
until the entire file arrives. Heartbeats include no URLs or headers. Partial
files have a saved identity and size. A resumed `206` response must have the exact
expected `Content-Range`; a `200` response replaces the partial bytes. Interrupted
transfers retry a bounded number of times. Complete bytes must match the
publisher's LFS SHA-256 or ordinary Git blob SHA-1 before an atomic rename.
Ordinary Git SHA-1 includes the `blob <size>\0` header: it is never presented as a
publisher SHA-256. Every complete file also receives a computed SHA-256.
Rejected checksum bytes are retained under a `.rejected-<id>` filename and never
enter the asset manifest. A process-safe destination lock survives restarts
without a stale lock; filesystem hashing and durable publication run off the
async executor.

`assets.json` has the native loader's `AssetManifest` schema. The accompanying
`hub-plan.json`, original `hub-metadata.json` and `hub-provenance.json` retain
the commit, publisher identities, sizes, canonical URLs, metadata hash, original
model card and licence evidence. Offline validation reconstructs the plan and
checks every file, shard reference and provenance relationship. The assessment
base validator keeps its original strict pin; comparative models use
`hub::validate_downloaded_assets`.

Some repositories, including SmolLM2, declare an SPDX licence in their original
README/model-card metadata but do not contain a separate licence file. A small
recognised SPDX allowlist permits that case only when the README is preserved;
provenance explicitly records this form of licence evidence. Retain the
canonical full licence text separately in the distribution and make the
applicable model-specific rights decision before use. An unrecognised or missing
declaration without a licence document fails closed. Asset retrieval does not
establish output-training rights.

Focused verification is `cargo test --locked --offline hub::tests --lib`. Local
HTTP tests use clearly labelled public placeholders, never user credentials, and
exercise same-origin and cross-origin redirects, publisher checksum corruption,
commit/metadata mismatch, licence provenance, traversal, symlinks and resumed
range validation. Those fixtures are software QA, rather than downloaded model
or learning evidence.

GGUF acquisition uses the separate `download_gguf` API and the
`hub_gguf_assets` Rust example. Selection names one exact `.gguf` file; the
immutable publisher hash and size can additionally be constrained to the
authorised candidate. The independently pinned original model supplies only
config/tokeniser/template/licence/documentation data under `source/`. Its
floating-point weights are excluded. The composite `gguf-assets.json` binds
the selected weight and every original source asset; `gguf-provenance.json`
retains both repository commits and original API responses. Model-card licence
declarations remain visible, and original source licences are preserved even
when a quantisation repository has no licence file.

`validate_gguf_assets` verifies both publisher identities, all manifest
relationships and GGUF magic/version. It records the declared source binding;
it does not establish that quantisation preserves every original tensor.
The native loader separately checks tensor/config compatibility. Safetensors
validation remains unchanged and rejects this different manifest format.
Transfer and hashing progress report bytes/total, mean MiB/s and ETA every
fifteen seconds, including stalled network waits, and print to stderr whether
or not a terminal is attached. Partial receipt/hash progress is explicitly
labelled as awaiting publisher checksum verification.

Large files use at most four conditional HTTP range streams. A stable strong
ETag, exact `206` boundaries/lengths and the pinned whole-file publisher hash
bind all segments to one object. Existing sequential prefixes are retained;
separate segment files and an immutable plan permit interrupted transfer and
assembly to resume without holding the model in RAM. Unexpected full responses,
changed ETags, mismatched plans and symlink/hardlink artefacts fail closed.
Segment data are removed only after whole-file hash verification succeeds.

A complete published GGUF cache is validated before credentials or any metadata
request. Availability failures have a sanitised typed classification; permanent
authentication/model errors and corrupted bytes do not become offline signals.
Explicit offline operation is controlled by the discovery loop and never calls
the online downloader.

Owned pruning records `.pruned-weights.json` before removing model files. It
binds the original manifest hash, immutable candidate identity and measured
weight hashes/sizes; completion is recorded afterwards and earlier proofs are
archived. A later online download may rehydrate only those known missing
weights after validating every remaining original metadata/source asset.
Changed existing bytes or a forged prune record remain errors. Pruning never
turns a missing checkpoint into a valid serving artifact, and offline runs
report pruned models as unavailable.

Historical teacher evidence uses the explicitly separate
`validate_downloaded_asset_metadata` and `validate_gguf_asset_metadata` APIs.
They revalidate the original commit metadata, immutable download plans,
manifests, publisher identities, licence notices, original tokeniser and chat
template bytes. Large LFS weights are not read; their recorded SHA-256 must
equal the original publisher's LFS identity. Missing weights additionally need
the preserved ownership tuple and durable prune proof bound to the original
manifest, hash and size. Unusual Git-blob weights retain full verification.
Callers must bind this historical identity to the original generation's frozen
manifest and raw output. These APIs deliberately do not prove that present
weights are undamaged, and cannot authorise new inference. All model loaders,
new generation and tensor-capture paths continue to use full weight checksums.
