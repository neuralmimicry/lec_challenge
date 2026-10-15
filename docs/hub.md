# Pinned model assets

`src/hub.rs` downloads data and documentation from the Hugging Face Hub for the
native model loader. This explicit online operation is separate from offline
model loading and assessment evaluation. It does not evaluate held-out tasks or
make a quality claim about a comparator.

Build the Rust helper with `cargo build --locked --offline --example hub_assets`.
For a public repository, run:

```sh
target/debug/examples/hub_assets \
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
