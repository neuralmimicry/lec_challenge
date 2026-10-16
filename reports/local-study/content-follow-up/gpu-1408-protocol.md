# Supplemental content collection: explicit GPU context

Declared before dispatch. This is a new operational protocol, not a rewrite of
round 001 or `round-002-cpu`. The CPU protocol's first three completed variants
timed out under their fixed 120-second generation limit. Its saved policy,
plans, raw outputs and records remain intact; remaining records are pending,
not successful. The integration owner stops that owned process to release its
resources. Repeating its original command resumes its validated work.

At the actual controlled SIGINT, four raw timeout records existed: the three
`ledger.sum` variants and `ledger.prefix` process. The process exited 130.
Those records and any pending plan remain unchanged; the interrupted round has
no complete successful-batch claim.

The actual model-free Rust receipt
`context-preflight-1408-001.json` counts all 12 original requests with the
publisher's original tokeniser/template. The largest complete request requires
548 input + 768 maximum output = 1,316 tokens. All requests therefore retain
their full allowance inside 1,408 tokens. The fresh GPU snapshot has
7,739,539,456 free bytes. The conservative guard needs 7,688,540,159 free bytes
for 1,408, versus 7,788,154,879 for 1,536. These are arithmetic guards, not a
guarantee against concurrent display allocations. Loading must recheck them.

Use the unchanged pinned application
`b45d32ca11ccc84633ab2de0bb9d195dc38a61c78e2ef1a87c03ea90091aa301`,
CPU thread settings Rayon/OMP/MKL 4 and assigned worker threads 5, and CUDA
12.9's local library directory. Run:

```sh
state/engines/b45d32ca11ccc84633ab2de0bb9d195dc38a61c78e2ef1a87c03ea90091aa301/lec-distil \
  collect-teacher-content \
  --bundle reports/local-study/qwen7b-evidence/bundle.json \
  --output reports/local-study/content-follow-up/round-003-gpu-1408 \
  --context-tokens 1408 --max-tokens 768 --maximum-tasks 4 \
  --content process,context,tool_trajectory --offline --device cuda --dtype f32
```

The collector freezes its exact requests and policy before generating, uses
train-only sources, independently grades complete code and retains failures.
Inherited 18 positive samples cannot count as successful new content. No
training round, matched quality gain, preference/trajectory capability or
promotion is claimed by this preregistration. The numerical preflight source,
executable and linked-library hashes are in the receipt. Its first standalone
link attempt lacked CUDA native library search paths and failed; the corrected
build and failed log are both retained under excluded diagnostics.
