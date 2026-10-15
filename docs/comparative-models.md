# Local comparative coding models

Public model metadata, configuration, model cards, tokenisers and licence evidence were retrieved on 15 October 2026, approximately 12:41 UTC. The registry is [config/comparative-models.toml](../config/comparative-models.toml). Public requests used no authentication. The root integration agent alone handles any Hugging Face credentials; this research neither reads nor stores them.

These models support a local comparative experiment on **training and development sources only**. They do not satisfy the assessment's three independently owned real provider businesses or establish a trained student, improvement or measured monetary cost. Hosting several models behind localhost endpoints does not convert those endpoints into independent provider businesses. Two Qwen sizes share a developer and family, with strongly related training; DeepSeek and SmolLM2 add distinct development lineages without proving independent errors or uncontaminated pretraining. Subsequent actual GPU observations are linked below, separately from this original metadata research.

## Selected models and immutable provenance

| Registry ID | Model / role | Developer and licence | Immutable revision | Weight download bytes |
| --- | --- | --- | --- | ---: |
| `qwen-coder-0_5b` | Qwen2.5-Coder-0.5B-Instruct; existing student baseline | Alibaba Qwen; Apache-2.0 | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | 988,097,824, already present |
| `qwen-coder-1_5b` | Qwen2.5-Coder-1.5B-Instruct; larger coding teacher | Alibaba Qwen; Apache-2.0 | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | 3,087,467,144 |
| `deepseek-coder-1_3b` | DeepSeek Coder 1.3B Instruct; distinct coding teacher | DeepSeek; DeepSeek Model Licence 1.0 | `e063262dac8366fc1f28a4da0ff3c50ea66259ca` | 2,692,969,128 |
| `smollm2-1_7b` | SmolLM2-1.7B-Instruct; optional general instruction control | Hugging Face SmolLM; Apache-2.0 | `31b70e2e869a7173562077fd711b654946d38674` | 3,422,777,952 |

The two additional coding teachers require 5,780,436,272 weight bytes. Including optional SmolLM2 requires 9,203,214,224 new weight bytes; including the existing student makes 10,191,312,048 bytes, before small metadata files and filesystem overhead. Download the individual safetensors files, not complete repositories: DeepSeek has duplicate PyTorch weights and SmolLM2 contains many large ONNX exports that this Rust runtime does not use.

| Weight file | Expected SHA-256 from the official Hugging Face LFS object |
| --- | --- |
| Qwen 0.5B `model.safetensors` | `f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44` |
| Qwen 1.5B `model.safetensors` | `c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8` |
| DeepSeek 1.3B `model.safetensors` | `4ff570bf9a558d23c5cb4171bad00daf663be8282150cdb5769e0f6975b0007a` |
| SmolLM2 1.7B `model.safetensors` | `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8` |

The original research retrieved no additional weight files. The subsequent native
Rust downloader has now verified and published all three new local model
directories against their immutable Git/LFS identities, including every weight
length and SHA-256 above. Each directory retains `assets.json`, `hub-plan.json`,
`hub-metadata.json` and `hub-provenance.json`; credentials are absent. Existing
student assets remain governed by their original manifest. Configuration,
tokeniser, generation settings, original card and applicable licence hashes are
separately listed under each model's `files` entries.

Snapshots live under `docs/comparative-models/provenance/<model>/`. Each `api.json` was requested through `https://huggingface.co/api/models/<repository>/revision/<commit>?blobs=true`; the returned revision and LFS hashes establish the reference point. Cards and ordinary files use `https://huggingface.co/<repository>/resolve/<commit>/<filename>`. API popularity counters can change even at an immutable revision; the retained API snapshot's own checksum records the observed bytes rather than promising those counters are immutable.

The [public-provenance audit manifest](comparative-models/provenance/manifest.json) was produced by a Rust check that parsed all four registry entries, matched repository/revision/architecture against their official API snapshots, compared published LFS weight sizes/hashes, checked every listed small file's actual size/hash, required unique identifiers and confirmed the new-weight download total. It records 29 inspected snapshot files, the registry checksum and a passing outcome. It explicitly does not claim additional weight verification, GPU runtime QA, credentials or held-out access.

## Licence decisions and output-training eligibility

Qwen 0.5B and 1.5B have an actual Apache-2.0 `LICENSE` file at their pinned revisions. Preserve its copyright and licence notice when distributing their weights or derived material. Their official cards identify them as coding instruction models. The card's benchmark claims are provider statements, not results measured by this repository.

DeepSeek's weights use the **DeepSeek License Agreement, Version 1.0, 23 October 2023**, not the code repository's MIT licence and not DeepSeek's API terms. Its pinned card explicitly says commercial use is supported. The weight licence defines derivatives to include another model created through transfer of outputs, including “generation of synthetic data by the Model for training the other model”. Paragraphs 2–6 grant derivative use and distribution subject to the stated conditions.

Consequently ordinary local coding evaluation is eligible, and output distillation can be eligible, but a student trained on those outputs is a derivative for this licence. Paragraph 4 requires inherited paragraph-5/Attachment-A use restrictions, a copy of the licence, retained attribution and prominent modification notices. These obligations must travel with the dataset's source rights decisions and the derived adapter's distribution terms. Such a student must not be described as unconditionally Apache-only merely because its Qwen base is Apache-2.0. The reviewed use is executable pure coding tasks; no additional API contract is being inferred. The licence text's SHA-256 is `a3a0e9449cb691a12f4de1d03725fd41326614fdeaf5d80b28c51187da0bed0e`.

SmolLM2's pinned official card declares Apache-2.0 in its front matter and explicitly links the Apache licence in its licence section. Its model repository contains no `LICENSE` sibling. Preserve that original card and the canonical text retrieved from `https://www.apache.org/licenses/LICENSE-2.0.txt`, whose SHA-256 is `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`. The registry records this external URL rather than inventing a missing Hugging Face file. SmolLM2 is a general instruction model with coding data in its documented pretraining, useful as a family/control comparison rather than an assumed stronger coding teacher.

Qwen2.5-Coder-3B-Instruct was reviewed and **excluded**. At revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`, its card says `license: other`, `license_name: qwen-research`; its actual licence limits use to non-commercial research/evaluation and requires a separate commercial licence. It also requires “Built with Qwen” or “Improved using Qwen” documentation when distributed models are trained using its outputs. The retained evidence prevents confusing this variant with the Apache-2.0 0.5B/1.5B variants. Its two shards total 6,171,927,000 bytes, and it is unnecessary for the selected workload and download ceiling.

## Native Rust engine compatibility

Qwen 1.5B uses `Qwen2ForCausalLM`, the same mathematical family as the pinned student, with QKV biases, tied embeddings and the Qwen ChatML template. It changes tensor dimensions: 28 layers, hidden size 1,536, 12 attention heads, two KV heads, head dimension 128 and intermediate size 8,960. Its EOS is 151645, BOS is 151643. Its official tokeniser bytes are identical to the baseline's, but each model still receives its own immutable manifest.

DeepSeek Coder uses `LlamaForCausalLM`: 24 layers, hidden size 2,048, 16 attention and KV heads, head dimension 128, intermediate size 5,504 and untied embeddings. Its RoPE theta is 100,000 and **linear RoPE scaling factor 4.0 is required**. Its BOS is 32013 and EOS is 32021. The original template uses `### Instruction:` / `### Response:` and a coding-specific default system instruction. Ignoring the scaling, using Qwen QKV biases, adding the BOS twice or applying the Qwen template would invalidate a comparison. Use the original pinned tokeniser/template and explicit Llama configuration support, then observe an actual complete generation before evaluation.

SmolLM2 also uses `LlamaForCausalLM`, without QKV biases: 24 layers, hidden size 2,048, 32 attention and KV heads, head dimension 64, intermediate size 8,192 and tied embeddings. Its RoPE theta is 130,000 with no scaling. Its BOS is 1, EOS/pad is 2. It uses its own ChatML template and default SmolLM system instruction. Its vocabulary and special-token IDs differ from Qwen despite the visually similar delimiters. Common Llama tensor operations can be reused; tokeniser/template, biases, scaling and tied-head choices require explicit configuration and QA.

These requirements are implemented in the native Rust Candle loader, including
the original tokeniser templates and DeepSeek rotary scaling. The assessment
student retains its original strict model pin. The [actual GPU endpoint report](../reports/comparative/gpu-serving-teachers-fixed/report.json)
records complete generations for all three new models, actual CUDA/F16 metadata
and driver process allocations. The [GPU guide](gpu.md) explains the original
Qwen attention overflow, its retained failure log and the corrected pre-product
float32 attention arithmetic. No Transformers/Python runtime is introduced.

## RTX 2080 memory and execution plan

The observed card has 8 GiB VRAM. These public safetensors store BF16 weights; use **F16 inference** on the Turing card and prove the actual kernels work. Full-weight F32 loading approximately doubles the tensor memory shown below; these estimates do not authorise placing several complete models on the card together.

| Model | Parameters from official metadata | F16 tensor bytes / GiB | F16 KV cache at 2,048 total tokens |
| --- | ---: | ---: | ---: |
| Qwen 0.5B | 494,032,768 | 988,065,536 / 0.92 GiB | 24 MiB |
| Qwen 1.5B | 1,543,714,304 | 3,087,428,608 / 2.88 GiB | 56 MiB |
| DeepSeek 1.3B | 1,346,471,936 | 2,692,943,872 / 2.51 GiB | 384 MiB |
| SmolLM2 1.7B | 1,711,376,384 | 3,422,752,768 / 3.19 GiB | 384 MiB |

Weights use `parameter_count × 2` bytes. A conventional cached attention layer uses `2 × layers × KV_heads × head_dimension × tokens × bytes_per_element`; the first factor stores keys and values. These are arithmetic estimates, not measured peak VRAM. CUDA context, allocator reservations, attention/MLP workspaces, input prefill, logits, copies and display use consume additional memory. The default batch is one, initial total context cap is 2,048, and **only one model is resident on the GPU at a time**. Bound requests with a single model worker, unload/synchronise between models, and measure real available/peak memory and complete-generation latency before increasing either context or concurrency. A small successful inference does not establish training memory feasibility.

First verify asset hashes and finite forward output for each selected model, then perform a short actual generation on a non-held-out smoke prompt. Compare matched train/development tasks with the same function contract, sandbox/grader and declared decoding settings; retain failed/truncated/empty generations in denominators. Record model/config/tokeniser/template hashes, seeds, device, dtype, timings and every executable verdict. No held-out outcome, original checkpoint gate or remote-provider completion claim follows from this supplemental comparison. Choose optional SmolLM2 only if native architecture support and the real smoke complete successfully.

## Completed development experiment — 15 October 2026

The experiment settings in `config/comparative-experiment.toml` were committed
before task outcomes. Both group plans bind the actual executable SHA-256
`aed864c2285423572c85c3481a80a61a9ebab7983ece13704323d1542db6c081`,
model assets, task contracts, sandbox policy and decoding parameters. All 16 tasks
from four development source packs were run for every model, with seed 1729,
temperature 0.2, top-p 0.95 and a 512-token output limit. One CUDA model and up to
two isolated CPU graders ran concurrently; each model was synchronised and
released before the next loaded. No sealed held-out tasks were evaluated.

| Model | Verified passes / planned tasks | Complete generations | Summed generation seconds | Mean seconds / generation |
| --- | ---: | ---: | ---: | ---: |
| Qwen Coder 0.5B baseline | 0 / 16 | 16 | 85.467 | 5.342 |
| Qwen Coder 1.5B teacher | 2 / 16 | 16 | 149.614 | 9.351 |
| DeepSeek Coder 1.3B teacher | 0 / 16 | 16 | 320.975 | 20.061 |
| SmolLM2 1.7B instruction control | 0 / 16 | 16 | 266.196 | 16.637 |

All 64 planned observations are present, with no generation or infrastructure
errors. Compile and executable contract failures remain failures. Qwen 1.5B's
two passes were `dispatch.finish_order` and `dispatch.waiting`; the other models
did not pass those tasks. This is weak performance on these Rust/JSON contracts,
not an established general ranking or evidence that a student learned. One seed
and four jointly authored sources do not support a promotion claim. Templates,
tokenisers and output lengths differ; mean generation times are observations
for these workloads, not a matched-token throughput benchmark.

The [coding group](../reports/comparative/coding-development/report.md) and
[instruction control](../reports/comparative/control-development/report.md)
retain immutable plans, actual execution metadata, raw outputs and trusted
per-task grader artifacts. The [native resource report](../reports/comparative/resources/resources.md)
binds the inputs and includes per-task times/tokens and sandbox durations in its
JSON counterpart. Generation times exclude loading, checksumming and grading;
sandbox and inference durations can overlap. Remote inference charges are zero.
Electricity, hardware depreciation, downloads and labour have not been priced,
so total monetary cost remains unknown. No training round or learning gain
occurred here; cost per point of learning is N/A.

## Research tooling exception

Public document and metadata retrieval used the installed `curl --disable` HTTP utility, and local byte-count/hash inspection used native `wc`/`sha256sum`. These are read-only operational research tools, consistent with the already documented public-rights snapshot workflow. They neither implement model inference/training nor substitute another language for application logic. `--disable` avoids user curl configuration; no authorisation header or token was supplied. All application logic, download publication/verification, comparative inference, grading and statistics remain Rust.

## Bounded teacher replacement and retention

The original 2/16 incumbent is the **Qwen 1.5B teacher**, while the student baseline
scored 0/16. The replacement protocol in
[config/teacher-search.toml](../config/teacher-search.toml) visits at most six
immutable candidates. The default target is three distinct original models,
each passing at least 80% separately on the original 16 development tasks and
on an additional, independently frozen 16-task development qualification pack.
At this denominator 80% requires at least 13 passes. The fresh pack is evaluated
only after the original threshold is reached. Both packs require all planned
observations and zero generation/infrastructure errors; missing, empty,
truncated and failed runs never become passes. This is exploratory teacher
selection, not the assessment's sealed held-out promotion gate.

```sh
./run-gpu.sh select-teachers --config config/teacher-search.toml --output reports/comparative/teacher-search-v1
./run-gpu.sh select-teachers --config config/teacher-search.toml --output reports/comparative/teacher-search-v1 --offline
```

The launcher discovers available hardware and builds the compatible Rust engine.
The search config requests `auto`: CUDA is chosen only when the compiled backend
and currently free VRAM satisfy a conservative estimate; otherwise the recorded
reason explains CPU selection. Explicit CPU/CUDA choices do not silently switch.
The verified original configuration drives estimates for quantised weights,
expanded embeddings, float32 KV caches, attention scores and activation workspace
at the complete 1,024-token context cap. Actual prompt plus 512 output tokens is
checked independently against that cap. One model is resident at a time, with
synchronisation and release before the next candidate. CPU inference remains
substantially slower; the protocol does not promise a completion time.

| Visit | Original source / quantisation | Quantisation repository | Expected weight bytes | Licence |
| --- | --- | --- | ---: | --- |
| 1 | Qwen2.5-Coder-7B-Instruct / Q4_K_M | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | 4,683,073,536 | Apache-2.0 |
| 2 | DeepSeek-Coder-6.7B-Instruct / Q4_K_M | `TheBloke/deepseek-coder-6.7B-instruct-GGUF` | 4,083,015,904 | DeepSeek Model Licence 1.0 |
| 3 | Magicoder-S-DS-6.7B / Q4_K_M | `TheBloke/Magicoder-S-DS-6.7B-GGUF` | 4,082,886,528 | DeepSeek Model Licence 1.0 |
| 4 | Qwen2.5-Coder-14B-Instruct / Q2_K | `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF` | 5,770,497,408 | Apache-2.0 |
| 5 | Qwen2.5-7B-Instruct / Q4_K_M | `bartowski/Qwen2.5-7B-Instruct-GGUF` | 4,683,074,240 | Apache-2.0 |
| 6 | Qwen2.5-Coder-7B-Instruct / Q6_K | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | 6,254,198,784 | Apache-2.0 |

The final row shares its repository and original source identity with visit 1;
the complete immutable repository/file/hash pins are retained in the TOML
config. Quantisation variants cannot increase the
distinct-source count. Qwen coding and general models share a developer;
Magicoder inherits DeepSeek's base, so neither source names nor endpoint counts
establish independent errors or three provider businesses. Third-party
quantisations retain their own immutable provenance alongside separately pinned
original tokenisers, chat templates, generation settings and licence documents.
Source snapshots reviewed before acquisition are under
`docs/comparative-models/replacements/`; the downloaded composite manifests bind
the actual files used.

The 32 GiB acquisition cap reserves the declared weight size plus 64 MiB for
auxiliaries before each first acquisition, with an additional measured 8 GiB
free-disk floor. This is a **logical asset allocation**, not a network traffic
meter or cash budget: retries, resumed partial transfers and filesystem overhead
have different physical costs. Local compute, bandwidth, storage and energy
cash costs remain unmeasured. No paid provider inference calls occur in this
supplemental search.

Every output directory freezes the config, original model registry, actual
executable, complete original development corpus and the fresh qualification
manifest before new candidate inference. A protocol/binary change needs a new
versioned output directory. Durable raw generations precede grading, and linked
plans/raw outputs/trusted sandbox audits are revalidated when resuming. The
original trial runtime and resource decision remain frozen; already recorded
generations can finish grading without allocating model or device memory.
Existing cache verification precedes all HTTP. Explicit `--offline` performs
zero metadata requests, downloads or health probes. A typed Hugging Face
connectivity failure changes the remainder of that invocation to cached-only
mode and records the precise limitation; authentication, configuration and
checksum failures do not become claims that the Internet is unavailable.

Qualified models are ranked by their lower pack score, then combined passes and
a deterministic identifier. Retention keeps the strongest qualified variant per
original source. If none qualifies, it retains the best available model solely
on the original selection score and exposes its fresh qualification status.
The original student, explicit keep-list entries and candidates awaiting an
observed fresh qualification remain protected. Poor scored alternatives are
pruned only after their complete run evidence is verified and their directories
are explicitly adopted by the owned-cache marker. Pruning touches validated
root weight files or expired untracked weight partials; model metadata, original
outputs, reports, assessment weights and checkpoints remain. A separate durable
deletion proof binds the identity, pre-prune manifest and removed bytes, allowing
online rehydration of known intentional deletions while present corruption is
rejected. Reclaimed logical bytes and removed allocated blocks are measured
separately; neither is a measured free-disk-space delta.

CodeQwen1.5-7B-Chat was additionally reviewed and excluded from automatic
distillation selection: its actual licence restricts using materials/outputs to
improve other LLMs except Tongyi Qianwen and derivatives. Apache terms from other
Qwen releases must not be assumed for that variant. Qwen3 and Phi variants remain
outside this roster because their native architectures require separate loader
work and verification. DeepSeek and Magicoder outputs remain subject to the
inherited derivative restrictions explained above. A search shortfall is
reported explicitly and does not relax these thresholds or claim learning.
