# Local comparative coding models

Public model metadata, configuration, model cards, tokenisers and licence evidence were retrieved on 15 October 2026, approximately 12:41 UTC. The registry is [config/comparative-models.toml](../config/comparative-models.toml). Public requests used no authentication. The root integration agent alone handles any Hugging Face credentials; this research neither reads nor stores them.

These models support a local comparative experiment on **training and development sources only**. They do not satisfy the assessment's three independently owned real provider businesses or establish a trained student, improvement, measured model cost or successful GPU execution. Hosting several models behind localhost endpoints does not convert those endpoints into independent provider businesses. Two Qwen sizes share a developer and family, with strongly related training; DeepSeek and SmolLM2 add distinct development lineages without proving independent errors or uncontaminated pretraining.

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

The research retrieved no additional weight files. An LFS object identifier is a published expected checksum, not a claim that locally downloaded bytes have passed verification. The root downloader must measure every downloaded file's length and hash before publishing the model directory. Existing student assets remain governed by their original manifest. Configuration, tokeniser, generation settings, original card and applicable licence hashes are separately listed under each model's `files` entries; every small-file checksum was measured from retrieved public bytes, except the already verified baseline tokeniser reused from its existing manifest.

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

These are compatibility requirements, not completed runtime smoke results. The comparison loader must preserve the assessment student's original strict model pin rather than silently widening the trust boundary used for training/promotion. Rust Candle is the intended inference engine; no Transformers/Python runtime is introduced.

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

## Research tooling exception

Public document and metadata retrieval used the installed `curl --disable` HTTP utility, and local byte-count/hash inspection used native `wc`/`sha256sum`. These are read-only operational research tools, consistent with the already documented public-rights snapshot workflow. They neither implement model inference/training nor substitute another language for application logic. `--disable` avoids user curl configuration; no authorisation header or token was supplied. All application logic, download publication/verification, comparative inference, grading and statistics remain Rust.
