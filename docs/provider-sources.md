# Sources and provider eligibility

Checked 14 October 2026. These are the retrieved public terms/documents, not an assurance that every account, endpoint, model or separately negotiated contract has identical permissions. Snapshot applicable terms and licences in the assessment repository before collecting training data. Missing or ambiguous permission must remain a blocking state for training eligibility.

| Provider/component | Evidence | Design consequence |
| --- | --- | --- |
| DeepSeek API | [Open Platform Terms, section 4.2](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html) explicitly includes training other models/distillation among permitted uses, subject to terms and law. | Strong first candidate for an output-training provider; verify the selected live model and applicable terms. |
| Together | [Terms](https://www.together.ai/terms-of-service), sections 2 and 7: output ownership and hosted-model terms. | Review the selected model licence as well as hosting terms; no blanket eligibility for every hosted model. |
| Mistral text API | [Commercial terms](https://legal.mistral.ai/terms/commercial-terms-of-service/), sections 2 and 3, and [Additional Product Terms](https://legal.mistral.ai/terms/additional-terms/): output ownership, reverse-engineering limits, an explicit competing-image-model restriction, and restrictions on third-party web-search content. | Candidate third provider: no general text-output training prohibition was identified in these reviewed terms. This is an inference, subject to the selected model/account terms. Use ordinary generated coding text without search/third-party content; retain a specific permission decision. |
| Anthropic | [Commercial terms, D.4](https://www.anthropic.com/legal/commercial-terms). | Competing-product/model restriction; exclude from the output-training roster absent applicable express approval. |
| Gemini | [API additional terms, Use Restrictions](https://ai.google.dev/gemini-api/terms). | Competing-model restriction; do not infer permission from output ownership. |
| Qwen student | [Official model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) and [Qwen coding-family release](https://qwenlm.github.io/blog/qwen2.5-coder-family/). | Established 0.5B coding student, Apache 2.0. Pin the exact revision and retain licence/notice files. |
| PEFT | [Transformers PEFT documentation](https://huggingface.co/docs/transformers/peft) and [LoRA documentation](https://huggingface.co/docs/peft/main/package_reference/lora). | Real low-rank training and saved adapters; pin working release versions at build time. |
| Container controls | [Docker run](https://docs.docker.com/engine/containers/run/), [resource constraints](https://docs.docker.com/engine/containers/resource_constraints), [seccomp](https://docs.docker.com/engine/security/seccomp/). | Use explicit controls and verify actual enforcement; Docker alone does not establish hidden-test secrecy. |
| Rootless resource limits | [Docker rootless tips](https://docs.docker.com/engine/security/rootless/tips/). | Check cgroup v2/systemd support; flags can be ignored in unsupported rootless configurations. Fail preflight if limits do not apply. |

## Claims these sources do not establish

- They do not prove that the proposed student improves, or that the selected teachers remain available.
- They do not provide the assumed rates in the design cost table. Those rates are explicitly hypothetical.
- They do not prove immunity to container escapes or establish the validity of a small-sample statistical approximation.
- They do not establish that the provider's decision not to train on your data grants you permission to train on its outputs. Those are separate questions.
- Third-party gateways cannot remove restrictions carried by a hosted model or service.
