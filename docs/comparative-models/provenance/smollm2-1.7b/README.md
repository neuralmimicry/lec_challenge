---
library_name: transformers
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- safetensors
- onnx
- transformers.js
base_model:
- HuggingFaceTB/SmolLM2-1.7B
---


# SmolLM2

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/y45hIMNREW7w_XpHYB_0q.png)

##  Table of Contents

1. [Model Summary](#model-summary)
2. [Evaluation](#evaluation)
3. [Examples](#examples)
4. [Limitations](#limitations)
5. [Training](#training)
6. [License](#license)
7. [Citation](#citation)

## Model Summary

SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device. More details in our paper: https://arxiv.org/abs/2502.02737v1

The 1.7B variant demonstrates significant advances over its predecessor SmolLM1-1.7B, particularly in instruction following, knowledge, reasoning, and mathematics. It was trained on 11 trillion tokens using a diverse dataset combination: FineWeb-Edu, DCLM, The Stack, along with new mathematics and coding datasets that we curated and will release soon. We developed the instruct version through supervised fine-tuning (SFT) using a combination of public datasets and our own curated datasets. We then applied Direct Preference Optimization (DPO) using [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

The instruct model additionally supports tasks such as text rewriting, summarization and function calling thanks to datasets developed by [Argilla](https://huggingface.co/argilla) such as [Synth-APIGen-v0.1](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1).
You can find the SFT dataset here: https://huggingface.co/datasets/HuggingFaceTB/smoltalk.

For more details refer to: https://github.com/huggingface/smollm. You will find pre-training, post-training, evaluation and local inference code.

### How to use

#### Transformers
```bash
pip install transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "What is the capital of France."}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```


#### Chat in TRL
You can also use the TRL CLI to chat with the model from the terminal:
```bash
pip install trl
trl chat --model_name_or_path HuggingFaceTB/SmolLM2-1.7B-Instruct --device cpu
```

#### Transformers.js

```bash
npm i @huggingface/transformers
```

```js
import { pipeline } from "@huggingface/transformers";

// Create a text generation pipeline
const generator = await pipeline(
  "text-generation",
  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
);

// Define the list of messages
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Tell me a joke." },
];

// Generate a response
const output = await generator(messages, { max_new_tokens: 128 });
console.log(output[0].generated_text.at(-1).content);
// "Why don't scientists trust atoms?\n\nBecause they make up everything!"
```

## Evaluation

In this section, we report the evaluation results of SmolLM2. All evaluations are zero-shot unless stated otherwise, and we use [lighteval](https://github.com/huggingface/lighteval) to run them.

## Base Pre-Trained Model

| Metric           | SmolLM2-1.7B | Llama-1B    | Qwen2.5-1.5B | SmolLM1-1.7B |
|------------------|--------------|-------------|---------------|--------------|
| HellaSwag        | **68.7**     | 61.2        | 66.4          | 62.9         |
| ARC (Average)    | **60.5**     | 49.2        | 58.5          | 59.9         |
| PIQA             | **77.6**     | 74.8        | 76.1          | 76.0         |
| MMLU-Pro (MCF)   | **19.4**     | 11.7        | 13.7          | 10.8         |
| CommonsenseQA    | **43.6**     | 41.2        | 34.1          | 38.0         |
| TriviaQA         | **36.7**     | 28.1        | 20.9          | 22.5         |
| Winogrande       | **59.4**     | 57.8        | 59.3          | 54.7         |
| OpenBookQA       | 42.2         | 38.4        | 40.0          | **42.4**     |
| GSM8K (5-shot)   | 31.0         | 7.2         | **61.3**      | 5.5          |

## Instruction Model

| Metric                       | SmolLM2-1.7B-Instruct | Llama-1B-Instruct | Qwen2.5-1.5B-Instruct | SmolLM1-1.7B-Instruct |
|:-----------------------------|:---------------------:|:-----------------:|:----------------------:|:----------------------:|
| IFEval (Average prompt/inst) | **56.7**             | 53.5             | 47.4                  | 23.1                  |
| MT-Bench                     | 6.13                | 5.48             | **6.52**              | 4.33                  |
| OpenRewrite-Eval (micro_avg RougeL) | 44.9           | 39.2             | **46.9**              | NaN                   |
| HellaSwag                    | **66.1**            | 56.1             | 60.9                  | 55.5                  |
| ARC (Average)                | **51.7**            | 41.6             | 46.2                  | 43.7                  |
| PIQA                         | **74.4**            | 72.3             | 73.2                  | 71.6                  |
| MMLU-Pro (MCF)               | 19.3               | 12.7             | **24.2**              | 11.7                  |
| BBH (3-shot)                 | 32.2               | 27.6             | **35.3**              | 25.7                  |
| GSM8K (5-shot)               | **48.2**           | 26.8             | 42.8                  | 4.62                  |


## Examples
Below are some system and instruct prompts that work well for special tasks

### Text rewriting

```python
system_prompt_rewrite = "You are an AI writing assistant. Your task is to rewrite the user's email to make it more professional and approachable while maintaining its main points and key message. Do not return any text other than the rewritten message."
user_prompt_rewrite = "Rewrite the message below to make it more friendly and approachable while maintaining its main points and key message. Do not add any new information or return any text other than the rewritten message\nThe message:"
messages = [{"role": "system", "content": system_prompt_rewrite}, {"role": "user", "content":f"{user_prompt_rewrite} The CI is failing after your last commit!"}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```
```
Hey there! I noticed that the CI isn't passing after your latest commit. Could you take a look and let me know what's going on? Thanks so much for your help!
```

### Summarization

```python
system_prompt_summarize = "Provide a concise, objective summary of the input text in up to three sentences, focusing on key actions and intentions without using second or third person pronouns."
messages = [{"role": "system", "content": system_prompt_summarize}, {"role": "user", "content": INSERT_LONG_EMAIL}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```

### Function calling

SmolLM2-1.7B-Instruct can handle function calling, it scores 27% on the [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html). Here's how you can leverage it:

```python
import json
import re
from typing import Optional

from jinja2 import Template
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema


system_prompt = Template("""You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>{{ tools }}</tools>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>""")


def prepare_messages(
    query: str,
    tools: Optional[dict[str, any]] = None,
    history: Optional[list[dict[str, str]]] = None
) -> list[dict[str, str]]:
    """Prepare the system and user messages for the given query and tools.
    
    Args:
        query: The query to be answered.
        tools: The tools available to the user. Defaults to None, in which case if a
            list without content will be passed to the model.
        history: Exchange of messages, including the system_prompt from
            the first query. Defaults to None, the first message in a conversation.
    """
    if tools is None:
        tools = []
    if history:
        messages = history.copy()
        messages.append({"role": "user", "content": query})
    else:
        messages = [
            {"role": "system", "content": system_prompt.render(tools=json.dumps(tools))},
            {"role": "user", "content": query}
        ]
    return messages


def parse_response(text: str) -> str | dict[str, any]:
    """Parses a response from the model, returning either the
    parsed list with the tool calls parsed, or the
    model thought or response if couldn't generate one.

    Args:
        text: Response from the model.
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    return text


model_name_smollm = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name_smollm, device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_smollm)

from datetime import datetime
import random

def get_current_time() -> str:
    """Returns the current time in 24-hour format.

    Returns:
        str: Current time in HH:MM:SS format.
    """
    return datetime.now().strftime("%H:%M:%S")


def get_random_number_between(min: int, max: int) -> int:
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return random.randint(min, max)


tools = [get_json_schema(get_random_number_between), get_json_schema(get_current_time)]

toolbox = {"get_random_number_between": get_random_number_between, "get_current_time": get_current_time}

query = "Give me a number between 1 and 300"

messages = prepare_messages(query, tools=tools)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
# [{'name': 'get_random_number_between', 'arguments': {'min': 1, 'max': 300}}

# Get tool responses
tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
# [63]

# For the second turn, rebuild the history of messages:
history = messages.copy()
# Add the "parsed response"
history.append({"role": "assistant", "content": result})
query = "Can you give me the hour?"
history.append({"role": "user", "content": query})

inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
# ['07:57:25']
```
More details such as parallel function calls and tools not available can be found [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/instructions_function_calling.md)

## Limitations

SmolLM2 models primarily understand and generate content in English. They can produce text on a variety of topics, but the generated content may not always be factually accurate, logically consistent, or free from biases present in the training data. These models should be used as assistive tools rather than definitive sources of information. Users should always verify important information and critically evaluate any generated content.

## Training

### Model

- **Architecture:** Transformer decoder
- **Pretraining tokens:** 11T
- **Precision:** bfloat16

### Hardware

- **GPUs:** 256 H100

### Software

- **Training Framework:** [nanotron](https://github.com/huggingface/nanotron/tree/main)
- **Alignment Handbook** [alignment-handbook](https://github.com/huggingface/alignment-handbook/)

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Citation
```bash
@misc{allal2025smollm2smolgoesbig,
      title={SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Guilherme Penedo and Lewis Tunstall and Andrés Marafioti and Hynek Kydlíček and Agustín Piqueres Lajarín and Vaibhav Srivastav and Joshua Lochner and Caleb Fahlgren and Xuan-Son Nguyen and Clémentine Fourrier and Ben Burtenshaw and Hugo Larcher and Haojun Zhao and Cyril Zakka and Mathieu Morlon and Colin Raffel and Leandro von Werra and Thomas Wolf},
      year={2025},
      eprint={2502.02737},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02737}, 
}
```