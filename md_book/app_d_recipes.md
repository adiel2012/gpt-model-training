# Quick-Start Recipes

## QLoRA SFT with TRL and PEFT

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B"
)

# LoRA config
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Train
dataset = load_dataset("your_dataset")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
    ),
)
trainer.train()
```

## DPO Alignment with TRL

```python
from trl import DPOConfig, DPOTrainer

# Dataset must have columns: prompt, chosen, rejected
dpo_trainer = DPOTrainer(
    model=model,           # SFT-trained model
    ref_model=ref_model,   # Frozen copy of the SFT model
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=DPOConfig(
        beta=0.1,          # KL penalty coefficient
        max_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        bf16=True,
        output_dir="./dpo_output",
    ),
)
dpo_trainer.train()
```

## GRPO Reasoning Training with TRL

```python
from trl import GRPOConfig, GRPOTrainer

def reward_fn(completions, ground_truths, **kwargs):
    """Binary reward: 1.0 if answer matches, 0.0 otherwise."""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        answer = extract_boxed_answer(completion)
        rewards.append(1.0 if answer == gt else 0.0)
    return rewards

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    train_dataset=math_dataset,
    args=GRPOConfig(
        num_generations=8,    # group size G
        max_new_tokens=512,
        temperature=0.9,
        beta=0.001,           # KL penalty weight
        output_dir="./grpo_output",
    ),
)
grpo_trainer.train()
```

## Serving with vLLM

```python
# CLI: vllm serve meta-llama/Llama-3.1-8B-Instruct \
#        --quantization awq \
#        --max-model-len 32768 \
#        --tensor-parallel-size 2 \
#        --enable-prefix-caching

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="awq",
    tensor_parallel_size=2,
    enable_prefix_caching=True,
)
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=512
)
outputs = llm.generate(
    ["Explain transformer attention briefly."],
    sampling_params,
)
print(outputs[0].outputs[0].text)
```

## Model Merging with mergekit

```python
# merge_config.yaml
models:
  - model: meta-llama/Llama-3.1-8B
    # base model -- no parameters
  - model: ./checkpoints/llama3-coding-lora
    parameters:
      weight: 0.6
      density: 0.7      # DARE: keep 70% of delta params
  - model: ./checkpoints/llama3-math-lora
    parameters:
      weight: 0.4
      density: 0.7
merge_method: ties           # TIES-Merging
base_model: meta-llama/Llama-3.1-8B
parameters:
  normalize: true            # normalize final weights
dtype: bfloat16
```

```python
# Install: pip install mergekit
# Run merge:
# mergekit-yaml merge_config.yaml ./merged_model

# Quick evaluation after merge
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Test merged capabilities
prompts = [
    "Write a Python function to compute fibonacci numbers.",
    "Solve: if 3x + 7 = 22, find x.",
]
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=200,
                             temperature=0.1, do_sample=True)
    print(tokenizer.decode(ids[0], skip_special_tokens=True))
    print("---")
```

## Continual Pre-training on Domain Data

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Mix: 80% new domain data, 20% replay from original distribution
domain_data  = load_dataset("your_domain_corpus", split="train")
replay_data  = load_dataset("fineweb", split="train").select(range(50_000))
mixed_dataset = concatenate_datasets([domain_data, replay_data])
mixed_dataset = mixed_dataset.shuffle(seed=42)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized = mixed_dataset.map(tokenize, batched=True,
                               remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./continual_pretrain",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,           # lower LR for continual training
    warmup_steps=200,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=500,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=collator,
)
trainer.train()
```

This document reflects the state of the art as of April 2026. The field evolves rapidly; techniques described here represent current best practices.



---

[← Previous Chapter](app_c_hardware.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](app_e_cloud.md)