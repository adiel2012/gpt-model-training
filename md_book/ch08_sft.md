# Chapter 8: Supervised Fine-Tuning (SFT)

> [!IMPORTANT]
> **What You Will Learn**
> - Understand when SFT helps, when it hurts, and the "less is more" principle.
> - Implement chat templates and loss masking correctly.
> - Master LoRA, QLoRA, and DoRA for parameter-efficient fine-tuning.
> - Select the right PEFT configuration for your model size and hardware.

---

## Purpose and Principles

SFT transitions the pre-trained base model to an instruction-following assistant. It is stable, inexpensive, and immediately effective — but cannot resolve preference trade-offs or handle long-tail failures alone.

**What SFT teaches:** Format (how to structure responses), tone (helpful, concise, safe), task execution (follow multi-step instructions).

**What SFT cannot teach:** Nuanced preference trade-offs between competing values, long-horizon reasoning, or behaviors that require trial-and-error learning. These require preference optimization (Chapter 9) and RL (Chapters 9–10).

> [!WARNING]
> **Heavy SFT hurts RL alignment.** Meta's Llama 4 team found that training on too many SFT examples before online RL made the policy rigid and harder to improve with GRPO. They removed 50%+ of SFT examples (using Llama-as-judge quality filtering) before RL. The lesson: SFT should establish baseline behavior, not over-specify it.

---

## Data Quality: Less Is More

The Llama 2 team found that **27K high-quality examples outperform 1M noisy examples**. This is the "LIMA principle" (Less Is More for Alignment): surface-level alignment is learnable from a small, carefully curated dataset.

### Quality Dimensions

| Dimension | Description | How to Enforce |
| :--- | :--- | :--- |
| Diversity | Cover the full distribution of user intents | Cluster prompts; ensure all cluster centroids are represented |
| Accuracy | Responses must be verifiably correct | Human review for factual claims; verifier for math/code |
| Format compliance | Consistent chat templates and response structure | Template validation; length distribution checks |
| Difficulty calibration | Include hard multi-step tasks; remove trivially easy ones | IFD scoring; reject lowest-difficulty quartile |
| Uniqueness | Avoid near-duplicate responses | MinHash or embedding dedup on responses |

### Quality Filtering Techniques

- **LLM-as-judge:** Prompt a strong model (GPT-4, Claude) to score each (instruction, response) pair 1–5. Keep top 70–80%.
- **IFD (Instruction Following Difficulty):** Score how difficult each instruction is; prioritize medium-to-hard difficulty.
- **Perplexity-based:** Remove examples the base model already generates near-perfectly (trivially easy) and those with extreme perplexity (possibly corrupted).
- **Human annotation of a seed set:** Label 500–2K examples; use these to train a quality classifier.

---

## Chat Templates and Loss Masking

Chat templates define how multi-turn conversations are serialized into token sequences. **Consistency between training and inference is critical** — a mismatch causes format errors and degraded performance.

### Llama 3 Format

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of France?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Paris.<|eot_id|>
```

### Loss Masking

Compute cross-entropy loss **only on assistant tokens**. Mask system and user tokens with `-100`:

```python
def apply_loss_mask(input_ids, tokenizer):
    """Set labels to -100 for all non-assistant tokens."""
    labels = input_ids.clone()
    # Find assistant token ranges and mask everything outside them
    assistant_start = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    # ... mask logic: labels[non_assistant_positions] = -100
    return labels
```

> [!WARNING]
> **Failure to mask causes the model to generate user messages.** A common symptom: the model injects `<|user|>` or `Human:` tokens mid-response. Always verify your masking implementation on a few examples before training.

---

## Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of a 70B model requires ~560 GB of optimizer state (AdamW). PEFT methods reduce this to 1–5% of parameters while maintaining near-full-fine-tuning quality.

Full formulations and implementations: [Appendix G](app_g_implementation_treasury.md).

### LoRA (Low-Rank Adaptation)

Hu et al. (2021). Adds trainable low-rank matrices to frozen weight matrices:

$$h = W_0 x + \frac{\alpha}{r}\,BAx, \qquad A \in \mathbb{R}^{r \times d_\mathrm{in}},\; B \in \mathbb{R}^{d_\mathrm{out} \times r}$$

$B$ is initialized to zero — no change to the model at the start of training. Only $A$ and $B$ are updated; $W_0$ is frozen.

**Configuration guide:**

| Model Size | Recommended Rank $r$ | Target Modules | Notes |
| :--- | :--- | :--- | :--- |
| 1B–7B | 8–16 | q\_proj, v\_proj | Minimal overhead |
| 7B–13B | 16–32 | q\_proj, k\_proj, v\_proj, o\_proj | Better coverage |
| 30B–70B | 32–64 | All attention + FFN gates | Higher rank needed at scale |
| 70B+ | 64–128 | All linear layers | DoRA recommended |

### QLoRA (Quantized LoRA)

Dettmers et al. (2023). Load the base model in **4-bit NF4** quantization; add BF16 LoRA adapters on top. Reduces VRAM from ~140 GB (70B BF16) to ~35 GB — fits on a single A100-40GB.

Key components:
- **NF4 quantization:** Normally distributed quantization grid, optimal for normally distributed weights.
- **Double quantization:** Quantize the quantization constants themselves, saving an additional 0.37 bits/param.
- **Paged optimizer:** Offload optimizer states to CPU RAM when GPU memory is full.

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Liu et al. (2024). Decomposes the weight update into **magnitude** and **direction** components separately:

$$W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

where $m$ is a learnable per-column magnitude vector and $\|\cdot\|_c$ is the column-wise norm.

DoRA matches or exceeds full fine-tuning on several benchmarks where LoRA falls short — especially on tasks requiring both format and content changes (e.g., style transfer + factual accuracy).

### Spectrum

Signal-to-noise ratio analysis identifies the most informative layers for fine-tuning. Instead of uniformly applying LoRA to all layers, Spectrum selects the top-$k$ layers by SNR and fine-tunes only those — better ROI per updated parameter. Useful when compute is severely limited.

### PEFT Method Comparison

| Method | Extra Params | VRAM vs Full FT | Quality vs Full FT | Best For |
| :--- | :--- | :--- | :--- | :--- |
| Full fine-tuning | 100% | Baseline | Baseline | Small models, highest quality needed |
| LoRA (r=16) | ~0.5% | −85% | 90–95% | General purpose |
| QLoRA (r=16) | ~0.5% | −95% | 85–92% | Large models, limited hardware |
| DoRA (r=16) | ~0.5% + m | −84% | 95–98% | Quality-critical tasks |
| Spectrum | Varies | −70–90% | 88–95% | Compute-constrained partial tuning |

---

## Multi-Turn SFT

Single-turn SFT (one instruction, one response) is insufficient for assistant models that must maintain coherent multi-turn conversations.

**Key considerations:**
- **History truncation:** When conversation history exceeds context length, truncate from the oldest turns first, always keeping the system prompt.
- **Role consistency:** The model must learn that the assistant role never asks questions that belong to the user role.
- **Context window packing:** Pack multiple short conversations into a single training example using separator tokens and cross-conversation attention masking.

---

## SFT Training Recipe

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# 4-bit quantization for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,507,392 || 0.52%

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch = 16
        bf16=True,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
    ),
)
trainer.train()
```

---

[← Previous Chapter](ch07_distributed_training.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](ch09_alignment.md)
