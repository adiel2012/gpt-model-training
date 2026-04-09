\part{Post-Training -- From Base Model to Product}

# Supervised Fine-Tuning (SFT)

> **What You Will Learn**
> - Master instruction tuning requirements and token-masking techniques.
> - Implement parameter-efficient fine-tuning (PEFT) with LoRA and QLoRA.
> - Analyze the transition to DoRA and rank-adaptive PEFT in 2026.
> - Understand the 2025 shift from large instruction sets to high-quality "Less Is More" (LIMA) datasets.

## Purpose and Principles

SFT transitions the base model to an instruction-following assistant. It is stable, inexpensive, and immediately effective---but cannot resolve preference trade-offs or handle long-tail failures alone.

## Data Quality for SFT

The Llama~2 team [touvron2023llama] found that 27K high-quality examples outperform 1M noisy examples. Key quality dimensions:

  - **Diversity:** Cover the full distribution of user intents.
  - **Accuracy:** Ground-truth responses must be verifiably correct.
  - **Format compliance:** Consistent chat templates and proper system prompts.
  - **Difficulty calibration:** Include challenging multi-step tasks; trivially easy examples waste model capacity.

Quality filtering approaches: LLM-as-judge scoring, perplexity-based selection, human annotation of a seed set, IFD (Instruction Following Difficulty) scoring.

## Chat Templates

Chat templates define how multi-turn conversations are serialized into token sequences. Consistent templates between training and inference are critical:

```python
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of France?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

Loss masking: compute cross-entropy loss *only* on assistant tokens, not on system or user tokens. Failure to mask causes the model to learn to generate user messages.

## Parameter-Efficient Fine-Tuning (PEFT)

Full derivations in [Appendix G](app_g_implementation_treasury.md): LoRA, DoRA; code: [Appendix G](app_g_implementation_treasury.md).

  - **LoRA [hu2022lora**:] Low-rank matrices on frozen weights. $<$1% extra parameters. Dominant PEFT method.
  - **QLoRA [dettmers2023qlora**:] LoRA + 4-bit quantization. Enables 70B+ models on a single consumer GPU.
  - **DoRA [liu2024dora**:] Weight-Decomposed Low-Rank Adaptation. Decomposes updates into magnitude and direction components---matches or exceeds full fine-tuning on several benchmarks.
  - **Spectrum:** Signal-to-noise ratio analysis identifies the most informative layers for fine-tuning, enabling partial fine-tuning with better ROI per updated parameter.

