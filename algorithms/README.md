# Algorithms Reference

Detailed explanations of every major algorithm referenced across the chapters. Each file covers the mathematical formulation, intuition, key hyperparameters, and comparison with alternatives.

---

## Files

### Core Architecture

- **[attention_mechanisms.md](attention_mechanisms.md)** — Scaled dot-product attention, MHA, MQA, GQA, MLA (DeepSeek), Flash Attention (v1/v2/v3), Sliding Window, Ring Attention.

- **[positional_encodings.md](positional_encodings.md)** — Sinusoidal, RoPE (rotation matrix + code), iRoPE (Llama 4), ALiBi, YaRN / LongRoPE (NTK-aware interpolation).

### Training

- **[optimizers.md](optimizers.md)** — AdamW (decoupled weight decay), Muon (Newton-Schulz orthogonalization), Lion (sign-based), Sophia (Hessian curvature), Cosine LR schedule, WSD schedule, gradient clipping.

- **[peft.md](peft.md)** — LoRA (low-rank adaptation + code), QLoRA (NF4, double quantization, paged optimizers), DoRA (magnitude+direction decomposition), Spectrum (SNR-based layer selection).

- **[parallelism.md](parallelism.md)** — Data Parallelism, ZeRO (stages 1/2/3), FSDP, Tensor Parallelism, Pipeline Parallelism (1F1B schedule), Sequence Parallelism, 3D Parallelism, activation checkpointing.

### Post-Training & Alignment

- **[alignment_objectives.md](alignment_objectives.md)** — Bradley-Terry model, PPO (clipped objective, 4-model setup), DPO (derivation), SimPO (length normalization, reference-free), KTO (Kahneman-Tversky, binary feedback), GRPO (group-normalized advantages), DAPO (clip-higher, token normalization), RLVR (verifiable rewards), Constitutional AI.

- **[reasoning_techniques.md](reasoning_techniques.md)** — Chain-of-Thought (zero-shot, few-shot), Self-Consistency, Process Reward Models (PRM vs ORM), Best-of-N / Rejection Sampling, MCTS (UCB + value function), Test-Time Compute Scaling, Extended Thinking / Thinking Tokens, R1-style Pure RL.

- **[distillation.md](distillation.md)** — Response distillation, Logit-level KD (Hinton, temperature scaling), SeqKD, MiniLLM (reverse KL, mode-seeking), Feature-map distillation (FitNets), Attention-pattern distillation (TinyBERT), On-policy GKD.

### Inference

- **[inference_optimization.md](inference_optimization.md)** — INT8 / INT4 / GPTQ / GGUF / FP8 / BitNet quantization, Speculative Decoding (rejection sampling proof), Medusa heads, PagedAttention, Continuous Batching, Prefix Caching / Radix Attention, KV Cache Quantization.

### Advanced

- **[model_merging.md](model_merging.md)** — Task Vectors, Linear Interpolation (Model Soup), SLERP, TIES-Merging (trim-elect-merge), DARE (drop-and-rescale), Model Breadcrumbs.

- **[data_methods.md](data_methods.md)** — Self-Instruct, Evol-Instruct, Magpie, Orca / Orca 2, Persona Hub, MinHash deduplication, SemDedup, quality filtering, curriculum learning, data mixture weighting, decontamination.

---

## Quick Reference by Task

| I want to... | See |
|---|---|
| Fine-tune a 70B model on one GPU | [peft.md → QLoRA](peft.md) |
| Extend context beyond training length | [positional_encodings.md → YaRN](positional_encodings.md) |
| Align model with preference data | [alignment_objectives.md → DPO / SimPO](alignment_objectives.md) |
| Train a reasoning model | [alignment_objectives.md → GRPO / RLVR](alignment_objectives.md) |
| Improve math accuracy at inference | [reasoning_techniques.md → Best-of-N, MCTS](reasoning_techniques.md) |
| Compress a model for deployment | [inference_optimization.md → GPTQ, GGUF](inference_optimization.md) |
| Speed up inference 2-3× | [inference_optimization.md → Speculative Decoding](inference_optimization.md) |
| Merge multiple fine-tuned models | [model_merging.md → TIES, DARE](model_merging.md) |
| Train at 100B+ scale | [parallelism.md → 3D Parallelism](parallelism.md) |
| Generate synthetic training data | [data_methods.md → Self-Instruct, Magpie](data_methods.md) |
| Reduce memory during training | [parallelism.md → FSDP / ZeRO-3](parallelism.md) |
| Transfer knowledge to small model | [distillation.md → GKD, SeqKD](distillation.md) |
