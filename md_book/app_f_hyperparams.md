# Training Hyperparameter Reference

A consolidated reference for common LLM training scenarios.

## Pre-training

[H]
L{2.5cm}L{2.5cm}L{3cm}}
|  | **70B Model** | **Notes** |
|---|---|---|---|
| Optimizer | AdamW | AdamW | Muon competitive at 7B |
| Learning rate (peak) | $3\times10^{-4}$ | $1\times10^{-4}$ | Scale down with model size |
| LR scheduler | Cosine + warmup | Cosine + warmup | WSD alternative |
| Warmup steps | 2,000 | 4,000 | More warmup for larger models |
| $\beta_1$ | 0.9 | 0.9 | Standard |
| $\beta_2$ | 0.95 | 0.95 | Slightly lower than Adam default |
| Weight decay | 0.1 | 0.1 | Higher than traditional 0.01 |
| Gradient clip norm | 1.0 | 1.0 | Standard |
| Batch size (tokens) | 4M | 8M | Scale with model size |
| Sequence length | 4,096 | 4,096 | Extend in Stage 2 |
| Precision | BF16 | BF16 / FP8 | FP8 for H100/H200 |

*Table: Pre-training hyperparameter reference*

## Supervised Fine-Tuning (SFT)

[H]
L{3cm}L{5cm}}
|  | **Notes** |
|---|---|---|---|
| Learning rate | $1\times10^{-5}$ -- $5\times10^{-5}$ | Lower than pre-training; avoid forgetting |
| Epochs | 2--5 | 3 is the most common sweet spot |
| Batch size | 32--128 | Larger = more stable; use grad accum |
| Scheduler | Cosine or linear decay | Linear often sufficient for short runs |
| Warmup ratio | 0.03--0.1 | 3--10\% of total steps |
| Max sequence length | 2,048--8,192 | Pack shorter sequences |
| Loss masking | On (assistant only) | Critical: never train on user/system tokens |

*Table: SFT hyperparameter reference*

## LoRA / QLoRA

[H]
L{3cm}L{5cm}}
|  | **Notes** |
|---|---|---|---|
| Rank $r$ | 8 -- 64 | 16 is the most common default; higher $r$ = more params |
| Alpha $\alpha$ | $2r$ (e.g., 32 for $r$=16) | $\alpha/r$ is the effective LR scale (usually 2) |
| Dropout | 0.05 -- 0.1 | Low dropout for small datasets |
| Target modules | q\_proj, v\_proj | Add k\_proj, o\_proj for harder tasks |
| Learning rate | $1\times10^{-4}$ -- $3\times10^{-4}$ | Higher than full FT; adapter weights start near zero |
| QLoRA quantization | nf4 (4-bit NormalFloat) | nf4 outperforms fp4 for most LLMs |
| Compute dtype | BF16 | Always use BF16 for LoRA computation |

*Table: LoRA and QLoRA hyperparameter reference*

## DPO / SimPO Alignment

[H]
L{3cm}L{5cm}}
|  | **Notes** |
|---|---|---|---|
| $\beta$ (KL penalty) | 0.01 -- 0.5 | Lower $\beta$ = more aggressive policy update |
| Learning rate | $5\times10^{-7}$ -- $1\times10^{-5}$ | Much lower than SFT |
| Epochs | 1 -- 3 | Over-training causes reward hacking |
| Batch size | 8 -- 64 | Larger = more stable preference gradient |
| Max length | 1,024 -- 4,096 | Truncate long responses symmetrically |
| Chosen/rejected ratio | 1:1 | Balanced preference pairs |
| SimPO $\gamma$ margin | 0.5 -- 2.0 | Target reward margin between chosen and rejected |

*Table: DPO and SimPO hyperparameter reference*

## GRPO Reasoning Training

[H]
L{3cm}L{5cm}}
|  | **Notes** |
|---|---|---|---|
| Group size $G$ | 8 -- 16 | DeepSeek uses 16; larger = more stable advantage estimate |
| $\beta$ (KL weight) | 0.001 -- 0.01 | Very small; reward signal dominates |
| Temperature (sampling) | 0.7 -- 1.0 | Higher = more diverse generations for better advantage estimation |
| Max new tokens | 512 -- 4096 | Match the complexity of reasoning tasks |
| Learning rate | $1\times10^{-6}$ -- $5\times10^{-6}$ | Lower than SFT; RL updates are noisier |
| Reward normalization | Group mean/std | Normalize within the group, not globally |
| Clip ratio $\varepsilon$ | 0.2 | Standard PPO-style clipping |

*Table: GRPO hyperparameter reference*

## Common Failure Modes and Fixes

[H]
L{3cm}L{4.5cm}}
|  | **Fix** |
|---|---|---|---|
| Loss NaN early in training | LR too high or warmup too short | Halve LR; double warmup steps |
| Loss plateau without improvement | LR too low or dataset too small | Increase LR by 2$\times$; check data quality |
| Loss spike then recovery | Corrupted data batch | Add data validation; inspect spike step's batch |
| Gradient norm spikes | Outlier data or no clipping | Enable grad clip norm 1.0; filter data |
| Model repeats outputs | Training data contamination or over-training | Reduce epochs; check for dedup failures |
| DPO loss goes negative | $\beta$ too small or LR too high | Increase $\beta$; reduce LR by 5$\times$ |
| GRPO reward stuck at zero | Reward function too strict or group too small | Soften reward; increase group size |
| OOM during GRPO | Too many generations in memory | Reduce group size; use gradient checkpointing |

*Table: Common training failure modes and remediation*

