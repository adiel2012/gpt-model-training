# Case Studies and Frontier Lab Analysis

> **What You Will Learn**
> - Analyze the architectural and data choices of Llama 4, GPT-5, and DeepSeek-V4.
> - Compare the alignment strategies of Anthropic (Constitutional AI) vs. OpenAI.
> - Review the open-weights development landscape at the 400B+ scale.
> - Reflect on the hardware-software co-design trends at Google, NVIDIA, and Meta.

## OpenAI -- GPT-5 Family

> **OpenAI -- GPT-5 / o3 / o4-mini**
> **Architecture:** Dense transformer (rumored MoE), Azure AI supercomputers. Unified intelligence, reasoning, coding, multimodality.
> 
> **Pre-training:** Massive-scale unsupervised learning. GPT-4.5: data from smaller models. GPT-5: 94.6% AIME, 74.9% SWE-bench, 84.2% MMMU.
> 
> **Post-training:** SFT + RLHF + ``safe completions'' (output-centric safety). Reasoning via dedicated RL chains of thought. Process reward models score intermediate reasoning steps.
> 
> **Deployment:** Real-time router (instant vs.\ deep reasoning). $\sim$80% fewer factual errors, 50--80% fewer output tokens.

## DeepSeek -- R1 and V3

> **DeepSeek -- R1 / V3**
> **Architecture:** Fine-grained MoE, shared expert isolation, multi-head latent attention (MLA).
> 
> **Pre-training:** Trillions of tokens at $\sim$$5.5M.
> 
> **R1 pipeline [guo2025deepseekr1** (4 stages):]
> [leftmargin=*]
> - Cold-Start SFT (thousands of examples)
> - RLVR + GRPO (16 responses/prompt, rule-based rewards, no neural RM)
> - Rejection Sampling + SFT Stage 2 (800K filtered examples)
> - Final RL: RLVR (reasoning) + RLHF (helpfulness/safety with separate RMs)
> 
> **Results:** R1-Zero AIME: 15.6% $\rightarrow$ 71.0% via pure RL. Emergent self-verification, ``aha moments.''
> 
> **Distillation:** 1.5B--70B models. 7B: 55.5% AIME 2024.

## Meta -- Llama 4 Family

> **Meta -- Llama 4 Scout / Maverick / Behemoth**
> **Architecture:** First Llama MoE. Scout: 17B/109B (16 exp.). Maverick: 17B/400B (128 exp.). Behemoth: 288B/$\sim$2T. iRoPE: 10M-token context.
> 
> **Pre-training:** Early fusion multimodality (text + image + video from start). MetaCLIP vision encoder.
> 
> **Post-training (key insight: heavy SFT hurts RL):**
> [leftmargin=*]
> - Lightweight SFT: removed 50%+ easy examples (Llama-as-judge)
> - Online RL: continuous, adaptive filtering, curriculum design
> - Lightweight DPO: final polish
> - Behemoth codistillation: teacher $\rightarrow$ Scout/Maverick
> 
> **Deployment:** Scout on 1$\times$H100 (INT4). 9--23$\times$ price-performance vs.\ GPT-4o.

## Anthropic -- Claude Family

> **Anthropic -- Claude 4 Family**
> **Architecture:** Dense transformer. Details not publicly disclosed.
> 
> **Constitutional AI (CAI):**
> [leftmargin=*]
> - Phase 1 (SL-CAI): Generate $\rightarrow$ self-critique against principles $\rightarrow$ revise. Revised outputs become SFT data.
> - Phase 2 (RL-CAI): Revised outputs train preference model for RLHF. RLAIF reduces human annotation needs.
> 
> **Safety:** Classifiers, harmlessness RMs, extensive red-teaming. Honest, harmless, helpful.

## Google DeepMind -- Gemini Family

> **Google DeepMind -- Gemini 2.5**
> **Architecture:** Natively multimodal transformer on TPU v5. Text, images, audio, video, code unified.
> 
> **Post-training:** SFT + RLHF + reasoning RL + ``thinking'' mode. Distillation: Pro $\rightarrow$ Flash $\rightarrow$ Nano.
> 
> **Deployment:** Google products, Vertex AI, on-device Nano.

## Alibaba / Qwen -- Qwen 3 Family

> **Alibaba -- Qwen 3 / Qwen 3-VL**
> **Architecture:** Dense + MoE. Qwen3-VL: vision-language MoE (30B-A3B, 235B-A22B).
> 
> **Post-training:** GSPO (Group Sequence Policy Optimization). ``Thinking'' toggle.
> 
> **Open ecosystem:** Full open weights 0.6B--235B. Popular base for R1 distillation.

## Cross-Cutting Patterns

> **Converging Trends Across Frontier Labs**
>
> - **Modular post-training:** Distinct stages, not monolithic. Heavy SFT hurts RL (Meta).
>   - **MoE is default:** Meta, DeepSeek, Qwen, rumored closed labs.
>   - **GRPO displaces PPO:** Critic-free, 33--50% memory savings.
>   - **Distillation is first-class:** Behemoth$\rightarrow$Scout, R1$\rightarrow$1.5B--70B, Pro$\rightarrow$Flash$\rightarrow$Nano.
>   - **Reasoning via RL:** Pure RL (DeepSeek), unified routing (OpenAI), thinking modes (Google/Qwen).
>   - **Safety diverges productively:** Safe completions, CAI, Llama-as-judge, separate RMs.

| **Company** | **Architecture** | **Post-Training Pipeline** | **Key Innovation** |
|---|---|---|---|
| OpenAI | Dense (MoE?) | SFT $\rightarrow$ Safe Compl.\ $\rightarrow$ RLHF $\rightarrow$ Reasoning RL | Safe completions |
| DeepSeek | Fine MoE | Cold SFT $\rightarrow$ GRPO $\rightarrow$ Rej.\ Samp.\ $\rightarrow$ SFT2 $\rightarrow$ RLHF+RLVR | Pure RL reasoning |
| Meta | MoE (128 exp.) | Light SFT $\rightarrow$ Online RL $\rightarrow$ Light DPO | Light SFT + heavy RL |
| Anthropic | Dense | SFT $\rightarrow$ CAI $\rightarrow$ RLHF/RLAIF | Constitutional AI |
| Google | Multimodal | SFT $\rightarrow$ RLHF $\rightarrow$ Reas.\ RL $\rightarrow$ Distill. | Native multimodality |
| Qwen | Dense+MoE | SFT $\rightarrow$ GSPO $\rightarrow$ Reasoning RL | GSPO, multilingual |

*Table: Comparison of frontier lab training approaches*

% ══════════════════════════════════════════════════════════════════
%  APPENDICES
% ══════════════════════════════════════════════════════════════════
