# Alignment -- RLHF, DPO, and Beyond

> **What You Will Learn**
> - Trace the evolution from RLHF to reference-free methods (DPO, SimPO).
> - Implement alignment with binary feedback (KTO) and group-relative rewards (GRPO).
> - Evaluate the role of verifiable rewards (RLVR) in reasoning-heavy models.
> - Understand the 2026 modular alignment stack for safety and helpfulness.

## The Alignment Problem

A grammatically perfect model may be unhelpful, evasive, or unsafe. The field has evolved from RLHF alone to a modular stack of complementary methods.

## Reward Modeling

Complete objectives in [Appendix G](app_g_implementation_treasury.md): PPO, DPO, SimPO, KTO, GRPO.

The reward model (RM) translates human preferences into scalar scores. RM quality determines the ceiling for all RM-based alignment methods.

  - **Architecture:** Language model with a regression head replacing the LM head.
  - **Training:** Bradley-Terry model on pairwise comparisons [ziegler2019finetuning]: $p(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$.
  - **Reward hacking:** An RM optimized too strongly diverges from true human preferences. KL penalty and early stopping mitigate this.
  - **Process Reward Models (PRM) [lightman2023lets**:] Score each intermediate reasoning step, not just the final output. More signal, less sparse. Used in OpenAI's o-series for math and coding.

## RLHF

Transformed GPT-3 into ChatGPT [ouyang2022training]. Pipeline: SFT $\rightarrow$ Reward Model $\rightarrow$ PPO [schulman2017proximal]. Essential for open-ended quality but requires four models in memory simultaneously and is prone to reward hacking.

## DPO

Eliminates the reward model entirely [rafailov2023direct] (formula [Appendix G](app_g_implementation_treasury.md), code [Appendix G](app_g_implementation_treasury.md)). 90--95% of RLHF performance at 40--60% less compute.

> **DPO Successors**
>
> - **SimPO [meng2024simpo**:] Eliminates the reference model entirely. It uses length-normalized log probabilities as an implicit reward and introduces a target reward margin $\gamma$ (see formulation [Appendix G](app_g_implementation_treasury.md), code [Appendix G](app_g_implementation_treasury.md)).
>   - **KTO (Kahneman-Tversky Optimization) [ethayarajh2024kto**:] Operates on binary thumbs-up/down signals rather than preferences. It maximizes the utility of model outputs by aligning with human loss aversion (see formulation [Appendix G](app_g_implementation_treasury.md), code [Appendix G](app_g_implementation_treasury.md)).
>   - **Iterative DPO:** Semi-online re-sampling from the current policy. Closes the distribution gap between training data and current model outputs.
>   - **IPO [azar2024ipo**:] Information-Preference Optimization. Regularizes DPO to prevent overfitting to preference margins.

## GRPO

DeepSeek's dominant algorithm for reasoning [guo2025deepseekr1] (formula [Appendix G](app_g_implementation_treasury.md), code [Appendix G](app_g_implementation_treasury.md)). Generates a group of 8--64 responses per prompt. Advantages normalized against group statistics. Eliminates the critic model, yielding 33--50% memory savings over PPO.

$$
\mathcal{L}_\text{GRPO} = \mathbb{E}\left[\sum_i \hat{A}_i \cdot \log \pi_\theta(o_i \mid q) - \beta \cdot \text{KL}[\pi_\theta \| \pi_\text{ref}]\right]
$$

where $\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ is the group-normalized advantage.

## DAPO

Removes reference model dependency from GRPO [yu2025dapo] (formulation [Appendix G](app_g_implementation_treasury.md), code [Appendix G](app_g_implementation_treasury.md)). The most streamlined RL alignment approach. Also introduces *clip-higher* (asymmetric policy clipping) and token-level loss normalization to stabilize training.

## RLAIF and Constitutional AI

AI-generated preferences replace human annotations. Anthropic's CAI [bai2022constitutional]: model self-critiques against constitutional principles, creating scalable alignment data without per-example human annotation.

## RLVR

Verifiable rewards for math and code. Binary, objective signal resists reward hacking. DeepSeek R1 demonstrated emergent reasoning from pure RLVR + GRPO without SFT (formulation [Appendix G](app_g_implementation_treasury.md), code [Appendix G](app_g_implementation_treasury.md)). The key insight: verifiable correctness is a cleaner training signal than a learned reward model.

## The Modern Alignment Stack

| **Alignment Goal** | **Method** | **What It Solves** |
|---|---|---|
| Instruction Following | SFT | Format, tone, task execution |
| Preference Alignment | DPO / SimPO / KTO | Human values, norms |
| Reasoning | GRPO / DAPO + RLVR | Math, code, planning |
| Safety \ | Helpfulness | RLHF (neural RM) | Open-ended quality, harm avoidance |
| Constitutional Guardrails | CAI / RLAIF | Scalable safety principles |

*Table: The modular alignment stack in 2026*

