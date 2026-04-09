# Knowledge Distillation and Model Compression

> **What You Will Learn**
> - Master the forward and reverse KL distillation loss functions.
> - Implement sequence-level distillation for complex reasoning tasks.
> - Analyze the trade-offs between student model size and knowledge retention.
> - Evaluate the role of speculative decoding as a dynamic distillation strategy.

Knowledge distillation (KD) transfers capability from a large *teacher* model to a smaller *student*, achieving performance beyond what the student could reach by training on ground-truth labels alone.

## Why Distillation Matters

Complete derivations in [Appendix G](app_g_implementation_treasury.md): forward KL, reverse KL (MiniLLM), speculative decoding acceptance; code: [Appendix G](app_g_implementation_treasury.md).

Frontier labs use distillation systematically: Behemoth $\rightarrow$ Scout/Maverick (Meta), R1-671B $\rightarrow$ 1.5B--70B (DeepSeek), Gemini Pro $\rightarrow$ Flash $\rightarrow$ Nano (Google). Distilled 7B models now routinely outperform undistilled 70B models on target tasks.

## Response (Output-Level) Distillation

The simplest form: generate responses from the teacher and use them as SFT labels for the student. The student minimizes the negative log-likelihood of teacher completions rather than human-annotated completions.

$$
\mathcal{L}_\text{resp} = -\sum_t \log p_\theta(y_t^\text{teacher} \mid x, y_{<t}^\text{teacher})
$$

**Advantages:** Simple pipeline; works with closed-source teachers (you only need teacher outputs, not logits). **Disadvantage:** Ignores the full teacher probability distribution---the student does not learn ``how uncertain'' the teacher is.

## Logit-Level Distillation

When teacher logits are available (open-weight teachers), the student matches the teacher's full output distribution via KL divergence:

$$
\mathcal{L}_\text{KD} = \alpha \cdot \mathcal{L}_\text{CE}(y^\text{gt}, p_\theta) + (1-\alpha) \cdot T^2 \cdot D_\text{KL}\!\left(\text{softmax}(z^\text{teacher}/T) \;\|\; \text{softmax}(z^\text{student}/T)\right)
$$

where $T$ is the temperature (commonly 2--4) and $\alpha$ balances ground-truth and teacher loss. Higher temperature softens the teacher distribution, providing richer signal on near-correct tokens.

## Sequence-Level Knowledge Distillation (SeqKD)

Rather than optimizing token-level KL, SeqKD [kim2016sequence] generates *mode sequences* from the teacher (beam search, temperature $= 0$) and treats them as hard labels. More cache-friendly than online logit distillation; works without access to logits. Used extensively when distilling from GPT-4 or Claude.

## MiniLLM: Reverse KL for Distillation

Standard KD minimizes forward KL $D_\text{KL}(p_\text{teacher} \| p_\text{student})$, which forces the student to cover all teacher modes and leads to ``mean-seeking'' behaviour. MiniLLM [gu2024minillm] minimizes reverse KL $D_\text{KL}(p_\text{student} \| p_\text{teacher})$, encouraging mode-seeking: the student becomes sharp and accurate on the teacher's most probable outputs, avoiding the diffusion of probability mass across irrelevant alternatives. Reverse KL is estimated via policy gradient on student-generated sequences with teacher log-probability as reward.

## Intermediate Layer Distillation

Align hidden states or attention patterns at intermediate layers, not just the output:

  - **Feature-map distillation (PKD, TinyBERT):** $\mathcal{L}_\text{feat} = \|W_\text{proj} h^\text{student}_l - h^\text{teacher}_{f(l)}\|_2^2$ where $f(l)$ maps student layer to teacher layer index and $W_\text{proj}$ is a learned projection.
  - **Attention-pattern distillation:** $\mathcal{L}_\text{attn} = D_\text{KL}(A^\text{teacher}_{l,h} \| A^\text{student}_{l,h})$ summed over heads and layers.
  - **Practical note:** Intermediate distillation requires matching architectures or learned projections. Most practical pipelines stick to output-level or logit-level distillation.

## On-Policy Distillation

A key failure mode of offline response distillation: the student is trained on teacher trajectories but evaluated on *its own* trajectories. Distribution mismatch grows with model capability gap. On-policy distillation:

  - Roll out the *student* to generate candidate completions.
  - Score each completion with the teacher (assign log-probability or RM score).
  - Minimize KL between student and teacher-scored distribution.

Requires either white-box access to teacher logits or a fast teacher inference endpoint. Used in GKD [agarwal2024onpolicy] (Generalized Knowledge Distillation).

## Speculative Decoding as Distillation

Speculative decoding [leviathan2023fast] uses a small draft model to propose $k$ tokens, verified in parallel by the target model. Beyond a decoding speedup, training the drafter to maximally predict acceptance by the target is a form of distillation---the drafter learns to approximate the target's conditional distribution on easy tokens, which constitute 70--90% of all tokens.

## Practical Distillation Workflow

  - Select teacher (strongest accessible model: open-weight 70B--671B or API).
  - Generate responses on your instruction set. For reasoning tasks, generate with high temperature ($T = 0.7$--$1.0$) and filter by verifiable correctness.
  - Optionally: collect teacher logits for logit-level KD (requires local teacher).
  - Fine-tune student with $\alpha = 0.5$: 50% KD loss, 50% standard CE loss on human labels.
  - Evaluate on target benchmarks; iterate on temperature and $\alpha$.

> **Distillation Rules of Thumb**
>
> - A 7B student distilled from a 70B teacher typically closes 60--80% of the gap to the teacher.
>   - Distillation data volume: 50K--500K high-quality teacher completions is sufficient for most fine-tuning objectives.
>   - For reasoning tasks, only keep teacher traces that lead to correct verified answers---incorrect reasoning chains hurt more than they help.
>   - On-policy distillation adds significant complexity; prefer offline SeqKD unless you observe severe distribution mismatch.

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits,
                      labels, alpha=0.5, temperature=2.0):
    """
    Combined cross-entropy + KL distillation loss.
    student_logits, teacher_logits: (batch, seq_len, vocab)
    labels: (batch, seq_len)  -100 for masked tokens
    """
    # Standard cross-entropy on ground-truth labels
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # KL divergence on teacher distribution
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = F.kl_div(
        soft_student, soft_teacher,
        reduction="batchmean", log_target=False,
    ) * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kd_loss
```

% ══════════════════════════════════════════════════════════════════
| %  PART V: EVALUATION | DEPLOYMENT |
% ══════════════════════════════════════════════════════════════════
