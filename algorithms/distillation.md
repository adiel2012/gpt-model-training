# Knowledge Distillation

Knowledge distillation transfers capability from a large teacher model to a smaller student model. Methods differ in what signal is transferred (outputs, logits, internal features) and how generation is handled (offline vs. on-policy).

---

## Response Distillation (Output-Level)

The simplest approach: generate responses from the teacher, then fine-tune the student via supervised learning.

$$
\mathcal{L}_\text{response} = -\sum_t \log p_S(y_t \mid x, y_{<t})
$$

where $y$ is sampled from the teacher $p_T$.

**Variants:**
- **Single-response:** One teacher output per prompt.
- **Best-of-N:** Generate $N$ teacher responses, keep only correct/highest-quality ones (rejection sampling).
- **Chain-of-thought distillation:** Teacher generates with reasoning traces; student learns to replicate the trace structure.

**Limitation:** The student only sees correct final outputs — no signal about teacher uncertainty or near-miss reasoning.

---

## Logit-Level Distillation (KD)

[hinton2015distilling] Transfer the teacher's full output distribution (soft labels) rather than just the top prediction.

**Loss:**

$$
\mathcal{L}_\text{KD} = (1 - \lambda)\, \mathcal{L}_\text{CE}(y, p_S) + \lambda\, T^2\, \mathcal{D}_\text{KL}(p_T^{(T)} \| p_S^{(T)})
$$

where $p^{(T)}$ denotes softmax at temperature $T > 1$ (softens the distribution):

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

**Why temperature matters:** At $T = 1$, the teacher distribution is nearly one-hot for confident predictions. At $T \in [2, 5]$, the distribution reveals relative probabilities of non-top tokens — the "dark knowledge" that the second-best guess is "cat" when the answer is "dog" tells the student about visual similarity.

**Typical settings:** $T = 2$–$4$, $\lambda = 0.5$–$0.9$.

---

## SeqKD (Sequence-Level Knowledge Distillation)

[kim2016sequence] Transfers knowledge through decoded output sequences rather than token-level distributions. The teacher decodes with beam search; the student trains on the resulting sequences.

**Algorithm:**
1. Run teacher with beam size $B$ on training prompts.
2. Use top-1 beam output as the distillation target.
3. Standard cross-entropy on student.

**Why this works beyond response distillation:** Beam search produces higher-quality outputs than sampling — the student trains on the teacher's best, not average, generation.

**Word-level SeqKD:** Use word-level BLEU/reward as a training signal instead of exact token matching — more flexible for paraphrase and style variation.

---

## MiniLLM (Reverse KL Distillation)

[gu2024minillm] Standard KD minimizes forward KL $\mathcal{D}_\text{KL}(p_T \| p_S)$, which is mean-seeking — the student spreads probability mass over all teacher modes, including ones the student can't model well. MiniLLM instead minimizes **reverse KL**:

$$
\mathcal{L}_\text{MiniLLM} = \mathcal{D}_\text{KL}(p_S \| p_T) = \mathbb{E}_{y \sim p_S}\left[\log \frac{p_S(y \mid x)}{p_T(y \mid x)}\right]
$$

**Effect:** Reverse KL is mode-seeking — the student concentrates on a subset of teacher modes it can model accurately, rather than spreading thinly over all modes.

**Practical result:** Smaller models with MiniLLM produce more focused, coherent outputs. They sometimes outperform forward-KL distillation on open-ended generation.

**Challenge:** Requires on-policy sampling from the student, which is expensive and introduces training instability.

---

## Feature-Map Distillation

Transfer intermediate representations, not just final outputs.

**FitNets [romero2014fitnets]:** Match the student's hidden layer $h_S$ to the teacher's hidden layer $h_T$ (projected to match dimensions):

$$
\mathcal{L}_\text{feat} = \|W_r\, h_S - h_T\|_2^2
$$

**Practical variants:**
- **Layer mapping:** Match every $k$-th student layer to a teacher layer ($k = d_T / d_S$).
- **Normalized features:** Use cosine similarity instead of L2 to avoid scale mismatch.
- **Block-level:** Match transformer block outputs, not individual layer hidden states.

**Limitation:** Requires architectural alignment — student and teacher must have compatible layer structure, or projection matrices must be trained simultaneously.

---

## Attention-Pattern Distillation

Transfer the teacher's attention matrices, not just output representations.

$$
\mathcal{L}_\text{attn} = \frac{1}{H} \sum_{h=1}^H \mathcal{D}_\text{KL}(A_h^T \| A_h^S)
$$

where $A_h$ is the attention matrix for head $h$.

**TinyBERT [jiao2019tinybert]** combines attention distillation with feature distillation across all layers in a two-stage pipeline:
1. General distillation on unlabeled data (attention + feature).
2. Task-specific distillation on labeled data (attention + feature + prediction layer).

---

## On-Policy Distillation (GKD)

[agarwal2024gkd] Standard offline distillation trains on teacher-generated sequences, but the student's own errors compound — it encounters states at inference that it never saw during training. Generalized Knowledge Distillation (GKD) fixes this by mixing on-policy and off-policy data.

**Algorithm:**
1. With probability $\lambda$: use teacher-generated sequence → standard KD loss.
2. With probability $1 - \lambda$: sample from student → compute $\mathcal{D}_\text{KL}(p_T \| p_S)$ on student's own generations.

**Loss (on-policy component):**

$$
\mathcal{L}_\text{on-policy} = \mathbb{E}_{y \sim p_S}\left[\sum_t \log \frac{p_T(y_t \mid x, y_{<t})}{p_S(y_t \mid x, y_{<t})}\right]
$$

**Effect:** The student learns to correct errors in its own distribution, not just imitate teacher sequences. Particularly important for long-form generation where early errors cascade.

---

## Speculative Decoding as Distillation

A form of inference-time distillation: a small draft model proposes tokens; a large verifier model accepts or rejects them. The accepted sequence matches what the large model would have produced.

- Draft model: small, fast, fine-tuned to approximate the large model's distribution.
- Can be trained explicitly via distillation from the large model.
- See `inference_optimization.md` for full speculative decoding details.

---

## Comparison

| Method | Signal | Data Needed | Quality | Cost |
|---|---|---|---|---|
| Response KD | Hard labels | Teacher generations | Good | Low |
| Logit KD | Soft distributions | Teacher logits | Better | Medium |
| SeqKD | Beam-decoded sequences | Teacher decoding | Good | Medium |
| MiniLLM | Reverse KL (mode-seeking) | On-policy student | Best for open-ended | High |
| Feature distillation | Hidden states | Teacher internals | Architectural fit req. | Medium |
| Attention distillation | Attention patterns | Teacher internals | Good (+ features) | Medium |
| GKD | On + off policy mixed | Mixed | Best for long-form | High |

**Practical recommendation:** Start with response distillation (cheapest). Add logit-level KD if the teacher logits are accessible. Use GKD for long-form generation where compounding errors matter.
