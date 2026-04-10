# Chapter 6: Pre-training Objectives and Strategies

> [!IMPORTANT]
> **What You Will Learn**
> - Master the next-token prediction (NTP) objective and its 2026 variants.
> - Understand Reinforcement Pre-Training (RPT) and Fill-in-the-Middle (FIM) objectives.
> - Apply curriculum learning and data mixing strategies (DoReMi).
> - Implement instruction-augmented pre-training to bridge to SFT.
> - Design a multi-stage compute-optimal context extension pipeline.

---

## Next-Token Prediction (NTP)

Self-supervised causal language modeling — the core objective for every modern autoregressive LLM. See [Appendix G](app_g_implementation_treasury.md) for the full implementation and perplexity metric.

$$\mathcal{L}_\mathrm{NTP}(\theta) = -\frac{1}{|\mathcal{D}|}\sum_{x \in \mathcal{D}}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

No labeled data is needed. Despite its simplicity, NTP is sufficient to produce emergent capabilities at scale: syntax, semantics, factual knowledge, multi-step reasoning, and in-context learning all emerge from this single objective applied to enough high-quality text.

**Perplexity** is the standard evaluation metric during pre-training:

$$\mathrm{PPL} = \exp(\mathcal{L}_\mathrm{NTP})$$

Lower perplexity = higher confidence predictions. A key property: perplexity improvements on held-out data reliably predict downstream benchmark improvements, making it the primary signal for pre-training health.

> [!NOTE]
> **Why NTP works so well.** To predict the next token accurately, a model must implicitly learn world knowledge (facts), syntax (grammar), and reasoning (multi-step inference). The objective is a proxy for compression — the best compressor of human text is also the most capable model.

---

## Compute-Optimal Training (Chinchilla Scaling)

Before choosing training duration, determine the compute-optimal allocation between model size and token count.

The Chinchilla scaling laws (Hoffmann et al., 2022) show that for a fixed compute budget $C$ (in FLOPs), optimal model size $N$ and training tokens $D$ satisfy:

$$N^* \propto C^{0.5}, \qquad D^* \propto C^{0.5}$$

This means earlier large models (GPT-3, PaLM) were systematically **undertrained** — they had too many parameters for the data they saw.

| Model | Params | Tokens | Chinchilla-Optimal? |
| :--- | :--- | :--- | :--- |
| GPT-3 | 175B | 300B | No — ~5× undertrained |
| Chinchilla | 70B | 1.4T | Yes |
| Llama 3.1 (8B) | 8B | 15T | Deliberately overtrained |
| DeepSeek-V3 | 671B (MoE) | 14.8T | Inference-optimal |

> [!TIP]
> **Inference-optimal vs. compute-optimal.** Frontier labs now deliberately train smaller models on more tokens than Chinchilla prescribes. Reason: the model is deployed at massive scale, so reducing inference cost (smaller model) is worth the extra training compute. A Chinchilla-optimal 70B model is more expensive to serve than an "overtrained" 8B model with comparable quality.

### The Scaling Law Formula

For a transformer trained on $D$ tokens:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

where $L_\infty$ is the irreducible loss from noise in the data, and typical values are $\alpha \approx 0.34$, $\beta \approx 0.28$ (Hoffmann et al., 2022).

---

## Reinforcement Pre-Training (RPT)

Microsoft Research (2025): reformulates next-token prediction as a **sequential decision-making problem**.

Standard NTP: maximize $\log p_\theta(x_t \mid x_{<t})$ for each token independently.

RPT: treat each token prediction as a policy action $a_t = x_t$ with state $s_t = x_{<t}$. The model receives a reward signal based on the quality of the generated sequence, not just local token-level likelihood.

**Key properties:**
- Fully self-supervised — no human labels or reward model needed.
- Richer gradient signals than maximum likelihood: the model receives feedback about long-horizon sequence quality.
- Improves coherence in long-form generation and reasoning trace quality.
- Acts as a bridge between NTP pre-training and GRPO-based alignment.

---

## Fill-in-the-Middle (FIM)

An objective specifically designed for **code models** that enables infilling (completing code given both prefix and suffix context).

$$\langle \mathrm{PRE}\rangle\, \text{prefix}\, \langle\mathrm{SUF}\rangle\, \text{suffix}\, \langle\mathrm{MID}\rangle\, \text{middle}$$

The model learns to predict the `middle` given `prefix` and `suffix`. This enables:
- Code completion at the cursor position (not just append-mode).
- Docstring generation from function signature + body.
- Test generation from implementation.

**Implementation:** With probability $p_\mathrm{FIM}$ (typically 0.5), transform a document into FIM format before packing. The remaining $(1 - p_\mathrm{FIM})$ fraction trains on standard NTP. Used in Code Llama, DeepSeek-Coder, and StarCoder.

```
Standard NTP:   def add(a, b):\n    return a + b
FIM transform:  <PRE>def add(a, b):\n<SUF>\n<MID>    return a + b
```

---

## Curriculum Learning

Data organized by difficulty (simple → complex) accelerates convergence and improves final performance compared to random shuffling.

### Difficulty Metrics

| Metric | How Computed | Best For |
| :--- | :--- | :--- |
| Perplexity under a small reference model | Proxy model scores each document | General text |
| Number of reasoning steps | Heuristic parsing of structure | Math, code |
| Domain specificity | Classifier confidence | Domain adaptation |
| IFD score | Instruction-following difficulty | SFT data selection |

### Practical Schedule

1. **Stage 1 (0–60% of training):** Easy/common examples — short documents, clean web text, Wikipedia. Builds vocabulary and basic syntax.
2. **Stage 2 (60–90%):** Medium difficulty — longer documents, code, scientific text. Builds factual knowledge and reasoning.
3. **Stage 3 (90–100%):** Hard examples — complex math, multi-step reasoning, long-form technical writing. Fine-tunes capabilities.

> [!NOTE]
> Curriculum order matters most in the **early** training stages. After sufficient exposure to easy examples, introducing hard examples yields diminishing returns from ordering.

---

## Data Mixing and DoReMi

For models trained on multiple data domains (web, code, books, math), the mixing weights across domains have a large effect on downstream performance.

### Manual Mixing (Baseline)

Human-specified fractions based on domain quality intuition. Fragile — requires extensive ablations.

### DoReMi (Domain Reweighting via Minimax Optimization)

DoReMi (Xie et al., 2023) automatically learns optimal domain mixing weights:

1. **Train a small proxy model** (30M–300M params) on uniform data.
2. **Compute per-domain excess loss** relative to a reference model trained on uniform data.
3. **Reweight domains** using a minimax objective that upweights domains where the proxy model is worst.
4. **Train the full model** with the learned weights.

$$\min_\theta \max_{\alpha \in \Delta} \sum_{k=1}^K \alpha_k \cdot \mathcal{L}_k(\theta)$$

where $\alpha_k$ are domain weights constrained to the simplex $\Delta$.

**Result:** DoReMi typically improves average downstream performance by 1–2% over hand-tuned mixing weights, with the gains concentrated in underrepresented but high-signal domains (math, scientific papers).

---

## Instruction-Response Augmented Pre-training

Synthetic instruction-response pairs mixed into the pre-training corpus bridge the gap to supervised fine-tuning.

**Typical mix:** 1–3% of total pre-training tokens.

**Effect:** Reduces SFT data required by 5–10× for comparable instruction-following quality. The model learns the assistant format during pre-training, so SFT need only refine tone and safety — not teach the format from scratch.

**Sources of synthetic instruction data:**
- Self-Instruct (Wang et al., 2022): bootstrap from 175 seed tasks using GPT-3.
- Evol-Instruct (Xu et al., 2023): iteratively rewrite instructions to increase complexity.
- Magpie (Xu et al., 2024): prompt Llama 3 with only the user turn prefix; it generates the instruction from scratch.

---

## Long-Context Pre-training

Training directly at long context from the start is compute-inefficient — most documents are short, so long-context positions receive sparse gradient signal.

### Two-Stage Extension Pipeline

```mermaid
graph LR
    A["Stage 1<br/>4K context<br/>~90% of compute"] -->|"RoPE freq rescaling<br/>(YaRN / LongRoPE)"| B["Stage 2<br/>32K–128K context<br/>~10% of compute<br/>long-doc filtered subset"]
    B --> C["Stage 3 (optional)<br/>1M+ context<br/>Ring Attention<br/>iRoPE layers"]

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#dcfce7,stroke:#16a34a
    style C fill:#fef9c3,stroke:#ca8a04
```

**Stage 1:** Train at 4K context on the full pre-training corpus. Builds core capabilities cheaply.

**Stage 2:** Increase to 32K–128K using a filtered long-document subset. Apply **YaRN** or **LongRoPE** to rescale rotary frequencies — avoids the positional out-of-distribution problem when extending beyond the training window.

**Stage 3 (optional):** Million-token context using Ring Attention (distributed attention across a device ring) and iRoPE-style interleaved non-positional layers (Llama 4).

### Context Extension Without Full Retraining: YaRN

YaRN (Peng et al., 2023) rescales the RoPE base frequency:

$$\theta_i' = \theta_i \cdot s, \qquad s = \frac{L_\mathrm{target}}{L_\mathrm{train}}$$

Combined with NTK-aware interpolation (different scaling per frequency band) and a temperature correction on attention logits. Achieves 4–8× context extension with fine-tuning on only ~1B tokens of long documents.

> [!WARNING]
> **Long-context fine-tuning without data filtering degrades quality.** Long documents in web crawls are often boilerplate, legal disclaimers, or spam. Use quality filters specifically tuned for long-document coherence (e.g., minimum 10K character documents from book/paper domains) for Stage 2 data.

---

## Summary: Pre-training Objective Comparison

| Objective | Supervision | Primary Benefit | Used In |
| :--- | :--- | :--- | :--- |
| NTP (causal LM) | Self-supervised | General capability | All LLMs |
| FIM (fill-in-middle) | Self-supervised | Code infilling | Code Llama, DeepSeek-Coder |
| RPT | Self-supervised | Long-horizon coherence | Microsoft 2025 models |
| Instruction-augmented | Synthetic labels | SFT efficiency | Llama 3, Phi series |
| Curriculum-ordered | Self-supervised | Convergence speed | Most frontier models |

---

[← Previous Chapter](ch05_synthetic_data.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](ch07_distributed_training.md)
