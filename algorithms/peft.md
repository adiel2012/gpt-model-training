# Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all model parameters — expensive in memory and compute. PEFT methods freeze most weights and train only a small set of new parameters, achieving comparable or better results at a fraction of the cost.

---

## LoRA (Low-Rank Adaptation)

[hu2022lora] The dominant PEFT method. Decomposes weight updates into a product of two low-rank matrices:

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

where:
- $W_0 \in \mathbb{R}^{d \times k}$ is the frozen pre-trained weight.
- $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are the trainable low-rank matrices.
- $r \ll \min(d, k)$ is the rank (typically 4–64).
- $\alpha / r$ is a scaling factor ($\alpha$ is a hyperparameter, commonly equal to $r$).

**Initialization:** $A$ is random Gaussian, $B$ is zero — so the update starts at zero and doesn't destabilize the pre-trained weights.

**Why it works:** The weight update $\Delta W = BA$ lives in a low-dimensional subspace. Empirically, the intrinsic dimensionality of task-specific adaptation is far lower than $d \times k$.

**Where to apply:** Typically on $W_Q$, $W_K$, $W_V$, $W_O$ in attention, and optionally on $W_\text{up}$, $W_\text{down}$ in the FFN. Modern practice applies LoRA to all linear layers.

**Parameter count:** $r(d + k)$ per weight matrix. At $r=16$ on a 7B model, this is $<0.5\%$ of total parameters.

**Merging:** At inference, $W = W_0 + \frac{\alpha}{r}BA$ — the LoRA weights can be merged into $W_0$ with zero inference overhead.

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return F.linear(x, self.weight) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)
```

---

## QLoRA (Quantized LoRA)

[dettmers2023qlora] Combines LoRA with 4-bit base model quantization to enable fine-tuning of 70B+ models on a single consumer GPU.

**Key techniques:**
1. **NF4 (NormalFloat4):** A 4-bit data type with values placed at quantiles of a standard normal distribution — optimal for normally-distributed weights.
2. **Double quantization:** Quantize the quantization constants themselves, saving an additional 0.37 bits/parameter.
3. **Paged optimizers:** Use NVIDIA unified memory to page optimizer states to CPU RAM during GPU memory spikes.

**Memory at 70B:**
- Full fine-tuning: ~280 GB (FP16).
- QLoRA: ~48 GB — fits on two A100 80GB GPUs.

**Trade-off:** 4-bit base introduces a small quality gap vs full FP16 fine-tuning, but QLoRA-tuned models typically match full fine-tuning quality on downstream tasks.

---

## DoRA (Weight-Decomposed Low-Rank Adaptation)

[liu2024dora] Extends LoRA by decomposing the weight into **magnitude** and **direction** components and training them separately:

$$
W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

where:
- $m \in \mathbb{R}^{1 \times k}$ is a learned magnitude vector (one scalar per output feature).
- $\|W_0 + BA\|_c$ normalizes along the column dimension.
- $BA$ is the standard LoRA update.

**Why it helps:** Standard LoRA couples magnitude and direction changes in a single low-rank matrix, which can limit expressivity. DoRA lets the model independently control how much each direction is emphasized vs which directions are active.

**Results:** Matches or exceeds full fine-tuning quality on commonsense reasoning and visual instruction tuning benchmarks, while using the same parameter budget as LoRA.

---

## Spectrum

[context2024spectrum] Selects *which* layers to fine-tune based on **signal-to-noise ratio (SNR)** analysis of the pre-trained weight matrices.

**Algorithm:**
1. Compute the singular value decomposition (SVD) of each weight matrix $W = U\Sigma V^\top$.
2. Define SNR for a matrix as the ratio of top singular values to the tail: $\text{SNR} = \sigma_1 / \sigma_r$.
3. Fine-tune only the top-SNR layers (most informative). Freeze the rest.

**Intuition:** Low-SNR layers are noisy and contribute little to task-specific representations — fine-tuning them wastes compute. High-SNR layers encode task-relevant structure worth adapting.

**Practical effect:** Fine-tuning 25–50% of layers by SNR ranking matches full fine-tuning performance while reducing trainable parameters further than LoRA.

---

## Comparison

| Method | Extra Params | Memory | Quality vs Full FT | Key Use Case |
|---|---|---|---|---|
| LoRA | $< 1\%$ | 1.1–1.5× base | ~95–98% | Standard fine-tuning |
| QLoRA | $< 1\%$ | 0.25–0.35× FP16 | ~93–97% | Large models, limited GPU |
| DoRA | $\approx$ LoRA | $\approx$ LoRA | ~98–100% | Quality-critical tasks |
| Spectrum | 25–50% layers | 0.5–0.75× | ~98–100% | Layer-selective tuning |

**Hyperparameter guidelines:**

| Param | Range | Notes |
|---|---|---|
| $r$ (rank) | 4–128 | 16–32 for general tasks; 64–128 for complex tasks |
| $\alpha$ | $= r$ or $2r$ | Higher $\alpha/r$ = stronger update |
| Dropout | 0.05–0.1 | Regularization, especially at higher $r$ |
| Target modules | all linear | Including FFN improves math/code |
| Learning rate | $1\text{–}3 \times 10^{-4}$ | Higher than full fine-tuning |
