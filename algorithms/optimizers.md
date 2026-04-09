# Optimizers and Learning Rate Schedules

---

## AdamW

The 2025–2026 default for LLM training. Adam with **decoupled weight decay** [loshchilov2019decoupled].

**Update rule:**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\,\theta_{t-1}\right)
$$

where $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$ are bias-corrected estimates.

**Why decoupled weight decay matters:** Standard Adam applies $L_2$ regularization by adding $\lambda\theta$ to the gradient before the adaptive scaling. This means heavily-updated parameters get less regularization. AdamW applies $\lambda\theta$ *after* the adaptive step, correctly regularizing all parameters equally regardless of gradient magnitude.

**LLM hyperparameters:**

| Param | Value | Note |
|---|---|---|
| $\beta_1$ | 0.9 | Momentum |
| $\beta_2$ | 0.95 | Second moment (lower than Adam's 0.999) |
| $\epsilon$ | $10^{-8}$ | Numerical stability |
| $\lambda$ | 0.1 | Higher than typical 0.01 for LLM pre-training |
| Memory | 3× model | $\theta$, $m$, $v$ |

---

## Muon (Momentum + Orthogonalization)

[kosson2024muon] Applies a Nesterov momentum update followed by **Newton-Schulz orthogonalization**.

**Key insight:** Adam treats each scalar weight independently. Muon treats each weight *matrix* as the unit of analysis, ensuring all directions in weight space receive equal update magnitude — no direction is starved.

**Algorithm:**

1. Compute Nesterov momentum gradient $G = \beta m_{t-1} + (1-\beta) g_t$.
2. Orthogonalize via Newton-Schulz iterations:

$$
X_0 = G / \|G\|_F, \quad X_{k+1} = \frac{1}{2} X_k (3I - X_k^\top X_k)
$$

3. Apply update: $\theta_t = \theta_{t-1} - \eta\, X_K$.

**Properties:**
- 2× model memory (no second moment).
- 10–25% faster optimizer step than AdamW.
- Matches or exceeds AdamW pre-training perplexity.
- Not suitable for embedding layers or 1D parameters (LayerNorm, biases) — use AdamW for those.

---

## Lion (EvoLved Sign Momentum)

[chen2023symbolic] Discovered by program search. Uses only the **sign** of the gradient momentum — the smallest possible update direction signal.

$$
c_t = \text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t)
$$

$$
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
$$

$$
\theta_t = \theta_{t-1} - \eta\, c_t - \eta\lambda\,\theta_{t-1}
$$

**Properties:**
- 2× model memory (no second moment, compared to Adam's 3×).
- Every parameter update has identical magnitude $\eta$ — acts as a sign-based gradient descent.
- Effective LR must be 3–10× *smaller* than AdamW (sign updates are larger in magnitude).
- Works well for fine-tuning; mixed results for large-scale pre-training.

---

## Sophia

[liu2023sophia] Diagonal Hessian preconditioned optimizer. Estimates curvature to scale updates inversely to the local loss landscape curvature.

$$
\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\max(h_t,\, \epsilon)}
$$

where $h_t$ is the diagonal Hessian estimate, computed every $k=10$ steps via the **Hutchinson estimator**:

$$
h_t \approx g_t \odot \hat{z}_t, \quad \hat{z}_t \sim \mathcal{N}(0, I)
$$

**Properties:**
- 4× model memory (adds Hessian estimate to Adam's three terms).
- Reported 2× faster convergence than AdamW.
- Not yet widely adopted at production scale due to implementation complexity.

---

## Cosine Learning Rate Schedule with Linear Warmup

The standard LLM training schedule:

$$
\eta(t) = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})\left(1 + \cos\left(\frac{t - t_\text{warmup}}{T - t_\text{warmup}}\pi\right)\right)
$$

**Phase 1 — Linear warmup** ($0 \leq t \leq t_\text{warmup}$):
- Ramp LR from 0 to $\eta_\text{max}$ over 1,000–5,000 steps.
- Prevents gradient explosions from random initial weight distributions.

**Phase 2 — Cosine decay** ($t_\text{warmup} < t \leq T$):
- Smooth decay to $\eta_\text{min} = 0.1 \times \eta_\text{max}$.
- Do **not** decay to zero: a residual LR prevents overfitting on the tail of training.

**Recommended peak LRs by model size:**

| Model Size | Peak LR |
|---|---|
| 1B–7B | $3 \times 10^{-4}$ |
| 13B–34B | $2 \times 10^{-4}$ |
| 70B | $1 \times 10^{-4}$ |
| 100B+ | $5 \times 10^{-5}$ |

Larger models require smaller LR because parameter interactions are stronger — a given gradient step has larger downstream effects.

---

## WSD (Warmup-Stable-Decay)

Three-phase schedule used by MiniCPM and Qwen:

1. **Warmup:** Linear ramp to peak LR (same as cosine).
2. **Stable:** Hold peak LR constant for the majority of training.
3. **Decay:** Rapid decay to near-zero over a short final window.

**Advantage over cosine:** The stable phase allows checkpointing at multiple scales — you can stop after any amount of stable-phase training and get a competitive model. Cosine schedules produce poor checkpoints mid-decay.

---

## Gradient Clipping

**Norm clipping:** Before the optimizer step, scale gradients if their global $\ell_2$ norm exceeds a threshold:

$$
g \leftarrow g \cdot \frac{c}{\max(\|g\|_2,\, c)}, \quad c = 1.0
$$

**Diagnostics:**
- More than 1% of steps hitting the clip threshold → hyperparameter issue (LR too high, bad data batch).
- A loss spike $> 3\times$ the rolling average → corrupted data batch, bad micro-batch, or numerical instability. Roll back to last checkpoint.

**Batch size scaling rule:** When doubling batch size, multiply LR by $\sqrt{2}$ (conservative) rather than $2\times$ (linear) for LLMs. Use gradient accumulation to simulate large batches without increasing per-device memory.

---

## Optimizer Comparison

| Optimizer | Memory | Speed | Best For |
|---|---|---|---|
| AdamW | 3× model | Baseline | Universal default |
| Muon | 2× model | +10–25% | Pre-training, large matrices |
| Lion | 2× model | +5–15% | Fine-tuning, memory-constrained |
| Sophia | 4× model | +80–120% | Research, curvature-sensitive tasks |
