# Appendix G: The Implementation Treasury

> Theory and code, side by side — every formula paired with its PyTorch equivalent.

This appendix provides a unified reference for the mathematical formulations and PyTorch implementations discussed throughout the book. Each theoretical concept is paired with a self-contained, runnable code snippet.

---

## Table of Contents

| Section | Subsections |
| :--- | :--- |
| [Shared Foundation](#shared-foundation) | Imports, helpers |
| [Training Objectives](#training-objectives) | Pre-training, SFT, DPO, SimPO, GRPO, KTO, DAPO, RLVR |
| [Attention Mechanisms](#attention-mechanisms) | SDPA, MHA, GQA, MQA, MLA, Mamba, MoE |
| [Positional Encodings](#positional-encodings) | RoPE, ALiBi |
| [Normalization & Activation](#normalization-and-activation) | RMSNorm, SwiGLU |
| [Optimizers](#optimizers) | AdamW, Muon, Lion, Cosine schedule |
| [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning) | LoRA, DoRA |
| [Alignment Objectives](#alignment-objectives) | PPO, DPO, SimPO, GRPO, KTO, DAPO, RLVR |
| [Inference & Distillation](#inference-and-distillation) | KD, MiniLLM, KV cache, Quantization, Speculative decoding |
| [Model Merging](#model-merging) | Task vectors, SLERP, DARE, TIES |
| [Continual Learning](#continual-learning) | EWC |
| [Final Assembly: TinyGPT](#final-assembly--tinygpt) | Full decoder stack |

---

## Shared Foundation

All snippets assume these standard imports:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable
```

### Helper: Sequence Log-Probability

Used throughout the alignment and distillation sections:

```python
def log_prob_seq(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Returns the mean per-token log-probability of labels under model.
    Labels with value -100 are excluded from the mean (standard SFT masking).
    """
    with torch.no_grad():
        logits = model(input_ids).logits if hasattr(model(input_ids), "logits") else model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)

def avg_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Alias for log_prob_seq; used in length-normalized objectives (SimPO)."""
    return log_prob_seq(model, input_ids, labels)
```

---

## Training Objectives

The full production pipeline runs five sequential optimization stages.

### Stage 1 — Pre-training

**Causal language modeling** (next-token prediction):

$$\theta_1^* = \arg\min_\theta\; \mathcal{L}_\mathrm{PT}(\theta) = -\frac{1}{|\mathcal{D}|}\sum_{x \in \mathcal{D}} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

```python
def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Shifts logits and labels so that position t predicts token t+1.
    Labels with value -100 are ignored (used to mask prompt tokens in SFT).
    """
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

**Perplexity** — the standard pre-training evaluation metric:

$$\mathrm{PPL} = \exp\!\left(\mathcal{L}_\mathrm{NTP}\right)$$

Lower perplexity = more confident predictions. It represents the effective branching factor of the model's distribution at each step.

### Stage 2 — Supervised Fine-Tuning (SFT)

Same cross-entropy objective as pre-training, but **loss is masked to assistant tokens only**:

```python
def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """SFT loss: identical to causal_lm_loss but labels must be pre-masked.

    Mask prompt / system / user tokens with -100 in the labels tensor before
    calling this function. Only assistant token positions contribute to the loss.
    """
    return causal_lm_loss(logits, labels)
```

> [!WARNING]
> **Loss Masking is Critical.** Failing to mask prompt tokens causes the model to learn to generate user messages. A common sign of this bug is the model injecting `<|user|>` tokens mid-response. Always set `labels[prompt_token_positions] = -100` before computing SFT loss.

### Stage 3 — Preference Optimization

See full implementations in [Alignment Objectives](#alignment-objectives).

**DPO:**
$$\mathcal{L}_\mathrm{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\mathrm{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\mathrm{ref}(y_l|x)}\right)\right]$$

**SimPO** (no reference model):
$$\mathcal{L}_\mathrm{SimPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

### Stage 4 — Reasoning RL (GRPO)

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$$

See full GRPO implementation in [Alignment Objectives](#alignment-objectives).

### Stage 5 — RAG / Citation Grounding (Optional)

Used to train the model to strictly ground responses in retrieved context:

```python
def citation_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_ids: torch.Tensor,
) -> torch.Tensor:
    """Augmented cross-entropy that penalizes hallucination relative to provided context.

    In production, a faithfulness reward model scores whether generated tokens
    are grounded in source_ids. This stub shows the base CE component.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
```

---

## Attention Mechanisms

### Scaled Dot-Product Attention

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where $M$ is the causal mask ($-\infty$ for future positions, $0$ elsewhere).

```python
def scaled_dot_product_attention(
    Q: torch.Tensor,   # (Batch, Heads, QueryLen, Dim)
    K: torch.Tensor,   # (Batch, Heads, KeyLen, Dim)
    V: torch.Tensor,   # (Batch, Heads, KeyLen, ValDim)
    causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with optional causal masking."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if causal:
        T, S = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(T, S, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V
```

> [!NOTE]
> In production, replace this with `F.scaled_dot_product_attention` (PyTorch 2.0+) to use Flash Attention kernels automatically when available.

### Multi-Head Attention (MHA)

$$\mathrm{MHA}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)\,W^O, \quad \mathrm{head}_h = \mathrm{Attention}(QW_h^Q,\,KW_h^K,\,VW_h^V)$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        Q, K, V = self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x))
        out = scaled_dot_product_attention(Q, K, V, causal=causal)
        B, H, T, dk = out.shape
        return self.Wo(out.transpose(1, 2).reshape(B, T, H * dk))
```

### Grouped Query Attention (GQA)

$G$ KV head groups shared across $H/G$ query heads. Setting $G=1$ recovers MQA; $G=H$ recovers MHA.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_groups: int):
        super().__init__()
        assert n_heads % n_kv_groups == 0
        self.h, self.g, self.d_k = n_heads, n_kv_groups, d_model // n_heads
        self.Wq = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_groups * self.d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_groups * self.d_k, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.g, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.g, self.d_k).transpose(1, 2)
        reps = self.h // self.g
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        out = scaled_dot_product_attention(q, k, v, causal=causal)
        return self.Wo(out.transpose(1, 2).reshape(B, T, self.h * self.d_k))
```

### Multi-Query Attention (MQA)

Single $K, V$ pair shared by all query heads. Special case of GQA with $G=1$.

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, self.d_k, bias=False)
        self.Wv = nn.Linear(d_model, self.d_k, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, 1, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, 1, self.d_k).transpose(1, 2)
        out = scaled_dot_product_attention(q, k, v, causal=causal)
        return self.Wo(out.transpose(1, 2).reshape(B, T, self.h * self.d_k))
```

### Multi-head Latent Attention (MLA)

DeepSeek-V3. Compresses the KV cache into a low-rank latent vector $c_t^{KV}$:

$$c_t^{KV} = W^{DKV}\, h_t, \qquad [k_t^C;\, v_t^C] = W^{UKV}\, c_t^{KV}$$

Only $c_t^{KV}$ (shape `d_latent`) is stored in the KV cache — far smaller than full $K, V$ tensors.

```python
class MLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_latent: int):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.Wq   = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_dk = nn.Linear(d_model, d_latent, bias=False)   # down-proj: h -> c_kv
        self.W_uk = nn.Linear(d_latent, n_heads * self.d_k, bias=False)  # up-proj: c_kv -> K
        self.W_uv = nn.Linear(d_latent, n_heads * self.d_k, bias=False)  # up-proj: c_kv -> V
        self.Wo   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        q   = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        c_kv = self.W_dk(x)                                    # (B, T, d_latent) — cached
        k   = self.W_uk(c_kv).view(B, T, self.h, self.d_k).transpose(1, 2)
        v   = self.W_uv(c_kv).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = scaled_dot_product_attention(q, k, v, causal=causal)
        return self.Wo(out.transpose(1, 2).reshape(B, T, self.h * self.d_k))
```

> [!NOTE]
> At inference, cache `c_kv` (size `d_latent`) instead of the full `k` and `v` tensors (size `2 × n_heads × d_k`). Re-project to `k, v` at each decode step. This is the key memory saving.

### State Space Models (Mamba / SSM)

Replaces attention with a linear-time selective scan. History is compressed into a fixed-size state vector — no KV cache, constant memory during generation.

$$\Delta = \tau_\Delta(\text{Linear}(x)), \quad \bar{A} = e^{\Delta A}, \quad \bar{B} = \Delta B, \quad h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

```python
class MambaBlock(nn.Module):
    """Simplified Selective State Space (SSM) block.

    In a full Mamba implementation, the scan over (A, B, C, delta) is performed
    via a parallel associative scan (O(L log L) work, O(L) memory) using
    custom CUDA kernels. This class illustrates the structure only.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.A      = nn.Parameter(torch.randn(d_model, d_state))
        self.B      = nn.Linear(d_model, d_state, bias=False)
        self.C      = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        dt  = F.softplus(self.dt_proj(x))      # (B, T, D) — input-dependent step size
        A   = -torch.exp(self.A.float())        # (D, N) — stable negative real part
        # Placeholder: a real impl uses torch.associative_scan or mamba_ssm kernels
        h   = (x * dt).cumsum(dim=1)            # structural approximation only
        return self.out_proj(h)
```

### Mixture of Experts (MoE)

Replaces the dense FFN with $E$ expert networks; a learned router activates the top-$k$ experts per token.

$$\mathrm{MoE}(x) = \sum_{i \in \mathcal{K}} g_i(x) \cdot E_i(x), \qquad g_i(x) = \frac{e^{r_i}}{\sum_{j \in \mathcal{K}} e^{r_j}}, \quad r = W_\mathrm{router}\, x$$

```python
class MoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.router  = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_model * 4) for _ in range(n_experts)])
        self.top_k   = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        logits = self.router(x)                                  # (B, T, E)
        probs  = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # renormalize

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Mask tokens routed to expert i
            mask        = (top_indices == i).any(dim=-1)          # (B, T)
            if not mask.any():
                continue
            weight      = (top_indices == i).float() * top_probs  # (B, T, k)
            weight      = weight.sum(dim=-1)                       # (B, T)
            out[mask]  += expert(x[mask]) * weight[mask].unsqueeze(-1)
        return out
```

> [!TIP]
> **Load Balancing Loss.** Without an auxiliary loss, the router collapses — routing all tokens to one or two experts. Add the following term to the total loss: `aux_loss = n_experts * (probs.mean(dim=[0,1]) * (top_indices == torch.arange(n_experts).view(1,1,-1)).float().mean(dim=[0,1])).sum()`.

---

## Positional Encodings

### Rotary Position Embedding (RoPE)

$$R_{\theta_i, t} = \begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}, \qquad \theta_i = 10000^{-2(i-1)/d}$$

Applied to $Q$ and $K$ before the dot product so that $q_m^\top k_n$ depends only on the relative position $(m - n)$.

```python
def precompute_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the RoPE frequency tensor for a given context length."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    return torch.outer(t, freqs).float()   # (seq_len, dim/2)

def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to the last dimension of x.

    Args:
        x:     (..., seq_len, dim)
        freqs: (seq_len, dim/2) — from precompute_freqs
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis  = torch.polar(torch.ones_like(freqs), freqs)   # e^{i * freqs}
    freqs_cis  = freqs_cis.view(1, 1, x.shape[-2], -1)
    x_rotated  = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
```

### ALiBi (Attention with Linear Biases)

$$\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

where the slope $m_h = 2^{-8h/H}$ is head-specific and fixed. No learned position parameters.

```python
def alibi_bias(n_heads: int, seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Generates the ALiBi linear penalty matrix.

    Returns: (n_heads, seq_len, seq_len) lower-triangular bias tensor.
    """
    slopes  = 2 ** (-8 * torch.arange(1, n_heads + 1, dtype=torch.float32) / n_heads)
    pos     = torch.arange(seq_len, dtype=torch.float32)
    dist    = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()           # (T, T)
    bias    = -slopes.view(-1, 1, 1) * dist.unsqueeze(0)            # (H, T, T)
    causal  = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    bias    = bias.masked_fill(~causal.unsqueeze(0), float("-inf"))
    return bias.to(device) if device else bias
```

---

## Normalization and Activation

### RMSNorm

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\|x\|_\mathrm{RMS} + \epsilon}, \qquad \|x\|_\mathrm{RMS} = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization. Standard in Llama 2/3, DeepSeek, Qwen."""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * (x / rms)
```

### SwiGLU Feed-Forward Network

$$\mathrm{SwiGLU}(x,\,W,\,V,\,W_2) = \bigl(\mathrm{Swish}(xW) \odot xV\bigr)\,W_2, \qquad \mathrm{Swish}(x) = x\,\sigma(x)$$

```python
class SwiGLU(nn.Module):
    """Gated linear unit with Swish activation. The 2024–2026 production FFN standard."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # gate branch
        self.W2 = nn.Linear(d_model, d_ff, bias=False)   # linear branch
        self.W3 = nn.Linear(d_ff, d_model, bias=False)   # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
```

---

## Optimizers

### AdamW

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_{t-1}\right)$$

Decoupled weight decay $\lambda$ is applied directly to $\theta$, not through the adaptive gradient estimate.

```python
def adamw_step(
    params: list, grads: list, m: list, v: list, t: int,
    lr: float = 3e-4, b1: float = 0.9, b2: float = 0.95,
    eps: float = 1e-8, wd: float = 0.1,
) -> int:
    """Single AdamW update step with decoupled weight decay."""
    t += 1
    for p, g, m_i, v_i in zip(params, grads, m, v):
        m_i.mul_(b1).add_(g, alpha=1 - b1)
        v_i.mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = m_i / (1 - b1 ** t)
        v_hat = v_i / (1 - b2 ** t)
        p.mul_(1 - lr * wd)                                  # weight decay
        p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr) # adaptive update
    return t
```

### Muon Optimizer

Maintains orthogonal weight updates via Newton-Schulz iterations:

$$X_\mathrm{new} = \tfrac{1}{2}\,X\,(3I - X^\top X)$$

```python
@torch.no_grad()
def muon_step(
    p: torch.Tensor, g: torch.Tensor, m: torch.Tensor,
    lr: float = 0.02, momentum: float = 0.95, n_iters: int = 5,
) -> None:
    """Orthogonalizing optimizer; strong results on LLM pre-training (2025–2026)."""
    m.mul_(momentum).add_(g)
    X = m.view(m.size(0), -1).clone()
    X = X / X.norm().clamp(min=1e-7)
    I = torch.eye(X.size(0), device=X.device)
    for _ in range(n_iters):
        X = 0.5 * X @ (3.0 * I - X.T @ X)
    p.add_(X.view_as(p), alpha=-lr)
```

### Lion Optimizer

Uses the `sign` of a momentum blend for updates — lower memory than Adam (no second moment):

$$c_t = \mathrm{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t), \qquad m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$$

```python
def lion_step(
    params: list, grads: list, exp_avg: list,
    lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, wd: float = 0.1,
) -> None:
    """Lion (sign-momentum) optimizer update."""
    for p, g, m in zip(params, grads, exp_avg):
        p.mul_(1 - lr * wd)
        update = torch.sign(m * beta1 + g * (1 - beta1))
        p.add_(update, alpha=-lr)
        m.mul_(beta2).add_(g, alpha=1 - beta2)
```

### Cosine LR Schedule with Linear Warmup

```python
def cosine_lr(
    step: int, total_steps: int,
    lr_max: float, lr_min: float, warmup_steps: int,
) -> float:
    """Cosine annealing with linear warmup. Standard for pre-training and SFT."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
```

---

## Parameter-Efficient Fine-Tuning

### LoRA (Low-Rank Adaptation)

$$h = W_0 x + \frac{\alpha}{r}\,B A x, \qquad A \in \mathbb{R}^{r \times d_\mathrm{in}},\; B \in \mathbb{R}^{d_\mathrm{out} \times r}$$

$B$ is initialized to zero so the adapter contributes nothing at the start of training.

```python
class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a trainable low-rank delta."""
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base  = base
        self.scale = alpha / rank
        self.base.weight.requires_grad_(False)   # freeze base weights
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Decomposes the weight update into **magnitude** and **direction** components:

$$W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}, \qquad m \in \mathbb{R}^{1 \times d_\mathrm{in}}$$

where $\|\cdot\|_c$ is the column-wise norm.

```python
class DoRALinear(nn.Module):
    """Weight-Decomposed LoRA. Matches or exceeds full fine-tuning on many benchmarks."""
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base  = base
        self.scale = alpha / rank
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        # Learnable magnitude, initialized to column norms of base weight
        col_norms = base.weight.norm(dim=0, keepdim=True)   # (1, d_in)
        self.m = nn.Parameter(col_norms.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_adapted = self.base.weight + (self.B @ self.A) * self.scale
        col_norms = W_adapted.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm    = W_adapted / col_norms                   # unit-column direction
        W_dora    = self.m * W_norm                         # scale by learned magnitude
        return F.linear(x, W_dora, self.base.bias)
```

---

## Alignment Objectives

### PPO (Proximal Policy Optimization)

$$\mathcal{L}_\mathrm{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}(r_t(\theta),\,1{-}\epsilon,\,1{+}\epsilon)\,\hat{A}_t\right)\right]$$

```python
def ppo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate objective."""
    ratio  = (policy_log_probs - old_log_probs).exp()
    surr1  = ratio * advantages
    surr2  = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(surr1, surr2).mean()
```

### DPO (Direct Preference Optimization)

$$\mathcal{L}_\mathrm{DPO}(\theta) = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\mathrm{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\mathrm{ref}(y_l|x)}\right)\right]$$

```python
def dpo_loss(
    policy: nn.Module,
    ref: nn.Module,
    x_w: torch.Tensor,
    x_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss for a batch of (prompt, chosen, rejected) triples."""
    pi_w   = log_prob_seq(policy, x_w, x_w)
    pi_l   = log_prob_seq(policy, x_l, x_l)
    ref_w  = log_prob_seq(ref,    x_w, x_w)
    ref_l  = log_prob_seq(ref,    x_l, x_l)
    reward_margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(reward_margin).mean()
```

### SimPO (Simple Preference Optimization)

No reference model required. Uses length-normalized average log-probability as the implicit reward:

$$\mathcal{L}_\mathrm{SimPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

```python
def simpo_loss(
    model: nn.Module,
    x_w: torch.Tensor,
    x_l: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """SimPO: length-normalized preference optimization without a reference model."""
    r_w = avg_log_prob(model, x_w, x_w)
    r_l = avg_log_prob(model, x_l, x_l)
    return -F.logsigmoid(beta * (r_w - r_l) - gamma).mean()
```

### GRPO (Group Relative Policy Optimization)

Generates $G$ responses per prompt and normalizes advantages within the group — no critic model needed:

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}, \qquad \mathcal{L}_\mathrm{GRPO} = -\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\min\!\left(\rho_i\hat{A}_i,\;\mathrm{clip}(\rho_i,1{-}\epsilon,1{+}\epsilon)\hat{A}_i\right) - \beta\,\mathrm{KL}[\pi_\theta\|\pi_\mathrm{ref}]\right]$$

```python
def grpo_step(
    policy: nn.Module,
    ref: nn.Module,
    prompt_ids: torch.Tensor,
    reward_fn: Callable,
    G: int = 8,
    beta: float = 0.001,
    eps: float = 0.2,
) -> torch.Tensor:
    """GRPO update step. reward_fn(completions) -> (G,) float tensor."""
    with torch.no_grad():
        completions = [policy.generate(prompt_ids) for _ in range(G)]
        rewards     = torch.stack(reward_fn(completions))
        old_logp    = torch.stack([log_prob_seq(policy, c, c) for c in completions])

    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = torch.tensor(0.0, requires_grad=True)
    for i, comp in enumerate(completions):
        curr_logp = log_prob_seq(policy, comp, comp)
        ratio     = (curr_logp - old_logp[i]).exp()
        surr1     = ratio * advantage[i]
        surr2     = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage[i]
        pg_loss   = -torch.min(surr1, surr2)
        kl_div    = curr_logp - log_prob_seq(ref, comp, comp)
        total_loss = total_loss + pg_loss + beta * kl_div

    return total_loss / G
```

### KTO (Kahneman-Tversky Optimization)

Aligns from binary feedback (thumbs up / down) instead of pairwise comparisons:

$$\mathcal{L}_\mathrm{KTO} = -\mathbb{E}\!\left[w(y) \cdot \log\sigma\!\left(\beta\,(r_\theta(x,y) - z_\mathrm{ref})\right)\right]$$

where $w(y) = \lambda_w$ for desirable outputs and $\lambda_l$ for undesirable, and $z_\mathrm{ref}$ is the expected reward under the reference model.

```python
def kto_loss(
    model: nn.Module,
    ref: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    desirable: torch.Tensor,  # bool mask: True = thumbs-up
    beta: float = 0.1,
    lam_w: float = 1.0,
    lam_l: float = 1.0,
) -> torch.Tensor:
    """KTO loss from binary per-sample feedback."""
    logp     = log_prob_seq(model, input_ids, labels)
    logp_ref = log_prob_seq(ref,   input_ids, labels)
    reward   = beta * (logp - logp_ref)
    weight   = torch.where(desirable, lam_w, lam_l)
    return -(weight * F.logsigmoid(reward)).mean()
```

### DAPO (Decoupled Alignment and Policy Optimization)

Removes the reference model from GRPO's gradient path, reducing memory overhead:

```python
def dapo_loss(
    policy: nn.Module,
    ref: nn.Module,
    x_w: torch.Tensor,
    x_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DAPO: reference model is detached (no gradient through ref)."""
    pi_w, pi_l = log_prob_seq(policy, x_w, x_w), log_prob_seq(policy, x_l, x_l)
    with torch.no_grad():                              # <-- decoupled: ref is frozen
        ref_w, ref_l = log_prob_seq(ref, x_w, x_w), log_prob_seq(ref, x_l, x_l)
    reward_margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(reward_margin).mean()
```

### RLVR (Reinforcement Learning with Verifiable Rewards)

Training on tasks with binary auto-checkable rewards (math answers, unit tests, compiler output):

```python
def rlvr_step(
    policy: nn.Module,
    prompt_ids: torch.Tensor,
    reward_fn: Callable,   # returns list of floats in {0.0, 1.0}
    G: int = 8,
) -> torch.Tensor:
    """DeepSeek-R1 style RLVR — uses GRPO with a verifiable binary reward signal."""
    with torch.no_grad():
        completions = [policy.generate(prompt_ids) for _ in range(G)]
        rewards     = torch.tensor(reward_fn(completions), dtype=torch.float32)
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    # Use advantage as policy gradient signal (via GRPO or REINFORCE)
    loss = torch.tensor(0.0, requires_grad=True)
    for i, comp in enumerate(completions):
        logp = log_prob_seq(policy, comp, comp)
        loss = loss - (logp * advantage[i])
    return loss / G
```

---

## Inference and Distillation

### Knowledge Distillation — Forward KL

Student learns to match teacher's full distribution:

$$\mathcal{L}_\mathrm{fwd} = D_\mathrm{KL}(p_\mathrm{teacher} \| p_\mathrm{student}) = \sum_v p_T(v)\log\frac{p_T(v)}{p_S(v)}$$

```python
def forward_kl_distill(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Forward KL distillation loss. Temp > 1 softens the teacher distribution."""
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)
```

### MiniLLM — Reverse KL

Student distribution leads the matching, avoiding mode-averaging:

$$\mathcal{L}_\mathrm{rev} = D_\mathrm{KL}(p_\mathrm{student} \| p_\mathrm{teacher}) = \sum_v p_S(v)\log\frac{p_S(v)}{p_T(v)}$$

```python
def reverse_kl_distill(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reverse KL distillation (MiniLLM). Prevents mode-averaging behavior."""
    p_student = F.softmax(student_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    return (p_student * (log_p_student - log_p_teacher)).sum(-1).mean()
```

### KV Cache Memory Formula

$$M_\mathrm{KV} = 2 \times L \times T \times H \times d_k \times \text{bytes\_per\_element}$$

where $L$ = layers, $T$ = sequence length, $H$ = KV heads (after GQA reduction), $d_k$ = head dimension. In BF16 ($2$ bytes): a 70B model (80 layers, 8 KV heads, $d_k=128$) at 32K context ≈ 4.2 GB.

### Linear Quantization

Quantize weights to $n$ bits:

$$x_q = \mathrm{round}\!\left(\frac{x}{\Delta}\right), \qquad \Delta = \frac{\max(|x|)}{2^{n-1}-1}$$

### Speculative Decoding

A small draft model proposes $k$ tokens; the target model verifies all $k$ in one forward pass. Token $t$ is accepted with probability $\alpha_t = \min\!\left(1,\, p_\mathrm{target}(t) / p_\mathrm{draft}(t)\right)$.

```python
@torch.no_grad()
def speculative_decode(
    target: nn.Module,
    draft: nn.Module,
    input_ids: torch.Tensor,
    max_new: int = 128,
    k: int = 4,
) -> torch.Tensor:
    """Draft-verify speculative decoding. Achieves 2–3× speedup at zero quality loss."""
    ids = input_ids.clone()
    while ids.shape[1] < input_ids.shape[1] + max_new:
        draft_ids, draft_probs = ids.clone(), []

        # Draft phase: generate k tokens with the small model
        for _ in range(k):
            d_logits = draft(draft_ids).logits[:, -1]
            p_draft  = F.softmax(d_logits, dim=-1)
            tok      = torch.multinomial(p_draft, 1)
            draft_probs.append(p_draft[0, tok[0, 0]].item())
            draft_ids = torch.cat([draft_ids, tok], dim=1)

        # Verify phase: target model scores all k+1 positions in one pass
        t_logits = target(draft_ids).logits
        accepted = 0
        for j in range(k):
            pos     = ids.shape[1] + j - 1
            p_tgt   = F.softmax(t_logits[0, pos], dim=-1)
            tok_j   = draft_ids[0, ids.shape[1] + j].item()
            alpha   = min(1.0, p_tgt[tok_j].item() / (draft_probs[j] + 1e-9))
            if torch.rand(1).item() < alpha:
                accepted += 1
            else:
                break

        ids = torch.cat([ids, draft_ids[:, ids.shape[1]:ids.shape[1] + accepted + 1]], dim=1)
    return ids
```

---

## Model Merging

### Task Vectors

$$\tau = \theta_\mathrm{fine\text{-}tuned} - \theta_\mathrm{base}$$

Task vectors can be added, subtracted, and scaled:

```python
def task_vector(base_sd: dict, ft_sd: dict) -> dict:
    """Computes the weight delta: tau = theta_ft - theta_base."""
    return {k: ft_sd[k].float() - base_sd[k].float() for k in base_sd}

def apply_task_vector(base_sd: dict, tau: dict, scale: float = 1.0) -> dict:
    """Applies a task vector to a base model at a given scale."""
    return {k: base_sd[k] + scale * tau[k].to(base_sd[k].dtype) for k in base_sd}
```

### SLERP (Spherical Linear Interpolation)

Interpolates along the geodesic of weight space:

$$\mathrm{SLERP}(\theta_A, \theta_B, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,\theta_A + \frac{\sin(t\Omega)}{\sin\Omega}\,\theta_B, \qquad \Omega = \arccos\!\left(\hat\theta_A \cdot \hat\theta_B\right)$$

```python
def slerp(w_a: torch.Tensor, w_b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation between two weight tensors."""
    a = w_a.flatten().float()
    b = w_b.flatten().float()
    cos_omega = (a / a.norm()).dot(b / b.norm()).clamp(-1.0, 1.0)
    omega     = cos_omega.acos()
    if omega.abs() < 1e-6:
        return ((1 - t) * w_a + t * w_b)        # fallback to linear when nearly parallel
    out = (math.sin((1 - t) * omega) * a + math.sin(t * omega) * b) / math.sin(omega)
    return out.view_as(w_a).to(w_a.dtype)
```

### DARE (Drop And REscale)

Randomly drops task vector entries and rescales survivors to reduce interference during merging:

$$\theta_\mathrm{merge} = \theta_\mathrm{base} + \frac{1}{1-p}\sum_{i=1}^n \mathrm{mask}_i \odot \tau_i$$

where $p$ is the per-element drop probability and the $1/(1-p)$ factor preserves expected magnitude.

### TIES-Merging

Three-step conflict resolution across multiple fine-tuned models: **T**rim → **E**lect sign → **D**isjoint merge.

```python
def ties_merge(
    base_sd: dict,
    ft_sds: list,
    lambdas: list,
    density: float = 0.7,
) -> dict:
    """TIES: Trim-Elect-Sign merging for combining multiple LoRA/SFT checkpoints."""
    taus = [task_vector(base_sd, ft) for ft in ft_sds]

    # Step 1 — Trim: keep top `density` fraction of deltas by magnitude
    trimmed = []
    for tau in taus:
        trimmed_tau = {}
        for k, v in tau.items():
            threshold = torch.quantile(v.abs().float(), 1.0 - density)
            trimmed_tau[k] = v * (v.abs() >= threshold)
        trimmed.append(trimmed_tau)

    # Steps 2 & 3 — Elect majority sign, then disjoint-merge agreeing vectors
    merged = {}
    for k in base_sd:
        stacked      = torch.stack([t[k].float() for t in trimmed], dim=0)  # (N, ...)
        elected_sign = torch.sign(stacked.sum(dim=0))
        agree_mask   = (torch.sign(stacked) == elected_sign.unsqueeze(0)).float()
        denom        = agree_mask.sum(dim=0).clamp(min=1)
        delta        = (stacked * agree_mask).sum(dim=0) / denom
        avg_lambda   = sum(lambdas) / len(lambdas)
        merged[k]    = base_sd[k] + (avg_lambda * delta).to(base_sd[k].dtype)

    return merged
```

---

## Continual Learning

### Elastic Weight Consolidation (EWC)

Penalizes updates to weights important for previously learned tasks, weighted by their Fisher information:

$$\mathcal{L}_\mathrm{EWC} = \mathcal{L}_\mathrm{new} + \frac{\lambda}{2}\sum_i F_i\,(\theta_i - \theta_i^*)^2$$

```python
class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""

    def __init__(self, model: nn.Module, dataset, lam: float = 1000.0):
        self.lam    = lam
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(model, dataset)

    def _compute_fisher(self, model: nn.Module, dataset) -> dict:
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        for x, y in dataset:
            model.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
        n_samples = len(dataset)
        return {name: f / n_samples for name, f in fisher.items()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC regularization term to add to the task loss."""
        loss = torch.tensor(0.0)
        for name, p in model.named_parameters():
            loss = loss + (self.fisher[name] * (p - self.params[name]) ** 2).sum()
        return self.lam * loss
```

---

## Final Assembly — TinyGPT

A minimal, production-style decoder-only model assembling all components from this treasury.

```python
class DecoderLayer(nn.Module):
    """Single transformer decoder layer: pre-norm GQA + SwiGLU FFN."""
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv: int = 2,
        d_ff: int = 2048,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn  = GroupedQueryAttention(d_model, n_heads, n_kv)
        self.ffn   = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    """Minimal production-style decoder LLM using GQA, RMSNorm, SwiGLU, and RoPE.

    Default config (d=512, 6 layers, 8 heads, 2 KV heads): ~50M parameters.
    """
    def __init__(
        self,
        vocab_size: int = 32_000,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv: int = 2,
        d_ff: int = 2048,
        n_layers: int = 6,
        max_seq: int = 2048,
    ):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, n_kv, d_ff) for _ in range(n_layers)
        ])
        self.norm   = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight               # weight tying
        self.register_buffer(
            "freqs", precompute_freqs(d_model // n_heads, max_seq)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(ids)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
```

---

[← Previous Chapter](app_f_hyperparams.md) | [Table of Contents](../README.md#table-of-contents)
