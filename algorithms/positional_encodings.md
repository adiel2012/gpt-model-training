# Positional Encodings

Transformers have no built-in notion of sequence order. Positional encodings inject position information either as absolute offsets or as relative biases on the attention scores.

---

## Absolute Sinusoidal (Original Transformer)

$$
PE_{t,2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad PE_{t,2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

Added directly to token embeddings. Fixed, not learned. Does not generalize beyond training context length. Replaced by RoPE in all modern LLMs.

---

## RoPE (Rotary Position Embedding)

RoPE [su2024roformer] encodes position by **rotating** query and key vectors before the dot product. Position is relative — only the difference $t - s$ affects the attention score between positions $t$ and $s$.

For a 2D sub-vector pair $(x_{2i}, x_{2i+1})$ at position $t$:

$$
R_{\theta_i, t} = \begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}, \qquad \theta_i = 10000^{-2(i-1)/d}
$$

The rotated query $\tilde{q}_t = R_t q_t$ and key $\tilde{k}_s = R_s k_s$ satisfy:

$$
\tilde{q}_t^\top \tilde{k}_s = q_t^\top R_{t-s} k_s
$$

so the dot product depends only on the relative displacement $t - s$.

**Properties:**
- No extra parameters.
- Generalizes beyond training length with interpolation techniques.
- Standard in LLaMA, Mistral, Qwen, Gemma, and virtually all 2024–2026 models.

```python
def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, d_head), freqs: (T, d_head//2) complex"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs).flatten(-2).type_as(x)
```

---

## iRoPE (Interleaved RoPE)

Used in Llama 4. Alternates between layers with and without positional encoding:

- **Even layers:** standard RoPE (position-aware).
- **Odd layers:** no positional encoding (position-agnostic, can attend anywhere).

This allows some layers to act as global "any-to-any" attention while others enforce locality. Enables 10M-token context windows without full positional interpolation at every layer.

---

## ALiBi (Attention with Linear Biases)

Rather than modifying the embeddings, ALiBi adds a linear penalty to the attention logits based on distance:

$$
\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot |i - j|
$$

where $m_h$ is a head-specific slope: $m_h = 2^{-8h/H}$ for head $h$ of $H$ total heads.

**Properties:**
- Zero extra parameters.
- Strong length generalization: models trained at 2K tokens can attend reasonably at 4K+.
- Each head has a different locality bias (flatter slopes = more global; steeper = more local).
- Used in BLOOM, MPT.

**Limitation:** Cannot be combined with KV-cache-dependent rotations (incompatible with RoPE's relative distance property in the same model).

---

## YaRN (Yet another RoPE extensioN)

Extends RoPE to longer contexts without full re-training by **NTK-aware interpolation**:

1. **NTK interpolation:** Scale the base frequency $b = 10000$ to $b' = b \cdot (L'/L)^{d/(d-2)}$ where $L$ is training length and $L'$ is target length. This preserves high-frequency (local) information while stretching low-frequency (global) components.
2. **Temperature scaling:** Apply an attention temperature $t < 1$ to compensate for distribution shift at longer contexts.
3. **Fine-tuning:** A short fine-tuning run (1000–5000 steps) on long-context data solidifies the extension.

**Results:** 4–8× context extension with minimal perplexity increase. Used in Mistral v0.2 (32K from 8K), LLaMA 2 Long, and others.

**LongRoPE** [ding2024longrope] extends this further with non-uniform per-dimension scaling, achieving 2M token contexts.

---

## Comparison

| Method | Extra Params | Relative Pos | Length Generalization | Used In |
|---|---|---|---|---|
| Sinusoidal | 0 | No | Poor | Original Transformer |
| RoPE | 0 | Yes | Moderate (needs YaRN) | LLaMA 3, Mistral, Qwen |
| iRoPE | 0 | Partial | Excellent | Llama 4 |
| ALiBi | 0 | Yes (linear) | Good | BLOOM, MPT |
| YaRN | 0 | Yes | Excellent (4–8×) | Mistral Long, LLaMA 2 Long |
