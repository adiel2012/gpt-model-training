# Appendix G: The Implementation Treasury

> Theory and code, side by side — every formula paired with its PyTorch equivalent.

This appendix provides a unified reference for the mathematical formulations and PyTorch implementations discussed throughout the book. Each theoretical concept is paired with a self-contained, runnable code snippet. All snippets are written against PyTorch 2.x.

---

## Table of Contents

| Section | Key Contents |
| :--- | :--- |
| [Shared Foundation](#shared-foundation) | Imports, `log_prob_seq`, `avg_log_prob` |
| [Training Objectives](#training-objectives) | Pre-training NTP, FIM, SFT loss masking, DPO formula, GRPO formula |
| [Data Utilities](#data-utilities) | Chat template, context packing, MinHash dedup |
| [Attention Mechanisms](#attention-mechanisms) | SDPA, Flash Attention, MHA, GQA (with RoPE), MQA, MLA, Mamba, MoE |
| [Positional Encodings](#positional-encodings) | RoPE, ALiBi |
| [Normalization & Activation](#normalization-and-activation) | RMSNorm, SwiGLU |
| [Optimizers](#optimizers) | AdamW, Muon, Lion, Cosine schedule, WSD schedule |
| [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning) | LoRA, DoRA |
| [Alignment Objectives](#alignment-objectives) | Reward Model, PPO, DPO, SimPO, GRPO, KTO, DAPO, RLVR |
| [Inference & Distillation](#inference-and-distillation) | Forward KL, Reverse KL, SeqKD, GKD, Rejection Sampling, Self-Consistency, PRM, Quantization, Speculative decoding |
| [Model Merging](#model-merging) | Task vectors, SLERP, DARE, TIES |
| [Continual Learning](#continual-learning) | EWC |
| [Training Utilities](#training-utilities) | Gradient checkpointing, Gradient accumulation, Mixed precision |
| [Final Assembly: TinyGPT](#final-assembly--tinygpt) | Full decoder stack with RoPE |

---

## Shared Foundation

All snippets assume these standard imports:

```python
import math
import contextlib
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
    """Returns mean per-token log-probability of labels under model.

    Gradients flow through the model by default — suitable for policy updates.
    Wrap in torch.no_grad() at the call site when evaluating a frozen reference model.
    Labels with value -100 are excluded from the mean (standard SFT masking convention).

    Args:
        model:     Any causal LM. May return a tensor or an object with a .logits attribute.
        input_ids: (B, T) token ids fed as input.
        labels:    (B, T) token ids for loss targets. Typically identical to input_ids;
                   set positions to -100 to exclude them from the mean.
    """
    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out  # (B, T, V)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (B,)


def avg_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Alias for log_prob_seq; used in length-normalized objectives (SimPO)."""
    return log_prob_seq(model, input_ids, labels)
```

> [!WARNING]
> **No-grad pattern for reference models.** `log_prob_seq` propagates gradients through the model. When scoring a *frozen* reference model (DPO, GRPO, KTO), always wrap the call in `torch.no_grad()`. The alignment functions in this appendix follow this convention consistently.

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

### Fill-in-the-Middle (FIM)

Used for code models (Bavarian et al., 2022). FIM reformats sequences so the model learns to complete a middle span given a prefix and suffix — critical for code completion and infill tasks.

A document `[prefix | middle | suffix]` is randomly transformed into `[<PRE>, prefix, <SUF>, suffix, <MID>, middle]`. Loss is computed on all tokens in the reformatted sequence via standard NTP. No loss function changes are required — only the data transformation.

```python
def fim_transform(
    tokens: torch.Tensor,
    pre_id: int,
    suf_id: int,
    mid_id: int,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
) -> torch.Tensor:
    """Applies Fill-in-the-Middle data transformation.

    With probability fim_rate, randomly selects a contiguous span as the
    middle and reformats the sequence. Otherwise returns the original sequence.

    Args:
        tokens:   1-D token tensor for a single document.
        pre_id:   Token id for <PRE> special token.
        suf_id:   Token id for <SUF> special token.
        mid_id:   Token id for <MID> special token.
        fim_rate: Probability of applying FIM to this document.
        spm_rate: Probability of using Suffix-Prefix-Middle format instead of PSM.
    """
    if torch.rand(1).item() > fim_rate or len(tokens) < 3:
        return tokens
    n = len(tokens)
    i = torch.randint(0, n, (1,)).item()
    j = torch.randint(i, n, (1,)).item()
    prefix, middle, suffix = tokens[:i], tokens[i:j], tokens[j:]
    if torch.rand(1).item() < spm_rate:
        # Suffix-Prefix-Middle (SPM) format
        return torch.cat([
            torch.tensor([suf_id]), suffix,
            torch.tensor([pre_id]), prefix,
            torch.tensor([mid_id]), middle,
        ])
    # Prefix-Suffix-Middle (PSM) format
    return torch.cat([
        torch.tensor([pre_id]), prefix,
        torch.tensor([suf_id]), suffix,
        torch.tensor([mid_id]), middle,
    ])
```

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
> **Loss Masking is Critical.** Failing to mask prompt tokens causes the model to learn to generate user messages. A common sign of this bug is the model injecting `<|user|>` tokens mid-response. Always set `labels[prompt_token_positions] = -100` before computing SFT loss. The TRL library's `DataCollatorForCompletionOnlyLM` handles this automatically.

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

```python
def citation_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_ids: torch.Tensor,
) -> torch.Tensor:
    """Augmented cross-entropy for grounded generation.

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

## Data Utilities

### Chat Template — Llama 3 Format

Chat templates serialize multi-turn conversations into the exact token sequences the model was trained on. Training/inference mismatch here is a common source of degraded performance.

```python
def format_chat_llama3(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Formats a list of messages into a Llama 3 chat template string.

    Each message is {'role': 'system'|'user'|'assistant', 'content': str}.
    The string should be tokenized with add_special_tokens=False after this call
    (BOS is already embedded).

    Args:
        add_generation_prompt: if True, appends the assistant header to prompt
                               the model to generate. Set False for training examples
                               that end with a complete assistant turn.

    Example:
        messages = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
        ]
        text = format_chat_llama3(messages, add_generation_prompt=False)
    """
    BOS = "<|begin_of_text|>"
    HDR = "<|start_header_id|>{role}<|end_header_id|>\n\n"
    EOT = "<|eot_id|>"

    out = BOS
    for msg in messages:
        out += HDR.format(role=msg["role"]) + msg["content"].strip() + EOT
    if add_generation_prompt:
        out += HDR.format(role="assistant")
    return out


def apply_chat_loss_mask(
    input_ids: list[int],
    tokenizer,
    assistant_start_token: str = "<|start_header_id|>assistant<|end_header_id|>",
    eot_token: str = "<|eot_id|>",
) -> list[int]:
    """Returns labels tensor with -100 for all non-assistant token positions.

    Scans for assistant turn boundaries using the chat template delimiters.
    Works on any template that uses consistent start/end markers.

    Returns a list of ints (labels) aligned with input_ids; pass to the model
    as the `labels` argument after converting to torch.Tensor.
    """
    labels = [-100] * len(input_ids)
    text   = tokenizer.decode(input_ids)

    start_marker = tokenizer.encode(
        assistant_start_token, add_special_tokens=False
    )
    end_marker = tokenizer.encode(eot_token, add_special_tokens=False)

    i = 0
    while i < len(input_ids):
        # Find next assistant start
        if input_ids[i: i + len(start_marker)] == start_marker:
            i += len(start_marker)
            # Copy label until the closing EOT
            while i < len(input_ids):
                if input_ids[i: i + len(end_marker)] == end_marker:
                    i += len(end_marker)
                    break
                labels[i] = input_ids[i]
                i += 1
        else:
            i += 1
    return labels
```

### Context Packing

Packing concatenates multiple short conversations into one training window, eliminating padding waste. Cross-sequence attention is blocked by resetting the causal mask at each boundary.

```python
def pack_sequences(
    token_lists: list[list[int]],
    max_len: int,
    eos_id: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs multiple token sequences into fixed-length windows.

    Sequences are separated by EOS tokens. Labels are set to -100 at padding
    and at the first token of each new sequence (no cross-sequence loss).

    Returns:
        input_ids:   (N, max_len) packed token ids
        labels:      (N, max_len) labels, -100 at non-loss positions
        seq_lengths: (N,) number of real (non-pad) tokens in each packed window
                     — used to construct block-diagonal attention masks.
    """
    packed_ids, packed_labels, seq_lens = [], [], []
    buf_ids: list[int] = []
    buf_labels: list[int] = []
    buf_boundaries: list[int] = []   # start positions of each sequence in buffer

    def flush():
        pad_len = max_len - len(buf_ids)
        packed_ids.append(buf_ids + [pad_id] * pad_len)
        packed_labels.append(buf_labels + [-100] * pad_len)
        seq_lens.append(len(buf_ids))

    for seq in token_lists:
        seq_with_eos = list(seq) + [eos_id]
        if len(buf_ids) + len(seq_with_eos) > max_len:
            flush()
            buf_ids, buf_labels, buf_boundaries = [], [], []

        start = len(buf_ids)
        buf_ids.extend(seq_with_eos)
        # Mask the first token of each new sequence (no BOS prediction loss)
        seq_labels = list(seq_with_eos)
        seq_labels[0] = -100
        buf_labels.extend(seq_labels)
        buf_boundaries.append(start)

    if buf_ids:
        flush()

    return (
        torch.tensor(packed_ids),
        torch.tensor(packed_labels),
        torch.tensor(seq_lens),
    )
```

### MinHash Near-Duplicate Detection

MinHash sketches approximate Jaccard similarity between documents for near-dedup in pre-training pipelines. Documents with `jaccard_estimate > threshold` (typically 0.8) are considered duplicates; the later one is removed.

```python
import hashlib


def minhash_signature(
    text: str,
    n_hashes: int = 128,
    n_grams: int = 5,
) -> list[int]:
    """Computes a MinHash signature for a text document.

    The signature is a list of n_hashes integers. Two documents' signatures
    can be compared with jaccard_estimate() to get an approximate Jaccard
    similarity without comparing every n-gram pair directly.

    Time complexity: O(n_grams_in_doc × n_hashes).
    Typical throughput: ~50K docs/s single-threaded at n_hashes=128.
    """
    tokens = text.lower().split()
    ngrams = {
        " ".join(tokens[i: i + n_grams])
        for i in range(max(1, len(tokens) - n_grams + 1))
    }
    signature = []
    for seed in range(n_hashes):
        min_hash = 2 ** 64  # sentinel — will be replaced immediately
        for gram in ngrams:
            h = int(hashlib.md5(f"{seed}:{gram}".encode()).hexdigest(), 16)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimates Jaccard similarity from two MinHash signatures.

    Exact Jaccard: |A ∩ B| / |A ∪ B|  (expensive to compute directly)
    MinHash estimate: fraction of signature positions where sig_a[i] == sig_b[i]

    Standard error ≈ 1 / sqrt(n_hashes). At n_hashes=128, error ≈ ±0.09.
    """
    assert len(sig_a) == len(sig_b), "Signatures must have equal length"
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)
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
> In production, replace this with `F.scaled_dot_product_attention` (PyTorch 2.0+) to use Flash Attention kernels automatically when available. This provides O(n) memory and ~2× throughput with zero code change.

### Flash Attention Drop-in (PyTorch 2.x)

`F.scaled_dot_product_attention` selects the optimal kernel at runtime — Flash Attention on CUDA, math fallback on CPU. The API is identical to manual SDPA but memory-efficient and faster.

```python
def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Flash Attention via PyTorch 2.0+ fused kernel.

    Requires PyTorch >= 2.0. On H100, uses FlashAttention-3 kernels automatically
    when installed via `pip install flash-attn`.

    Input shapes: (B, n_heads, T, head_dim) — same as scaled_dot_product_attention.
    Output shape: (B, n_heads, T, head_dim).

    Key advantage over manual SDPA: O(T) memory instead of O(T²) — mandatory
    for sequences > 8K tokens where the attention score matrix won't fit in SRAM.
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,     # causal mask is applied internally when is_causal=True
        dropout_p=dropout_p if Q.requires_grad else 0.0,
        is_causal=causal,
    )


def attention_with_kv_cache(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step decode with KV cache — the standard inference pattern.

    At each decode step:
    1. Compute Q, K, V for the new token only.
    2. Append K, V to the growing cache.
    3. Run attention with Q=(1 token) attending to all cached K, V.

    Returns (output, updated_K_cache, updated_V_cache).
    output shape: (B, n_heads, 1, head_dim)
    """
    K_full = torch.cat([K_cache, K_new], dim=2)   # (B, H, T_cache+1, dk)
    V_full = torch.cat([V_cache, V_new], dim=2)
    # Q attends to full history — no causal mask needed (Q is the last token)
    out = F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=False)
    return out, K_full, V_full
```

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

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        Q = self._split(self.Wq(x))
        K = self._split(self.Wk(x))
        V = self._split(self.Wv(x))
        if freqs is not None:
            Q = apply_rotary_emb(Q, freqs)
            K = apply_rotary_emb(K, freqs)
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

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, H, T, dk)
        k = self.Wk(x).view(B, T, self.g, self.d_k).transpose(1, 2)  # (B, G, T, dk)
        v = self.Wv(x).view(B, T, self.g, self.d_k).transpose(1, 2)  # (B, G, T, dk)
        if freqs is not None:
            # Apply RoPE to Q and K before KV expansion — more efficient
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)
        reps = self.h // self.g
        k = k.repeat_interleave(reps, dim=1)   # expand to (B, H, T, dk)
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

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, 1, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, 1, self.d_k).transpose(1, 2)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)
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
        self.W_dk = nn.Linear(d_model, d_latent, bias=False)          # down-proj: h -> c_kv
        self.W_uk = nn.Linear(d_latent, n_heads * self.d_k, bias=False)  # up-proj: c_kv -> K
        self.W_uv = nn.Linear(d_latent, n_heads * self.d_k, bias=False)  # up-proj: c_kv -> V
        self.Wo   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        q    = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        c_kv = self.W_dk(x)                                    # (B, T, d_latent) — cache this
        k    = self.W_uk(c_kv).view(B, T, self.h, self.d_k).transpose(1, 2)
        v    = self.W_uv(c_kv).view(B, T, self.h, self.d_k).transpose(1, 2)
        out  = scaled_dot_product_attention(q, k, v, causal=causal)
        return self.Wo(out.transpose(1, 2).reshape(B, T, self.h * self.d_k))
```

> [!NOTE]
> At inference, cache `c_kv` (size `d_latent`) instead of the full `k` and `v` tensors (size `2 × n_heads × d_k`). Re-project to `k, v` at each decode step. This is the key memory saving: `d_latent ≪ 2 × n_heads × d_k`.

### State Space Models (Mamba / SSM)

Replaces attention with a linear-time selective scan. History is compressed into a fixed-size state vector — no KV cache, constant memory during generation.

$$\Delta = \tau_\Delta(\mathrm{Linear}(x)), \quad \bar{A} = e^{\Delta A}, \quad \bar{B} = \Delta B, \quad h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

```python
class MambaBlock(nn.Module):
    """Simplified Selective State Space (SSM) block.

    In a full Mamba implementation, the scan over (A, B, C, delta) is performed
    via a parallel associative scan (O(L log L) work, O(L) memory) using
    custom CUDA kernels. This class illustrates the structure only.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state  = d_state
        self.A        = nn.Parameter(torch.randn(d_model, d_state))
        self.B        = nn.Linear(d_model, d_state, bias=False)
        self.C        = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        dt = F.softplus(self.dt_proj(x))       # (B, T, D) — input-dependent step size
        A  = -torch.exp(self.A.float())         # (D, N) — stable negative real part
        # Placeholder: a real impl uses torch.associative_scan or mamba_ssm kernels
        h  = (x * dt).cumsum(dim=1)             # structural approximation only
        return self.out_proj(h)
```

### Mixture of Experts (MoE)

Replaces the dense FFN with $E$ expert networks; a learned router activates the top-$k$ experts per token.

$$\mathrm{MoE}(x) = \sum_{i \in \mathcal{K}} g_i(x) \cdot E_i(x), \qquad g_i(x) = \frac{e^{r_i}}{\sum_{j \in \mathcal{K}} e^{r_j}}, \quad r = W_\mathrm{router}\, x$$

```python
def moe_aux_loss(
    router_probs: torch.Tensor,   # (B, T, E) — softmax output of the router
    top_indices: torch.Tensor,    # (B, T, k) — expert indices selected per token
    n_experts: int,
) -> torch.Tensor:
    """Load-balancing auxiliary loss (Fedus et al., 2021).

    Penalizes imbalanced routing by encouraging equal expert utilization.
    Add to total loss: total_loss = task_loss + aux_weight * moe_aux_loss(...)
    Typical aux_weight: 0.01.

    Minimizes the dot product of mean routing probability and mean dispatch fraction,
    which is minimized when each expert receives 1/E of the tokens.
    """
    # Mean routing probability per expert: (E,)
    avg_prob = router_probs.mean(dim=[0, 1])
    # Mean dispatch fraction per expert — fraction of tokens routed there: (E,)
    one_hot  = F.one_hot(top_indices[:, :, 0], num_classes=n_experts).float()
    avg_disp = one_hot.mean(dim=[0, 1])
    return n_experts * (avg_prob * avg_disp).sum()


class MoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.router  = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_model * 4) for _ in range(n_experts)])
        self.top_k   = top_k
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, aux_loss). Add aux_loss * 0.01 to total training loss."""
        B, T, D = x.shape
        logits     = self.router(x)                                  # (B, T, E)
        probs      = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_probs  = top_probs / top_probs.sum(dim=-1, keepdim=True)  # renormalize

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask   = (top_indices == i).any(dim=-1)
            if not mask.any():
                continue
            weight = (top_indices == i).float() * top_probs
            weight = weight.sum(dim=-1)
            out[mask] += expert(x[mask]) * weight[mask].unsqueeze(-1)

        aux = moe_aux_loss(probs, top_indices, self.n_experts)
        return out, aux
```

---

## Positional Encodings

### Rotary Position Embedding (RoPE)

$$R_{\theta_i, t} = \begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}, \qquad \theta_i = 10000^{-2(i-1)/d}$$

Applied to $Q$ and $K$ before the dot product so that $q_m^\top k_n$ depends only on the relative position $(m - n)$.

```python
def precompute_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the RoPE frequency tensor.

    Returns: (seq_len, dim // 2) float tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t     = torch.arange(seq_len, device=freqs.device)
    return torch.outer(t, freqs).float()   # (seq_len, dim/2)


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Applies RoPE rotations to x.

    Args:
        x:     (..., seq_len, dim) — typically (B, n_heads, T, head_dim)
        freqs: (seq_len, dim // 2) — from precompute_freqs, sliced to actual T

    The function treats pairs of consecutive dimensions as (real, imag) components
    of complex numbers, multiplies by e^{i * freqs}, then returns as real.
    """
    # Reshape to complex: (..., T, dim/2) of complex64
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Build unit phasors: (T, dim/2) -> (1, 1, T, dim/2) for broadcasting over (B, H)
    fc  = torch.polar(torch.ones_like(freqs), freqs).view(1, 1, freqs.shape[0], -1)
    return torch.view_as_real(x_c * fc).flatten(-2).type_as(x)
```

### ALiBi (Attention with Linear Biases)

$$\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

where the slope $m_h = 2^{-8h/H}$ is head-specific and fixed. No learned position parameters.

```python
def alibi_bias(n_heads: int, seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Generates the ALiBi linear penalty matrix.

    Returns: (n_heads, seq_len, seq_len) lower-triangular bias tensor.
    Add to attention scores before softmax.
    """
    slopes = 2 ** (-8 * torch.arange(1, n_heads + 1, dtype=torch.float32) / n_heads)
    pos    = torch.arange(seq_len, dtype=torch.float32)
    dist   = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()          # (T, T)
    bias   = -slopes.view(-1, 1, 1) * dist.unsqueeze(0)           # (H, T, T)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    bias   = bias.masked_fill(~causal.unsqueeze(0), float("-inf"))
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
        p.mul_(1 - lr * wd)                                  # decoupled weight decay
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

$$\eta(t) = \eta_\min + \frac{1}{2}(\eta_\max - \eta_\min)\left(1 + \cos\!\left(\pi \cdot \frac{t - t_\mathrm{wu}}{T - t_\mathrm{wu}}\right)\right)$$

```python
def cosine_lr(
    step: int, total_steps: int,
    lr_max: float, lr_min: float, warmup_steps: int,
) -> float:
    """Cosine annealing with linear warmup. Standard for pre-training and SFT."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
```

### WSD (Warmup-Stable-Decay) Schedule

Used by MiniCPM, Qwen3, and others. Keeps a stable phase at peak LR, enabling checkpoint reuse and mid-training data swaps. The decay phase uses cosine annealing.

$$\eta(t) = \begin{cases} \eta_\max \cdot t / t_\mathrm{wu} & t < t_\mathrm{wu} \\ \eta_\max & t_\mathrm{wu} \le t < t_\mathrm{wu} + t_\mathrm{stable} \\ \eta_\min + \frac{1}{2}(\eta_\max - \eta_\min)(1 + \cos\pi \cdot \frac{t - t_\mathrm{wu} - t_\mathrm{stable}}{t_\mathrm{decay}}) & \text{otherwise} \end{cases}$$

```python
def wsd_lr(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
) -> float:
    """Warmup-Stable-Decay learning rate schedule.

    Args:
        warmup_steps:  Linear ramp-up phase.
        stable_steps:  Constant peak LR phase — use for bulk of training.
        decay_steps:   Cosine decay to lr_min — triggered for final convergence.
    """
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    elif step < warmup_steps + stable_steps:
        return lr_max
    else:
        decay_step = step - warmup_steps - stable_steps
        progress   = decay_step / max(1, decay_steps)
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

Decomposes the weight update into **magnitude** and **direction** components separately (Liu et al., 2024):

$$W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}, \qquad m \in \mathbb{R}^{1 \times d_\mathrm{in}}$$

where $\|\cdot\|_c$ is the column-wise norm. The magnitude $m$ is learnable; the direction is updated via LoRA.

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
        # Learnable magnitude vector, one value per input column
        col_norms = base.weight.norm(dim=0, keepdim=True)   # (1, d_in)
        self.m = nn.Parameter(col_norms.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_adapted = self.base.weight + (self.B @ self.A) * self.scale   # (d_out, d_in)
        col_norms = W_adapted.norm(dim=0, keepdim=True).clamp(min=1e-8) # (1, d_in)
        W_norm    = W_adapted / col_norms                                # unit-column direction
        W_dora    = self.m * W_norm                                      # scale by learned mag
        return F.linear(x, W_dora, self.base.bias)
```

---

## Alignment Objectives

### Reward Model Training

PPO requires a reward model $r_\phi(x, y)$ trained on human preference pairs before the RL loop begins. The **Bradley-Terry** model frames preference as a probability:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)$$

Training minimises the negative log-likelihood of the observed preferences:

$$\mathcal{L}_\mathrm{RM} = -\mathbb{E}\!\left[\log\sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

```python
class RewardModel(nn.Module):
    """Scalar reward model built on top of a pretrained LM backbone.

    The backbone's final hidden state at the last non-padding token is projected
    to a single scalar reward. Trained on (prompt, chosen, rejected) preference pairs.

    Typical recipe: initialise from the SFT checkpoint, then fine-tune with
    rm_loss() on human preference data before the PPO/RLHF loop.
    """
    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()
        self.backbone   = backbone
        self.value_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns scalar rewards of shape (B,)."""
        out    = self.backbone(input_ids)
        hidden = out.logits if hasattr(out, "logits") else out  # (B, T, D)
        # Pool at the last real token (rightmost non-padding position)
        last_pos = attention_mask.sum(dim=1) - 1               # (B,)
        pooled   = hidden[torch.arange(hidden.size(0)), last_pos]  # (B, D)
        return self.value_head(pooled).squeeze(-1)             # (B,)


def rm_loss(
    reward_model: nn.Module,
    x_w_ids: torch.Tensor,   # (B, T) chosen sequence token ids
    x_l_ids: torch.Tensor,   # (B, T) rejected sequence token ids
    x_w_mask: torch.Tensor,  # (B, T) attention mask for chosen
    x_l_mask: torch.Tensor,  # (B, T) attention mask for rejected
) -> torch.Tensor:
    """Bradley-Terry preference loss for reward model training."""
    r_w = reward_model(x_w_ids, x_w_mask)   # (B,)
    r_l = reward_model(x_l_ids, x_l_mask)   # (B,)
    return -F.logsigmoid(r_w - r_l).mean()
```

> [!TIP]
> **Margin loss variant:** Add a margin $m > 0$ so the model must be confidently correct: `loss = -F.logsigmoid(r_w - r_l - m).mean()`. A margin of 0.5–1.0 improves reward model calibration and reduces reward hacking downstream.

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
    x_w: torch.Tensor,   # (B, T) chosen sequence ids (prompt + chosen response)
    x_l: torch.Tensor,   # (B, T) rejected sequence ids (prompt + rejected response)
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss for a batch of (prompt, chosen, rejected) triples.

    x_w and x_l should have labels set to -100 for prompt tokens so that
    log_prob_seq computes over response tokens only.
    """
    pi_w  = log_prob_seq(policy, x_w, x_w)
    pi_l  = log_prob_seq(policy, x_l, x_l)
    with torch.no_grad():                    # reference model: no gradient needed
        ref_w = log_prob_seq(ref, x_w, x_w)
        ref_l = log_prob_seq(ref, x_l, x_l)
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
    """GRPO update step. reward_fn(completions) -> list of G float rewards."""
    with torch.no_grad():
        completions = [policy.generate(prompt_ids) for _ in range(G)]
        rewards     = torch.tensor([r for r in reward_fn(completions)], dtype=torch.float32)
        old_logp    = torch.stack([log_prob_seq(policy, c, c) for c in completions])

    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # (G,)

    losses = []
    for i, comp in enumerate(completions):
        curr_logp = log_prob_seq(policy, comp, comp)
        with torch.no_grad():
            ref_logp = log_prob_seq(ref, comp, comp)
        ratio  = (curr_logp - old_logp[i]).exp()
        surr1  = ratio * advantage[i]
        surr2  = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage[i]
        pg     = -torch.min(surr1, surr2)
        kl_div = curr_logp - ref_logp          # approximate KL
        losses.append(pg + beta * kl_div)

    return torch.stack(losses).mean()
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
    desirable: torch.Tensor,  # bool mask: True = thumbs-up example
    beta: float = 0.1,
    lam_w: float = 1.0,
    lam_l: float = 1.0,
) -> torch.Tensor:
    """KTO loss from binary per-sample feedback (Ethayarajh et al., 2024)."""
    logp     = log_prob_seq(model, input_ids, labels)
    with torch.no_grad():
        logp_ref = log_prob_seq(ref, input_ids, labels)
    reward = beta * (logp - logp_ref)
    weight = torch.where(desirable, lam_w, lam_l)
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
    """DAPO: reference model is frozen and decoupled from the policy gradient."""
    pi_w = log_prob_seq(policy, x_w, x_w)
    pi_l = log_prob_seq(policy, x_l, x_l)
    with torch.no_grad():                       # ref is fully frozen — decoupled
        ref_w = log_prob_seq(ref, x_w, x_w)
        ref_l = log_prob_seq(ref, x_l, x_l)
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
    """DeepSeek-R1 style RLVR — GRPO with a verifiable binary reward signal.

    reward_fn receives a list of G completed token tensors and returns
    a list of G scalar rewards (0.0 = wrong, 1.0 = correct).
    """
    with torch.no_grad():
        completions = [policy.generate(prompt_ids) for _ in range(G)]
        rewards     = torch.tensor(reward_fn(completions), dtype=torch.float32)

    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    losses = []
    for i, comp in enumerate(completions):
        logp = log_prob_seq(policy, comp, comp)
        losses.append(-logp * advantage[i])

    return torch.stack(losses).mean()
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
    """Forward KL distillation loss.

    Temperature > 1 softens the teacher distribution, exposing more signal
    about the relative probabilities of non-top tokens.
    The temperature² rescaling keeps the loss magnitude consistent across temperatures.
    """
    p_teacher     = F.softmax(teacher_logits / temperature, dim=-1)
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
    """Reverse KL distillation (MiniLLM, Gu et al., 2024).

    Prevents mode-averaging: student concentrates mass on teacher's high-probability modes
    rather than spreading over all plausible continuations.
    """
    p_student     = F.softmax(student_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    return (p_student * (log_p_student - log_p_teacher)).sum(-1).mean()
```

### SeqKD (Sequence-Level Knowledge Distillation)

Kim & Rush (2016). Generate sequences from the teacher (greedy or beam), then train the student on those sequences with standard cross-entropy. No soft targets — entirely offline distillation.

```python
def seqkd_loss(
    student: nn.Module,
    teacher_seqs: torch.Tensor,  # (B, T) token ids greedy-decoded from teacher
) -> torch.Tensor:
    """SeqKD: train student on teacher's greedy output sequences.

    teacher_seqs are pre-generated offline via teacher.generate().
    This is equivalent to SFT on the teacher's outputs.
    """
    out = student(teacher_seqs)
    logits = out.logits if hasattr(out, "logits") else out
    # Labels are the same sequence shifted: predict token t+1 from tokens 0..t
    labels = teacher_seqs.clone()
    labels[:, :-1] = teacher_seqs[:, 1:]
    labels[:, -1]  = -100   # ignore last position (no target)
    return causal_lm_loss(logits, labels)
```

### GKD (Generalized Knowledge Distillation) — On-Policy Step

Agarwal et al. (2024). Mixes on-policy student completions with teacher-generated sequences, using the teacher's soft distribution as the training signal on both.

```python
def gkd_step(
    student: nn.Module,
    teacher: nn.Module,
    prompt_ids: torch.Tensor,
    lam: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """GKD update step.

    With probability lam: student generates on-policy, teacher provides soft targets.
    With probability 1-lam: teacher generates, student distills offline (SeqKD variant).
    The on-policy fraction is critical for distribution matching at inference time.
    """
    with torch.no_grad():
        if torch.rand(1).item() < lam:
            seq = student.generate(prompt_ids, temperature=temperature)  # on-policy
        else:
            seq = teacher.generate(prompt_ids)                           # off-policy
        t_out    = teacher(seq)
        t_logits = t_out.logits if hasattr(t_out, "logits") else t_out

    s_out    = student(seq)
    s_logits = s_out.logits if hasattr(s_out, "logits") else s_out

    return forward_kl_distill(s_logits, t_logits, temperature=temperature)
```

### Rejection Sampling for SFT Data Generation

Best-of-N selection: generate $N$ completions, keep the one with the highest reward. Used in the DeepSeek-R1 pipeline to create high-quality SFT data from the RLVR-trained policy.

```python
def rejection_sample(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    reward_fn: Callable,
    n_samples: int = 16,
    temperature: float = 0.8,
) -> tuple[torch.Tensor, float]:
    """Best-of-N rejection sampling.

    Returns (best_completion, best_reward).
    Use the returned completion as a training example for downstream SFT.

    Args:
        reward_fn: callable that takes a list of token tensors and returns
                   a list of scalar rewards.
    """
    completions = [
        model.generate(prompt_ids, temperature=temperature)
        for _ in range(n_samples)
    ]
    rewards  = reward_fn(completions)
    best_idx = max(range(n_samples), key=lambda i: rewards[i])
    return completions[best_idx], rewards[best_idx]
```

### KV Cache Memory Formula

$$M_\mathrm{KV} = 2 \times L \times T \times H_\mathrm{KV} \times d_k \times \text{bytes\_per\_element}$$

where $L$ = layers, $T$ = sequence length, $H_\mathrm{KV}$ = KV heads (after GQA), $d_k$ = head dimension.

Example: 70B Llama 3 (80 layers, 8 KV heads, $d_k=128$, BF16) at 32K context ≈ 4.2 GB.

### Absmax Symmetric Quantization

```python
def quantize_absmax(
    weight: torch.Tensor,
    n_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor absmax quantization to n_bits.

    Maps the full range [-max_val, max_val] to [-2^(n-1)-1, 2^(n-1)-1].
    Returns (quantized_int8_weight, scale_factor).

    Dequantize: weight_fp ≈ quantized * scale
    """
    qmax  = 2 ** (n_bits - 1) - 1
    scale = weight.abs().max() / qmax
    q     = (weight / scale.clamp(min=1e-8)).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_absmax(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct approximate float32 weight from absmax-quantized representation."""
    return q.float() * scale
```

### Speculative Decoding

A small draft model proposes $k$ tokens; the target model verifies all $k$ in one forward pass. Token $t$ is accepted with probability:

$$\alpha_t = \min\!\left(1,\, \frac{p_\mathrm{target}(t)}{p_\mathrm{draft}(t)}\right)$$

```python
@torch.no_grad()
def speculative_decode(
    target: nn.Module,
    draft: nn.Module,
    input_ids: torch.Tensor,
    max_new: int = 128,
    k: int = 4,
) -> torch.Tensor:
    """Draft-verify speculative decoding. Achieves 2–3× speedup at zero quality loss.

    The acceptance sampling preserves the exact target model distribution:
    tokens are accepted or rejected to correct for the draft distribution mismatch.
    """
    ids = input_ids.clone()
    while ids.shape[1] < input_ids.shape[1] + max_new:
        draft_ids, draft_probs = ids.clone(), []

        # Draft phase: generate k tokens with the small model
        for _ in range(k):
            d_logits = draft(draft_ids)
            d_logits = d_logits.logits if hasattr(d_logits, "logits") else d_logits
            p_draft  = F.softmax(d_logits[:, -1], dim=-1)
            tok      = torch.multinomial(p_draft, 1)
            draft_probs.append(p_draft[0, tok[0, 0]].item())
            draft_ids = torch.cat([draft_ids, tok], dim=1)

        # Verify phase: target model scores all k+1 positions in one forward pass
        t_out    = target(draft_ids)
        t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
        accepted = 0
        for j in range(k):
            pos   = ids.shape[1] + j - 1
            p_tgt = F.softmax(t_logits[0, pos], dim=-1)
            tok_j = draft_ids[0, ids.shape[1] + j].item()
            alpha = min(1.0, p_tgt[tok_j].item() / (draft_probs[j] + 1e-9))
            if torch.rand(1).item() < alpha:
                accepted += 1
            else:
                break

        ids = torch.cat([ids, draft_ids[:, ids.shape[1]: ids.shape[1] + accepted + 1]], dim=1)
    return ids
```

### Self-Consistency (Majority Vote)

Wang et al. (2022). Sample $N$ independent reasoning chains, extract the final answer from each, and return the most common answer. Improves accuracy without any model weight changes.

$$\hat{y} = \mathrm{mode}\!\left(\{y_1, y_2, \ldots, y_N\}\right)$$

```python
from collections import Counter


def self_consistency(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    answer_extractor: Callable[[torch.Tensor], str],
    n_samples: int = 16,
    temperature: float = 0.7,
) -> tuple[str, dict[str, int]]:
    """Self-consistency majority vote over N sampled reasoning chains.

    Args:
        answer_extractor: callable that takes a completion token tensor and returns
                          the final answer string (e.g. extracts the boxed answer
                          from a math solution, or the last line of code output).
        n_samples:        number of independent samples; 16–32 is typical.
        temperature:      >0 required — greedy (temp=0) gives the same answer each time.

    Returns:
        (best_answer, vote_counts) where vote_counts is a {answer: count} dict.

    Accuracy gain: typically +5–20 points on math benchmarks vs. greedy decoding.
    Cost: N × single-sample inference cost.
    """
    with torch.no_grad():
        answers = [
            answer_extractor(model.generate(prompt_ids, temperature=temperature))
            for _ in range(n_samples)
        ]
    votes = Counter(answers)
    best  = votes.most_common(1)[0][0]
    return best, dict(votes)
```

### Process Reward Model (PRM)

A PRM scores intermediate reasoning steps rather than only the final answer. Used to guide MCTS search and Best-of-N selection at test time (Lightman et al., 2023).

```python
class ProcessRewardModel(nn.Module):
    """Step-level reward model that scores each reasoning step independently.

    Architecture: same as RewardModel but applied at every step boundary
    rather than only at the final token.

    Training: supervised on human-labeled step correctness annotations
    (each step marked correct/incorrect), or via Monte Carlo rollout estimation
    — for each partial solution, estimate P(correct final answer) by sampling
    many completions and using the empirical success rate as the step label.
    """
    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()
        self.backbone   = backbone
        self.step_head  = nn.Linear(d_model, 1, bias=False)

    def score_steps(
        self,
        input_ids: torch.Tensor,         # (B, T) token ids of partial solution
        step_end_positions: torch.Tensor, # (B, S) token positions of step endings
    ) -> torch.Tensor:
        """Returns per-step scores of shape (B, S).

        step_end_positions[b, s] is the token index of the end of step s
        in example b. Typically the position of a newline or delimiter token.
        """
        out    = self.backbone(input_ids)
        hidden = out.logits if hasattr(out, "logits") else out   # (B, T, D)
        B, S   = step_end_positions.shape
        # Gather hidden states at each step boundary
        scores = []
        for s in range(S):
            pos    = step_end_positions[:, s]                     # (B,)
            pooled = hidden[torch.arange(B), pos]                 # (B, D)
            scores.append(self.step_head(pooled).squeeze(-1))     # (B,)
        return torch.stack(scores, dim=1)                         # (B, S)

    def score_solution(
        self,
        input_ids: torch.Tensor,
        step_end_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregated solution score: product of per-step probabilities.

        P(solution correct) ≈ ∏ P(step s correct)  (assumes step independence)
        Returns (B,) scalar scores in [0, 1].
        """
        step_scores = self.score_steps(input_ids, step_end_positions)  # (B, S)
        step_probs  = torch.sigmoid(step_scores)
        return step_probs.prod(dim=1)                                   # (B,)
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
    """Applies a task vector to a base model at a given scale.

    scale < 1.0: partial application (conservative merge).
    scale > 1.0: amplification (risky — can overshoot).
    """
    return {k: base_sd[k] + (scale * tau[k]).to(base_sd[k].dtype) for k in base_sd}
```

### SLERP (Spherical Linear Interpolation)

Interpolates along the geodesic of weight space:

$$\mathrm{SLERP}(\theta_A, \theta_B, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,\theta_A + \frac{\sin(t\Omega)}{\sin\Omega}\,\theta_B, \qquad \Omega = \arccos\!\left(\hat\theta_A \cdot \hat\theta_B\right)$$

```python
def slerp(w_a: torch.Tensor, w_b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation between two weight tensors at interpolation ratio t.

    t=0 returns w_a, t=1 returns w_b.
    Falls back to linear interpolation when vectors are nearly parallel (omega < 1e-6).
    """
    a         = w_a.flatten().float()
    b         = w_b.flatten().float()
    cos_omega = (a / a.norm()).dot(b / b.norm()).clamp(-1.0, 1.0)
    omega     = cos_omega.acos()
    if omega.abs() < 1e-6:
        return (1 - t) * w_a + t * w_b         # linear fallback when nearly parallel
    out = (math.sin((1 - t) * omega) * a + math.sin(t * omega) * b) / math.sin(omega)
    return out.view_as(w_a).to(w_a.dtype)
```

### DARE (Drop And REscale)

Randomly drops task-vector parameters and rescales survivors to reduce interference during merging (Yu et al., 2023):

$$\tau_\mathrm{DARE}[i] = \begin{cases} \tau[i] / (1-p) & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}$$

```python
def dare_merge(
    base_sd: dict,
    ft_sds: list,
    lambdas: list,
    drop_rate: float = 0.9,
) -> dict:
    """DARE: Drop And REscale task vectors before merging.

    Randomly drops drop_rate fraction of each task vector's parameters,
    rescales survivors by 1/(1-drop_rate) to preserve expected magnitude,
    then combines scaled task vectors linearly.

    Typical use: drop_rate=0.9 removes 90% of the delta, dramatically
    reducing interference when merging 3+ diverse fine-tunes.
    """
    merged = {k: base_sd[k].clone().float() for k in base_sd}
    for ft_sd, lam in zip(ft_sds, lambdas):
        for k in base_sd:
            delta   = ft_sd[k].float() - base_sd[k].float()
            mask    = (torch.rand_like(delta) > drop_rate).float()
            rescaled = delta * mask / (1.0 - drop_rate)
            merged[k] = merged[k] + lam * rescaled
    return {k: v.to(base_sd[k].dtype) for k, v in merged.items()}
```

### TIES-Merging

Three-step conflict resolution across multiple fine-tuned models: **T**rim → **E**lect sign → **D**isjoint merge.

```python
def ties_merge(
    base_sd: dict,
    ft_sds: list,
    lambdas: list,
    density: float = 0.7,
) -> dict:
    """TIES: Trim-Elect-Sign merging for combining multiple SFT/LoRA checkpoints.

    Args:
        density: fraction of task-vector parameters to keep (by magnitude).
                 0.7 = keep top 70%, zero out the rest.
    """
    taus = [task_vector(base_sd, ft) for ft in ft_sds]

    # Step 1 — Trim: zero out the bottom (1 - density) fraction by magnitude
    trimmed = []
    for tau in taus:
        trimmed_tau = {}
        for k, v in tau.items():
            threshold = torch.quantile(v.abs().float(), 1.0 - density)
            trimmed_tau[k] = v * (v.abs() >= threshold)
        trimmed.append(trimmed_tau)

    # Steps 2 & 3 — Elect majority sign, merge only parameters that agree
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

Penalizes updates to weights important for previously learned tasks, weighted by their Fisher information (Kirkpatrick et al., 2017):

$$\mathcal{L}_\mathrm{EWC} = \mathcal{L}_\mathrm{new} + \frac{\lambda}{2}\sum_i F_i\,(\theta_i - \theta_i^*)^2$$

```python
class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting.

    Usage:
        ewc = EWC(model, old_task_dataloader, lam=1000.0)
        for batch in new_task_data:
            loss = task_loss(model, batch) + ewc.penalty(model)
            loss.backward()
            optimizer.step()
    """

    def __init__(self, model: nn.Module, dataset, lam: float = 1000.0):
        self.lam    = lam
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(model, dataset)

    def _compute_fisher(self, model: nn.Module, dataset) -> dict:
        """Approximates diagonal Fisher information via squared gradients."""
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        n_samples = len(dataset)
        for x, y in dataset:
            model.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
        return {name: f / n_samples for name, f in fisher.items()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC regularization term to add to the task loss."""
        loss = torch.tensor(0.0)
        for name, p in model.named_parameters():
            loss = loss + (self.fisher[name] * (p - self.params[name]) ** 2).sum()
        return self.lam * loss
```

---

## Training Utilities

Practical utilities for writing production training loops: gradient accumulation, gradient clipping, mixed-precision, and gradient checkpointing. These are used across every training stage.

### Gradient Accumulation Training Loop

When a single batch does not fit in GPU memory, accumulate gradients across $k$ micro-batches before stepping the optimizer. Effective batch size = `micro_batch_size × accum_steps × n_gpus`.

```python
def train_with_accumulation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    accum_steps: int = 8,
    max_grad_norm: float = 1.0,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> float:
    """One epoch of training with gradient accumulation and optional AMP.

    Args:
        accum_steps:    number of micro-batches before each optimizer.step().
        max_grad_norm:  gradient clipping threshold (1.0 is standard for LLMs).
        scaler:         GradScaler for FP16/BF16 AMP; pass None for FP32.

    Returns:
        mean loss over the epoch.
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    n_batches  = len(dataloader)

    for step, (input_ids, labels) in enumerate(dataloader, 1):
        # Determine whether this is the last micro-batch in the accumulation window
        is_accum_step = (step % accum_steps == 0) or (step == n_batches)

        # Forward + backward
        ctx = contextlib.nullcontext() if scaler is None else torch.autocast("cuda")
        with ctx:
            logits = model(input_ids)
            loss   = causal_lm_loss(logits, labels) / accum_steps  # scale for accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum_steps   # undo scale for logging

        if is_accum_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / n_batches
```

### Gradient Checkpointing

Trades compute for memory by recomputing activations during the backward pass instead of storing them. Reduces activation memory from O(layers × sequence) to O(sqrt(layers × sequence)) at the cost of ~33% more compute.

```python
from torch.utils.checkpoint import checkpoint


class CheckpointedDecoderLayer(nn.Module):
    """DecoderLayer wrapper that applies gradient checkpointing during training.

    Wraps any nn.Module's forward call with torch.utils.checkpoint.checkpoint.
    Activations from the forward pass are NOT stored; they are recomputed
    during backprop, reducing peak VRAM at the cost of ~33% extra FLOPs.

    When to use:
    - Training large models where activation memory (not parameter memory) is the
      bottleneck. Typically: sequences > 4K tokens, models > 13B parameters.
    - Do NOT use during evaluation — the recomputation overhead is wasted.
    """
    def __init__(self, layer: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.layer        = layer
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.training:
            # checkpoint requires all inputs to be Tensors; pack kwargs separately
            def custom_forward(x_):
                return self.layer(x_, **kwargs)
            return checkpoint(custom_forward, x, use_reentrant=self.use_reentrant)
        return self.layer(x, **kwargs)


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Wraps all DecoderLayer instances in a model with gradient checkpointing.

    Modifies model.layers in-place. Compatible with LoRA — the LoRA adapters
    are recomputed normally during the checkpointed backward pass.

    Example:
        model = TinyGPT(...)
        model = enable_gradient_checkpointing(model)
        # Now trains with ~40% less activation memory
    """
    for i, layer in enumerate(model.layers):
        model.layers[i] = CheckpointedDecoderLayer(layer)
    return model
```

### Mixed-Precision Training Setup

BF16 (preferred on Ampere/Hopper) or FP16 (older hardware) with automatic loss scaling:

```python
def build_amp_components(
    dtype: str = "bf16",
) -> tuple[torch.dtype, Optional["torch.cuda.amp.GradScaler"]]:
    """Returns (autocast_dtype, grad_scaler) for mixed-precision training.

    BF16 (bfloat16):
    - Preferred on A100/H100 — native hardware support, no loss scaling needed.
    - Same exponent range as FP32 — immune to gradient overflow/underflow.
    - Scaler is None (not needed).

    FP16 (float16):
    - Required on V100/T4 and older hardware.
    - Smaller exponent range can cause gradient underflow — GradScaler required.
    - GradScaler dynamically scales loss up during forward, down before optimizer.

    Usage:
        dtype, scaler = build_amp_components("bf16")
        with torch.autocast("cuda", dtype=dtype):
            loss = model(input_ids)
        # If scaler is not None: scaler.scale(loss).backward(); scaler.step(...)
    """
    if dtype == "bf16":
        return torch.bfloat16, None
    elif dtype == "fp16":
        return torch.float16, torch.cuda.amp.GradScaler(
            init_scale=2 ** 16,
            growth_interval=2000,
        )
    else:
        raise ValueError(f"Unknown dtype '{dtype}'. Use 'bf16' or 'fp16'.")
```

> [!WARNING]
> **Never mix BF16 model weights with FP16 autocast** — the autocast dtype must match the dtype the model was loaded in. Use `model.to(torch.bfloat16)` before training with BF16, and verify with `next(model.parameters()).dtype`.

---

## Final Assembly — TinyGPT

A minimal, production-style decoder-only model assembling all components from this treasury: GQA, RoPE, RMSNorm, SwiGLU, and weight tying.

```python
class DecoderLayer(nn.Module):
    """Single transformer decoder layer: pre-norm GQA + SwiGLU FFN.

    Follows the pre-normalization (pre-norm) pattern: LayerNorm before
    each sub-layer, not after. This is the 2025–2026 production standard
    (Llama, Qwen, DeepSeek all use pre-norm RMSNorm).
    """
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

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs=freqs)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    """Minimal production-style decoder LLM using GQA, RMSNorm, SwiGLU, and RoPE.

    Default config (d=512, 6 layers, 8 heads, 2 KV groups): ~50M parameters.

    Architecture choices:
    - GQA (n_kv=2): 4× smaller KV cache vs MHA at 8 heads
    - RoPE: precomputed and sliced to actual sequence length in forward()
    - Pre-norm RMSNorm: training stability
    - Weight tying: embedding and output head share weights (reduces params ~10%)
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
        self.head.weight = self.embed.weight    # weight tying
        # Register frequency buffer — sliced per call to actual sequence length
        self.register_buffer("freqs", precompute_freqs(d_model // n_heads, max_seq))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) integer token indices, T ≤ max_seq.
        Returns:
            (B, T, vocab_size) logit tensor.
        """
        B, T = idx.shape
        x     = self.embed(idx)                 # (B, T, d_model)
        freqs = self.freqs[:T]                  # (T, d_head/2) — slice to actual length
        for layer in self.layers:
            x = layer(x, freqs=freqs)
        return self.head(self.norm(x))          # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive token generation with temperature and top-k sampling.

        Args:
            prompt_ids:     (1, T_prompt) or (B, T_prompt) integer token ids.
            max_new_tokens: number of new tokens to generate.
            temperature:    >1 = more random, <1 = more greedy, 1.0 = standard.
            top_k:          nucleus of top_k tokens to sample from; 0 = no restriction.
        Returns:
            (B, T_prompt + max_new_tokens) integer token ids.
        """
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(ids)[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids     = torch.cat([ids, next_id], dim=1)
        return ids
```

---

[← Previous Chapter](app_f_hyperparams.md) | [Table of Contents](../README.md#table-of-contents)
