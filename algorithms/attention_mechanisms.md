# Attention Mechanisms

Attention is the core operation in every transformer. It computes a weighted sum of values, where the weights express how relevant each key is to a given query.

---

## Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

- $Q \in \mathbb{R}^{T \times d_k}$, $K \in \mathbb{R}^{S \times d_k}$, $V \in \mathbb{R}^{S \times d_v}$
- $\sqrt{d_k}$ scaling prevents softmax saturation for large head dimensions.
- $M$ is an optional mask (causal, padding, or attention bias).

**Complexity:** $O(T \cdot S)$ time and memory — the quadratic bottleneck at long contexts.

---

## Multi-Head Attention (MHA)

Split queries, keys, and values into $H$ independent heads, compute attention in each, then concatenate:

$$
\text{head}_h = \text{Attention}(QW_h^Q,\, KW_h^K,\, VW_h^V)
$$

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\,W^O
$$

Each head learns a different attention pattern (local, global, syntactic, semantic). Standard in GPT-2/3, BERT, and early LLaMA models.

**KV cache size:** $2 \times H \times d_\text{head} \times L$ per token, where $L$ is the number of layers.

---

## Multi-Query Attention (MQA)

All query heads share a **single** key and value head:

$$
\text{head}_h = \text{Attention}(QW_h^Q,\, KW^K,\, VW^V)
$$

**Result:** KV cache shrinks by $H\times$ — critical for memory-bound inference. Slight quality drop vs MHA. Used in Falcon, PaLM-2.

---

## Grouped Query Attention (GQA)

Interpolates between MHA and MQA: $G$ groups of query heads, each sharing one KV head ($G < H$):

$$
\text{head}_h = \text{Attention}(QW_h^Q,\, KW_{g(h)}^K,\, VW_{g(h)}^V)
$$

where $g(h) = \lfloor h \cdot G / H \rfloor$ maps query head to its KV group.

**Trade-off:** KV cache reduced by $H/G\times$ with minimal quality loss. Standard in LLaMA 3, Mistral, Gemma. $G=1$ recovers MQA; $G=H$ recovers MHA.

| Config | KV Heads | Quality | Cache |
|---|---|---|---|
| MHA | $H$ | Best | $H\times$ |
| GQA ($G=8$) | $8$ | Near-MHA | $H/8\times$ |
| MQA ($G=1$) | $1$ | Good | $1\times$ |

---

## Multi-Head Latent Attention (MLA)

Introduced in DeepSeek-V2/V3. Compresses KV into a low-dimensional latent vector:

$$
c_t^{KV} = W^{DKV} h_t \qquad \text{(compress: } d_\text{model} \to d_c \ll d_\text{model}\text{)}
$$

$$
[k_t^C;\, v_t^C] = W^{UKV}\, c_t^{KV} \qquad \text{(expand at attention time)}
$$

Only $c_t^{KV}$ (the compressed latent, size $d_c$) is cached — not the full KV pair. This reduces the KV cache by $\sim 5\text{--}13\times$ vs MHA while matching or exceeding its quality.

**Decoupled RoPE:** A separate rope-compressed key $k_t^R$ carries positional information and is cached alongside $c_t^{KV}$.

---

## Flash Attention (v1/v2/v3)

Standard attention materializes the full $T \times T$ attention matrix in HBM (GPU memory), causing memory bandwidth bottleneck. Flash Attention rewrites the computation using **IO-aware tiling**:

1. Split $Q$, $K$, $V$ into tiles that fit in SRAM (fast on-chip memory).
2. Compute partial softmax and partial output in SRAM, accumulating with a running max trick.
3. Never write the $T \times T$ matrix to HBM.

**Results:**
- FA1: 2–4× faster, exact same output.
- FA2: further optimized warp scheduling; 2× over FA1.
- FA3: FP8 support, async pipeline; 2× over FA2 on H100.

Memory: $O(T)$ instead of $O(T^2)$.

---

## Sliding Window Attention

Restricts each token to attend only to the $w$ most recent tokens:

$$
\text{score}_{ij} = 0 \quad \text{if } i - j > w
$$

**Used in:** Mistral 7B ($w = 4096$), Gemma. Enables efficient processing of long sequences by keeping attention local. Information can still propagate globally across layers.

**Limitation:** Cannot directly attend to distant tokens within a single layer — relies on depth for global information aggregation.

---

## Ring Attention

Distributes the $T \times T$ attention computation across $N$ devices arranged in a ring topology:

1. Each device holds $T/N$ query tokens and $T/N$ KV tokens.
2. KV blocks rotate around the ring while each device computes its local attention contribution.
3. A running softmax accumulator merges contributions as KV blocks arrive.

**Enables:** Context lengths of 1M+ tokens by eliminating the $O(T^2/N)$ memory requirement per device. Used for ultra-long-context training (1M–10M tokens).

---

## Comparison Table

| Ring Attention | Distributed | Exact | Million-token training |

---

## State Space Models (Mamba / SSM)

Replaces attention with a linear-time selective scan. Unlike the $O(n^2)$ attention mechanism, SSMs maintain a fixed-size latent state $h_t$ that evolves over time:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
$$

- **Selective Scan:** $A, B, C$ are functions of the input $x_t$, allowing the model to focus on relevant tokens.
- **Complexity:** $O(n)$ time and $O(1)$ inference memory (no KV cache).
- **Hybrid (Jamba/Bolt):** Alternating Transformer and Mamba blocks to balance reasoning and efficiency.

