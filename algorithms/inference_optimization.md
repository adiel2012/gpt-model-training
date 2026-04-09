# Inference Optimization

Techniques to reduce latency, memory, and cost of LLM serving without (or with minimal) quality loss.

---

## Quantization

Reduce weight (and optionally activation) precision from FP16/BF16 to lower bit-width representations.

### INT8 (W8A8)

Quantize both weights and activations to 8-bit integers. Quality degradation is minimal for most tasks.

**Per-channel quantization:** Use a separate scale per output channel rather than a global scale:

$$
W_\text{int8} = \text{round}\!\left(\frac{W}{\Delta}\right), \quad \Delta = \frac{\max(|W|)}{127}
$$

**LLM.int8() [dettmers2022int8]:** Identifies "outlier" activation dimensions (large magnitude values that break uniform quantization) and keeps them in FP16. All other dimensions are quantized to INT8. This makes INT8 practical for large models.

### INT4 / GPTQ

**GPTQ [frantar2022gptq]:** Post-training quantization to 4-bit via second-order optimization. Processes weights layer by layer; uses the Hessian to compensate for quantization error in each weight group before moving to the next.

**Group quantization:** Use independent scales per group of $g = 128$ weights:

$$
\text{bits} = 4 + \frac{16}{g} \approx 4.125 \text{ bits effective}
$$

Quality at 4-bit is near FP16 for most models; 3-bit shows notable degradation.

### GGUF / llama.cpp formats

Quantization formats for CPU and consumer GPU inference:
- Q4_K_M, Q5_K_M, Q8_0 are common GGUF variants.
- K-quantization uses mixed precision (some layers stay at higher precision).
- Suitable for running 7B–70B models on consumer hardware.

### FP8

8-bit floating point (E4M3 or E5M2). Preserves more dynamic range than INT8.
- Supported natively on H100, B100/B200.
- Used in training (forward pass + activations) and inference.
- Flash Attention v3 adds FP8 support for the attention kernel.

### BitNet / 1-bit LLMs

[wang2023bitnet] Quantize weights to $\{-1, 0, +1\}$ (1.58 bits). Replaces multiplications with additions.
- Requires training from scratch (not a post-training technique).
- Energy and memory efficiency: up to 10× reduction in memory bandwidth.
- Quality still lags full-precision at current scales but gap is closing.

### Memory savings summary

| Format | Bits/Weight | Memory (7B model) | Quality vs BF16 |
|---|---|---|---|
| BF16 | 16 | 14 GB | Baseline |
| INT8 | 8 | 7 GB | ~99% |
| INT4 (GPTQ) | 4 | 3.5 GB | ~96–98% |
| INT3 | 3 | 2.6 GB | ~90–94% |
| FP8 | 8 | 7 GB | ~99.5% |

---

## Speculative Decoding

[leviathan2023speculative] Uses a small fast **draft model** to propose $K$ tokens at once, then the large **verifier model** evaluates all $K$ tokens in parallel (one forward pass).

**Algorithm:**
1. Draft model autoregressively generates $K$ tokens: $\tilde{y}_1, \ldots, \tilde{y}_K$.
2. Run one verifier forward pass over the $K$ draft tokens.
3. Accept or reject each draft token via rejection sampling:

$$
\text{Accept } \tilde{y}_i \text{ if } u_i \leq \frac{p_V(\tilde{y}_i)}{p_D(\tilde{y}_i)}, \quad u_i \sim \text{Uniform}(0, 1)
$$

4. If rejected at position $i$, sample from the corrected distribution and discard tokens $i+1, \ldots, K$.
5. If all $K$ accepted, the verifier generates one bonus token for free.

**Key property:** The accepted sequence has exactly the same distribution as the verifier alone — zero quality loss.

**Speedup:** Depends on the acceptance rate $\alpha$. Typical speedup: 2–3× for draft/verifier size ratio of 10–20×.

**Draft model options:** A smaller version of the same architecture, or a specialized draft model trained via distillation.

---

## Medusa

[cai2024medusa] Adds $K$ extra prediction heads to the base model, each predicting $k$ tokens ahead:

- Head 1: predicts token at position $t+1$ (offset 1).
- Head 2: predicts token at position $t+2$ (offset 2).
- ...
- Head $K$: predicts token at position $t+K$.

All heads run in parallel with one forward pass. A tree-structured verification scheme accepts subtrees of consistent predictions.

**Advantage over speculative decoding:** No separate draft model — lower memory overhead. The extra heads add ~5% parameters.

**Limitation:** Head accuracy drops with offset (harder to predict far ahead). In practice, 2–3 heads are effective; beyond that, acceptance rates fall.

---

## PagedAttention

[kwon2023vllm] The core innovation behind vLLM. Manages KV cache memory like virtual memory in an OS.

**Problem:** KV cache memory allocation is non-contiguous and wasteful under naive implementations:
- Sequences have variable length → over-allocation wastes GPU memory.
- Batch requests have different KV cache lifetimes.

**Solution:**
- Divide KV cache into fixed-size **pages** (blocks of, e.g., 16 tokens).
- Maintain a **block table** mapping logical KV positions to physical GPU memory blocks.
- Pages are allocated on-demand and freed when sequences complete.
- Pages can be **shared** across requests that share the same prefix (prefix caching).

**Result:** Near-zero KV cache memory waste, enabling 2–4× higher throughput vs naive KV cache management.

---

## Continuous Batching

[yu2022orca] Traditional static batching waits for all requests in a batch to finish before starting new ones. Continuous batching inserts new requests into the batch immediately after any sequence completes.

**Iteration-level scheduling:** At each forward pass iteration, the scheduler can:
1. Remove completed sequences and free their KV cache pages.
2. Insert new sequences from the queue.
3. Resume partially-decoded sequences.

**Result:** GPU utilization increases from 30–50% (static) to 80–95% (continuous), translating to 2–5× higher throughput at the same latency.

---

## Prefix Caching / KV Cache Reuse

For requests sharing a common prefix (system prompt, few-shot examples), compute and cache the KV activations once:

1. Hash the prefix tokens.
2. On a cache hit, skip the prefill computation for the prefix — start generation from the cached KV state.
3. LRU eviction when cache fills.

**Practical impact:** System prompts can be thousands of tokens. Prefix caching eliminates their cost entirely on cache hits.

**Radix attention [zheng2024sglang]:** Extends prefix caching to arbitrary shared prefixes using a radix tree, enabling cache reuse across requests with partially overlapping prompts.

---

## KV Cache Quantization

Quantize the stored KV cache (not just model weights) to reduce memory footprint:

- **INT8 KV:** 50% reduction in KV memory. Minimal quality loss.
- **INT4 KV:** 75% reduction. Noticeable quality loss on long contexts.
- **FP8 KV:** Supported on H100; better quality than INT8 at same memory.

**When to use:** When the KV cache is the memory bottleneck (long contexts, large batches). For a 70B model with 100K context, KV cache can exceed the model weights in memory.

---

## Comparison

| Technique | Memory | Throughput | Latency | Quality |
|---|---|---|---|---|
| INT8 quantization | 0.5× | 1.5–2× | No change | ~99% |
| INT4 quantization | 0.25× | 2–3× | No change | ~97% |
| Speculative decoding | +draft model | No change | 2–3× faster | 100% (exact) |
| Medusa heads | +5% | No change | 1.5–2× faster | ~100% |
| PagedAttention | 0.25–0.5× (KV) | 2–4× | No change | 100% |
| Continuous batching | No change | 2–5× | No change | 100% |
| Prefix caching | 0× (on hit) | High on repeated | Near-zero (on hit) | 100% |
| KV INT8 quant | 0.5× (KV only) | 1.2× | No change | ~99% |
