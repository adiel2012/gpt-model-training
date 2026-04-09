# Distributed Training Parallelism

Strategies for distributing LLM training across multiple GPUs and nodes. Modern large-scale training combines all three major axes simultaneously ("3D parallelism").

---

## Data Parallelism (DP)

The simplest form: each device holds a full copy of the model and processes a different mini-batch. Gradients are all-reduced (averaged) across devices after each backward pass.

**Synchronous SGD with all-reduce:**

$$
g_\text{avg} = \frac{1}{N} \sum_{i=1}^N g_i
$$

**Communication cost:** One all-reduce of size equal to the model parameters per step.

**Limitation:** Each device must hold the full model — infeasible beyond ~7B parameters on standard GPU memory.

---

## ZeRO (Zero Redundancy Optimizer)

[rajbhandari2020zero] Shards the optimizer state, gradients, and parameters across data-parallel ranks, eliminating redundancy.

**Three stages:**

| Stage | What Is Sharded | Memory per Device | Communication |
|---|---|---|---|
| ZeRO-1 | Optimizer states | $\frac{1}{N}$ optimizer + full params | All-reduce gradients |
| ZeRO-2 | Optimizer states + gradients | $\frac{1}{N}$ opt+grad + full params | Reduce-scatter gradients |
| ZeRO-3 | Optimizer states + gradients + parameters | $\frac{1}{N}$ everything | All-gather params on demand |

**ZeRO-3 memory per device:** $\approx \frac{16\,\text{bytes} \times M}{N}$ where $M$ is parameter count and $N$ is number of devices ($16$ bytes = FP16 param + FP32 copy + FP32 moments).

**ZeRO-Infinity:** Extends ZeRO-3 to offload to CPU RAM and NVMe SSD, enabling training of trillion-parameter models.

---

## FSDP (Fully Sharded Data Parallel)

PyTorch's native ZeRO-3 implementation. Parameters are sharded across ranks; each rank all-gathers the parameters for each layer as needed during the forward and backward pass, then discards them.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
)
```

**FSDP2 (PyTorch 2.4+):** Cleaner API, per-parameter sharding granularity, better async prefetching for overlapping compute and communication.

---

## Tensor Parallelism (TP)

[shoeybi2019megatron] Splits individual weight matrices across devices within a layer. Each device computes a partial result; results are combined with a single all-reduce per layer.

**Column-parallel linear:** Split $W$ column-wise across $N$ devices:

$$
Y = XW = X [W_1 | W_2 | \cdots | W_N] = [XW_1 | XW_2 | \cdots | XW_N]
$$

Each device computes $XW_i$ independently; results are concatenated (or all-reduced for row-parallel).

**Row-parallel linear:** Split $W$ row-wise; requires an all-reduce to sum partial outputs.

**For MHA:** Query, Key, Value projections are column-parallel (split across heads). Output projection is row-parallel.

**Communication:** One all-reduce per transformer layer per forward pass. Requires high-bandwidth interconnect (NVLink); inefficient over InfiniBand.

**Typical TP degree:** 4–8 within a node (NVLink). Cross-node TP is rare due to bandwidth constraints.

---

## Pipeline Parallelism (PP)

Splits the model's layers across devices (each device holds a consecutive subset of layers). Data flows sequentially through the pipeline.

**Naive PP (GPipe [huang2019gpipe]):**
1. Split model into $P$ stages across $P$ devices.
2. Process one micro-batch through all stages sequentially.
3. **Pipeline bubble:** Devices sit idle while waiting for upstream stages.

**Bubble fraction:** $\frac{P-1}{P + m - 1}$ where $m$ is the number of micro-batches. With $m \geq 4P$ micro-batches, bubble overhead is below 20%.

**1F1B Schedule (Interleaved) [narayanan2021efficient]:**
- Assigns multiple model chunks to each device (interleaved stages).
- Alternates between one forward and one backward pass per step.
- Reduces bubble fraction to $\frac{1}{m}$ — near-zero with enough micro-batches.

```
Device 0: F0 F1 F2 F3 | B3 B2 B1 B0
Device 1:    F0 F1 F2 F3 | B3 B2 B1 B0
```

**PP + gradient accumulation:** Micro-batches correspond to gradient accumulation steps — the local batch is split into $m$ micro-batches to fill the pipeline.

---

## Sequence Parallelism

[korthikanti2022reducing] Extends tensor parallelism to the sequence dimension for operations that don't parallelize easily along the hidden dimension (e.g., LayerNorm, Dropout).

In standard TP, each device holds the full sequence but a partial hidden dimension. Sequence parallelism splits the sequence across devices for the non-attention operations:

- Non-attention ops (LayerNorm, Dropout): split over sequence dimension.
- Attention ops: standard TP (heads split across devices).

**Result:** Further reduces activation memory by $N\times$ for the non-attention activations — critical for long-context training.

---

## 3D Parallelism

Combines all three axes for training at full scale (100B+ parameters):

```
Total GPUs = DP × TP × PP
```

**Example (LLaMA 3 405B training):**
- TP = 8 (within a node via NVLink)
- PP = 16 (across nodes)
- DP = 8 (ZeRO-1 or ZeRO-2)
- Total: 8 × 16 × 8 = 1024 GPUs

**Communication hierarchy:**
- TP: all-reduce within node (NVLink, ~600 GB/s).
- DP: all-reduce across nodes (InfiniBand, ~400 Gb/s).
- PP: send/recv across nodes (point-to-point, overlapped with compute).

---

## Activation Checkpointing (Gradient Checkpointing)

Not a parallelism strategy, but essential for memory at any scale. Instead of storing all intermediate activations for the backward pass, recompute them from checkpointed states.

**Memory vs. compute trade-off:**
- Without checkpointing: $O(L)$ activation memory, no extra compute.
- Full checkpointing: $O(1)$ activation memory (per layer), $+33\%$ compute (one extra forward pass).
- Selective checkpointing: checkpoint only the expensive layers (attention), saving 60–80% of activation memory at 5–10% compute overhead.

---

## Comparison

| Strategy | Splits | Communication | Best For |
|---|---|---|---|
| DP + ZeRO-1 | Optimizer state | All-reduce gradients | Up to ~7B |
| DP + ZeRO-3 / FSDP | Params + opt + grads | All-gather + reduce-scatter | 7B–70B |
| TP (within node) | Weight matrices | All-reduce per layer | Large matrices, fast interconnect |
| PP | Layers | Point-to-point | Very deep models, cross-node |
| 3D (DP + TP + PP) | All three | All of the above | 100B+ |
| Sequence Parallelism | Sequence dim | All-gather sequence | Long-context + TP |
