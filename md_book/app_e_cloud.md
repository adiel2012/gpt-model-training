# Cloud Provider Reference

## Overview

GPU compute is available through major cloud providers, specialized AI cloud providers, and spot-market brokers. Pricing varies significantly by reservation type, region, and demand.

| **Provider** | **GPU Examples** | **On-Demand Price** | **Best For** |
|---|---|---|---|
| AWS | p4d.24xl (8$\times$A100), p5.48xl (8$\times$H100) | $32/hr -- $98/hr | Enterprise, compliance |
| Google Cloud | A3 (8$\times$H100), TPU v5p pods | $30/hr -- $120/hr | JAX workloads, TPU |
| Azure | ND A100 v4, ND H100 v5 | $32/hr -- $100/hr | Microsoft ecosystem |
| Lambda Labs | 8$\times$H100 SXM | $24/hr | Cost-effective on-demand |
| CoreWeave | 8$\times$H100/A100 | $20--$28/hr | Flexible allocation |
| RunPod | 1--8$\times$ A100/H100 | $2--$5/hr per GPU | Spot compute, research |
| Vast.ai | Community GPUs | $0.80--$3/hr | Budget fine-tuning |

*Table: Cloud GPU pricing reference (April 2026; prices indicative, verify current rates)*

## Reservation vs. Spot Strategies

  - **On-demand:** Full price; available immediately. Use for production inference or short experiments.
  - **Reserved instances (1--3 years):** 30--60% discount vs. on-demand. Commit to this for sustained training workloads.
  - **Spot / preemptible:** 60--90% cheaper; can be reclaimed with 30--120 seconds notice. Viable only with robust checkpoint-and-resume infrastructure. Run FSDP training with async checkpointing every 30 minutes.
  - **Savings plans (AWS):** Commit to a spend level, not a specific instance type. Flexible across instance families.

## Cost Estimation: Representative Workloads

| **Workload** | **Hardware** | **Duration** | **Est. Cost** |
|---|---|---|---|
| QLoRA fine-tune 8B, 50K examples | 1$\times$A100 40GB | $\sim$4 hours | $8--$12 |
| Full fine-tune 8B, 100K examples | 4$\times$A100 80GB | $\sim$8 hours | $120--$160 |
| DPO alignment 8B | 2$\times$A100 80GB | $\sim$6 hours | $60--$90 |
| GRPO reasoning 7B | 8$\times$A100 80GB | $\sim$24 hours | $600--$800 |
| Pre-train 7B, 1T tokens | 64$\times$H100 | $\sim$21 days | $90K--$130K |
| Pre-train 70B, 2T tokens | 512$\times$H100 | $\sim$30 days | $1.5M--$2.5M |

*Table: Cost estimates for representative LLM workloads (indicative, 2026)*

## Infrastructure Checklist

> **Before Starting a Long Training Run**
>
> - [x] Async checkpointing configured and tested (simulate a restart).
>   - [x] Monitoring dashboard: GPU utilization, loss curve, gradient norm, memory.
>   - [x] Evaluation harness running on a held-out set every N steps.
>   - [x] Alert on loss spikes $>2\times$ rolling mean and OOM errors.
>   - [x] Estimated total cost approved; spot interruption rate acceptable.
>   - [x] Data pipeline throughput tested: can your dataloader saturate all GPUs?
