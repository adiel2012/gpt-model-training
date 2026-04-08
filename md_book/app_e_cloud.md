# Cloud Provider Reference

## Overview

GPU compute is available through major cloud providers, specialized AI cloud providers, and spot-market brokers. Pricing varies significantly by reservation type, region, and demand.

\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{2.2cm}L{3.2cm}L{2.8cm}L{3.5cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**Provider**} & \textcolor{white}{**GPU Options**} & \textcolor{white}{**On-Demand Price**} & \textcolor{white}{**Best For**} \\
\midrule
AWS & p4d.24xl (8$\times$A100), p5.48xl (8$\times$H100) & \$32/hr -- \$98/hr & Enterprise, compliance \\
Google Cloud & A3 (8$\times$H100), TPU v5p pods & \$30/hr -- \$120/hr & JAX workloads, TPU \\
Azure & ND A100 v4, ND H100 v5 & \$32/hr -- \$100/hr & Microsoft ecosystem \\
Lambda Labs & 8$\times$H100 SXM & \$24/hr & Cost-effective on-demand \\
CoreWeave & 8$\times$H100/A100 & \$20--\$28/hr & Flexible allocation \\
RunPod & 1--8$\times$ A100/H100 & \$2--\$5/hr per GPU & Spot compute, research \\
Vast.ai & Community GPUs & \$0.80--\$3/hr & Budget fine-tuning \\
\bottomrule
\end{tabular}
\caption{Cloud GPU pricing reference (April 2026; prices indicative, verify current rates)}
\end{table}

## Reservation vs.\ Spot Strategies


  - **On-demand:** Full price; available immediately. Use for production inference or short experiments.
  - **Reserved instances (1--3 years):** 30--60\% discount vs.\ on-demand. Commit to this for sustained training workloads.
  - **Spot / preemptible:** 60--90\% cheaper; can be reclaimed with 30--120 seconds notice. Viable only with robust checkpoint-and-resume infrastructure. Run FSDP training with async checkpointing every 30 minutes.
  - **Savings plans (AWS):** Commit to a spend level, not a specific instance type. Flexible across instance families.


## Cost Estimation: Representative Workloads

\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{3.8cm}L{2.5cm}L{2.5cm}L{2.2cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**Workload**} & \textcolor{white}{**Hardware**} & \textcolor{white}{**Duration**} & \textcolor{white}{**Est.\ Cost**} \\
\midrule
QLoRA fine-tune 8B, 50K examples & 1$\times$A100 40GB & $\sim$4 hours & \$8--\$12 \\
Full fine-tune 8B, 100K examples & 4$\times$A100 80GB & $\sim$8 hours & \$120--\$160 \\
DPO alignment 8B & 2$\times$A100 80GB & $\sim$6 hours & \$60--\$90 \\
GRPO reasoning 7B & 8$\times$A100 80GB & $\sim$24 hours & \$600--\$800 \\
Pre-train 7B, 1T tokens & 64$\times$H100 & $\sim$21 days & \$90K--\$130K \\
Pre-train 70B, 2T tokens & 512$\times$H100 & $\sim$30 days & \$1.5M--\$2.5M \\
\bottomrule
\end{tabular}
\caption{Cost estimates for representative LLM workloads (indicative, 2026)}
\end{table}

## Infrastructure Checklist

> **Before Starting a Long Training Run**
>
> - \checkmark Async checkpointing configured and tested (simulate a restart).
>   - \checkmark Monitoring dashboard: GPU utilization, loss curve, gradient norm, memory.
>   - \checkmark Evaluation harness running on a held-out set every N steps.
>   - \checkmark Alert on loss spikes $>2\times$ rolling mean and OOM errors.
>   - \checkmark Estimated total cost approved; spot interruption rate acceptable.
>   - \checkmark Data pipeline throughput tested: can your dataloader saturate all GPUs?
