# Continual Learning and Lifelong Adjustment

> **What You Will Learn**
> - Mitigate catastrophic forgetting using EWC and Low-Rank Adaptation.
> - Implement dynamic weight updates for real-time knowledge injection.
> - Analyze the 2026 architectures designed for memory and context retention.
> - Evaluate the trade-offs between architectural stability and plasticity.

## The Forgetting Problem

EWC Fisher penalty: [Appendix G](app_g_implementation_treasury.md) in Appendix~app:continual; code: [Appendix G](app_g_implementation_treasury.md).

When a model is fine-tuned on task B, performance on previously learned task A degrades---sometimes catastrophically. This limits the ability to update deployed models without full retraining.

## Mitigation Strategies

  - **Elastic Weight Consolidation (EWC) [kirkpatrick2017overcoming**:] Penalizes updates to weights important for previous tasks, identified via the Fisher information matrix approximation.
  - **LoRA-based updates:** Fine-tune only low-rank adapter weights; base model weights remain unchanged. Task-specific adapters can be swapped at inference time.
  - **Replay-based methods:** Mix a small fraction of old task data into new training. 5\% replay often sufficient to preserve 95\% of original task performance.
  - **Hierarchical optimization:** MAML-like meta-learning for fast adaptation with minimal forgetting.
  - **LoRA merging:** Train multiple domain-specific LoRA adapters and merge them periodically, avoiding full model retraining entirely.

## Continual Pre-training

Updating the base model on new data distributions (new knowledge, new language):

  - Learning rate warmup prevents early-stage gradient explosions on new data.
  - Data mixing: 80\% new data + 20\% replay from original pre-training corpus.
  - Context extension: interleave original context-length data with long-context data during extension training.

