\part{Pre-training}

# Pre-training Objectives and Strategies
\minitoc

\begin{chapteroverview}
  
    - Master the next-token prediction (NTP) objective and its 2026 variants.
    - Review Reinforcement Pre-Training (RPT) for enhanced reasoning.
    - Analyze curriculum learning and instruction-augmented pre-training.
    - Understand the multi-stage compute-optimal context extension pipeline.
  
\end{chapteroverview}

## Next-Token Prediction
Self-supervised causal language modeling (see also \S\ref*{form:ntp} for the full NTP objective and \S\ref*{form:ppl} for perplexity (Appendix~app:objectives); code: Listing~\ref*{lst:ntp_loss}):
$$
  \mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$
No labeled data needed. Captures syntax, semantics, factual knowledge, and reasoning patterns. Despite its simplicity, this objective is sufficient to produce emergent capabilities at scale.

## Reinforcement Pre-Training (RPT)
Microsoft (2025): next-token prediction reframed as a sequential decision-making process. Each token prediction is a policy action; the model receives richer gradient signals than maximum likelihood. Fully self-supervised. Improves long-horizon coherence and reasoning traces.

## Curriculum Learning
Data organized by difficulty (simple $\rightarrow$ complex) accelerates convergence and boosts final performance. Difficulty metrics: perplexity under a smaller reference model, task complexity (number of reasoning steps required), domain specificity.

## Instruction-Response Augmented Pre-training
Synthetic instruction-response pairs in the pre-training corpus bridge the gap to SFT. Typically 1--3\% of the pre-training mix. Reduces the SFT data required by 5--10$\times$ for comparable instruction-following capability.

## Long-Context Pre-training

Short-context training followed by gradual context extension:

  - Train at 4K context for 90\% of total compute.
  - Gradually increase to 32K--128K on a filtered long-document subset.
  - Apply YaRN or LongRoPE for positional interpolation.

Training from scratch at long context is compute-inefficient; the model wastes FLOPs on mostly short sequences in the dataset.
