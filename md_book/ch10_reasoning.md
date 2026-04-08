# Training for Reasoning
\minitoc

\begin{chapteroverview}
  
    - Analyze Chain-of-Thought (CoT) and its 2025--2026 training methods.
    - Master Process Reward Models (PRM) for step-level supervision.
    - Implement test-time compute scaling and search (MCTS, Best-of-N).
    - Understand Thinking Tokens (Quiet-STaR) for emergent reasoning.
  
\end{chapteroverview}

## The Rise of Reasoning Models

Models that break complex problems into explicit intermediate steps---the most significant capability development of 2025--2026. Reasoning models sacrifice token efficiency for accuracy on hard tasks.

## Chain-of-Thought (CoT)

Prompting or training models to generate intermediate reasoning steps before the final answer. Few-shot CoT [wei2022chain] demonstrated step-by-step prompting dramatically improves multi-step arithmetic, symbolic reasoning, and commonsense tasks.


  - **Zero-shot CoT:** Append ``Let's think step by step'' to the prompt.
  - **Few-shot CoT:** Provide example (question, reasoning chain, answer) triples in the prompt.
  - **Self-consistency:** Sample $k$ CoT paths and take the majority vote answer. +5--15\% on GSM8K vs.\ single-sample CoT.


## Process Reward Models (PRM)

Instead of rewarding only the final answer, PRMs assign a score to each reasoning step:

  - Denser supervision signal: $n$ steps $\rightarrow$ $n$ reward signals per example.
  - Enables early pruning of incorrect branches during tree search.
  - Used in OpenAI's o-series for competition math and coding [lightman2023lets].


Annotation challenge: step-level labels are expensive. Solutions: automated labeling via outcome consistency (steps on paths leading to correct answers are labeled positive) and synthetic step decomposition from a stronger model.

## Test-Time Search

Allocate more inference compute per query to find better solutions:

  - **Best-of-N:** Sample $N$ complete solutions; select via outcome RM or verifier.
  - **Beam Search over CoT:** Expand $k$ partial chains at each step; prune with PRM.
  - **Monte Carlo Tree Search (MCTS):** Full tree exploration with PRM scoring. Used in AlphaCode 2 and OpenAI o3.


> **Test-Time Compute Scaling**
>
> \textcite{snell2024scaling} showed that test-time compute can match training-time compute scaling: a model with 16$\times$ more test-time compute can match a model 16$\times$ larger. This creates a new performance axis independent of parameter count.

## TLT -- Accelerating Training

MIT: a smaller ``drafter'' model predicts outputs; a larger verifier model confirms them [leviathan2023fast]. 70--210\% acceleration with preserved accuracy. The drafter is reusable for speculative decoding at inference.

## R1-Style Pure RL Training

DeepSeek demonstrated that pure RL on verifiable tasks produces emergent reasoning without any SFT [guo2025deepseekr1]:

  - Start from the base model (no instruction tuning).
  - Apply GRPO (formulation \S\ref*{form:grpo}) with binary correctness reward on math/code tasks.
  - Emergent behaviors: self-verification (``wait, let me reconsider''), extended thinking, strategy switching.


> **Pure RL Training Instability**
>
> R1-Zero (pure RL, no SFT cold start) exhibits language mixing, repetitive patterns, and poor readability. The DeepSeek production pipeline uses a cold-start SFT phase (a few thousand high-quality CoT examples) before GRPO to stabilize training.
## Thinking Tokens (Quiet-STaR)

Rather than forcing reasoning into a separate pre-response phase, Quiet-STaR [schulman2024quietstar] trains the model to generate ``internal monologue'' tokens at every position in the sequence:

  - **Divergent Thinking:** The model generates multiple parallel reasoning traces (thoughts) in the background.
  - **Rationalization:** It selects the thought that most improves the prediction of the subsequent actual text tokens.
  - **Token-Level RL:** Thoughts are rewarded based on how much they reduce the cross-entropy loss of the next real token.

This allows models to benefit from reasoning even on tasks where explicit Chain-of-Thought is not requested, and it creates a unified architecture for both ``fast'' (direct) and ``slow'' (reasoned) processing.

> **The Thinking Token Trade-off**
>
> Thinking tokens increase inference compute (latency and FLOPs) but can dramatically reduce the model size required to achieve a certain level of performance. A 7B model with 16 thinking tokens can match a 70B model on several reasoning-heavy benchmarks.
