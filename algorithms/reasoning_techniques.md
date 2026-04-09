# Reasoning Techniques

Methods for improving LLM reasoning at training time (RL, process rewards) and inference time (sampling, search, extended compute).

---

## Chain-of-Thought (CoT)

[wei2022cot] Prompting the model to produce intermediate reasoning steps before the final answer dramatically improves performance on multi-step problems.

**Zero-shot CoT:** Append "Let's think step by step." to the prompt. Emergent at ~100B parameters; works on smaller models after fine-tuning.

**Few-shot CoT:** Provide exemplars with full reasoning chains:
```
Q: If there are 3 cars with 4 wheels each, how many wheels total?
A: Each car has 4 wheels. 3 cars × 4 wheels = 12 wheels. Answer: 12.

Q: [new question]
A: [model generates chain then answer]
```

**Why it works:** Forces the model to allocate computation to intermediate states, effectively simulating a search process in the forward pass.

---

## Self-Consistency

[wang2023self] Sample $N$ independent CoT completions, then majority-vote on the final answer:

$$
\hat{y} = \arg\max_y \sum_{i=1}^N \mathbb{1}[y_i = y]
$$

**Properties:**
- No fine-tuning required — pure inference-time compute.
- Consistent gains up to $N \approx 40$ samples; diminishing returns beyond.
- Works best when the answer space is discrete (math, code, classification).
- $N = 10$–$20$ is the practical sweet spot (cost vs. gain).

---

## Process Reward Model (PRM)

[lightman2023lets] A reward model that scores each **reasoning step** independently, rather than only the final answer.

**Training signal:** Human annotators label each step as correct/incorrect/neutral. The PRM $r_\phi(x, y_{1:t})$ predicts step-level correctness.

**Use cases:**
1. **Best-of-N with PRM:** Sample $N$ chains, score by minimum step reward (or product), return the best.
2. **MCTS guidance:** Use PRM as the value function for tree search (see MCTS section below).
3. **Step-DPO:** Construct step-level preference pairs for DPO training.

**ORM vs PRM:**
- ORM (Outcome Reward Model): one scalar at the end. Cheap to label but sparse signal.
- PRM: reward per step. Dense signal, expensive to label, but dramatically better at guiding search.

---

## Best-of-N Sampling (Rejection Sampling)

Generate $N$ candidate responses, score all with a reward model, return the highest-scoring:

$$
\hat{y} = \arg\max_{y \in \{y_1, \ldots, y_N\}} r(x, y)
$$

**Scaling law:** The best-of-$N$ pass rate for a problem with base solve rate $p$ is:

$$
P(\text{at least one correct}) = 1 - (1 - p)^N
$$

At $p = 0.1$ and $N = 100$: $P = 1 - 0.9^{100} \approx 99.997\%$.

**Compute-quality trade-off:** Doubling $N$ has diminishing returns. The effective gain scales as $O(\log N)$ for many distributions.

**Use in training:** Rejection sampling fine-tuning (RFT) — generate $N$ solutions, keep only correct ones, fine-tune on them. Bootstraps reasoning capability without RL.

---

## Monte Carlo Tree Search (MCTS)

Adapts the classical game-tree search algorithm to language generation. Enables deeper exploration of reasoning paths than linear chain sampling.

**Algorithm:**
1. **Selection:** Traverse the tree using UCB (Upper Confidence Bound):

$$
a^* = \arg\max_a \left[Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}\right]
$$

2. **Expansion:** Generate new child nodes (next reasoning steps) from the policy.
3. **Simulation (rollout):** Complete the reasoning chain from the expanded node.
4. **Backpropagation:** Update $Q$ and $N$ values along the path.

**Value function:** PRM provides $Q(s, a)$ — the expected correctness of continuing from a given step.

**Practical limitations:**
- Requires a reliable PRM — poor value estimates cause MCTS to diverge.
- Latency: $O(N \times T)$ tokens generated per query (vs. $O(T)$ for greedy).
- Memory: storing the tree for long sequences is expensive.

**Used in:** AlphaCode 2, o1/o3 (rumored), DeepSeek-R1 data generation phase.

---

## Test-Time Compute Scaling

The observation that allocating more computation at inference — via longer chains, more samples, or tree search — improves accuracy similarly to scaling model parameters.

**Scaling strategies:**
| Strategy | Compute | Quality Gain |
|---|---|---|
| Greedy decoding | $1\times$ | Baseline |
| Best-of-N (N=16) | $16\times$ | +10–30% on math |
| Self-consistency (N=16) | $16\times$ | +5–20% |
| MCTS (N=100 nodes) | $100\times$ | +15–40% on hard problems |
| Extended thinking chains | $2\text{–}10\times$ tokens | +10–25% |

**Key finding [snell2024scaling]:** For a fixed compute budget, it is often better to use a smaller model with more test-time compute than a larger model with greedy decoding.

---

## Extended Thinking / Thinking Tokens

Used in Claude 3.7 Sonnet, DeepSeek-R1, QwQ, Kimi k1.5. The model generates a long internal scratchpad (the "thinking" or `<think>` block) before producing the final response.

**Training approach:**
1. Fine-tune on data where correct answers are preceded by long reasoning traces.
2. Use RLVR/GRPO to reinforce reasoning traces that lead to correct answers.
3. The model learns to allocate thinking tokens proportionally to problem difficulty.

**Thinking token budget:** Some systems expose a `budget_tokens` parameter that caps the thinking length. Longer budgets improve hard problem accuracy at the cost of latency.

**Quiet-STaR [zelikman2024quietstar]:** Inserts thinking tokens at every token position during training (not just before final answers), teaching the model to maintain a continuous internal monologue.

---

## R1-Style Pure RL (Cold-Start Reasoning)

[deepseek2025r1] DeepSeek-R1 demonstrated that reasoning ability can emerge from RL on verifiable tasks without any supervised reasoning data.

**Training stages:**
1. **Cold-start data:** A small set of long-chain reasoning examples to initialize.
2. **GRPO on math/code:** RL with binary verifiable rewards — no PRM, no human labels.
3. **Rejection sampling SFT:** Generate solutions with the RL model, keep correct ones, fine-tune.
4. **Full RL pipeline:** Final GRPO run for reasoning + DPO for style/safety.

**Emergent behaviors observed:**
- Self-verification: model re-checks its own steps.
- Backtracking: model writes "Wait, let me reconsider..." and corrects errors.
- Reflection: model identifies which step went wrong.

These were not explicitly trained — they emerged from RL pressure to get correct answers.

---

## Comparison

| Technique | Train-Time Cost | Inference Cost | Best For |
|---|---|---|---|
| Zero-shot CoT | None | Minimal | General reasoning |
| Few-shot CoT | None | Minimal | Specific task format |
| Self-consistency | None | $N\times$ | Discrete answer tasks |
| Best-of-N + ORM | RM training | $N\times$ | Tasks with reliable reward |
| Best-of-N + PRM | PRM training | $N\times$ | Multi-step math/logic |
| MCTS | PRM training | $N \times T\times$ | Very hard, structured tasks |
| Extended thinking | RL fine-tuning | $2\text{–}10\times$ tokens | Open-ended hard problems |
| R1-style RL | RL on verifiable | $1\text{–}5\times$ | Math, code, reasoning |
