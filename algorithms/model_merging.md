# Model Merging

Model merging combines multiple fine-tuned models into one without additional training. This enables capability composition, continual learning without forgetting, and ensemble-like quality at single-model inference cost.

---

## Task Vectors

[ilharco2022editing] The key conceptual framework for model merging. Define the **task vector** as the weight difference between a fine-tuned model and its base:

$$
\tau_A = \theta_A - \theta_\text{base}
$$

Task vectors can be manipulated arithmetically:

- **Add a task:** $\theta' = \theta_\text{base} + \lambda\, \tau_A$ (amplify capability A).
- **Remove a task:** $\theta' = \theta_\text{base} - \lambda\, \tau_A$ (diminish capability A).
- **Combine tasks:** $\theta' = \theta_\text{base} + \lambda(\tau_A + \tau_B)$ (add both capabilities).

**Why this works:** Fine-tuning from the same base moves weights in directions that encode task-specific features. These directions are approximately orthogonal across diverse tasks — so their sum doesn't cancel out.

**Limitation:** Interference between task vectors when tasks are similar or the weight updates overlap significantly.

---

## Linear Interpolation (Model Soup)

[wortsman2022model] Average the weights of multiple models fine-tuned from the same base:

$$
\theta_\text{merged} = \frac{1}{N} \sum_{i=1}^N \theta_i
$$

**Greedy soup:** Start with the best single model; add models one by one only if they improve the validation metric.

**Why averaging works:** Models fine-tuned from the same base lie in a flat loss basin. Their average typically remains in this basin and exhibits lower variance, often outperforming any single model.

**Requirement:** All models must share the same base checkpoint. Merging models with different bases fails — the learned features are in different weight-space positions.

---

## SLERP (Spherical Linear Interpolation)

Interpolates between two models on the unit hypersphere rather than in linear weight space:

$$
\theta(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\, \theta_A + \frac{\sin(t\Omega)}{\sin\Omega}\, \theta_B
$$

where $\Omega = \arccos\!\left(\hat\theta_A \cdot \hat\theta_B\right)$ is the angle between the two weight vectors.

**Advantage over linear interpolation:** Maintains constant "distance" from the origin, preserving the scale of weight norms throughout the interpolation path. Linear interpolation can contract weight norms mid-path.

**When to use:** Merging two similarly-capable models for a smooth quality blend. Parameter $t \in [0, 1]$ controls the blend ratio.

---

## TIES-Merging

[yadav2023ties] Addresses **interference** between task vectors — conflicting signs in the same weight dimension cancel each other out. TIES applies three trimming steps:

1. **Trim:** Keep only the top-$k$% largest magnitude parameters in each task vector; zero out the rest.

$$
\tau_A^{\text{trim}} = \tau_A \odot \mathbb{1}[|\tau_A| \geq \text{topk}(|\tau_A|, k)]
$$

2. **Elect sign:** For each parameter, take the majority sign across all task vectors:

$$
\gamma_p = \text{sign}\!\left(\sum_A \tau_{A,p}^{\text{trim}}\right)
$$

3. **Disjoint merge:** Average only the task vectors that agree with the elected sign:

$$
\theta_p' = \theta_{\text{base},p} + \lambda \cdot \frac{\sum_A \tau_{A,p}^{\text{trim}} \cdot \mathbb{1}[\text{sign}(\tau_{A,p}^{\text{trim}}) = \gamma_p]}{|\{A : \tau_{A,p}^{\text{trim}} \neq 0, \text{sign}(\tau_{A,p}^{\text{trim}}) = \gamma_p\}|}
$$

**Result:** Significantly reduces interference. Outperforms simple task vector addition especially when merging 3+ models.

---

## DARE (Drop And REscale)

[yu2023language] Sparsifies task vectors by randomly dropping parameters and rescaling to preserve expected values:

$$
\tau' = \frac{1}{1 - p}\, \text{Bernoulli}(1-p) \odot \tau
$$

where $p \in [0.9, 0.99]$ is the drop rate (drop 90–99% of non-zero parameters).

**Intuition:** Fine-tuned models are over-parameterized for their tasks — most of the task vector weight is redundant. Dropping 99% of parameters with rescaling maintains the same expected update but eliminates interference between independent random projections.

**Combination with TIES:** DARE sparsification can be applied before TIES-merging for further interference reduction ("DARE-TIES").

---

## Model Breadcrumbs

[davari2023model] A variant of DARE that keeps only the very largest magnitude task vector parameters, discarding the long tail of small updates. Achieves similar quality to DARE with simpler implementation.

---

## Practical Merging Workflow

1. **Verify base checkpoint identity:** All models to merge must use exactly the same base weights.
2. **Compute task vectors:** $\tau_i = \theta_i - \theta_\text{base}$ for each model.
3. **Sparsify (DARE):** Drop 90–99% of task vector weights, rescale.
4. **Resolve conflicts (TIES):** Trim, elect signs, disjoint merge.
5. **Tune merge coefficient $\lambda$:** Grid search on validation set; $\lambda \in [0.3, 1.0]$ typically.

---

## Comparison

| Method | # Models | Quality | Interference Handling | Compute |
|---|---|---|---|---|
| Linear interpolation | 2+ | Good (same task) | None | Trivial |
| SLERP | 2 | Good (smooth blend) | None | Trivial |
| Task vectors | 2+ | Good | Minimal | Trivial |
| TIES | 3+ | Better | Explicit | Low |
| DARE | 3+ | Better | Probabilistic | Low |
| DARE-TIES | 3+ | Best | Both | Low |

**When merging fails:**
- Different base checkpoints — weights are in incompatible spaces.
- Very different tasks with large overlapping parameter directions (e.g., merging two models fine-tuned on conflicting styles).
- Large number of models ($> 8$) without sparsification — interference accumulates.
