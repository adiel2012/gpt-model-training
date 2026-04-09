# Alignment Objectives

Post-training alignment shapes model behavior toward human preferences. Methods range from full RL pipelines (PPO) to reference-free regression objectives (DPO, SimPO, KTO) and reasoning-specific RL (GRPO, RLVR).

---

## Bradley-Terry Preference Model

The foundation for most preference-based alignment. Given two responses $y_w$ (preferred) and $y_l$ (rejected) to a prompt $x$, the probability that a human prefers $y_w$ is:

$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

where $r(x, y)$ is a scalar reward model and $\sigma$ is the sigmoid function.

**Reward model training:** Minimize the binary cross-entropy loss over preference pairs:

$$
\mathcal{L}_\text{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]
$$

The trained $r_\phi$ is then used to supervise policy optimization (PPO) or to derive closed-form training objectives (DPO).

---

## PPO (Proximal Policy Optimization)

[schulman2017ppo] The classical RLHF pipeline. Optimizes the policy $\pi_\theta$ against a frozen reward model $r_\phi$ while staying close to a reference policy $\pi_\text{ref}$.

**Objective:**

$$
\mathcal{J}_\text{PPO}(\theta) = \mathbb{E}\left[\min\left(\rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

where $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the importance ratio and $A_t$ is the advantage estimate.

**KL penalty:** Added to prevent policy collapse:

$$
r_\text{shaped}(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}
$$

**Four models required simultaneously:**
1. Policy $\pi_\theta$ (trained)
2. Reference policy $\pi_\text{ref}$ (frozen)
3. Reward model $r_\phi$ (frozen)
4. Value/critic model $V_\psi$ (trained)

**Limitations:** Memory-intensive (4 models in GPU), sensitive to reward hacking, requires stable reward model. Largely replaced by DPO for offline alignment.

---

## DPO (Direct Preference Optimization)

[rafailov2023dpo] Eliminates the reward model by showing that the optimal policy under the KL-constrained RL objective can be expressed in closed form:

$$
r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)
$$

Substituting into the Bradley-Terry model and canceling $Z(x)$:

$$
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right]
$$

**Intuition:** Maximize the log-probability of preferred responses relative to the reference, while minimizing the log-probability of rejected responses relative to the reference.

**Only 2 models needed:** Policy $\pi_\theta$ (trained) + reference $\pi_\text{ref}$ (frozen). No reward model, no critic.

**Hyperparameter:** $\beta \in [0.01, 0.5]$ controls how far the policy can deviate from the reference. Typical value: 0.1–0.3.

**Limitation:** Requires paired preference data $(y_w, y_l)$. Sensitive to data quality — if pairs are noisy, training can degrade.

---

## SimPO (Simple Preference Optimization)

[meng2024simpo] Removes the reference model entirely. Uses sequence-level length-normalized log-probabilities as implicit reward:

$$
r_\text{SimPO}(x, y) = \frac{1}{|y|} \log \pi_\theta(y \mid x)
$$

**Loss:**

$$
\mathcal{L}_\text{SimPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)\right]
$$

where $\gamma > 0$ is a target margin that ensures the preferred response is not just slightly better.

**Advantages over DPO:**
- No reference model in memory — 50% memory reduction.
- Length normalization prevents the policy from preferring verbose responses.
- Target margin $\gamma$ prevents over-optimization near the decision boundary.

**Typical hyperparameters:** $\beta = 2.0$–$2.5$, $\gamma = 0.5$–$1.5$.

---

## KTO (Kahneman-Tversky Optimization)

[ethayarajh2024kto] Aligns from **unpaired** binary feedback — individual responses labeled as "good" or "bad", without requiring paired comparisons.

**Loss (inspired by prospect theory):**

$$
\mathcal{L}_\text{KTO}(\theta) = \mathbb{E}_x\!\left[\lambda_w \sigma\!\left(\beta \hat{r}(x, y_w) - z_\text{ref}\right) + \lambda_l \sigma\!\left(-\beta \hat{r}(x, y_l) - z_\text{ref}\right)\right]
$$

where $\hat{r}(x, y) = \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x)$ and $z_\text{ref}$ is the KL reference term.

**Key insight:** Humans are loss-averse — bad experiences carry more weight than equivalent good ones. KTO models this asymmetry with separate $\lambda_w$ and $\lambda_l$ weights.

**When to use:** When you have abundant binary feedback but cannot construct preference pairs (e.g., thumbs up/down from user logs).

---

## GRPO (Group Relative Policy Optimization)

[shao2024deepseekmath] Replaces the value network in PPO with group-normalized advantages, eliminating the critic model entirely.

**Algorithm:**
1. For each prompt $x$, sample a group of $G$ responses: $\{y_1, \ldots, y_G\}$.
2. Score each response with a reward function $r_i = r(x, y_i)$.
3. Compute group-normalized advantage:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}
$$

4. Optimize with a clipped PPO-style objective:

$$
\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \min\!\left(\rho_i \hat{A}_i,\; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta \mathbb{D}_\text{KL}[\pi_\theta \| \pi_\text{ref}]\right]
$$

**Advantages:**
- No critic model — 3 models instead of 4 (policy, reference, reward).
- Group baseline naturally adapts to difficulty: easy problems have low variance, hard ones have high variance.
- The core algorithm behind DeepSeek-R1's reasoning training.

**Typical settings:** $G = 4$–$16$ samples per prompt, $\epsilon = 0.2$, $\beta = 0.001$–$0.01$.

---

## DAPO (Decoupled Alignment Policy Optimization)

[yu2025dapo] Addresses training instabilities in GRPO with three targeted fixes:

1. **Clip-higher trick:** Use asymmetric clipping $[\epsilon_l, \epsilon_h]$ with $\epsilon_h > \epsilon_l$ to allow larger positive updates (learning from correct responses) while still bounding negative updates.
2. **Token-level normalization:** Normalize loss by total number of tokens rather than number of responses, preventing short responses from dominating gradients.
3. **Entropy bonus:** Add $\eta H(\pi_\theta)$ to the objective to maintain sufficient exploration and prevent premature policy collapse.

**Used in:** ByteDance's open reasoning model training, Qwen3 alignment pipeline.

---

## RLVR (Reinforcement Learning with Verifiable Rewards)

A training paradigm (not a specific algorithm) for reasoning tasks where correctness can be automatically verified:

**Reward function:**

$$
r(y, y^*) = \begin{cases} 1 & \text{if } y \text{ is correct (verifiable)} \\ 0 & \text{otherwise} \end{cases}
$$

**Verification methods:**
- **Math:** Symbolic evaluation via SymPy or Mathematica.
- **Code:** Unit test execution.
- **Logic:** Formal verification or answer matching.

**Why it works:** Eliminates reward hacking — the model cannot game a neural reward model because verification is deterministic. Binary signal is noisy but unbiased.

**Used in:** DeepSeek-R1 (math/code RL stage), OpenAI o1/o3 training (claimed), QwQ, Kimi k1.5.

**Process reward vs outcome reward:**
- Outcome reward (ORM): single reward at end of response.
- Process reward (PRM): reward at each reasoning step. Requires human annotation of step-level correctness. Higher signal but expensive to label.

---

## Constitutional AI (CAI)

[bai2022constitutional] Two-phase approach from Anthropic that uses the model itself to generate and apply alignment feedback.

**Phase 1 — SL-CAI (Supervised Learning):**
1. Generate harmful/unhelpful responses from the current model.
2. Ask the model to critique its own response against a written **constitution** (set of principles).
3. Ask the model to revise the response given the critique.
4. Fine-tune on the revised (self-improved) responses.

**Phase 2 — RL-CAI:**
1. Generate preference pairs: model response vs. AI-generated revision.
2. Use the model itself (not humans) to label which response better satisfies the constitution.
3. Train a preference model on AI-labeled pairs.
4. Apply standard RLHF/PPO with the AI-preference model as reward.

**Advantage:** Reduces reliance on human labeling of harmful content — the constitution encodes principles, not individual examples. Scales to diverse principles without proportionally increasing annotation cost.

---

## Comparison

| Method | Models Needed | Data Required | Primary Use |
|---|---|---|---|
| PPO | 4 (policy, ref, RM, critic) | Paired preferences | Classic RLHF |
| DPO | 2 (policy, ref) | Paired preferences | Offline alignment |
| SimPO | 1 (policy only) | Paired preferences | Memory-efficient alignment |
| KTO | 2 (policy, ref) | Binary labels (unpaired) | Large-scale implicit feedback |
| GRPO | 3 (policy, ref, verifier) | Prompts + reward fn | Reasoning/math RL |
| DAPO | 3 + entropy term | Prompts + reward fn | Stable long-horizon RL |
| RLVR | 2 (policy, ref) | Verifiable answers | Math, code tasks |
| CAI | 1 (+ constitution) | Principles text | Scalable safety alignment |

**$\beta$ tuning guidance:**
- Too high: policy stays close to reference → limited improvement.
- Too low: policy drifts → reward hacking, incoherence.
- DPO: $\beta = 0.1$–$0.3$. GRPO: $\beta = 0.001$–$0.01$ (much smaller, different scale).
