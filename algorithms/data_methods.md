# Data Methods

Techniques for generating, curating, and scheduling training data at scale.

---

## Synthetic Data Generation

### Self-Instruct

[wang2022self] Bootstraps instruction-following data using the model itself:

1. Start with a small seed set of 175 human-written instruction-response pairs.
2. Use the model to generate new instructions given the seed examples as context.
3. Filter: remove near-duplicates (ROUGE-L > 0.7), classify as classification vs. generation.
4. Use the model to generate responses for the new instructions.
5. Add accepted pairs back to the seed pool; repeat.

**Scale achieved:** 52K instruction-following pairs from GPT-3 at low cost. Basis for Alpaca.

### Evol-Instruct

[xu2023wizardlm] Evolves instructions to be progressively more complex and diverse:

**Evolution operations applied iteratively:**
- **In-depth:** Add constraints, deepen reasoning requirements, increase steps.
- **In-breadth:** Generate a new instruction on a different topic using the current as inspiration.
- **Concretizing:** Replace vague terms with specific constraints.
- **Reasoning:** Add multi-step reasoning requirements.

**Filter step:** Execute evolved instructions and discard those where the response quality degrades (detected by a judge model or quality classifier).

**Result:** Higher-difficulty training data that produces models with stronger instruction-following on hard tasks.

### Magpie

[xu2024magpie] Exploits the model's system prompt behavior to generate aligned instruction data without any seed instructions:

1. Feed only the system prompt + human turn prefix to the model.
2. The model auto-completes the user turn (generates an instruction).
3. Then generates the assistant response.

**Key insight:** Instruction-tuned models have learned to "expect" a user message after the system prompt. By stopping at the right token, you harvest what the model believes a plausible user would ask.

**Scale:** Generates millions of diverse, high-quality instruction pairs per day at near-zero cost.

### Orca / Orca 2

[mukherjee2023orca] Trains small models on **explanation traces** from GPT-4:

1. Sample diverse instructions from FLAN or other datasets.
2. Query GPT-4 with system prompts that elicit step-by-step reasoning, explanations, and self-reflection.
3. Fine-tune the small model on the GPT-4 traces (not just final answers).

**Orca 2:** Teaches the model to choose the appropriate reasoning strategy per task type (step-by-step, recall, guess-and-check) — rather than always defaulting to chain-of-thought.

### Persona Hub

[ge2024persona] Generates diverse synthetic data by conditioning generation on synthetic user personas:

1. Sample 1B+ personas from web text (each is a short description of a person's background, expertise, and interests).
2. For each target task, generate instructions conditioned on a randomly sampled persona.
3. Diversity is achieved through persona variation rather than instruction augmentation.

**Use case:** Creating diverse alignment data that represents the full distribution of user types rather than "typical" user queries.

---

## Data Curation

### MinHash Deduplication

Identifies near-duplicate documents at scale using Locality-Sensitive Hashing (LSH):

1. Tokenize documents into $n$-grams (typically 5-grams).
2. Compute MinHash signatures: for $k$ hash functions $h_1, \ldots, h_k$, the signature is $(\min_{g \in D} h_i(g))_{i=1}^k$.
3. Apply LSH: band documents by signature subsets; documents in the same band are candidate pairs.
4. Compute exact Jaccard similarity for candidates:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} \approx \frac{1}{k}\sum_{i=1}^k \mathbb{1}[\min_g h_i(g \in A) = \min_g h_i(g \in B)]
$$

5. Remove documents above a Jaccard threshold (typically 0.7–0.9).

**Scale:** Applied to trillion-token corpora. MinHash is the standard first-pass deduplication step (used in RedPajama, Dolma, DCLM).

### Semantic Deduplication (SemDedup)

[abbas2023semdedup] Deduplication at the semantic level using embedding similarity:

1. Embed all documents with a small encoder (e.g., SentenceTransformers).
2. Cluster embeddings; within each cluster, compute pairwise cosine similarities.
3. Remove one document from each pair with similarity above threshold $\epsilon$.

**Advantage over MinHash:** Catches near-paraphrases and reformulations that MinHash misses. Particularly important for web-scraped data with heavy copy-paste reuse.

### Quality Filtering

**Heuristic filters (fast):**
- Language detection (remove non-target language documents).
- Perplexity filtering: score documents with a small language model; remove outliers.
- Length filters, character-level noise filters (high symbol ratio, repeated characters).
- URL-based blacklists (spam, adult content).

**Classifier-based filters (expensive but effective):**
- Train a fastText or small BERT classifier on high-quality data (Wikipedia, books, curated web) vs. low-quality data.
- Threshold on classifier score.
- Used in GPT-3's WebText2 filter, LLaMA's C4 filter.

**GneissWeb approach [ibrahim2024gneissweb]:** Combines multiple quality signals into a joint score rather than hard thresholds; retains more data while filtering low-quality content proportionally.

---

## Curriculum Learning

Schedule training data from easy to hard over the course of training.

**Basic curriculum:**
1. Train on high-quality, diverse data for the majority of training.
2. In the final 5–10% of steps, upweight domain-specific high-quality data (books, math, code).

**Evidence for late-stage domain emphasis:** LLaMA 2, Qwen 2, and Mistral all use a "cool-down" phase with domain-specific data upweighting.

**Difficulty-based ordering:**
- Define difficulty via perplexity under a smaller reference model.
- Train on low-perplexity (easy/familiar) examples first; introduce high-perplexity examples later.
- Empirical results are mixed — benefits appear primarily at small scale or data-limited settings.

---

## Data Mixture and Weighting

**Domain weights:** The fraction of each domain in the training mixture significantly affects capabilities. Typical recipes:

| Domain | Pre-training Weight |
|---|---|
| Web text (filtered) | 70–80% |
| Books | 5–15% |
| Code | 5–15% |
| Math / papers | 2–8% |
| Wikipedia | 2–5% |

**Data Mixing Laws [ye2024data]:** Empirical scaling laws for estimating optimal domain weights. Train small proxy models at varied domain weights; extrapolate to the target compute budget.

**Importance sampling:** Down-weight documents that are similar to many others (over-represented topics) and up-weight rare, high-quality documents. Applied in DCLM and Dolma v2.

**DoReMi (Domain Reweighting with Mini-models):**
Automates domain weight discovery by training a small "reference" model on a baseline mixture, then adjusting weights for the main run based on where the reference model's loss exceeds theoretical bounds (optimizing the ratio of group-specific losses).


---

## Data Decontamination

Remove benchmark test sets from training data to prevent evaluation contamination:

1. Extract canonical $n$-grams from benchmark questions and answers (typically 13-grams).
2. Search training corpus for exact or near-exact matches.
3. Remove contaminated documents.

**Fuzzy decontamination:** Use MinHash or BM25 to catch paraphrases of benchmark questions — more conservative but reduces false positives.

**Best practice:** Report decontamination methodology alongside benchmark results. Run post-hoc contamination analysis if benchmarks weren't available at data collection time.
