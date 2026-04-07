# Training Production-Ready Large Language Models

## A Comprehensive Guide to Modern Techniques (2025–2026)

*From Data Curation to Deployment*
*Pre-training • Fine-tuning • Alignment • Inference Optimization*

---

April 2026

---

## Table of Contents

- [Part I: Foundations](#part-i-foundations)
  - [Chapter 1: The LLM Training Landscape in 2026](#chapter-1-the-llm-training-landscape-in-2026)
  - [Chapter 2: Transformer Architecture – The Foundation](#chapter-2-transformer-architecture--the-foundation)
- [Part II: Data – The Foundation of Everything](#part-ii-data--the-foundation-of-everything)
  - [Chapter 3: Pre-training Data Curation](#chapter-3-pre-training-data-curation)
  - [Chapter 4: Tokenization](#chapter-4-tokenization)
- [Part III: Pre-training](#part-iii-pre-training)
  - [Chapter 5: Pre-training Objectives and Strategies](#chapter-5-pre-training-objectives-and-strategies)
  - [Chapter 6: Distributed Training and Infrastructure](#chapter-6-distributed-training-and-infrastructure)
- [Part IV: Post-Training – From Base Model to Product](#part-iv-post-training--from-base-model-to-product)
  - [Chapter 7: Supervised Fine-Tuning (SFT)](#chapter-7-supervised-fine-tuning-sft)
  - [Chapter 8: Alignment – RLHF, DPO, and Beyond](#chapter-8-alignment--rlhf-dpo-and-beyond)
  - [Chapter 9: Training for Reasoning](#chapter-9-training-for-reasoning)
- [Part V: Evaluation, Safety & Deployment](#part-v-evaluation-safety--deployment)
  - [Chapter 10: Evaluation and Benchmarking](#chapter-10-evaluation-and-benchmarking)
  - [Chapter 11: Safety, Alignment, and Red-Teaming](#chapter-11-safety-alignment-and-red-teaming)
  - [Chapter 12: Inference Optimization and Deployment](#chapter-12-inference-optimization-and-deployment)
- [Part VI: Advanced Topics and Future Directions](#part-vi-advanced-topics-and-future-directions)
  - [Chapter 13: Domain-Specific and Multimodal Models](#chapter-13-domain-specific-and-multimodal-models)
  - [Chapter 14: Catastrophic Forgetting and Continual Learning](#chapter-14-catastrophic-forgetting-and-continual-learning)
  - [Chapter 15: The Future of LLM Training](#chapter-15-the-future-of-llm-training)
  - [Chapter 16: How Top Companies Train Their Models](#chapter-16-how-top-companies-train-their-models)
- [Appendices](#appendices)
  - [Appendix A: Glossary of Key Terms](#appendix-a-glossary-of-key-terms)
  - [Appendix B: Recommended Reading](#appendix-b-recommended-reading)
  - [Appendix C: Hardware Reference](#appendix-c-hardware-reference)

---

# Part I: Foundations

---

## Chapter 1: The LLM Training Landscape in 2026

### 1.1 The Evolution of Language Model Training

Large language models have undergone a dramatic transformation over the past several years. What began as research curiosities have become the computational backbone of search engines, coding assistants, data analysis platforms, and creative tools. The global market for LLMs was valued at $6.4 billion in 2024 and is projected to reach $36.1 billion by 2030. This growth reflects a fundamental shift: LLMs are no longer experimental—they are production infrastructure.

The training paradigm has also shifted substantially. Early models relied on a simple recipe: more data plus more parameters equals better performance. Scaling laws reinforced this trend, but by late 2025, several hard constraints became apparent. Marginal gains from more compute diminish as we double FLOPs. High-quality, deduplicated text is finite and increasingly expensive to obtain. And ever-larger models collide with real-time product requirements and energy budgets.

The modern approach recognizes that raw pre-training scale is only the beginning. Post-training—the combination of supervised fine-tuning, preference optimization, and reinforcement learning—now accounts for the majority of a model's usable capability. The field has moved from "make the base model bigger" to "given a strong base model, how do we make it safe, efficient, and excellent at specific jobs?"

### 1.2 The Modern Training Pipeline at a Glance

A production-ready LLM follows a multi-stage pipeline, each stage solving a different problem. The order matters:

1. **Stage 1 – Data Curation & Pre-training:** The model learns general language patterns, syntax, semantics, and factual knowledge through next-token prediction on trillions of tokens drawn from the web, books, code, and curated sources.
2. **Stage 2 – Supervised Fine-Tuning (SFT):** The model learns to follow instructions by training on high-quality instruction-response pairs. This establishes baseline behavior for helpfulness, format compliance, and task execution.
3. **Stage 3 – Preference Optimization:** The model learns which responses humans prefer over alternatives using techniques like DPO, SimPO, or KTO. This aligns the model with human values and conversational norms.
4. **Stage 4 – Reinforcement Learning:** Using verifiable rewards (math, code) or environment-based feedback (tool use, multi-step tasks), the model discovers new strategies through trial and error. This stage produces reasoning capabilities.
5. **Stage 5 – Evaluation, Safety & Deployment:** Rigorous benchmarking, red-teaming, quantization, and inference optimization prepare the model for production traffic.

### 1.3 Cost Realities

Training production LLMs demands enormous resources. OpenAI reportedly spent over $100 million developing ChatGPT, while DeepSeek demonstrated that comparable results can be achieved for roughly $5.5 million with the right architectural choices and training strategies. For smaller, domain-specific models, costs can range from tens of thousands to low millions of dollars.

| Component | Typical Cost Range | Key Driver |
|---|---|---|
| Pre-training (frontier model) | $5M – $100M+ | Compute (GPU hours) |
| Pre-training (7B–70B) | $50K – $5M | Data quality + GPU hours |
| Fine-tuning (LoRA/QLoRA) | $100 – $10K | Dataset size, GPU type |
| RLHF / DPO alignment | $10K – $500K | Human annotation costs |
| Evaluation & Red-teaming | $5K – $100K | Evaluator complexity |
| Inference infrastructure | $1K – $50K/month | Traffic volume, latency SLA |

---

## Chapter 2: Transformer Architecture – The Foundation

### 2.1 The Decoder-Only Transformer

Nearly all modern autoregressive LLMs use a decoder-only transformer architecture. This architecture is composed of sequential transformer blocks, each containing masked self-attention and feed-forward sub-layers. The self-attention mechanism enables the model to weigh the relevance of different parts of the input when generating each token, while the feed-forward network (FFN) transforms the contextual information to capture more complex relationships.

The FFN is the computational workhorse of the transformer. In large models like PaLM-540B, approximately 90% of parameters reside in the feed-forward layers. This is why architectural innovations like Mixture of Experts focus specifically on making the FFN more efficient.

### 2.2 Key Architectural Components

- **Multi-Head Attention:** Divides the attention computation across multiple heads, each learning different aspects of token relationships. Modern models use techniques like Grouped Query Attention (GQA) and Multi-Query Attention (MQA) to reduce the memory footprint of the key-value cache during inference.
- **Positional Encoding:** Since transformers have no inherent notion of sequence order, positional information must be injected. Rotary Position Embeddings (RoPE) have become the standard in 2025–2026, enabling efficient extension to longer contexts. ALiBi (Attention with Linear Biases) offers an alternative that avoids explicit position embeddings entirely.
- **Normalization:** RMSNorm has largely replaced LayerNorm in modern architectures due to its computational efficiency and equivalent performance. Pre-normalization (applying norm before attention/FFN rather than after) has become the dominant pattern.
- **Activation Functions:** SwiGLU has emerged as the preferred activation function for FFN layers, offering better performance than ReLU or GELU at similar computational cost.

### 2.3 Mixture of Experts (MoE)

Mixture of Experts has become one of the most important architectural innovations in production LLMs. Models like DeepSeek-V2/R1, Mixtral, Grok-1, and many closed-source systems (GPT-4 is rumored to use a sparse architecture) leverage MoE to scale model capacity without proportional increases in compute cost.

The core idea is elegant: replace the dense feed-forward network in each transformer block with multiple smaller expert networks and a router that selects which experts process each token. Only a subset of experts (typically 1–2 out of 8–64 total) are activated per token, creating a sparse computation pattern.

**How MoE works:** For each token, a gating network (router) produces a probability distribution over all experts. The top-k experts are selected, process the token independently, and their outputs are combined as a weighted sum. Mixtral 8x7B, for instance, has 8 experts per layer but activates only 2 per token, yielding a model with 47B total parameters but only ~13B active parameters per forward pass.

**Key MoE developments in 2025–2026:**

- **DeepSeekMoE:** Introduced finer-grained experts and shared expert isolation, achieving better specialization per expert.
- **Sparse Upcycling:** A technique where a pre-trained dense transformer is converted to MoE by duplicating its FFN layers with randomly initialized routing, then training further—avoiding the need to train MoE from scratch.
- **SwitchHead:** Applies MoE not only to FFN layers but also to attention projection layers (Q, K, V matrices), extending the sparsity concept across the entire transformer block.
- **Expert Parallelism:** Distributing experts across different GPUs, combined with techniques like expert buffering, enables efficient training and inference of MoE models at scale.

### 2.4 Context Length and Efficient Attention

The standard self-attention mechanism scales quadratically with sequence length, making long-context training prohibitively expensive. Several techniques address this:

- **Flash Attention (v2/v3):** Hardware-aware attention implementations that reduce memory usage from O(n²) to O(n) by tiling the computation and avoiding materialization of the full attention matrix. Flash Attention has become essentially mandatory for modern LLM training.
- **Ring Attention:** Distributes long sequences across multiple devices in a ring topology, enabling context lengths of millions of tokens during training.
- **Sliding Window Attention:** Used in Mistral and similar models, limits attention to a fixed local window while still allowing information flow across layers, significantly reducing compute for long sequences.

---

# Part II: Data – The Foundation of Everything

---

## Chapter 3: Pre-training Data Curation

### 3.1 Why Data Quality Trumps Quantity

The single most impactful factor in LLM training is data quality. Research has conclusively shown that a smaller, well-curated dataset consistently outperforms a larger, noisy one. The FineWeb dataset (15 trillion tokens, heavily curated) produces better models than RedPajama (20 trillion tokens, lightly filtered)—an extra 5 trillion tokens of noisy data actually hurt performance. This finding has reshaped how the industry approaches data curation.

Modern frontier models train on 10–20 trillion tokens, approaching the total pool of high-quality text available publicly. This ceiling has created intense interest in synthetic data generation and sophisticated curation techniques.

### 3.2 Data Sources

- **Web Crawl Data:** Common Crawl and curated derivatives like FineWeb, FineWeb-Edu, Ultra-FineWeb, and DCLM form the backbone of most pre-training corpora. Raw web data requires extensive filtering and deduplication.
- **Books and Academic Literature:** High-quality long-form text that provides coherent reasoning and domain knowledge. Ethical and legal concerns around copyrighted material have led to increased reliance on openly licensed sources.
- **Code Repositories:** GitHub, GitLab, and other code hosting platforms provide programming language data. Specialized datasets like The Stack offer permissively licensed code.
- **Curated Knowledge Bases:** Wikipedia, encyclopedias, and verified reference materials provide factual anchors for the model's knowledge.
- **Multilingual Sources:** FineWeb-2 and similar initiatives provide filtered, high-quality data across dozens of languages, essential for globally capable models.

### 3.3 Data Cleaning and Filtering Pipeline

Modern data curation follows a multi-level processing hierarchy, progressing from basic filtering to sophisticated model-driven refinement:

**Level 0–3: Rule-Based Filtering**

- URL and domain-level filtering (remove known low-quality domains)
- Language identification and script detection
- Boilerplate removal (navigation bars, ads, cookie notices)
- Heuristic quality filters (document length, character ratios, perplexity thresholds)

**Level 4–5: Model-Based Curation**

- Classifier-based quality scoring using FastText or small transformer models trained on high-quality examples
- GneissWeb (IBM, 2024) demonstrated sharded exact sub-string deduplication at 10 trillion token scale
- SoftDedup approaches avoid outright deletion, instead reweighting recurring data to preserve information while reducing redundancy

**Level 6+: LLM-Driven Refinement (2025–2026 frontier)**

- DataEvolve / Data Darwinism: uses LLMs to autonomously analyze data quality issues, generate cleaning strategies, and iteratively refine them through automated evaluation loops
- Category-specific strategies (SwallowCode for code, MegaMath for mathematics, MegaScience for scientific literature) that tailor cleaning approaches to domain characteristics
- Instruction-response augmented pre-training: enriching training corpora with synthetically generated Q&A pairs to improve downstream task performance

### 3.4 Deduplication

Deduplication is critical for preventing the model from over-memorizing patterns and for training efficiency. Modern approaches operate at multiple levels:

- **Exact deduplication:** Hash-based detection of identical documents or passages.
- **Near-duplicate detection:** MinHash / Locality-Sensitive Hashing (LSH) identifies semantically similar documents. GPU-accelerated connected components algorithms (as in NVIDIA NeMo Curator) handle the graph resolution at scale.
- **Fuzzy deduplication:** Embedding-based similarity detection catches paraphrased content.

After LSH bucketing, documents within the same bucket are connected via edges, and a connected components algorithm identifies all clusters of near-duplicates for removal or down-weighting.

### 3.5 Multi-Stage Training Strategy

Many state-of-the-art models employ a two-stage (or multi-stage) pre-training approach:

- **Stage 1:** Train on a very large corpus (10T+ tokens) to maximize coverage of knowledge and linguistic diversity. Some lower-quality material is acceptable here.
- **Stage 2:** Continue training on a smaller, meticulously curated dataset (0.5–1T tokens) of verified content—books, Wikipedia, curated code, and validated factual sources—to polish the model's capabilities.

This approach yields models that have both breadth (from Stage 1) and depth (from Stage 2), with the final weights focused on the most reliable information.

---

## Chapter 4: Tokenization

### 4.1 Subword Tokenization

LLMs do not process text as words or characters—they work with tokens, which are subword units mapped to numerical embeddings. Tokenization determines the granularity at which the model perceives language and has far-reaching implications for model performance, multilingual capability, and inference efficiency.

- **Byte-Pair Encoding (BPE):** The dominant tokenization algorithm, used by GPT-series and most modern LLMs. BPE iteratively merges the most frequent adjacent byte or character pairs to build a vocabulary of subword units, typically 32K–128K tokens.
- **WordPiece:** Similar to BPE but uses a likelihood-based criterion for merges. Used in BERT and some Google models.
- **SentencePiece:** A language-agnostic tokenizer that operates directly on raw text (including whitespace) without pre-tokenization steps. Widely used in multilingual models like LLaMA.

### 4.2 Universal and Multilingual Tokenizers

A major advancement in 2025 is the development of universal tokenizers trained on more languages than the primary pre-training languages. Research shows that models with universal tokenizers achieve up to 20.2% higher win rates in language adaptation tasks compared to language-group-specific tokenizers, while also improving adaptation to completely unseen languages by up to 5%. This "language plasticity" approach enables more efficient post-training expansion to new languages.

### 4.3 Sequence Packing

Since training operates on fixed-length sequences, documents must be assembled into training samples. The naive approach—concatenating all documents and splitting at sequence boundaries—creates many truncations that break document coherence. Recent research on length-aware combinatorial optimization proposes a data packing strategy that eliminates unnecessary truncations while maintaining training efficiency. Experiments show a 40% reduction in truncations, leading to improved language modeling performance.

---

# Part III: Pre-training

---

## Chapter 5: Pre-training Objectives and Strategies

### 5.1 Next-Token Prediction

The standard pre-training objective for autoregressive LLMs is next-token prediction (causal language modeling): given a sequence of tokens, predict the next token. This self-supervised task requires no labeled data—the training signal comes from the text itself. Despite its simplicity, this objective produces remarkably powerful representations that capture syntax, semantics, factual knowledge, and even reasoning patterns.

### 5.2 Reinforcement Pre-Training (RPT)

Introduced by Microsoft researchers in 2025, Reinforcement Pre-Training reframes next-token prediction as a sequential decision-making problem. Under RPT, the language model functions as an RL agent, receiving explicit reward-based feedback on its predictions. When predictions match actual tokens, the model receives positive rewards. This approach remains entirely self-supervised—it uses unlabeled text as its reward signal—but provides richer learning gradients than maximum likelihood training alone.

### 5.3 Curriculum Learning

Rather than presenting training data in random order, curriculum learning strategies organize data by difficulty or domain. Research on pacing-based sampling and interleaved curricula shows that curriculum learning can both accelerate convergence and boost final model performance. The intuition is straightforward: starting with cleaner, more structured text and gradually introducing more complex or noisy data helps the model build a solid foundation before tackling harder material.

### 5.4 Instruction-Response Augmented Pre-training

A more recent approach enriches the pre-training corpus with instruction-response pairs, often generated synthetically using stronger models. These pairs simulate task demonstrations (Q&A, summarization, classification) alongside the raw text. While still relatively uncommon, this technique bridges the gap between pre-training and supervised fine-tuning, resulting in models that are already partially instruction-tuned before the SFT stage begins.

---

## Chapter 6: Distributed Training and Infrastructure

### 6.1 Parallelism Strategies

Training models with billions of parameters requires distributing computation across hundreds or thousands of GPUs. Multiple parallelism strategies are combined:

- **Data Parallelism:** Replicates the full model on each GPU and splits the training data across replicas. Gradients are synchronized via all-reduce operations. Fully Sharded Data Parallelism (FSDP) from PyTorch and DeepSpeed ZeRO (Stages 1–3) shard optimizer states, gradients, and even parameters across devices to reduce per-GPU memory.
- **Tensor Parallelism:** Splits individual layers (attention heads, FFN weight matrices) across GPUs. Essential for models too large to fit on a single GPU. Megatron-LM is the canonical implementation.
- **Pipeline Parallelism:** Distributes different layers of the model across different GPUs, with micro-batching to keep the pipeline full. Reduces per-GPU memory at the cost of "bubble" time.
- **Expert Parallelism:** Specific to MoE models, distributes different experts across GPUs. Combined with tensor and data parallelism for the shared layers.
- **Sequence Parallelism:** Distributes the sequence dimension across devices, complementing tensor parallelism. Ring Attention enables this for the attention computation itself.

### 6.2 Mixed-Precision Training

Modern LLM training universally uses mixed-precision arithmetic:

- **BF16 (Brain Float 16):** The dominant training format in 2025–2026. BF16 maintains the same exponent range as FP32 (avoiding overflow/underflow) while halving memory. Preferred on NVIDIA A100/H100/H200 and Google TPUs.
- **FP8 Training:** Emerging as the next frontier, with NVIDIA H100/H200 GPUs providing native FP8 support. FP8 further halves memory and compute requirements, though careful loss scaling is required.

Master weights are typically maintained in FP32 for optimizer state, with forward/backward passes computed in BF16 or FP8.

### 6.3 Checkpointing and Fault Tolerance

At the scale of modern training runs (weeks to months on thousands of GPUs), hardware failures are inevitable. Robust checkpointing includes periodic model and optimizer state snapshots (every 1–4 hours), asynchronous checkpointing that does not block training, distributed checkpointing sharded across nodes, and automatic restart/recovery mechanisms.

### 6.4 Key Training Frameworks

| Framework | Primary Use Case | Key Features |
|---|---|---|
| Megatron-LM (NVIDIA) | Large-scale pre-training | Tensor/pipeline/sequence parallelism |
| DeepSpeed (Microsoft) | Efficient distributed training | ZeRO optimizer, pipeline parallelism |
| FSDP (PyTorch native) | Data-parallel training | Integrated with PyTorch ecosystem |
| NeMo (NVIDIA) | End-to-end LLM platform | Pre-training, fine-tuning, deployment |
| MosaicML / Composer | Efficient training recipes | Algorithmic optimizations, speed-ups |
| Axolotl | Fine-tuning workflows | LoRA, QLoRA, full fine-tuning |
| Unsloth | Memory-efficient fine-tuning | 2–5× faster with 80% less memory |
| TRL (Hugging Face) | Alignment training | PPO, DPO, GRPO trainers (v0.28+) |

---

# Part IV: Post-Training – From Base Model to Product

---

## Chapter 7: Supervised Fine-Tuning (SFT)

### 7.1 Purpose and Principles

Supervised fine-tuning is the first post-training stage, where the base model transitions from a generic text predictor to an instruction-following assistant. SFT trains the model on high-quality instruction-response pairs, establishing baseline behavior for helpfulness, format compliance, safety, and conversational tone.

SFT is stable, relatively inexpensive, and immediately improves instruction-following. However, SFT alone cannot resolve preference trade-offs (concise vs. thorough, safe vs. helpful) and struggles with long-tail failure modes because it only shows the model examples of good responses—never bad ones.

### 7.2 Data for SFT

The quality of SFT data is paramount. Typical sources include human-written instruction-response pairs covering diverse tasks, synthetically generated data from stronger models (distillation), domain-specific datasets for specialized applications, and conversation logs refined by human annotators. Datasets like no_robots, OpenAssistant, and ShareGPT have become standard references.

### 7.3 Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all model parameters and requires significant GPU memory. Parameter-efficient methods dramatically reduce this cost:

- **LoRA (Low-Rank Adaptation):** Adds small trainable low-rank matrices to frozen model weights. Typically adds less than 1% additional parameters while achieving near-full-fine-tuning quality. The dominant PEFT method in production.
- **QLoRA:** Combines LoRA with 4-bit quantization of the base model, enabling fine-tuning of 70B+ parameter models on a single consumer GPU. A breakthrough for accessibility.
- **DoRA (Weight-Decomposed Low-Rank Adaptation):** Decomposes weight updates into magnitude and direction components, achieving better performance than LoRA in many benchmarks.
- **Spectrum:** A newer method that selectively trains only the most informative layers, identified through signal-to-noise analysis of the weight matrices.

### 7.4 Tooling for SFT

The Hugging Face TRL (Transformer Reinforcement Learning) library, now at v0.28 as of early 2026, has become the most widely adopted framework for SFT and alignment. Combined with libraries like PEFT, Axolotl, Unsloth, and Torchtune, teams can fine-tune models with minimal boilerplate. For inference, frameworks like vLLM and TGI provide high-throughput, low-latency serving.

---

## Chapter 8: Alignment – RLHF, DPO, and Beyond

### 8.1 The Alignment Problem

A model that produces grammatically perfect responses may still be unhelpful, evasive, or unsafe. Bridging this gap—training models to behave in ways humans actually prefer—is the central problem of alignment. The field has evolved from a single dominant technique (RLHF) to a modular stack of complementary methods, each addressing different aspects of model behavior.

### 8.2 Reinforcement Learning from Human Feedback (RLHF)

RLHF was the technique that transformed GPT-3 into ChatGPT. OpenAI demonstrated that a 1.3B parameter model aligned with RLHF outperformed a 175B model with only SFT—proving that alignment matters more than raw size for usability.

The classic RLHF pipeline consists of three stages:

1. **Stage 1 – SFT:** Train the base model on instruction-response pairs to establish baseline behavior.
2. **Stage 2 – Reward Model Training:** Human annotators rank pairs of model-generated responses. These pairwise preferences train a reward model that predicts which response is better.
3. **Stage 3 – PPO Optimization:** The model generates responses, the reward model scores them, and Proximal Policy Optimization updates the model to maximize reward while staying close to the SFT baseline (via a KL-divergence penalty).

RLHF remains essential for tasks where there is no verifiable ground truth—tone, helpfulness, cultural sensitivity, and creative quality. However, RLHF is expensive (requiring four models in memory simultaneously: policy, reference, reward model, and critic), sensitive to hyperparameters, and prone to reward hacking.

### 8.3 Direct Preference Optimization (DPO)

DPO, proposed by Rafailov et al. in 2023, eliminates the reward model entirely. Instead of training a separate reward model and running an RL loop, DPO derives a reward signal implicitly from the LLM itself, treating alignment as a classification problem on preference pairs. The result: a single training run replaces the entire RLHF pipeline.

DPO can achieve approximately 90–95% of RLHF's alignment performance while requiring 40–60% less compute. The total codebase for a DPO implementation typically spans roughly 400 lines, compared to thousands for RLHF.

**Key DPO successors (2025–2026):**

- **SimPO:** Uses the average log probability of a response as an implicit reward, removing the reference model entirely. SimPO outperforms DPO by 6.4 points on AlpacaEval 2 and 7.5 points on Arena-Hard.
- **KTO (Kahneman-Tversky Optimization):** Works with simple thumbs-up/thumbs-down feedback instead of pairwise comparisons. Critical for production systems where binary feedback is cheap and abundant.
- **Iterative/Online DPO:** Periodically re-samples training data from the current policy, creating a semi-online training setup that captures the benefits of on-policy learning.

### 8.4 GRPO – Group Relative Policy Optimization

GRPO, introduced by DeepSeek and now the dominant algorithm for training reasoning models, eliminates the need for a separate critic (value function) model. For each prompt, the model generates a group of responses (typically 8–64). Rewards are computed for each response, then advantages are calculated by normalizing each response's reward against the group mean and standard deviation.

This approach reduces memory overhead by approximately 33–50% compared to PPO-based RLHF and is more stable in training because the reward signal varies less dramatically between updates. LLM development in 2025–2026 has been dominated by reasoning models using RLVR and GRPO.

### 8.5 DAPO – Decoupled Alignment and Policy Optimization

DAPO further removes the reference model dependency present in GRPO, simplifying the training setup. Along with GRPO, DAPO eliminates the need for critic models, reference models, and preference pair curation—representing the most streamlined RL approach to alignment available.

### 8.6 RLAIF – Reinforcement Learning from AI Feedback

RLAIF uses AI-generated preferences instead of human annotations to train reward models or preference pairs. Once a capable feedback model is available, thousands of comparisons can be generated in minutes, dramatically reducing annotation costs. Anthropic's Constitutional AI (CAI) was a pioneer in this approach—providing the model with a set of principles (a "constitution") and having it self-critique and revise its own outputs before using these revised outputs for RL training.

### 8.7 RLVR – Reinforcement Learning with Verifiable Rewards

RLVR trains models on tasks where correctness can be automatically verified—math problems (check the answer), code (run unit tests), and other structured domains. The reward signal is binary and objective: the answer is correct or it is not.

DeepSeek R1 demonstrated that pure RLVR with GRPO, applied directly to a base model without any SFT, could produce emergent reasoning capabilities including self-verification and extended chain-of-thought reasoning. Because verifiable rewards are less prone to reward hacking, RLVR training can run much longer than traditional RLHF without collapsing.

### 8.8 The Modern Production Alignment Stack

The cutting-edge alignment pipeline in 2026 is modular:

| Stage | Technique | What It Solves |
|---|---|---|
| 1. Instruction Following | Supervised Fine-Tuning (SFT) | Format, tone, task execution |
| 2. Preference Alignment | DPO / SimPO / KTO | Human values, conversational norms |
| 3. Reasoning | GRPO / DAPO + RLVR | Math, code, multi-step planning |
| 4. Safety & Helpfulness | RLHF (with neural reward model) | Open-ended quality, harm avoidance |
| 5. Constitutional Guardrails | CAI / RLAIF | Scalable safety principles |

DeepSeek R1's final training stage exemplifies this modularity: it used both RLVR (for reasoning tasks) and traditional RLHF with separate neural reward models for helpfulness and harmlessness, evaluated on different parts of the response.

---

## Chapter 9: Training for Reasoning

### 9.1 The Rise of Reasoning Models

Reasoning LLMs—models designed to solve complex problems by breaking them down into a series of smaller steps—have emerged as the most significant development in 2025–2026. These models excel at challenging tasks like advanced programming, mathematical proofs, and multi-step planning by generating extended chains of thought before producing final answers.

### 9.2 Training Reasoning Through RL

The standard approach relies on reinforcement learning with verifiable rewards. The model generates multiple potential answers to a query, receives a reward for correct solutions, and is updated based on the best candidates. This rollout-based process can consume up to 85% of the execution time for RL training, while the actual parameter update consumes very little time by comparison.

### 9.3 TLT – Accelerating Reasoning Model Training

Researchers from MIT developed TLT (Train with a Little help from a Teacher), a method that automatically trains a smaller, faster "drafter" model to predict the outputs of the larger reasoning LLM. The larger model then only needs to verify these predictions rather than generating everything from scratch. When tested across multiple reasoning LLMs, TLT doubled training speed (70–210% acceleration) while preserving accuracy. As a bonus, the small drafter model can be reused for efficient deployment through speculative decoding.

### 9.4 Test-Time Compute and Inference-Time Reasoning

An emerging frontier is using additional compute at inference time to improve reasoning. Rather than relying solely on the model's trained weights, these approaches allocate more computation per query—generating multiple solution paths, verifying intermediate steps, and selecting the best final answer. This test-time compute paradigm allows models to solve harder problems without additional training, effectively trading inference cost for accuracy.

---

# Part V: Evaluation, Safety & Deployment

---

## Chapter 10: Evaluation and Benchmarking

### 10.1 The Challenge of Evaluating Generative Models

Evaluating generative models is fundamentally more complex than traditional ML metrics. It requires tracing the model's reasoning chains, assessing open-ended quality, and measuring alignment with human preferences across diverse tasks. No single metric captures all dimensions of model quality.

### 10.2 Standard Benchmarks

| Benchmark | What It Measures | Type |
|---|---|---|
| MMLU / MMLU-Pro | Broad academic knowledge (57+ subjects) | Multiple choice |
| HellaSwag | Commonsense reasoning / sentence completion | Multiple choice |
| HumanEval / MBPP | Code generation correctness | Code execution |
| GSM8K / MATH | Mathematical problem solving | Verified answers |
| TruthfulQA | Resistance to common misconceptions | Open-ended |
| MT-Bench | Multi-turn conversational ability | LLM-judged |
| AlpacaEval 2 | Instruction-following quality | LLM-judged |
| Arena-Hard | Head-to-head comparison quality | LLM-judged |
| GPQA | Graduate-level reasoning | Multiple choice |
| SWE-bench | Real-world software engineering | Code execution |

### 10.3 LLM-as-Judge

Using a strong LLM (like GPT-4 or Claude) to evaluate another model's outputs has become standard practice. Tools like W&B Weave enable developers to build systematic evaluation pipelines that score models on nuanced dimensions: tone, faithfulness, safety, helpfulness, and reasoning quality.

### 10.4 Human Evaluation

For production deployments, human evaluation remains the gold standard. Pairwise comparison protocols (where evaluators choose between two outputs) provide more reliable signal than absolute quality ratings. Chatbot Arena, maintained by LMSYS, provides a crowdsourced ELO-based ranking system that has become the most trusted public benchmark for conversational model quality.

---

## Chapter 11: Safety, Alignment, and Red-Teaming

### 11.1 Safety as a First-Class Concern

Production LLMs must be safe by design, not as an afterthought. This means building safety into every stage: filtering harmful content from pre-training data, including safety-oriented examples in SFT, training reward models that penalize harmful outputs, and conducting extensive red-teaming before deployment.

### 11.2 Constitutional AI

Anthropic's Constitutional AI approach provides the model with explicit principles (a "constitution") that govern its behavior. The model generates responses, critiques them against these principles, and revises them—creating self-improvement data for further RL training. This scales safety alignment without proportional increases in human annotation.

### 11.3 Red-Teaming

Red-teaming involves systematically probing the model for failure modes, harmful outputs, and vulnerabilities. Modern red-teaming combines automated approaches (using other LLMs to generate adversarial prompts) with human experts who attempt creative attacks. Key areas include jailbreak resistance, factual accuracy under pressure, bias detection, and harmful content generation prevention.

### 11.4 Explainability

Tools like SHAP, LIME, and attention visualization help users and regulators understand how LLMs make decisions. This is critical in regulated industries like healthcare and finance. The EU AI Act, which took effect in August 2024 with full rollout planned through 2026, requires varying levels of transparency depending on the risk category of the AI system.

---

## Chapter 12: Inference Optimization and Deployment

### 12.1 Quantization

Quantization reduces model precision from 16-bit or 32-bit floating point to lower bit-widths, dramatically reducing memory requirements and improving inference speed:

- **INT8 Quantization:** Reduces model size by ~50% with minimal quality degradation. Widely supported by inference frameworks.
- **INT4 / GPTQ / AWQ:** Further reduces model size by ~75%. AWQ (Activation-Aware Weight Quantization) and GPTQ are the most popular methods, enabling 70B+ parameter models to run on consumer hardware.
- **1-bit and 1.58-bit Quantization:** Frontier research showing near-FP16 accuracy with extreme compression. Limited library support as of 2026 but indicates the direction of future optimization.

### 12.2 Inference Frameworks

- **vLLM:** High-throughput serving with PagedAttention for efficient KV-cache management. The most popular open-source inference engine for production.
- **TGI (Text Generation Inference):** Hugging Face's production inference server with built-in continuous batching, tensor parallelism, and quantization.
- **TensorRT-LLM:** NVIDIA's optimized inference library providing the highest performance on NVIDIA hardware, with FP8 and INT4 support.
- **llama.cpp / Llamafile:** Enables LLM inference on consumer hardware including CPUs and Apple Silicon, making models accessible for local and edge deployment.

### 12.3 Speculative Decoding

Speculative decoding uses a small, fast "draft" model to generate candidate tokens that the larger model verifies in parallel. Since verification is cheaper than generation, this can provide 2–3× speedups with no quality loss. This technique pairs naturally with TLT, where the drafter model trained during RL can be reused for inference.

### 12.4 Serving Architecture

Production LLM deployments require careful design: continuous batching to maximize GPU utilization, KV-cache management (PagedAttention), load balancing across replicas, request routing and priority queuing, monitoring and observability, model versioning and A/B testing infrastructure for safe rollouts.

---

# Part VI: Advanced Topics and Future Directions

---

## Chapter 13: Domain-Specific and Multimodal Models

### 13.1 Domain-Specific LLMs

The trend in 2026 is moving away from one-size-fits-all models toward models trained for specific fields. BloombergGPT for finance, Med-PaLM for medical applications, ChatLAW for legal work—these models achieve higher accuracy and reduced error rates by leveraging deeper domain understanding. The key decisions involve whether to pre-train from scratch, continue pre-training on domain data, or fine-tune with domain-specific SFT data.

### 13.2 Multimodal Capabilities

Modern LLMs are no longer limited to text. Multimodal models handle text, images, audio, and video. Vision-language MoE variants (like Qwen3-VL with 30B-A3B and 235B-A22B configurations) use MoE as a practical mechanism for compute allocation across modalities. This enables applications like medical image analysis, video understanding, and cross-modal search.

### 13.3 Agentic AI

LLM-powered autonomous agents that can make decisions, interact with tools, and take actions without ongoing human input represent a major frontier. Training agentic models requires specialized fine-tuning data (tool use demonstrations, multi-step planning traces) and evaluation frameworks that test end-to-end task completion.

---

## Chapter 14: Catastrophic Forgetting and Continual Learning

### 14.1 The Forgetting Problem

Naive fine-tuning can cause catastrophic forgetting—adding new knowledge erases old capabilities. This is particularly problematic for production systems that need updates while retaining existing competence.

### 14.2 Mitigation Strategies

- **Elastic Weight Consolidation:** Penalizes changes to weights important for previously learned tasks.
- **Hierarchical Optimization:** Adds new skills in structured layers to reduce interference.
- **Replay-Based Methods:** Mix samples from previous training stages into new training data.
- **LoRA Merging:** Train separate LoRA adapters for different capabilities and merge them, avoiding direct modification of the base model.

---

## Chapter 15: The Future of LLM Training

### 15.1 Key Trends

- **Post-training Dominance:** The leverage has shifted from pre-training scale to post-training sophistication. The most impactful improvements now come from better alignment, not bigger models.
- **Reasoning as Core Capability:** RLVR with GRPO has made reasoning a trainable skill, not just an emergent property of scale.
- **Synthetic Data at Scale:** As natural high-quality text becomes scarce, synthetically generated training data is becoming essential.
- **Hybrid Alignment Stacks:** Production models combine DPO for preference alignment, GRPO for reasoning, and RLHF for open-ended quality.
- **Efficiency Breakthroughs:** From MoE architectures to FP8 training to speculative decoding, the focus is on more capability with less compute.
- **Open-Source Convergence:** Open models like LLaMA, Mistral, DeepSeek, and Qwen are converging with closed models in quality.

### 15.2 Open Challenges

- Data scarcity: approaching the limits of available high-quality text
- Evaluation gaps: no reliable automated metric for open-ended quality
- Safety at scale: ensuring aligned behavior across all possible inputs
- Interpretability: understanding why models produce specific outputs
- Energy and environmental costs of training and inference
- Legal and ethical frameworks for training data usage

---

## Chapter 16: How Top Companies Train Their Models

Understanding the training approaches of frontier AI labs reveals both shared patterns and distinctive innovations. This chapter examines the pipelines used by six leading organizations, drawing on publicly available technical reports, system cards, and research publications as of early 2026.

### 16.1 OpenAI – GPT-5 Family

**Architecture:** Dense transformer (with rumored sparse/MoE elements in some variants), trained on Microsoft Azure AI supercomputers. GPT-5 unified general intelligence, reasoning depth, coding specialization, and multimodality under a single model line.

**Pre-training:** Massive-scale unsupervised learning on curated data. GPT-4.5 introduced techniques to train larger models using data derived from smaller models, improving steerability and conversational nuance. GPT-5 achieved state-of-the-art results across math (94.6% on AIME 2025), real-world coding (74.9% on SWE-bench Verified), and multimodal understanding (84.2% on MMMU).

**Post-training:** OpenAI combines traditional SFT and RLHF with a new "safe completions" paradigm that centers safety on the assistant's output rather than binary classification of user intent. This approach maximizes helpfulness subject to safety policy constraints. Reasoning models (o-series, now integrated into GPT-5) are trained through dedicated reinforcement learning to produce internal chains of thought.

**Deployment innovation:** GPT-5 uses a real-time router that decides between instant responses and deeper reasoning based on conversation type, complexity, tool needs, and user intent. The router is continuously trained on real signals including user model-switching behavior, preference rates, and measured correctness. GPT-5 thinking achieves approximately 80% fewer factual errors than previous reasoning models while using 50–80% fewer output tokens.

| Model | Key Innovation | Notable Result |
|---|---|---|
| GPT-4.5 | Scaled pre-training + data from smaller models | Reduced hallucinations, higher EQ |
| GPT-5 | Unified reasoning + instant in one model | 94.6% AIME, 74.9% SWE-bench |
| GPT-5 pro | Parallel test-time compute scaling | 88.4% GPQA without tools |
| o3 / o4-mini | Dedicated reasoning RL training | Tunable think-harder vs. respond-faster |

### 16.2 DeepSeek – R1 and V3

**Architecture:** MoE with fine-grained experts, shared expert isolation, multi-head latent attention, and auxiliary-loss-free load balancing. DeepSeek-V3 serves as the base model for all R1 variants.

**Pre-training:** Trained on trillions of tokens at approximately $5.5 million—a fraction of the cost of comparable frontier models. This cost efficiency comes from architectural innovations (MoE reducing active compute) and training infrastructure optimization.

**Post-training (R1 pipeline):** DeepSeek pioneered the most influential post-training innovation of 2025: proving that pure RL can produce reasoning without any supervised fine-tuning. The full R1 pipeline consists of four stages:

1. **Stage 1 – Cold-Start SFT:** A small set of thousands of high-quality examples establishes baseline instruction-following and format compliance.
2. **Stage 2 – RLVR with GRPO:** The core innovation. GRPO eliminates the critic model entirely. For each prompt, 16 responses are sampled. Rule-based accuracy rewards (checking final answers against ground truth) and format rewards (enforcing structured reasoning in `<think>` tags) provide the training signal. No neural reward model is used—avoiding reward hacking during long training runs.
3. **Stage 3 – Rejection Sampling + SFT Stage 2:** The model generates many reasoning examples, which are filtered for correctness and quality. Only the best 600K reasoning examples and 200K general-purpose examples survive. This synthetic dataset is used for a second round of SFT.
4. **Stage 4 – Final RL (RLVR + RLHF):** A combined RL stage using rule-based rewards for reasoning tasks AND neural reward models (separate models for helpfulness and safety) for general-purpose alignment.

**Key results:** R1-Zero's AIME 2024 score jumped from 15.6% to 71.0% through pure RL alone, with majority voting reaching 86.7%. The model spontaneously developed self-verification, reflection, and "aha moment" behaviors without any explicit training for these capabilities.

**Distillation:** R1's reasoning was successfully transferred to smaller models (1.5B to 70B parameters) based on Qwen and Llama architectures. The 7B distilled model achieved 55.5% on AIME 2024.

### 16.3 Meta – Llama 4 Family

**Architecture:** Meta's first MoE models. Scout has 17B active parameters with 16 experts (109B total). Maverick has 17B active with 128 experts (400B total). Behemoth, still in training, targets 288B active with nearly 2 trillion total parameters. All use alternating dense and MoE layers with a shared expert per layer. The iRoPE architecture enables a 10-million-token context window.

**Pre-training:** Llama 4 introduced early fusion multimodality—text, image, and video tokens are integrated into a unified model backbone from the start, rather than being bolted on post-training. Uses MetaCLIP as a vision encoder and per-layer learning rate optimization.

**Post-training:** Meta revamped its pipeline with a critical insight: heavy SFT and DPO over-constrain the model, restricting exploration during the RL stage. Their new approach:

- **Lightweight SFT (hard examples only):** Used Llama models as judges to identify and remove more than 50% of training examples tagged as easy, fine-tuning only on the harder set.
- **Online Reinforcement Learning:** Continuous RL training with adaptive filtering and curriculum design, focusing on hard prompts to maintain reasoning, coding, and conversational abilities simultaneously.
- **Lightweight DPO:** Applied after RL to fine-tune specific corner cases and response quality.
- **Behemoth Codistillation:** The Behemoth teacher model generates outputs used to train Scout and Maverick.

**Deployment:** Scout fits on a single H100 GPU with INT4 quantization. Maverick fits on a single H100 host. This yields approximately 9–23× better price-performance ratio compared to GPT-4o.

### 16.4 Anthropic – Claude Family

**Architecture:** Dense transformer architecture. Specific architectural details are not publicly disclosed.

**Pre-training:** Large-scale pre-training on curated corpora with emphasis on responsible data sourcing and safety from the earliest stages.

**Post-training – Constitutional AI (CAI):** Anthropic's most distinctive contribution. Rather than depending solely on human annotators for alignment, CAI provides the model with a set of explicit principles and asks it to self-critique. The process operates in two phases:

- **Phase 1 – SL-CAI:** The model generates a response, then critiques itself by applying a principle from the constitution. It then revises its response. These revised outputs become the SFT training data.
- **Phase 2 – RL-CAI:** The revised outputs train a preference model, which is then used for RLHF. This creates a scalable alignment pipeline where AI feedback (RLAIF) dramatically reduces the need for human annotation on every edge case.

**Safety approach:** Multiple layers including classifiers, separate harmlessness reward models, extensive red-teaming, and a focus on honest, harmless, and helpful outputs.

### 16.5 Google DeepMind – Gemini Family

**Architecture:** Natively multimodal transformer trained on custom TPU v5 infrastructure. Handles text, images, audio, video, and code in a unified architecture from the ground up.

**Pre-training:** Trained across massive multimodal corpora. Gemini 2.5 Pro combines a large context window (1M+ tokens) with strong reasoning capabilities. Custom TPU hardware enables cost-efficient training at unprecedented scale.

**Post-training:** Combines traditional SFT and RLHF with dedicated reasoning training. Gemini 2.5 Pro introduced a "thinking" mode with extended chain-of-thought. A strong distillation pipeline transfers capabilities from Pro to Flash (speed-optimized) to Nano (on-device).

**Deployment:** Tight integration across Google products (Search, Android, Workspace). Vertex AI provides enterprise access. Edge models run on-device for privacy-preserving inference.

### 16.6 Alibaba / Qwen – Qwen 3 Family

**Architecture:** Dense and MoE variants. Qwen3-VL offers vision-language MoE configurations (30B-A3B and 235B-A22B) that use MoE as a compute allocation mechanism across modalities.

**Pre-training:** Trained on trillions of multilingual tokens with strong coverage of Chinese, English, and dozens of other languages. Universal tokenizer approach enables efficient multilingual adaptation.

**Post-training:** Uses Group Sequence Policy Optimization (GSPO), a variant related to GRPO. Qwen 3 introduced a "thinking" toggle allowing users to switch between fast responses and extended reasoning on demand.

**Open ecosystem:** Full open-weight releases across model sizes (0.6B to 235B). Qwen serves as one of the most popular base models for community fine-tuning and was one of the architectures DeepSeek used for R1 distillation.

### 16.7 Cross-Cutting Patterns

Examining these six organizations reveals several converging trends:

- **Modular post-training has replaced monolithic pipelines:** Every lab now uses a multi-stage approach. The old "pre-train → SFT → RLHF" recipe has been replaced by more nuanced sequences—Meta explicitly found that heavy SFT hurts RL performance.
- **MoE is the default architecture for frontier models:** Meta adopted MoE for Llama 4, DeepSeek uses fine-grained MoE, Qwen offers MoE variants, and closed labs are rumored to use sparse architectures. The efficiency gains are too compelling to ignore.
- **GRPO has displaced PPO as the dominant RL algorithm:** DeepSeek pioneered GRPO, eliminating the critic model and reducing memory by 33–50%. Qwen adopted GSPO, and the broader community has embraced variants.
- **Distillation is a first-class strategy:** Meta trains Behemoth as a teacher for Scout/Maverick. DeepSeek distills R1 into 1.5B–70B models. Google cascades from Pro to Flash to Nano. The pattern: train the biggest model you can, then compress into deployable sizes.
- **Reasoning through RL is the frontier:** Every major lab has invested in RL-trained reasoning. DeepSeek proved pure RL works. OpenAI integrated reasoning into GPT-5. Google added "thinking" mode to Gemini.
- **Safety innovations are diverging productively:** OpenAI's safe completions, Anthropic's Constitutional AI, Meta's Llama-as-judge filtering, and DeepSeek's separate helpfulness/harmlessness reward models represent different but complementary approaches.

| Company | Architecture | Post-Training Pipeline | Key Innovation |
|---|---|---|---|
| OpenAI | Dense (rumored MoE) | SFT → Safe Completions → RLHF → Reasoning RL | Safe completions, unified routing |
| DeepSeek | Fine-grained MoE | Cold SFT → GRPO/RLVR → Rejection Sampling → SFT2 → RLHF+RLVR | Pure RL reasoning, GRPO |
| Meta | MoE (16–128 experts) | Light SFT (hard only) → Online RL → Light DPO | Light SFT + heavy RL, codistillation |
| Anthropic | Dense | SFT → Constitutional AI → RLHF/RLAIF | Constitutional AI, self-critique |
| Google | Multimodal transformer | SFT → RLHF → Reasoning RL → Distillation | Native multimodality, TPU scale |
| Qwen | Dense + MoE variants | SFT → GSPO → Reasoning RL | GSPO, multilingual, open weights |

---

# Appendices

---

## Appendix A: Glossary of Key Terms

- **RLHF:** Reinforcement Learning from Human Feedback – Alignment technique using human preference data and a reward model.
- **DPO:** Direct Preference Optimization – Alignment without a reward model, using preference pairs as a classification objective.
- **GRPO:** Group Relative Policy Optimization – RL algorithm that eliminates the critic model by comparing responses within a group.
- **DAPO:** Decoupled Alignment and Policy Optimization – Further simplifies GRPO by removing the reference model.
- **RLVR:** Reinforcement Learning with Verifiable Rewards – Training on tasks with automatically checkable correctness.
- **SFT:** Supervised Fine-Tuning – Training on instruction-response pairs.
- **LoRA:** Low-Rank Adaptation – Parameter-efficient fine-tuning using low-rank matrices.
- **QLoRA:** Quantized LoRA – Combines LoRA with 4-bit quantization for memory efficiency.
- **MoE:** Mixture of Experts – Sparse architecture activating only a subset of parameters per token.
- **BPE:** Byte-Pair Encoding – Subword tokenization algorithm.
- **KV Cache:** Key-Value Cache – Stored attention states that avoid redundant computation during autoregressive generation.
- **FSDP:** Fully Sharded Data Parallelism – Distributed training strategy that shards model state across devices.
- **PPO:** Proximal Policy Optimization – RL algorithm used in RLHF.
- **CAI:** Constitutional AI – Anthropic's approach to scalable safety alignment using self-critique against principles.
- **SimPO:** Simple Preference Optimization – DPO variant using average log probability as implicit reward.
- **KTO:** Kahneman-Tversky Optimization – Alignment from binary (thumbs up/down) feedback.
- **RLAIF:** Reinforcement Learning from AI Feedback – Uses AI-generated preferences instead of human annotations.

---

## Appendix B: Recommended Reading

- Vaswani et al. (2017) – "Attention Is All You Need" – The original Transformer paper
- Ouyang et al. (2022) – "Training language models to follow instructions with human feedback" – InstructGPT / RLHF
- Rafailov et al. (2023) – "Direct Preference Optimization" – The DPO paper
- Shazeer et al. (2017) – "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- DeepSeek-AI (2025) – "DeepSeek-R1" – GRPO and RLVR for reasoning
- Fedus et al. (2022) – "Switch Transformers" – Simplified MoE with top-1 routing
- Hu et al. (2021) – "LoRA: Low-Rank Adaptation of Large Language Models"
- Bai et al. (2022) – "Constitutional AI" – Anthropic's approach to alignment
- Chip Huyen (2025) – "AI Engineering" – O'Reilly – Production ML/AI systems
- Natural Language Processing with Transformers (2025 revised edition) – O'Reilly – Hugging Face authors

---

## Appendix C: Hardware Reference

| GPU | VRAM | FP16 TFLOPS | Best For |
|---|---|---|---|
| NVIDIA H100 SXM | 80 GB HBM3 | 989 | Frontier pre-training and RL |
| NVIDIA H200 | 141 GB HBM3e | 989 | Large model training, extended context |
| NVIDIA A100 80GB | 80 GB HBM2e | 312 | General-purpose training and inference |
| NVIDIA L40S | 48 GB GDDR6 | 362 | Inference and fine-tuning |
| NVIDIA RTX 4090 | 24 GB GDDR6X | 330 | QLoRA fine-tuning, local inference |
| Google TPU v5e | 16 GB HBM2e | 197 | Large-scale JAX/TensorFlow training |
| AMD MI300X | 192 GB HBM3 | 1,307 | Memory-intensive workloads |

---

*This document reflects the state of the art as of April 2026. The field of LLM training evolves rapidly; techniques described here represent current best practices but will continue to advance.*
