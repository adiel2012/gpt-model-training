# Data Curation -- The Oil for the Engine

> **What You Will Learn**
> - Implement quality filtering and deduplication at the trillion-token scale.
> - Review domain sampling proportions (Web, Code, Science, Books) for 2026.
> - Analyze strategies for removing PII and toxic content while preserving knowledge.
> - Understand the 2025 shift from quantity-driven to quality-driven curation.

## Why Data Quality Trumps Quantity

FineWeb [penedo2024fineweb] (15T tokens, heavily curated) outperforms RedPajama (20T tokens, lightly filtered)---5 trillion extra noisy tokens *hurt* performance. Modern frontier models train on 10--20T tokens, approaching available high-quality text limits.

## Data Sources

Web crawl (FineWeb, DCLM), books and academic literature, code repositories (The Stack), curated knowledge bases (Wikipedia), and multilingual sources (FineWeb-2).

| **Source Type** | **Proportion** | **Notes** |
|---|---|---|
| Web crawl | 50--70% | Requires aggressive cleaning |
| Code | 15--25% | Boosts reasoning and instruction following |
| Books | 5--15% | Long-form coherence |
| Scientific papers | 3--8% | Technical reasoning |
| Wikipedia / Wikidata | 2--5% | Factual grounding |
| Curated instruction data | 1--3% | Bridges to SFT |

*Table: Typical data source composition for pre-training*

## Cleaning and Filtering Pipeline

### Levels 0--3: Rule-Based
URL filtering, language identification, boilerplate removal, heuristic quality filters (perplexity thresholds, line length statistics, repetition detection).

### Levels 4--5: Model-Based
FastText/transformer classifiers for quality scoring. GneissWeb: sharded deduplication at 10T scale. SoftDedup: reweight rather than delete---preserves rare but informative examples.

### Levels 6+: LLM-Driven (2025--2026 Frontier)
DataEvolve / Data Darwinism: autonomous LLM-driven quality analysis and iterative cleaning. Category-specific strategies (SwallowCode, MegaMath). Instruction-response augmented pre-training.

## Deduplication

Exact (hash-based), near-duplicate (MinHash/LSH with GPU-accelerated connected components), and fuzzy (embedding-based similarity). Near-deduplication at the paragraph level catches reformatted reposts that exact dedup misses.

## Multi-Stage Training

Stage 1: large corpus (10T+) for breadth. Stage 2: curated subset (0.5--1T) for depth and polish. Some labs add a Stage 3: task-specific mix (code-heavy, math-heavy) aligned to deployment use case.

> **Data Contamination**
>
> Evaluation benchmarks present in pre-training data inflate benchmark scores without improving actual capability. Decontaminate training data by removing near-exact matches to test sets. Always report decontamination procedures in training reports.
