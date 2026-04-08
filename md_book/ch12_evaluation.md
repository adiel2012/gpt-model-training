

# Evaluating LLMs -- Benchmarks and Beyond

> **What You Will Learn**
> - Master the 2026 benchmark suite: MMLU-Pro, HumanEval, and GPQA.
> - Implement LLM-as-a-Judge and preference-based ELO evaluation systems.
> - Detect and mitigate data contamination in benchmark results.
> - Evaluate qualitative capabilities through red-teaming and side-by-side comparison.

| **Benchmark** | **Focus** | **Type** |
|---|---|---|
| MMLU / MMLU-Pro | Broad academic knowledge (57+ subjects) | Multiple choice |
| HumanEval / MBPP | Code generation correctness | Code execution |
| GSM8K / MATH | Mathematical problem solving | Verified answers |
| TruthfulQA | Resistance to misconceptions | Open-ended |
| MT-Bench | Multi-turn conversation | LLM-judged |
| AlpacaEval 2 | Instruction-following quality | LLM-judged |
| Arena-Hard | Head-to-head comparison | LLM-judged |
| SWE-bench | Real-world software engineering | Code execution |
| AIME 2024 | Competition mathematics | Verified answers |
| GPQA-Diamond | Expert-level science reasoning | Multiple choice |

*Table: Standard LLM evaluation benchmarks*

## LLM-as-Judge

LLM judges (GPT-4, Claude, Gemini) score open-ended responses. Pitfalls: position bias (judges prefer the first response), verbosity bias (longer responses rated higher), self-enhancement bias (models rate their own outputs higher). Mitigations: swap positions across evaluations, length-normalize scores, use multiple judge models.

## Chatbot Arena and ELO Rankings

Chatbot Arena (LMSYS) collects millions of pairwise human comparisons and computes ELO ratings. Statistically robust due to scale; reflects real user preferences better than academic benchmarks. High ELO and high MMLU are different things---they measure different aspects of model quality.

## Benchmark Contamination and Gaming

  - Evaluation data present in pre-training inflates scores. Always report decontamination procedures.
  - Optimizing specifically for a benchmark can improve scores without improving underlying capability (Goodhart's Law).
  - Mitigations: dynamic benchmarks with new questions each run, held-out test sets never used for selection, diverse multi-benchmark suites that resist narrow optimization.

