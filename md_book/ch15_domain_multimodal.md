# Specialized Domains and Multi-modality

> [!IMPORTANT]
> **What You Will Learn**
> - Master data curation and fine-tuning for Coding, Law, and Science.
> - Implement the 2026 multi-modal stack: Vision, Audio, and Action.
> - Analyze world models and agentic integration in specialized domains.
> - Evaluate the "Small Language Model" (SLM) trend for domain-specific edge AI.

## Domain-Specific Pre-training

Domain-specific models outperform general models on target tasks because domain vocabulary and reasoning patterns require exposure during pre-training, not just fine-tuning.

  - **BloombergGPT [wu2023bloomberggpt** (finance):] 50% financial text, 50% general. Outperforms GPT-3 on financial benchmarks while remaining competitive on general tasks. Key lesson: 50/50 mix preserves generality.
  - **Med-PaLM 2 [singhal2023large** (medical):] Expert-level performance on the USMLE. Multi-step reasoning over clinical evidence requires both medical knowledge and reasoning capabilities.
  - **CodeLlama / DeepSeek-Coder:** Code-specialized continued pre-training from a general base. Infill (fill-in-the-middle) pre-training objective for code completion.
  - **ChatLAW (legal):** Legal corpus with case law, statutes, and regulatory filings.

## Multimodal Models

  - **Vision-Language Models (VLM):** Connect a vision encoder (CLIP, SigLIP) to an LLM via a projection layer or cross-attention. LLaVA, InternVL, Qwen3-VL.
  - **Early fusion (Llama 4):** Text and image tokens processed jointly from the first layer. More parameter-efficient at scale than late-fusion adapters.
  - **Vision-Language MoE (Qwen3-VL):** Cross-modal compute allocation---vision and language tokens routed to specialized experts.
  - **Omni-modal:** Audio, video, and document understanding in a unified model (Gemini 2.5 Pro, GPT-4o).

## Agentic AI

LLMs as agents: perceive environment, plan actions, use tools, and produce observable effects. Training considerations:

  - End-to-end agent evaluation (SWE-bench, WebArena, GAIA) requires operating in real environments, not just text.
  - Tool-use fine-tuning: function calling, structured JSON output, code execution.
  - Multi-agent coordination: multiple specialized LLMs operating in concert.



---

[← Previous Chapter](ch14_inference.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](ch16_model_merging.md)