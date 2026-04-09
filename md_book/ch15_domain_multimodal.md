# Specialized Domains and Multi-modality

> [!IMPORTANT]
> **What You Will Learn**
> - Master data curation and fine-tuning for Coding, Law, and Science.
> - Implement the 2026 multi-modal stack: Vision, Audio, and Action.
> - Train for Agentic AI: Tool-use SFT, Function Calling, and Environment-Loop RL.
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

LLMs as agents: perceive environment, plan actions, use tools, and produce observable effects. Training a "frontier agent" in 2026 requires three specific technical layers.

### 1. Tool-use SFT (Supervised Fine-Tuning)
The model must learn to recognize when to use a tool and how to generate formatted calls (JSON/XML).
- **Function Calling Datasets:** Models like Llama-3-Agent or Berkeley Function Calling (BFCL) are trained on millions of synthetic tool-use traces (e.g., Magpie method).
- **Negative Constraint Training:** Training the model to *reject* tool-use when the user's intent is purely conversational, reducing hallucinated code execution.

### 2. Planning and Chain-of-Thought (RL)
Complex agency requires planning before execution.
- **R1-style Planning:** Inspired by DeepSeek-R1, models are trained to output a `<thought>` block detailing the intended multi-step strategy before generating the tool call.
- **Environment-Loop RL:** The model generates an action, receives the real output from the tool (e.g., a Python interpreter), and is rewarded for achieving the final goal. This allows the model to learn from execution errors.

### 3. Agentic Evaluation
End-to-end evaluation requires operating in real environments, not just measuring text similarity.
- **SWE-bench:** Solving GitHub issues in a live repository.
- **GAIA:** General AI Assistants benchmark requiring 2025-level cross-domain tool orchestration.
- **Multi-agent Coordination:** Training multiple LLMs to operate in hierarchical structures (e.g., Manager-Worker patterns).




---

[← Previous Chapter](ch14_inference.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](ch16_model_merging.md)