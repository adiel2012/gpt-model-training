\chapter*{Preface}
\addcontentsline{toc}{chapter}{Preface}

The large language model revolution has moved faster than any previous technology cycle in machine learning. What required a year-long research effort at a top lab in 2022 now fits in a weekend hackathon; models that once cost \$10 million to train can be replicated for tens of thousands of dollars.

This guide is written for engineers and researchers who need to *do the work*---not survey it from a distance. It covers the full training stack: data pipelines, pre-training at scale, the post-training alignment stack, evaluation, safety, and production deployment. Each chapter synthesizes the most important findings from 2024--2026 alongside the enduring principles from earlier years.

**Who this book is for:** ML engineers building fine-tuning pipelines; researchers designing new alignment methods; infrastructure engineers provisioning GPU clusters; technical leads making architectural decisions for LLM-powered products.

**How to read this book:** This guide is designed for both linear reading and targeted reference.

| - **Parts I--III (Foundations \ | Pre-training):** Essential for teams building or modifying base models. Covers data, architecture, and scaling. |
| - **Parts IV--V (Post-training \ | Alignment):** The high-value stack for 2026. Relevant to everyone from hobbyists to frontier researchers. |
  - **Part VI (Advanced Topics):** Specialized research on reasoning, multi-modality, and the 2027 roadmap.

Each chapter begins with a **What You Will Learn** overview and a **Mini Table of Contents** for quick navigation. Mathematical formulations are concentrated in Appendix~app:pipeline--app:continual, with direct links provided throughout the text. Ready-to-use PyTorch code exists in Appendix~H, serving as the executable companion to the theoretical chapters.
