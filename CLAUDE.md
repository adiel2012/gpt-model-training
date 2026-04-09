# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GPT model training project — training, fine-tuning, and evaluating GPT-style transformer models using Jupyter Notebooks.

## Structure

```
gpt-model-training/
├── md_book/                   # Markdown chapters and appendices
├── llm_implementations.ipynb  # Core implementations and training logic
└── README.md                 # Project overview
```

## Workflow

1.  **Read and Understand**: Follow the theoretical explanations in `README.md` and the `md_book/` chapters.
2.  **Implementation**: All code implementations (model modules, training loops, alignment objectives) reside in `llm_implementations.ipynb`.
3.  **Verification**: Use the "Verification Run" cell in the notebook to test new implementations against the described formulas.

## Commands

No separate training script is required. Execute cells within `llm_implementations.ipynb` to train and evaluate models.

```bash
# Install base dependencies
pip install torch pyyaml
```
