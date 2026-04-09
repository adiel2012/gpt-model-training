# Tokenization -- From Text to Vectors

> [!IMPORTANT]
> **What You Will Learn**
> - Compare BPE, Unigram, and WordPiece tokenization algorithms.
> - Master the 2026 standard 128K vocabularies for multilingual performance.
> - Implement multi-byte character handling and special token management.
> - Analyze the trade-offs between vocabulary size and inference latency.

Tokenization converts raw text into integer token IDs that the model consumes. The choice of tokenizer affects sequence length, coverage of rare words, multilingual quality, and downstream performance. It is fixed before training and cannot be changed without retraining from scratch.

## Subword Tokenization

Character-level models generalize to any input but produce very long sequences. Word-level models produce compact sequences but fail on rare or unseen words. **Subword tokenization** occupies the optimal middle ground: common words map to single tokens, rare words decompose into meaningful pieces, and nothing is truly out-of-vocabulary.

The three dominant families:

| **Algorithm** | **Used By** | **Vocab Size** | **Key Property** |
|---|---|---|---|
| BPE | GPT-2/3/4, LLaMA, Mistral | 32K--128K | Frequency-driven merges |
| WordPiece | BERT, mBERT | 30K--120K | Likelihood-driven merges |
| Unigram LM | SentencePiece, T5, Gemma | 32K--256K | Probabilistic, prunable |

### Byte-Pair Encoding (BPE)

BPE [sennrich2016neural] starts from a character (or byte) vocabulary and iteratively merges the most frequent adjacent pair into a new token:

1. Initialize vocabulary with all individual characters (or bytes).
2. Count all adjacent symbol pairs in the corpus.
3. Merge the most frequent pair into a single new token.
4. Repeat until the vocabulary reaches the target size.

**Byte-level BPE** (used by GPT-2, LLaMA, Mistral) operates on raw UTF-8 bytes rather than characters. Every possible byte (0–255) is in the initial vocabulary, so the model is truly open-vocabulary — no input can produce an unknown token.

**Strengths:** Deterministic, fast to encode, handles any Unicode input. **Weakness:** Merge order is corpus-dependent; two tokenizers trained on different corpora produce incompatible vocabularies.

### WordPiece

WordPiece [schuster2012japanese] uses a similar iterative merging strategy but chooses the merge that maximizes the likelihood of the training corpus under a unigram language model rather than the raw pair frequency. This favors merges that are linguistically meaningful. Tokens inside a word are prefixed with `##` (e.g., `playing` → `play`, `##ing`).

### Unigram Language Model (SentencePiece)

SentencePiece [kudo2018sentencepiece] starts from a large over-complete vocabulary and iteratively **removes** tokens whose removal causes the smallest drop in corpus likelihood. The result is a vocabulary where each token's probability is explicitly modeled.

**Advantages:** Language-agnostic (no whitespace assumptions), supports subword regularization (sampling different segmentations at training time as a data augmentation), and handles CJK and other script-native tokenization cleanly. Widely used in multilingual models (mT5, Gemma, Qwen).

### Vocabulary Size Trade-offs

Larger vocabularies reduce sequence length (faster attention, lower compute per token) but increase the embedding matrix size and require more data to learn good representations for rare tokens.

| **Vocab Size** | **Typical Use** | **Trade-off** |
|---|---|---|
| 32K | Monolingual English | Compact, well-trained embeddings |
| 64K--100K | Bilingual / code-heavy | Good balance for 2024-era models |
| 128K | Multilingual (2025 standard) | Short sequences; needs large corpus |
| 200K+ | Massively multilingual | Best coverage; embedding cost high |

The **2026 standard** for frontier models is 128K tokens. LLaMA 3 moved from 32K to 128K to improve multilingual and code coverage. For a 7B model, the embedding layer at 128K×4096 = 2.1B parameters — roughly 30% of total parameters.

**Rule of thumb:** Adding vocabulary tokens is cheap at inference but expensive at training (more parameters to learn). Over-large vocabularies hurt low-resource languages because their tokens are seen rarely.

## Multi-Byte Character Handling

Non-ASCII scripts (Chinese, Arabic, Hebrew, Devanagari) encode as multiple UTF-8 bytes. Byte-level BPE handles this transparently — every byte is a valid token, and frequent multi-byte sequences merge naturally. Character-level or word-level tokenizers require explicit Unicode normalization (NFC/NFKC) to avoid representing the same character as different token sequences.

**Practical pitfalls:**
- Punctuation normalization: curly quotes `"` vs straight `"` can produce different tokens.
- Whitespace: leading spaces matter in BPE (` dog` ≠ `dog`). The `add_prefix_space` parameter controls this.
- Digits: many tokenizers split numbers character-by-character (`2`, `0`, `2`, `6`), which hampers arithmetic. Models trained with number-aware tokenization (e.g., treating `2026` as one token) show better numeric reasoning.

## Special Token Management

Every LLM tokenizer defines a set of special tokens that signal structure to the model:

| **Token** | **Purpose** |
|---|---|
| `<\|begin_of_text\|>` | Marks start of sequence (LLaMA 3) |
| `<\|end_of_text\|>` / `<eos>` | End of generation signal |
| `<\|start_header_id\|>` | Role boundary in chat format |
| `<pad>` | Padding in batched inference |
| `<unk>` | Unknown token (byte-level BPE makes this unnecessary) |
| `<\|image\|>`, `<\|audio\|>` | Multimodal content placeholders |

**Adding special tokens** after pre-training requires updating both the tokenizer vocabulary and the embedding matrix. New token embeddings are typically initialized as the mean of related tokens (e.g., a new `<tool_call>` token initialized from the mean of `<`, `tool`, `>` embeddings).

## Universal Tokenizers

A tokenizer trained predominantly on English transfers poorly to other languages — fertility (tokens per word) explodes for under-represented scripts. Universal tokenizers [rust2021good] address this by:

- Training the vocabulary on a balanced multilingual corpus.
- Enforcing a maximum **fertility ratio** $\leq 2$ tokens per word across all target languages.
- Allocating vocabulary capacity proportional to target-language data budget, not source-corpus frequency.

Results: up to 20.2% higher win rates in language adaptation tasks, 5% improvement on unseen languages. Enables efficient post-training expansion to new languages without vocabulary surgery.

**LLaMA 3 / Gemma 2 approach:** 128K vocabulary trained on 30+ languages with explicit fertility constraints. Fertility for English is ~1.3; for Chinese ~1.5; for Arabic ~1.8 — all within budget.

## Sequence Packing

Naïve batching pads short sequences to the longest sequence in the batch, wasting compute. **Sequence packing** concatenates multiple short documents into a single training sequence up to the context length, separated by end-of-sequence tokens.

**Benefits:**
- 40% fewer truncated sequences compared to padding-based batching.
- Near 100% GPU utilization — no padding tokens consume attention compute.
- Improved modeling of document boundaries.

**Cross-document attention masking** is critical: without it, the attention mechanism treats packed documents as a single long text, creating spurious dependencies across document boundaries. A block-diagonal attention mask restricts each document to attend only to its own tokens.

```python
# Example: block-diagonal attention mask for sequence packing
import torch

def packed_attention_mask(doc_lengths: list[int]) -> torch.Tensor:
    """Returns a causal mask that blocks cross-document attention."""
    total_len = sum(doc_lengths)
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    offset = 0
    for length in doc_lengths:
        mask[offset:offset+length, offset:offset+length] = True
        offset += length
    # Apply causal masking within each block
    causal = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
    return mask & causal
```

## Practical Tokenizer Selection Guide

| **Use Case** | **Recommended Tokenizer** | **Reason** |
|---|---|---|
| English-only fine-tuning | Match the base model | Never change tokenizer mid-training |
| Multilingual expansion | SentencePiece Unigram, 128K | Low fertility across scripts |
| Code-heavy tasks | BPE with 64K+ vocab | Code tokens merge better with larger vocab |
| On-device / edge | Smaller vocab (32K) | Smaller embedding matrix |
| New language addition | Extend existing vocab | Add 2K--8K language-specific tokens |


---

[← Previous Chapter](ch03_data_curation.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](ch05_synthetic_data.md)