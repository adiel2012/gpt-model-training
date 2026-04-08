# Tokenization -- From Text to Vectors
\minitoc

\begin{chapteroverview}
  
    - Compare BPE, Unigram, and WordPiece tokenization algorithms.
    - Master the 2026 standard 128K vocabularies for multilingual performance.
    - Implement multi-byte character handling and special token management.
    - Analyze the trade-offs between vocabulary size and inference latency.
  
\end{chapteroverview}

## Subword Tokenization

BPE [sennrich2016neural] (dominant, GPT-series, 32K--128K vocabulary), WordPiece (BERT), SentencePiece [kudo2018sentencepiece] (language-agnostic, multilingual models).

### Vocabulary Size Trade-offs

Larger vocabularies reduce sequence length (faster attention) but increase embedding parameter count and can impair performance on rare tokens. 50K--100K is the 2025 sweet spot; some multilingual models use 200K+.

## Universal Tokenizers

Up to 20.2\% higher win rates in language adaptation. 5\% improvement on unseen languages. Enables efficient post-training language expansion. Key design criterion: fertility ratio (tokens per word) should be $\leq 2$ for all target languages.

## Sequence Packing

Length-aware combinatorial optimization: 40\% fewer truncations, improved modeling performance. Cross-document attention masking prevents spurious inter-document dependencies within packed batches.
