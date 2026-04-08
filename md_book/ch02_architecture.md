# Transformer Architecture -- The Foundation
\minitoc

\begin{chapteroverview}
  
    - Evaluate the decoder-only transformer and its 2026 architectural variations.
    - Analyze MQA, GQA, and MLA for efficient KV cache management at scale.
    - Compare modern positional encodings including RoPE, iRoPE, and ALiBi.
    - Review production-standard normalization (RMSNorm) and activation (SwiGLU) functions.
    - Understand the 2025 transition to fine-grained Mixture of Experts (MoE).
  
\end{chapteroverview}

## The Decoder-Only Transformer

Nearly all modern autoregressive LLMs use a decoder-only transformer architecture: sequential blocks of masked self-attention and feed-forward sub-layers. In PaLM-540B, approximately 90\% of parameters reside in feed-forward layers---which is why MoE focuses on FFN efficiency.

### Architectural Comparison: Key Modern Models

\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{2.8cm}L{2cm}L{2cm}L{3.5cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**Model**} & \textcolor{white}{**Params**} & \textcolor{white}{**Context**} & \textcolor{white}{**Notable Features**} \\
\midrule
Llama 3.1 (70B) & 70B & 128K & GQA, RoPE, SwiGLU \\
Mistral 7B & 7B & 32K & GQA, sliding window \\
Mixtral 8x7B & 47B total & 32K & MoE, 13B active \\
DeepSeek-V3 & 671B total & 128K & Fine-grained MoE, MLA \\
Llama 4 Maverick & 400B total & 1M & MoE, iRoPE, 128 experts \\
\bottomrule
\end{tabular}
\caption{Representative modern model architectures}
\end{table}

## Key Architectural Components

### Multi-Head Attention Variants

Standard multi-head attention (MHA) uses $H$ heads each with independent $Q$, $K$, $V$ projections (formulation: Appendix~app:attention; code: Appendix~H). The KV cache grows as $2 \times H \times d_\text{head} \times L$ per token at inference---a critical bottleneck at long context.


  - **Multi-Query Attention (MQA):** Single K,V head shared by all query heads. Maximum KV cache reduction; can hurt quality at small scale (formulation \S\ref*{form:mqa}).
  - **Grouped Query Attention (GQA):** $G$ KV head groups ($1 < G < H$) [ainslie2023gqa]. Balances cache reduction and quality. Used in Llama 3, Mistral, and most 2025 models (formulation \S\ref*{form:gqa}, code Listing~\ref*{lst:gqa}).
  - **Multi-head Latent Attention (MLA):** DeepSeek-V3. KV cache is compressed into a low-rank latent vector $c_{KV}$, from which heads are projected via an up-projection matrix $W_{UK}$. This achieves the memory footprint of MQA while maintaining the expressive power of MHA (see formulation \S\ref*{form:mla}, code Listing~\ref*{lst:mla}).


> **MLA Mechanics**
>
> MLA's core innovation is the decoupled latent vector:
> $$
> c_{KV} = W_{DK} h_t, \quad k_t, v_t = W_{UK} c_{KV}
> $$
> where $W_{DK}$ is a down-projection and $W_{UK}$ an up-projection. RoPE is applied to a separate, non-compressed portion of the query and key to maintain relative position awareness without bloating the KV cache.

### Positional Encodings

Full derivations in Appendix~app:pos.

  - **RoPE (Rotary Position Embedding):** Applies rotation matrices to Q/K vectors [su2024roformer]. Encodes relative positions implicitly. Dominant 2025--2026 standard.
  - **iRoPE (Llama 4):** Interleaved no-position and RoPE layers. Enables 10M-token context without explicit positional interpolation.
  - **ALiBi:** Linear bias on attention scores [press2022train]. Zero extra parameters, strong length generalization.
  - **YaRN / LongRoPE:** NTK-aware RoPE interpolation for 4--8$\times$ context extension without full retraining [peng2023yarn].


### Normalization and Activation

Full formulations in Appendix~app:norm (RMSNorm, SwiGLU).

  - **RMSNorm with pre-normalization:** More stable than post-norm LayerNorm. Dominant in all major 2025 models (formulation \S\ref*{form:rmsnorm}).
  - **SwiGLU:** Gated FFN activation combining Swish and GLU [shazeer2020glu]. Standard for 2024--2026 (formulation \S\ref*{form:swiglu}).


\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{2cm}L{3.5cm}L{5cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**Function**} & \textcolor{white}{**Formulation**} & \textcolor{white}{**Key Trade-off / Usage**} \\
\midrule
ReLU & $\max(0, x)$ & Simple, fast. Risk of ``dying ReLU'' (zero gradient). \\
GeLU & $x \Phi(x)$ & Smooth, probabilistic gating. GPT-2/3 standard. \\
Swish / SiLU & $x \sigma(x)$ & Non-monotonic, smoother than ReLU. Llama 1/2. \\
SwiGLU & $\text{Gated}(x, \text{Swish})$ & Most expressive. 2026 production standard. \\
\bottomrule
\end{tabular}
\caption{Comparative analysis of activation functions in LLMs}
\end{table}

## Mixture of Experts (MoE)

Replace the dense FFN with multiple smaller expert networks and a router selecting which experts process each token [shazeer2017outrageously] (formulation \S\ref*{form:moe}, code Listing~\ref*{lst:moe}). Only 1--2 of 8--64 experts activate per token. Mixtral 8x7B: 47B total parameters, $\sim$13B active per forward pass.

> **Key MoE Developments in 2025--2026**
>
> - **DeepSeekMoE:** Fine-grained experts with shared expert isolation. 256 routed experts + 1 shared expert per layer.
>   - **DeepSeekMoE [dai2024deepseekmoe**:] Fine-grained experts with shared expert isolation. 256 routed experts + 1 shared expert per layer.
>   - **Sparse Upcycling:** Convert dense models to MoE without training from scratch.
>   - **SwitchHead:** MoE applied to attention projection layers (Q, K, V).
>   - **Expert Parallelism:** Distributed experts across GPUs for efficient scale.
>   - **Load Balancing:** Auxiliary loss or expert-choice routing prevents expert collapse.

## Context Length and Efficient Attention


  - **Flash Attention (v2/v3):** $O(n^2) \rightarrow O(n)$ memory via IO-aware tiling [dao2022flashattention,dao2023flashattention2,shah2024flashattention3]. Mandatory for modern training. FA3 adds warp-specialization for $\sim$2$\times$ speedup over FA2 on H100.
  - **Ring Attention:** Million-token contexts across devices in a ring topology. Each device holds one segment; KV chunks circulate.
  - **Sliding Window Attention:** Fixed local window with cross-layer information flow (Mistral).


\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{3.5cm}C{2.5cm}L{4.5cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**Context Window**} & \textcolor{white}{**KV Cache (7B)**} & \textcolor{white}{**Use Case**} \\
\midrule
4K tokens & 512 MB & Short conversations \\
32K tokens & 4 GB & Document processing \\
128K tokens & 16 GB & Full codebase analysis \\
1M tokens & 128 GB & Entire book corpus \\
\bottomrule
\end{tabular}
\caption{Approximate KV cache memory at different context lengths (BF16, 32 heads)}
\end{table}


% ══════════════════════════════════════════════════════════════════
%  PART II: DATA
% ══════════════════════════════════════════════════════════════════
