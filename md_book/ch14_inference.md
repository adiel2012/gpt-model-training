# Inference and Deployment at Scale
\minitoc

\begin{chapteroverview}
  
    - Distinguish between throughput-bound and memory-bound inference.
    - Master KV cache management (PagedAttention, RadixAttention).
    - Implement the 2026 quantization stack (H100 FP8, post-training quantization).
    - Optimize throughput with speculative decoding and drafting.
  
\end{chapteroverview}

## Quantization

\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{2.5cm}L{2cm}L{3cm}L{3cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**Format**} & \textcolor{white}{**Size Reduction**} & \textcolor{white}{**Quality Loss**} & \textcolor{white}{**Best For**} \\
\midrule
FP16 / BF16 & 2$\times$ & Minimal & Training \\
INT8 (LLM.int8\nolinebreak[dettmers2022llmint8]) & $\sim$4$\times$ & Negligible & Serving large models \\
INT4 / GPTQ / AWQ & $\sim$8$\times$ & Small & Throughput-optimized serving \\
GGUF (Q4\_K\_M) & $\sim$6$\times$ & Small & Local / edge inference \\
FP8 & 4$\times$ & Minimal & H100/H200 native serving \\
1.58-bit (BitNet) & $\sim$20$\times$ & Moderate & Research / edge \\
\bottomrule
\end{tabular}
\caption{Quantization formats and trade-offs}
\end{table}

## Inference Frameworks

vLLM (PagedAttention), TGI (Hugging Face), TensorRT-LLM (NVIDIA), llama.cpp/Llamafile (local/edge).

## Speculative Decoding

A small draft model generates $k$ tokens; the large target model verifies all $k$ in parallel [leviathan2023fast]. 2--3$\times$ speedup, zero quality loss. Works best when the draft model acceptance rate is high. Medusa: multiple parallel draft heads on the target model itself, eliminating the separate draft model.

## Continuous Batching

Naive batching waits for all requests in a batch to complete before accepting new ones---wasteful when requests differ in length. Continuous batching [yu2022orca] inserts new requests as slots free up, achieving near-100\% GPU utilization.

## KV Cache Management

KV cache formula (\S\ref*{form:kvcache}), quantization bounds (\S\ref*{form:quant}) in Appendix~app:pipeline; speculative decoding code: Listing~\ref*{lst:spec_dec}.


  - **PagedAttention (vLLM) [kwon2023efficient**:] Non-contiguous physical KV cache blocks managed like OS virtual memory. Eliminates fragmentation, enables 2--4$\times$ more concurrent requests.
  - **Prefix caching:** Cache shared system prompt KV states across requests. 50--80\% cache hit rates for common deployments.
  - **KV quantization:** Quantize cached KV to INT8/INT4 to extend effective context within fixed memory.


## Multi-GPU Inference

For models exceeding single-GPU VRAM, use tensor parallelism at inference:

  - 2--4 GPUs: tensor parallelism within a node (NVLink bandwidth sufficient).
  - 8+ GPUs: tensor + pipeline parallelism.
  - Reference: a 70B model at BF16 requires 4$\times$A100-80GB; INT4 fits on 2$\times$A100-40GB.



% ══════════════════════════════════════════════════════════════════
%  PART VI: ADVANCED TOPICS
% ══════════════════════════════════════════════════════════════════
