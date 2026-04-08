# Hardware Reference

\begin{table}[H]
\centering\small\sffamily
\rowcolors{2}{tablealt}{white}
\begin{tabular}{L{3cm}L{2.5cm}C{2cm}L{3.5cm}}
\toprule
\rowcolor{tablehead}\textcolor{white}{**GPU / Accelerator**} & \textcolor{white}{**VRAM**} & \textcolor{white}{**FP16 TF**} & \textcolor{white}{**Best For**} \\
\midrule
NVIDIA B200 SXM & 192 GB HBM3e & 2,250 & Next-gen frontier training \\
NVIDIA GB200 NVL72 & 13.5 TB (rack) & 130,000 & Rack-scale cluster training \\
NVIDIA H200 & 141 GB HBM3e & 989 & Large models, long context \\
NVIDIA H100 SXM & 80 GB HBM3 & 989 & Frontier pre-training and RL \\
NVIDIA A100 80GB & 80 GB HBM2e & 312 & General training/inference \\
NVIDIA L40S & 48 GB GDDR6 & 362 & Inference and fine-tuning \\
NVIDIA RTX 4090 & 24 GB GDDR6X & 330 & QLoRA, local inference \\
Google TPU v5e & 16 GB HBM2e & 197 & Large-scale JAX/TF training \\
AMD MI300X & 192 GB HBM3 & 1,307 & Memory-intensive workloads \\
\bottomrule
\end{tabular}
\caption{GPU and accelerator reference for LLM training (TFLOPS are peak BF16/FP16)}
\end{table}
