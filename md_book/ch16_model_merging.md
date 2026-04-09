# Model Merging and Recombination

> [!IMPORTANT]
> **What You Will Learn**
> - Master task vector arithmetic and SLERP weight interpolation.
> - Implement TIES-Merging and DARE for combining fine-tuned experts.
> - Evaluate the "FrankenMoE" strategy for creating heterogeneous ensembles.
> - Analyze the 2025 shift from training singular models to evolving model families.

Model merging combines the weights of multiple fine-tuned models without additional training, enabling capability combination at near-zero compute cost.

## Merging Methods

  - **Linear interpolation:** $\theta = (1-\alpha)\theta_A + \alpha\theta_B$. Simple but creates interference between dissimilar capabilities.
  - **SLERP (Spherical Linear Interpolation):** Interpolates along the geodesic of weight space. Better than linear for preserving each model's characteristics.
  - **TIES-Merging [yadav2023ties]:** Trim redundant deltas $\rightarrow$ Elect sign direction $\rightarrow$ Disjoint merge. Handles sign conflicts between different fine-tuned models.
  - **DARE [yu2023language] (Drop And REscale):** Randomly drop task-vector parameters and rescale survivors. Reduces interference between merged models by eliminating redundant updates.
  - **Model Breadcrumbs:** Remove outlier parameters before merging. Outliers cause interference; removing them improves merge quality.

## Task Vectors

Full formulations in [Appendix G](app_g_implementation_treasury.md): task arithmetic, SLERP, TIES, DARE; code: [Appendix G](app_g_implementation_treasury.md).

Task vector [ilharco2022editing] $\tau = \theta_\text{fine-tuned} - \theta_\text{base}$. Arithmetic on task vectors enables:

  - **Capability addition:** $\theta_\text{base} + \tau_A + \tau_B$ combines coding and math capabilities.
  - **Capability subtraction:** $\theta_\text{base} - \tau_\text{toxic}$ removes unwanted behaviors.
  - **Analogy:** $\theta_\text{base} + \tau_\text{French} - \tau_\text{English}$ adapts a model to French.

> **When Merging Works Best**
>
> Merging works best when models share the same base and have been fine-tuned on complementary (not conflicting) tasks. Models fine-tuned on the same task but different data benefit more from ensemble methods than weight merging.

## Practical Merging Workflow

1. Fine-tune $n$ specialized models from the same base.
2. Compute task vectors $\tau_i = \theta_i - \theta_\text{base}$ for each.
3. Apply TIES or DARE to reduce interference.
4. Evaluate merged model on target benchmarks; tune merge coefficients $\alpha_i$.
5. Distribute as a single merged model.

Tools: `mergekit` [ilharco2022editing] (open source), Hugging Face `transformers` weight manipulation.


---

[← Previous Chapter](ch15_domain_multimodal.md) | [Table of Contents](../README.md#table-of-contents) | [Next Chapter →](ch17_continual_learning.md)