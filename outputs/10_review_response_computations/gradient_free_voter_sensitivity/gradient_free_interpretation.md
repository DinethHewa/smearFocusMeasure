# Gradient-Free Voter Sensitivity Interpretation

The gradient-free surrogate-label sensitivity analysis rebuilt labels using only Normalized Variance, Histogram Entropy, GLCM Contrast, and Fourier Transform Sharpness Index. No gradient or Laplacian voter was used. The resulting labels are an internal surrogate-label stress test and should not be described as optical ground truth.

Under gradient-free rank-based scoring, the top-ranked operator was Brenner Gradient. Under gradient-free value-based scoring, the top-ranked operator was Variance of Gradient. Gradient-family operators contributed 9 of the rank-based top 10 and 8 of the value-based top 10.

The gradient-family top tier largely survives this internal sensitivity analysis.

Operators with the largest upward rank movement under gradient-free labels were: Brenner Gradient, Intensity Variance, Normalized Variance, Absolute Central Moment 2, GLCM Contrast. Operators with the largest downward movement were: Intensity Range Index, Sum Modified Laplacian, Squared Gradient, Laplacian Energy, Variance of Laplacian.

Manuscript wording should state that this is a surrogate-label sensitivity analysis designed to test dependence on derivative-based voters, not an independent hardware-focus validation.
