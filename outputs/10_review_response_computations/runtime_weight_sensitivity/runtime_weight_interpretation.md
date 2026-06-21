# Runtime-Weight Sensitivity Interpretation

Runtime-weight sensitivity was recomputed without rerunning the benchmark by reusing saved raw metric tensors. The tested schemes reduced the execution-time weight to 0.05 with two redistribution rules and removed execution time entirely with renormalization of the remaining metrics.

Across single-only and union pools, the minimum Spearman correlation with the paper-default ranking was 0.773, and the maximum rank shift among paper-default top-10 entities was 15. At least one alternative runtime-weight scheme changed the top-ranked entity.

These results should be used to state whether runtime weighting is a robustness parameter rather than a driver of the main conclusions.
