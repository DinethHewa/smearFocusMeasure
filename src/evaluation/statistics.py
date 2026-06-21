"""Bootstrap and paired statistical helpers for corrected evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def bootstrap_ci_mean(
    values: Sequence[float],
    *,
    n_resamples: int,
    conf_level: float,
    seed: int = 12345,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means: List[float] = []
    for _ in range(int(n_resamples)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    alpha = 1.0 - float(conf_level)
    low = np.quantile(means, alpha / 2.0)
    high = np.quantile(means, 1.0 - alpha / 2.0)
    return float(low), float(high)


def bootstrap_ci_generalization(
    matrix: np.ndarray,
    measure_names: Sequence[str],
    *,
    n_resamples: int = 500,
    seed: int = 42,
) -> List[Dict[str, float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    num_measures, num_blocks = arr.shape
    for measure_idx in range(num_measures):
        samples: List[float] = []
        for _ in range(int(n_resamples)):
            idx = rng.integers(0, num_blocks, size=num_blocks)
            samples.append(float(np.nanmean(arr[measure_idx, idx])))
        low, high = np.percentile(samples, [2.5, 97.5])
        rows.append(
            {
                "measure_name": str(measure_names[measure_idx]),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return rows


def _rank_biserial(delta: np.ndarray) -> float:
    nonzero = np.asarray(delta, dtype=np.float64)
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero != 0)]
    if nonzero.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(nonzero))
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    denom = pos + neg
    return float((pos - neg) / denom) if denom > 0 else float("nan")


def friedman_wilcoxon_holm(
    rank_matrix: np.ndarray,
    measure_names: Sequence[str],
    block_names: Sequence[str],
    *,
    family: str = "single_measure_rank_cells",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matrix = np.asarray(rank_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D rank matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(measure_names) or matrix.shape[1] != len(block_names):
        raise ValueError("Rank matrix shape does not match measure/block names")

    friedman_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []

    valid_columns = np.all(np.isfinite(matrix), axis=0)
    aligned = matrix[:, valid_columns]
    aligned_block_names = [str(name) for idx, name in enumerate(block_names) if valid_columns[idx]]

    if aligned.shape[1] < 3 or aligned.shape[0] < 3:
        friedman_rows.append(
            {
                "family": family,
                "n": int(aligned.shape[1]),
                "k": int(aligned.shape[0]),
                "methods": ",".join(str(name) for name in measure_names),
                "blocks": ",".join(aligned_block_names),
                "statistic": float("nan"),
                "p_value": float("nan"),
                "valid": False,
                "note": "Need at least three measures and three finite blocks for Friedman.",
            }
        )
        return friedman_rows, pairwise_rows

    try:
        stat = friedmanchisquare(*[aligned[row_idx, :] for row_idx in range(aligned.shape[0])])
        friedman_rows.append(
            {
                "family": family,
                "n": int(aligned.shape[1]),
                "k": int(aligned.shape[0]),
                "methods": ",".join(str(name) for name in measure_names),
                "blocks": ",".join(aligned_block_names),
                "statistic": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "valid": True,
                "note": "",
            }
        )
    except Exception as exc:
        friedman_rows.append(
            {
                "family": family,
                "n": int(aligned.shape[1]),
                "k": int(aligned.shape[0]),
                "methods": ",".join(str(name) for name in measure_names),
                "blocks": ",".join(aligned_block_names),
                "statistic": float("nan"),
                "p_value": float("nan"),
                "valid": False,
                "note": f"Friedman failed: {exc}",
            }
        )

    raw_rows: List[Dict[str, Any]] = []
    for i in range(len(measure_names)):
        for j in range(i + 1, len(measure_names)):
            delta = aligned[j, :] - aligned[i, :]
            row: Dict[str, Any] = {
                "family": family,
                "measure_a": str(measure_names[i]),
                "measure_b": str(measure_names[j]),
                "n": int(np.isfinite(delta).sum()),
                "valid": False,
                "statistic": float("nan"),
                "p_raw": float("nan"),
                "p_holm": float("nan"),
                "delta_mean": float(np.nanmean(delta)) if np.isfinite(delta).any() else float("nan"),
                "delta_median": float(np.nanmedian(delta)) if np.isfinite(delta).any() else float("nan"),
                "rank_biserial": _rank_biserial(delta),
                "note": "",
            }

            nonzero = delta[np.isfinite(delta) & (delta != 0)]
            if nonzero.size < 3:
                row["note"] = "Too few non-zero paired differences for Wilcoxon."
                raw_rows.append(row)
                continue

            try:
                stat = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", correction=False, method="auto")
                row["valid"] = True
                row["statistic"] = float(stat.statistic)
                row["p_raw"] = float(stat.pvalue)
            except Exception as exc:
                row["note"] = f"Wilcoxon failed: {exc}"

            raw_rows.append(row)

    valid_rows = [row for row in raw_rows if np.isfinite(row["p_raw"])]
    valid_rows.sort(key=lambda row: row["p_raw"])
    m_tests = len(valid_rows)
    holm_values: List[float] = []
    for rank, row in enumerate(valid_rows, start=1):
        holm_values.append(min(1.0, (m_tests - rank + 1) * float(row["p_raw"])))
    if holm_values:
        holm_values = np.maximum.accumulate(np.asarray(holm_values, dtype=np.float64)).tolist()
    for row, holm in zip(valid_rows, holm_values):
        row["p_holm"] = float(holm)

    return friedman_rows, raw_rows


__all__ = [
    "bootstrap_ci_mean",
    "bootstrap_ci_generalization",
    "friedman_wilcoxon_holm",
]
