"""Rank-based and value-based aggregation helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from config.settings import (
    AUTOFOCUS_METRICS,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_NUM_RESAMPLES,
    EPS,
    GENERALIZATION_ALPHA,
    METRIC_WEIGHTS,
)
from src.evaluation.autofocus_metrics import METRIC_DIRECTION
from src.evaluation.statistics import bootstrap_ci_mean


RankCells = Dict[str, Dict[str, List[float]]]


def average_ranks(values: Sequence[float], *, lower_better: bool = True) -> List[float]:
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals if lower_better else -vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=np.float64)

    idx = 0
    while idx < len(vals):
        jdx = idx
        current_val = vals[order[idx]]
        while jdx + 1 < len(vals):
            next_val = vals[order[jdx + 1]]
            if np.isclose(next_val, current_val):
                jdx += 1
            else:
                break
        avg_rank = (idx + jdx) / 2.0 + 1.0
        ranks[order[idx:jdx + 1]] = avg_rank
        idx = jdx + 1

    return ranks.tolist()


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if np.sum(w) <= 0:
        return float(np.mean(vals))
    return float(np.sum(vals * w) / (np.sum(w) + EPS))


def weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if np.sum(w) <= 0:
        return float(np.std(vals, ddof=0))
    mean_value = weighted_mean(vals, w)
    variance = np.sum(w * (vals - mean_value) ** 2) / (np.sum(w) + EPS)
    return float(np.sqrt(variance))


def align_metric_value(
    metric_name: str,
    raw_value: float,
    *,
    metric_direction: Mapping[str, bool] = METRIC_DIRECTION,
) -> float:
    if np.isnan(raw_value):
        return float(raw_value)
    lower_is_better = bool(metric_direction[metric_name])
    if lower_is_better:
        return float(raw_value)
    return float(1.0 / (float(raw_value) + EPS))


def minmax_normalize_across_measures(values_by_measure: Mapping[str, float]) -> Dict[str, float]:
    keys = list(values_by_measure.keys())
    vals = np.asarray([values_by_measure[key] for key in keys], dtype=np.float64)
    finite_mask = np.isfinite(vals)
    out: Dict[str, float] = {}

    if finite_mask.sum() == 0:
        for key in keys:
            out[key] = float("nan")
        return out

    finite_vals = vals[finite_mask]
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if vmax - vmin <= EPS:
        for key in keys:
            out[key] = 0.0 if np.isfinite(values_by_measure[key]) else float("nan")
        return out

    for key in keys:
        value = float(values_by_measure[key])
        out[key] = float((value - vmin) / (vmax - vmin + EPS)) if np.isfinite(value) else float("nan")
    return out


def compute_rank_based_summary(
    *,
    dataset_metric_raw: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_subset: Sequence[str],
    metric_names: Sequence[str] = AUTOFOCUS_METRICS,
    alpha: float = GENERALIZATION_ALPHA,
    n_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    conf_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> Tuple[List[Dict[str, Any]], RankCells]:
    first_dataset = next(iter(dataset_metric_raw.values()))
    measures = list(first_dataset.keys())
    rank_cells: RankCells = {measure_name: {"ranks": []} for measure_name in measures}

    for dataset_name in dataset_subset:
        for metric_name in metric_names:
            aligned_values = [
                align_metric_value(metric_name, dataset_metric_raw[dataset_name][measure_name][metric_name])
                for measure_name in measures
            ]
            ranks = average_ranks(aligned_values, lower_better=True)
            for measure_name, rank_value in zip(measures, ranks):
                rank_cells[measure_name]["ranks"].append(float(rank_value))

    rows: List[Dict[str, Any]] = []
    for measure_name in measures:
        ranks = rank_cells[measure_name]["ranks"]
        mean_rank = float(np.mean(ranks))
        std_rank = float(np.std(ranks, ddof=0))
        rank_generalization_score = float(alpha * mean_rank + (1.0 - alpha) * std_rank)
        ci_low, ci_high = bootstrap_ci_mean(
            ranks,
            n_resamples=n_resamples,
            conf_level=conf_level,
        )
        rows.append(
            {
                "measure_name": measure_name,
                "overall_rank_mean": mean_rank,
                "overall_rank_std": std_rank,
                "rank_generalization_score": rank_generalization_score,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )

    rows.sort(key=lambda row: row["rank_generalization_score"])
    for rank, row in enumerate(rows, start=1):
        row["final_rank"] = rank
    return rows, rank_cells


def compute_value_based_summary(
    *,
    dataset_metric_raw: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_stack_counts: Mapping[str, int],
    dataset_subset: Sequence[str],
    weighting_mode: str,
    alpha: float,
    metric_names: Sequence[str] = AUTOFOCUS_METRICS,
    metric_weights: Mapping[str, float] = METRIC_WEIGHTS,
) -> List[Dict[str, Any]]:
    first_dataset = next(iter(dataset_metric_raw.values()))
    measures = list(first_dataset.keys())
    dataset_scores: Dict[str, Dict[str, float]] = {dataset_name: {} for dataset_name in dataset_subset}

    for dataset_name in dataset_subset:
        normalized_aligned_by_metric: Dict[str, Dict[str, float]] = {}
        for metric_name in metric_names:
            aligned_vals = {
                measure_name: align_metric_value(
                    metric_name,
                    dataset_metric_raw[dataset_name][measure_name][metric_name],
                )
                for measure_name in measures
            }
            normalized_aligned_by_metric[metric_name] = minmax_normalize_across_measures(aligned_vals)

        for measure_name in measures:
            vals: List[float] = []
            weights: List[float] = []
            for metric_name in metric_names:
                value = normalized_aligned_by_metric[metric_name][measure_name]
                if np.isfinite(value):
                    vals.append(value)
                    weights.append(float(metric_weights[metric_name]))
            dataset_scores[dataset_name][measure_name] = weighted_mean(vals, weights) if vals else float("nan")

    rows: List[Dict[str, Any]] = []
    for measure_name in measures:
        dataset_values = [dataset_scores[dataset_name][measure_name] for dataset_name in dataset_subset]
        if weighting_mode == "equal_dataset":
            weights = [1.0 for _ in dataset_subset]
        elif weighting_mode == "per_stack":
            weights = [float(dataset_stack_counts[dataset_name]) for dataset_name in dataset_subset]
        else:
            raise ValueError(f"Unknown weighting mode: {weighting_mode}")

        weighted_mean_value = weighted_mean(dataset_values, weights)
        weighted_std_value = weighted_std(dataset_values, weights)
        generalization_score = float(alpha * weighted_mean_value + (1.0 - alpha) * weighted_std_value)

        rows.append(
            {
                "measure_name": measure_name,
                "weighted_mean": weighted_mean_value,
                "weighted_std": weighted_std_value,
                "generalization_score": generalization_score,
                "dataset_weighting_mode": weighting_mode,
                "alpha": alpha,
            }
        )

    rows.sort(key=lambda row: row["generalization_score"])
    for rank, row in enumerate(rows, start=1):
        row["final_rank"] = rank
    return rows


def rank_cell_matrix(rank_cells: RankCells, measure_order: Sequence[str]) -> np.ndarray:
    return np.asarray([rank_cells[measure_name]["ranks"] for measure_name in measure_order], dtype=np.float64)


__all__ = [
    "RankCells",
    "average_ranks",
    "weighted_mean",
    "weighted_std",
    "align_metric_value",
    "minmax_normalize_across_measures",
    "compute_rank_based_summary",
    "compute_value_based_summary",
    "rank_cell_matrix",
]
