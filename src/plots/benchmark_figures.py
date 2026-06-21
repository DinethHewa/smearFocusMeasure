"""Benchmark heatmaps and confidence-summary figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.plots.focus_curves import (
    apply_publication_format,
    require_matplotlib,
    save_figure_multi,
    plt,
)


def top_measure_names(rank_rows: Sequence[Dict[str, str]], *, top_k: int = 10) -> List[str]:
    ordered = sorted(rank_rows, key=lambda row: float(row["final_rank"]))
    return [row["measure_name"] for row in ordered[:top_k]]


def plot_single_measure_heatmap(
    *,
    rank_rows: Sequence[Dict[str, str]],
    dataset_metric_rows: Sequence[Dict[str, str]],
    dataset_order: Sequence[str],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    metric_name: str = "absolute_peak_localization_error",
    top_k: int = 10,
) -> Dict[str, Any]:
    require_matplotlib()
    top_names = top_measure_names(rank_rows, top_k=top_k)
    matrix = np.full((len(dataset_order), len(top_names)), np.nan, dtype=np.float64)
    for i, dataset_name in enumerate(dataset_order):
        for j, measure_name in enumerate(top_names):
            matches = [
                row for row in dataset_metric_rows
                if row["dataset_name"] == dataset_name and row["measure_name"] == measure_name
            ]
            if matches:
                matrix[i, j] = float(matches[0][metric_name])

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(top_names)))
    ax.set_xticklabels(top_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels(dataset_order)
    ax.set_title(spec.title)
    ax.set_xlabel("Measure")
    ax.set_ylabel("Dataset")
    fig.colorbar(image, ax=ax, label=metric_name)
    apply_publication_format(ax)

    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {
        "figure_key": spec.key,
        "files": written,
        "description": spec.description,
        "metric_used": metric_name,
    }


def plot_bootstrap_ci(
    *,
    rank_rows: Sequence[Dict[str, str]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    top_k: int = 10,
) -> Dict[str, Any]:
    require_matplotlib()
    top_rows = sorted(rank_rows, key=lambda row: float(row["final_rank"]))[:top_k]
    names = [row["measure_name"] for row in top_rows]
    means = np.asarray([float(row["overall_rank_mean"]) for row in top_rows], dtype=np.float64)
    ci_lows = np.asarray([float(row["bootstrap_ci_low"]) for row in top_rows], dtype=np.float64)
    ci_highs = np.asarray([float(row["bootstrap_ci_high"]) for row in top_rows], dtype=np.float64)

    y = np.arange(len(names))
    xerr = np.vstack([means - ci_lows, ci_highs - means])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(means, y, xerr=xerr, fmt="o", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Overall rank mean")
    ax.set_title(spec.title)
    ax.grid(True, axis="x", alpha=0.3)
    apply_publication_format(ax)

    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {
        "figure_key": spec.key,
        "files": written,
        "description": spec.description,
    }


__all__ = [
    "top_measure_names",
    "plot_single_measure_heatmap",
    "plot_bootstrap_ci",
]
