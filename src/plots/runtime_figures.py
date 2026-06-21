"""Runtime and publication-grade rank-stability figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from src.plots.benchmark_figures import top_measure_names
from src.plots.focus_curves import (
    apply_publication_format,
    require_matplotlib,
    save_figure_multi,
    plt,
)


def plot_resolution_sensitivity(
    *,
    stability_rows: Sequence[Dict[str, str]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    top_k: int = 10,
) -> Dict[str, Any]:
    require_matplotlib()
    ordered = sorted(stability_rows, key=lambda row: float(row["native_value_final_rank"]))[:top_k]
    x = np.arange(2)
    xlabels = ["native subset", "1024 subset"]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for row in ordered:
        y = [
            float(row["native_value_final_rank"]),
            float(row["resolution_value_final_rank"]),
        ]
        ax.plot(x, y, marker="o", linewidth=2.0, label=row["measure_name"])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Evaluation regime")
    ax.set_ylabel("Value-based final rank")
    ax.set_title(spec.title)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    apply_publication_format(ax)

    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {
        "figure_key": spec.key,
        "files": written,
        "description": spec.description,
    }


def plot_runtime_scaling(
    *,
    timing_rows: Sequence[Dict[str, str]],
    rank_rows: Sequence[Dict[str, str]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    top_k: int = 10,
) -> Dict[str, Any]:
    require_matplotlib()
    top_names = top_measure_names(rank_rows, top_k=top_k)
    by_measure = {measure_name: [] for measure_name in top_names}

    for row in timing_rows:
        measure_name = row["measure_name"]
        if measure_name not in by_measure:
            continue
        value = row.get("native_avg_time_per_slice_sec", "")
        if value not in ("", None):
            try:
                by_measure[measure_name].append(float(value))
            except Exception:
                pass

    means = [float(np.mean(by_measure[measure_name])) if by_measure[measure_name] else np.nan for measure_name in top_names]

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(top_names))
    ax.bar(x, means)
    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=45, ha="right")
    ax.set_ylabel("Native average time per slice (s)")
    ax.set_title(spec.title)
    ax.grid(True, axis="y", alpha=0.3)
    apply_publication_format(ax)

    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {
        "figure_key": spec.key,
        "files": written,
        "description": spec.description,
    }


__all__ = [
    "plot_resolution_sensitivity",
    "plot_runtime_scaling",
]
