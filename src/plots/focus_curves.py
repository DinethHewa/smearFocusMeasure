"""Publication-oriented focus-curve figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = exc


CurvePathResolver = Callable[[str, str], Path]
ReferenceLabelResolver = Callable[[str], Tuple[Optional[np.ndarray], Optional[str]]]


def require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "This plotting module requires matplotlib. Install it in the environment used for the paper pipeline."
        ) from MATPLOTLIB_IMPORT_ERROR


def apply_publication_format(ax) -> None:
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.title.set_size(12)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(9)
    if not ax.images:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.figure.tight_layout()


def save_figure_multi(fig, output_base: Path, extensions: Sequence[str], *, figure_dpi: int) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for extension in extensions:
        out_path = output_base.with_suffix(extension)
        if extension.lower() == ".png":
            fig.savefig(out_path, dpi=figure_dpi, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        written.append(str(out_path))
    plt.close(fig)
    return written


def load_curve_file(path: Path) -> list[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(item, dtype=np.float64).reshape(-1) for item in arr]


def representative_curve_index(curves: Sequence[np.ndarray]) -> int:
    if not curves:
        raise ValueError("No curves provided")
    peak_indices = [int(np.argmax(curve)) for curve in curves]
    median_peak = float(np.median(peak_indices))
    return int(min(range(len(curves)), key=lambda idx: abs(peak_indices[idx] - median_peak)))


def plot_representative_single_focus_curves(
    *,
    measure_names: Sequence[str],
    dataset_names: Sequence[str],
    curve_path_resolver: CurvePathResolver,
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    reference_label_resolver: ReferenceLabelResolver | None = None,
    logger=None,
) -> Dict[str, Any]:
    require_matplotlib()
    fig, axes = plt.subplots(
        nrows=len(dataset_names),
        ncols=1,
        figsize=(8.0, 3.2 * len(dataset_names)),
        squeeze=False,
    )

    for row_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, 0]
        plotted_any = False
        for measure_name in measure_names:
            curve_path = curve_path_resolver(dataset_name, measure_name)
            if not curve_path.exists():
                if logger is not None:
                    logger.warning("[%s] missing normalized curve file for %s", dataset_name, measure_name)
                continue
            curves = load_curve_file(curve_path)
            if not curves:
                continue
            curve = curves[representative_curve_index(curves)]
            ax.plot(np.arange(len(curve)), curve, linewidth=2.0, label=measure_name)
            plotted_any = True

        if plotted_any and reference_label_resolver is not None:
            labels, provenance = reference_label_resolver(str(dataset_name))
            if labels is not None:
                curve_path = curve_path_resolver(dataset_name, str(measure_names[0]))
                curves = load_curve_file(curve_path) if curve_path.exists() else []
                if curves:
                    rep_idx = representative_curve_index(curves)
                    if rep_idx < len(labels):
                        label_idx = int(labels[rep_idx])
                        line_label = "Source best-focus" if provenance == "source" else "Reference focus"
                        linestyle = "-" if provenance == "source" else "--"
                        color = "black" if provenance == "source" else "gray"
                        ax.axvline(label_idx, color=color, linestyle=linestyle, linewidth=1.4, label=line_label)

        ax.set_title(str(dataset_name))
        ax.set_xlabel("Slice index")
        ax.set_ylabel("Normalized focus score")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=8)
        apply_publication_format(ax)

    fig.suptitle(spec.title, fontsize=12)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {
        "figure_key": spec.key,
        "files": written,
        "description": spec.description,
    }


def plot_dataset_specific_focus_curves(
    *,
    dataset_measure_map: Mapping[str, Sequence[str]],
    curve_path_resolver: CurvePathResolver,
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    reference_label_resolver: ReferenceLabelResolver | None = None,
    logger=None,
) -> Dict[str, Any]:
    require_matplotlib()
    dataset_names = list(dataset_measure_map.keys())
    fig, axes = plt.subplots(
        nrows=len(dataset_names),
        ncols=1,
        figsize=(8.5, 3.4 * len(dataset_names)),
        squeeze=False,
    )

    for row_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, 0]
        plotted_any = False
        for measure_name in dataset_measure_map[dataset_name]:
            curve_path = curve_path_resolver(dataset_name, measure_name)
            if not curve_path.exists():
                if logger is not None:
                    logger.warning("[%s] missing normalized curve file for %s", dataset_name, measure_name)
                continue
            curves = load_curve_file(curve_path)
            if not curves:
                continue
            curve = curves[representative_curve_index(curves)]
            ax.plot(np.arange(len(curve)), curve, linewidth=1.8, label=measure_name)
            plotted_any = True

        if plotted_any and reference_label_resolver is not None:
            labels, provenance = reference_label_resolver(str(dataset_name))
            primary_measure = next(iter(dataset_measure_map[dataset_name]), None)
            if labels is not None and primary_measure is not None:
                curve_path = curve_path_resolver(dataset_name, str(primary_measure))
                curves = load_curve_file(curve_path) if curve_path.exists() else []
                if curves:
                    rep_idx = representative_curve_index(curves)
                    if rep_idx < len(labels):
                        label_idx = int(labels[rep_idx])
                        line_label = "Source best-focus" if provenance == "source" else "Reference focus"
                        linestyle = "-" if provenance == "source" else "--"
                        color = "black" if provenance == "source" else "gray"
                        ax.axvline(label_idx, color=color, linestyle=linestyle, linewidth=1.2, label=line_label)

        ax.set_title(f"{dataset_name} representative normalized curves")
        ax.set_xlabel("Slice index")
        ax.set_ylabel("Normalized focus score")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=8, ncols=2)
        apply_publication_format(ax)

    fig.suptitle(spec.title, fontsize=12)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {
        "figure_key": spec.key,
        "files": written,
        "description": spec.description,
    }


__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "MATPLOTLIB_IMPORT_ERROR",
    "CurvePathResolver",
    "ReferenceLabelResolver",
    "require_matplotlib",
    "apply_publication_format",
    "save_figure_multi",
    "load_curve_file",
    "representative_curve_index",
    "plot_representative_single_focus_curves",
    "plot_dataset_specific_focus_curves",
]
