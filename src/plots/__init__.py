"""Plotting helpers for corrected paper figures."""

from src.plots.benchmark_figures import (
    plot_bootstrap_ci,
    plot_single_measure_heatmap,
    top_measure_names,
)
from src.plots.focus_curves import (
    MATPLOTLIB_AVAILABLE,
    MATPLOTLIB_IMPORT_ERROR,
    apply_publication_format,
    load_curve_file,
    plot_dataset_specific_focus_curves,
    plot_representative_single_focus_curves,
    representative_curve_index,
    require_matplotlib,
    save_figure_multi,
)
from src.plots.runtime_figures import (
    plot_resolution_sensitivity,
    plot_runtime_scaling,
)

__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "MATPLOTLIB_IMPORT_ERROR",
    "require_matplotlib",
    "apply_publication_format",
    "save_figure_multi",
    "load_curve_file",
    "representative_curve_index",
    "plot_representative_single_focus_curves",
    "plot_dataset_specific_focus_curves",
    "top_measure_names",
    "plot_single_measure_heatmap",
    "plot_bootstrap_ci",
    "plot_resolution_sensitivity",
    "plot_runtime_scaling",
]
