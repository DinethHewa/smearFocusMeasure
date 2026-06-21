# scripts/05_plot_single_measure_results.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paper_assets import get_figure_output_dir, get_figure_spec
from config.paths import (
    DATASET_ORDER,
    LOGS_DIR,
    RANK_STABILITY_RESULTS_FILE,
    SINGLE_EVAL_MAIN_DIR,
    SINGLE_EVAL_SUPP_DIR,
    SINGLE_TIMING_SUMMARY_FILE,
    ensure_output_dirs,
    get_single_norm_curve_file,
    get_source_label_file,
    get_surrogate_label_file,
)
from config.settings import (
    DEFAULT_MAIN_FIGURE_EXTENSIONS,
    DEFAULT_RUN_MODE,
    DEFAULT_SUPP_FIGURE_EXTENSIONS,
    FIGURE_DPI,
    PLOT_TOP_SINGLE_MEASURES,
    validate_all_settings,
)
from src.plots.benchmark_figures import (
    plot_bootstrap_ci,
    plot_single_measure_heatmap,
    top_measure_names,
)
from src.plots.focus_curves import (
    plot_dataset_specific_focus_curves,
    plot_representative_single_focus_curves,
    require_matplotlib,
)
from src.plots.runtime_figures import (
    plot_resolution_sensitivity,
    plot_runtime_scaling,
)
from src.utils.logging_utils import get_logger
from src.utils.validation import (
    load_csv_rows,
    save_json,
    validate_environment,
    validate_pipeline_prerequisites,
    write_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrected single-measure paper figures")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run full mode")
    return parser.parse_args()


def resolve_run_mode(args: argparse.Namespace) -> str:
    if args.smoke_test and args.full_run:
        raise ValueError("Use only one of --smoke-test or --full-run")
    if args.smoke_test:
        return "smoke"
    if args.full_run:
        return "full"
    return DEFAULT_RUN_MODE


def load_rank_rows() -> List[Dict[str, str]]:
    path = SINGLE_EVAL_SUPP_DIR / "all_single_rank_based.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing rank-based results: {path}")
    return load_csv_rows(path)


def load_dataset_metric_rows() -> List[Dict[str, str]]:
    path = SINGLE_EVAL_SUPP_DIR / "single_measure_dataset_level_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset-level metrics: {path}")
    return load_csv_rows(path)


def load_rank_stability_rows() -> List[Dict[str, str]]:
    if not RANK_STABILITY_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing publication rank-stability results: {RANK_STABILITY_RESULTS_FILE}. "
            "Run scripts/04_evaluate_single_measures.py first."
        )
    return load_csv_rows(RANK_STABILITY_RESULTS_FILE)


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    require_matplotlib()
    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=True)

    log_file = LOGS_DIR / f"plot_single_measure_results_{run_mode}.log"
    logger = get_logger("plot_single_measure_results", log_file=log_file)

    rank_rows = load_rank_rows()
    dataset_metric_rows = load_dataset_metric_rows()
    rank_stability_rows = load_rank_stability_rows()
    timing_rows = load_csv_rows(SINGLE_TIMING_SUMMARY_FILE) if SINGLE_TIMING_SUMMARY_FILE.exists() else []
    if not timing_rows:
        logger.warning("Timing summary not found or empty: %s", SINGLE_TIMING_SUMMARY_FILE)

    figure_records: List[Dict[str, Any]] = []
    curve_path_resolver = lambda dataset_name, measure_name: get_single_norm_curve_file(dataset_name, measure_name)

    def reference_label_resolver(dataset_name: str):
        source_path = get_source_label_file(dataset_name)
        if source_path.exists():
            return source_path, "source"
        surrogate_path = get_surrogate_label_file(dataset_name)
        if surrogate_path.exists():
            return surrogate_path, "surrogate"
        return None, None

    def load_reference_labels(dataset_name: str):
        label_path, provenance = reference_label_resolver(dataset_name)
        if label_path is None:
            return None, None
        labels = np.load(label_path, allow_pickle=False).astype(int).reshape(-1)
        return labels, provenance

    fig5_spec = get_figure_spec("Fig5_representative_single_focus_curves")
    logger.info("Generating %s", fig5_spec.key)
    figure_records.append(
        plot_representative_single_focus_curves(
            measure_names=PLOT_TOP_SINGLE_MEASURES,
            dataset_names=DATASET_ORDER,
            curve_path_resolver=curve_path_resolver,
            spec=fig5_spec,
            output_dir=get_figure_output_dir(fig5_spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
            reference_label_resolver=load_reference_labels,
            logger=logger,
        )
    )

    sfig1_spec = get_figure_spec("SFig1_dataset_specific_focus_curves")
    top_dataset_measures = top_measure_names(rank_rows, top_k=4)
    logger.info("Generating %s", sfig1_spec.key)
    figure_records.append(
        plot_dataset_specific_focus_curves(
            dataset_measure_map={dataset_name: top_dataset_measures for dataset_name in DATASET_ORDER},
            curve_path_resolver=curve_path_resolver,
            spec=sfig1_spec,
            output_dir=get_figure_output_dir(sfig1_spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
            reference_label_resolver=load_reference_labels,
            logger=logger,
        )
    )

    fig2_spec = get_figure_spec("Fig2_single_measure_heatmap")
    logger.info("Generating %s", fig2_spec.key)
    figure_records.append(
        plot_single_measure_heatmap(
            rank_rows=rank_rows,
            dataset_metric_rows=dataset_metric_rows,
            dataset_order=DATASET_ORDER,
            spec=fig2_spec,
            output_dir=get_figure_output_dir(fig2_spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
    )

    fig3_spec = get_figure_spec("Fig3_top_operator_bootstrap_ci")
    logger.info("Generating %s", fig3_spec.key)
    figure_records.append(
        plot_bootstrap_ci(
            rank_rows=rank_rows,
            spec=fig3_spec,
            output_dir=get_figure_output_dir(fig3_spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
    )

    fig4_spec = get_figure_spec("Fig4_resolution_sensitivity")
    logger.info("Generating %s", fig4_spec.key)
    figure_records.append(
        plot_resolution_sensitivity(
            stability_rows=rank_stability_rows,
            spec=fig4_spec,
            output_dir=get_figure_output_dir(fig4_spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
    )

    if timing_rows:
        fig10_spec = get_figure_spec("Fig10_runtime_scaling")
        logger.info("Generating %s", fig10_spec.key)
        figure_records.append(
            plot_runtime_scaling(
                timing_rows=timing_rows,
                rank_rows=rank_rows,
                spec=fig10_spec,
                output_dir=get_figure_output_dir(fig10_spec.output_group),
                extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
                figure_dpi=FIGURE_DPI,
            )
        )

    manifest_payload = {
        "stage": "plot_single_measure_results",
        "run_mode": run_mode,
        "figures_written": figure_records,
    }
    stage_manifest_path = SINGLE_EVAL_MAIN_DIR / "single_measure_figure_manifest.json"
    save_json(manifest_payload, stage_manifest_path)

    checkpoint_path = SINGLE_EVAL_MAIN_DIR / "plot_single_measure_results.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="plot_single_measure_results",
        status="complete",
        details=manifest_payload,
    )

    logger.info("Single-measure plotting stage complete")
    logger.info("Figure manifest written to %s", stage_manifest_path)
    logger.info("Checkpoint written to %s", checkpoint_path)


if __name__ == "__main__":
    main()
