# scripts/08_run_statistics_and_sensitivity.py

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paper_assets import get_figure_output_dir, get_figure_spec
from config.paths import (
    COMPOSITE_SUPP_DIR,
    LOGS_DIR,
    PAPER_TABLES_MAIN_CSV_DIR,
    PAPER_TABLES_SUPP_CSV_DIR,
    SINGLE_EVAL_SUPP_DIR,
    STATISTICS_DIR,
    SENSITIVITY_DIR,
    ensure_output_dirs,
)
from config.settings import (
    ALT_METRIC_WEIGHT_SCHEMES,
    AUTOFOCUS_METRICS,
    DEFAULT_MAIN_FIGURE_EXTENSIONS,
    DEFAULT_RUN_MODE,
    FIGURE_DPI,
    GENERALIZATION_ALPHA,
    USE_NEMENYI_POSTHOC,
    validate_all_settings,
)
from src.evaluation.aggregation import (
    align_metric_value,
    average_ranks,
    compute_value_based_summary,
)
from src.evaluation.statistics import bootstrap_ci_mean, friedman_wilcoxon_holm
from src.gp.terminal_selection import load_dataset_stack_counts
from src.plots.focus_curves import apply_publication_format, save_figure_multi
from src.utils.logging_utils import get_logger
from src.utils.validation import load_csv_rows, load_json, save_csv_rows, save_json, validate_environment, validate_pipeline_prerequisites, write_checkpoint

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = exc

try:
    from scipy.stats import studentized_range  # type: ignore

    SCIPY_AVAILABLE = True
except Exception:
    studentized_range = None  # type: ignore
    SCIPY_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected statistics and sensitivity analysis for single and composite outputs"
    )
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


def require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "This statistics stage requires matplotlib for CD figure export."
        ) from MATPLOTLIB_IMPORT_ERROR


def load_metric_raw(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric JSON: {path}")
    payload = load_json(path)
    return {
        str(dataset_name): {
            str(entity_name): {
                str(metric_name): float(metric_value)
                for metric_name, metric_value in metrics.items()
            }
            for entity_name, metrics in entities.items()
        }
        for dataset_name, entities in payload.items()
    }


def build_rank_matrix(
    dataset_metric_raw: Dict[str, Dict[str, Dict[str, float]]],
    metric_names: Sequence[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    entity_names = list(next(iter(dataset_metric_raw.values())).keys())
    rank_cells = {entity_name: [] for entity_name in entity_names}
    block_names: List[str] = []

    for dataset_name in dataset_metric_raw.keys():
        for metric_name in metric_names:
            aligned_values = [
                align_metric_value(metric_name, dataset_metric_raw[dataset_name][entity_name][metric_name])
                for entity_name in entity_names
            ]
            ranks = average_ranks(aligned_values, lower_better=True)
            for entity_name, rank_value in zip(entity_names, ranks):
                rank_cells[entity_name].append(float(rank_value))
            block_names.append(
                f"{dataset_name}" if len(metric_names) == 1 else f"{dataset_name}::{metric_name}"
            )

    matrix = np.asarray([rank_cells[entity_name] for entity_name in entity_names], dtype=np.float64)
    return matrix, entity_names, block_names


def average_rank_rows_from_matrix(rank_matrix: np.ndarray, entity_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows = [
        {
            "entity_name": str(entity_name),
            "average_rank": float(np.mean(rank_matrix[index, :])),
        }
        for index, entity_name in enumerate(entity_names)
    ]
    rows.sort(key=lambda row: float(row["average_rank"]))
    for display_order, row in enumerate(rows, start=1):
        row["display_order"] = int(display_order)
    return rows


def nemenyi_critical_difference(num_methods: int, num_blocks: int, *, alpha: float = 0.05) -> float:
    if not SCIPY_AVAILABLE or num_methods < 2 or num_blocks < 1:
        return float("nan")
    q_alpha = float(studentized_range.ppf(1.0 - alpha, num_methods, np.inf) / math.sqrt(2.0))
    return float(q_alpha * math.sqrt(num_methods * (num_methods + 1) / (6.0 * num_blocks)))


def plot_cd_diagram(
    *,
    average_rank_rows: List[Dict[str, Any]],
    cd_value: float,
    title: str,
    figure_key: str,
) -> Dict[str, Any]:
    require_matplotlib()
    spec = get_figure_spec(figure_key)
    output_dir = get_figure_output_dir(spec.output_group)

    names = [row["entity_name"] for row in average_rank_rows]
    avg_ranks = np.asarray([float(row["average_rank"]) for row in average_rank_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    y = np.arange(len(names))
    ax.scatter(avg_ranks, y, s=35)
    for x_val, y_val, name in zip(avg_ranks, y, names):
        ax.text(x_val + 0.03, y_val, name, va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    if np.isfinite(cd_value):
        bar_start = float(np.min(avg_ranks))
        bar_end = float(bar_start + cd_value)
        ybar = -0.8
        ax.plot([bar_start, bar_end], [ybar, ybar], linewidth=3)
        ax.plot([bar_start, bar_start], [ybar - 0.1, ybar + 0.1], linewidth=2)
        ax.plot([bar_end, bar_end], [ybar - 0.1, ybar + 0.1], linewidth=2)
        ax.text((bar_start + bar_end) / 2.0, ybar - 0.25, f"CD = {cd_value:.3f}", ha="center")

    apply_publication_format(ax)
    written = save_figure_multi(fig, output_dir / figure_key, DEFAULT_MAIN_FIGURE_EXTENSIONS, figure_dpi=FIGURE_DPI)
    return {
        "figure_key": figure_key,
        "files": written,
        "description": spec.description,
        "critical_difference": cd_value,
    }


def summarize_alpha_sensitivity(alpha_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_alpha: Dict[float, List[Dict[str, str]]] = {}
    for row in alpha_rows:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for alpha in sorted(by_alpha.keys()):
        ordered = sorted(by_alpha[alpha], key=lambda row: float(row["final_rank"]))
        summary_rows.append(
            {
                "alpha": float(alpha),
                "top1": ordered[0]["measure_name"] if ordered else "",
                "top3": " | ".join(row["measure_name"] for row in ordered[:3]),
                "top5": " | ".join(row["measure_name"] for row in ordered[:5]),
            }
        )
    return summary_rows


def summarize_metric_weight_sensitivity(
    dataset_metric_raw: Dict[str, Dict[str, Dict[str, float]]],
    dataset_stack_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for scheme_name, metric_weights in ALT_METRIC_WEIGHT_SCHEMES.items():
        full_rows = compute_value_based_summary(
            dataset_metric_raw=dataset_metric_raw,
            dataset_stack_counts=dataset_stack_counts,
            dataset_subset=list(dataset_metric_raw.keys()),
            weighting_mode="equal_dataset",
            alpha=GENERALIZATION_ALPHA,
            metric_weights=metric_weights,
        )
        save_csv_rows(full_rows, SENSITIVITY_DIR / f"metric_weight_scheme_{scheme_name}_full.csv")
        summary_rows.append(
            {
                "scheme_name": str(scheme_name),
                "top1": full_rows[0]["measure_name"] if full_rows else "",
                "top3": " | ".join(row["measure_name"] for row in full_rows[:3]),
                "top5": " | ".join(row["measure_name"] for row in full_rows[:5]),
            }
        )
    return summary_rows


def save_friedman_bundle(
    *,
    family: str,
    rank_matrix: np.ndarray,
    entity_names: Sequence[str],
    block_names: Sequence[str],
    output_prefix: str,
) -> Dict[str, Any]:
    friedman_rows, pairwise_rows = friedman_wilcoxon_holm(
        rank_matrix=rank_matrix,
        measure_names=entity_names,
        block_names=block_names,
        family=family,
    )
    average_rank_rows = average_rank_rows_from_matrix(rank_matrix, entity_names)
    num_methods = rank_matrix.shape[0]
    num_blocks = rank_matrix.shape[1]
    cd_value = nemenyi_critical_difference(num_methods=num_methods, num_blocks=num_blocks)

    save_csv_rows(average_rank_rows, STATISTICS_DIR / f"{output_prefix}_average_ranks.csv")
    save_csv_rows(friedman_rows, STATISTICS_DIR / f"{output_prefix}_friedman.csv")
    save_csv_rows(pairwise_rows, STATISTICS_DIR / f"{output_prefix}_pairwise_holm.csv")

    return {
        "friedman_rows": friedman_rows,
        "pairwise_rows": pairwise_rows,
        "average_rank_rows": average_rank_rows,
        "critical_difference": cd_value,
    }


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=True)

    log_file = LOGS_DIR / f"run_statistics_and_sensitivity_{run_mode}.log"
    logger = get_logger("run_statistics_and_sensitivity", log_file=log_file)

    dataset_metric_raw = load_metric_raw(SINGLE_EVAL_SUPP_DIR / "dataset_metric_raw.json")
    dataset_stack_counts = load_dataset_stack_counts()
    rank_rows = load_csv_rows(SINGLE_EVAL_SUPP_DIR / "all_single_rank_based.csv")
    alpha_rows = load_csv_rows(SINGLE_EVAL_SUPP_DIR / "alpha_sensitivity.csv")

    logger.info("Starting statistics and sensitivity stage")
    logger.info("Run mode: %s", run_mode)

    figure_records: List[Dict[str, Any]] = []

    overall_rank_matrix, overall_entities, overall_blocks = build_rank_matrix(
        dataset_metric_raw,
        AUTOFOCUS_METRICS,
    )
    overall_bundle = save_friedman_bundle(
        family="single_overall_rank",
        rank_matrix=overall_rank_matrix,
        entity_names=overall_entities,
        block_names=overall_blocks,
        output_prefix="overall",
    )

    accuracy_rank_matrix, accuracy_entities, accuracy_blocks = build_rank_matrix(
        dataset_metric_raw,
        ["absolute_peak_localization_error"],
    )
    accuracy_bundle = save_friedman_bundle(
        family="single_accuracy_rank",
        rank_matrix=accuracy_rank_matrix,
        entity_names=accuracy_entities,
        block_names=accuracy_blocks,
        output_prefix="accuracy",
    )

    if USE_NEMENYI_POSTHOC and SCIPY_AVAILABLE:
        figure_records.append(
            plot_cd_diagram(
                average_rank_rows=overall_bundle["average_rank_rows"][:15],
                cd_value=float(overall_bundle["critical_difference"]),
                title="Nemenyi critical-difference diagram for overall rank",
                figure_key="Fig7_nemenyi_cd_overall_rank",
            )
        )
        figure_records.append(
            plot_cd_diagram(
                average_rank_rows=accuracy_bundle["average_rank_rows"][:15],
                cd_value=float(accuracy_bundle["critical_difference"]),
                title="Nemenyi critical-difference diagram for localization accuracy",
                figure_key="Fig8_nemenyi_cd_accuracy_rank",
            )
        )

    bootstrap_top10 = []
    for row in sorted(rank_rows, key=lambda item: float(item["final_rank"]))[:10]:
        bootstrap_top10.append(
            {
                "measure_name": row["measure_name"],
                "overall_rank_mean": float(row["overall_rank_mean"]),
                "bootstrap_ci_low": float(row["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(row["bootstrap_ci_high"]),
                "rank_generalization_score": float(row["rank_generalization_score"]),
                "final_rank": int(float(row["final_rank"])),
            }
        )
    save_csv_rows(bootstrap_top10, PAPER_TABLES_MAIN_CSV_DIR / "Table_bootstrap_top10_single_rank_summary.csv")
    save_csv_rows(
        sorted(rank_rows, key=lambda item: float(item["final_rank"])),
        PAPER_TABLES_SUPP_CSV_DIR / "STable_bootstrap_all_single_rank_summary.csv",
    )

    alpha_summary_rows = summarize_alpha_sensitivity(alpha_rows)
    alpha_summary_csv = PAPER_TABLES_SUPP_CSV_DIR / "STable2_alpha_sensitivity.csv"
    save_csv_rows(alpha_summary_rows, alpha_summary_csv)

    metric_weight_summary_rows = summarize_metric_weight_sensitivity(dataset_metric_raw, dataset_stack_counts)
    metric_weight_summary_csv = PAPER_TABLES_SUPP_CSV_DIR / "STable3_metric_weight_sensitivity.csv"
    save_csv_rows(metric_weight_summary_rows, metric_weight_summary_csv)

    union_statistics: Dict[str, Any] = {}
    union_metric_path = COMPOSITE_SUPP_DIR / "union_metric_raw.json"
    if union_metric_path.exists():
        union_metric_raw = load_metric_raw(union_metric_path)
        union_overall_matrix, union_entities, union_blocks = build_rank_matrix(
            union_metric_raw,
            AUTOFOCUS_METRICS,
        )
        union_overall_bundle = save_friedman_bundle(
            family="union_overall_rank",
            rank_matrix=union_overall_matrix,
            entity_names=union_entities,
            block_names=union_blocks,
            output_prefix="union_overall",
        )
        union_accuracy_matrix, union_accuracy_entities, union_accuracy_blocks = build_rank_matrix(
            union_metric_raw,
            ["absolute_peak_localization_error"],
        )
        union_accuracy_bundle = save_friedman_bundle(
            family="union_accuracy_rank",
            rank_matrix=union_accuracy_matrix,
            entity_names=union_accuracy_entities,
            block_names=union_accuracy_blocks,
            output_prefix="union_accuracy",
        )
        union_statistics = {
            "union_overall_friedman": union_overall_bundle["friedman_rows"],
            "union_accuracy_friedman": union_accuracy_bundle["friedman_rows"],
            "union_overall_cd": float(union_overall_bundle["critical_difference"]),
            "union_accuracy_cd": float(union_accuracy_bundle["critical_difference"]),
        }

    statistics_summary = {
        "run_mode": run_mode,
        "overall_friedman": overall_bundle["friedman_rows"],
        "accuracy_friedman": accuracy_bundle["friedman_rows"],
        "overall_cd": float(overall_bundle["critical_difference"]),
        "accuracy_cd": float(accuracy_bundle["critical_difference"]),
        "alpha_summary_csv": str(alpha_summary_csv),
        "metric_weight_summary_csv": str(metric_weight_summary_csv),
        "figure_records": figure_records,
        "union_statistics": union_statistics,
    }
    save_json(statistics_summary, STATISTICS_DIR / "statistics_summary.json")

    friedman_summary_rows = [
        {
            "test": "overall_rank_friedman",
            "statistic": overall_bundle["friedman_rows"][0].get("statistic", float("nan")),
            "p_value": overall_bundle["friedman_rows"][0].get("p_value", float("nan")),
            "critical_difference": float(overall_bundle["critical_difference"]),
        },
        {
            "test": "accuracy_rank_friedman",
            "statistic": accuracy_bundle["friedman_rows"][0].get("statistic", float("nan")),
            "p_value": accuracy_bundle["friedman_rows"][0].get("p_value", float("nan")),
            "critical_difference": float(accuracy_bundle["critical_difference"]),
        },
    ]
    if union_statistics:
        friedman_summary_rows.extend(
            [
                {
                    "test": "union_overall_rank_friedman",
                    "statistic": union_statistics["union_overall_friedman"][0].get("statistic", float("nan")),
                    "p_value": union_statistics["union_overall_friedman"][0].get("p_value", float("nan")),
                    "critical_difference": float(union_statistics["union_overall_cd"]),
                },
                {
                    "test": "union_accuracy_rank_friedman",
                    "statistic": union_statistics["union_accuracy_friedman"][0].get("statistic", float("nan")),
                    "p_value": union_statistics["union_accuracy_friedman"][0].get("p_value", float("nan")),
                    "critical_difference": float(union_statistics["union_accuracy_cd"]),
                },
            ]
        )
    save_csv_rows(friedman_summary_rows, PAPER_TABLES_MAIN_CSV_DIR / "Table_friedman_and_cd_summary.csv")

    manifest_payload = {"stage": "run_statistics_and_sensitivity", "figures_written": figure_records}
    save_json(manifest_payload, STATISTICS_DIR / "statistics_figure_manifest.json")

    checkpoint_path = STATISTICS_DIR / "run_statistics_and_sensitivity.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="run_statistics_and_sensitivity",
        status="complete",
        details={
            "run_mode": run_mode,
            "statistics_summary_json": str(STATISTICS_DIR / "statistics_summary.json"),
            "alpha_summary_csv": str(alpha_summary_csv),
            "metric_weight_summary_csv": str(metric_weight_summary_csv),
            "num_figures_written": len(figure_records),
        },
    )

    logger.info("Statistics and sensitivity stage complete")
    logger.info("Statistics summary -> %s", STATISTICS_DIR / "statistics_summary.json")
    logger.info("Checkpoint -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
