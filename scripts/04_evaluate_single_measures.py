# scripts/04_evaluate_single_measures.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATASET_ORDER,
    LABEL_USAGE_DISCLOSURE_FILE,
    LABEL_USAGE_DISCLOSURE_JSON,
    LABEL_SOURCE_MANIFEST_FILE,
    LOGS_DIR,
    RANK_STABILITY_CHECKPOINT_FILE,
    RANK_STABILITY_DETAIL_FILE,
    RANK_STABILITY_RESULTS_FILE,
    RANK_STABILITY_SUMMARY_FILE,
    SINGLE_EVAL_MAIN_DIR,
    SINGLE_EVAL_SUPP_DIR,
    SINGLE_MEASURE_FREEZE_MANIFEST_FILE,
    SINGLE_OPERATOR_MANIFEST_FILE,
    ensure_output_dirs,
    get_loo_label_file,
    get_single_norm_curve_file,
    get_single_raw_curve_file,
    get_single_timing_file,
    get_source_label_file,
    get_stack_file,
    get_surrogate_label_file,
)
from config.settings import (
    DEFAULT_RUN_MODE,
    GENERALIZATION_ALPHA,
    get_run_profile,
    validate_all_settings,
)
from src.evaluation.aggregation import (
    compute_rank_based_summary,
    compute_value_based_summary,
    rank_cell_matrix,
)
from src.evaluation.autofocus_metrics import compute_dataset_metrics_for_measure
from src.evaluation.publication import (
    build_publication_measure_subset,
    build_single_measure_freeze_manifest,
    run_rank_stability_study,
)
from src.evaluation.sensitivity import (
    build_label_split_summary,
    compute_alpha_sensitivity,
)
from src.evaluation.statistics import friedman_wilcoxon_holm
from src.measures.focus_measure_library import build_focus_measure_registry
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import (
    load_csv_rows,
    load_json,
    save_csv_rows,
    save_json,
    validate_environment,
    validate_pipeline_prerequisites,
    write_checkpoint,
)


PROGRESS_METADATA_KEYS = {
    "dataset_name",
    "measure_name",
    "label_source_used",
    "dataset_label_mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single measures using corrected labels and normalized curves"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run in full-run mode")
    parser.add_argument(
        "--measure-names",
        type=str,
        default=None,
        help="Optional comma-separated subset of measures to evaluate",
    )
    parser.add_argument(
        "--skip-rrmse",
        action="store_true",
        help="Skip additive-noise RRMSE computation",
    )
    return parser.parse_args()


def resolve_run_mode(args: argparse.Namespace) -> str:
    if args.smoke_test and args.full_run:
        raise ValueError("Use only one of --smoke-test or --full-run")
    if args.smoke_test:
        return "smoke"
    if args.full_run:
        return "full"
    return DEFAULT_RUN_MODE


def load_operator_manifest() -> List[Dict[str, Any]]:
    if not SINGLE_OPERATOR_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing operator manifest: {SINGLE_OPERATOR_MANIFEST_FILE}. "
            "Run scripts/03_run_single_measure_benchmark.py first."
        )
    return load_json(SINGLE_OPERATOR_MANIFEST_FILE)


def resolve_measure_subset(
    operator_manifest: List[Dict[str, Any]],
    measure_names_arg: str | None,
) -> List[Dict[str, Any]]:
    if not measure_names_arg:
        return operator_manifest

    requested = [item.strip() for item in measure_names_arg.split(",") if item.strip()]
    by_name = {row["measure_name"]: row for row in operator_manifest}
    missing = [item for item in requested if item not in by_name]
    if missing:
        raise KeyError(f"Requested measures not found in operator manifest: {missing}")
    return [by_name[item] for item in requested]


def load_label_manifest() -> Dict[str, Dict[str, Any]]:
    if not LABEL_SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing label manifest: {LABEL_SOURCE_MANIFEST_FILE}. "
            "Run scripts/02_build_reference_labels.py first."
        )
    return load_json(LABEL_SOURCE_MANIFEST_FILE)


def load_curve_file(path: Path) -> List[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(item, dtype=np.float64).reshape(-1) for item in arr]


def load_labels_for_measure(
    dataset_name: str,
    measure_name: str,
    label_manifest: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, str]:
    dataset_record = label_manifest[dataset_name]
    label_mode = dataset_record["source_label_mode"]

    if label_mode == "source":
        labels = np.load(get_source_label_file(dataset_name), allow_pickle=False).astype(int).reshape(-1)
        return labels, "source"

    loo_path = get_loo_label_file(dataset_name, measure_name)
    if loo_path.exists():
        labels = np.load(loo_path, allow_pickle=False).astype(int).reshape(-1)
        return labels, "leave_one_out_surrogate"

    labels = np.load(get_surrogate_label_file(dataset_name), allow_pickle=False).astype(int).reshape(-1)
    return labels, "surrogate"


def load_timing_value(dataset_name: str, measure_name: str) -> float:
    timing_path = get_single_timing_file(dataset_name, measure_name)
    if not timing_path.exists():
        return float("nan")

    payload = load_json(timing_path)
    value = payload.get("native_avg_time_per_slice_sec", None)
    return float(value) if value is not None else float("nan")


def _progress_paths(run_mode: str, skip_rrmse: bool) -> Tuple[Path, Path]:
    suffix = f"{run_mode}_{'skiprrmse' if skip_rrmse else 'rrmse'}"
    return (
        SINGLE_EVAL_SUPP_DIR / f"single_measure_dataset_level_metrics.progress_{suffix}.csv",
        SINGLE_EVAL_SUPP_DIR / f"single_measure_dataset_level_metrics.progress_{suffix}.json",
    )


def _progress_signature(
    *,
    run_mode: str,
    skip_rrmse: bool,
    operator_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "run_mode": run_mode,
        "skip_rrmse": bool(skip_rrmse),
        "dataset_order": list(DATASET_ORDER),
        "measure_names": [str(row["measure_name"]) for row in operator_manifest],
    }


def _row_to_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in row.items():
        if key in PROGRESS_METADATA_KEYS:
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def load_progress_rows(
    *,
    progress_csv: Path,
    progress_state_json: Path,
    signature: Dict[str, Any],
    logger,
) -> List[Dict[str, Any]]:
    if not progress_csv.exists() or not progress_state_json.exists():
        return []
    try:
        state = load_json(progress_state_json)
        if state.get("signature") != signature:
            logger.info("Ignoring stale Stage 04 progress because the run signature changed")
            return []
        rows = load_csv_rows(progress_csv)
    except Exception as exc:
        logger.warning("Could not load Stage 04 progress; starting stage from scratch: %s", exc)
        return []

    valid_datasets = set(DATASET_ORDER)
    valid_measures = set(signature["measure_names"])
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        dataset_name = str(row.get("dataset_name", ""))
        measure_name = str(row.get("measure_name", ""))
        if dataset_name in valid_datasets and measure_name in valid_measures:
            deduped[(dataset_name, measure_name)] = dict(row)
    logger.info("Loaded %d completed Stage 04 dataset/measure rows from progress", len(deduped))
    return list(deduped.values())


def write_progress(
    *,
    progress_csv: Path,
    progress_state_json: Path,
    checkpoint_path: Path,
    signature: Dict[str, Any],
    detailed_rows: List[Dict[str, Any]],
    total_expected_rows: int,
) -> None:
    completed_keys = sorted(
        f"{row['dataset_name']}::{row['measure_name']}"
        for row in detailed_rows
        if "dataset_name" in row and "measure_name" in row
    )
    save_csv_rows(detailed_rows, progress_csv)
    save_json(
        {
            "signature": signature,
            "status": "running",
            "completed_rows": len(completed_keys),
            "total_expected_rows": int(total_expected_rows),
            "completed_keys": completed_keys,
            "progress_csv": str(progress_csv),
        },
        progress_state_json,
    )
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="evaluate_single_measures",
        status="running",
        details={
            **signature,
            "completed_rows": len(completed_keys),
            "total_expected_rows": int(total_expected_rows),
            "progress_csv": str(progress_csv),
            "progress_state_json": str(progress_state_json),
        },
    )


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=True)
    set_global_seed(42)

    log_file = LOGS_DIR / f"evaluate_single_measures_{run_mode}.log"
    logger = get_logger("evaluate_single_measures", log_file=log_file)

    operator_manifest = resolve_measure_subset(load_operator_manifest(), args.measure_names)
    label_manifest = load_label_manifest()
    registry = build_focus_measure_registry()
    profile = get_run_profile(run_mode)

    logger.info("Starting corrected single-measure evaluation stage")
    logger.info("Run mode: %s", run_mode)
    logger.info("Run profile: %s", profile)
    logger.info("Measures to evaluate: %d", len(operator_manifest))

    stacks_by_dataset: Dict[str, np.ndarray] = {
        dataset_name: np.load(get_stack_file(dataset_name), allow_pickle=True)
        for dataset_name in DATASET_ORDER
    }
    dataset_stack_counts = {dataset_name: len(stacks) for dataset_name, stacks in stacks_by_dataset.items()}
    dataset_label_modes = {
        dataset_name: label_manifest[dataset_name]["source_label_mode"]
        for dataset_name in DATASET_ORDER
    }

    checkpoint_path = SINGLE_EVAL_MAIN_DIR / "evaluate_single_measures.checkpoint.json"
    progress_csv, progress_state_json = _progress_paths(run_mode, bool(args.skip_rrmse))
    progress_signature = _progress_signature(
        run_mode=run_mode,
        skip_rrmse=bool(args.skip_rrmse),
        operator_manifest=operator_manifest,
    )
    total_expected_rows = len(operator_manifest) * len(DATASET_ORDER)

    dataset_metric_raw: Dict[str, Dict[str, Dict[str, float]]] = {dataset_name: {} for dataset_name in DATASET_ORDER}
    detailed_rows: List[Dict[str, Any]] = load_progress_rows(
        progress_csv=progress_csv,
        progress_state_json=progress_state_json,
        signature=progress_signature,
        logger=logger,
    )
    completed_keys = set()
    for row in detailed_rows:
        dataset_name = str(row["dataset_name"])
        measure_name = str(row["measure_name"])
        dataset_metric_raw[dataset_name][measure_name] = _row_to_metrics(row)
        completed_keys.add((dataset_name, measure_name))

    for measure_entry in operator_manifest:
        measure_name = measure_entry["measure_name"]
        if measure_name not in registry:
            logger.warning("Skipping measure not found in registry: %s", measure_name)
            continue
        measure_func = registry[measure_name]["func"]

        for dataset_name in DATASET_ORDER:
            if (dataset_name, measure_name) in completed_keys:
                logger.info("[%s] skipping completed %s from Stage 04 progress", dataset_name, measure_name)
                continue

            norm_curve_path = get_single_norm_curve_file(dataset_name, measure_name)
            raw_curve_path = get_single_raw_curve_file(dataset_name, measure_name)
            if not norm_curve_path.exists():
                raise FileNotFoundError(
                    f"Missing normalized curves: {norm_curve_path}. "
                    "Run scripts/03_run_single_measure_benchmark.py first."
                )
            if not raw_curve_path.exists():
                raise FileNotFoundError(
                    f"Missing raw curves: {raw_curve_path}. "
                    "Run scripts/03_run_single_measure_benchmark.py first."
                )

            norm_curves = load_curve_file(norm_curve_path)
            labels, label_source_used = load_labels_for_measure(
                dataset_name=dataset_name,
                measure_name=measure_name,
                label_manifest=label_manifest,
            )
            timing_value = load_timing_value(dataset_name, measure_name)

            metrics_summary = compute_dataset_metrics_for_measure(
                dataset_name=dataset_name,
                measure_name=measure_name,
                measure_func=measure_func,
                norm_curves=norm_curves,
                labels=labels,
                timing_value=timing_value,
                stacks=stacks_by_dataset[dataset_name],
                run_mode=run_mode,
                skip_rrmse=args.skip_rrmse,
            )

            dataset_metric_raw[dataset_name][measure_name] = metrics_summary
            completed_keys.add((dataset_name, measure_name))
            detailed_rows.append(
                {
                    "dataset_name": dataset_name,
                    "measure_name": measure_name,
                    "label_source_used": label_source_used,
                    "dataset_label_mode": dataset_label_modes[dataset_name],
                    **metrics_summary,
                }
            )
            write_progress(
                progress_csv=progress_csv,
                progress_state_json=progress_state_json,
                checkpoint_path=checkpoint_path,
                signature=progress_signature,
                detailed_rows=detailed_rows,
                total_expected_rows=total_expected_rows,
            )
            logger.info("[%s] evaluated %s using label source: %s", dataset_name, measure_name, label_source_used)

    detailed_csv = SINGLE_EVAL_SUPP_DIR / "single_measure_dataset_level_metrics.csv"
    save_csv_rows(detailed_rows, detailed_csv)

    rank_rows_all, rank_cells_all = compute_rank_based_summary(
        dataset_metric_raw=dataset_metric_raw,
        dataset_subset=DATASET_ORDER,
        alpha=GENERALIZATION_ALPHA,
    )
    value_rows_equal = compute_value_based_summary(
        dataset_metric_raw=dataset_metric_raw,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=DATASET_ORDER,
        weighting_mode="equal_dataset",
        alpha=GENERALIZATION_ALPHA,
    )
    value_rows_stack = compute_value_based_summary(
        dataset_metric_raw=dataset_metric_raw,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=DATASET_ORDER,
        weighting_mode="per_stack",
        alpha=GENERALIZATION_ALPHA,
    )

    rank_top10_csv = SINGLE_EVAL_MAIN_DIR / "top10_single_rank_based.csv"
    rank_all_csv = SINGLE_EVAL_SUPP_DIR / "all_single_rank_based.csv"
    value_equal_top10_csv = SINGLE_EVAL_MAIN_DIR / "top10_single_value_based_equal_dataset.csv"
    value_equal_all_csv = SINGLE_EVAL_SUPP_DIR / "all_single_value_based_equal_dataset.csv"
    value_stack_all_csv = SINGLE_EVAL_SUPP_DIR / "all_single_value_based_per_stack.csv"

    save_csv_rows(rank_rows_all[:10], rank_top10_csv)
    save_csv_rows(rank_rows_all, rank_all_csv)
    save_csv_rows(value_rows_equal[:10], value_equal_top10_csv)
    save_csv_rows(value_rows_equal, value_equal_all_csv)
    save_csv_rows(value_rows_stack, value_stack_all_csv)

    split_summary = build_label_split_summary(
        dataset_metric_raw=dataset_metric_raw,
        dataset_stack_counts=dataset_stack_counts,
        dataset_label_modes=dataset_label_modes,
        alpha=GENERALIZATION_ALPHA,
    )
    save_json(split_summary, SINGLE_EVAL_SUPP_DIR / "label_split_summaries.json")

    alpha_sensitivity_rows = compute_alpha_sensitivity(
        dataset_metric_raw=dataset_metric_raw,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=DATASET_ORDER,
    )
    save_csv_rows(alpha_sensitivity_rows, SINGLE_EVAL_SUPP_DIR / "alpha_sensitivity.csv")

    measure_order = [row["measure_name"] for row in rank_rows_all]
    rank_matrix = rank_cell_matrix(rank_cells_all, measure_order)
    block_names = [f"{dataset_name}:{metric_name}" for dataset_name in DATASET_ORDER for metric_name in next(iter(dataset_metric_raw[dataset_name].values())).keys()]
    friedman_rows, pairwise_rows = friedman_wilcoxon_holm(
        rank_matrix,
        measure_order,
        block_names,
    )
    save_csv_rows(friedman_rows, SINGLE_EVAL_SUPP_DIR / "single_rank_based_friedman.csv")
    save_csv_rows(pairwise_rows, SINGLE_EVAL_SUPP_DIR / "single_rank_based_pairwise_holm.csv")

    save_json(dataset_metric_raw, SINGLE_EVAL_SUPP_DIR / "dataset_metric_raw.json")
    save_json(dataset_stack_counts, SINGLE_EVAL_SUPP_DIR / "dataset_stack_counts.json")
    save_json(dataset_label_modes, SINGLE_EVAL_SUPP_DIR / "dataset_label_modes.json")
    save_json(rank_cells_all, SINGLE_EVAL_SUPP_DIR / "rank_cells_all.json")

    label_usage_rows = sorted(
        [
            {
                "dataset_name": row["dataset_name"],
                "measure_name": row["measure_name"],
                "dataset_label_mode": row["dataset_label_mode"],
                "label_source_used": row["label_source_used"],
                "uses_leave_one_out_surrogate": str(row["label_source_used"] == "leave_one_out_surrogate").lower(),
                "source_labels_available": str(label_manifest[row["dataset_name"]]["source_label_mode"] == "source").lower(),
            }
            for row in detailed_rows
        ],
        key=lambda item: (str(item["dataset_name"]), str(item["measure_name"])),
    )
    label_usage_summary: Dict[str, Any] = {
        "run_mode": run_mode,
        "datasets": {
            dataset_name: {
                "dataset_label_mode": dataset_label_modes[dataset_name],
                "source_labels_available": bool(dataset_label_modes[dataset_name] == "source"),
                "measures_using_source_labels": sum(
                    1
                    for row in label_usage_rows
                    if row["dataset_name"] == dataset_name and row["label_source_used"] == "source"
                ),
                "measures_using_leave_one_out_surrogate": sum(
                    1
                    for row in label_usage_rows
                    if row["dataset_name"] == dataset_name and row["label_source_used"] == "leave_one_out_surrogate"
                ),
                "measures_using_surrogate": sum(
                    1
                    for row in label_usage_rows
                    if row["dataset_name"] == dataset_name and row["label_source_used"] == "surrogate"
                ),
            }
            for dataset_name in DATASET_ORDER
        },
        "rows": label_usage_rows,
    }
    save_csv_rows(label_usage_rows, LABEL_USAGE_DISCLOSURE_FILE)
    save_json(label_usage_summary, LABEL_USAGE_DISCLOSURE_JSON)

    publication_measure_subset = build_publication_measure_subset(
        rank_rows_all,
        value_rows_equal,
        top_k=4 if run_mode == "smoke" else 10,
    )
    rank_stability_payload = run_rank_stability_study(
        dataset_names=DATASET_ORDER,
        measure_names=publication_measure_subset,
        stacks_by_dataset=stacks_by_dataset,
        label_loader=lambda dataset_name, measure_name: load_labels_for_measure(
            dataset_name=dataset_name,
            measure_name=measure_name,
            label_manifest=label_manifest,
        ),
        registry=registry,
        run_mode=run_mode,
        skip_rrmse=bool(args.skip_rrmse or run_mode == "smoke"),
    )
    rank_stability_summary = dict(rank_stability_payload["summary"])
    rank_spearman = float(rank_stability_summary.get("value_rank_spearman", float("nan")))
    rank_shift = float(rank_stability_summary.get("mean_absolute_value_rank_shift", float("nan")))
    top_overlap = int(rank_stability_summary.get("top_k_overlap_at_5", 0))
    overlap_threshold = min(3, max(1, len(publication_measure_subset)))
    rank_stability_passed = bool(
        top_overlap >= overlap_threshold
        and (not np.isfinite(rank_spearman) or rank_spearman >= 0.30)
        and (not np.isfinite(rank_shift) or rank_shift <= max(5.0, len(publication_measure_subset) / 2.0))
    )
    rank_stability_summary.update(
        {
            "passed": rank_stability_passed,
            "pass_thresholds": {
                "min_top_k_overlap_at_5": overlap_threshold,
                "min_value_rank_spearman": 0.30,
                "max_mean_absolute_value_rank_shift": max(5.0, len(publication_measure_subset) / 2.0),
            },
        }
    )
    save_csv_rows(rank_stability_payload["summary_rows"], RANK_STABILITY_RESULTS_FILE)
    save_csv_rows(rank_stability_payload["detail_rows"], RANK_STABILITY_DETAIL_FILE)
    save_json(rank_stability_summary, RANK_STABILITY_SUMMARY_FILE)
    write_checkpoint(
        checkpoint_path=RANK_STABILITY_CHECKPOINT_FILE,
        stage="rank_stability_validation",
        status="complete" if rank_stability_passed else "warning",
        details={
            "run_mode": run_mode,
            "summary_json": str(RANK_STABILITY_SUMMARY_FILE),
            "summary_csv": str(RANK_STABILITY_RESULTS_FILE),
            "detail_csv": str(RANK_STABILITY_DETAIL_FILE),
            "passed": rank_stability_passed,
            "measures_evaluated": publication_measure_subset,
        },
    )

    freeze_manifest = build_single_measure_freeze_manifest(
        run_mode=run_mode,
        file_paths={
            "rank_top10_csv": rank_top10_csv,
            "rank_all_csv": rank_all_csv,
            "value_equal_top10_csv": value_equal_top10_csv,
            "value_equal_all_csv": value_equal_all_csv,
            "value_per_stack_all_csv": value_stack_all_csv,
            "dataset_metric_raw_json": SINGLE_EVAL_SUPP_DIR / "dataset_metric_raw.json",
            "dataset_level_metrics_csv": detailed_csv,
            "label_usage_disclosure_csv": LABEL_USAGE_DISCLOSURE_FILE,
            "label_usage_disclosure_json": LABEL_USAGE_DISCLOSURE_JSON,
            "rank_stability_summary_json": RANK_STABILITY_SUMMARY_FILE,
            "rank_stability_summary_csv": RANK_STABILITY_RESULTS_FILE,
            "rank_stability_detail_csv": RANK_STABILITY_DETAIL_FILE,
            "label_source_manifest_json": LABEL_SOURCE_MANIFEST_FILE,
        },
        extra_details={
            "publication_measure_subset": publication_measure_subset,
            "rank_stability_passed": rank_stability_passed,
            "rank_stability_checkpoint": str(RANK_STABILITY_CHECKPOINT_FILE),
        },
    )
    save_json(freeze_manifest, SINGLE_MEASURE_FREEZE_MANIFEST_FILE)

    save_json(
        {
            "signature": progress_signature,
            "status": "complete",
            "completed_rows": len(completed_keys),
            "total_expected_rows": int(total_expected_rows),
            "progress_csv": str(progress_csv),
        },
        progress_state_json,
    )
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="evaluate_single_measures",
        status="complete",
        details={
            "run_mode": run_mode,
            "measures_evaluated": len(operator_manifest),
            "detailed_csv": str(detailed_csv),
            "rank_top10_csv": str(rank_top10_csv),
            "value_equal_top10_csv": str(value_equal_top10_csv),
            "label_usage_disclosure_csv": str(LABEL_USAGE_DISCLOSURE_FILE),
            "rank_stability_summary_json": str(RANK_STABILITY_SUMMARY_FILE),
            "freeze_manifest_json": str(SINGLE_MEASURE_FREEZE_MANIFEST_FILE),
            "skip_rrmse": args.skip_rrmse,
            "progress_csv": str(progress_csv),
        },
    )

    logger.info("Corrected single-measure evaluation stage complete")
    logger.info("Main outputs:")
    logger.info(" - %s", rank_top10_csv)
    logger.info(" - %s", value_equal_top10_csv)
    logger.info("Checkpoint written to %s", checkpoint_path)


if __name__ == "__main__":
    main()
