# scripts/07_evaluate_composites.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    COMPOSITE_FIGURE_MANIFEST_FILE,
    COMPOSITE_MAIN_DIR,
    COMPOSITE_METRIC_PROFILE_FILE,
    COMPOSITE_SUPP_DIR,
    DATASET_ORDER,
    GP_DEDUP_DIR,
    GP_SUMMARIES_DIR,
    LOGS_DIR,
    PAPER_TABLES_MAIN_CSV_DIR,
    PAPER_TABLES_SUPP_CSV_DIR,
    SINGLE_EVAL_SUPP_DIR,
    ensure_output_dirs,
    get_stack_file,
)
from config.settings import AUTOFOCUS_METRICS, DEFAULT_RUN_MODE, GENERALIZATION_ALPHA, validate_all_settings
from src.evaluation.aggregation import compute_rank_based_summary, compute_value_based_summary
from src.gp.baselines import (
    assign_composite_ids,
    build_best_composite_vs_best_single_rows,
    build_within_admissible_spread_rows,
)
from src.gp.deap_search import (
    compile_expression,
    evaluate_expression_raw_metrics,
    load_composite_labels,
    load_terminal_curves_for_dataset,
    load_timing_summary_map,
    require_deap,
)
from src.gp.terminal_selection import load_dataset_stack_counts
from src.measures.focus_measure_library import build_focus_measure_registry
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import load_csv_rows, load_json, save_csv_rows, save_json, validate_environment, validate_pipeline_prerequisites, write_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate corrected composite expressions under the same metric framework as singles"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run full mode")
    parser.add_argument(
        "--top-k-composites",
        type=int,
        default=10,
        help="Number of top composite rows to export to the main paper table",
    )
    parser.add_argument(
        "--max-eval-candidates",
        type=int,
        default=25,
        help="Maximum number of deduplicated composite candidates to evaluate",
    )
    parser.add_argument(
        "--skip-rrmse",
        action="store_true",
        help="Skip additive-noise RRMSE computation for composites",
    )
    parser.add_argument(
        "--full-rrmse-cap",
        type=int,
        default=100,
        help="Maximum stacks per dataset used for full-run composite RRMSE",
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


def load_deduplicated_candidates() -> List[Dict[str, Any]]:
    path = GP_DEDUP_DIR / "deduplicated_best_expressions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing deduplicated composite shortlist: {path}. "
            "Run scripts/06_run_composite_gp_lodo.py first."
        )
    candidates = list(load_json(path))

    final_path = GP_SUMMARIES_DIR / "final_composite_expression.json"
    if final_path.exists():
        final_result = load_json(final_path)
        final_expression = str(final_result.get("best_expression", ""))
        if final_expression:
            final_candidate = {
                "result_type": "final_refit",
                "best_expression": final_expression,
                "terminals": list(final_result.get("terminals", [])),
                "heldout_score": float("nan"),
                "source": str(final_result.get("source", "all_dataset_refit_after_lodo_validation")),
                "seed": int(final_result.get("seed", -1)),
                "best_training_objective": float(final_result.get("best_training_objective", float("nan"))),
                "best_all_dataset_score": float(final_result.get("best_all_dataset_score", float("nan"))),
                "best_complexity": float(final_result.get("best_complexity", float("nan"))),
                "num_nodes": int(final_result.get("num_nodes", 0)),
                "tree_height": int(final_result.get("tree_height", 0)),
            }
            candidates = [
                row for row in candidates
                if str(row.get("best_expression", "")) != final_expression
            ]
            candidates.insert(0, final_candidate)
    return candidates


def load_single_metric_reference() -> Dict[str, Dict[str, Dict[str, float]]]:
    path = SINGLE_EVAL_SUPP_DIR / "dataset_metric_raw.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing corrected single-measure metric reference: {path}. "
            "Run scripts/04_evaluate_single_measures.py first."
        )
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


def rename_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    renamed: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["entity_name"] = str(new_row.pop("measure_name"))
        renamed.append(new_row)
    return renamed


def load_stacks_for_rrmse(dataset_name: str):
    path = get_stack_file(dataset_name)
    try:
        return np.load(path, allow_pickle=True, mmap_mode="r")
    except Exception:
        return np.load(path, allow_pickle=True)


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    require_deap()
    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=True)
    set_global_seed(42)

    log_file = LOGS_DIR / f"evaluate_composites_{run_mode}.log"
    logger = get_logger("evaluate_composites", log_file=log_file)

    candidates = assign_composite_ids(load_deduplicated_candidates())
    if not candidates:
        raise RuntimeError("No deduplicated composite candidates were found")
    candidates = candidates[: args.max_eval_candidates]

    timing_map = load_timing_summary_map()
    measure_registry = build_focus_measure_registry()
    dataset_stack_counts = load_dataset_stack_counts()
    single_metric_reference = load_single_metric_reference()

    logger.info("Starting corrected composite evaluation stage")
    logger.info("Run mode: %s", run_mode)
    logger.info("Composite candidates to evaluate: %d", len(candidates))
    logger.info(
        "Composite RRMSE: skip=%s full_cap=%d",
        bool(args.skip_rrmse),
        int(args.full_rrmse_cap),
    )

    composite_dataset_rows: List[Dict[str, Any]] = []
    composite_metric_raw_by_dataset: Dict[str, Dict[str, Dict[str, float]]] = {
        str(dataset_name): {} for dataset_name in DATASET_ORDER
    }
    partial_dataset_csv = COMPOSITE_SUPP_DIR / "composite_dataset_level_metrics.partial.csv"
    partial_metric_json = COMPOSITE_SUPP_DIR / "composite_metric_raw_by_dataset.partial.json"

    for candidate_index, candidate in enumerate(candidates, start=1):
        composite_id = str(candidate["composite_id"])
        expression = str(candidate["best_expression"])
        terminal_names = list(candidate["terminals"])
        func = compile_expression(expression, terminal_names)
        logger.info(
            "Evaluating %s (%d/%d) -> %s",
            composite_id,
            candidate_index,
            len(candidates),
            expression,
        )

        for dataset_index, dataset_name in enumerate(DATASET_ORDER, start=1):
            logger.info(
                "[%s] dataset %s (%d/%d): loading curves%s",
                composite_id,
                dataset_name,
                dataset_index,
                len(DATASET_ORDER),
                "" if args.skip_rrmse else " and memory-mapped stacks",
            )
            labels, label_source_used = load_composite_labels(dataset_name)
            terminal_curves = load_terminal_curves_for_dataset(dataset_name, terminal_names)
            stacks = None if args.skip_rrmse else load_stacks_for_rrmse(dataset_name)
            logger.info("[%s] dataset %s: evaluating composite metrics", composite_id, dataset_name)
            eval_result = evaluate_expression_raw_metrics(
                func=func,
                terminal_names=terminal_names,
                terminal_curves=terminal_curves,
                labels=labels,
                dataset_name=dataset_name,
                timing_map=timing_map,
                measure_registry=measure_registry,
                stacks=stacks,
                skip_rrmse=args.skip_rrmse,
                run_mode=run_mode,
                metric_names=AUTOFOCUS_METRICS,
                full_rrmse_cap=max(0, int(args.full_rrmse_cap)),
            )
            logger.info(
                "[%s] dataset %s: complete absolute_peak_error=%s rrmse=%s",
                composite_id,
                dataset_name,
                eval_result["raw_metrics"].get("absolute_peak_localization_error", ""),
                eval_result["raw_metrics"].get("rrmse_under_additive_noise", ""),
            )

            composite_metric_raw_by_dataset[dataset_name][composite_id] = dict(eval_result["raw_metrics"])
            composite_dataset_rows.append(
                {
                    "dataset_name": dataset_name,
                    "composite_id": composite_id,
                    "expression": expression,
                    "terminals": " | ".join(terminal_names),
                    "label_source_used": label_source_used,
                    **eval_result["raw_metrics"],
                }
            )
            save_csv_rows(composite_dataset_rows, partial_dataset_csv)
            save_json(composite_metric_raw_by_dataset, partial_metric_json)

    composite_dataset_csv = COMPOSITE_SUPP_DIR / "composite_dataset_level_metrics.csv"
    composite_metric_json = COMPOSITE_SUPP_DIR / "composite_metric_raw_by_dataset.json"
    save_csv_rows(composite_dataset_rows, composite_dataset_csv)
    save_json(composite_metric_raw_by_dataset, composite_metric_json)

    union_metric_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name in DATASET_ORDER:
        union_metric_raw[dataset_name] = {}
        for single_name, metrics in single_metric_reference[dataset_name].items():
            union_metric_raw[dataset_name][single_name] = dict(metrics)
        for composite_id, metrics in composite_metric_raw_by_dataset[dataset_name].items():
            union_metric_raw[dataset_name][composite_id] = dict(metrics)

    common_value_rows = rename_summary_rows(
        compute_value_based_summary(
            dataset_metric_raw=union_metric_raw,
            dataset_stack_counts=dataset_stack_counts,
            dataset_subset=DATASET_ORDER,
            weighting_mode="equal_dataset",
            alpha=GENERALIZATION_ALPHA,
        )
    )
    common_rank_rows, _rank_cells = compute_rank_based_summary(
        dataset_metric_raw=union_metric_raw,
        dataset_subset=DATASET_ORDER,
        alpha=GENERALIZATION_ALPHA,
    )
    common_rank_rows = rename_summary_rows(common_rank_rows)

    common_value_by_entity = {row["entity_name"]: row for row in common_value_rows}
    common_rank_by_entity = {row["entity_name"]: row for row in common_rank_rows}
    single_entities = list(next(iter(single_metric_reference.values())).keys())

    composite_summary_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        composite_id = str(candidate["composite_id"])
        composite_summary_rows.append(
            {
                "composite_id": composite_id,
                "expression": str(candidate["best_expression"]),
                "terminals": " | ".join(candidate["terminals"]),
                "heldout_score_from_lodo_stage": float(candidate.get("heldout_score", float("nan"))),
                "common_value_final_rank": int(common_value_by_entity[composite_id]["final_rank"]),
                "common_value_generalization_score": float(common_value_by_entity[composite_id]["generalization_score"]),
                "common_value_weighted_mean": float(common_value_by_entity[composite_id]["weighted_mean"]),
                "common_value_weighted_std": float(common_value_by_entity[composite_id]["weighted_std"]),
                "common_rank_final_rank": int(common_rank_by_entity[composite_id]["final_rank"]),
                "common_rank_generalization_score": float(common_rank_by_entity[composite_id]["rank_generalization_score"]),
                "common_rank_mean": float(common_rank_by_entity[composite_id]["overall_rank_mean"]),
                "common_rank_std": float(common_rank_by_entity[composite_id]["overall_rank_std"]),
            }
        )
    composite_summary_rows.sort(key=lambda row: int(row["common_value_final_rank"]))

    top_metric_profile_rows: List[Dict[str, Any]] = []
    for summary_row in composite_summary_rows[: args.top_k_composites]:
        composite_id = str(summary_row["composite_id"])
        metric_rows = [
            row for row in composite_dataset_rows
            if str(row["composite_id"]) == composite_id
        ]
        profile_row: Dict[str, Any] = {
            "composite_id": composite_id,
            "expression": str(summary_row["expression"]),
            "heldout_score_from_lodo_stage": float(summary_row["heldout_score_from_lodo_stage"]),
            "common_value_final_rank": int(summary_row["common_value_final_rank"]),
            "common_rank_final_rank": int(summary_row["common_rank_final_rank"]),
        }
        for metric_name in AUTOFOCUS_METRICS:
            metric_values = [
                float(row[metric_name])
                for row in metric_rows
                if row.get(metric_name, "") not in ("", None)
            ]
            profile_row[f"{metric_name}_mean"] = (
                float(np.mean(metric_values)) if metric_values else float("nan")
            )
            profile_row[f"{metric_name}_std"] = (
                float(np.std(metric_values, ddof=0)) if metric_values else float("nan")
            )
        top_metric_profile_rows.append(profile_row)

    table7_csv = PAPER_TABLES_MAIN_CSV_DIR / "Table7_top10_composites_common_scoring.csv"
    comparison_csv = COMPOSITE_MAIN_DIR / "best_composite_vs_best_single.csv"
    within_spread_csv = COMPOSITE_SUPP_DIR / "within_admissible_spread.csv"
    composite_summary_csv = COMPOSITE_SUPP_DIR / "all_composites_common_scoring.csv"
    composite_rank_csv = COMPOSITE_SUPP_DIR / "all_composites_common_rank_scoring.csv"
    union_value_csv = COMPOSITE_SUPP_DIR / "union_singles_and_composites_common_value.csv"
    union_rank_csv = COMPOSITE_SUPP_DIR / "union_singles_and_composites_common_rank.csv"
    union_metric_json = COMPOSITE_SUPP_DIR / "union_metric_raw.json"

    save_csv_rows(composite_summary_rows[: args.top_k_composites], table7_csv)
    save_csv_rows(composite_summary_rows, composite_summary_csv)
    save_csv_rows(
        [row for row in common_rank_rows if row["entity_name"] not in set(single_entities)],
        composite_rank_csv,
    )
    save_csv_rows(common_value_rows, union_value_csv)
    save_csv_rows(common_rank_rows, union_rank_csv)
    save_json(union_metric_raw, union_metric_json)
    save_csv_rows(top_metric_profile_rows, COMPOSITE_METRIC_PROFILE_FILE)

    single_rank_rows = load_csv_rows(SINGLE_EVAL_SUPP_DIR / "all_single_rank_based.csv")
    single_value_rows = load_csv_rows(SINGLE_EVAL_SUPP_DIR / "all_single_value_based_equal_dataset.csv")
    comparison_rows = build_best_composite_vs_best_single_rows(
        composite_summary_rows=composite_summary_rows,
        common_value_rows=common_value_rows,
        single_entities=single_entities,
        single_rank_rows=single_rank_rows,
        single_value_rows=single_value_rows,
    )
    save_csv_rows(comparison_rows, comparison_csv)

    within_spread_rows = build_within_admissible_spread_rows(composite_summary_rows)
    save_csv_rows(within_spread_rows, within_spread_csv)
    save_json(
        {
            "stage": "evaluate_composites",
            "run_mode": run_mode,
            "best_composite_id": composite_summary_rows[0]["composite_id"] if composite_summary_rows else "",
            "best_composite_expression": composite_summary_rows[0]["expression"] if composite_summary_rows else "",
            "table7_csv": str(table7_csv),
            "composite_metric_profile_csv": str(COMPOSITE_METRIC_PROFILE_FILE),
            "foldwise_gp_summary_csv": str(PAPER_TABLES_SUPP_CSV_DIR / "STable4_gp_foldwise_results.csv"),
        },
        COMPOSITE_FIGURE_MANIFEST_FILE,
    )

    checkpoint_path = COMPOSITE_MAIN_DIR / "evaluate_composites.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="evaluate_composites",
        status="complete",
        details={
            "run_mode": run_mode,
            "num_candidates_evaluated": len(candidates),
            "top_k_composites": args.top_k_composites,
            "table7_csv": str(table7_csv),
            "comparison_csv": str(comparison_csv),
            "within_spread_csv": str(within_spread_csv),
            "composite_metric_json": str(composite_metric_json),
            "composite_metric_profile_csv": str(COMPOSITE_METRIC_PROFILE_FILE),
            "union_metric_json": str(union_metric_json),
            "skip_rrmse": args.skip_rrmse,
        },
    )

    logger.info("Composite evaluation stage complete")
    logger.info("Main composite table -> %s", table7_csv)
    logger.info("Composite comparison table -> %s", comparison_csv)
    logger.info("Composite spread table -> %s", within_spread_csv)
    logger.info("Checkpoint -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
