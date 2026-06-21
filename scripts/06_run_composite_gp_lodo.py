# scripts/06_run_composite_gp_lodo.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATASET_ORDER,
    GP_DEDUP_DIR,
    GP_RUNS_DIR,
    GP_SUMMARIES_DIR,
    LOGS_DIR,
    PAPER_TABLES_SUPP_CSV_DIR,
    RANK_STABILITY_SUMMARY_FILE,
    SINGLE_MEASURE_FREEZE_MANIFEST_FILE,
    ensure_output_dirs,
    get_gp_fold_dir,
)
from config.settings import (
    DEFAULT_RUN_MODE,
    DEFAULT_GP_FALLBACK_TERMINALS,
    GP_FULL_SETTINGS,
    GP_SMOKE_SETTINGS,
    USE_LODO_GP,
    validate_all_settings,
)
from src.gp.deduplication import deduplicate_seed_results
from src.gp.deap_search import build_reference_bounds, cupy_available, load_timing_summary_map, require_deap, run_gp_seed
from src.gp.lodo_runner import build_outer_folds, load_all_fold_data, summarize_fold_results
from src.gp.terminal_selection import (
    load_dataset_metric_reference,
    load_dataset_stack_counts,
    select_terminals_for_dataset_subset,
    select_terminals_for_fold,
)
from src.measures.focus_measure_library import build_focus_measure_registry
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import (
    load_json,
    save_csv_rows,
    save_json,
    validate_environment,
    validate_pipeline_prerequisites,
    write_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected leave-one-dataset-out composite GP search"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run full mode")
    parser.add_argument(
        "--top-k-rank",
        type=int,
        default=6,
        help="Top-k from corrected rank-based single-measure results per fold",
    )
    parser.add_argument(
        "--top-k-value",
        type=int,
        default=6,
        help="Top-k from corrected value-based single-measure results per fold",
    )
    parser.add_argument(
        "--terminal-names",
        type=str,
        default=None,
        help="Optional comma-separated explicit terminal list for all folds",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=35,
        help="Maximum allowed GP expression nodes during search",
    )
    parser.add_argument(
        "--max-eval-seconds",
        type=float,
        default=30.0,
        help="Soft per-individual fitness-evaluation time limit in seconds",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write/log GP progress every N generations",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Array backend for GP expression evaluation. Use cuda only after benchmarking this workload.",
    )
    parser.add_argument(
        "--no-resume-generations",
        action="store_true",
        help="Ignore generation_checkpoint.pkl and restart incomplete seeds from generation 0",
    )
    parser.add_argument(
        "--skip-final-refit",
        action="store_true",
        help="Skip the final all-dataset refit after the LODO GP runs",
    )
    parser.add_argument(
        "--final-refit-seeds",
        type=int,
        default=1,
        help="Number of final all-dataset refit seeds",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Recompute seeds even when compatible best_result.json/logbook.csv outputs already exist",
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


def explicit_terminals(args: argparse.Namespace) -> List[str] | None:
    if not args.terminal_names:
        return None
    return [item.strip() for item in args.terminal_names.split(",") if item.strip()]


def resolve_array_backend(requested_device: str, logger) -> str:
    requested = str(requested_device).strip().lower()
    if requested == "cpu":
        return "cpu"
    if cupy_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA/CuPy backend was requested, but CuPy cannot access the GPU")
    logger.warning("CUDA/CuPy backend is unavailable; GP expression evaluation will use CPU")
    return "cpu"


def validate_publication_gates(run_mode: str, logger) -> None:
    if not SINGLE_MEASURE_FREEZE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing single-measure freeze manifest: {SINGLE_MEASURE_FREEZE_MANIFEST_FILE}. "
            "Run scripts/04_evaluate_single_measures.py first."
        )
    if not RANK_STABILITY_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing rank-stability summary: {RANK_STABILITY_SUMMARY_FILE}. "
            "Run scripts/04_evaluate_single_measures.py first."
        )

    freeze_manifest = load_json(SINGLE_MEASURE_FREEZE_MANIFEST_FILE)
    rank_stability_summary = load_json(RANK_STABILITY_SUMMARY_FILE)

    if str(freeze_manifest.get("status", "")) != "frozen":
        raise RuntimeError(
            f"Single-measure freeze manifest is not frozen: {SINGLE_MEASURE_FREEZE_MANIFEST_FILE}"
        )

    if str(rank_stability_summary.get("status", "")) != "complete":
        raise RuntimeError(
            f"Rank-stability summary is not complete: {RANK_STABILITY_SUMMARY_FILE}"
        )

    if not bool(rank_stability_summary.get("passed", False)):
        message = (
            "Rank-stability gate did not pass. "
            f"Summary: {RANK_STABILITY_SUMMARY_FILE}"
        )
        if run_mode == "full":
            raise RuntimeError(message)
        logger.warning("%s", message)


GP_COMPATIBILITY_KEYS = (
    "population_size",
    "num_generations",
    "num_seeds",
    "crossover_probability",
    "mutation_probability",
    "tournament_size",
    "elitism",
    "max_tree_depth",
    "use_nsga2",
)


def _settings_compatible(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    for key in GP_COMPATIBILITY_KEYS:
        if key not in existing or key not in expected:
            return False
        if existing[key] != expected[key]:
            return False
    return True


def load_compatible_seed_result(
    *,
    held_out_dataset: str,
    seed: int,
    terminal_names: Sequence[str],
    gp_settings: Mapping[str, Any],
    logger,
) -> Optional[Dict[str, Any]]:
    out_dir = get_gp_fold_dir(held_out_dataset, seed)
    result_path = out_dir / "best_result.json"
    logbook_path = out_dir / "logbook.csv"
    if not result_path.exists() or not logbook_path.exists() or result_path.stat().st_size == 0:
        return None

    try:
        result = load_json(result_path)
    except Exception as exc:
        logger.warning(
            "[heldout=%s seed=%d] ignoring unreadable existing result %s: %s",
            held_out_dataset,
            seed,
            result_path,
            exc,
        )
        return None

    if not isinstance(result, dict):
        return None
    if str(result.get("held_out_dataset")) != str(held_out_dataset):
        return None
    if int(result.get("seed", -1)) != int(seed):
        return None
    if list(result.get("terminals", [])) != list(terminal_names):
        logger.info(
            "[heldout=%s seed=%d] existing seed output is incompatible: terminal set changed",
            held_out_dataset,
            seed,
        )
        return None
    if not _settings_compatible(dict(result.get("gp_settings", {})), gp_settings):
        logger.info(
            "[heldout=%s seed=%d] existing seed output is incompatible: GP settings changed",
            held_out_dataset,
            seed,
        )
        return None

    logbook = result.get("logbook", [])
    expected_generations = int(gp_settings["num_generations"])
    if not isinstance(logbook, list) or len(logbook) < expected_generations:
        logger.info(
            "[heldout=%s seed=%d] existing seed output is incomplete: logbook rows=%s expected=%d",
            held_out_dataset,
            seed,
            len(logbook) if isinstance(logbook, list) else "invalid",
            expected_generations,
        )
        return None

    logger.info(
        "[heldout=%s seed=%d] reusing completed seed output: %s",
        held_out_dataset,
        seed,
        result_path,
    )
    result["resume_status"] = "reused_completed_seed"
    return dict(result)


def get_final_refit_seed_dir(seed: int) -> Path:
    return GP_RUNS_DIR / "final_refit" / f"seed_{int(seed)}"


def load_compatible_final_refit_result(
    *,
    seed: int,
    terminal_names: Sequence[str],
    gp_settings: Mapping[str, Any],
    logger,
) -> Optional[Dict[str, Any]]:
    out_dir = get_final_refit_seed_dir(seed)
    result_path = out_dir / "best_result.json"
    logbook_path = out_dir / "logbook.csv"
    if not result_path.exists() or not logbook_path.exists() or result_path.stat().st_size == 0:
        return None

    try:
        result = load_json(result_path)
    except Exception as exc:
        logger.warning(
            "[final_refit seed=%d] ignoring unreadable existing result %s: %s",
            seed,
            result_path,
            exc,
        )
        return None

    if not isinstance(result, dict):
        return None
    if str(result.get("result_type", "")) != "final_refit":
        return None
    if int(result.get("seed", -1)) != int(seed):
        return None
    if list(result.get("terminals", [])) != list(terminal_names):
        logger.info("[final_refit seed=%d] existing output is incompatible: terminal set changed", seed)
        return None
    if not _settings_compatible(dict(result.get("gp_settings", {})), gp_settings):
        logger.info("[final_refit seed=%d] existing output is incompatible: GP settings changed", seed)
        return None

    logbook = result.get("logbook", [])
    expected_generations = int(gp_settings["num_generations"])
    if not isinstance(logbook, list) or len(logbook) < expected_generations:
        logger.info(
            "[final_refit seed=%d] existing output is incomplete: logbook rows=%s expected=%d",
            seed,
            len(logbook) if isinstance(logbook, list) else "invalid",
            expected_generations,
        )
        return None

    logger.info("[final_refit seed=%d] reusing completed seed output: %s", seed, result_path)
    result["resume_status"] = "reused_completed_seed"
    return dict(result)


def summarize_final_refit_results(seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not seed_results:
        raise ValueError("No final refit seed results are available")
    best_result = min(
        seed_results,
        key=lambda row: (
            float(row["best_training_objective"]),
            float(row.get("best_complexity", float("inf"))),
            int(row.get("seed", 10**9)),
        ),
    )
    objectives = [float(row["best_training_objective"]) for row in seed_results]
    all_dataset_scores = [float(row.get("all_dataset_score", float("nan"))) for row in seed_results]
    return {
        "result_type": "final_refit_summary",
        "num_seeds": int(len(seed_results)),
        "best_seed": int(best_result["seed"]),
        "best_expression": str(best_result["best_expression"]),
        "best_training_objective": float(best_result["best_training_objective"]),
        "best_complexity": float(best_result["best_complexity"]),
        "best_num_nodes": int(best_result["num_nodes"]),
        "best_tree_height": int(best_result["tree_height"]),
        "best_all_dataset_score": float(best_result.get("all_dataset_score", float("nan"))),
        "mean_training_objective": float(np.mean(objectives)),
        "std_training_objective": float(np.std(objectives, ddof=0)),
        "mean_all_dataset_score": float(np.nanmean(all_dataset_scores)),
        "std_all_dataset_score": float(np.nanstd(all_dataset_scores, ddof=0)),
        "terminals": list(best_result["terminals"]),
        "train_datasets": list(best_result["train_datasets"]),
        "best_result": dict(best_result),
    }


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    require_deap()
    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=True)
    set_global_seed(42)

    if not USE_LODO_GP:
        raise RuntimeError("This script requires USE_LODO_GP=True in config/settings.py")

    log_file = LOGS_DIR / f"run_composite_gp_lodo_{run_mode}.log"
    logger = get_logger("run_composite_gp_lodo", log_file=log_file)
    validate_publication_gates(run_mode, logger)
    array_backend = resolve_array_backend(args.device, logger)

    gp_settings = dict(GP_SMOKE_SETTINGS if run_mode == "smoke" else GP_FULL_SETTINGS)
    gp_settings["max_nodes"] = int(args.max_nodes)
    gp_settings["max_eval_seconds"] = None if float(args.max_eval_seconds) <= 0.0 else float(args.max_eval_seconds)
    gp_settings["device"] = array_backend
    dataset_metric_reference = load_dataset_metric_reference()
    dataset_stack_counts = load_dataset_stack_counts()
    measure_registry = build_focus_measure_registry()
    timing_map = load_timing_summary_map()

    logger.info("Starting corrected composite GP LODO stage")
    logger.info("Run mode: %s", run_mode)
    logger.info("GP settings: %s", gp_settings)
    logger.info("GP array backend: %s", array_backend)
    logger.info(
        "Seed resume policy: skip_completed=%s resume_generations=%s progress_every=%d",
        not bool(args.no_skip_completed),
        not bool(args.no_resume_generations),
        int(args.progress_every),
    )
    if not timing_map:
        logger.warning("Single-measure timing summary is missing; GP runtime scoring will use combination timing only")

    fold_summaries: List[Dict[str, Any]] = []
    all_seed_results: List[Dict[str, Any]] = []
    terminal_manifest_rows: List[Dict[str, Any]] = []
    terminal_manifest_payload: List[Dict[str, Any]] = []
    requested_terminals = explicit_terminals(args)
    outer_folds = build_outer_folds(DATASET_ORDER)
    reference_bounds = build_reference_bounds(
        dataset_metric_reference=dataset_metric_reference,
        dataset_subset=DATASET_ORDER,
    )

    for fold in outer_folds:
        if requested_terminals is None:
            selection = select_terminals_for_fold(
                held_out_dataset=fold.held_out_dataset,
                dataset_metric_reference=dataset_metric_reference,
                dataset_stack_counts=dataset_stack_counts,
                registry=measure_registry,
                top_k_rank=args.top_k_rank,
                top_k_value=args.top_k_value,
                fallback_terminals=DEFAULT_GP_FALLBACK_TERMINALS,
            )
            terminal_names = list(selection.selected_terminals)
            terminal_manifest_rows.extend(list(selection.selection_rows))
            terminal_manifest_payload.append(
                {
                    "held_out_dataset": fold.held_out_dataset,
                    "train_datasets": list(fold.train_datasets),
                    "selected_terminals": terminal_names,
                    "selection_rows": list(selection.selection_rows),
                }
            )
        else:
            terminal_names = list(requested_terminals)
            terminal_manifest_payload.append(
                {
                    "held_out_dataset": fold.held_out_dataset,
                    "train_datasets": list(fold.train_datasets),
                    "selected_terminals": terminal_names,
                    "selection_rows": [],
                    "selection_mode": "explicit",
                }
            )

        logger.info("[heldout=%s] selected %d terminals", fold.held_out_dataset, len(terminal_names))
        logger.info("[heldout=%s] terminals: %s", fold.held_out_dataset, terminal_names)

        curves_by_dataset, labels_by_dataset, label_modes = load_all_fold_data(terminal_names)

        seed_results: List[Dict[str, Any]] = []
        for seed_offset in range(int(gp_settings["num_seeds"])):
            seed = 42 + seed_offset
            out_dir = get_gp_fold_dir(fold.held_out_dataset, seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            result: Optional[Dict[str, Any]] = None

            if not args.no_skip_completed:
                result = load_compatible_seed_result(
                    held_out_dataset=fold.held_out_dataset,
                    seed=seed,
                    terminal_names=terminal_names,
                    gp_settings=gp_settings,
                    logger=logger,
                )

            if result is None:
                save_json(
                    {
                        "status": "running",
                        "held_out_dataset": fold.held_out_dataset,
                        "seed": int(seed),
                        "selected_terminals": terminal_names,
                        "gp_settings": gp_settings,
                        "progress_csv": str(out_dir / "progress.csv"),
                        "generation_checkpoint": str(out_dir / "generation_checkpoint.pkl"),
                    },
                    out_dir / "seed_status.json",
                )
                result = run_gp_seed(
                    held_out_dataset=fold.held_out_dataset,
                    seed=seed,
                    terminal_names=terminal_names,
                    curves_by_dataset=curves_by_dataset,
                    labels_by_dataset=labels_by_dataset,
                    reference_bounds=reference_bounds,
                    gp_settings=gp_settings,
                    timing_map=timing_map,
                    logger=logger,
                    progress_path=out_dir / "progress.csv",
                    progress_every=int(args.progress_every),
                    checkpoint_path=out_dir / "generation_checkpoint.pkl",
                    resume_checkpoint=not bool(args.no_resume_generations),
                    array_backend=array_backend,
                )
                result["resume_status"] = "computed"
                save_json(result, out_dir / "best_result.json")
                save_csv_rows(result["logbook"], out_dir / "logbook.csv")
                save_json(
                    {
                        "status": "complete",
                        "held_out_dataset": fold.held_out_dataset,
                        "seed": int(seed),
                        "best_result_json": str(out_dir / "best_result.json"),
                        "logbook_csv": str(out_dir / "logbook.csv"),
                        "progress_csv": str(out_dir / "progress.csv"),
                        "generation_checkpoint": str(out_dir / "generation_checkpoint.pkl"),
                    },
                    out_dir / "seed_status.json",
                )

            result["label_modes"] = label_modes
            seed_results.append(result)
            all_seed_results.append(result)

        fold_summary = summarize_fold_results(fold, seed_results)
        fold_summary["selected_terminals"] = " | ".join(terminal_names)
        fold_summaries.append(fold_summary)

        save_json(
            {
                "outer_fold": fold.outer_fold,
                "held_out_dataset": fold.held_out_dataset,
                "train_datasets": list(fold.train_datasets),
                "selected_terminals": terminal_names,
                "seed_results": seed_results,
            },
            GP_SUMMARIES_DIR / f"heldout_{fold.held_out_dataset}_summary.json",
        )

    final_refit_summary: Optional[Dict[str, Any]] = None
    final_refit_seed_results: List[Dict[str, Any]] = []
    if not bool(args.skip_final_refit):
        final_refit_seed_count = int(args.final_refit_seeds)
        if final_refit_seed_count <= 0:
            raise ValueError("--final-refit-seeds must be positive when final refit is enabled")

        if requested_terminals is None:
            final_selection = select_terminals_for_dataset_subset(
                selection_name="FINAL_ALL",
                train_datasets=DATASET_ORDER,
                dataset_metric_reference=dataset_metric_reference,
                dataset_stack_counts=dataset_stack_counts,
                registry=measure_registry,
                top_k_rank=args.top_k_rank,
                top_k_value=args.top_k_value,
                fallback_terminals=DEFAULT_GP_FALLBACK_TERMINALS,
            )
            final_terminal_names = list(final_selection.selected_terminals)
            terminal_manifest_rows.extend(list(final_selection.selection_rows))
            terminal_manifest_payload.append(
                {
                    "held_out_dataset": "FINAL_ALL",
                    "train_datasets": list(DATASET_ORDER),
                    "selected_terminals": final_terminal_names,
                    "selection_rows": list(final_selection.selection_rows),
                    "selection_mode": "final_refit_all_datasets",
                }
            )
        else:
            final_terminal_names = list(requested_terminals)
            terminal_manifest_payload.append(
                {
                    "held_out_dataset": "FINAL_ALL",
                    "train_datasets": list(DATASET_ORDER),
                    "selected_terminals": final_terminal_names,
                    "selection_rows": [],
                    "selection_mode": "explicit_final_refit_all_datasets",
                }
            )

        logger.info("[final_refit] selected %d terminals", len(final_terminal_names))
        logger.info("[final_refit] terminals: %s", final_terminal_names)
        final_curves_by_dataset, final_labels_by_dataset, final_label_modes = load_all_fold_data(final_terminal_names)

        for seed_offset in range(final_refit_seed_count):
            seed = 42 + seed_offset
            out_dir = get_final_refit_seed_dir(seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            result: Optional[Dict[str, Any]] = None

            if not args.no_skip_completed:
                result = load_compatible_final_refit_result(
                    seed=seed,
                    terminal_names=final_terminal_names,
                    gp_settings=gp_settings,
                    logger=logger,
                )

            if result is None:
                save_json(
                    {
                        "status": "running",
                        "result_type": "final_refit",
                        "seed": int(seed),
                        "selected_terminals": final_terminal_names,
                        "train_datasets": list(DATASET_ORDER),
                        "gp_settings": gp_settings,
                        "progress_csv": str(out_dir / "progress.csv"),
                        "generation_checkpoint": str(out_dir / "generation_checkpoint.pkl"),
                    },
                    out_dir / "seed_status.json",
                )
                result = run_gp_seed(
                    held_out_dataset="FINAL_ALL",
                    seed=seed,
                    terminal_names=final_terminal_names,
                    curves_by_dataset=final_curves_by_dataset,
                    labels_by_dataset=final_labels_by_dataset,
                    reference_bounds=reference_bounds,
                    gp_settings=gp_settings,
                    timing_map=timing_map,
                    logger=logger,
                    progress_path=out_dir / "progress.csv",
                    progress_every=int(args.progress_every),
                    checkpoint_path=out_dir / "generation_checkpoint.pkl",
                    resume_checkpoint=not bool(args.no_resume_generations),
                    array_backend=array_backend,
                    train_datasets_override=DATASET_ORDER,
                    heldout_evaluation_dataset=None,
                )
                result["result_type"] = "final_refit"
                result["resume_status"] = "computed"
                save_json(result, out_dir / "best_result.json")
                save_csv_rows(result["logbook"], out_dir / "logbook.csv")
                save_json(
                    {
                        "status": "complete",
                        "result_type": "final_refit",
                        "seed": int(seed),
                        "best_result_json": str(out_dir / "best_result.json"),
                        "logbook_csv": str(out_dir / "logbook.csv"),
                        "progress_csv": str(out_dir / "progress.csv"),
                        "generation_checkpoint": str(out_dir / "generation_checkpoint.pkl"),
                    },
                    out_dir / "seed_status.json",
                )

            result["label_modes"] = final_label_modes
            result["result_type"] = "final_refit"
            final_refit_seed_results.append(result)

        final_refit_summary = summarize_final_refit_results(final_refit_seed_results)
        final_refit_summary["selection_rule"] = {
            "top_k_rank": args.top_k_rank,
            "top_k_value": args.top_k_value,
            "fallback_terminals": list(DEFAULT_GP_FALLBACK_TERMINALS),
            "explicit_terminals": requested_terminals,
        }
        save_json(final_refit_seed_results, GP_SUMMARIES_DIR / "final_refit_seed_results.json")
        save_json(final_refit_summary, GP_SUMMARIES_DIR / "final_refit_summary.json")
        save_json(
            {
                "result_type": "final_refit_best_expression",
                "source": "all_dataset_refit_after_lodo_validation",
                "seed": int(final_refit_summary["best_seed"]),
                "best_expression": str(final_refit_summary["best_expression"]),
                "terminals": list(final_refit_summary["terminals"]),
                "train_datasets": list(final_refit_summary["train_datasets"]),
                "best_training_objective": float(final_refit_summary["best_training_objective"]),
                "best_all_dataset_score": float(final_refit_summary["best_all_dataset_score"]),
                "best_complexity": float(final_refit_summary["best_complexity"]),
                "num_nodes": int(final_refit_summary["best_num_nodes"]),
                "tree_height": int(final_refit_summary["best_tree_height"]),
                "gp_settings": dict(gp_settings),
                "summary_json": str(GP_SUMMARIES_DIR / "final_refit_summary.json"),
            },
            GP_SUMMARIES_DIR / "final_composite_expression.json",
        )

    deduplicated_results = sorted(
        deduplicate_seed_results(all_seed_results),
        key=lambda row: (
            float(row["heldout_score"]),
            float(row.get("best_complexity", float("inf"))),
            int(row.get("seed", 10**9)),
        ),
    )

    selected_manifest_json = GP_SUMMARIES_DIR / "selected_terminals.json"
    selected_manifest_csv = GP_SUMMARIES_DIR / "selected_terminal_manifest.csv"
    fold_summary_csv = PAPER_TABLES_SUPP_CSV_DIR / "STable4_gp_foldwise_results.csv"
    dedup_json = GP_DEDUP_DIR / "deduplicated_best_expressions.json"

    save_json(
        {
            "run_mode": run_mode,
            "selection_rule": {
                "top_k_rank": args.top_k_rank,
                "top_k_value": args.top_k_value,
                "fallback_terminals": list(DEFAULT_GP_FALLBACK_TERMINALS),
                "explicit_terminals": requested_terminals,
            },
            "folds": terminal_manifest_payload,
        },
        selected_manifest_json,
    )
    if terminal_manifest_rows:
        save_csv_rows(terminal_manifest_rows, selected_manifest_csv)
    save_csv_rows(fold_summaries, fold_summary_csv)
    save_json(all_seed_results, GP_SUMMARIES_DIR / "all_seed_results.json")
    save_json(deduplicated_results, dedup_json)

    checkpoint_path = GP_SUMMARIES_DIR / "run_composite_gp_lodo.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="run_composite_gp_lodo",
        status="complete",
        details={
            "run_mode": run_mode,
            "num_folds": len(outer_folds),
            "num_seed_results": len(all_seed_results),
            "final_refit_enabled": not bool(args.skip_final_refit),
            "final_refit_num_seed_results": len(final_refit_seed_results),
            "final_refit_summary_json": str(GP_SUMMARIES_DIR / "final_refit_summary.json") if final_refit_summary else "",
            "final_composite_expression_json": str(GP_SUMMARIES_DIR / "final_composite_expression.json") if final_refit_summary else "",
            "selected_terminal_manifest_json": str(selected_manifest_json),
            "selected_terminal_manifest_csv": str(selected_manifest_csv) if terminal_manifest_rows else "",
            "fold_summary_csv": str(fold_summary_csv),
            "deduplicated_shortlist_json": str(dedup_json),
        },
    )

    logger.info("Composite GP LODO stage complete")
    logger.info("Selected terminal manifest -> %s", selected_manifest_json)
    if terminal_manifest_rows:
        logger.info("Selected terminal CSV -> %s", selected_manifest_csv)
    logger.info("Fold summary CSV -> %s", fold_summary_csv)
    logger.info("Deduplicated shortlist -> %s", dedup_json)
    if final_refit_summary:
        logger.info("Final refit summary -> %s", GP_SUMMARIES_DIR / "final_refit_summary.json")
        logger.info("Final composite expression -> %s", GP_SUMMARIES_DIR / "final_composite_expression.json")
    logger.info("Checkpoint -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
