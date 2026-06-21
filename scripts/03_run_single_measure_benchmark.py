# scripts/03_run_single_measure_benchmark.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATASET_ORDER,
    LOGS_DIR,
    SINGLE_OPERATOR_MANIFEST_FILE,
    SINGLE_TIMING_SUMMARY_FILE,
    ensure_output_dirs,
    get_single_norm_curve_file,
    get_single_raw_curve_file,
    get_single_timing_file,
    get_stack_file,
)
from config.settings import (
    DEFAULT_RUN_MODE,
    EXPECTED_NUM_FOCUS_MEASURES,
    NORMALIZE_FOCUS_CURVES_PER_STACK,
    get_run_profile,
    validate_all_settings,
)
from src.evaluation.autofocus_metrics import (
    compute_focus_curve_for_stack,
    normalize_focus_curve,
)
from src.evaluation.sensitivity import (
    compute_timing_summary_for_measure,
    flatten_timing_record,
)
from src.measures.focus_measure_library import build_focus_measure_registry
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import (
    save_csv_rows,
    save_json,
    validate_environment,
    validate_pipeline_prerequisites,
    validate_stack_array,
    write_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-measure benchmark: raw curves, normalized curves, timing outputs"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run in full mode")
    parser.add_argument(
        "--measure-names",
        type=str,
        default=None,
        help="Optional comma-separated subset of measure names to run",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip timing study",
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


def load_dataset_stacks(dataset_name: str) -> np.ndarray:
    stacks = np.load(get_stack_file(dataset_name), allow_pickle=True)
    validate_stack_array(stacks, dataset_name)
    return stacks


def resolve_measure_subset(
    registry: Dict[str, Dict[str, Any]],
    measure_names_arg: str | None,
) -> Dict[str, Dict[str, Any]]:
    if not measure_names_arg:
        return registry

    requested = [item.strip() for item in measure_names_arg.split(",") if item.strip()]
    missing = [item for item in requested if item not in registry]
    if missing:
        raise KeyError(f"Requested measures not found in registry: {missing}")
    return {name: registry[name] for name in requested}


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=False)
    set_global_seed(42)

    log_file = LOGS_DIR / f"run_single_measure_benchmark_{run_mode}.log"
    logger = get_logger("run_single_measure_benchmark", log_file=log_file)

    profile = get_run_profile(run_mode)
    registry_full = build_focus_measure_registry()
    registry = resolve_measure_subset(registry_full, args.measure_names)

    logger.info("Starting single-measure benchmark stage")
    logger.info("Run mode: %s", run_mode)
    logger.info("Run profile: %s", profile)
    logger.info("Found %d measures in active registry", len(registry))
    logger.info("Expected paper target: %d measures", EXPECTED_NUM_FOCUS_MEASURES)
    if len(registry) != EXPECTED_NUM_FOCUS_MEASURES:
        logger.warning(
            "Registry currently has %d measures, not %d. This is acceptable for interim runs, but later prompts should extend or align it.",
            len(registry),
            EXPECTED_NUM_FOCUS_MEASURES,
        )

    operator_manifest: List[Dict[str, Any]] = []
    timing_summary_rows: List[Dict[str, Any]] = []

    for measure_name, entry in registry.items():
        measure_func = entry["func"]
        maximize = bool(entry.get("maximize", True))
        family = entry.get("family", "unknown")
        notes = entry.get("notes", "")

        logger.info("Running measure: %s", measure_name)
        operator_manifest.append(
            {
                "measure_name": measure_name,
                "maximize": maximize,
                "family": family,
                "notes": notes,
            }
        )

        for dataset_name in DATASET_ORDER:
            logger.info("[%s] computing curves for %s", dataset_name, measure_name)
            stacks = load_dataset_stacks(dataset_name)

            raw_curves: List[np.ndarray] = []
            norm_curves: List[np.ndarray] = []
            for stack in np.asarray(stacks, dtype=object):
                stack_arr = np.asarray(stack)
                if stack_arr.ndim != 3:
                    raise ValueError(
                        f"[{dataset_name}] invalid stack shape for measure {measure_name}: {stack_arr.shape}"
                    )
                curve_raw = compute_focus_curve_for_stack(stack_arr, measure_func)
                raw_curves.append(curve_raw)
                norm_curves.append(
                    normalize_focus_curve(curve_raw) if NORMALIZE_FOCUS_CURVES_PER_STACK else curve_raw.copy()
                )

            raw_out = np.array(raw_curves, dtype=object)
            norm_out = np.array(norm_curves, dtype=object)
            raw_path = get_single_raw_curve_file(dataset_name, measure_name)
            norm_path = get_single_norm_curve_file(dataset_name, measure_name)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, raw_out, allow_pickle=True)
            np.save(norm_path, norm_out, allow_pickle=True)

            logger.info("[%s] saved raw curves  -> %s", dataset_name, raw_path)
            logger.info("[%s] saved norm curves -> %s", dataset_name, norm_path)

            if not args.skip_timing:
                timing_record = compute_timing_summary_for_measure(
                    dataset_name=dataset_name,
                    stacks=stacks,
                    measure_name=measure_name,
                    measure_func=measure_func,
                    run_mode=run_mode,
                )
                timing_path = get_single_timing_file(dataset_name, measure_name)
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(timing_record, timing_path)
                timing_summary_rows.append(flatten_timing_record(timing_record))
                logger.info("[%s] saved timing -> %s", dataset_name, timing_path)

    save_json(operator_manifest, SINGLE_OPERATOR_MANIFEST_FILE)
    if not args.skip_timing:
        save_csv_rows(timing_summary_rows, SINGLE_TIMING_SUMMARY_FILE)

    checkpoint_path = SINGLE_OPERATOR_MANIFEST_FILE.parent / "run_single_measure_benchmark.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="run_single_measure_benchmark",
        status="complete",
        details={
            "run_mode": run_mode,
            "num_measures_run": len(registry),
            "operator_manifest": str(SINGLE_OPERATOR_MANIFEST_FILE),
            "timing_summary_csv": None if args.skip_timing else str(SINGLE_TIMING_SUMMARY_FILE),
            "datasets": list(DATASET_ORDER),
        },
    )

    logger.info("Single-measure benchmark stage complete")
    logger.info("Operator manifest written to %s", SINGLE_OPERATOR_MANIFEST_FILE)
    if not args.skip_timing:
        logger.info("Timing summary written to %s", SINGLE_TIMING_SUMMARY_FILE)
    logger.info("Checkpoint written to %s", checkpoint_path)


if __name__ == "__main__":
    main()
