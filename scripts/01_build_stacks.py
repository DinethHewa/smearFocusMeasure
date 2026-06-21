# scripts/01_build_stacks.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATASET_ORDER,
    RAW_DATASETS,
    ensure_output_dirs,
    get_stack_file,
    get_stack_metadata_file,
)
from config.settings import (
    CONVERT_TO_GRAYSCALE_WHEN_NEEDED,
    DEFAULT_RUN_MODE,
    GRAYSCALE_MODE,
    PRESERVE_NATIVE_RESOLUTION,
    REQUIRE_ROI_SAME_SIZE,
    ROI_MODE,
    ROI_SIZE,
    USE_ROI_CROPPING,
    get_run_profile,
    validate_all_settings,
)
from src.io.stack_builder import StackBuildConfig, build_dataset_stacks
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import save_json, validate_environment, write_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build native-resolution stack files from raw datasets")
    parser.add_argument("--smoke-test", action="store_true", help="Run small subset only")
    parser.add_argument("--full-run", action="store_true", help="Run full dataset build")
    parser.add_argument(
        "--max-stacks-per-dataset",
        type=int,
        default=None,
        help="Override max stacks per dataset",
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


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    set_global_seed(42)

    log_file = PROJECT_ROOT / "outputs" / "00_logs" / f"build_stacks_{run_mode}.log"
    logger = get_logger("build_stacks", log_file=log_file)

    profile = get_run_profile(run_mode)
    max_stacks = (
        args.max_stacks_per_dataset
        if args.max_stacks_per_dataset is not None
        else profile["max_stacks_per_dataset"]
    )
    build_config = StackBuildConfig(
        convert_to_grayscale_when_needed=CONVERT_TO_GRAYSCALE_WHEN_NEEDED,
        grayscale_mode=GRAYSCALE_MODE,
        preserve_native_resolution=PRESERVE_NATIVE_RESOLUTION,
        use_roi_cropping=USE_ROI_CROPPING,
        roi_mode=ROI_MODE,
        roi_size=ROI_SIZE,
        require_same_size_within_stack=REQUIRE_ROI_SAME_SIZE,
    )

    logger.info("Starting stack build stage")
    logger.info("Run mode: %s", run_mode)
    logger.info("Max stacks per dataset: %s", str(max_stacks))
    logger.info("Native resolution preserved: %s", PRESERVE_NATIVE_RESOLUTION)
    logger.info("ROI cropping enabled: %s", USE_ROI_CROPPING)

    stage_summary: Dict[str, Dict] = {}

    for dataset_name in DATASET_ORDER:
        dataset_root = Path(RAW_DATASETS[dataset_name]).expanduser().resolve()
        logger.info("[%s] building stacks from %s", dataset_name, dataset_root)

        dataset_array, metadata = build_dataset_stacks(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            max_stacks=max_stacks,
            logger=logger,
            config=build_config,
        )

        stack_file = get_stack_file(dataset_name)
        metadata_file = get_stack_metadata_file(dataset_name)

        stack_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        np.save(stack_file, dataset_array, allow_pickle=True)
        save_json(metadata, metadata_file)

        logger.info("[%s] saved stack file -> %s", dataset_name, stack_file)
        logger.info("[%s] saved metadata   -> %s", dataset_name, metadata_file)

        stage_summary[dataset_name] = {
            "stack_file": str(stack_file),
            "metadata_file": str(metadata_file),
            "stack_count": metadata["stack_count"],
            "stack_count_before_truncation": metadata["stack_count_before_truncation"],
            "discovery_mode": metadata["discovery_mode"],
            "planes_per_stack_min": metadata["planes_per_stack_min"],
            "planes_per_stack_max": metadata["planes_per_stack_max"],
        }

    checkpoint_path = PROJECT_ROOT / "outputs" / "01_stacks" / "build_stacks.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="build_stacks",
        status="complete",
        details={
            "run_mode": run_mode,
            "max_stacks_per_dataset": max_stacks,
            "datasets": stage_summary,
        },
    )

    logger.info("Stack build stage complete")
    logger.info("Checkpoint written to %s", checkpoint_path)


if __name__ == "__main__":
    main()
