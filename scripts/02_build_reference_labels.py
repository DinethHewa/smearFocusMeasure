# scripts/02_build_reference_labels.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATASET_ORDER,
    LABEL_PROVENANCE_MANIFEST_FILE,
    LABEL_SOURCE_MANIFEST_FILE,
    LABEL_SUMMARY_FILE,
    LOGS_DIR,
    LOO_VOTER_DISCLOSURE_FILE,
    RAW_DATASETS,
    ensure_output_dirs,
    get_loo_label_file,
    get_source_label_file,
    get_stack_file,
    get_stack_metadata_file,
    get_surrogate_label_file,
)
from config.settings import (
    DEFAULT_RUN_MODE,
    DEFAULT_SURROGATE_VOTERS,
    USE_LEAVE_ONE_OUT_SURROGATE_VOTING,
    get_run_profile,
    validate_all_settings,
)
from src.labels.loo_voting import (
    build_loo_label_sets,
    resolve_loo_measure_names,
    voter_pool_for_target_measure,
)
from src.labels.source_labels import discover_source_labels
from src.labels.surrogate_labels import (
    build_majority_vote_labels_from_predictions,
    compute_measure_peak_predictions,
)
from src.measures.focus_measure_library import (
    build_focus_measure_registry,
    validate_registry,
)
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import (
    load_json,
    save_json,
    validate_environment,
    validate_pipeline_prerequisites,
    validate_stack_and_label_alignment,
    validate_stack_array,
    write_checkpoint,
)


def load_measure_registry(logger) -> Dict[str, Dict[str, Any]]:
    registry = build_focus_measure_registry()
    logger.info("Loaded %d focus measures from registry", len(registry))
    return registry


def load_stack_array(dataset_name: str) -> np.ndarray:
    stack_file = get_stack_file(dataset_name)
    stacks = np.load(stack_file, allow_pickle=True)
    validate_stack_array(stacks, dataset_name)
    return stacks


def load_stack_metadata(dataset_name: str) -> Dict[str, Any]:
    metadata_file = get_stack_metadata_file(dataset_name)
    if not metadata_file.exists():
        return {}
    return load_json(metadata_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source, surrogate, and leave-one-out reference labels")
    parser.add_argument("--smoke-test", action="store_true", help="Run small subset mode")
    parser.add_argument("--full-run", action="store_true", help="Run full mode")
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional limit for debugging dataset loop",
    )
    parser.add_argument(
        "--loo-measures",
        type=str,
        default=None,
        help="Comma-separated measure names to build LOO labels for. Defaults to registry/all critical measures.",
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
    validate_pipeline_prerequisites(require_stacks=True, require_labels=False)
    set_global_seed(42)

    log_file = LOGS_DIR / f"build_reference_labels_{run_mode}.log"
    logger = get_logger("build_reference_labels", log_file=log_file)

    profile = get_run_profile(run_mode)
    logger.info("Starting reference-label stage")
    logger.info("Run mode: %s", run_mode)
    logger.info("Run profile: %s", profile)

    registry = load_measure_registry(logger)
    validate_registry(tuple(DEFAULT_SURROGATE_VOTERS))

    extra_loo_measures: Optional[List[str]] = None
    if args.loo_measures:
        extra_loo_measures = [item.strip() for item in args.loo_measures.split(",") if item.strip()]

    loo_measure_names = resolve_loo_measure_names(registry, extra_measures=extra_loo_measures)
    logger.info("LOO labels will be generated for %d measures", len(loo_measure_names))
    logger.info("Default surrogate voters: %s", list(DEFAULT_SURROGATE_VOTERS))

    dataset_names = list(DATASET_ORDER)
    if args.max_datasets is not None:
        dataset_names = dataset_names[: args.max_datasets]

    manifest: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for dataset_name in dataset_names:
        logger.info("[%s] processing dataset", dataset_name)

        stacks = load_stack_array(dataset_name)
        metadata = load_stack_metadata(dataset_name)
        raw_root = Path(RAW_DATASETS[dataset_name]).expanduser().resolve()
        num_stacks = len(stacks)

        source_discovery = discover_source_labels(
            dataset_name=dataset_name,
            dataset_root=raw_root,
            num_stacks=num_stacks,
            logger=logger,
        )

        source_label_file = get_source_label_file(dataset_name)
        surrogate_label_file = get_surrogate_label_file(dataset_name)

        dataset_record: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "stack_count": num_stacks,
            "raw_dataset_root": str(raw_root),
            "source_label_mode": None,
            "source_label_source_kind": source_discovery.source_kind,
            "source_label_file_detected": source_discovery.source_path,
            "source_label_note": source_discovery.note,
            "source_label_candidate_files": source_discovery.candidate_files,
            "source_label_auxiliary_files": source_discovery.auxiliary_files,
            "source_label_details": source_discovery.details,
            "saved_source_label_file": str(source_label_file),
            "saved_surrogate_label_file": str(surrogate_label_file),
            "surrogate_voters": list(DEFAULT_SURROGATE_VOTERS),
            "loo_labels_written": [],
            "metadata_file": str(get_stack_metadata_file(dataset_name)),
            "stack_file": str(get_stack_file(dataset_name)),
            "metadata_summary": metadata,
        }

        if source_discovery.labels is not None:
            validate_stack_and_label_alignment(stacks, source_discovery.labels, dataset_name)
            np.save(source_label_file, source_discovery.labels.astype(int), allow_pickle=False)
            dataset_record["source_label_mode"] = "source"
            logger.info("[%s] saved source labels -> %s", dataset_name, source_label_file)
        else:
            dataset_record["source_label_mode"] = "surrogate"
            logger.info(
                "[%s] no valid source labels found; building surrogate labels (%s)",
                dataset_name,
                source_discovery.note,
            )

            voter_predictions = compute_measure_peak_predictions(
                stacks=stacks,
                measure_names=DEFAULT_SURROGATE_VOTERS,
                registry=registry,
                dataset_name=dataset_name,
                logger=logger,
            )
            surrogate_labels = build_majority_vote_labels_from_predictions(
                predictions=voter_predictions,
                voter_names=DEFAULT_SURROGATE_VOTERS,
            )
            validate_stack_and_label_alignment(stacks, surrogate_labels, dataset_name)
            np.save(surrogate_label_file, surrogate_labels, allow_pickle=False)
            logger.info("[%s] saved surrogate labels -> %s", dataset_name, surrogate_label_file)

            if USE_LEAVE_ONE_OUT_SURROGATE_VOTING:
                loo_label_sets = build_loo_label_sets(
                    voter_predictions=voter_predictions,
                    target_measure_names=loo_measure_names,
                )
                for measure_name in loo_measure_names:
                    if measure_name not in loo_label_sets:
                        logger.warning(
                            "[%s] skipped LOO for %s because the voter pool became empty",
                            dataset_name,
                            measure_name,
                        )
                        continue

                    loo_labels = loo_label_sets[measure_name]
                    validate_stack_and_label_alignment(stacks, loo_labels, dataset_name)
                    loo_path = get_loo_label_file(dataset_name, measure_name)
                    loo_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(loo_path, loo_labels, allow_pickle=False)

                    voter_pool = voter_pool_for_target_measure(measure_name)
                    dataset_record["loo_labels_written"].append(
                        {
                            "measure_name": measure_name,
                            "voter_pool_size": len(voter_pool),
                            "voter_pool": list(voter_pool),
                            "path": str(loo_path),
                        }
                    )

                logger.info(
                    "[%s] wrote %d LOO label files",
                    dataset_name,
                    len(dataset_record["loo_labels_written"]),
                )

        manifest[dataset_name] = dataset_record
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "stack_count": num_stacks,
                "label_mode": dataset_record["source_label_mode"],
                "source_label_note": dataset_record["source_label_note"],
                "source_label_file_detected": dataset_record["source_label_file_detected"],
                "surrogate_written": dataset_record["source_label_mode"] == "surrogate",
                "loo_count": len(dataset_record["loo_labels_written"]),
            }
        )

    save_json(manifest, LABEL_SOURCE_MANIFEST_FILE)
    save_json(summary_rows, LABEL_SUMMARY_FILE)
    save_json(
        {
            dataset_name: {
                "dataset_name": dataset_name,
                "stack_count": record["stack_count"],
                "label_mode": record["source_label_mode"],
                "source_label_source_kind": record["source_label_source_kind"],
                "source_label_note": record["source_label_note"],
                "source_label_file_detected": record["source_label_file_detected"],
                "surrogate_voters": record["surrogate_voters"],
                "loo_measure_count": len(record["loo_labels_written"]),
            }
            for dataset_name, record in manifest.items()
        },
        LABEL_PROVENANCE_MANIFEST_FILE,
    )
    save_json(
        {
            "default_surrogate_voters": list(DEFAULT_SURROGATE_VOTERS),
            "datasets": {
                dataset_name: record["loo_labels_written"]
                for dataset_name, record in manifest.items()
            },
        },
        LOO_VOTER_DISCLOSURE_FILE,
    )

    checkpoint_path = LABEL_SOURCE_MANIFEST_FILE.parent / "build_reference_labels.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="build_reference_labels",
        status="complete",
        details={
            "run_mode": run_mode,
            "datasets_processed": dataset_names,
            "manifest_file": str(LABEL_SOURCE_MANIFEST_FILE),
            "summary_file": str(LABEL_SUMMARY_FILE),
            "label_provenance_manifest": str(LABEL_PROVENANCE_MANIFEST_FILE),
            "loo_voter_disclosure": str(LOO_VOTER_DISCLOSURE_FILE),
            "loo_enabled": USE_LEAVE_ONE_OUT_SURROGATE_VOTING,
            "default_voters": list(DEFAULT_SURROGATE_VOTERS),
        },
    )

    logger.info("Reference-label stage complete")
    logger.info("Manifest written to %s", LABEL_SOURCE_MANIFEST_FILE)
    logger.info("Summary written to %s", LABEL_SUMMARY_FILE)
    logger.info("Checkpoint written to %s", checkpoint_path)


if __name__ == "__main__":
    main()
