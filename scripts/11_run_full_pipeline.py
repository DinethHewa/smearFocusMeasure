# scripts/11_run_full_pipeline.py

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    LABEL_PROVENANCE_MANIFEST_FILE,
    LOGS_DIR,
    LOO_VOTER_DISCLOSURE_FILE,
    OUTPUTS_DIR,
    PAPER_ASSET_INDEX_FILE,
    PAPER_DIR,
    PAPER_FIGURE_MANIFEST_FILE,
    PAPER_FIGURES_MAIN_DIR,
    PAPER_FIGURES_SUPP_DIR,
    PAPER_MANIFESTS_DIR,
    PAPER_TABLE_MANIFEST_FILE,
    PAPER_TABLES_MAIN_CSV_DIR,
    PUBLICATION_PROCEDURE_SUMMARY_FILE,
    RANK_STABILITY_SUMMARY_FILE,
    SCRIPTS_DIR,
    SINGLE_MEASURE_FREEZE_MANIFEST_FILE,
    ensure_output_dirs,
)
from config.settings import (
    DEFAULT_RUN_MODE,
    get_run_profile,
    validate_all_settings,
)
from src.utils.logging_utils import get_logger
from src.utils.validation import (
    load_checkpoint,
    load_json,
    save_json,
    write_checkpoint,
    validate_environment,
)


# -----------------------------------------------------------------------------
# Stage definition
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    script_name: str
    checkpoint_path: Path
    optional_group: Optional[str] = None  # "gp", "downstream", "export"
    description: str = ""


PIPELINE_RESUME_STATE_FILE = PAPER_MANIFESTS_DIR / "pipeline_resume_state.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_stages() -> List[Stage]:
    return [
        Stage(
            script_name="01_build_stacks.py",
            checkpoint_path=OUTPUTS_DIR / "01_stacks" / "build_stacks.checkpoint.json",
            description="Build native-resolution stacks from raw datasets",
        ),
        Stage(
            script_name="02_build_reference_labels.py",
            checkpoint_path=OUTPUTS_DIR / "02_reference_labels" / "manifests" / "build_reference_labels.checkpoint.json",
            description="Build source/surrogate/leave-one-out labels",
        ),
        Stage(
            script_name="03_run_single_measure_benchmark.py",
            checkpoint_path=OUTPUTS_DIR / "03_single_measure_curves" / "run_single_measure_benchmark.checkpoint.json",
            description="Compute raw and normalized single-measure curves",
        ),
        Stage(
            script_name="04_evaluate_single_measures.py",
            checkpoint_path=OUTPUTS_DIR / "04_single_measure_eval" / "main" / "evaluate_single_measures.checkpoint.json",
            description="Evaluate single measures under corrected metric framework",
        ),
        Stage(
            script_name="05_plot_single_measure_results.py",
            checkpoint_path=OUTPUTS_DIR / "04_single_measure_eval" / "main" / "plot_single_measure_results.checkpoint.json",
            description="Generate single-measure benchmark figures",
        ),
        Stage(
            script_name="06_run_composite_gp_lodo.py",
            checkpoint_path=OUTPUTS_DIR / "05_gp_runs" / "summaries" / "run_composite_gp_lodo.checkpoint.json",
            optional_group="gp",
            description="Run corrected leave-one-dataset-out composite GP",
        ),
        Stage(
            script_name="07_evaluate_composites.py",
            checkpoint_path=OUTPUTS_DIR / "06_composite_eval" / "main" / "evaluate_composites.checkpoint.json",
            optional_group="gp",
            description="Evaluate composites under same metric framework as singles",
        ),
        Stage(
            script_name="08_run_statistics_and_sensitivity.py",
            checkpoint_path=OUTPUTS_DIR / "07_statistics" / "statistics" / "run_statistics_and_sensitivity.checkpoint.json",
            description="Run Friedman, Nemenyi, bootstrap, alpha and weight sensitivity",
        ),
        Stage(
            script_name="09_optional_downstream_baseline.py",
            checkpoint_path=PAPER_MANIFESTS_DIR / "optional_downstream_baseline.checkpoint.json",
            optional_group="downstream",
            description="Optional downstream / BSPC anchoring analysis",
        ),
        Stage(
            script_name="10_export_paper_assets.py",
            checkpoint_path=PAPER_MANIFESTS_DIR / "export_paper_assets.checkpoint.json",
            optional_group="export",
            description="Export final paper figures/tables/manifests",
        ),
    ]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected autofocus paper pipeline")

    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test profile")
    parser.add_argument("--full-run", action="store_true", help="Run full profile")

    parser.add_argument("--skip-gp", action="store_true", help="Skip GP search and composite evaluation stages")
    parser.add_argument("--skip-downstream", action="store_true", help="Skip optional downstream stage")
    parser.add_argument("--skip-export", action="store_true", help="Skip final paper asset export stage")

    parser.add_argument(
        "--export-paper-assets-only",
        action="store_true",
        help="Run only scripts/10_export_paper_assets.py",
    )
    parser.add_argument(
        "--run-inventory-first",
        action="store_true",
        help="Run scripts/00_inventory_repo.py before the pipeline if present",
    )
    parser.add_argument(
        "--publication-single-measure-only",
        action="store_true",
        help="Run the publication procedure only through stage 05 (single-measure foundation).",
    )
    parser.add_argument(
        "--publication-up-to-rank-stability",
        action="store_true",
        help="Run the publication procedure only through stage 04 (freeze + rank-stability gate).",
    )
    parser.add_argument(
        "--publication-up-to-gp",
        action="store_true",
        help="Run the publication procedure through stage 07 (composite evaluation) and stop before downstream/export.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if their checkpoints say complete",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the first incomplete, failed, or output-incomplete stage in the selected procedure.",
    )
    parser.add_argument(
        "--from-stage",
        type=str,
        default=None,
        help="Start from a specific script name, e.g. 04_evaluate_single_measures.py",
    )
    parser.add_argument(
        "--to-stage",
        type=str,
        default=None,
        help="Stop after a specific script name, e.g. 08_run_statistics_and_sensitivity.py",
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


# -----------------------------------------------------------------------------
# Execution helpers
# -----------------------------------------------------------------------------
def script_path(script_name: str) -> Path:
    return SCRIPTS_DIR / script_name


def checkpoint_run_mode(path: Path) -> str:
    payload = load_checkpoint(path)
    details = payload.get("details", {}) if payload else {}
    if isinstance(details, dict):
        return str(details.get("run_mode", "")).strip().lower()
    return str(payload.get("run_mode", "")).strip().lower() if payload else ""


def checkpoint_complete(path: Path, *, run_mode: Optional[str] = None) -> bool:
    payload = load_checkpoint(path)
    if not bool(payload and payload.get("status") == "complete"):
        return False
    if run_mode is not None:
        saved_run_mode = checkpoint_run_mode(path)
        if saved_run_mode and saved_run_mode != str(run_mode).strip().lower():
            return False
    return True


def checkpoint_status(path: Path) -> str:
    payload = load_checkpoint(path)
    status = str(payload.get("status", "")).strip().lower()
    return status


def build_common_args(run_mode: str) -> List[str]:
    if run_mode == "smoke":
        return ["--smoke-test"]
    return ["--full-run"]


def stage_specific_args(script_name: str, run_mode: str) -> List[str]:
    if script_name == "10_export_paper_assets.py":
        return ["--strict"] + (
            [] if run_mode != "smoke" else []
        )
    if run_mode != "smoke":
        return []
    if script_name == "07_evaluate_composites.py":
        return ["--max-eval-candidates", "1", "--top-k-composites", "1", "--skip-rrmse"]
    return []


def run_script(script_name: str, extra_args: List[str], logger) -> Dict[str, object]:
    path = script_path(script_name)

    if not path.exists():
        raise FileNotFoundError(f"Required stage script not found: {path}")

    cmd = [sys.executable, str(path)] + extra_args
    logger.info("Running stage: %s", script_name)
    logger.info("Command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    t1 = time.perf_counter()

    duration_sec = float(t1 - t0)

    if completed.returncode != 0:
        raise RuntimeError(f"Stage failed: {script_name}")

    return {
        "script_name": script_name,
        "status": "completed",
        "duration_sec": duration_sec,
        "command": cmd,
    }


def expected_stage_outputs(stage: Stage) -> List[Path]:
    if stage.script_name == "02_build_reference_labels.py":
        return [
            LABEL_PROVENANCE_MANIFEST_FILE,
            LOO_VOTER_DISCLOSURE_FILE,
        ]

    if stage.script_name == "04_evaluate_single_measures.py":
        return [
            SINGLE_MEASURE_FREEZE_MANIFEST_FILE,
            RANK_STABILITY_SUMMARY_FILE,
        ]

    if stage.script_name == "05_plot_single_measure_results.py":
        outputs = [
            PAPER_FIGURES_MAIN_DIR / "Fig2_single_measure_heatmap.png",
            PAPER_FIGURES_MAIN_DIR / "Fig3_top_operator_bootstrap_ci.png",
            PAPER_FIGURES_MAIN_DIR / "Fig5_representative_single_focus_curves.png",
            PAPER_FIGURES_SUPP_DIR / "SFig1_dataset_specific_focus_curves.png",
        ]
        timing_summary = OUTPUTS_DIR / "03_single_measure_curves" / "single_timing_summary.csv"
        if timing_summary.exists():
            outputs.extend([
                PAPER_FIGURES_MAIN_DIR / "Fig4_resolution_sensitivity.png",
                PAPER_FIGURES_MAIN_DIR / "Fig10_runtime_scaling.png",
            ])
        return outputs

    if stage.script_name == "08_run_statistics_and_sensitivity.py":
        return [
            PAPER_FIGURES_MAIN_DIR / "Fig7_nemenyi_cd_overall_rank.png",
            PAPER_FIGURES_MAIN_DIR / "Fig8_nemenyi_cd_accuracy_rank.png",
        ]

    if stage.script_name == "09_optional_downstream_baseline.py":
        return [
            PAPER_TABLES_MAIN_CSV_DIR / "Table11_optional_downstream_task.csv",
            PAPER_MANIFESTS_DIR / "optional_downstream_report.json",
        ]

    if stage.script_name == "10_export_paper_assets.py":
        return [
            PAPER_FIGURE_MANIFEST_FILE,
            PAPER_TABLE_MANIFEST_FILE,
            PAPER_ASSET_INDEX_FILE,
            PAPER_MANIFESTS_DIR / "paper_export_summary.json",
            PAPER_MANIFESTS_DIR / "asset_index.md",
        ]

    return []


def stage_outputs_complete(stage: Stage) -> bool:
    expected = expected_stage_outputs(stage)
    if expected and not all(path.exists() for path in expected):
        return False

    if stage.script_name == "10_export_paper_assets.py":
        try:
            figure_manifest = load_json(PAPER_FIGURE_MANIFEST_FILE)
            table_manifest = load_json(PAPER_TABLE_MANIFEST_FILE)
            asset_index = load_json(PAPER_ASSET_INDEX_FILE)
        except Exception:
            return False

        if not isinstance(figure_manifest, list):
            return False
        if not isinstance(table_manifest, list):
            return False
        if not isinstance(asset_index, dict):
            return False

    return True


def stage_resume_reason(stage: Stage, run_mode: str) -> Optional[str]:
    status = checkpoint_status(stage.checkpoint_path)
    outputs_complete = stage_outputs_complete(stage)
    saved_run_mode = checkpoint_run_mode(stage.checkpoint_path)
    if status == "complete" and outputs_complete:
        if saved_run_mode and saved_run_mode != str(run_mode).strip().lower():
            return f"checkpoint_run_mode={saved_run_mode}_expected_{run_mode}"
        return None
    if status in {"failed", "running", "warning"}:
        return f"checkpoint_status={status}"
    if status == "complete" and not outputs_complete:
        return "checkpoint_complete_but_outputs_incomplete"
    if status == "":
        return "checkpoint_missing"
    return f"checkpoint_status={status or 'unknown'}"


def resolve_resume_start_stage(stages: List[Stage], args: argparse.Namespace, logger, run_mode: str) -> Optional[str]:
    if args.from_stage is not None:
        logger.info("Resume requested but --from-stage is set explicitly; respecting --from-stage=%s", args.from_stage)
        return args.from_stage

    for stage in stages:
        if stage_should_be_skipped(stage, args):
            continue
        reason = stage_resume_reason(stage, run_mode)
        if reason is not None:
            logger.info("Resume selected stage %s (%s)", stage.script_name, reason)
            return stage.script_name

    logger.info("Resume found no incomplete stages in the selected procedure")
    return None


def write_pipeline_resume_state(payload: Dict[str, object]) -> None:
    PIPELINE_RESUME_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, PIPELINE_RESUME_STATE_FILE)


def maybe_run_inventory(run_mode: str, logger) -> None:
    inventory_path = script_path("00_inventory_repo.py")
    if not inventory_path.exists():
        logger.warning("Inventory script not found, skipping: %s", inventory_path)
        return

    cmd = [sys.executable, str(inventory_path)]
    logger.info("Running inventory stage before pipeline")
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise RuntimeError("Inventory stage failed: 00_inventory_repo.py")


def filter_stages_by_range(stages: List[Stage], from_stage: Optional[str], to_stage: Optional[str]) -> List[Stage]:
    names = [s.script_name for s in stages]

    if from_stage is not None and from_stage not in names:
        raise ValueError(f"--from-stage not found in pipeline: {from_stage}")
    if to_stage is not None and to_stage not in names:
        raise ValueError(f"--to-stage not found in pipeline: {to_stage}")

    start_idx = names.index(from_stage) if from_stage is not None else 0
    end_idx = names.index(to_stage) if to_stage is not None else len(stages) - 1

    if start_idx > end_idx:
        raise ValueError("--from-stage occurs after --to-stage")

    return stages[start_idx:end_idx + 1]


def stage_should_be_skipped(stage: Stage, args: argparse.Namespace) -> bool:
    if stage.optional_group == "gp" and args.skip_gp:
        return True
    if stage.optional_group == "downstream" and args.skip_downstream:
        return True
    if stage.optional_group == "export" and args.skip_export:
        return True
    return False


def apply_publication_shortcuts(args: argparse.Namespace) -> None:
    publication_flags = [
        args.publication_single_measure_only,
        args.publication_up_to_rank_stability,
        args.publication_up_to_gp,
    ]
    if sum(bool(flag) for flag in publication_flags) > 1:
        raise ValueError("Use at most one publication shortcut flag at a time")

    if args.publication_up_to_rank_stability:
        args.to_stage = "04_evaluate_single_measures.py"
        args.skip_gp = True
        args.skip_downstream = True
        args.skip_export = True

    if args.publication_single_measure_only:
        args.to_stage = "05_plot_single_measure_results.py"
        args.skip_gp = True
        args.skip_downstream = True
        args.skip_export = True

    if args.publication_up_to_gp:
        args.to_stage = "07_evaluate_composites.py"
        args.skip_downstream = True
        args.skip_export = True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    apply_publication_shortcuts(args)
    run_mode = resolve_run_mode(args)

    if args.export_paper_assets_only and args.skip_export:
        raise ValueError("--export-paper-assets-only cannot be combined with --skip-export")
    if args.resume and args.force:
        raise ValueError("--resume cannot be combined with --force")

    ensure_output_dirs()
    validate_all_settings()
    validate_environment()

    log_file = LOGS_DIR / f"full_pipeline_{run_mode}.log"
    logger = get_logger("full_pipeline", log_file=log_file)

    logger.info("Starting full pipeline")
    logger.info("Run mode: %s", run_mode)

    if args.run_inventory_first:
        maybe_run_inventory(run_mode, logger)

    common_args = build_common_args(run_mode)
    profile = get_run_profile(run_mode)

    stages = build_stages()
    stages = filter_stages_by_range(stages, args.from_stage, args.to_stage)

    if args.resume and not args.export_paper_assets_only:
        resume_from = resolve_resume_start_stage(stages, args, logger, run_mode)
        if resume_from is not None and args.from_stage != resume_from:
            args.from_stage = resume_from
            stages = filter_stages_by_range(build_stages(), args.from_stage, args.to_stage)

    if args.export_paper_assets_only:
        export_stage = next((s for s in stages if s.script_name == "10_export_paper_assets.py"), None)
        if export_stage is None:
            export_stage = next((s for s in build_stages() if s.script_name == "10_export_paper_assets.py"), None)
        if export_stage is None:
            raise RuntimeError("Could not find export stage in pipeline definition")

        stage_result = run_script(export_stage.script_name, common_args, logger)
        summary = {
            "run_mode": run_mode,
            "profile": profile,
            "export_only": True,
            "resume": args.resume,
            "results": [stage_result],
            "paper_dir": str(PAPER_DIR),
            "paper_export_summary_path": str(PAPER_MANIFESTS_DIR / "paper_export_summary.json"),
        }
        save_json(summary, PAPER_MANIFESTS_DIR / "full_pipeline_summary.json")
        write_pipeline_resume_state(
            {
                "timestamp_utc": _now_iso(),
                "run_mode": run_mode,
                "resume": args.resume,
                "status": "complete",
                "current_stage": "10_export_paper_assets.py",
                "next_stage": None,
                "results": [stage_result],
            }
        )
        save_json(
            {
                "run_mode": run_mode,
                "publication_order": [
                    "Phase A: 01 -> 02 -> 03 -> 04 (freeze + rank stability) -> 05",
                    "Phase B: 06 -> 07 -> 08",
                    "Phase C: 09 -> 10",
                ],
                "export_only": True,
                "summary_path": str(PAPER_MANIFESTS_DIR / "full_pipeline_summary.json"),
                "results": [stage_result],
            },
            PUBLICATION_PROCEDURE_SUMMARY_FILE,
        )
        logger.info("Export-only run complete")
        return

    stage_results: List[Dict[str, object]] = []
    write_pipeline_resume_state(
        {
            "timestamp_utc": _now_iso(),
            "run_mode": run_mode,
            "resume": args.resume,
            "status": "running",
            "current_stage": None,
            "next_stage": stages[0].script_name if stages else None,
            "from_stage": args.from_stage,
            "to_stage": args.to_stage,
            "skip_gp": args.skip_gp,
            "skip_downstream": args.skip_downstream,
            "skip_export": args.skip_export,
            "results": [],
        }
    )

    for stage in stages:
        if stage_should_be_skipped(stage, args):
            logger.info("Skipping stage by option: %s", stage.script_name)
            stage_results.append(
                {
                    "script_name": stage.script_name,
                    "status": "skipped_by_option",
                    "checkpoint_path": str(stage.checkpoint_path),
                    "description": stage.description,
                }
            )
            write_pipeline_resume_state(
                {
                    "timestamp_utc": _now_iso(),
                    "run_mode": run_mode,
                    "resume": args.resume,
                    "status": "running",
                    "current_stage": stage.script_name,
                    "next_stage": None,
                    "results": stage_results,
                }
            )
            continue

        if (not args.force) and checkpoint_complete(stage.checkpoint_path, run_mode=run_mode):
            if stage_outputs_complete(stage):
                logger.info("Skipping completed stage via checkpoint: %s", stage.script_name)
                stage_results.append(
                    {
                        "script_name": stage.script_name,
                        "status": "skipped_completed_checkpoint",
                        "checkpoint_path": str(stage.checkpoint_path),
                        "description": stage.description,
                    }
                )
                write_pipeline_resume_state(
                    {
                        "timestamp_utc": _now_iso(),
                        "run_mode": run_mode,
                        "resume": args.resume,
                        "status": "running",
                        "current_stage": stage.script_name,
                        "next_stage": None,
                        "results": stage_results,
                    }
                )
                continue
            logger.info(
                "Checkpoint exists but expected outputs are missing, rerunning: %s",
                stage.script_name,
            )

        stage_args = common_args + stage_specific_args(stage.script_name, run_mode)
        write_checkpoint(
            checkpoint_path=stage.checkpoint_path,
            stage=stage.script_name.replace(".py", ""),
            status="running",
            details={
                "command": [sys.executable, str(script_path(stage.script_name)), *stage_args],
                "run_mode": run_mode,
                "resume": args.resume,
                "description": stage.description,
            },
        )
        write_pipeline_resume_state(
            {
                "timestamp_utc": _now_iso(),
                "run_mode": run_mode,
                "resume": args.resume,
                "status": "running",
                "current_stage": stage.script_name,
                "next_stage": stage.script_name,
                "results": stage_results,
            }
        )
        try:
            result = run_script(stage.script_name, stage_args, logger)
        except Exception as exc:
            write_checkpoint(
                checkpoint_path=stage.checkpoint_path,
                stage=stage.script_name.replace(".py", ""),
                status="failed",
                details={
                    "run_mode": run_mode,
                    "resume": args.resume,
                    "description": stage.description,
                    "error": str(exc),
                    "command": [sys.executable, str(script_path(stage.script_name)), *stage_args],
                },
            )
            if stage.optional_group == "downstream":
                logger.warning("Optional downstream stage failed and will not block the pipeline: %s", exc)
                stage_results.append(
                    {
                        "script_name": stage.script_name,
                        "status": "optional_failed",
                        "checkpoint_path": str(stage.checkpoint_path),
                        "description": stage.description,
                        "error": str(exc),
                    }
                )
                write_pipeline_resume_state(
                    {
                        "timestamp_utc": _now_iso(),
                        "run_mode": run_mode,
                        "resume": args.resume,
                        "status": "running",
                        "current_stage": stage.script_name,
                        "next_stage": None,
                        "results": stage_results,
                    }
                )
                continue
            write_pipeline_resume_state(
                {
                    "timestamp_utc": _now_iso(),
                    "run_mode": run_mode,
                    "resume": args.resume,
                    "status": "failed",
                    "current_stage": stage.script_name,
                    "next_stage": stage.script_name,
                    "error": str(exc),
                    "results": stage_results,
                }
            )
            raise

        result["checkpoint_path"] = str(stage.checkpoint_path)
        result["description"] = stage.description
        stage_results.append(result)
        remaining = [s.script_name for s in stages[stages.index(stage) + 1:] if not stage_should_be_skipped(s, args)]
        write_pipeline_resume_state(
            {
                "timestamp_utc": _now_iso(),
                "run_mode": run_mode,
                "resume": args.resume,
                "status": "running",
                "current_stage": stage.script_name,
                "next_stage": remaining[0] if remaining else None,
                "results": stage_results,
            }
        )

    summary = {
        "run_mode": run_mode,
        "profile": profile,
        "force": args.force,
        "resume": args.resume,
        "skip_gp": args.skip_gp,
        "skip_downstream": args.skip_downstream,
        "skip_export": args.skip_export,
        "publication_single_measure_only": args.publication_single_measure_only,
        "publication_up_to_rank_stability": args.publication_up_to_rank_stability,
        "publication_up_to_gp": args.publication_up_to_gp,
        "from_stage": args.from_stage,
        "to_stage": args.to_stage,
        "results": stage_results,
        "paper_dir": str(PAPER_DIR),
        "paper_export_summary_path": str(PAPER_MANIFESTS_DIR / "paper_export_summary.json"),
        "optional_downstream_summary_path": str(PAPER_MANIFESTS_DIR / "optional_downstream_report.json"),
    }

    PAPER_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PAPER_MANIFESTS_DIR / "full_pipeline_summary.json"
    save_json(summary, summary_path)
    write_pipeline_resume_state(
        {
            "timestamp_utc": _now_iso(),
            "run_mode": run_mode,
            "resume": args.resume,
            "status": "complete",
            "current_stage": None,
            "next_stage": None,
            "summary_path": str(summary_path),
            "results": stage_results,
        }
    )
    publication_summary = {
        "run_mode": run_mode,
        "publication_order": [
            "Phase A: 01 -> 02 -> 03 -> 04 (freeze + rank stability) -> 05",
            "Phase B: 06 -> 07 -> 08",
            "Phase C: 09 -> 10",
        ],
        "single_measure_freeze_manifest_exists": SINGLE_MEASURE_FREEZE_MANIFEST_FILE.exists(),
        "rank_stability_summary_exists": RANK_STABILITY_SUMMARY_FILE.exists(),
        "phase_a_ready_for_gp": SINGLE_MEASURE_FREEZE_MANIFEST_FILE.exists() and RANK_STABILITY_SUMMARY_FILE.exists(),
        "summary_path": str(summary_path),
        "results": stage_results,
        "resume_state_path": str(PIPELINE_RESUME_STATE_FILE),
    }
    save_json(publication_summary, PUBLICATION_PROCEDURE_SUMMARY_FILE)

    logger.info("Pipeline complete")
    logger.info("Summary -> %s", summary_path)
    logger.info("Publication procedure summary -> %s", PUBLICATION_PROCEDURE_SUMMARY_FILE)
    logger.info("Paper assets dir -> %s", PAPER_DIR)


if __name__ == "__main__":
    main()
