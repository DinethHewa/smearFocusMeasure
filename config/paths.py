"""Canonical project paths for the corrected Q1 scaffold."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# -----------------------------------------------------------------------------
# Canonical raw datasets
# -----------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("FOCUS_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()

RAW_DATASETS: Dict[str, str] = {
    dataset_name: str(
        Path(
            os.environ.get(
                f"FOCUS_{dataset_name}_DIR",
                DATA_ROOT / dataset_name,
            )
        ).expanduser()
    )
    for dataset_name in ("WBC", "TBI", "PBS", "BMA", "TBF")
}

DATASET_ORDER: Tuple[str, ...] = ("WBC", "TBI", "PBS", "BMA", "TBF")


# -----------------------------------------------------------------------------
# Output roots
# -----------------------------------------------------------------------------
LOGS_DIR = OUTPUTS_DIR / "00_logs"

STACKS_DIR = OUTPUTS_DIR / "01_stacks"
STACK_ARRAYS_DIR = STACKS_DIR / "arrays"
STACK_METADATA_DIR = STACKS_DIR / "metadata"

REFERENCE_LABELS_DIR = OUTPUTS_DIR / "02_reference_labels"
LABEL_SOURCE_DIR = REFERENCE_LABELS_DIR / "source"
LABEL_SURROGATE_DIR = REFERENCE_LABELS_DIR / "surrogate"
LABEL_LOO_DIR = REFERENCE_LABELS_DIR / "leave_one_out"
LABEL_MANIFESTS_DIR = REFERENCE_LABELS_DIR / "manifests"

SINGLE_CURVES_DIR = OUTPUTS_DIR / "03_single_measure_curves"
SINGLE_RAW_CURVES_DIR = SINGLE_CURVES_DIR / "raw"
SINGLE_NORM_CURVES_DIR = SINGLE_CURVES_DIR / "normalized"
SINGLE_TIMING_DIR = SINGLE_CURVES_DIR / "timing"

SINGLE_EVAL_DIR = OUTPUTS_DIR / "04_single_measure_eval"
SINGLE_EVAL_MAIN_DIR = SINGLE_EVAL_DIR / "main"
SINGLE_EVAL_SUPP_DIR = SINGLE_EVAL_DIR / "supplementary"

GP_STAGE_DIR = OUTPUTS_DIR / "05_gp_runs"
GP_RUNS_DIR = GP_STAGE_DIR / "folds"
GP_SUMMARIES_DIR = GP_STAGE_DIR / "summaries"
GP_DEDUP_DIR = GP_STAGE_DIR / "dedup"

COMPOSITE_EVAL_DIR = OUTPUTS_DIR / "06_composite_eval"
COMPOSITE_MAIN_DIR = COMPOSITE_EVAL_DIR / "main"
COMPOSITE_SUPP_DIR = COMPOSITE_EVAL_DIR / "supplementary"

STATS_STAGE_DIR = OUTPUTS_DIR / "07_statistics"
STATISTICS_DIR = STATS_STAGE_DIR / "statistics"
SENSITIVITY_DIR = STATS_STAGE_DIR / "sensitivity"

PAPER_DIR = OUTPUTS_DIR / "09_paper"
PAPER_CAPTIONS_DIR = PAPER_DIR / "captions"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_FIGURES_MAIN_DIR = PAPER_FIGURES_DIR / "main"
PAPER_FIGURES_SUPP_DIR = PAPER_FIGURES_DIR / "supplementary"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_TABLES_MAIN_DIR = PAPER_TABLES_DIR / "main"
PAPER_TABLES_MAIN_CSV_DIR = PAPER_TABLES_MAIN_DIR / "csv"
PAPER_TABLES_MAIN_LATEX_DIR = PAPER_TABLES_MAIN_DIR / "latex"
PAPER_TABLES_SUPP_DIR = PAPER_TABLES_DIR / "supplementary"
PAPER_TABLES_SUPP_CSV_DIR = PAPER_TABLES_SUPP_DIR / "csv"
PAPER_TABLES_SUPP_LATEX_DIR = PAPER_TABLES_SUPP_DIR / "latex"
PAPER_MANIFESTS_DIR = PAPER_DIR / "manifests"


# -----------------------------------------------------------------------------
# Stage-level manifest files
# -----------------------------------------------------------------------------
LABEL_SOURCE_MANIFEST_FILE = LABEL_MANIFESTS_DIR / "label_source_manifest.json"
LABEL_SUMMARY_FILE = LABEL_MANIFESTS_DIR / "label_summary.json"
LABEL_PROVENANCE_MANIFEST_FILE = LABEL_MANIFESTS_DIR / "label_provenance_disclosure.json"
LOO_VOTER_DISCLOSURE_FILE = LABEL_MANIFESTS_DIR / "loo_voter_disclosure.json"

SINGLE_OPERATOR_MANIFEST_FILE = SINGLE_CURVES_DIR / "single_operator_manifest.json"
SINGLE_TIMING_SUMMARY_FILE = SINGLE_CURVES_DIR / "single_timing_summary.csv"

SINGLE_MEASURE_FREEZE_MANIFEST_FILE = SINGLE_EVAL_MAIN_DIR / "single_measure_results_freeze.json"
RANK_STABILITY_SUMMARY_FILE = SINGLE_EVAL_MAIN_DIR / "rank_stability_summary.json"
RANK_STABILITY_CHECKPOINT_FILE = SINGLE_EVAL_MAIN_DIR / "rank_stability.checkpoint.json"
RANK_STABILITY_RESULTS_FILE = SINGLE_EVAL_SUPP_DIR / "rank_stability_1024.csv"
RANK_STABILITY_DETAIL_FILE = SINGLE_EVAL_SUPP_DIR / "rank_stability_detail_by_dataset.csv"
LABEL_USAGE_DISCLOSURE_FILE = SINGLE_EVAL_SUPP_DIR / "label_source_usage_by_measure.csv"
LABEL_USAGE_DISCLOSURE_JSON = SINGLE_EVAL_SUPP_DIR / "label_source_usage_by_measure.json"

COMPOSITE_METRIC_PROFILE_FILE = COMPOSITE_SUPP_DIR / "top_composites_10_metric_profile.csv"
COMPOSITE_FIGURE_MANIFEST_FILE = COMPOSITE_MAIN_DIR / "composite_figure_manifest.json"

PAPER_FIGURE_MANIFEST_FILE = PAPER_MANIFESTS_DIR / "figure_manifest.json"
PAPER_TABLE_MANIFEST_FILE = PAPER_MANIFESTS_DIR / "table_manifest.json"
PAPER_ASSET_INDEX_FILE = PAPER_MANIFESTS_DIR / "asset_index.json"
PUBLICATION_PROCEDURE_SUMMARY_FILE = PAPER_MANIFESTS_DIR / "publication_procedure_summary.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def measure_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip())
    slug = slug.strip("_").lower()
    return slug or "unnamed_measure"


def dataset_path(dataset_name: str) -> Path:
    if dataset_name not in RAW_DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    return Path(RAW_DATASETS[dataset_name]).expanduser().resolve()


def _dataset_dirs() -> Iterable[Path]:
    for dataset_name in DATASET_ORDER:
        yield SINGLE_RAW_CURVES_DIR / dataset_name
        yield SINGLE_NORM_CURVES_DIR / dataset_name
        yield SINGLE_TIMING_DIR / dataset_name
        yield LABEL_LOO_DIR / dataset_name


def ensure_output_dirs() -> Dict[str, Path]:
    created = {
        "reports": ensure_dir(REPORTS_DIR),
        "logs": ensure_dir(LOGS_DIR),
        "stack_arrays": ensure_dir(STACK_ARRAYS_DIR),
        "stack_metadata": ensure_dir(STACK_METADATA_DIR),
        "label_source": ensure_dir(LABEL_SOURCE_DIR),
        "label_surrogate": ensure_dir(LABEL_SURROGATE_DIR),
        "label_loo": ensure_dir(LABEL_LOO_DIR),
        "label_manifests": ensure_dir(LABEL_MANIFESTS_DIR),
        "single_curves": ensure_dir(SINGLE_CURVES_DIR),
        "single_raw": ensure_dir(SINGLE_RAW_CURVES_DIR),
        "single_norm": ensure_dir(SINGLE_NORM_CURVES_DIR),
        "single_timing": ensure_dir(SINGLE_TIMING_DIR),
        "single_eval_main": ensure_dir(SINGLE_EVAL_MAIN_DIR),
        "single_eval_supp": ensure_dir(SINGLE_EVAL_SUPP_DIR),
        "gp_runs": ensure_dir(GP_RUNS_DIR),
        "gp_summaries": ensure_dir(GP_SUMMARIES_DIR),
        "gp_dedup": ensure_dir(GP_DEDUP_DIR),
        "composite_main": ensure_dir(COMPOSITE_MAIN_DIR),
        "composite_supp": ensure_dir(COMPOSITE_SUPP_DIR),
        "statistics": ensure_dir(STATISTICS_DIR),
        "sensitivity": ensure_dir(SENSITIVITY_DIR),
        "paper_captions": ensure_dir(PAPER_CAPTIONS_DIR),
        "paper_figures_main": ensure_dir(PAPER_FIGURES_MAIN_DIR),
        "paper_figures_supp": ensure_dir(PAPER_FIGURES_SUPP_DIR),
        "paper_tables_main_csv": ensure_dir(PAPER_TABLES_MAIN_CSV_DIR),
        "paper_tables_main_latex": ensure_dir(PAPER_TABLES_MAIN_LATEX_DIR),
        "paper_tables_supp_csv": ensure_dir(PAPER_TABLES_SUPP_CSV_DIR),
        "paper_tables_supp_latex": ensure_dir(PAPER_TABLES_SUPP_LATEX_DIR),
        "paper_manifests": ensure_dir(PAPER_MANIFESTS_DIR),
    }
    for path in _dataset_dirs():
        ensure_dir(path)
    return created


def get_stack_file(dataset_name: str) -> Path:
    return STACK_ARRAYS_DIR / f"{dataset_name}_stacks.npy"


def get_stack_metadata_file(dataset_name: str) -> Path:
    return STACK_METADATA_DIR / f"{dataset_name}_stack_metadata.json"


def get_source_label_file(dataset_name: str) -> Path:
    return LABEL_SOURCE_DIR / f"{dataset_name}_source_labels.npy"


def get_surrogate_label_file(dataset_name: str) -> Path:
    return LABEL_SURROGATE_DIR / f"{dataset_name}_surrogate_labels.npy"


def get_loo_label_file(dataset_name: str, measure_name: str) -> Path:
    return LABEL_LOO_DIR / dataset_name / f"{measure_slug(measure_name)}_loo_labels.npy"


def get_single_raw_curve_file(dataset_name: str, measure_name: str) -> Path:
    return SINGLE_RAW_CURVES_DIR / dataset_name / f"{measure_slug(measure_name)}.npy"


def get_single_norm_curve_file(dataset_name: str, measure_name: str) -> Path:
    return SINGLE_NORM_CURVES_DIR / dataset_name / f"{measure_slug(measure_name)}.npy"


def get_single_timing_file(dataset_name: str, measure_name: str) -> Path:
    return SINGLE_TIMING_DIR / dataset_name / f"{measure_slug(measure_name)}.json"


def get_gp_fold_dir(held_out_dataset: str, seed: int) -> Path:
    return GP_RUNS_DIR / f"heldout_{held_out_dataset}" / f"seed_{seed}"


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "SRC_DIR",
    "SCRIPTS_DIR",
    "REPORTS_DIR",
    "OUTPUTS_DIR",
    "RAW_DATASETS",
    "DATASET_ORDER",
    "LOGS_DIR",
    "STACKS_DIR",
    "STACK_ARRAYS_DIR",
    "STACK_METADATA_DIR",
    "REFERENCE_LABELS_DIR",
    "LABEL_SOURCE_DIR",
    "LABEL_SURROGATE_DIR",
    "LABEL_LOO_DIR",
    "LABEL_MANIFESTS_DIR",
    "SINGLE_CURVES_DIR",
    "SINGLE_RAW_CURVES_DIR",
    "SINGLE_NORM_CURVES_DIR",
    "SINGLE_TIMING_DIR",
    "SINGLE_EVAL_DIR",
    "SINGLE_EVAL_MAIN_DIR",
    "SINGLE_EVAL_SUPP_DIR",
    "GP_STAGE_DIR",
    "GP_RUNS_DIR",
    "GP_SUMMARIES_DIR",
    "GP_DEDUP_DIR",
    "COMPOSITE_EVAL_DIR",
    "COMPOSITE_MAIN_DIR",
    "COMPOSITE_SUPP_DIR",
    "STATS_STAGE_DIR",
    "STATISTICS_DIR",
    "SENSITIVITY_DIR",
    "PAPER_DIR",
    "PAPER_CAPTIONS_DIR",
    "PAPER_FIGURES_DIR",
    "PAPER_FIGURES_MAIN_DIR",
    "PAPER_FIGURES_SUPP_DIR",
    "PAPER_TABLES_DIR",
    "PAPER_TABLES_MAIN_DIR",
    "PAPER_TABLES_MAIN_CSV_DIR",
    "PAPER_TABLES_MAIN_LATEX_DIR",
    "PAPER_TABLES_SUPP_DIR",
    "PAPER_TABLES_SUPP_CSV_DIR",
    "PAPER_TABLES_SUPP_LATEX_DIR",
    "PAPER_MANIFESTS_DIR",
    "LABEL_SOURCE_MANIFEST_FILE",
    "LABEL_SUMMARY_FILE",
    "LABEL_PROVENANCE_MANIFEST_FILE",
    "LOO_VOTER_DISCLOSURE_FILE",
    "SINGLE_OPERATOR_MANIFEST_FILE",
    "SINGLE_TIMING_SUMMARY_FILE",
    "SINGLE_MEASURE_FREEZE_MANIFEST_FILE",
    "RANK_STABILITY_SUMMARY_FILE",
    "RANK_STABILITY_CHECKPOINT_FILE",
    "RANK_STABILITY_RESULTS_FILE",
    "RANK_STABILITY_DETAIL_FILE",
    "LABEL_USAGE_DISCLOSURE_FILE",
    "LABEL_USAGE_DISCLOSURE_JSON",
    "COMPOSITE_METRIC_PROFILE_FILE",
    "COMPOSITE_FIGURE_MANIFEST_FILE",
    "PAPER_FIGURE_MANIFEST_FILE",
    "PAPER_TABLE_MANIFEST_FILE",
    "PAPER_ASSET_INDEX_FILE",
    "PUBLICATION_PROCEDURE_SUMMARY_FILE",
    "ensure_dir",
    "measure_slug",
    "dataset_path",
    "ensure_output_dirs",
    "get_stack_file",
    "get_stack_metadata_file",
    "get_source_label_file",
    "get_surrogate_label_file",
    "get_loo_label_file",
    "get_single_raw_curve_file",
    "get_single_norm_curve_file",
    "get_single_timing_file",
    "get_gp_fold_dir",
]
