# config/paper_assets.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from config.paths import (
    PAPER_CAPTIONS_DIR,
    PAPER_FIGURE_MANIFEST_FILE,
    PAPER_FIGURES_MAIN_DIR,
    PAPER_FIGURES_SUPP_DIR,
    PAPER_MANIFESTS_DIR,
    PAPER_TABLE_MANIFEST_FILE,
    PAPER_TABLES_MAIN_CSV_DIR,
    PAPER_TABLES_MAIN_LATEX_DIR,
    PAPER_TABLES_SUPP_CSV_DIR,
    PAPER_TABLES_SUPP_LATEX_DIR,
)


# -----------------------------------------------------------------------------
# Asset specs
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSpec:
    key: str
    title: str
    section: str
    output_group: str  # "main" or "supplementary"
    description: str = ""


@dataclass(frozen=True)
class TableSpec:
    key: str
    title: str
    section: str
    output_group: str  # "main" or "supplementary"
    description: str = ""


# -----------------------------------------------------------------------------
# Canonical figure registry
# -----------------------------------------------------------------------------
MAIN_FIGURES: List[FigureSpec] = [
    FigureSpec(
        key="Fig1_pipeline_overview",
        title="Corrected study pipeline",
        section="Methods",
        output_group="main",
        description="End-to-end corrected benchmarking and composite synthesis workflow.",
    ),
    FigureSpec(
        key="Fig2_single_measure_heatmap",
        title="Single-measure benchmark heatmap",
        section="Results",
        output_group="main",
        description="Cross-dataset summary heatmap for single handcrafted focus measures.",
    ),
    FigureSpec(
        key="Fig3_top_operator_bootstrap_ci",
        title="Bootstrap confidence intervals for top operators",
        section="Results",
        output_group="main",
        description="Confidence interval summary for leading single operators.",
    ),
    FigureSpec(
        key="Fig4_resolution_sensitivity",
        title="Resolution sensitivity of top operators",
        section="Results",
        output_group="main",
        description="Ranking or metric variation across native and reduced resolutions.",
    ),
    FigureSpec(
        key="Fig5_representative_single_focus_curves",
        title="Representative single-measure focus curves",
        section="Results",
        output_group="main",
        description="Normalized focus curves for selected top single measures.",
    ),
    FigureSpec(
        key="Fig6_representative_composite_vs_single_curves",
        title="Representative composite versus single focus curves",
        section="Results",
        output_group="main",
        description="Normalized focus curves comparing the best composite against key single measures.",
    ),
    FigureSpec(
        key="Fig7_nemenyi_cd_overall_rank",
        title="Nemenyi critical-difference diagram for overall rank",
        section="Results",
        output_group="main",
        description="Critical-difference diagram for overall rank-based comparison.",
    ),
    FigureSpec(
        key="Fig8_nemenyi_cd_accuracy_rank",
        title="Nemenyi critical-difference diagram for localization accuracy",
        section="Results",
        output_group="main",
        description="Critical-difference diagram for accuracy-specific ranking.",
    ),
    FigureSpec(
        key="Fig9_gp_lodo_summary",
        title="Leave-one-dataset-out GP summary",
        section="Results",
        output_group="main",
        description="Foldwise held-out performance and best-expression summary for corrected GP.",
    ),
    FigureSpec(
        key="Fig10_runtime_scaling",
        title="Runtime scaling across image resolutions",
        section="Results",
        output_group="main",
        description="Operator runtime comparison across native and reduced resolutions.",
    ),
    FigureSpec(
        key="Fig11_optional_downstream_task",
        title="Downstream biomedical anchor analysis",
        section="Results",
        output_group="main",
        description="Optional downstream or proxy biomedical anchor comparing poor focus, Brenner Gradient, and the best composite.",
    ),
    FigureSpec(
        key="Fig12_lodo_vs_final_refit",
        title="LODO validation versus final all-dataset GP refit",
        section="Results",
        output_group="main",
        description="Comparison between leave-one-dataset-out held-out GP performance and the final all-dataset refit used as the proposed composite.",
    ),
]

SUPPLEMENTARY_FIGURES: List[FigureSpec] = [
    FigureSpec(
        key="SFig1_dataset_specific_focus_curves",
        title="Dataset-specific representative focus curves",
        section="Supplementary",
        output_group="supplementary",
        description="One or more representative normalized focus-curve plots per dataset.",
    ),
    FigureSpec(
        key="SFig2_gp_convergence",
        title="GP convergence across folds and seeds",
        section="Supplementary",
        output_group="supplementary",
        description="Convergence traces for corrected leave-one-dataset-out GP runs.",
    ),
    FigureSpec(
        key="SFig3_terminal_frequency",
        title="Terminal usage frequency in GP solutions",
        section="Supplementary",
        output_group="supplementary",
        description="Frequency of terminal usage across retained composite expressions.",
    ),
    FigureSpec(
        key="SFig4_expression_equivalence_clusters",
        title="Functional equivalence clustering of composite expressions",
        section="Supplementary",
        output_group="supplementary",
        description="Correlation-based clustering of near-equivalent composite expressions.",
    ),
    FigureSpec(
        key="SFig5_gp_convergence_by_fold",
        title="Generation-wise GP convergence by held-out fold",
        section="Supplementary",
        output_group="supplementary",
        description="Mean and seed-level convergence traces for each leave-one-dataset-out GP fold.",
    ),
    FigureSpec(
        key="SFig6_final_refit_convergence",
        title="Generation-wise convergence of the final all-dataset GP refit",
        section="Supplementary",
        output_group="supplementary",
        description="Best objective value over generations for the final all-dataset refit.",
    ),
    FigureSpec(
        key="SFig7_gp_seedwise_score_distribution",
        title="Seed-wise GP score distribution across LODO and final refit",
        section="Supplementary",
        output_group="supplementary",
        description="Distribution of seed-level GP scores for held-out folds and final all-dataset refit.",
    ),
]


# -----------------------------------------------------------------------------
# Canonical table registry
# -----------------------------------------------------------------------------
MAIN_TABLES: List[TableSpec] = [
    TableSpec(
        key="Table2_dataset_summary",
        title="Dataset summary",
        section="Methods",
        output_group="main",
        description="Summary of all benchmark datasets and their acquisition characteristics.",
    ),
    TableSpec(
        key="Table3_metric_definitions",
        title="Autofocus metric definitions",
        section="Methods",
        output_group="main",
        description="Definitions of the quantitative autofocus evaluation metrics.",
    ),
    TableSpec(
        key="Table4_metric_weights_and_alpha",
        title="Metric weights and alpha settings",
        section="Methods",
        output_group="main",
        description="Scalarization settings for corrected value-based analysis.",
    ),
    TableSpec(
        key="Table5_top10_single_rank_based",
        title="Top-10 single measures under rank-based analysis",
        section="Results",
        output_group="main",
        description="Final top-10 operator list under corrected rank-based evaluation.",
    ),
    TableSpec(
        key="Table6_top10_single_value_based",
        title="Top-10 single measures under normalized value-based analysis",
        section="Results",
        output_group="main",
        description="Final top-10 operator list under corrected normalized value-based evaluation.",
    ),
    TableSpec(
        key="Table7_top10_composites_common_scoring",
        title="Top-10 composites under common scoring regime",
        section="Results",
        output_group="main",
        description="Top composite expressions under the same corrected scoring regime as singles.",
    ),
    TableSpec(
        key="Table8_true_vs_surrogate_label_split",
        title="Results split by source-label and surrogate-label datasets",
        section="Results",
        output_group="main",
        description="Comparison of performance on datasets with source labels versus surrogate labels.",
    ),
    TableSpec(
        key="Table9_equal_vs_stack_weighted_comparison",
        title="Equal-weighted versus per-stack-weighted comparison",
        section="Results",
        output_group="main",
        description="Sensitivity of aggregate conclusions to weighting choice.",
    ),
    TableSpec(
        key="Table10_runtime_multi_resolution",
        title="Runtime comparison across image resolutions",
        section="Results",
        output_group="main",
        description="Execution time comparison across native and reduced resolutions.",
    ),
    TableSpec(
        key="Table11_optional_downstream_task",
        title="Downstream biomedical anchor summary",
        section="Results",
        output_group="main",
        description="Performance summary for the downstream biomedical anchor or proxy fallback analysis.",
    ),
    TableSpec(
        key="Table12_final_generalized_composite",
        title="Final generalized composite focus measure",
        section="Results",
        output_group="main",
        description="Final all-dataset GP refit expression proposed for general use after LODO validation.",
    ),
]

SUPPLEMENTARY_TABLES: List[TableSpec] = [
    TableSpec(
        key="STable1_all_single_measure_results",
        title="Full single-measure results for all operators",
        section="Supplementary",
        output_group="supplementary",
        description="Complete single-measure benchmark results for all handcrafted operators.",
    ),
    TableSpec(
        key="STable2_alpha_sensitivity",
        title="Alpha sensitivity analysis",
        section="Supplementary",
        output_group="supplementary",
        description="Sensitivity of corrected value-based rankings to alpha selection.",
    ),
    TableSpec(
        key="STable3_metric_weight_sensitivity",
        title="Metric-weight sensitivity analysis",
        section="Supplementary",
        output_group="supplementary",
        description="Sensitivity of corrected rankings to alternate metric-weight schemes.",
    ),
    TableSpec(
        key="STable4_gp_foldwise_results",
        title="Foldwise leave-one-dataset-out GP results",
        section="Supplementary",
        output_group="supplementary",
        description="Held-out foldwise performance and best expressions from corrected GP.",
    ),
    TableSpec(
        key="STable5_composite_deduplication",
        title="Composite deduplication and equivalence summary",
        section="Supplementary",
        output_group="supplementary",
        description="Deduplication summary for syntactic and functional composite redundancy.",
    ),
    TableSpec(
        key="STable6_label_provenance_and_loo_disclosure",
        title="Dataset label provenance and leave-one-out disclosure",
        section="Supplementary",
        output_group="supplementary",
        description="Per-dataset disclosure of source-label availability, surrogate-label fallback, and leave-one-out voter usage.",
    ),
    TableSpec(
        key="STable7_rank_stability_summary",
        title="Native versus 1024 rank-stability study",
        section="Supplementary",
        output_group="supplementary",
        description="Rank-stability summary for top single measures under native-subset versus 1024-resolution reruns.",
    ),
    TableSpec(
        key="STable8_top_composites_10_metric_profile",
        title="Top composites under the full 10 autofocus metrics",
        section="Supplementary",
        output_group="supplementary",
        description="Per-composite metric profile under the same 10 autofocus metrics used for single measures.",
    ),
    TableSpec(
        key="STable9_gp_lodo_vs_final_refit",
        title="LODO GP versus final all-dataset refit summary",
        section="Supplementary",
        output_group="supplementary",
        description="Protocol-level comparison of held-out LODO results and the final all-dataset GP refit.",
    ),
    TableSpec(
        key="STable10_gp_generation_traces",
        title="Generation-wise GP progress traces",
        section="Supplementary",
        output_group="supplementary",
        description="Per-generation best objective values for all LODO and final-refit GP seeds.",
    ),
    TableSpec(
        key="STable11_gp_seedwise_lodo_and_final",
        title="Seed-wise LODO and final-refit GP results",
        section="Supplementary",
        output_group="supplementary",
        description="Seed-level best expressions, scores, and complexity for all LODO folds and final refit runs.",
    ),
    TableSpec(
        key="STable12_gp_terminal_selection_lodo_and_final",
        title="Terminal selection for LODO folds and final refit",
        section="Supplementary",
        output_group="supplementary",
        description="Selected GP terminals, ranking sources, and terminal families for each held-out fold and final all-dataset refit.",
    ),
]


# -----------------------------------------------------------------------------
# Combined registries
# -----------------------------------------------------------------------------
ALL_FIGURES: List[FigureSpec] = MAIN_FIGURES + SUPPLEMENTARY_FIGURES
ALL_TABLES: List[TableSpec] = MAIN_TABLES + SUPPLEMENTARY_TABLES


# -----------------------------------------------------------------------------
# Basic accessors
# -----------------------------------------------------------------------------
def build_figure_manifest() -> List[Dict]:
    return [asdict(x) for x in ALL_FIGURES]


def build_table_manifest() -> List[Dict]:
    return [asdict(x) for x in ALL_TABLES]


def get_figure_output_dir(group: str) -> Path:
    if group == "main":
        return PAPER_FIGURES_MAIN_DIR
    if group == "supplementary":
        return PAPER_FIGURES_SUPP_DIR
    raise ValueError(f"Unknown figure output group: {group}")


def get_table_output_dirs(group: str) -> Dict[str, Path]:
    if group == "main":
        return {
            "csv": PAPER_TABLES_MAIN_CSV_DIR,
            "latex": PAPER_TABLES_MAIN_LATEX_DIR,
        }
    if group == "supplementary":
        return {
            "csv": PAPER_TABLES_SUPP_CSV_DIR,
            "latex": PAPER_TABLES_SUPP_LATEX_DIR,
        }
    raise ValueError(f"Unknown table output group: {group}")


def get_figure_spec(key: str) -> FigureSpec:
    for spec in ALL_FIGURES:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown figure key: {key}")


def get_table_spec(key: str) -> TableSpec:
    for spec in ALL_TABLES:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown table key: {key}")


def list_figure_keys(group: str | None = None) -> List[str]:
    figures = ALL_FIGURES if group is None else [x for x in ALL_FIGURES if x.output_group == group]
    return [x.key for x in figures]


def list_table_keys(group: str | None = None) -> List[str]:
    tables = ALL_TABLES if group is None else [x for x in ALL_TABLES if x.output_group == group]
    return [x.key for x in tables]


# -----------------------------------------------------------------------------
# Expected file helpers
# -----------------------------------------------------------------------------
def expected_figure_paths(extensions: Sequence[str] = (".png", ".pdf")) -> List[Path]:
    paths: List[Path] = []
    for spec in ALL_FIGURES:
        out_dir = get_figure_output_dir(spec.output_group)
        for ext in extensions:
            paths.append(out_dir / f"{spec.key}{ext}")
    return paths


def expected_table_paths() -> List[Path]:
    paths: List[Path] = []
    for spec in ALL_TABLES:
        dirs = get_table_output_dirs(spec.output_group)
        paths.append(dirs["csv"] / f"{spec.key}.csv")
        paths.append(dirs["latex"] / f"{spec.key}.tex")
    return paths


def expected_caption_paths() -> List[Path]:
    paths: List[Path] = []
    for spec in ALL_FIGURES:
        paths.append(PAPER_CAPTIONS_DIR / f"{spec.key}.md")
    for spec in ALL_TABLES:
        paths.append(PAPER_CAPTIONS_DIR / f"{spec.key}.md")
    return paths


# -----------------------------------------------------------------------------
# Manifest-friendly enrichment
# -----------------------------------------------------------------------------
def build_enriched_figure_manifest(
    extensions: Sequence[str] = (".png", ".pdf"),
) -> List[Dict]:
    manifest: List[Dict] = []
    for spec in ALL_FIGURES:
        entry = asdict(spec)
        out_dir = get_figure_output_dir(spec.output_group)
        entry["caption_file"] = str(PAPER_CAPTIONS_DIR / f"{spec.key}.md")
        entry["outputs"] = {ext: str(out_dir / f"{spec.key}{ext}") for ext in extensions}
        manifest.append(entry)
    return manifest


def build_enriched_table_manifest() -> List[Dict]:
    manifest: List[Dict] = []
    for spec in ALL_TABLES:
        entry = asdict(spec)
        dirs = get_table_output_dirs(spec.output_group)
        entry["caption_file"] = str(PAPER_CAPTIONS_DIR / f"{spec.key}.md")
        entry["outputs"] = {
            "csv": str(dirs["csv"] / f"{spec.key}.csv"),
            "latex": str(dirs["latex"] / f"{spec.key}.tex"),
        }
        manifest.append(entry)
    return manifest


# -----------------------------------------------------------------------------
# Exported symbols
# -----------------------------------------------------------------------------
__all__ = [
    "FigureSpec",
    "TableSpec",
    "MAIN_FIGURES",
    "SUPPLEMENTARY_FIGURES",
    "MAIN_TABLES",
    "SUPPLEMENTARY_TABLES",
    "ALL_FIGURES",
    "ALL_TABLES",
    "build_figure_manifest",
    "build_table_manifest",
    "build_enriched_figure_manifest",
    "build_enriched_table_manifest",
    "get_figure_output_dir",
    "get_table_output_dirs",
    "get_figure_spec",
    "get_table_spec",
    "list_figure_keys",
    "list_table_keys",
    "expected_figure_paths",
    "expected_table_paths",
    "expected_caption_paths",
    "PAPER_FIGURE_MANIFEST_FILE",
    "PAPER_TABLE_MANIFEST_FILE",
    "PAPER_CAPTIONS_DIR",
    "PAPER_MANIFESTS_DIR",
]
