"""Regenerate scaffold and migration inventory reports after prompt 3."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT.parent / "focus_measure"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEEDED_SCAFFOLD_FILES: Sequence[str] = (
    "config/paper_assets.py",
    "config/settings.py",
    "src/measures/focus_measure_library.py",
    "scripts/01_build_stacks.py",
    "scripts/02_build_reference_labels.py",
    "scripts/03_run_single_measure_benchmark.py",
    "scripts/04_evaluate_single_measures.py",
    "scripts/05_plot_single_measure_results.py",
    "scripts/06_run_composite_gp_lodo.py",
    "scripts/07_evaluate_composites.py",
    "scripts/08_run_statistics_and_sensitivity.py",
    "scripts/09_optional_downstream_baseline.py",
    "scripts/10_export_paper_assets.py",
    "scripts/11_run_full_pipeline.py",
)

FOUNDATION_FILES: Sequence[str] = (
    "config/paths.py",
    "src/utils/seeds.py",
    "src/utils/logging_utils.py",
    "src/utils/validation.py",
    "scripts/00_inventory_repo.py",
)

PRESERVE_SCAFFOLD_FILES: Sequence[str] = (
    *SEEDED_SCAFFOLD_FILES,
    "config/paths.py",
    "src/utils/seeds.py",
    "src/utils/logging_utils.py",
    "src/utils/validation.py",
    "src/io/dataset_loader.py",
    "src/io/stack_builder.py",
    "src/io/metadata.py",
    "src/labels/source_labels.py",
    "src/labels/surrogate_labels.py",
    "src/labels/loo_voting.py",
    "src/evaluation/autofocus_metrics.py",
    "src/evaluation/aggregation.py",
    "src/evaluation/statistics.py",
    "src/evaluation/sensitivity.py",
    "src/plots/focus_curves.py",
    "src/plots/benchmark_figures.py",
    "src/plots/runtime_figures.py",
    "src/gp/terminal_selection.py",
    "src/gp/deap_search.py",
    "src/gp/lodo_runner.py",
    "src/gp/deduplication.py",
    "src/gp/baselines.py",
    "scripts/00_inventory_repo.py",
)

IMPORTANT_LEGACY_FILES: Dict[str, Sequence[str]] = {
    "stack_loading_and_reference_peak_monolith": (
        "focus_measure_selection.py",
        "paper2_gpu_backend.py",
    ),
    "single_measure_reporting_and_rebuild": (
        "single_focus_fitness.py",
        "paper_eval_common.py",
        "paper_eval_rebuild.py",
        "paper_significance.py",
    ),
    "gp_and_composite_search": (
        "gp_focus_competitive_explainable.py",
        "gp_focus_paper2_nsga2.py",
        "genetic_composite.py",
        "genetic_composite_competitive.py",
        "paper2_eval_backbone.py",
        "paper2_terminal_screening.py",
    ),
    "paper_tables_and_figures": (
        "paper2_report_figures.py",
        "paper2_report_tables.py",
        "paper_ablation_report.py",
        "composite_report.py",
        "make_final_viz.py",
    ),
}

MIGRATION_ROWS: Sequence[tuple[str, str, str]] = (
    (
        "focus_measure_selection.py",
        "src/io/dataset_loader.py + src/io/stack_builder.py + src/labels/surrogate_labels.py + src/measures/focus_measure_library.py + scripts/01_build_stacks.py + scripts/02_build_reference_labels.py + scripts/03_run_single_measure_benchmark.py + scripts/04_evaluate_single_measures.py + scripts/05_plot_single_measure_results.py",
        "Legacy monolith covering stack loading, reference peaks, focus measures, ranking, and visualization. Prompt 1 extracted the stage-01/02 foundations from here.",
    ),
    (
        "Raw dataset folder structure (WBC/TBI/PBS/BMA/TBF)",
        "src/io/dataset_loader.py + src/io/stack_builder.py",
        "Stage-01 stack discovery now follows the real canonical raw folder layout directly instead of relying on notebook-only assumptions.",
    ),
    (
        "paper2_gpu_backend.py",
        "future src/measures/gpu_backend.py",
        "GPU-backed operator kernels; not needed for prompt 1.",
    ),
    (
        "single_focus_fitness.py",
        "src/evaluation/autofocus_metrics.py + src/evaluation/aggregation.py + scripts/04_evaluate_single_measures.py",
        "Soft peak localization, curve-quality penalties, and single-measure objective alignment informed the corrected stage-04 extraction.",
    ),
    (
        "focus_measure_selection.py (Stage E/G/H)",
        "src/evaluation/autofocus_metrics.py + src/evaluation/statistics.py + src/plots/benchmark_figures.py",
        "Prompt 2 extracted the reusable single-measure metric, generalization-bootstrap, Friedman/Wilcoxon-Holm, and publication-figure patterns from the legacy monolith.",
    ),
    (
        "gp_focus_competitive_explainable.py",
        "src/gp/deap_search.py + src/gp/deduplication.py + scripts/06_run_composite_gp_lodo.py",
        "Legacy expression evaluation, protected operators, and syntax-level deduplication informed the prompt-3 GP extraction.",
    ),
    (
        "gp_focus_paper2_nsga2.py",
        "src/gp/deap_search.py + src/gp/lodo_runner.py + scripts/06_run_composite_gp_lodo.py",
        "Prompt 3 extracted the leave-one-dataset-out fold orchestration, DEAP search structure, and held-out scoring path.",
    ),
    (
        "genetic_composite.py + genetic_composite_competitive.py",
        "src/gp/deap_search.py + scripts/07_evaluate_composites.py",
        "Composite-expression primitives and runtime-aware evaluation informed the corrected composite scoring helpers.",
    ),
    (
        "paper_eval_common.py",
        "src/evaluation/aggregation.py + future src/evaluation/paper_common.py",
        "Reusable weighted summaries and paper-facing aggregation helpers; Prompt 2 extracted the single-measure pieces and left composite reporting helpers for later.",
    ),
    (
        "paper_eval_rebuild.py + paper_significance.py",
        "src/evaluation/statistics.py + scripts/04_evaluate_single_measures.py + scripts/07_evaluate_composites.py + scripts/08_run_statistics_and_sensitivity.py",
        "Paper-safe significance tests and reporting patterns informed the corrected single/composite statistics exports; Prompt 3 added composite-aware union statistics on top.",
    ),
    (
        "paper2_report_figures.py + make_final_viz.py + paper_figures.py",
        "src/plots/focus_curves.py + src/plots/benchmark_figures.py + src/plots/runtime_figures.py + scripts/05_plot_single_measure_results.py + scripts/10_export_paper_assets.py",
        "Prompt 2 extracted publication-format figure helpers and single-measure figure patterns; later prompts still need composite figure extraction.",
    ),
    (
        "paper2_report_tables.py + composite_report.py + paper_ablation_report.py",
        "src/gp/baselines.py + scripts/07_evaluate_composites.py + scripts/10_export_paper_assets.py",
        "Composite summary tables, ablations, and manuscript-facing exports informed the prompt-3 comparison helpers.",
    ),
    (
        "paper2_eval_backbone.py + paper2_terminal_screening.py",
        "src/labels/source_labels.py + src/gp/terminal_selection.py + src/gp/lodo_runner.py",
        "Prompt 3 adopted fold-local leave-one-dataset-out terminal selection and outer-fold manifests from these paper-2 components without reintroducing leakage.",
    ),
    (
        "paper2_final_selection.py",
        "src/gp/deduplication.py + src/gp/baselines.py + scripts/07_evaluate_composites.py",
        "Prompt 3 extracted shortlist deduplication and composite-vs-single comparison patterns from the paper-2 final-selection logic.",
    ),
)

LATER_EXTRACTION_MODULES: Sequence[str] = (
    "src/evaluation/paper_common.py",
    "src/plots/composite_figures.py",
    "src/downstream/baseline_model.py",
    "src/export/table_writer.py",
)


@dataclass(frozen=True)
class RepoScan:
    python_files: List[str]
    notebook_files: List[str]
    scripts: List[str]


def relative_files(root: Path, pattern: str) -> List[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob(pattern))


def scan_repo(root: Path) -> RepoScan:
    python_files = relative_files(root, "*.py")
    notebook_files = relative_files(root, "*.ipynb")
    scripts = [path for path in python_files if path.startswith("scripts/") or path.startswith("notebooks/")]
    return RepoScan(python_files=python_files, notebook_files=notebook_files, scripts=scripts)


def top_level_legacy_scripts() -> List[str]:
    return sorted(
        str(path.relative_to(LEGACY_ROOT))
        for path in LEGACY_ROOT.glob("*.py")
        if path.is_file()
    )


def detect_hardcoded_path_hits(root: Path) -> Dict[str, List[str]]:
    patterns = {
        "absolute_dataset_paths": (
            "/absolute/path/to/focus_measure",
            "WBC_dataset1",
            "TBSI",
            "bma pbf tfa",
        ),
        "legacy_output_roots": (
            "focus_eval",
            "paper2_full",
            "paper_outputs_old",
        ),
        "space_or_windows_metadata": (
            "New folder",
            "Zone.Identifier",
        ),
    }
    hits: Dict[str, List[str]] = {key: [] for key in patterns}
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(root))
        for key, needles in patterns.items():
            if any(needle in text or needle in rel for needle in needles):
                hits[key].append(rel)
    return {key: sorted(values) for key, values in hits.items() if values}


def detect_blockers() -> List[str]:
    blockers = []
    missing_now = [rel for rel in FOUNDATION_FILES if not (PROJECT_ROOT / rel).exists()]
    if missing_now:
        blockers.append("Foundation files still missing: " + ", ".join(missing_now))

    if any("Zone.Identifier" in path.name for path in LEGACY_ROOT.rglob("*")):
        blockers.append("Legacy repo contains Windows Zone.Identifier artifacts that should not be migrated into the scaffold.")

    blockers.append("Source-provided labels are still unresolved dataset-by-dataset; later prompts need targeted extraction of label discovery and auditing logic.")
    blockers.append("Prompt 3 extracted working src/gp foundations and corrected composite stages 06 to 08, but optional downstream baselines, final paper-asset export alignment, and broader paper-table helpers are still pending.")
    return blockers


def markdown_list(items: Iterable[str], *, empty_message: str = "- None") -> str:
    materialized = list(items)
    if not materialized:
        return empty_message
    return "\n".join(f"- `{item}`" for item in materialized)


def scaffold_inventory_files() -> List[str]:
    files: List[str] = []
    for rel_root in ("config", "src", "scripts", "reports"):
        root = PROJECT_ROOT / rel_root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files.append(str(path.relative_to(PROJECT_ROOT)))
    return files


def build_repo_inventory_markdown(scaffold_scan: RepoScan, legacy_scan: RepoScan) -> str:
    hardcoded_hits = detect_hardcoded_path_hits(LEGACY_ROOT)
    blockers = detect_blockers()
    likely_sources = {
        "stack_loading": ["focus_measure_selection.py"],
        "labels_and_reference_peaks": ["focus_measure_selection.py", "paper2_eval_backbone.py"],
        "focus_measure_library_and_gpu_subset": ["focus_measure_selection.py", "paper2_gpu_backend.py"],
        "single_measure_evaluation": ["single_focus_fitness.py", "paper_eval_common.py", "paper_eval_rebuild.py"],
        "gp_and_expression_runtime": [
            "gp_focus_competitive_explainable.py",
            "gp_focus_paper2_nsga2.py",
            "genetic_composite.py",
            "genetic_composite_competitive.py",
            "paper2_terminal_screening.py",
        ],
        "statistics_and_reporting": ["paper_significance.py", "paper2_report_tables.py", "paper_ablation_report.py"],
        "plotting_and_export": ["paper2_report_figures.py", "make_final_viz.py", "paper_figures.py"],
    }

    duplicated_logic = [
        "Legacy `focus_measure_selection.py` overlaps scaffold stages 01 through 05 and is the main monolith to split later.",
        "Legacy GP logic is duplicated across `gp_focus_competitive_explainable.py`, `gp_focus_paper2_nsga2.py`, `genetic_composite.py`, and `genetic_composite_competitive.py`.",
        "Legacy reporting logic is duplicated across `paper_eval_rebuild.py`, `paper_significance.py`, `paper2_report_tables.py`, and `paper_ablation_report.py`.",
        "Legacy figure/export logic is spread across `paper2_report_figures.py`, `paper_figures.py`, `make_final_viz.py`, and `make_all_figures.py`.",
    ]

    hardcoded_section_lines = []
    for category, files in hardcoded_hits.items():
        hardcoded_section_lines.append(f"- `{category}`: {', '.join(f'`{item}`' for item in files[:12])}")
    hardcoded_section = "\n".join(hardcoded_section_lines) if hardcoded_section_lines else "- None detected"

    likely_source_lines = []
    for topic, files in likely_sources.items():
        likely_source_lines.append(f"- `{topic}`: {', '.join(f'`{item}`' for item in files)}")

    legacy_sections: List[str] = []
    for group, files in IMPORTANT_LEGACY_FILES.items():
        legacy_sections.append(f"### {group.replace('_', ' ').title()}")
        legacy_sections.append(markdown_list(files))
        legacy_sections.append("")

    scaffold_files = scaffold_inventory_files()

    return "\n".join(
        [
            "# Repo Inventory",
            "",
            f"- Scaffold root: `{PROJECT_ROOT}`",
            f"- Legacy root: `{LEGACY_ROOT}`",
            "",
            "## Current Scaffold Files",
            markdown_list(scaffold_files),
            "",
            "## Missing Foundation Files",
            "- Initial prompt-0 gaps were: `config/paths.py`, `src/utils/seeds.py`, `src/utils/logging_utils.py`, `src/utils/validation.py`, `scripts/00_inventory_repo.py`.",
            "- Current status: all prompt-0 foundation files now exist.",
            "- Prompt-1 extraction added `src/io/dataset_loader.py`, `src/io/stack_builder.py`, `src/io/metadata.py`, `src/labels/source_labels.py`, `src/labels/surrogate_labels.py`, and `src/labels/loo_voting.py`.",
            "- Prompt-2 extraction added `src/evaluation/autofocus_metrics.py`, `src/evaluation/aggregation.py`, `src/evaluation/statistics.py`, `src/evaluation/sensitivity.py`, `src/plots/focus_curves.py`, `src/plots/benchmark_figures.py`, and `src/plots/runtime_figures.py`.",
            "- Prompt-3 extraction added `src/gp/terminal_selection.py`, `src/gp/deap_search.py`, `src/gp/lodo_runner.py`, `src/gp/deduplication.py`, and `src/gp/baselines.py`.",
            "",
            "## Notebooks",
            f"- Scaffold notebooks found: {len(scaffold_scan.notebook_files)}",
            f"- Legacy notebooks found: {len(legacy_scan.notebook_files)}",
            "- No `.ipynb` notebooks were found in either repository.",
            "",
            "## Legacy Scripts",
            markdown_list(top_level_legacy_scripts()),
            "",
            "## Important Legacy Files Likely To Be Mined Later",
            *legacy_sections,
            "## Likely Source Files By Concern",
            *likely_source_lines,
            "",
            "## Duplicated Logic",
            *[f"- {item}" for item in duplicated_logic],
            "",
            "## Obvious Path Inconsistencies",
            "- Canonical raw dataset paths include directories with spaces such as `New folder`, so all future path handling must stay on `pathlib.Path` and avoid shell-string concatenation.",
            "- Legacy code mixes `focus_eval`, `paper2_full - Copy`, `paper2_full_v2 - Copy`, and `paper_outputs_old`, so old output roots are not canonical for the scaffold.",
            "- Legacy repo contains `Zone.Identifier` artifacts that must be ignored during migration.",
            "- Prompt-0 fixed scaffold path-type bugs where seeded scripts were still passing raw dataset strings directly into path-walking helpers.",
            hardcoded_section,
            "",
            "## Obvious Blockers",
            *[f"- {item}" for item in blockers],
            "",
        ]
    )


def build_migration_map_markdown() -> str:
    table_lines = [
        "| Legacy file(s) | New target | Notes |",
        "| --- | --- | --- |",
    ]
    for legacy, target, notes in MIGRATION_ROWS:
        table_lines.append(f"| `{legacy}` | `{target}` | {notes} |")

    return "\n".join(
        [
            "# Migration Map",
            "",
            "## Old File To New Module Mapping",
            *table_lines,
            "",
            "## Scaffold Files To Preserve",
            markdown_list(PRESERVE_SCAFFOLD_FILES),
            "",
            "## Missing Modules For Later Extraction",
            markdown_list(LATER_EXTRACTION_MODULES),
            "",
            "## Notes",
            "- Preserve the seeded stage scripts as orchestration entry points; later prompts should extract shared logic into `src/*` modules behind them instead of replacing the scripts wholesale.",
            "- Keep `config/settings.py`, `config/paper_assets.py`, and `src/measures/focus_measure_library.py` as the canonical corrected scaffold anchors unless a later prompt finds a concrete defect.",
            "- Prompt 1 moved stage-01 stack traversal/loading and stage-02 label discovery/voting into reusable `src/io/*` and `src/labels/*` modules.",
            "- Prompt 2 moved stage-03/04/05 metric, aggregation, statistics, sensitivity, and plotting logic into reusable `src/evaluation/*` and `src/plots/*` modules.",
        ]
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scaffold_scan = scan_repo(PROJECT_ROOT)
    legacy_scan = scan_repo(LEGACY_ROOT)

    repo_inventory = build_repo_inventory_markdown(scaffold_scan, legacy_scan)
    migration_map = build_migration_map_markdown()

    (REPORTS_DIR / "repo_inventory.md").write_text(repo_inventory + "\n", encoding="utf-8")
    (REPORTS_DIR / "migration_map.md").write_text(migration_map + "\n", encoding="utf-8")

    print(f"[SAVE] {REPORTS_DIR / 'repo_inventory.md'}")
    print(f"[SAVE] {REPORTS_DIR / 'migration_map.md'}")


if __name__ == "__main__":
    main()
