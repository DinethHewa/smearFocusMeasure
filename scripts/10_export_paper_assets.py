# scripts/10_export_paper_assets.py

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paper_assets import (
    ALL_FIGURES,
    ALL_TABLES,
    build_enriched_figure_manifest,
    build_enriched_table_manifest,
    expected_caption_paths,
    expected_figure_paths,
    expected_table_paths,
    get_figure_output_dir,
    get_figure_spec,
    get_table_output_dirs,
)
from config.paths import (
    COMPOSITE_FIGURE_MANIFEST_FILE,
    COMPOSITE_MAIN_DIR,
    COMPOSITE_METRIC_PROFILE_FILE,
    COMPOSITE_SUPP_DIR,
    DATASET_ORDER,
    GP_DEDUP_DIR,
    GP_RUNS_DIR,
    LABEL_PROVENANCE_MANIFEST_FILE,
    GP_SUMMARIES_DIR,
    LABEL_SOURCE_MANIFEST_FILE,
    LOGS_DIR,
    LOO_VOTER_DISCLOSURE_FILE,
    PAPER_ASSET_INDEX_FILE,
    PAPER_CAPTIONS_DIR,
    PAPER_DIR,
    PAPER_FIGURE_MANIFEST_FILE,
    PAPER_FIGURES_MAIN_DIR,
    PAPER_FIGURES_SUPP_DIR,
    PAPER_MANIFESTS_DIR,
    PAPER_TABLE_MANIFEST_FILE,
    PAPER_TABLES_DIR,
    PAPER_TABLES_MAIN_CSV_DIR,
    PAPER_TABLES_MAIN_LATEX_DIR,
    PAPER_TABLES_SUPP_CSV_DIR,
    PAPER_TABLES_SUPP_LATEX_DIR,
    RANK_STABILITY_RESULTS_FILE,
    RANK_STABILITY_SUMMARY_FILE,
    SINGLE_EVAL_MAIN_DIR,
    SINGLE_EVAL_SUPP_DIR,
    SINGLE_MEASURE_FREEZE_MANIFEST_FILE,
    SINGLE_TIMING_SUMMARY_FILE,
    SENSITIVITY_DIR,
    STATISTICS_DIR,
    STACK_METADATA_DIR,
    ensure_output_dirs,
    get_single_norm_curve_file,
    get_source_label_file,
    get_stack_metadata_file,
    get_surrogate_label_file,
)
from config.settings import (
    ALPHA_SENSITIVITY_VALUES,
    ALT_METRIC_WEIGHT_SCHEMES,
    AUTOFOCUS_METRICS,
    DEFAULT_MAIN_FIGURE_EXTENSIONS,
    DEFAULT_RUN_MODE,
    DEFAULT_SUPP_FIGURE_EXTENSIONS,
    FIGURE_DPI,
    GENERALIZATION_ALPHA,
    METRIC_WEIGHTS,
    validate_all_settings,
)
from src.plots.composite_figures import (
    load_best_composite_candidate,
    plot_composite_vs_single_focus_curves,
    plot_expression_equivalence_clusters,
    plot_gp_convergence,
    plot_gp_convergence_by_fold,
    plot_gp_final_refit_convergence,
    plot_gp_lodo_summary,
    plot_gp_lodo_vs_final_refit,
    plot_gp_seedwise_score_distribution,
    plot_pipeline_overview,
    plot_terminal_frequency,
)
from src.utils.logging_utils import get_logger
from src.utils.validation import (
    load_checkpoint,
    load_csv_rows,
    load_json,
    save_csv_rows,
    save_json,
    summarize_existing_files,
    validate_environment,
    validate_expected_asset_files,
    write_checkpoint,
)


COMPAT_TABLE_DIRS = {
    "main_csv": PAPER_TABLES_DIR / "main_csv",
    "main_latex": PAPER_TABLES_DIR / "main_latex",
    "supplementary_csv": PAPER_TABLES_DIR / "supplementary_csv",
    "supplementary_latex": PAPER_TABLES_DIR / "supplementary_latex",
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and validate final paper assets")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run full-run mode")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if expected assets are missing",
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
# Generic helpers
# -----------------------------------------------------------------------------
def write_caption_placeholders() -> None:
    PAPER_CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in ALL_FIGURES:
        caption_file = PAPER_CAPTIONS_DIR / f"{spec.key}.md"
        if not caption_file.exists():
            caption_file.write_text(
                "\n".join(
                    [
                        f"# {spec.key}",
                        "",
                        f"Title: {spec.title}",
                        f"Section: {spec.section}",
                        "",
                        "Caption:",
                        "",
                        "[Write final caption here.]",
                        "",
                        "Notes:",
                        "- Source script:",
                        "- Source data files:",
                        "- Main message:",
                    ]
                ),
                encoding="utf-8",
            )

    for spec in ALL_TABLES:
        caption_file = PAPER_CAPTIONS_DIR / f"{spec.key}.md"
        if not caption_file.exists():
            caption_file.write_text(
                "\n".join(
                    [
                        f"# {spec.key}",
                        "",
                        f"Title: {spec.title}",
                        f"Section: {spec.section}",
                        "",
                        "Caption:",
                        "",
                        "[Write final caption here.]",
                        "",
                        "Notes:",
                        "- Source script:",
                        "- Source data files:",
                        "- Main message:",
                    ]
                ),
                encoding="utf-8",
            )


def copy_if_exists(src: Path, dst: Path, logger) -> bool:
    if not src.exists():
        logger.warning("Missing source asset: %s", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Copied %s -> %s", src, dst)
    return True


def enrich_asset_manifest_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in entries:
        entry = dict(item)
        outputs = dict(entry.get("outputs", {}))
        entry["output_exists"] = {name: Path(path_str).exists() for name, path_str in outputs.items()}
        caption_file = entry.get("caption_file")
        entry["caption_exists"] = bool(caption_file and Path(caption_file).exists())
        enriched.append(entry)
    return enriched


def ensure_table_compatibility_dirs(logger) -> Dict[str, Any]:
    mirrored: Dict[str, List[str]] = {}
    pairs = [
        (PAPER_TABLES_MAIN_CSV_DIR, COMPAT_TABLE_DIRS["main_csv"]),
        (PAPER_TABLES_MAIN_LATEX_DIR, COMPAT_TABLE_DIRS["main_latex"]),
        (PAPER_TABLES_SUPP_CSV_DIR, COMPAT_TABLE_DIRS["supplementary_csv"]),
        (PAPER_TABLES_SUPP_LATEX_DIR, COMPAT_TABLE_DIRS["supplementary_latex"]),
    ]

    for src_dir, dst_dir in pairs:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for stale in dst_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
        copied: List[str] = []
        if src_dir.exists():
            for src in sorted(src_dir.glob("*")):
                if not src.is_file():
                    continue
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst))
        mirrored[str(dst_dir)] = copied
        logger.info("Mirrored %d table assets into %s", len(copied), dst_dir)

    return {
        "compatibility_dirs": {name: str(path) for name, path in COMPAT_TABLE_DIRS.items()},
        "mirrored_files": mirrored,
    }


def table_csv_path(key: str, group: str) -> Path:
    return get_table_output_dirs(group)["csv"] / f"{key}.csv"


def table_tex_path(key: str, group: str) -> Path:
    return get_table_output_dirs(group)["latex"] / f"{key}.tex"


def write_simple_latex_table(csv_path: Path, tex_path: Path, caption: Optional[str] = None) -> bool:
    if not csv_path.exists():
        return False

    rows = load_csv_rows(csv_path)
    if not rows:
        tex_path.write_text(
            "\\begin{table}[t]\n\\centering\n\\caption{Placeholder caption}\n\\begin{tabular}{c}\nNo data\\\\\n\\end{tabular}\n\\end{table}\n",
            encoding="utf-8",
        )
        return True

    headers = list(rows[0].keys())
    colspec = "l" + "c" * (len(headers) - 1)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption or 'Placeholder caption'}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")

    # keep LaTeX export compact: top 15 rows max in auto-exported tables
    max_rows = min(15, len(rows))
    for row in rows[:max_rows]:
        values = [str(row.get(h, "")).replace("_", "\\_") for h in headers]
        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def load_reference_labels_for_dataset(dataset_name: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    source_path = get_source_label_file(dataset_name)
    if source_path.exists():
        return np.load(source_path, allow_pickle=False).astype(int).reshape(-1), "source"
    surrogate_path = get_surrogate_label_file(dataset_name)
    if surrogate_path.exists():
        return np.load(surrogate_path, allow_pickle=False).astype(int).reshape(-1), "surrogate"
    return None, None


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_gp_fold_summary_rows() -> List[Dict[str, Any]]:
    preferred = PAPER_TABLES_SUPP_CSV_DIR / "STable4_gp_foldwise_results.csv"
    if preferred.exists():
        return [dict(row) for row in load_csv_rows(preferred)]

    rows: List[Dict[str, Any]] = []
    for path in sorted(GP_SUMMARIES_DIR.glob("heldout_*_summary.json")):
        payload = load_json(path)
        seed_results = payload.get("seed_results", payload.get("fold_results", []))
        if not seed_results:
            continue
        heldout_scores = [safe_float(row.get("heldout_score")) for row in seed_results]
        heldout_scores = [value for value in heldout_scores if np.isfinite(value)]
        if not heldout_scores:
            continue
        best = min(seed_results, key=lambda row: safe_float(row.get("heldout_score"), 1e18))
        rows.append(
            {
                "outer_fold": payload.get("outer_fold", ""),
                "held_out_dataset": payload.get("held_out_dataset", ""),
                "train_datasets": " | ".join(payload.get("train_datasets", [])),
                "num_seeds": len(seed_results),
                "mean_heldout_score": float(np.mean(heldout_scores)),
                "std_heldout_score": float(np.std(heldout_scores, ddof=0)),
                "best_heldout_score": safe_float(best.get("heldout_score")),
                "best_expression": best.get("best_expression", ""),
                "best_seed": best.get("seed", ""),
                "selected_terminal_count": len(best.get("terminals", [])),
            }
        )
    return rows


def load_final_refit_summary() -> Optional[Dict[str, Any]]:
    path = GP_SUMMARIES_DIR / "final_refit_summary.json"
    if not path.exists():
        return None
    payload = load_json(path)
    return dict(payload) if isinstance(payload, dict) else None


def load_final_composite_expression() -> Optional[Dict[str, Any]]:
    path = GP_SUMMARIES_DIR / "final_composite_expression.json"
    if not path.exists():
        return None
    payload = load_json(path)
    return dict(payload) if isinstance(payload, dict) else None


def build_gp_lodo_vs_final_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in load_gp_fold_summary_rows():
        rows.append(
            {
                "protocol": "lodo",
                "dataset_scope": row.get("held_out_dataset", ""),
                "train_datasets": row.get("train_datasets", ""),
                "score_kind": "heldout_score",
                "num_seeds": row.get("num_seeds", ""),
                "mean_score": row.get("mean_heldout_score", ""),
                "std_score": row.get("std_heldout_score", ""),
                "best_score": row.get("best_heldout_score", ""),
                "best_seed": row.get("best_seed", ""),
                "best_complexity": row.get("best_complexity", ""),
                "num_nodes": row.get("num_nodes", ""),
                "tree_height": row.get("tree_height", ""),
                "best_expression": row.get("best_expression", ""),
                "selected_terminals": row.get("selected_terminals", ""),
            }
        )

    final_summary = load_final_refit_summary()
    if final_summary:
        rows.append(
            {
                "protocol": "final_refit",
                "dataset_scope": "ALL",
                "train_datasets": " | ".join(str(x) for x in final_summary.get("train_datasets", DATASET_ORDER)),
                "score_kind": "all_dataset_training_score",
                "num_seeds": final_summary.get("num_seeds", ""),
                "mean_score": final_summary.get("mean_all_dataset_score", ""),
                "std_score": final_summary.get("std_all_dataset_score", ""),
                "best_score": final_summary.get("best_all_dataset_score", ""),
                "best_seed": final_summary.get("best_seed", ""),
                "best_complexity": final_summary.get("best_complexity", ""),
                "num_nodes": final_summary.get("best_num_nodes", ""),
                "tree_height": final_summary.get("best_tree_height", ""),
                "best_expression": final_summary.get("best_expression", ""),
                "selected_terminals": " | ".join(str(x) for x in final_summary.get("terminals", [])),
            }
        )
    return rows


def _progress_source_for_seed(seed_dir: Path) -> Optional[Path]:
    progress_path = seed_dir / "progress.csv"
    if progress_path.exists() and load_csv_rows(progress_path):
        return progress_path
    logbook_path = seed_dir / "logbook.csv"
    if logbook_path.exists():
        return logbook_path
    return None


def collect_gp_generation_trace_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dirs = list(sorted(GP_RUNS_DIR.glob("heldout_*/seed_*")))
    run_dirs.extend(sorted((GP_RUNS_DIR / "final_refit").glob("seed_*")))

    for seed_dir in run_dirs:
        source = _progress_source_for_seed(seed_dir)
        if source is None:
            continue
        parent_name = seed_dir.parent.name
        if parent_name == "final_refit":
            protocol = "final_refit"
            dataset_scope = "ALL"
        else:
            protocol = "lodo"
            dataset_scope = parent_name.replace("heldout_", "")
        seed = seed_dir.name.replace("seed_", "")

        for row in load_csv_rows(source):
            generation_index = int(safe_float(row.get("generation"), -1))
            if generation_index < 0:
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "dataset_scope": dataset_scope,
                    "held_out_dataset": dataset_scope if protocol == "lodo" else "",
                    "seed": seed,
                    "generation_index": generation_index,
                    "generation_number": generation_index + 1,
                    "best_generalization_score": row.get("best_generalization_score", ""),
                    "best_complexity": row.get("best_complexity", ""),
                    "best_nodes": row.get("best_nodes", ""),
                    "best_height": row.get("best_height", ""),
                    "best_expression": row.get("best_expression", ""),
                    "source_file": str(source),
                }
            )
    return rows


def collect_gp_seedwise_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    result_paths = list(sorted(GP_RUNS_DIR.glob("heldout_*/seed_*/best_result.json")))
    result_paths.extend(sorted((GP_RUNS_DIR / "final_refit").glob("seed_*/best_result.json")))

    for path in result_paths:
        payload = load_json(path)
        parent_name = path.parents[1].name
        protocol = "final_refit" if parent_name == "final_refit" else "lodo"
        dataset_scope = "ALL" if protocol == "final_refit" else parent_name.replace("heldout_", "")
        rows.append(
            {
                "protocol": protocol,
                "dataset_scope": dataset_scope,
                "held_out_dataset": payload.get("held_out_dataset", ""),
                "evaluation_dataset": payload.get("evaluation_dataset", ""),
                "seed": payload.get("seed", ""),
                "train_datasets": " | ".join(str(x) for x in payload.get("train_datasets", [])),
                "heldout_score": payload.get("heldout_score", ""),
                "all_dataset_score": payload.get("all_dataset_score", ""),
                "all_dataset_score_std": payload.get("all_dataset_score_std", ""),
                "best_training_objective": payload.get("best_training_objective", ""),
                "best_complexity": payload.get("best_complexity", ""),
                "num_nodes": payload.get("num_nodes", ""),
                "tree_height": payload.get("tree_height", ""),
                "best_expression": payload.get("best_expression", ""),
                "terminals": " | ".join(str(x) for x in payload.get("terminals", [])),
                "result_json": str(path),
            }
        )
    return rows


def export_publication_figures(logger) -> Dict[str, bool]:
    results: Dict[str, bool] = {}

    try:
        spec = get_figure_spec("Fig1_pipeline_overview")
        plot_pipeline_overview(
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "Fig1_pipeline_overview", exc)
        results["Fig1_pipeline_overview"] = False

    try:
        best_composite = load_best_composite_candidate(
            table7_csv=PAPER_TABLES_MAIN_CSV_DIR / "Table7_top10_composites_common_scoring.csv",
            comparison_csv=COMPOSITE_MAIN_DIR / "best_composite_vs_best_single.csv",
            dedup_json=GP_DEDUP_DIR / "deduplicated_best_expressions.json",
        )
        if best_composite is None:
            raise FileNotFoundError("Best composite candidate could not be resolved from evaluation outputs")
        spec = get_figure_spec("Fig6_representative_composite_vs_single_curves")
        plot_composite_vs_single_focus_curves(
            dataset_names=DATASET_ORDER,
            curve_path_resolver=lambda dataset_name, measure_name: get_single_norm_curve_file(dataset_name, measure_name),
            reference_label_resolver=load_reference_labels_for_dataset,
            best_composite=best_composite,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
            logger=logger,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "Fig6_representative_composite_vs_single_curves", exc)
        results["Fig6_representative_composite_vs_single_curves"] = False

    try:
        spec = get_figure_spec("Fig9_gp_lodo_summary")
        fold_csv = PAPER_TABLES_SUPP_CSV_DIR / "STable4_gp_foldwise_results.csv"
        if not fold_csv.exists():
            raise FileNotFoundError(f"Missing GP fold summary CSV: {fold_csv}")
        plot_gp_lodo_summary(
            fold_rows=load_csv_rows(fold_csv),
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "Fig9_gp_lodo_summary", exc)
        results["Fig9_gp_lodo_summary"] = False

    try:
        spec = get_figure_spec("Fig12_lodo_vs_final_refit")
        summary_rows = build_gp_lodo_vs_final_rows()
        if not summary_rows:
            raise FileNotFoundError("No LODO/final-refit summary rows are available")
        plot_gp_lodo_vs_final_refit(
            summary_rows=summary_rows,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_MAIN_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "Fig12_lodo_vs_final_refit", exc)
        results["Fig12_lodo_vs_final_refit"] = False

    try:
        spec = get_figure_spec("SFig2_gp_convergence")
        plot_gp_convergence(
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "SFig2_gp_convergence", exc)
        results["SFig2_gp_convergence"] = False

    try:
        trace_rows = collect_gp_generation_trace_rows()

        spec = get_figure_spec("SFig5_gp_convergence_by_fold")
        plot_gp_convergence_by_fold(
            trace_rows=trace_rows,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "SFig5_gp_convergence_by_fold", exc)
        results["SFig5_gp_convergence_by_fold"] = False

    try:
        trace_rows = collect_gp_generation_trace_rows()

        spec = get_figure_spec("SFig6_final_refit_convergence")
        plot_gp_final_refit_convergence(
            trace_rows=trace_rows,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "SFig6_final_refit_convergence", exc)
        results["SFig6_final_refit_convergence"] = False

    try:
        seed_rows = collect_gp_seedwise_rows()

        spec = get_figure_spec("SFig7_gp_seedwise_score_distribution")
        plot_gp_seedwise_score_distribution(
            seed_rows=seed_rows,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export %s: %s", "SFig7_gp_seedwise_score_distribution", exc)
        results["SFig7_gp_seedwise_score_distribution"] = False

    try:
        dedup_path = GP_DEDUP_DIR / "deduplicated_best_expressions.json"
        if not dedup_path.exists():
            raise FileNotFoundError(f"Missing deduplicated composite JSON: {dedup_path}")
        dedup_rows = load_json(dedup_path)

        spec = get_figure_spec("SFig3_terminal_frequency")
        plot_terminal_frequency(
            dedup_rows=dedup_rows,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True

        spec = get_figure_spec("SFig4_expression_equivalence_clusters")
        plot_expression_equivalence_clusters(
            dedup_rows=dedup_rows,
            spec=spec,
            output_dir=get_figure_output_dir(spec.output_group),
            extensions=DEFAULT_SUPP_FIGURE_EXTENSIONS,
            figure_dpi=FIGURE_DPI,
        )
        results[spec.key] = True
    except Exception as exc:
        logger.warning("Could not export supplementary GP figures: %s", exc)
        results.setdefault("SFig3_terminal_frequency", False)
        results.setdefault("SFig4_expression_equivalence_clusters", False)

    return results


# -----------------------------------------------------------------------------
# Main table exporters
# -----------------------------------------------------------------------------
def export_table2_dataset_summary(logger) -> bool:
    rows: List[Dict[str, Any]] = []

    for dataset_name in DATASET_ORDER:
        meta_path = get_stack_metadata_file(dataset_name)
        if not meta_path.exists():
            logger.warning("Missing stack metadata for dataset summary: %s", meta_path)
            continue

        meta = load_json(meta_path)
        rows.append(
            {
                "dataset_name": dataset_name,
                "stack_count": meta.get("stack_count", ""),
                "planes_per_stack_min": meta.get("planes_per_stack_min", ""),
                "planes_per_stack_max": meta.get("planes_per_stack_max", ""),
                "planes_per_stack_median": meta.get("planes_per_stack_median", ""),
                "native_resolution_preserved": meta.get("native_resolution_preserved", ""),
                "use_roi_cropping": meta.get("use_roi_cropping", ""),
            }
        )

    csv_path = table_csv_path("Table2_dataset_summary", "main")
    tex_path = table_tex_path("Table2_dataset_summary", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(csv_path, tex_path, caption="Dataset summary.")


def export_table3_metric_definitions(logger) -> bool:
    descriptions = {
        "absolute_peak_localization_error": "Absolute difference between predicted and reference peak index.",
        "fwhm": "Full width at half maximum of the normalized focus curve.",
        "curvature_at_peak": "Peak sharpness proxy using local second-order curvature.",
        "steep_slope_width": "Width of the steep-response region around the main peak.",
        "steep_to_gradual_slope_ratio": "Ratio of local peak slope to background slope level.",
        "false_maxima_count": "Count of local maxima excluding the global maximum.",
        "noise_level": "Second-difference fluctuation level of the focus curve.",
        "rrmse_under_additive_noise": "Relative RMSE between clean and noisy normalized curves.",
        "range_around_global_maximum": "Width of the high-response plateau around the global maximum.",
        "execution_time_per_slice": "Average processing time per slice.",
    }

    rows = []
    for metric in AUTOFOCUS_METRICS:
        rows.append(
            {
                "metric_name": metric,
                "direction": "lower_is_better" if metric in {
                    "absolute_peak_localization_error",
                    "fwhm",
                    "steep_slope_width",
                    "false_maxima_count",
                    "noise_level",
                    "rrmse_under_additive_noise",
                    "range_around_global_maximum",
                    "execution_time_per_slice",
                } else "higher_is_better",
                "description": descriptions.get(metric, ""),
            }
        )

    csv_path = table_csv_path("Table3_metric_definitions", "main")
    tex_path = table_tex_path("Table3_metric_definitions", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(csv_path, tex_path, caption="Definitions of autofocus evaluation metrics.")


def export_table4_metric_weights_and_alpha(logger) -> bool:
    rows = []
    for metric_name, weight in METRIC_WEIGHTS.items():
        rows.append(
            {
                "metric_name": metric_name,
                "weight": weight,
                "alpha": GENERALIZATION_ALPHA,
                "alpha_sensitivity_values": " | ".join(str(x) for x in ALPHA_SENSITIVITY_VALUES),
            }
        )

    csv_path = table_csv_path("Table4_metric_weights_and_alpha", "main")
    tex_path = table_tex_path("Table4_metric_weights_and_alpha", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(csv_path, tex_path, caption="Metric weights and alpha settings.")


def export_table5_rank_based(logger) -> bool:
    src = SINGLE_EVAL_MAIN_DIR / "top10_single_rank_based.csv"
    dst = table_csv_path("Table5_top10_single_rank_based", "main")
    ok = copy_if_exists(src, dst, logger)
    if ok:
        write_simple_latex_table(dst, table_tex_path("Table5_top10_single_rank_based", "main"),
                                 caption="Top-10 single measures under rank-based analysis.")
    return ok


def export_table6_value_based(logger) -> bool:
    src = SINGLE_EVAL_MAIN_DIR / "top10_single_value_based_equal_dataset.csv"
    dst = table_csv_path("Table6_top10_single_value_based", "main")
    ok = copy_if_exists(src, dst, logger)
    if ok:
        write_simple_latex_table(dst, table_tex_path("Table6_top10_single_value_based", "main"),
                                 caption="Top-10 single measures under normalized value-based analysis.")
    return ok


def export_table7_composites(logger) -> bool:
    dst = table_csv_path("Table7_top10_composites_common_scoring", "main")
    if dst.exists():
        write_simple_latex_table(dst, table_tex_path("Table7_top10_composites_common_scoring", "main"),
                                 caption="Top-10 composites under the common corrected scoring regime.")
        return True

    src_candidates = [
        COMPOSITE_SUPP_DIR / "all_composites_common_scoring.csv",
        COMPOSITE_SUPP_DIR / "union_singles_and_composites_common_value.csv",
        COMPOSITE_MAIN_DIR / "best_composite_vs_best_single.csv",
    ]

    composite_rows: List[Dict[str, Any]] = []
    for src in src_candidates:
        if not src.exists():
            continue
        rows = load_csv_rows(src)
        if not rows:
            continue

        if src.name == "all_composites_common_scoring.csv":
            rows = sorted(rows, key=lambda x: float(x.get("common_value_final_rank", "1e18")))
            composite_rows = rows[:10]
            break

        if src.name == "union_singles_and_composites_common_value.csv":
            rows = [row for row in rows if row.get("candidate_type", "").lower() == "composite"]
            rows = sorted(rows, key=lambda x: float(x.get("final_rank", "1e18")))
            composite_rows = rows[:10]
            break

        if src.name == "best_composite_vs_best_single.csv":
            rows = [row for row in rows if row.get("comparison_item") == "best_composite_under_common_value_scoring"]
            if rows:
                composite_rows = rows
                break

    if not composite_rows:
        logger.warning("Missing Table7 composite source CSV")
        return False

    save_csv_rows(composite_rows, dst)

    write_simple_latex_table(dst, table_tex_path("Table7_top10_composites_common_scoring", "main"),
                             caption="Top-10 composites under the common corrected scoring regime.")
    return True


def export_table8_label_split(logger) -> bool:
    src = SINGLE_EVAL_SUPP_DIR / "label_split_summaries.json"
    if not src.exists():
        logger.warning("Missing label split summary JSON: %s", src)
        return False

    payload = load_json(src)

    rows: List[Dict[str, Any]] = []

    source_rank = payload.get("source_label_only_rank", [])
    source_value = payload.get("source_label_only_value", [])
    surrogate_rank = payload.get("surrogate_label_only_rank", [])
    surrogate_value = payload.get("surrogate_label_only_value", [])

    rows.append(
        {
            "group": "source_label_datasets",
            "datasets": " | ".join(payload.get("source_datasets", [])),
            "top_rank_measure": source_rank[0]["measure_name"] if source_rank else "",
            "top_value_measure": source_value[0]["measure_name"] if source_value else "",
        }
    )
    rows.append(
        {
            "group": "surrogate_label_datasets",
            "datasets": " | ".join(payload.get("surrogate_datasets", [])),
            "top_rank_measure": surrogate_rank[0]["measure_name"] if surrogate_rank else "",
            "top_value_measure": surrogate_value[0]["measure_name"] if surrogate_value else "",
        }
    )

    csv_path = table_csv_path("Table8_true_vs_surrogate_label_split", "main")
    tex_path = table_tex_path("Table8_true_vs_surrogate_label_split", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Comparison of results on source-label and surrogate-label dataset subsets.",
    )


def export_table9_weighting_comparison(logger) -> bool:
    eq_path = SINGLE_EVAL_SUPP_DIR / "all_single_value_based_equal_dataset.csv"
    ps_path = SINGLE_EVAL_SUPP_DIR / "all_single_value_based_per_stack.csv"

    if not eq_path.exists() or not ps_path.exists():
        logger.warning("Missing equal/per-stack weighting files")
        return False

    eq_rows = {r["measure_name"]: r for r in load_csv_rows(eq_path)}
    ps_rows = {r["measure_name"]: r for r in load_csv_rows(ps_path)}

    common = sorted(set(eq_rows.keys()) & set(ps_rows.keys()))
    rows = []
    for name in common[:15]:
        rows.append(
            {
                "measure_name": name,
                "equal_dataset_rank": eq_rows[name]["final_rank"],
                "per_stack_rank": ps_rows[name]["final_rank"],
                "equal_dataset_score": eq_rows[name]["generalization_score"],
                "per_stack_score": ps_rows[name]["generalization_score"],
            }
        )

    csv_path = table_csv_path("Table9_equal_vs_stack_weighted_comparison", "main")
    tex_path = table_tex_path("Table9_equal_vs_stack_weighted_comparison", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Comparison between equal-dataset and per-stack weighting strategies.",
    )


def export_table10_runtime(logger) -> bool:
    if not SINGLE_TIMING_SUMMARY_FILE.exists():
        logger.warning("Missing timing summary: %s", SINGLE_TIMING_SUMMARY_FILE)
        return False

    rows_in = load_csv_rows(SINGLE_TIMING_SUMMARY_FILE)
    by_measure: Dict[str, Dict[str, List[float]]] = {}

    for row in rows_in:
        m = row["measure_name"]
        by_measure.setdefault(m, {
            "native": [],
            "128": [],
            "512": [],
            "1024": [],
        })

        for key, dst_key in [
            ("native_avg_time_per_slice_sec", "native"),
            ("avg_time_per_slice_sec_128", "128"),
            ("avg_time_per_slice_sec_512", "512"),
            ("avg_time_per_slice_sec_1024", "1024"),
        ]:
            val = row.get(key, "")
            if val not in ("", None):
                try:
                    by_measure[m][dst_key].append(float(val))
                except Exception:
                    pass

    rows = []
    for measure_name, vals in by_measure.items():
        rows.append(
            {
                "measure_name": measure_name,
                "native_avg_time_per_slice_sec": np.mean(vals["native"]) if vals["native"] else np.nan,
                "avg_time_per_slice_sec_128": np.mean(vals["128"]) if vals["128"] else np.nan,
                "avg_time_per_slice_sec_512": np.mean(vals["512"]) if vals["512"] else np.nan,
                "avg_time_per_slice_sec_1024": np.mean(vals["1024"]) if vals["1024"] else np.nan,
            }
        )

    rows = sorted(rows, key=lambda x: x["native_avg_time_per_slice_sec"] if np.isfinite(x["native_avg_time_per_slice_sec"]) else 1e18)
    rows = rows[:15]

    csv_path = table_csv_path("Table10_runtime_multi_resolution", "main")
    tex_path = table_tex_path("Table10_runtime_multi_resolution", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Runtime comparison across image resolutions.",
    )


def export_table11_downstream(logger) -> bool:
    src = PAPER_TABLES_MAIN_CSV_DIR / "Table11_optional_downstream_task.csv"
    dst = table_csv_path("Table11_optional_downstream_task", "main")
    if not src.exists():
        logger.warning("Missing optional downstream CSV: %s", src)
        return False
    if src.resolve() != dst.resolve():
        copy_if_exists(src, dst, logger)
    return write_simple_latex_table(
        dst,
        table_tex_path("Table11_optional_downstream_task", "main"),
        caption="Optional downstream biomedical anchoring analysis. Proxy mode is reported explicitly when true labels are unavailable.",
    )


def export_table12_final_generalized_composite(logger) -> bool:
    final_expr = load_final_composite_expression()
    final_summary = load_final_refit_summary()
    if not final_expr:
        logger.warning("Missing final generalized composite expression JSON")
        return False

    rows = [
        {
            "result_type": final_expr.get("result_type", "final_refit_best_expression"),
            "source": final_expr.get("source", "all_dataset_refit_after_lodo_validation"),
            "seed": final_expr.get("seed", ""),
            "best_training_objective": final_expr.get("best_training_objective", ""),
            "best_all_dataset_score": final_expr.get("best_all_dataset_score", ""),
            "best_complexity": final_expr.get("best_complexity", ""),
            "num_nodes": final_expr.get("num_nodes", ""),
            "tree_height": final_expr.get("tree_height", ""),
            "train_datasets": " | ".join(str(x) for x in final_expr.get("train_datasets", [])),
            "terminals": " | ".join(str(x) for x in final_expr.get("terminals", [])),
            "best_expression": final_expr.get("best_expression", ""),
            "num_final_refit_seeds": final_summary.get("num_seeds", "") if final_summary else "",
            "mean_all_dataset_score": final_summary.get("mean_all_dataset_score", "") if final_summary else "",
            "std_all_dataset_score": final_summary.get("std_all_dataset_score", "") if final_summary else "",
        }
    ]

    csv_path = table_csv_path("Table12_final_generalized_composite", "main")
    tex_path = table_tex_path("Table12_final_generalized_composite", "main")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Final generalized composite focus measure obtained by all-dataset GP refit after LODO validation.",
    )


# -----------------------------------------------------------------------------
# Supplementary table exporters
# -----------------------------------------------------------------------------
def export_stable1_all_single_measure_results(logger) -> bool:
    rank_path = SINGLE_EVAL_SUPP_DIR / "all_single_rank_based.csv"
    value_path = SINGLE_EVAL_SUPP_DIR / "all_single_value_based_equal_dataset.csv"

    if not rank_path.exists() or not value_path.exists():
        logger.warning("Missing full single-measure result files")
        return False

    rank_rows = {r["measure_name"]: r for r in load_csv_rows(rank_path)}
    value_rows = {r["measure_name"]: r for r in load_csv_rows(value_path)}

    rows = []
    for name in sorted(set(rank_rows.keys()) & set(value_rows.keys())):
        rows.append(
            {
                "measure_name": name,
                "rank_based_final_rank": rank_rows[name]["final_rank"],
                "rank_based_score": rank_rows[name]["rank_generalization_score"],
                "value_based_final_rank": value_rows[name]["final_rank"],
                "value_based_score": value_rows[name]["generalization_score"],
            }
        )

    csv_path = table_csv_path("STable1_all_single_measure_results", "supplementary")
    tex_path = table_tex_path("STable1_all_single_measure_results", "supplementary")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(csv_path, tex_path, caption="Full single-measure summary table.")


def export_stable2_alpha_sensitivity(logger) -> bool:
    src_candidates = [
        PAPER_TABLES_SUPP_CSV_DIR / "STable2_alpha_sensitivity.csv",
        SINGLE_EVAL_SUPP_DIR / "alpha_sensitivity.csv",
    ]

    src = None
    for c in src_candidates:
        if c.exists():
            src = c
            break

    if src is None:
        logger.warning("Missing alpha sensitivity source CSV")
        return False

    dst = table_csv_path("STable2_alpha_sensitivity", "supplementary")
    if src.resolve() != dst.resolve():
        copy_if_exists(src, dst, logger)

    return write_simple_latex_table(dst, table_tex_path("STable2_alpha_sensitivity", "supplementary"),
                                    caption="Alpha sensitivity analysis.")


def export_stable3_metric_weight_sensitivity(logger) -> bool:
    src_candidates = [
        PAPER_TABLES_SUPP_CSV_DIR / "STable3_metric_weight_sensitivity.csv",
        SENSITIVITY_DIR / "metric_weight_scheme_paper_default_full.csv",
    ]

    dst = table_csv_path("STable3_metric_weight_sensitivity", "supplementary")

    # Preferred already-exported summary
    preferred = PAPER_TABLES_SUPP_CSV_DIR / "STable3_metric_weight_sensitivity.csv"
    if preferred.exists():
        if preferred.resolve() != dst.resolve():
            copy_if_exists(preferred, dst, logger)
        return write_simple_latex_table(dst, table_tex_path("STable3_metric_weight_sensitivity", "supplementary"),
                                        caption="Metric-weight sensitivity analysis.")

    # Fallback: synthesize summary from all per-scheme files
    rows = []
    for scheme_name in ALT_METRIC_WEIGHT_SCHEMES.keys():
        scheme_csv = SENSITIVITY_DIR / f"metric_weight_scheme_{scheme_name}_full.csv"
        if not scheme_csv.exists():
            continue
        scheme_rows = load_csv_rows(scheme_csv)
        if not scheme_rows:
            continue
        scheme_rows = sorted(scheme_rows, key=lambda x: float(x["final_rank"]))
        rows.append(
            {
                "scheme_name": scheme_name,
                "top1": scheme_rows[0]["measure_name"],
                "top3": " | ".join(r["measure_name"] for r in scheme_rows[:3]),
            }
        )

    if not rows:
        logger.warning("Missing metric-weight sensitivity sources")
        return False

    save_csv_rows(rows, dst)
    return write_simple_latex_table(dst, table_tex_path("STable3_metric_weight_sensitivity", "supplementary"),
                                    caption="Metric-weight sensitivity analysis.")


def export_stable4_gp_foldwise_results(logger) -> bool:
    preferred = PAPER_TABLES_SUPP_CSV_DIR / "STable4_gp_foldwise_results.csv"
    dst = table_csv_path("STable4_gp_foldwise_results", "supplementary")

    if preferred.exists():
        if preferred.resolve() != dst.resolve():
            copy_if_exists(preferred, dst, logger)
        return write_simple_latex_table(dst, table_tex_path("STable4_gp_foldwise_results", "supplementary"),
                                        caption="Foldwise leave-one-dataset-out GP results.")

    # fallback synthesis from GP summary json files
    if not GP_SUMMARIES_DIR.exists():
        logger.warning("Missing GP summaries directory")
        return False

    rows = []
    for path in sorted(GP_SUMMARIES_DIR.glob("heldout_*_summary.json")):
        payload = load_json(path)
        heldout = payload.get("held_out_dataset", "")
        fold_results = payload.get("seed_results", payload.get("fold_results", []))
        if not fold_results:
            continue
        heldout_scores = [float(r["heldout_score"]) for r in fold_results]
        best = min(fold_results, key=lambda x: float(x["heldout_score"]))
        rows.append(
            {
                "held_out_dataset": heldout,
                "num_seeds": len(fold_results),
                "mean_heldout_score": float(np.mean(heldout_scores)),
                "std_heldout_score": float(np.std(heldout_scores, ddof=0)),
                "best_expression": best["best_expression"],
            }
        )

    if not rows:
        logger.warning("Could not synthesize GP foldwise results")
        return False

    save_csv_rows(rows, dst)
    return write_simple_latex_table(dst, table_tex_path("STable4_gp_foldwise_results", "supplementary"),
                                    caption="Foldwise leave-one-dataset-out GP results.")


def export_stable5_composite_deduplication(logger) -> bool:
    src = GP_DEDUP_DIR / "deduplicated_best_expressions.json"
    if not src.exists():
        logger.warning("Missing deduplicated composite JSON: %s", src)
        return False

    payload = load_json(src)
    rows = []
    for row in payload:
        rows.append(
            {
                "expression": row.get("best_expression", row.get("expression", "")),
                "heldout_score": row.get("heldout_score", ""),
                "best_training_objective": row.get("best_training_objective", ""),
                "best_complexity": row.get("best_complexity", ""),
                "held_out_dataset": row.get("held_out_dataset", ""),
            }
        )

    csv_path = table_csv_path("STable5_composite_deduplication", "supplementary")
    tex_path = table_tex_path("STable5_composite_deduplication", "supplementary")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(csv_path, tex_path, caption="Deduplicated composite-expression summary.")


def export_stable6_label_provenance(logger) -> bool:
    if not LABEL_PROVENANCE_MANIFEST_FILE.exists():
        logger.warning("Missing label provenance disclosure: %s", LABEL_PROVENANCE_MANIFEST_FILE)
        return False
    provenance = load_json(LABEL_PROVENANCE_MANIFEST_FILE)
    loo_payload = load_json(LOO_VOTER_DISCLOSURE_FILE) if LOO_VOTER_DISCLOSURE_FILE.exists() else {}
    usage_rows = load_csv_rows(SINGLE_EVAL_SUPP_DIR / "label_source_usage_by_measure.csv") if (SINGLE_EVAL_SUPP_DIR / "label_source_usage_by_measure.csv").exists() else []

    usage_by_dataset: Dict[str, Dict[str, int]] = {}
    for row in usage_rows:
        dataset_name = str(row["dataset_name"])
        usage_by_dataset.setdefault(dataset_name, {"source": 0, "leave_one_out_surrogate": 0, "surrogate": 0})
        label_mode = str(row["label_source_used"])
        usage_by_dataset[dataset_name][label_mode] = usage_by_dataset[dataset_name].get(label_mode, 0) + 1

    rows: List[Dict[str, Any]] = []
    dataset_rows = provenance if isinstance(provenance, dict) else {}
    loo_by_dataset = loo_payload.get("datasets", {}) if isinstance(loo_payload, dict) else {}
    for dataset_name, info in dataset_rows.items():
        dataset_loo_rows = loo_by_dataset.get(dataset_name, [])
        default_voters = info.get("surrogate_voters", loo_payload.get("default_surrogate_voters", []))
        rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_label_mode": info.get("label_mode", ""),
                "source_labels_available": str(info.get("label_mode", "") == "source").lower(),
                "source_labels_file": info.get("source_label_file_detected", ""),
                "source_label_note": info.get("source_label_note", ""),
                "source_label_source_kind": info.get("source_label_source_kind", ""),
                "stack_count": info.get("stack_count", ""),
                "surrogate_labels_file": str(get_surrogate_label_file(dataset_name)),
                "loo_label_files_written": len(dataset_loo_rows),
                "default_voter_count": len(default_voters),
                "default_voters": " | ".join(str(v) for v in default_voters),
                "measures_using_source_labels": usage_by_dataset.get(dataset_name, {}).get("source", 0),
                "measures_using_leave_one_out_surrogate": usage_by_dataset.get(dataset_name, {}).get("leave_one_out_surrogate", 0),
                "measures_using_surrogate": usage_by_dataset.get(dataset_name, {}).get("surrogate", 0),
            }
        )

    csv_path = table_csv_path("STable6_label_provenance_and_loo_disclosure", "supplementary")
    tex_path = table_tex_path("STable6_label_provenance_and_loo_disclosure", "supplementary")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Dataset-level label provenance, source-versus-surrogate disclosure, and leave-one-out voter usage.",
    )


def export_stable7_rank_stability(logger) -> bool:
    if not RANK_STABILITY_RESULTS_FILE.exists():
        logger.warning("Missing rank-stability CSV: %s", RANK_STABILITY_RESULTS_FILE)
        return False
    rows = load_csv_rows(RANK_STABILITY_RESULTS_FILE)
    summary = load_json(RANK_STABILITY_SUMMARY_FILE) if RANK_STABILITY_SUMMARY_FILE.exists() else {}
    annotated_rows = []
    for row in rows:
        entry = dict(row)
        entry["stability_passed"] = summary.get("passed", "")
        entry["comparison_resolution"] = summary.get("comparison_resolution", "")
        annotated_rows.append(entry)
    csv_path = table_csv_path("STable7_rank_stability_summary", "supplementary")
    tex_path = table_tex_path("STable7_rank_stability_summary", "supplementary")
    save_csv_rows(annotated_rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Publication rank-stability comparison between the native subset and the 1024-pixel subset.",
    )


def export_stable8_composite_metric_profile(logger) -> bool:
    if not COMPOSITE_METRIC_PROFILE_FILE.exists():
        logger.warning("Missing composite metric profile CSV: %s", COMPOSITE_METRIC_PROFILE_FILE)
        return False
    dst = table_csv_path("STable8_top_composites_10_metric_profile", "supplementary")
    if COMPOSITE_METRIC_PROFILE_FILE.resolve() != dst.resolve():
        copy_if_exists(COMPOSITE_METRIC_PROFILE_FILE, dst, logger)
    return write_simple_latex_table(
        dst,
        table_tex_path("STable8_top_composites_10_metric_profile", "supplementary"),
        caption="Top composite operators evaluated under the same 10 autofocus metrics used for single measures.",
    )


def export_stable9_gp_lodo_vs_final_refit(logger) -> bool:
    rows = build_gp_lodo_vs_final_rows()
    if not rows:
        logger.warning("Missing LODO/final-refit rows")
        return False

    csv_path = table_csv_path("STable9_gp_lodo_vs_final_refit", "supplementary")
    tex_path = table_tex_path("STable9_gp_lodo_vs_final_refit", "supplementary")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Leave-one-dataset-out GP validation compared with the final all-dataset refit.",
    )


def export_stable10_gp_generation_traces(logger) -> bool:
    rows = collect_gp_generation_trace_rows()
    if not rows:
        logger.warning("Missing GP generation trace rows")
        return False

    csv_path = table_csv_path("STable10_gp_generation_traces", "supplementary")
    tex_path = table_tex_path("STable10_gp_generation_traces", "supplementary")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Generation-wise best-objective traces for all LODO and final-refit GP seeds.",
    )


def export_stable11_gp_seedwise_lodo_and_final(logger) -> bool:
    rows = collect_gp_seedwise_rows()
    if not rows:
        logger.warning("Missing seed-wise GP result rows")
        return False

    csv_path = table_csv_path("STable11_gp_seedwise_lodo_and_final", "supplementary")
    tex_path = table_tex_path("STable11_gp_seedwise_lodo_and_final", "supplementary")
    save_csv_rows(rows, csv_path)
    return write_simple_latex_table(
        csv_path,
        tex_path,
        caption="Seed-wise GP results for all LODO folds and the final all-dataset refit.",
    )


def export_stable12_gp_terminal_selection(logger) -> bool:
    src = GP_SUMMARIES_DIR / "selected_terminal_manifest.csv"
    if not src.exists():
        logger.warning("Missing selected terminal manifest CSV: %s", src)
        return False

    dst = table_csv_path("STable12_gp_terminal_selection_lodo_and_final", "supplementary")
    if src.resolve() != dst.resolve():
        copy_if_exists(src, dst, logger)
    return write_simple_latex_table(
        dst,
        table_tex_path("STable12_gp_terminal_selection_lodo_and_final", "supplementary"),
        caption="Terminal selection for each LODO fold and the final all-dataset refit.",
    )


# -----------------------------------------------------------------------------
# Index + manifests
# -----------------------------------------------------------------------------
def required_figure_paths_for_strict() -> List[Path]:
    required: List[Path] = [
        PAPER_FIGURES_MAIN_DIR / "Fig1_pipeline_overview.png",
        PAPER_FIGURES_MAIN_DIR / "Fig1_pipeline_overview.pdf",
    ]

    plot_ckpt = load_checkpoint(SINGLE_EVAL_MAIN_DIR / "plot_single_measure_results.checkpoint.json")
    if plot_ckpt.get("status") == "complete":
        required.extend([
            PAPER_FIGURES_MAIN_DIR / "Fig2_single_measure_heatmap.png",
            PAPER_FIGURES_MAIN_DIR / "Fig2_single_measure_heatmap.pdf",
            PAPER_FIGURES_MAIN_DIR / "Fig3_top_operator_bootstrap_ci.png",
            PAPER_FIGURES_MAIN_DIR / "Fig3_top_operator_bootstrap_ci.pdf",
            PAPER_FIGURES_MAIN_DIR / "Fig5_representative_single_focus_curves.png",
            PAPER_FIGURES_MAIN_DIR / "Fig5_representative_single_focus_curves.pdf",
            PAPER_FIGURES_SUPP_DIR / "SFig1_dataset_specific_focus_curves.png",
            PAPER_FIGURES_SUPP_DIR / "SFig1_dataset_specific_focus_curves.pdf",
        ])
        if SINGLE_TIMING_SUMMARY_FILE.exists():
            required.extend([
                PAPER_FIGURES_MAIN_DIR / "Fig4_resolution_sensitivity.png",
                PAPER_FIGURES_MAIN_DIR / "Fig4_resolution_sensitivity.pdf",
                PAPER_FIGURES_MAIN_DIR / "Fig10_runtime_scaling.png",
                PAPER_FIGURES_MAIN_DIR / "Fig10_runtime_scaling.pdf",
            ])

    gp_eval_ckpt = load_checkpoint(COMPOSITE_MAIN_DIR / "evaluate_composites.checkpoint.json")
    if gp_eval_ckpt.get("status") == "complete":
        required.extend([
            PAPER_FIGURES_MAIN_DIR / "Fig6_representative_composite_vs_single_curves.png",
            PAPER_FIGURES_MAIN_DIR / "Fig6_representative_composite_vs_single_curves.pdf",
            PAPER_FIGURES_MAIN_DIR / "Fig9_gp_lodo_summary.png",
            PAPER_FIGURES_MAIN_DIR / "Fig9_gp_lodo_summary.pdf",
            PAPER_FIGURES_SUPP_DIR / "SFig2_gp_convergence.png",
            PAPER_FIGURES_SUPP_DIR / "SFig2_gp_convergence.pdf",
            PAPER_FIGURES_SUPP_DIR / "SFig3_terminal_frequency.png",
            PAPER_FIGURES_SUPP_DIR / "SFig3_terminal_frequency.pdf",
            PAPER_FIGURES_SUPP_DIR / "SFig4_expression_equivalence_clusters.png",
            PAPER_FIGURES_SUPP_DIR / "SFig4_expression_equivalence_clusters.pdf",
        ])
        if (GP_SUMMARIES_DIR / "final_composite_expression.json").exists():
            required.extend([
                PAPER_FIGURES_MAIN_DIR / "Fig12_lodo_vs_final_refit.png",
                PAPER_FIGURES_MAIN_DIR / "Fig12_lodo_vs_final_refit.pdf",
                PAPER_FIGURES_SUPP_DIR / "SFig5_gp_convergence_by_fold.png",
                PAPER_FIGURES_SUPP_DIR / "SFig5_gp_convergence_by_fold.pdf",
                PAPER_FIGURES_SUPP_DIR / "SFig6_final_refit_convergence.png",
                PAPER_FIGURES_SUPP_DIR / "SFig6_final_refit_convergence.pdf",
                PAPER_FIGURES_SUPP_DIR / "SFig7_gp_seedwise_score_distribution.png",
                PAPER_FIGURES_SUPP_DIR / "SFig7_gp_seedwise_score_distribution.pdf",
            ])

    stats_ckpt = load_checkpoint(STATISTICS_DIR / "run_statistics_and_sensitivity.checkpoint.json")
    if stats_ckpt.get("status") == "complete":
        required.extend([
            PAPER_FIGURES_MAIN_DIR / "Fig7_nemenyi_cd_overall_rank.png",
            PAPER_FIGURES_MAIN_DIR / "Fig7_nemenyi_cd_overall_rank.pdf",
            PAPER_FIGURES_MAIN_DIR / "Fig8_nemenyi_cd_accuracy_rank.png",
            PAPER_FIGURES_MAIN_DIR / "Fig8_nemenyi_cd_accuracy_rank.pdf",
        ])

    downstream_ckpt = load_checkpoint(PAPER_MANIFESTS_DIR / "optional_downstream_baseline.checkpoint.json")
    if downstream_ckpt.get("status") == "complete":
        required.extend([
            PAPER_FIGURES_MAIN_DIR / "Fig11_optional_downstream_task.png",
            PAPER_FIGURES_MAIN_DIR / "Fig11_optional_downstream_task.pdf",
        ])

    return required


def required_table_paths_for_strict(export_results: Mapping[str, bool]) -> List[Path]:
    required: List[Path] = []
    if LABEL_SOURCE_MANIFEST_FILE.exists():
        required.extend([
            table_csv_path("STable6_label_provenance_and_loo_disclosure", "supplementary"),
            table_tex_path("STable6_label_provenance_and_loo_disclosure", "supplementary"),
        ])

    single_ckpt = load_checkpoint(SINGLE_EVAL_MAIN_DIR / "evaluate_single_measures.checkpoint.json")
    if single_ckpt.get("status") == "complete":
        required.extend([
            table_csv_path("STable7_rank_stability_summary", "supplementary"),
            table_tex_path("STable7_rank_stability_summary", "supplementary"),
        ])

    composite_ckpt = load_checkpoint(COMPOSITE_MAIN_DIR / "evaluate_composites.checkpoint.json")
    if composite_ckpt.get("status") == "complete":
        required.extend([
            table_csv_path("STable8_top_composites_10_metric_profile", "supplementary"),
            table_tex_path("STable8_top_composites_10_metric_profile", "supplementary"),
        ])
        if (GP_SUMMARIES_DIR / "final_composite_expression.json").exists():
            required.extend([
                table_csv_path("Table12_final_generalized_composite", "main"),
                table_tex_path("Table12_final_generalized_composite", "main"),
                table_csv_path("STable9_gp_lodo_vs_final_refit", "supplementary"),
                table_tex_path("STable9_gp_lodo_vs_final_refit", "supplementary"),
                table_csv_path("STable10_gp_generation_traces", "supplementary"),
                table_tex_path("STable10_gp_generation_traces", "supplementary"),
                table_csv_path("STable11_gp_seedwise_lodo_and_final", "supplementary"),
                table_tex_path("STable11_gp_seedwise_lodo_and_final", "supplementary"),
                table_csv_path("STable12_gp_terminal_selection_lodo_and_final", "supplementary"),
                table_tex_path("STable12_gp_terminal_selection_lodo_and_final", "supplementary"),
            ])

    for spec in ALL_TABLES:
        if not export_results.get(spec.key, False):
            continue
        dirs = get_table_output_dirs(spec.output_group)
        required.append(dirs["csv"] / f"{spec.key}.csv")
        required.append(dirs["latex"] / f"{spec.key}.tex")
    return required


def collect_auxiliary_assets() -> List[Dict[str, Any]]:
    candidates = [
        PAPER_TABLES_MAIN_CSV_DIR / "Table11_optional_downstream_task.csv",
        PAPER_FIGURES_MAIN_DIR / "Fig11_optional_downstream_task.png",
        PAPER_FIGURES_MAIN_DIR / "Fig11_optional_downstream_task.pdf",
        PAPER_MANIFESTS_DIR / "optional_downstream_report.json",
        PAPER_MANIFESTS_DIR / "optional_downstream_limitation.md",
        PAPER_MANIFESTS_DIR / "optional_downstream_baseline.checkpoint.json",
        PAPER_MANIFESTS_DIR / "paper_export_summary.json",
        PAPER_MANIFESTS_DIR / "full_pipeline_summary.json",
        PAPER_MANIFESTS_DIR / "publication_procedure_summary.json",
        GP_SUMMARIES_DIR / "final_composite_expression.json",
        GP_SUMMARIES_DIR / "final_refit_summary.json",
        GP_SUMMARIES_DIR / "final_refit_seed_results.json",
    ]
    auxiliary: List[Dict[str, Any]] = []
    for path in candidates:
        if path.exists():
            auxiliary.append({
                "path": str(path),
                "name": path.name,
                "suffix": path.suffix,
            })
    return auxiliary


def write_asset_index(
    figure_manifest: List[Dict[str, Any]],
    table_manifest: List[Dict[str, Any]],
    compatibility_layout: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "paper_root": str(PAPER_DIR),
        "figure_manifest": figure_manifest,
        "table_manifest": table_manifest,
        "captions_dir": str(PAPER_CAPTIONS_DIR),
        "compatibility_layout": dict(compatibility_layout),
        "auxiliary_assets": collect_auxiliary_assets(),
    }
    save_json(payload, PAPER_ASSET_INDEX_FILE)

    md_lines: List[str] = []
    md_lines.append("# Paper Asset Index")
    md_lines.append("")
    md_lines.append(f"- Root: `{PAPER_DIR}`")
    md_lines.append(f"- Machine-readable index: `{PAPER_ASSET_INDEX_FILE.name}`")
    md_lines.append("")
    md_lines.append("## Figures")
    md_lines.append("")
    for item in figure_manifest:
        outputs = item.get("output_exists", {})
        status = ", ".join(f"{name}:{'ok' if exists else 'missing'}" for name, exists in outputs.items())
        md_lines.append(f"- `{item['key']}` ({item['output_group']}): {status}")
    md_lines.append("")
    md_lines.append("## Tables")
    md_lines.append("")
    for item in table_manifest:
        outputs = item.get("output_exists", {})
        status = ", ".join(f"{name}:{'ok' if exists else 'missing'}" for name, exists in outputs.items())
        md_lines.append(f"- `{item['key']}` ({item['output_group']}): {status}")

    PAPER_ASSET_INDEX_FILE.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")
    return payload


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    ensure_output_dirs()
    validate_all_settings()
    validate_environment()

    log_file = LOGS_DIR / f"export_paper_assets_{run_mode}.log"
    logger = get_logger("export_paper_assets", log_file=log_file)

    logger.info("Starting final paper asset export stage")
    write_caption_placeholders()
    figure_export_results = export_publication_figures(logger)

    # Export main tables
    export_results = {
        "Table2_dataset_summary": export_table2_dataset_summary(logger),
        "Table3_metric_definitions": export_table3_metric_definitions(logger),
        "Table4_metric_weights_and_alpha": export_table4_metric_weights_and_alpha(logger),
        "Table5_top10_single_rank_based": export_table5_rank_based(logger),
        "Table6_top10_single_value_based": export_table6_value_based(logger),
        "Table7_top10_composites_common_scoring": export_table7_composites(logger),
        "Table8_true_vs_surrogate_label_split": export_table8_label_split(logger),
        "Table9_equal_vs_stack_weighted_comparison": export_table9_weighting_comparison(logger),
        "Table10_runtime_multi_resolution": export_table10_runtime(logger),
        "Table11_optional_downstream_task": export_table11_downstream(logger),
        "Table12_final_generalized_composite": export_table12_final_generalized_composite(logger),
        "STable1_all_single_measure_results": export_stable1_all_single_measure_results(logger),
        "STable2_alpha_sensitivity": export_stable2_alpha_sensitivity(logger),
        "STable3_metric_weight_sensitivity": export_stable3_metric_weight_sensitivity(logger),
        "STable4_gp_foldwise_results": export_stable4_gp_foldwise_results(logger),
        "STable5_composite_deduplication": export_stable5_composite_deduplication(logger),
        "STable6_label_provenance_and_loo_disclosure": export_stable6_label_provenance(logger),
        "STable7_rank_stability_summary": export_stable7_rank_stability(logger),
        "STable8_top_composites_10_metric_profile": export_stable8_composite_metric_profile(logger),
        "STable9_gp_lodo_vs_final_refit": export_stable9_gp_lodo_vs_final_refit(logger),
        "STable10_gp_generation_traces": export_stable10_gp_generation_traces(logger),
        "STable11_gp_seedwise_lodo_and_final": export_stable11_gp_seedwise_lodo_and_final(logger),
        "STable12_gp_terminal_selection_lodo_and_final": export_stable12_gp_terminal_selection(logger),
    }

    # Build manifests
    figure_manifest = enrich_asset_manifest_entries(build_enriched_figure_manifest())
    table_manifest = enrich_asset_manifest_entries(build_enriched_table_manifest())

    save_json(figure_manifest, PAPER_FIGURE_MANIFEST_FILE)
    save_json(table_manifest, PAPER_TABLE_MANIFEST_FILE)
    compatibility_layout = ensure_table_compatibility_dirs(logger)
    asset_index_payload = write_asset_index(figure_manifest, table_manifest, compatibility_layout)

    # Validate expected files
    figure_status = validate_expected_asset_files(expected_figure_paths(), strict=False)
    table_status = validate_expected_asset_files(expected_table_paths(), strict=False)
    caption_status = validate_expected_asset_files(expected_caption_paths(), strict=False)
    strict_required_figure_status = validate_expected_asset_files(required_figure_paths_for_strict(), strict=False)
    strict_required_table_status = validate_expected_asset_files(required_table_paths_for_strict(export_results), strict=False)

    if args.strict:
        validate_expected_asset_files(expected_caption_paths(), strict=True)
        validate_expected_asset_files(required_figure_paths_for_strict(), strict=True)
        validate_expected_asset_files(required_table_paths_for_strict(export_results), strict=True)

    # Export summary
    export_summary = {
        "run_mode": run_mode,
        "figure_exports": figure_export_results,
        "table_exports": export_results,
        "figure_status": figure_status,
        "table_status": table_status,
        "caption_status": caption_status,
        "strict_required_figure_status": strict_required_figure_status,
        "strict_required_table_status": strict_required_table_status,
        "figure_manifest": str(PAPER_FIGURE_MANIFEST_FILE),
        "table_manifest": str(PAPER_TABLE_MANIFEST_FILE),
        "asset_index": str(PAPER_ASSET_INDEX_FILE),
        "asset_index_markdown": str(PAPER_ASSET_INDEX_FILE.with_suffix(".md")),
        "compatibility_layout": compatibility_layout,
        "asset_index_payload": asset_index_payload,
        "existing_figure_files": summarize_existing_files(PAPER_FIGURE_MANIFEST_FILE.parent.parent / "figures", suffixes=[".png", ".pdf"]),
        "existing_table_files": summarize_existing_files(PAPER_TABLE_MANIFEST_FILE.parent.parent / "tables", suffixes=[".csv", ".tex"]),
    }
    save_json(export_summary, PAPER_MANIFESTS_DIR / "paper_export_summary.json")

    checkpoint_path = PAPER_MANIFESTS_DIR / "export_paper_assets.checkpoint.json"
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="export_paper_assets",
        status="complete",
        details=export_summary,
    )

    logger.info("Paper asset export stage complete")
    logger.info("Figure manifest -> %s", PAPER_FIGURE_MANIFEST_FILE)
    logger.info("Table manifest  -> %s", PAPER_TABLE_MANIFEST_FILE)
    logger.info("Asset index     -> %s", PAPER_ASSET_INDEX_FILE)
    logger.info("Checkpoint      -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
