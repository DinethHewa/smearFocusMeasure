"""Publication-oriented composite and GP figures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.paths import GP_RUNS_DIR
from src.evaluation.autofocus_metrics import normalize_focus_curve
from src.gp.deap_search import compile_expression, load_terminal_curves_for_dataset
from src.plots.focus_curves import (
    apply_publication_format,
    load_curve_file,
    representative_curve_index,
    require_matplotlib,
    save_figure_multi,
    plt,
)
from src.utils.validation import load_csv_rows, load_json


ReferenceLabelResolver = Callable[[str], Tuple[Optional[np.ndarray], Optional[str]]]


def _split_terminals(value: str) -> List[str]:
    return [item.strip() for item in re.split(r"[;|]", str(value)) if item.strip()]


def load_best_composite_candidate(
    *,
    table7_csv: Path,
    comparison_csv: Path,
    dedup_json: Path,
) -> Dict[str, Any] | None:
    if table7_csv.exists():
        rows = load_csv_rows(table7_csv)
        if rows:
            best = sorted(rows, key=lambda row: float(row.get("common_value_final_rank", "1e18")))[0]
            return {
                "composite_id": best.get("composite_id", "best_composite"),
                "expression": str(best["expression"]),
                "terminals": _split_terminals(best.get("terminals", "")),
            }

    if comparison_csv.exists():
        rows = load_csv_rows(comparison_csv)
        for row in rows:
            if row.get("comparison_item") == "best_composite_under_common_value_scoring":
                return {
                    "composite_id": row.get("name", "best_composite"),
                    "expression": str(row["expression"]),
                    "terminals": [],
                }

    if dedup_json.exists():
        payload = load_json(dedup_json)
        if payload:
            row = payload[0]
            return {
                "composite_id": row.get("composite_id", "best_composite"),
                "expression": str(row["best_expression"]),
                "terminals": list(row.get("terminals", [])),
            }
    return None


def plot_pipeline_overview(
    *,
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.axis("off")

    phase_boxes = [
        ("Phase A", "Corrected single-measure foundation\n01 stacks -> 02 labels -> 03 curves -> 04 evaluation -> 05 figures"),
        ("Phase B", "Corrected composite stage\n06 LODO GP -> 07 composite evaluation -> 08 statistics and sensitivity"),
        ("Phase C", "Scope anchoring and export\n09 downstream anchor -> 10 paper assets -> 11 orchestrator summary"),
    ]
    xs = [0.05, 0.37, 0.69]
    width = 0.26
    for (title, body), x in zip(phase_boxes, xs):
        rect = plt.Rectangle((x, 0.35), width, 0.35, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width / 2.0, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + width / 2.0, 0.48, body, ha="center", va="center", fontsize=10)
    for x in (0.31, 0.63):
        ax.annotate("", xy=(x + 0.04, 0.525), xytext=(x, 0.525), arrowprops=dict(arrowstyle="->", linewidth=2))

    ax.text(
        0.5,
        0.18,
        "Publication gates: freeze single-measure outputs, verify 1024 rank stability, then allow GP and strict export.",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.set_title(spec.title, fontsize=13)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_composite_vs_single_focus_curves(
    *,
    dataset_names: Sequence[str],
    curve_path_resolver,
    reference_label_resolver: ReferenceLabelResolver,
    best_composite: Mapping[str, Any],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    logger=None,
) -> Dict[str, Any]:
    require_matplotlib()
    terminal_names = list(best_composite.get("terminals", []))
    if not terminal_names:
        raise ValueError("Best composite terminal list is required to plot Fig6")
    func = compile_expression(str(best_composite["expression"]), terminal_names)

    fig, axes = plt.subplots(
        nrows=len(dataset_names),
        ncols=1,
        figsize=(8.8, 3.2 * len(dataset_names)),
        squeeze=False,
    )
    plotted_any = False

    for row_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, 0]
        br_path = curve_path_resolver(dataset_name, "Brenner Gradient")
        glcm_path = curve_path_resolver(dataset_name, "GLCM Contrast")
        if not br_path.exists() or not glcm_path.exists():
            if logger is not None:
                logger.warning("[%s] missing Brenner/GLCM curves for Fig6", dataset_name)
            continue
        br_curves = load_curve_file(br_path)
        glcm_curves = load_curve_file(glcm_path)
        terminal_curves = load_terminal_curves_for_dataset(dataset_name, terminal_names)

        composite_curves: List[np.ndarray] = []
        for stack_idx in range(len(br_curves)):
            args = [np.asarray(terminal_curves[name][stack_idx], dtype=np.float64) for name in terminal_names]
            curve_raw = np.asarray(func(*args), dtype=np.float64).reshape(-1)
            if curve_raw.size == 1:
                curve_raw = np.repeat(curve_raw.item(), len(args[0]))
            composite_curves.append(normalize_focus_curve(curve_raw))

        rep_idx = representative_curve_index(br_curves)
        ax.plot(np.arange(len(br_curves[rep_idx])), br_curves[rep_idx], linewidth=2.0, label="Brenner Gradient")
        ax.plot(np.arange(len(glcm_curves[rep_idx])), glcm_curves[rep_idx], linewidth=2.0, label="GLCM Contrast")
        ax.plot(
            np.arange(len(composite_curves[rep_idx])),
            composite_curves[rep_idx],
            linewidth=2.2,
            label=str(best_composite.get("composite_id", "Best composite")),
        )

        labels, provenance = reference_label_resolver(str(dataset_name))
        title_suffix = "reference unavailable"
        if labels is not None and rep_idx < len(labels):
            label_idx = int(labels[rep_idx])
            if provenance == "source":
                ax.axvline(label_idx, color="black", linestyle="-", linewidth=1.4, label="Source best-focus")
                title_suffix = "source reference"
            else:
                ax.axvline(label_idx, color="gray", linestyle="--", linewidth=1.4, label="Surrogate reference")
                title_suffix = f"{provenance or 'surrogate'} reference"

        ax.set_title(f"{dataset_name} ({title_suffix})")
        ax.set_xlabel("Slice index")
        ax.set_ylabel("Normalized focus score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncols=2)
        apply_publication_format(ax)
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("Could not plot Fig6 because no dataset had the required curves")

    fig.suptitle(spec.title, fontsize=12)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_gp_lodo_summary(
    *,
    fold_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    ordered = list(fold_rows)
    datasets = [str(row["held_out_dataset"]) for row in ordered]
    means = np.asarray([float(row["mean_heldout_score"]) for row in ordered], dtype=np.float64)
    stds = np.asarray([float(row.get("std_heldout_score", 0.0)) for row in ordered], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(datasets))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Held-out dataset")
    ax.set_ylabel("Held-out corrected score")
    ax.set_title(spec.title)
    ax.grid(True, axis="y", alpha=0.3)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_gp_convergence(
    *,
    gp_runs_dir: Path = GP_RUNS_DIR,
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    logbooks = sorted(gp_runs_dir.glob("heldout_*/seed_*/logbook.csv"))
    if not logbooks:
        raise FileNotFoundError(f"No GP logbooks found under {gp_runs_dir}")

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for path in logbooks:
        rows = load_csv_rows(path)
        gens = [int(row["generation"]) for row in rows]
        scores = [float(row["best_generalization_score"]) for row in rows]
        heldout = path.parents[1].name.replace("heldout_", "")
        seed = path.parent.name.replace("seed_", "")
        ax.plot(gens, scores, linewidth=1.6, alpha=0.85, label=f"{heldout}/seed{seed}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best training objective")
    ax.set_title(spec.title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def plot_gp_lodo_vs_final_refit(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    lodo_rows = [row for row in summary_rows if str(row.get("protocol", "")) == "lodo"]
    final_rows = [row for row in summary_rows if str(row.get("protocol", "")) == "final_refit"]
    if not lodo_rows and not final_rows:
        raise ValueError("No LODO/final-refit rows are available for Fig12")

    labels: List[str] = []
    means: List[float] = []
    errors: List[float] = []
    colors: List[str] = []
    for row in lodo_rows:
        labels.append(str(row.get("dataset_scope", row.get("held_out_dataset", ""))))
        means.append(_safe_float(row.get("mean_score")))
        errors.append(_safe_float(row.get("std_score"), 0.0))
        colors.append("#4C78A8")

    for row in final_rows:
        labels.append("Final\nall datasets")
        means.append(_safe_float(row.get("best_score")))
        errors.append(_safe_float(row.get("std_score"), 0.0))
        colors.append("#F58518")

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Corrected GP score")
    ax.set_xlabel("Validation or refit scope")
    ax.set_title(spec.title)
    ax.grid(True, axis="y", alpha=0.3)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_gp_convergence_by_fold(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    lodo_rows = [row for row in trace_rows if str(row.get("protocol", "")) == "lodo"]
    if not lodo_rows:
        raise ValueError("No LODO generation traces are available")

    by_fold_seed: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for row in lodo_rows:
        fold = str(row.get("dataset_scope", row.get("held_out_dataset", "")))
        seed = str(row.get("seed", ""))
        generation = int(_safe_float(row.get("generation_number"), 0.0))
        score = _safe_float(row.get("best_generalization_score"))
        if generation <= 0 or not np.isfinite(score):
            continue
        by_fold_seed.setdefault(fold, {}).setdefault(seed, []).append((generation, score))

    if not by_fold_seed:
        raise ValueError("No finite LODO generation traces are available")

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for fold_index, (fold, seed_map) in enumerate(sorted(by_fold_seed.items())):
        color = color_cycle[fold_index % len(color_cycle)] if color_cycle else None
        generation_values = sorted({gen for rows in seed_map.values() for gen, _score in rows})
        mean_scores: List[float] = []
        lower_scores: List[float] = []
        upper_scores: List[float] = []
        for generation in generation_values:
            values = [
                score
                for rows in seed_map.values()
                for gen, score in rows
                if gen == generation and np.isfinite(score)
            ]
            mean_scores.append(float(np.mean(values)))
            lower_scores.append(float(np.min(values)))
            upper_scores.append(float(np.max(values)))
        ax.plot(generation_values, mean_scores, linewidth=2.0, label=fold, color=color)
        ax.fill_between(generation_values, lower_scores, upper_scores, alpha=0.12, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best training objective")
    ax.set_title(spec.title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Held-out fold", fontsize=8)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_gp_final_refit_convergence(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    final_rows = [row for row in trace_rows if str(row.get("protocol", "")) == "final_refit"]
    if not final_rows:
        raise ValueError("No final-refit generation traces are available")

    by_seed: Dict[str, List[Tuple[int, float]]] = {}
    for row in final_rows:
        seed = str(row.get("seed", ""))
        generation = int(_safe_float(row.get("generation_number"), 0.0))
        score = _safe_float(row.get("best_generalization_score"))
        if generation <= 0 or not np.isfinite(score):
            continue
        by_seed.setdefault(seed, []).append((generation, score))

    if not by_seed:
        raise ValueError("No finite final-refit generation traces are available")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for seed, rows in sorted(by_seed.items()):
        ordered = sorted(rows)
        ax.plot(
            [gen for gen, _score in ordered],
            [score for _gen, score in ordered],
            linewidth=2.0,
            label=f"seed {seed}",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best all-dataset training objective")
    ax.set_title(spec.title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_gp_seedwise_score_distribution(
    *,
    seed_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    grouped: Dict[str, List[float]] = {}
    for row in seed_rows:
        protocol = str(row.get("protocol", ""))
        if protocol == "final_refit":
            label = "Final"
            score = _safe_float(row.get("all_dataset_score", row.get("best_training_objective")))
        else:
            label = str(row.get("dataset_scope", row.get("held_out_dataset", "")))
            score = _safe_float(row.get("heldout_score"))
        if label and np.isfinite(score):
            grouped.setdefault(label, []).append(score)

    if not grouped:
        raise ValueError("No seed-wise GP scores are available")

    labels = sorted([label for label in grouped if label != "Final"])
    if "Final" in grouped:
        labels.append("Final")
    data = [grouped[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_xlabel("Held-out fold or final refit")
    ax.set_ylabel("Seed-level GP score")
    ax.set_title(spec.title)
    ax.grid(True, axis="y", alpha=0.3)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_terminal_frequency(
    *,
    dedup_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
) -> Dict[str, Any]:
    require_matplotlib()
    counts: Dict[str, int] = {}
    for row in dedup_rows:
        for terminal in row.get("terminals", []):
            counts[str(terminal)] = counts.get(str(terminal), 0) + 1
    if not counts:
        raise ValueError("No terminal counts available for SFig3")

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    names = [item[0] for item in ordered]
    values = [item[1] for item in ordered]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    y = np.arange(len(names))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency across retained composites")
    ax.set_title(spec.title)
    ax.grid(True, axis="x", alpha=0.3)
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


def plot_expression_equivalence_clusters(
    *,
    dedup_rows: Sequence[Mapping[str, Any]],
    spec,
    output_dir: Path,
    extensions: Sequence[str],
    figure_dpi: int,
    top_n: int = 10,
) -> Dict[str, Any]:
    require_matplotlib()
    usable = [row for row in dedup_rows if row.get("mean_heldout_curve")]
    if len(usable) < 2:
        raise ValueError("Need at least two retained composites to plot expression equivalence clusters")

    usable = usable[:top_n]
    names = [str(row.get("composite_id", row.get("best_expression", f"C{i+1}"))) for i, row in enumerate(usable)]
    curves = [np.asarray(row["mean_heldout_curve"], dtype=np.float64).reshape(-1) for row in usable]
    matrix = np.full((len(curves), len(curves)), np.nan, dtype=np.float64)
    for i, curve_a in enumerate(curves):
        for j, curve_b in enumerate(curves):
            if np.std(curve_a) <= 0.0 or np.std(curve_b) <= 0.0:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = float(np.corrcoef(curve_a, curve_b)[0, 1])

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(spec.title)
    fig.colorbar(image, ax=ax, label="Curve correlation")
    apply_publication_format(ax)
    output_base = output_dir / spec.key
    written = save_figure_multi(fig, output_base, extensions, figure_dpi=figure_dpi)
    return {"figure_key": spec.key, "files": written, "description": spec.description}


__all__ = [
    "ReferenceLabelResolver",
    "load_best_composite_candidate",
    "plot_pipeline_overview",
    "plot_composite_vs_single_focus_curves",
    "plot_gp_lodo_summary",
    "plot_gp_convergence",
    "plot_gp_lodo_vs_final_refit",
    "plot_gp_convergence_by_fold",
    "plot_gp_final_refit_convergence",
    "plot_gp_seedwise_score_distribution",
    "plot_terminal_frequency",
    "plot_expression_equivalence_clusters",
]
