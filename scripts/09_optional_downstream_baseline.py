# scripts/09_optional_downstream_baseline.py

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    COMPOSITE_MAIN_DIR,
    DATASET_ORDER,
    LOGS_DIR,
    PAPER_FIGURES_MAIN_DIR,
    PAPER_MANIFESTS_DIR,
    PAPER_TABLES_MAIN_CSV_DIR,
    RAW_DATASETS,
    ensure_output_dirs,
    get_single_norm_curve_file,
    get_source_label_file,
    get_stack_file,
    get_surrogate_label_file,
)
from config.settings import (
    DEFAULT_RUN_MODE,
    FIGURE_DPI,
    validate_all_settings,
)
from src.measures.focus_measure_library import build_focus_measure_registry
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed
from src.utils.validation import (
    load_csv_rows,
    load_json,
    save_csv_rows,
    save_json,
    validate_environment,
    validate_pipeline_prerequisites,
    write_checkpoint,
)

# -----------------------------------------------------------------------------
# Optional sklearn import
# -----------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import accuracy_score, balanced_accuracy_score  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BEST_COMPOSITE_TABLE = COMPOSITE_MAIN_DIR / "best_composite_vs_best_single.csv"
BEST_COMPOSITE_SUMMARY = PAPER_TABLES_MAIN_CSV_DIR / "Table7_top10_composites_common_scoring.csv"
OPTIONAL_DOWNSTREAM_TABLE = PAPER_TABLES_MAIN_CSV_DIR / "Table11_optional_downstream_task.csv"
OPTIONAL_DOWNSTREAM_FIGURE = PAPER_FIGURES_MAIN_DIR / "Fig11_optional_downstream_task"
OPTIONAL_DOWNSTREAM_REPORT = PAPER_MANIFESTS_DIR / "optional_downstream_report.json"
OPTIONAL_DOWNSTREAM_CHECKPOINT = PAPER_MANIFESTS_DIR / "optional_downstream_baseline.checkpoint.json"
OPTIONAL_DOWNSTREAM_LIMITATION_NOTE = PAPER_MANIFESTS_DIR / "optional_downstream_limitation.md"


def require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "This script requires matplotlib. Install it in the environment used for the paper pipeline."
        ) from MATPLOTLIB_IMPORT_ERROR


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional downstream validation / BSPC anchoring experiment"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test mode")
    parser.add_argument("--full-run", action="store_true", help="Run full mode")
    parser.add_argument(
        "--dataset",
        type=str,
        default="WBC",
        choices=["WBC", "TBI"],
        help="Preferred dataset for downstream analysis",
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
# Helper utilities
# -----------------------------------------------------------------------------
def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if is_image_file(p)], key=lambda p: natural_key(p.name))


def list_all_image_dirs(root: Path) -> List[Path]:
    candidate_dirs: List[Path] = []
    if any(is_image_file(p) for p in root.iterdir()):
        candidate_dirs.append(root)

    for p in root.rglob("*"):
        if p.is_dir():
            try:
                if any(is_image_file(x) for x in p.iterdir()):
                    candidate_dirs.append(p)
            except PermissionError:
                continue

    return sorted(set(candidate_dirs), key=lambda x: natural_key(str(x)))


def discover_stack_folders(dataset_root: Path) -> List[Path]:
    candidate_dirs = list_all_image_dirs(dataset_root)
    valid = []
    for folder in candidate_dirs:
        if len(list_image_files(folder)) >= 2:
            valid.append(folder)
    return valid


def load_curve_file(path: Path) -> List[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(x, dtype=np.float64).reshape(-1) for x in arr]


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    cmin = float(np.min(curve))
    cmax = float(np.max(curve))
    if cmax - cmin <= 1e-12:
        return np.zeros_like(curve, dtype=np.float64)
    return (curve - cmin) / (cmax - cmin + 1e-12)


def load_labels(dataset_name: str) -> np.ndarray:
    source_path = get_source_label_file(dataset_name)
    if source_path.exists():
        return np.load(source_path, allow_pickle=False).astype(int).reshape(-1)
    surrogate_path = get_surrogate_label_file(dataset_name)
    if surrogate_path.exists():
        return np.load(surrogate_path, allow_pickle=False).astype(int).reshape(-1)
    raise FileNotFoundError(f"No labels found for dataset {dataset_name}")


def laplacian_variance(image: np.ndarray) -> float:
    x = np.asarray(image, dtype=np.float64)
    kernel = np.array(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=np.float64,
    )
    padded = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + x.shape[0], j:j + x.shape[1]]
    return float(np.var(out))


def gradient_energy(image: np.ndarray) -> float:
    x = np.asarray(image, dtype=np.float64)
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, 1:-1] = x[:, 2:] - x[:, :-2]
    gy[1:-1, :] = x[2:, :] - x[:-2, :]
    return float(np.sum(gx ** 2 + gy ** 2))


def extract_simple_features(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    hist, _ = np.histogram(x, bins=16, range=(float(x.min()), float(x.max()) + 1e-12))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-12)

    feats = [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.var(x)),
        laplacian_variance(x),
        gradient_energy(x),
        float(np.max(x) - np.min(x)),
    ]
    return np.concatenate([np.asarray(feats, dtype=np.float64), hist])


# -----------------------------------------------------------------------------
# Strategy selection
# -----------------------------------------------------------------------------
def get_brenner_predicted_indices(dataset_name: str) -> List[int]:
    curve_path = get_single_norm_curve_file(dataset_name, "Brenner Gradient")
    curves = load_curve_file(curve_path)
    return [int(np.argmax(c)) for c in curves]


def load_best_composite_expression() -> Optional[Dict[str, Any]]:
    if BEST_COMPOSITE_SUMMARY.exists():
        rows = load_csv_rows(BEST_COMPOSITE_SUMMARY)
        if rows:
            rows_sorted = sorted(rows, key=lambda x: float(x["common_value_final_rank"]))
            best = rows_sorted[0]
            return {
                "expression": best["expression"],
                "terminals": [x.strip() for x in re.split(r"[;|]", best["terminals"]) if x.strip()],
                "source": str(BEST_COMPOSITE_SUMMARY),
            }

    if BEST_COMPOSITE_TABLE.exists():
        rows = load_csv_rows(BEST_COMPOSITE_TABLE)
        for row in rows:
            if row["comparison_item"] == "best_composite_under_common_value_scoring":
                return {
                    "expression": row["expression"],
                    "terminals": [],
                    "source": str(BEST_COMPOSITE_TABLE),
                }

    return None


def p_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a) + np.asarray(b)


def p_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a) - np.asarray(b)


def p_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a) * np.asarray(b)


def p_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) / (np.asarray(b, dtype=np.float64) + 1e-12)


def p_abs(a: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a))


def p_sqrt(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(np.asarray(a)) + 1e-12)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def compile_expression(expr_str: str, terminal_names: Sequence[str]):
    from deap import gp  # type: ignore

    pset = gp.PrimitiveSet("MAIN", len(terminal_names))
    rename_map = {f"ARG{i}": sanitize_name(name) for i, name in enumerate(terminal_names)}
    pset.renameArguments(**rename_map)
    pset.addPrimitive(p_add, 2, name="add")
    pset.addPrimitive(p_sub, 2, name="sub")
    pset.addPrimitive(p_mul, 2, name="mul")
    pset.addPrimitive(p_div, 2, name="pdiv")
    pset.addPrimitive(p_abs, 1, name="pabs")
    pset.addPrimitive(p_sqrt, 1, name="psqrt")
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    return gp.compile(tree, pset)


def get_best_composite_predicted_indices(dataset_name: str, logger) -> Optional[List[int]]:
    best = load_best_composite_expression()
    if best is None:
        logger.warning("Best composite expression not available")
        return None

    expr = best["expression"]
    terminals = best["terminals"]

    if not terminals:
        logger.warning("Best composite terminals missing, skipping composite downstream strategy")
        return None

    try:
        func = compile_expression(expr, terminals)
    except Exception as exc:
        logger.warning("Failed to compile best composite expression: %s", exc)
        return None

    terminal_curves = {}
    lengths = []
    for terminal in terminals:
        path = get_single_norm_curve_file(dataset_name, terminal)
        if not path.exists():
            logger.warning("Missing terminal curves for composite terminal: %s", terminal)
            return None
        curves = load_curve_file(path)
        terminal_curves[terminal] = curves
        lengths.append(len(curves))

    if len(set(lengths)) != 1:
        logger.warning("Composite terminal curve length mismatch")
        return None

    predicted = []
    n = lengths[0]
    for i in range(n):
        args = [terminal_curves[t][i] for t in terminals]
        curve = np.asarray(func(*args), dtype=np.float64).reshape(-1)
        if curve.size == 1:
            curve = np.repeat(curve.item(), len(args[0]))
        curve = normalize_curve(curve)
        predicted.append(int(np.argmax(curve)))

    return predicted


def get_poor_focus_indices(labels: np.ndarray, num_slices_per_stack: Sequence[int]) -> List[int]:
    out = []
    for label_idx, n in zip(labels, num_slices_per_stack):
        if n <= 1:
            out.append(0)
            continue
        candidates = [0, n - 1]
        worst = max(candidates, key=lambda x: abs(x - int(label_idx)))
        out.append(int(worst))
    return out


# -----------------------------------------------------------------------------
# WBC class-label inference
# -----------------------------------------------------------------------------
def infer_wbc_class_labels_from_structure(dataset_root: Path) -> Tuple[Optional[List[str]], Optional[List[Path]], str]:
    """
    Conservative inference:
    - use folder-based stacks
    - infer class label from the parent folder immediately below dataset root
    only if multiple distinct parent labels exist
    """
    stack_dirs = discover_stack_folders(dataset_root)
    if not stack_dirs:
        return None, None, "no_stack_folders_found"

    labels = []
    for folder in stack_dirs:
        rel = folder.relative_to(dataset_root)
        parts = rel.parts
        if len(parts) < 2:
            return None, None, "stack_folders_not_nested_under_class_folders"
        labels.append(parts[0])

    unique = sorted(set(labels))
    if len(unique) < 2:
        return None, None, "could_not_infer_multiple_classes_from_folder_structure"

    return labels, stack_dirs, "class_labels_inferred_from_parent_folder"


# -----------------------------------------------------------------------------
# Selected image extraction
# -----------------------------------------------------------------------------
def load_stack_array(dataset_name: str) -> np.ndarray:
    return np.load(get_stack_file(dataset_name), allow_pickle=True)


def extract_selected_images(
    dataset_name: str,
    selected_indices: Sequence[int],
) -> List[np.ndarray]:
    stacks = load_stack_array(dataset_name)
    images = []

    for stack, idx in zip(stacks, selected_indices):
        stack_arr = np.asarray(stack)
        idx = int(np.clip(idx, 0, stack_arr.shape[0] - 1))
        images.append(np.asarray(stack_arr[idx], dtype=np.float64))

    return images


# -----------------------------------------------------------------------------
# Downstream modes
# -----------------------------------------------------------------------------
def run_wbc_classification_if_possible(run_mode: str, logger) -> Optional[Dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available; skipping WBC downstream classification")
        return None

    dataset_root = Path(RAW_DATASETS["WBC"]).expanduser().resolve()
    class_labels, stack_dirs, label_note = infer_wbc_class_labels_from_structure(dataset_root)
    if class_labels is None or stack_dirs is None:
        logger.warning("WBC class labels unavailable: %s", label_note)
        return None

    stacks = load_stack_array("WBC")
    labels_focus = load_labels("WBC")

    if len(class_labels) != len(stacks):
        logger.warning(
            "WBC class label count does not match stack count (%d vs %d); skipping classification",
            len(class_labels), len(stacks)
        )
        return None

    num_slices = [int(np.asarray(s).shape[0]) for s in stacks]

    strategies: Dict[str, List[int]] = {
        "poor_autofocus": get_poor_focus_indices(labels_focus, num_slices),
        "brenner_gradient": get_brenner_predicted_indices("WBC"),
    }

    comp_idx = get_best_composite_predicted_indices("WBC", logger)
    if comp_idx is not None:
        strategies["best_composite"] = comp_idx

    unique_classes = sorted(set(class_labels))
    if len(unique_classes) < 2:
        logger.warning("WBC class label inference produced fewer than two classes")
        return None

    class_counts = {name: class_labels.count(name) for name in unique_classes}
    if min(class_counts.values()) < 2:
        logger.warning(
            "WBC class label inference produced underrepresented classes: %s",
            class_counts,
        )
        return None

    class_to_int = {c: i for i, c in enumerate(unique_classes)}
    y = np.asarray([class_to_int[c] for c in class_labels], dtype=int)

    rows = []

    for strategy_name, selected_indices in strategies.items():
        images = extract_selected_images("WBC", selected_indices)
        X = np.stack([extract_simple_features(img) for img in images], axis=0)

        test_size = 0.2 if len(X) >= 20 else 0.5
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        except Exception as exc:
            logger.warning(
                "Skipping WBC downstream strategy %s because classification failed: %s",
                strategy_name,
                exc,
            )
            continue

        rows.append(
            {
                "analysis_mode": "wbc_classification",
                "strategy": strategy_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "num_classes": len(unique_classes),
                "num_samples": len(X),
                "class_label_source": label_note,
            }
        )

    if not rows:
        logger.warning("No valid WBC downstream classification rows were produced")
        return None

    rows_sorted = sorted(rows, key=lambda x: x["balanced_accuracy"], reverse=True)
    return {
        "mode": "wbc_classification",
        "rows": rows_sorted,
        "note": label_note,
    }


def run_focus_quality_proxy(dataset_name: str, logger) -> Dict[str, Any]:
    """
    Fallback analysis if true downstream class labels are unavailable.
    Compares strategy quality using:
    - mean absolute focus index error
    - mean Laplacian variance of selected images
    - mean gradient energy of selected images
    """
    stacks = load_stack_array(dataset_name)
    labels_focus = load_labels(dataset_name)
    num_slices = [int(np.asarray(s).shape[0]) for s in stacks]

    strategies: Dict[str, List[int]] = {
        "poor_autofocus": get_poor_focus_indices(labels_focus, num_slices),
        "brenner_gradient": get_brenner_predicted_indices(dataset_name),
    }

    comp_idx = get_best_composite_predicted_indices(dataset_name, logger)
    if comp_idx is not None:
        strategies["best_composite"] = comp_idx

    rows = []

    for strategy_name, selected_indices in strategies.items():
        images = extract_selected_images(dataset_name, selected_indices)

        focus_errors = [abs(int(p) - int(gt)) for p, gt in zip(selected_indices, labels_focus)]
        lap_vars = [laplacian_variance(img) for img in images]
        grad_energies = [gradient_energy(img) for img in images]

        rows.append(
            {
                "analysis_mode": "focus_quality_proxy",
                "dataset_name": dataset_name,
                "strategy": strategy_name,
                "mean_absolute_focus_index_error": float(np.mean(focus_errors)),
                "mean_laplacian_variance": float(np.mean(lap_vars)),
                "mean_gradient_energy": float(np.mean(grad_energies)),
                "num_samples": len(images),
            }
        )

    # lower error is better, higher sharpness proxies are better
    return {
        "mode": "focus_quality_proxy",
        "rows": rows,
        "note": "Used because true downstream class labels were not reliably available.",
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(rows: List[Dict[str, Any]], output_base: Path, title: str) -> List[str]:
    strategies = [r["strategy"] for r in rows]

    if rows and "balanced_accuracy" in rows[0]:
        values = [float(r["balanced_accuracy"]) for r in rows]
        ylabel = "Balanced accuracy"
    else:
        # proxy: lower focus error is better
        values = [float(r["mean_absolute_focus_index_error"]) for r in rows]
        ylabel = "Mean absolute focus index error"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(strategies))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in [".png", ".pdf"]:
        out = output_base.with_suffix(ext)
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        written.append(str(out))
    plt.close(fig)
    return written


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)

    require_matplotlib()
    ensure_output_dirs()
    validate_all_settings()
    validate_environment()
    validate_pipeline_prerequisites(require_stacks=True, require_labels=True)
    set_global_seed(42)

    log_file = LOGS_DIR / f"optional_downstream_baseline_{run_mode}.log"
    logger = get_logger("optional_downstream_baseline", log_file=log_file)

    logger.info("Starting optional downstream baseline stage")
    logger.info("Preferred dataset: %s", args.dataset)

    result: Optional[Dict[str, Any]] = None

    # Prefer true downstream WBC classification if possible
    if args.dataset == "WBC":
        result = run_wbc_classification_if_possible(run_mode=run_mode, logger=logger)

    # Fallback to proxy analysis
    if result is None:
        logger.info("Falling back to proxy downstream analysis")
        result = run_focus_quality_proxy(dataset_name=args.dataset, logger=logger)

    rows = result["rows"]

    table_path = OPTIONAL_DOWNSTREAM_TABLE
    save_csv_rows(rows, table_path)

    figure_base = OPTIONAL_DOWNSTREAM_FIGURE
    figure_files = plot_results(
        rows=rows,
        output_base=figure_base,
        title="Optional downstream validation",
    )

    json_report = {
        "run_mode": run_mode,
        "preferred_dataset": args.dataset,
        "result_mode": result["mode"],
        "note": result.get("note", ""),
        "rows": rows,
        "table_csv": str(table_path),
        "figure_files": figure_files,
    }
    report_path = OPTIONAL_DOWNSTREAM_REPORT
    save_json(json_report, report_path)
    limitation_lines = [
        "# Optional Downstream Limitation Note",
        "",
        f"Mode: {result['mode']}",
        "",
        result.get("note", "No limitation note provided."),
    ]
    if result["mode"] == "focus_quality_proxy":
        limitation_lines.extend(
            [
                "",
                "Interpretation:",
                "- This is a proxy anchoring analysis, not a definitive downstream biomedical task.",
                "- Publication claims should state that true downstream labels or a downstream learning stack were not available in the current environment.",
                "- The figure and table should be described as scope anchoring evidence rather than a primary efficacy result.",
            ]
        )
    OPTIONAL_DOWNSTREAM_LIMITATION_NOTE.write_text("\n".join(limitation_lines), encoding="utf-8")

    checkpoint_path = OPTIONAL_DOWNSTREAM_CHECKPOINT
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        stage="optional_downstream_baseline",
        status="complete",
        details=json_report,
    )

    logger.info("Optional downstream baseline stage complete")
    logger.info("Table -> %s", table_path)
    logger.info("Figure -> %s", figure_files)
    logger.info("Report -> %s", report_path)
    logger.info("Limitation note -> %s", OPTIONAL_DOWNSTREAM_LIMITATION_NOTE)
    logger.info("Checkpoint -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
