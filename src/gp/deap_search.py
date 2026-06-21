"""Reusable DEAP search helpers for corrected composite GP."""

from __future__ import annotations

import operator
import csv
import pickle
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore

    CUPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    cp = None  # type: ignore
    CUPY_IMPORT_ERROR = exc

from config.paths import DATASET_ORDER, SINGLE_TIMING_SUMMARY_FILE, get_single_norm_curve_file, get_source_label_file, get_surrogate_label_file
from config.settings import (
    AUTOFOCUS_METRICS,
    EPS,
    GENERALIZATION_ALPHA,
    GP_PRIMARY_OBJECTIVE,
    GP_SECONDARY_OBJECTIVE,
    METRIC_WEIGHTS,
)
from src.evaluation.aggregation import align_metric_value, weighted_mean
from src.evaluation.autofocus_metrics import (
    absolute_peak_localization_error,
    add_noise_to_slice,
    curvature_at_peak,
    false_maxima_count,
    full_width_half_maximum,
    noise_level,
    normalize_focus_curve,
    range_around_global_maximum,
    steep_slope_width,
    steep_to_gradual_slope_ratio,
)
from src.utils.seeds import set_global_seed
from src.utils.validation import load_csv_rows

try:
    from deap import algorithms, base, creator, gp, tools  # type: ignore

    DEAP_AVAILABLE = True
    DEAP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    algorithms = base = creator = gp = tools = None  # type: ignore
    DEAP_AVAILABLE = False
    DEAP_IMPORT_ERROR = exc


GP_SEARCH_METRICS: Tuple[str, ...] = tuple(
    metric_name for metric_name in AUTOFOCUS_METRICS if metric_name != "rrmse_under_additive_noise"
)

_WEIGHT_TOTAL = sum(METRIC_WEIGHTS[metric_name] for metric_name in GP_SEARCH_METRICS)
GP_SEARCH_METRIC_WEIGHTS: Dict[str, float] = {
    metric_name: float(METRIC_WEIGHTS[metric_name] / (_WEIGHT_TOTAL + EPS))
    for metric_name in GP_SEARCH_METRICS
}


def require_deap() -> None:
    if not DEAP_AVAILABLE:
        raise ImportError(
            "This stage requires DEAP. Install it in the environment used for the paper pipeline."
        ) from DEAP_IMPORT_ERROR


def cupy_available() -> bool:
    if cp is None:
        return False
    try:
        test = cp.asarray([1.0], dtype=cp.float64)  # type: ignore[union-attr]
        _ = float(cp.asnumpy(test.sum()).item())  # type: ignore[union-attr]
        return True
    except Exception:
        return False


def require_cupy() -> None:
    if not cupy_available():
        raise RuntimeError("CUDA/CuPy backend requested, but CuPy is not available or cannot access the GPU") from CUPY_IMPORT_ERROR


def is_cupy_array(value: Any) -> bool:
    return cp is not None and isinstance(value, cp.ndarray)  # type: ignore[union-attr]


def array_namespace(*values: Any):
    if cp is not None and any(is_cupy_array(value) for value in values):
        return cp
    return np


def to_cpu_array(value: Any) -> np.ndarray:
    if is_cupy_array(value):
        return cp.asnumpy(value)  # type: ignore[union-attr]
    return np.asarray(value)


def load_curve_file(path) -> List[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(item, dtype=np.float64).reshape(-1) for item in arr]


def load_terminal_curves_for_dataset(dataset_name: str, terminal_names: Sequence[str]) -> Dict[str, List[np.ndarray]]:
    curves_by_terminal: Dict[str, List[np.ndarray]] = {}
    lengths: List[int] = []
    for terminal_name in terminal_names:
        path = get_single_norm_curve_file(dataset_name, terminal_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing normalized terminal curve file: {path}")
        curves = load_curve_file(path)
        curves_by_terminal[str(terminal_name)] = curves
        lengths.append(len(curves))

    if lengths and len(set(lengths)) != 1:
        raise ValueError(f"[{dataset_name}] terminal curve count mismatch: {lengths}")
    return curves_by_terminal


def load_all_terminal_curves(terminal_names: Sequence[str]) -> Dict[str, Dict[str, List[np.ndarray]]]:
    return {
        str(dataset_name): load_terminal_curves_for_dataset(str(dataset_name), terminal_names)
        for dataset_name in DATASET_ORDER
    }


def load_composite_labels(dataset_name: str) -> Tuple[np.ndarray, str]:
    source_path = get_source_label_file(dataset_name)
    if source_path.exists():
        labels = np.load(source_path, allow_pickle=False).astype(int).reshape(-1)
        return labels, "source"

    surrogate_path = get_surrogate_label_file(dataset_name)
    if surrogate_path.exists():
        labels = np.load(surrogate_path, allow_pickle=False).astype(int).reshape(-1)
        return labels, "surrogate"

    raise FileNotFoundError(f"No source or surrogate labels found for dataset {dataset_name}")


def load_timing_summary_map(path=SINGLE_TIMING_SUMMARY_FILE) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    rows = load_csv_rows(path)
    timing_map: Dict[str, Dict[str, float]] = {}
    for row in rows:
        dataset_name = str(row["dataset_name"])
        measure_name = str(row["measure_name"])
        try:
            value = float(row.get("native_avg_time_per_slice_sec", "nan"))
        except Exception:
            value = float("nan")
        timing_map.setdefault(dataset_name, {})[measure_name] = value
    return timing_map


def resample_curve(curve: np.ndarray, *, target_length: int = 64) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(target_length, dtype=np.float64)
    if arr.size == target_length:
        return arr.copy()
    if arr.size == 1:
        return np.repeat(arr.item(), target_length).astype(np.float64)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, target_length)
    return np.interp(x_new, x_old, arr).astype(np.float64)


def build_reference_bounds(
    dataset_metric_reference: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_subset: Sequence[str],
    *,
    metric_names: Sequence[str] = GP_SEARCH_METRICS,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for dataset_name in dataset_subset:
        by_measure = dataset_metric_reference[str(dataset_name)]
        bounds[str(dataset_name)] = {}
        for metric_name in metric_names:
            aligned_values = [
                align_metric_value(metric_name, float(metrics[metric_name]))
                for metrics in by_measure.values()
            ]
            finite = [value for value in aligned_values if np.isfinite(value)]
            if finite:
                bounds[str(dataset_name)][str(metric_name)] = (float(np.min(finite)), float(np.max(finite)))
            else:
                bounds[str(dataset_name)][str(metric_name)] = (0.0, 1.0)
    return bounds


def normalize_against_reference(
    dataset_name: str,
    metric_name: str,
    raw_metric_value: float,
    reference_bounds: Mapping[str, Mapping[str, Tuple[float, float]]],
) -> float:
    aligned_value = align_metric_value(metric_name, raw_metric_value)
    ref_min, ref_max = reference_bounds[str(dataset_name)][str(metric_name)]
    denom = float(ref_max - ref_min)
    if denom <= EPS:
        return 0.0
    return float((aligned_value - ref_min) / (denom + EPS))


def p_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xp = array_namespace(a, b)
    return xp.asarray(a) + xp.asarray(b)


def p_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xp = array_namespace(a, b)
    return xp.asarray(a) - xp.asarray(b)


def p_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xp = array_namespace(a, b)
    return xp.asarray(a) * xp.asarray(b)


def p_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xp = array_namespace(a, b)
    return xp.asarray(a, dtype=xp.float64) / (xp.asarray(b, dtype=xp.float64) + EPS)


def p_abs(a: np.ndarray) -> np.ndarray:
    xp = array_namespace(a)
    return xp.abs(xp.asarray(a))


def p_sqrt(a: np.ndarray) -> np.ndarray:
    xp = array_namespace(a)
    return xp.sqrt(xp.abs(xp.asarray(a)) + EPS)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")


def build_pset(terminal_names: Sequence[str]):
    require_deap()
    pset = gp.PrimitiveSet("MAIN", len(terminal_names))
    rename_map = {f"ARG{i}": sanitize_name(name) for i, name in enumerate(terminal_names)}
    pset.renameArguments(**rename_map)
    pset.addPrimitive(p_add, 2, name="add")
    pset.addPrimitive(p_sub, 2, name="sub")
    pset.addPrimitive(p_mul, 2, name="mul")
    pset.addPrimitive(p_div, 2, name="pdiv")
    pset.addPrimitive(p_abs, 1, name="pabs")
    pset.addPrimitive(p_sqrt, 1, name="psqrt")
    return pset


def compile_expression(expr_str: str, terminal_names: Sequence[str]):
    require_deap()
    pset = build_pset(terminal_names)
    tree = gp.PrimitiveTree.from_string(str(expr_str), pset)
    return gp.compile(tree, pset)


def ensure_deap_creators() -> None:
    require_deap()
    if not hasattr(creator, "FitnessCompositeMin"):
        creator.create("FitnessCompositeMin", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "IndividualComposite"):
        creator.create("IndividualComposite", gp.PrimitiveTree, fitness=creator.FitnessCompositeMin)


def build_toolbox(pset, gp_settings: Mapping[str, Any]):
    ensure_deap_creators()
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.IndividualComposite, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    max_depth = int(gp_settings["max_tree_depth"])
    max_nodes = int(gp_settings.get("max_nodes", max(40, max_depth * 5)))
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_nodes))
    return toolbox


def composite_execution_time_per_slice(
    *,
    dataset_name: str,
    terminal_names: Sequence[str],
    terminal_curves: Mapping[str, Sequence[np.ndarray]],
    func,
    timing_map: Mapping[str, Mapping[str, float]] | None,
) -> float:
    terminal_runtime = 0.0
    if timing_map:
        for terminal_name in terminal_names:
            value = float(timing_map.get(str(dataset_name), {}).get(str(terminal_name), float("nan")))
            if np.isfinite(value):
                terminal_runtime += value

    total_combo = 0.0
    total_slices = 0
    num_stacks = len(next(iter(terminal_curves.values())))
    for stack_idx in range(num_stacks):
        args = [np.asarray(terminal_curves[terminal_name][stack_idx], dtype=np.float64) for terminal_name in terminal_names]
        t0 = time.perf_counter()
        curve = np.asarray(func(*args), dtype=np.float64).reshape(-1)
        if curve.size == 1:
            curve = np.repeat(curve.item(), len(args[0]))
        t1 = time.perf_counter()
        total_combo += (t1 - t0)
        total_slices += len(curve)

    combo_time = float(total_combo / (total_slices + EPS))
    return float(terminal_runtime + combo_time)


def composite_rrmse_under_additive_noise(
    *,
    clean_curve_norm: np.ndarray,
    stack: np.ndarray,
    terminal_names: Sequence[str],
    measure_registry: Mapping[str, Mapping[str, Any]],
    func,
    rng: np.random.Generator,
) -> float:
    noisy_terminal_curves: List[np.ndarray] = []
    for terminal_name in terminal_names:
        measure_func = measure_registry[str(terminal_name)]["func"]
        scores = [
            float(measure_func(add_noise_to_slice(slice_2d, rng)))
            for slice_2d in np.asarray(stack)
        ]
        noisy_terminal_curves.append(normalize_focus_curve(np.asarray(scores, dtype=np.float64)))

    noisy_curve = np.asarray(func(*noisy_terminal_curves), dtype=np.float64).reshape(-1)
    if noisy_curve.size == 1:
        noisy_curve = np.repeat(noisy_curve.item(), len(clean_curve_norm))
    noisy_curve_norm = normalize_focus_curve(noisy_curve)

    diff = np.asarray(clean_curve_norm, dtype=np.float64) - noisy_curve_norm
    numerator = np.sqrt(np.mean(diff ** 2))
    denominator = np.sqrt(np.mean(np.asarray(clean_curve_norm, dtype=np.float64) ** 2)) + EPS
    return float(numerator / denominator)


def predict_peak_indices_matrix(curves: np.ndarray) -> np.ndarray:
    max_vals = np.max(curves, axis=1)
    ties = np.isclose(curves, max_vals[:, None])
    tie_counts = np.sum(ties, axis=1)
    target_ranks = tie_counts // 2 + 1
    cumulative = np.cumsum(ties, axis=1)
    return np.argmax(cumulative >= target_ranks[:, None], axis=1).astype(int)


def normalize_focus_curves_matrix(curves: np.ndarray) -> np.ndarray:
    cmin = np.min(curves, axis=1)
    cmax = np.max(curves, axis=1)
    denom = cmax - cmin
    normalized = np.zeros_like(curves, dtype=np.float64)
    valid = denom > EPS
    if np.any(valid):
        normalized[valid] = (curves[valid] - cmin[valid, None]) / (denom[valid, None] + EPS)
    return normalized


def summarize_focus_metrics_matrix(curves: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    curve_arr = np.asarray(curves, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=int).reshape(-1)
    if curve_arr.ndim != 2:
        raise ValueError(f"Expected focus-curve matrix with shape (n, slices), got {curve_arr.shape}")
    num_rows, curve_len = curve_arr.shape
    rows = np.arange(num_rows)
    peak_idx = predict_peak_indices_matrix(curve_arr)
    peak_val = np.max(curve_arr, axis=1)
    positions = np.arange(curve_len)[None, :]

    metrics: Dict[str, np.ndarray] = {}
    metrics["absolute_peak_localization_error"] = np.abs(peak_idx - label_arr).astype(np.float64)

    above_half = curve_arr >= (0.5 * peak_val[:, None])
    first_half = np.argmax(above_half, axis=1)
    last_half = curve_len - 1 - np.argmax(above_half[:, ::-1], axis=1)
    fwhm_values = (last_half - first_half + 1).astype(np.float64)
    fwhm_values[peak_val <= EPS] = float(curve_len)
    metrics["fwhm"] = fwhm_values

    curvature_values = np.zeros(num_rows, dtype=np.float64)
    interior = (peak_idx > 0) & (peak_idx < curve_len - 1)
    if np.any(interior):
        interior_rows = rows[interior]
        interior_peaks = peak_idx[interior]
        second_diff = (
            curve_arr[interior_rows, interior_peaks - 1]
            - 2.0 * curve_arr[interior_rows, interior_peaks]
            + curve_arr[interior_rows, interior_peaks + 1]
        )
        curvature_values[interior] = np.maximum(0.0, -second_diff)
    metrics["curvature_at_peak"] = curvature_values

    minima = np.zeros_like(curve_arr, dtype=bool)
    if curve_len >= 3:
        minima[:, 1:-1] = (
            (curve_arr[:, 1:-1] <= curve_arr[:, :-2])
            & (curve_arr[:, 1:-1] <= curve_arr[:, 2:])
        )
    left_candidates = np.where(minima & (positions < peak_idx[:, None]), positions, -1)
    right_candidates = np.where(minima & (positions > peak_idx[:, None]), positions, curve_len)
    left = np.max(left_candidates, axis=1)
    right = np.min(right_candidates, axis=1)
    left = np.where(left < 0, 0, left)
    right = np.where(right >= curve_len, curve_len - 1, right)
    metrics["steep_slope_width"] = (right - left).astype(np.float64)

    local_sum = np.zeros(num_rows, dtype=np.float64)
    local_count = np.zeros(num_rows, dtype=np.float64)
    has_left = peak_idx > 0
    if np.any(has_left):
        left_rows = rows[has_left]
        left_peaks = peak_idx[has_left]
        local_sum[has_left] += np.abs(curve_arr[left_rows, left_peaks] - curve_arr[left_rows, left_peaks - 1])
        local_count[has_left] += 1.0
    has_right = peak_idx < curve_len - 1
    if np.any(has_right):
        right_rows = rows[has_right]
        right_peaks = peak_idx[has_right]
        local_sum[has_right] += np.abs(curve_arr[right_rows, right_peaks] - curve_arr[right_rows, right_peaks + 1])
        local_count[has_right] += 1.0
    local_mean = np.divide(local_sum, local_count, out=np.zeros_like(local_sum), where=local_count > 0)

    diffs = np.abs(np.diff(curve_arr, axis=1))
    diff_mask = np.ones_like(diffs, dtype=bool)
    diff_positions = np.arange(max(0, curve_len - 1))[None, :]
    if diffs.size:
        diff_mask &= diff_positions != (peak_idx - 1)[:, None]
        diff_mask &= diff_positions != peak_idx[:, None]
        background_count = np.sum(diff_mask, axis=1)
        background_sum = np.sum(np.where(diff_mask, diffs, 0.0), axis=1)
        background_mean = np.divide(
            background_sum,
            background_count,
            out=np.zeros_like(background_sum),
            where=background_count > 0,
        )
    else:
        background_mean = np.zeros(num_rows, dtype=np.float64)
    metrics["steep_to_gradual_slope_ratio"] = local_mean / (background_mean + EPS)

    local_maxima = np.zeros_like(curve_arr, dtype=bool)
    if curve_len >= 3:
        local_maxima[:, 1:-1] = (
            (curve_arr[:, 1:-1] > curve_arr[:, :-2])
            & (curve_arr[:, 1:-1] > curve_arr[:, 2:])
        )
    local_maxima[rows, peak_idx] = False
    metrics["false_maxima_count"] = np.sum(local_maxima, axis=1).astype(np.float64)

    if curve_len < 3:
        metrics["noise_level"] = np.zeros(num_rows, dtype=np.float64)
    else:
        second = np.diff(curve_arr, n=2, axis=1)
        metrics["noise_level"] = np.mean(second ** 2, axis=1)

    above_near_peak = curve_arr >= (0.95 * peak_val[:, None])
    false_before = (~above_near_peak) & (positions < peak_idx[:, None])
    false_after = (~above_near_peak) & (positions > peak_idx[:, None])
    last_false_before = np.max(np.where(false_before, positions, -1), axis=1)
    first_false_after = np.min(np.where(false_after, positions, curve_len), axis=1)
    range_values = (first_false_after - last_false_before - 1).astype(np.float64)
    range_values[peak_val <= EPS] = float(curve_len)
    metrics["range_around_global_maximum"] = range_values

    return metrics


def evaluate_expression_raw_metrics_batched(
    *,
    func,
    terminal_names: Sequence[str],
    terminal_curves: Mapping[str, Sequence[np.ndarray]],
    labels: np.ndarray,
    dataset_name: str,
    timing_map: Mapping[str, Mapping[str, float]] | None = None,
    metric_names: Sequence[str] = AUTOFOCUS_METRICS,
    array_backend: str = "cpu",
    collect_mean_curve: bool = True,
) -> Dict[str, Any]:
    if str(array_backend) == "cuda":
        require_cupy()

    requested_metrics = tuple(str(metric_name) for metric_name in metric_names)
    label_arr = np.asarray(labels, dtype=int).reshape(-1)
    if not terminal_names:
        raise ValueError("No GP terminal names were provided")
    num_stacks = len(label_arr)

    grouped_indices: Dict[int, List[int]] = {}
    first_terminal = str(terminal_names[0])
    for stack_idx in range(num_stacks):
        curve_len = int(len(to_cpu_array(terminal_curves[first_terminal][stack_idx])))
        grouped_indices.setdefault(curve_len, []).append(stack_idx)

    metric_sums: Dict[str, float] = {metric_name: 0.0 for metric_name in requested_metrics}
    metric_counts: Dict[str, int] = {metric_name: 0 for metric_name in requested_metrics}
    combo_total_time = 0.0
    combo_total_slices = 0
    mean_curve_sum = np.zeros(64, dtype=np.float64)
    mean_curve_count = 0

    for curve_len, group_indices in grouped_indices.items():
        group_size = len(group_indices)
        args_cpu = []
        for terminal_name in terminal_names:
            matrix = np.stack(
                [
                    np.asarray(to_cpu_array(terminal_curves[str(terminal_name)][idx]), dtype=np.float64).reshape(-1)
                    for idx in group_indices
                ],
                axis=0,
            )
            if matrix.shape != (group_size, curve_len):
                raise ValueError(f"[{dataset_name}] inconsistent terminal curve lengths for {terminal_name}")
            args_cpu.append(matrix)

        t0 = time.perf_counter()
        if str(array_backend) == "cuda":
            args = [cp.asarray(arg, dtype=cp.float64) for arg in args_cpu]  # type: ignore[union-attr]
            curve_matrix = np.asarray(to_cpu_array(cp.asarray(func(*args))), dtype=np.float64)  # type: ignore[union-attr]
        else:
            curve_matrix = np.asarray(func(*args_cpu), dtype=np.float64)
        t1 = time.perf_counter()

        if curve_matrix.ndim == 0:
            curve_matrix = np.full((group_size, curve_len), float(curve_matrix), dtype=np.float64)
        elif curve_matrix.ndim == 1:
            if group_size == 1 and curve_matrix.size == curve_len:
                curve_matrix = curve_matrix.reshape(1, curve_len)
            elif curve_matrix.size == 1:
                curve_matrix = np.full((group_size, curve_len), float(curve_matrix.item()), dtype=np.float64)
            else:
                raise ValueError(f"[{dataset_name}] expression returned unexpected curve shape {curve_matrix.shape}")
        elif curve_matrix.shape != (group_size, curve_len):
            try:
                curve_matrix = np.broadcast_to(curve_matrix, (group_size, curve_len)).astype(np.float64)
            except Exception as exc:
                raise ValueError(f"[{dataset_name}] expression returned unexpected curve shape {curve_matrix.shape}") from exc

        if not np.all(np.isfinite(curve_matrix)):
            raise ValueError(f"[{dataset_name}] composite expression produced non-finite values")

        curve_norm = normalize_focus_curves_matrix(curve_matrix)
        metrics = summarize_focus_metrics_matrix(curve_norm, label_arr[group_indices])
        for metric_name in requested_metrics:
            if metric_name == "execution_time_per_slice":
                continue
            if metric_name in metrics:
                values = np.asarray(metrics[metric_name], dtype=np.float64)
                metric_sums[metric_name] += float(np.sum(values))
                metric_counts[metric_name] += int(values.size)

        if collect_mean_curve:
            for row in curve_norm:
                mean_curve_sum += resample_curve(row)
                mean_curve_count += 1

        combo_total_time += float(t1 - t0)
        combo_total_slices += int(group_size * curve_len)

    raw_metrics: Dict[str, float] = {}
    for metric_name in requested_metrics:
        if metric_name == "execution_time_per_slice":
            terminal_runtime = 0.0
            if timing_map:
                for terminal_name in terminal_names:
                    value = float(timing_map.get(str(dataset_name), {}).get(str(terminal_name), float("nan")))
                    if np.isfinite(value):
                        terminal_runtime += value
            combo_time = combo_total_time / float(combo_total_slices + EPS)
            raw_metrics[metric_name] = float(terminal_runtime + combo_time)
            continue
        count = metric_counts.get(metric_name, 0)
        raw_metrics[metric_name] = float(metric_sums[metric_name] / count) if count else float("nan")

    mean_curve = (mean_curve_sum / float(mean_curve_count)).tolist() if mean_curve_count else []
    return {
        "raw_metrics": raw_metrics,
        "mean_curve": mean_curve,
        "combo_only_time_per_slice": float(combo_total_time / float(combo_total_slices + EPS)),
    }


def evaluate_expression_raw_metrics(
    *,
    func,
    terminal_names: Sequence[str],
    terminal_curves: Mapping[str, Sequence[np.ndarray]],
    labels: np.ndarray,
    dataset_name: str,
    timing_map: Mapping[str, Mapping[str, float]] | None = None,
    measure_registry: Mapping[str, Mapping[str, Any]] | None = None,
    stacks: np.ndarray | None = None,
    skip_rrmse: bool = True,
    run_mode: str = "smoke",
    metric_names: Sequence[str] = AUTOFOCUS_METRICS,
    smoke_rrmse_cap: int = 5,
    full_rrmse_cap: int = 100,
    array_backend: str = "cpu",
    collect_mean_curve: bool = True,
) -> Dict[str, Any]:
    if str(array_backend) == "cuda":
        require_cupy()

    requested_metric_names = tuple(str(metric_name) for metric_name in metric_names)
    needs_rrmse = "rrmse_under_additive_noise" in set(requested_metric_names)
    non_rrmse_metric_names = tuple(
        metric_name for metric_name in requested_metric_names
        if metric_name != "rrmse_under_additive_noise"
    )
    if not needs_rrmse or skip_rrmse or stacks is None or measure_registry is None:
        raw_eval = evaluate_expression_raw_metrics_batched(
            func=func,
            terminal_names=terminal_names,
            terminal_curves=terminal_curves,
            labels=labels,
            dataset_name=dataset_name,
            timing_map=timing_map,
            metric_names=non_rrmse_metric_names,
            array_backend=array_backend,
            collect_mean_curve=collect_mean_curve,
        )
        if needs_rrmse:
            raw_eval["raw_metrics"]["rrmse_under_additive_noise"] = float("nan")
        return raw_eval

    raw_eval = evaluate_expression_raw_metrics_batched(
        func=func,
        terminal_names=terminal_names,
        terminal_curves=terminal_curves,
        labels=labels,
        dataset_name=dataset_name,
        timing_map=timing_map,
        metric_names=non_rrmse_metric_names,
        array_backend=array_backend,
        collect_mean_curve=collect_mean_curve,
    )

    rrmse_cap = int(smoke_rrmse_cap if run_mode == "smoke" else full_rrmse_cap)
    rng = np.random.default_rng(123)
    rrmse_values: List[float] = []
    for stack_idx in range(min(rrmse_cap, len(np.asarray(labels).reshape(-1)))):
        args_cpu = [
            np.asarray(to_cpu_array(terminal_curves[terminal_name][stack_idx]), dtype=np.float64)
            for terminal_name in terminal_names
        ]
        if str(array_backend) == "cuda":
            args = [cp.asarray(arg, dtype=cp.float64) for arg in args_cpu]  # type: ignore[union-attr]
            curve_raw = np.asarray(to_cpu_array(cp.asarray(func(*args)).reshape(-1)), dtype=np.float64)  # type: ignore[union-attr]
        else:
            curve_raw = np.asarray(func(*args_cpu), dtype=np.float64).reshape(-1)
        if curve_raw.size == 1:
            curve_raw = np.repeat(curve_raw.item(), len(args_cpu[0]))
        if not np.all(np.isfinite(curve_raw)):
            raise ValueError(f"[{dataset_name}] composite expression produced non-finite values")
        curve_norm = normalize_focus_curve(curve_raw)
        rrmse_values.append(
            composite_rrmse_under_additive_noise(
                clean_curve_norm=curve_norm,
                stack=np.asarray(stacks[stack_idx]),
                terminal_names=terminal_names,
                measure_registry=measure_registry,
                func=func,
                rng=rng,
            )
        )
    raw_eval["raw_metrics"]["rrmse_under_additive_noise"] = (
        float(np.nanmean(rrmse_values)) if rrmse_values else float("nan")
    )
    return raw_eval

    metric_values: Dict[str, List[float]] = {str(metric_name): [] for metric_name in metric_names}
    normalized_curve_fingerprints: List[np.ndarray] = []
    combo_only_timings: List[float] = []

    rrmse_cap = int(smoke_rrmse_cap if run_mode == "smoke" else full_rrmse_cap)
    rng = np.random.default_rng(123)

    for stack_idx, label_idx in enumerate(np.asarray(labels, dtype=int).reshape(-1)):
        args_cpu = [np.asarray(to_cpu_array(terminal_curves[terminal_name][stack_idx]), dtype=np.float64) for terminal_name in terminal_names]
        t0 = time.perf_counter()
        if str(array_backend) == "cuda":
            args = [cp.asarray(arg, dtype=cp.float64) for arg in args_cpu]  # type: ignore[union-attr]
            curve_raw = np.asarray(to_cpu_array(cp.asarray(func(*args)).reshape(-1)), dtype=np.float64)  # type: ignore[union-attr]
        else:
            curve_raw = np.asarray(func(*args_cpu), dtype=np.float64).reshape(-1)
        t1 = time.perf_counter()
        if curve_raw.size == 1:
            curve_raw = np.repeat(curve_raw.item(), len(args_cpu[0]))
        if not np.all(np.isfinite(curve_raw)):
            raise ValueError(f"[{dataset_name}] composite expression produced non-finite values")

        curve_norm = normalize_focus_curve(curve_raw)
        if collect_mean_curve:
            normalized_curve_fingerprints.append(resample_curve(curve_norm))
        combo_only_timings.append(float((t1 - t0) / max(1, len(curve_norm))))

        if "absolute_peak_localization_error" in metric_values:
            metric_values["absolute_peak_localization_error"].append(
                absolute_peak_localization_error(curve_norm, int(label_idx))
            )
        if "fwhm" in metric_values:
            metric_values["fwhm"].append(full_width_half_maximum(curve_norm))
        if "curvature_at_peak" in metric_values:
            metric_values["curvature_at_peak"].append(curvature_at_peak(curve_norm))
        if "steep_slope_width" in metric_values:
            metric_values["steep_slope_width"].append(steep_slope_width(curve_norm))
        if "steep_to_gradual_slope_ratio" in metric_values:
            metric_values["steep_to_gradual_slope_ratio"].append(steep_to_gradual_slope_ratio(curve_norm))
        if "false_maxima_count" in metric_values:
            metric_values["false_maxima_count"].append(false_maxima_count(curve_norm))
        if "noise_level" in metric_values:
            metric_values["noise_level"].append(noise_level(curve_norm))
        if "range_around_global_maximum" in metric_values:
            metric_values["range_around_global_maximum"].append(range_around_global_maximum(curve_norm))
        if (
            "rrmse_under_additive_noise" in metric_values
            and not skip_rrmse
            and stacks is not None
            and measure_registry is not None
            and stack_idx < rrmse_cap
        ):
            metric_values["rrmse_under_additive_noise"].append(
                composite_rrmse_under_additive_noise(
                    clean_curve_norm=curve_norm,
                    stack=np.asarray(stacks[stack_idx]),
                    terminal_names=terminal_names,
                    measure_registry=measure_registry,
                    func=func,
                    rng=rng,
                )
            )

    raw_metrics: Dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name == "execution_time_per_slice":
            terminal_runtime = 0.0
            if timing_map:
                for terminal_name in terminal_names:
                    value = float(timing_map.get(str(dataset_name), {}).get(str(terminal_name), float("nan")))
                    if np.isfinite(value):
                        terminal_runtime += value
            raw_metrics[metric_name] = float(terminal_runtime + np.nanmean(combo_only_timings))
            continue
        values = metric_values.get(str(metric_name), [])
        raw_metrics[str(metric_name)] = float(np.nanmean(values)) if values else float("nan")

    mean_curve = (
        np.mean(np.stack(normalized_curve_fingerprints, axis=0), axis=0).tolist()
        if normalized_curve_fingerprints else []
    )
    return {
        "raw_metrics": raw_metrics,
        "mean_curve": mean_curve,
        "combo_only_time_per_slice": float(np.nanmean(combo_only_timings)) if combo_only_timings else float("nan"),
    }


def evaluate_expression_on_dataset(
    *,
    func,
    terminal_names: Sequence[str],
    terminal_curves: Mapping[str, Sequence[np.ndarray]],
    labels: np.ndarray,
    dataset_name: str,
    reference_bounds: Mapping[str, Mapping[str, Tuple[float, float]]],
    timing_map: Mapping[str, Mapping[str, float]] | None = None,
    array_backend: str = "cpu",
    collect_diagnostics: bool = True,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    raw_eval = evaluate_expression_raw_metrics(
        func=func,
        terminal_names=terminal_names,
        terminal_curves=terminal_curves,
        labels=labels,
        dataset_name=dataset_name,
        timing_map=timing_map,
        skip_rrmse=True,
        metric_names=GP_SEARCH_METRICS,
        array_backend=array_backend,
        collect_mean_curve=collect_diagnostics,
    )

    dataset_metric_scores: Dict[str, float] = {}
    weighted_scores: List[float] = []
    weights: List[float] = []
    for metric_name in GP_SEARCH_METRICS:
        score = normalize_against_reference(
            dataset_name=str(dataset_name),
            metric_name=str(metric_name),
            raw_metric_value=float(raw_eval["raw_metrics"][metric_name]),
            reference_bounds=reference_bounds,
        )
        dataset_metric_scores[str(metric_name)] = score
        if np.isfinite(score):
            weighted_scores.append(score)
            weights.append(GP_SEARCH_METRIC_WEIGHTS[str(metric_name)])

    dataset_score = weighted_mean(weighted_scores, weights) if weighted_scores else float("inf")
    return (
        float(dataset_score),
        dict(raw_eval["raw_metrics"]),
        {
            "dataset_metric_scores": dataset_metric_scores,
            "stack_fingerprint_mean_curve": raw_eval["mean_curve"],
        },
    )


def make_fitness_function(
    *,
    toolbox,
    terminal_names: Sequence[str],
    train_datasets: Sequence[str],
    curves_by_dataset: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    labels_by_dataset: Mapping[str, np.ndarray],
    reference_bounds: Mapping[str, Mapping[str, Tuple[float, float]]],
    timing_map: Mapping[str, Mapping[str, float]] | None,
    max_nodes: Optional[int] = None,
    max_eval_seconds: Optional[float] = None,
    array_backend: str = "cpu",
):
    def evaluate_individual(individual):
        complexity = float(len(individual))
        if max_nodes is not None and int(complexity) > int(max_nodes):
            return (1e9, complexity)

        try:
            func = toolbox.compile(expr=individual)
        except Exception:
            return (1e9, complexity)

        dataset_scores: List[float] = []
        t0 = time.perf_counter()
        for dataset_name in train_datasets:
            try:
                dataset_score, _raw_metrics, _diag = evaluate_expression_on_dataset(
                    func=func,
                    terminal_names=terminal_names,
                    terminal_curves=curves_by_dataset[str(dataset_name)],
                    labels=labels_by_dataset[str(dataset_name)],
                    dataset_name=str(dataset_name),
                    reference_bounds=reference_bounds,
                    timing_map=timing_map,
                    array_backend=array_backend,
                    collect_diagnostics=False,
                )
            except Exception:
                return (1e9, complexity)
            if not np.isfinite(dataset_score):
                return (1e9, complexity)
            dataset_scores.append(float(dataset_score))
            if max_eval_seconds is not None and (time.perf_counter() - t0) > float(max_eval_seconds):
                return (1e9, complexity)

        mean_score = float(np.mean(dataset_scores)) if dataset_scores else 1e9
        std_score = float(np.std(dataset_scores, ddof=0)) if dataset_scores else 1e9
        generalization_score = float(
            GENERALIZATION_ALPHA * mean_score + (1.0 - GENERALIZATION_ALPHA) * std_score
        )
        return (generalization_score, complexity)

    return evaluate_individual


def append_progress_row(progress_path: Path, row: Mapping[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    with progress_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def write_progress_rows(progress_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def save_gp_generation_checkpoint(
    checkpoint_path: Path,
    *,
    held_out_dataset: str,
    seed: int,
    terminal_names: Sequence[str],
    gp_settings: Mapping[str, Any],
    completed_generations: int,
    population: Sequence[Any],
    hall_of_fame: Any,
    logbook_rows: Sequence[Mapping[str, Any]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "held_out_dataset": str(held_out_dataset),
        "seed": int(seed),
        "terminal_names": list(terminal_names),
        "gp_settings": dict(gp_settings),
        "completed_generations": int(completed_generations),
        "population": list(population),
        "hall_of_fame": hall_of_fame,
        "logbook_rows": [dict(row) for row in logbook_rows],
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "saved_at_unix": time.time(),
    }
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(checkpoint_path)


def load_gp_generation_checkpoint(
    checkpoint_path: Path,
    *,
    held_out_dataset: str,
    seed: int,
    terminal_names: Sequence[str],
    gp_settings: Mapping[str, Any],
    logger=None,
) -> Optional[Dict[str, Any]]:
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return None

    try:
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "[heldout=%s seed=%d] ignoring unreadable generation checkpoint %s: %s",
                held_out_dataset,
                seed,
                checkpoint_path,
                exc,
            )
        return None

    if not isinstance(payload, dict):
        return None
    if str(payload.get("held_out_dataset")) != str(held_out_dataset):
        return None
    if int(payload.get("seed", -1)) != int(seed):
        return None
    if list(payload.get("terminal_names", [])) != list(terminal_names):
        if logger is not None:
            logger.info(
                "[heldout=%s seed=%d] generation checkpoint is incompatible: terminal set changed",
                held_out_dataset,
                seed,
            )
        return None
    if dict(payload.get("gp_settings", {})) != dict(gp_settings):
        if logger is not None:
            logger.info(
                "[heldout=%s seed=%d] generation checkpoint is incompatible: GP settings changed",
                held_out_dataset,
                seed,
            )
        return None

    completed_generations = int(payload.get("completed_generations", -1))
    expected_generations = int(gp_settings["num_generations"])
    if completed_generations < 0 or completed_generations > expected_generations:
        return None

    try:
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "[heldout=%s seed=%d] generation checkpoint has invalid RNG state: %s",
                held_out_dataset,
                seed,
                exc,
            )
        return None

    return payload


def run_gp_seed(
    *,
    held_out_dataset: str,
    seed: int,
    terminal_names: Sequence[str],
    curves_by_dataset: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    labels_by_dataset: Mapping[str, np.ndarray],
    reference_bounds: Mapping[str, Mapping[str, Tuple[float, float]]],
    gp_settings: Mapping[str, Any],
    timing_map: Mapping[str, Mapping[str, float]] | None = None,
    logger=None,
    progress_path: Optional[Path] = None,
    progress_every: int = 1,
    checkpoint_path: Optional[Path] = None,
    resume_checkpoint: bool = True,
    array_backend: str = "cpu",
    train_datasets_override: Optional[Sequence[str]] = None,
    heldout_evaluation_dataset: Optional[str] = None,
) -> Dict[str, Any]:
    require_deap()
    set_global_seed(int(seed))
    array_backend = str(array_backend).strip().lower()
    if array_backend not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported GP array backend: {array_backend}")
    if array_backend == "cuda":
        require_cupy()

    train_datasets = (
        [str(dataset_name) for dataset_name in train_datasets_override]
        if train_datasets_override is not None
        else [dataset_name for dataset_name in DATASET_ORDER if dataset_name != held_out_dataset]
    )
    evaluation_dataset = str(heldout_evaluation_dataset) if heldout_evaluation_dataset is not None else str(held_out_dataset)
    if str(held_out_dataset) == "FINAL_ALL" and heldout_evaluation_dataset is None:
        evaluation_dataset = ""
    pset = build_pset(terminal_names)
    toolbox = build_toolbox(pset, gp_settings)
    max_nodes = int(gp_settings.get("max_nodes", max(40, int(gp_settings["max_tree_depth"]) * 5)))
    max_eval_seconds_value = gp_settings.get("max_eval_seconds", None)
    max_eval_seconds = None if max_eval_seconds_value is None else float(max_eval_seconds_value)
    toolbox.register(
        "evaluate",
        make_fitness_function(
            toolbox=toolbox,
            terminal_names=terminal_names,
            train_datasets=train_datasets,
            curves_by_dataset=curves_by_dataset,
            labels_by_dataset=labels_by_dataset,
            reference_bounds=reference_bounds,
            timing_map=timing_map,
            max_nodes=max_nodes,
            max_eval_seconds=max_eval_seconds,
            array_backend=array_backend,
        ),
    )

    pop_size = int(gp_settings["population_size"])
    num_generations = int(gp_settings["num_generations"])
    crossover_probability = float(gp_settings["crossover_probability"])
    mutation_probability = float(gp_settings["mutation_probability"])
    progress_every = max(1, int(progress_every))

    if logger is not None:
        logger.info(
            "[heldout=%s seed=%d] starting GP seed pop=%d gens=%d max_nodes=%d max_eval_seconds=%s backend=%s",
            held_out_dataset,
            seed,
            pop_size,
            num_generations,
            max_nodes,
            "none" if max_eval_seconds is None else f"{max_eval_seconds:.3f}",
            array_backend,
        )

    logbook_rows: List[Dict[str, Any]] = []
    start_generation = 0
    checkpoint_payload = None
    if checkpoint_path is not None and resume_checkpoint:
        checkpoint_payload = load_gp_generation_checkpoint(
            checkpoint_path,
            held_out_dataset=held_out_dataset,
            seed=seed,
            terminal_names=terminal_names,
            gp_settings=gp_settings,
            logger=logger,
        )

    if checkpoint_payload is not None:
        population = list(checkpoint_payload["population"])
        hall_of_fame = checkpoint_payload["hall_of_fame"]
        logbook_rows = [dict(row) for row in checkpoint_payload.get("logbook_rows", [])]
        start_generation = int(checkpoint_payload["completed_generations"])
        if progress_path is not None:
            write_progress_rows(progress_path, logbook_rows)
        if logger is not None:
            logger.info(
                "[heldout=%s seed=%d] resumed GP generation checkpoint at generation %d/%d",
                held_out_dataset,
                seed,
                start_generation,
                num_generations,
            )
    else:
        if progress_path is not None and progress_path.exists():
            progress_path.unlink()
        population = toolbox.population(n=pop_size)
        hall_of_fame = tools.HallOfFame(10)

        invalid = [individual for individual in population if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness
        population = toolbox.select(population, len(population))
        hall_of_fame.update(population)
        initial_best = min(population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1]))
        if checkpoint_path is not None:
            save_gp_generation_checkpoint(
                checkpoint_path,
                held_out_dataset=held_out_dataset,
                seed=seed,
                terminal_names=terminal_names,
                gp_settings=gp_settings,
                completed_generations=0,
                population=population,
                hall_of_fame=hall_of_fame,
                logbook_rows=logbook_rows,
            )
        if logger is not None:
            logger.info(
                "[heldout=%s seed=%d] initial population evaluated best_score=%.6f nodes=%d",
                held_out_dataset,
                seed,
                float(initial_best.fitness.values[0]),
                int(len(initial_best)),
            )
            if float(initial_best.fitness.values[0]) >= 1e9:
                logger.warning(
                    "[heldout=%s seed=%d] all initial candidates received the invalid penalty; "
                    "increase --max-eval-seconds or inspect terminal/label compatibility",
                    held_out_dataset,
                    seed,
                )

    for generation in range(start_generation, num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness
        population = toolbox.select(population + offspring, pop_size)
        hall_of_fame.update(population)

        best = min(population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1]))
        row = {
            "generation": generation,
            "best_generalization_score": float(best.fitness.values[0]),
            "best_complexity": float(best.fitness.values[1]),
            "best_expression": str(best),
            "population_size": int(len(population)),
            "best_nodes": int(len(best)),
            "best_height": int(getattr(best, "height", 0)),
        }
        logbook_rows.append(row)
        if checkpoint_path is not None:
            save_gp_generation_checkpoint(
                checkpoint_path,
                held_out_dataset=held_out_dataset,
                seed=seed,
                terminal_names=terminal_names,
                gp_settings=gp_settings,
                completed_generations=generation + 1,
                population=population,
                hall_of_fame=hall_of_fame,
                logbook_rows=logbook_rows,
            )
        if progress_path is not None and (generation % progress_every == 0 or generation == num_generations - 1):
            append_progress_row(progress_path, row)
        if logger is not None and (generation % progress_every == 0 or generation == num_generations - 1):
            logger.info(
                "[heldout=%s seed=%d] gen=%d/%d best_score=%.6f nodes=%d height=%d",
                held_out_dataset,
                seed,
                generation + 1,
                num_generations,
                float(best.fitness.values[0]),
                int(len(best)),
                int(getattr(best, "height", 0)),
            )

    best = min(population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1]))
    best_func = toolbox.compile(expr=best)

    train_dataset_scores = []
    train_dataset_metric_scores: Dict[str, Dict[str, float]] = {}
    for dataset_name in train_datasets:
        score, _raw_metrics, _diag = evaluate_expression_on_dataset(
            func=best_func,
            terminal_names=terminal_names,
            terminal_curves=curves_by_dataset[str(dataset_name)],
            labels=labels_by_dataset[str(dataset_name)],
            dataset_name=str(dataset_name),
            reference_bounds=reference_bounds,
            timing_map=timing_map,
            array_backend=array_backend,
        )
        train_dataset_scores.append(float(score))
        train_dataset_metric_scores[str(dataset_name)] = dict(_diag["dataset_metric_scores"])

    if evaluation_dataset:
        heldout_score, heldout_metrics_raw, heldout_diag = evaluate_expression_on_dataset(
            func=best_func,
            terminal_names=terminal_names,
            terminal_curves=curves_by_dataset[evaluation_dataset],
            labels=labels_by_dataset[evaluation_dataset],
            dataset_name=evaluation_dataset,
            reference_bounds=reference_bounds,
            timing_map=timing_map,
            array_backend=array_backend,
        )
        heldout_metric_scores = heldout_diag["dataset_metric_scores"]
        mean_heldout_curve = heldout_diag["stack_fingerprint_mean_curve"]
    else:
        heldout_score = float(np.mean(train_dataset_scores)) if train_dataset_scores else float("nan")
        heldout_metrics_raw = {}
        heldout_metric_scores = {}
        mean_heldout_curve = []

    result = {
        "held_out_dataset": str(held_out_dataset),
        "evaluation_dataset": evaluation_dataset or "ALL_TRAINING_DATASETS",
        "seed": int(seed),
        "train_datasets": list(train_datasets),
        "terminals": list(terminal_names),
        "best_expression": str(best),
        "best_training_objective": float(best.fitness.values[0]),
        "best_complexity": float(best.fitness.values[1]),
        "num_nodes": int(len(best)),
        "tree_height": int(getattr(best, "height", 0)),
        "train_dataset_scores": train_dataset_scores,
        "train_dataset_metric_scores": train_dataset_metric_scores,
        "all_dataset_score": float(np.mean(train_dataset_scores)) if train_dataset_scores else float("nan"),
        "all_dataset_score_std": float(np.std(train_dataset_scores, ddof=0)) if train_dataset_scores else float("nan"),
        "heldout_score": float(heldout_score),
        "heldout_metrics_raw": heldout_metrics_raw,
        "heldout_metric_scores": heldout_metric_scores,
        "mean_heldout_curve": mean_heldout_curve,
        "hall_of_fame": [
            {
                "expression": str(individual),
                "fitness_generalization_score": float(individual.fitness.values[0]),
                "fitness_complexity": float(individual.fitness.values[1]),
            }
            for individual in hall_of_fame
        ],
        "logbook": logbook_rows,
        "gp_settings": dict(gp_settings),
        "array_backend": array_backend,
        "gp_primary_objective": GP_PRIMARY_OBJECTIVE,
        "gp_secondary_objective": GP_SECONDARY_OBJECTIVE,
    }
    if logger is not None:
        logger.info(
            "[heldout=%s seed=%d] best=%s heldout_score=%.6f",
            held_out_dataset,
            seed,
            result["best_expression"],
            result["heldout_score"],
        )
    return result


__all__ = [
    "DEAP_AVAILABLE",
    "GP_SEARCH_METRICS",
    "GP_SEARCH_METRIC_WEIGHTS",
    "build_pset",
    "build_reference_bounds",
    "build_toolbox",
    "compile_expression",
    "cupy_available",
    "evaluate_expression_on_dataset",
    "evaluate_expression_raw_metrics",
    "load_all_terminal_curves",
    "load_composite_labels",
    "load_curve_file",
    "load_terminal_curves_for_dataset",
    "load_timing_summary_map",
    "make_fitness_function",
    "normalize_against_reference",
    "require_deap",
    "require_cupy",
    "resample_curve",
    "run_gp_seed",
]
