"""Corrected autofocus metric helpers for single-measure evaluation."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np

from config.settings import (
    AUTOFOCUS_METRICS,
    EPS,
    NOISE_RANDOM_SEED,
    NOISE_STD_FOR_RRMSE,
)
from src.utils.validation import validate_curve_range


MetricSummary = Dict[str, float]
MeasureCallable = Callable[[np.ndarray], float]

METRIC_DIRECTION: Dict[str, bool] = {
    "absolute_peak_localization_error": True,
    "fwhm": True,
    "curvature_at_peak": False,
    "steep_slope_width": True,
    "steep_to_gradual_slope_ratio": False,
    "false_maxima_count": True,
    "noise_level": True,
    "rrmse_under_additive_noise": True,
    "range_around_global_maximum": True,
    "execution_time_per_slice": True,
}


def normalize_focus_curve(curve: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    cmin = float(np.min(curve_arr))
    cmax = float(np.max(curve_arr))
    denom = cmax - cmin
    if denom <= eps:
        normalized = np.zeros_like(curve_arr, dtype=np.float64)
    else:
        normalized = (curve_arr - cmin) / (denom + eps)
    validate_curve_range(normalized, normalized=True)
    return normalized


def compute_focus_curve_for_stack(stack: np.ndarray, measure_func: MeasureCallable) -> np.ndarray:
    stack_arr = np.asarray(stack)
    if stack_arr.ndim != 3:
        raise ValueError(f"Expected stack shape (num_slices, H, W), got {stack_arr.shape}")

    scores = [float(measure_func(slice_2d)) for slice_2d in stack_arr]
    curve = np.asarray(scores, dtype=np.float64)
    validate_curve_range(curve, normalized=False)
    return curve


def tie_break_peak(indices: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in indices)
    return ordered[len(ordered) // 2]


def predict_peak_index(curve: np.ndarray) -> int:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    best = float(np.max(curve_arr))
    tied = np.where(np.isclose(curve_arr, best))[0].tolist()
    return tie_break_peak(tied)


def absolute_peak_localization_error(curve: np.ndarray, label_idx: int) -> float:
    return float(abs(predict_peak_index(curve) - int(label_idx)))


def full_width_half_maximum(curve: np.ndarray, *, eps: float = EPS) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_val = float(np.max(curve_arr))
    if peak_val <= eps:
        return float(len(curve_arr))
    threshold = 0.5 * peak_val
    idx = np.where(curve_arr >= threshold)[0]
    if idx.size == 0:
        return float(len(curve_arr))
    return float(idx[-1] - idx[0] + 1)


def curvature_at_peak(curve: np.ndarray) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(curve_arr)
    if peak_idx == 0 or peak_idx == len(curve_arr) - 1:
        return 0.0
    second_diff = curve_arr[peak_idx - 1] - 2.0 * curve_arr[peak_idx] + curve_arr[peak_idx + 1]
    return float(max(0.0, -second_diff))


def _local_minima_indices(curve: np.ndarray) -> List[int]:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    minima: List[int] = []
    for idx in range(1, len(curve_arr) - 1):
        if curve_arr[idx] <= curve_arr[idx - 1] and curve_arr[idx] <= curve_arr[idx + 1]:
            minima.append(idx)
    return minima


def steep_slope_width(curve: np.ndarray) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(curve_arr)
    minima = _local_minima_indices(curve_arr)

    left_candidates = [idx for idx in minima if idx < peak_idx]
    right_candidates = [idx for idx in minima if idx > peak_idx]
    left = max(left_candidates) if left_candidates else 0
    right = min(right_candidates) if right_candidates else len(curve_arr) - 1
    return float(right - left)


def steep_to_gradual_slope_ratio(curve: np.ndarray, *, eps: float = EPS) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(curve_arr)

    local_slopes: List[float] = []
    if peak_idx > 0:
        local_slopes.append(abs(float(curve_arr[peak_idx] - curve_arr[peak_idx - 1])))
    if peak_idx < len(curve_arr) - 1:
        local_slopes.append(abs(float(curve_arr[peak_idx] - curve_arr[peak_idx + 1])))
    local_mean = float(np.mean(local_slopes)) if local_slopes else 0.0

    diffs = np.abs(np.diff(curve_arr))
    mask = np.ones_like(diffs, dtype=bool)
    if peak_idx - 1 >= 0 and peak_idx - 1 < len(mask):
        mask[peak_idx - 1] = False
    if peak_idx < len(mask):
        mask[peak_idx] = False
    background = diffs[mask]
    background_mean = float(np.mean(background)) if background.size else 0.0
    return float(local_mean / (background_mean + eps))


def false_maxima_count(curve: np.ndarray) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(curve_arr)
    count = 0
    for idx in range(1, len(curve_arr) - 1):
        if idx == peak_idx:
            continue
        if curve_arr[idx] > curve_arr[idx - 1] and curve_arr[idx] > curve_arr[idx + 1]:
            count += 1
    return float(count)


def noise_level(curve: np.ndarray) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    if len(curve_arr) < 3:
        return 0.0
    second_diff = np.diff(curve_arr, n=2)
    return float(np.mean(second_diff ** 2))


def range_around_global_maximum(curve: np.ndarray, *, eps: float = EPS) -> float:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_val = float(np.max(curve_arr))
    if peak_val <= eps:
        return float(len(curve_arr))

    peak_idx = predict_peak_index(curve_arr)
    threshold = 0.95 * peak_val

    left = peak_idx
    while left - 1 >= 0 and curve_arr[left - 1] >= threshold:
        left -= 1

    right = peak_idx
    while right + 1 < len(curve_arr) and curve_arr[right + 1] >= threshold:
        right += 1

    return float(right - left + 1)


def add_noise_to_slice(slice_2d: np.ndarray, rng: np.random.Generator, *, noise_std: float = NOISE_STD_FOR_RRMSE, eps: float = EPS) -> np.ndarray:
    image = np.asarray(slice_2d, dtype=np.float64)
    xmin = float(np.min(image))
    xmax = float(np.max(image))
    if xmax - xmin <= eps:
        normalized = np.zeros_like(image)
    else:
        normalized = (image - xmin) / (xmax - xmin + eps)
    noisy = normalized + rng.normal(0.0, noise_std, size=normalized.shape)
    return np.clip(noisy, 0.0, 1.0)


def rrmse_under_additive_noise(
    clean_curve_norm: np.ndarray,
    stack: np.ndarray,
    measure_func: MeasureCallable,
    rng: np.random.Generator,
    *,
    eps: float = EPS,
) -> float:
    noisy_scores = [float(measure_func(add_noise_to_slice(slice_2d, rng))) for slice_2d in np.asarray(stack)]
    noisy_curve = np.asarray(noisy_scores, dtype=np.float64)
    noisy_curve_norm = normalize_focus_curve(noisy_curve, eps=eps)
    diff = np.asarray(clean_curve_norm, dtype=np.float64) - noisy_curve_norm
    numerator = np.sqrt(np.mean(diff ** 2))
    denominator = np.sqrt(np.mean(np.asarray(clean_curve_norm, dtype=np.float64) ** 2)) + eps
    return float(numerator / denominator)


def compute_dataset_metrics_for_measure(
    *,
    dataset_name: str,
    measure_name: str,
    measure_func: MeasureCallable,
    norm_curves: Sequence[np.ndarray],
    labels: np.ndarray,
    timing_value: float,
    stacks: np.ndarray,
    run_mode: str,
    skip_rrmse: bool,
    smoke_rrmse_cap: int = 5,
    full_rrmse_cap: int = 200,
    noise_seed: int = NOISE_RANDOM_SEED,
) -> MetricSummary:
    if len(norm_curves) != len(labels):
        raise ValueError(
            f"[{dataset_name}] measure {measure_name}: curve/label length mismatch "
            f"({len(norm_curves)} vs {len(labels)})"
        )

    metric_values: Dict[str, List[float]] = {metric: [] for metric in AUTOFOCUS_METRICS}
    rng = np.random.default_rng(noise_seed)
    rrmse_cap = smoke_rrmse_cap if run_mode == "smoke" else full_rrmse_cap

    for idx, (curve_norm, label_idx) in enumerate(zip(norm_curves, labels)):
        curve_arr = np.asarray(curve_norm, dtype=np.float64).reshape(-1)
        metric_values["absolute_peak_localization_error"].append(
            absolute_peak_localization_error(curve_arr, int(label_idx))
        )
        metric_values["fwhm"].append(full_width_half_maximum(curve_arr))
        metric_values["curvature_at_peak"].append(curvature_at_peak(curve_arr))
        metric_values["steep_slope_width"].append(steep_slope_width(curve_arr))
        metric_values["steep_to_gradual_slope_ratio"].append(steep_to_gradual_slope_ratio(curve_arr))
        metric_values["false_maxima_count"].append(false_maxima_count(curve_arr))
        metric_values["noise_level"].append(noise_level(curve_arr))
        metric_values["range_around_global_maximum"].append(range_around_global_maximum(curve_arr))
        metric_values["execution_time_per_slice"].append(float(timing_value))

        if not skip_rrmse and idx < rrmse_cap:
            stack_arr = np.asarray(stacks[idx])
            if stack_arr.ndim == 3:
                metric_values["rrmse_under_additive_noise"].append(
                    rrmse_under_additive_noise(
                        clean_curve_norm=curve_arr,
                        stack=stack_arr,
                        measure_func=measure_func,
                        rng=rng,
                    )
                )

    summary: MetricSummary = {}
    for metric_name, values in metric_values.items():
        summary[metric_name] = float(np.nanmean(values)) if values else float("nan")
    return summary


__all__ = [
    "MetricSummary",
    "MeasureCallable",
    "METRIC_DIRECTION",
    "normalize_focus_curve",
    "compute_focus_curve_for_stack",
    "tie_break_peak",
    "predict_peak_index",
    "absolute_peak_localization_error",
    "full_width_half_maximum",
    "curvature_at_peak",
    "steep_slope_width",
    "steep_to_gradual_slope_ratio",
    "false_maxima_count",
    "noise_level",
    "range_around_global_maximum",
    "add_noise_to_slice",
    "rrmse_under_additive_noise",
    "compute_dataset_metrics_for_measure",
]
