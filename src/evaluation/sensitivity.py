"""Timing-only resized analysis and reusable sensitivity helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

import cv2 as cv
import numpy as np

from config.settings import (
    ALPHA_SENSITIVITY_VALUES,
    INCLUDE_NATIVE_IN_TIMING,
    TIMING_RESOLUTIONS,
)
from src.evaluation.aggregation import compute_rank_based_summary, compute_value_based_summary


MeasureCallable = Callable[[np.ndarray], float]

SMOKE_TIMING_MAX_STACKS_PER_DATASET = 3
SMOKE_TIMING_MAX_SLICES_PER_STACK = 3
FULL_TIMING_MAX_STACKS_PER_DATASET = 10
FULL_TIMING_MAX_SLICES_PER_STACK = 5


def resize_slice_for_timing(slice_2d: np.ndarray, target_size: int) -> np.ndarray:
    arr = np.asarray(slice_2d)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale slice for timing resize, got {arr.shape}")

    arr_float = arr.astype(np.float64, copy=False)
    amin = float(arr_float.min())
    amax = float(arr_float.max())
    if amax - amin <= 1e-12:
        scaled = np.zeros_like(arr_float, dtype=np.uint8)
    else:
        scaled = ((arr_float - amin) / (amax - amin + 1e-12) * 255.0).clip(0, 255).astype(np.uint8)
    resized = cv.resize(scaled, (int(target_size), int(target_size)), interpolation=cv.INTER_LINEAR)
    return resized.astype(np.float64, copy=False)


def select_timing_subset(
    stacks: np.ndarray,
    *,
    run_mode: str,
    smoke_max_stacks: int = SMOKE_TIMING_MAX_STACKS_PER_DATASET,
    smoke_max_slices: int = SMOKE_TIMING_MAX_SLICES_PER_STACK,
    full_max_stacks: int = FULL_TIMING_MAX_STACKS_PER_DATASET,
    full_max_slices: int = FULL_TIMING_MAX_SLICES_PER_STACK,
) -> List[np.ndarray]:
    if run_mode == "smoke":
        max_stacks = smoke_max_stacks
        max_slices = smoke_max_slices
    else:
        max_stacks = full_max_stacks
        max_slices = full_max_slices

    subset: List[np.ndarray] = []
    for stack in np.asarray(stacks, dtype=object)[:max_stacks]:
        stack_arr = np.asarray(stack)
        if stack_arr.ndim != 3:
            continue
        subset.append(stack_arr[:max_slices])
    return subset


def compute_timing_summary_for_measure(
    *,
    dataset_name: str,
    stacks: np.ndarray,
    measure_name: str,
    measure_func: MeasureCallable,
    run_mode: str,
    include_native: bool = INCLUDE_NATIVE_IN_TIMING,
    timing_resolutions: Sequence[int] = TIMING_RESOLUTIONS,
) -> Dict[str, Any]:
    subset = select_timing_subset(stacks, run_mode=run_mode)
    timing_record: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "measure_name": measure_name,
        "num_stacks_used": len(subset),
        "native_avg_time_per_slice_sec": None,
        "resized_avg_time_per_slice_sec": {},
    }
    if not subset:
        return timing_record

    if include_native:
        total_time = 0.0
        total_slices = 0
        for stack in subset:
            for slice_2d in stack:
                t0 = time.perf_counter()
                _ = float(measure_func(slice_2d))
                total_time += time.perf_counter() - t0
                total_slices += 1
        timing_record["native_avg_time_per_slice_sec"] = total_time / total_slices if total_slices > 0 else None

    for target_size in timing_resolutions:
        total_time = 0.0
        total_slices = 0
        for stack in subset:
            for slice_2d in stack:
                resized = resize_slice_for_timing(slice_2d, int(target_size))
                t0 = time.perf_counter()
                _ = float(measure_func(resized))
                total_time += time.perf_counter() - t0
                total_slices += 1
        timing_record["resized_avg_time_per_slice_sec"][str(target_size)] = total_time / total_slices if total_slices > 0 else None

    return timing_record


def flatten_timing_record(timing_record: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "dataset_name": timing_record["dataset_name"],
        "measure_name": timing_record["measure_name"],
        "num_stacks_used": timing_record["num_stacks_used"],
        "native_avg_time_per_slice_sec": timing_record["native_avg_time_per_slice_sec"],
    }
    for resolution, value in dict(timing_record.get("resized_avg_time_per_slice_sec", {})).items():
        row[f"avg_time_per_slice_sec_{resolution}"] = value
    return row


def summarize_dataset_label_modes(dataset_label_modes: Mapping[str, str]) -> Dict[str, List[str]]:
    source_datasets = [dataset_name for dataset_name, mode in dataset_label_modes.items() if mode == "source"]
    surrogate_datasets = [dataset_name for dataset_name, mode in dataset_label_modes.items() if mode == "surrogate"]
    return {
        "source_datasets": source_datasets,
        "surrogate_datasets": surrogate_datasets,
    }


def build_label_split_summary(
    *,
    dataset_metric_raw: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_stack_counts: Mapping[str, int],
    dataset_label_modes: Mapping[str, str],
    alpha: float,
) -> Dict[str, Any]:
    split_sets = summarize_dataset_label_modes(dataset_label_modes)
    summary: Dict[str, Any] = {
        "source_datasets": split_sets["source_datasets"],
        "surrogate_datasets": split_sets["surrogate_datasets"],
        "source_label_only_rank": [],
        "surrogate_label_only_rank": [],
        "source_label_only_value": [],
        "surrogate_label_only_value": [],
    }

    if split_sets["source_datasets"]:
        summary["source_label_only_rank"], _ = compute_rank_based_summary(
            dataset_metric_raw=dataset_metric_raw,
            dataset_subset=split_sets["source_datasets"],
            alpha=alpha,
        )
        summary["source_label_only_value"] = compute_value_based_summary(
            dataset_metric_raw=dataset_metric_raw,
            dataset_stack_counts=dataset_stack_counts,
            dataset_subset=split_sets["source_datasets"],
            weighting_mode="equal_dataset",
            alpha=alpha,
        )

    if split_sets["surrogate_datasets"]:
        summary["surrogate_label_only_rank"], _ = compute_rank_based_summary(
            dataset_metric_raw=dataset_metric_raw,
            dataset_subset=split_sets["surrogate_datasets"],
            alpha=alpha,
        )
        summary["surrogate_label_only_value"] = compute_value_based_summary(
            dataset_metric_raw=dataset_metric_raw,
            dataset_stack_counts=dataset_stack_counts,
            dataset_subset=split_sets["surrogate_datasets"],
            weighting_mode="equal_dataset",
            alpha=alpha,
        )

    return summary


def compute_alpha_sensitivity(
    *,
    dataset_metric_raw: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_stack_counts: Mapping[str, int],
    dataset_subset: Sequence[str],
    weighting_mode: str = "equal_dataset",
    alpha_values: Sequence[float] = ALPHA_SENSITIVITY_VALUES,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alpha in alpha_values:
        rows.extend(
            compute_value_based_summary(
                dataset_metric_raw=dataset_metric_raw,
                dataset_stack_counts=dataset_stack_counts,
                dataset_subset=dataset_subset,
                weighting_mode=weighting_mode,
                alpha=float(alpha),
            )
        )
    return rows


__all__ = [
    "MeasureCallable",
    "SMOKE_TIMING_MAX_STACKS_PER_DATASET",
    "SMOKE_TIMING_MAX_SLICES_PER_STACK",
    "FULL_TIMING_MAX_STACKS_PER_DATASET",
    "FULL_TIMING_MAX_SLICES_PER_STACK",
    "resize_slice_for_timing",
    "select_timing_subset",
    "compute_timing_summary_for_measure",
    "flatten_timing_record",
    "summarize_dataset_label_modes",
    "build_label_split_summary",
    "compute_alpha_sensitivity",
]
