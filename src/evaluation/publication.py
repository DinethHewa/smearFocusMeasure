"""Publication-focused evaluation helpers and gating artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import cv2 as cv
import numpy as np

from config.settings import (
    GENERALIZATION_ALPHA,
    PUBLICATION_RANK_STABILITY_FULL_MAX_STACKS_PER_DATASET,
    PUBLICATION_RANK_STABILITY_SMOKE_MAX_STACKS_PER_DATASET,
    PUBLICATION_RANK_STABILITY_TARGET_RESOLUTION,
    PUBLICATION_RANK_STABILITY_TOP_K,
)
from src.evaluation.aggregation import compute_rank_based_summary, compute_value_based_summary
from src.evaluation.autofocus_metrics import (
    compute_dataset_metrics_for_measure,
    compute_focus_curve_for_stack,
    normalize_focus_curve,
)


MeasureCallable = Callable[[np.ndarray], float]


def select_evenly_spaced_indices(num_items: int, max_items: int) -> List[int]:
    if num_items <= 0:
        return []
    if max_items <= 0 or num_items <= max_items:
        return list(range(num_items))
    raw = np.linspace(0, num_items - 1, num=max_items)
    indices = sorted({int(round(value)) for value in raw})
    while len(indices) < max_items:
        for idx in range(num_items):
            if idx not in indices:
                indices.append(idx)
            if len(indices) == max_items:
                break
    return sorted(indices[:max_items])


def resize_stack_to_square(stack: np.ndarray, target_size: int) -> np.ndarray:
    stack_arr = np.asarray(stack, dtype=np.float64)
    if stack_arr.ndim != 3:
        raise ValueError(f"Expected stack shape (num_slices, H, W), got {stack_arr.shape}")

    resized_slices: List[np.ndarray] = []
    for slice_2d in stack_arr:
        resized = cv.resize(
            np.asarray(slice_2d, dtype=np.float64),
            (int(target_size), int(target_size)),
            interpolation=cv.INTER_LINEAR,
        )
        resized_slices.append(np.asarray(resized, dtype=np.float64))
    return np.stack(resized_slices, axis=0)


def compute_norm_curves_and_timing(
    stacks: Sequence[np.ndarray],
    measure_func: MeasureCallable,
    *,
    target_resolution: int | None,
) -> tuple[List[np.ndarray], np.ndarray, float]:
    norm_curves: List[np.ndarray] = []
    processed_stacks: List[np.ndarray] = []
    total_time = 0.0
    total_slices = 0

    for stack in stacks:
        stack_arr = np.asarray(stack, dtype=np.float64)
        processed = resize_stack_to_square(stack_arr, target_resolution) if target_resolution is not None else stack_arr
        processed_stacks.append(processed)

        t0 = time.perf_counter()
        curve_raw = compute_focus_curve_for_stack(processed, measure_func)
        total_time += time.perf_counter() - t0
        total_slices += int(processed.shape[0])
        norm_curves.append(normalize_focus_curve(curve_raw))

    timing_value = float(total_time / max(1, total_slices))
    return norm_curves, np.asarray(processed_stacks, dtype=object), timing_value


def build_publication_measure_subset(
    rank_rows: Sequence[Mapping[str, Any]],
    value_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = PUBLICATION_RANK_STABILITY_TOP_K,
) -> List[str]:
    ordered: List[str] = []
    for rows in (rank_rows, value_rows):
        for row in sorted(rows, key=lambda item: float(item["final_rank"]))[:top_k]:
            name = str(row["measure_name"])
            if name not in ordered:
                ordered.append(name)
    return ordered


def _spearman_from_rank_pairs(rank_pairs: Sequence[tuple[float, float]]) -> float:
    if len(rank_pairs) < 2:
        return float("nan")
    arr = np.asarray(rank_pairs, dtype=np.float64)
    a = arr[:, 0]
    b = arr[:, 1]
    if np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_rank_stability_study(
    *,
    dataset_names: Sequence[str],
    measure_names: Sequence[str],
    stacks_by_dataset: Mapping[str, np.ndarray],
    label_loader,
    registry: Mapping[str, Mapping[str, Any]],
    run_mode: str,
    target_resolution: int = PUBLICATION_RANK_STABILITY_TARGET_RESOLUTION,
    smoke_max_stacks_per_dataset: int = PUBLICATION_RANK_STABILITY_SMOKE_MAX_STACKS_PER_DATASET,
    full_max_stacks_per_dataset: int = PUBLICATION_RANK_STABILITY_FULL_MAX_STACKS_PER_DATASET,
    skip_rrmse: bool = False,
) -> Dict[str, Any]:
    max_stacks = smoke_max_stacks_per_dataset if run_mode == "smoke" else full_max_stacks_per_dataset

    native_metric_raw: Dict[str, Dict[str, Dict[str, float]]] = {str(dataset_name): {} for dataset_name in dataset_names}
    target_metric_raw: Dict[str, Dict[str, Dict[str, float]]] = {str(dataset_name): {} for dataset_name in dataset_names}
    detail_rows: List[Dict[str, Any]] = []
    dataset_stack_counts: Dict[str, int] = {}

    for dataset_name in dataset_names:
        stack_arr = np.asarray(stacks_by_dataset[str(dataset_name)], dtype=object)
        subset_indices = select_evenly_spaced_indices(len(stack_arr), max_stacks)
        dataset_stack_counts[str(dataset_name)] = len(subset_indices)

        for measure_name in measure_names:
            measure_func = registry[str(measure_name)]["func"]
            labels_all, label_source_used = label_loader(str(dataset_name), str(measure_name))
            subset_stacks = [np.asarray(stack_arr[idx], dtype=np.float64) for idx in subset_indices]
            subset_labels = np.asarray(labels_all, dtype=int).reshape(-1)[subset_indices]

            native_curves, native_stacks, native_timing = compute_norm_curves_and_timing(
                subset_stacks,
                measure_func,
                target_resolution=None,
            )
            target_curves, target_stacks, target_timing = compute_norm_curves_and_timing(
                subset_stacks,
                measure_func,
                target_resolution=target_resolution,
            )

            native_metrics = compute_dataset_metrics_for_measure(
                dataset_name=str(dataset_name),
                measure_name=str(measure_name),
                measure_func=measure_func,
                norm_curves=native_curves,
                labels=subset_labels,
                timing_value=native_timing,
                stacks=native_stacks,
                run_mode=run_mode,
                skip_rrmse=skip_rrmse,
            )
            target_metrics = compute_dataset_metrics_for_measure(
                dataset_name=str(dataset_name),
                measure_name=str(measure_name),
                measure_func=measure_func,
                norm_curves=target_curves,
                labels=subset_labels,
                timing_value=target_timing,
                stacks=target_stacks,
                run_mode=run_mode,
                skip_rrmse=skip_rrmse,
            )

            native_metric_raw[str(dataset_name)][str(measure_name)] = native_metrics
            target_metric_raw[str(dataset_name)][str(measure_name)] = target_metrics
            detail_rows.append(
                {
                    "dataset_name": str(dataset_name),
                    "measure_name": str(measure_name),
                    "label_source_used": str(label_source_used),
                    "num_stacks_used": len(subset_indices),
                    "resolution_regime": "native_subset",
                    **native_metrics,
                }
            )
            detail_rows.append(
                {
                    "dataset_name": str(dataset_name),
                    "measure_name": str(measure_name),
                    "label_source_used": str(label_source_used),
                    "num_stacks_used": len(subset_indices),
                    "resolution_regime": f"{int(target_resolution)}_subset",
                    **target_metrics,
                }
            )

    native_rank_rows, _ = compute_rank_based_summary(
        dataset_metric_raw=native_metric_raw,
        dataset_subset=dataset_names,
        alpha=GENERALIZATION_ALPHA,
    )
    target_rank_rows, _ = compute_rank_based_summary(
        dataset_metric_raw=target_metric_raw,
        dataset_subset=dataset_names,
        alpha=GENERALIZATION_ALPHA,
    )
    native_value_rows = compute_value_based_summary(
        dataset_metric_raw=native_metric_raw,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=dataset_names,
        weighting_mode="equal_dataset",
        alpha=GENERALIZATION_ALPHA,
    )
    target_value_rows = compute_value_based_summary(
        dataset_metric_raw=target_metric_raw,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=dataset_names,
        weighting_mode="equal_dataset",
        alpha=GENERALIZATION_ALPHA,
    )

    native_rank_by_measure = {row["measure_name"]: row for row in native_rank_rows}
    target_rank_by_measure = {row["measure_name"]: row for row in target_rank_rows}
    native_value_by_measure = {row["measure_name"]: row for row in native_value_rows}
    target_value_by_measure = {row["measure_name"]: row for row in target_value_rows}

    summary_rows: List[Dict[str, Any]] = []
    for measure_name in measure_names:
        native_rank = native_rank_by_measure[str(measure_name)]
        target_rank = target_rank_by_measure[str(measure_name)]
        native_value = native_value_by_measure[str(measure_name)]
        target_value = target_value_by_measure[str(measure_name)]
        summary_rows.append(
            {
                "measure_name": str(measure_name),
                "native_rank_final_rank": int(native_rank["final_rank"]),
                "resolution_rank_final_rank": int(target_rank["final_rank"]),
                "rank_shift": int(target_rank["final_rank"]) - int(native_rank["final_rank"]),
                "native_value_final_rank": int(native_value["final_rank"]),
                "resolution_value_final_rank": int(target_value["final_rank"]),
                "value_rank_shift": int(target_value["final_rank"]) - int(native_value["final_rank"]),
                "native_generalization_score": float(native_value["generalization_score"]),
                "resolution_generalization_score": float(target_value["generalization_score"]),
                "native_weighted_mean": float(native_value["weighted_mean"]),
                "resolution_weighted_mean": float(target_value["weighted_mean"]),
            }
        )

    summary_rows.sort(key=lambda row: int(row["native_value_final_rank"]))
    top_k = min(5, len(summary_rows))
    native_top = {row["measure_name"] for row in sorted(summary_rows, key=lambda row: int(row["native_value_final_rank"]))[:top_k]}
    target_top = {row["measure_name"] for row in sorted(summary_rows, key=lambda row: int(row["resolution_value_final_rank"]))[:top_k]}
    rank_pairs = [
        (float(row["native_value_final_rank"]), float(row["resolution_value_final_rank"]))
        for row in summary_rows
    ]
    rank_shift_mean = float(np.mean([abs(float(row["value_rank_shift"])) for row in summary_rows])) if summary_rows else float("nan")

    summary = {
        "run_mode": str(run_mode),
        "comparison_resolution": int(target_resolution),
        "dataset_names": list(dataset_names),
        "dataset_stack_counts_used": dataset_stack_counts,
        "measures_evaluated": list(measure_names),
        "top_k_overlap_at_5": int(len(native_top & target_top)),
        "value_rank_spearman": _spearman_from_rank_pairs(rank_pairs),
        "mean_absolute_value_rank_shift": rank_shift_mean,
        "status": "complete",
        "skip_rrmse": bool(skip_rrmse),
    }

    return {
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "summary": summary,
        "native_metric_raw": native_metric_raw,
        "resolution_metric_raw": target_metric_raw,
        "native_value_rows": native_value_rows,
        "resolution_value_rows": target_value_rows,
        "native_rank_rows": native_rank_rows,
        "resolution_rank_rows": target_rank_rows,
    }


def build_single_measure_freeze_manifest(
    *,
    run_mode: str,
    file_paths: Mapping[str, Path],
    extra_details: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    files: Dict[str, Dict[str, Any]] = {}
    for key, path in file_paths.items():
        entry = {
            "path": str(path),
            "exists": bool(path.exists()),
        }
        if path.exists():
            stat = path.stat()
            entry["size_bytes"] = int(stat.st_size)
            entry["mtime_ns"] = int(stat.st_mtime_ns)
        files[str(key)] = entry
    manifest = {
        "run_mode": str(run_mode),
        "status": "frozen",
        "files": files,
    }
    if extra_details:
        manifest["details"] = dict(extra_details)
    return manifest


__all__ = [
    "MeasureCallable",
    "select_evenly_spaced_indices",
    "resize_stack_to_square",
    "compute_norm_curves_and_timing",
    "build_publication_measure_subset",
    "run_rank_stability_study",
    "build_single_measure_freeze_manifest",
]
