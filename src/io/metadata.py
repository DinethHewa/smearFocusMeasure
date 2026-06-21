"""Metadata builders for dataset stack artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.io.dataset_loader import StackDiscoveryResult


def _to_list_or_none(value: Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(x) for x in value]


def _stack_shape_signature(stack: np.ndarray) -> list[int]:
    arr = np.asarray(stack)
    if arr.ndim == 3:
        return [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])]
    if arr.ndim == 1:
        return [int(len(arr))]
    return [int(x) for x in arr.shape]


def _iter_slice_shapes(stack: np.ndarray) -> Iterable[str]:
    arr = np.asarray(stack)
    if arr.ndim == 3:
        for slice_2d in arr:
            yield str(tuple(int(x) for x in slice_2d.shape))
        return
    for slice_2d in arr:
        yield str(tuple(int(x) for x in np.asarray(slice_2d).shape))


def build_dataset_metadata(
    *,
    dataset_name: str,
    dataset_root: Path,
    discovery: StackDiscoveryResult,
    stacks: Sequence[np.ndarray],
    grayscale_conversion: bool,
    native_resolution_preserved: bool,
    use_roi_cropping: bool,
    roi_mode: str,
    roi_size: Sequence[int] | None,
    preserve_native_resolution_note: str = "Native resolution preserved by default",
) -> Dict[str, Any]:
    planes_per_stack = [int(np.asarray(stack).shape[0] if np.asarray(stack).ndim == 3 else len(np.asarray(stack))) for stack in stacks]
    image_shape_histogram: Counter[str] = Counter()
    stack_shape_examples: list[list[int]] = []

    for stack in stacks:
        image_shape_histogram.update(_iter_slice_shapes(stack))
        if len(stack_shape_examples) < 5:
            stack_shape_examples.append(_stack_shape_signature(stack))

    return {
        "dataset_name": dataset_name,
        "raw_dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "stack_count": len(stacks),
        "stack_count_before_truncation": int(discovery.stack_count_before_truncation),
        "discovery_mode": discovery.discovery_mode,
        "planes_per_stack_min": int(min(planes_per_stack)) if planes_per_stack else 0,
        "planes_per_stack_max": int(max(planes_per_stack)) if planes_per_stack else 0,
        "planes_per_stack_median": float(np.median(planes_per_stack)) if planes_per_stack else 0.0,
        "native_resolution_preserved": bool(native_resolution_preserved),
        "native_resolution_note": preserve_native_resolution_note,
        "use_roi_cropping": bool(use_roi_cropping),
        "roi_mode": roi_mode,
        "roi_size": _to_list_or_none(roi_size),
        "grayscale_conversion": bool(grayscale_conversion),
        "stack_folder_examples": [str(path) for path in discovery.stack_folders[:5]],
        "image_shape_histogram": dict(image_shape_histogram),
        "stack_shape_examples": stack_shape_examples,
        "sidecar_counts": dict(discovery.sidecar_counts),
        "rejected_low_count_examples": [
            {"folder": str(path), "image_count": int(count)}
            for path, count in discovery.rejected_low_count[:10]
        ],
    }


def add_metadata_note(metadata: Mapping[str, Any], *, key: str, value: Any) -> Dict[str, Any]:
    payload = dict(metadata)
    payload[key] = value
    return payload


__all__ = [
    "build_dataset_metadata",
    "add_metadata_note",
]
