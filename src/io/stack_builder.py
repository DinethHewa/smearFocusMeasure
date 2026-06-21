"""Native-resolution stack loading and dataset build helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2 as cv
import numpy as np

from src.io.dataset_loader import StackDiscoveryResult, discover_stack_folders, list_image_files
from src.io.metadata import build_dataset_metadata
from src.utils.validation import validate_stack_array


@dataclass(frozen=True)
class StackBuildConfig:
    convert_to_grayscale_when_needed: bool = True
    grayscale_mode: str = "luminance"
    preserve_native_resolution: bool = True
    use_roi_cropping: bool = False
    roi_mode: str = "center"
    roi_size: Tuple[int, int] | None = None
    require_same_size_within_stack: bool = False


def _to_grayscale(image: np.ndarray, config: StackBuildConfig) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]

    if not config.convert_to_grayscale_when_needed:
        raise ValueError(f"Expected grayscale image but got shape {arr.shape}")

    if config.grayscale_mode != "luminance":
        raise ValueError(f"Unsupported grayscale mode: {config.grayscale_mode}")

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {arr.shape}")

    if arr.shape[2] == 3:
        return cv.cvtColor(arr, cv.COLOR_BGR2GRAY)
    if arr.shape[2] == 4:
        return cv.cvtColor(arr, cv.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported channel count for grayscale conversion: {arr.shape}")


def _center_crop(arr: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    height, width = arr.shape[:2]
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"Requested ROI {(crop_h, crop_w)} exceeds image size {(height, width)}"
        )
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return arr[y0:y0 + crop_h, x0:x0 + crop_w]


def _apply_roi(arr: np.ndarray, config: StackBuildConfig) -> np.ndarray:
    if not config.use_roi_cropping or config.roi_mode == "none":
        return arr

    if config.roi_size is None:
        raise ValueError("ROI cropping is enabled but roi_size is None")

    if config.roi_mode == "center":
        return _center_crop(arr, int(config.roi_size[0]), int(config.roi_size[1]))

    raise ValueError(f"Unsupported ROI mode: {config.roi_mode}")


def load_single_image(path: Path, config: StackBuildConfig) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    gray = _to_grayscale(image, config)
    gray = _apply_roi(gray, config)

    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image at {path}, got shape {gray.shape}")

    return np.asarray(gray)


def build_stack_from_folder(
    folder: Path,
    *,
    dataset_name: str,
    config: StackBuildConfig,
    logger=None,
) -> np.ndarray:
    image_files = list_image_files(folder)
    if len(image_files) < 2:
        raise ValueError(f"[{dataset_name}] folder does not contain a valid stack: {folder}")

    slices: List[np.ndarray] = []
    slice_shapes: List[Tuple[int, int]] = []

    for image_path in image_files:
        arr = load_single_image(image_path, config)
        slices.append(arr)
        slice_shapes.append(tuple(int(x) for x in arr.shape))

    unique_shapes = sorted(set(slice_shapes))
    if len(unique_shapes) > 1:
        message = (
            f"[{dataset_name}] inconsistent slice shapes inside stack folder {folder}: "
            f"{unique_shapes}"
        )
        if config.require_same_size_within_stack:
            raise ValueError(message)
        if logger is not None:
            logger.warning(message)
        stack = np.empty(len(slices), dtype=object)
        for idx, slice_2d in enumerate(slices):
            stack[idx] = slice_2d
        return stack

    return np.stack(slices, axis=0)


def build_dataset_stacks(
    *,
    dataset_name: str,
    dataset_root: Path,
    max_stacks: int | None,
    logger,
    config: StackBuildConfig,
) -> Tuple[np.ndarray, Dict]:
    discovery: StackDiscoveryResult = discover_stack_folders(
        dataset_root,
        dataset_name,
        max_stacks=max_stacks,
        logger=logger,
    )

    stack_list: List[np.ndarray] = []
    for idx, folder in enumerate(discovery.stack_folders, start=1):
        stack = build_stack_from_folder(
            folder,
            dataset_name=dataset_name,
            config=config,
            logger=logger,
        )
        stack_list.append(stack)
        if logger is not None and idx % 100 == 0:
            logger.info("[%s] processed %d stacks", dataset_name, idx)

    dataset_array = np.empty(len(stack_list), dtype=object)
    for idx, stack in enumerate(stack_list):
        dataset_array[idx] = stack

    validate_stack_array(dataset_array, dataset_name)

    metadata = build_dataset_metadata(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        discovery=discovery,
        stacks=stack_list,
        grayscale_conversion=config.convert_to_grayscale_when_needed,
        native_resolution_preserved=config.preserve_native_resolution,
        use_roi_cropping=config.use_roi_cropping,
        roi_mode=config.roi_mode,
        roi_size=config.roi_size,
    )
    return dataset_array, metadata


__all__ = [
    "StackBuildConfig",
    "load_single_image",
    "build_stack_from_folder",
    "build_dataset_stacks",
]
