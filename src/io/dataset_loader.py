"""Dataset traversal helpers for raw autofocus stack discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SIDECAR_EXTENSIONS = {".mat", ".csv", ".json", ".npy"}
WINDOWS_STREAM_MARKER = ":Zone.Identifier"


def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def natural_path_key(path: Path, root: Path | None = None) -> List[object]:
    target = path if root is None else path.relative_to(root)
    return natural_key(str(target))


def is_windows_stream(path: Path | str) -> bool:
    return WINDOWS_STREAM_MARKER in str(path)


def is_image_file(path: Path) -> bool:
    return (
        path.is_file()
        and not is_windows_stream(path.name)
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_sidecar_file(path: Path) -> bool:
    return (
        path.is_file()
        and not is_windows_stream(path.name)
        and path.suffix.lower() in SIDECAR_EXTENSIONS
    )


def list_image_files(folder: Path) -> List[Path]:
    try:
        children = list(folder.iterdir())
    except PermissionError:
        return []
    return sorted(
        [path for path in children if is_image_file(path)],
        key=lambda path: natural_path_key(path, folder),
    )


def list_sidecar_files(folder: Path) -> List[Path]:
    try:
        children = list(folder.iterdir())
    except PermissionError:
        return []
    return sorted(
        [path for path in children if is_sidecar_file(path)],
        key=lambda path: natural_path_key(path, folder),
    )


@dataclass(frozen=True)
class StackDiscoveryResult:
    dataset_name: str
    dataset_root: Path
    discovery_mode: str
    stack_folders: Tuple[Path, ...]
    stack_count_before_truncation: int
    rejected_low_count: Tuple[Tuple[Path, int], ...]
    sidecar_counts: Dict[str, int]


def _collect_immediate_child_stack_dirs(
    dataset_root: Path,
    *,
    min_images_per_stack: int,
) -> Tuple[List[Path], List[Tuple[Path, int]]]:
    valid: List[Path] = []
    rejected: List[Tuple[Path, int]] = []

    child_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir()],
        key=lambda path: natural_path_key(path, dataset_root),
    )
    for folder in child_dirs:
        image_files = list_image_files(folder)
        if len(image_files) >= min_images_per_stack:
            valid.append(folder)
        elif image_files:
            rejected.append((folder, len(image_files)))

    return valid, rejected


def _collect_recursive_stack_dirs(
    dataset_root: Path,
    *,
    min_images_per_stack: int,
) -> Tuple[List[Path], List[Tuple[Path, int]]]:
    valid: List[Path] = []
    rejected: List[Tuple[Path, int]] = []

    for folder in sorted(
        [path for path in dataset_root.rglob("*") if path.is_dir()],
        key=lambda path: natural_path_key(path, dataset_root),
    ):
        image_files = list_image_files(folder)
        if len(image_files) >= min_images_per_stack:
            valid.append(folder)
        elif image_files:
            rejected.append((folder, len(image_files)))

    return valid, rejected


def _count_sidecars(stack_folders: Sequence[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for folder in stack_folders:
        for sidecar in list_sidecar_files(folder):
            suffix = sidecar.suffix.lower()
            counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def discover_stack_folders(
    dataset_root: Path,
    dataset_name: str,
    *,
    min_images_per_stack: int = 2,
    max_stacks: int | None = None,
    logger=None,
) -> StackDiscoveryResult:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"[{dataset_name}] dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"[{dataset_name}] dataset root is not a directory: {dataset_root}")

    immediate_valid, rejected = _collect_immediate_child_stack_dirs(
        dataset_root,
        min_images_per_stack=min_images_per_stack,
    )

    if immediate_valid:
        discovery_mode = "immediate_child_dirs"
        stack_folders = immediate_valid
    else:
        root_images = list_image_files(dataset_root)
        if len(root_images) >= min_images_per_stack:
            discovery_mode = "dataset_root"
            stack_folders = [dataset_root]
        else:
            discovery_mode = "recursive_fallback"
            stack_folders, recursive_rejected = _collect_recursive_stack_dirs(
                dataset_root,
                min_images_per_stack=min_images_per_stack,
            )
            rejected.extend(recursive_rejected)

    if not stack_folders:
        raise RuntimeError(
            f"[{dataset_name}] no stack folders found under {dataset_root}. "
            "Expected directories containing multiple image files."
        )

    stack_count_before_truncation = len(stack_folders)
    if max_stacks is not None:
        stack_folders = stack_folders[:max_stacks]

    sidecar_counts = _count_sidecars(stack_folders)

    if logger is not None:
        logger.info(
            "[%s] stack discovery mode=%s, found=%d, using=%d, rejected_low_count=%d",
            dataset_name,
            discovery_mode,
            stack_count_before_truncation,
            len(stack_folders),
            len(rejected),
        )

    return StackDiscoveryResult(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        discovery_mode=discovery_mode,
        stack_folders=tuple(stack_folders),
        stack_count_before_truncation=stack_count_before_truncation,
        rejected_low_count=tuple(rejected),
        sidecar_counts=sidecar_counts,
    )


__all__ = [
    "IMAGE_EXTENSIONS",
    "SIDECAR_EXTENSIONS",
    "WINDOWS_STREAM_MARKER",
    "StackDiscoveryResult",
    "natural_key",
    "natural_path_key",
    "is_windows_stream",
    "is_image_file",
    "is_sidecar_file",
    "list_image_files",
    "list_sidecar_files",
    "discover_stack_folders",
]
