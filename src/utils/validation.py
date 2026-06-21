"""Validation, serialization, and checkpoint helpers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from config.paths import (
    DATASET_ORDER,
    LABEL_SOURCE_MANIFEST_FILE,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_output_dirs,
    get_source_label_file,
    get_stack_file,
    get_surrogate_label_file,
)


JSONDict = Dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_json(payload: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonify(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                fieldnames.append(key_str)

    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return path
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonify(value) for key, value in row.items()})
    return path


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def load_checkpoint(path: Path) -> JSONDict:
    path = Path(path)
    if not path.exists():
        return {}
    payload = load_json(path)
    return dict(payload) if isinstance(payload, dict) else {}


def write_checkpoint(
    *,
    checkpoint_path: Path,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "stage": stage,
        "status": status,
        "timestamp_utc": _now_iso(),
        "details": _jsonify(dict(details or {})),
    }
    return save_json(payload, checkpoint_path)


def _iter_stacks(stacks: np.ndarray) -> Iterable[np.ndarray]:
    arr = np.asarray(stacks, dtype=object)
    if arr.ndim >= 4:
        for idx in range(arr.shape[0]):
            yield np.asarray(arr[idx])
        return
    for stack in arr:
        yield np.asarray(stack)


def _stack_num_slices(stack: np.ndarray) -> int:
    arr = np.asarray(stack)
    if arr.ndim == 3:
        return int(arr.shape[0])
    if arr.ndim == 1:
        return int(len(arr))
    raise ValueError(f"Unsupported stack shape: {arr.shape}")


def validate_stack_array(stacks: np.ndarray, dataset_name: str) -> None:
    arr = np.asarray(stacks, dtype=object)
    if arr.size == 0:
        raise ValueError(f"[{dataset_name}] empty stack array")

    num_valid = 0
    for idx, stack in enumerate(_iter_stacks(arr)):
        stack_arr = np.asarray(stack)
        if stack_arr.ndim == 3:
            if stack_arr.shape[0] < 1:
                raise ValueError(f"[{dataset_name}] stack {idx} has no slices")
        elif stack_arr.ndim == 1:
            if len(stack_arr) < 1:
                raise ValueError(f"[{dataset_name}] stack {idx} has no slices")
            for slice_idx, slice_arr in enumerate(stack_arr):
                slice_2d = np.asarray(slice_arr)
                if slice_2d.ndim != 2:
                    raise ValueError(
                        f"[{dataset_name}] stack {idx} slice {slice_idx} is not 2D: {slice_2d.shape}"
                    )
        else:
            raise ValueError(f"[{dataset_name}] invalid stack {idx} shape: {stack_arr.shape}")
        num_valid += 1

    if num_valid == 0:
        raise ValueError(f"[{dataset_name}] no valid stacks found")


def validate_stack_and_label_alignment(
    stacks: np.ndarray,
    labels: Sequence[int],
    dataset_name: str,
) -> None:
    labels_arr = np.asarray(labels).reshape(-1)
    stack_list = list(_iter_stacks(np.asarray(stacks, dtype=object)))

    if len(stack_list) != len(labels_arr):
        raise ValueError(
            f"[{dataset_name}] label count {len(labels_arr)} does not match stack count {len(stack_list)}"
        )

    for idx, (stack, label) in enumerate(zip(stack_list, labels_arr)):
        label_int = int(label)
        num_slices = _stack_num_slices(stack)
        if not 0 <= label_int < num_slices:
            raise ValueError(
                f"[{dataset_name}] label {label_int} at stack {idx} is outside [0, {num_slices - 1}]"
            )


def validate_curve_range(curve: Sequence[float], *, normalized: bool) -> None:
    arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Curve is empty")
    if not np.isfinite(arr).all():
        raise ValueError("Curve contains non-finite values")
    if normalized and ((arr < -1e-6).any() or (arr > 1.000001).any()):
        raise ValueError("Normalized curve must stay within [0, 1]")


def validate_environment() -> Dict[str, str]:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project root not found: {PROJECT_ROOT}")
    ensure_output_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "project_root": str(PROJECT_ROOT),
        "outputs_dir": str(OUTPUTS_DIR),
        "reports_dir": str(REPORTS_DIR),
    }


def validate_pipeline_prerequisites(
    *,
    require_stacks: bool,
    require_labels: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "require_stacks": require_stacks,
        "require_labels": require_labels,
        "datasets_checked": list(DATASET_ORDER),
    }

    if require_stacks:
        missing_stacks = [str(get_stack_file(dataset)) for dataset in DATASET_ORDER if not get_stack_file(dataset).exists()]
        if missing_stacks:
            raise FileNotFoundError(
                "Missing stack files. Run scripts/01_build_stacks.py first.\n"
                + "\n".join(missing_stacks)
            )
        summary["stack_files_present"] = True

    if require_labels:
        if not LABEL_SOURCE_MANIFEST_FILE.exists():
            raise FileNotFoundError(
                f"Missing label manifest: {LABEL_SOURCE_MANIFEST_FILE}. "
                "Run scripts/02_build_reference_labels.py first."
            )
        missing_labels = []
        for dataset in DATASET_ORDER:
            if not get_source_label_file(dataset).exists() and not get_surrogate_label_file(dataset).exists():
                missing_labels.append(dataset)
        if missing_labels:
            raise FileNotFoundError(
                "Missing source/surrogate labels for datasets: " + ", ".join(missing_labels)
            )
        summary["label_files_present"] = True

    return summary


def summarize_existing_files(root: Path, *, suffixes: Sequence[str] | None = None) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []

    suffix_set = {s.lower() for s in suffixes} if suffixes else None
    files: List[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if suffix_set is not None and path.suffix.lower() not in suffix_set:
            continue
        files.append(str(path))
    return files


def validate_expected_asset_files(
    expected_paths: Iterable[Path],
    *,
    strict: bool,
) -> Dict[str, Any]:
    expected = [Path(p) for p in expected_paths]
    existing = [str(path) for path in expected if path.exists()]
    missing = [str(path) for path in expected if not path.exists()]

    if strict and missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing expected asset files ({len(missing)} total):\n{preview}")

    return {
        "expected_count": len(expected),
        "existing_count": len(existing),
        "missing_count": len(missing),
        "existing_files": existing,
        "missing_files": missing,
    }


__all__ = [
    "save_json",
    "load_json",
    "save_csv_rows",
    "load_csv_rows",
    "load_checkpoint",
    "write_checkpoint",
    "validate_stack_array",
    "validate_stack_and_label_alignment",
    "validate_curve_range",
    "validate_environment",
    "validate_pipeline_prerequisites",
    "summarize_existing_files",
    "validate_expected_asset_files",
]

