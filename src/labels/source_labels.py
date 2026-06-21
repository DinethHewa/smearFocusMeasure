"""Source-label discovery helpers for real dataset sidecars and manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

from src.io.dataset_loader import discover_stack_folders, list_sidecar_files


SOURCE_LABEL_BASENAMES: Tuple[str, ...] = (
    "source_labels",
    "labels",
    "best_focus",
    "best_focus_index",
    "focus_peak_array",
    "focus_peaks",
    "peak_indices",
    "peak_index",
    "reference_peak",
    "reference_peaks",
)

SOURCE_LABEL_SUFFIXES: Tuple[str, ...] = (".npy", ".json", ".csv", ".mat")

LABEL_KEYS: Tuple[str, ...] = (
    "label",
    "labels",
    "best_focus",
    "best_focus_index",
    "focus_peak",
    "focus_peak_index",
    "focus_peaks",
    "peak",
    "peak_index",
    "peak_indices",
    "reference_peak",
    "reference_peaks",
)

KNOWN_METRIC_CURVE_PREFIXES: Tuple[str, ...] = (
    "Total_M",
    "TotalM",
)


@dataclass
class SourceLabelDiscovery:
    labels: Optional[np.ndarray]
    source_path: Optional[str]
    source_kind: Optional[str]
    note: str
    candidate_files: List[str] = field(default_factory=list)
    auxiliary_files: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _as_label_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([int(float(v)) for v in values], dtype=int).reshape(-1)


def _try_scalar_label(value: Any) -> Optional[int]:
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    return int(float(arr.reshape(-1)[0]))


def _extract_label_array_from_mapping(mapping: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in LABEL_KEYS:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            flat = np.asarray(value).reshape(-1)
            if flat.size >= 1:
                return _as_label_array(flat.tolist())
        scalar = _try_scalar_label(value)
        if scalar is not None:
            return np.asarray([scalar], dtype=int)
    return None


def _load_npy_labels(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    return _as_label_array(np.asarray(arr).reshape(-1).tolist())


def _load_json_labels(path: Path) -> np.ndarray:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _as_label_array(payload)
    if isinstance(payload, dict):
        arr = _extract_label_array_from_mapping(payload)
        if arr is not None:
            return arr
    raise ValueError(f"Unsupported JSON source-label structure: {path}")


def _load_csv_labels(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            matched = next((col for col in LABEL_KEYS if col in reader.fieldnames), None)
            if matched is None and len(reader.fieldnames) == 1:
                matched = reader.fieldnames[0]
            if matched is None:
                raise ValueError(f"Could not locate a label column in {path}")
            values = [int(float(row[matched])) for row in reader]
            return np.asarray(values, dtype=int).reshape(-1)

    # Headerless fallback.
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _as_label_array(lines)


def _mat_payload(path: Path) -> Dict[str, Any]:
    payload = loadmat(path)
    return {key: value for key, value in payload.items() if not key.startswith("__")}


def _load_mat_labels(path: Path) -> np.ndarray:
    payload = _mat_payload(path)
    arr = _extract_label_array_from_mapping(payload)
    if arr is not None:
        return arr
    raise ValueError(f"No explicit source-label field found in {path}")


def load_label_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy_labels(path)
    if suffix == ".json":
        return _load_json_labels(path)
    if suffix == ".csv":
        return _load_csv_labels(path)
    if suffix == ".mat":
        return _load_mat_labels(path)
    raise ValueError(f"Unsupported source-label file type: {path}")


def _dataset_candidate_names(dataset_name: str) -> List[str]:
    prefix = dataset_name.lower()
    names: List[str] = []
    for basename in SOURCE_LABEL_BASENAMES:
        for suffix in SOURCE_LABEL_SUFFIXES:
            names.append(f"{basename}{suffix}")
            names.append(f"{prefix}_{basename}{suffix}")
    return sorted(set(names))


def find_candidate_source_label_files(dataset_name: str, dataset_root: Path) -> List[Path]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    names = set(_dataset_candidate_names(dataset_name))
    search_dirs = [dataset_root]
    search_dirs.extend(sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()))

    candidates: List[Path] = []
    for folder in search_dirs:
        for name in names:
            path = folder / name
            if path.exists() and path.is_file():
                candidates.append(path)
    return sorted(set(candidates))


def _source_note_from_mat_payload(path: Path) -> Optional[str]:
    payload = _mat_payload(path)
    if any(key in payload for key in LABEL_KEYS):
        return None
    if any(key.startswith(KNOWN_METRIC_CURVE_PREFIXES) for key in payload):
        return f"mat_sidecar_without_explicit_label_field:{path.name}"
    return f"mat_sidecar_unrecognized:{path.name}"


def _try_load_dataset_level_source_labels(
    *,
    dataset_name: str,
    dataset_root: Path,
    num_stacks: int,
    logger=None,
) -> SourceLabelDiscovery:
    candidate_files = find_candidate_source_label_files(dataset_name, dataset_root)
    if not candidate_files:
        return SourceLabelDiscovery(
            labels=None,
            source_path=None,
            source_kind=None,
            note="no_candidate_source_label_file_found",
            candidate_files=[],
        )

    for path in candidate_files:
        try:
            labels = load_label_file(path)
        except Exception as exc:
            if logger is not None:
                logger.warning("[%s] failed to load candidate source labels %s: %s", dataset_name, path, exc)
            continue

        if len(labels) != num_stacks:
            if logger is not None:
                logger.warning(
                    "[%s] source-label length mismatch for %s: expected=%d got=%d",
                    dataset_name,
                    path,
                    num_stacks,
                    len(labels),
                )
            continue

        return SourceLabelDiscovery(
            labels=labels.astype(int),
            source_path=str(path),
            source_kind="dataset_level_file",
            note="loaded_dataset_level_source_labels",
            candidate_files=[str(item) for item in candidate_files],
        )

    return SourceLabelDiscovery(
        labels=None,
        source_path=None,
        source_kind=None,
        note="candidate_source_label_files_failed_or_mismatched",
        candidate_files=[str(item) for item in candidate_files],
    )


def _stack_sidecar_candidates(folder: Path, dataset_name: str) -> List[Path]:
    named = set(_dataset_candidate_names(dataset_name))
    candidates = [path for path in list_sidecar_files(folder) if path.name in named]
    mat_sidecars = [path for path in list_sidecar_files(folder) if path.suffix.lower() == ".mat"]
    return sorted(set(candidates + mat_sidecars), key=lambda path: path.name.lower())


def _try_load_per_stack_source_labels(
    *,
    dataset_name: str,
    dataset_root: Path,
    num_stacks: int,
    logger=None,
) -> SourceLabelDiscovery:
    discovery = discover_stack_folders(
        dataset_root,
        dataset_name,
        max_stacks=num_stacks,
        logger=logger,
    )

    labels: List[int] = []
    candidate_files: List[str] = []
    auxiliary_files: List[str] = []
    mat_notes: List[str] = []

    for folder in discovery.stack_folders:
        candidates = _stack_sidecar_candidates(folder, dataset_name)
        candidate_files.extend(str(path) for path in candidates)

        loaded_label: Optional[int] = None
        for path in candidates:
            try:
                arr = load_label_file(path)
            except Exception:
                if path.suffix.lower() == ".mat":
                    note = _source_note_from_mat_payload(path)
                    if note is not None:
                        mat_notes.append(note)
                        auxiliary_files.append(str(path))
                continue

            if len(arr) == 1:
                loaded_label = int(arr[0])
                break

        if loaded_label is None:
            break

        labels.append(loaded_label)

    if len(labels) == num_stacks:
        return SourceLabelDiscovery(
            labels=np.asarray(labels, dtype=int),
            source_path=None,
            source_kind="per_stack_sidecars",
            note="loaded_per_stack_source_labels",
            candidate_files=sorted(set(candidate_files)),
            auxiliary_files=sorted(set(auxiliary_files)),
        )

    if mat_notes:
        return SourceLabelDiscovery(
            labels=None,
            source_path=None,
            source_kind=None,
            note="per_stack_mat_sidecars_present_without_explicit_labels",
            candidate_files=sorted(set(candidate_files)),
            auxiliary_files=sorted(set(auxiliary_files)),
            details={"mat_notes": sorted(set(mat_notes))},
        )

    return SourceLabelDiscovery(
        labels=None,
        source_path=None,
        source_kind=None,
        note="no_per_stack_source_labels_found",
        candidate_files=sorted(set(candidate_files)),
        auxiliary_files=sorted(set(auxiliary_files)),
    )


def _survey_dataset_sidecars(
    *,
    dataset_name: str,
    dataset_root: Path,
    max_stack_folders_to_scan: int = 200,
) -> SourceLabelDiscovery:
    discovery = discover_stack_folders(
        dataset_root,
        dataset_name,
        max_stacks=max_stack_folders_to_scan,
        logger=None,
    )

    sidecars: List[Path] = []
    mat_notes: List[str] = []
    for folder in discovery.stack_folders:
        for path in list_sidecar_files(folder):
            sidecars.append(path)
            if path.suffix.lower() == ".mat":
                note = _source_note_from_mat_payload(path)
                if note is not None:
                    mat_notes.append(note)

    if mat_notes:
        return SourceLabelDiscovery(
            labels=None,
            source_path=None,
            source_kind=None,
            note="dataset_contains_mat_sidecars_without_explicit_source_labels",
            candidate_files=[],
            auxiliary_files=sorted(str(path) for path in sidecars[:20]),
            details={
                "mat_notes": sorted(set(mat_notes))[:20],
                "survey_stack_folders": int(len(discovery.stack_folders)),
            },
        )

    if sidecars:
        return SourceLabelDiscovery(
            labels=None,
            source_path=None,
            source_kind=None,
            note="dataset_contains_sidecars_without_explicit_source_labels",
            candidate_files=[],
            auxiliary_files=sorted(str(path) for path in sidecars[:20]),
            details={"survey_stack_folders": int(len(discovery.stack_folders))},
        )

    return SourceLabelDiscovery(
        labels=None,
        source_path=None,
        source_kind=None,
        note="no_dataset_sidecars_found",
        candidate_files=[],
        auxiliary_files=[],
        details={"survey_stack_folders": int(len(discovery.stack_folders))},
    )


def discover_source_labels(
    *,
    dataset_name: str,
    dataset_root: Path,
    num_stacks: int,
    logger=None,
) -> SourceLabelDiscovery:
    dataset_root = Path(dataset_root).expanduser().resolve()

    dataset_level = _try_load_dataset_level_source_labels(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        num_stacks=num_stacks,
        logger=logger,
    )
    if dataset_level.labels is not None:
        return dataset_level

    per_stack = _try_load_per_stack_source_labels(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        num_stacks=num_stacks,
        logger=logger,
    )

    # Prefer the more informative outcome between the two failed passes.
    if per_stack.note != "no_per_stack_source_labels_found":
        if not per_stack.candidate_files:
            per_stack.candidate_files = dataset_level.candidate_files
        return per_stack

    survey = _survey_dataset_sidecars(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )
    if survey.note != "no_dataset_sidecars_found":
        survey.candidate_files.extend(dataset_level.candidate_files)
        return survey

    dataset_level.auxiliary_files.extend(per_stack.auxiliary_files)
    dataset_level.details.update(per_stack.details)
    return dataset_level


__all__ = [
    "SourceLabelDiscovery",
    "SOURCE_LABEL_BASENAMES",
    "SOURCE_LABEL_SUFFIXES",
    "LABEL_KEYS",
    "find_candidate_source_label_files",
    "load_label_file",
    "discover_source_labels",
]
