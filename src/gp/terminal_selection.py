"""Fold-local terminal selection for corrected composite GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from config.paths import DATASET_ORDER, SINGLE_EVAL_SUPP_DIR, get_single_norm_curve_file, get_stack_file
from config.settings import DEFAULT_GP_FALLBACK_TERMINALS, GENERALIZATION_ALPHA
from src.evaluation.aggregation import compute_rank_based_summary, compute_value_based_summary
from src.utils.validation import load_json


@dataclass(frozen=True)
class FoldTerminalSelection:
    held_out_dataset: str
    train_datasets: Tuple[str, ...]
    selected_terminals: Tuple[str, ...]
    selection_rows: Tuple[Dict[str, Any], ...]
    rank_rows: Tuple[Dict[str, Any], ...]
    value_rows: Tuple[Dict[str, Any], ...]


def load_dataset_metric_reference() -> Dict[str, Dict[str, Dict[str, float]]]:
    path = SINGLE_EVAL_SUPP_DIR / "dataset_metric_raw.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing corrected single-measure metric reference: {path}. "
            "Run scripts/04_evaluate_single_measures.py first."
        )
    payload = load_json(path)
    return {
        str(dataset_name): {
            str(measure_name): {
                str(metric_name): float(metric_value)
                for metric_name, metric_value in metrics.items()
            }
            for measure_name, metrics in by_measure.items()
        }
        for dataset_name, by_measure in payload.items()
    }


def load_dataset_stack_counts() -> Dict[str, int]:
    path = SINGLE_EVAL_SUPP_DIR / "dataset_stack_counts.json"
    if path.exists():
        payload = load_json(path)
        return {str(dataset_name): int(count) for dataset_name, count in payload.items()}

    counts: Dict[str, int] = {}
    for dataset_name in DATASET_ORDER:
        stack_path = get_stack_file(dataset_name)
        if not stack_path.exists():
            raise FileNotFoundError(
                f"Missing stack array for dataset {dataset_name}: {stack_path}. "
                "Run scripts/01_build_stacks.py first."
            )
        counts[str(dataset_name)] = int(len(np.load(stack_path, allow_pickle=True)))
    return counts


def _train_datasets(held_out_dataset: str, dataset_order: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(name) for name in dataset_order if str(name) != str(held_out_dataset))


def _preserve_unique(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(str(item))
    return ordered


def _terminal_family(measure_name: str, registry: Mapping[str, Mapping[str, Any]]) -> str:
    return str(registry.get(measure_name, {}).get("family", "unknown"))


def _curve_files_exist(measure_name: str, dataset_order: Sequence[str]) -> bool:
    return all(get_single_norm_curve_file(dataset_name, measure_name).exists() for dataset_name in dataset_order)


def select_terminals_for_fold(
    *,
    held_out_dataset: str,
    dataset_metric_reference: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_stack_counts: Mapping[str, int],
    registry: Mapping[str, Mapping[str, Any]],
    top_k_rank: int,
    top_k_value: int,
    fallback_terminals: Sequence[str] = DEFAULT_GP_FALLBACK_TERMINALS,
    dataset_order: Sequence[str] = DATASET_ORDER,
    max_per_family: int = 2,
) -> FoldTerminalSelection:
    train_datasets = _train_datasets(held_out_dataset, dataset_order)
    if not train_datasets:
        raise ValueError("Terminal selection requires at least one training dataset")

    rank_rows, _rank_cells = compute_rank_based_summary(
        dataset_metric_raw=dataset_metric_reference,
        dataset_subset=train_datasets,
        alpha=GENERALIZATION_ALPHA,
    )
    value_rows = compute_value_based_summary(
        dataset_metric_raw=dataset_metric_reference,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=train_datasets,
        weighting_mode="equal_dataset",
        alpha=GENERALIZATION_ALPHA,
    )

    rank_by = {str(row["measure_name"]): row for row in rank_rows}
    value_by = {str(row["measure_name"]): row for row in value_rows}

    candidate_names = _preserve_unique(
        [str(row["measure_name"]) for row in rank_rows[:top_k_rank]]
        + [str(row["measure_name"]) for row in value_rows[:top_k_value]]
        + [str(name) for name in fallback_terminals]
    )

    candidate_rows: List[Dict[str, Any]] = []
    for measure_name in candidate_names:
        rank_rank = float(rank_by.get(measure_name, {}).get("final_rank", np.inf))
        value_rank = float(value_by.get(measure_name, {}).get("final_rank", np.inf))
        selection_sources: List[str] = []
        if measure_name in rank_by and rank_rank <= float(top_k_rank):
            selection_sources.append("top_rank")
        if measure_name in value_by and value_rank <= float(top_k_value):
            selection_sources.append("top_value")
        if measure_name in fallback_terminals:
            selection_sources.append("fallback")

        candidate_rows.append(
            {
                "held_out_dataset": str(held_out_dataset),
                "measure_name": measure_name,
                "family": _terminal_family(measure_name, registry),
                "rank_based_final_rank": rank_rank,
                "value_based_final_rank": value_rank,
                "priority_score": float(rank_rank + value_rank),
                "selection_sources": "|".join(selection_sources),
                "curve_files_available": _curve_files_exist(measure_name, dataset_order),
            }
        )

    valid_rows = [row for row in candidate_rows if bool(row["curve_files_available"])]
    if not valid_rows:
        raise RuntimeError(
            f"No valid terminals available for held-out dataset {held_out_dataset}. "
            "Check corrected stage-03 and stage-04 outputs."
        )

    valid_rows.sort(
        key=lambda row: (
            float(row["priority_score"]),
            float(row["rank_based_final_rank"]),
            float(row["value_based_final_rank"]),
            str(row["measure_name"]),
        )
    )

    selected_rows: List[Dict[str, Any]] = []
    selected_names: List[str] = []
    family_counts: Dict[str, int] = {}

    for row in valid_rows:
        family = str(row["family"])
        if family in family_counts:
            continue
        updated = dict(row)
        updated["selection_stage"] = "family_coverage"
        selected_rows.append(updated)
        selected_names.append(str(updated["measure_name"]))
        family_counts[family] = 1

    for row in valid_rows:
        measure_name = str(row["measure_name"])
        if measure_name in selected_names:
            continue
        family = str(row["family"])
        if family_counts.get(family, 0) >= max_per_family:
            continue
        updated = dict(row)
        updated["selection_stage"] = "priority_fill"
        selected_rows.append(updated)
        selected_names.append(measure_name)
        family_counts[family] = family_counts.get(family, 0) + 1

    if len(selected_names) < 2:
        raise RuntimeError(
            f"Terminal selection produced too few valid terminals ({len(selected_names)}) "
            f"for held-out dataset {held_out_dataset}."
        )

    for order, row in enumerate(selected_rows, start=1):
        row["selection_order"] = order
        row["train_datasets"] = "|".join(train_datasets)

    return FoldTerminalSelection(
        held_out_dataset=str(held_out_dataset),
        train_datasets=train_datasets,
        selected_terminals=tuple(selected_names),
        selection_rows=tuple(selected_rows),
        rank_rows=tuple(dict(row) for row in rank_rows),
        value_rows=tuple(dict(row) for row in value_rows),
    )


def select_terminals_for_dataset_subset(
    *,
    selection_name: str,
    train_datasets: Sequence[str],
    dataset_metric_reference: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_stack_counts: Mapping[str, int],
    registry: Mapping[str, Mapping[str, Any]],
    top_k_rank: int,
    top_k_value: int,
    fallback_terminals: Sequence[str] = DEFAULT_GP_FALLBACK_TERMINALS,
    dataset_order: Sequence[str] = DATASET_ORDER,
    max_per_family: int = 2,
) -> FoldTerminalSelection:
    selected_train_datasets = tuple(str(name) for name in train_datasets)
    if not selected_train_datasets:
        raise ValueError("Terminal selection requires at least one training dataset")

    rank_rows, _rank_cells = compute_rank_based_summary(
        dataset_metric_raw=dataset_metric_reference,
        dataset_subset=selected_train_datasets,
        alpha=GENERALIZATION_ALPHA,
    )
    value_rows = compute_value_based_summary(
        dataset_metric_raw=dataset_metric_reference,
        dataset_stack_counts=dataset_stack_counts,
        dataset_subset=selected_train_datasets,
        weighting_mode="equal_dataset",
        alpha=GENERALIZATION_ALPHA,
    )

    rank_by = {str(row["measure_name"]): row for row in rank_rows}
    value_by = {str(row["measure_name"]): row for row in value_rows}

    candidate_names = _preserve_unique(
        [str(row["measure_name"]) for row in rank_rows[:top_k_rank]]
        + [str(row["measure_name"]) for row in value_rows[:top_k_value]]
        + [str(name) for name in fallback_terminals]
    )

    candidate_rows: List[Dict[str, Any]] = []
    for measure_name in candidate_names:
        rank_rank = float(rank_by.get(measure_name, {}).get("final_rank", np.inf))
        value_rank = float(value_by.get(measure_name, {}).get("final_rank", np.inf))
        selection_sources: List[str] = []
        if measure_name in rank_by and rank_rank <= float(top_k_rank):
            selection_sources.append("top_rank")
        if measure_name in value_by and value_rank <= float(top_k_value):
            selection_sources.append("top_value")
        if measure_name in fallback_terminals:
            selection_sources.append("fallback")

        candidate_rows.append(
            {
                "held_out_dataset": str(selection_name),
                "measure_name": measure_name,
                "family": _terminal_family(measure_name, registry),
                "rank_based_final_rank": rank_rank,
                "value_based_final_rank": value_rank,
                "priority_score": float(rank_rank + value_rank),
                "selection_sources": "|".join(selection_sources),
                "curve_files_available": _curve_files_exist(measure_name, dataset_order),
            }
        )

    valid_rows = [row for row in candidate_rows if bool(row["curve_files_available"])]
    if not valid_rows:
        raise RuntimeError(
            f"No valid terminals available for selection {selection_name}. "
            "Check corrected stage-03 and stage-04 outputs."
        )

    valid_rows.sort(
        key=lambda row: (
            float(row["priority_score"]),
            float(row["rank_based_final_rank"]),
            float(row["value_based_final_rank"]),
            str(row["measure_name"]),
        )
    )

    selected_rows: List[Dict[str, Any]] = []
    selected_names: List[str] = []
    family_counts: Dict[str, int] = {}

    for row in valid_rows:
        family = str(row["family"])
        if family in family_counts:
            continue
        updated = dict(row)
        updated["selection_stage"] = "family_coverage"
        selected_rows.append(updated)
        selected_names.append(str(updated["measure_name"]))
        family_counts[family] = 1

    for row in valid_rows:
        measure_name = str(row["measure_name"])
        if measure_name in selected_names:
            continue
        family = str(row["family"])
        if family_counts.get(family, 0) >= max_per_family:
            continue
        updated = dict(row)
        updated["selection_stage"] = "priority_fill"
        selected_rows.append(updated)
        selected_names.append(measure_name)
        family_counts[family] = family_counts.get(family, 0) + 1

    if len(selected_names) < 2:
        raise RuntimeError(
            f"Terminal selection produced too few valid terminals ({len(selected_names)}) "
            f"for selection {selection_name}."
        )

    for order, row in enumerate(selected_rows, start=1):
        row["selection_order"] = order
        row["train_datasets"] = "|".join(selected_train_datasets)

    return FoldTerminalSelection(
        held_out_dataset=str(selection_name),
        train_datasets=selected_train_datasets,
        selected_terminals=tuple(selected_names),
        selection_rows=tuple(selected_rows),
        rank_rows=tuple(dict(row) for row in rank_rows),
        value_rows=tuple(dict(row) for row in value_rows),
    )


__all__ = [
    "FoldTerminalSelection",
    "load_dataset_metric_reference",
    "load_dataset_stack_counts",
    "select_terminals_for_dataset_subset",
    "select_terminals_for_fold",
]
