"""LODO fold helpers for corrected composite GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from config.paths import DATASET_ORDER
from src.gp.deap_search import load_composite_labels, load_terminal_curves_for_dataset


@dataclass(frozen=True)
class OuterFold:
    outer_fold: int
    held_out_dataset: str
    train_datasets: Tuple[str, ...]


def build_outer_folds(dataset_order: Sequence[str] = DATASET_ORDER) -> List[OuterFold]:
    ordered = [str(dataset_name) for dataset_name in dataset_order]
    folds: List[OuterFold] = []
    for fold_index, held_out_dataset in enumerate(ordered):
        train_datasets = tuple(dataset_name for dataset_name in ordered if dataset_name != held_out_dataset)
        folds.append(
            OuterFold(
                outer_fold=int(fold_index),
                held_out_dataset=str(held_out_dataset),
                train_datasets=train_datasets,
            )
        )
    return folds


def load_all_fold_data(
    terminal_names: Sequence[str],
    dataset_order: Sequence[str] = DATASET_ORDER,
) -> Tuple[Dict[str, Dict[str, List[np.ndarray]]], Dict[str, np.ndarray], Dict[str, str]]:
    curves_by_dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
    labels_by_dataset: Dict[str, np.ndarray] = {}
    label_modes: Dict[str, str] = {}

    for dataset_name in dataset_order:
        curves_by_dataset[str(dataset_name)] = load_terminal_curves_for_dataset(str(dataset_name), terminal_names)
        labels, label_mode = load_composite_labels(str(dataset_name))
        labels_by_dataset[str(dataset_name)] = labels
        label_modes[str(dataset_name)] = str(label_mode)

    return curves_by_dataset, labels_by_dataset, label_modes


def summarize_fold_results(fold: OuterFold, seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not seed_results:
        raise ValueError(f"No seed results available for fold {fold.held_out_dataset}")

    heldout_scores = np.asarray([float(row["heldout_score"]) for row in seed_results], dtype=np.float64)
    best_result = min(seed_results, key=lambda row: float(row["heldout_score"]))
    return {
        "outer_fold": int(fold.outer_fold),
        "held_out_dataset": str(fold.held_out_dataset),
        "train_datasets": "|".join(fold.train_datasets),
        "num_seeds": int(len(seed_results)),
        "mean_heldout_score": float(np.mean(heldout_scores)),
        "std_heldout_score": float(np.std(heldout_scores, ddof=0)),
        "best_heldout_score": float(best_result["heldout_score"]),
        "best_expression": str(best_result["best_expression"]),
        "best_seed": int(best_result["seed"]),
        "selected_terminal_count": int(len(best_result["terminals"])),
    }


__all__ = [
    "OuterFold",
    "build_outer_folds",
    "load_all_fold_data",
    "summarize_fold_results",
]
