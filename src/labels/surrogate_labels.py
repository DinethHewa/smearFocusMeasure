"""Surrogate-label voting helpers built on focus-measure curves."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from config.settings import EXCLUDE_ENDPOINT_SLICES_IN_VOTING, VOTING_TIE_BREAK_POLICY


MeasureCallable = Callable[[np.ndarray], float]


def allowed_slice_indices(
    num_slices: int,
    *,
    exclude_endpoint_slices: bool = EXCLUDE_ENDPOINT_SLICES_IN_VOTING,
) -> List[int]:
    if exclude_endpoint_slices and num_slices >= 3:
        return list(range(1, num_slices - 1))
    return list(range(num_slices))


def tie_break_vote(indices: Sequence[int], *, policy: str = VOTING_TIE_BREAK_POLICY) -> int:
    ordered = sorted(int(value) for value in indices)
    if len(ordered) == 1:
        return ordered[0]
    if policy == "first":
        return ordered[0]
    if policy == "last":
        return ordered[-1]
    if policy == "central_index":
        return ordered[len(ordered) // 2]
    raise ValueError(f"Unsupported voting tie-break policy: {policy}")


def best_index_from_curve(
    curve: np.ndarray,
    *,
    maximize: bool = True,
    exclude_endpoint_slices: bool = EXCLUDE_ENDPOINT_SLICES_IN_VOTING,
    tie_break_policy: str = VOTING_TIE_BREAK_POLICY,
) -> int:
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    if curve.size == 0:
        raise ValueError("Cannot vote over an empty focus curve")
    if not np.isfinite(curve).all():
        raise ValueError("Focus curve contains non-finite values")

    valid_indices = allowed_slice_indices(
        len(curve),
        exclude_endpoint_slices=exclude_endpoint_slices,
    )
    valid_scores = curve[valid_indices]

    best_value = np.max(valid_scores) if maximize else np.min(valid_scores)
    tied = [
        valid_indices[idx]
        for idx, value in enumerate(valid_scores)
        if np.isclose(value, best_value)
    ]
    return tie_break_vote(tied, policy=tie_break_policy)


def compute_focus_curve_for_measure(
    stack: np.ndarray,
    measure_func: MeasureCallable,
) -> np.ndarray:
    stack_arr = np.asarray(stack)
    if stack_arr.ndim != 3:
        raise ValueError(f"Expected stack shape (num_slices, H, W), got {stack_arr.shape}")
    scores = [float(measure_func(slice_2d)) for slice_2d in stack_arr]
    curve = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(curve).all():
        raise ValueError("Focus curve contains NaN or Inf values")
    return curve


def _stack_iter(stacks: np.ndarray) -> Iterable[np.ndarray]:
    for stack in np.asarray(stacks, dtype=object):
        yield np.asarray(stack)


def compute_measure_peak_predictions(
    *,
    stacks: np.ndarray,
    measure_names: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
    dataset_name: str,
    logger=None,
    exclude_endpoint_slices: bool = EXCLUDE_ENDPOINT_SLICES_IN_VOTING,
    tie_break_policy: str = VOTING_TIE_BREAK_POLICY,
) -> Dict[str, np.ndarray]:
    missing = [name for name in measure_names if name not in registry]
    if missing:
        raise KeyError(f"[{dataset_name}] missing measures in registry: {missing}")

    stack_list = [np.asarray(stack) for stack in _stack_iter(stacks)]
    predictions: Dict[str, np.ndarray] = {
        name: np.zeros(len(stack_list), dtype=int) for name in measure_names
    }

    for stack_idx, stack in enumerate(stack_list):
        if stack.ndim != 3:
            raise ValueError(
                f"[{dataset_name}] expected per-stack array of shape (num_slices,H,W), got {stack.shape}"
            )

        for measure_name in measure_names:
            entry = registry[measure_name]
            curve = compute_focus_curve_for_measure(stack, entry["func"])
            predictions[measure_name][stack_idx] = best_index_from_curve(
                curve,
                maximize=bool(entry.get("maximize", True)),
                exclude_endpoint_slices=exclude_endpoint_slices,
                tie_break_policy=tie_break_policy,
            )

        if logger is not None and (stack_idx + 1) % 100 == 0:
            logger.info("[%s] computed surrogate voter peaks for %d stacks", dataset_name, stack_idx + 1)

    return predictions


def build_majority_vote_labels_from_predictions(
    *,
    predictions: Mapping[str, np.ndarray],
    voter_names: Sequence[str],
    tie_break_policy: str = VOTING_TIE_BREAK_POLICY,
) -> np.ndarray:
    if not voter_names:
        raise ValueError("No voter names supplied for surrogate voting")

    missing = [name for name in voter_names if name not in predictions]
    if missing:
        raise KeyError(f"Missing voter predictions: {missing}")

    num_stacks = len(np.asarray(predictions[voter_names[0]]).reshape(-1))
    labels = np.zeros(num_stacks, dtype=int)

    for stack_idx in range(num_stacks):
        votes = [int(np.asarray(predictions[name]).reshape(-1)[stack_idx]) for name in voter_names]
        counts = Counter(votes)
        max_count = max(counts.values())
        tied = [idx for idx, count in counts.items() if count == max_count]
        labels[stack_idx] = tie_break_vote(tied, policy=tie_break_policy)

    return labels


def build_surrogate_labels(
    *,
    stacks: np.ndarray,
    voter_names: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
    dataset_name: str,
    logger=None,
    exclude_endpoint_slices: bool = EXCLUDE_ENDPOINT_SLICES_IN_VOTING,
    tie_break_policy: str = VOTING_TIE_BREAK_POLICY,
) -> np.ndarray:
    predictions = compute_measure_peak_predictions(
        stacks=stacks,
        measure_names=voter_names,
        registry=registry,
        dataset_name=dataset_name,
        logger=logger,
        exclude_endpoint_slices=exclude_endpoint_slices,
        tie_break_policy=tie_break_policy,
    )
    return build_majority_vote_labels_from_predictions(
        predictions=predictions,
        voter_names=voter_names,
        tie_break_policy=tie_break_policy,
    )


__all__ = [
    "MeasureCallable",
    "allowed_slice_indices",
    "tie_break_vote",
    "best_index_from_curve",
    "compute_focus_curve_for_measure",
    "compute_measure_peak_predictions",
    "build_majority_vote_labels_from_predictions",
    "build_surrogate_labels",
]
