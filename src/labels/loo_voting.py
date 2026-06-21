"""Leave-one-out surrogate voting helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from config.settings import (
    CRITICAL_MEASURE_NAMES,
    DEFAULT_SURROGATE_VOTERS,
    USE_LEAVE_ONE_OUT_SURROGATE_VOTING,
    VOTING_TIE_BREAK_POLICY,
)
from src.labels.surrogate_labels import build_majority_vote_labels_from_predictions
from src.measures.focus_measure_library import canonicalize_measure_name


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def resolve_loo_measure_names(
    registry: Mapping[str, Mapping[str, object]],
    *,
    extra_measures: Sequence[str] | None = None,
) -> List[str]:
    names = list(DEFAULT_SURROGATE_VOTERS)
    names.extend(CRITICAL_MEASURE_NAMES)
    names.extend(registry.keys())
    if extra_measures:
        names.extend(extra_measures)
    return sorted(_unique_preserve_order(names))


def voter_pool_for_target_measure(
    target_measure: str,
    *,
    default_voters: Sequence[str] = DEFAULT_SURROGATE_VOTERS,
    use_leave_one_out: bool = USE_LEAVE_ONE_OUT_SURROGATE_VOTING,
) -> List[str]:
    if not use_leave_one_out:
        return list(default_voters)

    target_canonical = canonicalize_measure_name(target_measure)
    pool: List[str] = []
    for voter in default_voters:
        if canonicalize_measure_name(voter) == target_canonical:
            continue
        pool.append(voter)
    return pool


def build_loo_label_sets(
    *,
    voter_predictions: Mapping[str, np.ndarray],
    target_measure_names: Sequence[str],
    default_voters: Sequence[str] = DEFAULT_SURROGATE_VOTERS,
    use_leave_one_out: bool = USE_LEAVE_ONE_OUT_SURROGATE_VOTING,
    tie_break_policy: str = VOTING_TIE_BREAK_POLICY,
) -> Dict[str, np.ndarray]:
    loo_labels: Dict[str, np.ndarray] = {}
    for target_measure in target_measure_names:
        voter_pool = voter_pool_for_target_measure(
            target_measure,
            default_voters=default_voters,
            use_leave_one_out=use_leave_one_out,
        )
        if not voter_pool:
            continue
        loo_labels[target_measure] = build_majority_vote_labels_from_predictions(
            predictions=voter_predictions,
            voter_names=voter_pool,
            tie_break_policy=tie_break_policy,
        )
    return loo_labels


__all__ = [
    "resolve_loo_measure_names",
    "voter_pool_for_target_measure",
    "build_loo_label_sets",
]
