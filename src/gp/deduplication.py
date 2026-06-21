"""Expression and functional deduplication for composite GP outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from config.settings import FUNCTIONAL_EQUIVALENCE_CORRELATION_THRESHOLD


def _record_sort_key(record: Mapping[str, Any]) -> tuple[float, float, int]:
    return (
        float(record.get("heldout_score", float("inf"))),
        float(record.get("best_complexity", float("inf"))),
        int(record.get("seed", 10**9)),
    )


def _curve_correlation(curve_a: Sequence[float], curve_b: Sequence[float]) -> float:
    arr_a = np.asarray(curve_a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(curve_b, dtype=np.float64).reshape(-1)
    if arr_a.size == 0 or arr_b.size == 0 or arr_a.size != arr_b.size:
        return float("nan")
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        return float("nan")
    if np.std(arr_a) <= 0.0 or np.std(arr_b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def deduplicate_seed_results(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    correlation_threshold: float = FUNCTIONAL_EQUIVALENCE_CORRELATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    best_by_expression: Dict[str, Dict[str, Any]] = {}
    for record in seed_results:
        expression = str(record["best_expression"])
        current = best_by_expression.get(expression)
        if current is None or _record_sort_key(record) < _record_sort_key(current):
            best_by_expression[expression] = dict(record)

    expression_deduped = sorted(best_by_expression.values(), key=_record_sort_key)
    kept: List[Dict[str, Any]] = []

    for candidate in expression_deduped:
        candidate_curve = candidate.get("mean_heldout_curve", [])
        candidate_dataset = str(candidate.get("held_out_dataset", ""))
        equivalent_to = None
        for kept_record in kept:
            if str(kept_record.get("held_out_dataset", "")) != candidate_dataset:
                continue
            corr = _curve_correlation(candidate_curve, kept_record.get("mean_heldout_curve", []))
            if np.isfinite(corr) and corr >= float(correlation_threshold):
                equivalent_to = str(kept_record["best_expression"])
                break
        if equivalent_to is not None:
            continue
        kept.append(dict(candidate))

    return kept


__all__ = [
    "deduplicate_seed_results",
]
