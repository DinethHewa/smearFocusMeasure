"""Baseline and comparison helpers for corrected composite evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def assign_composite_ids(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        record = dict(candidate)
        record["composite_id"] = f"CFM{index}"
        out.append(record)
    return out


def build_best_composite_vs_best_single_rows(
    *,
    composite_summary_rows: Sequence[Mapping[str, Any]],
    common_value_rows: Sequence[Mapping[str, Any]],
    single_entities: Sequence[str],
    single_rank_rows: Sequence[Mapping[str, Any]],
    single_value_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not composite_summary_rows:
        return []

    best_composite = dict(composite_summary_rows[0])
    single_common_rows = sorted(
        [dict(row) for row in common_value_rows if str(row["entity_name"]) in set(single_entities)],
        key=lambda row: float(row["generalization_score"]),
    )
    best_single_common = single_common_rows[0] if single_common_rows else None

    single_rank_sorted = sorted(single_rank_rows, key=lambda row: float(row["final_rank"]))
    single_value_sorted = sorted(single_value_rows, key=lambda row: float(row["final_rank"]))

    rows: List[Dict[str, Any]] = [
        {
            "comparison_item": "best_composite_under_common_value_scoring",
            "name": str(best_composite["composite_id"]),
            "expression": str(best_composite["expression"]),
            "score": float(best_composite["common_value_generalization_score"]),
            "final_rank": int(best_composite["common_value_final_rank"]),
        }
    ]
    if best_single_common is not None:
        rows.append(
            {
                "comparison_item": "best_single_under_common_value_scoring",
                "name": str(best_single_common["entity_name"]),
                "expression": "",
                "score": float(best_single_common["generalization_score"]),
                "final_rank": int(best_single_common["final_rank"]),
            }
        )
    if single_rank_sorted:
        rows.append(
            {
                "comparison_item": "best_single_under_original_rank_based_analysis",
                "name": str(single_rank_sorted[0]["measure_name"]),
                "expression": "",
                "score": float(single_rank_sorted[0]["rank_generalization_score"]),
                "final_rank": int(float(single_rank_sorted[0]["final_rank"])),
            }
        )
    if single_value_sorted:
        rows.append(
            {
                "comparison_item": "best_single_under_original_value_based_analysis",
                "name": str(single_value_sorted[0]["measure_name"]),
                "expression": "",
                "score": float(single_value_sorted[0]["generalization_score"]),
                "final_rank": int(float(single_value_sorted[0]["final_rank"])),
            }
        )
    return rows


def build_within_admissible_spread_rows(
    composite_summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not composite_summary_rows:
        return []

    ordered = list(composite_summary_rows)
    best_score = float(ordered[0]["common_value_generalization_score"])
    worst_score = float(ordered[-1]["common_value_generalization_score"])
    return [
        {
            "composite_id": str(row["composite_id"]),
            "common_value_generalization_score": float(row["common_value_generalization_score"]),
            "delta_from_best": float(row["common_value_generalization_score"]) - best_score,
            "delta_from_worst": worst_score - float(row["common_value_generalization_score"]),
        }
        for row in ordered
    ]


__all__ = [
    "assign_composite_ids",
    "build_best_composite_vs_best_single_rows",
    "build_within_admissible_spread_rows",
]
