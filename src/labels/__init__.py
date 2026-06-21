"""Source and surrogate label helpers."""

from src.labels.loo_voting import build_loo_label_sets, resolve_loo_measure_names, voter_pool_for_target_measure
from src.labels.source_labels import SourceLabelDiscovery, discover_source_labels, find_candidate_source_label_files, load_label_file
from src.labels.surrogate_labels import (
    MeasureCallable,
    allowed_slice_indices,
    best_index_from_curve,
    build_majority_vote_labels_from_predictions,
    build_surrogate_labels,
    compute_focus_curve_for_measure,
    compute_measure_peak_predictions,
    tie_break_vote,
)

__all__ = [
    "SourceLabelDiscovery",
    "MeasureCallable",
    "find_candidate_source_label_files",
    "load_label_file",
    "discover_source_labels",
    "allowed_slice_indices",
    "tie_break_vote",
    "best_index_from_curve",
    "compute_focus_curve_for_measure",
    "compute_measure_peak_predictions",
    "build_majority_vote_labels_from_predictions",
    "build_surrogate_labels",
    "resolve_loo_measure_names",
    "voter_pool_for_target_measure",
    "build_loo_label_sets",
]
