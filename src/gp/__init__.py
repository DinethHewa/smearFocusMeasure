"""Reusable GP and composite-search helpers."""

from src.gp.baselines import (
    assign_composite_ids,
    build_best_composite_vs_best_single_rows,
    build_within_admissible_spread_rows,
)
from src.gp.deap_search import (
    DEAP_AVAILABLE,
    GP_SEARCH_METRICS,
    build_pset,
    build_reference_bounds,
    build_toolbox,
    compile_expression,
    evaluate_expression_on_dataset,
    evaluate_expression_raw_metrics,
    load_all_terminal_curves,
    load_composite_labels,
    load_terminal_curves_for_dataset,
    load_timing_summary_map,
    require_deap,
    run_gp_seed,
)
from src.gp.deduplication import deduplicate_seed_results
from src.gp.lodo_runner import (
    OuterFold,
    build_outer_folds,
    load_all_fold_data,
    summarize_fold_results,
)
from src.gp.terminal_selection import (
    FoldTerminalSelection,
    load_dataset_metric_reference,
    load_dataset_stack_counts,
    select_terminals_for_fold,
)

__all__ = [
    "DEAP_AVAILABLE",
    "GP_SEARCH_METRICS",
    "OuterFold",
    "FoldTerminalSelection",
    "assign_composite_ids",
    "build_best_composite_vs_best_single_rows",
    "build_outer_folds",
    "build_pset",
    "build_reference_bounds",
    "build_toolbox",
    "build_within_admissible_spread_rows",
    "compile_expression",
    "deduplicate_seed_results",
    "evaluate_expression_on_dataset",
    "evaluate_expression_raw_metrics",
    "load_all_fold_data",
    "load_all_terminal_curves",
    "load_composite_labels",
    "load_dataset_metric_reference",
    "load_dataset_stack_counts",
    "load_terminal_curves_for_dataset",
    "load_timing_summary_map",
    "require_deap",
    "run_gp_seed",
    "select_terminals_for_fold",
    "summarize_fold_results",
]
