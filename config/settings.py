# config/settings.py

from __future__ import annotations

from typing import Dict, Optional, Tuple
import math

from config.paths import DATASET_ORDER

# -----------------------------------------------------------------------------
# Global reproducibility
# -----------------------------------------------------------------------------
GLOBAL_SEED: int = 42
NUMPY_SEED: int = 42
PYTHON_RANDOM_SEED: int = 42

# -----------------------------------------------------------------------------
# Pipeline modes
# -----------------------------------------------------------------------------
VALID_RUN_MODES: Tuple[str, ...] = ("smoke", "full")
DEFAULT_RUN_MODE: str = "smoke"

# -----------------------------------------------------------------------------
# Data / preprocessing behavior
# -----------------------------------------------------------------------------
PRESERVE_NATIVE_RESOLUTION: bool = True
ALLOW_GLOBAL_RESIZE_IN_MAIN_BENCHMARK: bool = False
MAIN_BENCHMARK_RESIZE_TARGET: Optional[int] = None  # None => native resolution

# Used for timing/sensitivity only, not for the main benchmark
TIMING_RESOLUTIONS: Tuple[int, ...] = (128, 512, 1024)
INCLUDE_NATIVE_IN_TIMING: bool = True

CONVERT_TO_GRAYSCALE_WHEN_NEEDED: bool = True
GRAYSCALE_MODE: str = "luminance"  # supported: "luminance"

# -----------------------------------------------------------------------------
# ROI behavior
# -----------------------------------------------------------------------------
USE_ROI_CROPPING: bool = False
ROI_MODE: str = "center"  # supported: "center", "bbox", "none"
ROI_SIZE: Optional[Tuple[int, int]] = None  # e.g. (512, 512)
REQUIRE_ROI_SAME_SIZE: bool = False

# -----------------------------------------------------------------------------
# Numerical safety
# -----------------------------------------------------------------------------
EPS: float = 1e-12

# -----------------------------------------------------------------------------
# Dataset-level behavior
# -----------------------------------------------------------------------------
# Leave these as "auto" unless you know with certainty which datasets contain
# source-provided labels. The label-building script should resolve and save
# the final label manifest.
DATASET_LABEL_MODE: Dict[str, str] = {
    "WBC": "auto",
    "TBI": "auto",
    "PBS": "auto",
    "BMA": "auto",
    "TBF": "auto",
}

PRIMARY_DATASET_WEIGHTING: str = "equal_dataset"
SECONDARY_DATASET_WEIGHTING: str = "per_stack"

# -----------------------------------------------------------------------------
# Surrogate label voting
# -----------------------------------------------------------------------------
DEFAULT_SURROGATE_VOTERS: Tuple[str, ...] = (
    "Tenengrad",
    "Brenner Gradient",
    "Variance of Laplacian",
    "Sum Modified Laplacian",
    "Normalized Variance",
    "Energy of Gradient",
    "Histogram Entropy",
    "GLCM Contrast",
    "Variance of Gradient",
    "Fourier Transform Sharpness Index",
)

USE_LEAVE_ONE_OUT_SURROGATE_VOTING: bool = True
EXCLUDE_ENDPOINT_SLICES_IN_VOTING: bool = True
VOTING_TIE_BREAK_POLICY: str = "central_index"  # supported: "central_index", "first", "last"

# -----------------------------------------------------------------------------
# Focus measure library
# -----------------------------------------------------------------------------
EXPECTED_NUM_FOCUS_MEASURES: int = 50

# Critical subset commonly discussed in the paper.
CRITICAL_MEASURE_NAMES: Tuple[str, ...] = (
    "Brenner Gradient",
    "GLCM Contrast",
    "Fourier High Frequency Energy Ratio",
    "Variance of Laplacian",
    "Sum Modified Laplacian",
    "Tenengrad",
    "Roberts Focus Measure",
    "Wavelet W1",
    "Wavelet W2",
    "Wavelet W3",
    "Curvelet Transform Sharpness Index",
    "DCT Focus Measure",
)

# -----------------------------------------------------------------------------
# Curve normalization
# -----------------------------------------------------------------------------
NORMALIZE_FOCUS_CURVES_PER_STACK: bool = True
NORMALIZATION_METHOD: str = "minmax_per_stack"


def normalize_formula_description() -> str:
    return "F_norm = (F - min(F)) / (max(F) - min(F) + eps)"


# -----------------------------------------------------------------------------
# Benchmark metrics
# -----------------------------------------------------------------------------
AUTOFOCUS_METRICS: Tuple[str, ...] = (
    "absolute_peak_localization_error",
    "fwhm",
    "curvature_at_peak",
    "steep_slope_width",
    "steep_to_gradual_slope_ratio",
    "false_maxima_count",
    "noise_level",
    "rrmse_under_additive_noise",
    "range_around_global_maximum",
    "execution_time_per_slice",
)

GENERALIZATION_ALPHA: float = 0.7
ALPHA_SENSITIVITY_VALUES: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)

# Metric weights used in the manuscript
METRIC_WEIGHTS: Dict[str, float] = {
    "absolute_peak_localization_error": 0.20,
    "range_around_global_maximum": 0.15,
    "false_maxima_count": 0.10,
    "fwhm": 0.10,
    "noise_level": 0.10,
    "steep_to_gradual_slope_ratio": 0.10,
    "execution_time_per_slice": 0.10,
    "steep_slope_width": 0.05,
    "curvature_at_peak": 0.05,
    "rrmse_under_additive_noise": 0.05,
}

ALT_METRIC_WEIGHT_SCHEMES: Dict[str, Dict[str, float]] = {
    "paper_default": dict(METRIC_WEIGHTS),
    "accuracy_heavy": {
        "absolute_peak_localization_error": 0.35,
        "range_around_global_maximum": 0.10,
        "false_maxima_count": 0.10,
        "fwhm": 0.10,
        "noise_level": 0.08,
        "steep_to_gradual_slope_ratio": 0.07,
        "execution_time_per_slice": 0.08,
        "steep_slope_width": 0.04,
        "curvature_at_peak": 0.04,
        "rrmse_under_additive_noise": 0.04,
    },
}

# -----------------------------------------------------------------------------
# Noise robustness settings
# -----------------------------------------------------------------------------
NOISE_STD_FOR_RRMSE: float = 0.01
NOISE_RANDOM_SEED: int = 123

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
USE_FRIEDMAN_TEST: bool = True
USE_NEMENYI_POSTHOC: bool = True
BOOTSTRAP_NUM_RESAMPLES: int = 2000
BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95

# -----------------------------------------------------------------------------
# Expression equivalence / deduplication
# -----------------------------------------------------------------------------
FUNCTIONAL_EQUIVALENCE_CORRELATION_THRESHOLD: float = 0.99

# -----------------------------------------------------------------------------
# GP settings
# -----------------------------------------------------------------------------
USE_LODO_GP: bool = True  # leave-one-dataset-out
GP_PRIMARY_OBJECTIVE: str = "corrected_generalization_score"
GP_SECONDARY_OBJECTIVE: str = "expression_complexity"

# Stronger than the legacy pipeline, but still practical
GP_FULL_SETTINGS: Dict[str, object] = {
    "population_size": 500,
    "num_generations": 100,
    "num_seeds": 10,
    "crossover_probability": 0.5,
    "mutation_probability": 0.2,
    "tournament_size": 3,
    "elitism": 1,
    "max_tree_depth": 10,
    "use_nsga2": True,
}

# Fast sanity-check profile
GP_SMOKE_SETTINGS: Dict[str, object] = {
    "population_size": 60,
    "num_generations": 10,
    "num_seeds": 1,
    "crossover_probability": 0.5,
    "mutation_probability": 0.2,
    "tournament_size": 3,
    "elitism": 1,
    "max_tree_depth": 6,
    "use_nsga2": True,
}

# Fallback terminal set only. The corrected pipeline should derive terminals
# from corrected single-measure benchmark outputs.
DEFAULT_GP_FALLBACK_TERMINALS: Tuple[str, ...] = (
    "GLCM Contrast",
    "Intensity Skewness Index",
    "Fourier Transform Sharpness Index",
    "Brenner Gradient",
    "Fourier High Frequency Energy Ratio",
    "Wavelet W1",
    "Wavelet W2",
    "Wavelet W3",
    "Curvelet Transform Sharpness Index",
    "Sum Modified Laplacian",
    "Squared Gradient",
    "Gradient Squared Energy",
    "Roberts Focus Measure",
)

# -----------------------------------------------------------------------------
# Optional baseline / downstream stages
# -----------------------------------------------------------------------------
RUN_OPTIONAL_LEARNING_BASELINE: bool = False
LEARNING_BASELINE_NAME: str = "small_cnn_regression"
LEARNING_BASELINE_TRAIN_ON_4_TEST_ON_1: bool = True

RUN_OPTIONAL_DOWNSTREAM_EXPERIMENT: bool = False

# -----------------------------------------------------------------------------
# Plotting / export behavior
# -----------------------------------------------------------------------------
FIGURE_DPI: int = 300
SAVE_FIGURES_AS_PNG: bool = True
SAVE_FIGURES_AS_PDF: bool = True
DEFAULT_MAIN_FIGURE_EXTENSIONS: Tuple[str, ...] = (".png", ".pdf")
DEFAULT_SUPP_FIGURE_EXTENSIONS: Tuple[str, ...] = (".png", ".pdf")

EXPORT_LATEX_TABLES: bool = True
EXPORT_CSV_TABLES: bool = True

NUM_REPRESENTATIVE_STACKS_PER_DATASET: int = 1
PLOT_TOP_SINGLE_MEASURES: Tuple[str, ...] = (
    "Brenner Gradient",
    "GLCM Contrast",
)
PLOT_INCLUDE_BEST_COMPOSITE: bool = True

# Publication-critical stability rerun
PUBLICATION_RANK_STABILITY_TARGET_RESOLUTION: int = 1024
PUBLICATION_RANK_STABILITY_TOP_K: int = 10
PUBLICATION_RANK_STABILITY_SMOKE_MAX_STACKS_PER_DATASET: int = 2
PUBLICATION_RANK_STABILITY_FULL_MAX_STACKS_PER_DATASET: int = 5

# -----------------------------------------------------------------------------
# Final paper asset policy
# -----------------------------------------------------------------------------
EXPORT_ONLY_APPROVED_ASSETS: bool = True
PAPER_MAIN_TABLE_LIMIT_TOP_K: int = 10
PAPER_SUPP_INCLUDE_FULL_OPERATOR_LIST: bool = True

# -----------------------------------------------------------------------------
# Run profiles
# -----------------------------------------------------------------------------
RUN_PROFILE: Dict[str, Dict[str, object]] = {
    "smoke": {
        "max_stacks_per_dataset": 5,
        "run_gp": False,
        "run_downstream": False,
        "run_learning_baseline": False,
        "export_paper_assets": False,
        "gp_settings": GP_SMOKE_SETTINGS,
    },
    "full": {
        "max_stacks_per_dataset": None,
        "run_gp": True,
        "run_downstream": RUN_OPTIONAL_DOWNSTREAM_EXPERIMENT,
        "run_learning_baseline": RUN_OPTIONAL_LEARNING_BASELINE,
        "export_paper_assets": True,
        "gp_settings": GP_FULL_SETTINGS,
    },
}


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def validate_metric_weights(weights: Dict[str, float]) -> None:
    missing = set(AUTOFOCUS_METRICS) - set(weights.keys())
    extra = set(weights.keys()) - set(AUTOFOCUS_METRICS)

    if missing:
        raise ValueError(f"Metric weights missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Metric weights contain unknown keys: {sorted(extra)}")

    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Metric weights must sum to 1.0, got {total}")


def validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"GENERALIZATION_ALPHA must be in [0, 1], got {alpha}")


def validate_run_mode(mode: str) -> None:
    if mode not in VALID_RUN_MODES:
        raise ValueError(f"Invalid run mode '{mode}'. Valid options: {VALID_RUN_MODES}")


def get_run_profile(mode: str = DEFAULT_RUN_MODE) -> Dict[str, object]:
    validate_run_mode(mode)
    return RUN_PROFILE[mode]


def validate_all_settings() -> None:
    validate_alpha(GENERALIZATION_ALPHA)
    validate_metric_weights(METRIC_WEIGHTS)

    for scheme_name, scheme_weights in ALT_METRIC_WEIGHT_SCHEMES.items():
        try:
            validate_metric_weights(scheme_weights)
        except ValueError as exc:
            raise ValueError(f"Invalid metric-weight scheme '{scheme_name}': {exc}") from exc

    for dataset_name in DATASET_ORDER:
        if dataset_name not in DATASET_LABEL_MODE:
            raise ValueError(f"Missing DATASET_LABEL_MODE entry for dataset '{dataset_name}'")

    if not PRESERVE_NATIVE_RESOLUTION and MAIN_BENCHMARK_RESIZE_TARGET is None:
        raise ValueError(
            "If PRESERVE_NATIVE_RESOLUTION is False, MAIN_BENCHMARK_RESIZE_TARGET must be set."
        )

    if NORMALIZATION_METHOD != "minmax_per_stack":
        raise ValueError(
            "Corrected paper pipeline expects NORMALIZATION_METHOD='minmax_per_stack'."
        )


__all__ = [
    "GLOBAL_SEED",
    "NUMPY_SEED",
    "PYTHON_RANDOM_SEED",
    "VALID_RUN_MODES",
    "DEFAULT_RUN_MODE",
    "PRESERVE_NATIVE_RESOLUTION",
    "ALLOW_GLOBAL_RESIZE_IN_MAIN_BENCHMARK",
    "MAIN_BENCHMARK_RESIZE_TARGET",
    "TIMING_RESOLUTIONS",
    "INCLUDE_NATIVE_IN_TIMING",
    "CONVERT_TO_GRAYSCALE_WHEN_NEEDED",
    "GRAYSCALE_MODE",
    "USE_ROI_CROPPING",
    "ROI_MODE",
    "ROI_SIZE",
    "REQUIRE_ROI_SAME_SIZE",
    "EPS",
    "DATASET_LABEL_MODE",
    "PRIMARY_DATASET_WEIGHTING",
    "SECONDARY_DATASET_WEIGHTING",
    "DEFAULT_SURROGATE_VOTERS",
    "USE_LEAVE_ONE_OUT_SURROGATE_VOTING",
    "EXCLUDE_ENDPOINT_SLICES_IN_VOTING",
    "VOTING_TIE_BREAK_POLICY",
    "EXPECTED_NUM_FOCUS_MEASURES",
    "CRITICAL_MEASURE_NAMES",
    "NORMALIZE_FOCUS_CURVES_PER_STACK",
    "NORMALIZATION_METHOD",
    "AUTOFOCUS_METRICS",
    "GENERALIZATION_ALPHA",
    "ALPHA_SENSITIVITY_VALUES",
    "METRIC_WEIGHTS",
    "ALT_METRIC_WEIGHT_SCHEMES",
    "NOISE_STD_FOR_RRMSE",
    "NOISE_RANDOM_SEED",
    "USE_FRIEDMAN_TEST",
    "USE_NEMENYI_POSTHOC",
    "BOOTSTRAP_NUM_RESAMPLES",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "FUNCTIONAL_EQUIVALENCE_CORRELATION_THRESHOLD",
    "USE_LODO_GP",
    "GP_PRIMARY_OBJECTIVE",
    "GP_SECONDARY_OBJECTIVE",
    "GP_FULL_SETTINGS",
    "GP_SMOKE_SETTINGS",
    "DEFAULT_GP_FALLBACK_TERMINALS",
    "RUN_OPTIONAL_LEARNING_BASELINE",
    "LEARNING_BASELINE_NAME",
    "LEARNING_BASELINE_TRAIN_ON_4_TEST_ON_1",
    "RUN_OPTIONAL_DOWNSTREAM_EXPERIMENT",
    "FIGURE_DPI",
    "SAVE_FIGURES_AS_PNG",
    "SAVE_FIGURES_AS_PDF",
    "DEFAULT_MAIN_FIGURE_EXTENSIONS",
    "DEFAULT_SUPP_FIGURE_EXTENSIONS",
    "EXPORT_LATEX_TABLES",
    "EXPORT_CSV_TABLES",
    "NUM_REPRESENTATIVE_STACKS_PER_DATASET",
    "PLOT_TOP_SINGLE_MEASURES",
    "PLOT_INCLUDE_BEST_COMPOSITE",
    "PUBLICATION_RANK_STABILITY_TARGET_RESOLUTION",
    "PUBLICATION_RANK_STABILITY_TOP_K",
    "PUBLICATION_RANK_STABILITY_SMOKE_MAX_STACKS_PER_DATASET",
    "PUBLICATION_RANK_STABILITY_FULL_MAX_STACKS_PER_DATASET",
    "EXPORT_ONLY_APPROVED_ASSETS",
    "PAPER_MAIN_TABLE_LIMIT_TOP_K",
    "PAPER_SUPP_INCLUDE_FULL_OPERATOR_LIST",
    "RUN_PROFILE",
    "normalize_formula_description",
    "validate_metric_weights",
    "validate_alpha",
    "validate_run_mode",
    "get_run_profile",
    "validate_all_settings",
]
