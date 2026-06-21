# BSPC Smear Microscopy Autofocus Benchmark

Reproducibility repository for:

**Weakly Supervised Cross-Dataset Benchmarking and Interpretable Symbolic Fusion of Focus Measures for Smear Microscopy Autofocus**

This repository contains the complete analysis code, normalized focus curves, aggregate results, publication assets, and reviewer-response computations. Raw microscopy images and the 46 GB derived stack cache are not redistributed.

## Study scope

- Five smear-microscopy datasets: WBC, TBI, PBS, BMA, and TBF.
- Thirty-two implemented single focus measures.
- Ten autofocus evaluation metrics.
- Leave-one-dataset-out genetic programming with ten seeds per fold.
- Ten-seed final all-dataset refit.
- Fourteen retained composite candidates evaluated under common scoring.
- Gradient-free voter, CFM4, runtime-weight, and GP-audit reviewer analyses.

## Repository structure

```text
config/                 Experiment settings, paths, weights, and asset registry
scripts/                Pipeline stages 00-11
src/                    Measures, labels, evaluation, GP, plotting, and utilities
outputs/
  01_stacks/metadata/   Stack metadata only; image arrays are excluded
  02_reference_labels/  Source/surrogate labels and provenance manifests
  03_single_measure_curves/
    normalized/         Saved normalized curves for all 32 measures
    timing/             Saved timing records
  04_single_measure_eval/
  05_gp_runs/           Compact GP summaries and deduplicated expressions
  06_composite_eval/
  07_statistics/
  09_paper/             Final tables, figures, captions, and manifests
  10_review_response_computations/
docs/                   Reproduction, data, output, and release instructions
```

The corresponding Zenodo archive contains raw focus curves and complete per-seed GP artifacts in addition to the files provided here.

## Installation

Recommended:

```bash
conda env create -f environment.yml
conda activate bspc-focus
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA/CuPy is optional. Production GP results in the archive used the CPU backend.

## Dataset configuration

Raw datasets are not included. Before rebuilding stacks, set `FOCUS_DATA_ROOT` to a directory containing `WBC/`, `TBI/`, `PBS/`, `BMA/`, and `TBF/`, or set the five per-dataset variables shown in `.env.example`. See [docs/DATASETS.md](docs/DATASETS.md).

Do not publish or redistribute source images unless their licenses and participant/privacy conditions permit it.

## Reproducing the method

After configuring authorized raw datasets, run a smoke test:

```bash
python scripts/11_run_full_pipeline.py --smoke-test --force
```

Full pipeline:

```bash
python scripts/11_run_full_pipeline.py --full-run --skip-downstream
```

The production GP stage is computationally expensive. Existing results are included so inspection and downstream reporting do not require rerunning GP.

Reviewer-response computations from saved normalized curves:

```bash
python tools/review_response/run_review_response_computations.py
```

Validate the release structure and included derived outputs:

```bash
python tools/validate_release.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for staged commands and checkpoint behavior.

## Important interpretation notes

- Surrogate labels are consensus labels, not optical ground truth.
- `Curvelet Transform Sharpness Index` is a legacy internal name mapped to Wavelet Detail Energy (db1); it is not an independent curvelet implementation.
- CFM4 is rank-competitive but highly curve-similar to FTSI; its contribution should not be described as value-dominant.
- The optional downstream analysis is a focus-quality proxy when true class labels are unavailable, not diagnostic validation.
- Single-only and union value scores use different min-max normalization pools and are not numerically interchangeable.

## Citation

Use `CITATION.cff`. Update the author details, GitHub URL, and Zenodo DOI before the public release.

## Licenses

- Source code: MIT License, see `LICENSE`.
- Repository-authored tables, figures, and documentation: CC BY 4.0, see `OUTPUTS_LICENSE.md`.
- No rights are granted to source microscopy datasets, which are not included.

## Release status

This folder is prepared for GitHub but has not been pushed. Complete [docs/GITHUB_RELEASE_CHECKLIST.md](docs/GITHUB_RELEASE_CHECKLIST.md) before publication.
