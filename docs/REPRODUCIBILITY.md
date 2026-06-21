# Reproducibility guide

Run commands from the repository root.

## Environment

```bash
conda env create -f environment.yml
conda activate bspc-focus
```

## Full staged pipeline

```bash
python scripts/01_build_stacks.py --full-run
python scripts/02_build_reference_labels.py --full-run
python scripts/03_run_single_measure_benchmark.py --full-run
python scripts/04_evaluate_single_measures.py --full-run
python scripts/05_plot_single_measure_results.py --full-run
python scripts/06_run_composite_gp_lodo.py --full-run --final-refit-seeds 10
python scripts/07_evaluate_composites.py --full-run
python scripts/08_run_statistics_and_sensitivity.py --full-run
python scripts/09_optional_downstream_baseline.py --full-run
python scripts/10_export_paper_assets.py --full-run --strict
```

Equivalent orchestrated command:

```bash
python scripts/11_run_full_pipeline.py --full-run
```

## Checkpoints and resumption

The GP stage skips compatible completed seeds by default and resumes incomplete seeds from generation checkpoints. Do not use `--no-skip-completed` unless a deliberate full GP recomputation is required.

Resume the orchestrator from the first incomplete stage:

```bash
python scripts/11_run_full_pipeline.py --full-run --resume
```

Start from a specified stage:

```bash
python scripts/11_run_full_pipeline.py --full-run --from-stage 08_run_statistics_and_sensitivity.py
```

## Reviewer-response computations

The normalized curves and required aggregate metrics are included here:

```bash
python tools/review_response/run_review_response_computations.py
```

This command does not rerun LODO GP or the full pipeline.

## Production GP configuration

- Population: 500
- Generations: 100
- Seeds: 10 per held-out fold
- Crossover probability: 0.5
- Mutation probability: 0.2
- Tournament size: 3
- Elitism: 1
- Maximum tree depth: 10
- Maximum nodes: 35 where recorded after the run-control patch
- NSGA-II: enabled
- Production backend: CPU

Saved per-seed artifacts in the Zenodo package are the authoritative run audit.
