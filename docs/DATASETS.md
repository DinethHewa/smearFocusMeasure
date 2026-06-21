# Dataset configuration and availability

## Dataset order

The pipeline uses the fixed order:

```text
WBC, TBI, PBS, BMA, TBF
```

## Raw-data configuration

Raw images are not redistributed. Before Stage 01, either set one common root:

```bash
export FOCUS_DATA_ROOT=/path/to/data/raw
```

with this structure:

```text
data/raw/
  WBC/
  TBI/
  PBS/
  BMA/
  TBF/
```

or set each dataset independently:

```bash
export FOCUS_WBC_DIR=/path/to/WBC
export FOCUS_TBI_DIR=/path/to/TBI
export FOCUS_PBS_DIR=/path/to/PBS
export FOCUS_BMA_DIR=/path/to/BMA
export FOCUS_TBF_DIR=/path/to/TBF
```

The default is the repository-local `data/raw/<dataset>` layout. Environment variables are read by `config/paths.py`; source files do not need to be edited.

Each dataset directory must preserve the stack/focus-sequence organization expected by `src/io/dataset_loader.py`. Inspect `outputs/01_stacks/metadata/` for the production stack counts and slice structure.

## Excluded cache

`outputs/01_stacks/arrays/` is intentionally excluded. It is a 46 GB derived cache that can be regenerated from authorized source images with:

```bash
python scripts/01_build_stacks.py --full-run
```

## Data-governance requirement

Before publishing links or instructions for obtaining the source datasets, verify each dataset's redistribution license, participant/privacy conditions, and citation requirements. Do not treat surrogate labels as optical ground truth.
