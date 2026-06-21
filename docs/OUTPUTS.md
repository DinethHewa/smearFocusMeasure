# Included outputs

## GitHub package

The GitHub package includes:

- Label provenance, source labels, and surrogate labels.
- Normalized curves for all 32 active focus measures.
- Timing records and aggregate single-measure metrics.
- Compact GP summaries and retained/deduplicated expressions.
- Composite evaluation and statistics outputs.
- All paper tables, figures, captions, and manifests.
- All reviewer-response computations.

It excludes:

- Raw microscopy images.
- The 46 GB derived stack cache.
- Raw focus curves.
- Full GP generation checkpoints and per-seed progress files.
- Execution logs and duplicate ZIP archives.

## Zenodo package

The Zenodo package additionally includes:

- Raw and normalized focus curves.
- Complete LODO and final-refit per-seed GP artifacts.

## Authoritative manifests

- `outputs/09_paper/manifests/asset_index.json`
- `outputs/09_paper/manifests/figure_manifest.json`
- `outputs/09_paper/manifests/table_manifest.json`
- `outputs/10_review_response_computations/review_response_manifest.json`

Historical absolute paths were sanitized in these release packages. Paths in exported manifests are repository-relative; raw-data roots are represented as external data locations.
