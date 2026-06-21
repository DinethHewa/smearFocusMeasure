# Review Response Computations

This folder contains isolated reviewer-response computations for the BSPC manuscript. The scripts and outputs here read existing pipeline artifacts and write only under `outputs/10_review_response_computations/`.

No LODO GP run was rerun. No full pipeline stage was rerun. Existing manuscript outputs under `outputs/03_*` through `outputs/09_paper/` were used as inputs only.

Subfolders:

- `gradient_free_voter_sensitivity/`: gradient-free surrogate-label sensitivity analysis.
- `cfm4_diagnostics/`: CFM4 vs FTSI and BG-GSE diagnostic outputs.
- `runtime_weight_sensitivity/`: metric-weight sensitivity for runtime weight.
- `gp_audit/`: saved GP hyperparameter and composite fold-origin audit.
- `supplementary_methods/`: formal metric definitions.
- `manuscript_insert_text/`: concise text blocks for reviewer response and manuscript insertion.
