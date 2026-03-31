# exp_015e_v18_calibrated_submit

## Goal

Add a low-risk calibration-first refinement on top of the working V18 artifact split.

## Notebooks

- export: `notebooks/exp_015e_v18_calibrated_artifact_export.ipynb`
- submit: `notebooks/kaggle_submission_exp_015e_v18_calibrated_submit.ipynb`

## Design

- keep the `exp_015d` thin-submit structure intact
- fit per-class isotonic calibrators from OOF blended probabilities during export
- optimize per-class thresholds on calibrated OOF probabilities
- save both calibrators and calibrated thresholds as artifact files
- apply calibration in thin submit before downstream file-level scaling and threshold sharpening

## Rationale

This branch targets one of the clearest remaining active differences suggested by the `0.927` reference family: calibration-aware postprocessing. It is intentionally narrower and safer than another full-stack V18 rewrite.

## Expected Inputs

- the same Kaggle inputs as `exp_015d`
- a new artifact dataset exported by `exp_015e_v18_calibrated_artifact_export.ipynb`

## Status

- notebooks scaffolded
- AST validation passed for both notebooks
- Kaggle run not executed yet
