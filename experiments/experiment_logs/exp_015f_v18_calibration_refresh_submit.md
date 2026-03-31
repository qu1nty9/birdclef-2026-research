# exp_015f_v18_calibration_refresh_submit

## Goal

Provide a thin Kaggle submit notebook for the refreshed `exp_015f` calibration artifacts.

## Notebooks

- export: `notebooks/exp_015f_v18_calibration_refresh_export.ipynb`
- submit: `notebooks/kaggle_submission_exp_015f_v18_calibration_refresh_submit.ipynb`

## Design

- load the fixed V18 Pantanal/ProtoSSM stack through exported artifacts
- load the refreshed calibration files from `exp_015f`
- keep the notebook CPU-only for Kaggle code-competition stability
- apply refreshed calibrators before file-level scaling and threshold sharpening

## Expected Inputs

- competition data
- TensorFlow wheels dataset
- `perch_v2_cpu`
- refresh artifact dataset exported by `exp_015f_v18_calibration_refresh_export.ipynb`

## Status

- notebook scaffolded
- AST validation passed
- Kaggle run not executed yet
