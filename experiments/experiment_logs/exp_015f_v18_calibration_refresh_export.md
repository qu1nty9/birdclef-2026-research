# exp_015f_v18_calibration_refresh_export

## Goal

Run a thin calibration refresh on top of the fixed `exp_015d` artifact stack without retraining `ProtoSSM`, probe models, or `ResidualSSM`.

## Notebook

- export: `notebooks/exp_015f_v18_calibration_refresh_export.ipynb`

## Inputs

- competition data
- full-file Perch cache dataset:
  - `full_perch_meta.parquet`
  - `full_perch_arrays.npz`
- existing `exp_015d` artifact dataset

## Output

A new self-contained artifact folder with:

- copied fixed `exp_015d` artifacts
- refreshed `per_class_thresholds.npy`
- new `calibrators.pkl`
- new `calibration_manifest.json`
- updated `artifacts_manifest.json`

## Design

- replay the fixed V18 stack on cached labeled soundscape rows
- compute in-sample first-pass probabilities on the exact downstream path
- fit isotonic calibrators at file-max level
- re-optimize thresholds on calibrated probabilities

## Motivation

`exp_015e` kept the calibration idea but still reused the heavy export pipeline, which made timeout likely. This refresh branch isolates calibration into the cheapest possible artifact-only update.

## Status

- notebook scaffolded
- initial Kaggle run failed on ProtoSSM checkpoint load because the local `ProtoSSMv2` class used old field names (`proto`, plain `class_to_family`) instead of the artifact-compatible V18 layout (`prototypes`, registered `class_to_family` buffer)
- notebook patched to match the `exp_015d` artifact state-dict layout
- the notebook was also found to be hard-coded to `cpu`, which made the refresh replay path much slower than intended
- replay now uses normal `cuda -> mps -> cpu` selection plus batched ProtoSSM / ResidualSSM forward passes
- after reviewing the competition rules and timing prints, the real blocker turned out to be Kaggle code-competition policy rather than replay cost: GPU-attached submissions are effectively limited to about one minute
- the notebook now fails fast with an explicit message if a GPU accelerator is attached, instructing to run it as a CPU-only `Save Version` export job
- AST validation passed after the compatibility fix
- AST validation passed
- Kaggle run not executed yet
