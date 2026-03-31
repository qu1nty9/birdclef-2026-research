# exp_015c_v18_artifact_export

## Goal

Turn the timed-out monolithic V18 submit path into an offline/export stage that trains the full downstream stack once and saves all artifacts needed for a later thin Kaggle submission.

## Inputs

- `perch_v2_cpu`
- `full_perch_meta.parquet`
- `full_perch_arrays.npz`
- competition metadata / taxonomy

## Outputs

- `proto_ssm_state.pt`
- `residual_ssm_state.pt` when residual is enabled
- `sklearn_artifacts.pkl`
- `prior_tables.pkl`
- `per_class_thresholds.npy`
- `artifacts_manifest.json`
- `artifact_export_logs.json`

## Motivation

`exp_015c_full_v18_submit_path` timed out on Kaggle even with `P100` because submit mode still performed expensive train-time work. This export notebook isolates that expensive fitting stage so the later Kaggle run only does hidden-test inference.
