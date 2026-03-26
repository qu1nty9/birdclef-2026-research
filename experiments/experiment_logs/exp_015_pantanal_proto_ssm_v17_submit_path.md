# Experiment Log

Experiment ID:
`exp_015`

Experiment Name:
`pantanal_proto_ssm_v17_submit_path`

Date:
`2026-03-27`

Research Question:
Can the strongest known external `0.924` ProtoSSM / Pantanal Distill notebook be operationalized as a faithful Kaggle submission path in our repository without prematurely simplifying away the high-value parts of the stack?

Baseline Reference:
`references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.924.ipynb`

Change Introduced:
- create a faithful submit-ready notebook copy rather than another local simplification
- replace hard-coded Kaggle paths with dynamic input discovery
- make TensorFlow 2.20 wheel installation dynamic
- require attached cached full-file Perch outputs in submit mode
- preserve the original high-value stack:
  - Perch v2 inference
  - ProtoSSM v5
  - MLP probe branch
  - residual SSM second pass
  - TTA
  - file-level scaling
  - rank-aware scaling
  - delta-shift smoothing
  - per-class threshold sharpening logic

Status:
- notebook created
- notebook AST-validated
- not yet Kaggle-run

Notebook:
- `notebooks/kaggle_submission_exp_015_pantanal_proto_ssm_v17.ipynb`

Expected Kaggle Inputs:
- competition data `BirdCLEF+ 2026`
- TensorFlow 2.20 wheel dataset
- `perch_v2_cpu` SavedModel dataset
- cached full-file Perch outputs dataset containing:
  - `full_perch_meta.parquet`
  - `full_perch_arrays.npz`

Key Operational Decisions:
- keep `MODE = "submit"` by default
- keep `DEVICE = cpu` to match the reference competition-safe path
- set `require_full_cache_in_submit = True` so the notebook fails fast if the trusted Perch cache is missing
- add discovery helpers for:
  - competition dir
  - Perch model dir
  - Perch cache dir
  - TensorFlow wheel files

Why This Branch Matters:
- `exp_012` and `exp_012b` showed that local simplified Perch reproductions are not reliable.
- `exp_015` therefore takes the opposite approach:
  - operationalize the high-ceiling notebook almost as-is
  - and only patch the engineering surfaces needed to make it portable in our Kaggle environment

Next Step:
- attach the required Kaggle datasets
- run the notebook in submit mode
- record runtime and public LB
