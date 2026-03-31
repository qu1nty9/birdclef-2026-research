# Experiment Log

Experiment ID:
`exp_015c`

Experiment Name:
`full_v18_submit_path`

Date:
`2026-03-28`

Research Question:
Can we exceed the confirmed `exp_015 = 0.925` score by porting the active V18 changes from the `0.927` Pantanal / ProtoSSM notebook while preserving the working engineering scaffold of the faithful `exp_015` submit path?

Baseline References:
- `exp_015 = 0.925`
- `references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.927.ipynb`

Change Introduced:
- create a separate high-risk submit notebook instead of mutating the stable `exp_015`
- keep the same Kaggle input resolution and cache handling as `exp_015`
- port the active V18 changes that appear wired into the final submit path:
  - larger `ProtoSSM`
  - larger `ResidualSSM`
  - updated fusion lambdas
  - updated MLP probe defaults
  - finer threshold grid
  - `tta_shifts = [0]`
  - adaptive delta smoothing
- do not blindly copy helper blocks that appear defined but unused in the reference notebook

Notebook:
- `notebooks/kaggle_submission_exp_015c_full_v18_submit_path.ipynb`

Status:
- notebook created
- notebook JSON validated
- all code cells pass `ast.parse`
- PyTorch branch updated to select `cuda/mps/cpu` dynamically instead of forcing CPU
- Residual SSM now has a stricter wall-time guard on CPU paths
- first Kaggle run pending

Why This Branch Matters:
- It is the cleanest way to test whether the `0.927` reference is a real incremental upgrade over the already-working `exp_015 = 0.925`.
- It protects the stable production baseline by isolating the V18 changes in a separate notebook.
- It keeps the project on the strongest external track without another full re-port from scratch.

Expected Kaggle Inputs:
- competition data `BirdCLEF+ 2026`
- TensorFlow 2.20 wheel dataset
- `perch_v2_cpu` model dataset
- cached full-file Perch outputs dataset (`full_perch_meta.parquet`, `full_perch_arrays.npz`)

Next Step:
- run the first Kaggle submission for `exp_015c`
- compare runtime and public LB directly against `exp_015 = 0.925`
