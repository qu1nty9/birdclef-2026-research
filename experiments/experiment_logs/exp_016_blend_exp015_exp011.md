# Experiment Log

Experiment ID:
`exp_016`

Experiment Name:
`blend_exp015_exp011`

Date:
`2026-03-27`

Research Question:
Do the strongest faithful external path (`exp_015`) and the strongest repository-native path (`exp_011`) make sufficiently different errors on hidden test to justify a simple submission-level blend?

Baseline References:
- `exp_015 = 0.925`
- `exp_011 = 0.850`

Change Introduced:
- create a lightweight Kaggle notebook that blends two attached `submission.csv` files
- avoid rerunning both heavy model stacks in the first ensemble attempt
- default first blend:
  - `0.90 * exp_015 + 0.10 * exp_011`
- after Kaggle artifact mismatch (`240` vs `48` rows), pivot to a runtime blend notebook that runs both branches in one submit pass
- keep TensorFlow on CPU for the Perch/ProtoSSM path and allow PyTorch to use GPU for the `exp_011` branch

Notebook:
- `notebooks/kaggle_submission_exp_016_blend_exp015_exp011.ipynb`
- `notebooks/kaggle_submission_exp_016_runtime_blend_exp015_exp011.ipynb`

Expected Kaggle Inputs:
- competition data `BirdCLEF+ 2026`
- TF 2.20 wheel dataset for the Pantanal/ProtoSSM path
- `perch_v2_cpu` SavedModel dataset
- full-file Perch cache dataset with `full_perch_meta.parquet` and `full_perch_arrays.npz`
- `exp_011` 4-fold model dataset with `fold_00..fold_03/best_model.pt`

Status:
- lightweight CSV-blend notebook created and AST-validated, but rejected as unreliable because Kaggle-downloaded artifacts did not share a compatible row set
- runtime-blend notebook created
- runtime-blend notebook AST-validated
- runtime-blend notebook optimized to batch `exp_011` soundscape inference across files and reuse `read_soundscape_60s(...)` instead of per-file `librosa.load(...)`
- first Kaggle run pending

Why This Branch Matters:
- It is the cheapest direct test of complementarity between the best external and best native branches.
- If it helps, the project gets a stronger public path without any additional model training.
- If it does not help, we learn that the current native branch is not adding enough diversity to beat the already strong `exp_015`.

Next Step:
- run the runtime blend notebook on Kaggle with `GPU = On`
- start from `EXP011_BLEND_WEIGHT = 0.10`
- if promising, try nearby weights such as `0.05` and `0.15`
