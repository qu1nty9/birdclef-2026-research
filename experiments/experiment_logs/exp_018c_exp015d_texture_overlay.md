# exp_018c_exp015d_texture_overlay

## Goal

Build the first Kaggle-facing targeted specialist overlay on top of the strongest production path:

- base submit path: `notebooks/kaggle_submission_exp_015d_v18_artifact_submit.ipynb`
- specialist source: `notebooks/exp_018a_texture_specialist_oof.ipynb`
- local merge benchmark: `notebooks/exp_018b_targeted_merge_benchmark.ipynb`

The design goal is to change only `Amphibia + Insecta` columns while leaving the rest of the `exp_015d` V18 stack untouched.

## Local Motivation

`exp_018a` finished as a real positive signal on the weak texture-heavy taxa:

- four-fold specialist mean target soundscape macro AUC: `0.8057`
- same-taxa generic `exp_011` mean: `0.7874`
- delta: `+0.0183`

`exp_018b` then showed that the strongest local merge is a soft target-only blend:

- generic target soundscape macro AUC: `0.7218`
- specialist target soundscape macro AUC: `0.7450`
- best tested targeted blend: `w_spec = 0.75`
- best target soundscape macro AUC: `0.7465`

Because `exp_015d` is much stronger than the `exp_011` proxy used in `exp_018b`, the first public overlay should be more conservative.

## Packaged Assets

Prepared Kaggle specialist dataset:

- `submissions/kaggle_datasets/birdclef-exp018a-texture-specialist-4fold`

Contents:

- `fold_00/best_model.pt`
- `fold_01/best_model.pt`
- `fold_02/best_model.pt`
- `fold_03/best_model.pt`
- `target_config.json`
- `dataset_manifest.json`
- `README.md`

Notebook scaffold:

- `notebooks/kaggle_submission_exp_018c_exp015d_texture_overlay.ipynb`

## First Kaggle Configuration

Recommended first run:

```python
EXP018A_MODEL_DATASET_HINT = "birdclef-exp018a-texture-specialist-4fold"
RUN_EXP018A_OVERLAY = True
EXP018A_BLEND_WEIGHT = 0.35
EXP018A_FOLD_IDS = (0, 1)
EXP018A_BATCH_FILES = 8
```

Why these defaults:

- lower risk than the local `0.75` optimum from `exp_018b`
- better runtime margin in code competition submit mode
- easier to interpret against the already strong `exp_015d = 0.929`

## Validation Status

- notebook JSON validated
- all code cells pass `ast.parse`
- overlay code resolves the specialist dataset dynamically
- overlay only touches the target indices from `target_config.json`
- target row order is checked against `meta_test["row_id"]` before blending

## Next Step

Run the first Kaggle submission for:

- `notebooks/kaggle_submission_exp_018c_exp015d_texture_overlay.ipynb`

If runtime is safe and score is neutral-to-positive, the next sweep should stay conservative first:

- `EXP018A_BLEND_WEIGHT = 0.25`
- `EXP018A_BLEND_WEIGHT = 0.35`
- `EXP018A_BLEND_WEIGHT = 0.45`
