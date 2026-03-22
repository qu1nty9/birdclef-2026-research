# Experiment Log

Experiment ID:
`exp_008b`

Experiment Name:
`long_context_priors_postproc`

Date:
`2026-03-22`

Research Question:
Does the proven `exp_007` priors plus texture-smoothing layer still improve the stronger long-context `exp_008` branch, or did the move to `20s -> 4 x 5s` context already absorb most of that gain?

Baseline Reference:
`exp_008_long_context_native_sed`

Change Introduced:
Apply the existing native soundscape-aware inference layer on top of exported `exp_008` fold `0` predictions:
- `site/hour/site-hour` metadata priors with shrinkage
- stronger prior fusion on texture classes
- texture-aware smoothing across adjacent `5s` windows inside each file

Dataset:
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/taxonomy.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_00/best_valid_meta.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_00/best_valid_outputs.npz`

Feature Extraction:
- No new features
- Reuses exported row-level probabilities from `exp_008`

Model / Inference Layer:
- Raw `exp_008` long-context native predictions
- Added priors and smoothing only at inference time

Validation Strategy:
- Reconstruct the exact `exp_008` fold `0` grouped filename split
- Verify `row_id` alignment between reconstructed validation rows and exported `exp_008` outputs
- Compare five variants:
  - `raw`
  - `event_priors_only`
  - `texture_priors_only`
  - `event_texture_priors`
  - `event_texture_priors_smooth`

Status:
- notebook created
- local run completed

Results:
- Raw `exp_008` fold `0`: `0.8377433031`
- Best variant: `event_texture_priors_smooth`
- Best macro ROC-AUC: `0.8434672644`
- Absolute gain vs raw: `+0.0057239613`
- Kaggle leaderboard score: `0.707`
- Scored classes: `29`
- Coverage statistics reused from `exp_008` export:
  - mean: `3.0`
  - min: `1`
  - max: `4`
- Variant ablations:
  - `event + texture priors + smoothing`: `0.8435`
  - `event + texture priors`: `0.8402`
  - `raw`: `0.8377`
  - `texture priors only`: `0.8357`
  - `event priors only`: `0.8188`

Observations:
- Priors still help the long-context branch, so the soundscape-aware inference layer remains relevant even after the architecture jump.
- The gain is much smaller than in the short-context branch (`+0.0057` here versus `+0.0361` in `exp_005`), which is a good sign rather than a bad one:
  - long context already captures part of the file-level and neighborhood structure that priors used to patch in later
  - the model itself is now doing more of the adaptation work
- The best variant remained the same as in the earlier native branch:
  - `event + texture priors + smoothing`
- Texture priors alone no longer dominate raw performance the way they did earlier; this again suggests that the stronger long-context model is reducing dependence on inference-only fixes.
- The leaderboard result was negative despite the local gain:
  - `exp_008b` public LB: `0.707`
  - previous native public best (`exp_007`): `0.758`
  - previous single-fold native hybrid (`exp_005`): `0.737`
- This means the current long-context branch should not replace the short-context native default yet.

Failure Cases:
- Still only one sparse fold with `29` scored classes.
- Several previously strong classes regressed while texture-heavy classes gained, so the long-context branch should still be checked on Kaggle before it becomes the default native path.
- The first Kaggle check shows that the single-fold long-context result was not a reliable leaderboard proxy.
- Most likely failure mode: fold `0` was too sparse and too optimistic, so the branch needs more folds or a stricter local protocol before another leaderboard attempt.

Planned Outputs:
- `experiments/outputs/exp_008b_long_context_priors_postproc/ablation_results.csv`
- `experiments/outputs/exp_008b_long_context_priors_postproc/classwise_auc_comparison.csv`
- `experiments/outputs/exp_008b_long_context_priors_postproc/best_valid_predictions.parquet`
- `experiments/outputs/exp_008b_long_context_priors_postproc/result_snapshot.json`

Notebook:
- `notebooks/exp_008b_long_context_priors_postproc.ipynb`
