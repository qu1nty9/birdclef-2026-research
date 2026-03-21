# Experiment Log

Experiment ID:
`exp_005`

Experiment Name:
`native_priors_texture_postproc`

Date:
`2026-03-21`

Research Question:
How much can the native `exp_004` soundscape model gain from `site/hour/site-hour` priors and texture-aware smoothing without any new training?

Baseline Reference:
`exp_004_soundscape_finetuning`

Change Introduced:
Keep the native `exp_004` checkpoint fixed and apply soundscape-aware postprocessing inspired by the Perch branch:
- metadata priors in logit space
- separate treatment of texture taxa
- smoothing over 12-window soundscape sequences

Dataset:
- Competition soundscape labels under `data/birdclef-2026/`
- Validation fold reconstructed to match `exp_004`
- Native checkpoint from `experiments/outputs/exp_004_soundscape_finetuning/best_model.pt`

Feature Extraction:
- No new acoustic training
- Uses native `exp_004` validation logits as the starting point
- Adds metadata priors derived from the soundscape training fold

Model Architecture:
- Same `exp_004` EfficientNet-B0 + GeM + attention SED head
- No new neural parameters are trained in this experiment

Training Setup:
- None
- This is a postprocessing and evaluation notebook

Augmentations:
- None

Validation Strategy:
- Same grouped soundscape validation fold as `exp_004`
- Compare:
  - raw native predictions
  - event priors only
  - texture priors only
  - event + texture priors
  - event + texture priors + smoothing

Results:
- Raw native macro ROC-AUC: `0.7796052180`
- Best variant: `event_texture_priors_smooth`
- Best macro ROC-AUC: `0.8156599403`
- Gain over raw native predictions: `+0.0360547223`
- Full ablation table:
  - `event + texture priors + smoothing`: `0.8157`
  - `event + texture priors`: `0.8151`
  - `texture priors only`: `0.8059`
  - `event priors only`: `0.7888`
  - `raw`: `0.7796`
- Validation scored `29` classes on this fold
- Kaggle leaderboard score: `n/a`

Training Time:
- Fast notebook-only diagnostic run
- Main runtime cost is one forward pass of the `exp_004` checkpoint over the soundscape validation fold

Observations:
- The uplift is real and strong enough to matter.
- The native branch clearly benefits from the same soundscape-aware logic that helped the Perch branch.
- Texture handling remains more important than event priors alone:
  - `texture priors only` beat `event priors only` by a wide margin
- Smoothing adds a small but positive final increment on top of the prior fusion.
- This result materially narrows the local gap between the native branch and `exp_003`.

Failure Cases:
- The fold is still sparse and only scores `29` classes, so this is not yet a final native-vs-Perch verdict.
- Some classes regress under the best variant, including `22961`, `24321`, and `25092`.

Next Experiment Ideas:
- Improve native soundscape validation coverage with more folds or OOF aggregation
- Build a native submission path from `exp_004 + priors`
- Decide later whether the native branch needs a second-stage stacker, or whether priors are already enough for the next Kaggle test

Submission Notebook:
- `notebooks/kaggle_submission_exp_005_native_hybrid.ipynb`
- This Kaggle path keeps the same lightweight recipe as the best local variant:
  - `exp_004` checkpoint
  - `site/hour/site-hour` priors
  - texture-aware smoothing
  - no Perch, no external checkpoints, no probe fitting
