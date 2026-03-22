# Experiment Log

Experiment ID:
`exp_007`

Experiment Name:
`native_priors_on_exp006_oof`

Date:
`2026-03-22`

Research Question:
Does the proven `exp_005` metadata-prior and texture-aware postprocessing recipe still help when applied to the stronger, fold-aware `exp_006` validation exports instead of the earlier single-fold `exp_004` outputs?

Baseline Reference:
`exp_006_soundscape_finetuning_v2`

Change Introduced:
Keep the native `exp_006` predictions fixed and test fold-safe inference adaptations only:
- `site/hour/site-hour` priors fit on each fold's training partition
- stronger texture weighting than event weighting
- optional filename-level texture smoothing
- honest pooled OOF evaluation across folds `0`, `1`, `2`

Dataset:
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_00/`
- `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_01/`
- `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_02/`

Feature Extraction:
- No new acoustic features
- Reuse exported `exp_006` probabilities and row metadata
- Priors computed from soundscape label frequencies per fold

Model Architecture:
- No new neural model
- Inference-only fusion on top of native logits

Training Setup:
- none
- postprocessing-only notebook experiment

Validation Strategy:
- Reconstruct folds `0-2` from the same grouped-soundscape split protocol as `exp_006`
- Fit priors only on the non-held-out files for each fold
- Report both:
  - fold-wise macro ROC-AUC
  - pooled OOF macro ROC-AUC across all exported rows

Results:
- Raw pooled OOF macro ROC-AUC: `0.6646442720`
- Best pooled OOF macro ROC-AUC: `0.7108902338`
- Delta versus raw pooled OOF: `+0.0462459618`
- Best variant: `event_texture_priors_smooth`
- Pooled OOF scored classes: `54`
- Fold-wise raw macro ROC-AUC:
  - fold `0`: `0.7796052180`
  - fold `1`: `0.8312950828`
  - fold `2`: `0.7724515414`
- Fold-wise best macro ROC-AUC:
  - fold `0`: `0.8229967977`
  - fold `1`: `0.8640179581`
  - fold `2`: `0.8286392475`
- Mean fold-wise raw macro ROC-AUC: `0.7944506141`
- Mean fold-wise best macro ROC-AUC: `0.8385513344`
- Kaggle leaderboard score: `0.758`

Training Time:
- under one minute locally

Observations:
- The `exp_005` recipe transfers cleanly to the `exp_006` branch.
- The same direction also transferred to the public leaderboard:
  - previous native public score: `0.737`
  - `exp_007` 3-fold public score: `0.758`
  - absolute public uplift: `+0.021`
- Texture-aware priors remain the dominant gain source:
  - raw: `0.6646`
  - event priors only: `0.6672`
  - texture priors only: `0.7013`
  - event + texture priors: `0.7086`
  - event + texture priors + smoothing: `0.7109`
- Filename-level smoothing is a small but real extra improvement after priors are already active.
- The pooled OOF score is much lower than the fold-wise mean because pooled evaluation scores more classes (`54`) than the sparse per-fold readouts.
- The public gain is real but smaller than the local OOF delta, which suggests that the current bottleneck is no longer just inference calibration.

Failure Cases:
- Even with the positive `+0.0462` uplift, the best native pooled OOF remains below the stronger external Perch-based soundscape reference.
- The experiment improves inference quality, but it does not prove that the underlying `exp_006` training recipe is a stronger model than `exp_004`.
- Even after the successful public LB improvement to `0.758`, the native branch still remains far below the current `0.890` reference blend.
- This means more lightweight priors alone are unlikely to close the gap; the next jump should come from stronger soundscape modeling.

Notebook:
- `notebooks/exp_007_native_priors_on_exp006_oof.ipynb`
- `notebooks/kaggle_submission_exp_007_native_3fold.ipynb`
