# Experiment Log

Experiment ID:
`exp_008`

Experiment Name:
`long_context_native_sed`

Date:
`2026-03-22`

Research Question:
Can a repository-native long-context SED branch trained on `20s` soundscape contexts with `4` aligned `5s` outputs outperform the current short-context native branch and become the next serious native submission base?

Baseline Reference:
`exp_007_native_priors_on_exp006_oof`

Change Introduced:
Move from short-context clip classification to a longer soundscape model:
- input waveform context: `20s`
- output windows per sample: `4 x 5s`
- overlap-aware aggregation back to row-level validation predictions
- background mixing retained
- partial initialization from the best `exp_002` clipwise checkpoint

Dataset:
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/train_soundscapes/`
- Initialization from `experiments/outputs/exp_002_train_audio_reproduction/best_model.pt`

Feature Extraction:
- Native mel frontend
- No Perch features
- No pseudo-labels yet

Model Architecture:
- timm EfficientNet-B0 backbone
- GeM-style frequency pooling
- temporal 1D head that emits `4` aligned `5s` windows per `20s` context

Training Setup:
- supervised soundscape training only in the first version
- background mixing retained
- no noisy-student branch yet
- no metadata priors inside the training notebook itself

Validation Strategy:
- grouped soundscape split by filename
- context-level prediction followed by overlap-aware mean aggregation back to row-level `row_id`
- export row-level validation predictions for later `exp_007`-style postprocessing reuse

Status:
- notebook created
- first three folds completed

Results:
- Fold `0` best validation macro ROC-AUC: `0.8377433031`
- Fold `1` best validation macro ROC-AUC: `0.8222946825`
- Fold `2` best validation macro ROC-AUC: `0.8094790079`
- Mean across folds `0-2`: `0.8231724344`
- Best epochs:
  - fold `0`: `6 / 6`
  - fold `1`: `6 / 6`
  - fold `2`: `5 / 6`
- Scored classes:
  - fold `0`: `29`
  - fold `1`: `29`
  - fold `2`: `35`
- Skipped no-positive classes:
  - fold `0`: `205`
  - fold `1`: `205`
  - fold `2`: `199`
- Coverage statistics after overlap aggregation:
  - mean: `3.0`
  - min: `1`
  - max: `4`
- Best validation loss:
  - fold `0`: `0.0511462778`
  - fold `1`: `0.0408654824`
  - fold `2`: `0.0429244949`
- Kaggle leaderboard score: `n/a`

Observations:
- This is the first clear native modeling gain that does not come only from postprocessing.
- Across folds `0-2`, raw `exp_008` is stronger and more stable locally than the short-context `exp_006` branch:
  - `exp_006` mean over folds `0-2`: `0.7945`
  - `exp_008` mean over folds `0-2`: `0.8232`
  - absolute gain: `+0.0287`
- The gain is not limited to the easiest sparse fold:
  - fold `2` still reached `0.8095` while scoring `35` classes instead of `29`
- The longer-context branch therefore still looks genuinely promising before any metadata-prior layer is applied.
- The overlap-aware aggregation is active in practice because most validation rows were seen in multiple contexts.

Failure Cases:
- Even after three local folds, the first single-fold Kaggle test through `exp_008b` scored only `0.707`.
- This means the local gain is real, but the local protocol is still not strong enough to guarantee leaderboard transfer.
- The branch should continue, but only with a stricter multi-fold local readout before another leaderboard attempt.

Planned Outputs:
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/history.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/result_snapshot.json`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/best_model.pt`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/best_valid_predictions.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/best_valid_outputs.npz`

Notebook:
- `notebooks/exp_008_long_context_native_sed.ipynb`
