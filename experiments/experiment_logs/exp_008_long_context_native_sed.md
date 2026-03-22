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
- first fold run completed

Results:
- Fold `0` best validation macro ROC-AUC: `0.8377433031`
- Best epoch: `6 / 6`
- Scored classes: `29`
- Skipped no-positive classes: `205`
- Coverage statistics after overlap aggregation:
  - mean: `3.0`
  - min: `1`
  - max: `4`
- Best validation loss: `0.0511462778`
- Kaggle leaderboard score: `n/a`

Observations:
- This is the first clear native modeling gain that does not come only from postprocessing.
- On the same sparse fold, raw `exp_008` improved over:
  - raw `exp_004`: `0.7796 -> 0.8377`
  - fold `0` raw `exp_006`: `0.7796 -> 0.8377`
  - fold-local best `exp_005`: `0.8157 -> 0.8377`
  - fold-local best `exp_007`: `0.8230 -> 0.8377`
- The longer-context branch therefore looks genuinely promising before any metadata-prior layer is applied.
- The overlap-aware aggregation is active in practice because most validation rows were seen in multiple contexts.

Failure Cases:
- This is still only one sparse fold with `29` scored classes.
- The result is strong enough to continue, but not yet strong enough to declare the branch solved.
- No Kaggle submission has been run yet for this branch.

Planned Outputs:
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/history.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/result_snapshot.json`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/best_model.pt`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/best_valid_predictions.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_XX/best_valid_outputs.npz`

Notebook:
- `notebooks/exp_008_long_context_native_sed.ipynb`
