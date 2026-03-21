# Experiment Log

Experiment ID:
`exp_006`

Experiment Name:
`soundscape_finetuning_v2`

Date:
`2026-03-21`

Research Question:
Can a stronger native soundscape finetuning recipe with real `secondary_labels` weighting and fold-specific validation exports improve both checkpoint quality and evaluation reliability beyond `exp_004`?

Baseline Reference:
`exp_004_soundscape_finetuning`

Change Introduced:
Keep the native soundscape branch, but strengthen the training and evaluation loop:
- non-zero `secondary_weight`
- slightly larger `train_audio` replay bank
- fold-specific output directories
- exported best validation predictions for later OOF analysis

Dataset:
- Competition soundscape labels under `data/birdclef-2026/`
- Same `train_audio` replay source as `exp_004`
- Initialization from `experiments/outputs/exp_002_train_audio_reproduction/best_model.pt`

Feature Extraction:
- Same native mel frontend as `exp_004`
- No external Perch features
- No borrowed public-LB checkpoints

Model Architecture:
- Same repository-native EfficientNet-B0 + GeM + attention SED head

Training Setup:
- Soundscape finetuning from the `exp_002` checkpoint
- Background mixing retained
- Replay branch retained
- `secondary_labels` now have a non-zero contribution

Validation Strategy:
- Grouped soundscape validation by filename
- Fold-aware outputs under `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_XX/`
- Best validation epoch should export fold predictions for later OOF aggregation

Results:
- Completed folds: `0`, `1`, `2`
- Fold 0 best validation macro ROC-AUC: `0.7796052180`
- Fold 1 best validation macro ROC-AUC: `0.8312950828`
- Fold 2 best validation macro ROC-AUC: `0.7724515414`
- Mean validation macro ROC-AUC across folds 0-2: `0.7944506141`
- Best epochs by fold:
  - fold 0: `4`
  - fold 1: `6`
  - fold 2: `6`
- Scored classes by fold:
  - fold 0: `29`
  - fold 1: `29`
  - fold 2: `35`
- Exported validation artifacts: yes
- Kaggle leaderboard score: `n/a`

Training Time:
- pending

Observations:
- Fold `0` matched the earlier `exp_004` result exactly, so the branch did not open with a clean gain.
- Fold `1` was notably stronger and shows that the branch still has upside.
- Fold `2` dropped again while covering more classes, reinforcing how sensitive the current protocol still is to fold composition.
- The new `secondary_weight` and larger replay bank therefore look mixed rather than clearly helpful or clearly useless.
- The fold-aware infrastructure worked as intended:
  - best checkpoint saved
  - `result_snapshot.json` saved
  - validation predictions exported

Failure Cases:
- The current validation coverage is still limited and uneven across folds.
- Because of that, the recipe cannot yet be declared a real modeling improvement over `exp_004`.
- The branch is strong enough to keep for follow-up, but not strong enough yet to justify a Kaggle submission without another postprocessing comparison.

Notebook:
- `notebooks/exp_006_soundscape_finetuning_v2.ipynb`
