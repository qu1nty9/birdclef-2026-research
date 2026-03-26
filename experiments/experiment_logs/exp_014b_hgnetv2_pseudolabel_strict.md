# Experiment Log

Experiment ID:
`exp_014b`

Experiment Name:
`hgnetv2_pseudolabel_strict`

Date:
`2026-03-26`

Research Question:
Can a stricter pseudo-label recipe recover the early positive gains from `exp_014` while avoiding the regressions that appeared on folds `2-3`?

Baseline Reference:
`exp_014_hgnetv2_pseudolabel`

Change Introduced:
- keep the same `HGNetV2-B0` student and `exp_011` teacher family
- raise `pseudo_min_confidence` from `0.30` to `0.45`
- reduce `pseudo_loss_weight` from `0.50` to `0.25`
- reduce `max_pseudo_segments_per_file` from `6` to `4`
- cap the pseudo cache globally at `12000` rows
- delay pseudo usage until `epoch >= 2`
- shorten the run to `6` epochs with a slightly lower learning rate

Dataset:
- `data/birdclef-2026/train.csv`
- `data/birdclef-2026/train_soundscapes/`
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/taxonomy.csv`
- `experiments/outputs/exp_011_hgnetv2_soundscape_supervised/fold_00/`
- `experiments/outputs/exp_011_hgnetv2_soundscape_supervised/fold_01/`
- `experiments/outputs/exp_011_hgnetv2_soundscape_supervised/fold_02/`
- `experiments/outputs/exp_011_hgnetv2_soundscape_supervised/fold_03/`

Validation Strategy:
- keep the same grouped supervised folds as `exp_011` / `exp_014`
- keep the current validation fold fully excluded from the teacher ensemble
- exclude labeled soundscape files from pseudo candidates
- continue selecting checkpoints by `soundscape_macro_auc`

Status:
- notebook created
- notebook AST-validated
- safe setup validated in the project `.venv`
- pseudo generation and training not yet run

Setup Validation Readout:
- Notebook:
  - `notebooks/exp_014b_hgnetv2_pseudolabel_strict.ipynb`
- Fold checked during setup:
  - `0`
- Device during terminal validation:
  - `cpu`
- Labeled rows:
  - `36078`
- Train rows in fold `0`:
  - `26991`
- Valid rows in fold `0`:
  - `9087`
- Valid soundscape rows in fold `0`:
  - `144`
- Teacher folds:
  - `[1, 2, 3]`
- Pseudo manifest rows / files:
  - `127104 / 10592`
- Delayed pseudo start:
  - `epoch 2`

Artifacts Planned:
- `experiments/outputs/exp_014b_hgnetv2_pseudolabel_strict/fold_00/pseudo_manifest.parquet`
- `experiments/outputs/exp_014b_hgnetv2_pseudolabel_strict/fold_00/pseudo_label_meta.parquet`
- `experiments/outputs/exp_014b_hgnetv2_pseudolabel_strict/fold_00/pseudo_label_probs.npz`
- `experiments/outputs/exp_014b_hgnetv2_pseudolabel_strict/fold_00/teacher_summary.json`
- `experiments/outputs/exp_014b_hgnetv2_pseudolabel_strict/fold_00/history.csv`
- `experiments/outputs/exp_014b_hgnetv2_pseudolabel_strict/fold_00/result_snapshot.json`

Observations:
- This is intentionally not a cosmetic rerun of `exp_014`.
- Every main knob is moved in the “more conservative pseudo-labeling” direction.
- The branch is designed to answer a narrow question:
  - was the failure of `exp_014` caused by pseudo labels in general,
  - or by letting too many low-to-medium confidence pseudo rows influence the student too strongly?

## Fold 0 Pseudo Generation

Confirmed Result:
- Manifest rows / files:
  - `127104 / 10592`
- Pseudo rows / files:
  - `8724 / 3237`
- Retain rate:
  - `0.0686`
- Mean pseudo confidence:
  - `0.6358`
- `p75` pseudo confidence:
  - `0.7378`

Interpretation:
- The stricter recipe keeps far fewer pseudo rows than `exp_014`.
- At the same time, the kept rows are clearly cleaner:
  - higher mean confidence
  - higher upper quantiles
  - lower exposure to medium-confidence windows

## Fold 0 First Full Training Run

Confirmed Result:
- Best epoch:
  - `4 / 6`
- Best selection metric:
  - soundscape-only macro ROC-AUC `0.8682576606`
- Best overall macro ROC-AUC observed during the run:
  - `0.9702049724`
- Pseudo rows / files used:
  - `8724 / 3237`

Direct Comparison:
- `exp_014` fold `0` soundscape-only macro ROC-AUC:
  - `0.8684479825`
- `exp_014b` fold `0` soundscape-only macro ROC-AUC:
  - `0.8682576606`
- Delta:
  - `-0.00019`

Pseudo-Cache Comparison:
- `exp_014` pseudo rows / files:
  - `20688 / 5471`
- `exp_014b` pseudo rows / files:
  - `8724 / 3237`

Interpretation:
- This is the strongest possible first signal for the strict branch.
- We preserved essentially the same fold `0` quality while cutting the pseudo cache drastically.
- That suggests the failure mode of `exp_014` may indeed have been “too much noisy pseudo supervision” rather than the pseudo-label idea itself.

Practical Conclusion:
- `exp_014b` is a meaningful improvement over `exp_014` as a recipe, even before more folds.
- It is still only one fold, so it is not a Kaggle promotion candidate yet.
- But it is already a successful diagnostic experiment.
