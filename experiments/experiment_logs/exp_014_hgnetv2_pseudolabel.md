# Experiment Log

Experiment ID:
`exp_014`

Experiment Name:
`hgnetv2_pseudolabel`

Date:
`2026-03-25`

Research Question:
Can the strongest repository-native supervised branch (`exp_011`) be pushed beyond its stabilized `0.850` Kaggle score by turning unlabeled `train_soundscapes` windows into soft target-domain training data?

Baseline Reference:
`exp_011_hgnetv2_soundscape_supervised`

Change Introduced:
- keep the `HGNetV2-B0` recipe from `exp_011`
- use the existing `4-fold exp_011` checkpoints as teachers
- generate soft pseudo labels only on unlabeled `train_soundscapes`
- train a student on:
  - labeled `train_audio`
  - labeled soundscape clips
  - pseudo-labeled soundscape windows
- continue selecting checkpoints by `soundscape_macro_auc`

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
- rebuild the same grouped multi-label fold structure used in `exp_011`
- keep the current validation fold completely out of the pseudo-teacher ensemble
- exclude all labeled soundscape files from pseudo candidates
- evaluate the student on the held-out mixed fold
- use `soundscape_macro_auc` as the selection metric

Status:
- notebook created
- notebook AST-validated
- safe setup validated in the project `.venv`
- fold `0` pseudo manifest built successfully
- pseudo generation and training not yet run

Setup Validation Readout:
- Notebook:
  - `notebooks/exp_014_hgnetv2_pseudolabel.ipynb`
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
- Pseudo manifest rows:
  - `127104`
- Pseudo manifest files:
  - `10592`

Artifacts Planned:
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/pseudo_manifest.parquet`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/pseudo_label_meta.parquet`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/pseudo_label_probs.npz`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/teacher_summary.json`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/history.csv`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/best_model.pt`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/last_model.pt`
- `experiments/outputs/exp_014_hgnetv2_pseudolabel/fold_00/result_snapshot.json`

Observations:
- The branch is intentionally conservative:
  - same backbone family as `exp_011`
  - same mel frontend
  - same soundscape-aware checkpoint selection
- This makes `exp_014` a much cleaner research test than the earlier `exp_009` pseudo-label branch.
- The safe setup already confirms that the notebook can:
  - rebuild the supervised frame
  - resolve all four teacher checkpoints
  - build the unlabeled soundscape manifest without path or IO errors
- The most important next gate is not Kaggle, but whether pseudo generation produces a healthy confidence distribution and a non-trivial number of retained windows.

Next Step:
- Run full pseudo-label generation for fold `0`
- inspect `teacher_summary.json`
- then launch the first full student training run on the same fold

## Fold 0 Pseudo Generation

Confirmed Result:
- Teacher folds:
  - `[1, 2, 3]`
- Pseudo rows:
  - `20688`
- Pseudo files:
  - `5471`
- Mean pseudo confidence:
  - `0.5025`
- `p75` pseudo confidence:
  - `0.6014`
- Threshold:
  - `0.30`
- Per-file cap:
  - `6`

Interpretation:
- The pseudo cache is meaningfully selective rather than exploding to nearly all windows.
- Compared with the manifest size (`127104` windows), the retained pseudo set is about `16.28%`.
- Confidence is moderate rather than overconfident, which is a good fit for soft-label training.

## Fold 0 First Full Training Run

Confirmed Result:
- Best epoch:
  - `5 / 8`
- Best selection metric:
  - soundscape-only macro ROC-AUC `0.8684479825`
- Best overall macro ROC-AUC observed during the run:
  - `0.9660572606`
- Train / valid rows:
  - `26991 / 9087`
- Pseudo rows / files used:
  - `20688 / 5471`
- Resume mode:
  - `init_from_exp011_fold`

Epoch Trace:
- epoch `1`: soundscape macro ROC-AUC `0.8472`
- epoch `2`: soundscape macro ROC-AUC `0.8491`
- epoch `3`: soundscape macro ROC-AUC `0.8261`
- epoch `4`: soundscape macro ROC-AUC `0.8534`
- epoch `5`: soundscape macro ROC-AUC `0.8684`
- epoch `6`: soundscape macro ROC-AUC `0.8629`
- epoch `7`: soundscape macro ROC-AUC `0.8602`
- epoch `8`: soundscape macro ROC-AUC `0.8608`

Interpretation:
- This is a clear positive signal over `exp_011` on the same fold.
- `exp_011` fold `0` soundscape-only macro ROC-AUC:
  - `0.8508523324`
- `exp_014` fold `0` soundscape-only macro ROC-AUC:
  - `0.8684479825`
- Delta:
  - `+0.0176`
- The comparison is also more trustworthy than many earlier branches because:
  - soundscape-scored classes increased from `42` to `46`
  - the branch kept the same backbone family and the same checkpoint-selection rule
- The curve peaked at epoch `5` and then softened slightly, which fits the expected pseudo-label behavior:
  - early target-domain gain
  - then mild overfitting / noise absorption

Practical Conclusion:
- `exp_014` is the strongest first-fold native modeling result after `exp_011`.
- Unlike `exp_009`, this branch improves over an already leaderboard-positive backbone family.
- The next gate should be fold `1`, not Kaggle promotion yet.

## Fold 1 Update

Confirmed Result:
- Best epoch:
  - `1 / 8`
- Best selection metric:
  - soundscape-only macro ROC-AUC `0.8153869305`
- Best overall macro ROC-AUC observed during the run:
  - `0.9617269745`
- Train / valid rows:
  - `27032 / 9046`
- Pseudo rows / files used:
  - `22188 / 5646`
- Teacher folds:
  - `[0, 2, 3]`
- Mean pseudo confidence:
  - `0.5264`
- `p75` pseudo confidence:
  - `0.6425`

Direct Comparison To `exp_011` Fold 1:
- `exp_011` soundscape-only macro ROC-AUC:
  - `0.8042338925`
- `exp_014` soundscape-only macro ROC-AUC:
  - `0.8153869305`
- Delta:
  - `+0.0112`

Interpretation:
- This is a second consecutive positive fold over the supervised `exp_011` baseline.
- The gain is smaller than on fold `0`, but still real.
- Unlike fold `0`, the best epoch arrived immediately at epoch `1`, and later epochs never exceeded it.
- That likely means the branch is more sensitive to pseudo-label noise on this split and saturates very early.

Two-Fold Readout:
- `exp_014` folds `0-1` soundscape-only macro ROC-AUC:
  - `0.8684 / 0.8154`
- mean:
  - `0.8419`
- `exp_011` folds `0-1` comparison mean:
  - `0.8275`
- mean delta:
  - `+0.0144`

Practical Conclusion:
- `exp_014` is now past the point of being a one-fold curiosity.
- It is still too early for Kaggle promotion, but fold `2` has become a meaningful gate rather than a formality.

## Fold 2 Update

Confirmed Result:
- Best epoch:
  - `6 / 8`
- Best selection metric:
  - soundscape-only macro ROC-AUC `0.8300901176`
- Best overall macro ROC-AUC observed during the run:
  - `0.9707828628`
- Train / valid rows:
  - `27084 / 8994`
- Pseudo rows / files used:
  - `20211 / 5333`
- Teacher folds:
  - `[0, 1, 3]`
- Mean pseudo confidence:
  - `0.4974`
- `p75` pseudo confidence:
  - `0.5939`

Direct Comparison To `exp_011` Fold 2:
- `exp_011` soundscape-only macro ROC-AUC:
  - `0.8543629181`
- `exp_014` soundscape-only macro ROC-AUC:
  - `0.8300901176`
- Delta:
  - `-0.0243`

Interpretation:
- This is the first negative fold for the branch.
- The drop is meaningful enough that `exp_014` can no longer be described as a clean win over `exp_011`.
- It is especially important that the strongest supervised baseline fold (`exp_011` fold `2`) was not matched here.

Three-Fold Readout:
- `exp_014` folds `0-2` soundscape-only macro ROC-AUC:
  - `0.8684 / 0.8154 / 0.8301`
- mean:
  - `0.8380`
- `exp_011` folds `0-2` comparison mean:
  - `0.8365`
- mean delta:
  - `+0.0015`

Practical Conclusion:
- After three folds, `exp_014` is mixed rather than clearly better.
- The branch still looks interesting, but it no longer justifies immediate Kaggle promotion.
- Fold `3` is now the real decision point.

## Fold 3 Update

Confirmed Result:
- Best epoch:
  - `3 / 8`
- Best selection metric:
  - soundscape-only macro ROC-AUC `0.7702235931`
- Best overall macro ROC-AUC observed during the run:
  - `0.9661132492`
- Train / valid rows:
  - `27127 / 8951`
- Pseudo rows / files used:
  - `20595 / 5293`
- Teacher folds:
  - `[0, 1, 2]`
- Mean pseudo confidence:
  - `0.5141`
- `p75` pseudo confidence:
  - `0.6184`

Direct Comparison To `exp_011` Fold 3:
- `exp_011` soundscape-only macro ROC-AUC:
  - `0.7992063670`
- `exp_014` soundscape-only macro ROC-AUC:
  - `0.7702235931`
- Delta:
  - `-0.0290`

Interpretation:
- This is the second clearly negative fold in the branch.
- The loss is slightly larger than on fold `2`, so the four-fold view does not recover the earlier optimism from folds `0-1`.

Four-Fold Readout:
- `exp_014` folds `0-3` soundscape-only macro ROC-AUC:
  - `0.8684 / 0.8154 / 0.8301 / 0.7702`
- mean:
  - `0.8210`
- `exp_011` folds `0-3` comparison mean:
  - `0.8272`
- mean delta:
  - `-0.0061`

Final Conclusion:
- `exp_014` should not be promoted to Kaggle in its current form.
- The branch is valuable as a research result because it shows:
  - pseudo labels can help some folds
  - but the current recipe does not beat the supervised `exp_011` baseline consistently enough
- The operational focus should now return to:
  - `exp_011` as the best native submit path
  - faithful operationalization of a stronger external `0.924` branch
