# Experiment Log

Experiment ID:
`exp_009`

Experiment Name:
`noisy_student_pseudolabel`

Date:
`2026-03-22`

Research Question:
Can a fold-safe noisy-student branch improve the native soundscape model by turning unlabeled `train_soundscapes` windows into training data rather than relying only on supervised labeled rows and inference-time priors?

Baseline Reference:
`exp_007_native_priors_on_exp006_oof`

Change Introduced:
Create the first repository-native pseudo-label training notebook with:
- teacher ensemble from `exp_006` folds that exclude the current validation fold
- pseudo-label generation on unlabeled `train_soundscapes` windows
- metadata-prior and texture-aware teacher postprocessing before pseudo-label caching
- probability power-transform denoising
- confidence-weighted pseudo sampling
- student training on labeled soundscapes + `train_audio` replay + pseudo-labeled soundscape windows

Dataset:
- `data/birdclef-2026/train_soundscapes/`
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/train.csv`
- `data/birdclef-2026/sample_submission.csv`
- `data/birdclef-2026/taxonomy.csv`
- `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_00/`
- `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_01/`
- `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_02/`
- `experiments/outputs/exp_002_train_audio_reproduction/best_model.pt`

Validation Strategy:
- keep grouped soundscape validation fold-safe through `GroupKFold`
- exclude the current validation files from the pseudo-label teacher pool
- exclude already labeled windows from pseudo candidates
- warm-start the student from the current fold checkpoint when available
- evaluate student improvement only on the held-out labeled soundscape fold

Status:
- notebook created
- safe setup validated in the project `.venv`
- pseudo generation completed for fold `0`
- first full training run completed for fold `0`

Setup Validation Readout:
- Notebook: `notebooks/exp_009_noisy_student_pseudolabel.ipynb`
- Fold checked during setup: `0`
- Resolved teacher folds: `[1, 2]`
- Pseudo candidates before filtering: `127157`
- Resume/init mode during safe run: `init_from_exp006_fold`
- Student init checkpoint: `experiments/outputs/exp_006_soundscape_finetuning_v2/fold_00/best_model.pt`

Pseudo Generation Readout:
- Fold completed: `0`
- Manifest rows: `127896`
- Pseudo candidates before filtering: `127157`
- Kept pseudo rows: `69593`
- Keep rate vs pseudo candidates: `54.73%`
- Pseudo files with at least one kept segment: `9552`
- Teacher folds used: `[1, 2]`
- Mean pseudo confidence: `0.6552`
- Median pseudo confidence: `0.6797`
- `p75` pseudo confidence: `0.9063`
- `p90` pseudo confidence: `0.9789`
- Max pseudo confidence: `0.99995`
- Probability tensor shape: `(69593, 234)`
- Per-file pseudo cap check: top files have `8` kept windows, matching `max_pseudo_segments_per_file=8`

Artifacts Planned:
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/soundscape_manifest.parquet`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/pseudo_label_meta.parquet`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/pseudo_label_probs.npz`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/teacher_summary.json`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/history.csv`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/best_model.pt`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/best_valid_outputs.npz`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/best_valid_meta.csv`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/result_snapshot.json`

Observations:
- The notebook is intentionally based on the stronger short-context native branch rather than `exp_008`, because `exp_008c` did not beat `exp_007` under honest pooled OOF.
- Teacher inference is already wired to reuse the proven native metadata-prior and texture-aware postprocessing before caching pseudo labels.
- The setup check confirmed that the notebook can resolve all required paths and checkpoints without starting the heavy stages.
- The generated pseudo artifacts are internally consistent:
  - `pseudo_label_meta.parquet` and `pseudo_label_probs.npz` have the same row count
  - `pseudo_index` spans `0..69592` without gaps
  - the kept rows cover most soundscape files while still leaving the confidence filter selective
- The confidence distribution looks healthy for a first run:
  - the median is comfortably above the `0.20` threshold
  - the upper tail is strong, which means a confidence-weighted sampler should have useful high-certainty rows to prioritize

Training Readout:
- Fold trained: `0`
- Best epoch: `2 / 6`
- Best macro ROC-AUC: `0.8494899256`
- Best valid loss: `0.0684762717`
- Scored classes: `29`
- Resume/init mode: `init_from_exp006_fold`
- Pseudo rows used: `69593`
- Pseudo files used: `9552`

Epoch Trace:
- epoch `1`: macro ROC-AUC `0.8306`, valid loss `0.0637`
- epoch `2`: macro ROC-AUC `0.8495`, valid loss `0.0685`
- epoch `3`: macro ROC-AUC `0.8343`, valid loss `0.0709`
- epoch `4`: macro ROC-AUC `0.7950`, valid loss `0.0735`
- epoch `5`: macro ROC-AUC `0.8253`, valid loss `0.0693`
- epoch `6`: macro ROC-AUC `0.8028`, valid loss `0.0714`

Practical Conclusion:
- `exp_009` has now passed both major first-run checkpoints:
  - pseudo labels are cached
  - student training did not collapse
- The first full run is a genuine positive signal.
- On this fold, noisy-student training is already stronger than:
  - `exp_006` fold `0`
  - raw `exp_008` fold `0`
  - postprocessed `exp_008b` fold `0`
- The next high-signal sequence should be:
  - apply the proven priors/texture postprocess on top of the exported `exp_009` validation outputs
  - run at least one more fold before trusting the branch for Kaggle promotion

## Fold 1 Update

Confirmed Result:
- Fold `1` best macro ROC-AUC: `0.8768579395`
- Best epoch: `3 / 6`
- Scored classes: `29`
- Best valid loss: `0.0457730132`
- Output dir: `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_01`
- Pseudo rows / files: `69640 / 9569`
- Teacher folds: `[0, 2]`
- Pseudo confidence mean: `0.6845`

Interpretation:
- This is the strongest native fold result in the project so far.
- It improves over fold `0` (`0.8495`) by about `+0.0274`.
- The branch is now past the point of looking like a one-fold anomaly.
- Folds `0-1` average `0.8632`, which is a materially stronger local signal than any earlier native training branch has produced.

Practical Conclusion:
- `exp_009` should stay raw by default.
- The next gate is fold `2`.
- If fold `2` remains strong, the branch deserves its first raw Kaggle submission.

## Fold 2 Update

Confirmed Result:
- Fold `2` best macro ROC-AUC: `0.8848806193`
- Best epoch: `3 / 6`
- Scored classes: `35`
- Best valid loss: `0.0452111292`
- Output dir: `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_02`
- Pseudo rows / files: `70142 / 9669`
- Teacher folds: `[0, 1]`
- Pseudo confidence mean: `0.6458`

Interpretation:
- This is the strongest fold in the branch so far.
- It is also the broadest fold so far, which makes it more trustworthy than folds `0-1`.
- Three-fold summary:
  - fold `0`: `0.8495`
  - fold `1`: `0.8769`
  - fold `2`: `0.8849`
  - mean: `0.8704`
- This is now strong enough evidence that the branch is not a sparse-fold illusion.

Practical Conclusion:
- `exp_009` is ready for its first raw Kaggle submission.
- The next step should be a submit notebook, not another local postprocess pass.
