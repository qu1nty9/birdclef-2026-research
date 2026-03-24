# Experiment Log

Experiment ID:
`exp_009b`

Experiment Name:
`noisy_student_priors_postproc`

Date:
`2026-03-23`

Research Question:
Does the proven native metadata-prior and texture-smoothing layer still improve validation quality after the model has already gone through noisy-student pseudo-label training?

Baseline Reference:
`exp_009_noisy_student_pseudolabel`

Change Introduced:
Apply the same postprocessing family used in `exp_005`, `exp_007`, and `exp_008b` on top of the exported fold `0` validation outputs from `exp_009`:
- `event_priors_only`
- `texture_priors_only`
- `event_texture_priors`
- `event_texture_priors_smooth`

Dataset:
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/taxonomy.csv`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/best_valid_meta.csv`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/best_valid_outputs.npz`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00/result_snapshot.json`

Validation Strategy:
- reconstruct the exact grouped soundscape fold `0`
- verify exported `row_id` alignment against the reconstructed validation fold
- fit priors only on the non-validation labeled soundscape rows
- compare raw noisy-student probabilities against four postprocessed variants

Status:
- notebook created
- local run completed

Results:
- Raw macro ROC-AUC: `0.8494899256`
- Best variant: `raw`
- Best macro ROC-AUC after postprocess search: `0.8494899256`
- Delta vs raw: `+0.0000`
- Scored classes: `29`

Ablation Readout:
- `raw`: `0.8495`
- `event_texture_priors`: `0.8112`
- `event_texture_priors_smooth`: `0.8108`
- `texture_priors_only`: `0.8075`
- `event_priors_only`: `0.7964`

Observations:
- This is the first native branch where the old metadata-prior recipe is not just low-value but actively harmful.
- The degradation is large rather than marginal:
  - full priors hurt by about `-0.0383`
  - texture-only priors hurt by about `-0.0420`
  - event-only priors hurt by about `-0.0531`
- The most plausible interpretation is positive:
  - the noisy-student branch is already internalizing more of the site/hour/texture structure during training
  - the old inference-time repair layer is no longer calibrated for this student

Practical Conclusion:
- `exp_009` should now be treated as a raw-model branch first, not a branch that automatically inherits `exp_007` postprocessing.
- The next correct step is not a Kaggle submission from fold `0`.
- The next correct step is to run at least one more `exp_009` fold and see whether the raw branch stays strong without priors.

Artifacts:
- `experiments/outputs/exp_009b_noisy_student_priors_postproc/ablation_results.csv`
- `experiments/outputs/exp_009b_noisy_student_priors_postproc/classwise_auc_comparison.csv`
- `experiments/outputs/exp_009b_noisy_student_priors_postproc/best_valid_predictions.csv`
- `experiments/outputs/exp_009b_noisy_student_priors_postproc/result_snapshot.json`

Notebook:
- `notebooks/exp_009b_noisy_student_priors_postproc.ipynb`
