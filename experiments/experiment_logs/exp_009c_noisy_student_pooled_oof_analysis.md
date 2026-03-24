# Experiment Log

## Metadata

- Experiment ID:
  - `exp_009c`
- Experiment Name:
  - `noisy_student_pooled_oof_analysis`
- Parent Experiment:
  - `exp_009`
- Date:
  - `2026-03-24`
- Notebook:
  - `notebooks/exp_009c_noisy_student_pooled_oof_analysis.ipynb`
- Output Directory:
  - `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis`

## Objective

- Build the honest pooled OOF view for `exp_009`.
- Quantify how optimistic the three-fold summary was.
- Test whether lightweight context repair, prior fusion, or temperature-style calibration rescues the branch before another Kaggle attempt.

## Inputs

- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_00`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_01`
- `experiments/outputs/exp_009_noisy_student_pseudolabel/fold_02`
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/sample_submission.csv`
- `data/birdclef-2026/taxonomy.csv`

## Variants Tested

- `raw`
- `weak_priors`
- `old_priors`
- `filemax_all_020`
- `filemax_all_035`
- `smooth_all_020`
- `event_smooth_texture_filemax`
- `event_smooth_texture_filemax_strong`
- fold-safe variant selection across the family above
- fold-safe global temperature scaling for calibration metrics

## Key Results

- Raw pooled OOF macro ROC-AUC:
  - `0.7933963541`
- Raw fold-mean macro ROC-AUC:
  - `0.8704094948`
- Optimism gap:
  - `0.0770131407`
- Best fixed variant:
  - `raw`
- Fold-safe selected macro ROC-AUC:
  - `0.7852925095`
- Temperature-scaled raw macro ROC-AUC:
  - `0.7674671923`
- Raw pooled Brier / ECE:
  - `0.0091121 / 0.00708`
- Temperature-scaled Brier / ECE:
  - `0.0090323 / 0.00369`

## Interpretation

- The pooled OOF view is much less optimistic than the three-fold mean.
- That explains a large part of why raw `exp_009` disappointed on Kaggle.
- Raw remains the best fixed OOF variant.
- Lightweight context repair can improve calibration-style losses on individual folds, but it does not improve pooled OOF over raw.
- Weak priors and old priors both hurt, confirming the earlier fold-level signal.
- Temperature scaling improves reliability metrics but not ranking quality.

## Practical Conclusion

- `exp_009` should not get another leaderboard attempt in lightly modified form.
- The branch remains useful as research signal, but no lightweight inference tweak currently rescues it.
- The next better use of time is a new modeling branch, especially the `HGNetV2-B0 + wav-cache + soundscape-clip` supervised path suggested by the strong `0.856` reference.

## Artifacts

- `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis/fold_results.csv`
- `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis/oof_ablation_results.csv`
- `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis/fold_safe_variant_selection.csv`
- `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis/calibration_results.csv`
- `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis/classwise_auc_comparison.csv`
- `experiments/outputs/exp_009c_noisy_student_pooled_oof_analysis/result_snapshot.json`
