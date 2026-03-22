# Experiment Log

Experiment ID:
`exp_008c`

Experiment Name:
`long_context_priors_on_oof`

Date:
`2026-03-22`

Research Question:
Does the long-context native branch actually beat the current short-context native baseline under a strict pooled OOF protocol, or were the earlier fold-local gains too optimistic?

Baseline Reference:
`exp_007_native_priors_on_exp006_oof`

Change Introduced:
Apply the `exp_008b` postprocessing recipe fold-safely on top of `exp_008` folds `0-2`:
- `site/hour/site-hour` priors
- stronger prior fusion on texture classes
- texture-aware smoothing
- pooled OOF evaluation instead of single-fold reading

Dataset:
- `data/birdclef-2026/train_soundscapes_labels.csv`
- `data/birdclef-2026/taxonomy.csv`
- `experiments/outputs/exp_008_long_context_native_sed/fold_00/`
- `experiments/outputs/exp_008_long_context_native_sed/fold_01/`
- `experiments/outputs/exp_008_long_context_native_sed/fold_02/`

Validation Strategy:
- reconstruct grouped filename folds exactly as in `exp_008`
- verify `row_id` alignment between reconstructed fold targets and exported `best_valid_meta.csv`
- compare five variants:
  - `raw`
  - `event_priors_only`
  - `texture_priors_only`
  - `event_texture_priors`
  - `event_texture_priors_smooth`
- report both fold-mean and pooled OOF

Status:
- notebook created
- local run completed

Results:
- Fold-mean raw macro ROC-AUC: `0.8231723312`
- Fold-mean best macro ROC-AUC: `0.8380978440`
- Pooled OOF raw macro ROC-AUC: `0.6682152899`
- Best pooled OOF variant: `event_texture_priors_smooth`
- Best pooled OOF macro ROC-AUC: `0.7004623407`
- Absolute pooled OOF gain vs raw: `+0.0322470508`
- Pooled OOF scored classes: `54`

Comparative Readout:
- `exp_007` pooled OOF best: `0.7109`
- `exp_008c` pooled OOF best: `0.7005`
- `exp_007` pooled OOF raw: `0.6646`
- `exp_008c` pooled OOF raw: `0.6682`

Observations:
- The long-context branch still benefits from the same postprocessing recipe.
- But under honest pooled OOF it does **not** beat `exp_007`.
- This resolves the earlier contradiction:
  - fold-local means looked strong for `exp_008`
  - Kaggle score was weak (`0.707`)
  - pooled OOF now agrees with the Kaggle direction more than the fold means did
- The gap is small at the raw OOF level (`0.6682` vs `0.6646`) and negative at the best OOF level (`0.7005` vs `0.7109`).

Failure Cases:
- Fold means were too optimistic because they hid the harsher pooled class coverage view.
- The long-context branch has not earned another Kaggle attempt yet.

Practical Conclusion:
- `exp_007` remains the best validated native branch.
- `exp_008` is still a useful architectural exploration, but not the next promoted submission path.
- The roadmap should now move to the stronger next modeling jump rather than spending more leaderboard attempts on this branch.

Planned Outputs:
- `experiments/outputs/exp_008c_long_context_priors_on_oof/fold_results.csv`
- `experiments/outputs/exp_008c_long_context_priors_on_oof/oof_ablation_results.csv`
- `experiments/outputs/exp_008c_long_context_priors_on_oof/classwise_auc_comparison.csv`
- `experiments/outputs/exp_008c_long_context_priors_on_oof/best_oof_predictions.parquet`
- `experiments/outputs/exp_008c_long_context_priors_on_oof/result_snapshot.json`

Notebook:
- `notebooks/exp_008c_long_context_priors_on_oof.ipynb`
