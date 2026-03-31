# exp_015g — smoke_submit

## Purpose

Minimal Kaggle code-competition smoke notebook to separate:
- environment / run-mode timeouts
- from actual model-pipeline timeouts

## Notebook

- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/exp_015g_smoke_submit.ipynb`

## Behavior

- resolves `sample_submission.csv` from attached competition data
- prints lightweight environment diagnostics
- writes a valid `submission.csv`
- writes `/kaggle/working/smoke_submit_runtime.json`

## Intended Use

- run with the exact same Kaggle settings that previously timed out
- if this notebook also times out, the blocker is almost certainly Kaggle mode / hardware / UI configuration rather than model code
- if this notebook succeeds, heavier submit notebooks should be debugged against the same settings and inputs

## Outcome

- The smoke submit completed successfully under the same Kaggle settings that previously timed out for heavier notebooks.
- User-reported end-to-end completion time was about `26 minutes`.
- Interpretation:
  - Kaggle run mode itself is valid
  - there can still be substantial queue / platform / scoring overhead even for a trivial submit
  - remaining timeout issues should now be treated as notebook-pipeline issues rather than environment-mode issues
