# Experiment Log

Experiment ID:
`exp_016b`

Experiment Name:
`blend_exp015d_exp011`

Date:
`2026-03-29`

Research Question:
Does the new strongest overall path (`exp_015d = 0.929`) still benefit from a small runtime blend with the strongest repository-native branch (`exp_011 = 0.850`)?

Baseline References:
- `exp_015d = 0.929`
- `exp_011 = 0.850`

Change Introduced:
- keep the original `exp_016` notebooks as historical attempts tied to `exp_015`
- create a new runtime-blend notebook on top of the thin artifactized V18 submit path
- reuse the optimized runtime `exp_011` inference branch from the older `exp_016` work
- default first blend:
  - `0.95 * exp_015d + 0.05 * exp_011`
- after the first runtime timeout, trim the native branch to the strongest local soundscape folds only:
  - `exp_011` folds `0` and `2`
  - larger exp011 GPU batches (`batch_files=24`, `batch_chunks=96`)
  - add wall-time guards so the notebook can safely fall back to pure `exp_015d` instead of timing out again if the hidden run is too large

Notebook:
- `notebooks/kaggle_submission_exp_016b_runtime_blend_exp015d_exp011.ipynb`

Expected Kaggle Inputs:
- competition data `BirdCLEF+ 2026`
- TF 2.20 wheel dataset
- `perch_v2_cpu` SavedModel dataset
- V18 artifact dataset from `exp_015c_v18_artifact_export`
- `exp_011` 4-fold model dataset with `fold_00..fold_03/best_model.pt`

Status:
- runtime-blend notebook created
- notebook AST-validated
- first full 4-fold native runtime blend timed out
- current notebook revised to a lighter 2-fold native runtime blend
- Kaggle rerun completed
- public LB matched `exp_015d` exactly: `0.929`

Why This Branch Matters:
- It is now the cleanest complementarity test after `exp_015d` became the strongest overall path.
- It preserves the timeout-safe artifactized V18 submit route instead of regressing to the older monolithic runtime stack.
- It tells us whether the repository-native HGNet branch still adds useful diversity even after the external path reached `0.929`.

Outcome:
- first successful blend run did not improve public LB over `exp_015d`
- observed public result:
  - `exp_015d = 0.929`
  - `exp_016b = 0.929`

Interpretation:
- the simple runtime blend does not show meaningful public complementarity so far
- this does not prove `exp_011` is useless, but it does mean that a small generic add-on from `exp_011` is not currently a high-priority leaderboard path
- further blend-weight sweeps should now be treated as optional, not as the main next move

Next Step:
- keep `exp_015d` as the default overall Kaggle path
- only revisit nearby blend weights such as `0.03` and `0.08` if we want a low-priority complementarity check
