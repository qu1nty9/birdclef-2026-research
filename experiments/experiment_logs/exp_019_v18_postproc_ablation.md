# exp_019_v18_postproc_ablation

## Goal

Build a low-risk postprocessing ablation notebook on top of the fixed `exp_015d` V18 artifact stack.

Notebook:

- `notebooks/exp_019_v18_postproc_ablation.ipynb`

Reference parent:

- `notebooks/exp_015f_v18_calibration_refresh_export.ipynb`
- `notebooks/kaggle_submission_exp_015d_v18_artifact_submit.ipynb`

## Why This Exists

After the recent Kaggle checks:

- `exp_015d = 0.929`
- `exp_015h = 0.920`
- `exp_018d = 0.929`

the highest-value next move is no longer another heavy runtime branch.

Instead, this notebook tests cheap postprocess hypotheses on top of the already strongest artifactized V18 path.

## Design

`exp_019` is intentionally **not** a submit notebook.

It:

1. loads the fixed `exp_015d` artifacts
2. loads cached full-file Perch outputs for trusted labeled soundscapes
3. replays the frozen downstream stack
4. benchmarks a small set of low-risk postprocess variants
5. writes compact comparison outputs

## Current Variant Set

- `manifest_baseline`
- `no_thresholds`
- `no_file_scale`
- `no_rank_aware`
- `no_delta_smooth`
- `topk3_manifest`
- `rank035_manifest`
- `rank045_manifest`
- `delta010_manifest`
- `aves_smooth005_manifest`
- optional TTA variants:
  - `tta_shift_p1`
  - `tta_shift_pm1`

## Outputs

If the notebook is run successfully, it saves:

- `experiments/outputs/exp_019_v18_postproc_ablation/variant_results.csv`
- `experiments/outputs/exp_019_v18_postproc_ablation/report_snapshot.json`

## Validation Status

- notebook scaffolded
- syntax checked with `ast.parse`
- executed by the user on Kaggle-style inputs and produced a first proxy ranking

## First Result

- artifact dataset used:
  - `exp_015c_v18_artifacts`
- calibrators available in this run:
  - no (`use_calibration = False` for all variants)
- number of tested variants:
  - `12`

Best proxy variant by row macro AUC:

- `no_file_scale`
- row macro AUC: `0.993906`
- baseline row macro AUC: `0.993120`
- delta vs baseline: `+0.000786`

Important nuance:

- the same `no_file_scale` variant reduced file-level macro AUC:
  - baseline file macro AUC: `0.992126`
  - `no_file_scale`: `0.991105`

Other notable observations:

- `no_rank_aware` was the second-best row-level variant and preserved file-level macro AUC:
  - row macro AUC: `0.993606`
  - file macro AUC: `0.992126`
- `no_thresholds` was effectively identical to the baseline on this proxy benchmark
- both TTA variants were slightly worse than the manifest baseline

## Practical Interpretation

This is a useful but cautious result:

- there is some signal that the current V18 postprocess may be slightly over-engineered
- the cleanest next cheap public test is not a full sweep, but a very thin submit patch with:
  - either `no_rank_aware`
  - or `no_file_scale`

The safer of those two is probably `no_rank_aware`, because:

- it improves proxy row AUC
- it does not hurt proxy file AUC

`no_file_scale` still has upside, but its mixed row/file behavior makes it higher-risk for the first public check.

## Intended Interpretation

This is a **proxy benchmark**, not a fully honest OOF or hidden-test evaluation.

It should be used to decide whether a very cheap postprocess tweak is worth a later thin submit check, not to replace the leaderboard as the final arbiter.
