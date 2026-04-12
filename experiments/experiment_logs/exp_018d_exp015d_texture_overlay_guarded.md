# exp_018d_exp015d_texture_overlay_guarded

## Goal

Create a timeout-safe follow-up to the first Kaggle-facing texture overlay attempt.

Notebook:

- `notebooks/kaggle_submission_exp_018d_exp015d_texture_overlay_guarded.ipynb`

Baseline:

- `notebooks/kaggle_submission_exp_018c_exp015d_texture_overlay.ipynb`

## Why This Exists

The first Kaggle-facing `exp_018c` overlay timed out.

Interpretation:

- the idea is still locally justified by `exp_018b`
- but the first public implementation was too heavy when added on top of full `exp_015d`

So the next goal is not “more overlay”, but “learn whether any overlay can survive code-competition runtime at all”.

## Main Safety Changes

- save a valid pure `exp_015d` `submission.csv` snapshot before the specialist branch
- reduce the specialist branch to the single strongest fold:
  - `EXP018A_FOLD_IDS = (1,)`
- reduce the first public specialist weight:
  - `EXP018A_BLEND_WEIGHT = 0.25`
- increase per-batch specialist throughput modestly:
  - `EXP018A_BATCH_FILES = 12`
- add wall-time guards:
  - `EXP018A_MAX_START_WALL_SECONDS = 2100`
  - `EXP018A_ABORT_WALL_SECONDS = 4200`
- if the overlay overruns, keep the valid baseline submission instead of timing out blindly

## Runtime Artifacts

Expected files in `/kaggle/working`:

- `submission.csv`
- `submission_before_exp018a_overlay.csv`
- optionally `submission_after_exp018a_overlay.csv`
- `exp_018d_texture_overlay_logs.json`

## First Kaggle Configuration

```python
EXP018A_MODEL_DATASET_HINT = "birdclef-exp018a-texture-specialist-4fold"
RUN_EXP018A_OVERLAY = True
EXP018A_BLEND_WEIGHT = 0.25
EXP018A_FOLD_IDS = (1,)
EXP018A_BATCH_FILES = 12
EXP018A_MAX_START_WALL_SECONDS = 2100
EXP018A_ABORT_WALL_SECONDS = 4200
EXP018A_SAVE_BASELINE_FIRST = True
```

## Decision Logic

If this branch still times out, the practical conclusion is strong:

- runtime HGNetV2 overlay on top of `exp_015d` is too expensive in current code-competition constraints
- future texture correction should move toward:
  - thinner feature-level correction
  - artifactized specialist outputs
  - or a different postprocess-style correction path

If it completes, then the next step is a conservative weight sweep:

- `0.25`
- `0.35`
- `0.45`

## First Kaggle Result

- Public LB: `0.929`

Interpretation:

- this exactly matches the current baseline `exp_015d = 0.929`
- so the guarded overlay produced no immediate leaderboard gain
- without the runtime log we cannot tell whether:
  - the overlay branch was skipped by the wall-time guards
  - or the overlay actually ran but was neutral on public LB

Practical decision:

- keep `exp_015d` as the main production submit path
- treat `exp_018d` as a useful no-gain control rather than the next promoted submission branch
