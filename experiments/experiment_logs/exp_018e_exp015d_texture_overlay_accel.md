# exp_018e_exp015d_texture_overlay_accel

## Goal

Give the strongest runtime-blocked specialist overlay idea one cleaner engineering rescue.

Notebook:

- `notebooks/kaggle_submission_exp_018e_exp015d_texture_overlay_accel.ipynb`

Baseline:

- `notebooks/kaggle_submission_exp_018d_exp015d_texture_overlay_guarded.ipynb`

## Why This Exists

`exp_018c` timed out, and `exp_018d` completed but gave no public gain.

The unresolved question from `exp_018d` is still important:

- did the specialist branch actually run and stay neutral,
- or did the guarded runtime path mostly skip the overlay?

## Main Engineering Changes

- keep the same conservative single-fold overlay structure:
  - `EXP018A_FOLD_IDS = (1,)`
  - `EXP018A_BLEND_WEIGHT = 0.25`
- add specialist backend selection:
  - `openvino`
  - `torchscript`
  - eager Torch
- `EXP018A_ACCEL_BACKEND = "auto"` tries:
  - `openvino -> torchscript -> eager`
- support optional prebuilt OpenVINO IR reuse from the attached specialist dataset
- otherwise try inline `ov.convert_model(...)` export into a runtime cache
- write richer runtime logs to:
  - `/kaggle/working/exp_018e_texture_overlay_accel_logs.json`

## Key Runtime Outputs

Expected files in `/kaggle/working`:

- `submission.csv`
- `submission_before_exp018a_overlay.csv`
- optionally `submission_after_exp018a_overlay.csv`
- `exp_018e_texture_overlay_accel_logs.json`

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

EXP018A_ACCEL_BACKEND = "auto"
EXP018A_TRACE_BATCH_WAVES = 32
EXP018A_OPENVINO_NUM_REQUESTS = 2
EXP018A_OPENVINO_CHUNK_ROWS = 128
```

## Decision Logic

If this branch improves over `0.929`, then the old runtime form really was part of the problem.

If it again lands at `0.929`, the log becomes crucial:

- `runtime_backend = openvino` or `torchscript` with `active = True` would mean the overlay itself is probably neutral
- repeated fallback or skip behavior would mean the idea is still deployment-limited rather than model-limited

## Current Status

- first Kaggle run completed
- AST-valid

## First Kaggle Result

- Public LB:
  - `0.929`
- Runtime setup:
  - `requested_backend = auto`
  - `selected_backend = torchscript`
  - `openvino` fallback reason:
    - `ModuleNotFoundError: No module named 'openvino'`
- Overlay execution:
  - `active = true`
  - `elapsed_seconds ≈ 18.89`
  - `elapsed_before_overlay ≈ 348.43`
- Snapshots:
  - `submission_before_exp018a_overlay.csv`
  - `submission_after_exp018a_overlay.csv`

## Interpretation

This is the key value of `exp_018e`:

- it removes the main ambiguity left by `exp_018d`
- the specialist branch really executed
- and it still did not improve over `exp_015d`

So the current single-fold deployable texture-overlay form now looks genuinely neutral on public LB, not just runtime-blocked.
