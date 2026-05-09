# exp_030 — exp_029c multi-fold texture overlay

Date: 2026-04-13

## Goal

Reopen the strongest remaining score-side patch around the stable `0.929` V18 path, but on top of the new ONNX-first runtime scaffold from `exp_029c`.

The main question is no longer whether the specialist idea can run fast enough. `exp_029c` already solved the runtime pressure. The new question is whether the previous neutral public result from the texture overlay came from an overly thin deployable form:

- one specialist fold only
- conservative overlay weight

## Notebook

- [kaggle_submission_exp_030_exp029c_texture_overlay_multifold.ipynb](/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/kaggle_submission_exp_030_exp029c_texture_overlay_multifold.ipynb)

## Design

Base path:

- `exp_029c` ONNX-first `exp_015d` scaffold

Added specialist branch:

- `exp_018a` `Amphibia + Insecta` HGNetV2 overlay
- accelerated overlay runtime block from `exp_018e`

## Default First-Run Settings

- `EXP018A_FOLD_IDS = (0, 1, 3)`
  - uses the strongest positive specialist folds from `exp_018a`
  - intentionally excludes fold `2`, which was the main local negative fold
- `EXP018A_BLEND_WEIGHT = 0.35`
  - stronger than the earlier guarded public test, but still not an all-in overwrite
- `EXP018A_ACCEL_BACKEND = "torchscript"`
  - keep the runtime path explicit and simple, since `torchscript` already worked in `exp_018e`

## Why This Branch Exists

The previous public texture-overlay line is now split into two separate conclusions:

- `exp_018e` showed that the single-fold overlay can execute cleanly and still stay neutral
- `exp_029c` showed that the V18 path now has a very large runtime margin

That combination makes `exp_030` the first honest submit-facing test of a stronger specialist overlay form.

## Result

- First Kaggle run completed successfully
- Public LB: `0.920`
- ONNX Perch active: yes
- Specialist runtime backend: `torchscript`

## Key Log Signals

- `runtime_port.onnx_backend_active = true`
- `exp018a_runtime_setup.active = true`
- `exp018a_overlay.active = true`
- `exp018a_overlay.fold_ids = [0, 1, 3]`
- `exp018a_overlay.blend_weight = 0.35`

Important calibration readout:

- target mean before overlay: `0.04209`
- specialist target mean: `0.01426`
- target mean after overlay: `0.03235`

## Interpretation

This is a strong negative result for the deployable texture-overlay idea.

The branch did exactly what it was supposed to do operationally:

- fast ONNX-first V18 base
- real multi-fold specialist execution
- no timeout ambiguity

But the public score dropped sharply. That means the line is no longer blocked by runtime; it is blocked by the deployed specialist behavior itself. In practical terms, the specialist predictions are too suppressive relative to the already-strong `exp_029c` / `exp_015d` baseline.
