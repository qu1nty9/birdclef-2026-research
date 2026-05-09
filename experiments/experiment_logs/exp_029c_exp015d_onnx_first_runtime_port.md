# exp_029c — exp015d ONNX-first runtime port

Date: 2026-04-13

## Goal

Follow up on `exp_029b`, which still timed out on Kaggle. The working hypothesis is that the notebook never actually activated the ONNX path because `onnxruntime` was not installed from the attached dataset, and the branch silently fell back to the slower TensorFlow Perch path.

## What Changed

- New notebook: `notebooks/kaggle_submission_exp_029c_exp015d_onnx_first_runtime_port.ipynb`
- ONNX-first bootstrap:
  - installs `onnxruntime-*.whl` from the attached ONNX dataset before imports
  - resolves `*.onnx` and matching `labels.csv` directly from the ONNX dataset
- Strict runtime behavior:
  - `REQUIRE_ONNX_PERCH = True` by default
  - if ONNX cannot be activated, the notebook now fails early instead of spending the run on a slow TensorFlow fallback
- TensorFlow is now optional and only used for explicit fallback
- Runtime log path:
  - `/kaggle/working/v18_onnx_first_runtime_port_submit_logs.json`

## Why This Matters

`exp_029b` was meant to test the `bird26` runtime ideas, but if `onnxruntime` was missing, it inherited most of the original TensorFlow runtime cost. `exp_029c` is designed to answer the question much more cleanly:

- either ONNX really activates and we get an honest timing result
- or the notebook fails quickly with a clear missing-input signal

## Expected Inputs

- BirdCLEF competition dataset
- V18 artifacts dataset used by `exp_015d`
- ONNX Perch dataset containing:
  - `perch_v2.onnx`
  - `labels.csv`
  - `onnxruntime-*.whl`

TensorFlow wheels and `perch_v2_cpu` are no longer required for the default ONNX-first path.

## Result

- First Kaggle run completed successfully
- Public LB: `0.929`
- End-to-end runtime: about `23` minutes

## Interpretation

This confirms the engineering hypothesis behind `exp_029c`:

- the earlier `exp_029b` timeout was not a clean argument against ONNX-first Perch
- once `onnxruntime` was installed from the attached dataset and ONNX activation became strict, the runtime problem largely disappeared
- the score matched the stable `exp_015d` public baseline exactly, so the ONNX-first path appears numerically safe enough for practical deployment

This does **not** raise the public-score ceiling by itself, but it is still a major project improvement because it removes a lot of runtime pressure from the strongest current recipe.
