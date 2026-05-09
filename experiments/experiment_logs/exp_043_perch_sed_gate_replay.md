# exp_043: perch_sed_gate_replay

## Status

First Kaggle run completed. Follow-up cache-control variant scaffolded.

## Goal

Replay the new public `0.945` notebook as closely as possible while making the input resolution robust for both Kaggle and the local repository layout.

## Source Notebooks

- `references/private-notebooks/birdclef-2026-onnx-perch-sequence-modeling-0.944.ipynb`
- `references/private-notebooks/birdclef-2026-gate-fake008-head0015-0.945.ipynb`

## Key Recipe

- ONNX Perch window logits and `1536`-d embeddings.
- Lightweight in-notebook ProtoSSM + MLP probe + ResidualSSM branch.
- Tucker Arrants public distilled SED ONNX ensemble with `sed_fold0.onnx` through `sed_fold4.onnx`.
- Final rank blend with SED weight `0.40`.
- `0.945` gate layer:
  - Proto-only rescue when ProtoSSM is confident and SED is very low.
  - Proto temporal-continuity rescue using a fat-tailed `+/-3` window context.
  - Rare local SED spike rescue.

## Local Assets Resolved

- `data/models/Perch-onnx-for-birdclef+2026/perch_v2.onnx`
- `data/models/Perch-onnx-for-birdclef+2026/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `data/models/Perch-onnx-for-birdclef+2026/labels.csv`
- `data/models/BC2026-Distilled-SED-Public/sed_fold0.onnx` through `sed_fold4.onnx`
- `data/perch_meta/full_perch_meta.parquet`
- `data/perch_meta/full_perch_arrays.npz`

## Notebook

- `notebooks/kaggle_submission_exp_043_perch_sed_gate_replay.ipynb`

## Kaggle Inputs To Attach

- Competition: `BirdCLEF+ 2026`
- Dataset: `perch-onnx-for-birdclef-2026`
- Dataset: `bc2026-distilled-sed-public`
- Dataset: `perch-meta`
- Optional fallback only: Google `bird-vocalization-classifier` Perch v2 TensorFlow model and TF wheels

## Validation

- Static notebook JSON and code-cell syntax check passed.
- Local cells `0-3` execute and resolve the local competition directory and cached metadata.
- Full local execution was not attempted because the attached `onnxruntime` wheel is Linux/cp312 and cannot install on local macOS. The first meaningful runtime validation should be a Kaggle CPU submission run.

## Expected Outcome

This is the highest-upside current branch because it adds a genuinely different SED model family on top of the saturated V18/ProtoSSM path. The first target is to reproduce the public `0.945` behavior before attempting any further improvement.

## Result

- `kaggle_submission_exp_043_perch_sed_gate_replay.ipynb`: public LB `0.943`.

## Interpretation

The branch is a clear success versus the previous stable plateau (`0.929 -> 0.943`), but it is slightly below the referenced `0.945`. The most likely implementation-level difference is cache resolution: the public reference output rebuilt the train Perch cache with ONNX, while this replay can resolve an attached/local `perch-meta` cache. If that cache was produced by a different Perch backend, the train/test feature distribution can drift enough to cost a small amount of LB.

## Follow-up

- `notebooks/kaggle_submission_exp_043a_perch_sed_gate_rebuild_cache.ipynb`
- Same model logic as `exp_043`.
- `USE_EXTERNAL_PERCH_CACHE = False` by default.
- Rebuilds train cache with the currently active backend, matching the public reference path more closely.

## Follow-up Result

- `kaggle_submission_exp_043a_perch_sed_gate_rebuild_cache.ipynb`: public LB `0.942`.

## Updated Interpretation

The cache-rebuild hypothesis did not explain the gap to the public `0.945` reference. Disabling the external cache slightly worsened the result (`0.943 -> 0.942`), so `exp_043` remains the stronger local replay of the `0.945` family. Treat `exp_043a` as a diagnostic negative, not as the new base.

For the next submit, the higher-upside path is still `exp_044`: it ports the actual V8 `0.946` deltas (`apply_per_class_thresholds`, sonotype mirroring, and rare-taxon adaptive suppression). If `exp_044` underperforms, the next controlled variant should combine V8 final postprocess with the `exp_043` cache behavior rather than continuing the rebuild-cache line.
