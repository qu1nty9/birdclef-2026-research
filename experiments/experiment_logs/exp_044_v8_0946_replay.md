# exp_044: v8_0946_replay

## Status

First Kaggle run completed.

## Source

- `references/private-notebooks/birdclef-2026-v8-0.946.ipynb`

## Relationship To exp_043

`exp_043` reproduced the public Perch ONNX + ProtoSSM + distilled SED gate family and scored `0.943`. `exp_044` keeps the robust `exp_043a` resolver/cache behavior but ports the `0.946` V8 deltas.

## Key V8 Deltas

- Adds a global seed cell with seed `42`.
- Keeps external Perch cache disabled by default, so train cache is rebuilt with the active ONNX Perch backend.
- Enables `apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)` in the ProtoSSM branch.
- Uses the V8 final blend:
  - rank blend `60%` ProtoSSM + `40%` SED,
  - Proto-only rescue,
  - Proto temporal-continuity rescue,
  - rare SED spike rescue,
  - sonotype mirroring for the `47158son*` groups,
  - adaptive suppression of weak `Amphibia`, `Mammalia`, and `Reptilia` predictions.

## Notebook

- `notebooks/kaggle_submission_exp_044_v8_0946_replay.ipynb`

## Kaggle Inputs To Attach

- Competition: `BirdCLEF+ 2026`
- Dataset: `perch-onnx-for-birdclef-2026`
- Dataset: `bc2026-distilled-sed-public`
- Optional: `perch-meta` can be attached, but the notebook intentionally does not use it by default.
- Optional fallback only: Google `bird-vocalization-classifier` Perch v2 TensorFlow model and TF wheels.

## Validation

- Static notebook JSON and code-cell syntax check passed.
- Full local execution was not attempted because the Kaggle ONNX Runtime wheel is Linux/cp312 and cannot install on local macOS.

## Expected Outcome

Primary target is reproducing the public `0.946` improvement over `exp_043`. If the score lands below `0.946`, compare against `exp_043a` to separate cache/backend drift from the V8 final postprocess changes.

## Result

- `kaggle_submission_exp_044_v8_0946_replay.ipynb`: public LB `0.944`.

## Interpretation

The V8 deltas are positive relative to `exp_043a` (`0.942 -> 0.944`) and slightly positive relative to `exp_043` (`0.943 -> 0.944`), but they still do not fully reproduce the referenced `0.946`. This suggests the V8 postprocess is useful, while some remaining implementation or environment detail still differs from the source run.

The cleanest next control is not a new family. It is `exp_044b`: keep the V8 final postprocess and ProtoSSM threshold activation, but restore the stronger `exp_043` cache behavior instead of forcing ONNX train-cache rebuild.

## Follow-up Variant

- `notebooks/kaggle_submission_exp_044b_v8_postprocess_external_cache.ipynb`
- Same V8 postprocess and ProtoSSM threshold activation as `exp_044`.
- Sets `USE_EXTERNAL_PERCH_CACHE = True`, restoring the `exp_043` cache behavior.
- Purpose: test whether the stronger `exp_043` cache path plus V8 postprocess closes the remaining gap toward `0.946`.

## Follow-up Result

- `kaggle_submission_exp_044b_v8_postprocess_external_cache.ipynb`: public LB `0.944`.

## Updated Interpretation

The cache path is no longer the main suspect. `exp_044` and `exp_044b` both land at `0.944`, so the V8 postprocess is reproducibly positive but not sufficient to reproduce `0.946` in our environment. The remaining search should be small ablations inside the V8 final block, especially separating sonotype mirroring from rare-taxon adaptive suppression.

## Follow-up Ablation

- `notebooks/kaggle_submission_exp_044c_v8_no_rare_suppression.ipynb`
- Keeps V8 ProtoSSM thresholds, rank/gates, external-cache behavior, and sonotype mirroring.
- Disables only the final adaptive suppression for weak `Amphibia`, `Mammalia`, and `Reptilia` predictions.
- Purpose: check whether the V8 rare-taxon suppression is too aggressive for our replay and is masking a small gain.

## Follow-up Ablation Result

- `kaggle_submission_exp_044c_v8_no_rare_suppression.ipynb`: public LB `0.944`.

## Updated Interpretation After exp_044c

Rare-taxon adaptive suppression is effectively neutral in public LB for this replay. The repeated `0.944` scores from `exp_044`, `exp_044b`, and `exp_044c` suggest that the reproducible gain is coming from the broader V8 recipe, not from this final suppression line. The next clean ablation is to disable sonotype mirroring while keeping ProtoSSM thresholds and the rank/gate blend.

## Follow-up Ablation 2

- `notebooks/kaggle_submission_exp_044d_v8_no_mirror_no_rare_suppression.ipynb`
- Keeps V8 ProtoSSM thresholds and the rank/gate blend.
- Disables both sonotype mirroring and rare-taxon adaptive suppression.
- Purpose: isolate whether hard `47158son*` mirroring is helping, neutral, or harmful relative to the core V8 rank/gate recipe.
