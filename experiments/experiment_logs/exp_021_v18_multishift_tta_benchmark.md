# exp_021_v18_multishift_tta_benchmark

## Goal

Run a controlled benchmark for broader temporal-shift TTA sets on top of the fixed `exp_015d` V18 artifact stack.

Notebook:

- `notebooks/exp_021_v18_multishift_tta_benchmark.ipynb`

## Why This Exists

The current state of the project points to a narrow next low-risk idea:

- `exp_015d = 0.929` remains the strongest production path
- `exp_019a/019b` closed the cheap postprocess-deletion line
- `exp_020a/020b` closed the current deployable texture-correction line
- several reviewed monolithic V18 references still repeated one active difference:
  - broader multi-shift TTA, especially `[0, 1, -1, 2, -2]`

So this branch asks a simple question:

- does a wider temporal-shift TTA set help if we keep everything else in `exp_015d` fixed

## Design

`exp_021` reuses the same fixed `exp_015d` artifact stack and cached full-file Perch outputs as `exp_019`.

It:

1. replays the fixed V18 scorer with different `tta_shifts`
2. keeps the rest of the manifest postprocess unchanged
3. compares row-level and file-level macro AUC under each shift set
4. records replay wall time so we can judge whether any local gain is even deployable

## TTA Sets

The benchmark includes:

- manifest baseline
- `(0, 1)`
- `(0, 1, -1)`
- `(0, 2, -2)`
- `(0, 1, -1, 2, -2)`

## Planned Outputs

- `experiments/outputs/exp_021_v18_multishift_tta_benchmark/report_snapshot.json`
- `experiments/outputs/exp_021_v18_multishift_tta_benchmark/variant_results.csv`

## Validation Status

- notebook scaffolded from the fixed `exp_019` replay path
- syntax checked with `ast.parse`
- Kaggle/local artifact-backed run has now completed on attached `exp_015d` artifacts and full Perch cache

## Result Snapshot

- baseline variant: `manifest_baseline`
- baseline row macro AUC: `0.993120`
- baseline file macro AUC: `0.992126`
- best variant by file-level ordering: `tta_shift_p1`
- `tta_shift_p1` row macro AUC: `0.993041`
- `tta_shift_p1` file macro AUC: `0.992144`
- delta vs baseline:
  - row: `-0.000079`
  - file: `+0.000019`

Variant pattern:

- `(0, 1)` was the only shift set with a visible positive file-level delta
- `(0, 1, -1)`, `(0, 2, -2)`, and `(0, 1, -1, 2, -2)` were effectively neutral or slightly worse
- replay wall time scaled roughly as expected with the number of shifts, but the quality gain stayed negligible

Main interpretation:

- broader multi-shift TTA does **not** show a convincing local win on top of the fixed `exp_015d` stack
- the tiny file-level lift from `(0, 1)` is too small and too offset by a row-level drop to justify an immediate public promotion
- practical consequence:
  - treat this branch as near-neutral / weakly negative for now
  - do not yet spend a Kaggle attempt on a TTA patch unless we specifically want one final cheap confirmation
  - the wider monolithic-reference TTA sets can be considered effectively closed

## Intended Interpretation

If a broader shift set improves file-level AUC without a prohibitive replay-time jump, it becomes the next thin candidate to port into the production path.

If it is neutral or negative, then the multi-shift TTA idea can be closed as another monolithic-reference detail that does not transfer cleanly to our artifactized stack.
