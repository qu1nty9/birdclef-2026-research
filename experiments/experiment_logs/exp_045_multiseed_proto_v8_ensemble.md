# exp_045: multiseed_proto_v8_ensemble

## Status

Scaffolded for first Kaggle run.

## Motivation

The `exp_044` family repeatedly lands at public LB `0.944`. Small final-postprocess toggles are not moving the score, so this experiment targets a more structural source of variance: the in-notebook ProtoSSM/ResidualSSM branch is trained from scratch on only `708` trusted train-window rows.

## Baseline

- `exp_044b`: public LB `0.944`
- Same Perch ONNX + Tucker SED + V8 final blend family.

## Key Change

Perch and SED are still run once. The ProtoSSM branch is trained for multiple seeds and ensembled before the final V8 blend.

Seeds:

- `17`
- `42`
- `777`

The shared prior + MLP probe branch is computed once. For each seed, the notebook trains:

- `LightProtoSSM`
- per-seed calibration thresholds
- `ResidualSSM`

Then it writes:

- `submission_protossm_seed17.csv`
- `submission_protossm_seed42.csv`
- `submission_protossm_seed777.csv`
- `protossm_multiseed_summary.csv`
- final `submission_protossm.csv` as mean probability ensemble

The final submission still uses the V8 rank/gate blend with the distilled SED branch.

## Notebook

- `notebooks/kaggle_submission_exp_045_multiseed_proto_v8_ensemble.ipynb`

## Expected Effect

This is a low-risk attempt to reduce seed variance in the most stochastic branch of the current solution. Expected upside is modest but real if the branch is variance-limited: roughly `+0.001` to `+0.003`. Runtime should increase only by the extra ProtoSSM/ResidualSSM training passes, not by repeating Perch or SED inference.

## Validation

- Static notebook JSON and code-cell syntax check passed.
- Local setup/data cells `0-4` execute successfully.
- Full local execution was not attempted because Kaggle ONNX Runtime wheels are Linux/cp312 and cannot install on local macOS.
