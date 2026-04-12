# exp_023a_v18x_dmodel384_artifact_export

## Goal

Build the first focused heavy research fork from the stable V18 artifact export path by testing the only clearly new active modeling ingredient found in `pantanal-distill-birdclef2026-v18x-dmodel-0.929.ipynb`:

- larger `ProtoSSM d_model = 384`

while preserving the cleaner artifactized engineering shape from the repository.

Notebook:

- `notebooks/exp_023a_v18x_dmodel384_artifact_export.ipynb`

## Why This Exists

Recent reference analysis showed an important nuance:

- many `0.927-0.930` notebooks were near-duplicates of the same monolithic V18 family
- but `pantanal-distill-birdclef2026-v18x-dmodel-0.929.ipynb` contains a genuinely different active training recipe branch
- most of that recipe was already present in our `exp_015c` export path
- the cleanest remaining new ingredient is therefore not another postprocess trick, but the larger `ProtoSSM` width

This makes `exp_023a` a cleaner test than porting yet another monolithic submit notebook.

## Design

`exp_023a` starts from:

- `notebooks/exp_015c_v18_artifact_export.ipynb`

and changes only the highest-value active differences:

- `CFG["proto_ssm"]["d_model"] = 384`
- `CFG["tta_shifts"] = [0, 1, -1, 2, -2]`
- dedicated artifact/output namespace:
  - `exp_023a`
  - `v18x_dmodel384_artifact_export`
  - `exp_023a_v18x_dmodel384_artifacts`

Everything else intentionally stays aligned with the already working V18 export scaffold.

## What Not To Import

This branch explicitly does **not** import the weaker engineering choices from the reference notebook:

- CPU-only execution
- monolithic submit-time training
- hardcoded Kaggle paths
- fixed `0.5` thresholds in the final path

## Validation Status

- notebook scaffolded from `exp_015c_v18_artifact_export.ipynb`
- `ProtoSSM d_model` updated from `320 -> 384`
- wider TTA config copied in for local/OOF benchmarking
- syntax checked with `ast.parse`

## First Run Result

The first export run completed and produced a clean artifact bundle:

- `proto_ssm_state.pt`
- `sklearn_artifacts.pkl`
- `prior_tables.pkl`
- `per_class_thresholds.npy`
- `oof_meta_features.npz`
- `artifacts_manifest.json`
- `artifact_export_logs.json`

Main metrics from `artifact_export_logs.json`:

- `oof_auc_proto = 0.653477`
- `ensemble_auc = 0.961724`
- `mlp_only_auc = 0.961724`
- `ensemble_weight = 0.0`
- `train_time_final = 95.81s`
- `oof_time = 229.88s`
- `proto_parameters = 8,110,842`

Important behavioral signal:

- the larger-width `ProtoSSM` branch did train successfully
- but it was not useful enough to survive the downstream fusion stage
- the final blend collapsed to pure probe/base behavior:
  - `ensemble_weight = 0.0`
  - `ensemble_auc == mlp_only_auc`

Residual behavior:

- `ResidualSSM` was skipped
- wall time had already reached about `9.24` minutes by that stage
- final artifact manifest therefore reports:
  - `has_residual = false`

Threshold artifact:

- learned thresholds were still exported successfully
- threshold mean was about `0.513`

## Interpretation

This is a strong negative local result for the `d_model=384` hypothesis in this exact deployment shape.

The branch did not fail operationally; it failed causally:

- increasing `ProtoSSM` width made the model much larger
- but the resulting `ProtoSSM` OOF signal remained weak
- and the downstream stack chose to ignore it completely

Practical consequence:

- do **not** promote this exact branch into `exp_023b` thin submit
- treat the simple `320 -> 384` width increase as not justified on top of the current V18 artifactized recipe
- if the `v18x` family is revisited later, it should be through a more substantial training redesign rather than width alone

## Intended Interpretation

The first run now points to the second outcome:

- the larger-width `ProtoSSM` idea is not the missing source of gain beyond the already strong `exp_015d` line
