# Experiment Log

## Metadata

- Experiment ID:
  - `exp_012b`
- Experiment Name:
  - `perch_temporal_ablation`
- Parent Experiment:
  - `exp_012`
- Date:
  - `2026-03-25`
- Notebook:
  - `notebooks/exp_012b_perch_temporal_ablation.ipynb`
- Output Directory:
  - `experiments/outputs/exp_012b_perch_temporal_ablation`

## Objective

- Diagnose the negative `exp_012` result with a simpler ablation ladder.
- Keep the same trusted `perch_meta` file-level setup and grouped OOF protocol.
- Remove prototype and gated-fusion complexity so the next result is easier to interpret.

## Variants

- `raw_perch`
- `pooled_mlp_rawfeat`
- `ssm_linear`
- `ssm_linear_rawfeat`

## Planned Decision Rule

- Primary metric:
  - grouped pooled OOF macro ROC-AUC
- Secondary view:
  - per-fold OOF macro ROC-AUC

## Interpretation Guide

- If `pooled_mlp_rawfeat` beats raw Perch:
  - there is still usable signal in simple file-level context even without the heavier temporal stack
- If `ssm_linear` beats `pooled_mlp_rawfeat`:
  - temporal modeling itself is helping
- If `ssm_linear_rawfeat` is best:
  - temporal modeling plus raw-score conditioning is the most viable repaired Perch path
- If none beat raw Perch:
  - the immediate Perch direction should be deprioritized relative to the stabilized `exp_011` native branch

## Setup Status

- Notebook created and AST-clean on `2026-03-25`
- Uses the same aligned `perch_meta` tensors as `exp_012`
- First grouped OOF run pending

## First Grouped OOF Result

- `raw_perch`:
  - `0.7390178442`
- `pooled_mlp_rawfeat`:
  - `0.6175990663`
- `ssm_linear`:
  - `0.3355242629`
- `ssm_linear_rawfeat`:
  - `0.4285695910`
- Best variant:
  - `raw_perch`

## Interpretation

- The ablation ladder did its job: it showed that the current failure is not limited to the prototype / gated-fusion logic from `exp_012`.
- Even much simpler learned downstream variants still lose to raw Perch on grouped pooled OOF.

## Next Step

- Do not promote the current local Perch branch to Kaggle.
- Treat `exp_011` as the main optimization branch again unless a materially different Perch recipe is introduced.
