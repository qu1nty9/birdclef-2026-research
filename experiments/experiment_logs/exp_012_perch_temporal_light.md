# Experiment Log

## Metadata

- Experiment ID:
  - `exp_012`
- Experiment Name:
  - `perch_temporal_light`
- Parent Experiment:
  - `0.924` Pantanal Distill / ProtoSSM Perch reference
- Date:
  - `2026-03-25`
- Notebook:
  - `notebooks/exp_012_perch_temporal_light.ipynb`
- Output Directory:
  - `experiments/outputs/exp_012_perch_temporal_light`

## Objective

- Rebuild the strongest new Perch idea in a lighter, research-friendly form.
- Use cached `Perch v2` embeddings and logits as the representation space.
- Model the full `12 x 5s` sequence of each soundscape file instead of treating windows independently.
- Test whether in-model temporal reasoning plus `site/hour` embeddings beats the older `priors + probe` downstream stack.

## Reference Assets

- `references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.924.ipynb`
- `data/perch_meta/full_perch_meta.parquet`
- `data/perch_meta/full_perch_arrays.npz`

## Planned Model

- Input:
  - cached `Perch v2` embeddings and logits
- Sequence unit:
  - one full `60s` file represented as `12` consecutive `5s` windows
- Core temporal block:
  - lightweight bidirectional selective SSM
- Heads:
  - direct temporal classifier
  - prototype-similarity classifier
  - family auxiliary head
- Fusion:
  - per-class gated fusion with raw Perch logits
- Metadata:
  - learned `site` and `hour` embeddings injected inside the model

## Deliberate Simplifications

- No residual second-pass SSM yet
- No long leaderboard-oriented postprocess chain yet
- No threshold sharpening yet
- No heavyweight ensemble yet

## Validation Plan

- Use grouped OOF by `site` over the trusted fully labeled soundscape files cached in `perch_meta`
- Export:
  - `fold_summary.csv`
  - `oof_outputs.npz`
  - `file_meta.csv`
  - `result_snapshot.json`

## Setup Status

- Notebook scaffolded and filled on `2026-03-25`
- AST validation passed for all code cells
- Local setup sanity is logically complete:
  - taxonomy / sample submission parsing
  - soundscape label expansion
  - `perch_meta` alignment
  - file-level reshaping
  - grouped OOF / final-train switches

## Immediate Next Step

- Run the first grouped OOF pass for `exp_012`
- In parallel, finish `exp_011` fold `3` so the current best native branch can be promoted to a `4-fold` Kaggle submission

## First Grouped OOF Result

- Raw Perch AUC on the cached trusted files:
  - `0.7390178442`
- Grouped pooled OOF AUC of `exp_012`:
  - `0.6247943480`
- Delta vs raw:
  - `-0.1142234962`
- Fold summary:
  - fold `1`: `0.8336697497`
  - fold `2`: `0.6862099010`
  - fold `3`: `0.8705334548`
  - fold mean: `0.7968043685`

## Interpretation

- The first OOF run is a clear negative result.
- Fold-level scores looked decent, but the honest pooled OOF score is materially worse than raw Perch.
- So the current light temporal stack is not yet a valid downstream upgrade.

## Next Step

- Do not promote this branch to Kaggle in its current form.
- Build a simpler ablation sequence to identify which piece is hurting:
  - sequence MLP without SSM
  - temporal model without prototype head
  - temporal model without gated fusion
