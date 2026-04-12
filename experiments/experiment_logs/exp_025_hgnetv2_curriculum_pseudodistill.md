# exp_025_hgnetv2_curriculum_pseudodistill

## Goal

Build the next native pseudo/distillation branch by changing the **schedule** rather than the pseudo cache itself.

Notebook:

- `notebooks/exp_025_hgnetv2_curriculum_pseudodistill.ipynb`

## Why This Exists

`exp_022` gave a very clean diagnosis:

- the pseudo cache quality itself was already strong
- the best epoch still stayed in the `labeled_only` phase
- once pseudo supervision was switched on, soundscape AUC declined

That points to a schedule problem more than a filtering problem.

## Design

`exp_025` keeps the same OOF-teacher pseudo generation idea as `exp_022`, but changes how pseudo rows enter training:

- later pseudo start: `epoch 4`
- weaker final pseudo influence: `pseudo_loss_weight = 0.10`
- curriculum ramp over `3` epochs
- start from only the top-ranked `20%` pseudo rows
- gradually expand toward the full retained pseudo pool
- keep the ranking signal explicit via `pseudo_rank_score`

## Main Recipe Changes vs `exp_022`

- `epochs: 6 -> 8`
- `pseudo_start_epoch: 2 -> 4`
- `pseudo_loss_weight: 0.20 -> 0.10`
- new curriculum controls:
  - `pseudo_curriculum_ramp_epochs = 3`
  - `pseudo_curriculum_weight_start = 0.20`
  - `pseudo_curriculum_keep_start = 0.20`
  - `pseudo_curriculum_keep_end = 1.00`
  - `pseudo_min_curriculum_rows = 512`

## Intended Interpretation

If `exp_025` beats the last `labeled_only` stage from `exp_022`, then native pseudo/distillation is still alive and the missing ingredient was the introduction schedule.

If it still fails, that is much stronger evidence that the current HGNetV2 pseudo line is largely exhausted in this form.

## Fold 0 Result

- teacher folds: `[1, 2, 3]`
- pseudo rows / files: `4895 / 2310`
- pseudo start: `epoch 4`
- curriculum ramp epochs: `3`
- initial pseudo keep ratio / weight scale: `0.20 / 0.20`

Training result:

- best epoch: `1`
- best selection metric: `0.851456`
- best soundscape macro AUC: `0.851456`
- best macro AUC: `0.971498`

Comparison:

- vs `exp_022` fold `0`: effectively unchanged on the selection metric (`0.851456` vs `0.851456`)
- vs `exp_011` fold `0`: still only about `+0.000603`

Most important behavioral signal:

- the best epoch still stayed in the `labeled_only` phase
- delaying pseudo start from `epoch 2` to `epoch 4` did not change that
- once pseudo curriculum activated, the soundscape metric still dropped:
  - epoch `4`: `0.838077`
  - epoch `5`: `0.840747`
  - epoch `6`: `0.838425`
  - epoch `7`: `0.843304`
  - epoch `8`: `0.842653`

Main interpretation:

- the cleaner curriculum schedule did not rescue the native pseudo line
- it reproduced the same causal picture as `exp_022`:
  - labeled-only training reaches the best checkpoint
  - pseudo supervision still hurts after activation
- this is stronger evidence that the bottleneck is no longer just pseudo timing
- the current HGNetV2 pseudo/distillation line now looks largely exhausted in this form

Practical consequence:

- do not continue the current `exp_025` recipe to more folds
- if native pseudo is revisited again, it should be through a more substantial redesign:
  - two-stage fine-tuning
  - offline distillation targets
  - or a meaningfully different target-domain pseudo construction
