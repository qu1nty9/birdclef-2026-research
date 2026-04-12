# exp_022_hgnetv2_oof_pseudodistill_agreement

## Goal

Build the next serious native pseudo/distillation branch on top of the strong `exp_011` HGNetV2 baseline, but with a stricter OOF-teacher recipe than `exp_014/014b`.

Notebook:

- `notebooks/exp_022_hgnetv2_oof_pseudodistill_agreement.ipynb`

## Why This Exists

The recent landscape is now quite clear:

- `exp_015d` remains the strongest external production path
- `exp_019`, `exp_020`, and `exp_021` mostly closed the thin patch / overlay lines
- the strongest remaining research branch is therefore no longer a tiny postprocess tweak
- it is a cleaner second attempt at native pseudo/distillation

`exp_014` and `exp_014b` already taught us something important:

- pseudo labels can help on top of strong `HGNetV2`
- but the branch became unstable when medium-confidence pseudo rows had too much influence

So this branch asks a tighter question:

- can an OOF pseudo/distillation recipe become more stable if we require teacher agreement and soften pseudo weighting

## Design

`exp_022` starts from the same strong native base as `exp_014b`:

1. reuse the `exp_011` grouped supervised frame
2. initialize each fold from the matching `exp_011` checkpoint
3. generate pseudo labels only from the other teacher folds
4. keep only pseudo windows that satisfy:
   - minimum ensemble confidence
   - minimum teacher vote count on the top class
   - maximum teacher disagreement (`teacher_top_std`)
5. use a probability power transform and softer pseudo weighting instead of a more aggressive pseudo pool

## Main Recipe Changes vs `exp_014b`

- lower pseudo cache aggressiveness:
  - `max_pseudo_segments_per_file = 3`
  - `max_pseudo_segments_total = 8000`
- slightly stronger denoising:
  - `pseudo_power = 1.25`
- explicit teacher-agreement filtering:
  - `pseudo_vote_threshold = 0.30`
  - `pseudo_min_teacher_votes = 2`
  - `pseudo_max_teacher_std = 0.18`
- softer pseudo influence:
  - `pseudo_loss_weight = 0.20`
  - `pseudo_sampler_power = 1.5`

## Pseudo Metadata Additions

The pseudo cache now also records:

- `teacher_vote_count`
- `teacher_vote_ratio`
- `teacher_top_std`
- `pseudo_rank_score`
- `pseudo_top_label`

This should make later diagnostics much cleaner than in `exp_014/014b`.

## Validation Status

- notebook scaffolded from `exp_014b`
- syntax checked with `ast.parse`
- operational fix applied:
  - if `pseudo_manifest.parquet` is missing, the notebook now rebuilds it automatically even when `RUN_PREPARE = False`
- operational fix applied:
  - `pseudo_top_label` now resolves from `CLASSES`, matching the notebook's actual class-list variable name during pseudo generation
- fold `0` pseudo generation and training have now completed end-to-end

## Fold 0 Result

- teacher folds: `[1, 2, 3]`
- pseudo rows / files: `4895 / 2310`
- pseudo retain rate: `3.85%`
- pseudo confidence mean / p75: `0.6508 / 0.7890`
- teacher agreement mean: `0.9927`
- teacher top-std mean: `0.1004`

Training result:

- best epoch: `1`
- best selection metric: `0.851456`
- best soundscape macro AUC: `0.851456`
- best macro AUC: `0.970205`

Comparison:

- vs `exp_011` fold `0`: `0.851456 - 0.850852 = +0.000603`
- vs `exp_014b` fold `0`: `0.851456 - 0.868258 = -0.016802`

Most important behavioral signal:

- epoch `1` was still `labeled_only`
- pseudo labels only started at epoch `2`
- once pseudo training began, the soundscape metric declined monotonically:
  - epoch `2`: `0.845814`
  - epoch `3`: `0.842588`
  - epoch `4`: `0.843165`
  - epoch `5`: `0.831291`
  - epoch `6`: `0.835010`

Main interpretation:

- the pseudo cache is indeed much cleaner than earlier branches
- but the current pseudo/distillation recipe still does not help once pseudo supervision is activated
- this is strong evidence that the remaining issue is no longer simple pseudo noise
- it is more likely a mismatch in how pseudo supervision is introduced or weighted against the already strong supervised branch

Practical consequence:

- do not continue this exact `exp_022` recipe to more folds
- if we revisit the native pseudo line later, it should be with a materially different schedule:
  - later pseudo start
  - much weaker pseudo weight
  - or a two-stage fine-tuning design instead of immediate mixed sampling

## Intended Interpretation

If this branch gives a cleaner first-fold signal than `exp_014b`, then pseudo/distillation remains the strongest next native research direction.

If it still behaves inconsistently, then we should treat the current HGNetV2 pseudo line as mostly exhausted and move future research effort elsewhere.
