# exp_018a — texture_specialist_oof

## Notebook

- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/exp_018a_texture_specialist_oof.ipynb`

## Status

- scaffolded
- AST-validated
- executed locally in safe setup mode with `RUN_TRAINING=False`

## Setup Snapshot

- target taxa: `Amphibia + Insecta`
- target classes: `63`
- total target rows: `1017`
- `train_audio` rows: `650`
- `soundscape_clip` rows: `367`

## Fold 0 Overview

- train rows: `748`
- valid rows: `269`
- train soundscape rows: `297`
- valid soundscape rows: `70`

## Warm Start Plan

- backbone: `hgnetv2_b0.ssld_stage2_ft_in1k`
- target head: `63` specialist classes
- optional warm start from:
  - `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_011_hgnetv2_soundscape_supervised/fold_00/best_model.pt`

## First Interpretation

- The specialist branch is intentionally narrow and light:
  it keeps the proven `exp_011` training recipe, but trims the label space down to the texture-heavy taxa identified by `exp_017`.
- The first useful question is not leaderboard score yet.
- The first useful question is whether target-only soundscape macro AUC improves materially on `Amphibia/Insecta` compared with the generic native branch.

## Next Step

- Run fold `0` with:
  - `RUN_TRAINING = True`
  - `CFG.fold = 0`
  - `RESUME_TRAINING = True`
- Then compare classwise target-domain metrics against the weak-taxa report from `exp_017`.

## Fold 0 Result

- best epoch: `9`
- best macro AUC: `0.9198`
- best target soundscape macro AUC: `0.8056`
- target soundscape scored classes: `27`
- train / valid rows: `748 / 269`
- train / valid soundscape rows: `297 / 70`
- device: `mps`

## First Comparison Against Generic exp_011

- On the same `Amphibia + Insecta` target taxa, the generic `exp_011` fold `0` soundscape-only macro AUC was about `0.7934`.
- So the first specialist fold improved over the generic native branch by about `+0.0122`.
- Important caveat:
  - `exp_018a` scored `27` target soundscape classes
  - the generic `exp_011` comparison on the same fold had only `11` scorable target classes in its stored soundscape subset
- Even with that caveat, the initial signal is clearly positive enough to justify at least one more fold before deciding whether this becomes a real targeted correction branch.

## Fold 1 Result

- best epoch: `10`
- best macro AUC: `0.9231`
- best target soundscape macro AUC: `0.8254`
- target soundscape scored classes: `25`
- train / valid rows: `756 / 261`
- train / valid soundscape rows: `279 / 88`
- device: `mps`

## Two-Fold Comparison Against Generic exp_011

- same-taxa soundscape-only comparison against stored generic `exp_011`:
  - fold `0`: `0.8056` vs `0.7934` (`+0.0122`)
  - fold `1`: `0.8254` vs `0.7720` (`+0.0534`)
- two-fold mean:
  - `exp_018a`: `0.8155`
  - same-taxa generic `exp_011`: `0.7827`
  - delta: `+0.0328`

## Current Interpretation

- The specialist branch is no longer just a one-fold curiosity.
- Across folds `0-1`, it is consistently ahead of the generic native branch on the target texture taxa.
- This is already strong enough to justify continuing to at least `fold 2`.
- If fold `2` stays positive, the branch becomes a credible candidate for a later targeted correction layer on top of `exp_015d`.

## Fold 2 Result

- best epoch: `7`
- best macro AUC: `0.9527`
- best target soundscape macro AUC: `0.7914`
- target soundscape scored classes: `18`
- train / valid rows: `780 / 237`
- train / valid soundscape rows: `279 / 88`
- device: `mps`

## Three-Fold Comparison Against Generic exp_011

- same-taxa soundscape-only comparison against stored generic `exp_011`:
  - fold `0`: `0.8056` vs `0.7934` (`+0.0122`)
  - fold `1`: `0.8254` vs `0.7720` (`+0.0534`)
  - fold `2`: `0.7914` vs `0.8293` (`-0.0380`)
- three-fold mean:
  - `exp_018a`: `0.8075`
  - same-taxa generic `exp_011`: `0.7983`
  - delta: `+0.0092`

## Updated Interpretation

- The specialist branch is still slightly ahead on average across folds `0-2`, but the signal is now much weaker than after folds `0-1`.
- `fold 2` shows that the branch is not yet a clear, stable upgrade over the generic native baseline.
- This is still valuable:
  - the idea did not collapse
  - but it now looks like a conditional or fragile improvement rather than an automatic promotion candidate
- The right next step is `fold 3`, not Kaggle.

## Fold 3 Result

- best epoch: `10`
- best macro AUC: `0.9218`
- best target soundscape macro AUC: `0.8004`
- target soundscape scored classes: `33`
- train / valid rows: `767 / 250`
- train / valid soundscape rows: `246 / 121`
- device: `mps`

## Full Four-Fold Comparison Against Generic exp_011

- same-taxa soundscape-only comparison against stored generic `exp_011`:
  - fold `0`: `0.8056` vs `0.7934` (`+0.0122`)
  - fold `1`: `0.8254` vs `0.7720` (`+0.0534`)
  - fold `2`: `0.7914` vs `0.8293` (`-0.0380`)
  - fold `3`: `0.8004` vs `0.7548` (`+0.0456`)
- four-fold mean:
  - `exp_018a`: `0.8057`
  - same-taxa generic `exp_011`: `0.7874`
  - delta: `+0.0183`

## Final Interpretation

- The branch finished as a real but not perfectly stable positive signal.
- It is not strong enough to replace the generic native branch globally.
- But it is strong enough to justify a next-stage targeted experiment:
  - either a packaged multi-fold specialist correction branch
  - or a controlled targeted-merge benchmark on `Amphibia/Insecta` columns only
- The most important caveat remains unchanged:
  - scored class counts differ between the specialist evaluation and the generic stored comparison on each fold
  - so this should be treated as a strong directional result, not as a final apples-to-apples leaderboard proxy
