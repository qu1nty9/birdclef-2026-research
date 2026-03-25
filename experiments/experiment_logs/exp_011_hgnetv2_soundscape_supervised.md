# Experiment Log

## Metadata

- Experiment ID:
  - `exp_011`
- Experiment Name:
  - `hgnetv2_soundscape_supervised`
- Parent Experiment:
  - `0.856` HGNetV2 BirdCLEF 2026 reference
- Date:
  - `2026-03-24`
- Notebook:
  - `notebooks/exp_011_hgnetv2_soundscape_supervised.ipynb`
- Output Directory:
  - `experiments/outputs/exp_011_hgnetv2_soundscape_supervised/fold_00`

## Objective

- Rebuild the strong HGNetV2 supervised recipe as a repository-native notebook branch.
- Use a unified dataframe over `train_audio + labeled soundscape clips`.
- Keep optional wav-cache support without making the first local run depend on preconverted datasets.
- Track both overall validation ROC-AUC and soundscape-only validation ROC-AUC.

## Reference Assets

- `references/private-solutions/birdclef2026-score=0.856/birdclef-2026-hgnetv2-b0-baseline-training.ipynb`
- `references/private-solutions/birdclef2026-score=0.856/birdclef-2026-hgnetv2-b0-baseline-inference.ipynb`
- `references/private-solutions/birdclef2026-score=0.856/hgnetv2_b0_baseline.docx`

## Local Setup Facts Confirmed

- `train_soundscapes_labels.csv` stores multi-label segment strings separated by `;`.
- After deduplication and token expansion:
  - `739` unique soundscape segments
  - `3122` per-label soundscape rows
  - `75` target classes
- After contiguous same-label merging:
  - `529` supervised soundscape clips
  - `66` source files

## Fold 0 Result

- Train rows:
  - `26996`
- Valid rows:
  - `9082`
- Train soundscape rows:
  - `393`
- Valid soundscape rows:
  - `136`
- Wav-cache rows:
  - `0`
- Best epoch:
  - `8 / 12`
- Best checkpoint selection metric:
  - soundscape-only macro ROC-AUC `0.8508523324`
- Overall macro ROC-AUC at the selected epoch:
  - `0.9555301905`
- Overall scored classes:
  - `220`
- Soundscape-only scored classes:
  - `42`
- Best validation loss at the selected epoch:
  - `0.0123390177`

## Practical Notes

- The notebook passed full setup validation in the project `.venv` before training.
- Fold `0` completed directly from `ogg` offset reads on `mps`.
- `train_audio` wav caching and soundscape clip wav export remain optional accelerators, not hard requirements.
- The best checkpoint is not the last epoch.
- Overall macro ROC-AUC keeps improving after epoch `8`, but soundscape-only ROC-AUC peaks at epoch `8` and then declines.
- That confirms the branch needs soundscape-aware checkpoint selection.

## Fold 1 Result

- Best epoch:
  - `4 / 12`
- Best checkpoint selection metric:
  - soundscape-only macro ROC-AUC `0.8042338925`
- Overall macro ROC-AUC at the selected epoch:
  - `0.9406223787`
- Overall scored classes:
  - `217`
- Soundscape-only scored classes:
  - `37`
- Best validation loss at the selected epoch:
  - `0.0152798412`

## Fold 2 Result

- Best epoch:
  - `9 / 12`
- Best checkpoint selection metric:
  - soundscape-only macro ROC-AUC `0.8543629181`
- Overall macro ROC-AUC at the selected epoch:
  - `0.9622501589`
- Overall scored classes:
  - `224`
- Soundscape-only scored classes:
  - `52`
- Best validation loss at the selected epoch:
  - `0.0118059591`

## Three-Fold Readout

- Mean overall macro ROC-AUC across folds `0-2`:
  - `0.9528009094`
- Mean soundscape-only macro ROC-AUC across folds `0-2`:
  - `0.8364830477`
- Interpretation:
  - fold `1` is lower, but the branch remains clearly strong
  - fold `2` is broad and nearly matches fold `0`
  - the branch is now stable enough for a first Kaggle test

## Fold 3 Result

- Best epoch:
  - `9 / 12`
- Best checkpoint selection metric:
  - soundscape-only macro ROC-AUC `0.7992063670`
- Overall macro ROC-AUC at the selected epoch:
  - `0.9662019379`
- Overall scored classes:
  - `208`
- Soundscape-only scored classes:
  - `39`
- Best validation loss at the selected epoch:
  - `0.0115137544`

## Four-Fold Readout

- Mean overall macro ROC-AUC across folds `0-3`:
  - `0.9561511665`
- Mean soundscape-only macro ROC-AUC across folds `0-3`:
  - `0.8271638775`
- Interpretation:
  - fold `3` is weaker than folds `0` and `2`, but still strong enough to keep the branch credible
  - the mean drops compared with the first three-fold summary, which is useful because it makes the branch estimate less optimistic before the second Kaggle test
  - we now have the full four-fold checkpoint set, so the next submission can test stability rather than just additional modeling

## Next Step

- First Kaggle submission completed.
- Public leaderboard score: `0.844`
- This beats the old native public best `exp_007 = 0.758` by `+0.086`.
- Notebook: `notebooks/kaggle_submission_exp_011_hgnetv2_3fold.ipynb`
- Dataset package: `submissions/kaggle_datasets/birdclef-exp011-hgnetv2-3fold`
- Second submission assets are now also prepared:
  - notebook: `notebooks/kaggle_submission_exp_011_hgnetv2_4fold.ipynb`
  - dataset package: `submissions/kaggle_datasets/birdclef-exp011-hgnetv2-4fold`
- Second Kaggle submission completed.
- Public leaderboard score (`4-fold`): `0.850`
- Gain over the first `3-fold` Kaggle submission:
  - `0.844 -> 0.850`
  - delta `+0.006`
- Interpretation:
  - fold expansion was worthwhile
  - but the gain is modest, so the branch now looks stabilized rather than massively under-ensembled
- Next decision: compare the stabilized `exp_011` result against the simplified `exp_012` Perch temporal branch and decide whether an ensemble is justified.
