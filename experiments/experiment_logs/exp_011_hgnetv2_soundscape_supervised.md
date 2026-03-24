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

## Fold 0 Ready State

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

## Practical Notes

- The notebook passes a full setup execution in the project `.venv` with `RUN_TRAINING = False`.
- The first run can proceed directly from `ogg` offset reads.
- `train_audio` wav caching and soundscape clip wav export remain optional accelerators, not hard requirements.

## Next Step

- Run fold `0` and record:
  - best overall validation macro ROC-AUC
  - best soundscape-only validation macro ROC-AUC
  - runtime characteristics with and without wav-cache
