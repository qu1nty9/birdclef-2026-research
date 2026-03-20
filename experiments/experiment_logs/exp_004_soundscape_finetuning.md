# Experiment Log

Experiment ID:
`exp_004`

Experiment Name:
`soundscape_finetuning`

Date:
`2026-03-21`

Research Question:
Can the native `exp_002` checkpoint recover a strong local soundscape metric once it is finetuned directly on labeled `train_soundscapes` with grouped validation, soundscape background mixing, and a masked-BCE-compatible loss?

Baseline Reference:
`exp_002_train_audio_reproduction`

Change Introduced:
Move from isolated-audio training to a soundscape-aware finetuning stage. The new notebook starts from the best `exp_002` checkpoint, trains on 5-second labeled soundscape windows, keeps validation grouped by soundscape file, and adds soundscape-domain augmentation plus optional replay from `train_audio`.

Dataset:
- Competition data under `data/birdclef-2026/`
- Labeled `train_soundscapes`
- `train_soundscapes_labels.csv`
- Optional capped replay subset from `train_audio`

Feature Extraction:
- 32 kHz mono audio
- Fixed 5-second windows aligned to soundscape labels
- Same mel frontend as the reference EfficientNet-B0 reproduction from `exp_002`

Model Architecture:
- `tf_efficientnet_b0.ns_jft_in1k`
- GeM pooling across frequency
- Attention-style SED head
- Initialized from `experiments/outputs/exp_002_train_audio_reproduction/best_model.pt`

Training Setup:
- Grouped soundscape validation using fully labeled files only
- AdamW optimizer
- Cosine schedule
- Resume support through `last_model.pt` and `best_model.pt`
- Optional replay from `train_audio` so the masked-secondary-label loss can be exercised on true secondary annotations

Augmentations:
- Soundscape background mixing with soft target blending
- Optional replay clips from isolated `train_audio`

Validation Strategy:
- File-grouped local validation on fully labeled soundscape files
- Macro ROC-AUC over classes with both positive and negative examples

Results:
- Best validation score in the first run: `0.779605` macro ROC-AUC at epoch `4 / 6`
- Full first-run trajectory:
  - epoch 1: `0.686650`
  - epoch 2: `0.758403`
  - epoch 3: `0.768842`
  - epoch 4: `0.779605`
  - epoch 5: `0.778186`
  - epoch 6: `0.772404`
- Validation scored `29` classes and skipped `205` classes without positives in the held-out fold
- Kaggle leaderboard score: `n/a`

Training Time:
- Much shorter than `exp_002` because this is a targeted finetuning stage on labeled 5-second soundscape windows rather than a full isolated-audio pretraining run

Observations:
- The notebook has been created at `notebooks/exp_004_soundscape_finetuning.ipynb`.
- This is the first native branch that directly targets the soundscape domain rather than isolated recordings.
- The design intentionally absorbs only the most transferable ideas from the new training references:
  - masked secondary-label handling
  - soundscape background mixing
- Heavier ideas such as full mmap preprocessing, PCEN replacement, and target-domain pseudo-labeling are intentionally deferred to later experiments.
- The first run is encouraging but not yet definitive:
  - the soundscape metric improved clearly through epoch 4
  - the best epoch arrived before the end of the cosine schedule
  - later epochs drifted slightly downward, suggesting the first finetuning stage may already be close to saturation
- The biggest current limitation of the metric is coverage:
  - only `29` classes were scored in the validation fold
  - this is much noisier than the `exp_003` OOF setup and should not yet be treated as a final native-vs-Perch verdict
- Even with that caveat, the first run supports the broader hypothesis that soundscape finetuning is materially more relevant than another isolated-audio-only experiment.

Failure Cases:
- The first grouped validation fold is too sparse to support a strong class-coverage comparison against `exp_003`.
- Because `205` classes had no positives in the held-out fold, fold-to-fold variance is expected to be high.

Next Experiment Ideas:
- Run additional soundscape folds or an honest OOF pass before making strong conclusions
- Add `site/hour` priors and texture-aware postprocessing as the next native hybrid step
- Compare `exp_004 + priors` against `exp_003` on the same soundscape validation protocol
- Decide later whether a pseudo-label branch is worth the added submission complexity
