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
- Pending first run
- Kaggle leaderboard score: `n/a`

Training Time:
- Pending first run

Observations:
- The notebook has been created at `notebooks/exp_004_soundscape_finetuning.ipynb`.
- This is the first native branch that directly targets the soundscape domain rather than isolated recordings.
- The design intentionally absorbs only the most transferable ideas from the new training references:
  - masked secondary-label handling
  - soundscape background mixing
- Heavier ideas such as full mmap preprocessing, PCEN replacement, and target-domain pseudo-labeling are intentionally deferred to later experiments.

Failure Cases:
- Pending first run

Next Experiment Ideas:
- Compare the best `exp_004` checkpoint directly against `exp_003`
- Add `site/hour` priors and texture-aware postprocessing as the next native hybrid step
- Decide later whether a pseudo-label branch is worth the added submission complexity
