# Experiment Log

Experiment ID:
`exp_002`

Experiment Name:
`train_audio_reproduction`

Date:
`2026-03-18`

Research Question:
How much of the current public baseline can be recovered by training the same reference architecture on isolated `train_audio` inside this repository?

Baseline Reference:
`exp_001_soundscape_reference_blend`

Change Introduced:
Start a repository-native training baseline using the same mel frontend and EfficientNet-B0 + attention SED architecture that underlies the current reference checkpoints.

Dataset:
- Competition data under `data/birdclef-2026/`
- Isolated recordings from `train_audio`
- Metadata from `train.csv`
- Class order from `sample_submission.csv`

Feature Extraction:
- 32 kHz mono audio
- Fixed 5-second crops
- Mel spectrogram with `n_mels=224`, `n_fft=2048`, `hop_length=512`, `fmax=16000`
- Per-window min-max normalization
- 3-channel mel image replication

Model Architecture:
- `tf_efficientnet_b0.ns_jft_in1k`
- GeM pooling across frequency
- Attention-style SED head

Training Setup:
- Primary-label multi-hot targets
- Class-aware hold-out split with at least one training example preserved per label when possible
- AdamW optimizer
- BCE-style multi-label loss

Augmentations:
- Random train crop from each isolated recording
- No extra augmentation in the first reproduction baseline

Validation Strategy:
- Hold-out validation split built from isolated recordings
- Macro ROC-AUC over classes with both positive and negative examples

Results:
- Final best validation score: `0.9135256778` macro ROC-AUC at epoch `8`
- Leaderboard score: `0.647`

Training Time:
- The initial run was interrupted after epoch `6 / 8`, then resumed successfully and completed through epoch `8 / 8`

Observations:
- This experiment is necessary to separate external checkpoint quality from the underlying architecture quality.
- The first goal is reproducibility, not novelty.
- Validation quality improved strongly and monotonically overall through epoch 6:
  - epoch 1: `0.6925`
  - epoch 2: `0.8172`
  - epoch 3: `0.8623`
  - epoch 4: `0.8577`
  - epoch 5: `0.8958`
  - epoch 6: `0.9093`
- Resume-to-completion added:
  - epoch 7: `0.9095`
  - epoch 8: `0.9135`
- The best checkpoint and training history were written to `experiments/outputs/exp_002_train_audio_reproduction/`.
- The resumed notebook now writes `last_model.pt` after every completed epoch and still preserves `best_model.pt`.
- The final curve suggests the cosine schedule was still productive at the end rather than obviously overfitting.
- A dedicated Kaggle submission notebook now exists at `notebooks/kaggle_submission_exp_002_native.ipynb` to measure the pure single-checkpoint public score of `exp_002`.
- The pure native checkpoint does not transfer well enough to hidden soundscapes yet: the public score of `0.647` is far below the earlier `0.890` reference blend.
- This makes the main research conclusion very clear: isolated-clip validation is useful for pretraining and architecture checks, but leaderboard-facing work now has to center soundscape adaptation.

Failure Cases:
- The first long run was interrupted manually before epochs 7 and 8 completed.
- This was fixed by adding notebook-native resume support plus checkpoint loading for model, optimizer, scheduler, and scaler state.

Next Experiment Ideas:
- Finetune the best `exp_002` checkpoint on labeled soundscape segments
- Add secondary labels as weak supervision
- Compare 128-bin and 224-bin mel settings once the reproduction baseline is stable
- Transfer `site/hour` priors and texture-aware postprocessing into the repository-native stack
