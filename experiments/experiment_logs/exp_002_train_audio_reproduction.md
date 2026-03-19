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
- Best validation score so far: `0.9093025611` macro ROC-AUC at epoch `6`
- Leaderboard score: pending

Training Time:
- Long CPU / local notebook run; the first full attempt was manually interrupted after epoch `6 / 8`

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
- The best checkpoint and training history were written to `experiments/outputs/exp_002_train_audio_reproduction/`.
- Because `save_every_epoch=False`, the run currently preserves the best model rather than every epoch state.

Failure Cases:
- The first long run was interrupted manually before epochs 7 and 8 completed.
- Resume-from-checkpoint support is not yet built into the notebook, so a true continuation would require a notebook patch.

Next Experiment Ideas:
- Finetune the best `exp_002` checkpoint on labeled soundscape segments
- Add secondary labels as weak supervision
- Compare 128-bin and 224-bin mel settings once the reproduction baseline is stable
- Add notebook-native resume support and modernize AMP calls away from deprecated `torch.cuda.amp.autocast(...)`
