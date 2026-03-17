# Experiment Log

Experiment ID:
`exp_001`

Experiment Name:
`soundscape_reference_blend`

Date:
`2026-03-17`

Research Question:
Can the two available reference checkpoints be reconstructed locally and compared fairly on labeled soundscapes to establish a trustworthy baseline for future research?

Baseline Reference:
None. This is the initial validation baseline for the repository.

Change Introduced:
Add a local evaluation pipeline for labeled `train_soundscapes` and compare four reference inference strategies:

1. `LB862.pt` only
2. `LB872.pt` only
3. probability blend `0.8 * LB872 + 0.2 * LB862`
4. the same blend with temporal smoothing and file-max leakage from the reference notebook

Dataset:
- Competition data under `data/birdclef-2026/`
- Validation targets from `train_soundscapes_labels.csv`
- Taxonomy order from `sample_submission.csv`

Feature Extraction:
- 32 kHz mono audio
- 5-second windows
- Mel spectrogram with `n_mels=224`, `n_fft=2048`, `hop_length=512`, `fmax=16000`
- Per-window min-max normalization
- 3-channel mel image replication

Model Architecture:
- `tf_efficientnet_b0.ns_jft_in1k`
- GeM pooling across frequency
- Attention-style SED head

Training Setup:
- No new training in this experiment
- This is an inference and validation reconstruction experiment using existing checkpoints

Augmentations:
- None in this experiment

Validation Strategy:
- Build multi-hot targets from labeled soundscape segments
- Deduplicate identical duplicated label rows by `(filename, start, end, primary_label)`
- Evaluate only labeled segments and compute per-class ROC-AUC when both positive and negative examples exist

Results:
- Validation score: pending
- Leaderboard score: pending

Training Time:
- Not applicable

Runtime Notes:
- The repository now includes a dry-run mode for verifying local labels and file coverage without Torch.
- Full inference still needs a Torch-enabled environment.

Observations:
- The local label file contains duplicated identical segment rows, so naïve evaluation would double-count the same target segment.
- Soundscape labels cover only 75 classes, so any local metric must report how many classes were actually scored.

Failure Cases:
- Pending execution

Next Experiment Ideas:
- Train a clean soundscape-aware finetuning stage initialized from the best reference checkpoint
- Validate heuristics separately from checkpoint blending
- Compare `n_mels=128` vs `n_mels=224`
