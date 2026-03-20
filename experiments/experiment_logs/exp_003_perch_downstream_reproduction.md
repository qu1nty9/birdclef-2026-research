# Experiment Log

Experiment ID:
`exp_003`

Experiment Name:
`perch_downstream_reproduction`

Date:
`2026-03-20`

Research Question:
How much of the strong Perch public-baseline behavior can be reproduced locally using only cached `perch_meta` outputs, trusted soundscape labels, metadata priors, and classwise embedding probes?

Baseline Reference:
External `Perch V2` starter notebooks in `references/private-notebooks/`

Change Introduced:
Rebuild the downstream part of the Perch stack locally without rerunning TensorFlow inference. The experiment uses cached logits and embeddings, then adds fold-safe metadata priors, texture-aware smoothing, and classwise logistic probes.

Dataset:
- Competition metadata under `data/birdclef-2026/`
- Trusted soundscape labels from `train_soundscapes_labels.csv`
- Cached Perch outputs from `data/perch_meta/`
- Perch label assets from `data/models/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1.tar.gz`

Feature Extraction:
- No new acoustic frontend is trained here
- Inputs are frozen cached Perch outputs:
  - `scores_full_raw` in BirdCLEF class space
  - `emb_full` with `1536`-dimensional embeddings
- Additional features are metadata priors and file-context features derived from 12-window soundscape blocks

Model Architecture:
- Frozen Perch V2 features
- No end-to-end neural training
- Second stage:
  - metadata prior fusion in logit space
  - texture smoothing for `Amphibia` and `Insecta`
  - classwise `LogisticRegression` probes on top of PCA-compressed embeddings and file-context features

Training Setup:
- `GroupKFold(n_splits=5)` by filename
- Prior tables fit out-of-fold
- Probe params:
  - `pca_dim=64`
  - `min_pos=8`
  - `C=0.50`
  - `alpha=0.40`
- Fusion params:
  - `lambda_event=0.4`
  - `lambda_texture=1.0`
  - `lambda_proxy_texture=0.8`
  - `smooth_texture=0.35`

Augmentations:
- None
- This is a downstream analysis notebook, not a waveform training run

Validation Strategy:
- Honest OOF macro ROC-AUC on the `59` fully labeled soundscape files
- Comparison between:
  - raw cached scores
  - OOF metadata-prior baseline
  - OOF embedding-probe stack

Results:
- Raw local macro ROC-AUC: `0.7390178442`
- OOF prior baseline: `0.8044348180`
- OOF embedding-probe score: `0.8353024140`
- Probe delta over the OOF baseline: `+0.0308675960`
- Modeled probe classes: `52`
- Kaggle leaderboard score: `n/a`

Training Time:
- Local notebook-only downstream analysis
- Runtime is short compared with model training because the expensive Perch inference step is reused from cache

Observations:
- Texture-aware priors are the dominant source of gain:
  - `texture priors only`: `0.8017`
  - `event priors only`: `0.7404`
- Temporal smoothing adds only a small extra boost once texture priors are active.
- The probe helps selectively and recovers several classes that the prior baseline underserves, including:
  - `nacnig1`
  - `trsowl`
  - `redjun`
  - `24321`
  - `517063`
- The experiment confirms that strong BirdCLEF+ 2026 soundscape performance depends heavily on domain adaptation and not only on raw classifier quality.
- A dedicated Kaggle submission notebook now exists at `notebooks/kaggle_submission_exp_003_perch.ipynb` for public-LB validation of this branch.

Failure Cases:
- The probe is not universally beneficial; some already-strong classes regress.
- The whole stack still depends on external Perch embeddings, so it is not yet a repository-native solution.
- Only `52` classes are modeled by the probe because low-positive classes are intentionally skipped.

Next Experiment Ideas:
- Build `exp_004_soundscape_finetuning` starting from the native `exp_002` checkpoint
- Compare `exp_004` directly against this soundscape-aware Perch baseline
- Transfer the strongest ideas into the native branch:
  - metadata priors
  - texture-aware handling
  - file-context second-stage features
