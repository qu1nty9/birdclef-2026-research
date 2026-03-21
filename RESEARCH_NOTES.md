# Research Notes

## 2026-03-17 Initial Project Read

### Repository State

- The project documentation already defines a strong experimental workflow, but the execution layer was mostly empty at the start of this session.
- `src/`, `notebooks/`, and `experiments/` had no code or experiment logs yet.
- `MASTER_EXPERIMENT_TABLE.md` was referenced by the docs but did not exist.
- `TODO_RESEARCH.md`, `PROJECT_STATE.md`, and `SOLUTION_OUTLINE.md` existed but were empty.

### Dataset Facts Confirmed Locally

- `train.csv` contains 35,549 isolated training recordings.
- The target taxonomy contains 234 classes.
- `train_audio` covers 206 classes, so 28 target classes are missing from isolated recordings.
- Labeled soundscapes contain 75 classes total, including 28 soundscape-only classes.
- The local copy of `train_soundscapes_labels.csv` has 1,478 rows but only 739 unique `(filename, start, end)` segments.
- Every duplicated segment currently has an identical `primary_label` string, so deduplicating identical rows is safe for local evaluation.
- Mean number of labels per labeled 5-second segment is about 4.23, with a maximum of 10.
- Only about 10.98% of isolated recordings fall inside a rough Pantanal latitude/longitude bounding box.

### Reference Notebook Takeaways

Source: `references/private-notebooks/birdclef-26-acoustic-species-identification-eda.ipynb`

- The most important structural finding is that soundscape labels are essential, not optional.
- A local validation set should be built from labeled soundscapes because the competition test domain is Pantanal soundscapes, not isolated global recordings.
- Multi-label behavior is central to the task.
- Filtering too aggressively by quality rating would throw away all unrated iNaturalist data.

Source: `references/private-notebooks/birdclef-2026-lb-0-89.ipynb`

- Reference inference uses 32 kHz audio, 5-second chunks, 224-bin mel spectrograms, `n_fft=2048`, `hop_length=512`, and `fmax=16000`.
- The backbone is `tf_efficientnet_b0.ns_jft_in1k` with GeM pooling across frequency and an attention-style SED head.
- The notebook blends two checkpoints, `LB862.pt` and `LB872.pt`, with the strongest public setting presented as `0.8 * finetuned + 0.2 * baseline`.
- Two important postprocessing heuristics are used for ranking:
  - confidence-sharpened temporal smoothing
  - global file-max leakage

Source: `references/private-notebooks/birdclef-2026-smart-audio-bird-detector.ipynb`

- This notebook largely mirrors the same inference stack in a shorter form.
- It is useful as a compact submission template but less careful about local evaluation setup.

### Working Hypotheses

- Early progress should come from better validation and cleaner use of soundscape labels before chasing larger backbones.
- Public-LB heuristics may not transfer cleanly to local macro ROC-AUC because local labels are partial and class coverage is sparse.
- The strongest first training baseline should likely be a two-stage recipe:
  - pretrain on isolated `train_audio`
  - finetune on labeled soundscape segments

## 2026-03-18 First Kaggle Score

### Confirmed Outcome

- The Kaggle submission notebook based on the reference blend produced a public leaderboard score of `0.890`.
- This establishes the first end-to-end working baseline for the repository.

### Interpretation

- The project now has a real competitive anchor, not just a dry-run or local scaffold.
- At the same time, the score is still tied to borrowed reference checkpoints, so it does not yet tell us how strong our own training pipeline is.
- The most informative next step is to reproduce the same architecture on local `train_audio` and then finetune on labeled soundscapes.

### Updated Priority

1. Recover interpretability through local CV for `exp_001`.
2. Build a repository-native `train_audio` baseline in `exp_002`.
3. Use soundscape finetuning as the first genuinely novel training improvement over the `0.890` public baseline.

## 2026-03-19 Perch V2 Reference Analysis

### Source Artifacts Reviewed

Source: `references/private-notebooks/perch-v2-starter-train-infer.ipynb`

- Public score reported by the author is `0.899`.
- The notebook is a two-stage stack, not a plain end-to-end classifier.
- It uses Google Perch V2 as a frozen TensorFlow SavedModel and then adds BirdCLEF-specific adaptation on top.

Source: `references/private-notebooks/bc26-tensorflow-2-20-0.ipynb`

- This is not a modeling notebook.
- It packages local wheels for `tensorflow==2.20.0`, `tensorboard==2.20.0`, and `perch-hoplite[tf]` so the Kaggle notebook can run without internet.

Source: `data/perch_meta/`

- `full_perch_meta.parquet` and `full_perch_arrays.npz` are not extra labeled data.
- They are a cache of Perch outputs on fully labeled training soundscapes:
  - `59` fully labeled files
  - `708` windows of 5 seconds
  - raw Perch-derived class scores of shape `(708, 234)`
  - Perch embeddings of shape `(708, 1536)`
- This cache allows the notebook to fit priors and probes in `submit` mode without recomputing the expensive training-side Perch pass.

Source: `data/models/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1.tar.gz`

- The archive is a TensorFlow SavedModel with `saved_model.pb`, variables, and label assets.
- `assets/labels.csv` contains `14,795` Perch label names.
- The notebook uses two outputs from the model:
  - `label` logits
  - `embedding` vectors

### What The Method Actually Does

- It splits every 60-second soundscape into `12` windows of `5` seconds at `32 kHz`.
- It runs Perch V2 once on the batch of windows and keeps both BirdCLEF-mapped logits and a `1536`-dimensional embedding per window.
- It maps BirdCLEF labels to Perch labels by exact scientific-name match.
- In the local copy, `203 / 234` BirdCLEF classes map directly to Perch labels.
- Among the `75` soundscape-active classes, only `46` map directly, so the rest must be handled indirectly.
- It builds fold-safe metadata priors from labeled soundscapes using:
  - global prevalence
  - site prevalence
  - hour prevalence
  - site-hour prevalence
- It applies stronger prior fusion to `texture` taxa (`Amphibia`, `Insecta`) than to event-like classes.
- It smooths texture-class scores across neighboring 5-second windows.
- It then trains classwise logistic-regression probes on top of:
  - PCA-projected Perch embeddings
  - raw class score
  - prior logit
  - fused baseline score
  - previous / next / mean / max baseline score inside the 60-second file
- Probe training is done with `GroupKFold` by filename and only on classes with at least `8` positives in the training folds.

### Why The Score Is Strong

- The core strength is the pretrained Perch representation, which is much broader than the competition taxonomy.
- The biggest domain adaptation gain comes from metadata priors on soundscapes, not from end-to-end retraining.
- The method treats persistent background textures differently from short acoustic events, which fits BirdCLEF+ 2026 well.
- The final probe fixes classes where priors are too crude and restores class-specific discrimination from embeddings.
- The whole stack is careful about leakage:
  - only fully labeled files are trusted for training the second stage
  - priors are fit out-of-fold
  - probes are trained on OOF meta-features

### Local Ablations Reproduced From The Cache

- Using only the cached `perch_meta` outputs, the downstream stack can be analyzed without TensorFlow.
- On the `708` trusted windows from `59` fully labeled files:
  - raw Perch-derived BirdCLEF scores: `0.7390` macro ROC-AUC
  - honest OOF baseline after priors and texture smoothing: `0.7960`
  - honest OOF baseline plus embedding probe: `0.8259`
- Baseline ablation shows that almost all of the first-stage gain comes from texture handling:
  - raw: `0.7390`
  - inactive-unmapped suppression only: `0.7290`
  - event priors only: `0.7304`
  - texture priors only: `0.7936`
  - event + texture priors: `0.7950`
  - event + texture priors + smoothing: `0.7960`
- This strongly suggests that BirdCLEF+ 2026 rewards explicit modeling of amphibian / insect background texture.

### Where The Gain Comes From By Class

- The largest baseline gains come from insect sonotype classes and other non-bird texture classes, many of which sit near random (`AUC ~ 0.5`) before priors.
- The probe stage adds the most value where coarse priors hurt bird/event classes or underfit complex texture classes.
- Examples of strong probe recovery in the local reproduction include:
  - `nacnig1`
  - `trsowl`
  - `redjun`
  - `22961`
  - `24321`
  - `517063`
- Interpretation:
  - priors are excellent for coarse localization of texture classes
  - priors alone can oversmooth or miscalibrate some bird classes
  - embeddings recover class-specific detail

### Limits And Caveats

- The frog genus-proxy block is present in the reference notebook but did not activate in the local copy, so it does not appear to be a major source of gain for our current assets.
- The method depends on fully labeled files only, reducing the effective adaptation set from `66` raw files to `59` trusted files.
- This is a strong inference-and-stacking method, but it is not yet a repository-native training solution.

### Working Methods Worth Reusing

- Use frozen foundation-model embeddings as a reusable second-stage feature space.
- Fit site / hour / site-hour priors with shrinkage and strict OOF discipline.
- Treat `Amphibia` and `Insecta` as texture-style classes rather than standard event-only classes.
- Add file-context features such as previous, next, mean, and max window score.
- Cache expensive model outputs for trusted soundscapes so notebooks can reproduce downstream experiments quickly.
- Build stackers only on fully labeled files.
- Keep direct scientific-name mapping as the first bridge between external foundation models and competition labels.

## 2026-03-20 Exp_002 Partial Training Outcome

### Confirmed Result

- The first long run of `exp_002_train_audio_reproduction` was interrupted manually after epoch `6 / 8`.
- The current best hold-out validation score is `0.9093025611` macro ROC-AUC at epoch `6`.
- The run wrote:
  - `experiments/outputs/exp_002_train_audio_reproduction/history.csv`
  - `experiments/outputs/exp_002_train_audio_reproduction/best_model.pt`

### Validation Trajectory

- epoch 1: `0.6925`
- epoch 2: `0.8172`
- epoch 3: `0.8623`
- epoch 4: `0.8577`
- epoch 5: `0.8958`
- epoch 6: `0.9093`

### Interpretation

- This is already a strong repository-native baseline signal.
- The architecture is clearly capable of learning useful isolated-recording representations without borrowed checkpoints.
- The run was interrupted late rather than early, so the lost work is much smaller than restarting from zero would suggest.
- Because the notebook currently saves only the best checkpoint by default, the best epoch is preserved even though the full optimizer-state continuation path is not yet exposed in the notebook UI.

### Practical Conclusion

- Restarting from scratch is not necessary just to keep a usable baseline result.
- The next decision is operational rather than scientific:
  - either accept epoch 6 as the current endpoint for `exp_002`
  - or patch the notebook with proper resume support and finish epochs 7 and 8 from the saved checkpoint state

### Final Outcome After Resume

- Resume support was added to the notebook and the run completed successfully through epoch `8 / 8`.
- Final validation trajectory:
  - epoch 7: `0.9095`
  - epoch 8: `0.9135`
- The final `exp_002` result is therefore a completed repository-native isolated-audio baseline rather than a partial run.
- The experiment now exceeds the repository's first public baseline score contextually in local validation strength, though local hold-out AUC and public Kaggle LB remain different metrics.
- The next research step should move from isolated-audio reproduction to soundscape adaptation.

## 2026-03-20 Exp_002 Kaggle Submission Check

### Confirmed Outcome

- The dedicated Kaggle submission notebook for the pure `exp_002` checkpoint produced a public leaderboard score of `0.647`.

### Interpretation

- The result is much worse than the earlier `0.890` reference-blend baseline despite the much stronger isolated-recording hold-out AUC.
- This confirms that isolated `train_audio` validation is not sufficient as a leaderboard proxy for BirdCLEF+ 2026.
- The model learned useful species discrimination on clean clips, but it did not transfer to hidden Pantanal soundscapes without explicit soundscape adaptation, metadata priors, or texture-aware postprocessing.
- The gap also fits the dataset facts established earlier:
  - `28` target classes are absent from isolated `train_audio`
  - many evaluation-relevant classes are soundscape-only or texture-like
  - the hidden test domain is geographically and acoustically narrower than the global isolated-recording pool

### Practical Conclusion

- `exp_002` remains valuable as a repository-native pretraining checkpoint.
- It should not be treated as a competitive standalone submission recipe.
- The next experiments should prioritize:
  - finetuning on labeled soundscape segments
  - metadata priors
  - texture-aware handling for `Amphibia` and `Insecta`
  - second-stage stackers or probes on top of soundscape-aware features

## 2026-03-20 Basic Tutorial Reference Check

### Source Artifact Reviewed

Source: `references/private-notebooks/basic-tutorial.ipynb`

### Findings

- The notebook is functionally the same method as `references/private-notebooks/perch-v2-starter-train-infer.ipynb`.
- A cell-by-cell text comparison shows no meaningful algorithmic differences in the flattened notebook source.
- It uses the same supporting assets:
  - TensorFlow packaging notebook `bc26-tensorflow-2-20-0.ipynb`
  - Google Perch V2 SavedModel
  - cached `perch_meta` arrays
- It follows the same modeling recipe:
  - Perch logits plus embeddings
  - exact scientific-name mapping into BirdCLEF labels
  - site/hour/site-hour priors
  - texture-specific smoothing and stronger prior fusion
  - classwise embedding probes trained on fully labeled files only

### Research Value

- The notebook is still useful as a second copy of the same Perch stack, because it confirms that this approach is not a one-off implementation accident.
- It does not currently add a new methodological direction beyond what was already extracted from the Perch starter notebook.
- The working ideas worth reusing remain the same:
  - soundscape-aware metadata priors
  - explicit handling of texture classes
  - frozen foundation embeddings as second-stage features
  - strict use of fully labeled files for trusted adaptation

## 2026-03-20 Exp_003 Perch Downstream Reproduction

### Confirmed Outcome

- A notebook-only local reproduction of the Perch downstream stack now exists at `notebooks/exp_003_perch_downstream_reproduction.ipynb`.
- The experiment uses only local assets:
  - `data/perch_meta/full_perch_meta.parquet`
  - `data/perch_meta/full_perch_arrays.npz`
  - `data/birdclef-2026/taxonomy.csv`
  - `data/models/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1.tar.gz`
- The aligned trusted subset is:
  - `59` fully labeled files
  - `708` windows
  - `71` active classes in the trusted windows
  - `203 / 234` direct scientific-name mappings
  - `42` active texture classes

### Reproduced Metrics

- Raw Perch-derived BirdCLEF scores: `0.7390` macro ROC-AUC
- Honest OOF baseline after priors and texture smoothing: `0.8044`
- Honest OOF embedding-probe stack: `0.8353`
- Probe lift over the OOF baseline: `+0.0309`
- Modeled probe classes: `52`

### Baseline Ablation

- `event + texture priors + smoothing`: `0.8044`
- `event + texture priors`: `0.8031`
- `texture priors only`: `0.8017`
- `event priors only`: `0.7404`
- `raw`: `0.7390`
- `inactive unmapped suppression only`: `0.7390`

### Interpretation

- The reproduction confirms the earlier qualitative story very clearly:
  - texture-aware priors explain almost all of the first-stage gain
  - event priors alone barely move the metric
  - smoothing helps, but only slightly after texture priors are already active
- The classwise probe is useful but selective.
- The largest probe recoveries in the local reproduction include:
  - `nacnig1`
  - `trsowl`
  - `redjun`
  - `24321`
  - `517063`
- Some classes still regress under the probe, so the stack should be treated as a strong comparison target, not as a universal monotonic improvement.

### Practical Conclusion

- `exp_003` is now the strongest soundscape-aware local reference in the repository.
- `exp_004_soundscape_finetuning` should be compared against `exp_003`, not against the isolated `exp_002` hold-out alone.
- The first native hybrid should likely import the following ideas from `exp_003`:
  - site/hour/site-hour priors
  - texture-aware handling for `Amphibia` and `Insecta`
  - optional second-stage features using previous, next, mean, and max file-context scores

## 2026-03-20 BirdCLEF Training Stack Reference Analysis

### Source Artifacts Reviewed

Source: `references/private-notebooks/birdclef-training/birdclef-2026-i-o-preprocessing.ipynb`

- This notebook is an engineering-first preprocessing stage rather than a modeling notebook.
- It converts filtered BirdCLEF audio into fixed-length 5-second waveform clips stored in memory-mapped `.npy` arrays.
- It separates the corpus into:
  - focal labeled clips
  - background soundscape clips for later mixing
- The main filters applied before decoding are:
  - `rating >= 3.0`
  - broad South America / Pantanal-style latitude-longitude bounds
- Core constants:
  - `sample_rate = 32_000`
  - `clip_duration = 5`
  - `SAMPLES_PER_CLIP = 160_000`
- The most useful implementation ideas are:
  - header-only clip counting with `soundfile.info()` before allocation
  - explicit disk budgeting before building the mmap arrays
  - precomputed `mmap_index` metadata and per-file offsets
  - a capped background bank with `MAX_BG_CLIPS_TOTAL = 5_000`

Source: `references/private-notebooks/birdclef-training/birdclef-2026-training.ipynb`

- This notebook is a full dual-GPU DDP training system built on top of the preprocessed waveform arrays.
- Core configuration extracted from the checked-in code:
  - `sample_rate = 32_000`
  - `clip_duration = 5`
  - `n_fft = 1024`
  - `hop_length = 512`
  - `n_mels = 128`
  - `fmin = 40`
  - `fmax = 14_000`
  - `backbone = tf_efficientnet_b1`
  - `batch_size = 64`
  - `epochs = 30`
  - `lr = 3e-4`
  - `weight_decay = 1e-4`
  - `backbone_lr_mult = 0.1`
  - `snapshot_epochs = [20, 22, 24, 26, 28, 30]`
- The acoustic frontend is not a standard torchaudio mel stack.
- It uses a custom GPU-native `AudioToMelPCEN` module:
  - `torch.stft`
  - mel filterbank projection
  - causal smoother for the PCEN background estimate
  - learnable PCEN parameters
- The label handling is more careful than naive multi-hot BCE.
- `MaskedBCEWithLogitsLoss` combines primary and secondary labels into one target tensor but masks loss on secondary-only positions, so uncertain secondary tags do not behave like fully trusted positives.
- The training augmentation recipe is especially notable:
  - background waveform mixing with probability `0.80`
  - random SNR between `10` and `30` dB
  - time shift with probability `0.50`
  - gain perturbation with probability `0.40`
  - SpecAugment with `2` frequency masks and `2` time masks
  - additive MixUp with probability `0.50`
- Pseudo-label integration is done at the waveform-mixing stage:
  - pseudo targets are loaded for background clips
  - background targets are blended into the focal target vector in proportion to the mixing scale
- Validation appears to be `StratifiedKFold` on `label_id` from focal-clip metadata rather than a soundscape-aware split.
- The notebook logs very high fold AUC values around `0.966`, but they should be treated carefully because this validation still looks isolated-recording-centric.

Source: `references/private-notebooks/birdclef-training/birdclef-2026-target-domain-pseudo-labeling.ipynb`

- This notebook is the stage that adapts the training pipeline to the target soundscape domain.
- It runs a 5-fold ONNX ensemble over `train_soundscapes` and writes pseudo labels for stage-2 training.
- The inference pipeline uses:
  - 5-second windows
  - 2.5-second hop
  - temporal smoothing with `uniform_filter1d(size=3)`
  - class-specific quantile thresholding
- The checked-in code currently sets:
  - `gamma = 1.0`
  - `dynamic_threshold_percentile = 0.70`
- This is important because the markdown text describes a more aggressive variant:
  - Babych-style power transform with `gamma = 2.0`
  - `95th` percentile filtering
- So the notebook documentation and the executed code are not perfectly aligned; we should trust the code more than the prose when porting ideas.
- Operationally the notebook is very optimized:
  - one producer process for audio decoding
  - two GPU consumer processes for ONNX inference
  - shared-memory transport between them
- Conceptually the key idea is simple and very relevant for us:
  - pseudo-label the target-domain soundscapes rather than only the isolated recordings

### What Likely Drives This Stack

- These three notebooks together form a coherent recipe, not three disconnected tricks:
  - preprocess fixed waveform clips for fast random access
  - train a strong clip classifier with heavy augmentation and careful target handling
  - adapt to the real test domain through pseudo labels on soundscapes
- The main lesson is that this approach treats BirdCLEF+ 2026 as a domain-adaptation systems problem.
- The likely biggest gains come from:
  - soundscape-like background mixing
  - masked use of secondary labels
  - PCEN for noisy long-form audio
  - target-domain pseudo-labeling with overlapping windows and smoothing

### Most Transferable Ideas For Our Project

- Add a soundscape-bank background mixing stage to native training instead of training only on clean focal clips.
- Replace naive secondary-label supervision with a masked-BCE formulation.
- Evaluate PCEN as a stronger frontend for noisy Pantanal soundscapes.
- Keep target-domain pseudo-labeling as a later-stage branch once our soundscape validation loop is stable.
- Reuse the overlap-and-smooth idea even if we do not copy the full shared-memory ONNX system.

### Cautions

- The preprocessing notebook is strong operationally, but it is a large infrastructure commitment; we should only port the full mmap pipeline if our current notebook training becomes I/O-bound.
- The training notebook's validation protocol still appears too close to isolated-recording CV to be trusted as a leaderboard proxy.
- The pseudo-labeling notebook uses `train_soundscapes`, so any local evaluation must remain fold-safe to avoid leakage.

## 2026-03-21 Exp_004 Launch

### Notebook Prepared

- A new experiment notebook now exists at `notebooks/exp_004_soundscape_finetuning.ipynb`.
- The experiment is intentionally the first native soundscape adaptation branch rather than another isolated-audio baseline.

### Design Choices

- Initialization comes from `exp_002` through `experiments/outputs/exp_002_train_audio_reproduction/best_model.pt`.
- Training data is shifted to labeled `train_soundscapes` segments.
- Validation is grouped by soundscape file and restricted to fully labeled files.
- The notebook adds:
  - soundscape background mixing with soft target blending
  - a masked-BCE-compatible loss wrapper
  - optional capped replay from `train_audio` so true `secondary_labels` can be exercised without turning the whole run back into isolated-audio training

### Why This Is The Right Next Step

- `exp_002` showed that isolated-audio CV is not enough.
- `exp_003` showed that soundscape-aware adaptation is the main path to better leaderboard behavior.
- The newest training references suggest that the highest-value transferable ideas before pseudo-labeling are:
  - background mixing
  - careful treatment of secondary labels

### Open Question

- The first `exp_004` run should answer whether native soundscape finetuning alone can move materially toward the `exp_003` local soundscape baseline, or whether priors must be added immediately in the next step.

## 2026-03-21 Exp_004 First Run

### Confirmed Result

- The first run of `exp_004_soundscape_finetuning` completed through `6 / 6` epochs.
- Best validation score: `0.779605` macro ROC-AUC at epoch `4`.
- Full trajectory:
  - epoch 1: `0.686650`
  - epoch 2: `0.758403`
  - epoch 3: `0.768842`
  - epoch 4: `0.779605`
  - epoch 5: `0.778186`
  - epoch 6: `0.772404`
- Validation scored `29` classes and skipped `205` classes without positives in the held-out fold.

### Interpretation

- The result is directionally good:
  - the model adapts to soundscape supervision
  - the metric improves quickly in the first few epochs
  - the best epoch arrives before the run ends, so more epochs alone are unlikely to be the main source of the next gain
- The result is not yet a clean head-to-head against `exp_003`.
- The current validation fold is too sparse:
  - only `29` classes contribute to the metric
  - fold variance is likely to be high
- Even with that limitation, the run still supports the main research story:
  - native soundscape finetuning is much more meaningful than more isolated-audio-only training
  - priors and texture-aware postprocessing are the most logical next additions

### Practical Conclusion

- `exp_004` is a successful first native soundscape branch, but not yet the strongest local soundscape recipe.
- The next best step is not a Kaggle submission of raw `exp_004`.
- The next best step is a native hybrid:
  - `exp_004` predictions
  - plus `site/hour` priors
  - plus texture-aware postprocessing

## 2026-03-21 Exp_005 Native Priors Transfer Check

### Confirmed Result

- `exp_005_native_priors_texture_postproc` was run on the same validation fold as `exp_004`.
- Raw native `exp_004` score on that fold: `0.7796052180`
- Best postprocessed native score: `0.8156599403`
- Uplift over raw native predictions: `+0.0360547223`

### Ablation Pattern

- `event + texture priors + smoothing`: `0.8157`
- `event + texture priors`: `0.8151`
- `texture priors only`: `0.8059`
- `event priors only`: `0.7888`
- `raw`: `0.7796`

### Interpretation

- This is a strong validation of the current research direction.
- The native branch is not missing only training signal; it was also missing a soundscape-aware inference layer.
- The transfer pattern mirrors the Perch branch:
  - texture priors matter much more than event priors alone
  - smoothing is a small but positive final increment
- The result substantially narrows the local gap between the first native soundscape branch and the soundscape-aware Perch reference.

### Practical Conclusion

- The best next public test should not be raw `exp_004`.
- The best next public test should be a lightweight native submission built around:
  - the `exp_004` checkpoint
  - metadata priors
  - texture-aware smoothing

## 2026-03-21 Exp_005 First Public Kaggle Score

### Confirmed Result

- The lightweight native hybrid submission notebook scored `0.737` on the public leaderboard.
- This score comes from the `exp_004` checkpoint plus the best local `exp_005` postprocessing recipe:
  - `site/hour/site-hour` priors
  - texture-aware smoothing
  - no Perch
  - no external reference checkpoints

### Interpretation

- This is a meaningful leaderboard improvement over the pure native `exp_002` checkpoint:
  - `exp_002`: `0.647`
  - `exp_005` native hybrid: `0.737`
- The direction from local validation transferred correctly:
  - soundscape-aware priors are not just a local-CV artifact
  - texture-aware logic helps public LB too
- But the remaining gap is still large:
  - `exp_005`: `0.737`
  - reference blend baseline: `0.890`
- The implication is important:
  - better inference logic helps
  - but postprocessing alone is not enough to make the native branch competitive with the strongest currently available reference path

### Practical Conclusion

- `exp_005` should be considered a successful proof that native hybrid inference works.
- The next main gain should likely come from one of three heavier directions:
  - stronger native soundscape training
  - a second-stage native stacker with file-context features
  - target-domain pseudo-labeling
- Before choosing among them, the validation protocol for native soundscape runs still needs to be strengthened beyond the current sparse single fold.
