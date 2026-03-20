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
