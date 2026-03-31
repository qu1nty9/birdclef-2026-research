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

## 2026-03-22 Exp_008b Kaggle Submission Check

### Confirmed Outcome

- The first Kaggle submission built from the long-context native branch (`exp_008 + exp_008b` postprocessing) scored `0.707`.

### Interpretation

- This is a clearly negative leaderboard result.
- The branch underperformed:
  - `exp_007` 3-fold native hybrid: `0.758`
  - `exp_005` single-fold native hybrid: `0.737`
  - current long-context submission: `0.707`
- So the strong local fold `0` result did not transfer to the public leaderboard.

### Most Likely Explanation

- The local long-context evidence was built on a single sparse fold with only `29` scored classes.
- That fold now looks too optimistic to guide leaderboard choices by itself.
- The architectural idea may still be valid, but the current evidence says:
  - single-fold long-context validation is not reliable enough
  - the branch should not replace the current native public baseline yet

### Practical Conclusion

- `exp_007` remains the strongest repository-native public recipe for now.
- The long-context branch should only continue in one of two ways:
  - validate it on more folds first
  - or pause it and move to a stronger modeling jump such as target-domain pseudo-labeling
- The key lesson is methodological:
  - local fold gains from soundscape data are useful
  - but sparse single-fold wins are still too fragile to trust for leaderboard promotion

## 2026-03-22 Exp_008 Additional Folds

### Confirmed Outcome

- `exp_008` now has three completed local folds:
  - fold `0`: `0.8377433031`
  - fold `1`: `0.8222946825`
  - fold `2`: `0.8094790079`
- Mean across folds `0-2`: `0.8231724344`

### Interpretation

- This materially changes the local interpretation of the long-context branch.
- Compared with `exp_006` over the same folds:
  - `exp_006` mean: `0.7945`
  - `exp_008` mean: `0.8232`
  - absolute gain: `+0.0287`
- So the architecture jump was not a one-fold fluke locally.
- Fold `2` is especially useful evidence because it covered `35` scored classes and still stayed above `0.809`.

### Practical Conclusion

- The negative `0.707` Kaggle result should not be read as “long context is dead”.
- The better reading is:
  - single-fold promotion to Kaggle was premature
  - local gains are real
  - the next correct step is a fold-safe OOF postprocess comparison on top of `exp_008` folds `0-2`
- In other words, the branch has earned one more local validation stage before we either discard it or try another submission.

## 2026-03-22 Exp_008c Honest Pooled OOF Check

### Confirmed Outcome

- `exp_008c_long_context_priors_on_oof` completed locally on `exp_008` folds `0-2`.
- Raw pooled OOF macro ROC-AUC: `0.6682152899`
- Best pooled OOF variant: `event_texture_priors_smooth`
- Best pooled OOF macro ROC-AUC: `0.7004623407`
- Absolute pooled OOF gain vs raw: `+0.0322470508`
- Scored classes in pooled OOF: `54`

### Why This Matters

- This is the honest comparison the branch needed after the misleading single-fold story.
- The result explains the weak leaderboard outcome much better than the fold means did.

### Comparison To The Current Native Baseline

- `exp_007` pooled OOF raw: `0.6646`
- `exp_008c` pooled OOF raw: `0.6682`
- `exp_007` pooled OOF best: `0.7109`
- `exp_008c` pooled OOF best: `0.7005`

### Interpretation

- Long context is not useless; it slightly improves the raw pooled OOF baseline.
- But once both branches receive the same metadata-prior and texture-aware inference layer, the long-context branch still loses to `exp_007`.
- So the main lesson is:
  - long-context modeling alone was not the missing competition-level ingredient
  - the branch is interesting architecturally
  - but it is not the next promoted submission route

### Practical Conclusion

- The current best validated native recipe remains `exp_007`.
- The long-context branch should be parked for now rather than pushed further on Kaggle.
- The roadmap should move on to the stronger next modeling jump: noisy-student pseudo-labeling.

## 2026-03-22 Exp_008b Long-Context Postprocessing Check

### Confirmed Outcome

- `exp_008b_long_context_priors_postproc` completed locally on top of the exported `exp_008` fold `0` predictions.
- Raw long-context `exp_008` fold `0` macro ROC-AUC: `0.8377433031`
- Best postprocessed variant: `event_texture_priors_smooth`
- Best macro ROC-AUC after priors: `0.8434672644`
- Absolute gain versus raw: `+0.0057239613`

### Variant Ablation

- `event + texture priors + smoothing`: `0.8435`
- `event + texture priors`: `0.8402`
- `raw`: `0.8377`
- `texture priors only`: `0.8357`
- `event priors only`: `0.8188`

### Interpretation

- The long-context branch still benefits from the existing soundscape-aware inference layer, so priors are not obsolete.
- At the same time, the gain is much smaller than what we saw in the short-context branch:
  - `exp_005`: `0.7796 -> 0.8157` (`+0.0361`)
  - `exp_008b`: `0.8377 -> 0.8435` (`+0.0057`)
- This is a strong sign that the move to `20s` context is already absorbing part of the structure that priors previously had to repair:
  - neighboring windows are now modeled jointly
  - overlap-aware aggregation already injects local file context
  - the backbone and temporal head are carrying more of the soundscape adaptation load directly

### Classwise Pattern

- Texture-heavy and sonotype-like classes still produced the largest gains.
- Some already-strong bird/event classes regressed, which means postprocessing should still be validated on Kaggle rather than assumed universally beneficial.
- The best recipe remained stable:
  - `event + texture priors + smoothing`

### Practical Conclusion

- `exp_008 + priors` is now the strongest repository-native long-context candidate for the next public leaderboard test.
- The next high-signal action should be a Kaggle submission from this branch, not another small local postprocessing ablation on the old short-context models.

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

## 2026-03-21 Exp_006 Native Branch Plan

### Why This Branch Exists

- The first native hybrid public score of `0.737` answered one question clearly:
  - soundscape-aware priors do help the leaderboard
  - but inference-only adaptation is not enough
- That pushes the next main bet toward stronger native soundscape training rather than another lightweight postprocessing-only experiment.

### Planned Changes

- Keep the same native `exp_002 -> exp_004` initialization path.
- Keep background mixing from the first soundscape finetuning notebook.
- Turn on a non-zero weight for replay `secondary_labels` so the masked loss is no longer a placeholder.
- Expand the replay bank slightly to make the secondary-label signal show up more often.
- Save fold-specific best validation predictions so future OOF analysis and native postprocessing do not depend on rerunning training.

### Practical Objective

- `exp_006` is meant to improve two things at once:
  - the checkpoint quality
  - the evaluation protocol
- This is the first native notebook designed explicitly as an OOF-ready training branch rather than a single-fold proof of concept.

## 2026-03-21 Exp_006 Fold 0 First Result

### Confirmed Result

- `exp_006_soundscape_finetuning_v2` completed fold `0`.
- Best validation score: `0.7796052180` macro ROC-AUC.
- Best epoch: `4`.
- Scored classes: `29`.
- Validation exports were written successfully alongside the checkpoint.

### Interpretation

- The scientific result is currently neutral:
  - the score exactly matches the best `exp_004` fold-0 result
  - the current `secondary_weight=0.2` and larger replay bank did not produce a measurable lift on this fold
- The operational result is still valuable:
  - fold-specific outputs work
  - best-model checkpointing works
  - best validation predictions are now exported for later OOF analysis

### Practical Conclusion

- `exp_006` cannot yet be claimed as a real modeling improvement.
- `exp_006` can already be claimed as a successful protocol improvement.
- The next decision should wait for additional folds:
  - if more folds also show no lift, keep the export pipeline and move on to a stronger modeling change
  - if later folds improve, then `exp_006` becomes the right base for the next native hybrid

## 2026-03-21 Exp_006 Folds 1-2 Update

### Confirmed Result

- Fold `1` best macro ROC-AUC: `0.8312950828`
- Fold `2` best macro ROC-AUC: `0.7724515414`
- Fold `2` scored `35` classes, compared with `29` on folds `0` and `1`
- The first three-fold summary for `exp_006` is now:
  - fold `0`: `0.7796052180`
  - fold `1`: `0.8312950828`
  - fold `2`: `0.7724515414`
  - mean over folds `0-2`: `0.7944506141`

### Interpretation

- The result is now clearly mixed rather than purely negative.
- Fold `1` is strong enough to show that the branch still has upside.
- Fold `2` falling below fold `0` while scoring more classes reinforces the main methodological warning:
  - these soundscape folds are sparse
  - coverage differences still change the apparent difficulty a lot
- The current evidence does not justify claiming that `exp_006` is strictly better than `exp_004`.
- The current evidence also does not justify discarding `exp_006` as a dead end.

### Practical Conclusion

- `exp_006` should now be treated as a usable native base, but not yet a proven stronger checkpoint.
- The most informative next test is no longer another raw training ablation.
- The most informative next test is to reuse the exported `exp_006` fold predictions and apply the proven `exp_005` priors/texture recipe on top of them.

## 2026-03-22 BirdCLEF 2025 1st Place Solution Analysis

### Source Artifacts Reviewed

Source: `references/private-solutions/birdclef2025_1st_place_solution/birdclef2025_1st_place_solution.docx`

- The writeup is unusually strong because it explains not only the final ensemble, but also why the pseudo-labeling loop eventually worked.
- The solution title is very descriptive: `Multi-Iterative Noisy Student Is All You Need`.

Source: `references/private-solutions/birdclef2025_1st_place_solution/birdclef2025-1st-place-inference.ipynb`

- The submission path confirms the writeup claims in code:
  - long-context SED models
  - overlap-aware frame aggregation
  - delta-shift TTA
  - smoothing
  - a separate `Amphibia/Insecta` model
  - CPU-optimized OpenVINO inference

### What The 2025 Winner Actually Did

- Trained SED CNNs on `20`-second chunks while still predicting competition outputs at `5`-second resolution.
- Reused the SED head to keep framewise predictions, then aggregated overlapping neighboring chunks instead of treating each chunk independently.
- Used multi-iterative pseudo-labeling on unlabeled soundscapes.
- Made self-training work through a Noisy Student recipe:
  - mixup between clean labeled samples and pseudo-labeled soundscapes
  - stochastic depth during self-training
  - confidence-aware pseudo-label sampling
  - power-transform denoising of pseudo-label probabilities
- Added a dedicated model for texture-heavy groups (`Amphibia`, `Insecta`) using extra Xeno-Canto species.
- Finalized with a diverse ensemble across iterations and backbones.

### High-Value Transfer Ideas For BirdCLEF 2026

- Long-context SED is the single most important architectural idea from this solution.
  - Our current branch still treats the task too much like independent `5`-second clip classification.
  - A `20`-second context branch that still emits `5`-second outputs is highly aligned with the soundscape-heavy nature of BirdCLEF 2026.
- Overlap-aware frame aggregation is a very strong inference idea for our setting.
  - It is more principled than making each `5`-second row independent.
  - It fits especially well if we keep or extend SED-style heads.
- Noisy Student style pseudo-label training is the most important training idea.
  - The key transferable lesson is not just “use pseudo-labels.”
  - The real lesson is that pseudo-labels worked only after adding enough noise and denoising:
    - mixup with labeled data
    - stochastic depth
    - confidence-weighted pseudo sampling
    - probability power transform
- Separate treatment of texture groups is strongly reinforced again.
  - We already saw this story in `exp_003` and `exp_005`.
  - The 2025 winning solution confirms that a dedicated `Amphibia/Insecta` branch can be worth real leaderboard points.
- Extra external data should be targeted, not indiscriminate.
  - The writeup explicitly says that extra target-species audio often hurt.
  - The useful external data was much more specific: extra texture-heavy species for a dedicated texture model.
- CPU inference engineering matters once the ensemble becomes non-trivial.
  - OpenVINO/ONNX export, shared spectrogram computation, and multiprocess loading are all practical ideas for Kaggle runtime limits.

### Ideas To Treat Carefully

- We should not import the “validate on LB only” practice.
  - The 2025 winner explicitly says CV did not correlate well and they trusted public LB.
  - For our 2026 project, we already have a better path: labeled soundscapes plus fold-aware exports.
- We should not blindly copy the exact `20`-second choice as dogma.
  - It is a very strong ablation target, not yet a guaranteed optimum for 2026.
- We should not jump directly into a huge 7-model ensemble.
  - The ensemble only became powerful after the training stages themselves were already strong.

### Practical Conclusion

- The most actionable ideas for us, in order, are:
  - build a long-context native SED branch with overlap-aware `5`-second outputs
  - add a real noisy-student pseudo-label branch with denoising and weighted pseudo sampling
  - consider a dedicated `Amphibia/Insecta` model with targeted extra data
  - only after that, invest in heavier ensemble and inference optimization work

## 2026-03-22 Exp_007 Native Priors On Exp006 OOF

### Confirmed Result

- The `exp_005` postprocessing recipe transfers positively to the fold-aware `exp_006` exports.
- Pooled OOF raw macro ROC-AUC: `0.6646442720`
- Pooled OOF best macro ROC-AUC: `0.7108902338`
- Absolute uplift: `+0.0462459618`
- Best variant: `event_texture_priors_smooth`
- Pooled OOF scored classes: `54`

### Fold-Level Readout

- Fold `0`:
  - raw: `0.7796052180`
  - best: `0.8229967977`
- Fold `1`:
  - raw: `0.8312950828`
  - best: `0.8640179581`
- Fold `2`:
  - raw: `0.7724515414`
  - best: `0.8286392475`
- Mean fold-wise raw macro ROC-AUC: `0.7944506141`
- Mean fold-wise best macro ROC-AUC: `0.8385513344`

### Interpretation

- The positive result is real, but the methodological warning is just as important as the gain.
- Simple fold means are too optimistic here:
  - pooled OOF raw is only `0.6646`, far below the fold mean `0.7945`
  - pooled OOF best is `0.7109`, far below the fold mean `0.8386`
- The main reason is broader pooled class coverage:
  - per-fold evaluation scored only `29`, `29`, and `35` classes
  - pooled OOF scored `54` classes
- This makes pooled OOF a much better comparison tool for future native branches than single sparse splits.
- The pattern from earlier experiments survives:
  - event priors alone help only slightly (`0.6646 -> 0.6672`)
  - texture priors deliver most of the gain (`0.6646 -> 0.7013`)
  - smoothing adds a smaller final improvement (`0.7086 -> 0.7109`)

### Practical Conclusion

- `exp_007` is a successful consolidation step for the native branch.
- The next native Kaggle submission should use `exp_006 + priors + texture smoothing`.
- At the same time, `exp_007` does not remove the need for a larger modeling jump:
  - even improved native OOF is still clearly below the stronger soundscape-aware external branch
  - the roadmap toward long-context native SED and noisy-student pseudo-labeling remains justified

## 2026-03-22 Exp_007 Kaggle Submission Readout

### Confirmed Result

- Public leaderboard score: `0.758`
- Previous best native public score: `0.737`
- Absolute native public gain: `+0.021`

### Interpretation

- The gain is real, so `exp_006 + priors + texture smoothing` is not just a local-OOF artifact.
- The public leaderboard confirms that fold-aware ensembling and metadata priors do transfer to hidden test.
- At the same time, the gain is modest relative to the remaining benchmark gap:
  - current native public best: `0.758`
  - current strong reference blend: `0.890`
- This strongly suggests that the current main bottleneck is no longer “missing priors.”
- The bottleneck is more likely the native model's soundscape representation itself:
  - short context
  - limited target-domain adaptation
  - no pseudo-label branch yet

### Practical Conclusion

- Native progress is now clear and monotonic:
  - `0.647` from pure `exp_002`
  - `0.737` from `exp_005`
  - `0.758` from `exp_007`
- That is encouraging, but still not competitive enough.
- The next likely meaningful gain should come from `exp_008` long-context native SED rather than from another small postprocessing tweak.

## 2026-03-22 Exp_008 Notebook Design

### Planned Experiment

- `exp_008` is now prepared as a notebook-first long-context native SED branch:
  - `20s` waveform input
  - `4` aligned `5s` outputs
  - overlap-aware aggregation back to row-level validation predictions

### Why This Design

- It directly targets the main limitation of the current native branch:
  - short local context
  - heavy dependence on inference-only postprocessing for leaderboard gains
- It is also the cleanest next transfer from the 2025 first-place solution without jumping too early into noisy-student pseudo-labeling.

### First-Version Scope

- Keep the first version simple enough to debug:
  - labeled soundscapes only
  - background mixing retained
  - partial warm-start from `exp_002`
  - no pseudo-labels yet
  - no extra stacker yet

### Practical Goal

- First confirm that longer context alone improves the native local soundscape metric.
- If that happens, reuse the proven `exp_007` priors and texture-aware smoothing on top of `exp_008` exports before going back to Kaggle.

## 2026-03-22 Exp_008 Fold 0 Result

### Confirmed Result

- Fold `0` best macro ROC-AUC: `0.8377433031`
- Best epoch: `6 / 6`
- Scored classes: `29`
- Coverage after overlap aggregation:
  - mean: `3.0`
  - min: `1`
  - max: `4`

### Interpretation

- This is the first strong native gain that comes from changing the model context itself rather than only from inference-time priors.
- On the same sparse fold, raw `exp_008` is stronger than:
  - raw `exp_004`: `0.7796`
  - raw `exp_006` fold `0`: `0.7796`
  - best local `exp_005` fold result: `0.8157`
  - best local `exp_007` fold result: `0.8230`
- That comparison is important because it means long context is already competitive before adding the priors layer back in.
- The coverage statistics also confirm that overlap-aware aggregation is active and meaningful:
  - many rows are being seen in multiple `20s` contexts
  - the branch is not degenerating back to one independent view per row

### Practical Conclusion

- `exp_008` is the strongest native modeling direction we have seen so far.
- The immediate next test should not be a second raw architecture change.
- The immediate next test should be to reuse the validated `exp_007` priors/texture layer on top of the `exp_008` exported validation outputs.

## 2026-03-22 Exp_009 Notebook Setup

### What Was Created

- The first repository-native noisy-student notebook now exists:
  - `notebooks/exp_009_noisy_student_pseudolabel.ipynb`
- The branch intentionally uses the stronger short-context native path as its base:
  - student init from `exp_006` fold checkpoints when available
  - fallback init from `exp_002`
  - teacher ensemble from `exp_006` folds that exclude the active validation fold

### Setup Validation Result

- The notebook passed a safe setup run in the project `.venv` with both heavy stages disabled.
- Fold `0` resolved:
  - teacher folds: `[1, 2]`
  - pseudo-candidate windows: `127157`
  - student init: `exp_006` fold `0`
- No pseudo rows were cached yet and no training has run yet, so this is an operational checkpoint, not a quality result.

### Why This Branch Is The Right Next Step

- `exp_008c` showed that long context is still weaker than `exp_007` under honest pooled OOF.
- That means the next likely gain source is not another promotion of the long-context branch.
- The next likely gain source is to improve the training data itself:
  - fold-safe pseudo labels on target-domain soundscapes
  - confidence-aware sampling
  - denoising by probability power transform
  - continued use of background mixing and replay

### Practical First-Run Sequence

- Generate pseudo labels once and write:
  - `pseudo_label_meta.parquet`
  - `pseudo_label_probs.npz`
  - `teacher_summary.json`
- Inspect pseudo count and confidence distribution before training.
- Run a short smoke-train after pseudo caching succeeds.
- Only then commit to the full `exp_009` student run.

## 2026-03-23 Exp_009 Pseudo-Generation Readout

### Confirmed Result

- Fold `0` pseudo-label generation completed successfully.
- Manifest rows: `127896`
- Pseudo candidates before filtering: `127157`
- Kept pseudo rows: `69593`
- Pseudo files covered: `9552`
- Keep rate vs pseudo candidates: `54.73%`
- Teacher folds used: `[1, 2]`
- Probability tensor shape: `(69593, 234)`

### Confidence Distribution

- Mean confidence: `0.6552`
- Median confidence: `0.6797`
- `p75` confidence: `0.9063`
- `p90` confidence: `0.9789`
- Max confidence: `0.99995`

### Interpretation

- This is a strong first pseudo-label cache:
  - it is selective enough to remove a large fraction of low-confidence candidates
  - it still keeps a large training pool rather than collapsing to a tiny trusted subset
- The confidence profile is especially encouraging:
  - median confidence is far above the `0.20` threshold
  - the upper tail is very strong, which should work well with confidence-weighted pseudo sampling
- The per-file cap is behaving as intended:
  - top files keep exactly `8` pseudo windows
  - the notebook is not letting a few high-activity files dominate the pseudo pool

### Practical Conclusion

- `exp_009` has cleared the pseudo-generation checkpoint.
- The next correct step is not another pseudo-generation rerun.
- The next correct step is a short smoke-train on fold `0` using these cached pseudo labels.

## 2026-03-23 Exp_009 Fold 0 Training Result

### Confirmed Result

- Best epoch: `2 / 6`
- Best macro ROC-AUC: `0.8494899256`
- Best valid loss: `0.0684762717`
- Scored classes: `29`
- Pseudo rows used: `69593`
- Teacher folds: `[1, 2]`

### Epoch Curve

- epoch `1`: `0.8306`
- epoch `2`: `0.8495`
- epoch `3`: `0.8343`
- epoch `4`: `0.7950`
- epoch `5`: `0.8253`
- epoch `6`: `0.8028`

### Interpretation

- This is the strongest raw fold-0 native training result in the project so far.
- On the same sparse fold, `exp_009` beats:
  - `exp_006` fold `0` by about `+0.0699`
  - raw `exp_008` fold `0` by about `+0.0117`
  - postprocessed `exp_008b` by about `+0.0060`
- That is exactly the kind of signal we wanted from the noisy-student branch:
  - pseudo labels did not destabilize training
  - the student improved materially over the short-context supervised baseline
  - the branch is competitive even against the stronger long-context local fold result
- The curve also suggests early saturation:
  - the best score came at epoch `2`
  - later epochs regressed
  - so future runs may need stronger early stopping or slightly lighter late-epoch learning

### Important Caution

- The result is still only one sparse fold with `29` scored classes.
- So this is a strong positive direction, but not yet enough evidence for immediate Kaggle promotion.
- We already learned from `exp_008` that single-fold enthusiasm can be misleading.

### Practical Conclusion

- `exp_009` is now the most promising native training branch.
- The next high-signal steps are:
  - apply the proven priors/texture postprocess on top of `exp_009` exports
  - run at least one more fold
  - only then decide whether it should replace `exp_007` as the default native submit path

## 2026-03-23 Exp_009b Priors Postprocess Result

### Confirmed Result

- Raw `exp_009` fold `0`: `0.8494899256`
- Best postprocess variant: `raw`
- Delta vs raw: `+0.0000`
- All prior variants regressed:
  - `event_texture_priors`: `0.8112`
  - `event_texture_priors_smooth`: `0.8108`
  - `texture_priors_only`: `0.8075`
  - `event_priors_only`: `0.7964`

### Interpretation

- This is one of the most informative negative results in the project so far.
- Earlier native branches depended heavily on inference-time priors.
- `exp_009` does not.
- More than that, the old priors are now miscalibrated enough to hurt strongly.
- The most plausible explanation is that noisy-student training has already internalized a meaningful part of:
  - site/hour structure
  - texture-heavy background patterns
  - target-domain calibration that earlier supervised branches lacked

### Practical Conclusion

- `exp_009` should currently be treated as a raw model branch first.
- We should not automatically inherit the old `exp_007` postprocess into the noisy-student branch.
- The next correct test is not another postprocess tweak.
- The next correct test is another `exp_009` fold, and then a raw Kaggle submission if the gain survives.

## 2026-03-23 Exp_009 Fold 1 Result

### Confirmed Result

- Fold `1` best macro ROC-AUC: `0.8768579395`
- Best epoch: `3 / 6`
- Scored classes: `29`
- Best valid loss: `0.0457730132`
- Pseudo rows / files: `69640 / 9569`
- Teacher folds: `[0, 2]`
- Pseudo confidence mean: `0.6845`

### Interpretation

- This is the strongest native fold result in the project so far.
- Fold `1` improves over fold `0` by about `+0.0274`.
- Across folds `0-1`, the branch now averages `0.8632`, which is materially above the earlier supervised native branches.
- The result also weakens the idea that fold `0` was a lucky spike:
  - the second fold stayed in the same sparse validation regime
  - and still improved again
- This makes the raw noisy-student path the first native branch that looks genuinely promotion-worthy rather than merely interesting.

### Practical Conclusion

- `exp_009` should remain raw by default.
- The next high-signal step is now fold `2`, not more postprocess work.
- If fold `2` stays in the same quality band, the branch deserves its first raw Kaggle submission.

## 2026-03-24 Exp_009 Fold 2 Result

### Confirmed Result

- Fold `2` best macro ROC-AUC: `0.8848806193`
- Best epoch: `3 / 6`
- Scored classes: `35`
- Best valid loss: `0.0452111292`
- Pseudo rows / files: `70142 / 9669`
- Teacher folds: `[0, 1]`
- Pseudo confidence mean: `0.6458`

### Interpretation

- This is the strongest `exp_009` fold so far.
- More importantly, it is also the broadest fold so far, with `35` scored classes instead of `29`.
- That sharply reduces the risk that the branch is only exploiting an easy sparse split.
- The three-fold picture is now:
  - fold `0`: `0.8495`
  - fold `1`: `0.8769`
  - fold `2`: `0.8849`
  - mean: `0.8704`
- This is the first native branch in the project that looks both strong and stable enough to justify direct leaderboard validation in raw form.

### Practical Conclusion

- The main local validation question is answered.
- `exp_009` should be promoted to the next Kaggle candidate.
- The correct next step is the first raw `exp_009` submission, not more local postprocess ablations.

## 2026-03-24 Exp_009 First Raw Kaggle Result

### Confirmed Result

- Raw `exp_009` 3-fold Kaggle LB: `0.735`
- Previous best native public result: `exp_007 = 0.758`
- Earlier single-fold native hybrid: `exp_005 = 0.737`

### Interpretation

- This is a strong negative transfer result.
- The branch looked excellent locally:
  - fold `0`: `0.8495`
  - fold `1`: `0.8769`
  - fold `2`: `0.8849`
  - mean: `0.8704`
- But that local picture did not survive leaderboard validation.
- So the issue is not that `exp_009` lacks local signal.
- The issue is that our current summary for it is still not honest enough as a leaderboard proxy.
- In other words:
  - raw noisy-student training improved local folds
  - but the branch is still miscalibrated or otherwise mismatched to the hidden test domain

### Practical Conclusion

- `exp_009` should not replace `exp_007` as the default native public baseline yet.
- The next correct step is not another blind Kaggle submission.
- The next correct step is a pooled OOF / calibration analysis for `exp_009`.

## 2026-03-24 Exp_009c Pooled OOF / Calibration Analysis

Source:
- `notebooks/exp_009c_noisy_student_pooled_oof_analysis.ipynb`

### Confirmed Result

- Raw pooled OOF macro ROC-AUC: `0.7933963541`
- Raw fold-mean macro ROC-AUC: `0.8704094948`
- Optimism gap: `0.0770131407`
- Best fixed variant: `raw`
- Fold-safe selected macro ROC-AUC: `0.7852925095`
- Temperature-scaled raw macro ROC-AUC: `0.7674671923`
- Raw pooled Brier: `0.0091121`
- Raw pooled ECE: `0.00708`
- Temperature scaling improved calibration:
  - Brier: `0.0090323`
  - ECE: `0.00369`
  - but not ranking quality

### Interpretation

- This analysis explains the main local-vs-Kaggle disconnect much better than the fold means did.
- `exp_009` is still a real branch:
  - pooled OOF is not weak in absolute terms
  - and it remains clearly above the old `exp_007` pooled OOF reference (`0.7109`)
- But its fold summary was substantially too optimistic.
- More importantly, the calibration story is now much clearer:
  - raw remains the best fixed OOF variant
  - fold-safe context repair does not beat raw
  - both weak and old priors hurt
  - temperature scaling improves probability calibration but not AUC
- So the current problem is not “we forgot one small inference trick”.
- The current problem is that this branch still does not transfer cleanly to the hidden leaderboard domain.

### Practical Conclusion

- No lightweight postprocess rescue has justified another `exp_009` leaderboard attempt.
- `exp_009` remains useful research signal, but not the next promoted submit path.
- The next high-signal move should be a new modeling branch, with the `HGNetV2-B0` supervised reference now looking especially attractive.

## 2026-03-23 BirdCLEF 2026 HGNetV2-B0 Baseline (`0.856`) Analysis

Source:
- `references/private-solutions/birdclef2026-score=0.856/hgnetv2_b0_baseline.docx`
- `references/private-solutions/birdclef2026-score=0.856/birdclef-2026-hgnetv2-b0-baseline-training.ipynb`
- `references/private-solutions/birdclef2026-score=0.856/birdclef-2026-hgnetv2-b0-baseline-inference.ipynb`
- `references/private-solutions/birdclef2026-score=0.856/birdclef-2026-download-wheels.ipynb`

### What The Method Actually Is

- This is a strong supervised baseline rather than a foundation-model stack.
- Backbone:
  - `hgnetv2_b0.ssld_stage2_ft_in1k` from `timm`
- Input:
  - `32 kHz`
  - `5s` audio crops
  - single-channel log-mel image resized to `(256, 256)`
- Loss:
  - plain `nn.BCEWithLogitsLoss`
- Training data:
  - `train_audio`
  - segmented `train_soundscapes`
  - `secondary_labels` from `train_audio` merged into the target vector
- CV:
  - custom `MultiLabelStratifiedGroupKFold`
  - groups are `audio_id`
  - `4` folds
- Augmentation:
  - spectrogram-space MixUp with `alpha=1.0`, `theta=0.8`
- Scheduler:
  - `OneCycleLR`
- Inference:
  - convert trained PyTorch fold models to ONNX and then OpenVINO
  - CPU inference with async request queue
  - optional rank averaging across folds

### Why The Score Is Good

- The score does not come from one exotic trick.
- It comes from combining a few pragmatic choices that fit the competition well:
  - stronger image-style backbone than our current B0 baseline
  - direct use of labeled soundscape segments inside supervised training
  - multi-label targets through `secondary_labels`
  - very fast audio IO, which makes full 4-fold training cheap enough to run often
  - efficient CPU-safe inference through OpenVINO
- The writeup reports:
  - OOF: `0.9574`
  - public LB: `0.856`
- The OOF is probably not directly comparable to our pooled soundscape OOF because it mixes isolated recordings and segmented soundscapes in one broad CV protocol.
- Still, the public `0.856` confirms that the supervised recipe is genuinely strong.

### High-Value Transfer Ideas

- Preconvert `train_audio` from `.ogg` to `.wav` and use partial reads through `soundfile.SoundFile`.
  - This is the single most practical engineering idea in the solution.
  - It cuts training time enough to make 4-fold supervised training realistic.
- Build a unified supervised dataframe from:
  - `train_audio` with `primary + secondary`
  - labeled `train_soundscapes` cut into explicit supervised clips
- Use grouped multi-label CV by `audio_id`.
  - This is a cleaner supervised protocol than naive random clip splitting.
- Try `HGNetV2-B0` as a backbone candidate.
  - The solution suggests that a stronger efficient image backbone can beat our current EfficientNet-B0 baseline even without complicated downstream machinery.
- Keep the head simple at first.
  - A plain classification head already reaches a good public score here.
- Export the best native models to OpenVINO for Kaggle inference.
  - This is highly relevant for our future CPU-limited submission notebooks.

### Important Details About Their Data Handling

- `train_soundscapes_labels.csv` is deduplicated first.
- The notebook then writes explicit soundscape clip files:
  - contiguous rows with the same `filename` and `primary_label` are merged into longer segments
  - these are saved as `.wav`
- Those soundscape-derived clips are treated as labeled training examples.
- `train_audio` rows keep multi-label targets using `primary_label + secondary_labels`.
- This means the method does target-domain adaptation already at the supervised dataset level, not only through postprocessing.

### Inference Engineering Ideas

- They do not run a heavy neural postprocess stack.
- Instead they optimize the submission path:
  - precompute log-mels in batches with `joblib`
  - compile fold models with OpenVINO
  - run async CPU inference
  - average folds in probability space by default
  - optionally test rank averaging
- This is very useful for us later even if we keep a different training recipe.

### What To Treat Carefully

- The very high OOF should not be treated as a reliable leaderboard proxy for our project.
- The baseline still uses only `5s` independent clips and a plain linear head.
  - So it is strong, but probably not the final ceiling.
- The inference notebook reads only the first `60s` of each file and slices fixed `5s` windows.
  - That matches the competition format, but it is not a general soundscape detector.
- There is no metadata-prior layer here.
  - So the method is complementary to `exp_007`, not a replacement for our soundscape-aware inference ideas.

### Practical Conclusion

- This is one of the most useful references in the repository because it is strong, simple, and reproducible.
- The most transferable ideas for our project are:
  - wav-cache plus partial-read audio loading
  - unified supervised training on `train_audio + labeled soundscape clips`
  - `HGNetV2-B0` backbone test
  - OpenVINO export for native submission notebooks
- This reference suggests a valuable future branch:
  - a fast supervised native branch that is simpler than Perch and cheaper than noisy-student training
  - especially as a comparison branch once `exp_009` is finished

## 2026-03-24 Exp_011 HGNetV2 Supervised Branch Scaffold

### What Was Turned Into A Notebook

- New experiment notebook:
  - `notebooks/exp_011_hgnetv2_soundscape_supervised.ipynb`
- Core recipe kept from the `0.856` reference:
  - `hgnetv2_b0.ssld_stage2_ft_in1k`
  - unified supervised dataframe
  - grouped multi-label folds by `audio_id`
  - `soundfile.SoundFile` partial reads
  - log-mel frontend with `(256, 256)` resizing
  - spectrogram MixUp
  - optional wav-cache for `train_audio`

### Important Local Data Nuance Confirmed

- The local `train_soundscapes_labels.csv` does **not** behave like one-label-per-row data.
- It stores multi-label segment strings separated by `;`.
- After deduplication and token expansion:
  - unique labeled soundscape segments: `739`
  - expanded per-label rows: `3122`
  - target classes covered: `75`
- After contiguous same-label merging inside each file:
  - supervised soundscape clips: `529`
  - files covered: `66`

### Why This Matters

- The first naive notebookization attempt almost treated the raw `primary_label` field as a single class string.
- That would have collapsed most of the soundscape supervision incorrectly.
- Fixing this inside the notebook makes `exp_011` the cleanest supervised soundscape-clip branch we have built so far.

### Setup Validation

- The notebook now passes full setup execution in the local `.venv` with `RUN_TRAINING = False`.
- Fold `0` ready state:
  - train rows: `26996`
  - valid rows: `9082`
  - train soundscape rows: `393`
  - valid soundscape rows: `136`
- No wav cache exists yet, so the first run will use direct `ogg` offset reads unless cache building is enabled manually.

### Practical Conclusion

- `exp_011` is ready for the first real fold run.
- This branch is attractive because it tests a stronger supervised backbone and cleaner target-domain supervision without the complexity of Perch or noisy-student inference stacks.

## 2026-03-24 Exp_011 Fold 0 Result

### Confirmed Result

- `exp_011` fold `0` completed successfully on `mps`.
- Best epoch:
  - `8 / 12`
- Best checkpoint selection metric:
  - soundscape-only macro ROC-AUC `0.8508523324`
- Overall validation macro ROC-AUC at the selected epoch:
  - `0.9555301905`
- Overall scored classes:
  - `220`
- Soundscape-only scored classes:
  - `42`
- Best validation loss at the selected epoch:
  - `0.0123390177`

### Why This Is Important

- This is the strongest native supervised fold we have produced so far.
- It is also a much healthier validation signal than the older soundscape-only branches because the soundscape subset now scores `42` classes instead of the older `29-35` range.
- The result sits very close to the external supervised reference OOF scale while still being fully repository-native.

### Learning Curve Interpretation

- Overall macro ROC-AUC keeps increasing through the later epochs:
  - epoch `8`: `0.9555`
  - epoch `12`: `0.9585`
- But soundscape-only macro ROC-AUC peaks at epoch `8`:
  - epoch `8`: `0.8509`
  - epoch `9`: `0.8277`
  - epoch `10`: `0.8323`
  - epoch `11`: `0.8353`
  - epoch `12`: `0.8354`
- This is exactly the type of domain-gap signal we wanted to capture:
  - the model can still improve on the mixed validation fold
  - while already drifting away from the target-domain soundscape objective

### Practical Conclusion

- The HGNetV2 branch has now earned immediate expansion to folds `1-2`.
- The best checkpoint criterion should remain soundscape-aware, not generic full-fold AUC or raw validation loss.
- If folds `1-2` stay strong, this branch becomes the next most justified native Kaggle submission candidate.

## 2026-03-25 Exp_011 Folds 1-2 Result

### Confirmed Result

- Fold `1`:
  - best epoch: `4 / 12`
  - overall macro ROC-AUC: `0.9406223787`
  - soundscape-only macro ROC-AUC: `0.8042338925`
  - soundscape-only scored classes: `37`
- Fold `2`:
  - best epoch: `9 / 12`
  - overall macro ROC-AUC: `0.9622501589`
  - soundscape-only macro ROC-AUC: `0.8543629181`
  - soundscape-only scored classes: `52`
- Mean across folds `0-2`:
  - overall macro ROC-AUC: `0.9528009094`
  - soundscape-only macro ROC-AUC: `0.8364830477`

### Interpretation

- Fold `1` is visibly weaker than folds `0` and `2`, but it is still a strong result rather than a collapse.
- Fold `2` is especially important because it covers the broadest soundscape subset so far (`52` scored classes) and still remains extremely strong.
- The branch therefore looks stable enough for a first Kaggle check.

### Repeated Pattern Worth Keeping

- Fold `1` repeats the same lesson as fold `0` even more strongly:
  - full-fold macro ROC-AUC keeps rising all the way to epoch `12`
  - soundscape-only macro ROC-AUC peaks much earlier at epoch `4`
- Fold `2` is slightly gentler, but still peaks on soundscape metric before the final epoch.
- So `exp_011` has now clearly validated soundscape-aware checkpointing across multiple folds, not just once.

### Practical Conclusion

- `exp_011` has passed the “multi-fold stability” bar.
- The next justified step is no longer another local fold.
- The next justified step is the first Kaggle submission for the `exp_011` branch.
- That submission package is now prepared as:
  - `notebooks/kaggle_submission_exp_011_hgnetv2_3fold.ipynb`
  - `submissions/kaggle_datasets/birdclef-exp011-hgnetv2-3fold`

## 2026-03-25 Exp_011 First Kaggle Result

- Public leaderboard score: `0.844`
- Previous best native public result: `exp_007 = 0.758`
- Strong reference blend: `0.890`

Interpretation:
- This is a strong positive transfer result.
- `exp_011` improves the repository-native public baseline by `+0.086` over `exp_007`.
- The branch is now the strongest repository-native public path in the project.
- The gap to the stronger reference blend shrinks from `0.132` (`0.758 -> 0.890`) to only `0.046` (`0.844 -> 0.890`).

Research implication:
- The `HGNetV2 + unified train_audio + labeled soundscape clips` branch is not just locally strong; it transfers meaningfully to the hidden leaderboard.
- This makes `exp_011` the first native branch that is good enough to serve as a serious launchpad for the final solution.
- The next strategic choice is now between:
  - scaling `exp_011` further with more folds / stronger inference engineering
  - or jumping to the simplified `0.924` Perch ProtoSSM branch as the next major modeling leap

## 2026-03-25 Pantanal Distill ProtoSSM (`0.924`) Analysis

Source:
- `references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.924.ipynb`

What this notebook really is:
- Not a plain `Perch + priors` submission.
- It builds a full downstream stack on top of `Perch v2` embeddings and logits over all `12` windows of each `60s` file.
- The trusted training subset is still the `59` fully labeled soundscape files, cached through `perch_meta`-style arrays.

Core pipeline:
- `Perch -> ProtoSSM(pass1) + MLP probe -> weighted fusion -> ResidualSSM(pass2) -> TTA -> temperature -> file-level scaling -> rank-aware scaling -> delta smoothing -> threshold sharpening`

What appears genuinely new relative to the earlier Perch notebooks:
- File-level temporal modeling of the full `12 x 5s` sequence, instead of windowwise stacking only.
- Bidirectional selective SSM layers plus cross-attention over windows.
- Metadata embeddings (`site`, `hour`) injected inside the temporal model, not only as post-hoc priors.
- Prototypical head with learnable class prototypes and cosine-similarity logits.
- Gated fusion between the learned temporal model and raw Perch logits on a per-class basis.
- Multi-task training: focal BCE, label smoothing, distillation toward Perch logits, taxonomic auxiliary loss, mixup, and SWA.
- A second-pass residual SSM that learns additive corrections after the first ensemble.

What is probably doing most of the work:
- The biggest conceptual jump is the file-level temporal model on top of `Perch` embeddings/logits.
- The second biggest is moving metadata from pure postprocess priors into the model through `site/hour` embeddings.
- The `MLP probe + sequential features` branch is still important, but it now acts as one leg of a larger ensemble rather than the whole downstream story.

What looks more leaderboard-specific or lower-priority for first reproduction:
- Per-class threshold sharpening.
- The long postprocess chain after the main fusion.
- The residual SSM second pass before we have reproduced the simpler first-pass ProtoSSM branch.

Research implication:
- The strongest current external ceiling in this repo is no longer the older `0.899` Perch stack, but this `0.924` temporal downstream branch.
- This suggests that, for the Perch direction, the next serious experiment should be a simplified `Perch embeddings + file-level temporal model` reproduction rather than another minor priors/probe tweak.

## 2026-03-25 Exp_012 Perch Temporal Light Branch

- Notebook created:
  - `notebooks/exp_012_perch_temporal_light.ipynb`
- Status:
  - scaffolded and AST-clean
  - first grouped OOF run still pending

### Why This Branch Matters

- `exp_011 = 0.844` has now proved that a strong native supervised branch can transfer to Kaggle.
- But the new external ceiling is much higher:
  - `0.924` from the Pantanal Distill / ProtoSSM Perch stack
- That makes `exp_012` a very useful control experiment:
  - not just "can we score higher?"
  - but "how much extra structure do strong foundation embeddings expose once we model the whole file sequence?"

### What Exp_012 Intentionally Keeps

- cached `Perch v2` logits and embeddings from `data/perch_meta`
- file-level `12 x 5s` sequence modeling instead of independent window stacking
- in-model `site` and `hour` embeddings
- a prototype head
- per-class gated fusion between learned temporal logits and raw Perch logits

### What Exp_012 Intentionally Drops For Now

- residual second-pass SSM correction
- long leaderboard-oriented postprocess chain
- threshold sharpening
- heavier ensemble logic

### Research Value

- If `exp_012` is strong locally, we get evidence that the next major gains are not just from better labels or stronger native supervision.
- They are from better temporal modeling of a richer embedding space.
- If `exp_012` underperforms `exp_011`, that is also a valuable result:
  - it would suggest our supervised HGNet reformulation is already capturing much of the competition signal without the full external stack.

### Immediate Next Step

- Run `exp_011` fold `3` to create the first `4-fold` native submission candidate.
- In parallel, run grouped OOF for `exp_012` and compare it first against:
  - `exp_003` Perch downstream reproduction
  - `exp_011` soundscape-only folds

## 2026-03-25 Exp_012 First Grouped OOF Result

- Raw Perch AUC on the cached fully labeled files:
  - `0.7390`
- `exp_012` grouped pooled OOF AUC:
  - `0.6248`
- Delta vs raw:
  - `-0.1142`
- Fold AUCs:
  - fold `1`: `0.8337`
  - fold `2`: `0.6862`
  - fold `3`: `0.8705`
- Fold mean:
  - `0.7968`

### Interpretation

- This is a strong warning against trusting fold means in this branch.
- The fold-level scores look decent, but the honest pooled OOF score is much worse and even drops below raw Perch.
- So the current `ProtoTemporalLight` stack is not yet learning a useful file-level correction.

### Most Likely Takeaway

- The failure is probably not "Perch is bad".
- It is more likely one of these:
  - the current SSM/prototype/fusion stack is too ambitious for only `59` trusted files
  - the grouped split by `site` is highly imbalanced, so the model overfits fold-specific structure
  - the gated fusion is already hurting before the temporal branch has learned a stable representation

### Research Implication

- This is still a valuable result.
- It tells us the simplified ProtoSSM direction should not be promoted as-is.
- The next useful move is not Kaggle submission, but a cleaner ablation ladder:
  - raw Perch
  - simple MLP / probe on file sequences
  - temporal model without prototype head
  - temporal model without gated fusion

## 2026-03-25 Exp_012b Ablation Notebook

- Notebook created:
  - `notebooks/exp_012b_perch_temporal_ablation.ipynb`

### Purpose

- Turn the negative `exp_012` result into a precise diagnosis rather than abandoning the Perch direction entirely.
- Compare four variants under the same grouped pooled OOF protocol:
  - `raw_perch`
  - `pooled_mlp_rawfeat`
  - `ssm_linear`
  - `ssm_linear_rawfeat`

### Why This Is The Right Next Step

- It removes the two highest-risk pieces from `exp_012`:
  - prototype head
  - gated fusion
- That makes the next result much easier to interpret:
  - if even `pooled_mlp_rawfeat` loses to raw Perch, the issue is probably the training protocol or grouped split
  - if `pooled_mlp_rawfeat` helps but `ssm_*` hurts, the issue is the temporal block
  - if `ssm_linear_rawfeat` wins, the Perch temporal path is still alive but needed a much simpler downstream head

## 2026-03-25 Exp_012b First Grouped OOF Result

- `raw_perch`: `0.7390`
- `pooled_mlp_rawfeat`: `0.6176`
- `ssm_linear`: `0.3355`
- `ssm_linear_rawfeat`: `0.4286`
- Best variant:
  - still `raw_perch`

### Interpretation

- This is a very strong negative control.
- The problem is not only the heavy prototype / gated-fusion stack from `exp_012`.
- Even much simpler learned downstream variants still regress materially below raw Perch on honest pooled OOF.

### What This Tells Us

- On the current trusted `59` files and site-grouped split, the local Perch direction is not yet a stable upgrade path.
- That could mean:
  - the available trusted subset is too small for these learned downstream models
  - the grouped split is so harsh that simple local training does not recover robust cross-site behavior
  - the strong external `0.924` notebook owes a lot more to its full recipe than to any one simplified component we can peel off easily

### Research Implication

- This is actually a high-value result for the project.
- We now have a clearer contrast:
  - `exp_011` gives a stable, real Kaggle-positive native path (`0.850`)
  - the current simplified Perch local branches (`exp_012`, `exp_012b`) are not yet competitive even with raw Perch on grouped pooled OOF
- So the Perch line should be treated as a high-ceiling but currently unresolved research path, not as the immediate main optimization branch.

## 2026-03-25 Exp_011 Fold 3 And 4-Fold Promotion

- Fold `3` best epoch:
  - `9 / 12`
- Fold `3` overall macro ROC-AUC:
  - `0.9662`
- Fold `3` soundscape-only macro ROC-AUC:
  - `0.7992`
- Fold `3` soundscape-only scored classes:
  - `39`

### Interpretation

- Fold `3` is weaker than folds `0` and `2`, but still fully consistent with the branch being strong.
- More importantly, it reduces the risk that we are over-trusting an optimistic three-fold picture.
- The four-fold summary is now:
  - overall macro ROC-AUC mean: `0.9562`
  - soundscape-only macro ROC-AUC mean: `0.8272`

### Practical Outcome

- `exp_011` is now promotion-ready for a second Kaggle test as a `4-fold` ensemble.
- Submission assets prepared:
  - `notebooks/kaggle_submission_exp_011_hgnetv2_4fold.ipynb`
  - `submissions/kaggle_datasets/birdclef-exp011-hgnetv2-4fold`

### Why This Matters Research-Wise

- We now have a fuller answer to the native supervised hypothesis:
  - strong transfer was real at `0.844`
  - but the branch also has non-trivial fold variance on the soundscape-only metric
- That makes the next Kaggle submission more scientifically meaningful than another local ablation:
  - if `4-fold` improves, the gain is likely coming from a more stable ensemble estimate
  - if it does not, that is a strong sign that the remaining gap is architectural, not just fold count

## 2026-03-25 Exp_011 Second Kaggle Result (`4-fold`)

- Public leaderboard score:
  - `0.850`
- Previous `3-fold` score:
  - `0.844`
- Delta from adding fold `3`:
  - `+0.006`

### Interpretation

- This is a real and useful gain, so the fold-expansion idea was worth trying.
- But the gain is small enough that it changes the strategic interpretation of the branch:
  - `exp_011` is now a strong stabilized native baseline
  - it is probably not hiding another big jump from "just one more fold" or another tiny inference tweak

### Research Implication

- We now have a much firmer native anchor for the whole project:
  - repository-native public score `0.850`
  - gap to reference blend `0.890`: `0.040`
  - gap to external ProtoSSM ceiling `0.924`: `0.074`
- This is exactly the point where the next experiment should become more explanatory, not just more incremental.
- So the highest-value next step is no longer more HGNet bookkeeping.
- It is the first grouped OOF run of `exp_012`, because that branch can answer a deeper question:
  - does file-level temporal modeling over strong Perch embeddings unlock a qualitatively different source of gains than our native supervised path?

## 2026-03-25 Exp_014 Setup Validation

- New branch:
  - `exp_014_hgnetv2_pseudolabel`
- Notebook:
  - `notebooks/exp_014_hgnetv2_pseudolabel.ipynb`
- Parent branch:
  - `exp_011`

### Why This Branch Matters

- `exp_009` already told us that pseudo-labeling can create strong local signals.
- But it used the older EfficientNet-based native stack and transferred poorly to Kaggle.
- `exp_014` is a cleaner scientific test:
  - keep the strongest successful native supervised recipe
  - change only the training signal by adding pseudo-labeled soundscape windows

### Setup Readout

- Fold validated:
  - `0`
- Teacher folds:
  - `[1, 2, 3]`
- Labeled rows:
  - `36078`
- Fold `0` train / valid rows:
  - `26991 / 9087`
- Fold `0` valid soundscape rows:
  - `144`
- Pseudo manifest rows / files:
  - `127104 / 10592`

### Research Implication

- This is now the most informative next native modeling branch in the repository.
- If `exp_014` improves over `exp_011`, we learn that the missing gain is in target-domain supervision rather than only architecture choice.
- If it fails, that is equally useful:
  - `exp_011` may already be close to saturating what this training family can extract from the available labels
  - and the next large jump likely requires either a faithful `0.924` external stack or a genuinely different native architecture

## 2026-03-26 Exp_014 Fold 0 First Readout

- Pseudo generation summary:
  - retained pseudo rows / files: `20688 / 5471`
  - mean confidence: `0.5025`
  - `p75` confidence: `0.6014`
  - retain rate vs manifest: about `16.28%`

### Training Result

- Fold `0` best epoch:
  - `5 / 8`
- Best soundscape-only macro ROC-AUC:
  - `0.8684479825`
- Best overall macro ROC-AUC seen during the run:
  - `0.9660572606`
- Soundscape-scored classes:
  - `46`

### Interpretation

- This is the strongest first-fold result we have seen from a native pseudo-label continuation.
- It is also a cleaner comparison than `exp_009` because the parent branch is already leaderboard-positive and the validation protocol is aligned with `exp_011`.
- Direct fold `0` comparison:
  - `exp_011`: `0.8509`
  - `exp_014`: `0.8684`
  - delta: `+0.0176`

### Research Implication

- This is the first concrete sign that pseudo labels may help more when layered on top of a strong supervised soundscape-aware branch than when attached to the older EfficientNet recipe.
- That makes `exp_014` one of the highest-value current experiments in the project.
- But it is still only one fold, so the next scientific question is stability, not Kaggle score yet.

## 2026-03-26 Exp_014 Fold 1 Update

- Fold `1` best epoch:
  - `1 / 8`
- Best soundscape-only macro ROC-AUC:
  - `0.8153869305`
- `exp_011` fold `1` comparison:
  - `0.8042338925`
- Delta:
  - `+0.0112`

### Interpretation

- This is a second positive fold for the HGNetV2 pseudo-label continuation.
- The gain is smaller than on fold `0`, but it still survives the split change.
- That matters more than the absolute number, because it means `exp_014` is no longer behaving like a one-fold anomaly.

### Important Nuance

- Fold `1` peaked immediately at epoch `1`.
- Later epochs hovered just below the best value instead of collapsing, which suggests:
  - the pseudo labels are still useful
  - but this split may saturate much earlier than fold `0`
- So if the same pattern repeats on fold `2`, early stopping may matter more than a longer cosine tail.

## 2026-03-26 Exp_014 Fold 2 Update

- Fold `2` best epoch:
  - `6 / 8`
- Best soundscape-only macro ROC-AUC:
  - `0.8300901176`
- `exp_011` fold `2` comparison:
  - `0.8543629181`
- Delta:
  - `-0.0243`

### Interpretation

- This is the first fold where the pseudo-label continuation loses clearly to the supervised baseline.
- That matters a lot because fold `2` was one of the strongest `exp_011` folds, so this is not just harmless noise around a weak split.

### Research Implication

- `exp_014` is now a genuinely mixed branch.
- That is still useful:
  - it suggests pseudo labels may help some grouped splits while hurting others
  - which is exactly the kind of instability we would want to diagnose before any Kaggle promotion
- The current three-fold summary is:
  - `exp_014`: `0.8380`
  - `exp_011`: `0.8365`
  - mean delta: only `+0.0015`
- So the next question is no longer “is pseudo-labeling promising at all?”
- It is:
  - “is this branch truly better after the last fold, or are we looking at a near-tie with higher variance?”

## 2026-03-26 Exp_014 Fold 3 And Final Four-Fold Verdict

- Fold `3` best soundscape-only macro ROC-AUC:
  - `0.7702235931`
- `exp_011` fold `3` comparison:
  - `0.7992063670`
- Delta:
  - `-0.0290`

### Four-Fold Summary

- `exp_014` mean soundscape-only macro ROC-AUC:
  - `0.8210`
- `exp_011` mean soundscape-only macro ROC-AUC:
  - `0.8272`
- Mean delta:
  - `-0.0061`

### Final Interpretation

- `exp_014` is not a real upgrade over the supervised HGNetV2 branch.
- The branch is still valuable scientifically:
  - it shows that pseudo-label continuation can improve some folds
  - but it also shows that the same recipe can degrade the broader fold picture enough to lose on the full four-fold view
- This is exactly the kind of result that saves wasted Kaggle attempts:
  - without the four-fold readout, the first two folds could easily have led to an overconfident promotion

## 2026-03-26 Exp_014b Setup

- New branch:
  - `exp_014b_hgnetv2_pseudolabel_strict`
- Notebook:
  - `notebooks/exp_014b_hgnetv2_pseudolabel_strict.ipynb`

### Motivation

- `exp_014` failed overall, but not in a way that kills the whole idea.
- The most plausible failure mode is:
  - too many medium-confidence pseudo rows
  - too much pseudo influence
  - pseudo supervision arriving too early

### What Changes In `exp_014b`

- `pseudo_min_confidence: 0.45`
- `pseudo_loss_weight: 0.25`
- `max_pseudo_segments_per_file: 4`
- `max_pseudo_segments_total: 12000`
- `pseudo_start_epoch: 2`
- `epochs: 6`
- `learning_rate: 2e-4`

### Setup Readout

- Safe setup validated on fold `0`
- Same grouped supervised frame as `exp_014`
- Same fold `0` pseudo manifest size:
  - `127104` rows across `10592` files
- Teacher folds:
  - `[1, 2, 3]`

### Research Implication

- `exp_014b` is worth exactly because it is a narrow diagnostic branch.
- If it helps, we learn that pseudo labels themselves were not the problem; the problem was how aggressively we used them.
- If it still fails, that makes the case much stronger that this HGNetV2 pseudo-label direction should be paused in favor of the stronger external `0.924` line.

## 2026-03-26 Exp_014b Fold 0

- Best soundscape-only macro ROC-AUC:
  - `0.8682576606`
- `exp_014` fold `0` comparison:
  - `0.8684479825`
- Delta:
  - `-0.00019`

### Pseudo Cache Comparison

- `exp_014` pseudo rows / files:
  - `20688 / 5471`
- `exp_014b` pseudo rows / files:
  - `8724 / 3237`
- `exp_014b` mean confidence / `p75`:
  - `0.6358 / 0.7378`

### Interpretation

- This is a very strong diagnostic result.
- The stricter recipe keeps essentially the same validation quality while using far fewer pseudo rows.
- That strongly supports the idea that the instability of `exp_014` was caused by an overly aggressive pseudo recipe, not by the pseudo-label concept itself.

## 2026-03-27 Exp_015 Submit-Path Operationalization

- New notebook:
  - `notebooks/kaggle_submission_exp_015_pantanal_proto_ssm_v17.ipynb`
- Source reference:
  - `references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.924.ipynb`

### What Was Done

- The notebook was not reinterpreted as a local ablation.
- Instead, it was ported as a faithful Kaggle submit path with only the following changes:
  - dynamic TensorFlow 2.20 wheel discovery
  - dynamic competition directory discovery
  - dynamic `perch_v2_cpu` model discovery
  - dynamic cached full-file Perch output discovery
  - explicit failure in submit mode if the full-file cache is missing

### Why This Matters

- This is the first time the project has a real operational path for the strongest known external branch.
- That is important because `exp_012` and `exp_012b` strongly suggested that simplified local Perch reproductions were not the right way to capture this ceiling.
- `exp_015` therefore becomes the correct next Kaggle-facing test of the external high-ceiling hypothesis.

### First Kaggle Result

- The first Kaggle submission for `exp_015` scored `0.925` on the public leaderboard.
- This slightly exceeds the reported `0.924` reference score.
- It also becomes the new best public result in the repository:
  - vs `exp_001` reference blend: `+0.035`
  - vs `exp_011` native four-fold HGNetV2: `+0.075`

### Interpretation

- This is a major positive result for the project.
- It validates the decision to stop over-simplifying the external Perch / ProtoSSM family and instead operationalize one strong notebook faithfully.
- In hindsight, `exp_012` and `exp_012b` were still very useful:
  - they showed that light local reproductions were not enough
  - but they did not mean the external ceiling itself was weak
- So the current strategic split becomes much clearer:
  - `exp_015` is now the main Kaggle-facing path
  - `exp_011` remains the strongest repository-native branch
  - the next high-value question is whether `exp_015` and `exp_011` are complementary enough to justify a blend or ensemble

## 2026-03-27 Exp_016 Submission-Level Blend Scaffold

- New notebook:
  - `notebooks/kaggle_submission_exp_016_blend_exp015_exp011.ipynb`

### What It Does

- Takes one attached `submission.csv` from `exp_015`
- Takes one attached `submission.csv` from `exp_011`
- Aligns them against `sample_submission.csv`
- Produces a weighted blend

## 2026-03-29 Exp_016b Runtime Blend Refresh

- New notebook:
  - `notebooks/kaggle_submission_exp_016b_runtime_blend_exp015d_exp011.ipynb`

### Why A New Blend Branch Was Needed

- The original `exp_016` question remains valid, but its execution path became outdated after `exp_015d = 0.929`.
- The old CSV-based approach was also too brittle because Kaggle-downloaded artifacts were not reliable for direct row-aligned blending.
- `exp_016b` keeps the useful runtime `exp_011` inference block from the old work, but anchors it on the stronger artifactized V18 submit path.

### Default Hypothesis

- Start with a small native weight:
  - `0.95 * exp_015d + 0.05 * exp_011`
- If this helps, then the native branch still contributes diversity even after the external path reached `0.929`.

### First Outcome

- The first public run of `exp_016b` matched `exp_015d` exactly:
  - `0.929`
- So the simple runtime blend did not show useful public complementarity.
- This is still a valuable result because it tells us the current strongest native branch is not adding obvious generic ensemble gain on top of the V18 external path.
- Default first test:
  - `0.90 * exp_015 + 0.10 * exp_011`

### Why This Is The Right First Ensemble Test

- It is much cheaper and safer than building a first dual-model runtime notebook.
- It directly answers the complementarity question.
- It keeps the first ensemble experiment focused on signal, not engineering complexity.
- If this already helps, we can justify heavier ensemble work later.

## 2026-03-26 HGNetV2 Inference Notebook `0.859`

- New file reviewed:
  - `references/private-notebooks/birdclef-2026-hgnetv2-b0-baseline-inference-0.859.ipynb`
- Compared against:
  - `references/private-solutions/birdclef2026-score=0.856/birdclef-2026-hgnetv2-b0-baseline-inference.ipynb`

### Main Finding

- The two notebooks are operationally the same at the source-code level.
- All cell sources match.
- The difference is not in the inference algorithm itself.
- So the reported `0.859` should not be interpreted as a new modeling or inference trick relative to the earlier `0.856` HGNetV2 solution.

### What The Notebook Actually Does

- Uses the same `HGNetV2-B0` model family and the same `4-fold` inference setup.
- Builds `256 x 256` log-mel spectrograms from `5s` windows at `32 kHz`.
- Converts trained fold checkpoints to OpenVINO IR and runs CPU async inference.
- Supports either:
  - plain probability averaging across folds
  - optional rank averaging (`RANK_AVG = False` by default)
- Reads one `60s` soundscape at a time, slices it into `12` non-overlapping `5s` windows, and predicts rows directly.

### Why This Matters

- This notebook does not introduce a new scientific direction.
- Its value is engineering confirmation:
  - the OpenVINO CPU-safe submit path is stable
  - the simple non-overlapping `5s` inference recipe is competitive enough to reach around the same public-LB range as the original HGNetV2 baseline
- For our roadmap, that means:
  - no new experiment should be created just from this notebook
  - the useful ideas were already extracted into the `exp_011` branch
  - if we want to improve the HGNet line further, we should focus on training changes (`exp_014`) or later ensembling, not on re-copying this inference notebook

## 2026-03-28 Pantanal Distill ProtoSSM (`0.927`) Analysis

- New file reviewed:
  - `references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.927.ipynb`
- Compared against:
  - `references/private-notebooks/pantanal-distill-birdclef2026-improvement-0.924.ipynb`
  - operational project port:
    - `notebooks/kaggle_submission_exp_015_pantanal_proto_ssm_v17.ipynb`

### Main Finding

- The `0.927` notebook is not a new modeling family.
- It is a V18-style continuation of the same Pantanal / ProtoSSM submit path that we already ported as `exp_015`.
- So the right interpretation is:
  - `0.927` is an incremental upgrade opportunity over `exp_015`
  - not a reason to abandon the current external line and start from scratch again

### Changes That Look Real And Active

- Larger first-pass ProtoSSM config:
  - `d_model: 320`
  - `n_ssm_layers: 4`
  - `n_prototypes: 2`
  - `meta_dim: 24`
  - `cross_attn_heads: 8`
- Updated train config:
  - `n_epochs: 80`
  - `lr: 8e-4`
  - `patience: 20`
  - stronger regularization / label smoothing / focal settings
- Stronger residual SSM config:
  - `d_model: 128`
  - `correction_weight: 0.35`
- Changed fusion coefficients:
  - `lambda_event = 0.45`
  - `lambda_texture = 1.1`
  - `lambda_proxy_texture = 0.9`
- Stronger probe defaults:
  - larger `MLPClassifier`
  - `pca_dim = 128`
  - looser `min_pos`
  - `alpha = 0.45`
- Inference/postprocess changes that really appear in the final path:
  - finer `threshold_grid`
  - `tta_shifts = [0]` (so effectively TTA is removed)
  - `rank_aware_power = 0.4`
  - `delta_shift_alpha = 0.20`
  - adaptive confidence-aware delta smoothing replaces the fixed delta smoother

### Changes That Appear Defined But Not Actually Wired In

- The notebook defines several new helpers, but they do not appear to be called later in the provided source:
  - cosine restart scheduler
  - `mixup_cutmix(...)`
  - `species_focal_loss(...)`
  - class-frequency weights via `CLASS_WEIGHTS`
  - isotonic calibration + threshold optimizer helper
  - ensemble-weight sweep helper
- So these should not be over-interpreted as proven causal contributors to the `0.927` score unless we later verify they are used in a different companion training notebook or asset pipeline.

### Practical Interpretation For The Project

- `exp_015 = 0.925` remains the strongest operational path right now.
- The `0.927` reference suggests that the next external improvement should be an incremental V18 upgrade on top of `exp_015`, not a wholesale rewrite.
- The highest-signal next port would therefore be:
  - keep the faithful `exp_015` engineering scaffold
  - port only the active V18 changes first
  - especially:
    - larger ProtoSSM / residual configs
    - updated probe config
    - adaptive delta smoothing
    - revised fusion lambdas
    - lighter/no-TTA setting (`[0]`)
- This is attractive because it has a realistic chance of improving score while avoiding unnecessary engineering risk from copying unused or unverified helper blocks.

### exp_015c Timeout Interpretation And Artifactized Response

- The first direct V18 submit attempt (`exp_015c_full_v18_submit_path`) hit Kaggle `Notebook Timeout` even after moving to `P100`.
- This does not point to hidden-test inference alone as the main issue.
- Inspection of the notebook shows that in submit mode it still performs heavy train-time work:
  - builds OOF meta-features when not already cached in working dir
  - trains the final `ProtoSSM`
  - trains all classwise `MLP probes`
  - may also train `ResidualSSM`
- Therefore the correct next step is not more micro-optimizing of the same monolithic submit notebook, but splitting the path into:
  - `exp_015c_v18_artifact_export.ipynb`
  - `kaggle_submission_exp_015d_v18_artifact_submit.ipynb`
- The export notebook saves the downstream artifacts once.
- The thin submit notebook then loads those artifacts and performs only hidden-test inference plus postprocess.
- This should preserve the V18 recipe much more faithfully while directly targeting the real timeout source.
- This artifactized path is now confirmed:
  - `kaggle_submission_exp_015d_v18_artifact_submit.ipynb`
  - public LB: `0.929`
  - relative gain over `exp_015`: `+0.004`
- This is an important result because it shows the V18 stack itself was strong enough all along; the blocking issue was the monolithic submit engineering, not the modeling direction.
- Practically, `exp_015d` now becomes the strongest overall Kaggle path in the repository.

### exp_015e Calibration-First Follow-Up

- The next low-risk refinement is now scaffolded as:
  - `notebooks/exp_015e_v18_calibrated_artifact_export.ipynb`
  - `notebooks/kaggle_submission_exp_015e_v18_calibrated_submit.ipynb`
- This branch does not rewrite the working V18 recipe again.
- Instead, it isolates one plausible remaining gain source from the `0.927` family:
  per-class isotonic calibration plus calibrated threshold export.
- The export notebook fits isotonic calibrators from OOF blended probabilities, then re-optimizes per-class thresholds on calibrated probabilities.
- The submit notebook stays thin and only loads those calibrators as extra artifacts, applying them before the existing file-level scaling and threshold sharpening.
- Research value:
  this should tell us whether there is still residual quality left in calibration, as opposed to architecture or training changes.
- Score value:
  it is one of the cheapest remaining ways to test for a `0.930+` move without destabilizing the proven `0.929` artifactized path.

### exp_015f Thin Calibration Refresh

- After the timeout on the heavier `exp_015e` export path, a thinner refresh branch is now prepared:
  - `notebooks/exp_015f_v18_calibration_refresh_export.ipynb`
- This branch no longer retrains `ProtoSSM`, probes, or `ResidualSSM`.
- Instead it loads the fixed `exp_015d` artifacts, replays that exact downstream stack on cached labeled soundscape rows, and updates only:
  - isotonic calibrators
  - calibrated per-class thresholds
  - artifact manifests
- This is less honest than full OOF calibration because it is an in-sample refresh.
- But it is much cheaper, much less timeout-prone, and is still a useful causal test of whether calibration alone can move the already strong `0.929` path.

### `luck-factor-0.928.ipynb` Analysis

- The newly added reference [luck-factor-0.928.ipynb](/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/references/private-notebooks/luck-factor-0.928.ipynb) is not a new modeling family.
- It is another monolithic `Perch -> ProtoSSM -> MLP probe -> ResidualSSM -> postprocess` notebook in the same V17/V18 lineage as the already studied `0.927` Pantanal notebook and our artifactized `exp_015d`.
- Its active config is very close to the V18 recipe we already operationalized:
  - larger `ProtoSSM` (`d_model=320`, `4` SSM layers, `2` prototypes, `meta_dim=24`, `8` attention heads)
  - larger `ResidualSSM`
  - updated fusion lambdas
  - stronger probe config (`pca_dim=128`, `hidden_layer_sizes=(256, 128)`)
  - `rank_aware_power=0.4`
  - `delta_shift_alpha=0.20`
  - V18 hardcoded per-class thresholds
- It still keeps the old monolithic engineering style:
  - explicit `DEVICE = torch.device("cpu")`
  - no artifact split
  - submit path still depends on in-notebook downstream training and wall-time guards
- This makes it less attractive than `exp_015d` as an operational path even though the public score is strong (`0.928`).
- Research interpretation:
  - the notebook reinforces that the V18 ProtoSSM family is real and reproducible across multiple variants
  - but it does not currently expose a clearly new causal ingredient beyond the already captured `exp_015d` stack
  - so it should be treated as confirmatory evidence, not as a new top-priority port

### `pantanal-distill-birdclef2026-improvement-a4dc68-0.930.ipynb` Analysis

- The newly added [pantanal-distill-birdclef2026-improvement-a4dc68-0.930.ipynb](/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/references/private-notebooks/pantanal-distill-birdclef2026-improvement-a4dc68-0.930.ipynb) does **not** appear to introduce a new code path.
- A direct notebook-source comparison shows:
  - code-cell content is identical to [luck-factor-0.928.ipynb](/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/references/private-notebooks/luck-factor-0.928.ipynb)
  - similarity versus `0.928`: `1.0000`
  - similarity versus the older `0.927` Pantanal notebook: `0.9894`
- So the `0.930` result should currently be interpreted as:
  - the same V18-family monolithic notebook
  - with a slightly better leaderboard outcome
  - not as evidence for a genuinely new architectural or inference idea
- Practical interpretation:
  - this strengthens confidence that the V18 ProtoSSM family can live in the `0.928-0.930` range
  - but it does **not** create a new porting priority beyond the already operationalized `exp_015d` / calibration-refresh line
- Research interpretation:
  - the score delta from `0.928` to `0.930` is more plausibly due to leaderboard variance, run context, or ancillary notebook state than to source-code novelty
  - therefore it should be treated as additional confirmatory evidence for the same family, not as a separate breakthrough

### `birdclef-2026-protossm-v5-time-optimised.ipynb` Runtime Interpretation

- The saved notebook source does **not** support the claim that it is a `7s` end-to-end replacement for our current path.
- The attached execution metadata shows:
  - papermill duration: `610.918562s` (about `10.2` minutes)
  - internal logged wall time near the end: `361.9s`
- Those two numbers are already much closer to our current operational runtime than to `7s`.
- The run saved inside the notebook is also not a true hidden-test submission replay:
  - it explicitly prints `Hidden test not mounted. Dry-run on first 20 train soundscapes.`
  - so the observed inference timing is not directly comparable to a scored hidden-test submit run
- There are also structural reasons the notebook can look faster:
  - smaller, older model family than `exp_015d` (`ProtoSSM` around `723,610` params in the saved run)
  - default submit-time thresholds (`0.5`) instead of artifactized V18 thresholds
  - monolithic path with simpler/older postprocess
- Practical interpretation:
  - it is not evidence of a magic engineering trick that makes our current `0.929` recipe obsolete
  - it is better understood as a lighter, older, dry-run-observed monolithic notebook whose apparent speed is easy to over-read

### `0-928-luck-factor-just-edit-run-instantly.ipynb`

- This notebook is not a new branch.
- Its code is an exact duplicate of:
  - `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/references/private-notebooks/luck-factor-0.928.ipynb`
  - `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/references/private-notebooks/pantanal-distill-birdclef2026-improvement-a4dc68-0.930.ipynb`
- Exact code hash match:
  - all three share the same code-cell MD5: `be9b3784195bcb6eee7e5519bc6ff432`
- Interpretation:
  - no new causal idea
  - no new engineering scaffold
  - just another renamed copy of the already studied monolithic V18 ProtoSSM family

### `bird26-reprod-perch-proto-residualssm-train-s7177.ipynb`

- This notebook is not a new modeling family.
- It stays in the same `Perch -> MLP probe -> ProtoSSM -> ResidualSSM -> postprocess` family as the existing Pantanal / V18 notebooks.
- Line-overlap comparison shows it is closer to the older `0.927` path than to our artifactized `exp_015d`:
  - Jaccard vs `pantanal-distill-birdclef2026-improvement-0.927.ipynb`: about `0.915`
  - Jaccard vs `luck-factor-0.928.ipynb`: about `0.902`
  - Jaccard vs `kaggle_submission_exp_015d_v18_artifact_submit.ipynb`: about `0.540`
- What is genuinely useful here:
  - explicit reproducibility fixes for PyTorch init / augmentation / dropout
  - a documented two-stage workflow:
    - set `ProtoSSM_PATH = None`, `ProtoSSM_JSON = None`, `ResidualSSM_PATH = None`
    - train locally first
    - then reattach the notebook and skip training at inference time
- This is conceptually aligned with our artifact split:
  - local/offline downstream training
  - thinner inference later
- What it still does poorly:
  - remains monolithic
  - still mixes training and submission logic in one notebook via `MODE`
  - still depends on manual path editing instead of a clean artifact dataset contract
  - still writes raw `.pt` checkpoints rather than a robust packaged artifact schema

### `bird26-reproduce-perch-protossm-resssm-inf-train.ipynb`

- This notebook is effectively the paired inference/reuse version of `bird26-reprod-perch-proto-residualssm-train-s7177.ipynb`.
- It directly hardcodes notebook-input paths back to the first notebook:
  - `ProtoSSM_PATH = "/kaggle/input/notebooks/.../bird26-reprod-perch-proto-residualssm-train-s7177/..."`
  - `ResidualSSM_PATH = "/kaggle/input/notebooks/.../bird26-reprod-perch-proto-residualssm-train-s7177/..."`
- So it is not an independent recipe; it is a notebook-to-notebook reuse scaffold.
- The only materially new engineering idea relative to the first `bird26` notebook is:
  - a batched `temporal_shift_tta(..., max_batch_size=512)` implementation
  - this replaces the slower one-pass-per-shift TTA loop
- That TTA batching is the main reusable idea from this notebook.
- The rest remains the same monolithic V18-family stack with `LightGBM`, `IsotonicRegression`, `mixup/cutmix`, `SWA`, `ResidualSSM`, and cached Perch arrays.
- Practical interpretation:
  - useful as corroboration that pretraining plus lighter inference is the right direction
  - but still weaker than our cleaner `exp_015d` artifactized path as an engineering scaffold
  - the only likely transplant candidate is the batched `temporal_shift_tta` idea, and only if we later need it in a controlled way

### `exp_018a_texture_specialist_oof`

- The first targeted specialist branch has now been scaffolded as:
  - `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/exp_018a_texture_specialist_oof.ipynb`
- Design choice:
  - keep the strong native `exp_011` HGNetV2 recipe
  - narrow the label space to the weakest texture-heavy taxa from `exp_017`
  - start with `Amphibia + Insecta`
- Local safe setup completed successfully without training.
- Current target-only data footprint:
  - `63` target classes
  - `1017` total rows
  - `650` isolated `train_audio` rows
  - `367` merged `soundscape_clip` rows
- Fold `0` setup snapshot:
  - train rows: `748`
  - valid rows: `269`
  - train soundscape rows: `297`
  - valid soundscape rows: `70`
- Fold `0` training result:
  - best epoch: `9`
  - best target soundscape macro AUC: `0.8056`
  - target soundscape scored classes: `27`
- A direct same-taxa comparison against stored `exp_011` fold `0` soundscape predictions gives about:
  - generic `exp_011`: `0.7934`
  - specialist `exp_018a`: `0.8056`
  - delta: `+0.0122`
- Fold `1` training result:
  - best epoch: `10`
  - best target soundscape macro AUC: `0.8254`
  - target soundscape scored classes: `25`
- A direct same-taxa comparison against stored `exp_011` fold `1` soundscape predictions gives about:
  - generic `exp_011`: `0.7720`
  - specialist `exp_018a`: `0.8254`
  - delta: `+0.0534`
- Fold `2` training result:
  - best epoch: `7`
  - best target soundscape macro AUC: `0.7914`
  - target soundscape scored classes: `18`
- A direct same-taxa comparison against stored `exp_011` fold `2` soundscape predictions gives about:
  - generic `exp_011`: `0.8293`
  - specialist `exp_018a`: `0.7914`
  - delta: `-0.0380`
- Fold `3` training result:
  - best epoch: `10`
  - best target soundscape macro AUC: `0.8004`
  - target soundscape scored classes: `33`
- A direct same-taxa comparison against stored `exp_011` fold `3` soundscape predictions gives about:
  - generic `exp_011`: `0.7548`
  - specialist `exp_018a`: `0.8004`
  - delta: `+0.0456`
- Four-fold mean comparison:
  - specialist `exp_018a`: `0.8057`
  - same-taxa generic `exp_011`: `0.7874`
  - delta: `+0.0183`
- Research value:
  - this is the first clean test of whether the native weakness highlighted by `exp_017` is concentrated enough to justify a specialist correction branch rather than another generic ensemble
  - current status: modest but real positive signal; the next ensemble should be targeted by taxon, not global, and should be evaluated as a correction layer rather than as a full replacement model

### `exp_018b_targeted_merge_benchmark`

- This benchmark uses pooled aligned OOF from:
  - generic `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_011_hgnetv2_soundscape_supervised`
  - specialist `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018a_texture_specialist_oof`
- Alignment is exact on:
  - `filename`
  - `source`
  - `clip_start_frame`
  - `clip_end_frame`
  - `primary_label`
- Pooled aligned target rows:
  - `1017`
- Pooled aligned soundscape target rows:
  - `367`
- Baselines on the target classes:
  - generic target macro AUC: `0.8486`
  - generic target soundscape macro AUC: `0.7218`
  - specialist target macro AUC: `0.8904`
  - specialist target soundscape macro AUC: `0.7450`
- Weight sweep result:
  - the best target-only blend is not full overwrite
  - best tested weight is `w_spec = 0.75`
  - best target macro AUC: `0.8913`
  - best target soundscape macro AUC: `0.7465`
- On the aligned all-class subset, replacing only the target columns with the same `0.75` specialist blend lifts the local proxy macro AUC from:
  - `0.8383` -> `0.8766`
- Interpretation:
  - the specialist branch is useful not only as a standalone target expert
  - the stronger result is a soft targeted merge, not a hard overwrite
  - this is now the strongest local justification for trying a later Kaggle overlay on top of `exp_015d`

### `exp_018c_exp015d_texture_overlay`

- This branch is the first Kaggle-facing attempt to turn the `exp_018a` specialist into a real correction layer on top of the strongest current submit path:
  - base path: `notebooks/kaggle_submission_exp_015d_v18_artifact_submit.ipynb`
  - overlay path: `notebooks/kaggle_submission_exp_018c_exp015d_texture_overlay.ipynb`
- Engineering principle:
  - keep the `exp_015d` V18 path unchanged for all non-target classes
  - run the HGNetV2 texture specialist only for `Amphibia + Insecta`
  - blend only those target columns into the final probability table
- Packaged Kaggle specialist assets now exist in:
  - `submissions/kaggle_datasets/birdclef-exp018a-texture-specialist-4fold`
- The specialist dataset contains:
  - `4` fold checkpoints
  - `63` target classes
  - `target_config.json` with the target label list
- First planned Kaggle settings are intentionally conservative:
  - `RUN_EXP018A_OVERLAY = True`
  - `EXP018A_BLEND_WEIGHT = 0.35`
  - `EXP018A_FOLD_IDS = (0, 1)`
- Rationale:
  - the local `exp_018b` optimum (`w_spec = 0.75`) was measured against pooled `exp_011`, not against the much stronger `exp_015d`
  - the first public test should therefore protect both score and runtime
  - if the overlay is positive and runtime-safe, weight can be increased later in a controlled sweep
