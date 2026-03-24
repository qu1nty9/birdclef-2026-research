# TODO Research

## Completed Setup

- [x] Analyze project documentation and current repository state
- [x] Review reference notebooks in `references/private-notebooks/`
- [x] Identify first experiment and a local validation strategy based on labeled soundscapes
- [x] Confirm that `train_soundscapes_labels.csv` contains duplicated identical rows per segment and should be deduplicated during evaluation

## Immediate Next Steps

- [x] Submit the reference blend baseline to Kaggle and record the first public LB
- [ ] Run `exp_001_soundscape_reference_blend` in a Torch-enabled environment and record local macro ROC-AUC for all strategies
- [ ] Decide whether notebook heuristics help local validation or only public LB ranking
- [x] Start `exp_002_train_audio_reproduction` and train the same architecture on isolated `train_audio`
- [x] Submit the pure `exp_002` checkpoint through a dedicated Kaggle notebook and record the public LB
- [x] Create a notebook-only reproduction of the Perch downstream stack using the cached `perch_meta` arrays
- [ ] Submit the dedicated Perch `exp_003` Kaggle notebook and record the public LB
- [ ] Test whether site/hour priors and texture smoothing transfer to our non-Perch baselines
- [x] Promote the best locally validated inference recipe into the default Kaggle submission notebook
- [x] Create `exp_004_soundscape_finetuning` starting from the `exp_002` checkpoint because the pure isolated-audio model scored only `0.647` on Kaggle
- [x] Run `exp_004_soundscape_finetuning` and record the first local soundscape macro ROC-AUC
- [ ] Improve `exp_004` validation coverage with additional folds or OOF aggregation because the first fold scored only `29` classes
- [x] Build `exp_005_native_priors_texture_postproc` on top of `exp_004`
- [x] Test whether site/hour priors and texture smoothing transfer to our non-Perch baselines
- [ ] Compare `exp_004` and `exp_005` against `exp_003_perch_downstream_reproduction` on local soundscape metrics before another Kaggle submission
- [x] Build a lightweight native submission notebook using the best `exp_005` postprocessing recipe
- [x] Submit the dedicated `exp_005` native hybrid Kaggle notebook and record the public LB
- [x] Choose stronger native soundscape training as the next main branch after the `0.737` public LB result
- [x] Create `exp_006_soundscape_finetuning_v2` as a fold-aware notebook with exported validation predictions
- [x] Run `exp_006_soundscape_finetuning_v2` on fold `0` and record best macro ROC-AUC
- [x] Run at least `2-3` folds of `exp_006_soundscape_finetuning_v2` before trusting small native deltas
- [ ] Decide whether `exp_006` is a real improvement or just a protocol upgrade after additional folds
- [x] Apply the `exp_005` priors/texture recipe on top of exported `exp_006` fold predictions and compare against raw `exp_006`
- [x] Build the next native Kaggle submission from `exp_006 + priors`
- [x] Submit the `exp_006 + priors` native notebook to Kaggle and record the public LB
- [x] Start the long-context native SED branch (`exp_008`) because `exp_007` public LB `0.758` is still too far from `0.890`
- [x] Run fold `0` of `exp_008_long_context_native_sed` and record best macro ROC-AUC
- [x] Apply the `exp_007` priors/texture layer on top of `exp_008` fold `0` exports
- [x] Build the first Kaggle submission from the long-context branch using the `exp_008b` winning recipe
- [x] Record the first Kaggle LB for the long-context branch and compare it against the current native public best `0.758`
- [x] Run at least one more `exp_008` fold before trusting the long-context branch again on Kaggle
- [x] Build a fold-safe long-context OOF postprocess comparison on top of `exp_008` folds `0-2`
- [x] Decide whether long-context is a real branch or a single-fold mirage using pooled OOF before spending more leaderboard attempts on it
- [x] After the long-context branch, start a noisy-student pseudo-label branch (`exp_009`)
- [x] Run pseudo-label generation in `exp_009` and inspect pseudo confidence / coverage before any long training run
- [x] Launch a short smoke-train for `exp_009` after pseudo-label caching succeeds
- [x] Run the first full `exp_009` fold and record best macro ROC-AUC
- [x] Apply the `exp_007` priors/texture recipe on top of exported `exp_009` fold predictions
- [x] Run at least one more `exp_009` fold before treating the branch as Kaggle-ready
- [x] Run `exp_009` fold `2` to get a three-fold view before the first raw Kaggle promotion
- [x] Prepare a Kaggle submission notebook for raw `exp_009` without the old priors layer
- [x] Run the first Kaggle submission for raw `exp_009` and record the public LB
- [x] Build a pooled OOF / calibration analysis for `exp_009` before another leaderboard attempt
- [x] Prototype the `HGNetV2-B0 + wav-cache + soundscape-clip` supervised branch from the `0.856` reference
- [ ] Run `exp_011_hgnetv2_soundscape_supervised` on fold `0` and record both overall and soundscape-only validation ROC-AUC
- [ ] Prepare a dedicated `Amphibia/Insecta` specialist branch (`exp_010`) if the generic native branch still underperforms on texture-heavy classes

## Data And Validation

- [ ] Build a stable local validation report split by animal class and by rare classes
- [ ] Quantify performance on the 28 soundscape-only classes
- [ ] Inspect whether duplicate labels in `train_soundscapes_labels.csv` are a packaging artifact or intentional redundancy
- [ ] Evaluate how much performance changes if training ignores `secondary_labels`
- [ ] Evaluate fully labeled files as a dedicated trusted adaptation subset for second-stage models

## Training Ideas

- [ ] Train a clean baseline on `train_audio` only with the same mel frontend and backbone as the reference checkpoints
- [ ] Train a soundscape-aware finetuning stage on labeled `train_soundscapes`
- [x] Prototype a long-context native SED branch (`20s` chunks -> `5s` outputs) instead of treating each row as an independent `5s` clip
- [ ] Compare BCE vs focal-style loss for long-tail classes
- [ ] Test adding `secondary_labels` as weak multi-label supervision
- [ ] Port masked BCE for primary-plus-secondary labels from the training references and compare it against the current loss
- [ ] Test class-balanced sampling for species with `<= 20` isolated recordings
- [ ] Add a background-bank mixing augmentation stage using soundscape clips or trusted background windows
- [ ] Test whether a PCEN frontend improves soundscape transfer over the current mel frontend
- [x] Implement a real noisy-student pseudo-label branch with mixup between labeled data and pseudo-labeled soundscape chunks
- [ ] Test probability power-transform denoising and confidence-weighted sampling for pseudo-labeled soundscapes
- [ ] Try a dedicated `Amphibia/Insecta` model with targeted extra Xeno-Canto species instead of mixing those labels into the generic branch
- [x] Prototype a fast supervised HGNetV2-B0 branch using `train_audio + labeled soundscape clips` as a simpler alternative to heavier pseudo-label pipelines
- [ ] Test whether explicit soundscape clip extraction by contiguous `(filename, primary_label)` segments is stronger than our current raw soundscape-window supervision
- [ ] Build a wav-cache + partial-read audio loader path so full 4-fold supervised training is cheap enough to rerun often

## Feature And Architecture Ideas

- [ ] Compare `n_mels=128` vs `n_mels=224`
- [ ] Test lower `fmin` for amphibians and insects
- [ ] Compare EfficientNet-B0 against a stronger timm backbone once the baseline is stable
- [ ] Compare EfficientNet-B0 against `hgnetv2_b0.ssld_stage2_ft_in1k`
- [ ] Evaluate whether segmentwise logits from the attention head can improve sound event localization
- [ ] Test frozen embedding stackers on top of model outputs instead of only end-to-end finetuning
- [ ] Separate texture-style classes (`Amphibia`, `Insecta`) from bird/event classes in postprocessing or auxiliary heads

## Inference And Ensemble Ideas

- [ ] Validate temporal smoothing separately from global file-max leakage
- [ ] Compare probability blending vs logit blending
- [ ] Try class-aware thresholds only for diagnostic analysis, not as the main AUC optimization target
- [ ] Build a CPU-safe submission script that mirrors the best validated inference recipe
- [ ] Test overlap-average-max-delta aggregation on framewise SED predictions instead of independent `5s` clipwise inference
- [ ] Add metadata priors based on `site`, `hour_utc`, and `site-hour` combinations
- [x] Add metadata priors based on `site`, `hour_utc`, and `site-hour` combinations
- [ ] Compare plain logits against logits plus file-context features (`prev`, `next`, `mean`, `max`) in a second-stage classifier
- [ ] Evaluate OpenVINO or ONNX export once the native ensemble is strong enough to make CPU runtime a bottleneck
- [ ] Reproduce the HGNetV2/OpenVINO CPU-safe submit path as a native inference engineering branch
- [ ] Reproduce target-domain pseudo-labeling with overlapping `5s` windows, `2.5s` hop, temporal smoothing, and classwise quantile filtering in a notebook-only experiment
