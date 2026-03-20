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
- [ ] Promote the best locally validated inference recipe into the default Kaggle submission notebook
- [x] Create `exp_004_soundscape_finetuning` starting from the `exp_002` checkpoint because the pure isolated-audio model scored only `0.647` on Kaggle
- [ ] Run `exp_004_soundscape_finetuning` and record the first local soundscape macro ROC-AUC
- [ ] Compare `exp_004` against `exp_003_perch_downstream_reproduction` on local soundscape metrics before another Kaggle submission

## Data And Validation

- [ ] Build a stable local validation report split by animal class and by rare classes
- [ ] Quantify performance on the 28 soundscape-only classes
- [ ] Inspect whether duplicate labels in `train_soundscapes_labels.csv` are a packaging artifact or intentional redundancy
- [ ] Evaluate how much performance changes if training ignores `secondary_labels`
- [ ] Evaluate fully labeled files as a dedicated trusted adaptation subset for second-stage models

## Training Ideas

- [ ] Train a clean baseline on `train_audio` only with the same mel frontend and backbone as the reference checkpoints
- [ ] Train a soundscape-aware finetuning stage on labeled `train_soundscapes`
- [ ] Compare BCE vs focal-style loss for long-tail classes
- [ ] Test adding `secondary_labels` as weak multi-label supervision
- [ ] Port masked BCE for primary-plus-secondary labels from the training references and compare it against the current loss
- [ ] Test class-balanced sampling for species with `<= 20` isolated recordings
- [ ] Add a background-bank mixing augmentation stage using soundscape clips or trusted background windows
- [ ] Test whether a PCEN frontend improves soundscape transfer over the current mel frontend

## Feature And Architecture Ideas

- [ ] Compare `n_mels=128` vs `n_mels=224`
- [ ] Test lower `fmin` for amphibians and insects
- [ ] Compare EfficientNet-B0 against a stronger timm backbone once the baseline is stable
- [ ] Evaluate whether segmentwise logits from the attention head can improve sound event localization
- [ ] Test frozen embedding stackers on top of model outputs instead of only end-to-end finetuning
- [ ] Separate texture-style classes (`Amphibia`, `Insecta`) from bird/event classes in postprocessing or auxiliary heads

## Inference And Ensemble Ideas

- [ ] Validate temporal smoothing separately from global file-max leakage
- [ ] Compare probability blending vs logit blending
- [ ] Try class-aware thresholds only for diagnostic analysis, not as the main AUC optimization target
- [ ] Build a CPU-safe submission script that mirrors the best validated inference recipe
- [ ] Add metadata priors based on `site`, `hour_utc`, and `site-hour` combinations
- [ ] Compare plain logits against logits plus file-context features (`prev`, `next`, `mean`, `max`) in a second-stage classifier
- [ ] Reproduce target-domain pseudo-labeling with overlapping `5s` windows, `2.5s` hop, temporal smoothing, and classwise quantile filtering in a notebook-only experiment
