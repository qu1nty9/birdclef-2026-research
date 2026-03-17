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
- [ ] Start `exp_002_train_audio_reproduction` and train the same architecture on isolated `train_audio`
- [ ] Promote the best locally validated inference recipe into the default Kaggle submission notebook

## Data And Validation

- [ ] Build a stable local validation report split by animal class and by rare classes
- [ ] Quantify performance on the 28 soundscape-only classes
- [ ] Inspect whether duplicate labels in `train_soundscapes_labels.csv` are a packaging artifact or intentional redundancy
- [ ] Evaluate how much performance changes if training ignores `secondary_labels`

## Training Ideas

- [ ] Train a clean baseline on `train_audio` only with the same mel frontend and backbone as the reference checkpoints
- [ ] Train a soundscape-aware finetuning stage on labeled `train_soundscapes`
- [ ] Compare BCE vs focal-style loss for long-tail classes
- [ ] Test adding `secondary_labels` as weak multi-label supervision
- [ ] Test class-balanced sampling for species with `<= 20` isolated recordings

## Feature And Architecture Ideas

- [ ] Compare `n_mels=128` vs `n_mels=224`
- [ ] Test lower `fmin` for amphibians and insects
- [ ] Compare EfficientNet-B0 against a stronger timm backbone once the baseline is stable
- [ ] Evaluate whether segmentwise logits from the attention head can improve sound event localization

## Inference And Ensemble Ideas

- [ ] Validate temporal smoothing separately from global file-max leakage
- [ ] Compare probability blending vs logit blending
- [ ] Try class-aware thresholds only for diagnostic analysis, not as the main AUC optimization target
- [ ] Build a CPU-safe submission script that mirrors the best validated inference recipe
