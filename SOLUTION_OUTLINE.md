# Solution Outline

## 1. Problem Setup

- Multi-label acoustic species recognition in Pantanal soundscapes
- 5-second prediction windows
- Macro ROC-AUC with classes without positives skipped

## 2. Data Strategy

- Isolated `train_audio` for broad species coverage
- Labeled `train_soundscapes` for local-domain validation and finetuning
- Careful handling of soundscape-only classes

## 3. Baseline Model Family

- Mel spectrogram frontend
- CNN backbone from timm
- Multi-label attention-style prediction head

## 4. Training Recipe

- Stage 1: isolated recording pretraining
- Stage 2: soundscape-aware finetuning
- Ablations on augmentations, label usage, and sampling

## 5. Inference Recipe

- 5-second chunking
- Checkpoint blending
- Optional temporal smoothing and file-level priors

## 6. Error Analysis

- Rare classes
- Soundscape-only classes
- Cross-class confusion
- Domain shift between global recordings and Pantanal soundscapes

## 7. Final Ensemble

- To be filled after validated experiments

## 8. Research Roadmap

### Phase A. Native Hybrid Consolidation

- Goal:
  - decide whether the current native branch becomes stronger when `exp_005` priors and texture-aware smoothing are applied on top of `exp_006` fold exports
- Main experiment:
  - `exp_007`: postprocess exported `exp_006` validation predictions with metadata priors and texture-aware logic
- Outcome:
  - positive
  - best pooled OOF variant: `event + texture priors + smoothing`
  - pooled OOF improvement: `0.6646 -> 0.7109`
- Success criterion:
  - stronger and more stable local soundscape metric than the older `exp_004 + exp_005` branch
- If successful:
  - build the next native Kaggle submission from `exp_006 + priors`
- If not:
  - keep `exp_006` as a protocol improvement and move to a larger modeling change

### Phase B. Long-Context Native SED

- Goal:
  - move beyond independent `5s` clip classification and model broader soundscape context
- Main experiment:
  - `exp_008`: native SED branch on `20s` chunks with `5s` outputs
- Core ideas:
  - long-context mel frontend
  - SED head with framewise predictions
  - overlap-aware aggregation for `5s` rows
- Success criterion:
  - better soundscape-aware validation than the short-context native branch

### Phase C. Noisy-Student Pseudo-Label Branch

- Goal:
  - use unlabeled soundscapes as a real training asset rather than only at inference time
- Main experiment:
  - `exp_009`: pseudo-label training with noisy-student style mixup
- Core ideas:
  - pseudo-labeled soundscape chunks
  - mixup with labeled data
  - stochastic depth
  - confidence-weighted pseudo sampling
  - probability power-transform denoising
- Success criterion:
  - reproducible gain over the best purely supervised native branch

### Phase D. Texture Specialist Branch

- Goal:
  - improve texture-heavy taxa that repeatedly behave differently from bird/event classes
- Main experiment:
  - `exp_010`: dedicated `Amphibia/Insecta` model, optionally with targeted extra Xeno-Canto species
- Success criterion:
  - measurable uplift on texture-heavy validation classes and positive ensemble contribution

### Phase E. Native Stacker And Ensemble

- Goal:
  - combine the strongest native branches into a solution that can realistically challenge the current reference baseline
- Main experiments:
  - file-context stacker over logits or embeddings
  - ensemble of:
    - best native supervised branch
    - best pseudo-label branch
    - optional texture specialist branch
- Success criterion:
  - public LB materially above the current native hybrid `0.737`
  - ideally closes a meaningful part of the gap to the `0.890` reference blend

### Execution Order

1. Native Kaggle submission from `exp_006 + priors`
2. `exp_008`: long-context native SED
3. `exp_009`: noisy-student pseudo-label branch
4. `exp_010`: texture specialist branch
5. native stacker and final ensemble
