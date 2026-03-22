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
  - public LB after the dedicated 3-fold submission: `0.758`
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
- Follow-up experiment:
  - `exp_008b`: apply metadata priors and texture-aware smoothing on top of exported `exp_008` predictions
- Status:
  - first fold completed with a promising raw gain
  - postprocessing follow-up also completed and improved the fold again
  - first Kaggle submission was negative and scored below the current native public best
- Core ideas:
  - long-context mel frontend
  - SED head with framewise predictions
  - overlap-aware aggregation for `5s` rows
- Success criterion:
  - better soundscape-aware validation than the short-context native branch
- Current readout:
  - raw `exp_008` fold `0`: `0.8377`
  - best `exp_008b` variant: `0.8435`
  - best variant remains `event + texture priors + smoothing`
  - first long-context public LB: `0.707`
- Updated local readout after more folds:
  - `exp_008` fold `1`: `0.8223`
  - `exp_008` fold `2`: `0.8095`
  - `exp_008` mean across folds `0-2`: `0.8232`
- Honest pooled OOF check:
  - `exp_008c` raw pooled OOF: `0.6682`
  - `exp_008c` best pooled OOF: `0.7005`
  - this is below `exp_007` best pooled OOF: `0.7109`
- Updated operational step:
  - do not promote long-context to the default native submit path yet
  - the fold-safe OOF check is now complete and still not strong enough

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

1. Keep `exp_007` as the default native public baseline
2. `exp_009`: noisy-student pseudo-label branch
3. `exp_010`: texture specialist branch
4. native stacker and final ensemble
