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
- Current status:
  - notebook created: `notebooks/exp_009_noisy_student_pseudolabel.ipynb`
  - safe setup validated on fold `0`
  - pseudo labels generated on fold `0`
  - first full run completed on fold `0` with best macro ROC-AUC `0.8495`
  - postprocess check completed through `exp_009b`, and raw remains best
  - fold `1` completed with best macro ROC-AUC `0.8769`
  - fold `2` completed with best macro ROC-AUC `0.8849`
  - folds `0-2` now average `0.8704`
  - first raw Kaggle submission scored `0.735`
  - that is below the current native public best `0.758`
  - pooled OOF follow-up completed through `exp_009c`
  - raw pooled OOF is only `0.7934`, with an optimism gap of `0.0770` vs fold mean
  - raw remains the best fixed `exp_009` OOF variant
  - no lightweight calibration rescue justified another leaderboard attempt
  - raw submission notebook prepared: `notebooks/kaggle_submission_exp_009_raw_3fold.ipynb`
  - raw model dataset prepared: `submissions/kaggle_datasets/birdclef-exp009-raw-3fold`
- Core ideas:
  - pseudo-labeled soundscape chunks
  - mixup with labeled data
  - stochastic depth
  - confidence-weighted pseudo sampling
  - probability power-transform denoising
- Success criterion:
  - reproducible gain over the best purely supervised native branch

### Phase D. HGNetV2 Supervised Branch

- Goal:
  - test whether a simpler but stronger supervised branch can outperform the older native path without Perch or noisy-student machinery
- Main experiment:
  - `exp_011`: `HGNetV2-B0 + train_audio + labeled soundscape clips + grouped multi-label CV`
- Current status:
  - notebook created: `notebooks/exp_011_hgnetv2_soundscape_supervised.ipynb`
  - setup validated end-to-end in the local `.venv`
  - local soundscape labels were correctly reinterpreted as multi-label segment strings
  - those expand to `3122` per-label rows and `529` merged soundscape clips across `75` target classes
  - folds `0-2` completed with:
    - fold `0` soundscape-only macro ROC-AUC `0.8509`
    - fold `1` soundscape-only macro ROC-AUC `0.8042`
    - fold `2` soundscape-only macro ROC-AUC `0.8544`
    - soundscape-only mean across folds `0-2`: `0.8365`
    - overall macro ROC-AUC mean across folds `0-2`: `0.9528`
  - later epochs often improved mixed-fold AUC while degrading the soundscape-only objective, so soundscape-aware checkpointing is clearly necessary
- Core ideas:
  - `hgnetv2_b0.ssld_stage2_ft_in1k`
  - optional wav-cache for `train_audio`
  - soundscape-clip supervision through merged contiguous intervals
  - grouped multi-label validation by `audio_id`
  - export of validation metadata and predictions for later OOF analysis
- Success criterion:
  - a stronger supervised local branch than the current EfficientNet-based native path
  - especially on soundscape-only validation readouts

#### Follow-Up: Exp_014 HGNetV2 Pseudo-Label Continuation

- Goal:
  - test whether the stabilized `exp_011` branch can move beyond pure supervised training by adding target-domain pseudo labels from unlabeled `train_soundscapes`
- Main experiment:
  - `exp_014`: `HGNetV2-B0` student trained on:
    - labeled `train_audio`
    - labeled soundscape clips
    - soft pseudo-labeled soundscape windows
- Current status:
  - notebook created: `notebooks/exp_014_hgnetv2_pseudolabel.ipynb`
  - safe setup validated on fold `0`
  - fold `0` teacher folds resolved to `[1, 2, 3]`
  - fold `0` pseudo manifest already built with `127104` windows across `10592` files
  - fold `0` pseudo generation retained `20688` windows across `5471` files
  - fold `0` best soundscape-only macro ROC-AUC reached `0.8684`
  - this beats `exp_011` fold `0` (`0.8509`) by `+0.0176`
  - fold `1` pseudo generation retained `22188` windows across `5646` files
  - fold `1` best soundscape-only macro ROC-AUC reached `0.8154`
  - this beats `exp_011` fold `1` (`0.8042`) by `+0.0112`
  - fold `2` pseudo generation retained `20211` windows across `5333` files
  - fold `2` best soundscape-only macro ROC-AUC reached `0.8301`
  - this loses to `exp_011` fold `2` (`0.8544`) by `-0.0243`
  - fold `3` pseudo generation retained `20595` windows across `5293` files
  - fold `3` best soundscape-only macro ROC-AUC reached `0.7702`
  - this loses to `exp_011` fold `3` (`0.7992`) by `-0.0290`
  - four-fold mean is now `0.8210`, below `exp_011` at `0.8272`
- Follow-up rescue branch:
  - `exp_014b`: stricter pseudo-label continuation with:
    - higher confidence threshold
    - smaller pseudo cache
    - lower pseudo loss weight
    - delayed pseudo start
  - first fold result:
    - soundscape-only macro ROC-AUC `0.8683`
    - almost identical to `exp_014` fold `0`
    - but with a much smaller pseudo cache (`8724` vs `20688`)
- Success criterion:
  - beat the local soundscape-aware validation picture of `exp_011`
  - and justify a new Kaggle submission path beyond the current `0.850` native baseline

### Phase E. Texture Specialist Branch

- Goal:
  - improve texture-heavy taxa that repeatedly behave differently from bird/event classes
- Main experiment:
  - `exp_010`: dedicated `Amphibia/Insecta` model, optionally with targeted extra Xeno-Canto species
- Success criterion:
  - measurable uplift on texture-heavy validation classes and positive ensemble contribution

### Phase F. Native Stacker And Ensemble

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

### Phase G. Faithful External ProtoSSM Path

- Goal:
  - operationalize the strongest external `Perch + ProtoSSM + probe + residual SSM` notebook without prematurely simplifying away the components that make it competitive
- Main experiment:
  - `exp_015`: faithful Kaggle submit-path port of the Pantanal Distill / ProtoSSM `0.924` notebook
- Current status:
  - notebook created and engineering fixes applied
  - first real Kaggle submission completed
  - artifactized V18 continuation (`exp_015d`) now completed
  - public LB improved to `0.929`
  - this is now the best overall public result in the repository
- Core ideas:
  - frozen Perch foundation model
  - file-level ProtoSSM temporal modeling
  - in-model metadata handling
  - MLP probe branch
  - residual SSM correction
  - strong leaderboard-oriented postprocess chain
- Success criterion:
  - confirm that the strongest external ceiling remains strong when ported faithfully into our own Kaggle environment
- Current interpretation:
  - achieved
  - this branch is now the main Kaggle-facing path
  - the next question is no longer whether the external stack works, but whether it complements `exp_011` well enough to justify a blend or ensemble

### Phase H. Submission-Level Blend Check

- Goal:
  - test complementarity between the strongest overall path (`exp_015d = 0.929`) and the strongest repository-native path (`exp_011 = 0.850`) with minimal extra engineering risk
- Main experiment:
  - `exp_016`: historical CSV-blend scaffold for `exp_015 + exp_011`
  - `exp_016b`: runtime blend on top of the artifactized V18 path plus `exp_011` 4-fold HGNet inference
- Current status:
  - historical notebook kept: `notebooks/kaggle_submission_exp_016_blend_exp015_exp011.ipynb`
  - active notebook created: `notebooks/kaggle_submission_exp_016b_runtime_blend_exp015d_exp011.ipynb`
  - default runtime blend: `0.95 * exp_015d + 0.05 * exp_011`
  - first public run: `0.929`, exactly equal to `exp_015d`
- Why this form:
  - it tests complementarity directly against the current best public path
  - it keeps the timeout-safe artifactized V18 route intact
  - and it lets us probe whether the native branch still adds useful diversity despite the large raw score gap
- Current interpretation:
  - the first simple blend did not improve leaderboard score
  - therefore plain `exp_015d + exp_011` ensembling is not currently a top-priority optimization path
  - ideally closes a meaningful part of the gap to the `0.890` reference blend

### Execution Order

1. Treat `exp_011 = 0.850` as the default native public baseline
2. Use the modest `4-fold` gain (`0.844 -> 0.850`) as evidence that `exp_011` is now stabilized rather than massively under-ensembled
3. Treat the current local Perch line as unresolved after `exp_012` and `exp_012b` both failed to beat raw Perch on pooled OOF
4. Compare any future repaired Perch line against `exp_011` only after it first beats raw Perch locally
5. Run `exp_014` as the next native modeling test on top of the stabilized `exp_011` branch
6. In parallel, operationalize one faithful `0.924` external notebook as a separate high-ceiling submit path
   - now prepared as `exp_015`
   - Kaggle score
   - research value / explanatory power
   - ensemble complementarity
7. `exp_010`: texture specialist branch
8. native stacker and final ensemble
