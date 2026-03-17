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
