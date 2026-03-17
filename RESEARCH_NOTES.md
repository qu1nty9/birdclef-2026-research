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
