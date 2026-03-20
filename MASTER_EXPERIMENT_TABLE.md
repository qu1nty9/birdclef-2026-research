# Master Experiment Table

This table is the internal registry for all experiments in the BirdCLEF+ 2026 project.

| ID | Name | Status | Baseline | Model | Key Change | Local CV | Kaggle LB | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_001 | soundscape_reference_blend | completed | none | EfficientNet-B0 + GeM + attention SED head | Rebuild local validation around labeled soundscapes and compare `LB862`, `LB872`, `0.8 * LB872 + 0.2 * LB862`, and the same blend with notebook heuristics | pending | 0.890 | First Kaggle submission baseline validated on 2026-03-18 through `notebooks/kaggle_submission_reference_blend.ipynb`; local labels are still deduplicated by `(filename, start, end)` for future CV |
| exp_002 | train_audio_reproduction | completed | exp_001 | EfficientNet-B0 + GeM + attention SED head | Train the same architecture from scratch on isolated `train_audio` to replace borrowed checkpoints with a repository-native baseline | 0.9135 hold-out macro ROC-AUC at epoch 8 / 8 | pending | Resume-from-checkpoint completed successfully on 2026-03-20; final artifacts include `history.csv`, `best_model.pt`, `last_model.pt`, and `result_snapshot.json` |
