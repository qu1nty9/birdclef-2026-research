# Master Experiment Table

This table is the internal registry for all experiments in the BirdCLEF+ 2026 project.

| ID | Name | Status | Baseline | Model | Key Change | Local CV | Kaggle LB | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_001 | soundscape_reference_blend | in_progress | none | EfficientNet-B0 + GeM + attention SED head | Rebuild local validation around labeled soundscapes and compare `LB862`, `LB872`, `0.8 * LB872 + 0.2 * LB862`, and the same blend with notebook heuristics | pending | pending | Reference checkpoints from `data/BirdCLEF-2026-model/`; local labels deduplicated by `(filename, start, end)` before evaluation |
