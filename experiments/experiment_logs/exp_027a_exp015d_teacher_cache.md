# `exp_027a_exp015d_teacher_cache`

- Status:
  - scaffolded
- Goal:
  - replay the fixed `exp_015d` V18 artifact stack on fully labeled soundscape rows and save a fold-aware teacher cache for later native distillation
- Notebook:
  - `notebooks/exp_027a_exp015d_teacher_cache.ipynb`
- Planned outputs:
  - `teacher_meta.parquet`
  - `teacher_outputs.npz`
  - `fold_summary.csv`
  - `report_snapshot.json`
- Design notes:
  - keep the strongest external path fixed instead of inventing another V18 variant
  - save both logit-like teacher scores and submit-like probabilities
  - assign local folds directly inside the teacher cache so downstream native runs can reuse the same split without rebuilding alignment logic
- Decision rule:
  - if the teacher replay looks healthy and the fold assignment is stable, promote the cache into `exp_027b`

