# `exp_027b_hgnetv2_soundscape_distill_from_exp015d`

- Status:
  - scaffolded
- Goal:
  - train a soundscape-only HGNetV2 student initialized from `exp_011` and distilled from the fixed `exp_015d` teacher cache
- Notebook:
  - `notebooks/exp_027b_hgnetv2_soundscape_distill_from_exp015d.ipynb`
- Planned outputs:
  - `history.csv`
  - `best_model.pt`
  - `last_model.pt`
  - `best_valid_meta.csv`
  - `best_valid_outputs.npz`
  - `result_snapshot.json`
- Design notes:
  - no unlabeled pseudo cache
  - no runtime overlay tricks
  - the experiment is intentionally narrow: target-domain soundscape supervision plus teacher-target distillation only
- Hardening note:
  - the notebook now auto-resolves `exp_027a` outputs from both `experiments/outputs` and `~/Downloads`
  - if the cache still cannot be found, set `TEACHER_CACHE_DIR` explicitly to the completed `exp_027a` output folder
  - the notebook now also remaps Kaggle-style soundscape paths stored inside `teacher_meta.parquet` back onto the local `data/birdclef-2026/train_soundscapes` tree
- Decision rule:
  - if fold `0` cannot beat or at least closely match the corresponding supervised `exp_011` checkpoint, close the branch early

## Fold 0 Result

- best epoch:
  - `5`
- best selection metric:
  - `0.961055`
- scored classes:
  - `43`
- train / valid rows:
  - `540 / 168`
- device:
  - `mps`
- teacher cache source:
  - `/Users/yaroslav/Downloads/exp_027a_exp015d_teacher_cache`

Important interpretation:

- this validation is on the small fully labeled soundscape subset from `exp_027a`, so the absolute AUC is not directly comparable to the old broad `exp_011` fold metrics
- on the *same* `168` validation rows, the fixed `exp_015d` teacher scores about `0.996472`
- the student scores only `0.961055`
- simple blends between teacher and student also regress immediately, with the best local weight staying at pure teacher (`w_student = 0.00`)

Current conclusion:

- the first `exp_027b` fold is a useful negative result
- the student does learn a strong soundscape classifier in absolute terms
- but it is not yet a complementary or competitive approximation of the `exp_015d` teacher on this trusted subset
- unless a later redesign changes that picture materially, the current direct teacher-student line should be treated as weak rather than submit-ready
