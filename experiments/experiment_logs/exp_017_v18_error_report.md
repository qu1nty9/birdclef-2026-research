# exp_017 — v18_error_report

## Notebook

- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/exp_017_v18_error_report.ipynb`

## Status

- scaffolded
- AST-validated
- executed locally end-to-end on pooled `exp_011` validation outputs

## Local Outputs

- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/report_snapshot.json`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_classwise_auc.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_weak_soundscape_classes.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_taxon_summary.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_soundscape_site_macro_auc.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_soundscape_site_confidence.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_soundscape_hour_macro_auc.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_017_v18_error_report/exp011_soundscape_hour_confidence.csv`

## Main Findings (exp_011 pooled validation)

- pooled rows: `36078`
- soundscape rows: `529`
- pooled macro AUC over all rows: `0.9468`
- pooled macro AUC over soundscape subset: `0.7612`
- scored soundscape classes: `75`

### Weakest Soundscape Taxa

- `Amphibia` had the lowest mean soundscape AUC: about `0.672`
- `Insecta` was next: about `0.755`
- `Aves` remained strongest on average: about `0.812`

### Hardest Sites

- worst soundscape sites by classwise macro AUC:
  - `S19`
  - `S08`
  - `S03`
  - `S13`
  - `S15`

### Hardest Hours

- worst soundscape hours by classwise macro AUC:
  - `07`
  - `19`
  - `21`
  - `06`
  - `02`

## Interpretation

- `exp_011` is strong overall but still has a large drop from mixed validation to soundscape-only evaluation.
- The largest native weakness is concentrated in texture-heavy taxa, especially `Amphibia` and parts of `Insecta`.
- This supports a targeted next step rather than a generic new blend:
  - texture specialist branch
  - or V18 / external-path analysis focused on those difficult taxa and site/hour regimes

## V18 Crosswalk

- The notebook supports optional crosswalk against attached V18 artifact datasets.
- In the local run, no V18 artifact dataset was found, so that section was skipped safely.
