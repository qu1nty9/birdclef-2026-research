# exp_018b — targeted_merge_benchmark

## Notebook

- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/notebooks/exp_018b_targeted_merge_benchmark.ipynb`

## Status

- scaffolded
- AST-validated
- executed locally end-to-end

## Purpose

- Benchmark a taxon-targeted merge where the `exp_018a` specialist only modifies `Amphibia + Insecta` columns.
- Use pooled aligned local OOF as a proxy benchmark before spending a Kaggle attempt on an `exp_015d` overlay.

## Inputs

- generic baseline:
  - `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_011_hgnetv2_soundscape_supervised`
- specialist branch:
  - `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018a_texture_specialist_oof`

## Alignment

- pooled aligned rows: `1017`
- pooled aligned soundscape rows: `367`
- alignment key:
  - `filename`
  - `source`
  - `clip_start_frame`
  - `clip_end_frame`
  - `primary_label`

## Main Results

- generic target-only baseline:
  - target macro AUC: `0.8486`
  - target soundscape macro AUC: `0.7218`
- specialist target-only baseline:
  - target macro AUC: `0.8904`
  - target soundscape macro AUC: `0.7450`
- best target-only blend:
  - `w_spec = 0.75`
  - target macro AUC: `0.8913`
  - target soundscape macro AUC: `0.7465`

## Overall Aligned-Subset Effect

- generic baseline overall macro AUC on the aligned subset: `0.8383`
- best targeted overwrite/blend in the tested grid:
  - `w_spec = 0.75`
  - overall macro AUC on aligned subset: `0.8766`

## Interpretation

- The specialist is not just better on its own target space.
- A soft target-only merge beats both:
  - the generic baseline
  - and the pure specialist overwrite
- The best merge is not `1.0` overwrite.
- The best local proxy is a conservative targeted blend around:
  - `75% specialist`
  - `25% generic`

## Outputs

- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018b_targeted_merge_benchmark/baseline_summary.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018b_targeted_merge_benchmark/weight_sweep.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018b_targeted_merge_benchmark/overall_merge_sweep.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018b_targeted_merge_benchmark/taxon_summary.csv`
- `/Users/yaroslav/Documents/Kaggle/BirdCLEF_2026/experiments/outputs/exp_018b_targeted_merge_benchmark/report_snapshot.json`

## Caveat

- This benchmark uses aligned pooled `exp_011` OOF as the generic proxy.
- It is a strong directional signal for a later `exp_015d` overlay, but not a direct guarantee of Kaggle gain.
