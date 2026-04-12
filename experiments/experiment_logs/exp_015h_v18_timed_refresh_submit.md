# exp_015h_v18_timed_refresh_submit

## Goal

Create a timer-aware variant of the abandoned `exp_015f` calibration-refresh submit path without mutating the original notebook.

Notebook:

- `notebooks/kaggle_submission_exp_015h_v18_timed_refresh_submit.ipynb`

Baseline source:

- `notebooks/kaggle_submission_exp_015f_v18_calibration_refresh_submit.ipynb`

## Why This Exists

The Kaggle discussion script found by the user is not a real in-notebook timeout fix.

What that script actually does:

- repeatedly calls `kaggle competitions submissions -v ...`
- tracks pending / completed submissions outside the competition notebook
- writes local `submissions_pending.csv` and `submissions_finished.csv`

So it is useful as an external monitor, but it cannot help a code-competition notebook survive a wall-time limit because:

- notebook internet is disabled
- Kaggle CLI polling is external to the scoring runtime
- it does not change the internal execution path of the model notebook

## Actual Integration Strategy

Instead of copying the external polling loop literally, `exp_015h` ports the useful idea into the notebook itself:

- explicit wall-time accounting
- stage-by-stage logging
- per-stage timer guards
- optional early-safe snapshot writes for debugging:
  - `submission_after_temperature.csv`
  - `submission_after_calibration.csv`
  - `submission_after_file_level_scaling.csv`
  - `submission_after_rank_scaling.csv`
  - `submission_after_delta_smooth.csv`
  - `submission_after_thresholds.csv`
- `submission.csv` is refreshed after each safe stage

## Timer Defaults

```python
SUBMIT_TIMER_ENABLED = True
SUBMIT_MAX_WALL_SECONDS = 4800
SUBMIT_RESERVE_SECONDS = 180
SUBMIT_SAVE_EARLY_SNAPSHOTS = False
```

Stage requirements:

```python
SUBMIT_STAGE_REQUIREMENTS = {
    "isotonic_calibration": 300,
    "file_level_scaling": 120,
    "rank_aware_scaling": 90,
    "adaptive_delta_smooth": 90,
    "threshold_sharpening": 60,
}
```

## Intended Diagnostic Value

This branch answers a more useful question than plain timeout / no-timeout:

- does `exp_015f` fail because one late postprocess stage is too expensive?
- or does it fail earlier, before even a valid calibrated fallback can be materialized?

Expected runtime artifacts:

- `/kaggle/working/v18_timed_refresh_submit_logs.json`
- staged submission snapshots under `/kaggle/working/`

## Next Step

First Kaggle run result:

- `Notebook Timeout`

Interpretation after the first run:

- the watchdog logic was not enough to rescue the branch
- the remaining bottleneck is likely earlier than the guarded late postprocess stages
- this matches the structural diff versus the working `exp_015d` path:
  - `exp_015h` forces CPU
  - `exp_015d` can use faster device selection when available
  - `exp_015h` only starts skipping work after the full hidden-test replay has already happened

Decision:

- pause `exp_015h` as a negative engineering branch
- keep `exp_015d = 0.929` as the main production submit path

Later Kaggle rerun result:

- Public LB: `0.920`

Final interpretation:

- the runtime tweaks were enough to make the notebook finish
- but the branch still underperformed the stable `exp_015d = 0.929` baseline by `-0.009`
- so `exp_015h` is now a completed negative branch in both senses:
  - originally fragile on runtime
  - and ultimately not good enough on score
