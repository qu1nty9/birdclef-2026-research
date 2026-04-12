# exp_019a_exp015d_no_rank_aware

## Goal

Create the thinnest possible public submit patch from `exp_019`:

- keep the full `exp_015d` artifactized V18 stack
- change only one postprocess choice
- disable rank-aware scaling after artifact load

Notebook:

- `notebooks/kaggle_submission_exp_019a_exp015d_no_rank_aware.ipynb`

Base notebook:

- `notebooks/kaggle_submission_exp_015d_v18_artifact_submit.ipynb`

## Why This Exists

The first `exp_019` proxy benchmark suggested:

- top row-level proxy winner: `no_file_scale`
- safer cheap candidate: `no_rank_aware`

`no_rank_aware` is the cleaner first public test because it:

- improved proxy row macro AUC
- did not worsen proxy file macro AUC
- requires only a tiny patch to the already working submit notebook

## Patch Design

This branch introduces only two explicit controls in the Kaggle hints cell:

```python
FORCE_DISABLE_RANK_AWARE = True
FORCED_RANK_AWARE_POWER = 0.0
```

Then, after artifact manifest load, it overrides:

```python
CFG["rank_aware_scale"] = False
CFG["rank_aware_power"] = 0.0
```

All other model, artifact, residual, threshold, and temperature logic remains unchanged.

## Validation Status

- notebook scaffolded from the working `exp_015d` submit path
- syntax checked with `ast.parse`
- first Kaggle submission completed
- public LB: `0.928`

## Intended Interpretation

If this branch improves `0.929`, it is strong evidence that the current V18 stack is slightly over-postprocessed and that a simpler thin submit is preferable.

If it matches or underperforms `0.929`, then rank-aware scaling should be considered part of the stable `exp_015d` recipe again.

## Result

- `exp_015d`: `0.929`
- `exp_019a`: `0.928`
- delta: `-0.001`

This means the cheap proxy signal from `exp_019` did not transfer to public LB in this case.

Practical conclusion:

- keep rank-aware scaling in the stable `exp_015d` recipe
- treat `exp_019a` as a useful thin negative control rather than a promotion path
