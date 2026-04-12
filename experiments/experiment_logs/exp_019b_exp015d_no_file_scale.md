# exp_019b_exp015d_no_file_scale

## Goal

Create the thinnest possible public submit patch from `exp_019`:

- keep the full `exp_015d` artifactized V18 stack
- change only one postprocess choice
- disable file-level confidence scaling after artifact load

Notebook:

- `notebooks/kaggle_submission_exp_019b_exp015d_no_file_scale.ipynb`

Base notebook:

- `notebooks/kaggle_submission_exp_015d_v18_artifact_submit.ipynb`

## Why This Exists

The first `exp_019` proxy benchmark suggested:

- top row-level proxy winner: `no_file_scale`
- safer cheap candidate: `no_rank_aware`

`no_file_scale` is the higher-upside but higher-risk second public test because it:

- delivered the strongest proxy row macro AUC gain
- is still only a tiny patch to the working submit notebook
- requires only a tiny patch to the already working submit notebook

The risk is that the same proxy run also reduced file-level macro AUC, so this branch is explicitly more aggressive than `exp_019a`.

## Patch Design

This branch introduces only two explicit controls in the Kaggle hints cell:

```python
FORCE_DISABLE_FILE_SCALE = True
FORCED_FILE_LEVEL_TOP_K = 0
```

Then, after artifact manifest load, it overrides:

```python
CFG["file_level_top_k"] = 0
```

All other model, artifact, rank-aware, residual, threshold, and temperature logic remains unchanged.

## Validation Status

- notebook scaffolded from the working `exp_015d` submit path
- syntax checked with `ast.parse`
- first Kaggle submission attempt timed out
- second Kaggle submission attempt timed out

## Runtime Notes

The first public attempt timed out, and a second clean retry timed out again.

Why:

- relative to `exp_015d`, this notebook changes only one late postprocess control:
  - `CFG["file_level_top_k"] = 0`
- that change happens after hidden-test Perch inference and after the downstream V18 stack replay
- so it should not materially increase runtime by itself

Practical interpretation after two retries:

- this still looks more like a deployment-budget problem than a direct modeling failure
- but two consecutive timeouts are enough to make the branch unattractive as a practical public test
- `exp_019b` should therefore be treated as operationally closed unless we later revisit it by editing the stable `exp_015d` notebook in place rather than as a separate branch

## Intended Interpretation

If this branch improves `0.929`, it is strong evidence that the current V18 stack is slightly over-postprocessed specifically at the file-level scaling stage.

If it matches or underperforms `0.929`, then file-level confidence scaling should remain part of the stable `exp_015d` recipe and the `exp_019` proxy row winner should be treated as a misleading public direction.

## Final Current Verdict

- no public score obtained
- two consecutive Kaggle timeouts
- branch is practically closed for now
