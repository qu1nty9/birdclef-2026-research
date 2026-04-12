# exp_020b_texture_soundscape_file_correction_oof

## Goal

Redesign the failed `exp_020a` texture-correction proxy around grouped soundscape file targets instead of trusted full-row macro AUC.

Notebook:

- `notebooks/exp_020b_texture_soundscape_file_correction_oof.ipynb`

## Why This Exists

`exp_020a` was a useful negative control:

- the target-only correction idea itself was not disproven
- but the chosen proxy was too easy and too close to ceiling
- baseline target AUC on trusted full rows was already near-saturated

So the next correct test is:

- keep the fixed `exp_015d` V18 stack
- keep the correction branch lightweight and artifact-oriented
- but train and select it on grouped soundscape file targets, which are much closer to the actual deployment bottleneck

## Design

`exp_020b` still reuses the fixed `exp_015d` artifact stack and cached full-file Perch outputs.

It:

1. replays the full V18 stack on trusted labeled soundscape windows
2. aggregates each file into target-aware statistics
3. trains lightweight per-class `LogisticRegression` correction models only for `Amphibia + Insecta`
4. selects the blend weight by grouped file-level macro AUC
5. exports reusable artifacts for a later thin file-level correction overlay if the proxy is positive

## Feature Strategy

Each target class now sees file-level features built from:

- mean / max / top-k stats of:
  - raw Perch score
  - prior score
  - base fused score
  - ProtoSSM score
  - MLP score
  - final V18 score
  - baseline postprocessed probability
- pooled embedding summaries:
  - file-level mean of `Z_FULL`
  - file-level max of `Z_FULL`
- file metadata:
  - site id
  - hour sin/cos

The overlay form is file-level rescaling of target columns, not a second heavy model family.

## Planned Outputs

- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/report_snapshot.json`
- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_weight_sweep.csv`
- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_classwise_auc.csv`
- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_oof_outputs.npz`

Artifact bundle:

- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_artifacts/texture_models.pkl`
- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_artifacts/texture_target_labels.json`
- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_artifacts/texture_blend_config.json`
- `experiments/outputs/exp_020b_texture_soundscape_file_correction_oof/texture_artifacts/texture_manifest.json`

## Validation Status

- notebook scaffolded from the fixed `exp_020a` replay path
- proxy redesigned around grouped file-level targets
- syntax checked with `ast.parse`
- Kaggle/local artifact-backed run has now completed on attached `exp_015d` artifacts and full Perch cache

## Result Snapshot

- baseline overall row macro AUC: `0.993120`
- baseline overall file macro AUC: `0.992126`
- baseline target row macro AUC: `0.997988`
- baseline target file macro AUC: `0.998281`
- correction-only target file macro AUC: `0.991418`
- best correction blend weight: `0.00`
- trained target models: `10`
- skipped low-support labels: `32`

Weight sweep summary:

- every positive correction weight was worse than the baseline
- best blended result stayed exactly at the pure baseline (`w_corr = 0.00`)
- both overall and target file-level AUC degraded monotonically as correction weight increased

Classwise pattern:

- only `10` target classes had enough support to train file-level models
- all `10` trained classes got worse
- `0` improved
- most `Insecta` labels were skipped entirely because file-level support was too low

Main interpretation:

- this still does **not** validate the artifactized texture-correction idea in deployable form
- compared with `exp_020a`, the proxy is better aligned, but the signal is still negative
- the current target domain remains too small / too clean at file level to justify a correction layer over the already very strong `exp_015d` baseline
- practical consequence:
  - do not promote `exp_020b` into a thin submit overlay
  - the current `exp_020` line should be treated as closed in deployment form
  - if the specialist idea is revisited later, it likely needs a different supervision source or broader target design rather than another thin correction on this proxy

## Intended Interpretation

If this branch improves grouped file-level AUC while preserving or slightly improving overall row/file metrics, it becomes the best thin-correction candidate after `exp_015d`.

If it is still neutral or negative, then the specialist idea should stay in the research bucket and not be promoted into the deployment path.
