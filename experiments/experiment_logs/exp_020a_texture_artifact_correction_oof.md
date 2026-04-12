# exp_020a_texture_artifact_correction_oof

## Goal

Build a thin artifact-oriented correction benchmark for `Amphibia + Insecta` on top of the fixed `exp_015d` V18 stack.

Notebook:

- `notebooks/exp_020a_texture_artifact_correction_oof.ipynb`

## Why This Exists

The recent sequence of results points to the same conclusion:

- `exp_017` showed that the strongest native weakness is concentrated in texture-heavy taxa
- `exp_018a` gave a positive 4-fold specialist signal
- `exp_018b` showed that a soft target-only merge can beat both the generic branch and pure overwrite locally
- `exp_018d` then showed that a runtime HGNetV2 overlay is not a practical promotion path on Kaggle

So the next useful test is no longer another runtime specialist notebook.

Instead, this branch asks:

- can we keep `exp_015d` fixed
- and add only a cheap target-only correction layer as reusable artifacts

## Design

`exp_020a` reuses the fixed `exp_015d` artifact stack and cached full-file Perch outputs.

It:

1. replays the full V18 stack on trusted labeled rows
2. builds target-only features for `Amphibia + Insecta`
3. trains lightweight per-class `LogisticRegression` correction models with grouped OOF
4. chooses a target-only correction blend weight
5. exports reusable correction artifacts for a later thin submit overlay

## Feature Strategy

Each target class sees features built from:

- projected embedding features `Z_FULL`
- raw Perch class score
- prior score
- base fused score
- ProtoSSM score
- MLP score
- final V18 logit-like score
- baseline postprocessed probability
- simple site / hour context features

The design intentionally avoids a second heavy model family inside submit mode.

## Planned Outputs

- `experiments/outputs/exp_020a_texture_artifact_correction_oof/report_snapshot.json`
- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_weight_sweep.csv`
- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_classwise_auc.csv`
- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_oof_outputs.npz`

Artifact bundle:

- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_artifacts/texture_models.pkl`
- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_artifacts/texture_target_labels.json`
- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_artifacts/texture_blend_config.json`
- `experiments/outputs/exp_020a_texture_artifact_correction_oof/texture_artifacts/texture_manifest.json`

## Validation Status

- notebook scaffolded
- syntax checked with `ast.parse`
- Kaggle/local artifact-backed run has now completed on attached `exp_015d` artifacts and full Perch cache

## Result Snapshot

- baseline overall row macro AUC: `0.993120`
- baseline overall file macro AUC: `0.992126`
- baseline target row macro AUC: `0.997988`
- baseline target file macro AUC: `0.998281`
- correction-only target row macro AUC: `0.482546`
- best correction blend weight: `0.00`
- trained target models: `36`
- skipped low-support labels: `6`

Weight sweep summary:

- every positive correction weight was worse than the baseline
- best blended result stayed exactly at the pure baseline (`w_corr = 0.00`)
- as `w_corr` increased, both overall and target AUC degraded monotonically

Classwise pattern:

- only `1` trained target class improved at all
- `29` trained target classes got worse
- `6` were unchanged

Main interpretation:

- this does **not** validate the artifactized texture-correction idea in its current form
- the benchmark proxy is almost certainly too easy / misaligned:
  - target baseline AUC is already near-saturated on trusted full rows
  - that does not reproduce the real soundscape weakness seen in `exp_017` and the earlier specialist experiments
- practical consequence:
  - do not promote this exact `exp_020a` output into a thin submit overlay
  - if we revisit the specialist-correction idea, it should be redesigned around soundscape-aware or grouped file-level targets rather than full-row macro AUC

## Intended Interpretation

If this branch improves local overall macro AUC while helping target taxa, it becomes the strongest candidate for the next thin submit overlay after `exp_015d`.

If it only improves target-only metrics but hurts overall AUC, then the specialist idea remains interesting scientifically but should stay out of the deployment path.
