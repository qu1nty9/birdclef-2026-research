# TODO Research

## Completed Setup

- [x] Analyze project documentation and current repository state
- [x] Review reference notebooks in `references/private-notebooks/`
- [x] Identify first experiment and a local validation strategy based on labeled soundscapes
- [x] Confirm that `train_soundscapes_labels.csv` contains duplicated identical rows per segment and should be deduplicated during evaluation

## Immediate Next Steps

- [x] Submit the reference blend baseline to Kaggle and record the first public LB
- [ ] Run `exp_001_soundscape_reference_blend` in a Torch-enabled environment and record local macro ROC-AUC for all strategies
- [ ] Decide whether notebook heuristics help local validation or only public LB ranking
- [x] Start `exp_002_train_audio_reproduction` and train the same architecture on isolated `train_audio`
- [x] Submit the pure `exp_002` checkpoint through a dedicated Kaggle notebook and record the public LB
- [x] Create a notebook-only reproduction of the Perch downstream stack using the cached `perch_meta` arrays
- [ ] Submit the dedicated Perch `exp_003` Kaggle notebook and record the public LB
- [ ] Test whether site/hour priors and texture smoothing transfer to our non-Perch baselines
- [x] Promote the best locally validated inference recipe into the default Kaggle submission notebook
- [x] Create `exp_004_soundscape_finetuning` starting from the `exp_002` checkpoint because the pure isolated-audio model scored only `0.647` on Kaggle
- [x] Run `exp_004_soundscape_finetuning` and record the first local soundscape macro ROC-AUC
- [ ] Improve `exp_004` validation coverage with additional folds or OOF aggregation because the first fold scored only `29` classes
- [x] Build `exp_005_native_priors_texture_postproc` on top of `exp_004`
- [x] Test whether site/hour priors and texture smoothing transfer to our non-Perch baselines
- [ ] Compare `exp_004` and `exp_005` against `exp_003_perch_downstream_reproduction` on local soundscape metrics before another Kaggle submission
- [x] Build a lightweight native submission notebook using the best `exp_005` postprocessing recipe
- [x] Submit the dedicated `exp_005` native hybrid Kaggle notebook and record the public LB
- [x] Choose stronger native soundscape training as the next main branch after the `0.737` public LB result
- [x] Create `exp_006_soundscape_finetuning_v2` as a fold-aware notebook with exported validation predictions
- [x] Run `exp_006_soundscape_finetuning_v2` on fold `0` and record best macro ROC-AUC
- [x] Run at least `2-3` folds of `exp_006_soundscape_finetuning_v2` before trusting small native deltas
- [ ] Decide whether `exp_006` is a real improvement or just a protocol upgrade after additional folds
- [x] Apply the `exp_005` priors/texture recipe on top of exported `exp_006` fold predictions and compare against raw `exp_006`
- [x] Build the next native Kaggle submission from `exp_006 + priors`
- [x] Submit the `exp_006 + priors` native notebook to Kaggle and record the public LB
- [x] Start the long-context native SED branch (`exp_008`) because `exp_007` public LB `0.758` is still too far from `0.890`
- [x] Run fold `0` of `exp_008_long_context_native_sed` and record best macro ROC-AUC
- [x] Apply the `exp_007` priors/texture layer on top of `exp_008` fold `0` exports
- [x] Build the first Kaggle submission from the long-context branch using the `exp_008b` winning recipe
- [x] Record the first Kaggle LB for the long-context branch and compare it against the current native public best `0.758`
- [x] Run at least one more `exp_008` fold before trusting the long-context branch again on Kaggle
- [x] Build a fold-safe long-context OOF postprocess comparison on top of `exp_008` folds `0-2`
- [x] Decide whether long-context is a real branch or a single-fold mirage using pooled OOF before spending more leaderboard attempts on it
- [x] After the long-context branch, start a noisy-student pseudo-label branch (`exp_009`)
- [x] Run pseudo-label generation in `exp_009` and inspect pseudo confidence / coverage before any long training run
- [x] Launch a short smoke-train for `exp_009` after pseudo-label caching succeeds
- [x] Run the first full `exp_009` fold and record best macro ROC-AUC
- [x] Apply the `exp_007` priors/texture recipe on top of exported `exp_009` fold predictions
- [x] Run at least one more `exp_009` fold before treating the branch as Kaggle-ready
- [x] Run `exp_009` fold `2` to get a three-fold view before the first raw Kaggle promotion
- [x] Prepare a Kaggle submission notebook for raw `exp_009` without the old priors layer
- [x] Run the first Kaggle submission for raw `exp_009` and record the public LB
- [x] Build a pooled OOF / calibration analysis for `exp_009` before another leaderboard attempt
- [x] Prototype the `HGNetV2-B0 + wav-cache + soundscape-clip` supervised branch from the `0.856` reference
- [x] Run `exp_011_hgnetv2_soundscape_supervised` on fold `0` and record both overall and soundscape-only validation ROC-AUC
- [x] Run `exp_011_hgnetv2_soundscape_supervised` on folds `1-2` before any Kaggle promotion
- [x] Build a dedicated Kaggle submission notebook for `exp_011` if folds `1-2` confirm the fold `0` strength
- [x] Run the first Kaggle submission for `exp_011` and compare it against `exp_007 = 0.758`
- [x] Decide whether to prioritize `exp_011` fold `3` / stronger inference or jump directly to the simplified `0.924` ProtoSSM branch
- [x] Analyze `pantanal-distill-birdclef2026-improvement-0.924.ipynb` and extract the genuinely new ideas beyond the older Perch notebooks
- [x] Prototype a simplified `Perch embeddings + file-level temporal model` branch inspired by the `0.924` ProtoSSM stack
- [x] Run `exp_011_hgnetv2_soundscape_supervised` on fold `3` and select the checkpoint by soundscape-only ROC-AUC
- [x] Build a `4-fold` Kaggle submission package for `exp_011`
- [x] Run the second Kaggle submission for `exp_011` as a `4-fold` ensemble and compare it against the current `0.844` public baseline
- [x] Run the first grouped OOF experiment for `exp_012_perch_temporal_light` on cached `perch_meta`
- [x] Compare `exp_011` 4-fold vs `exp_012` on Kaggle readiness, research value, and ensemble potential
- [ ] Add a simpler `exp_012` ablation without the temporal/prototype stack to isolate whether the failure comes from grouping, gated fusion, or the SSM block itself
- [x] Add a simpler `exp_012` ablation without the temporal/prototype stack to isolate whether the failure comes from grouping, gated fusion, or the SSM block itself
- [x] Run the first grouped OOF comparison for `exp_012b_perch_temporal_ablation`
- [x] Decide whether the Perch local branch should be paused after `exp_012` and `exp_012b` both failed to beat raw Perch on pooled OOF
- [x] Scaffold `exp_014_hgnetv2_pseudolabel` as the next native modeling branch on top of `exp_011`
- [x] Run pseudo generation for `exp_014` fold `0` and inspect the retained confidence distribution before training
- [x] Run the first full `exp_014` fold and compare it against `exp_011` on soundscape-aware validation
- [x] Run `exp_014` fold `1` to test whether the first positive signal survives beyond one fold
- [x] Run `exp_014` fold `2` to decide whether the branch is stable enough for a first Kaggle submission
- [x] Run `exp_014` fold `3` to settle whether the branch is a real upgrade over `exp_011` or just a mixed continuation
- [ ] Decide whether to revise `exp_014` with stricter early stopping / pseudo filtering or leave it as a negative control and move the main effort to the `0.924` external branch
- [x] Scaffold `exp_014b_hgnetv2_pseudolabel_strict` as the conservative pseudo-label follow-up to `exp_014`
- [x] Run pseudo generation for `exp_014b` fold `0` and inspect whether the stricter recipe produces a meaningfully cleaner cache
- [x] Run the first full `exp_014b` fold and compare it against both `exp_014` and `exp_011`
- [ ] Decide whether to run more `exp_014b` folds later or freeze it as a successful diagnostic branch and move on
- [x] Choose one `0.924` reference notebook and operationalize it as a faithful submission path instead of another simplified local Perch rewrite
- [x] Run the first Kaggle submission for `exp_015_pantanal_proto_ssm_v17_submit_path`
- [x] Record runtime behavior and public LB for `exp_015`
- [x] Compare `exp_015d = 0.929` against `exp_011 = 0.850` at the design level and choose runtime blending as the safest first ensemble path
- [ ] Decide whether `exp_015` should fully replace `exp_001 = 0.890` as the external anchor in future blend experiments
- [x] Scaffold a lightweight Kaggle blend notebook for `exp_015 + exp_011` that works from attached `submission.csv` outputs instead of rerunning both heavy model stacks
- [x] Run the first Kaggle submission for `exp_016b_runtime_blend_exp015d_exp011`
- [ ] Compare `0.95 / 0.05` against nearby blend weights like `0.97 / 0.03` and `0.92 / 0.08` only if we explicitly want one more low-priority complementarity check
- [ ] Test whether in-model `site/hour` metadata embeddings outperform our older post-hoc priors on the same trusted full-file subset
- [ ] Prepare a dedicated `Amphibia/Insecta` specialist branch (`exp_010`) if the generic native branch still underperforms on texture-heavy classes

## Data And Validation

- [ ] Build a stable local validation report split by animal class and by rare classes
- [ ] Quantify performance on the 28 soundscape-only classes
- [ ] Inspect whether duplicate labels in `train_soundscapes_labels.csv` are a packaging artifact or intentional redundancy
- [ ] Evaluate how much performance changes if training ignores `secondary_labels`
- [ ] Evaluate fully labeled files as a dedicated trusted adaptation subset for second-stage models

## Training Ideas

- [ ] Train a clean baseline on `train_audio` only with the same mel frontend and backbone as the reference checkpoints
- [ ] Train a soundscape-aware finetuning stage on labeled `train_soundscapes`
- [x] Prototype a long-context native SED branch (`20s` chunks -> `5s` outputs) instead of treating each row as an independent `5s` clip
- [ ] Compare BCE vs focal-style loss for long-tail classes
- [ ] Test adding `secondary_labels` as weak multi-label supervision
- [ ] Port masked BCE for primary-plus-secondary labels from the training references and compare it against the current loss
- [ ] Test class-balanced sampling for species with `<= 20` isolated recordings
- [ ] Add a background-bank mixing augmentation stage using soundscape clips or trusted background windows
- [ ] Test whether a PCEN frontend improves soundscape transfer over the current mel frontend
- [x] Implement a real noisy-student pseudo-label branch with mixup between labeled data and pseudo-labeled soundscape chunks
- [ ] Test probability power-transform denoising and confidence-weighted sampling for pseudo-labeled soundscapes
- [ ] Try a dedicated `Amphibia/Insecta` model with targeted extra Xeno-Canto species instead of mixing those labels into the generic branch
- [x] Prototype a fast supervised HGNetV2-B0 branch using `train_audio + labeled soundscape clips` as a simpler alternative to heavier pseudo-label pipelines
- [ ] Test whether explicit soundscape clip extraction by contiguous `(filename, primary_label)` segments is stronger than our current raw soundscape-window supervision
- [ ] Build a wav-cache + partial-read audio loader path so full 4-fold supervised training is cheap enough to rerun often

## Feature And Architecture Ideas

- [ ] Compare `n_mels=128` vs `n_mels=224`
- [ ] Test lower `fmin` for amphibians and insects
- [ ] Compare EfficientNet-B0 against a stronger timm backbone once the baseline is stable
- [ ] Compare EfficientNet-B0 against `hgnetv2_b0.ssld_stage2_ft_in1k`
- [ ] Evaluate whether segmentwise logits from the attention head can improve sound event localization
- [ ] Test frozen embedding stackers on top of model outputs instead of only end-to-end finetuning
- [ ] Separate texture-style classes (`Amphibia`, `Insecta`) from bird/event classes in postprocessing or auxiliary heads

## Inference And Ensemble Ideas

- [ ] Validate temporal smoothing separately from global file-max leakage
- [ ] Compare probability blending vs logit blending
- [ ] Try class-aware thresholds only for diagnostic analysis, not as the main AUC optimization target
- [ ] Build a CPU-safe submission script that mirrors the best validated inference recipe
- [ ] Test overlap-average-max-delta aggregation on framewise SED predictions instead of independent `5s` clipwise inference
- [ ] Add metadata priors based on `site`, `hour_utc`, and `site-hour` combinations
- [x] Add metadata priors based on `site`, `hour_utc`, and `site-hour` combinations
- [ ] Compare plain logits against logits plus file-context features (`prev`, `next`, `mean`, `max`) in a second-stage classifier
- [ ] Evaluate OpenVINO or ONNX export once the native ensemble is strong enough to make CPU runtime a bottleneck
- [ ] Reproduce the HGNetV2/OpenVINO CPU-safe submit path as a native inference engineering branch
- [x] Verify whether the newly added `birdclef-2026-hgnetv2-b0-baseline-inference-0.859.ipynb` introduces any new inference logic beyond the already studied `0.856` HGNetV2 reference
- [ ] Port the active V18 changes from `pantanal-distill-birdclef2026-improvement-0.927.ipynb` on top of `exp_015`, starting with the larger ProtoSSM / residual configs, revised probe settings, adaptive delta smoothing, and updated fusion lambdas
- [x] Create a separate high-risk V18 submit notebook on top of `exp_015` instead of mutating the stable `0.925` baseline
- [x] Attempt the first Kaggle submission for `exp_015c_full_v18_submit_path`
- [x] Run `exp_015c_v18_artifact_export` to save V18 downstream artifacts (`ProtoSSM`, probes, residual, priors, thresholds)
- [x] Package the exported artifacts as a Kaggle dataset
- [x] Run the first Kaggle submission for `kaggle_submission_exp_015d_v18_artifact_submit`
- [x] Scaffold `exp_015e` as a calibration-first refinement with isotonic calibration artifacts and a thin calibrated submit path
- [ ] Run `exp_015e_v18_calibrated_artifact_export`
- [ ] Package the calibrated artifact dataset for Kaggle
- [ ] Run the first Kaggle submission for `kaggle_submission_exp_015e_v18_calibrated_submit`
- [x] Scaffold `exp_015f` as a thin calibration-refresh export on top of fixed `exp_015d` artifacts
- [ ] Run `exp_015f_v18_calibration_refresh_export`
- [x] Scaffold `kaggle_submission_exp_015f_v18_calibration_refresh_submit`
- [ ] Package the refreshed artifact dataset for the `exp_015f` thin submit notebook
- [ ] Run the first Kaggle submission for `kaggle_submission_exp_015f_v18_calibration_refresh_submit`
- [ ] Decide whether to pause `exp_015f` after repeated timeout behavior despite a passing `exp_015g` smoke-submit
- [x] Scaffold `exp_015g_smoke_submit` as a minimal Kaggle timeout diagnostic notebook
- [x] Run `exp_015g_smoke_submit` in the exact same Kaggle settings that previously timed out
- [x] Scaffold `exp_017_v18_error_report` as a pooled native error-analysis notebook
- [x] Run `exp_017_v18_error_report` locally on pooled `exp_011` outputs
- [ ] Attach a V18 artifact dataset to `exp_017_v18_error_report` and generate the optional external-path threshold/calibration crosswalk
- [x] Scaffold `exp_018a_texture_specialist_oof` as the first `Amphibia + Insecta` specialist notebook on top of the `exp_011` training recipe
- [x] Validate `exp_018a_texture_specialist_oof` locally in safe setup mode with `RUN_TRAINING=False`
- [x] Run fold `0` training for `exp_018a_texture_specialist_oof`
- [x] Compare `exp_018a` target soundscape macro AUC against the weak-taxa diagnostics from `exp_017`
- [x] Run fold `1` training for `exp_018a_texture_specialist_oof`
- [x] Run fold `2` training for `exp_018a_texture_specialist_oof`
- [x] Run fold `3` training for `exp_018a_texture_specialist_oof`
- [x] Decide whether to promote `exp_018a` into `exp_018b` as a packaged multi-fold specialist correction branch
- [x] Design and run a local targeted merge benchmark that overwrites or blends only `Amphibia/Insecta` columns using pooled aligned OOF
- [x] Decide whether to promote `exp_018b` into a Kaggle-facing targeted overlay on top of `exp_015d`
- [x] If promoted, build a submit notebook that runs specialist inference only for `Amphibia/Insecta` columns and blends them into `exp_015d`
- [ ] Run the first Kaggle submission for `kaggle_submission_exp_018c_exp015d_texture_overlay`
- [ ] Sweep conservative overlay weights around the first `exp_018c` run, starting with `0.25 / 0.35 / 0.45`, only if the first submit is runtime-safe
- [x] Review whether the newly added `luck-factor-0.928.ipynb` introduces a new top-priority path beyond the current `exp_015d` / V18 family
- [x] Review whether `pantanal-distill-birdclef2026-improvement-a4dc68-0.930.ipynb` adds source-code novelty beyond `luck-factor-0.928.ipynb`
- [x] Review `0-928-luck-factor-just-edit-run-instantly.ipynb`
- [x] Review `bird26-reprod-perch-proto-residualssm-train-s7177.ipynb`
- [x] Review `bird26-reproduce-perch-protossm-resssm-inf-train.ipynb`
- [ ] Decide whether batched `temporal_shift_tta` from the second `bird26` notebook is worth porting into a controlled local benchmark
- [x] Freeze the original `exp_016` as a historical scaffold and move the active blend plan to `exp_016b = exp_015d + exp_011`
- [ ] Reproduce target-domain pseudo-labeling with overlapping `5s` windows, `2.5s` hop, temporal smoothing, and classwise quantile filtering in a notebook-only experiment
