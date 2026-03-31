# exp_015d_v18_artifact_submit

## Goal

Provide a thin Kaggle submission notebook for the V18 Pantanal/ProtoSSM stack that loads pre-exported downstream artifacts instead of training them during submit.

## Expected Inputs

- competition data
- TensorFlow wheels dataset
- `perch_v2_cpu`
- V18 artifact dataset exported by `exp_015c_v18_artifact_export`

## Behavior

- loads `ProtoSSM` state
- loads optional `ResidualSSM` state
- loads `StandardScaler`, `PCA`, and classwise `MLP probes`
- loads prior tables and per-class thresholds
- runs only hidden-test Perch inference, fusion, and postprocess

## Motivation

This notebook is the timeout-safe continuation of `exp_015c`. It keeps the V18 recipe but removes submit-time training, which was identified as the main cause of Kaggle notebook timeout.

## Outcome

- first Kaggle submission completed successfully
- public LB: `0.929`
- delta vs `exp_015`: `+0.004`
- practical impact reported during the run: movement from roughly public-rank `120` to public-rank `21`

## Interpretation

This result validates the artifactized split directly. The V18 branch did not need a simpler model; it needed a better execution path. Once the train-time components were exported offline, the same modeling family became both runnable and stronger than the original `0.925` baseline.
