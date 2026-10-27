# Supervised errors by base, arm and individual joint

> Coverage update, 27 October 2026: the original report below uses only the fixed
> 26-goal anchor. Complete own-validation and paired-union results are now in
> [SUPERVISED_FULL_VALIDATION.md](SUPERVISED_FULL_VALIDATION.md). Do not interpret
> the following values as full-validation scores for the larger datasets.


Analysis requested by Rodrigo on 27 October 2026. This report reuses the completed
CPU/sklearn offline results; no retraining or simulation is required. The same
26 held-out expert-state conditions and equal episode weights are used at every
size. The PyTorch pilot remains separate. This is supervised **command** error,
not base position, arm pose, or online trajectory error.

Base error means the average RMSE of the two independent wheel commands, in
rad/s. Arm error means the average of the seven individual torque RMSEs, in N·m.
For each output, compute within-episode RMSE, then average equally over episodes.
Do not confuse this with pooled-row RMSE or RMSE of the average arm torque.

The reference normalizers remain the original uniform64 training statistics,
allowing the saved NRMSEs to be checked against physical RMSEs. A fixed baseline
predicts the original64 training mean on each channel, evaluated on the same26
conditions. Its mean was not fitted on validation data.

## Aggregate physical errors

|Episodes|Uniform wheel RMSE(rad/s)|Stratified wheel RMSE(rad/s)|Uniform mean joint RMSE(Nm)|Stratified mean joint RMSE(Nm)|
|---:|---:|---:|---:|---:|
|64|0.562544|0.545047|0.385437|0.461315|
|128|0.494021|0.369318|0.393672|0.323439|
|256|0.272548|0.251589|0.276886|0.263278|
|512|0.133003|0.170888|0.207429|0.223222|
|1024|0.124279|0.128841|0.191745|0.203759|
|2048|0.097845|0.127135|0.186289|0.202357|

![Base and arm RMSE](../runs/diagnostics/supervised_subsystem_scaling_20261027/base_arm_rmse.png)

## What doubling from1024 to2048 actually buys

Positive reductions mean improvement; negative values mean deterioration. Quote
physical reductions alongside percentages: a large relative reduction of an
already small error can have a small physical effect.

|Design|Output group|Previous RMSE|New RMSE|Absolute reduction|Relative reduction(%)|Unit|
|---|---|---:|---:|---:|---:|---|
|uniform|base|0.124279|0.097845|+0.026433|+21.27|rad/s|
|uniform|arm|0.191745|0.186289|+0.005456|+2.85|N*m|
|stratified|base|0.128841|0.127135|+0.001706|+1.32|rad/s|
|stratified|arm|0.203759|0.202357|+0.001402|+0.69|N*m|

The uniform aggregate gain is concentrated in wheel commands:21.27% versus2.85%
for arm torques. The stratified model changes little in both groups at this
step. The single global balanced NRMSE hid these different learning curves.

## Arm: every joint in physical units

### Uniform sampling

|Joint|64|128|256|512|1024|2048|
|---:|---:|---:|---:|---:|---:|---:|
|1|0.424744|0.523779|0.298234|0.238220|0.209484|0.208139|
|2|0.469132|0.413734|0.264477|0.222797|0.199010|0.193367|
|3|0.470403|0.430003|0.375722|0.246939|0.238041|0.225036|
|4|0.400270|0.504553|0.289454|0.213367|0.191965|0.190840|
|5|0.460136|0.360669|0.374225|0.269412|0.252895|0.247594|
|6|0.331657|0.404635|0.280923|0.234539|0.225469|0.220925|
|7|0.141715|0.118333|0.055167|0.026727|0.025347|0.018120|

All cells are equal-episode torque RMSE in N·m.

### Stratified sampling

|Joint|64|128|256|512|1024|2048|
|---:|---:|---:|---:|---:|---:|---:|
|1|0.534135|0.371123|0.332729|0.258184|0.230087|0.233021|
|2|0.532420|0.409242|0.280710|0.235408|0.202059|0.203847|
|3|0.468144|0.363841|0.291323|0.267254|0.241206|0.240801|
|4|0.527928|0.322021|0.264912|0.212995|0.211919|0.205275|
|5|0.522998|0.412338|0.357788|0.298262|0.271521|0.262794|
|6|0.535408|0.326343|0.269609|0.248739|0.235411|0.231850|
|7|0.108170|0.059164|0.045873|0.041709|0.034107|0.038909|

All cells are equal-episode torque RMSE in N·m.

![Per-joint learning curves](../runs/diagnostics/supervised_subsystem_scaling_20261027/joint_rmse.png)

### Last doubling by joint

|Joint|Uniform change1024→2048(Nm)|Uniform reduction(%)|Stratified change1024→2048(Nm)|Stratified reduction(%)|
|---:|---|---:|---|---:|
|1|0.209484→0.208139|+0.64|0.230087→0.233021|-1.28|
|2|0.199010→0.193367|+2.84|0.202059→0.203847|-0.89|
|3|0.238041→0.225036|+5.46|0.241206→0.240801|+0.17|
|4|0.191965→0.190840|+0.59|0.211919→0.205275|+3.14|
|5|0.252895→0.247594|+2.10|0.271521→0.262794|+3.21|
|6|0.225469→0.220925|+2.02|0.235411→0.231850|+1.51|
|7|0.025347→0.018120|+28.51|0.034107→0.038909|-14.08|

Uniform joint7 improves28.51%, but the absolute improvement is only0.00723Nm.
Uniform joints1 and4 improve less than1% at this step. Stratified joints1,2 and7
worsen, despite the small improvement of the arm average. Its joint7 RMSE rises
14.08%, an absolute0.00480Nm deterioration. There is no monotonic per-joint
benefit merely from increasing demonstration count in this realization.

## Wheels individually and comparison against a fixed mean predictor

The independent outputs are right/left wheel commands. The other two physical
wheels are duplicates and are not counted as additional independent outputs.

|Design|Output|RMSE1024(rad/s)|RMSE2048(rad/s)|Reduction(%)|
|---|---|---:|---:|---:|
|uniform|wheel_right|0.125765|0.095239|+24.27|
|uniform|wheel_left|0.122792|0.100452|+18.19|
|stratified|wheel_right|0.126897|0.128250|-1.07|
|stratified|wheel_left|0.130786|0.126020|+3.64|

|Output|Unit|Fixed mean predictor RMSE|Uniform2048 RMSE|Stratified2048 RMSE|Reference training std|
|---|---|---:|---:|---:|---:|
|wheel_right|rad/s|1.566877|0.095239|0.128250|1.641322|
|wheel_left|rad/s|1.711097|0.100452|0.126020|1.696670|
|arm_torque_1|N*m|2.193982|0.208139|0.233021|2.307339|
|arm_torque_2|N*m|1.456663|0.193367|0.203847|1.392260|
|arm_torque_3|N*m|1.306544|0.225036|0.240801|1.832342|
|arm_torque_4|N*m|2.079716|0.190840|0.205275|2.156871|
|arm_torque_5|N*m|1.430785|0.247594|0.262794|1.474745|
|arm_torque_6|N*m|2.100413|0.220925|0.231850|2.084557|
|arm_torque_7|N*m|0.715272|0.018120|0.038909|1.169525|

The CSV also records reference training mean, validation mean absolute expert
command, MAE, fixed-scale NRMSE and strict5% nonzero-command fidelity. Error
magnitude and relative tolerance are complementary; channels have different
expert command magnitudes and variability.

## Interpretation and reproducibility

An approximate offline compromise should be assessed per output group. Arm
improvement flattens substantially by1024 in this realization; uniform wheel
prediction still benefits from2048. A universal256/512/1024 cutoff cannot be
inferred from one averaged NRMSE curve. No practical per-channel tolerance or
statistical plateau criterion was declared, and only one realization per design
was run. Different stopping epochs and update budgets remain confounds.

These command errors do not establish accurate base navigation, correct arm
pose or stable autonomous control. The completed [partial online report](PATIENCE_PARTIAL_REPORT.md)
shows the distinction. No isolated arm pose score is introduced here.

Run from the project with the RoboEnv environment active:

```bash
python scripts/report_supervised_scaling.py --bank anchor26 \
  --output-dir runs/diagnostics/NEW_SUPERVISED_ERROR_REPORT
```

The reporter validates complete model/split checksums,26-goal bank sizes, exact
agreement of per-channel and grouped saved metrics, and physical RMSE divided by
the fixed reference std. Files include24 group rows,108 per-output rows and110
adjacent-size/output comparisons. There are five scientific PNG/PDF figure
families. Existing run directories are not overwritten.

- [metrics.json](../runs/diagnostics/supervised_subsystem_scaling_20261027/metrics.json).
- [base_arm.csv](../runs/diagnostics/supervised_subsystem_scaling_20261027/base_arm.csv).
- [per_output.csv](../runs/diagnostics/supervised_subsystem_scaling_20261027/per_output.csv).
- [incremental_changes.csv](../runs/diagnostics/supervised_subsystem_scaling_20261027/incremental_changes.csv).
- [base_arm_rmse PNG](../runs/diagnostics/supervised_subsystem_scaling_20261027/base_arm_rmse.png), [PDF](../runs/diagnostics/supervised_subsystem_scaling_20261027/base_arm_rmse.pdf).
- [base_arm_mae PNG](../runs/diagnostics/supervised_subsystem_scaling_20261027/base_arm_mae.png), [PDF](../runs/diagnostics/supervised_subsystem_scaling_20261027/base_arm_mae.pdf).
- [joint_rmse PNG](../runs/diagnostics/supervised_subsystem_scaling_20261027/joint_rmse.png), [PDF](../runs/diagnostics/supervised_subsystem_scaling_20261027/joint_rmse.pdf).
- [joint_nrmse_train_std PNG](../runs/diagnostics/supervised_subsystem_scaling_20261027/joint_nrmse_train_std.png), [PDF](../runs/diagnostics/supervised_subsystem_scaling_20261027/joint_nrmse_train_std.pdf).
- [wheel_rmse PNG](../runs/diagnostics/supervised_subsystem_scaling_20261027/wheel_rmse.png), [PDF](../runs/diagnostics/supervised_subsystem_scaling_20261027/wheel_rmse.pdf).
