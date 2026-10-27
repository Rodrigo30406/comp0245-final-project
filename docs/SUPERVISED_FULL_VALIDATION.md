# Supervised errors over complete validation sets

Coverage clarification requested by Rodrigo on 27 October 2026. The earlier
[per-subsystem report](SUPERVISED_ERROR_BY_SUBSYSTEM.md) used a **fixed26-goal
anchor**, not the full validation of larger datasets. Those measurements remain
valid for their labelled scope; they must not be described as full-validation
results. This report adds every held-out episode at each dataset size.

## Three distinct evaluation views

|Collected episodes|Own full validation, per model|Paired full union, per model|Fixed common anchor|
|---:|---:|---:|---:|
|64|13|26|26|
|128|26|52|26|
|256|52|104|26|
|512|103|206|26|
|1024|205|410|26|
|2048|410|820|26|

**Own validation** uses each model's entire original held-out split. Uniform and
stratified models see different goals in this view. **Paired union** evaluates
both models on all goals from both splits at a given size, making their
within-size comparison use identical conditions. **Anchor26** keeps the same
conditions across sizes, but has limited coverage. No unfinished online bank is
needed: all these supervised results were already available offline.

The full sets change across sizes, so1024→2048 error differences also reflect
changed held-out conditions. They do not isolate the causal effect of adding
training data. Repeating seeds and/or using a larger fixed common holdout would
address different uncertainties; no such new experiment is claimed here.

All entries remain CPU/sklearn model command errors on expert states. Physical
units and equal-episode weighting are unchanged: wheel RMSE averages2outputs,
arm RMSE averages7individual torque RMSEs. No pose/online score is introduced.

## Full own-validation results

|Episodes|Held-out goals per model|Uniform base RMSE(rad/s)|Stratified base RMSE(rad/s)|Uniform mean joint RMSE(Nm)|Stratified mean joint RMSE(Nm)|
|---:|---:|---:|---:|---:|---:|
|64|13|0.397360|0.562069|0.293183|0.549651|
|128|26|0.380261|0.339823|0.273397|0.279599|
|256|52|0.212712|0.236164|0.207328|0.238714|
|512|103|0.127150|0.204918|0.193301|0.262943|
|1024|205|0.131584|0.152741|0.188004|0.208614|
|2048|410|0.109563|0.151192|0.177688|0.220414|

![Full own validation](../runs/diagnostics/supervised_full_validation_20261027/own_validation/base_arm_rmse.png)

## Full paired-union results

|Episodes|Same held-out goals per model|Uniform base RMSE(rad/s)|Stratified base RMSE(rad/s)|Uniform mean joint RMSE(Nm)|Stratified mean joint RMSE(Nm)|
|---:|---:|---:|---:|---:|---:|
|64|26|0.562544|0.545047|0.385437|0.461315|
|128|52|0.423926|0.310607|0.291701|0.261127|
|256|104|0.243148|0.200030|0.236098|0.214945|
|512|206|0.156183|0.176229|0.219042|0.233309|
|1024|410|0.159949|0.142625|0.195933|0.202476|
|2048|820|0.179978|0.141635|0.227950|0.205000|

![Full paired validation](../runs/diagnostics/supervised_full_validation_20261027/native_union/base_arm_rmse.png)

## How coverage changes the1024→2048 interpretation

Positive numbers mean RMSE reduction; negative numbers mean deterioration.
The full banks vary across sizes; the anchor bank stays fixed.

|View|Design|Base reduction(%)|Arm reduction(%)|
|---|---|---:|---:|
|anchor26|uniform|+21.27|+2.85|
|anchor26|stratified|+1.32|+0.69|
|own_validation|uniform|+16.74|+5.49|
|own_validation|stratified|+1.01|-5.66|
|native_union|uniform|-12.52|-16.34|
|native_union|stratified|+0.69|-1.25|

The earlier anchor-only observation of general improvement cannot be extended
to every full validation population. For example, stratified own-validation arm
RMSE rises0.208614→0.220414Nm. On the full paired union, uniform base and arm
errors are higher at2048 than at1024. These results weaken a blanket claim that
more demonstrations always reduce held-out error, but do not by themselves prove
overfitting or that larger datasets cause worse models: the goals also changed.

Comparing samplers on their own splits can reverse the ranking seen on their
shared union. Quote the bank definition and number of held-out episodes beside
any score. The26-goal anchor and full sets answer complementary questions.

## Individual joint errors on full validation

### own_validation

All cells are equal-episode torque RMSE(Nm), with every goal in that view included.

|Joint|Uniform512|Uniform1024|Uniform2048|Stratified512|Stratified1024|Stratified2048|
|---:|---:|---:|---:|---:|---:|---:|
|1|0.239769|0.231535|0.220654|0.312439|0.252682|0.276581|
|2|0.191814|0.188405|0.174974|0.247513|0.202177|0.204680|
|3|0.251492|0.253323|0.227697|0.320494|0.270265|0.274294|
|4|0.189315|0.175625|0.171481|0.278208|0.202288|0.210335|
|5|0.222575|0.217321|0.206916|0.302722|0.251948|0.266065|
|6|0.235220|0.227630|0.224866|0.335449|0.250878|0.260520|
|7|0.022925|0.022188|0.017231|0.043776|0.030058|0.050419|

![own_validation joint curves](../runs/diagnostics/supervised_full_validation_20261027/own_validation/joint_rmse.png)

### native_union

All cells are equal-episode torque RMSE(Nm), with every goal in that view included.

|Joint|Uniform512|Uniform1024|Uniform2048|Stratified512|Stratified1024|Stratified2048|
|---:|---:|---:|---:|---:|---:|---:|
|1|0.265342|0.240896|0.279424|0.280838|0.250252|0.256180|
|2|0.219294|0.199137|0.228245|0.224584|0.196376|0.192997|
|3|0.297442|0.265374|0.273492|0.290983|0.263883|0.258018|
|4|0.216526|0.180378|0.206507|0.242293|0.193012|0.193828|
|5|0.251575|0.228224|0.272567|0.263060|0.244575|0.245400|
|6|0.253587|0.232976|0.314466|0.293620|0.241158|0.244489|
|7|0.029528|0.024545|0.020950|0.037786|0.028078|0.044089|

![native_union joint curves](../runs/diagnostics/supervised_full_validation_20261027/native_union/joint_rmse.png)

## Reproduction and checks

```bash
python scripts/report_supervised_scaling.py --bank all \
  --output-dir runs/diagnostics/NEW_FULL_VALIDATION_REPORT
```

The CLI now defaults to `--bank native_union`, covering the full paired sets.
Choose `own_validation` for each model's complete holdout and `anchor26` to
reproduce the original fixed-bank comparison. `all` creates three separate
folders with explicit bank names/counts in every CSV/JSON result and figure.
No training or physics is rerun. Output directories must be new.

Verified: own-bank physical errors match the full validation metrics stored by
each original training run; the regenerated anchor reproduces the earlier report
exactly; all expected bank sizes and model/split checksums pass. The paired
union's equal-episode metrics equal the mean of the two equal-size banks.
Fixed training-mean predictor baselines are recomputed on the selected full bank,
not reused from26cases. Each view has24group rows,108output rows and110adjacent
size comparisons, plus five PNG/PDF figure families. The old report is preserved.

- **own_validation:** [group CSV](../runs/diagnostics/supervised_full_validation_20261027/own_validation/base_arm.csv), [each output](../runs/diagnostics/supervised_full_validation_20261027/own_validation/per_output.csv), [incremental changes](../runs/diagnostics/supervised_full_validation_20261027/own_validation/incremental_changes.csv), [JSON with coverage/provenance](../runs/diagnostics/supervised_full_validation_20261027/own_validation/metrics.json).
- **native_union:** [group CSV](../runs/diagnostics/supervised_full_validation_20261027/native_union/base_arm.csv), [each output](../runs/diagnostics/supervised_full_validation_20261027/native_union/per_output.csv), [incremental changes](../runs/diagnostics/supervised_full_validation_20261027/native_union/incremental_changes.csv), [JSON with coverage/provenance](../runs/diagnostics/supervised_full_validation_20261027/native_union/metrics.json).
- **anchor26:** [group CSV](../runs/diagnostics/supervised_full_validation_20261027/anchor26/base_arm.csv), [each output](../runs/diagnostics/supervised_full_validation_20261027/anchor26/per_output.csv), [incremental changes](../runs/diagnostics/supervised_full_validation_20261027/anchor26/incremental_changes.csv), [JSON with coverage/provenance](../runs/diagnostics/supervised_full_validation_20261027/anchor26/metrics.json).
