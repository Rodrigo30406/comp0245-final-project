# Partial report: larger datasets and patience-only MLP training

Snapshot: **2026-10-26T23:22:17.589061+00:00 UTC**, taken while the campaign is running.
The immutable supporting files are in `runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z`. This report concerns the
**CPU sklearn sampling/epoch campaign**. The separate PyTorch64 pilot is described
in [PyTorch training](PYTORCH_TRAINING.md) and is not mixed into these curves.

## Current evidence and scope

All four new collections and all five new MLP fits are complete. All12 MLPs in
the six-size comparison stopped by patience; seven eligible earlier models were
reused. Complete offline results exist through2048 for both sampling designs.
Paired online results through1024 include both complete native validation banks
and the same26 held-out anchor conditions. Online2048 is still pending in this
snapshot: no statistics are estimated from an unfinished bank.

At the snapshot, **2872/4206** new online jobs had
finished. Some jobs in a running bank have finished but are excluded from this
report's outcome aggregates until the complete bank is available. Repeated
expert/model rollouts and repeated uniform prefixes are not independent samples.

Physical rules, expert, box, initial state,21inputs/9outputs,128/128 architecture,
Adam settings and loss remain fixed. Complete evaluation forces40s and uses
0.5s dwell with1cm primary task tolerance. NRMSE uses the same original uniform64
training output scales for every model, with equal episodes and50/50 wheel/arm
group weighting. It is **not percent command accuracy**. Strict5% fidelity is
relative to each nonzero expert command; numerical zeros are assessed separately.

## Main findings

1. More demonstrations lower aggregate expert-state command error in this
   realization. Uniform anchor NRMSE improves from0.279545
   to0.081433;
   stratified improves from0.293001
   to0.095009.
2. Lower offline error has not produced stable autonomous control through1024.
   Both models have0/410 task arrivals on the complete1024 native union; the
   expert has410/410. All these rollouts are valid full-horizon simulations.
3. Removing the60-epoch cap from uniform512 did not resolve its instability.
   The first60 losses/scores reproduce exactly; the extended fit stops at79,
   selects71 and improves anchor NRMSE only0.58%. Final-position medians worsen.
4. Stratification strongly reduces the scale of simulated divergence in the
   completed512/1024 comparisons, despite sometimes higher supervised error.
   It still does not give a useful controller. A single realization and changing
   selected epochs do not prove universal sampler superiority.
5. Zero-command predictions improve substantially with more data, while strict
   nonzero5% accuracy remains uneven across wheels and joints. Better stopping
   imitation on expert states is insufficient evidence of stopping online.

## Offline results on the same26 held-out goals

|Episodes|Uniform balanced NRMSE|Stratified balanced NRMSE|Uniform wheel RMSE(rad/s)|Stratified wheel RMSE(rad/s)|Uniform mean joint RMSE(Nm)|Stratified mean joint RMSE(Nm)|
|---:|---:|---:|---:|---:|---:|---:|
|64|0.279545|0.293001|0.56254|0.54505|0.38544|0.46131|
|128|0.258103|0.202679|0.49402|0.36932|0.39367|0.32344|
|256|0.159855|0.149597|0.27255|0.25159|0.27689|0.26328|
|512|0.098067|0.114251|0.13300|0.17089|0.20743|0.22322|
|1024|0.091104|0.095810|0.12428|0.12884|0.19174|0.20376|
|2048|0.081433|0.095009|0.09785|0.12713|0.18629|0.20236|

From1024 to2048, uniform NRMSE decreases another
10.62%, whereas stratified decreases only
0.84%.
The stratified curve is flattening for this aggregate metric in this realization;
this does not establish a universal optimal dataset size. Updates, validation
signal and selected epochs differ between fits, and no final test set is reserved.

![offline anchor curve](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/offline_anchor_curve.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/offline_anchor_curve.pdf)
## Controlled epoch-cap comparison: uniform512

Same dataset, ordered split, seed and exact training-only normalizers. All60
historical epoch losses and validation scores are identical before extending the
budget. This isolates the budget intervention for this specific run; it does not
establish that extra epochs generally worsen control.

|Quantity, fixed26 anchor|Historical cap60 / best58|Patience79 / best71|
|---|---:|---:|
|Balanced NRMSE|0.098639|0.098067|
|Median final baseXY error(m)|204.6487|233.6493|
|Median final EE error(m)|441.4852|746.9644|
|Mean wheel clipping(%)|91.72|91.26|
|Stable task arrivals|0/26|0/26|

The same adverse pattern appears on the full206 native goals: median final
baseXY error288.7709→458.1579m and
EE error501.2948→1037.2507m, still0/206 task arrivals.
The improved aggregate command score does not select a better closed-loop policy
in this run. Large distances are signs of unstable simulation, not physically
credible travel by the robot.

![cap512 trajectories](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/cap512_trajectories.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/cap512_trajectories.pdf)
## Completed autonomous comparisons

Same26 conditions for every row below; forced40s horizon. Position errors are
medians of final distances; yaw is final navigation bearing error, not an arm
score. Clipping is the mean full-step wheel raw-versus-applied fraction. Base
arrival is a strategy diagnostic; original-target EE success remains primary.

|Design|Episodes|Task1cm|Base arrival|Median baseXY(m)|Median EE(m)|Median nav yaw error(deg)|Wheel clipping(%)|Torque clipping(%)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|uniform|64|0/26|0/26|203.5631|899.7475|76.94|93.00|82.34|
|stratified|64|0/26|0/26|2.3867|2.7069|80.71|31.43|15.01|
|uniform|128|0/26|0/26|84.2155|85.0073|94.23|92.27|73.57|
|stratified|128|0/26|0/26|7.8624|8.1155|88.94|82.68|49.76|
|uniform|256|0/26|0/26|3.5482|3.7447|87.91|87.08|26.05|
|stratified|256|0/26|0/26|2.9396|3.2730|113.43|35.83|12.91|
|uniform|512|0/26|0/26|233.6493|746.9644|84.20|91.26|84.27|
|stratified|512|0/26|0/26|10.6650|11.6979|88.25|58.31|41.42|
|uniform|1024|0/26|0/26|242.5784|338.2837|103.04|79.58|68.84|
|stratified|1024|0/26|0/26|10.6287|10.7328|106.90|48.52|37.28|

![online anchor curve](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/online_anchor_curve.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/online_anchor_curve.pdf)
Full native comparisons are also complete for512 and1024, avoiding conclusions
based only on the common26-case anchor. Within each size the two models use the
same union of both validation banks. Targets differ between sizes.

|Design|Episodes|Full native goals|Task1cm|Task5cm|Base arrivals|Invalid|Median final baseXY(m)|Median final EE(m)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|expert|512|206|206|206|188|0|0.0511|0.0016|
|uniform|512|206|0|0|0|0|458.1579|1037.2507|
|stratified|512|206|0|1|1|0|12.8080|12.1721|
|expert|1024|410|410|410|377|0|0.0506|0.0016|
|uniform|1024|410|0|0|0|0|235.6947|397.7261|
|stratified|1024|410|0|0|0|0|16.1002|16.1129|

At1024 the stratified model has much less divergence than uniform, but its16.1m
native median base error remains far from the desired navigation behavior. On the
fixed26 anchor its median is10.63m, similar to10.66m at512. More expert-state data
has not provided a consistent monotonic improvement of autonomous navigation.

The next figure uses three original uniform anchor cases selected by sorted
scenario IDs at indices0,6,12, before looking at outcomes. Full and first2s base
paths, wrapped yaw differences and time-varying separation are shown. Later live
commands act on different states and must not be called supervised errors.

![design1024 trajectories](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/design1024_trajectories.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/design1024_trajectories.pdf)
## Command fidelity, zeros and per-joint errors

The following complete2048 offline results still concern expert-state replay.
MAE/RMSE use equal episode weights; strict5% fractions and conditional error tails
pool transitions. These are different weightings and are labelled accordingly.
Only commands with absolute expert value>1e-6 enter the relative criterion.
Conditional mean/P95 are **absolute physical errors among failed nonzero commands**.
A large percentage near zero need not imply an enormous physical command error.

|Design|Output|Unit|MAE|RMSE|Nonzero within5%(%)|Failed5% mean abs error|Failed5% P95 abs error|
|---|---|---|---:|---:|---:|---:|---:|
|uniform|wheel_right|rad/s|0.05058|0.09524|80.37|0.18296|0.42296|
|uniform|wheel_left|rad/s|0.05687|0.10045|76.63|0.17719|0.41117|
|uniform|arm_torque_1|N*m|0.06815|0.20814|29.14|0.08511|0.23783|
|uniform|arm_torque_2|N*m|0.05610|0.19337|99.63|2.12289|6.84094|
|uniform|arm_torque_3|N*m|0.05498|0.22504|67.23|0.13799|0.38918|
|uniform|arm_torque_4|N*m|0.06360|0.19084|51.82|0.09209|0.30187|
|uniform|arm_torque_5|N*m|0.07459|0.24759|37.01|0.10027|0.29951|
|uniform|arm_torque_6|N*m|0.04299|0.22092|97.31|0.38939|0.73669|
|uniform|arm_torque_7|N*m|0.01128|0.01812|36.89|0.01197|0.02856|
|stratified|wheel_right|rad/s|0.07234|0.12825|74.73|0.21452|0.50836|
|stratified|wheel_left|rad/s|0.07113|0.12602|74.00|0.21353|0.53745|
|stratified|arm_torque_1|N*m|0.08096|0.23302|36.02|0.10901|0.40977|
|stratified|arm_torque_2|N*m|0.05782|0.20385|99.48|1.86895|5.19217|
|stratified|arm_torque_3|N*m|0.06752|0.24080|56.60|0.13097|0.40909|
|stratified|arm_torque_4|N*m|0.08673|0.20527|31.36|0.09950|0.38220|
|stratified|arm_torque_5|N*m|0.09070|0.26279|28.05|0.10796|0.22864|
|stratified|arm_torque_6|N*m|0.04579|0.23185|95.47|0.39276|0.83552|
|stratified|arm_torque_7|N*m|0.02104|0.03891|33.58|0.02584|0.08785|

![per output rmse](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/per_output_rmse.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/per_output_rmse.pdf)

![relative 5pct anchor curve](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/relative_5pct_anchor_curve.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/relative_5pct_anchor_curve.pdf)
When both expert wheel commands are numerically zero,70898 pooled anchor rows
are evaluated separately with no percentage division. Mean predicted absolute
commands are shown inrad/s; these are imitation errors, not actual base speeds.

|Design|Episodes|Right mean abs command(rad/s)|Left mean abs command(rad/s)|
|---|---:|---:|---:|
|uniform|512|0.089680|0.090649|
|stratified|512|0.090021|0.097979|
|uniform|1024|0.037365|0.030844|
|stratified|1024|0.043347|0.057123|
|uniform|2048|0.007516|0.013074|
|stratified|2048|0.030793|0.024374|

Zero-command imitation improves, especially for uniform2048, but errors may
amplify once the policy induces different states. Wheel nonzero5% fidelity is
not monotonic with sample count even where aggregate RMSE falls. Joint2 and6
have high relative fidelity, while several other joints remain much weaker.
Separate units, command magnitudes and numerical zeros before interpreting this.

## Training, data and reproducibility

|Model|Completed epochs|Best epoch|Stop|Training rows|Adam updates|
|---|---:|---:|---|---:|---:|
|mlp_uniform64|22|21|patience|490730|10560|
|mlp_stratified64|11|3|patience|480899|5170|
|mlp_uniform128|17|9|patience|978961|16269|
|mlp_stratified128|19|11|patience|952907|17689|
|mlp_uniform256|24|16|patience|1911967|44832|
|mlp_stratified256|45|37|patience|1937067|85140|
|mlp_uniform512|79|71|patience|3775046|291273|
|mlp_stratified512|35|27|patience|3733369|127610|
|mlp_uniform1024|55|47|patience|7534366|404690|
|mlp_stratified1024|47|39|patience|7488122|343711|
|mlp_uniform2048|60|52|patience|15095908|884580|
|mlp_stratified2048|34|26|patience|15265547|506872|

The uniform2048 run stopped at60 by **patience**, not a60-epoch cap. Its recipe
has no cap and its best epoch was52. All current MLPs satisfy the same stopping
rule; this does not prove a global optimum or control stability.

New collection quality, with all stalled demonstrations retained:

|Design|Episodes|Training/validation|Recorded transitions|Collection successes|Collection stalled|Execution failures|
|---|---:|---|---:|---:|---:|---:|
|uniform|1024|819/205|9425506|1005|19|0|
|stratified|1024|819/205|9399971|1003|21|0|
|uniform|2048|1638/410|18875221|2015|33|0|
|stratified|2048|1638/410|19100770|2010|38|0|

Collection termination uses the original expert rules, while evaluation continues
through40s. A collected stalled episode is not a proof of physical impossibility.
Uniform prefixes and frozen512 memberships are checked and extended; larger
stratified grids are independent, with geometry-only balanced holdouts. No failed
demonstration filtering, temporal windows or extra state inputs were introduced.

![new goal layouts](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/new_goal_layouts.png)

[Standalone PDF](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/new_goal_layouts.pdf)

All selected evaluation conditions are held out by every compared model. Model
artifacts/split hashes were checked; the first60 uniform512 epoch scores/losses
reproduce exactly. This partial snapshot includes52 repeated expert-anchor metric
checks and104 valid paired anchor trace diagnostics. Exact records, source model
paths and checksums remain in the run folders. No running source/config is edited.

- [Snapshot](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/snapshot.json) and [full partial summary](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/partial_summary.json).
- [Per-output metrics and failed5% tails](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/command_metrics.csv).
- [Per-case online metrics](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/online_per_episode.csv) and [training table](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/training.csv).
- [Paired anchor trajectory diagnostics](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/anchor_trajectory_metrics.json).
- [Controlled cap comparison](../runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/budget_comparison.json).
- Snapshot-preserving report builder: `runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/build_report.py`.

## Interpretation and pending work

The tested cap intervention did not rescue uniform512 control. Increasing
expert demonstrations improves command regression and zero-command imitation,
but autonomous failures persist across complete1024 validation banks. This is
consistent with state-distribution shift and accumulated policy errors; hidden
expert memory remains another hypothesis. These observations do not isolate a
cause, prove classical overfitting, or quantify clipping's causal contribution.
Preserve the recorded raw/applied distinction and early trajectory onsets.

Online2048 must finish before its navigation/control results are ranked. The
fixed26 anchor, both full native banks and updated plots will appear in the
completed [patience-only study](PATIENCE_SERIES.md). No full GPU backend comparison,
DAgger, temporal-window or output-bounding experiment is part of this snapshot.

Regenerate this snapshot from the project root with `PYTHONPATH=src python runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z/build_report.py --output-dir runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z`. The saved complete-bank list is reused; no running bank is aggregated.
