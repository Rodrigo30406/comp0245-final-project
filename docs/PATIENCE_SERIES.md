# Patience-only training and extension to 2048 episodes

Authorized by Rodrigo on26 October2026. The frozen
[64–512 series](SAMPLING_SERIES.md) remains unchanged. This new experiment tests
whether the historical uniform512 epoch cap constrained validation performance,
and extends both sampling designs to1024 and2048 episodes.

The MLP architecture21→128→128→9, ReLU/linear outputs, Adam learning rate0.001,
L2alpha0.0001, batch1024, seed20261025 and all physical rules remain unchanged.
`experiment_specs/training/mlp_patience.json` removes the epoch cap (`null`).
Patience8 and significant improvement threshold1e-4 on equal-episode balanced
validation NRMSE remain. Select the lowest validation score, not the final epoch.
Seven previous MLPs already satisfied this stopping condition and are reused.
Uniform512 is rerun from the original seed with the exact frozen normalizers;
its first60 validation scores must reproduce before attributing later changes
solely to budget. Historical OLS is retained as a reference, not retrained here.

New train/validation splits are819/205 and1638/410. Uniform sampling extends the
verified original target/trajectory prefix and frozen512 split. Stratified grids
are16×8×8 and16×16×8, independent jittered realizations. Holdouts balance axis
margins, represent all octants and retain a training target per adjacent vertical
pair, using geometry only and fixed seeds before outcomes. Same box and start.
All models are compared on the same26-condition anchor and both complete native
validation banks; selected conditions must be absent from every model's training.
No final test set exists. One realization and different selected epoch/update
counts cannot establish general superiority or isolate sampling causally.

Training uses float64 disk-backed raw arrays and float32 normalized arrays.
New training-only normalization uses incremental StandardScaler per episode;
the512 budget comparison loads exact original training-only statistics. Full
Adam and shuffle state is checkpointed every epoch with context and checksums.
The disk workspace is removed after successful training. Offline paired metrics
stream episode observations and use temporary disk-backed truth/prediction banks,
retaining exact pooled quantiles, strict5% nonzero command fidelity and separate
numerical-zero diagnostics. Complete model/mean-baseline errors and joint-level
errors remain available. No clipping or policy architecture changes are added.

```bash
python scripts/compare_patience_series.py \
  --output-dir runs/sampling_comparison/patience_series_20261026 --workers 16
```

Status: collection, training, offline comparisons and full autonomous evaluation complete. Measured results follow.

Live evidence: `runs/sampling_comparison/patience_series_20261026/run.json`.
Read a compact progress snapshot with:

```bash
python runs/sampling_comparison/patience_series_20261026/progress.py
```
Allowing patience to stop training does not guarantee stability or prove the cap
caused historical divergence. Online40s matched rollouts retain arrival/dwell,
baseXY/yaw, stopping, saturation and full-system end-effector metrics.


## Measured stopping and command accuracy

All12 selected MLPs stopped by patience. Seven frozen smaller runs already met
that criterion; five were trained in this extension. The rerun uniform512 exactly
reproduces all60 historical epoch losses and validation scores before extending
the budget. The new runs preserve full optimizer checkpoints and training-only
normalizers. This is a validation-selected comparison, not a final test.

|Design|Episodes|Completed epochs|Selected epoch|Stop|Adam updates|
|---|---:|---:|---:|---|---:|
|uniform|64|22|21|patience|10560|
|stratified|64|11|3|patience|5170|
|uniform|128|17|9|patience|16269|
|stratified|128|19|11|patience|17689|
|uniform|256|24|16|patience|44832|
|stratified|256|45|37|patience|85140|
|uniform|512|79|71|patience|291273|
|stratified|512|35|27|patience|127610|
|uniform|1024|55|47|patience|404690|
|stratified|1024|47|39|patience|343711|
|uniform|2048|60|52|patience|884580|
|stratified|2048|34|26|patience|506872|

Same26 held-out expert-state conditions and fixed original64 output scales:

|Design|Episodes|Balanced NRMSE|Wheel RMSE(rad/s)|Mean joint torque RMSE(Nm)|
|---|---:|---:|---:|---:|
|uniform|64|0.279545|0.56254|0.38544|
|stratified|64|0.293001|0.54505|0.46131|
|uniform|128|0.258103|0.49402|0.39367|
|stratified|128|0.202679|0.36932|0.32344|
|uniform|256|0.159855|0.27255|0.27689|
|stratified|256|0.149597|0.25159|0.26328|
|uniform|512|0.098067|0.13300|0.20743|
|stratified|512|0.114251|0.17089|0.22322|
|uniform|1024|0.091104|0.12428|0.19174|
|stratified|1024|0.095810|0.12884|0.20376|
|uniform|2048|0.081433|0.09785|0.18629|
|stratified|2048|0.095009|0.12713|0.20236|

## Autonomous control

Same26 matched conditions,40s forced horizon, stable arrival requires0.5s dwell.
Extreme simulated distances indicate divergence, not physically credible travel.
Base arrival is a diagnostic; the original-target EE task remains primary.

|Design|Episodes|Task1cm|Base arrival|Median final baseXY(m)|Median final EE(m)|Median final nav yaw error(deg)|Wheel clipping(%)|
|---|---:|---:|---:|---:|---:|---:|---:|
|uniform|64|0/26|0/26|203.5631|899.7475|76.94|93.00|
|stratified|64|0/26|0/26|2.3867|2.7069|80.71|31.43|
|uniform|128|0/26|0/26|84.2155|85.0073|94.23|92.27|
|stratified|128|0/26|0/26|7.8624|8.1155|88.94|82.68|
|uniform|256|0/26|0/26|3.5482|3.7447|87.91|87.08|
|stratified|256|0/26|0/26|2.9396|3.2730|113.43|35.83|
|uniform|512|0/26|0/26|233.6493|746.9644|84.20|91.26|
|stratified|512|0/26|0/26|10.6650|11.6979|88.25|58.31|
|uniform|1024|0/26|0/26|242.5784|338.2837|103.04|79.58|
|stratified|1024|0/26|0/26|10.6287|10.7328|106.90|48.52|
|uniform|2048|0/26|0/26|325.0103|811.5534|60.82|83.64|
|stratified|2048|0/26|0/26|12.7403|13.6626|106.81|68.21|

Both complete native banks per size, jointly evaluated by both designs. Conditions
differ across sizes; these rows do not replace the fixed26 comparison.

|Design|Episodes|Task1cm|Invalid|Median final baseXY(m)|Median final EE(m)|
|---|---:|---:|---:|---:|---:|
|uniform|64|0/26|0|203.5631|899.7475|
|stratified|64|0/26|0|2.3867|2.7069|
|uniform|128|0/52|0|99.6067|100.0616|
|stratified|128|0/52|0|6.8302|7.5351|
|uniform|256|0/104|0|4.5037|4.4255|
|stratified|256|0/104|0|3.1483|3.3926|
|uniform|512|0/206|0|458.1579|1037.2507|
|stratified|512|0/206|0|12.8080|12.1721|
|uniform|1024|0/410|0|235.6947|397.7261|
|stratified|1024|0/410|0|16.1002|16.1129|
|uniform|2048|0/820|0|316.3095|571.8082|
|stratified|2048|0/820|0|14.5920|14.9717|

## Isolated uniform512 budget comparison

Historical capped versus patience-only, same data/split/seed/normalizers and
exact60-epoch prefix: anchor NRMSE0.098639→0.098067;
median final baseXY204.6487→233.6493m;
task arrivals0/26→0/26. Check these outcomes separately:
a lower offline score alone does not establish that the cap caused divergence.

## Artifacts and interpretation

Complete evidence: `runs/sampling_comparison/patience_series_20261026`.
The frozen previous study and historical cap remain untouched.

- `series_summary.json`: all six sizes, fixed26 and full-native summaries.
- `offline/offline_512.json`, `offline_1024.json`, `offline_2048.json`: complete
  command metrics, mean baseline, strict5% fidelity, zero diagnostics and tails
  outside5%, including physical mean/median/P95 errors per output.
- `command_metrics.csv`, `online_per_episode.csv`, `training_efficiency.csv`,
  `paired_mlp_outcomes.csv`: per-channel/per-case comparisons and training cost.
- `full_trajectory_metrics.json`: paired baseXY/yaw/EE trajectory errors and
  first recorded divergence/clipping; dense1ms first2s,100ms later.
- `verification.json`: exact historical prefix, patience and repeat checks.
- `target_geometry.json`, `targets.csv`, `target_layout_full_series.png` and
  `targets_3d_interactive.html`: train/validation clouds and geometric coverage.
- `offline_anchor_learning_curve.png`, `online_anchor_learning_curve.png`,
  `online_native_learning_curve.png`, `relative_5pct_learning_curve.png`,
  `training_stopping_curves.png`, `target_coverage_learning_curve.png`: all
  have standalonePDF counterparts. Four paired anchor trajectory figures show
  base paths, wrapped orientation difference and time-varying base separation.

Patience is not a proof of global convergence. Geometry coverage is not
state/history coverage. One training seed per design, validation reuse and
different update budgets limit causal or statistical claims. No memory, extra
inputs, DAgger, bounded outputs or expert intervention was added.
