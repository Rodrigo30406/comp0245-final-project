# Full stratified-versus-uniform series: results

Completed on 26 October 2026. Authorized counts **64/128/256/512**. All new
data/model/evaluation artifacts retain their actual sizes; the corrected uniform256
cohort was collected and trained, never relabelled from a historical experiment.

## Frozen design

| Episodes | Training | Validation | Stratified cells |
|---:|---:|---:|---|
| 64 | 51 | 13 | 4x4x4 |
| 128 | 102 | 26 | 8x4x4 |
| 256 | 204 | 52 | 8x8x4 |
| 512 | 409 | 103 | 8x8x8 |

Unchanged box X/Y [-2,2] m, Z [0.3,0.8] m, nominal start, expert, physical
period, limits, 21 inputs/9 outputs, OLS and two-layer MLP. Original max60 epochs,
patience8,min_delta1e-4; scalers fitted training only. Completed pilot64 and frozen
uniform64/128/512 models reused. New uniform256 extends the128 split to204/52;
its holdout is not assumed nested in frozen512. Uniform first128 NPZ/scenario
hashes reproduce exactly. Larger stratified layouts are independent jittered grids,
not nested prefixes. Their geometry-only holdouts balance each axis, include all
octants and retain training points in every adjacent vertical pair.
No simulator outcome entered selection; plans were saved before collection.

Primary comparison uses the identical26 anchor conditions (original uniform13
plus stratified-pilot13), checked absent from every training set. Every size also
gets both complete native validation banks, evaluated by both sampling designs
and both regressors, plus expert. Native unions have26/52/104/206 goals and differ
across sizes; compare designs within a size, not unmatched banks as data-only evidence.

One realization per design and development validation, not a reserved final test
or multi-seed confirmation. Same epoch recipe does not equalize optimizer updates,
selected epoch or episode-duration weighting. No memory/DAgger/limits/budget
intervention is mixed into this study.

## MLP comparison on the same 26 anchor conditions

| Episodes | Uniform NRMSE | Stratified NRMSE | Uniform task arrivals | Stratified task arrivals | Uniform median base XY (m) | Stratified median base XY (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.2795 | 0.2930 | 0/26 | 0/26 | 203.5631 | 2.3867 |
| 128 | 0.2581 | 0.2027 | 0/26 | 0/26 | 84.2155 | 7.8624 |
| 256 | 0.1599 | 0.1496 | 0/26 | 0/26 | 3.5482 | 2.9396 |
| 512 | 0.0986 | 0.1143 | 0/26 | 0/26 | 204.6487 | 10.6650 |

NRMSE uses fixed original uniform64 output scales and equal episode weights.
Primary command criterion remains abs(pred-expert)<=0.05*abs(expert), no additive
floor; abs(expert)<=1e-6 is excluded from relative denominators and evaluated
with absolute mean/P95 separately. Command fractions pool nonzero transitions.
Complete per-wheel/joint MAE/RMSE, 5%/zero metrics and absolute tails are preserved.

## MLP full native unions

| Episodes | Goals per native union | Uniform task arrivals | Stratified task arrivals | Uniform median EE error (m) | Stratified median EE error (m) | Stratified base XY better (paired) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 26 | 0/26 | 0/26 | 899.7475 | 2.7069 | 26/26 |
| 128 | 52 | 0/52 | 0/52 | 100.0616 | 7.5351 | 47/52 |
| 256 | 104 | 0/104 | 0/104 | 4.4255 | 3.3926 | 56/104 |
| 512 | 206 | 0/206 | 0/206 | 501.2948 | 12.1721 | 189/206 |

The expert reference is evaluated on the same banks. Full-horizon finite results
are not physical validity; extreme distances represent simulated divergence.
Base goal is the expert standoff strategy, distinct from EE-to-original-target
task error. A brief0.5s arrival does not certify holding position until40s.
Invalid rollouts, if any, remain in denominators and receive false success;
physical medians report valid trajectories only, with valid counts in JSON.

## Interpretation and limits

On the identical26-anchor bank, stratified sampling has lower offline NRMSE at
128 and256, while uniform has lower NRMSE at64 and512. There is no universal
offline winner in this realization. Stratified MLPs have lower median final base
error at every size, yet all MLPs still have0/26 task arrivals; the expert has26/26.
The full native unions also have zero MLP task arrivals at every size, including
0/206 for each512 policy versus206/206 for the expert. These are not usable
controllers. Larger stratified datasets improve offline error without monotonic
control improvement: median base error is2.39/7.86/2.94/10.66m over64/128/256/512.
Extreme values are simulated divergence, not physically plausible travel.

This compares complete practical recipes, not target geometry in isolation.
Selected epochs, trajectory lengths and one-realization variability differ;
more data under the same epoch cap also means more updates. The findings do not
prove a unique failure cause or that stratification alone fixes feedback stability.
Time/history inputs and the unresolved DAgger expert-memory protocol remain
separate research questions, not implemented variants in this series.

## Training stopping: no convergence claim at a cap

| Design | Episodes | Epochs attempted | Selected epoch | Stop reason |
|---|---:|---:|---:|---|
| uniform | 64 | 22 | 21 | patience |
| uniform | 128 | 17 | 9 | patience |
| uniform | 256 | 24 | 16 | patience |
| uniform | 512 | 60 | 58 | max_epochs |
| stratified | 64 | 11 | 3 | patience |
| stratified | 128 | 19 | 11 | patience |
| stratified | 256 | 45 | 37 | patience |
| stratified | 512 | 35 | 27 | patience |

Any max_epochs stop means the run reached its imposed budget, not that validation
ceased improving. Extending the budget is a separate experiment. Full efficiency
metadata includes parameters, model bytes, updates, fit time and inference time.
Timings describe local runs and reused historical measurements, not a controlled
hardware-scaling benchmark. Conditional tail exports also include mean-predictor
MAE/RMSE references on the same banks with the same fixed normalization.

## Figures and numerical exports

- [All target clouds](../runs/sampling_comparison/full_series_20261026/target_layout_full_series.png),
  with explicitly labelled height expansion; targets.csv preserves actual coordinates.
- [Interactive target viewer](../runs/sampling_comparison/full_series_20261026/targets_3d_interactive.html), self-contained HTML.
- [Target-space coverage](../runs/sampling_comparison/full_series_20261026/target_coverage_learning_curve.png).
- [Offline anchor curves](../runs/sampling_comparison/full_series_20261026/offline_anchor_learning_curve.png).
- [5% relative command fidelity](../runs/sampling_comparison/full_series_20261026/relative_5pct_learning_curve.png).
- [Online anchor metrics](../runs/sampling_comparison/full_series_20261026/online_anchor_learning_curve.png).
- [Full native online curves](../runs/sampling_comparison/full_series_20261026/online_native_learning_curve.png).
- [Paired base outcomes](../runs/sampling_comparison/full_series_20261026/paired_base_outcomes.png).
- [All stopping curves](../runs/sampling_comparison/full_series_20261026/training_stopping_full_series.png).

Artifacts: runs/sampling_comparison/full_series_20261026.
series_summary.json, online_per_episode.csv, command_metrics.csv,
training_efficiency.csv, paired_mlp_outcomes.csv, full_trajectory_metrics.json,
anchor_trajectory_metrics.json (common-bank subset), conditional_5pct_tails.json,
target_geometry.json, dataset manifests, plans/splits/configs, checkpoints, raw
evaluation traces and logs preserve the evidence. Reference outcomes are reused
only with explicit source/condition checks. Repeated experts are integrity checks,
not independent training replicates. The64 pilot remains unchanged.

```bash
python scripts/compare_sampling_series.py \
  --output-dir runs/sampling_comparison/NEW_FULL_SERIES \
  --workers 16
```

Run with RoboEnv; optional interactive exporter uses a temporary Plotly installation,
not a change to project dependencies. Existing output directories are rejected.
