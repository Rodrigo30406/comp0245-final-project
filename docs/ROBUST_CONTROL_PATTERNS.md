# Robust trajectory summaries and exploratory difficulty patterns

Completed 27 October 2026. Rodrigo requested a way to prevent a few runaway
episodes from dominating the description of otherwise useful trajectories,
while identifying conditions the policy finds difficult. This analysis reuses
completed simulations; no expert, policy, physical limit or stopping rule changed.

## Reporting rule

Compute each metric per episode, then give every episode equal weight. Report:

- **Typical behavior:** median, Q1 and Q3; IQR = Q3 − Q1.
- **Failure tail:** P90, P95, maximum, mean and invalid/missing counts separately.
- **Distribution:** empirical cumulative distribution (CDF), showing the fraction
  of episodes below each error or path ratio. Display scales may be logarithmic;
  the stored measurements remain unchanged.
- **Task outcome:** original-target EE success with the existing 1 cm/0.5 s dwell
  rule, independently of route similarity. Good geometric similarity alone is
  not successful control.

The IQR is a robust spread measure that is less affected by extremes than the
standard deviation ([NIST](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/iqrange.htm)).
We do not remove failures as statistical outliers, cap distances, winsorize them,
pool all timesteps across episodes or select only the best trajectories. Mean
and tail statistics remain visible. The contribution of the two largest errors
to the sum quantifies whether two episodes actually dominate a particular mean.
Missing/nonfinite values are counted explicitly; summaries of finite values
must be read with that count. Here all 2,460 pairs are valid and finite.

This protects the aggregate against **a few bad episodes**, not against a large
excursion within an episode: discrete Fréchet deliberately measures its worst
ordered geometric discrepancy. Timed XY RMSE provides a complementary measure.

## Coverage and definitions

Use the complete paired native validation union for each dataset size:

| Training dataset size | Goals per policy | Policies | Matched pairs |
|---|---:|---|---:|
| 1024 | 410 | Uniform, stratified | 820 |
| 2048 | 820 | Uniform, stratified | 1640 |

Both policies at a given size see identical evaluation goals and expert
references. These are the completed sklearn/CPU models, not the PyTorch pilot.
The banks change across dataset sizes; within-size comparisons are paired.
This is not the supplementary 26-goal anchor. Initial conditions remain nominal
and fixed, with the original 40 s evaluation horizon. All experts succeed;
all four learned-policy groups have **zero primary task successes**.

Route similarity is discrete Fréchet distance of base XY, in metres, using all
401 common 100 ms vertices in the original world frame, without alignment.
It preserves traversal order while allowing different speeds; it is not exact
continuous-path distance. The recurrence is from
[Eiter and Mannila (1994)](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf).
Timed XY/yaw errors retain sampling-aware time weights. Full-step path lengths
give both learner/expert length ratio and excess metres; a zero expert length
makes the ratio undefined, not zero. Full-step wheel clipping fraction is exact
for the recorded run. Normalized clipping mismatch is an approximation from
saved samples; accumulated quaternion rotation is a sampled SO(3) travel lower
bound, not an exact yaw revolution count. These diagnostics do not establish a
cause of instability. See [metric definitions](CONTROL_METRIC_PROPOSAL.md).

## Results: extremes are not the entire problem

Discrete Fréchet XY, metres; quartile range shown as Q1–Q3:

| Size | Policy | Median | Q1–Q3 | Mean | P95 | Maximum | Two largest / sum |
|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | Uniform | 242.17 | 90.82–508.95 | 335.65 | 957.27 | 1587.37 | 2.26% |
| 1024 | Stratified | 18.26 | 4.62–63.36 | 76.62 | 320.34 | 1416.28 | 8.42% |
| 2048 | Uniform | 324.05 | 131.48–578.36 | 422.12 | 1212.02 | 1987.27 | 1.14% |
| 2048 | Stratified | 16.48 | 8.39–38.04 | 60.72 | 311.57 | 1324.16 | 4.74% |

The stratified1024 mean is much larger than its median, so a mean alone gives an
incomplete picture. However, even its median route discrepancy is large and the
two largest cases contribute only 8.42% of the sum. We cannot characterize the
remaining cases as stable. Extreme distances describe simulator divergence,
not credible real-robot travel.

For example, the fraction with Fréchet ≤10 m is 12/410 uniform versus 162/410
stratified at1024, and 2/820 versus 261/820 at2048. This is a descriptive CDF
readout, **not a success tolerance**. Stratified1024 has only 12/410 within1 m;
the other three groups have none. Final base-goal distance medians are
235.69/16.10 m at1024 and316.31/14.59 m at2048 (uniform/stratified).

## Looking for difficult conditions

Define geometric bins from the configured box, before examining errors:

- Initial horizontal target radius: 0–1, 1–2 and2–3 m.
- Target height: four equal bins covering0.3–0.8 m.
- Initial navigation direction: eight equal signed sectors over−π toπ,
  relative to initial base yaw and using the expert's calculated standoff goal.

Each group shows median/IQR and its number of goals; require at least20 goals
for interpretation. The map retains every goal, with the same color scale
across policies. Additional exploratory Spearman associations cover navigation
distance, absolute required initial turn and actual nearest training-goal
distance in box-normalized XYZ. These are marginal associations, not causal
feature importance, a difficulty classifier or an independently tested pattern.

There are tentative differences in stratified1024: the0–45° navigation sector
has median Fréchet3.75 m (48 goals), compared with31.65 m for−135 to−90°
(52 goals). The highest Z quarter has median32.62 m versus11.42 m in the
preceding quarter. These differences **do not replicate clearly at2048**:
height medians are15.29–17.78 m and sector medians approximately13.22–18.21 m.
IQRs are wide; no consistent strong geometry-based difficulty rule has been
established. Confirm candidate regions with fixed held-out goals and additional
training seeds before changing sampling or claiming a general limitation.

### Translation departure can hide earlier orientation error

Use the existing first observed thresholds0.1 m XY and5° yaw separately.
They are descriptive markers, not calibrated stability boundaries.

| Size | Policy | Median first XY departure (s) | Median first yaw departure (s) |
|---|---|---:|---:|
| 1024 | Uniform | 1.601 | 0.780 |
| 1024 | Stratified | 2.600 | 0.781 |
| 2048 | Uniform | 2.300 | 0.784 |
| 2048 | Stratified | 1.975 | 0.550 |

For stratified1024, absolute initial turn correlates with later XY departure
(Spearmanρ=0.690), but almost not with the first departure in either XY or yaw
(ρ=0.035). At2048 these are0.553 and0.001. This is compatible with a longer
turn-in-place phase postponing translation; it does not demonstrate improved
orientation or that large turns cause failure. Every case here crosses both
thresholds. Future cases with no observed crossing must remain right-censored
at their horizon, not assigned an invented departure time. Trace cadence is1 ms
through2 s and100 ms afterwards; observed times are sampling-limited.

## Artifacts and reproduction

All generated files are ignored experiment artifacts in
[`runs/diagnostics/robust_control_patterns_20261027`](../runs/diagnostics/robust_control_patterns_20261027).

- [Interactive goal map](../runs/diagnostics/robust_control_patterns_20261027/goal_difficulty_interactive.html):
  rotate four3D panels, hover for individual errors/geometry and select the color
  metric. Standalone HTML with bundled JavaScript; no upload is required.
- [Full CDFs](../runs/diagnostics/robust_control_patterns_20261027/episode_error_cdfs.png),
  [direction groups](../runs/diagnostics/robust_control_patterns_20261027/bearing_groups.png),
  [height groups](../runs/diagnostics/robust_control_patterns_20261027/height_groups.png),
  [radius groups](../runs/diagnostics/robust_control_patterns_20261027/radius_groups.png),
  [3D static maps](../runs/diagnostics/robust_control_patterns_20261027/goal_difficulty_maps.png).
  Each static figure also has a PDF copy.
- `per_episode.csv`, `robust_summary.csv`, `subgroups.csv`,
  `geometry_associations.csv`, `distance_threshold_fractions.csv`,
  `departure_times.csv`, `departure_associations.csv` and
  `pose_departure_diagnostics.csv` retain individual and grouped measurements.
- `run.json`, `jobs.json`, `verification.json`, source snapshots and interactive
  provenance preserve source identities and coverage. SHA256 of `per_episode.csv`:
  `9275c6d03c4ea80325b3852d731117cc2b5052b57af911471603319822c3e741`.
  A superseded first aggregate-schema draft is separately archived and labelled;
  it is not used here. Original simulation and training artifacts are unchanged.

The maintained postprocessor generates the numeric report and static figures:

```bash
../RoboEnv/.pixi/envs/default/bin/python scripts/report_control_patterns.py \
  --counts 1024 2048 --workers 4 \
  --output-dir runs/diagnostics/robust_control_patterns_NEW_ID
```

Use a new output directory. The four workers only analyze saved traces; this is
not a new physics campaign. The optional interactive exporter and extra timing
analysis are preserved with this result, and use Plotly7.1.0 from a temporary
analysis environment. The main postprocessor introduces no dependency.

Three targeted tests cover ordered geometry, robust aggregation with extreme
failures and dataset-size/group-count separation. Artifact checks verify all
2,460 pairs, source hashes, summary medians/counts, subgroup coverage and all six
independent prototype Fréchet/path-ratio comparisons. Generalization to random
initial states, multiple training seeds and a final untouched test bank remains
unmeasured. Exact full-step rotation/mismatch logging remains future work.
