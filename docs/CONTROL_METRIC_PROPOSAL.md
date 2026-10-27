# Control quality and expert-trajectory similarity: metric proposal

Research and definitions requested by Rodrigo on27 October2026. The purpose is
to distinguish useful closed-loop behavior from policies that spin, wander or
request commands far outside the software envelope. Metrics describe behavior;
they neither improve a controller by themselves nor prove stability or a cause
of failure. No production evaluator, expert, clipping limit or stopping rule is
changed by this document. The initial six illustrative post-hoc comparisons
are preserved as examples. A subsequent full-bank postprocessor now evaluates
all410 paired goals at1024 and820 at2048 for both policies, including robust
summaries and exploratory difficulty maps. See
[completed robust control analysis](ROBUST_CONTROL_PATTERNS.md). This extends
saved-trace analysis; exact full-step observer extensions remain pending.

## What is already available

The existing evaluator records original-target EE success at1/2/5cm with0.5s
dwell, final/minimum/time-mean target distance, base navigation error/arrival,
full-step path length, terminal motion and full-step per-command clipping counts.
`evaluation/trajectory.py` also computes time-weighted XY, wrapped yaw, baseZ and
whole-system EE separation from the expert, RMSE/max/final errors, and first
observed separation/clipping. Additions must complement these, not rename them.

Time weighting is essential: traces contain every1ms sample for the first2s,
then every100ms. A plain row mean would over-weight the first2s. Expert and learner
start from the same physical condition and act in their own simulators. Their
later command differences are not supervised errors on a common state.

## Recommended additions

Start with two new trajectory measures: **discrete Frechet distance** and
**path-length ratio to the expert**. Add cumulative rotation and clipping severity
as diagnostics of the spinning/saturation failures. Preserve target success,
timed trajectory errors and departure times beside them. Do not combine them
into an arbitrary weighted score.

### 1. Ordered geometric similarity: discrete Frechet distance

Compare the baseXY curves in their original world coordinates. It matches their
progress in order while allowing different speeds; the reported distance is the
smallest possible worst separation over these monotone matchings, in metres.
Keep timestamp-aligned RMSE beside it: a delayed correct path and a wrong path
should not be indistinguishable. The discrete approximation depends on saved
vertices; identical continuous routes with different sampling need not score
exactly zero. Source: [Eiter and Mannila,1994](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf).

Use a common100ms grid for historical40s traces:401points, explicitly named
`discrete_frechet_xy_100ms_m`. This is a metric of sampled trajectories, not an
exact continuous-path distance. Record maximum spatial/time sampling gaps.
No rigid alignment, translation, rotation, rescaling or trajectory truncation is
allowed to hide errors: both simulations already share a frame and initial pose.
For a future3D whole-system comparison use EE world coordinates separately;
this is not an isolated arm pose score. [CGAL's explanation of geometric Frechet
comparison](https://doc.cgal.org/latest/Frechet_distance/index.html).

A trace that wanders hundreds of metres gets a large distance even if it briefly
returns near the goal. A stationary or spinning-in-place robot can still have a
small geometric error in some cases, so this never replaces task/orientation
metrics. Benchmark software should retain the original timestamps and fixed
horizon even though this particular distance does not enforce time alignment.

### 2. Unnecessary travel: path length relative to the expert

Let L_M and L_E be model and expert baseXY path lengths over the same full40s.
Report both raw lengths and:

`R_L = L_M / L_E`, and `excess_path_length_m = L_M - L_E`.

A ratio of5 means five times the expert's travelled distance. Values below1 do
not automatically mean improvement: a robot that never moves has zero travel and
can fail completely. Interpret jointly with goal error and success. If L_E is
zero, report the ratio as undefined and retain absolute distances; do not invent
a small denominator that makes the score look meaningful.

Path efficiency has established navigation-evaluation precedent, but this is an
**expert-referenced diagnostic, not standard SPL**. The expert is not claimed to
be shortest-path optimal. SPL is success-weighted and therefore collapses to zero
for our currently unsuccessful policies; it would not distinguish these failures.
[Anderson et al.,2018](https://arxiv.org/pdf/1807.06757).

Our full-step `subsystems.base_navigation.path_length_xy_m` already supplies both
lengths. Forming their ratio requires no new simulation, and avoids estimating
length from sparse saved positions, which can miss loops between samples.

### 3. Spinning and oscillation: accumulated rotation

A wrapped final bearing error can be near zero after three full turns. Add the
accumulated absolute yaw increments at every control step for planar motion:

`A_yaw = sum(abs(wrap(yaw[k] - yaw[k-1])))`, in radians.

`A_yaw / (2*pi)` expresses equivalent angular travel, not net rotation. Both
back-and-forth oscillation and repeated turns contribute. Report signed winding
separately if direction matters. Compare the expert under the same protocol;
turning is not inherently undesirable if the expert must reorient.

This is a proposed diagnostic, not a standard stability certificate. For large
roll/pitch or flips, yaw is insufficient: use full quaternion angular travel and
label it as3D rotational motion rather than planar turns. Quaternion increments
can use `2*acos(abs(dot(q[k],q[k-1])))` after unit normalization.

Historical100ms tails cannot recover exact turn counts: a robot may complete
extra rotations between saved poses. The demonstration reports only sampledSO(3)
travel, a lower bound on unobserved rotational travel. Exact per-control-step
aggregation needs new logger instrumentation/new runs; do not relabel this lower
bound as the true number of yaw turns. No numerical differentiation of sparse
positions is needed or recommended here.

### 4. Saturation severity: how far raw requests exceed applied commands

We already have exact full-step clipping fractions and first recorded clipping.
Frequency alone cannot distinguish requesting5.4 versus500rad/s when the
software envelope is5.273rad/s. For channel j propose:

`C_j = (1/T) integral( abs(u_raw_j - u_applied_j) / limit_j dt )`.

It is dimensionless: a value1 means the time-average command mismatch equals
that channel's software limit. Report right/left and each joint, plus separate
base and arm means; count the two wheel sides once. Also retain raw peak requests,
which the evaluator already stores. This is a **project-defined normalized
clipping-mismatch diagnostic**, not an established universal score or measured
motor-energy quantity. Literature treats saturation mismatch as relevant to
tracking performance, but this exact formula is our adaptation.
[Original saturation/tracking study](https://doi.org/10.1016/j.aej.2014.06.004).

For future runs accumulate dt-weighted mismatch at every1ms control decision,
and record the longest continuous clipped burst in seconds per channel. Do not
increase bounds to improve scores. Applied commands are the clipped commands
sent by the policy wrapper, not independently measured actuator torques.

For old traces this integral can only be estimated from saved samples, with
proper time weights and explicit sampling/coverage labels. Clipping frequency
remains exact from full-step counters. Observing high clipping does not prove it
initiated the deviation; earlier dense audits found trajectory departures before
clipping in their tested cases.

## Additional documented alternatives

- **Relative Pose Error at a fixed interval:** compare the expert and model's
  incremental motion over100ms or1s. Translation and rotation errors remain
  separate. This can reveal whether local turning/motion is wrong before large
  absolute position error accumulates. Adapt the pose formulation carefully;
  SLAM alignment conventions must not hide control errors in our shared frame.
  [Sturm et al.,2012](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf).
- **DTW/nDTW:** useful ordered path-fidelity alternatives with flexible timing.
  nDTW needs a declared distance scale and resampling protocol; its navigation
  benchmark defaults should not be copied blindly into this mobile-manipulator
  task. It cannot replace timed tracking or goal success, and aggressive warping
  may conceal delays. Use it as an alternative to Frechet initially, rather than
  adding many overlapping scores. [Ilharco et al.,2019](https://arxiv.org/html/1907.05446v2).
- **IAE/ITAE and control total variation:** integral tracking error and command
  variation are established control-performance tools. Our time-mean target
  distance already equals IAE/T for that scalar distance; adding it under another
  name contributes little. ITAE would emphasize late error; total variation can
  flag chatter, but must be computed per channel at a fixed rate and must not be
  called physical energy. Neither a still robot nor a smooth wrong trajectory is
  good control by itself. [Bacci di Capaci and Scali,2017](https://arpi.unipi.it/handle/11568/903087).

## Proposed compact dashboard

|Question|Metric|Units|Availability|
|---|---|---|---|
|Does it reach and stay at the actual task goal?|Existing task success, final/meanEE distance, arrival+dwell|%,m,s|Already full-step|
|Does it follow the expert at the right time?|Existing time-weighted baseXY/yaw/EE trajectory errors|m,rad|Already from stored traces|
|When does it depart?|Existing threshold-crossing times; optional future sustained-departure criterion|s|Sampling intervals/censoring retained|
|Does it follow a similar ordered route?|New discrete Frechet distance|m|Historical traces, explicit100ms discretization|
|Does it travel excessively?|New expert path ratio and excess distance|ratio,m|Existing full-step path lengths|
|Does it spin or oscillate?|New accumulated rotation|rad,equivalent turns|Only a historical sampled lower bound; full-step aggregation pending|
|How often/how severely does clipping occur?|Existing full-step fractions + proposed normalized mismatch/burst duration|%,ratio,s|Frequency exact; historical severity approximate; full-step extensions pending|

If a trajectory never departs, mark its departure time as right-censored at40s,
not a measured failure at40s. Invalid/truncated runs remain explicit failures;
do not reward them with short low-error averages. Aggregate per episode over the
entire intended holdout bank, report medians/tails and valid counts, and retain
per-case curves. No tolerances have been selected after viewing these examples.

## Illustrative measurements from existing1024 trajectories

The prototype chooses the same three original uniform anchor conditions by
sorted scenario IDs at indices0,6,12, without selecting by outcomes. Each has an
expert, uniform1024 and stratified1024 full40s run. It checks stored trace hashes
and four analytic Frechet examples (identity, translation, pauses and ordering).
These six comparisons illustrate interpretation; they are **not full-validation
results** and must not be used to select a winning model.

|Scenario prefix|Model|Discrete FrechetXY(m)|Full-step path ratio|Exact wheel clipping(%)|Sampled normalized clipping excess, wheels|
|---|---|---:|---:|---:|---:|
|18273f3b|mlp_uniform1024|383.931|276.39|74.88|1893.82|
|18273f3b|mlp_stratified1024|5.182|9.93|50.39|5.33|
|6ce6e240|mlp_uniform1024|188.108|317.95|78.80|346.61|
|6ce6e240|mlp_stratified1024|59.561|67.41|54.92|26.10|
|e71c81f4|mlp_uniform1024|196.335|221.97|66.75|384.76|
|e71c81f4|mlp_stratified1024|20.339|40.18|74.05|65.34|

The last column is a time-weighted interpolation of saved raw/applied samples,
not an exact per-step integral. It must retain its sampled label. Extremely
large values describe raw policy requests far beyond the software envelope,
not physically applied velocities. These trajectories are simulated divergence.
In the first illustrative scenario, the model travels276.39times the expert
length for uniform and9.93times for stratified; both still fail the task.

Raw [CSV](../runs/diagnostics/control_metric_research_20261027/examples.csv),
[JSON with trace hashes and definitions](../runs/diagnostics/control_metric_research_20261027/examples.json),
and [prototype](../runs/diagnostics/control_metric_research_20261027/demo.py).
The prototype has no production-side integration and changes no frozen artifact.

## Proposed next implementation boundary

1. Add a tested postprocessor for Frechet and full-step path ratios to all paired
   complete validation traces, retaining bank IDs/counts, fixed grids and hashes.
2. Extend the evaluation observer (not the expert/control loop) to aggregate
   full-step rotation travel, normalized clip mismatch and burst duration on
   future runs. Verify exact aggregation with controlled synthetic sequences.
3. Keep historical approximate quantities separate; do not silently compare them
   against exact new counters or infer unrecorded turns between sparse samples.
4. Evaluate candidate windows/architectures using the same frozen full holdouts,
   original40s horizon and arrival rules. Compare per-episode outcomes, medians
   and tails; no new4k-rollout campaign is launched by this research request.
