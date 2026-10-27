# Validation geometry and the interpretation of sampling comparisons

Requested by Rodrigo on 27 October 2026. Own-validation results can change when
the holdout bank changes. In the current supervised own-validation tables,
stratified is not always lower: uniform has lower mean arm torque RMSE at all six
sizes and lower wheel RMSE except at128. This statement concerns command error;
the earlier lower simulated divergence of stratified policies is a different
metric and does not imply stable control.

## Actual stratified holdout rule

The collector assigns one jittered target to each grid cell. Before outcomes or
target jitter are observed, holdout selection balances axis levels, represents
all eight octants and retains training coverage in adjacent vertical cell pairs.
Candidate holdout subsets are selected by normalized cell-center separation
between held-out points, not by model loss or measured distance to training.
No target or trajectory is copied from training into validation.

This design encourages coverage of the same bounded domain. It evaluates
interpolation to new goals inside that domain, not guaranteed extrapolation into
unseen regions. Geometry constraints may affect difficulty, but they do not
establish that its validation goals are closer to training than uniform goals.

## Measured holdout-to-training distances

For every actual held-out goal, find the nearest target in that model's own
training split. Distances use the physical XYZ coordinates; means and P95 are
in metres. The reference data and split hashes are retained with the diagnostic.

|Episodes|Uniform mean(m)|Stratified mean(m)|Uniform P95(m)|Stratified P95(m)|
|---:|---:|---:|---:|---:|
|64|0.320403|0.385508|0.529637|0.562360|
|128|0.265888|0.291074|0.399360|0.428563|
|256|0.209383|0.222978|0.333389|0.339788|
|512|0.161860|0.161606|0.283630|0.250684|
|1024|0.123097|0.132891|0.218841|0.204749|
|2048|0.096350|0.105340|0.157823|0.158306|

At2048, mean nearest-training distance is0.09635m uniform versus0.10534m
stratified. With each axis normalized by its box extent, the corresponding means
are0.04801 versus0.05485. Thus the specific claim that stratified holdouts are
closer *on average* is not supported by this geometry measurement. At1024,
stratified has a lower physical P95/maximum despite a larger mean: distribution
shape matters, not only an average. No exact target duplicates are observed.

XYZ proximity is not21-dimensional state/history proximity or a measure of
control difficulty. Different targets can produce different phases, velocities,
command magnitudes and trajectories. These distances do not isolate the cause
of command-error or online-control differences, nor prove equal bank difficulty.

For fair within-size model comparison use the same full paired union. Across
sizes the native union still changes; the26-goal anchor fixes conditions but has
limited coverage. A future larger shared holdout or spatial-region holdout would
address other questions; no such experiment was run by this diagnostic.

See [complete supervised validation](SUPERVISED_FULL_VALIDATION.md) and
[the sampling design](SAMPLING_SERIES.md). Raw measured distances:
[CSV](../runs/diagnostics/validation_geometry_20261027/distances.csv) and
[JSON with split checksums](../runs/diagnostics/validation_geometry_20261027/metrics.json).
