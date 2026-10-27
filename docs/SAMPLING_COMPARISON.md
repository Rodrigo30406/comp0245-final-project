# Stratified versus uniform sampling: 64-episode pilot

Completed on 26 October 2026. Corrected planned series: **64/128/256/512**.
This report records the original64 pilot only. The later complete
[64/128/256/512 series](SAMPLING_SERIES.md) now supplies the larger cohorts. One realization per design, not multi-seed evidence.

## Design and fairness

Same box X/Y [-2,2] m, Z [0.3,0.8] m, nominal start, expert, limits and MLP/OLS
recipes. New design: one uniformly jittered goal per 4x4x4 cell; geometry-only
13-cell validation selection balances axis levels (3/4 each) and all octants.
Every coarse 4x4x2 cell retains a training target. Cell selection uses normalized
coordinates, before target jitter and outcomes. 51 training / 13 validation.

The original uniform64 models remain frozen. Each policy is evaluated on both
complete validation banks and the same26-condition union. All selected physical
conditions are checked absent from every model training set. Each model used its
own native validation for checkpoint selection; this is comparative development
validation, not a pristine independent test. No memory/window or bounded-output
change was mixed into the sampling experiment.

[Target layout](../runs/sampling_comparison/stratified64_20261026/target_layout.png).

## Coverage

Mean distances from100,000 shared uniform probe locations to nearest target:

| Design | Occupied cells (all64, out of64) | Training mean nearest distance (cm) | Validation mean nearest distance (cm) | Validation octants |
|---|---:|---:|---:|---:|
| uniform13 | 41 | 33.54 | 63.73 | 6/8 |
| stratified13 | 64 | 33.58 | 55.13 | 8/8 |

Coverage here describes target XYZ, not21D states or recovery behavior.

## Offline commands, fixed original output scales

[Comparison figure](../runs/sampling_comparison/stratified64_20261026/offline_comparison.png).

| Model | Original uniform13 NRMSE | New stratified13 NRMSE | Common26 NRMSE |
|---|---:|---:|---:|
| linear_stratified64 | 0.6066 | 0.6627 | 0.6347 |
| linear_uniform64 | 0.5975 | 0.6459 | 0.6217 |
| mlp_stratified64 | 0.2636 | 0.3224 | 0.2930 |
| mlp_uniform64 | 0.2031 | 0.3560 | 0.2795 |

Primary command fidelity remains abs(pred-expert)<=5%*abs(expert), no additive
floor, with numerical zeros<=1e-6 evaluated separately. Offline JSON includes
all nine outputs, physical MAE/RMSE, relative/zero/stop profiles and native reports.

## Autonomous control

130/130 rollouts completed: expert and four policies on both13-case banks,40s
each. 0.5s dwell, original action envelopes, dense first2s; all valid full horizon.
Finite simulation does not certify physical plausibility.

[Online comparison](../runs/sampling_comparison/stratified64_20261026/online_comparison.png).

| Policy | Bank | Task arrivals | Base arrivals | Median final base XY (m) | Median final EE (m) |
|---|---|---:|---:|---:|---:|
| expert | uniform13 | 13/13 | 13/13 | 0.0453 | 0.0016 |
| expert | stratified13 | 13/13 | 11/13 | 0.1060 | 0.0014 |
| expert | common26 | 26/26 | 24/26 | 0.0734 | 0.0014 |
| linear_stratified64 | uniform13 | 0/13 | 0/13 | 3.4785 | 3.6871 |
| linear_stratified64 | stratified13 | 0/13 | 0/13 | 2.9199 | 3.3122 |
| linear_stratified64 | common26 | 0/26 | 0/26 | 3.2996 | 3.4181 |
| linear_uniform64 | uniform13 | 0/13 | 0/13 | 1.9058 | 2.4417 |
| linear_uniform64 | stratified13 | 0/13 | 0/13 | 2.1621 | 2.5815 |
| linear_uniform64 | common26 | 0/26 | 0/26 | 2.0545 | 2.4912 |
| mlp_stratified64 | uniform13 | 0/13 | 0/13 | 2.0235 | 2.7717 |
| mlp_stratified64 | stratified13 | 0/13 | 0/13 | 2.7498 | 1.9366 |
| mlp_stratified64 | common26 | 0/26 | 0/26 | 2.3867 | 2.7069 |
| mlp_uniform64 | uniform13 | 0/13 | 0/13 | 563.1744 | 926.5750 |
| mlp_uniform64 | stratified13 | 0/13 | 0/13 | 179.6286 | 646.1251 |
| mlp_uniform64 | common26 | 0/26 | 0/26 | 203.5631 | 899.7475 |

Interpret the common26 within this experiment; do not compare its medians directly
to the old13-bank medians. Different validation goals and checkpoint selection
matter. More evenly distributed targets do not by themselves guarantee better
prediction or stable control. One seed/design cannot establish a universal winner.

## Interpretation of this pilot

The stratified layout occupies all64 fine cells versus41 for the uniform layout;
validation covers8 octants versus6. Physical mean nearest-training-target distance
is practically unchanged (33.58 versus33.54 cm), while validation coverage improves.
Do not equate balanced cell occupancy with universally smaller Euclidean gaps.

On the common26 bank, MLP stratified NRMSE is0.2930 versus0.2795 for uniform,
so this pilot does not improve overall offline prediction. It improves on its
new stratified13 bank but worsens on the old uniform13 bank. Online median base
error decreases203.56→2.39m and median EE error899.75→2.71m, with wheel clipping
92.998%→31.427% and torque clipping82.339%→15.012%. Nevertheless, both MLPs have
0/26 whole-system and base arrivals, while the expert reaches26/26 task targets.
Lower divergence is not successful control.

The selected stratified checkpoint is epoch3 versus21 for uniform. Identical
patience/epoch-cap rules do not equalize optimizer updates or selected training
age. This result compares the practical recipes; sampling geometry, selected
checkpoint, per-episode duration/weight and one-seed variability are not causally
separated. A fixed-epoch/update control would be a subsequent experiment.

## Training stopping

[Validation curves](../runs/sampling_comparison/stratified64_20261026/training_stopping.png).

| Model | Epochs completed | Best epoch | Stop reason |
|---|---:|---:|---|
| Uniform64 | 22 | 21 | patience |
| Stratified64 | 11 | 3 | patience |
| Uniform128 | 17 | 9 | patience |
| Uniform512 | 60 | 58 | max_epochs |

The original uniform512 model hit max_epochs=60, best58, so convergence is
unproven. The new sampling pilot deliberately retains that recipe. Patience is8
epochs without a meaningful validation improvement>=1e-4; a tiny best-checkpoint
improvement need not reset patience. See [stopping audit](TRAINING_STOPPING.md).

## Evidence and reproduction

Artifacts: runs/sampling_comparison/stratified64_20261026. Sampling plan, explicit
targets/split, dataset, old/new frozen model hashes, configs/logs, offline per-bank
and common26 metrics,130 traces/results, geometric coverage and four PNG/PDF
figure families are retained. Old uniform policy/expert physical metrics on
the original13 exactly reproduce the previous run. No frozen data was overwritten.

```bash
python scripts/compare_sampling.py \
  --output-dir runs/sampling_comparison/NEW_COMPARISON \
  --workers 16
```

Run inside RoboEnv. Existing output directories are rejected. Extended training
budget and other episode counts are separate experiments, not executed here.
