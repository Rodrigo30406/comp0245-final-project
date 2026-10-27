# MLP depth and expanding/contracting hidden layers

Rodrigo requests additional hidden layers and a32→64→128 progression on27 October
2026. This is a separate architecture screen after the temporal-window comparison.
Current code supports any nonempty list of positive hidden-layer widths, retaining
ReLU in every hidden layer and a linear nine-command output. The loss remains
expert command regression. An expanding/contracting width profile is not itself
an autoencoder or an input-reconstruction objective.

## Predeclared candidates

Use both instantaneous21D observations and the agreed200ms/20ms measured stack
(eleven18D states plus current3D goal once,201D). Do not vary temporal settings
while testing these architectures.

| Shape | Hidden widths | Hidden depth | Parameters,21D | Parameters,201D |
|---|---|---:|---:|---:|
| Reference |128,128|2|20489|43529|
| Increasing |32,64,128|3|12297|18057|
| Expand/contract |32,64,128,64,32|5|21769|27529|

Four new fits: `instant_pyramid`, `window200ms_pyramid`,
`instant_expand_contract`, `window200ms_expand_contract`. Reuse the existing
PyTorch `instant128` and `window200ms` baselines, yielding six policies in the
final comparison. All figures and tables must report input representation,
hidden widths, depth and parameter count together. This screen changes width,
depth and parameter budget; it is not a pure causal depth-only comparison.
In particular,201D→32 is already compression at the first hidden layer.

## Controls and evaluation

Reuse the exact stratified1024 dataset,819/205 episode split, frozen training-only
normalizers, seed20261025, Adam/objective convention, batch1024, alpha0.0001 and
learning rate0.001. Use PyTorch/CUDA FP32, no AMP/TF32, patience8/min_delta0.0001
and no epoch cap. Select the minimum equal-episode balanced validation NRMSE.
The learned action remains two wheel-side commands plus seven joint torques.

No expert behavior, physics, clipping, target projection, standoff or stopping
rule changes. Do not add dropout, BatchNorm, bounded outputs, extra sensors or
DAgger to these variants. Keep all failures. Changes in wall time during another
active campaign are not a controlled speed comparison.

Train new models sequentially on the GPU. Before starting new physics processes,
wait for the original temporal campaign to finish; use16 simulation workers.
Evaluate all205 own and205 external uniform validation goals,410 per policy,
with the existing40s protocol and1cm/0.5s dwell success. Reuse the two baselines'
saved full-bank trajectories rather than simulating those models again. New
architecture evaluation also records expert references through the existing
evaluator. Per-policy goal sets must match exactly before aggregation.

Report supervised wheel/joint errors, strict5% nonzero fidelity and zero-command
diagnostics, plus target success, base/system pose errors, yaw, departures,
clipping and stopping. Use per-episode Fréchet, timed XY RMSE and path ratio CDFs,
median/IQR and tail statistics on the full paired bank. A lower validation loss
does not establish better closed-loop control. One seed and repeatedly inspected
development holdouts limit the strength of architectural conclusions.

## Pipeline and provenance

`MLPPolicy` validates all consecutive indexed weight/bias arrays and infers the
layer count when loading NPZ, preserving historical two-layer artifacts. The
Torch builder, initialization, regularization, checkpoint export, and NumPy
inference all iterate over the configured depth. The sklearn path also supports
variable-depth checkpoint reconstruction. Existing window encoding/reset rules
remain shared. Simulator workers continue to run without a Torch dependency.

Launcher:

```bash
.venv/bin/python scripts/compare_mlp_architectures.py \
  --reference-campaign runs/temporal_comparison/stratified1024_window200ms_20261027 \
  --output-dir runs/architecture_comparison/stratified1024_depth_shape_20261027 \
  --workers 16
```

The output must be new. A shared launcher freezes source, scripts, recipes and
package metadata so active jobs are isolated from workspace changes. Each fit
saves configuration, split/scalers, optimizer recovery state, history, selected
model, timing/memory and complete metrics. Campaign state records whether it is
training, waiting for the reference, evaluating or reporting. Source data and
historical baselines are not rewritten.

Expected artifacts:
`runs/architecture_comparison/stratified1024_depth_shape_20261027`, ignored by Git;
coordinator log `/tmp/mlp_depth_shape_20261027.log`. Read `run.json` and each
`logs/train_<variant>.log` for actual execution status. Results are pending.

## Verification

Three new tests pass: arbitrary-depth NumPy/Torch agreement for both selected
shapes and both input representations; preservation of negative linear outputs;
full deep-layer save/load and malformed index rejection; live/window batch
agreement; actual deep sklearn and Torch fitting with exact interrupted/resumed
weights and training-only scalers; and actual CUDA deep200ms fitting/export with
patience. All57 repository tests pass before campaign launch. The new loader
also exactly reproduces frozen legacy predictions for instant128/window200ms
on600 actual held-out observations each. Source compilation passes.
No experimental improvement is claimed from these
implementation tests.
