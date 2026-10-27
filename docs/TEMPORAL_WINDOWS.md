# First temporal-window ablation

**Current design — agreed 27 October 2026: 200 ms duration, 20 ms spacing.**
Rodrigo corrected a brief100ms acceptance to200ms before relaunch. The original
100ms campaign remains interrupted and preserved. The new200ms campaign recovers
the unchanged instantaneous baseline from its hash-verified epoch21 checkpoint.
Temporal training and control results remain pending.

Authorized and launched 27 October 2026 after Rodrigo selected a fixed1024-episode
budget. This begins the next P3 representation experiment. Implementation and
regression checks pass; training and autonomous results must be read from the
campaign manifests before claiming any improvement.

## Question and controlled comparison

Does a short causal measurement history improve closed-loop imitation, beyond
the effect of adding parameters? Use the completed **stratified1024** dataset,
its exact819 training /205 validation episodes and seed20261025. Keep uniform1024
as the independent additional205-goal validation bank. Both banks remain outside
training; together they provide410 paired goals per policy. This is development
validation, not an untouched final test set.

| Variant | Encoded inputs | Hidden layers | Parameters |
|---|---:|---|---:|
| `instant128` | 21 | 128,128 | 20489 |
| `window200ms` | 201 | 128,128 | 43529 |
| `instant193` | 21 | 193,193 | 43434 |

All three train with PyTorch/CUDA, FP32, the existing Adam/objective convention,
learning rate0.001, alpha0.0001, batch1024, patience8, min_delta0.0001 and no epoch
cap. The selected checkpoint minimizes equal-episode balanced validation NRMSE.
The historical1024 reference is sklearn; retraining the instantaneous baseline
on the same backend prevents a backend change from being mistaken for memory.
No new expert demonstrations or additional sensors are collected.

## Exact window semantics

At physical step t, concatenate the measured18D states at
`t−200, t−180, t−160, t−140, t−120, t−100, t−80, t−60, t−40, t−20, t`,
followed by the **current original3D target**
once. With1ms physics these eleven states cover200ms at20ms spacing. Each18D state consists of
baseXYZ/bearing, seven arm joint positions and seven joint velocities. Commands
are still predicted and applied at every1ms physical step.

History contains current and earlier measurements only. Expert phase,
`navigation_done`, desired references, expert actions and future states remain
excluded. Missing initial history repeats the first measured state. The online
buffer resets per episode and rejects a simulator timestep different from the
stored1ms setting. Its201-slot ring stores only18D measured states.

Offline data must include dense consecutive steps starting at zero, compatible
times and metadata dt; sparse logs are rejected rather than misinterpreted as
1ms samples. The episode split precedes window construction. Every shuffled
training row gathers history from its own episode; clamping at that episode's
first row prevents windows crossing boundaries.

Reuse the **exact frozen21D current-observation training scaler** at every lag,
and normalize the goal once. This is intentional shared normalization, not a
new independently fitted scaler for each lag. Target/output normalization is
identical across all three models. No validation observations fit statistics.
The21D raw and normalized banks remain disk-backed/device-resident as before;
201D windows are expanded only for the current minibatch. Overlapping windows
are not permanently materialized eleven times in RAM, VRAM or on disk.

## Deployment and evaluation

The portable NPZ contains window metadata as well as weights/scalers. Existing
21D models still load without this metadata. Simulation workers use NumPy and
need no Torch/CUDA. `MLPPolicy.predict` on a matrix treats it as **one complete,
dense episode**, starting with a fresh padded history; use separate calls for
separate episodes. For live control, `predict_action` appends one observation per
physical step, and `reset(dt)` initializes each new rollout. Do not call the live
method twice for the same physical step or on sparse observations.

Evaluate all205 own holdouts offline by wheel/joint MAE/RMSE, strict5% nonzero
command fidelity and separate zero-command diagnostics. Train metrics include
the complete819 training episodes. Evaluate all410 paired native goals online
for expert plus three models, using16 simulation workers and the existing40s
horizon, clipping limits and original-target1cm/0.5s dwell success.

Report base/whole-system target error, yaw, departure times, stopping and clipping.
The final robust report adds per-episode Fréchet, timed XY RMSE and expert path
ratio with median/IQR plus tail statistics and CDFs; no metric outlier removal.
Four analysis workers process saved traces after simulation, independently of
the16 physics workers. Input history update is included in inference latency;
parameter counts and GPU peak memory remain recorded. A200ms stack does not
guarantee recovery of the expert's integrated references or25s timeout state.

## Execution and artifacts

Launcher: `scripts/compare_temporal_windows.py`. It freezes the source, scripts,
recipes and package metadata before running subprocesses, preserving a separate
code tree for the campaign. Work in the main checkout can continue without
invalidating its live source hashes. Original datasets, experts and historical
models are unchanged.

Current output:
`runs/temporal_comparison/stratified1024_window200ms_20261027` (ignored by Git).
Coordinator log: `/tmp/temporal1024_window200ms_20261027.log`.

Inside the output, inspect:

- `run.json`: overall status, completed training runs and evaluation banks.
- `configs/`, `frozen_source.json`, `frozen_project/`: reproducible settings/code.
- `training/<variant>/history.json`, `optimizer_latest.json`, `model_info.json`,
  `metrics.json`, `model.npz`: epoch progress, recovery, selected checkpoint,
  full training/validation errors and portable export.
- `logs/train_<variant>.log`: preprocessing, actual device and epoch progress.
- `evaluation/<bank>/`: every scenario, expert/model trace and task result.
- `reports/command_fidelity/`, `reports/<bank>/` and
  `reports/paired_control/`: complete offline/control diagnostics and final CDFs.

For another run, pass a **new** output directory:

```bash
.venv/bin/python scripts/compare_temporal_windows.py \
  --reference-run runs/sampling_comparison/patience_series_20261026/training/mlp_stratified1024 \
  --external-split runs/sampling_comparison/patience_series_20261026/training/mlp_uniform1024/split.json \
  --output-dir runs/temporal_comparison/NEW_ID --workers 16 \
  --window-ms 200 --spacing-ms 20
```

## Verification and interpretation limits

The earlier53 repository tests passed, including the new three temporal tests and existing
actual CUDA fitting/export/recovery and simulator regressions. Temporal tests
check future-data exclusion, initial padding, multiple episode boundaries,
ring-buffer wraparound, batch/live prediction agreement, saved metadata/reset,
sparse-log rejection, CPU/CUDA gather equality, training-only statistics, streamed
versus resident training, and exact optimizer resume. Source compilation passes.

A separate synthetic-label CUDA training and real physics smoke rollout also
passes:250 successive1ms policy calls, including ring wraparound, match offline
window predictions exactly (maximum absolute raw-command difference0). This
checks deployment integration, not control quality. Evidence:
`runs/diagnostics/window_cuda_rollout_smoke_20261027/verification.json`.

This first experiment uses one seed. Improvement over both instantaneous controls
would support useful temporal information, but needs repeat seeds and a fixed
test protocol for stronger claims. Lower supervised error alone is not evidence
of stable autonomous control. Keep every failed trajectory and publish negative
results. No LSTM, DAgger, dropout, BatchNorm or output-bounding intervention is
introduced in this experiment.

Correction verification: all four temporal tests pass for the updated launcher,
including the201D representation and201-slot ring/offline agreement. Seven
checkpoint-relevant training modules remain byte-identical and were hash-verified
before resuming; the full suite was not repeated for these launcher/report edits.
The earlier CUDA/physics smoke used100ms and is integration evidence, not a200ms
performance result. Use --baseline-checkpoint to recover the original interrupted
instant128 checkpoint without overwriting that historical campaign.
