# PyTorch MLP training on CUDA

Rodrigo authorized a PyTorch implementation on 26 October 2026 after the hardware
check showed that the original sklearn training used one CPU thread per model.
This backend is an explicit experimental variant. The existing CPU campaign is
preserved, and its remaining evaluations run from an immutable source snapshot.
Its models and measurements must not be silently replaced by GPU-trained models.

The shared training launcher is still `scripts/train_model.py`. Select
`experiment_specs/training/mlp_pytorch.json` to use PyTorch. Data inventory,
episode splits, training-only normalization, final per-output/phase metrics,
model serialization, and autonomous evaluation remain shared with the baseline.

## Environment and execution

The simulator continues using the sibling RoboEnv Pixi environment. PyTorch is
installed in a local ignored `.venv` that inherits RoboEnv's installed packages;
this adds CUDA packages without changing the environment of the running CPU
campaign. No environment directory, data or trained weights are versioned.
The optional project dependency pins torch 2.11.0. This machine was verified with
**torch 2.11.0+cu128, CUDA runtime 12.8, RTX 4060 Laptop GPU (8 GB)**.

One-time setup on this Linux/NVIDIA machine:

```bash
uv venv --python ../RoboEnv/.pixi/envs/default/bin/python --system-site-packages .venv
uv pip install --python .venv/bin/python 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu128
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Train against an existing complete dataset and its frozen episode split:

```bash
python scripts/train_model.py \
  --dataset /path/to/complete/dataset \
  --split /path/to/frozen/split.json \
  --config experiment_specs/training/mlp_pytorch.json \
  --mapped \
  --output-dir runs/training/NEW_PYTORCH_RUN
```

For an isolated backend comparison on identical data, also pass
`--preprocessing-from /path/to/complete/reference/model/run`. This reuses exact
training-only normalizers after verifying the identical ordered training
trajectory checksums. The new output directory must not already exist.

Recover an interrupted PyTorch fit into a new run directory with
`--resume-checkpoint /path/to/interrupted/run/optimizer_latest.json` and identical
config/data/backend/version/source context. Add `--preprocessing-from` pointing
to a complete run with the same ordered training inventory when exact frozen
normalizers were used originally. Otherwise reconstruct the same training-only
incremental normalizers from the original dataset. sklearn optimizer checkpoints
are a different format and cannot be resumed by this backend.

## Fixed scientific choices

- Same 21 inputs, 128/128 ReLU hidden layers, nine linear outputs and 20,489
  regression parameters. No extra state/history, output bounds or expert changes.
- Same seed, batch size 1024, learning rate 0.001 and alpha 0.0001. Preserve the
  NumPy RandomState Glorot weight/bias initialization and epoch permutation stream.
- Actual sklearn multioutput backpropagation sums the output-channel gradients,
  while its reported loss averages channels. PyTorch explicitly preserves both
  conventions: optimize half the output-summed MSE and report half output-mean
  MSE, with weight-only alpha/(2 × actual batch rows) L2. Biases are not decayed.
- Adam uses beta1=0.9, beta2=0.999 and the sklearn epsilon convention. PyTorch's
  epsilon is adjusted each update to account for its bias-corrected denominator.
  The CUDA optimizer uses PyTorch's fused implementation.
- Float32 training, float64 normalizers and physical-error reductions. Mixed
  precision and TF32 are disabled. Deterministic algorithms are requested; record
  Torch/CUDA/device details. These controls do not imply bit-identical results
  across frameworks, hardware or releases.
- No epoch cap (`max_epochs: null`), unchanged patience 8 and min_delta 1e-4 on
  equal-episode balanced validation NRMSE. Save the lowest-validation checkpoint.
  A small finite cap may be used only for explicitly labelled tests/benchmarks.
- Validation forward passes use Torch on the selected device; physical MAE/RMSE
  use float64 and the same metrics. Final reporting uses the exported NumPy
  policy. Tiny forward-pass rounding differences are expected and checked.

These details were checked against the installed sklearn implementation, rather
than assuming its loss/Adam conventions match PyTorch defaults. A numerical test
compares the first actual sklearn Adam update and reported loss with PyTorch.
Interrupted/resumed CPU and CUDA Torch fits reproduce uninterrupted Torch fits
exactly in the tested environment. Resident and streaming CPU batches also agree.

## Memory and device settings

The recipe contains a separate `torch` section:

| Setting | Meaning |
|---|---|
| `device: cuda` | Require CUDA. Fail explicitly if unavailable; do not silently train on CPU. `cpu` and `auto` are available for deliberate variants. |
| `data_placement: auto` | Put normalized training arrays and validation inputs on the GPU when a conservative free-VRAM estimate permits it; otherwise stream minibatches. |
| `validation_batch_size: 16384` | Bound validation activations independently of training minibatch size. Preserve whole-episode metric weighting. |
| `deterministic: true` | Request deterministic Torch algorithms and configure cuBLAS workspace. |

The CLI automatically uses the disk-backed training path for PyTorch; `--mapped`
is shown explicitly above for clarity. Inputs and targets are normalized with
training-only statistics. GPU-resident data are transferred once, and each epoch
transfers one permutation, rather than copying the full dataset per minibatch.
Streaming retains the same minibatch/optimizer semantics with bounded memory.
Peak allocated/reserved VRAM, actual device and placement are stored in
`model_info.json`. A small MLP need not occupy 100% of GPU compute to use CUDA.
Do not change batch size or numerical precision merely to report high utilization.

## Artifacts and simulator compatibility

Each run retains the existing `config.json`, `split.json`, `run.json`,
`history.json`, `model_info.json`, `metrics.json`, and per-output CSV reports.
It additionally saves immutable `.pt` optimizer checkpoints and a hash/context
manifest. Safe weights-only loading restores model, Adam, epoch, patience,
selected-best weights and NumPy/Torch/CUDA RNG states.

The final `model.npz` is the existing `MLPPolicy` format. Autonomous simulation
uses the same NumPy inference and four-wheel expansion as before, so the default
RoboEnv environment needs **no Torch or CUDA** for evaluating the model:

```bash
python scripts/evaluate_policy.py \
  --split runs/training/NEW_PYTORCH_RUN/split.json \
  --model pytorch_mlp=runs/training/NEW_PYTORCH_RUN \
  --config experiment_specs/evaluation/trajectory_diagnostic.json \
  --workers 16 --output-dir runs/evaluation/NEW_PYTORCH_EVALUATION
```

CUDA training alone does not accelerate PyBullet simulation or establish stable
control. Supervised fidelity and matched autonomous task metrics remain separate.

## Verification and current experiments

All 47 tests passed, including real CUDA tests, first-update/loss comparison,
CPU/CUDA optimizer recovery, portable export, patience, geometry and the previous
expert/data/evaluation regressions. The real uniform64 pilot uses the original
51/13 split and exact original training-only normalizers. Its artifacts are in
`runs/training/mlp_pytorch_uniform64_20261026`. Training and simulator integration
verification are complete. This is one development run, not a replacement for the
frozen learning-curve campaign.

Sources: [PyTorch CUDA execution and timing](https://docs.pytorch.org/docs/stable/notes/cuda),
[reproducibility limits](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[official installation](https://pytorch.org/get-started/locally/).


### Measured uniform64 pilot

| Quantity | Measurement |
|---|---:|
| Whole episodes, train / validation | 51 / 13 |
| Regression parameters | 20,489 |
| Completed epochs / selected epoch | 54 / 46 |
| Stop reason | patience |
| Native validation balanced NRMSE | 0.182987 |
| Median epoch time, training plus validation | 2.227 s |
| Whole training command wall time | 122.11 s |
| Peak Torch tensor allocation | 148.64 MiB |
| Preview maximum Torch / exported NumPy physical difference | 3.3194065e-06 |

The last difference is across the nine physical channels (mixed units), an export
consistency check rather than a control tolerance. CUDA context/driver allocations
are additional to Torch's recorded tensor allocation.

The original sklearn64 reference had native NRMSE0.203130, stopped at22 epochs,
and a historical median1.418s/epoch. This Torch pilot has different selected
weights/epoch and numerical kernels. Its approximately2.227s/epoch does **not**
establish a speedup; historical/current workload and sampling of timing differ.
The MLP is small and GPU launch overhead matters. Preserve batch1024 and FP32 for
this variant; future throughput tuning needs its own measured comparison.

The exported policy was evaluated with the original RoboEnv interpreter, where
Torch is absent: one held-out condition, expert plus model, forced40s horizon,
both runs valid. Expert task arrival1/1; model0/1, final EE error110.9338m
(simulated divergence). This verifies compatibility, **not stable control** or
full offline/online backend superiority. The exact records and traces are in
`runs/evaluation/pytorch_uniform64_integration_20261026`.
`integration_verification.json` beside the trained model links these checks.

## Configurable hidden depth:27 October 2026

The backend and portable exporter now accept any nonempty list of positive
hidden widths, with ReLU hidden activations and a linear command output. Existing
two-layer artifacts remain loadable; deeper NPZ files store all consecutively
indexed weight/bias arrays. Source/configuration checks intentionally reject
optimizer recovery across changed training implementations. Do not relabel an
older checkpoint as a different architecture. The current controlled candidates,
data/representation settings and verification are documented in
[MLP depth and shape](MLP_DEPTH_AND_SHAPE.md).
