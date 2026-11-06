# 3. Models

The progression of policies from the baseline that never arrives to the ones that do, their architectures and training
recipe, and the commands that reproduce every model of the report. Code: `src/robot_learning/learning/` (`mlp.py`,
`gru.py`, `transformer.py`), recipes `experiment_specs/training/`. Results: [results.md](results.md) 1.1.

## 3.1 The progression

| model | what it adds | problem it addresses | 1 cm, nominal (five seeds) |
|---|---|---|---|
| **MLP-Base** | an MLP on the 21 inputs + clock | the baseline of the brief | 1.6 ± 3.6 |
| **MLP-Ego** | + egocentric transforms ([02](02_inputs_and_outputs.md)) | the base parks at the dock | 2.0 ± 1.6 |
| **MLP-Ego-IC** | + ignition copies ([05](05_arm_ignition.md)) | the arm starts | **189.0 ± 9.3** |
| MLP-Ego-IC + DB | + wheel dead band at evaluation ([06](06_base_drift.md)) | the wheels stop exactly | 189.2 ± 9.3 |
| MLP-Ego-Cat | categorical wheels + ignition copies ([06](06_base_drift.md)) | a learned exact zero | 188.2 ± 11.3 |
| GRU-Ego | memory instead of copies | ignition and stop from history | 149.2 ± 23.9 |
| TF-Ego | attention over the episode instead of copies | same, with a longer memory | 158.4 ± 43.2 |

All models share the 28 egocentric inputs (except MLP-Base), the nine outputs, the training data (3 221 expert-success
episodes of the training mixture, [01](01_data_collection.md)) and the selection rule.

## 3.2 Architectures

**MLP** (`mlp.py`): 28 → 128 → 128 → 9, ReLU, linear outputs (21 385 parameters). Stateless: one decision per
observation, every 1 ms. MLP-Ego-Cat replaces the two wheel outputs by two 41-way softmax heads (31 705 parameters).

**GRU** (`gru.py`): memory at a slow rate, reaction at the control rate. A single-layer GRU (128 units) reads the
features every 20 ms and keeps its state for the whole episode; an output head (128 ReLU units) reads the latest state
and the current features at **every** 1 ms step (81 929 parameters). The control period stays 1 ms because the expert
itself fails when its command is held for 20 ms; only the memory update is strided. Trained on whole episodes with
truncated backpropagation through time (chunks of 1 000 memory updates = 20 s, state carried across chunks and reset at
every episode).

**Causal transformer** (`transformer.py`): same two-rate structure. Every 20 ms the current features become a token; two
layers of causal self-attention (d 64, 4 heads, feed-forward 128) attend over the last 1 000 tokens (20 s); the same kind
of head reacts at every step (81 993 parameters). No positional encoding: order comes from the causal mask and time from
the clock input, so nothing extrapolates when a 40 s evaluation outlasts the demonstrations.

All three are trained in PyTorch and exported to portable NumPy files (`model.npz`, no pickle); simulation needs no
torch. At the end of every run the trainer checks that NumPy inference reproduces the torch network and that live
inference (one step at a time) equals batch inference.

## 3.3 Training recipe

| | MLP | GRU / transformer |
|---|---|---|
| loss | half-sum squared error on the normalised outputs (+ cross-entropy for categorical wheels) + L2 1e-4 on the weights | masked half-sum squared error over every physics step of whole episodes + L2 1e-4; gradient-norm clip 10 |
| optimiser | Adam, lr 1e-3 | Adam, lr 1e-3 |
| batch | 1 024 rows drawn from all training episodes | 8 whole episodes |
| stopping | early stopping on the validation balanced NRMSE, patience 8 | patience 40 (slow, noisy improvement) |
| checkpoint | lowest validation balanced NRMSE | same |
| data | expert successes only; inputs and outputs standardised with the training rows | same |
| seeds | seed 0 of the report = 20261025; seeds 1–4 = 1–4 (`--seed`) | same |
| determinism | deterministic CUDA algorithms, no TF32: a rerun reproduces the weights bit for bit (verified) | |

Model selection uses the **offline** validation error only; closed-loop results are never used to pick a checkpoint
(chapter 4 shows why the offline error is a weak guide, which is why every model is reported over five seeds).

## 3.4 Efficiency

| model | parameters | inference per step (µs, one CPU thread, NumPy) | training (min, one GPU) |
|---|---|---|---|
| MLP-Ego(-IC) | 21 385 | 35 | 27–29 |
| MLP-Ego-Cat | 31 705 | 40 | 27 |
| GRU-Ego | 81 929 | 33 | 20 |
| TF-Ego | 81 993 | 31 | 22 |

All run 30 times faster than the 1 ms control period.

## 3.5 Reproducing the models

With `PY="pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python"`, the training mixture from
[01](01_data_collection.md) 1.9 (`NOMINAL_DART_DIR` = primary dataset of `runs/splits/reaching_train_val.json`) and `N`
the seed (0 → `--seed 20261025`, else `--seed N`):

```bash
COMMON="--dataset NOMINAL_DART_DIR --split runs/splits/reaching_train_val.json --device cuda --horizon-s 40"
# MLP-Base / MLP-Ego / MLP-Ego-IC / MLP-Ego-Cat
$PY scripts/train_mlp.py $COMMON --inputs raw21_clock_v1             --seed N --output-dir runs/training/mlp_base_seedN
$PY scripts/train_mlp.py $COMMON --inputs raw21_targetpolar_clock_v1 --seed N --output-dir runs/training/mlp_ego_seedN
$PY scripts/train_mlp.py $COMMON --inputs raw21_targetpolar_clock_v1 --seed N --option trigger_copies=144 \
    --output-dir runs/training/mlp_ego_ic_seedN
$PY scripts/train_mlp.py $COMMON --inputs raw21_targetpolar_clock_v1 --seed N --option trigger_copies=144 \
    --option wheel_output=categorical --output-dir runs/training/mlp_ego_cat_seedN
# GRU-Ego, TF-Ego
$PY scripts/train_gru.py $COMMON --inputs raw21_targetpolar_clock_v1 --seed N --hidden 128 --head-width 128 --stride 20 \
    --chunk-steps 1000 --episodes-per-batch 8 --output-dir runs/training/gru_ego_seedN
$PY scripts/train_transformer.py $COMMON --inputs raw21_targetpolar_clock_v1 --seed N --stride 20 --window 1000 \
    --output-dir runs/training/tf_ego_seedN
# data controls: the same MLP-Ego-IC command on the clean mixture (no DART) or on the nominal split only;
# data-size curves: --option training_target_cells=experiment_specs/data_size/balanced_cells_512.json (or 128, 256)
#                   --option training_episode_cap=400 (random subsets)
```

Each run directory holds `model.npz`, `model_info.json` (architecture, data filter, epochs, offline error, inference
time), `metrics.json`, `history.json`, `run.json` (command, hashes of the data and of the source) and a copy of the
source. The trained models of the report are in the `rodrigo` branch (`runs/training/v2_*`, names in
[01](01_data_collection.md) 1.9).
