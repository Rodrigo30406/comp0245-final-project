# Using the code

A task-by-task guide: for each thing you may want to do, the command, its parameters, what to choose, and what it
produces. Setup (installing pixi and RoboEnv, next to this repository) is in the main [README](../README.md#1-setup). All commands run from the
project root with:

```bash
PY="pixi run --manifest-path ../RoboEnv/pixi.toml python"            # simulation, collection, evaluation
PYML="pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python"    # training (PyTorch)
```

| I want to … | section |
|---|---|
| watch the expert or a trained policy | [1](#1-watch-the-expert-or-a-trained-policy) |
| collect expert demonstrations | [2](#2-collect-expert-demonstrations) |
| split a dataset into training and validation | [3](#3-split-the-data) |
| train a model | [4](#4-train-a-model) |
| evaluate models in closed loop | [5](#5-evaluate-in-closed-loop) |
| get the tables and figures of the report | [6](#6-tables-and-figures) |
| work on the obstacle tasks | [7](#7-obstacle-tasks) |
| reproduce a table or figure of the report | [8](#8-reproducing-the-report) |

---

## 1. Watch the expert or a trained policy

```bash
$PY scripts/simulate.py   --target 1.5 1.0 0.6                                         # the expert
$PY scripts/run_policy.py --model weights/v2_mlp28_trig144_seed0 --target 1.5 1.0 0.6  # a trained policy
```

| parameter | meaning | default |
|---|---|---|
| `--target X Y Z` | target of the end effector, world frame (m); the training range is x, y ∈ [−2, 2], z ∈ [0.3, 0.8] | required |
| `--model DIR` / `--expert` | a trained model (any directory of `weights/`, or your own training run) / the expert | — |
| `--start X Y YAW` | base start pose (m, m, rad) | the origin, facing +x |
| `--seconds S` | episode length; there is no stop rule, the policy runs the whole time | 40 |
| `--dead-band 0.45` | wheel commands below 0.45 rad/s become exactly zero (the +DB variant) | off |
| `--headless` | no window (faster than real time) | GUI |

Which model to watch: see the table in [4](#4-train-a-model) (report name → `weights/` directory). Good contrasts for a
demonstration: `v2_mlp28_control_seed0` (MLP-Ego: the base arrives, the arm never starts) against
`v2_mlp28_trig144_seed0` (MLP-Ego-IC: it reaches).

---

## 2. Collect expert demonstrations

```bash
$PY scripts/collect_dataset.py --config experiment_specs/collection/<spec>.json --workers 16
```

The collector runs the expert in the simulator once per scenario and records every 1 ms step (observations, the
expert's clean command as label, the executed command). It prints the new dataset directory
(`datasets/<timestamp>_<name>_<hash>/`).

| parameter | meaning | default |
|---|---|---|
| `--config SPEC.json` | the collection spec: which scenarios, how many, which noise (below) | required (or `--resume`) |
| `--workers N` | episodes simulated in parallel (16 is a good value on 32 cores) | 1 |
| `--episodes N` | collect only the first N scenarios of the spec | all |
| `--max-steps N` | episode cap in 1 ms steps | 40 000 (40 s) |
| `--plan-only` | write the scenarios (targets, start poses, seeds) without simulating | off |
| `--resume DIR` | continue an interrupted collection | — |
| `--gui` | watch the episodes (with `--workers 1`) | off |
| `--output-dir DIR` | where datasets are created | `datasets/` |

**Which spec** (`experiment_specs/collection/`):

| I want … | spec | episodes |
|---|---|---|
| a quick check that everything works | `smoke.json` | 2 short |
| the training data of the report, nominal start | `reaching_nominal_dart.json` | 1 024 |
| the training data of the report, random starts | `reaching_random_start_dart.json` | 3 072 |
| the re-collection of the expert's failed scenarios | `reaching_recollect_failures_dart.json` | 353 |
| the random-start validation bank | `reaching_validation_random_start_dart.json` | 1 024 |
| the nominal validation bank / the no-DART control | `reaching_nominal_clean.json` (+ `reaching_random_start_clean.json`, `reaching_validation_random_start_clean.json`) | 1 024 (+ 3 072, 1 024) |
| banks outside the training range | `ood_far_targets.json`, `ood_target_height.json`, `ood_far_start.json` | 210 each |

**Making your own spec**: copy one and edit its fields. The useful ones:

| field | meaning |
|---|---|
| `episodes`, `seed` | how many scenarios; scenario i depends only on (seed, i) |
| `targets` or `target_box` | explicit target list, or uniform sampling in a box `{"x": [lo, hi], "y": …, "z": …}` |
| `initial_base` | `{"mode": "nominal"}` or `{"mode": "uniform_xy", "x": [lo, hi], "y": [lo, hi], "yaw": [lo, hi]}` |
| `goal_reference` | `"start"` for random starts (the expert's base goal is computed from the start pose) |
| `action_noise` | DART noise on the executed command: `{"wheel_std": 0.5, "torque_std": 1.0, "hold_steps": 50, "seed": N}`; `null` = clean |
| `base_controller`, `arm_controller`, `stop`, `standoff`, `physics` | override any rule of the expert (`experiment_specs/expert/default.json`) |

Why these choices (DART, random starts, the target grid, the amount of data): [01_data_collection.md](01_data_collection.md).

---

## 3. Split the data

Training and validation are separated by **target**: an episode is validation when its target lies in one of the 205
held-out cells of the target grid, whatever its start pose, so no validation target is ever seen in training.

```bash
$PY scripts/data/make_train_val_split.py OUT.json PRIMARY_DATASET [EXTRA_DATASET ...]
$PY scripts/data/make_bank_split.py BANK_DATASET OUT.json            # evaluation-only banks (ood_*)
```

| split | command | used for |
|---|---|---|
| the training mixture of the report | `make_train_val_split.py runs/splits/reaching_train_val.json NOMINAL_DART RANDOM_START_DART RECOLLECT` | training every model |
| nominal validation bank | `make_train_val_split.py runs/splits/nominal_bank.json NOMINAL_CLEAN` | evaluation (205 targets, nominal start) |
| random-start validation bank | `make_train_val_split.py runs/splits/reaching_validation_random_start.json VALIDATION_RANDOM_START_DART` | evaluation (205 targets, random starts) |
| a bank outside the range | `make_bank_split.py OOD_DATASET runs/splits/bank_<name>.json` | evaluation only |
| no-DART control | `make_train_val_split.py runs/splits/clean_train_val.json NOMINAL_CLEAN RANDOM_START_CLEAN` | training the controls |

Expert failures stay in the split (every episode is accounted for); the trainers drop them.

---

## 4. Train a model

**Choose the model** (all on the training mixture; report names as in [03_models.md](03_models.md)):

| report name | script | `--inputs` | options | trained models in `weights/` |
|---|---|---|---|---|
| MLP-Base | `train_mlp.py` | `raw21_clock_v1` | — | `v2_mlp22raw_control_seed{0..4}` |
| MLP-Ego | `train_mlp.py` | `raw21_targetpolar_clock_v1` | — | `v2_mlp28_control_seed{0..4}` |
| **MLP-Ego-IC** | `train_mlp.py` | `raw21_targetpolar_clock_v1` | `--option trigger_copies=144` | `v2_mlp28_trig144_seed{0..4}` |
| MLP-Ego-IC + DB | same model as MLP-Ego-IC | | dead band at evaluation (section 5) | same |
| MLP-Ego-Cat | `train_mlp.py` | `raw21_targetpolar_clock_v1` | `--option trigger_copies=144 --option wheel_output=categorical` | `v2_mlp28_cat41_trig144_seed{0..4}` |
| GRU-Ego | `train_gru.py` | `raw21_targetpolar_clock_v1` | `--hidden 128 --head-width 128 --stride 20 --chunk-steps 1000 --episodes-per-batch 8` | `v2_gru28_patience40_seed{0..4}` |
| TF-Ego | `train_transformer.py` | `raw21_targetpolar_clock_v1` | defaults (`--stride 20 --window 1000 --d-model 64 --layers 2 --heads 4`) | `v2_tf28_p40_seed{0..4}` |

**Parameters common to the three trainers:**

| parameter | meaning | report value |
|---|---|---|
| `--dataset DIR` | the primary dataset of the split | the nominal DART collection |
| `--split FILE` | the train/validation split (section 3) | `runs/splits/reaching_train_val.json` |
| `--config FILE` | the recipe: layers, learning rate, batch, early stopping (`experiment_specs/training/{mlp,gru,transformer}.json`) | default of each trainer |
| `--inputs VARIANT` | input variant: `raw21_clock_v1` (21 + clock), `raw21_targetpolar_clock_v1` (+ egocentric transforms), `raw21_targetpolar_clock_lidar64_v1` (+ lidar, obstacle tasks) | see the table |
| `--seed N` | training seed | 20261025 (seed 0), 1, 2, 3, 4 |
| `--output-dir DIR` | the run directory to create | — |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--horizon-s S` | normalisation of the clock input (the evaluation horizon) | 40 |

**MLP options** (`--option KEY=VALUE`): `trigger_copies=K` (ignition copies, [05](05_arm_ignition.md)),
`wheel_output=categorical` (+ `wheel_bins`, [06](06_base_drift.md)), `training_episode_cap=N` and
`training_target_cells=experiment_specs/data_size/balanced_cells_<128|256|512>.json` (data-size curves,
[01](01_data_collection.md) 1.5).

**Recipes** (`experiment_specs/training/`): `mlp.json` (2 × 128 ReLU, Adam 1e-3, batch 1 024 rows, L2 1e-4, early
stopping with patience 8 on the validation error), `gru.json` and `transformer.json` (same optimiser, patience 40).
Copy and edit one to change hyperparameters.

Example, MLP-Ego-IC seed 0 (~30 min on one GPU):

```bash
$PYML scripts/train_mlp.py --dataset datasets/<…>_reaching_nominal_dart_<…> --split runs/splits/reaching_train_val.json \
    --inputs raw21_targetpolar_clock_v1 --option trigger_copies=144 --seed 20261025 --output-dir runs/training/mlp_ego_ic_seed0
```

**What it produces** (`runs/training/<name>/`): `model.npz` (the policy, NumPy), `model_info.json` (architecture, data,
epochs, offline validation error, inference time), `metrics.json`, `history.json` (per epoch), `run.json` (command and
hashes), a copy of the source. The run directory is what sections 1 and 5 take as a model.

---

## 5. Evaluate in closed loop

The policy drives the simulated robot alone from the start pose of each held-out scenario, for 40 s with no stop rule;
the expert is always evaluated on the same scenarios as the reference.

```bash
$PY scripts/evaluate_policy.py --split runs/splits/nominal_bank.json \
    --model ic_seed0=weights/v2_mlp28_trig144_seed0 --model ego_seed0=weights/v2_mlp28_control_seed0 \
    --allow-cross-split --allow-external-holdout --workers 16 --output-dir runs/evaluation/my_eval
```

| parameter | meaning | default |
|---|---|---|
| `--split FILE` | the bank: its validation episodes are the scenarios evaluated (section 3) | required |
| `--model LABEL=DIR` | a model to evaluate (repeat for several); LABEL names it in the results | the expert only |
| `--config FILE` | the protocol (below) | `validation.json` |
| `--workers N` | episodes in parallel | 1 |
| `--allow-cross-split` | allow models trained on another split (always needed with the banks above); scenarios used in a model's training are still rejected | off |
| `--allow-external-holdout` | allow a bank collected separately (`ood_*`, random-start bank) | off |
| `--limit N` | only the first N scenarios (quick check) | all |
| `--output-dir DIR` | where the results go | `runs/evaluation/<time>` |

**Choose the protocol** (`experiment_specs/evaluation/`):

| protocol | what changes |
|---|---|
| `validation.json` | the standard protocol: 40 s, tolerances 1 / 2 / 5 cm with 0.5 s dwell, commands clipped to the expert's envelope |
| `validation_deadband045.json` | + wheel dead band 0.45 rad/s on the policies (the +DB variant) |
| `validation_gain_left095.json` | the left wheels execute 95 % of the command |
| `validation_payload0p1kg.json`, `0p25kg`, `0p5kg` | extra mass on the last arm link, unknown to the expert's gravity compensation |
| `validation_obsnoise_tenth.json`, `_half.json`, `validation_obsnoise.json` | Gaussian noise on what the policy reads (base 2 cm / 0.02 rad, arm 0.01 rad / 0.05 rad/s at full level) |
| `validation_delay2ms.json`, `5ms`, `50ms` | the executed command is the one issued 2 / 5 / 50 ms earlier |

**What it produces** (`runs/evaluation/<name>/`): `summary.json` (per policy: stable arrivals at each tolerance, final
and closest distances, clipping), `episodes/*.json` (every rollout), `traces/*.npz` (100 ms traces: base pose, commands,
end-effector distance).

**Statistics:**

```bash
python3 scripts/evaluation/summarize_eval.py runs/evaluation/my_eval            # one line per policy
python3 scripts/evaluation/seed_stats.py runs/evaluation/<dir> [...]           # mean ± sd (min–max) over seeds
python3 scripts/evaluation/arm_base_decomposition.py runs/evaluation/my_eval    # base parked, arm active, hold
```

Re-evaluating the shipped `weights/` needs the datasets of their split (the evaluator checks that no evaluated scenario
was used in training); they are found by directory name under `datasets/` or `$DATASETS_ROOT`.

---

## 6. Tables and figures

| I want … | command | input |
|---|---|---|
| every reaching table of the report | `python3 scripts/evaluation/report_tables.py` | `results/evaluations/` (shipped), `weights/` |
| one cell over seeds | `python3 scripts/evaluation/seed_stats.py DIR [DIR …]` | evaluation directories |
| parking, final stop, after-arrival, arm-error figures | `python3 scripts/figures/problem_figures.py OUT LABEL=EVAL_DIR:POLICY …` | evaluation traces |
| offline error per expert phase | `$PY scripts/figures/phase_errors.py OUT SPLIT LABEL=RUN …` | datasets + models |
| offline error vs closed-loop success | `python3 scripts/figures/offline_vs_closed_loop.py OUT RUNS "LABEL=RUN_PREFIX:EVAL_DIRS" …` | models + evaluation summaries |
| the arm ignition probe | `$PY scripts/figures/arm_switch_probe.py EVAL_DIR RUN POLICY_LABEL` | evaluation traces + model |
| the report figures (trajectories, generalisation, geometry) | `python3 scripts/figures/report_figures.py EPISODE OUT EVAL_ROOT` | evaluation directories |
| the egocentric-frame drawing | `python3 scripts/figures/egocentric_frame.py OUT.png` | — |

Which evaluations feed each table and figure of the report: [section 8](#8-reproducing-the-report).

---

## 7. Obstacle tasks

The group's obstacle-task code lives in `controlled_experiments/` (link the robot models once:
`ln -s ../../RoboEnv/models controlled_experiments/models`).

| I want … | command |
|---|---|
| static obstacle: collect with the 10 s hold and our noise | `cd controlled_experiments && OBSTACLE_HOLD_WHEEL_NOISE=0.5 $PY ../scripts/obstacle/collect_static_obstacle.py --set-name train2 --noise-frac 1.0 --noise-v 0.0389 --noise-w 0.1178 --noise-torque 1.0 --noise-hold 0.05 --noise-fade-base 0 --noise-fade-arm 0 --workers 12 --out-dir RAW` |
| convert an obstacle collection for the trainers | `$PYML scripts/obstacle/convert_obstacle_dataset.py --source RAW --output datasets/<name> --split-out runs/splits/<name>.json` |
| train on it | `train_mlp.py` / `train_gru.py` with `--inputs raw21_targetpolar_clock_lidar64_v1 --horizon-s 60` |
| static obstacle: collect the evaluation set dev2 (300 scenes, seed 300) | `cd controlled_experiments && $PY -m obstacle_task.collect --set-name dev2 --workers 16` (writes `data/obstacle/dev2`) |
| evaluate a static-obstacle policy | `cd controlled_experiments && $PY -m obstacle_task.evaluate --policy ../scripts/obstacle/rl_obstacle_policy.py:make_policy --policy-arg RUN_DIR --eval-set data/obstacle/dev2 --workers 16 --out-dir OUT` |
| moving obstacle: watch the expert | `$PY scripts/obstacle/dynamic_obstacle.py --seed 1 --episodes 0-11 --gif --out-dir OUT` |
| moving obstacle: training collection / policy test | `… dynamic_obstacle.py --seed 3 --episodes 0-1599 --dart --workers 28 --out-dir OUT` / `--seed 4 --episodes 0-99 --policy RUN_DIR` |

`RUN_DIR` can be any trained model of `weights/obstacle/` (e.g. `weights/obstacle/static_mlp_seed0`,
`weights/obstacle/dynamic_mlp_seed0`). The report's tables without simulating:
`python3 scripts/obstacle/dev2_table.py` (static, from `results/obstacle/static_dev2/`) and the per-scene rows of
`results/obstacle/dynamic_test/` (moving). Parameters of each: [01_data_collection.md](01_data_collection.md) section 1.8;
what the tasks and the data changes are: [07_obstacles.md](07_obstacles.md).

---

## 8. Reproducing the report

How the pieces above produce every number and figure of the report. Three levels, from seconds to days of compute:

| level | what you get | needs | time |
|---|---|---|---|
| **1. tables from the evaluation summaries** | every reaching number of the report | this repository | seconds |
| **2. re-evaluate the trained models** | the closed-loop runs, traces and figures | `weights/` + the datasets of their splits (~64 GB, copy from the group's machine) | 15 min per evaluation of 205 episodes × 4 policies (16 workers) |
| **3. from scratch** | everything, including new datasets and models | RoboEnv, a GPU | ~1 h collection, ~30 GPU h training (all families × 5 seeds), ~20 h evaluation |

Each stage was checked against the report's artifacts: the target grid and the splits are regenerated identical,
collections reproduce the report's scenarios (and the obstacle recordings bit for bit; the expert itself is checked
against recorded lecturer episodes by the tests), retraining reproduces the weights bit for bit, and the evaluator
reproduces the closed-loop traces and metrics exactly.

### 8.1 Level 1: the tables, without simulating

```bash
python3 scripts/evaluation/report_tables.py          # tables 1.1–1.4 of docs/results.md, from results/evaluations/
python3 scripts/evaluation/seed_stats.py results/evaluations/<dir> [...]     # any cell, mean ± sd (min–max)
```

### 8.2 Levels 2 and 3: each table and figure → what to run

Evaluation directory names are those of the experiment records (`results/evaluations/`); "both banks" = run the same
command with `--split runs/splits/nominal_bank.json` and with `--split runs/splits/reaching_validation_random_start.json`.

| report item | models | evaluation | then |
|---|---|---|---|
| results 1.1, main table | the seven families of [03](03_models.md) 3.1, five seeds | both banks, protocol `validation.json`; +DB: `validation_deadband045.json` on the MLP-Ego-IC models | `seed_stats.py` |
| results 1.2, data controls | `v2_mlp28clean_*`, `v2_mlp28nom_trig144` | both banks | `seed_stats.py` |
| results 1.2, data design validation | `v2_mlp28_trig144_bal{128,256,512}`, `cap{400…2400}` | both banks | `summarize_eval.py` |
| results 1.3, generalisation | MLP-Ego-IC, Cat, GRU-Ego seeds 0–2 (+DB) | `--split runs/splits/bank_<far_targets|target_height|far_start>.json` (`make_bank_split.py` on the `ood_*` collections) | `summarize_eval.py` |
| results 1.4, robustness | MLP-Ego-IC, GRU-Ego seeds 0–2 | nominal bank, `--config experiment_specs/evaluation/validation_<adversity>.json` | `summarize_eval.py` |
| results 1.5, efficiency | all | — | `model_info.json` of each run |
| [02](02_inputs_and_outputs.md) 2.6, base parked / arm active | MLP-Base, MLP-Ego (+ copies) | nominal bank | `arm_base_decomposition.py` |
| [04](04_evaluation.md) offline vs closed loop | all families | nominal bank | `scripts/figures/offline_vs_closed_loop.py` |
| [05](05_arm_ignition.md) / [06](06_base_drift.md) figures | MLP-Ego, MLP-Ego-IC (+DB), Cat, GRU-Ego | nominal bank | `scripts/figures/problem_figures.py`, `phase_errors.py`, `arm_switch_probe.py` |
| report figures (trajectories, generalisation, geometry) | as above | as above | `scripts/figures/report_figures.py EPISODE OUT_DIR EVAL_DIR` |
| results 2, static obstacle | `scripts/obstacle/` (chapter 7) | group's harness on dev2 | `scripts/obstacle/dev2_table.py` |
| results 3, moving obstacle | `scripts/obstacle/dynamic_obstacle.py --policy` | seed 4, 100 scenes | `index.csv` of the run |

Example, the MLP-Ego-IC row of the main table from the shipped weights (level 2):

```bash
$PY scripts/data/make_train_val_split.py runs/splits/nominal_bank.json <datasets>/<…>_stratified1024_nominal_<…>
M=""; for s in 0 1 2 3 4; do M="$M --model ic_seed$s=weights/v2_mlp28_trig144_seed$s"; done
$PY scripts/evaluate_policy.py --split runs/splits/nominal_bank.json $M --allow-cross-split --allow-external-holdout \
    --workers 16 --output-dir runs/evaluation/ic_nominal
python3 scripts/evaluation/seed_stats.py runs/evaluation/ic_nominal        # 189.0 ± 9.3 (175–199) at 1 cm
```

From scratch (level 3), in this order: collect the datasets (section 2; the full list and order in
[01](01_data_collection.md) 1.9), make the splits (section 3), train the five seeds of each family (section 4),
evaluate them on both banks (section 5), then `seed_stats.py` (section 5) and the figure scripts (section 6).

### 8.3 Known differences when reproducing

- **Figures**: rendering depends on the matplotlib version and fonts (same data, slightly different glyphs).
- **Training on another GPU or driver**: deterministic algorithms make a rerun identical on the same machine and software;
  another GPU model can change the last bits and, through the closed loop, a few arrivals per seed. The five-seed
  spread (sd 9 arrivals for MLP-Ego-IC) is the scale against which such differences should be read.
- **Re-collected datasets** contain the same scenarios and trajectories but get new directory names and file hashes
  (time stamps); the shipped weights must then be evaluated against the original datasets (level 2), or retrained.
