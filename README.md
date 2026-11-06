# COMP0245 — Final Project (2026/27): behaviour cloning of a mobile manipulator

Group project for Machine Learning for Robotics, MSc Robotics and AI, UCL. A skid-steer base with a 7-joint arm must
reach a target in 3D; the lecturer's controller (the *expert*) does it with a base state machine and an IK arm
controller. We train neural policies that map the robot's observations to its wheel and arm commands, from expert
demonstrations only, and evaluate them driving the simulated robot alone.

**Main result** (205 held-out targets, five training seeds, end effector held within 1 cm): the expert 205 / 205; an MLP
on the brief's 21 inputs 2 / 205; the same MLP with egocentric inputs and *ignition copies* (MLP-Ego-IC) **189 / 205**
(and 193 / 205 from random start poses). The path between the two, the failures behind it and every table of the report
are documented in [`docs/`](docs/README.md).

## 1. Setup

The project runs on **RoboEnv**, the course environment: the PyBullet simulator, the robot models (`models/`,
`configs/`) and the lecturer's `simulation_and_control` library, managed with [pixi](https://pixi.sh). RoboEnv is cloned
**next to** this repository; the code finds it there (or through `ROBOENV_ROOT`).

```
<parent>/RoboEnv/                   # the environment: pixi.toml, models/, configs/, simulation_and_control/
<parent>/comp0245-final-project/    # this repository
```

**1. Install pixi** (once per machine):

```bash
curl -fsSL https://pixi.sh/install.sh | sh          # Linux / macOS (Windows PowerShell: irm https://pixi.sh/install-pixi.ps1 | iex)
```

**2. Get RoboEnv with its submodule, at the version used for the report:**

```bash
cd <parent>
git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
cd RoboEnv
git checkout 7bb49439677c2ac111347b40e997025ce87bff50 && git submodule update --init --recursive   # simulation_and_control df58505
```

The datasets and evaluations of the report record these two commits and the hashes of every robot model and simulator
file; the evaluator refuses to compare runs made with different physics or robot assets. A newer RoboEnv works for
new experiments, but compare results only within one version.

**3. Create the two environments** (pixi downloads everything; no compiler needed):

```bash
pixi install                  # default: simulation, data collection, evaluation (numpy, pybullet, pinocchio, scipy, ...)
pixi install -e ml            # ml: the same + PyTorch, for training
pixi run smoke-test           # prints "simulation_and_control OK"
```

Training uses a CUDA GPU when available (`--device cuda`, PyTorch wheels with CUDA on Linux); `--device cpu` works for
small runs. Everything else (simulation, evaluation, the trained policies) runs on the CPU, without PyTorch.

**4. Get this repository next to it and check the installation:**

```bash
cd <parent>
git clone https://github.com/Rodrigo30406/comp0245-final-project.git
cd comp0245-final-project
PY="pixi run --manifest-path ../RoboEnv/pixi.toml python"            # simulation, collection, evaluation
PYML="pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python"    # training (PyTorch)
ln -s ../../RoboEnv/models controlled_experiments/models             # robot models for the group's obstacle code
$PY scripts/run_tests.py                                             # 45 tests, about 1 min (torch tests: $PYML)
```

RoboEnv in another folder: `export ROBOENV_ROOT=/path/to/RoboEnv` (and use that path in `--manifest-path`). Datasets
stored elsewhere: `export DATASETS_ROOT=/path`. Both can also go in a local `.env` file (`KEY=value` lines; ignored by Git).

## 2. Try it in five minutes

```bash
$PY scripts/simulate.py --target 1.5 1.0 0.6                                   # the expert, PyBullet window
$PY scripts/run_policy.py --model weights/v2_mlp28_trig144_seed0 --target 1.5 1.0 0.6   # the report's MLP-Ego-IC
$PY scripts/run_policy.py --model weights/v2_mlp28_control_seed0 --target 1.5 1.0 0.6   # MLP-Ego: the arm never starts
$PY scripts/run_policy.py --model weights/v2_gru28_patience40_seed4 --target 1.5 1.0 0.6 --start 1.0 -1.0 2.5
python3 scripts/evaluation/report_tables.py                                    # every reaching table of the report
```

`run_policy.py` takes any trained model in `weights/` (see [`weights/README.md`](weights/README.md)), `--start X Y YAW`
for another start pose, `--dead-band 0.45` for the +DB variant and `--headless` to run without a window.

## 3. The pipeline

| step | command | documentation |
|---|---|---|
| collect expert demonstrations | `scripts/collect_dataset.py --config experiment_specs/collection/<spec>.json --workers 16` | [01 data collection](docs/01_data_collection.md) |
| train/validation split by target cell | `scripts/data/make_train_val_split.py OUT.json DATASET [EXTRA ...]` | [01](docs/01_data_collection.md) 1.4, 1.9 |
| train a policy | `scripts/train_mlp.py`, `train_gru.py`, `train_transformer.py` (`$PYML`) | [02 inputs](docs/02_inputs_and_outputs.md), [03 models](docs/03_models.md) |
| evaluate in closed loop | `scripts/evaluate_policy.py --split BANK.json --model LABEL=RUN --workers 16` | [04 evaluation](docs/04_evaluation.md) |
| statistics over seeds | `scripts/evaluation/seed_stats.py EVAL_DIR [...]` | [04](docs/04_evaluation.md) |
| obstacle tasks | `scripts/obstacle/` | [07 obstacles](docs/07_obstacles.md); [results](docs/results.md) 2–3 |

**[`docs/usage.md`](docs/usage.md) is the task-by-task guide** (collect, split, train, evaluate, tables and figures,
obstacles: each with its command, parameters and what to choose), ending with what to run for every table and
figure of the report.

## 4. Repository layout

```
src/robot_learning/
  expert/controllers.py   the lecturer's base and arm controllers and stop rule (unchanged)
  simulation/             the lecturer's episode loop, shared by every consumer; interactive viewer
  data/                   collection (spec, scenarios, parallel workers, DART noise, storage), target grid, splits
  learning/               inputs (features.py), outputs, episode loading, metrics, MLP / GRU / transformer
  evaluation/             closed-loop protocol, rollout worker, adversities, task and base metrics
scripts/                  entry points (collect, train, evaluate, run_policy, simulate) and data/, evaluation/,
                          figures/, obstacle/ tools
experiment_specs/         expert rules, collection specs, training recipes, evaluation protocols, data-size subsets
weights/                  the 55 reaching and 8 obstacle models of the report (portable NumPy files)
results/                  evaluations/: summaries of the 85 reaching evaluations; obstacle/: per-scene rows of the obstacle tests
docs/                     one chapter per part of the report, results.md, usage.md, figures/
controlled_experiments/   the group's controlled experiments and obstacle-task code (own README)
final/robotnik_arm_task/  the lecturer's starter code, unchanged
tests/                    unit and integration tests
```

Generated data (`datasets/`, `runs/`) are not versioned; every dataset and run directory records its configuration,
the hashes of the code and robot assets that produced it, and a copy of the source.

## 5. External code, and what was modified or written from scratch

**External code used as is**

| source | what | where it is used |
|---|---|---|
| Lecturer's starter code, [LAB_SESSION_COMP_0245_2026_PUBLIC](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC), commit `cae4fb69` | the expert: base state machine, IK arm controller, stop rule (`robotnik_arm_control.py`) and the task script (`robotnik_arm_task.py`) | `final/robotnik_arm_task/` unchanged; `src/robot_learning/expert/controllers.py` is a **verbatim copy** of `robotnik_arm_control.py` |
| [RoboEnv](https://github.com/VModugno/RoboEnv) and its `simulation_and_control` library (lecturer) | PyBullet simulation interface, Pinocchio wrapper, robot models and configs, the pixi environments | imported, not modified (pinned commits in section 1) |
| Open-source libraries | NumPy, SciPy (`milp` for the target-grid holdout), PyBullet, Pinocchio, PyTorch (training only), Matplotlib | through RoboEnv's environments |
| The group's obstacle task (`controlled_experiments/obstacle_task/`, `collect_data.py`, `evaluate_policy.py`) | static-obstacle scenarios, lidar, expert, collector and evaluation harness | imported unmodified by our obstacle scripts, which wrap them |

**Modified (refactored from the lecturer's code, same behaviour)**

| file | change | check |
|---|---|---|
| `src/robot_learning/simulation/episode.py`, `setup.py`, `config.py` | the main loop and setup of `robotnik_arm_task.py` moved into one reusable episode engine (shared by the viewer, the collector and the evaluator); constants moved to `experiment_specs/expert/default.json`; added: optional start pose, `goal_reference: "start"` (base goal from the start pose, identical to the lecturer's at the origin), an action filter for DART noise, a policy hook and an evaluation horizon | `tests/test_episode.py` replays recorded lecturer episodes (actions, states and stop decisions identical) |
| `src/robot_learning/simulation/interactive.py`, `scripts/simulate.py` | the lecturer's interactive run on the shared engine | same tests |

**Written from scratch by the group**

| part | files |
|---|---|
| data collection: specs, deterministic scenarios, parallel workers, DART noise on the executed command (idea: Laskey et al. 2017), recording, provenance | `src/robot_learning/data/`, `scripts/collect_dataset.py`, `experiment_specs/collection/` |
| target grid with geometry-only holdout, splits, expert-success filter | `src/robot_learning/data/sampling.py`, `splits.py`, `scripts/data/` |
| policy inputs (egocentric transforms, clock, lidar) and outputs | `src/robot_learning/learning/features.py`, `outputs.py` |
| models and trainers: MLP (ignition copies, categorical wheels, data-size options), GRU, causal transformer, NumPy inference | `src/robot_learning/learning/`, `scripts/train_*.py`, `experiment_specs/training/` |
| closed-loop evaluator, metrics, adversities, statistics, tables and figures | `src/robot_learning/evaluation/`, `scripts/evaluate_policy.py`, `scripts/evaluation/`, `scripts/figures/`, `scripts/run_policy.py` |
| obstacle extensions: 10 s hold and base noise during the hold (wrapper of the group's collector), dataset converter, harness adapter, the moving-obstacle task and its one-cost avoidance expert (design informed by open-source local planners such as DWA, VO/RVO and TEB; no code taken from them) | `scripts/obstacle/` |
| the group's controlled experiments (copycat, base/arm split, DAgger, chunking, …) | `controlled_experiments/` (own README) |
| tests | `tests/` |

The starter script itself still runs as the lecturer wrote it: follow `final/robotnik_arm_task/README.md`
(`python robotnik_arm_task.py` with RoboEnv's `models/` and `configs/` reachable).

## 6. Working as a group

```bash
git clone https://github.com/Rodrigo30406/comp0245-final-project.git
git switch -c feature/your-task                 # work on a branch, open a pull request into main
git remote add upstream https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC.git   # the lecturer's updates
```

Keep datasets and run directories outside Git. The full research log (every experiment in the order it was run) is in
the `rodrigo` branch; this branch keeps only what the report uses.

## 7. Submission

Report: at most 6 pages (IEEE template) plus 1 page of contributions; in-person demonstration
(`scripts/run_policy.py` above). Deadline per the slides: 6 November 2026.
