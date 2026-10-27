# COMP0245 — Machine Learning for Robotics

Behaviour cloning for a Robotnik mobile base with a Kinova arm, MSc Robotics and
AI, UCL, 2026/27. The baseline predicts two side velocities and seven arm torques from 21 robot-state
and target inputs, then duplicates each side velocity into four wheel commands.

## Repository layout

```text
src/robot_learning/
  settings.py              # Shared .env loading and RoboEnv discovery
  expert/controllers.py    # Lecturer's expert controllers and stop criteria
  simulation/              # Shared episode engine, rules, setup and interactive view
  data/                    # Collection CLI, scenarios, schema, recording, provenance
  learning/                # Episode split, normalization, models and offline metrics
  evaluation/              # Paired autonomous rollouts and external task metrics
scripts/                   # Small launchers for simulation, collection and training
experiment_specs/expert/     # Experiment-wide expert rules
experiment_specs/collection/ # Dataset scenarios and explicit rule overrides
experiment_specs/training/   # Model and preprocessing recipes
experiment_specs/evaluation/ # Fixed-horizon task evaluation protocol
tests/                     # Data integrity and behavioural regression checks
docs/                      # Assignment brief, research plan and usage guides
datasets/                  # Generated local datasets; excluded from Git
```

All application code lives in one Python package. The interactive simulator and
collector use the same episode engine, expert rules and simulation setup. New functionality can be
refactored freely while preserving the expert's behaviour; see
[architecture and provenance](docs/ARCHITECTURE.md).

## Environment

Use the environment supplied by RoboEnv, with the repositories side by side:

```text
Machine Learning for Robotics/
├── RoboEnv/
└── Final Project/         # or comp0245-final-project/
```

Activate it and enter this project:

```bash
cd "/path/to/RoboEnv"
pixi shell
cd "../Final Project"
```

Alternatively, from this project prefix Python commands with:

```bash
pixi run --manifest-path "../RoboEnv/pixi.toml" python ...
```

`pyproject.toml` describes our Python package; it does not replace RoboEnv's
`pixi.toml` and `pixi.lock`. The launchers in `scripts/` work without installing
our package or adding dependencies to RoboEnv.

Optional local settings go in `.env`, using `.env.example` as a template.
The file is discovered relative to the project, independently of terminal cwd.
Terminal variables override `.env`; omitted settings use defaults. Leave
`ROBOENV_ROOT` unset for sibling detection, or set an absolute path to another
RoboEnv checkout. Relative roots are resolved from this project. No asset copies
or links are required.

## Open the simulator

With the RoboEnv environment active and your terminal in this project:

```bash
python scripts/simulate.py
```

Choose a target without editing Python:

```bash
python scripts/simulate.py --target 1.5 1.0 0.7
```

With no target argument, the previous interactive target `[0, 0, 0.1]` is retained.
Experiment-wide expert rules live in
[experiment_specs/expert/default.json](experiment_specs/expert/default.json): fixed
finger angles, 0.85 m reach limit, 0.01 m tolerance, original standoff table and
25 s base navigation timeout. Both modes inherit them. JSON overrides define a
separate experimental variant; targets remain episode inputs. Press **q** in the
simulation window to stop; close plots to finish. See the
[simulator guide](docs/SIMULATOR.md).

## Collect expert datasets

```bash
# Quick technical check, not complete training demonstrations:
python scripts/collect_dataset.py --config experiment_specs/collection/smoke.json

# Pilot: 64 uniform XYZ targets, nominal initial pose:
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --workers 4
```

`--workers N` runs up to N independent episode processes (default: 1). Parallel
collection requires headless mode; GUI collection uses one worker. Change N on
resume without changing the frozen scenarios or expert rules. Each collection
invocation records processes, workers and wall time in `executions/`. Numerical
libraries default to one thread per worker, respecting explicit environment values.

Each invocation creates a unique dataset with requested/effective configuration,
seeded scenarios, source snapshots, versions and hashes, episode trajectories,
metrics and logs. Datasets stay local; experiment specifications are versioned.

`--plan-only` saves scenarios without simulation; `--gui` shows the episodes;
`--resume datasets/NAME` continues a frozen collection with matching code and
environment. The collector's JSON/CLI settings control GUI and step cap rather
than the corresponding `.env` defaults. See the
[data collection guide](docs/DATA_COLLECTION.md).

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover deterministic sampling, independent episodes, state/action alignment,
terminal recording, subsampling, resume integrity, equality of interactive and
recorded runs, sequential/parallel array equality, worker crashes and Ctrl+C
recovery, and lecturer trajectories for success, stall, timeout and step cap. Re-run relevant behavioural tests when refactoring control or setup.

## Project documents

Documentation, comments, CLI messages and plot labels are maintained in English
for all three teammates. Historical experiment snapshots remain unchanged.

- [Assignment slides](docs/COMP0245-Assignment-2026-2.pdf).
- [Research plan](docs/RESEARCH_PLAN.md).
- [Experiment notes](docs/EXPERIMENT_NOTES.md).
- [Architecture and code provenance](docs/ARCHITECTURE.md).

## Collaboration and attribution

Work on individual branches and review pull requests into `main`. Our group
repository is private. The lecturer's repository can be added as `upstream` for
manual review of updates; changes are not synchronized automatically.

The expert and interactive task originate from
[V. Modugno's COMP0245 starter](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/cae4fb69386872ae73c601ce77e182aad7abe284/final/robotnik_arm_task).
The Git history is retained. The lab-only `week1/` and `week2/` folders were removed.
The original controls are in `expert/controllers.py`; shared physics setup was
extracted into `simulation/setup.py`. The original episode sequence now lives
in `simulation/episode.py`; `simulation/interactive.py` provides the CLI and plots.
Both interactive and collected episodes use the original pre-step stop measurement
after applying the action. Schema v2 distinguishes this stop decision from the
post-step terminal state; executed steps can slightly exceed the configured cap. Our collection pipeline, configuration loading,
launchers and tests are project additions. Details and behavioural preservation
checks are recorded in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

The assignment requires a maximum 6-page IEEE LaTeX report plus a maximum
1-page contribution statement. The slides state submission on 6 November 2026
and an obligatory in-person demonstration during 30 October–6 November. The
submission README must describe execution, outputs, external sources and our
modifications. Document data generation, training and evaluation here as they
are implemented.

## Train the linear baseline

Select one complete dataset, then run:

```bash
python scripts/train_model.py --dataset "datasets/DATASET_NAME"
```

The pipeline splits whole episodes (80/20), fits input/output scalers on training
only, trains a 21-to-9 linear regressor and compares it with a training-mean
predictor. Run artifacts and weights stay under ignored `runs/training/`.
See [training methodology, commands and first results](docs/BASELINE_TRAINING.md).
The initial training reports are offline. Autonomous evaluation is now available;
see [protocol and first results](docs/AUTONOMOUS_EVALUATION.md).

To train the two-hidden-layer MLP with the same data pipeline:

```bash
python scripts/train_model.py --dataset "datasets/DATASET_NAME" --config experiment_specs/training/mlp_baseline.json
```

See [MLP architecture, selection protocol and measured comparison](docs/MLP_BASELINE.md).
Both models preserve the same 21 inputs and reconstruct 11 actuator commands
from 9 independent outputs. Validation is split by episodes, including for
early stopping; no automatic row-level holdout is used.

Research progress, deviations from the initial plan, measured results and report
evidence are tracked in the [research log](docs/RESEARCH_LOG.md).
The [episode-count comparison](docs/DATASET_SIZE_COMPARISON.md) records the
planned 64/128/256/512 series, frozen splits and evaluation protocol. The corrected256 cohort has now been measured; the initial sampling pilot is documented in the
[stratified comparison](docs/SAMPLING_COMPARISON.md).

## Evaluate autonomous control

```bash
python scripts/evaluate_policy.py \
  --split runs/training/linear_pilot_20261025/split.json \
  --model linear=runs/training/linear_pilot_20261025 \
  --model mlp=runs/training/mlp_pilot_20261025 \
  --workers 16
```

The expert is included automatically. Each policy controls its own fresh simulator
for the same 40 s horizon; stable arrival and final distance are measured externally.
Use your own training run paths if the local ignored artifacts are absent.

[Base, arm-relative trajectories and stopping diagnostics](docs/BASE_ARM_STOPPING_DIAGNOSTICS.md)
separate navigation failures from arm geometry and command fidelity.

## Compare trajectories and locate early departures

The current evaluation report focuses on base navigation and whole-system task
performance, without an isolated arm score. Use dense initial logging with
`--config experiment_specs/evaluation/trajectory_diagnostic.json` when evaluating.

```bash
python scripts/report_subsystems.py \
  --evaluation-run runs/evaluation/trajectory_validation_20261025 \
  --output-dir runs/diagnostics/NEW_TRAJECTORY_REPORT
```

This generates per-scenario PNG/PDF paths, early/full-horizon error curves,
wheel-command plots and `trajectory_analysis.json`. Existing run directories
are never overwritten. See [method and measured onset](docs/BASE_AND_SYSTEM_TRAJECTORIES.md)
and the [figure index](docs/TRAJECTORY_FIGURES.md).


Current diagnostics: [baseline limitations](docs/BASELINE_LIMITATIONS.md),
[saturation and orientation](docs/SATURATION_AND_ORIENTATION.md), and the
[English figure gallery](docs/TRAJECTORY_FIGURES.md). Lecturer questions are in
[research plan §17.1](docs/RESEARCH_PLAN.md#171-questions-for-the-lecturer-command-limits-and-expert-rules).


The [full sampling-design series](docs/SAMPLING_SERIES.md) extends the stratified
pilot to 64/128/256/512 with shared and complete native validation.

The [patience-only extension](docs/PATIENCE_SERIES.md) repeats the capped uniform
512 MLP and extends both designs to 1024/2048 episodes with unchanged patience
and no fixed epoch cap. Collection, training and full held-out comparisons use
`scripts/compare_patience_series.py`; the guide records current execution status.

[PyTorch/CUDA training](docs/PYTORCH_TRAINING.md) is available through the same
training launcher with `--config experiment_specs/training/mlp_pytorch.json`.
The optional ML environment adds GPU support; exported models still run in the
existing NumPy simulator evaluator without PyTorch.

The [temporal-window experiment](docs/TEMPORAL_WINDOWS.md) compares instantaneous
and200ms observation-stack models (20ms spacing) on the frozen1024-episode stratified split,
including a parameter-capacity control and complete paired validation.

The [depth/shape screen](docs/MLP_DEPTH_AND_SHAPE.md) adds32→64→128 and
32→64→128→64→32 hidden layers with both input representations, preserving the
same data, optimization and full control evaluation.
