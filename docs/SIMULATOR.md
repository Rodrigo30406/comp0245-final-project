# Run and explore the expert

The entry point is `scripts/simulate.py`. The simulation code is located at
[`src/robot_learning/simulation/interactive.py`](../src/robot_learning/simulation/interactive.py)
and the controllers are in
[`expert/controllers.py`](../src/robot_learning/expert/controllers.py).
The organization may change; the equations, parameters, and behavior of the
expert must be preserved and verified.

## Preparation

Install RoboEnv and its submodules according to its README. Keep it alongside the project,
or configure `ROBOENV_ROOT` in `.env`. Assets are located directly in RoboEnv via `conf_file_path_ext`;
no copies or symlinks are created.

From a terminal:

```bash
cd "/home/rodrigo/Documents/MSc Robotics and AI/Machine Learning for Robotics/RoboEnv"
pixi shell
cd "../Final Project"
python scripts/simulate.py
```

With the environment already activated, only the last command is needed. Without activating it, from the
root of the project:

```bash
pixi run --manifest-path "../RoboEnv/pixi.toml" python scripts/simulate.py
```

The launchers resolve the package `src/robot_learning` without requiring additional installation.
If run from another directory, specify the path to the launcher. `.env` is always searched in the root of the project.

## Test targets

Pass the target as input to the episode, without editing Python:

```bash
python scripts/simulate.py --target 1.5 1.0 0.7
```

Without `--target`, the target that Rodrigo was testing is retained: `[0, 0, 0.1]`.
The common procedure is defined in [`simulation/episode.py`](../src/robot_learning/simulation/episode.py);
`interactive.py` adds arguments, messages, and graphics.

## Experiment rules

[`experiment_specs/expert/default.json`](../experiment_specs/expert/default.json)
contains the teacher's parameters: fingers `[0.1] × 6`, reach `0.85 m`,
tolerance `0.01 m`, base timeout `25 s`, and original standoff table.
These remain unchanged throughout all episodes of an experiment.

The standoff relates heights `[0.1, 0.5, 1.1, 1.25]` with horizontal separations
`[0.8, 0.65, 0.65, 0.35]` from the base center. It is interpolated linearly;
outside the range, the endpoint is preserved. It does not guarantee joint reachability.

For an experimental variant, create a JSON file with only the changes and pass it via `--config path/variant.json`.
This JSON supports episode rules, not collector fields such as `episodes` or `target_box`.
Example content:

```json
{"base_controller": {"nav_timeout": 30.0}}
```

That example serves as a future comparison; the base value remains 25 s.

The base navigates first, and the arm attempts to reach the target. After completion, graphs are displayed. Pressing **q** within PyBullet stops the execution; closing the graphs ends the program.

The main inputs are base position/bearing, arm positions/velocities, and target. The expert outputs wheel velocities and torques for seven arm joints; the fingers maintain a fixed posture.

## Local Configuration

Copy `.env.example` to `.env` if desired. Terminal variables take precedence. `ROBOENV_ROOT` locates RoboEnv; `ROBOENV_HEADLESS=1` runs without GUI; `ROBOENV_MAX_STEPS` sets the stopping threshold in steps. The loader uses the standard library and supports one-line assignments, quotes, and comments, without shell variable interpolation.

For a brief test from the project root:

```bash
ROBOENV_HEADLESS=1 ROBOENV_MAX_STEPS=100 python scripts/simulate.py
```

The arguments `--headless` and `--max-steps` take precedence over the JSON; the JSON takes precedence over `.env` for these options. The target is passed separately via `--target`. The base navigation timeout is independent of the total step threshold. The teacher checks for stopping after applying an action, using the previous measurement: the executed number of steps may slightly exceed `max_steps`, also due to clock rounding. The summary distinguishes the decision instant from the final physical state.  
Details are in [Experiment Notes](EXPERIMENT_NOTES.md).

## Data Collection

For dataset registration, another entry point is used:

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json
```

It uses exactly the same episode engine, rules, and stopping criteria. The collector adds scenario sampling, logging, and storage; it does not include any additional control loop. Targets come from the collection JSON, not from the target selected in the interactive interface. See the [Data Collection Guide](DATA_COLLECTION.md).
