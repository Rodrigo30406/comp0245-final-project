# Data Collection Pipeline of the Expert

Each collection command creates an independent dataset. It saves effective parameters, scenarios, seeds, code, versions, trajectories, results, and logs. Goals are generated per episode from the recipe, independently of the goal being tested in the interactive interface.

The code resides in `src/robot_learning/data/` and shares configuration, expert, and the same episode engine (`simulation/episode.py`) with the simulator. The scripts in `scripts/` are only launchers. See [Architecture](ARCHITECTURE.md) for structure and provenance.

Common rules are defined in  
[`experiment_specs/expert/default.json`](../experiment_specs/expert/default.json).  
Fingers, reach, tolerance, standoff, and timeout are kept constant within each experiment. A collection recipe can explicitly declare changes to these rules; `config.json` always stores all effective values. The common file is also included in the snapshot and provenance check.

## 1. Activate the environment and enter the project

With the standard structure:

```bash
cd "/home/rodrigo/Documents/MSc Robotics and AI/Machine Learning for Robotics/RoboEnv"
pixi shell
cd "../Final Project"
```

From here, `python` is used directly. Each command can also be run from the project using `pixi run --manifest-path ../RoboEnv/pixi.toml python ...`. No additional packages need to be installed.

## 2. Verify the pipeline with two very short episodes

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/smoke.json
```

This trial uses a threshold of 100 steps per episode: approximately 0.1 s with the current `dt`. It is useful for checking files and alignment; **it is not a complete demonstration dataset nor a test of goal reaching**.

For a full expert reference execution:

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/expert_reference.json
```

## 3. Pilot: 64 uniform goals in a box, fixed start

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json
```

The example proposes X/Y between −2 and 2 m and Z between 0.3 and 0.8 m. These bounds need to be validated by the expert; failures are not silently filtered. All these episodes are labeled as `development`: the pilot does not automatically become the final test.

For a smaller exploration:

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --episodes 4
```

To view episodes in a window, add `--gui`. Episodes are executed sequentially;  
`q` ends the visible episode and is recorded as `user_stop`.

### Collect in parallel

**Decision of October 25, 2026: use `--workers 16` on Rodrigo's machine, with cooler enabled.** After comparing 4/6/8/12/16/20/24, Rodrigo selected 16 as the equilibrium point: approximately 52.7 seconds per 64 episodes, average CPU usage of 49%, and maximum RSS of 3.90 GiB. Increases beyond this provide less than 10% speed gain per increase in worker count. This choice replaces the previous recommendations of 24 and 20. The CLI default remains 1 for portability; each machine must be measured. See [percentages, repetitions, and limits](MARGINAL_WORKER_SCALING.md).

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --workers 16
```

`--workers` must be a positive integer; its default value is 1. It allows up to N episode processes to run simultaneously, each with its own simulator and controller. With GUI, only 1 process is allowed. Objectives and seeds are generated beforehand; the order of completion does not alter the scenarios. Each process writes its episode and log; only the coordinator updates the manifest and summary.

This is a CLI operational option, not an expert rule nor a field in the collection JSON configuration. It can be changed upon resuming:

```bash
python scripts/collect_dataset.py --resume "datasets/FULL_DATASET_NAME" --workers 2
```

The `executions/` directory stores a JSON record for each collection/resume invocation: requested workers, maximum simultaneous workers, processes, start/end times, real time, CPU usage, host, and numeric thread variables. Full episodes are not repeated. To compare speed, measure real time with 1, 2, and 4 processes using the same scenarios; do not assume linear acceleration. Each process consumes RAM and CPU. Numerical libraries default to one thread per process; explicit values of `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` are respected and recorded.

`Ctrl+C` interrupts all workers and waits for their exit, allowing partial data to be saved when a worker reaches its interruption handler. Completed episodes are preserved. A failed worker is recorded and the others can finish; resumption restarts only the incomplete attempts. If the source fingerprint changes, active episodes are marked invalid and execution is halted.

## 4. Define a custom experiment

Copy a JSON from `experiment_specs/collection/` and edit it. That directory is versioned. Data is saved by default in `datasets/`, which is ignored by Git.

| Field | Meaning |
|------|--------|
| `name` | Descriptive name of the experiment |
| `episodes` | Number of episodes; not the number of training rows |
| `seed` | Seed that determines targets, initial states, and episode-level seeds |
| `split` | `development`, `train`, `validation`, or `test`; label for the entire dataset |
| `target_box.x/y/z` | Independent uniform sampling intervals, in meters |
| `targets` | Optional: explicit list of XYZ coordinates, one per episode; replaces box sampling |
| `initial_base` | `nominal` maintains the robot's startup file |
| `headless` | `true` collects without a window and without waiting for visualization |
| `max_steps` | Teacher termination threshold; checked after physics advancement and may be slightly exceeded |
| `record_every_steps` | Saves one out of every N transitions without changing control frequency |
| `base_controller.nav_timeout` | Maximum simulated navigation time, in seconds |
| `arm_controller` | Parameters of the arm controller, including `reach_limit` |
| `stop` | Tolerance, minimum time, and criteria for stagnation |
| `standoff`, `physics` | Base placement geometry and physical parameters applied by the starter |

The complete effective configuration is materialized in `config.json`, including values omitted from the input file. Unknown names and invalid values are rejected before dataset creation.

Priority: CLI overrides > JSON > collector defaults. The `.env` file is used to locate RoboEnv, but **its GUI options and step limits do not alter experiments**: these are determined by the JSON/CLI and are saved. Thus, a custom `.env` does not silently change duration or collection mode.

To extend runtime, check both budgets: 25 s of navigation and 40,000 global steps are independent. At `dt=0.001`, allowing up to 1,000 s globally requires `max_steps=1000000`; this does not force execution of the entire duration if the simulation ends earlier or stalls. Changing the maximum time constitutes another experiment.

To test random initial positions:

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/random_start_box.json
```

This variant only changes the initial X/Y of the base; it maintains the arm posture and initial robot velocities. The example fixes yaw to 0 rad; a range of `yaw` in radians can be provided for further study. The nominal Z of the base is preserved. Each episode saves the requested pose and the actual initial state. With the same seed, box, and number of episodes, fixed-start and random-start configurations share the same objectives, enabling comparisons.

## 5. Inspect scenarios before simulating and resume

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --plan-only
```

Prints a path to a folder with state `planned`: it already has configurations and scenarios, but does not yet have trajectories. To execute it or resume a interrupted collection:

```bash
python scripts/collect_dataset.py --resume "datasets/FULL_DATASET_NAME"
```

Do not modify saved JSON files to change an experiment: create a new collection with a different configuration. When resuming, configuration hashes, scenarios, schema, model assets, Python code, and package versions are verified. The checkout path may change if the content is identical. Completed episodes are verified and not repeated; previous failed attempts are archived in `attempts/` before being retried.

The code and environment fingerprint is also verified before and after each episode. If it changes during collection, the dataset is halted; affected active episodes are marked as invalid rather than silently mixing implementations.

`Ctrl+C` interrupts the dataset and preserves completed episodes. A sudden process termination may leave a `.collection.lock` file; check that no other collector is active before removing it and resuming. Do not run two collectors on the same directory.

Each new invocation creates a name with UTC timestamp, name, and unique suffix; it does not overwrite a previous dataset. `--output-dir` allows an alternative destination. CLI configuration, resume, and output paths are interpreted from the terminal.

## 6. Files of each dataset

```text
datasets/DATE_name_ID/
├── config.requested.json       # original requested file
├── request.json                # command and overrides of that invocation
├── config.json                 # effective complete configuration
├── scenarios.json              # target and initial condition of each episode
├── schema.json                 # names, dimensions, units, and alignment
├── provenance.json             # hashes, commits, local changes, packages, and paths
├── source_snapshot/            # sources and manifests used
├── dataset.json                # identity, state, and hashes of the protocol
├── executions/                # Workers, processes, and execution/restart times
├── summary.json                # global metrics and episode-level results
├── logs/episode_000000.log
└── episodes/
    ├── episode_000000.json      # result, times, error, and actual initial state
    └── episode_000000.npz       # transitions and terminal state
```

The entire RoboEnv and models are not copied into the project. The dataset snapshot  
preserves code and manifests, and the robot configuration file; it records hashes of  
files in its model directories. Assets continue to be resolved from RoboEnv.  
Reproduction requires having those assets and dependencies—just the code snapshot is not sufficient. The complete content of `.env` or other personal variables is not captured.

## 7. What a trajectory contains

```python
import numpy as np

with np.load("datasets/MY_DATASET/episodes/episode_000000.npz", allow_pickle=False) as data:
    X = data["observation"]  # N x 21
    y = data["action"]       # N x 11
    print(X.shape, y.shape)
```

Base inputs in order: base XYZ position, bearing, 7 arm positions, 7 arm velocities, and original target XYZ. Outputs: front right wheel speed, front left wheel speed, rear left wheel speed, rear right wheel speed; followed by the 7 arm torques. Exact units and names are defined in `schema.json`. State and action values are stored as float32, and times as float64.

Additional records include full orientation, states of the 17 joints, base velocities, causal velocity differences of joint velocities, previous actions and their validity indicators, end-effector and shoulder positions, errors, saturations, and internal expert diagnostics. **Do not use references, expert phase, or auxiliary target as inputs to the baseline without explicitly declaring that variant.** Acceleration is a numerically derived value, not an additional sensor.

Each row contains `observation` at `time`, the `action` decided upon it, and `next_observation` at `next_time` after a physical step. The following observations are reused as the state for the next step: sensors are not re-read to construct another reading at that instant. The terminal state is stored even if it does not align with the subsampling. Episodes' windows are not mixed.

With `record_every_steps=5` and `dt=0.001`, decisions are saved every 5 ms, but each transition still describes a 1 ms step. `previous_action` corresponds to the physical step prior, not to the previous row saved. Do not confuse these two time intervals when preparing windows or derivatives.

## 8. Interpreting results and quality

`status=complete` means the episode was executed and saved, **not that the objective was reached**. `termination` distinguishes between `success`, `stalled`, `max_steps`, and `user_stop`; execution errors remain as `error` with logs. Both successes and failures are preserved. `summary.json` reports success rates over complete episodes and over all requested episodes, to avoid hiding errors from the collector.

The current success metric uses `check_stop` with the original teacher's sequence: the action is applied, and then stop is checked **using measured time/error prior to the step**. Navigation is complete, error < tolerance, and time > `min_time`; the order of criteria is preserved, including stalled states before success. `stop_decision_time_s` and `stop_decision_distance_m` record this decision. `duration_s`, `final_distance_m`, `final_within_tolerance`, and terminal states correspond to the physical state after the last step. For this reason, success and final threshold are recorded separately.

The schema is now version 2. The clock accumulates `dt` as in the starter, and the `max_steps` threshold is checked after applying an action: additional steps may be executed; for example, a threshold of 25 with `dt=0.001` executes 26 steps in the reference case. No other limit is imposed that alters the original sequence. `configured_step_cap` records the threshold, and `steps` records the actual number executed. Version 1 datasets retain their previous semantics (stop on post-step state and strict limit), their snapshots, and their files intact.

**The 0.5 s permanence proposed for final evaluation has not yet been implemented**. `stalled` does not verify that the objective is physically unattainable.

Save `final_distance_m`, `minimum_distance_m`, `mean_distance_m` over the actually executed duration, time, navigation, and distance traveled to enable auditing of demonstrations. This mean does not replace a common horizon comparison between policies with different durations.

`scenario_id` identifies the target and the requested initial condition; repetitions share this identifier. The label `split` is assigned to the full dataset; **it does not automatically create a safe partition between multiple datasets**. Before training, group by scenario/episode and verify that no reserved conditions overlap. Demonstration selection and windowing are subsequent steps, preserving the raw data intact.

## 9. Collector Verification

From the root of the project with RoboEnv active:

```bash
python -m unittest discover -s tests -v
```

These tests check for invalid configurations, seeds, and reproducible prefixes, common objectives when changing startup, transition alignment, final state, independent restarts, subsampling, integrity, and recovery without overwriting. The short simulations in these tests are technical controls, not scientific results from the pilot.
