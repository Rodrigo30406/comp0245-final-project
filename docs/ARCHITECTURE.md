# Architecture and preservation of the expert code

**Update: 24/10/2026.** The goal of the structure is to enable the project to be developed as a coherent system. The provided code can be moved, split, and refactored; what is preserved is its functional behavior, maintaining clear traceability of origin and modifications.

## Organization

```text
src/robot_learning/
├── settings.py                # Local configuration and RoboEnv path
├── expert/
│   └── controllers.py         # Base/arm control, stopping criteria, and fingers control
├── simulation/
│   ├── setup.py               # Shared physical setup and Pinocchio model
│   ├── config.py              # Loading and validation of common rules
│   ├── episode.py             # Single execution procedure for an episode
│   └── interactive.py         # Arguments, messages, and graphics
├── data/
    ├── cli.py                 # Orchestrate datasets and episode processes
    ├── config.py              # Validate recipes and sample scenarios
    ├── scheduler.py           # Coordinate processes, concurrency, and resumption
    ├── schema.py              # Fields, units, and transition contract
    ├── worker.py              # Records an episode executed by the common engine
    └── storage.py             # Logging, files, provenance, and summaries
└── learning/                  # ETL, split, model, and offline metrics

scripts/
├── simulate.py                # Launch the simulation
├── collect_dataset.py         # Start/resume dataset collection
└── train_model.py             # Train and validate a regressor
```

`experiment_specs/` contains versioned JSON recipes. `datasets/` contains local results ignored by Git. `docs/` explains methodology and execution. `tests/` verifies integrity, behavior, and data separation/normalization. `runs/training/` preserves local learning artifacts ignored by Git.

Launchers only locate the package and call its main function. Operations reside in `src/`; rules are loaded from the common JSON. Both modes consume the same `Episode.transitions()`. The expert does not depend on `data/`, and the simulation engine does not care about the collector.

`pyproject.toml` declares our package and its entry points. The scripts work directly with the RoboEnv environment; no installation of the package is required to use them. The environment definition remains in RoboEnv.

## Code Provenance

Source: [Valerio Modugno starter, commit cae4fb69386872ae73c601ce77e182aad7abe284](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/cae4fb69386872ae73c601ce77e182aad7abe284/final/robotnik_arm_task).

| Before | Now | Change |
|-------|-----|--------|
| `final/robotnik_arm_task/robotnik_arm_control.py` | `src/robot_learning/expert/controllers.py` | Moved; equations and parameters preserved |
| `final/robotnik_arm_task/robotnik_arm_task.py` | `simulation/episode.py`, `simulation/interactive.py` and `experiment_specs/expert/default.json` | Common procedure, separated presentation, and parameters extracted |
| Physical setup and Pinocchio initialization from starter | `src/robot_learning/simulation/setup.py` | Shared functions with the collector, same order and values |
| `final/robotnik_arm_task/project_settings.py`, added by us | `src/robot_learning/settings.py` | Shared configuration independent of simulator |
| `data_collection/`, added by us | `src/robot_learning/data/` | Unified package, separated schema/storage from worker |
| `collect_dataset.py`, added by us | `scripts/collect_dataset.py` + `data/cli.py` | Separation between launcher and logic |
| Task README | `docs/SIMULATOR.md` | Guide adapted to new commands |

The previous interactive target `[0, 0, 0.1]` is preserved as default value;  
it is changed via `--target X Y Z`. Collector targets still come from their JSON scenarios.  
No Python constants are edited to select points.

## What it means to preserve behavior

For the same configuration, state, internal history, target, and environment,  
the expert must generate the same wheel velocities/arm torques and phase transitions.  
The meaning of variables and the reachability policy, saturations, gains, integrated reference,  
or stopping criteria must not be changed merely to reorganize the code.  
The original control sequence is preserved in a single loop in `simulation/episode.py`.  
It maintains the original sequence: measure, compute base and arm, apply commands, update visualizer,  
check keyboard input, and evaluate `check_stop` using the previous measurement's time/error.  
The subsequent reading is used to store the transition and prepare the next iteration;  
it does not replace the measurement used to decide stopping. The clock accumulates `dt` as taught by the instructor,  
preserving its rounding limits as well.

The previous collector checked stopping based on the post-state and imposed a strict step limit. Now both modes follow the expert. Since their stopping condition is checked after applying an action, additional steps beyond `max_steps` can be executed: in the captured cases, threshold 25 executes 26 steps and threshold 40,000 executes 40,002. No additional stopping condition is added. `stop_decision_time_s` and `stop_decision_distance_m` allow distinguishing the decision measure from the post-terminal state. The data schema is upgraded to version 2.

## Experiment rules and inputs per episode

[`experiment_specs/expert/default.json`](../experiment_specs/expert/default.json) is the common source of expert rules: fingers `[0.1] × 6`, reach `0.85 m`, tolerance `0.01 m`, base timeout `25 s`, original standoff and other gains. A standalone copy is loaded; mutable navigation states, references, and stagnation states are recreated anew in each episode.

Collection recipes add count, seed, sampling, and logging. They can override rules for an explicit experimental variant, but those rules remain common across all episodes of that variant. Interactive mode allows the same type of override via `--config`; its `--target` is a separate input.

Each episode receives only its target and initial conditions. The collector spawns a new process per episode, with concurrency limited by `--workers` (1 by default), and freezes the entire effective configuration, including the common rules file and its hashes/snapshots. Changing a rule requires a new dataset; it is not silently mixed with an existing one.

If a future improvement changes the expert's behavior, it must be treated as an explicit experimental variant. Refactoring while preserving results is a separate operation and does not require keeping the original rules file frozen.

## Parallel coordination

`data/scheduler.py` maintains up to N active processes in headless mode. Each worker writes only its own episode; the coordinator is the sole writer of summary, manifest, and operational logs. The lock prevents two coordinators from operating on the same dataset. Seeds and scenarios do not depend on the number of workers. `executions/` records resources and times; changing workers upon resuming does not alter the scientific configuration. Cleanup on interruption stops and waits for all child processes. Individual failures are preserved and can be retried.

## How it is verified

- `tests/fixtures/lecturer_episode.json` captures the previous interactive loop, along with its source hash. It contains success cases, high-target with stagnation, distant target with timeout and short step limit. Actions, observations, steps, and stopping decisions are compared against the controller.  
- `tests/fixtures/expert_reference.json` also preserves checkpoints from the previous trajectory of the 8,001 transitions collected. It remains useful for verifying controls and states; it does not define the new semantics of stopping.  
- A test runs the interface and collector with the same target and an override, and compares all episode results, including the terminal state.  
- Data tests verify sampling, episode restart, alignment, terminal state, subsampling, hashes, and resumption.

These checks detect regressions in the evaluated cases; they do not constitute an exhaustive equivalence test for all states and platforms. The 13 tests pass in the local environment (24/10/2026), including the four original loop cases captured, matrix equality between one and two workers, resumption with a different number of workers, process crash, and Ctrl+C. The laws in `expert/controllers.py` have not been modified in this unification.

## Existing Datasets

Previous datasets retain their files, schema, original metrics, and snapshots. They are not rewritten to conform to new paths. They can be read normally. Provenance checking will reject resuming a dataset created with prior code using this new version; to complete an old dataset, one must recover its corresponding code/environment or start a new dataset.

## Supervised Learning (25/10/2026)

`learning/` implements data inventory, episode splitting, normalization, linear regression, and offline metrics. The launcher `scripts/train_model.py` uses `experiment_specs/training/linear_baseline.json`; artifacts are stored in `runs/training/`. It does not add another simulation loop nor modify the expert. The adapter learns 9 outputs and reconstructs the original 11 commands.  
[Methodology and results](BASELINE_TRAINING.md).

`learning/fitting.py` contains specific OLS/MLP adjustments; `learning/train.py` preserves the common orchestration. The MLP uses `model.MLPPolicy`, two ReLU layers, and a linear output; it does not introduce dependencies to the simulation.

## Autonomous Evaluation (25/10/2026)

`evaluation/` coordinates frozen policies across matched scenarios and evaluates tasks with a fixed horizon. It reuses `Episode`, which supports policy/horizon_steps/action_filter only as explicit options: original collection mode/GUI preserved. The launcher is `scripts/evaluate_policy.py`; rules are defined in `experiment_specs/evaluation/`.  
[Protocol, verification, and first results](AUTONOMOUS_EVALUATION.md).

## Optional PyTorch training backend

`learning/train.py` remains the shared inventory, split, normalization, provenance
and reporting entry point. `learning/fitting.py` dispatches to sklearn by default
or to `learning/torch_fitting.py` for `backend: pytorch`. The latter owns CUDA
batches, validation forward passes and Torch optimizer checkpoints; it exports
`learning/model.py`'s existing NumPy MLPPolicy. Evaluation workers and the expert
engine do not change. The optional ignored ML environment inherits the sibling
RoboEnv packages. See [PyTorch setup and decisions](PYTORCH_TRAINING.md).
