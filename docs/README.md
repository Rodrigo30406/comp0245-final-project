# Documentation

**All result tables of the report: [results.md](results.md)** (reaching, static obstacle, moving obstacle).
**How to use the code, task by task, and what to run for each table and figure: [usage.md](usage.md).** Setup and quick start: the main
[README](../README.md).

Reading order, following the structure of the report. Each chapter states the criteria behind the choices, the evidence
(v2 data, five seeds unless marked), and the commands that reproduce it.

| # | chapter | status |
|---|---|---|
| 1 | [Data collection](01_data_collection.md): DART noise, randomised targets and starts and their bounds, data size, expert-failure filter, the hold | done |
| 2 | [Inputs and outputs](02_inputs_and_outputs.md): the 21 observations, the egocentric transforms, clock, what is not given (copycat), the 9 outputs | done |
| 3 | [Models](03_models.md): the progression, MLP / GRU / transformer, training recipe, efficiency, reproduction commands | done |
| 4 | [Evaluation](04_evaluation.md): closed-loop protocol, metrics, banks, offline error vs closed-loop success, commands | done |
| 5 | [Arm ignition](05_arm_ignition.md): the arm that never starts, the ignition copies, memory | done |
| 6 | [Base drift](06_base_drift.md): park, creep and restart; dead band, categorical wheels, memory | done |
| 7 | Static obstacle avoidance (lidar) | to do |
| 8 | Dynamic obstacles | to do |

The full research log (every experiment in the order it was run) is in the `rodrigo` branch.
