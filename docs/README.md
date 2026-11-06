# Documentation

Reading order, following the structure of the report. Each chapter states the criteria behind the choices, the evidence
(v2 data, five seeds unless marked), and the commands that reproduce it.

| # | chapter | status |
|---|---|---|
| 1 | [Data collection](01_data_collection.md): DART noise, randomised targets and starts and their bounds, data size, expert-failure filter, the hold | done |
| 2 | Inputs and outputs: the 21 observations, egocentric transforms, clock, lidar; the 9 independent outputs | to do |
| 3 | Models: MLP, GRU, transformer; training recipe and model selection | to do |
| 4 | Evaluation: offline error vs closed-loop success, metrics and banks | to do |
| 5 | Copycat / ignition: the trigger-row copies, two-network variants | to do |
| 6 | Base drift: the residual creep, categorical wheels, dead band, GRU, restart of the navigation | to do |
| 7 | Static obstacle avoidance (lidar) | to do |
| 8 | Dynamic obstacles | to do |

All results of the report use the **v2** data definition (see 1.8 for why v1 is not used). The full research log,
including v1, is in the `rodrigo` branch.
