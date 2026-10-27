# Command Fidelity: Tolerances, Error Queues, and Stability

Post-diagnosis dated 25/10/2026, requested by Rodrigo after observing that improving offline error did not result in stable control. No retraining was performed, no episodes were changed, and the primary experiment metric was not replaced.

## Different Questions

- **Imitation:** Given the same state, how much does each command deviate from the expert?
- **Control:** When executing successive commands, does the robot reach the target and maintain it?

MAE/RMSE address the first question through averages and penalties for large errors. NRMSE divides by a training scale; **0.203 does not mean 20.3% error nor 79.7% correctness**. The 66% reduction in NRMSE compared to OLS represents a relative improvement in that metric, not a rate of correct actions.

No aggregated imitation score guarantees stability on its own: state, phase, direction, temporal persistence, and dynamic response are all important. Stability and reaching still require rollouts and external criteria.

## Interpretable Metric: Percentage Within Tolerance

For each output $j$ and physical tolerance $\delta_j$:

`fraction_j = mean( |prediction_j - expert_j| <= delta_j )`.

The simultaneous fraction is reported per wheel/joint and for the full vector:

`fraction_simultaneous = mean( all nine differences are within their tolerances )`.

The simultaneous metric is directly computed over each transition; percentages are not multiplied assuming independence. Using nine outputs avoids double-counting the same lateral wheel commands.

The curve is saved for tolerance values of 0.05, 0.1, 0.2, 0.5, and 1.0, in rad/s for wheel velocities and N·m for torques. This is a **descriptive grid**, with the same numerical value but different units per output; it does not represent a physical equivalence between wheels and arm. An approval threshold or robot safety margin has not been selected. To establish such a threshold, sensitivity of control would need to be studied per channel/phase under declared conditions; neither actuator limits nor a standard defines this margin.

Two aggregations are reported: all rows combined, and per-episode fraction averages to give equal weight. The table below uses the second. As a tail diagnostic, the **95th percentile of absolute error** across all rows is added: 95% of the rows in that channel have error equal to or below this value, except for quantile interpolation conventions. It is not a maximum bound nor a guarantee for each episode. Errors from short phases are not masked by a single average.

## First comparison on expert validation

Same 13 episodes / 127,147 rows; original models without clipping, same normalizers. Predictions are made on **expert-recorded states**, not states visited by the learned agents. Therefore, this does not measure on-policy fidelity.

A tolerance of 0.1 is shown as an example for interpreting the curve; it was not selected to stabilize the controller. Complete results and alternatives are in `runs/diagnostics/command_fidelity_verified_20261025/command_fidelity.json`.

| Output | Linear within ±0.1 | MLP within ±0.1 | P95 linear error | P95 MLP error |
|---|---:|---:|---:|---:|
| wheel_right (rad/s) | 8.4 % | 33.0 % | 2.361314 | 0.768874 |
| wheel_left (rad/s) | 8.0 % | 31.7 % | 2.387404 | 0.893938 |
| arm_torque_1 (N·m) | 19.0 % | 65.1 % | 2.495235 | 0.454142 |
| arm_torque_2 (N·m) | 10.2 % | 57.3 % | 2.022565 | 0.786394 |
| arm_torque_3 (N·m) | 27.9 % | 67.7 % | 1.584372 | 0.485471 |
| arm_torque_4 (N·m) | 17.3 % | 55.8 % | 1.713836 | 0.547673 |
| arm_torque_5 (N·m) | 16.1 % | 73.8 % | 1.984917 | 0.487680 |
| arm_torque_6 (N·m) | 25.0 % | 66.5 % | 1.098529 | 0.402456 |
| arm_torque_7 (N·m) | 100.0 % | 90.2 % | 0.000391 | 0.145171 |

P95 in the units specified by output. Percentages with equal weight per episode; percentiles over grouped rows.

| Output tolerance (rad/s or N·m) | Nine simultaneous: linear | Nine simultaneous: MLP |
|---:|---:|---:|
| ±0.05 | 0.00 % | 0.23 % |
| ±0.1 | 0.00 % | 5.06 % |
| ±0.2 | 0.02 % | 26.61 % |
| ±0.5 | 3.80 % | 71.61 % |
| ±1.0 | 32.56 % | 91.64 % |

The MLP improves over OLS, but at ±0.1, its nine outputs coincide simultaneously only 5.1% of the time on average per episode. This necessitates the statement of "lower error"; it does not define a stability score. Even a high fraction could conceal persistent errors during critical moments.

To study states visited by the model itself, one would need to consult an expert reference. The expert in this project has memory (phase and joint references), so the comparison would require an explicit protocol for internal state; it is not sufficient to simply reuse any command from an original trajectory or to initialize the expert from scratch in each query. This diagnosis has not yet been implemented.

## Reproduction

```bash
python scripts/report_command_fidelity.py \
  --run linear=runs/training/linear_pilot_20261025 \
  --run mlp=runs/training/mlp_pilot_20261025
```

Uses the same split and normalizers; checks model/split and NPZ data hashes, compares with training mean, generates a new directory with JSON output, episode-level details, and a copy of the report script. Does not modify training artifacts and does not require new rollouts.

Specific tests: an incorrect channel is not masked among eight correct ones, exact equality at the threshold, and rejection of finite/non-aligned data or invalid tolerances.

The diagnosis is post-hoc and was not used for weight selection. Maintain MAE/RMSE and NRMSE with their original definitions; add percentages and tails as interpretable descriptions. [Autonomous Evaluation](AUTONOMOUS_EVALUATION.md) and [Research Logging](RESEARCH_LOG.md).

Future extension: [separate diagnosis of base, arm, and stopping](BASE_ARM_STOPPING_DIAGNOSTICS.md), including full geometry, relative/zero commands, and terminal velocities.
