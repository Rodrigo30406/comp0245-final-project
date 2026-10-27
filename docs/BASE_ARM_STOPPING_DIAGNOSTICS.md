# Separate diagnosis of base, arm, and stopping

**Historical report:** By a later decision of Rodrigo, the [current report](BASE_AND_SYSTEM_TRAJECTORIES.md) focuses on base and full system, without isolated scoring for the arm. Here, previous methods and figures are retained with revised interpretations.

Rodrigo's decision, 25/10/2026: the global distance to the target is insufficient to understand the failure. Evaluate base trajectory and arrival, arm position, and stopping separately. Do not modify or retrain policies during this analysis.

## 1. Base: navigation and trajectory

At each physical step, the following are recorded:

- Distance XY to the **navigation position calculated by the expert**, not to the effector's target. The standoff causes these two positions to differ.
- Angular error relative to the navigation bearing, wrapped in [-π, π].
- Planar arrival: position ≤ 0.08 m and orientation ≤ 0.05 rad for at least 0.5 s; tolerances from the original controller's pose and external protocol persistence. This is a planar diagnosis, not a 3D stability certification.
- XY path length, arrival time, and drift after arrival.
- Actual velocity and rotation during the final 0.5 s window, without automatically converting a post-hoc chosen value into a "stopped" threshold.

Additionally, every 100 steps, the trajectory is compared against the same scenario's reference:
RMSE XY at equal times and orientation difference. This measures trajectory imitation and timing; a different path that arrives correctly should not be judged solely on this RMSE. The geometric distance between sampled paths ignores order and time, and must be accompanied by arrival, duration, and path length. Base height and inclination are stored so that a trajectory that ascends or rolls over does not appear incorrect when judged solely on XY.

## 2. Arm: global position and position relative to base

The effector and shoulder positions are recorded in the world frame, along with the **full orientation** of the base. The effector's position relative to the base is computed as:

`p_EE_base = R_base^T · (p_EE_world - p_base_world)`.

This removes base translation and rotation, including roll/pitch; using only yaw would be incorrect for trajectories that incline. The RMSE of that position is measured against the expert at the same times, in addition to the final local error. This allows one to observe whether the relative arm motion resembles the expert's, even if the base movement is poor. It does not demonstrate equality across all joints: a redundant arm may achieve the same end-effector with a different posture.

Distance from shoulder to target and fraction of steps within 0.85 m are also recorded. **This sphere is an expert rule**, a geometric diagnostic condition, not a feasibility test with joint limits, contacts, or orientation constraints. The global error EE–target and stable arrival as a result of the full task are preserved; transforming it to the base frame does not physically eliminate a navigation error. The distance from the EE to its target expressed in the same frame would remain the same norm, which is why a local comparison with reference is added.

Old traces only stored yaw and a scalar EE error. No roll/pitch or geometries were invented to reconstruct the arm: the same 39 scenarios/policies were repeated with enriched recordings. The previous batch is preserved to verify that the instrumentation did not alter common actions or states.

## 3. Stopping: order and motion are distinct diagnostic criteria

**Stopping order:** evaluate predictions when the expert sends zero in both wheels. Numerical threshold for classifying zeros: 1e-6 rad/s, not an acceptable tolerance for motion. Report median magnitude/P95 per side; do not divide by zero. A zero torque does not imply that a joint is stopped, and a stopped joint may require torque to remain in position. For this reason, these subsets of torques are called "zero commands," without labeling them as a physical arm stop.

In 34,111 validation rows with both expert wheels at zero, raw predictions without clipping on those states:

| Model | Mean magnitude right | Mean magnitude left |
|---|---:|---:|
| OLS | 0.516 rad/s | 0.540 rad/s |
| MLP | 0.291 rad/s | 0.349 rad/s |

**Effective movement:** in each rollout, measure XY velocity and base rotation velocity; for the arm, joint velocities and end-effector displacement within the same window. These are real simulator data, different from the predicted commands in expert states. The terminal window does not imply that the robot has reached its goal; it is reported alongside the position error.

## 4. Relative tolerances and zero handling

The command fidelity report is expanded to include relative tolerances of 5%/10%/20%/50% for non-zero commands. Zeros are described separately. The mixed curve is preserved:

`|prediction - expert| ≤ rtol·|expert| + atol_j`.

Exploratory floors are set at 0%/1%/5% of the training channel standard deviation, stored in physical units: comparing alternatives without arbitrarily fixing a single minimum. These are not safety margins nor stability tolerances calibrated for robustness. For channels that are nearly constant, the standard deviation may be a poor physical scale; review before interpretation. These curves are secondary and do not replace the primary training metric. The relative fractions in this report use all grouped rows; do not confuse them with the macro fractions per episode from the previous absolute report.

```bash
python scripts/report_command_fidelity.py \
  --run linear=runs/training/linear_pilot_20261025 \
  --run mlp=runs/training/mlp_pilot_20261025
```

Relative results/zero in `relative_and_stop_pooled`, under each model, in `runs/diagnostics/relative_stop_20261025/command_fidelity.json`.

## Interpretation limits

These indicators **localize symptoms, not isolate causes**: the arm and base are coupled, and arm torques can influence base motion. To attribute causality to one module, explicit controlled comparisons (e.g., hybrid variants) would be required—not inferred solely from a single curve. Such interventions were not performed here.

Comparison with the reference is a measure of imitation, not a requirement to exactly follow that path to solve the task. Both the curves and navigation geometry, reach, and overall success are preserved.

## First result and reproducibility

Enriched batch: `runs/evaluation/subsystems_validation_20261025/`, 39 rollouts, same models/scenarios and 16 workers. The 234 common trace matrices and original summaries exactly match those from the previous batch; no behavioral changes were introduced. Geometry and diagnostics have been added.

| Model | Base: planar stable arrival | Final XY error to base target, median | Median local imitation RMSE vs expert |
|---|---:|---:|---:|
| Expert | 13/13 | 0.0453 m | 0 m (self-reference) |
| OLS | 0/13 | 1.906 m | 0.757 m |
| MLP | 0/13 | 563.174 m | 0.873 m |

The extreme distance of MLP indicates divergence in the simulated system, not a physically plausible trajectory. Local RMSE uses synchronized traces every 0.1 s, then takes the median across episodes; arrival and reach fractions use all steps. At different temporal alignments, part of the imitation RMSE may be phase-shifted.

Median fraction of steps with target within the shoulder sphere of 0.85 m:
expert: 87.96%, OLS: 0.40%, MLP: 3.38%. This does not mean that those steps are physically reachable nor that the remaining steps cannot be reached after moving the base.

The median final XY velocity across episodes is 0.00231 m/s for the expert and 0.3829 m/s for OLS; MLP ~30.57 m/s expresses its simulated physical divergence and is not a plausible real-world platform velocity. The data is retained without being interpreted as a hardware assertion or filtered to remove the failure.

In these rollouts, navigation arrival fails and the local effector trajectory differs from the expert. This second difference **does not alone prove poor arm control**: a different base pose may require a different local placement to reach the same target. The case "base achieves arrival and only arm fails" is not observed, but causal failures within modules have not been isolated.

```bash
python scripts/report_subsystems.py \
  --evaluation-run runs/evaluation/subsystems_validation_20261025 \
  --output-dir runs/diagnostics/NUEVO_REPORTE
```

The report directory must be new. Trace and time-aligned hashes are verified; episode summaries and metrics are preserved. No re-simulation is performed. Tests include full transformation with tilt/translation, yaw wrapped, joint position/orientation co-continuity, and drift, in addition to the eight previous evaluator tests and three fidelity tests (including zero separation).

Evidence: `runs/diagnostics/subsystems_20261025/subsystem_analysis.json`, `instrumentation_audit.json`, report code and figure. The figure uses the first scenario from the manifest, not a subsequent selection based on result.

![Base and arm separately](../runs/diagnostics/subsystems_20261025/base_brazo.png)

## Expert dependencies and possible model separation

Audit of 25/10/2026, in response to Rodrigo's question about two regressors.  
`base_controller` uses planar base pose, navigation/bearing goal, time, and its phase memory; it does not directly query arm positions/velocities.  
The goal includes standoff based on target height. It updates `nav.done`, also via a 25-second timeout, which does not guarantee geometric arrival.

`Episode` passes `reach_active=nav.done` to the arm controller. Prior to this, the arm maintains `q_hold` with PD control and gravity compensation; after this, it generates reach via inverse kinematics (IK) and integrates `q_des`. Its kinematics uses base pose and joint configuration. Therefore, there is explicit logical coordination base→arm and physical coupling.

Two models (2 base outputs and 7 arm torques) represent a reasonable variant to compare, not implemented here. Separating outputs does not alone resolve phase memory or `q_des`: the baseline's 21 inputs do not contain these internal variables. Learned phase signals or temporal history are alternatives; using the true expert phase during execution would be a hybrid/oracle explicit variant, not silently adding it to the baseline. To properly attribute improvements to model separation, inputs, data, and parameter/training budget must be controlled, and subsystem and full-task metrics must be preserved. Training of this variant has not yet been decided.

## Correction of interpretation: base-arm compensation (25/10/2026)

Rodrigo observes that, from a different base pose, the arm can reach the goal with a different correct relative position. Therefore, the local RMSE relative to the expert reflects **trajectory/fidelity in timing**, not isolated arm task error. The previous values .757/.873 m do not alone indicate how poorly the arm performs for the base pose that was actually achieved. Do not alter these values or the historical JSONs; correct the interpretation and preserve this metric as a secondary diagnostic.

For each real base pose, transform **the goal as well**:

`g_base(t) = R_base(t)^T · (g_world - p_base_world(t))`.

The distance `||p_EE_base(t) - g_base(t)||` is identical to the global distance `||p_EE_world(t) - g_world||`. It is not a new causal separation of error; rather, it is the correct task-level question for the arm in its current state. Do not sum as scalars the base's planar error and the arm's RMSE against reference: compensations and orientation changes may exist.

Criteria for future comparisons, without retrospectively replacing current ones:

1. System primary metrics: stable arrival/error to the original target across all scenarios, horizons, and common denominators. A base that fails to bring the robot to the target remains a global failure.

2. Base: navigation geometry and path, indicating that the expert's standoff target is a reference strategy and not the only valid pose for solving the global task.

3. Arm: error to the actual target from its current pose, accompanied by temporal coverage of reachability conditions. It is not necessary to require that the base be exactly at its target if, from another pose, the target can be reached.

4. Reachability: maintain .85m as a labeled proxy, **without calling it a certified physical reachability or a physical maximum boundary**. For verified reachability, search for a valid joint configuration (forward kinematics/inverse kinematics and limits); failure to find one via a solver does not demonstrate impossibility.

5. If a conditional error related to "in reach" is presented, also show the fraction and duration of those intervals, scenarios covered, and the criterion used. Do not hide failures outside of reach, nor compare policies on distinct subsets as if they were causally paired experiments.

6. To isolate arm control: conduct a separate experiment with the same base fixed in a pose with demonstrated reachability for all policies. Declare protocol or physical changes, without confusing them with the full mobile robot system.

This refined definition is a methodological agreement under development for P2. The experiment with a fixed base and the kinematic verification of reachability have not yet been implemented or executed. Current reports describe proxies and observed geometric results, not universal guarantees of reachability.
