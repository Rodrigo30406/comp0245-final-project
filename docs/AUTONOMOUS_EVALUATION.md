# Autonomous Evaluator: Closure of the P1 evaluation test bench

Protocol implemented on 25/10/2026. First, expert regressions, metric tests, and short integration were verified. Then, the first complete validation comparison was initiated, with the protocol saved prior to observing its results.

## What does "autonomous" mean?

Offline, the model is asked what command it predicts for a state recorded by the expert. In an autonomous rollout, the model acts upon **its own current state**:

`observe robot → normalize → predict 9 commands → denormalize → expand wheels → apply declared limits → advance physically → measure result`.

Expert states are not restored, errors are not corrected with expert actions, and expert commands are not mixed into the learned rollout. The objective remains the original, never the projected auxiliary of the arm.

## Shared episode engine and expert behaviour

`simulation/episode.py:Episode` is extended with explicit evaluation options: `policy`, `horizon_steps`, and `action_filter`. Without these, interactive and collector modules retain original actions, states, and stopping conditions; previous regressions continue to pass. No new physics/control loop has been created.

- Expert: the same controllers and memory, with commands calculated identically.
- Learned: the callable receives the current 21 inputs and returns 11 reconstructed commands via the adapter. The expert controllers are not called.
- During evaluation, all agents continue until a fixed horizon. For the expert, the **first stopping condition that would have been returned by the original script** is separately recorded, but the agent does not stop there. This protocol extends the observed time: it is neither a reproduction of the originally truncated episode nor a correction of its controller.

## Frozen Protocol for the First Comparison

| Element | Selection |
|---|---|
| Scenarios | The same 13 validation episodes from the historical split |
| Policies | Expert, OLS, and MLP 128×128 already frozen |
| New Data | None for training; only development evaluation |
| Horizon | 40 s simulated, integer number of physical steps according to effective dt |
| Primary stable-arrival criterion | EE–target error ≤1 cm sustained for at least 0.5 s, measured at each step |
| Secondary Thresholds | 2 cm and 5 cm, with equal duration requirements |
| Early Stopping | Only due to execution failure or user intervention; no termination/standstill based on reaching thresholds |
| Initialization | New simulation and new temporal memory per rollout; same seed/original state |
| Local Parallelism | 16 workers, one process per episode |
| Traces | Every 100 steps, including start and end; metrics use **all** steps |

A numerical equality of the initial observation is required at startup relative to the scenario source. Each policy uses the same physical/expert configuration saved in that dataset. Model artifacts, split, data, assets, and expert code are checked; a snapshot of the evaluator is preserved and provenance is verified upon completion of the batch. Code changes during a batch invalidate its global state. Do not resume or overwrite a directory: start a new run.

Stable arrival is detected at step resolution (1 ms in this configuration), without interpolation between samples. Its recorded time is **the end of the first sustained interval**, not the first fleeting threshold crossing. Continuing the simulation distinguishes having arrived from remaining within tolerance at the end. This does not mathematically guarantee continuous dwell between measurements.

## Command Limits: declared part of the evaluated policy

For learned policies, commands are clipped to the bounds defined by the expert parameters, with the same clipping applied to the reference:

- Wheels: `±(max_lin_vel + max_ang_vel * wheel_base_width/2) / wheel_radius`, **±5.272727 rad/s** in the current configuration.
- Arm torques: **±30 N·m** per joint.

The wheel bounds are a bound on the commands computed by the expert, **not a certification of hardware limits** nor the maximum observed in the pilot. No model was retrained with this filter. The offline metrics previously reported still describe predictions without clipping; the autonomous system evaluated includes this clipping explicitly. Save commands before/after and the fraction clipped for each of the 11 applied outputs. If the expert were clipped, review it; under normal conditions its formula already respects this bound.

Joint position out-of-bounds in the simulator is only counted for joints with bounds (`lower < upper`); continuous joints do not have such a limit. **The nominal posture of the expert already has joint 6 below its declared lower limit** (1.0 versus ~1.134 rad). For this reason, the initial condition and per-joint count are saved: do not attribute all violations to the model, nor use this diagnosis to silently discard the reference. The posture was not corrected and the dynamics were not changed.

## Task metrics and failures

Each episode preserves:

- Stable arrival and time of completion by tolerance.
- Final, minimum, and average distance over time (trapezoidal integral / 40 s).
- Final and average distance penalized to integrate failures without favoring short rollouts: declared penalty of 10 m; failed final distance set to `max(last, 10)`, and the rest of the horizon set to 10 m in the integral. Do not confuse this with measured distance.
- Valid rollout completion, failure exception/reason, and steps executed.
- Command saturation, prediction maxima, joint violations, and observed average prediction latency during that rollout.
- For the expert, its original first stop label, separated from external geometric criteria.

A failure invalidates the success of that rollout even if it entered the target earlier; its physical final distance to the horizon is `null`, not the last partial value presented as final. The partial minimum distance remains a diagnosed value. Summaries count failures in the denominator; the average of valid physical distances explicitly reports its denominator. Contact and self-collision events and electrical consumption are not yet evaluated. Command saturation and joint violation alone do not equate to these phenomena.

## Organization and usage

- `evaluation/metrics.py`: external metrics independent of expert phase/nav.  
- `evaluation/worker.py`: a rollout via the shared engine and current data.  
- `evaluation/cli.py`: validation selection, frozen models, provenance, dynamic queue, and summary. Reuses existing process/provenance utilities.  
- `experiment_specs/evaluation/validation.json`: shared protocol.  
- `scripts/evaluate_policy.py`: single launcher; select policies via arguments.  

With RoboEnv enabled, from the project root:

```bash
python scripts/evaluate_policy.py \
  --split runs/training/linear_pilot_20261025/split.json \
  --model linear=runs/training/linear_pilot_20261025 \
  --model mlp=runs/training/mlp_pilot_20261025 \
  --workers 16
```

The expert is included automatically. `--limit N` is used to verify integration with the first N scenarios; do not present it as a full evaluation. The default number of workers is 1 (portable). `Ctrl+C` interrupts, waits and collects all child processes, and preserves completed files; no evaluation resumption is implemented.

Each run under `runs/evaluation/` saves protocol, effective configuration, split, policies, model copies, provenance/snapshot, jobs, logs and NPZ traces, per-episode JSON results, and summary. The directories are ignored by Git. A terminated process does not equate to successful task completion: consult `valid`, metrics, and the batch global state. The first comparison does not modify weights or select a new model based on these rollouts.

## Evidence necessary to consider P1 closed

1. Original mode regressions preserved.  
2. Tests of persistence, integrality, failures, limits, horizon, and autonomous states.  
3. Integration with serialized real models and paired scenarios.  
4. Complete development batch executed with auditably traceable states, results, and errors.  

This closes the P1 infrastructure, not all P2 questions nor the confirmatory study. The 13 scenarios have already been used for development/checkpoint selection: they are not test cases. Method conclusions will require additional scenarios, seeds, and uncertainty analysis according to the [plan](RESEARCH_PLAN.md).

## First complete batch: results and verifications

Run `runs/evaluation/validation_20261025/`: 39 simulations (13 per policy),  
40,000 steps each; 16 workers, ~113 s real-time coordinator time.  
All completed the horizon with finite states. **"Valid" means numerically complete, not physically reliable, safe, or successful**: do not confuse this with the scientific validity of the simulator for extreme motions.

| Policy | Stable arrival at 1 cm | Within 1 cm at end | Median final distance | Mean final distance |
|--------|------------------------|--------------------|------------------------|----------------------|
| Expert | 13/13 | 11/13 | 0.00162 m | 0.00366 m |
| OLS | 0/13 | 0/13 | 2.442 m | 5.234 m |
| MLP | 0/13 | 0/13 | 926.575 m | 940.479 m |

**The extreme distances from MLP represent simulator/robot divergence, not a credible prediction of real robot displacement.** These are retained to make the failure visible, without post-hoc filtering of episodes. The MLP did not reach even stably within 5 cm in any of the 13 cases; neither did OLS. The models were evaluated using the declared filter, without expert intervention.

The reference demonstrates another distinction: arriving with a 0.5 s dwell does not guarantee remaining there at second 40. Of the 13 cases, all achieve stable arrival, but 2 end outside of 1 cm. This does not contradict the original tolerance error at termination: simulation continued beyond that point.

Median fraction of clipped commands (uniform average per episode/channel):  
wheel velocities OLS ~90.0%, MLP ~94.2%; torques OLS ~48.1%, MLP ~83.2%; expert 0%.  
The learned inference leaves the domain visited by the expert and frequently requires commands outside the observed range. The maximum z-score of inputs in subsampled traces increases dramatically; this is a descriptive diagnosis of extrapolation, not a calibrated probabilistic detector nor a causal attribution.

The origin of the failure has not yet been isolated—whether it stems from missing temporal information, state representation, sensitive scales, accumulated small errors, or control/contact interactions. **Do not train a new architecture and attribute the solution to a single cause without controls.** The 66% offline improvement of the MLP remains correct within its evaluation domain and does not imply closed-loop superiority; these are distinct questions.

Checks performed:

- 18 previous simulation/recollection/learning tests and 8 specific ones pass (local logs: `runs/evaluation_regression_tests.log` and `runs/evaluation_new_tests.log`).
- The original stopping points from the 13 experts exactly match historical metadata. 1,267 sampled transitions from their prefixes align with stored actions/observations at dataset precision.
- Identical initial states across policies, and artifacts/provenance verified.
- Short test with three policies and one/two workers: 18 identical trace matrices; raw commands learned exactly match the applied model to the **previous** state at each step.

Reproducible sources: `analyze.py`, `analysis.json`, `integration_audit.json`, `summary.json`, episode results, and `plot.py` under the run.

![Autonomous Results](../runs/evaluation/validation_20261025/evaluacion.png)

**P1 is closed for the nominal development test bench:** episode engine preserved, external metrics verified, models connected, and paired comparison executed. This does not declare models ready nor close P2: now it is necessary to diagnose failed trajectories before expanding data, inputs, or models. A contact/robustness evaluator and statistical confirmation will belong to subsequent phases.

Post-batch integrity enforcement: the coordinator now verifies protocol, configuration, job, policy, and copied model hashes before launching each new task and upon completion. A modified protocol test was rejected, and smoke tests for the three policies were fully completed (`smoke_guarded_20261025`). Control and metrics remained unchanged, no 39 rollouts were repeated; snapshots preserve the executed version. Historical protocol, job, and model hashes were also verified.

Post-expansion: [Separate diagnosis of base, arm, and stopping](BASE_ARM_STOPPING_DIAGNOSTICS.md), with complete geometry, relative/zero commands, and terminal velocities.

Active P2 Report: [base metrics/system and first deviations](BASE_AND_SYSTEM_TRAJECTORIES.md); [index of graphs for all scenarios](TRAJECTORY_FIGURES.md). The diagnostic recipe adds dense initial logging without altering control.
