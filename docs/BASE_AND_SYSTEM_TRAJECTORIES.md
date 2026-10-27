# Current Metrics and First Deviations from Trajectory

Decision by Rodrigo, 25/10/2026: focus P2 on **base and full system**, remove isolated arm scoring and identify when executions diverge. No policy was trained or tuned. Historical arm reports are preserved with their interpretation limits and are not included in the active report.

## Task Metrics for Future Experiments

| Level | Current Metrics | Interpretation |
|---|---|---|
| Full System | Arrival ≤1 cm within 0.5 s; secondary components ≤2/5 cm; final distance/min distance and integral of temporal distance to original target; terminal motion | If the robot solves the task and maintains its behavior |
| Base | XY distance and bearing error to navigation target with standoff; planar arrival at 8 cm / 0.05 rad within 0.5 s; path length, time, and drift; terminal velocity/rotation | Quality of the navigation strategy reference, not a unique final pose achievable |
| Trajectory | XY/yaw/Z separation between base and end-effector position versus expert at equal times | Trajectory imitation and timing, separate diagnosis of success |
| First Deviation | First observations with XY separation of 1 mm / 1 cm / 10 cm, yaw of 0.1° / 1° / 5°, and global EE separation of 1 mm / 1 cm / 10 cm; first command difference | Describe where the discrepancy begins, without treating thresholds as stability certification |

No RMSE of arm relative error is calculated as an indicator of isolated control. The end-effector error to target is retained as the **system-wide result**. Improved navigation may help, but does not guarantee improvement of this error on its own: we will verify this across both metric families. Do not sum base and arm errors as scalars, nor ignore task failures due to the target being out of reach.

The base is compared both to its navigation target and to the expert trajectory. An execution may follow a different trajectory and still solve the task; therefore, trajectory RMSE does not replace arrival/error to the original target.

## Logging for Detecting First Error

New recipe: `experiment_specs/evaluation/trajectory_diagnostic.json`, same 40 s horizon and rules, with **logging of each step at 1 ms up to 2 s**; thereafter, every 100 steps (0.1 s). Physics and control continue at 1 ms throughout. Sampling only changes storage. The following are explicitly logged:

- Previous state/time step prior to the step that generated each command.  
- Raw command from the model and the command applied after clipping limits.  
- Global state, orientation, and position of the effector after the step.  
- Expert phase/nav.done used solely for diagnostic purposes; in models, it is shown as "not available" and never used as input or as an auxiliary expert calculation.

The initial recording contains the startup state and placeholder zero commands:  
`has_action=False` prevents confusing them with real predictions. The first command is decided at t=0 and produces the state after t=0.001 s.

The **first numerical difference in command** (threshold of 1e-9 in its units) is separated from the **first observable trajectory separation** at different scales. Commands from different runs are computed on distinct states after the first step: not attributing all subsequent delta to supervised error on the same input. The equality of the input for the first command is verified.

Each detection indicates a sample, the previous time, and the separation between samples. Outside the dense segment, there may be unobserved prior transients; the time is the first **observed** crossing, not a guarantee of the first physical continuous crossing. The arrival/task metrics from the evaluator still use every step, not just traces.

Since the temporal grid is uneven, trajectory RMSE/MAE are integrated using time steps (`trapezoid / duration`), not by averaging rows: giving 2000 rows to the first 2 seconds should not make them weigh more than the following 38 seconds. After 2 seconds, the integral of trajectory separations between runs is an approximation with samples every 0.1 s; the internal task error uses all steps.

## Replay the simulation and generate graphs

From the project with RoboEnv activated:

```bash
python scripts/evaluate_policy.py \
  --split runs/training/linear_pilot_20261025/split.json \
  --model linear=runs/training/linear_pilot_20261025 \
  --model mlp=runs/training/mlp_pilot_20261025 \
  --config experiment_specs/evaluation/trajectory_diagnostic.json \
  --workers 16
```

```bash
python scripts/report_subsystems.py \
  --evaluation-run runs/evaluation/trajectory_validation_20261025 \
  --output-dir runs/diagnostics/NUEVO_REPORTE
```

The launcher retains its name for continuity, but its current report is based on base + system. JSON: `trajectory_analysis.json`, version 2. Does not include arm-independent scoring. `--no-plots` enables only metrics, and `--zoom-until-s` selects the window to be drawn (does not alter physical dynamics or metric definitions).

PNG and PDF files are exported for the summary and trajectories/errors, and PNGs of commands.  
There are plots for **the 13 scenarios**, without selecting only favorable examples:

1. XY trajectories of the expert versus each model: start and full 40 s duration,  
   linear axes with equal spatial scale, start/end and navigation target points.  
2. Separation of base position and yaw, global EE separation, and distance to target: full horizon and first 2 s, with logarithmic scale where helpful.  
3. Wheel commands applied: first 2 s and zoom of 0.25 s, PRE-step time.  
4. Summary of the 13 conditions: medians per time instant, P10–P90 spread, and base/system arrival times. The bands are not confidence intervals.

The enormous excursions of the MLP still represent system divergence, not physically plausible paths. They are not filtered to make curves look smoother. Zooms allow viewing the start without the MLP divergence hiding the reference trajectory.

## Dense diagnosis results, 25/10/2026

Run `runs/evaluation/trajectory_validation_20261025/`, 39 simulations and the same  
13 development scenarios. The **390 common trace matrices** exactly match the previous batch, and their task summary/saturation reports remain unchanged.  
Repetition is additional instrumentation, not an increase in independent scenarios.  
Each trace contains 2381 samples; 2000 initial controls were verified per rollout with PRE-step observations aligned. No new training or modifications to the expert were performed.

The models differ in their wheel commands starting from **t=0** across the 13 scenarios, with the same initial input as the expert. It is not sufficient to describe this as an error appearing only after long accumulation: initial disagreement already exists.  
Subsequent evolution occurs under the specific states of each execution.

Median time to first observed XY crossing relative to reference, across all 13 scenarios:

| XY Separation | OLS | MLP |
|---|---:|---:|
| 1 mm | 24 ms | 115 ms |
| 1 cm | 77 ms | 313 ms |
| 10 cm | 942 ms | 1484 ms |

Crossovers preserve in JSON their separation between samples: 1 ms in the first 2 s and up to 100 ms thereafter. These are descriptive thresholds, not definitions of stability nor guarantees of exact physical onset. Medians correspond to the first crossings of each scenario, not to the crossing of the median curve in the figure.

### First scenario from the manifest: episode_000002

The first is used by fixed order, not by visual appearance of results. Commands applied at t=0, rad/s per side:

| Policy | Right | Left |
|---|---:|---:|
| Expert | +3.000 | −3.000 |
| OLS | +5.273 | +3.821 |
| MLP | +2.182 | −2.206 |

OLS already commands both wheels forward where the expert rotates. Its right wheel is clipped from a raw prediction of 5.760. The MLP preserves the sign of rotation, but modifies its magnitudes. This describes commands, not solely attributes the subsequent divergence to that difference.

In this case, OLS achieves XY separation of 1 mm at 24 ms and 1 cm at 79 ms; MLP reaches 1 mm at 110 ms and 1 cm at 413 ms. For MLP, the **global** effector trajectory is already 1 cm away from the reference at 154 ms, prior to base XY separation. Therefore, assuming improved XY separation is sufficient to correct the entire system is invalid; orientation, height, and dynamics also change. Causality has not been isolated from this observation.

**40 PNG and 27 PDF files** were generated: three figures per scenario and one global summary; PDFs of trajectories/errors/summary for export. [Index of all figures](TRAJECTORY_FIGURES.md).

![Summary of the 13 scenarios](../runs/diagnostics/trajectories_english_20261025/trajectory_overview.png)

![Spatial comparison of the first scenario](../runs/diagnostics/trajectories_english_20261025/base_trajectories_episode_000002.png)

[Errors and zoom of the first scenario](../runs/diagnostics/trajectories_english_20261025/trajectory_differences_episode_000002.png)  
· [Commands at the start](../runs/diagnostics/trajectories_english_20261025/initial_commands_episode_000002.png).

Verification: eight evaluator tests and four new metric/report temporal tests pass; short integration generates the four families of plots, ignores the placeholder command, and reproduces exactly the model inference with its PRE-step state.  
Metrics and snapshot code/JSON in the diagnostic directory; the repetition audit is in `dense_logging_audit.json` within the run.


English presentation copies of these figures, including orientation and saturation,
are indexed in [the current gallery](TRAJECTORY_FIGURES.md). Historical exports
remain unchanged; the new presentation uses the same frozen traces and metrics.
