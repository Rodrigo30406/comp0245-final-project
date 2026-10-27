# Saturation and orientation: diagnosis P2

25/10/2026. Post-hoc analysis of the same 39 rollouts from  
`runs/evaluation/trajectory_validation_20261025`: 13 scenarios, expert, OLS and  
MLP frozen. No training, no repeated physics, no changes to control rules.  
Does not constitute an independent population nor a causal experiment.

## What we measure

The evaluator preserves the **raw** and **applied** orders. Clipping occurs when  
`abs(raw - applied) > 1e-9` in any channel. Achieving exactly the limit without clipping is recorded separately. Protocol limits: wheels ±5.272727 rad/s, torques ±30 Nm; these are declared command envelopes, not certified limits nor measured mechanical saturation. Nine learned outputs are expanded to eleven; the duplicated wheel outputs do not represent independent predictions.

Each channel and the wheel/torque groups have their first clipping observed at time **PRE-step**. Up to 2 seconds, every control is recorded at 1 ms intervals, so the start is exactly aligned with control resolution. After that, we retain only one control every 100: the first **recorded** clipping may occur after an unobserved transient, or even before the last control without clipping. Do not interpret the time between samples as a guaranteed bound on the true onset. `exact_from_episode_start` distinguishes these cases. The full fractions per channel, accumulated across **all** controls by the worker, are preserved separately; do not estimate them by counting these rows with unequal density.

Saturation figures show `max_j(abs(raw_j)/limit_j)` per group; 1 is the limit, and an X marks the first recorded clipping. The vertical axis uses symlog with a minimum of zero. This does not imply that all channels in the group are saturated. The lower panel shows the expert phase with `nav.done` prevailing as the done state. This is a reference from another run, not the phase of the learned model.

## Results

| First observed clipping | OLS | MLP |
|---|---:|---:|
| Wheels: episodes with clipping | 13/13 | 13/13 |
| Wheels: median onset | 0 s | 0.526 s |
| Torques: episodes with clipping | 13/13 | 13/13 |
| Torques: median onset (recorded) | 2.399 s | 0.350 s |

OLS clips wheel commands in 10/13 cases. All initial wheel clips in MLP are observed before 1 s with continuous recording. For MLP torques, 12/13 initial events occur within the dense segment; one is observed at 3.299 s. In OLS, several initial wheel and torque events fall within the sparse segment: their times are reported as observations, not as exact starts. The expert had no clipping in any of the controls, according to the original counters.

In `episode_000002`, MLP clips torques first at 0.347 s and wheels at 0.552 s. The heading error EE of 1 cm had already been recorded at 0.154 s, and the base XY error of 1 cm at 0.413 s. OLS already clips the right wheel from t=0. These differences precede the expert timeout of 25 s; their absence in the model does not alone explain the onset of failure. The temporal order does not establish causality between angular error, torques, dynamics, and saturation. Clipping also does not stop the robot.

## Orientation

Yaw arrows are added to the XY paths, and for the 13 scenarios, a figure showing expert/OLS/MLP yaw and signed angular difference relative to the expert is included. Positive indicates counterclockwise rotation when viewed from above. Two columns: 40 s and first 2 s. Continuous yaw is reconstructed using `unwrap` to visualize rotations; it may lose revolutions if the rotation exceeds 180° between samples, especially during rapid divergence with 100 ms sampling. Do not use it as an exact revolution counter. The signed difference uses the minimum angle in [-180°, 180°]; its unwrapping jumps are not instantaneous physical rotations. Yaw does not measure roll/pitch inclination.

## Evidence and Reproduction

Final local report:
`runs/diagnostics/saturation_orientation_verified_20261025/trajectory_analysis.json`
(schema 3, trace/hash results, snapshots of the three modules).  
All scenarios include PNG/PDF of orientation and saturation, in addition to the previous figures with arrows added. Run files are ignored by Git.

- [Orientation, scenario 2](../runs/diagnostics/trajectories_english_20261025/orientation_episode_000002.png)  
- [Saturation, scenario 2](../runs/diagnostics/trajectories_english_20261025/saturation_episode_000002.png)  
- [XY Path with Orientation, scenario 2](../runs/diagnostics/trajectories_english_20261025/base_trajectories_episode_000002.png)  

```bash
python scripts/report_subsystems.py \
  --evaluation-run runs/evaluation/trajectory_validation_20261025 \
  --output-dir runs/diagnostics/nuevo_reporte
```

The initial export `saturation_orientation_20261025` is retained as a draft:  
the verified version corrects the lower bound of the ratio axis and shows "done" from the expert flag. Its numerical metrics match. To save rendering time, the verified export reuses the figures from unmodified families from the draft; the procedure is implemented in its generation script.

Tests: five trajectory tests, including a new one that distinguishes touching a limit from clipping, ignores the initial placeholder, and preserves the scattered recording uncertainty. P2 remains in diagnostics; no supervisor was implemented, and the failure was not exclusively attributed to memory, base, or arm.


The linked figures now use the English presentation export in
`runs/diagnostics/trajectories_english_20261025`. Historical reports and snapshots
remain unchanged; numerical analysis is identical. See the
[English gallery](TRAJECTORY_FIGURES.md) for all 13 cases.
