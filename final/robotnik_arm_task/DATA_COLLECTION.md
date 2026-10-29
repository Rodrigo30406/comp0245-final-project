# Expert data collection (behaviour cloning)

Records expert demonstrations for the Robotnik base + Kinova arm reaching task: for every
simulation step (1 kHz) the robot measurements (model inputs), the expert commands (labels)
and the expert's internal state.

## Files

| File | Status | Purpose |
|---|---|---|
| `collect_data.py` | new | Runs the expert for one target or a random batch, saves one `.npz` per episode + `index.csv` |
| `analyze_index.py` | new | Figures, summary and data-quality check of one or more collected runs |
| `evaluate_policy.py` | new | Closed-loop evaluation of a policy (or the expert) on an evaluation set, see `EVALUATION.md` |
| `analyze_dataset.py` | new | Data-level analysis without a model (behaviour modes, phases, copycat baseline, what the base label depends on, arm label quality, task coverage); `results/data_analysis/` |
| `robotnik_arm_task.py`, `robotnik_arm_control.py` | course template, **unmodified** | The expert. `collect_data.py` imports its controllers and settings from them |
| `report.txt` | notes (Chinese) | Design decisions, experiments and findings collected for the report |

## Setup

1. Get the template folder `final/robotnik_arm_task/` from the course repo (`git pull`).
2. Put the files above into that folder.
3. Copy `models/` and `configs/` from RoboEnv into it (as in the template README).
4. Run everything from that folder with the RoboEnv pixi environment, for example
   `pixi run --manifest-path <path to RoboEnv>/pixi.toml python collect_data.py --help`.

## Quick start

```bash
# one episode with the GUI (target x y z in the world frame)
python collect_data.py --target 0.0 1.1 0.5 --gui --out-dir data/debug

# same, from another start pose (x, y, yaw in degrees)
python collect_data.py --target 1.0 0.5 0.6 --init-pose 1.0 -1.0 90 --gui --out-dir data/debug

# batch: random targets and start poses, half of the episodes with command noise
python collect_data.py --n-episodes 200 --workers 16 --init-radius 2.0 --init-yaw-range -180 180 \
    --noise-frac 0.5 --out-dir data/run_x

# continue an interrupted batch (same command + --resume): saved episodes are skipped
python collect_data.py ... --resume

# figures + summary (+ per-sample quality check) -> data/run_x/analysis/
python analyze_index.py data/run_x --check-files
```

A batch prints progress with an ETA. With 20 worker processes it runs about 1 episode per
second; each episode takes about 2-2.7 MB on disk.

## What is saved

`<out-dir>/ep_XXXX.npz`, one per episode. Per-step arrays (T = number of steps):

| Key | Shape | Meaning |
|---|---|---|
| `t` | (T,) | simulation time (s) |
| `base_pos` | (T, 3) | base position, world frame (m) |
| `base_ori` | (T, 4) | base quaternion (x, y, z, w) |
| `base_bearing` | (T,) | base yaw (rad), wraps at +-pi |
| `q_mes`, `qd_mes` | (T, 17) | joint angles / velocities: wheels 0-3, arm 4-10, fingers 11-16 |
| `wheel_cmds` | (T, 4) | **label**: wheel speeds `[front_right, front_left, back_left, back_right]` (rad/s) = `[rw, lw, lw, rw]` |
| `torque_arm` | (T, 7) | **label**: arm joint torques (N m) |
| `wheel_cmds_exec`, `torque_arm_exec` | (T, 4), (T, 7) | commands actually sent (= label + noise; equal to the label in clean episodes) |
| `q_des_arm`, `qd_des_arm` | (T, 7) | expert internal joint reference (not available to a learned policy at test time) |
| `phase` | (T,) | 0 align, 1 drive, 2 orient (base navigation), 3 reach (arm) |
| `nav_done` | (T,) | base navigation finished |
| `ee_pos` | (T, 3) | end-effector position, world frame (for evaluation) |

Plus `target` (3,) and `meta`, a JSON string with the episode settings and outcome (start pose,
status, final error, duration, noise parameters, seed, base goal, ...).

```python
import json, numpy as np
d = np.load("data/train/ep_0000.npz")
meta = json.loads(str(d["meta"]))
```

`<out-dir>/index.csv` has one row per episode: status, final error, duration, navigation
timeout flag, target (world frame and `r`, `theta_deg` relative to the start pose), start pose,
noise settings and seed. `config.json` stores the command-line settings of the run.

## How the expert is run

- Same controllers, gains and stop criteria as the template (imported, not copied). Running
  `collect_data.py --target 0.0 1.1 0.5` reproduces `robotnik_arm_task.py` exactly.
- **Start pose**: reset in code after the simulator is created (the config always starts at
  the origin facing +x; editing the shared config would race between parallel workers).
- **Modified expert planning**: the template takes the approach direction to the target from the
  world origin. It is taken from the robot's start position instead, which is identical when the
  robot starts at the origin.
- **Targets** are sampled uniformly in cylindrical coordinates relative to the start pose
  (r, bearing from the start heading, height z), then converted to the world frame.
- **Status correction**: the template's `check_stop` tests "stalled" before "reached", so an arm
  that reaches the target early and waits for the 8 s minimum is reported as "not reachable".
  Episodes ending within 10 mm of the target are recorded as `reached`.
- **Command noise (DART-style)**: in noisy episodes, Gaussian noise is added to the executed
  commands, while the clean expert commands are saved as labels. The data therefore shows how
  the expert corrects deviations.
  - Each noise sample is held for 0.3 s. Noise drawn independently at 1 kHz averages out and
    barely moves the robot.
  - Base noise is added on (v, w) and only while navigating, because the expert never corrects
    the base after navigation is done. It fades out within 0.5 m of the base goal. Arm torque
    noise fades out within 0.15 m of the target while reaching.
  - Defaults (`v` 0.02 m/s, `w` 0.05 rad/s, torque 2 N m) were chosen by experiment. With the
    earlier `w` = 0.2 rad/s the base often circled around its goal and timed out (29/40 reached,
    against 40/40 with the defaults). The defaults still deviate the base path by about 3 cm
    (median of the per-episode maximum).
- **Self-collision stays off** (`collision_enable: 0`, as in the template). With it on, the base
  cannot turn on the spot and drives away instead.

## Collected datasets

All runs use theta -180..180 deg, max 40000 steps and log every step.

| Run | Episodes | Seed | Targets | Start pose | Noisy | Purpose |
|---|---|---|---|---|---|---|
| `data/probe` | 200 | 0 | r 0.1-4.0, z 0.0-1.6 | origin | 0% | find the expert's working range |
| `data/pilot` | 100 | 1000 | r 0.1-3.0, z 0.2-1.2 | radius 2 m, any yaw | 50% | pilot run |
| `data/train` | 2000 | 1 | r 0.1-3.0, z 0.2-1.2 | radius 2 m, any yaw | 50% | training + validation |
| `data/test` | 300 | 100 | r 0.1-3.0, z 0.2-1.2 | radius 2 m, any yaw | 0% | final evaluation only |
| `data/dev` | 300 | 200 | r 0.1-3.0, z 0.2-1.2 | radius 2 m, any yaw | 0% | closed-loop evaluation: development decisions |
| `data/hard` | 200 | 201 | r 0.1-1.6, z 0.2-1.2 | radius 2 m, any yaw | 0% | closed-loop evaluation: short drives |
| `data/confirm` | 300 | 202 | r 0.1-3.0, z 0.2-1.2 | radius 2 m, any yaw | 0% | closed-loop evaluation: confirming adopted changes |

The last three are evaluation sets for `evaluate_policy.py`; their roles and commands are in `EVALUATION.md`.

Commands (reproduce the data exactly):

```bash
python collect_data.py --n-episodes 200 --workers 16 --seed 0 --r-range 0.1 4.0 --z-range 0.0 1.6 --out-dir data/probe
python collect_data.py --n-episodes 100 --workers 20 --seed 1000 --r-range 0.1 3.0 --z-range 0.2 1.2 --init-radius 2.0 --init-yaw-range -180 180 --noise-frac 0.5 --out-dir data/pilot
python collect_data.py --n-episodes 2000 --workers 20 --seed 1 --r-range 0.1 3.0 --z-range 0.2 1.2 --init-radius 2.0 --init-yaw-range -180 180 --noise-frac 0.5 --out-dir data/train
python collect_data.py --n-episodes 300 --workers 20 --seed 100 --r-range 0.1 3.0 --z-range 0.2 1.2 --init-radius 2.0 --init-yaw-range -180 180 --noise-frac 0.0 --out-dir data/test
```

Results so far:

- Expert success: train 95.2% (clean 98.4%, noisy 91.6%), test 298/300, dev 298/300, hard 194/200,
  confirm 294/300.
- No NaN in the data, wheel labels always match `[rw, lw, lw, rw]`, 0.15% of torque labels at
  the 30 N m limit.
- Samples per phase in train: drive 49%, reach 27%, align 17%, orient 7%.
- `data/probe`: the expert reaches every target with z in 0.2-1.2 m, whatever r or bearing.
  Above 1.4 m it never reaches.

## Using the data: recommendations

- **Filter episodes**: keep `status == reached` and `nav_timed_out == False`. Most failures
  (94 of 97 in train) are the expert circling around its base goal after a short drive. That
  behaviour should not be imitated.
- **Split train / validation by episode**, never by time step: neighbouring samples at 1 kHz are
  near-duplicates.
- **Downsample** (for example every 10-20 steps) for training; plot learning curves against the
  number of episodes.
- Base labels: convert the 4 wheel speeds to (v, w) with the skid-steer model in
  `base_controller` (`wheel_radius` 0.11, `wheel_base_width` 0.66).
- Encode the yaw as (sin, cos), since it wraps at +-pi.

## Known limitations

- The expert sometimes circles around its base goal on short drives (r < 0.5 m or 0.8-1.6 m),
  in about 5% of episodes. A cloned policy will probably inherit this, so compare against the
  expert on the same test episodes.
- `final_error` is not a precision measure: any episode longer than 8 s stops as soon as the
  error drops below 10 mm.
- With self-collision off, the arm may pass through the robot body. Low targets close to the
  robot should be checked in the GUI.
- `data/probe` was collected before the status correction. `analyze_index.py` re-labels its 3
  affected episodes (final error < 10 mm) as reached.
