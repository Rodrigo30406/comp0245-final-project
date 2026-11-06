# Closed-loop evaluation (protocol v1.1)

How every policy, learned or the expert, is evaluated in closed loop, so that results from different
models, datasets and teammates can be compared episode by episode. The rules below were fixed on
2026-10-28, before any learned model was evaluated with them (pre-registration, see `report.txt`).
Protocol v1.1 (2026-10-29, [below](#protocol-v11)) keeps every v1 rule and every v1 result column, and only
adds secondary metrics and diagnostics.

## Files

| File | Status | Purpose |
|---|---|---|
| `evaluate_policy.py` | new | Runs a policy on an evaluation set and writes one result row per episode |
| `results/<run>/` | output | `rollouts.csv`, `summary.txt`, `config.json` (+ `logs/`, `traj/`, not versioned) |
| `tests/test_harness_v11.py` | new (v1.1) | Unit tests of the v1.1 metrics; `RUN_SLOW=1` adds an expert check against the v1 run |
| `tests/check_v11_against_v1.py` | new (v1.1) | Checks, episode by episode, that a v1.1 run reproduces every v1 column of a v1 run |

## Evaluation sets

All are collected by `collect_data.py` with the clean expert and the same task distribution as
`data/train` and `data/test` (theta -180..180 deg, z 0.2-1.2 m, start within 2 m of the origin, any yaw),
so the expert's own outcome and trajectory on every episode are already recorded.

| Set | Episodes | Seed | r (m) | Role |
|---|---|---|---|---|
| `data/dev` | 300 | 200 | 0.1-3.0 | every development decision (model choice, data settings, hyperparameters) |
| `data/hard` | 200 | 201 | 0.1-1.6 | short drives, where the expert's base and the noise experiments fail (start-to-base-goal distance up to 1.15 m) |
| `data/confirm` | 300 | 202 | 0.1-3.0 | each adopted change is re-checked here once, after the decision |
| `data/test` | 300 | 100 | 0.1-3.0 | final numbers only, opened once at the end (never used for a decision) |

```bash
COMMON="--workers 20 --z-range 0.2 1.2 --theta-range -180 180 --init-radius 2.0 --init-yaw-range -180 180 --noise-frac 0.0"
python collect_data.py $COMMON --n-episodes 300 --seed 200 --r-range 0.1 3.0 --out-dir data/dev
python collect_data.py $COMMON --n-episodes 200 --seed 201 --r-range 0.1 1.6 --out-dir data/hard
python collect_data.py $COMMON --n-episodes 300 --seed 202 --r-range 0.1 3.0 --out-dir data/confirm
```

`evaluate_policy.py` takes the target and start pose of each episode from the set's `config.json`
(re-drawn with the same seed, so the floats are exactly those the expert ran with) and checks them
against `index.csv`.

## Protocol v1

The same rules apply to every policy, the expert included. They do not depend on the expert's internal
state (`nav_done`, phase), which a learned policy does not have, and they replace the template's
`check_stop` (8 s minimum time, stall test before the reached test).

| Rule | Value |
|---|---|
| Simulation | exactly as in the data collection (1 kHz, floor friction 1.0, wheel friction/contact settings, motor gains, fixed fingers, start pose reset) |
| Actuator limit | arm torques clipped to +-30 N m (the expert's limit) for every policy; wheel speeds are applied as given |
| **Success** (`success`) | EE within **10 mm** of the target at some time t <= **40 s**, and the episode ends normally (`stop_reason == reached`) |
| Hold window | after the first reach the episode continues for **1 s** |
| Settled (`settled`) | EE stays within 15 mm during the whole hold window |
| Precision (`hold_error`) | mean EE error over the hold window (`final_error` of the data collection is not a precision measure) |
| Early stop `stalled` | before reaching, for 5 s: base speed < 1 cm/s, yaw rate < 0.02 rad/s, every arm joint < 0.05 rad/s, EE error not improved by 1 mm and above 30 mm |
| Early stop `tipped` | base roll or pitch > 0.5 rad |
| Other stops | `timeout` (40 s without reaching), `invalid_action` (NaN/inf command), `policy_error` (exception in the policy's reset or act) |

Every stop other than `reached` is a failure, including an error or a NaN command inside the hold window
(`t_reach` is still reported). The early stops only save simulation time; they must never end an episode
the policy would still have completed:

- The stall rule never fires within 30 mm of the target (a slow final approach may still succeed) and needs
  the robot to be still for 5 s.
- With the expert it never ends an episode the expert completes (see the verification below).
- **For every new model family**, run DEV once with `--no-early-stop` and check that no episode stopped as
  `stalled` succeeds without the early stop. Record the check in `report.txt`. If one does, the rule is
  wrong for that model: use `--no-early-stop` for it.
- A policy that uses `t` as an input could wait for a time-triggered event (the expert's base navigation
  times out at 25 s). Evaluate such a policy with `--no-early-stop`.

### Columns of `rollouts.csv`

- Outcome: `success`, `settled`, `t_reach`, `stop_reason`, `min_error`, `hold_error`, `hold_error_max`,
  `final_error`, `duration`, `n_steps`.
- Episode: `r`, `theta_deg` (relative to the start pose), `target_z`, `d0 = r - standoff(z)` (signed distance
  from the start to the expert's base goal, negative when the goal is behind the start), world-frame
  target and start pose, and the expert's outcome recorded with the template rules.
- Base diagnostics, measured against the expert's planned base goal (evaluation only, never given to
  the policy): `base_goal_dist_start / _min / _final`, `base_path_length`, `base_turn_total_deg` (large
  values mean circling).
- `arm_move_time`, `base_goal_dist_at_arm_move`: when the arm first leaves its start pose (> 0.05 rad), and
  how far the base still was from its goal (the expert only reaches after parking: a large distance means
  the policy mixes up the phases). v1.1: invalid for the expert (its hold posture sags more than 0.05 rad,
  so it always records about 0.58 s); use `arm_extend_time` instead, see Protocol v1.1.
- Commands: `torque_saturated_frac` (steps with a torque at the 30 N m limit before clipping), `wheel_cmd_max`.
- Efficiency: `policy_ms_mean`, `policy_ms_p99`, `policy_ms_max` (time of one `act` call, single thread).
- With `--compare-recorded`: `cmp_identical`, `cmp_first_diff_step`, `cmp_max_<signal>`.

## Running

```bash
# verify the harness with the expert (results/expert_v11_<set>/ are also the paired baselines; results/expert_<set>/
# are the v1 runs, kept as the reference of the v1 columns: do not overwrite them)
python evaluate_policy.py --policy expert --eval-set data/dev --workers 20 --compare-recorded --out-dir results/expert_v11_dev

# a learned policy
python evaluate_policy.py --policy my_policy.py:make_policy --policy-arg path/to/model.pkl \
    --eval-set data/dev --workers 20 --out-dir results/<model>_dev

# quick checks: a subset, or the GUI
python evaluate_policy.py --policy my_policy.py:make_policy --eval-set data/dev --limit 20 --workers 20 --out-dir results/tmp
python evaluate_policy.py --policy my_policy.py:make_policy --eval-set data/dev --episodes 7 --gui --out-dir results/tmp
```

Each episode runs in a fresh process (clean pybullet state), so the policy is created again for every
episode: keep model loading light. Workers run single-threaded (`OMP_NUM_THREADS=1` etc.). The policy is
also loaded once before the run starts, so a wrong path, factory or model file fails immediately.

The policy gets copies of the observations, the target and `info`: changing them in place is harmless.
Episodes recorded with command noise (when a noisy set such as `data/train` is used as an evaluation
set) are left out of `--compare-recorded` and of the comparison with the recorded expert, because that
expert executed noisy commands.

### Policy interface

```python
# my_policy.py
import numpy as np

class MyPolicy:
    def __init__(self, model_path):
        ...                              # load the model, scalers, feature settings

    def reset(self, target, info):
        ...                              # new episode; target (3,) in the world frame

    def act(self, obs):
        # obs: "t", "base_pos" (3,), "base_ori" (4,), "base_bearing", "q_mes" (17,), "qd_mes" (17,), "target" (3,)
        # same names and meaning as the recorded .npz files: build the features with the same function
        # as in training
        ...
        return wheel_cmds, torque_arm    # (4,) [front_right, front_left, back_left, back_right] rad/s, (7,) N m

def make_policy(arg):                    # arg: the --policy-arg string, None when not given
    return MyPolicy(arg)
```

Errors in `reset` or `act`, and non-finite commands, end the episode as a failure (`policy_error`,
`invalid_action`) with the message in `error_message` and the traceback in `logs/ep_XXXX.log`.

A model that predicts the base command as (v, w) converts it with `collect_data.wheel_speeds(v, w, params)`
and `params = BaseControllerParams(nav_timeout=25.0)` from `robotnik_arm_control.py`.

`info` holds the robot model (`dyn_model`, frame ids, joint limits), the time step and the start state.
The expert uses it; a learned policy that uses any of it takes an extra input and must say so in the report.

## How results are compared

- Report `success` with its 95% Wilson interval (printed in `summary.txt`). For 300 episodes the
  half-width is about 2.5 points at 95% success and 3.4 points at 90%; for 100 episodes, 4.5 and 6 points.
- Two policies (or a policy and the expert) are compared **on the same episodes**: 2x2 table of outcomes,
  exact McNemar test on the discordant episodes, paired bootstrap over episodes for the difference.
- Policies with training randomness: at least 3 training seeds. Average each episode's success over the
  seeds first, then bootstrap over episodes; also list the difference of every seed. Do not pool seeds into
  one McNemar test (the episodes are the same).
- A difference that is not significant is reported as "no difference larger than about X points could be
  detected", not as "no difference".
- Stratify by `d0` (the mechanism behind the expert's failures), not by `r`.

## Verification of the harness

The expert run by `evaluate_policy.py` (`results/expert_dev`, `expert_hard`, `expert_confirm`):

| Set | Identical to recording | Success (protocol v1) | Same outcome as recorded | Settled |
|---|---|---|---|---|
| DEV | 300/300 | 298/300 = 99.3% [97.6, 99.8] | 300/300 | 296 |
| HARD | 200/200 | 194/200 = 97.0% [93.6, 98.6] | 200/200 | 194 |
| CONFIRM | 300/300 | 294/300 = 98.0% [95.7, 99.1] | 300/300 | 292 |

- "Identical" means zero difference in base pose, joint angles and velocities, and commands at every
  recorded step (float32). Hence the simulation setup, the expert, and the regenerated targets and start
  poses all match the data collection.
- The 3 expert episodes stopped as `stalled` also fail without the early stop (`--no-early-stop`: timeout
  at 40 s).
- All 14 expert failures are the base circling around its goal (closest distance 8.0-8.5 cm, just outside
  `pos_tol`).
- The expert's step time is 0.17-0.20 ms (median), p99 about 0.41 ms.

These runs are the expert baselines for paired comparisons. Re-run them if the protocol changes.

## Protocol v1.1

Decided on 2026-10-29 (step 3 plan, `report.txt`), before any learned model was evaluated in closed loop.
**The primary metric is unchanged**: `success`, `settled`, `t_reach`, `stop_reason`, `hold_error`,
`hold_error_max` and every other v1 column keep their exact v1 meaning and values (two documented exceptions:
`error_message`, and the `--compare-recorded` columns, which now cover the whole run; see the rules below). v1.1
only runs longer after a reach and adds secondary metrics and diagnostics.

### What changed and why

- **10 s hold.** After the first reach (EE error < 10 mm) a v1.1 episode keeps running until t_reach + 10 s
  (v1: 1 s). The template's `check_stop` stopped at the reach and v1 stopped 1 s later, so a policy that
  reaches and then fails to hold, or drifts away, looked like a success. Teammate Rodrigo's models do exactly
  that: they leave the 5 cm band a median 4-13 s after arriving. The 10 s hold makes this failure visible
  (`held_10s` and the `hold10_*` columns).
- **Arm diagnostic.** `arm_move_time` (> 0.05 rad from the start pose) is invalid for the expert. Its hold
  posture sags 0.050-0.058 rad in j2, so all 800/800 v1 expert episodes record about 0.58 s. v1.1 adds
  `arm_extend_time` (Rodrigo's definition): the EE moves more than 0.10 m in the base frame from where it was
  at t = 0. `arm_move_time` stays unchanged for v1 compatibility, but do not use it.
- **Parking ring distance** `| ||target_xy - base_xy|| - standoff(z) |`. It does not depend on the expert's
  parking azimuth. Our models never see the start position, so they may park anywhere on the ring and still
  reach. The `base_goal_dist_*` columns (expert's parking point) stay as secondary diagnostics.
- **Stable arrival** (Rodrigo's success measure), so his and our results can be reported side by side.

### Rules of the extended window

- Until the v1 stop point the episode runs exactly as in v1, with the same stop rules before the reach. The
  v1 stop point is the end of the 1 s hold window for a reached episode, and the stop step otherwise. The v1
  row is computed there and frozen, including `n_steps`, `duration`, `final_error`, `min_error`, the base
  diagnostics, `torque_saturated_frac`, `wheel_cmd_max` and `policy_ms_*`.
- A reached episode then continues until t_reach + 10 s. Only three things end the extended window early:
  a policy exception, a non-finite command, or tipping (roll or pitch > 0.5 rad). Tipping is checked only
  after the 1 s window, so the first second is identical to v1. There are no stall or timeout rules in the
  extended window. Why it ended is in `hold10_stop`.
- Nothing in the extended window changes `success`, `settled`, `stop_reason` or any other v1 column, with
  two exceptions. (1) `error_message`: an exception in the extended window is written there as
  `extended hold: <error>` when the v1 part had no error. (2) The `--compare-recorded` columns, next bullet.
- Episodes that fail before reaching end exactly as in v1. Their new columns cover the run up to that stop.
- `wall_time` and the saved or compared trajectory (`--save-traj`, `--compare-recorded`) cover the whole run.
  The recorded expert stops at the template's stop, so `cmp_steps` (the common length) can be larger than in
  v1. `cmp_identical`, `cmp_first_diff_step` and `cmp_max_*` are computed over the whole v1.1 run, so they
  are a stronger check than in v1 (the extended hold is compared too, as far as the recording goes). For the
  expert they equal the v1 values (`cmp_identical` True in 800/800 episodes). For a policy that diverges from
  the recording only after the v1 stop point they can differ from the v1 values (e.g. `cmp_identical` False,
  `cmp_first_diff_step` inside the extended hold) while every other v1 column is unchanged.

### New columns of `rollouts.csv` (appended after the v1 columns)

| Column | Meaning |
|---|---|
| `hold10_stop` | why the extended window ended: `completed`, `policy_error`, `invalid_action`, `tipped`; empty if the episode never reached |
| `held_10s` | reached, the 10 s window after the first reach completed, and EE error < 15 mm at every step of it |
| `hold10_error_end` | EE error at t_reach + 10 s (NaN unless completed) |
| `hold10_error_mean`, `hold10_error_max` | over the window, from the reach step; over the part that ran if it ended early |
| `hold10_leave_time` | seconds after t_reach when the EE error is first >= 15 mm (NaN if never within the window) |
| `hold10_base_disp_max` | max planar distance of the base from its position at t_reach during the window (m) |
| `hold10_yaw_change_max_deg` | max abs(yaw change) from t_reach during the window |
| `hold10_wheel_cmd_max` | max abs(wheel command) during the window (rad/s) |
| `stable_1cm_time`, `stable_2cm_time`, `stable_5cm_time` | Rodrigo's stable arrival: END time of the first interval of >= 0.5 s with EE error <= 1 / 2 / 5 cm at every step, counted only if it ends at t <= 40 s; over the whole run; NaN if none |
| `stable_5cm_leave_time` | first time after `stable_5cm_time` with EE error > 5 cm (NaN if never within the run) |
| `run_end_time` | simulated time at which the run actually ended (extended hold included); accumulated like `duration`, see below |
| `arm_extend_time` | first time the EE displacement in the base frame, relative to its base-frame position at t = 0, exceeds 0.10 m |
| `ee_base_disp_max` | max of that displacement |
| `ring_dist_start`, `ring_dist_min`, `ring_dist_final` | parking ring distance (m) |
| `ring_dist_at_arm_extend`, `target_bearing_err_at_arm_extend_deg` | at `arm_extend_time` |
| `target_bearing_err_start_deg`, `target_bearing_err_final_deg` | abs(wrap(atan2(target - base) - yaw)), degrees |
| `target_dist_start`, `target_dist_final` | planar distance from the base to the target (m) |
| `parked_time`, `parked` | END time of the first interval of >= 1.0 s with ring distance < 0.20 m and the base still (planar speed < 0.01 m/s and abs(yaw rate) < 0.02 rad/s, simulator velocities) at every step, ending by the v1 stop point (t_reach + 1 s for a reached episode); `parked` = `parked_time` not NaN |
| `target_bearing_err_at_park_deg` | at `parked_time` |
| `wheel_over_envelope_frac` | share of steps with max abs(wheel command) > 3.6 rad/s (1.2 x the expert's max 3.0); descriptive only |

From `arm_extend_time` to `wheel_over_envelope_frac`, the columns cover the run up to the v1 stop point
(included). The `stable_*` columns and `run_end_time` cover the whole run.

- `parked` of a reached episode: the 1 s interval must end by the v1 stop point t_reach + 1 s, so the base
  must have been on the ring and still for a full second, without a break, at some time that ends no later
  than t_reach + 1 s. A base that stops shortly before the reach and is jolted right after it misses this,
  even if it is parked during the extended hold. Expert DEV 67 (ring distance 0.07 m): still from 7.953 s,
  jolted to 0.033 m/s 0.075 s after t_reach = 8.141 s, then still again from 8.224 s to the v1 stop at
  9.141 s (0.917 s), so it is not parked (over the extended hold it would be, at 9.224 s).
- The EE position in the base frame is R(base_ori)^T (ee_pos - base_pos), with the full base quaternion.
- standoff(z) = `np.interp(z, STANDOFF_Z_POINTS, STANDOFF_POINTS)`.
- An interval from step k0 to step k1 lasts (k1 - k0) ms, like the v1 hold window. So 0.5 s means 501
  consecutive samples at 1 kHz.
- All times are step x 1 ms, like `t_reach`, except `run_end_time`. It is the accumulated simulated time,
  like `duration` (a float sum of the 1 ms steps), so `run_end_time - t_reach` of a completed window can come
  out 1e-12 s or so below (or above) 10 s (expert DEV 11: t_reach 1.574, run_end_time 11.573999999999025; up
  to about 1e-11 s on DEV / HARD). Compare it with a tolerance.
- The constants (`HOLD_EXT_TIME`, `ARM_EXTEND_DIST`, `PARK_RING_TOL`, `PARK_TIME`, `STABLE_TOLS`,
  `STABLE_TIME`, `STABLE_HORIZON`, `WHEEL_ENVELOPE`) are in the protocol block of `evaluate_policy.py` and
  in every run's `config.json`.
- `summary.txt` adds these lines: `held_10s` with its Wilson interval and the `hold10_stop` counts; stable
  arrival (5 cm with its Wilson interval, plus the 2 cm and 1 cm counts); `parked`; and arm extended.

### Known limitations (expert: the full v1.1 runs `results/expert_v11_<set>`; zero policy: DEV 0)

- `arm_extend_time` does not tell an extending arm from a collapsing one: a collapsing arm (zero torques)
  also counts as extended. Under the zero policy the arm falls under gravity: on DEV 0 `arm_extend_time` is
  0.14 s and the EE moves 0.78 m in the base frame (`ee_base_disp_max`); about 0.14 s in 5/5 checked
  episodes. A learned policy whose arm torques are too weak can therefore count as "extended".
- The expert itself counts as extended in only DEV 245/300 = 81.7%, HARD 181/200 = 90.5% and CONFIRM
  253/300 = 84.3% of the episodes. In the others the reach is short: the base-frame EE displacement never
  exceeds 0.10 m (at most 0.099 m in a DEV / HARD success). So do not read a policy's arm-extension rate
  against 100%: compare it paired against the expert on the same episodes.
- For the expert, `arm_extend_time - nav_done` is >= 0.36 s (median about 0.47 s, max about 2.4 s; DEV and
  HARD successes), never before `nav_done`. The Cartesian speed limit of 0.25 m/s makes 0.10 m take
  >= 0.4 s; the 0.36 s minimum is a little shorter because the displacement is counted from the EE position
  at t = 0, and the EE has already moved a few cm in the base frame at `nav_done` (next bullet).
- Before `nav_done` the expert's base-frame EE displacement stays small (per-episode maximum 0.032-0.048 m
  on DEV 0-19), so navigation is never counted as an extension: none of the 426 DEV / HARD expert
  extensions comes before `nav_done` (the 8 failures included, 0.33-0.42 s after their 25 s timeout).
- `PARK_RING_TOL` is 0.20 m because the expert itself parks up to 0.193 m off the ring on DEV / HARD: its
  drive phase ends within 0.15 m of the goal on the passed-goal rule (goal behind), and the in-place orient
  turn then moves the base centre. With 0.20 m the expert is parked in DEV 297/300, HARD 194/200 and CONFIRM
  293/300 episodes. The only expert successes that are not parked are DEV 67 (cut off at the v1 stop point,
  see above) and CONFIRM 185 (ring distance 0.076 m, t_reach 10.084 s; not examined further).
- The expert's failures (all 8 on DEV / HARD are navigation timeouts at 25 s, then `stalled` or `timeout`)
  end far from the ring: `ring_dist_final` 1.9-2.6 m, although the base came within 0.08 m of the expert's
  goal (`base_goal_dist_min` 0.081-0.084 m) and crossed the ring (`ring_dist_min` below 1 mm). After the
  25 s navigation timeout its wheel commands are zero, and the base slides about 1.6-1.8 m (lateral wheel
  friction 0.01). The final ring and goal distances of a failure therefore say little about how close the
  base came; use the `_min` columns.
- `stable_*` cover our run, which ends at t_reach + 10 s or at the v1 stop. Rodrigo always simulates 40 s,
  so an interval that would only come later is not seen here.

### Verification

`tests/check_v11_against_v1.py <v1_run_dir> <v11_run_dir>` compares every v1 column of `rollouts.csv`
episode by episode. Values are compared as the strings written in the CSV, because both runs use the same
code path. Excluded are `wall_time`, `policy_ms_*` and `cmp_steps`; `error_message` may only change from
empty to `extended hold: <error>`. The script fails unless the reference run is protocol v1 and the other
run v1.1 (`config.json`), on a duplicated episode row in either file, and when an episode of the v1 run is
missing from the v1.1 run (`--allow-subset` accepts a partial run, e.g. `--limit`). `cmp_identical` must be
True in the v1.1 run wherever it is present; with `--require-compare` it must be True in every compared
episode. It prints "N of M v1 episodes compared" and PASS or FAIL with the differing episodes and columns
(exit status 0 / 1).

```bash
python evaluate_policy.py --policy expert --eval-set data/dev --workers 20 --compare-recorded --out-dir results/expert_v11_dev
python tests/check_v11_against_v1.py results/expert_dev results/expert_v11_dev --require-compare
python -m unittest tests/test_harness_v11.py -v              # about 10 s
RUN_SLOW=1 python -m unittest tests/test_harness_v11.py -v   # + expert on 2 DEV episodes, about 30 s
```

- Full v1.1 expert re-runs (2026-10-29, `results/expert_v11_<set>` against `results/expert_<set>`,
  `--require-compare`): PASS on all three sets. All 41 compared v1 columns (every v1 column except
  `wall_time`, `policy_ms_*` and `cmp_steps`; `error_message` is empty in every episode of both runs) are
  identical in every episode (DEV 300 of 300, HARD 200 of 200, CONFIRM 300 of 300), and 800/800 runs are
  identical to the recordings (`cmp_identical`, over the common length, extended hold included).
- The slow unit test passes on DEV 67 (reached, not settled) and DEV 169 (stalled). Episode 11 passes with
  an error injected in the extended hold.

The v1.1 expert baselines (from `results/expert_v11_<set>/summary.txt`; 95% Wilson intervals):

| Set | Success (v1) | `held_10s` | Stable 5 cm (2 / 1 cm) | Parked | Arm extended |
|---|---|---|---|---|---|
| DEV | 298/300 = 99.3% [97.6, 99.8] | 296/300 = 98.7% [96.6, 99.5] | 298/300 = 99.3% [97.6, 99.8] (298 / 298) | 297/300 = 99.0% [97.1, 99.7] | 245/300 = 81.7% [76.9, 85.6] |
| HARD | 194/200 = 97.0% [93.6, 98.6] | 194/200 = 97.0% [93.6, 98.6] | 194/200 = 97.0% [93.6, 98.6] (194 / 194) | 194/200 = 97.0% [93.6, 98.6] | 181/200 = 90.5% [85.6, 93.8] |
| CONFIRM | 294/300 = 98.0% [95.7, 99.1] | 292/300 = 97.3% [94.8, 98.6] | 294/300 = 98.0% [95.7, 99.1] (294 / 294) | 293/300 = 97.7% [95.3, 98.9] | 253/300 = 84.3% [79.8, 88.0] |

The extended window completed in every reached episode (298 / 194 / 294), and `held_10s` equals `settled`
episode by episode on all three sets: every expert success that settles in the first second also holds for
10 s.
