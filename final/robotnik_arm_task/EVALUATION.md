# Closed-loop evaluation (protocol v1)

How every policy, learned or the expert, is evaluated in closed loop, so that results from different
models, datasets and teammates can be compared episode by episode. The rules below were fixed on
2026-10-28, before any learned model was evaluated with them (pre-registration, see `report.txt`).

## Files

| File | Status | Purpose |
|---|---|---|
| `evaluate_policy.py` | new | Runs a policy on an evaluation set and writes one result row per episode |
| `results/<run>/` | output | `rollouts.csv`, `summary.txt`, `config.json` (+ `logs/`, `traj/`, not versioned) |

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
  the policy mixes up the phases).
- Commands: `torque_saturated_frac` (steps with a torque at the 30 N m limit before clipping), `wheel_cmd_max`.
- Efficiency: `policy_ms_mean`, `policy_ms_p99`, `policy_ms_max` (time of one `act` call, single thread).
- With `--compare-recorded`: `cmp_identical`, `cmp_first_diff_step`, `cmp_max_<signal>`.

## Running

```bash
# verify the harness with the expert (results/expert_<set>/ are also the paired baselines)
python evaluate_policy.py --policy expert --eval-set data/dev --workers 20 --compare-recorded --out-dir results/expert_dev

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
