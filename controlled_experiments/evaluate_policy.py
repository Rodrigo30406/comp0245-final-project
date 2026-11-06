"""Closed-loop evaluation of a policy (a learned model, or the expert itself) on a fixed evaluation set.

The policy controls the robot from the start pose of every episode of an evaluation set (a folder made by
collect_data.py: config.json + index.csv + ep_XXXX.npz), with the same simulation setup as the data
collection. Every policy, the expert included, is judged with the same success and stop rules
(evaluation protocol v1.1, see EVALUATION.md), so the results are directly comparable episode by episode.

Protocol v1.1 keeps every v1 rule and every v1 column of rollouts.csv exactly as in v1 (computed at the v1 stop
point: the end of the 1 s hold window of a reached episode, the stop step otherwise). It only adds secondary
metrics and diagnostics, and keeps simulating a reached episode until 10 s after the first reach, so that a
policy that reaches and then drifts away is visible (v1 and the template stopped at the reach).

Examples:
  # the expert, checked step by step against its own recorded episodes (verifies this harness), then its v1
  # columns checked against the v1 run results/expert_dev (kept as the v1 reference: do not overwrite it)
  python evaluate_policy.py --policy expert --eval-set data/dev --workers 20 --compare-recorded --out-dir results/expert_v11_dev
  python tests/check_v11_against_v1.py results/expert_dev results/expert_v11_dev
  # a learned policy: my_policy.py defines make_policy(arg) returning an object with reset() and act()
  python evaluate_policy.py --policy my_policy.py:make_policy --policy-arg models/mlp.pkl \
      --eval-set data/dev --workers 20 --out-dir results/mlp_dev
  # a few episodes with the GUI
  python evaluate_policy.py --policy expert --eval-set data/dev --episodes 7 --gui --out-dir results/debug

Policy interface (see EVALUATION.md for a template):
  policy.reset(target, info)   called once at the start of each episode
  policy.act(obs)              -> (wheel_cmds (4,), torque_arm (7,)), called at every 1 ms step
  obs has the same names and meaning as the recorded datasets:
    "t", "base_pos" (3,), "base_ori" (4,), "base_bearing", "q_mes" (17,), "qd_mes" (17,), "target" (3,)
  info holds the robot model and the start state (used by the expert). A learned policy that uses any of
  it takes an extra input and must say so in the report.
"""
import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pinocchio as pin
import pybullet as p
from simulation_and_control import pb, MotorCommands, PinWrapper, wrap_angle
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState,
    arm_controller, base_controller, configure_gripper_fingers, quaternion2bearing,
)
from robotnik_arm_task import (
    FIXED_FINGER_ANGLES, BASE_NAV_TIMEOUT, ARM_REACH_LIMIT, STANDOFF_Z_POINTS, STANDOFF_POINTS,
)
from collect_data import sample_targets, sample_init_poses, relative_to_world


# ==============================================================================
# EVALUATION PROTOCOL v1 (pre-registered 2026-10-28, see report.txt). Changing any of these values makes
# results from different versions incomparable: bump PROTOCOL_VERSION and re-run the baselines.
# v1.1 (2026-10-29) leaves every v1 value below unchanged and adds the block after it.
PROTOCOL_VERSION = "v1.1"
T_MAX = 40.0               # s allowed to reach the target (the data collection step cap)
REACH_TOL = 0.010          # m, success: EE within this distance of the target at some time t <= T_MAX
HOLD_TIME = 1.0            # s simulated after the first reach, to measure whether the EE stays there
SETTLE_TOL = 0.015         # m, "settled": EE within this distance during the whole hold window
STALL_TIME = 5.0           # s, early stop: robot still and EE error not improved for this long
STALL_IMPROVEMENT = 0.001  # m, an improvement smaller than this does not count as progress
STALL_MIN_ERROR = 0.030    # m, never stalled closer than this: a slow final approach may still succeed
STILL_LIN = 0.01           # m/s, base speed below which the base counts as still
STILL_ANG = 0.02           # rad/s, base yaw rate below which the base counts as still
STILL_ARM = 0.05           # rad/s, max arm joint speed below which the arm counts as still
TIP_ANGLE = 0.5            # rad of roll or pitch: the robot has tipped over
ARM_MOVE_ANGLE = 0.05      # rad, the arm counts as having left its start pose (diagnostic only; v1.1: invalid
                           # for the expert, whose hold posture sags more than this, see arm_extend_time)
TORQUE_LIMIT = ArmControllerParams.max_torque  # N*m actuator limit applied to every policy's torques
# ---- protocol v1.1: secondary metrics and diagnostics only (the primary metric is the v1 one) ----
HOLD_EXT_TIME = 10.0       # s simulated after the first reach (the v1 row is frozen at HOLD_TIME); only tipping
                           # (checked after HOLD_TIME), a policy error or a non-finite command end it earlier
ARM_EXTEND_DIST = 0.10     # m, the arm has extended: EE displaced this far in the base frame from its t = 0 position
PARK_RING_TOL = 0.20       # m, parked: base within this distance of the ring of radius standoff(z) around the target
                           # (the expert parks up to 0.19 m off the ring: its drive ends within 0.15 m of the goal on the
                           # passed-goal rule, then the in-place orient turn moves the base centre; DEV/HARD recordings)
PARK_TIME = 1.0            # s, ... and still (STILL_LIN, STILL_ANG) at every step for this long
STABLE_TOLS = (0.01, 0.02, 0.05)  # m, "stable arrival" (teammate Rodrigo): EE within the tolerance at every step
STABLE_TIME = 0.5          # s, ... for this long,
STABLE_HORIZON = 40.0      # s, ... with the interval ending at t <= this time
WHEEL_ENVELOPE = 3.6       # rad/s, 1.2 x the expert's max wheel command (3.0 rad/s); descriptive only
# ==============================================================================

PROTOCOL = {name: value for name, value in globals().items()
            if name in ("PROTOCOL_VERSION", "T_MAX", "REACH_TOL", "HOLD_TIME", "SETTLE_TOL", "STALL_TIME",
                        "STALL_IMPROVEMENT", "STALL_MIN_ERROR", "STILL_LIN", "STILL_ANG", "STILL_ARM",
                        "TIP_ANGLE", "ARM_MOVE_ANGLE", "TORQUE_LIMIT",
                        "HOLD_EXT_TIME", "ARM_EXTEND_DIST", "PARK_RING_TOL", "PARK_TIME", "STABLE_TOLS",
                        "STABLE_TIME", "STABLE_HORIZON", "WHEEL_ENVELOPE")}

# Per-step signals kept when the trajectory is saved or compared (same names as the recorded datasets;
# wheel_cmds / torque_arm are the commands applied to the robot)
TRAJ_FIELDS = ["t", "base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "ee_pos", "wheel_cmds", "torque_arm"]
# Recorded signals compared with --compare-recorded
COMPARED_FIELDS = ["base_pos", "q_mes", "qd_mes", "wheel_cmds", "torque_arm"]


# ==============================================================================
# Protocol v1.1 metric helpers. They only see the per-step values passed to them (no simulator), so they are
# unit-tested on synthetic sequences in tests/test_harness_v11.py. Times are counted in steps (exact) and
# converted to seconds with step * time_step, like t_reach.
# ==============================================================================

def ee_in_base_frame(ee_pos, base_pos, base_ori):
    """EE position in the base frame, R(base_ori)^T (ee_pos - base_pos), with the full base quaternion.

    base_ori is (x, y, z, w), as returned by the simulator (see quaternion2bearing's argument order)."""
    x, y, z, w = (float(c) for c in base_ori)
    rotation = pin.Quaternion(w, x, y, z).normalized().toRotationMatrix()
    return rotation.T @ (np.asarray(ee_pos, dtype=float) - np.asarray(base_pos, dtype=float))


def ring_distance(base_xy, target, standoff):
    """Parking ring distance | ||target_xy - base_xy|| - standoff(z) | (m): how far the base is from the circle of
    radius standoff around the target, wherever on the circle. It does not depend on the expert's parking azimuth
    (a policy that does not know the start position may park anywhere on the ring and still reach)."""
    return abs(float(np.hypot(target[0] - base_xy[0], target[1] - base_xy[1])) - standoff)


def target_bearing_error(base_xy, yaw, target):
    """|wrap(atan2(target - base) - yaw)| (rad): how far the base is from facing the target (planar)."""
    return abs(float(wrap_angle(np.arctan2(target[1] - base_xy[1], target[0] - base_xy[0]) - yaw)))


def _cm(tol):
    """Column label of a tolerance in m: 0.05 -> '5cm'."""
    return f"{round(tol * 100)}cm"


class DwellDetector:
    """First interval of at least min_steps steps during which a condition holds at every step.

    update(step, holds) is called once per step with consecutive step indices. An interval from step k0 to step k1
    (both included, the condition true at every step in between) lasts (k1 - k0) steps, as the v1 hold window
    lasts HOLD_TIME from the reach step to the stop step. end_step is the step at which the first such interval
    reached min_steps steps ("END time of the first interval of >= T s"), or None."""

    def __init__(self, min_steps):
        self.min_steps = int(min_steps)
        self.start_step = None
        self.end_step = None

    def update(self, step, holds):
        """Returns True at the step at which the interval is found."""
        if self.end_step is not None:
            return False
        if not holds:
            self.start_step = None
            return False
        if self.start_step is None:
            self.start_step = step
        if step - self.start_step >= self.min_steps:
            self.end_step = step
            return True
        return False


class StableArrival:
    """Teammate Rodrigo's "stable arrival" for one tolerance: END time of the first interval of >= STABLE_TIME during
    which the EE error is <= tol at every step, counted only if that interval ends at t <= STABLE_HORIZON; and the
    first time after it at which the EE error is > tol (leaving)."""

    def __init__(self, tol, time_step, stable_time=STABLE_TIME, horizon=STABLE_HORIZON):
        self.tol = tol
        self.time_step = time_step
        self.dwell = DwellDetector(round(stable_time / time_step))
        self.horizon_step = round(horizon / time_step)
        self.leave_step = None

    def update(self, step, error):
        if self.dwell.end_step is None:
            if step <= self.horizon_step:  # a later interval cannot end by the horizon
                self.dwell.update(step, error <= self.tol)
        elif self.leave_step is None and error > self.tol:
            self.leave_step = step

    def times(self):
        """(stable time, leave time) in s, NaN when there is none."""
        if self.dwell.end_step is None:
            return np.nan, np.nan
        leave = self.leave_step * self.time_step if self.leave_step is not None else np.nan
        return self.dwell.end_step * self.time_step, leave


class ApproachDiagnostics:
    """Protocol v1.1 diagnostics of the base approach and of the arm extension, up to the v1 stop point.

    update() gets the measured state of every step from t = 0 up to the v1 stop point included (like the v1
    diagnostics), command() every command applied before it. Nothing here uses the expert's internal state."""

    def __init__(self, target, standoff, time_step):
        self.target = np.asarray(target, dtype=float)
        self.standoff = standoff
        self.time_step = time_step
        self.n_updates = 0
        self.ee_base_start = None
        self.ee_base_disp_max = 0.0
        self.arm_extend_step = None
        self.ring_at_arm_extend, self.bearing_at_arm_extend = np.nan, np.nan
        self.park = DwellDetector(round(PARK_TIME / time_step))
        self.bearing_at_park = np.nan
        self.ring_start = self.ring_final = self.bearing_start = self.bearing_final = np.nan
        self.dist_start = self.dist_final = np.nan
        self.ring_min = np.inf
        self.n_commands, self.over_envelope = 0, 0

    def update(self, step, ee_pos, base_pos, base_ori, base_bearing, base_lin_vel, base_yaw_rate):
        """base_lin_vel: world-frame base velocity (the planar part is used), base_yaw_rate: rad/s (simulator)."""
        ee_base = ee_in_base_frame(ee_pos, base_pos, base_ori)
        if self.ee_base_start is None:
            self.ee_base_start = ee_base
        disp = float(np.linalg.norm(ee_base - self.ee_base_start))
        self.ee_base_disp_max = max(self.ee_base_disp_max, disp)
        ring = ring_distance(base_pos, self.target, self.standoff)
        bearing = target_bearing_error(base_pos, base_bearing, self.target)
        dist = float(np.hypot(self.target[0] - base_pos[0], self.target[1] - base_pos[1]))
        if self.n_updates == 0:
            self.ring_start, self.bearing_start, self.dist_start = ring, bearing, dist
        self.n_updates += 1
        self.ring_final, self.bearing_final, self.dist_final = ring, bearing, dist
        self.ring_min = min(self.ring_min, ring)
        if self.arm_extend_step is None and disp > ARM_EXTEND_DIST:
            self.arm_extend_step = step
            self.ring_at_arm_extend, self.bearing_at_arm_extend = ring, bearing
        still = (float(np.hypot(base_lin_vel[0], base_lin_vel[1])) < STILL_LIN
                 and abs(float(base_yaw_rate)) < STILL_ANG)
        if self.park.update(step, ring < PARK_RING_TOL and still):
            self.bearing_at_park = bearing

    def command(self, wheel_cmds):
        self.n_commands += 1
        if np.max(np.abs(wheel_cmds)) > WHEEL_ENVELOPE:
            self.over_envelope += 1

    def columns(self):
        dt = self.time_step
        return {
            "arm_extend_time": self.arm_extend_step * dt if self.arm_extend_step is not None else np.nan,
            "ee_base_disp_max": self.ee_base_disp_max if self.n_updates else np.nan,
            "ring_dist_start": self.ring_start,
            "ring_dist_min": self.ring_min if self.n_updates else np.nan,
            "ring_dist_final": self.ring_final,
            "ring_dist_at_arm_extend": self.ring_at_arm_extend,
            "target_bearing_err_at_arm_extend_deg": float(np.rad2deg(self.bearing_at_arm_extend)),
            "target_bearing_err_start_deg": float(np.rad2deg(self.bearing_start)),
            "target_bearing_err_final_deg": float(np.rad2deg(self.bearing_final)),
            "target_dist_start": self.dist_start,
            "target_dist_final": self.dist_final,
            "parked_time": self.park.end_step * dt if self.park.end_step is not None else np.nan,
            "parked": self.park.end_step is not None,
            "target_bearing_err_at_park_deg": float(np.rad2deg(self.bearing_at_park)),
            "wheel_over_envelope_frac": self.over_envelope / max(1, self.n_commands),
        }


class ExtendedHold:
    """Protocol v1.1 extended post-reach window, from the first reach (t_reach) to t_reach + HOLD_EXT_TIME.

    Created at the reach step with the base pose there; update() gets the measured state of every step of the
    window, the reach step included (as the v1 hold window), command() every command applied in it. stop is set by
    the harness to why the window ended: completed, policy_error, invalid_action or tipped."""

    STOP_REASONS = ("completed", "policy_error", "invalid_action", "tipped")

    def __init__(self, base_xy, yaw):
        self.base_xy0 = np.array(base_xy[:2], dtype=float)
        self.yaw0 = float(yaw)
        self.errors = []
        self.base_disp_max, self.yaw_change_max = 0.0, 0.0
        self.wheel_max = np.nan
        self.stop = None

    def update(self, error, base_xy, yaw):
        self.errors.append(float(error))
        self.base_disp_max = max(self.base_disp_max, float(np.hypot(base_xy[0] - self.base_xy0[0],
                                                                    base_xy[1] - self.base_xy0[1])))
        self.yaw_change_max = max(self.yaw_change_max, abs(float(wrap_angle(yaw - self.yaw0))))

    def command(self, wheel_cmds):
        m = float(np.max(np.abs(wheel_cmds)))
        self.wheel_max = m if np.isnan(self.wheel_max) else max(self.wheel_max, m)

    def columns(self, time_step):
        errors = np.array(self.errors) if self.errors else np.array([np.nan])
        completed = self.stop == "completed"
        left = np.flatnonzero(errors >= SETTLE_TOL)
        return {
            "hold10_stop": self.stop or "",
            "held_10s": bool(completed and np.max(errors) < SETTLE_TOL),
            "hold10_error_end": float(errors[-1]) if completed else np.nan,
            "hold10_error_mean": float(np.mean(errors)),
            "hold10_error_max": float(np.max(errors)),
            "hold10_leave_time": left[0] * time_step if left.size else np.nan,
            "hold10_base_disp_max": self.base_disp_max if self.errors else np.nan,
            "hold10_yaw_change_max_deg": float(np.rad2deg(self.yaw_change_max)) if self.errors else np.nan,
            "hold10_wheel_cmd_max": self.wheel_max,
        }

    @staticmethod
    def empty_columns():
        """Columns of an episode that never reached the target."""
        return {"hold10_stop": "", "held_10s": False, "hold10_error_end": np.nan, "hold10_error_mean": np.nan,
                "hold10_error_max": np.nan, "hold10_leave_time": np.nan, "hold10_base_disp_max": np.nan,
                "hold10_yaw_change_max_deg": np.nan, "hold10_wheel_cmd_max": np.nan}


def stable_columns(stable):
    """Columns of a list of StableArrival (one per STABLE_TOLS); the leave time only for the largest tolerance."""
    out = {}
    for s in stable:
        out[f"stable_{_cm(s.tol)}_time"] = s.times()[0]
    largest = max(stable, key=lambda s: s.tol)
    out[f"stable_{_cm(largest.tol)}_leave_time"] = largest.times()[1]
    return out


# New columns of protocol v1.1, appended after the v1 columns of rollouts.csv
V11_COLUMNS = [
    "hold10_stop", "held_10s", "hold10_error_end", "hold10_error_mean", "hold10_error_max", "hold10_leave_time",
    "hold10_base_disp_max", "hold10_yaw_change_max_deg", "hold10_wheel_cmd_max",
    *[f"stable_{_cm(tol)}_time" for tol in STABLE_TOLS], f"stable_{_cm(max(STABLE_TOLS))}_leave_time",
    "run_end_time", "arm_extend_time", "ee_base_disp_max", "ring_dist_start", "ring_dist_min", "ring_dist_final",
    "ring_dist_at_arm_extend", "target_bearing_err_at_arm_extend_deg", "target_bearing_err_start_deg",
    "target_bearing_err_final_deg", "target_dist_start", "target_dist_final", "parked_time", "parked",
    "target_bearing_err_at_park_deg", "wheel_over_envelope_frac",
]


def empty_v11_columns():
    """v1.1 columns of an episode that ended before its first step (an error in the policy's reset or creation)."""
    return {**{name: np.nan for name in V11_COLUMNS}, **ExtendedHold.empty_columns(), "parked": False}


def expert_base_goal(target, init_base_pos, init_bearing):
    """Base goal pose the expert plans for (collect_data.run_episode): standoff from the target, facing it,
    on the line from the start position to the target. Also used to evaluate any policy's base motion."""
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    to_target = target[:2] - init_base_pos[:2]
    target_bearing = np.arctan2(to_target[1], to_target[0]) if np.linalg.norm(to_target) > 1e-4 else init_bearing
    goal_pos = target[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    return goal_pos, target_bearing, standoff


class ExpertPolicy:
    """The expert of the data collection: template controllers, approach direction from the start position."""

    def reset(self, target, info):
        self.target = target
        self.model = info["dyn_model"]
        self.ee_frame_id = info["ee_frame_id"]
        self.shoulder_frame_id = info["shoulder_frame_id"]
        self.arm_lower = info["joint_lower"][4:11]
        self.arm_upper = info["joint_upper"][4:11]
        self.time_step = info["time_step"]
        self.base_goal_pos, self.base_goal_bearing, _ = expert_base_goal(
            target, info["init_base_pos"], info["init_base_bearing"])
        self.base_state = BaseNavState()
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.q_init_arm = info["init_q_mes"][4:11].copy()
        self.q_des_arm = self.q_init_arm.copy()
        self.qd_des_arm = np.zeros(7)

    def act(self, obs):
        q_mes, qd_mes = obs["q_mes"], obs["qd_mes"]
        q_pin = self.model.ReoderJoints2PinVec(np.concatenate((obs["base_pos"], obs["base_ori"], q_mes)), "pos")
        pin.forwardKinematics(self.model.pin_model, self.model.pin_data, q_pin)
        pin.updateFramePlacements(self.model.pin_model, self.model.pin_data)
        ee_pos = self.model.pin_data.oMf[self.ee_frame_id].translation.copy()
        wheel_cmds = base_controller(self.base_state, obs["base_pos"], obs["base_bearing"], self.base_goal_pos,
                                     self.base_goal_bearing, obs["t"], self.base_params)
        torque_arm, self.q_des_arm, self.qd_des_arm = arm_controller(
            self.model, q_pin, q_mes[4:11], qd_mes[4:11], self.q_des_arm, self.q_init_arm, self.target, ee_pos,
            self.ee_frame_id, self.shoulder_frame_id, self.arm_lower, self.arm_upper, self.time_step,
            self.arm_params, reach_active=self.base_state.done,
        )
        return wheel_cmds, torque_arm


class ZeroPolicy:
    """All commands zero: the base stays still and the arm falls under gravity. A lower bound, and a quick
    check of the failure and early-stop paths of the harness."""

    def reset(self, target, info):
        pass

    def act(self, obs):
        return np.zeros(4), np.zeros(7)


BUILTIN_POLICIES = {"expert": ExpertPolicy, "zero": ZeroPolicy}


def load_policy(spec, arg=None):
    """'expert' / 'zero', or 'path/to/file.py:factory' where factory(arg) returns the policy object
    (arg is the --policy-arg string, None when it is not given)."""
    if spec in BUILTIN_POLICIES:
        return BUILTIN_POLICIES[spec]()
    path, _, factory_name = spec.rpartition(":")
    if not path.endswith(".py") or not factory_name:
        raise ValueError(f"policy must be one of {list(BUILTIN_POLICIES)} or 'file.py:factory', got {spec!r}")
    path = os.path.abspath(path)
    module_name = os.path.splitext(os.path.basename(path))[0]
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module  # needed by e.g. dataclasses and pickle inside the module
    sys.path.insert(0, os.path.dirname(path))
    module_spec.loader.exec_module(module)
    return getattr(module, factory_name)(arg)


def load_eval_set(eval_dir):
    """Episodes of an evaluation set: target and start pose, regenerated exactly from config.json.

    The (target, start pose) pairs are re-drawn with the same seed and draw order as collect_data.collect_batch,
    which gives the exact floats the expert ran with (the yaw in index.csv went through a degree conversion).
    They are checked against index.csv. Returns a list of dicts sorted by episode.
    """
    with open(os.path.join(eval_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("target") is not None:
        raise SystemExit(f"{eval_dir} is a single-target run, not an evaluation set")
    n = cfg["n_episodes"]
    rng = np.random.default_rng(cfg["seed"])
    r, theta, z = sample_targets(n, rng, cfg["r_range"], cfg["theta_range"], cfg["z_range"])
    noisy = rng.random(n) < cfg.get("noise_frac", 0.0)
    init_xy, init_yaw = sample_init_poses(n, rng, cfg.get("init_radius", 0.0), cfg.get("init_yaw_range", [0.0, 0.0]))

    with open(os.path.join(eval_dir, "index.csv")) as f:
        index = {int(row["episode"]): row for row in csv.DictReader(f)}
    episodes = []
    for i in range(n):
        target = relative_to_world(init_xy[i], init_yaw[i], r[i], theta[i], z[i])
        row = index.get(i)
        if row is None:
            raise SystemExit(f"episode {i} missing from {eval_dir}/index.csv")
        # sets made before start poses were sampled have no init_* columns: they start at the origin
        recorded = np.array([float(row["target_x"]), float(row["target_y"]), float(row["target_z"]),
                             float(row.get("init_x", 0.0)), float(row.get("init_y", 0.0))])
        if not np.allclose(recorded, [*target, *init_xy[i]], atol=1e-9):
            raise SystemExit(f"episode {i}: regenerated target / start pose do not match {eval_dir}/index.csv")
        episodes.append({
            "episode": i, "target": target, "init_xy": init_xy[i], "init_yaw": float(init_yaw[i]),
            "r": float(r[i]), "theta_deg": float(np.rad2deg(theta[i])), "z": float(z[i]),
            "recorded_file": os.path.join(eval_dir, row["file"]) if row.get("file") else None,
            "recorded_every": cfg.get("log_every", 1),
            "expert_status": row["status"], "expert_nav_timed_out": row["nav_timed_out"],
            # the recorded expert of a noisy episode executed noisy commands: not a clean-expert reference
            "expert_noisy": bool(noisy[i]),
        })
    if noisy.any():
        print(f"Note: {noisy.sum()} episodes of {eval_dir} were recorded with command noise; they are left out of "
              f"--compare-recorded and of the comparison with the recorded expert")
    return episodes, cfg


def setup_simulation(init_base_xy, init_base_yaw, gui=False, target=None):
    """Simulation in the same state as collect_data.run_episode at its first step."""
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=gui)

    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    spawn_pos, _ = client.getBasePositionAndOrientation(rid)
    client.resetBasePositionAndOrientation(
        rid, [init_base_xy[0], init_base_xy[1], spawn_pos[2]],
        client.getQuaternionFromEuler([0.0, 0.0, init_base_yaw]),
    )
    sim.ReceiveObservation()

    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    if gui and target is not None:
        visual_shape_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=list(target))

    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())
    info = {
        "dyn_model": dyn_model,
        "ee_frame_id": dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector"),
        "shoulder_frame_id": dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2"),
        "joint_lower": np.array(lower_limits),
        "joint_upper": np.array(upper_limits),
        "time_step": sim.GetTimeStep(),
        "init_base_pos": init_base_pos,
        "init_base_bearing": quaternion2bearing(init_base_ori[3], init_base_ori[0], init_base_ori[1], init_base_ori[2]),
        "init_q_mes": np.array(sim.GetMotorAngles(0)),
    }
    return sim, client, rid, info


def run_policy_episode(policy, episode, gui=False, keep_traj=False, early_stop=True, verbose=True):
    """Run one evaluation episode under protocol v1.1. Returns (summary row, trajectory dict or None).

    Until the v1 stop point the episode runs exactly as in v1, and the v1 columns are computed there: for a reached
    episode at the end of the 1 s hold window (the v1 row is frozen then), otherwise at the stop step. A reached
    episode then keeps running until t_reach + HOLD_EXT_TIME, unless tipping, a policy error or a non-finite command
    ends that extended window first (hold10_stop); nothing in it changes a v1 column. The trajectory covers the whole
    run. The policy only gets copies of the harness's arrays, so changing them in place cannot affect the evaluation.
    early_stop=False disables the 'stalled' stop (to check that it never cuts off an episode that would succeed).
    """
    wall_start = time.time()
    target = np.array(episode["target"], dtype=float)
    sim, client, rid, info = setup_simulation(episode["init_xy"], episode["init_yaw"], gui=gui, target=target)
    dyn_model, ee_frame_id, time_step = info["dyn_model"], info["ee_frame_id"], info["time_step"]

    # Evaluation-only reference: where the expert would park the base (never given to the policy)
    goal_pos, _, standoff = expert_base_goal(target, info["init_base_pos"], info["init_base_bearing"])
    q_init_arm = info["init_q_mes"][4:11].copy()
    init_xy, init_bearing = info["init_base_pos"][:2].copy(), info["init_base_bearing"]
    policy_info = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in info.items()}
    try:
        policy.reset(target.copy(), policy_info)
    except Exception as e:
        traceback.print_exc()
        client.disconnect()
        return {**empty_v11_columns(), "success": False, "stop_reason": "policy_error", "error_message": repr(e),
                "wall_time": time.time() - wall_start}, None

    step_cap = round(T_MAX / time_step)
    hold_steps = round(HOLD_TIME / time_step)
    hold_ext_steps = round(HOLD_EXT_TIME / time_step)
    stall_steps = round(STALL_TIME / time_step)
    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6

    step, current_time = 0, 0.0
    reach_step, hold_errors = None, []
    best_error, last_progress_step = np.inf, 0
    min_error = np.inf
    prev_xy, prev_bearing = init_xy, init_bearing
    path_length, turn_total = 0.0, 0.0
    goal_dist_start = np.linalg.norm(init_xy - goal_pos)
    goal_dist_min = goal_dist_start
    arm_move_time, goal_dist_at_arm_move = None, np.nan
    saturated_steps, wheel_max = 0, 0.0
    policy_times = []
    traj = {name: [] for name in TRAJ_FIELDS} if keep_traj else None
    stop_reason, error_message = None, ""
    # Protocol v1.1
    approach = ApproachDiagnostics(target, standoff, time_step)    # up to the v1 stop point
    stable = [StableArrival(tol, time_step) for tol in STABLE_TOLS]  # whole run
    hold10 = None           # ExtendedHold, from the first reach
    v1_row = None           # the v1 columns, frozen at the v1 stop point of a reached episode
    hold10_error_message = ""

    def v1_columns():
        """The v1 columns from the state of the run at the v1 stop point (exactly the v1 row)."""
        times_ms = np.array(policy_times) * 1000.0 if policy_times else np.array([np.nan])
        hold = np.array(hold_errors) if hold_errors else np.array([np.nan])
        return {
            # every stop other than "reached" is a failure, also an error or NaN command inside the hold window
            "success": stop_reason == "reached",
            "t_reach": reach_step * time_step if reach_step is not None else np.nan,
            # settled needs the full hold window: it is False when the episode stopped inside it
            "settled": stop_reason == "reached" and bool(np.max(hold) < SETTLE_TOL),
            "hold_error": float(np.mean(hold)),
            "hold_error_max": float(np.max(hold)),
            "min_error": float(min_error),
            "final_error": float(error),
            "stop_reason": stop_reason,
            "duration": current_time,
            "n_steps": step,
            "standoff": standoff,
            "base_goal_dist_start": goal_dist_start,
            "base_goal_dist_min": goal_dist_min,
            "base_goal_dist_final": goal_dist,
            "base_path_length": path_length,
            "base_turn_total_deg": float(np.rad2deg(turn_total)),
            "arm_move_time": arm_move_time if arm_move_time is not None else np.nan,
            "base_goal_dist_at_arm_move": goal_dist_at_arm_move,
            "torque_saturated_frac": saturated_steps / max(1, len(policy_times)),
            "wheel_cmd_max": wheel_max,
            "policy_ms_mean": float(np.mean(times_ms)),
            "policy_ms_p99": float(np.percentile(times_ms, 99)),
            "policy_ms_max": float(np.max(times_ms)),
            "error_message": error_message,
        }

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        error = np.linalg.norm(target - ee_pos)

        if v1_row is None:
            # Diagnostics of the measured state (v1), up to the v1 stop point
            min_error = min(min_error, error)
            path_length += np.linalg.norm(base_pos[:2] - prev_xy)
            turn_total += abs(wrap_angle(base_bearing - prev_bearing))
            prev_xy, prev_bearing = base_pos[:2], base_bearing
            goal_dist = np.linalg.norm(base_pos[:2] - goal_pos)
            goal_dist_min = min(goal_dist_min, goal_dist)
            if arm_move_time is None and np.max(np.abs(q_mes[4:11] - q_init_arm)) > ARM_MOVE_ANGLE:
                arm_move_time, goal_dist_at_arm_move = current_time, goal_dist
            # v1.1 diagnostics, also up to the v1 stop point (the simulator velocity getters have no side effects)
            approach.update(step, ee_pos, base_pos, base_ori, base_bearing,
                            sim.GetBaseLinVelocity(), sim.GetBaseAngVelocity()[2])
        for detector in stable:
            detector.update(step, error)

        # Stop rules, evaluated on the measured state before acting
        if v1_row is None:
            # protocol v1, unchanged
            if reach_step is None and error < REACH_TOL:
                reach_step = step
                hold10 = ExtendedHold(base_pos[:2], base_bearing)
            if reach_step is not None:
                hold_errors.append(error)
                hold10.update(error, base_pos[:2], base_bearing)
                if step - reach_step >= hold_steps:
                    # v1 stop point: freeze the v1 row; the run continues to t_reach + HOLD_EXT_TIME
                    stop_reason = "reached"
                    v1_row = v1_columns()
            else:
                roll, pitch, _ = client.getEulerFromQuaternion(base_ori)
                if max(abs(roll), abs(pitch)) > TIP_ANGLE:
                    stop_reason = "tipped"
                    break
                if step >= step_cap:
                    stop_reason = "timeout"
                    break
                if error < best_error - STALL_IMPROVEMENT:
                    best_error, last_progress_step = error, step
                moving = (np.linalg.norm(np.asarray(sim.GetBaseLinVelocity())[:2]) > STILL_LIN
                          or abs(sim.GetBaseAngVelocity()[2]) > STILL_ANG
                          or np.max(np.abs(qd_mes[4:11])) > STILL_ARM)
                if moving or error < STALL_MIN_ERROR:
                    last_progress_step = step
                if early_stop and step - last_progress_step >= stall_steps:
                    stop_reason = "stalled"
                    break
        else:
            # v1.1 extended hold, after the 1 s window: no stall or timeout rule, tipping ends it
            hold10.update(error, base_pos[:2], base_bearing)
            roll, pitch, _ = client.getEulerFromQuaternion(base_ori)
            if max(abs(roll), abs(pitch)) > TIP_ANGLE:
                hold10.stop = "tipped"
                break
            if step - reach_step >= hold_ext_steps:
                hold10.stop = "completed"
                break

        obs = {"t": current_time, "base_pos": base_pos.copy(), "base_ori": base_ori.copy(),
               "base_bearing": base_bearing, "q_mes": q_mes.copy(), "qd_mes": qd_mes.copy(), "target": target.copy()}
        try:
            start = time.perf_counter()
            wheel_cmds, torque_arm = policy.act(obs)
            policy_times.append(time.perf_counter() - start)
            # np.array copies: a policy that reuses its output buffers cannot change the stored commands
            wheel_cmds = np.array(wheel_cmds, dtype=float).reshape(4)
            torque_arm = np.array(torque_arm, dtype=float).reshape(7)
        except Exception as e:
            traceback.print_exc()
            if v1_row is None:
                stop_reason, error_message = "policy_error", repr(e)
            else:
                hold10_error_message = f"extended hold: {e!r}"
            if hold10 is not None:
                hold10.stop = "policy_error"
            break
        if not (np.all(np.isfinite(wheel_cmds)) and np.all(np.isfinite(torque_arm))):
            if v1_row is None:
                stop_reason = "invalid_action"
            if hold10 is not None:
                hold10.stop = "invalid_action"
            break
        if v1_row is None:
            if np.max(np.abs(torque_arm)) >= TORQUE_LIMIT:
                saturated_steps += 1
            wheel_max = max(wheel_max, np.max(np.abs(wheel_cmds)))
            approach.command(wheel_cmds)
        if hold10 is not None:
            hold10.command(wheel_cmds)
        torque_arm = np.clip(torque_arm, -TORQUE_LIMIT, TORQUE_LIMIT)

        if keep_traj:
            for name, value in zip(TRAJ_FIELDS, (current_time, base_pos, base_ori, base_bearing, q_mes, qd_mes,
                                                 ee_pos, wheel_cmds, torque_arm)):
                traj[name].append(value)

        cmd.SetControlCmd(np.concatenate((wheel_cmds, torque_arm, np.zeros(6))), control_modes)
        sim.Step(cmd, "torque")
        if gui:
            time.sleep(0.001)

        current_time += time_step
        step += 1
        if verbose and step % 1000 == 0:
            print(f"t={current_time:.2f}s | base {np.round(base_pos[:2], 3)} | EE error {error:.4f}m")

    client.disconnect()  # GUI mode runs several episodes in one process, and pybullet allows one GUI at a time
    if v1_row is None:  # the run ended at the v1 stop point (never reached, or a failure inside the 1 s window)
        v1_row = v1_columns()
    row = {
        **v1_row,
        **(hold10.columns(time_step) if hold10 is not None else ExtendedHold.empty_columns()),
        **stable_columns(stable),
        "run_end_time": current_time,
        **approach.columns(),
        "wall_time": time.time() - wall_start,  # of the whole run, the extended hold included
    }
    if hold10_error_message and not row["error_message"]:
        row["error_message"] = hold10_error_message
    if verbose:
        print(f"Stopped: {row['stop_reason']} at t={row['duration']:.2f}s, success {row['success']}, "
              f"min error {row['min_error']:.4f} m; extended hold: {row['hold10_stop'] or 'not reached'}, "
              f"run ended at t={current_time:.2f}s")
    if keep_traj:
        traj = {name: np.asarray(values) for name, values in traj.items()}
    return row, traj


def compare_with_recorded(traj, recorded_file, recorded_every=1):
    """Compare a trajectory with the recorded episode over their common length.

    The trajectory covers the whole run (v1.1: the extended hold included); the recording of a reached episode
    ends at the template's stop, so the common length is usually the recording's (cmp_steps can therefore be
    larger than in v1 when the recording continued after the v1 stop point).

    The recordings are float32, so the trajectory is rounded to float32 first: an exact reproduction gives
    zero differences. recorded_every is the --log-every of the recorded set. Returns a dict of comparison
    columns; cmp_first_diff_step counts recorded samples.
    """
    with np.load(recorded_file) as rec:
        n = min(len(rec["t"]), len(traj["t"][::recorded_every]))
        if n == 0:  # the policy stopped before its first command: nothing was reproduced
            return {"cmp_steps": 0, "cmp_identical": False, "cmp_first_diff_step": 0}
        out = {"cmp_steps": n}
        first_diff = None
        for name in COMPARED_FIELDS:
            a = traj[name][::recorded_every][:n].astype(np.float32)
            b = rec[name][:n]
            diff = np.abs(a - b).reshape(n, -1).max(axis=1)
            out[f"cmp_max_{name}"] = float(diff.max())
            nonzero = np.flatnonzero(diff)
            if nonzero.size and (first_diff is None or nonzero[0] < first_diff):
                first_diff = int(nonzero[0])
    out["cmp_identical"] = first_diff is None
    out["cmp_first_diff_step"] = first_diff if first_diff is not None else -1
    return out


def save_traj(traj, path, every):
    np.savez_compressed(path, **{name: values[::every] for name, values in traj.items()})


def _worker(job):
    """One episode in a fresh process (clean pybullet state), console output redirected to a log file."""
    fd = os.open(job["log_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    episode = job["episode"]
    compare = job["compare"] and episode["recorded_file"] and not episode["expert_noisy"]
    try:
        policy = load_policy(job["policy"], job["policy_arg"])
    except Exception as e:
        traceback.print_exc()
        return {**episode_info(episode), **empty_v11_columns(), "success": False, "stop_reason": "policy_error",
                "error_message": repr(e)}
    try:
        row, traj = run_policy_episode(policy, episode, keep_traj=compare or job["traj_path"] is not None,
                                       early_stop=job["early_stop"])
    except Exception as e:
        traceback.print_exc()
        return {**episode_info(episode), **empty_v11_columns(), "success": False, "stop_reason": "harness_error",
                "error_message": repr(e)}
    # Post-processing errors must not discard the episode's result
    try:
        if compare and traj is not None:
            row.update(compare_with_recorded(traj, episode["recorded_file"], episode["recorded_every"]))
        if job["traj_path"] is not None and traj is not None:
            save_traj(traj, job["traj_path"], job["traj_every"])
    except Exception as e:
        traceback.print_exc()
        row["error_message"] = f"post-processing: {e!r}"
    sys.stdout.flush()
    return {**episode_info(episode), **row}


def episode_info(episode):
    """Columns describing the episode: index, target relative to the start pose and in the world frame,
    start pose, signed start-to-base-goal distance d0 = r - standoff(z), recorded expert outcome."""
    standoff = float(np.interp(episode["z"], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    return {
        "episode": episode["episode"], "r": episode["r"], "theta_deg": episode["theta_deg"],
        "target_z": episode["z"], "d0": episode["r"] - standoff,
        "target_x": float(episode["target"][0]), "target_y": float(episode["target"][1]),
        "init_x": float(episode["init_xy"][0]), "init_y": float(episode["init_xy"][1]),
        "init_yaw_deg": float(np.rad2deg(episode["init_yaw"])),
        "expert_status_recorded": episode["expert_status"],
        "expert_nav_timed_out_recorded": episode["expert_nav_timed_out"],
        "expert_noisy_recorded": episode["expert_noisy"],
    }


ROW_COLUMNS = [
    "episode", "success", "settled", "t_reach", "stop_reason", "min_error", "hold_error", "hold_error_max",
    "final_error", "duration", "n_steps", "r", "theta_deg", "target_z", "d0", "standoff", "base_goal_dist_start",
    "base_goal_dist_min", "base_goal_dist_final", "base_path_length", "base_turn_total_deg", "arm_move_time",
    "base_goal_dist_at_arm_move", "torque_saturated_frac", "wheel_cmd_max", "policy_ms_mean", "policy_ms_p99",
    "policy_ms_max", "target_x", "target_y", "init_x", "init_y", "init_yaw_deg", "expert_status_recorded",
    "expert_nav_timed_out_recorded", "expert_noisy_recorded", "cmp_steps", "cmp_identical", "cmp_first_diff_step",
    *[f"cmp_max_{name}" for name in COMPARED_FIELDS], "wall_time", "error_message",
    # protocol v1.1 (the columns above keep their exact v1 meaning and values)
    *V11_COLUMNS,
]


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval of a binomial proportion k / n."""
    if n == 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def parse_episode_selection(text, n):
    """'0-49', '3,7,10' or '5' -> sorted list of episode indices."""
    if not text:
        return list(range(n))
    selected = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            selected.update(range(int(lo), int(hi) + 1))
        else:
            selected.add(int(part))
    return sorted(i for i in selected if 0 <= i < n)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        return ""


def summarize(rows, policy_name, eval_set):
    n = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    lo, hi = wilson_interval(successes, n)
    settled = sum(bool(row.get("settled")) for row in rows)
    lines = [f"Policy {policy_name} on {eval_set}: success {successes}/{n} = {100 * successes / max(1, n):.1f}% "
             f"[{100 * lo:.1f}, {100 * hi:.1f}] (95% Wilson), settled {settled}/{n}"]
    reasons = [str(row.get("stop_reason")) for row in rows]
    lines.append("Stop reasons: " + ", ".join(f"{r} {reasons.count(r)}" for r in sorted(set(reasons))))
    means = [row["policy_ms_mean"] for row in rows if "policy_ms_mean" in row]
    p99s = [row["policy_ms_p99"] for row in rows if "policy_ms_p99" in row]
    if means:
        lines.append(f"Policy step time: mean {np.nanmedian(means):.3f} ms, p99 {np.nanmedian(p99s):.3f} ms "
                     f"(median over episodes; control period 1 ms)")
    # Against the clean expert's recorded outcome (template stop rules) on the same episodes
    table = {}
    clean = [row for row in rows if not row.get("expert_noisy_recorded")]
    for row in clean:
        key = (row.get("expert_status_recorded") == "reached", bool(row["success"]))
        table[key] = table.get(key, 0) + 1
    lines.append(f"Recorded clean expert reached / this policy succeeded ({len(clean)} episodes): "
                 f"both {table.get((True, True), 0)}, expert only {table.get((True, False), 0)}, "
                 f"policy only {table.get((False, True), 0)}, neither {table.get((False, False), 0)}")
    compared = [row for row in rows if "cmp_identical" in row and row["cmp_identical"] != ""]
    if compared:
        identical = sum(bool(row["cmp_identical"]) for row in compared)
        lines.append(f"Recorded comparison: {identical}/{len(compared)} episodes identical over the common length "
                     f"(float32), max differences: " + ", ".join(
                         f"{name} {max(row.get(f'cmp_max_{name}', 0.0) for row in compared):.3g}"
                         for name in COMPARED_FIELDS))

    # Protocol v1.1 secondary metrics (the primary metric is the success above)
    def count(name):  # episodes where a time column is set (not NaN)
        return sum(_is_set(row.get(name)) for row in rows)

    def rate(k):
        lo_k, hi_k = wilson_interval(k, n)
        return f"{k}/{n} = {100 * k / max(1, n):.1f}% [{100 * lo_k:.1f}, {100 * hi_k:.1f}]"

    held = sum(bool(row.get("held_10s")) for row in rows)
    stops = [str(row.get("hold10_stop") or "not reached") for row in rows]
    lines.append(f"v1.1 held_10s (EE < {1000 * SETTLE_TOL:.0f} mm for {HOLD_EXT_TIME:.0f} s after the first reach): "
                 f"{rate(held)}; extended hold ended: "
                 + ", ".join(f"{s} {stops.count(s)}" for s in sorted(set(stops))))
    tols = sorted(STABLE_TOLS, reverse=True)
    lines.append(f"v1.1 stable arrival (EE within tol for >= {STABLE_TIME} s, ending by {STABLE_HORIZON:.0f} s): "
                 f"{_cm(tols[0])} {rate(count(f'stable_{_cm(tols[0])}_time'))}, "
                 + ", ".join(f"{_cm(tol)} {count(f'stable_{_cm(tol)}_time')}" for tol in tols[1:]))
    lines.append(f"v1.1 parked (ring distance < {PARK_RING_TOL} m and base still for >= {PARK_TIME} s): "
                 f"{rate(sum(bool(row.get('parked')) for row in rows))}; arm extended (EE > {ARM_EXTEND_DIST} m from "
                 f"its start in the base frame): {rate(count('arm_extend_time'))} (both up to the v1 stop point)")
    return "\n".join(lines)


def _is_set(value):
    """A value of a result row that is present: not None, not '' and not NaN."""
    if value is None or (isinstance(value, str) and value == ""):
        return False
    try:
        return not np.isnan(value)
    except TypeError:
        return True


def run_batch(args):
    episodes, set_config = load_eval_set(args.eval_set)
    selected = parse_episode_selection(args.episodes, len(episodes))
    if args.limit:
        selected = selected[:args.limit]
    episodes = [episodes[i] for i in selected]
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    traj_dir = os.path.join(out_dir, "traj") if args.save_traj else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
    policy_name = args.name or os.path.basename(args.policy.rpartition(":")[0] or args.policy)
    # Load the policy once here, so a wrong path, factory or model file stops the run before any episode
    load_policy(args.policy, args.policy_arg)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "protocol": PROTOCOL, "eval_set_config": set_config,
                   "git_commit": git_commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "episodes": selected}, f, indent=2)

    jobs = [{
        "episode": ep, "policy": args.policy, "policy_arg": args.policy_arg, "compare": args.compare_recorded,
        "traj_path": os.path.join(traj_dir, f"ep_{ep['episode']:04d}.npz") if traj_dir else None,
        "traj_every": args.traj_every, "early_stop": not args.no_early_stop,
        "log_path": os.path.join(out_dir, "logs", f"ep_{ep['episode']:04d}.log"),
    } for ep in episodes]

    print(f"Evaluating {policy_name} on {len(jobs)} episodes of {args.eval_set} with {args.workers} workers "
          f"(protocol {PROTOCOL_VERSION})")
    rows = []
    rollouts_path = os.path.join(out_dir, "rollouts.csv")
    start = time.time()
    with open(rollouts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for k, row in enumerate(pool.imap_unordered(_worker, jobs), start=1):
                rows.append(row)
                writer.writerow(row)
                f.flush()
                elapsed = time.time() - start
                print(f"[{k}/{len(jobs)}] ep {row['episode']:04d} {str(row.get('stop_reason')):<14} "
                      f"min err {row.get('min_error', np.nan):.4f} m  sim {row.get('duration', 0):.1f}s  "
                      f"hold10 {row.get('hold10_stop') or '-':<14} "
                      f"| elapsed {elapsed / 60:.1f} min, ETA {elapsed / k * (len(jobs) - k) / 60:.1f} min")

    rows.sort(key=lambda row: row["episode"])
    with open(rollouts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, policy_name, args.eval_set)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"Evaluation time: {(time.time() - start) / 60:.1f} min. Results in {out_dir}")
    errors = [row for row in rows if row.get("stop_reason") in ("harness_error", "policy_error")]
    if errors:
        print(f"WARNING: {len(errors)} episodes ended with an error, first: {errors[0].get('error_message')} "
              f"(see {out_dir}/logs/ep_{errors[0]['episode']:04d}.log)")


def run_gui(args):
    """Episodes one after another in this process, with the pybullet GUI."""
    episodes, _ = load_eval_set(args.eval_set)
    selected = parse_episode_selection(args.episodes, len(episodes))[:args.limit or None]
    for i in selected:
        policy = load_policy(args.policy, args.policy_arg)
        row, _ = run_policy_episode(policy, episodes[i], gui=True, early_stop=not args.no_early_stop)
        print({key: row[key] for key in ("success", "settled", "t_reach", "stop_reason", "min_error", "held_10s",
                                         "hold10_stop", "stable_5cm_time", "arm_extend_time", "parked_time")})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True,
                        help="'expert', 'zero', or 'path/to/file.py:factory' returning a policy object")
    parser.add_argument("--policy-arg", default=None, help="string passed to the policy factory (e.g. a model path)")
    parser.add_argument("--name", default=None, help="policy name used in the summary")
    parser.add_argument("--eval-set", required=True, help="evaluation set folder made by collect_data.py")
    parser.add_argument("--episodes", default=None, help="subset of episodes, e.g. '0-49' or '3,7,10'")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N selected episodes")
    parser.add_argument("--out-dir", required=True, help="output folder for rollouts.csv, summary.txt, logs")
    parser.add_argument("--workers", type=int, default=1, help="parallel simulation processes")
    parser.add_argument("--compare-recorded", action="store_true",
                        help="compare each trajectory with the recorded episode (to verify the harness with the expert)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="disable the 'stalled' stop: run to success or T_MAX (to check the early stop on a new model)")
    parser.add_argument("--save-traj", action="store_true",
                        help="save each trajectory (the whole run, extended hold included) to <out-dir>/traj/")
    parser.add_argument("--traj-every", type=int, default=10, help="keep one saved sample every N steps")
    parser.add_argument("--gui", action="store_true", help="run the selected episodes one by one with the GUI")
    return parser.parse_args()


def main():
    args = parse_args()
    # One thread per worker: parallel workers would otherwise compete for the cores and slow each other down
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if args.gui:
        run_gui(args)
    else:
        run_batch(args)


if __name__ == '__main__':
    main()
