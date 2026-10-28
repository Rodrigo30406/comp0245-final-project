"""Closed-loop evaluation of a policy (a learned model, or the expert itself) on a fixed evaluation set.

The policy controls the robot from the start pose of every episode of an evaluation set (a folder made by
collect_data.py: config.json + index.csv + ep_XXXX.npz), with the same simulation setup as the data
collection. Every policy, the expert included, is judged with the same success and stop rules
(evaluation protocol v1, see EVALUATION.md), so the results are directly comparable episode by episode.

Examples:
  # the expert, checked step by step against its own recorded episodes (verifies this harness)
  python evaluate_policy.py --policy expert --eval-set data/dev --workers 20 --compare-recorded --out-dir results/expert_dev
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
PROTOCOL_VERSION = "v1"
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
ARM_MOVE_ANGLE = 0.05      # rad, the arm counts as having left its start pose (diagnostic only)
TORQUE_LIMIT = ArmControllerParams.max_torque  # N*m actuator limit applied to every policy's torques
# ==============================================================================

PROTOCOL = {name: value for name, value in globals().items()
            if name in ("PROTOCOL_VERSION", "T_MAX", "REACH_TOL", "HOLD_TIME", "SETTLE_TOL", "STALL_TIME",
                        "STALL_IMPROVEMENT", "STALL_MIN_ERROR", "STILL_LIN", "STILL_ANG", "STILL_ARM",
                        "TIP_ANGLE", "ARM_MOVE_ANGLE", "TORQUE_LIMIT")}

# Per-step signals kept when the trajectory is saved or compared (same names as the recorded datasets;
# wheel_cmds / torque_arm are the commands applied to the robot)
TRAJ_FIELDS = ["t", "base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "ee_pos", "wheel_cmds", "torque_arm"]
# Recorded signals compared with --compare-recorded
COMPARED_FIELDS = ["base_pos", "q_mes", "qd_mes", "wheel_cmds", "torque_arm"]


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
    """Run one evaluation episode under protocol v1. Returns (summary row, trajectory dict or None).

    The policy only gets copies of the harness's arrays, so changing them in place cannot affect the evaluation.
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
        return {"success": False, "stop_reason": "policy_error", "error_message": repr(e),
                "wall_time": time.time() - wall_start}, None

    step_cap = round(T_MAX / time_step)
    hold_steps = round(HOLD_TIME / time_step)
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

        # Diagnostics of the measured state
        min_error = min(min_error, error)
        path_length += np.linalg.norm(base_pos[:2] - prev_xy)
        turn_total += abs(wrap_angle(base_bearing - prev_bearing))
        prev_xy, prev_bearing = base_pos[:2], base_bearing
        goal_dist = np.linalg.norm(base_pos[:2] - goal_pos)
        goal_dist_min = min(goal_dist_min, goal_dist)
        if arm_move_time is None and np.max(np.abs(q_mes[4:11] - q_init_arm)) > ARM_MOVE_ANGLE:
            arm_move_time, goal_dist_at_arm_move = current_time, goal_dist

        # Stop rules, evaluated on the measured state before acting
        if reach_step is None and error < REACH_TOL:
            reach_step = step
        if reach_step is not None:
            hold_errors.append(error)
            if step - reach_step >= hold_steps:
                stop_reason = "reached"
                break
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
            stop_reason, error_message = "policy_error", repr(e)
            break
        if not (np.all(np.isfinite(wheel_cmds)) and np.all(np.isfinite(torque_arm))):
            stop_reason = "invalid_action"
            break
        if np.max(np.abs(torque_arm)) >= TORQUE_LIMIT:
            saturated_steps += 1
        wheel_max = max(wheel_max, np.max(np.abs(wheel_cmds)))
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
    times_ms = np.array(policy_times) * 1000.0 if policy_times else np.array([np.nan])
    hold = np.array(hold_errors) if hold_errors else np.array([np.nan])
    row = {
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
        "wall_time": time.time() - wall_start,
    }
    if verbose:
        print(f"Stopped: {stop_reason} at t={current_time:.2f}s, success {row['success']}, "
              f"min error {min_error:.4f} m")
    if keep_traj:
        traj = {name: np.asarray(values) for name, values in traj.items()}
    return row, traj


def compare_with_recorded(traj, recorded_file, recorded_every=1):
    """Compare a trajectory with the recorded episode over their common length.

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
        return {**episode_info(episode), "success": False, "stop_reason": "policy_error", "error_message": repr(e)}
    try:
        row, traj = run_policy_episode(policy, episode, keep_traj=compare or job["traj_path"] is not None,
                                       early_stop=job["early_stop"])
    except Exception as e:
        traceback.print_exc()
        return {**episode_info(episode), "success": False, "stop_reason": "harness_error", "error_message": repr(e)}
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
    return "\n".join(lines)


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
        print({key: row[key] for key in ("success", "settled", "t_reach", "stop_reason", "min_error")})


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
    parser.add_argument("--save-traj", action="store_true", help="save each trajectory to <out-dir>/traj/")
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
