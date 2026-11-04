"""Closed-loop evaluation of a policy on the obstacle task: evaluation protocol v2.0 (step 9, plan section 4).

Same rules and columns as evaluate_policy.py v1.1 (its helpers are imported, never edited), except:
  - T_MAX = 60 s (v1.1: 40 s) and the stable-arrival horizon = T_MAX (v1.1: 40 s); every other v1.1 constant is unchanged;
  - the scenario's obstacle is spawned before the first step; a robot-obstacle contact (non-empty contact manifold,
    obstacle_task/sim.py) ends the run at once. success = EE < 10 mm at some measured state within T_MAX, no contact
    at or before it, and the 1 s hold window ended normally (no policy error / invalid action), as v1. A contact
    inside the 1 s or the 10 s hold window ends that window and fails the hold metrics (settled, held_10s;
    hold10_stop "collision"), not success. A contact before the reach: stop_reason "collision", a failure;
  - every observation carries obs["lidar"] = lidar.scan(base_pos[:2], base_bearing, scenario.obstacles);
  - new columns (V2_COLUMNS, after the v1.1 ones): collision, collision_time (s, measured state at which the contact
    shows), collision_link (robot link name), min_clearance (m, getClosestPoints every 50 ms and at a contact, whole
    run), passed_obstacle (the base's coordinate along the start->dock line exceeded the obstacle centre's),
    side_taken (+1 left / -1 right: sign of the base's lateral position from that line at that moment; 0 if never),
    docked (= v1.1 parked: within 0.20 m of the standoff ring and still for 1 s, up to the v1 stop point),
    time_to_dock (= parked_time, the end of that 1 s), docked_clean (docked before any contact), ambiguous,
    scenario_side (the expert's side), lateral_offset, obstacle_r (from the scenario).

Policy interface (as v1.1): policy.reset(target, info) once per episode, then policy.act(obs) -> (wheel_cmds (4,),
torque_arm (7,)) at every 1 ms step. Arrays are copies (a policy cannot change the harness's state).
  obs:  t (float, s), base_pos (3,), base_ori (4,) quaternion x, y, z, w, base_bearing (float, rad), q_mes (17,),
        qd_mes (17,), target (3,), lidar (64,) float64 m (beam k at angle 2 pi k / 64 from the base's forward axis,
        counter-clockwise, clipped at 4 m): the only obstacle information a learned policy receives
  info: dyn_model, ee_frame_id, shoulder_frame_id, joint_lower (17,), joint_upper (17,), time_step,
        init_base_pos (3,), init_base_bearing, init_q_mes (17,) (INFO_KEYS; no obstacles, waypoint, side or dock).
        Only the builtin expert (exactly ExpertPolicy) also gets info["scenario"].
--policy: "expert" (obstacle_task.expert, privileged), "zero", "path/to/file.py:factory" or "package.module:factory";
factory(arg) returns the policy, arg = --policy-arg (None if not given). A file inside obstacle_task/ (e.g.
obstacle_task/policies.py:make_policy) is imported as the package module obstacle_task.<name>.

Evaluation sets are folders made by obstacle_task.collect (scenarios.json gives the exact floats replayed; index.csv
the recorded expert outcome). The expert replays its clean recordings exactly (--compare-recorded, gate A6).

Examples (from the task folder):
  python -m obstacle_task.evaluate --policy expert --eval-set data/obstacle/dev2 --workers 20 --compare-recorded \
      --out-dir results/step9/expert_dev2
  python -m obstacle_task.evaluate --policy obstacle_task/policies.py:<factory> --policy-arg <models> \
      --eval-set data/obstacle/dev2 --workers 20 --out-dir results/step9/<run>     # Track B's factories
test2 is sealed: --allow-test2.
"""
import argparse
import csv
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pinocchio as pin
from simulation_and_control import MotorCommands, wrap_angle

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import evaluate_policy as v1  # noqa: E402
from robotnik_arm_control import quaternion2bearing  # noqa: E402
from obstacle_task.collect import check_sealed  # noqa: E402
from obstacle_task.expert import ExpertPolicy  # noqa: E402
from obstacle_task.lidar import scan  # noqa: E402
from obstacle_task.scenario import Scenario, load_scenarios  # noqa: E402
from obstacle_task.sim import link_name, min_clearance, robot_obstacle_contact, spawn_obstacles  # noqa: E402

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# EVALUATION PROTOCOL v2.0: the v1.1 block of evaluate_policy.py, except T_MAX, plus the clearance sampling.
# The stable-arrival horizon follows T_MAX (Claude's integration decision, 2026-11-04, before any learner run: with
# v1.1's 40 s a stable arrival between 40 and 60 s would not count while success does).
PROTOCOL_VERSION = "v2.0"
T_MAX = 60.0               # s allowed to reach the target (v1.1: 40 s)
CLEARANCE_INTERVAL = 0.05  # s between two min_clearance samples
# ==============================================================================
STABLE_HORIZON = T_MAX     # s, a stable arrival must end by this time (v1.1: 40 s)
PROTOCOL = {**v1.PROTOCOL, "PROTOCOL_VERSION": PROTOCOL_VERSION, "T_MAX": T_MAX,
            "STABLE_HORIZON": STABLE_HORIZON, "CLEARANCE_INTERVAL": CLEARANCE_INTERVAL}
INFO_KEYS = ("dyn_model", "ee_frame_id", "shoulder_frame_id", "joint_lower", "joint_upper",
             "time_step", "init_base_pos", "init_base_bearing", "init_q_mes")
V2_COLUMNS = ["collision", "collision_time", "collision_link", "min_clearance", "passed_obstacle",
              "side_taken", "docked", "docked_clean", "time_to_dock", "ambiguous", "scenario_side",
              "lateral_offset", "obstacle_r", "policy_latched_side", "policy_p_left"]
ROW_COLUMNS = [*v1.ROW_COLUMNS, *V2_COLUMNS]
TRAJ_FIELDS = [*v1.TRAJ_FIELDS, "lidar", "segment"]
COMPARED_FIELDS = v1.COMPARED_FIELDS


def sanitize_policy_info(info):
    """The info of a learned policy: copies of the INFO_KEYS entries only (an allowlist, so nothing added to the
    harness's info later can reach a learner)."""
    return {key: value.copy() if isinstance(value, np.ndarray) else value
            for key in INFO_KEYS for value in [info[key]]}


def build_observation(t, base_pos, base_ori, base_bearing, q_mes, qd_mes, target, obstacles):
    """The observation given to every policy (copies), lidar scanned from the measured base pose."""
    return {"t": float(t), "base_pos": np.array(base_pos, copy=True),
            "base_ori": np.array(base_ori, copy=True), "base_bearing": float(base_bearing),
            "q_mes": np.array(q_mes, copy=True), "qd_mes": np.array(qd_mes, copy=True),
            "target": np.array(target, copy=True), "lidar": scan(base_pos[:2], base_bearing, obstacles)}


def setup_simulation(init_base_xy, init_base_yaw, gui=False, target=None, obstacles=()):
    """Reuse v1's setup, then add static cylinders before the first simulation step.

    Returns (sim, client, robot_id, info, obstacle_ids). No simulation step occurs
    between v1's base reset and spawning, so this shares the collector's initial state.
    """
    sim, client, rid, info = v1.setup_simulation(init_base_xy, init_base_yaw, gui=gui, target=target)
    return sim, client, rid, info, spawn_obstacles(client, obstacles)


def load_policy(spec, arg=None):
    """'expert' (the obstacle expert), 'zero', 'path/to/file.py:factory' or 'package.module:factory' -> policy.

    factory(arg) returns the policy object. A file inside obstacle_task/ is imported as obstacle_task.<name>, so its
    package imports work and it cannot shadow a top-level module of the same name (v1's loader would register
    obstacle_task/policies.py as the module 'policies', the folder of the A-to-B policies); any other file is
    loaded by evaluate_policy.load_policy."""
    if spec == "expert":
        return ExpertPolicy()
    if spec in v1.BUILTIN_POLICIES:
        return v1.load_policy(spec, arg)
    target, _, factory_name = spec.rpartition(":")
    if not target or not factory_name:
        raise ValueError(f"policy must be 'expert', 'zero', 'file.py:factory' or 'module:factory', got {spec!r}")
    if target.endswith(".py"):
        path = os.path.abspath(target)
        if os.path.dirname(path) != PACKAGE_DIR:
            return v1.load_policy(spec, arg)
        target = "obstacle_task." + os.path.splitext(os.path.basename(path))[0]
    return getattr(importlib.import_module(target), factory_name)(arg)


def _scenario(episode):
    value = episode["scenario"]
    return value if isinstance(value, Scenario) else Scenario.from_dict(value)


def scenario_columns(scenario):
    return {"ambiguous": scenario.ambiguous, "scenario_side": scenario.side,
            "lateral_offset": scenario.lateral_offset,
            "obstacle_r": scenario.obstacles[0]["r"] if scenario.obstacles else np.nan}


def empty_v2_columns():
    return {"collision": False, "collision_time": np.nan, "collision_link": None,
            "min_clearance": np.nan, "passed_obstacle": False, "side_taken": 0,
            "docked": False, "docked_clean": False, "time_to_dock": np.nan}


class ObstacleDiagnostics:
    """passed_obstacle / side_taken from the measured base positions only (never the policy's state).

    The first measured state whose coordinate along the start->dock line exceeds the obstacle centre's latches the
    passage and side = sign of the base's lateral position from that line (+1 left, as scenario.side). Without
    obstacles nothing is ever passed (side 0)."""

    def __init__(self, scenario):
        self.start = np.asarray(scenario.init_xy, dtype=float)
        delta = np.asarray(scenario.dock_xy, dtype=float) - self.start
        length = float(np.linalg.norm(delta))
        self.tangent = delta / length if length else np.array([1.0, 0.0])
        self.normal = np.array([-self.tangent[1], self.tangent[0]])
        self.obstacle_along = None
        if scenario.obstacles:
            ob = scenario.obstacles[0]
            self.obstacle_along = float((np.array([ob["x"], ob["y"]]) - self.start) @ self.tangent)
        self.passed = False
        self.side = 0

    def update(self, base_xy):
        offset = np.asarray(base_xy)[:2] - self.start
        if (not self.passed and self.obstacle_along is not None
                and float(offset @ self.tangent) > self.obstacle_along):
            self.passed = True
            self.side = int(np.sign(float(offset @ self.normal)))


def run_policy_episode(policy, episode, gui=False, keep_traj=False, early_stop=True, verbose=True):
    """Run one scenario, returning (metric row, optional trajectory dict).

    ``episode`` needs target, init_xy, init_yaw, and scenario (Scenario or dict).
    Privilege is granted only to the exact builtin ExpertPolicy type; subclasses
    and factory policies receive the same sanitized info as any learned policy.
    ``early_stop=False`` disables only the still/stalled rule, never contact stops.
    """
    wall_start = time.time()
    scenario = _scenario(episode)
    target = np.array(episode["target"], dtype=float)
    sim, client, rid, info, obstacle_ids = setup_simulation(
        episode["init_xy"], episode["init_yaw"], gui=gui, target=target, obstacles=scenario.obstacles)
    model, ee_frame, dt = info["dyn_model"], info["ee_frame_id"], info["time_step"]
    goal_pos, _, standoff = v1.expert_base_goal(target, info["init_base_pos"], info["init_base_bearing"])
    q_init_arm = info["init_q_mes"][4:11].copy()
    init_xy, init_yaw = info["init_base_pos"][:2].copy(), info["init_base_bearing"]
    policy_info = sanitize_policy_info(info)
    if type(policy) is ExpertPolicy:
        policy_info["scenario"] = Scenario.from_dict(scenario.to_dict())
    try:
        policy.reset(target.copy(), policy_info)
    except Exception as exc:
        traceback.print_exc()
        client.disconnect()
        return {**v1.empty_v11_columns(), **empty_v2_columns(), **scenario_columns(scenario),
                "success": False, "stop_reason": "policy_error", "error_message": repr(exc),
                "wall_time": time.time() - wall_start}, None

    step_cap = round(T_MAX / dt)
    hold_steps, hold_ext_steps = round(v1.HOLD_TIME / dt), round(v1.HOLD_EXT_TIME / dt)
    stall_steps, clearance_steps = round(v1.STALL_TIME / dt), max(1, round(CLEARANCE_INTERVAL / dt))
    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 13
    step, current_time = 0, 0.0
    reach_step, hold_errors = None, []
    best_error, last_progress_step, minimum_error = np.inf, 0, np.inf
    previous_xy, previous_yaw = init_xy, init_yaw
    path_length, turn_total = 0.0, 0.0
    goal_dist_start = float(np.linalg.norm(init_xy - goal_pos))
    goal_dist_min = goal_dist_start
    arm_move_time, goal_dist_at_arm_move = None, np.nan
    saturated_steps, wheel_max = 0, 0.0
    policy_times = []
    traj = {name: [] for name in TRAJ_FIELDS} if keep_traj else None
    stop_reason, error_message, hold_error_message = None, "", ""
    approach = v1.ApproachDiagnostics(target, standoff, dt)
    stable = [v1.StableArrival(tol, dt, horizon=STABLE_HORIZON) for tol in v1.STABLE_TOLS]
    passage = ObstacleDiagnostics(scenario)
    hold10, primary_row = None, None
    collision, collision_time, collision_link = False, np.nan, None
    clearance = np.inf
    hold_completed = False

    def primary_columns():
        times_ms = np.array(policy_times) * 1000.0 if policy_times else np.array([np.nan])
        hold = np.array(hold_errors) if hold_errors else np.array([np.nan])
        return {
            "success": stop_reason == "reached",
            "t_reach": reach_step * dt if reach_step is not None else np.nan,
            "settled": stop_reason == "reached" and hold_completed and bool(np.max(hold) < v1.SETTLE_TOL),
            "hold_error": float(np.mean(hold)), "hold_error_max": float(np.max(hold)),
            "min_error": float(minimum_error), "final_error": float(error), "stop_reason": stop_reason,
            "duration": current_time, "n_steps": step, "standoff": standoff,
            "base_goal_dist_start": goal_dist_start, "base_goal_dist_min": goal_dist_min,
            "base_goal_dist_final": goal_dist, "base_path_length": path_length,
            "base_turn_total_deg": float(np.rad2deg(turn_total)),
            "arm_move_time": arm_move_time if arm_move_time is not None else np.nan,
            "base_goal_dist_at_arm_move": goal_dist_at_arm_move,
            "torque_saturated_frac": saturated_steps / max(1, len(policy_times)),
            "wheel_cmd_max": wheel_max, "policy_ms_mean": float(np.mean(times_ms)),
            "policy_ms_p99": float(np.percentile(times_ms, 99)), "policy_ms_max": float(np.max(times_ms)),
            "error_message": error_message,
        }

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        yaw = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes, qd_mes = np.array(sim.GetMotorAngles(0)), np.array(sim.GetMotorVelocities(0))
        q_pin = model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(model.pin_model, model.pin_data, q_pin)
        pin.updateFramePlacements(model.pin_model, model.pin_data)
        ee_pos = model.pin_data.oMf[ee_frame].translation.copy()
        error = float(np.linalg.norm(target - ee_pos))

        if primary_row is None:
            minimum_error = min(minimum_error, error)
            path_length += float(np.linalg.norm(base_pos[:2] - previous_xy))
            turn_total += abs(wrap_angle(yaw - previous_yaw))
            previous_xy, previous_yaw = base_pos[:2], yaw
            goal_dist = float(np.linalg.norm(base_pos[:2] - goal_pos))
            goal_dist_min = min(goal_dist_min, goal_dist)
            if arm_move_time is None and np.max(np.abs(q_mes[4:11] - q_init_arm)) > v1.ARM_MOVE_ANGLE:
                arm_move_time, goal_dist_at_arm_move = current_time, goal_dist
            approach.update(step, ee_pos, base_pos, base_ori, yaw,
                            sim.GetBaseLinVelocity(), sim.GetBaseAngVelocity()[2])
        for detector in stable:
            detector.update(step, error)
        passage.update(base_pos[:2])

        # Check contact before registering a first reach: a simultaneous contact is
        # not a successful arrival. Queries do not step or otherwise alter physics.
        collision, link = robot_obstacle_contact(client, rid, obstacle_ids)
        if step % clearance_steps == 0 or collision:
            clearance = min(clearance, min_clearance(client, rid, obstacle_ids))
        if collision:
            collision_time, collision_link = step * dt, link_name(client, rid, link)
            if hold10 is not None:
                if primary_row is None:
                    hold_errors.append(error)
                hold10.update(error, base_pos[:2], yaw)
                hold10.stop = "collision"
            if primary_row is None:
                stop_reason = "reached" if reach_step is not None else "collision"
            break

        if primary_row is None:
            if reach_step is None and error < v1.REACH_TOL and step <= step_cap:
                reach_step = step
                hold10 = v1.ExtendedHold(base_pos[:2], yaw)
            if reach_step is not None:
                hold_errors.append(error)
                hold10.update(error, base_pos[:2], yaw)
                if step - reach_step >= hold_steps:
                    hold_completed, stop_reason = True, "reached"
                    primary_row = primary_columns()
            else:
                roll, pitch, _ = client.getEulerFromQuaternion(base_ori)
                if max(abs(roll), abs(pitch)) > v1.TIP_ANGLE:
                    stop_reason = "tipped"
                    break
                if step >= step_cap:
                    stop_reason = "timeout"
                    break
                if error < best_error - v1.STALL_IMPROVEMENT:
                    best_error, last_progress_step = error, step
                moving = (np.linalg.norm(np.asarray(sim.GetBaseLinVelocity())[:2]) > v1.STILL_LIN
                          or abs(sim.GetBaseAngVelocity()[2]) > v1.STILL_ANG
                          or np.max(np.abs(qd_mes[4:11])) > v1.STILL_ARM)
                if moving or error < v1.STALL_MIN_ERROR:
                    last_progress_step = step
                if early_stop and step - last_progress_step >= stall_steps:
                    stop_reason = "stalled"
                    break
        else:
            hold10.update(error, base_pos[:2], yaw)
            roll, pitch, _ = client.getEulerFromQuaternion(base_ori)
            if max(abs(roll), abs(pitch)) > v1.TIP_ANGLE:
                hold10.stop = "tipped"
                break
            if step - reach_step >= hold_ext_steps:
                hold10.stop = "completed"
                break

        obs = build_observation(current_time, base_pos, base_ori, yaw, q_mes, qd_mes, target, scenario.obstacles)
        # The saved sensor observation is protected from a policy mutating its input.
        ranges = obs["lidar"].copy() if keep_traj else None
        try:
            start = time.perf_counter()
            wheel_cmds, torque_arm = policy.act(obs)
            policy_times.append(time.perf_counter() - start)
            wheel_cmds = np.array(wheel_cmds, dtype=float).reshape(4)
            torque_arm = np.array(torque_arm, dtype=float).reshape(7)
        except Exception as exc:
            traceback.print_exc()
            if primary_row is None:
                stop_reason, error_message = "policy_error", repr(exc)
            else:
                hold_error_message = f"extended hold: {exc!r}"
            if hold10 is not None:
                hold10.stop = "policy_error"
            break
        if not (np.all(np.isfinite(wheel_cmds)) and np.all(np.isfinite(torque_arm))):
            if primary_row is None:
                stop_reason = "invalid_action"
            if hold10 is not None:
                hold10.stop = "invalid_action"
            break
        if primary_row is None:
            if np.max(np.abs(torque_arm)) >= v1.TORQUE_LIMIT:
                saturated_steps += 1
            wheel_max = max(wheel_max, float(np.max(np.abs(wheel_cmds))))
            approach.command(wheel_cmds)
        if hold10 is not None:
            hold10.command(wheel_cmds)
        torque_arm = np.clip(torque_arm, -v1.TORQUE_LIMIT, v1.TORQUE_LIMIT)
        if keep_traj:
            segment = policy.segment if type(policy) is ExpertPolicy else -1
            for name, value in zip(TRAJ_FIELDS, (current_time, base_pos, base_ori, yaw, q_mes, qd_mes,
                                                ee_pos, wheel_cmds, torque_arm, ranges, segment)):
                traj[name].append(value)
        cmd.SetControlCmd(np.concatenate((wheel_cmds, torque_arm, np.zeros(6))), control_modes)
        sim.Step(cmd, "torque")
        if gui:
            time.sleep(0.001)
        current_time += dt
        step += 1
        if verbose and step % 1000 == 0:
            print(f"t={current_time:.2f}s | base {np.round(base_pos[:2], 3)} | EE error {error:.4f}m")

    client.disconnect()
    if primary_row is None:
        primary_row = primary_columns()
    approach_columns = approach.columns()
    dock_time = approach_columns["parked_time"]
    row = {**primary_row, **(hold10.columns(dt) if hold10 is not None else v1.ExtendedHold.empty_columns()),
           **v1.stable_columns(stable), "run_end_time": current_time, **approach_columns,
           "wall_time": time.time() - wall_start,
           "collision": collision, "collision_time": collision_time, "collision_link": collision_link,
           "min_clearance": clearance, "passed_obstacle": passage.passed, "side_taken": passage.side,
           "docked": approach_columns["parked"], "time_to_dock": dock_time,
           "docked_clean": approach_columns["parked"] and (not collision or dock_time < collision_time),
           **scenario_columns(scenario)}
    # Policy diagnostics (e.g. SideLatchedPolicy's latched side, for the M4 check); empty for other policies
    log = getattr(policy, "episode_log", None)
    log = log() if callable(log) else {}
    row["policy_latched_side"] = log.get("latched_side")
    row["policy_p_left"] = log.get("p_left")
    if hold_error_message and not row["error_message"]:
        row["error_message"] = hold_error_message
    if verbose:
        print(f"Stopped: {row['stop_reason']} at t={row['duration']:.2f}s, success {row['success']}, "
              f"min error {row['min_error']:.4f}m, collision {collision}; hold {row['hold10_stop'] or '-'}")
    if keep_traj:
        traj = {name: np.asarray(values, dtype=np.int8 if name == "segment" else float)
                for name, values in traj.items()}
    return row, traj


def load_eval_set(eval_dir):
    """Read exact scenario floats and recorded outcomes; never regenerate the sampler.

    Partially collected sets are allowed: only rows with recordings are returned,
    with their original episode ids. The CLI rejects requested missing episodes.
    """
    with open(os.path.join(eval_dir, "config.json")) as handle:
        config = json.load(handle)
    scenarios, _ = load_scenarios(os.path.join(eval_dir, "scenarios.json"))
    with open(os.path.join(eval_dir, "index.csv")) as handle:
        index = {int(row["episode"]): row for row in csv.DictReader(handle)}
    episodes = []
    for scenario in scenarios:
        row = index.get(scenario.episode)
        if row is None:
            continue
        target = np.asarray(scenario.target)
        recorded = np.array([float(row["target_x"]), float(row["target_y"]), float(row["target_z"]),
                             float(row["init_x"]), float(row["init_y"])])
        if not np.allclose(recorded, [*target, *scenario.init_xy], atol=1e-9, rtol=0):
            raise ValueError(f"episode {scenario.episode}: scenarios.json disagrees with index.csv")
        delta = target[:2] - scenario.init_xy
        relative_bearing = float(wrap_angle(np.arctan2(delta[1], delta[0]) - scenario.init_yaw))
        episodes.append({"episode": scenario.episode, "scenario": scenario, "target": target,
                         "init_xy": np.asarray(scenario.init_xy), "init_yaw": scenario.init_yaw,
                         "r": float(np.linalg.norm(delta)), "theta_deg": float(np.rad2deg(relative_bearing)),
                         "z": float(target[2]), "recorded_every": config.get("log_every", 1),
                         "recorded_file": os.path.join(eval_dir, row["file"]) if row.get("file") else None,
                         "expert_status": row.get("status", ""),
                         "expert_nav_timed_out": row.get("nav_timed_out", ""),
                         "expert_noisy": scenario.noisy})
    episodes.sort(key=lambda ep: ep["episode"])
    if not episodes:
        raise ValueError(f"no recorded episodes in {eval_dir}")
    if any(ep["expert_noisy"] for ep in episodes):
        print("Note: noisy recordings are excluded from --compare-recorded; the harness expert is clean.")
    return episodes, config


def episode_info(episode):
    return {**v1.episode_info(episode), **scenario_columns(_scenario(episode))}


compare_with_recorded = v1.compare_with_recorded
save_traj = v1.save_traj


def _worker(job):
    """Exactly one episode per spawned process; pybullet console output goes to its log."""
    fd = os.open(job["log_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    episode = job["episode"]
    compare = job["compare"] and episode["recorded_file"] and not episode["expert_noisy"]
    try:
        policy = load_policy(job["policy"], job["policy_arg"])
    except Exception as exc:
        traceback.print_exc()
        return {**v1.empty_v11_columns(), **empty_v2_columns(), **episode_info(episode), "success": False,
                "stop_reason": "policy_error", "error_message": repr(exc)}
    try:
        row, traj = run_policy_episode(policy, episode, keep_traj=bool(compare or job["traj_path"]),
                                       early_stop=job["early_stop"])
    except Exception as exc:
        traceback.print_exc()
        return {**v1.empty_v11_columns(), **empty_v2_columns(), **episode_info(episode), "success": False,
                "stop_reason": "harness_error", "error_message": repr(exc)}
    try:
        if compare and traj is not None:
            row.update(compare_with_recorded(traj, episode["recorded_file"], episode["recorded_every"]))
        if job["traj_path"] and traj is not None:
            save_traj(traj, job["traj_path"], job["traj_every"])
    except Exception as exc:
        traceback.print_exc()
        row["error_message"] = f"post-processing: {exc!r}"
    sys.stdout.flush()
    return {**episode_info(episode), **row}


def summarize(rows, policy_name, eval_set):
    """v1.1 summary (its stable-arrival line now uses the v2.0 horizon = T_MAX) plus the protocol v2.0 obstacle lines."""
    n = len(rows)

    def rate(k, m=n):
        lo, hi = v1.wilson_interval(k, m)
        return f"{k}/{m} = {100 * k / max(1, m):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"

    lines = [f"Protocol {PROTOCOL_VERSION} (T_MAX {T_MAX:g} s, contact ends the run)",
             v1.summarize(rows, policy_name, eval_set)]
    collided = [row for row in rows if bool(row.get("collision"))]
    lines.append(f"v2.0 collision: {rate(len(collided))}; links "
                 + (", ".join(f"{k} {v}" for k, v in sorted(_counts(row.get("collision_link") for row in collided)))
                    or "-"))
    no_pass = sum(not bool(row.get("passed_obstacle")) and not bool(row.get("collision")) for row in rows)
    lines.append(f"v2.0 passed_obstacle: {rate(sum(bool(row.get('passed_obstacle')) for row in rows))}; "
                 f"no_pass (never passed, no collision): {rate(no_pass)}")
    passes = [row for row in rows if bool(row.get("passed_obstacle"))]
    match = sum(row.get("side_taken") == row.get("scenario_side") for row in passes)
    lines.append(f"v2.0 side_taken == scenario_side among passes: {match}/{len(passes)}")
    lines.append(f"v2.0 docked (= v1.1 parked): {rate(sum(bool(row.get('docked')) for row in rows))}; "
                 f"docked_clean: {rate(sum(bool(row.get('docked_clean')) for row in rows))} (95% Wilson)")
    clearances = [float(row["min_clearance"]) for row in rows if v1._is_set(row.get("min_clearance"))]
    if clearances:
        lines.append(f"v2.0 min_clearance (m): min {min(clearances):.3f}, median {np.median(clearances):.3f}")
    for flag in (True, False):
        band = [row for row in rows if bool(row.get("ambiguous")) is flag]
        if band:
            lines.append(f"v2.0 {'inside' if flag else 'outside'} the ambiguous band: success "
                         f"{rate(sum(bool(row.get('success')) for row in band), len(band))}, docked_clean "
                         f"{rate(sum(bool(row.get('docked_clean')) for row in band), len(band))}")
    return "\n".join(lines)


def _counts(values):
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out.items()


def _selected_episodes(episodes, text, limit):
    by_id = {ep["episode"]: ep for ep in episodes}
    if text:
        selected = v1.parse_episode_selection(text, max(by_id) + 1)
        # v1 silently clips ids outside n; v2 must not silently omit a requested
        # uncollected episode from a resumable set.
        requested = set()
        for part in text.split(","):
            bounds = part.split("-")
            requested.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        missing = requested - by_id.keys()
        if missing:
            raise ValueError(f"requested episodes have no recordings: {sorted(missing)}")
    else:
        selected = sorted(by_id)
    if limit is not None:
        selected = selected[:limit]
    return [by_id[i] for i in selected]


def run_batch(args):
    check_sealed(args.eval_set, args.allow_test2)
    episodes, set_config = load_eval_set(args.eval_set)
    episodes = _selected_episodes(episodes, args.episodes, args.limit)
    if not episodes:
        raise ValueError("no episodes selected")
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    traj_dir = os.path.join(args.out_dir, "traj") if args.save_traj else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
    policy_name = args.name or os.path.basename(args.policy.rpartition(":")[0] or args.policy)
    load_policy(args.policy, args.policy_arg)
    with open(os.path.join(args.out_dir, "config.json"), "w") as handle:
        json.dump({"args": vars(args), "protocol": PROTOCOL, "eval_set_config": set_config,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "episodes": [ep["episode"] for ep in episodes]}, handle, indent=2)
    jobs = [{"episode": ep, "policy": args.policy, "policy_arg": args.policy_arg,
             "compare": args.compare_recorded, "early_stop": not args.no_early_stop,
             "traj_path": os.path.join(traj_dir, f"ep_{ep['episode']:04d}.npz") if traj_dir else None,
             "traj_every": args.traj_every,
             "log_path": os.path.join(args.out_dir, "logs", f"ep_{ep['episode']:04d}.log")}
            for ep in episodes]
    print(f"Evaluating {policy_name}: {len(jobs)} episodes, {args.workers} workers, {PROTOCOL_VERSION}")
    rows, start = [], time.time()
    path = os.path.join(args.out_dir, "rollouts.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for k, row in enumerate(pool.imap_unordered(_worker, jobs), 1):
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                elapsed = time.time() - start
                print(f"[{k}/{len(jobs)}] ep {row['episode']:04d} {str(row.get('stop_reason')):<14} "
                      f"min err {row.get('min_error', np.nan):.4f}m collision {row.get('collision')} "
                      f"| elapsed {elapsed / 60:.1f}min, ETA {elapsed / k * (len(jobs) - k) / 60:.1f}min")
    rows.sort(key=lambda row: row["episode"])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, policy_name, args.eval_set)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")
    print(summary)
    print(f"Evaluation time {(time.time() - start) / 60:.1f}min; results in {args.out_dir}")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True,
                        help="'expert', 'zero', 'path/to/file.py:factory' or 'package.module:factory'")
    parser.add_argument("--policy-arg", default=None, help="string passed to the policy factory (e.g. a model path)")
    parser.add_argument("--name", default=None, help="policy name used in the summary")
    parser.add_argument("--eval-set", required=True, help="set folder made by obstacle_task.collect")
    parser.add_argument("--workers", type=int, default=1, help="parallel simulation processes")
    parser.add_argument("--out-dir", required=True, help="output folder for rollouts.csv, summary.txt, logs")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="disable the 'stalled' stop (contacts always end the run)")
    parser.add_argument("--episodes", default=None, help="subset of episode ids, e.g. '0-19' or '3,7,10'")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N selected episodes")
    parser.add_argument("--compare-recorded", action="store_true",
                        help="compare each trajectory with the recorded clean episode (verifies the harness)")
    parser.add_argument("--save-traj", action="store_true",
                        help="save each trajectory (whole run, lidar and expert segment included) to <out-dir>/traj/")
    parser.add_argument("--traj-every", type=int, default=10, help="keep one saved sample every N steps")
    parser.add_argument("--gui", action="store_true", help="run the selected episodes one by one with the GUI")
    parser.add_argument("--allow-test2", action="store_true", help="allow the sealed test2 set")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.traj_every < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers, traj-every and limit must be positive")
    return args


def main():
    args = parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if args.gui:
        check_sealed(args.eval_set, args.allow_test2)
        episodes, _ = load_eval_set(args.eval_set)
        for ep in _selected_episodes(episodes, args.episodes, args.limit):
            row, _ = run_policy_episode(load_policy(args.policy, args.policy_arg), ep,
                                        gui=True, early_stop=not args.no_early_stop)
            print(row)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
