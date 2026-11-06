"""Expert data collection for the Robotnik base + Kinova arm reaching task (behaviour cloning).

Runs the provided expert (base_controller + arm_controller from robotnik_arm_control.py) towards many
targets and records, at every simulation step, the robot measurements (model inputs), the expert
commands (labels) and the expert's internal state. Each episode is saved as one compressed .npz file,
and every batch run writes an index.csv with one summary row per episode.

Examples:
  # one episode with the GUI, to check visually what is being recorded
  python collect_data.py --target 0.0 1.1 0.5 --gui --out-dir data/debug
  # 200 random targets on 4 worker processes, 30% of the episodes with command noise
  python collect_data.py --n-episodes 200 --workers 4 --noise-frac 0.3 --out-dir data/run1
  # same, with random start poses: within 2 m of the origin, any heading
  python collect_data.py --n-episodes 200 --workers 4 --init-radius 2.0 --init-yaw-range -180 180 --out-dir data/run2
  # single episode from a given start pose (x, y, yaw in degrees)
  python collect_data.py --target 1.0 0.5 0.6 --init-pose 1.0 -1.0 90 --gui --out-dir data/debug

Loading an episode:
  d = np.load("data/run1/ep_0000.npz")
  meta = json.loads(str(d["meta"]))
  X_parts = d["base_pos"], d["base_bearing"], d["q_mes"], d["qd_mes"], d["target"]
  Y_parts = d["wheel_cmds"], d["torque_arm"]
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pybullet as p
from simulation_and_control import pb, MotorCommands, PinWrapper
import pinocchio as pin
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)
# Expert settings are imported rather than copied, so the recorded expert is exactly the template one
from robotnik_arm_task import (
    FIXED_FINGER_ANGLES, BASE_NAV_TIMEOUT, ARM_REACH_LIMIT, STANDOFF_Z_POINTS, STANDOFF_POINTS,
    REACH_TOLERANCE,
)

# Phase label: the base navigation phases, plus "reach" once navigation is done. BaseNavState keeps
# phase == "orient" after it sets done = True, so phase alone cannot tell orient and reach apart.
PHASE_CODES = {"align": 0, "drive": 1, "orient": 2, "reach": 3}

# Per-step signals saved in each episode file (name -> dtype)
LOG_FIELDS = {
    # measurements (candidate model inputs)
    "t": np.float32,              # (T,)    simulation time (s)
    "base_pos": np.float32,       # (T, 3)  base position in the world frame (m)
    "base_ori": np.float32,       # (T, 4)  base orientation quaternion as returned by the simulator
    "base_bearing": np.float32,   # (T,)    base yaw (rad), wraps at +-pi
    "q_mes": np.float32,          # (T, 17) joint angles: wheels 0-3, arm 4-10, fingers 11-16
    "qd_mes": np.float32,         # (T, 17) joint velocities, same order
    # expert commands (labels), computed from the measurements above
    "wheel_cmds": np.float32,     # (T, 4)  wheel speeds [front_right, front_left, back_left, back_right] (rad/s)
    "torque_arm": np.float32,     # (T, 7)  arm joint torques (N*m)
    # commands actually sent to the simulator (= labels + noise; equal to the labels in clean episodes)
    "wheel_cmds_exec": np.float32,
    "torque_arm_exec": np.float32,
    # expert internal state (not measurable by a learned policy at test time)
    "q_des_arm": np.float32,      # (T, 7)  joint reference integrated by the differential IK
    "qd_des_arm": np.float32,     # (T, 7)  joint velocity reference
    "phase": np.int8,             # (T,)    PHASE_CODES
    "nav_done": np.bool_,         # (T,)    base navigation finished, i.e. the arm is reaching
    # for evaluation
    "ee_pos": np.float32,         # (T, 3)  end-effector position in the world frame (m)
}


class HeldNoise:
    """Piecewise-constant Gaussian noise: a new sample is drawn every `hold_steps` steps.

    Independent noise at every 1 ms step averages out through the robot's inertia and barely moves
    it off the expert trajectory. Holding each sample for a few hundred ms produces real deviations
    that the expert then has to correct, which is what we want the data to show.
    """

    def __init__(self, rng, std, hold_steps):
        self.rng = rng
        self.std = np.asarray(std, dtype=float)
        self.hold_steps = max(1, int(hold_steps))
        self.value = np.zeros_like(self.std)
        self.count = 0

    def sample(self):
        if self.count % self.hold_steps == 0:
            self.value = self.rng.normal(0.0, 1.0, self.std.shape) * self.std
        self.count += 1
        return self.value


def fade(distance, fade_distance):
    """Noise scale: 1 beyond fade_distance, then linearly down to 0 at the goal (fade_distance <= 0: always 1)."""
    return 1.0 if fade_distance <= 0 else min(1.0, distance / fade_distance)


def wheel_speeds(v, w, params):
    """(v, w) -> [front_right, front_left, back_left, back_right], same skid-steer model as base_controller."""
    lw = (v - w * params.wheel_base_width / 2) / params.wheel_radius
    rw = (v + w * params.wheel_base_width / 2) / params.wheel_radius
    return np.array([rw, lw, lw, rw])


def stop_status(stop_message):
    if stop_message.startswith("Target reached"):
        return "reached"
    if stop_message.startswith("Target not reachable"):
        return "unreachable"
    if stop_message.startswith("Reached step cap"):
        return "step_cap"
    return "quit"


def run_episode(target, out_path, init_base_xy=(0.0, 0.0), init_base_yaw=0.0, noise_std=None,
                noise_hold=0.3, noise_fade=(0.5, 0.15), seed=0, gui=False, max_steps=40000, log_every=1,
                verbose=True):
    """Run the expert once towards `target` and save the recorded episode to `out_path` (.npz).

    init_base_xy, init_base_yaw: initial base position (m) and yaw (rad) in the world frame.
    noise_std: dict with keys "v" (m/s), "w" (rad/s), "torque" (N*m), or None for a clean episode.
    The noise is added to the commands sent to the simulator, while the clean expert commands are
    stored as labels (DART-style), so the data shows how the expert recovers from deviations.
    noise_fade: (base, arm) distances (m) below which the noise fades out, see the main loop.
    Returns a summary dict (one row of index.csv).
    """
    wall_start = time.time()
    target = np.asarray(target, dtype=float)
    noise_std = noise_std or {}
    std_v, std_w, std_tau = (float(noise_std.get(k, 0.0)) for k in ("v", "w", "torque"))
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------------
    # Simulation setup (same as robotnik_arm_task.py)
    # --------------------------------------------------------------------------
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

    # Initial base pose. The config always spawns the robot at the origin facing +x, so the pose is reset
    # here per episode (editing the shared config file would race between parallel workers).
    # pybullet's base position is the base centre of mass, the same point sim.GetBasePosition() measures.
    spawn_pos, _ = client.getBasePositionAndOrientation(rid)
    client.resetBasePositionAndOrientation(
        rid, [init_base_xy[0], init_base_xy[1], spawn_pos[2]],
        client.getQuaternionFromEuler([0.0, 0.0, init_base_yaw]),
    )
    sim.ReceiveObservation()

    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    if gui:
        visual_shape_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target.tolist())

    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())
    init_bearing = quaternion2bearing(init_base_ori[3], init_base_ori[0], init_base_ori[1], init_base_ori[2])
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    time_step = sim.GetTimeStep()

    # --------------------------------------------------------------------------
    # Expert planning and controllers (as in robotnik_arm_task.py, except for the approach direction)
    # --------------------------------------------------------------------------
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target[:2]
    # MODIFIED: the template takes the approach direction from the world origin (arctan2 of target_xy),
    # which assumes the robot starts there. It is taken from the robot's initial position instead, so the
    # expert is unchanged for a start at the origin and still approaches the target directly from elsewhere.
    to_target = target_xy - init_base_pos[:2]
    target_bearing = np.arctan2(to_target[1], to_target[0]) if np.linalg.norm(to_target) > 1e-4 else init_bearing
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)

    cmd = MotorCommands()
    control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6

    hold_steps = round(noise_hold / time_step)
    base_noise = HeldNoise(rng, [std_v, std_w], hold_steps)
    arm_noise = HeldNoise(rng, [std_tau] * 7, hold_steps)

    log = {name: [] for name in LOG_FIELDS}
    step = 0
    current_time = 0.0
    nav_done_time = None

    # --------------------------------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------------------------------
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target - ee_pos)

        # Expert commands for the measured state: these are the labels
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params
        )
        # Read after the call: the controller can switch phase within a step, and the command
        # comes from the phase it ends in
        phase = base_state.phase
        nav_done = base_state.done
        if nav_done and nav_done_time is None:
            nav_done_time = current_time
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=nav_done,
        )

        # Commands actually executed = expert commands + noise. Base noise only while navigating:
        # once navigation is done the expert never moves the base again, so it could not correct it.
        # The noise fades out near the goals: it should create deviations the expert recovers from on the
        # way, not spoil the final precision, nor push the base off a goal it is about to reach (with its
        # limited turn rate while driving, the expert can then circle around the goal until the timeout)
        wheel_exec = wheel_cmds.copy()
        if not nav_done:
            dv, dw = base_noise.sample() * fade(np.linalg.norm(base_pos[:2] - base_goal_pos), noise_fade[0])
            wheel_exec += wheel_speeds(dv, dw, base_params)
        arm_scale = fade(ee_dist_error, noise_fade[1]) if nav_done else 1.0
        torque_exec = np.clip(torque_arm + arm_noise.sample() * arm_scale,
                              -arm_params.max_torque, arm_params.max_torque)

        # Log the measurements together with the commands computed from them (same time index)
        if step % log_every == 0:
            log["t"].append(current_time)
            log["base_pos"].append(base_pos)
            log["base_ori"].append(base_ori)
            log["base_bearing"].append(base_bearing)
            log["q_mes"].append(q_mes)
            log["qd_mes"].append(qd_mes)
            log["wheel_cmds"].append(wheel_cmds)
            log["torque_arm"].append(torque_arm)
            log["wheel_cmds_exec"].append(wheel_exec)
            log["torque_arm_exec"].append(torque_exec)
            log["q_des_arm"].append(q_des_arm)
            log["qd_des_arm"].append(qd_des_arm)
            log["phase"].append(PHASE_CODES["reach"] if nav_done else PHASE_CODES[phase])
            log["nav_done"].append(nav_done)
            log["ee_pos"].append(ee_pos)

        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (held by configure_gripper_fingers)
        cmd.SetControlCmd(np.concatenate((wheel_exec, torque_exec, np.zeros(6))), control_modes)
        sim.Step(cmd, "torque")

        if gui:
            keys = client.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & client.KEY_WAS_TRIGGERED:
                stop_message = "Stopped by user"
                break
            time.sleep(0.001)

        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, nav_done)
        if stop_message:
            break

        current_time += time_step
        step += 1
        if verbose and step % 1000 == 0:
            print(f"t={current_time:.2f}s | base {np.round(base_pos[:2], 3)} | EE error {ee_dist_error:.4f}m "
                  f"| phase {phase} nav_done {nav_done}")

    # --------------------------------------------------------------------------
    # Save the episode
    # --------------------------------------------------------------------------
    status = stop_status(stop_message)
    # The template's check_stop tests the stall condition before the reached condition: an arm that reaches
    # the target early and then waits for min_time is reported "not reachable" although it is on target
    if status == "unreachable" and ee_dist_error < REACH_TOLERANCE:
        status = "reached"
    meta = {
        "target": target.tolist(),
        "init_base_pos": init_base_pos.tolist(),
        "init_base_yaw": float(init_bearing),
        "status": status,
        "success": status == "reached",
        "stop_message": stop_message,
        "final_error": float(ee_dist_error),
        "duration": float(current_time),
        "n_steps": step + 1,
        "log_every": log_every,
        "time_step": time_step,
        "nav_done_time": nav_done_time,
        "nav_timed_out": nav_done_time is not None and nav_done_time > BASE_NAV_TIMEOUT,
        "standoff": standoff,
        "base_goal_pos": base_goal_pos.tolist(),
        "base_goal_bearing": float(base_goal_bearing),
        "noise_v": std_v,
        "noise_w": std_w,
        "noise_torque": std_tau,
        "noise_hold": noise_hold,
        "noise_fade_base": noise_fade[0],
        "noise_fade_arm": noise_fade[1],
        "seed": seed,
    }
    arrays = {name: np.asarray(log[name], dtype=dtype) for name, dtype in LOG_FIELDS.items()}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **arrays, target=target, meta=json.dumps(meta))

    if verbose:
        print(f"{stop_message}\nSaved {arrays['t'].shape[0]} samples to {out_path}")
    return {**meta, "file": os.path.basename(out_path), "wall_time": time.time() - wall_start}


def sample_targets(n, rng, r_range, theta_range_deg, z_range):
    """Sample targets uniformly in cylindrical coordinates (r, theta, z) relative to the robot start pose.

    theta is measured from the robot's initial heading. Returns r (m), theta (rad), z (m).
    """
    r = rng.uniform(*r_range, size=n)
    theta = np.deg2rad(rng.uniform(*theta_range_deg, size=n))
    z = rng.uniform(*z_range, size=n)
    return r, theta, z


def sample_init_poses(n, rng, radius, yaw_range_deg):
    """Sample initial base poses: position uniform over a disc of `radius` around the origin, yaw uniform."""
    rho = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))  # sqrt makes the density uniform over the disc area
    phi = rng.uniform(-np.pi, np.pi, size=n)
    xy = np.stack([rho * np.cos(phi), rho * np.sin(phi)], axis=1)
    yaw = np.deg2rad(rng.uniform(*yaw_range_deg, size=n))
    return xy, yaw


def relative_to_world(init_xy, init_yaw, r, theta, z):
    """Target given relative to the start pose (r, theta from the heading, z) -> world frame (x, y, z)."""
    heading = init_yaw + theta
    return np.array([init_xy[0] + r * np.cos(heading), init_xy[1] + r * np.sin(heading), z])


def _worker(job):
    """Run one episode in a fresh process, with its console output redirected to a log file."""
    # dup2 also captures the C-level prints of pybullet, which would otherwise flood the terminal
    fd = os.open(job["log_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        row = run_episode(**job["kwargs"])
    except Exception as e:
        traceback.print_exc()
        row = {"status": "error", "success": False, "stop_message": repr(e)}
    sys.stdout.flush()
    return {**job["info"], **row}


# r and theta_deg are relative to the robot start pose; target_* and init_* are in the world frame
INDEX_COLUMNS = [
    "episode", "file", "status", "success", "final_error", "duration", "n_steps", "nav_done_time",
    "nav_timed_out", "target_x", "target_y", "target_z", "r", "theta_deg", "init_x", "init_y", "init_yaw_deg",
    "noise_v", "noise_w", "noise_torque", "noise_hold", "noise_fade_base", "noise_fade_arm", "seed",
    "wall_time", "stop_message",
]
# Settings that define the dataset: a resumed run must use the same ones (workers etc. may change)
DATASET_ARGS = ["n_episodes", "seed", "r_range", "theta_range", "z_range", "init_radius", "init_yaw_range",
                "noise_frac", "noise_v", "noise_w", "noise_torque", "noise_hold", "noise_fade_base",
                "noise_fade_arm", "max_steps", "log_every"]


def row_from_npz(path, info):
    """Index row of an episode already saved by an earlier (interrupted) run, or None if unreadable."""
    try:
        with np.load(path) as d:
            meta = json.loads(str(d["meta"]))
    except Exception:  # e.g. the run was killed while this file was being written
        return None
    return {**info, **meta, "file": os.path.basename(path), "wall_time": 0.0}


def collect_batch(args):
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Draw order matters for reproducibility: targets and noise flags first (as in earlier runs), so the
    # same --seed with the default (fixed) start pose regenerates exactly the earlier datasets
    r, theta, z = sample_targets(args.n_episodes, rng, args.r_range, args.theta_range, args.z_range)
    noisy = rng.random(args.n_episodes) < args.noise_frac
    init_xy, init_yaw = sample_init_poses(args.n_episodes, rng, args.init_radius, args.init_yaw_range)
    noise_std = {"v": args.noise_v, "w": args.noise_w, "torque": args.noise_torque}

    jobs = []
    for i in range(args.n_episodes):
        name = f"ep_{i:04d}"
        target = relative_to_world(init_xy[i], init_yaw[i], r[i], theta[i], z[i])
        jobs.append({
            "log_path": os.path.join(out_dir, "logs", name + ".log"),
            "info": {"episode": i, "r": float(r[i]), "theta_deg": float(np.rad2deg(theta[i])),
                     "target_x": float(target[0]), "target_y": float(target[1]), "target_z": float(target[2]),
                     "init_x": float(init_xy[i, 0]), "init_y": float(init_xy[i, 1]),
                     "init_yaw_deg": float(np.rad2deg(init_yaw[i]))},
            "kwargs": {"target": target, "out_path": os.path.join(out_dir, name + ".npz"),
                       "init_base_xy": init_xy[i], "init_base_yaw": float(init_yaw[i]),
                       "noise_std": noise_std if noisy[i] else None, "noise_hold": args.noise_hold,
                       "noise_fade": (args.noise_fade_base, args.noise_fade_arm),
                       "seed": args.seed * 100_000 + i, "max_steps": args.max_steps,
                       "log_every": args.log_every},
        })

    # Save the settings of this run so the dataset can be regenerated exactly
    config_path = os.path.join(out_dir, "config.json")
    settings = {key: getattr(args, key) for key in DATASET_ARGS}
    rows = []
    if args.resume and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        changed = [key for key in DATASET_ARGS if saved.get(key) != settings[key]]
        if changed:
            raise SystemExit(f"--resume with different dataset settings than {config_path}: {changed}")
        for job in jobs:
            if os.path.exists(job["kwargs"]["out_path"]):
                row = row_from_npz(job["kwargs"]["out_path"], job["info"])
                if row is not None:
                    rows.append(row)
        done = {row["episode"] for row in rows}
        jobs = [job for job in jobs if job["info"]["episode"] not in done]
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Collecting {args.n_episodes} episodes ({noisy.sum()} noisy) into {out_dir} with {args.workers} "
          f"workers" + (f", {len(rows)} already done" if rows else ""))
    index_path = os.path.join(out_dir, "index.csv")
    start = time.time()
    # The index is written while collecting, so an interrupted run keeps the rows of finished episodes
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        # One fresh process per episode (maxtasksperchild=1): each simulation starts from a clean pybullet state
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for k, row in enumerate(pool.imap_unordered(_worker, jobs), start=1):
                rows.append(row)
                writer.writerow(row)
                f.flush()
                elapsed = time.time() - start
                eta = elapsed / k * (len(jobs) - k)
                print(f"[{len(rows)}/{args.n_episodes}] ep {row['episode']:04d} {row['status']:<11} "
                      f"err {row.get('final_error', float('nan')):.4f} m  sim {row.get('duration', 0):.1f}s  "
                      f"| elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")

    # Final index sorted by episode
    rows.sort(key=lambda row: row["episode"])
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    statuses = [row["status"] for row in rows]
    print("Summary: " + ", ".join(f"{s} {statuses.count(s)}" for s in sorted(set(statuses))))
    print(f"Collection time: {(time.time() - start) / 60:.1f} min for {len(jobs)} episodes")
    errors = [row for row in rows if row["status"] == "error"]
    if errors:
        first = errors[0]
        log_path = os.path.join(out_dir, "logs", "ep_%04d.log" % first["episode"])
        print(f"WARNING: {len(errors)} episodes crashed, first error: {first['stop_message']} "
              f"(full traceback in {log_path})")
    print(f"Index written to {index_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/run", help="output folder for .npz files, index.csv and logs")
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="run a single episode towards this target instead of a random batch")
    parser.add_argument("--name", default="single", help="file name of the single-target episode")
    parser.add_argument("--gui", action="store_true", help="show the pybullet GUI (single-target mode only)")
    parser.add_argument("--noisy", action="store_true", help="add command noise in single-target mode")
    parser.add_argument("--init-pose", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "YAW_DEG"), help="initial base pose in single-target mode")

    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="parallel simulation processes")
    parser.add_argument("--seed", type=int, default=0, help="seed for target sampling and noise")
    # Target relative to the start pose. Defaults = the chosen task range: the expert reaches every height
    # in 0.2-1.2 m (data/probe), and r is capped at 3 m to bound the episode length
    parser.add_argument("--r-range", type=float, nargs=2, default=[0.1, 3.0],
                        help="horizontal distance from the start position (m)")
    parser.add_argument("--theta-range", type=float, nargs=2, default=[-180.0, 180.0],
                        help="bearing from the start heading (deg)")
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.2, 1.2], help="height (m)")
    # Initial base pose; the defaults keep the template start (origin, facing +x)
    parser.add_argument("--init-radius", type=float, default=0.0,
                        help="start position sampled uniformly within this radius of the origin (m)")
    parser.add_argument("--init-yaw-range", type=float, nargs=2, default=[0.0, 0.0], help="start yaw (deg)")

    parser.add_argument("--noise-frac", type=float, default=0.0, help="fraction of batch episodes with noise")
    parser.add_argument("--noise-v", type=float, default=0.02, help="std of base linear speed noise (m/s)")
    parser.add_argument("--noise-w", type=float, default=0.05, help="std of base angular speed noise (rad/s)")
    parser.add_argument("--noise-torque", type=float, default=2.0, help="std of arm torque noise (N*m)")
    parser.add_argument("--noise-hold", type=float, default=0.3, help="time each noise sample is held (s)")
    parser.add_argument("--noise-fade-base", type=float, default=0.5,
                        help="base noise fades out within this distance of the base goal (m, 0 = no fade)")
    parser.add_argument("--noise-fade-arm", type=float, default=0.15,
                        help="arm noise fades out within this EE distance of the target while reaching (m, 0 = no fade)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted batch in --out-dir: keep saved episodes, run the missing ones")

    parser.add_argument("--max-steps", type=int, default=40000, help="step cap per episode (0 disables it)")
    parser.add_argument("--log-every", type=int, default=1, help="save one sample every N simulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target is not None:
        noise_std = {"v": args.noise_v, "w": args.noise_w, "torque": args.noise_torque} if args.noisy else None
        run_episode(args.target, os.path.join(args.out_dir, args.name + ".npz"),
                    init_base_xy=args.init_pose[:2], init_base_yaw=np.deg2rad(args.init_pose[2]),
                    noise_std=noise_std, noise_hold=args.noise_hold,
                    noise_fade=(args.noise_fade_base, args.noise_fade_arm), seed=args.seed, gui=args.gui,
                    max_steps=args.max_steps, log_every=args.log_every)
    else:
        collect_batch(args)


if __name__ == '__main__':
    main()
