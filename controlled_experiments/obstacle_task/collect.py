"""Expert data collection of the obstacle task (step 9, protocol v2.0): one cylinder between start and dock.

Runs the expert v2 (obstacle_task.expert: two base segments over the unchanged template controllers) on the scenarios
of a set (obstacle_task.scenario) and records every simulation step, like collect_data.py. Differences from the A-to-B
collector, nothing else: the obstacle (spawned right after the shared setup), the segments, nav_timeout 45 s, step cap
60 000 (60 s), and a robot-obstacle contact ends the episode with status "collision". Same DART noise model and
defaults (v 0.02 m/s, w 0.05 rad/s, torque 2.0 N*m, held 0.3 s; arm noise fades within 0.15 m of the target, base noise
only while navigating and fades within 0.5 m of the CURRENT segment goal: the waypoint, then the dock), labels = the
clean expert commands, same stop rule (template check_stop + collect_data's reached fix).

A set folder (default data/obstacle/<set-name>/) holds:
  scenarios.json  every scenario of the set (scenario.save_scenarios, format obstacle_scenarios_v1), also those not
                  collected yet (--episodes); the exact floats the harness replays
  config.json     the arguments of the last run (+ "protocol": "v2.0", "format": "obstacle_collection_v1")
  index.csv       one row per collected episode: collect_data.INDEX_COLUMNS + INDEX_COLUMNS_V2
  ep_XXXX.npz     every collect_data.LOG_FIELDS array (same names, shapes, dtypes; length T = n_steps) plus
                  segment (T,) int8 (0 = to the waypoint, 1 = to the dock / reaching), target (3,) float64 and
                  meta (0-d str, JSON): every collect_data meta key + scenario (the full Scenario.to_dict(): obstacles,
                  waypoint_xy, dock_xy, side, ambiguous, lateral_offset, noisy, ...), collided, collision_time,
                  collision_link, segment_switch_time, no_obstacle, ambiguous, side, lateral_offset, obstacle_r, noisy
Randomness: scenarios from the per-episode streams of scenario.py ([seed, episode, 9]; noisy flag [seed, episode, 10]);
the command noise from default_rng(seed * 100000 + episode), the collect_data convention (used by gate A2).

Examples (from the task folder):
  python -m obstacle_task.collect --set-name pilot2 --workers 20 --out-dir results/step9/smoke/pilot2
  python -m obstacle_task.collect --set-name train2 --workers 20 --episodes 0-99 --resume   # data/obstacle/train2
  python -m obstacle_task.collect --set-name a2 --no-obstacle --seed 909 --n-episodes 3 --noise-frac 1 \\
      --init-radius 2 --init-yaw-range -180 180 --out-dir results/step9/smoke/a2/obstacle      # gate A2
--no-obstacle samples the A-to-B task with collect_data's draw order and skips segment 0 (25 s timeout, 40 000 step
cap): the same loop must then reproduce collect_data.py bitwise. test2 (seed 301) is sealed: --allow-test2.
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pinocchio as pin
from simulation_and_control import MotorCommands

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import collect_data as legacy  # noqa: E402
from evaluate_policy import expert_base_goal, parse_episode_selection, setup_simulation  # noqa: E402
from robotnik_arm_control import StopCriteria, check_stop, quaternion2bearing  # noqa: E402
from robotnik_arm_task import REACH_TOLERANCE  # noqa: E402
from obstacle_task.expert import ExpertPolicy  # noqa: E402
from obstacle_task.scenario import Scenario, load_scenarios, make_scenarios, save_scenarios  # noqa: E402
from obstacle_task.sim import link_name, robot_obstacle_contact, spawn_obstacles  # noqa: E402

PROTOCOL_VERSION = "v2.0"
STEP_CAP = 60000             # steps (60 s at 1 kHz); --no-obstacle: collect_data's 40000
LOG_FIELDS = {**legacy.LOG_FIELDS, "segment": np.int8}
INDEX_COLUMNS_V2 = ["ambiguous", "side", "lateral_offset", "obstacle_r", "collided", "collision_time",
                    "collision_link", "segment_switch_time", "noisy"]
INDEX_COLUMNS = legacy.INDEX_COLUMNS + INDEX_COLUMNS_V2
# Protocol v2.0 sets (plan section 4): name -> (seed, episodes, noise_frac)
SET_DEFAULTS = {
    "train2": (3, 2000, 0.5), "pilot2": (3000, 100, 0.0), "pilot2n": (3000, 100, 1.0),
    "dev2": (300, 300, 0.0), "confirm2": (302, 300, 0.0), "test2": (301, 300, 0.0),
}
SEALED_SET, SEALED_SEED = "test2", SET_DEFAULTS["test2"][0]
# Settings that define a set: a resumed run must use the same ones (workers, episodes etc. may change)
DATASET_ARGS = [
    "set_name", "n_episodes", "seed", "no_obstacle", "r_range", "theta_range", "z_range", "init_radius",
    "init_yaw_range", "noise_frac", "noise_v", "noise_w", "noise_torque", "noise_hold", "noise_fade_base",
    "noise_fade_arm", "max_steps", "log_every",
]


def check_sealed(path, allow_test2=False):
    """Refuse A-to-B data/test always, and a test2 folder (a path component named test2) unless allow_test2.
    Used by the collector, the harness and the classifier before anything in the folder is opened."""
    parts = os.path.realpath(path).split(os.sep)
    if any(parts[i:i + 2] == ["data", "test"] for i in range(len(parts) - 1)):
        raise SystemExit(f"{path}: data/test is sealed and is never opened in step 9")
    if SEALED_SET in parts and not allow_test2:
        raise SystemExit(f"{path}: {SEALED_SET} is sealed (final frozen model only): pass --allow-test2")


def run_episode(scenario, out_path, noise_std=None, noise_hold=0.3, noise_fade=(0.5, 0.15), seed=0, gui=False,
                max_steps=None, log_every=1, verbose=True):
    """Run the expert v2 on one Scenario (or its dict) and save the episode to out_path (.npz, written atomically).

    noise_std: dict with keys "v" (m/s), "w" (rad/s), "torque" (N*m), or None for a clean episode (the scenario's
    noisy flag is applied by collect_batch). seed: the command-noise seed (collect_batch: seed * 100000 + episode).
    noise_fade: (base, arm) fade distances (m). max_steps None: STEP_CAP, or 40000 without obstacles.
    Returns a summary dict (one row of index.csv).
    """
    if isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)
    wall_start = time.time()
    target = np.asarray(scenario.target, dtype=float)
    no_obstacle = not scenario.obstacles
    if max_steps is None:
        max_steps = 40000 if no_obstacle else STEP_CAP
    noise_std = noise_std or {}
    std_v, std_w, std_tau = (float(noise_std.get(k, 0.0)) for k in ("v", "w", "torque"))
    rng = np.random.default_rng(seed)

    # Same setup as collect_data.run_episode (shared helper), then the obstacles before the first step
    sim, client, rid, info = setup_simulation(scenario.init_xy, scenario.init_yaw, gui=gui, target=target)
    try:
        obstacle_ids = spawn_obstacles(client, scenario.obstacles)
        expert = ExpertPolicy()
        expert.reset(target, {**info, "scenario": scenario})
        dyn_model, ee_frame_id, time_step = info["dyn_model"], info["ee_frame_id"], info["time_step"]
        stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
        cmd = MotorCommands()
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        # One generator for both noises, drawn in collect_data's order (base, then arm)
        hold_steps = round(noise_hold / time_step)
        base_noise = legacy.HeldNoise(rng, [std_v, std_w], hold_steps)
        arm_noise = legacy.HeldNoise(rng, [std_tau] * 7, hold_steps)
        log = {name: [] for name in LOG_FIELDS}
        step, current_time = 0, 0.0
        nav_done_time = collision_time = collision_link = None
        collided = False

        while True:
            # Measure current states
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))
            q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
            pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
            pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
            ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
            ee_dist_error = np.linalg.norm(target - ee_pos)

            # Expert commands (labels); its state is read after the call (a segment switch happens inside it)
            wheel_cmds, torque_arm = expert.act({
                "t": current_time, "base_pos": base_pos, "base_ori": base_ori, "base_bearing": base_bearing,
                "q_mes": q_mes, "qd_mes": qd_mes, "target": target,
            })
            phase = expert.base_state.phase
            nav_done = expert.segment == 1 and expert.base_state.done
            if nav_done and nav_done_time is None:
                nav_done_time = current_time

            # Executed = labels + noise (collect_data's rules; the base noise fades near the CURRENT segment goal)
            wheel_exec = wheel_cmds.copy()
            if not nav_done:
                goal_distance = np.linalg.norm(base_pos[:2] - expert.current_goal_pos)
                dv, dw = base_noise.sample() * legacy.fade(goal_distance, noise_fade[0])
                wheel_exec += legacy.wheel_speeds(dv, dw, expert.base_params)
            arm_scale = legacy.fade(ee_dist_error, noise_fade[1]) if nav_done else 1.0
            torque_exec = np.clip(torque_arm + arm_noise.sample() * arm_scale,
                                  -expert.arm_params.max_torque, expert.arm_params.max_torque)

            if step % log_every == 0:
                values = {
                    "t": current_time, "base_pos": base_pos, "base_ori": base_ori, "base_bearing": base_bearing,
                    "q_mes": q_mes, "qd_mes": qd_mes, "wheel_cmds": wheel_cmds, "torque_arm": torque_arm,
                    "wheel_cmds_exec": wheel_exec, "torque_arm_exec": torque_exec,
                    "q_des_arm": expert.q_des_arm, "qd_des_arm": expert.qd_des_arm,
                    "phase": legacy.PHASE_CODES["reach"] if nav_done else legacy.PHASE_CODES[phase],
                    "nav_done": nav_done, "ee_pos": ee_pos, "segment": expert.segment,
                }
                for name, value in values.items():
                    log[name].append(value)

            cmd.SetControlCmd(np.concatenate((wheel_exec, torque_exec, np.zeros(6))), control_modes)
            sim.Step(cmd, "torque")

            # Contact ends the episode at once. It shows in this step's contact manifold, i.e. at the measured
            # state of step + 1: the harness records it there too (same collision_time, exact step multiple)
            collided, link = robot_obstacle_contact(client, rid, obstacle_ids)
            if collided:
                collision_time = (step + 1) * time_step
                collision_link = link_name(client, rid, link)
                stop_message = f"Robot-obstacle contact at t={collision_time:.3f}s ({collision_link})"
                break
            if gui:
                keys = client.getKeyboardEvents()
                if ord("q") in keys and keys[ord("q")] & client.KEY_WAS_TRIGGERED:
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
                      f"| segment {expert.segment} phase {phase} nav_done {nav_done}")

        status = "collision" if collided else legacy.stop_status(stop_message)
        # collect_data's fix of the template stop order: on target but reported "not reachable"
        if status == "unreachable" and ee_dist_error < REACH_TOLERANCE:
            status = "reached"
        meta = {
            # the collect_data meta keys, same meanings (duration = t of the last sample, n_steps = samples)
            "target": target.tolist(), "init_base_pos": info["init_base_pos"].tolist(),
            "init_base_yaw": float(info["init_base_bearing"]), "status": status, "success": status == "reached",
            "stop_message": stop_message, "final_error": float(ee_dist_error), "duration": float(current_time),
            "n_steps": step + 1, "log_every": log_every, "time_step": time_step, "nav_done_time": nav_done_time,
            "nav_timed_out": nav_done_time is not None and nav_done_time > expert.base_params.nav_timeout,
            "standoff": float(scenario.standoff), "base_goal_pos": list(scenario.dock_xy),
            "base_goal_bearing": float(scenario.dock_bearing),
            "noise_v": std_v, "noise_w": std_w, "noise_torque": std_tau, "noise_hold": noise_hold,
            "noise_fade_base": noise_fade[0], "noise_fade_arm": noise_fade[1], "seed": seed,
            # protocol v2.0
            "scenario": scenario.to_dict(), "no_obstacle": no_obstacle, "collided": bool(collided),
            "collision_time": collision_time, "collision_link": collision_link,
            "segment_switch_time": expert.segment_switch_time, "ambiguous": bool(scenario.ambiguous),
            "side": int(scenario.side), "lateral_offset": float(scenario.lateral_offset),
            "obstacle_r": float(scenario.obstacles[0]["r"]) if scenario.obstacles else 0.0,
            "noisy": bool(scenario.noisy),
        }
        arrays = {name: np.asarray(log[name], dtype=dtype) for name, dtype in LOG_FIELDS.items()}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        # Written to a temporary file first: a resumed run never takes an interrupted write for a saved episode
        temporary = str(out_path) + ".tmp"
        with open(temporary, "wb") as f:
            np.savez_compressed(f, **arrays, target=target, meta=json.dumps(meta))
        os.replace(temporary, out_path)
        if verbose:
            print(f"{stop_message}\nSaved {len(arrays['t'])} samples to {out_path}")
        return {**meta, "file": os.path.basename(out_path), "wall_time": time.time() - wall_start}
    finally:
        client.disconnect()


def make_no_obstacle_scenarios(args):
    """collect_data.collect_batch's targets, noise flags and start poses (same draw order) as obstacle-free
    Scenarios: the dock is evaluate_policy.expert_base_goal, segment 0 is skipped."""
    rng = np.random.default_rng(args.seed)
    r, theta, z = legacy.sample_targets(args.n_episodes, rng, args.r_range, args.theta_range, args.z_range)
    noisy = rng.random(args.n_episodes) < args.noise_frac
    xy, yaw = legacy.sample_init_poses(args.n_episodes, rng, args.init_radius, args.init_yaw_range)
    scenarios = []
    for ep in range(args.n_episodes):
        target = legacy.relative_to_world(xy[ep], yaw[ep], r[ep], theta[ep], z[ep])
        dock, bearing, standoff = expert_base_goal(target, np.r_[xy[ep], 0.0], yaw[ep])
        scenarios.append(Scenario(
            seed=args.seed, episode=ep, target=target.tolist(), init_xy=xy[ep].tolist(), init_yaw=float(yaw[ep]),
            dock_xy=dock.tolist(), dock_bearing=float(bearing), standoff=standoff, obstacles=[], along=0.0,
            lateral_offset=0.0, ambiguous=False, side=0, waypoint_xy=dock.tolist(), noisy=bool(noisy[ep]),
            meta={"no_obstacle": True, "r": float(r[ep]), "theta_deg": float(np.rad2deg(theta[ep]))},
        ))
    return scenarios


def scenario_index_info(scenario):
    """index.csv columns known before the run (also written for an episode that crashes)."""
    delta = np.asarray(scenario.target[:2]) - scenario.init_xy
    theta = np.arctan2(delta[1], delta[0]) - scenario.init_yaw
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return {
        "episode": scenario.episode,
        "r": scenario.meta.get("r", float(np.linalg.norm(delta))),
        "theta_deg": scenario.meta.get("theta_deg", float(np.rad2deg(theta))),
        "target_x": scenario.target[0], "target_y": scenario.target[1], "target_z": scenario.target[2],
        "init_x": scenario.init_xy[0], "init_y": scenario.init_xy[1],
        "init_yaw_deg": float(np.rad2deg(scenario.init_yaw)),
        "ambiguous": scenario.ambiguous, "side": scenario.side, "lateral_offset": scenario.lateral_offset,
        "obstacle_r": scenario.obstacles[0]["r"] if scenario.obstacles else 0.0, "noisy": scenario.noisy,
    }


def _worker(job):
    """Run one episode in a fresh process, with its console output (pybullet's included) in a log file."""
    fd = os.open(job["log_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        row = run_episode(**job["kwargs"])
    except Exception as exc:
        traceback.print_exc()
        row = {"status": "error", "success": False, "stop_message": repr(exc)}
    sys.stdout.flush()
    sys.stderr.flush()
    return {**job["info"], **row}


def plan_jobs(args, scenarios):
    """(rows of the episodes already saved, jobs to run). With --resume an episode whose .npz is readable and was
    recorded on the same scenario is kept; the others among --episodes are (re-)run."""
    selected = set(parse_episode_selection(args.episodes, args.n_episodes))
    noise_std = {"v": args.noise_v, "w": args.noise_w, "torque": args.noise_torque}
    rows, jobs = [], []
    for sc in scenarios:
        name = f"ep_{sc.episode:04d}"
        path = os.path.join(args.out_dir, name + ".npz")
        info = scenario_index_info(sc)
        if args.resume and os.path.exists(path):
            row = legacy.row_from_npz(path, info)
            if row is not None and row.get("scenario") == sc.to_dict():
                rows.append(row)
                continue
        if sc.episode not in selected:
            continue
        jobs.append({
            "log_path": os.path.join(args.out_dir, "logs", name + ".log"), "info": info,
            "kwargs": {"scenario": sc, "out_path": path, "noise_std": noise_std if sc.noisy else None,
                       "noise_hold": args.noise_hold, "noise_fade": (args.noise_fade_base, args.noise_fade_arm),
                       "seed": args.seed * 100_000 + sc.episode, "max_steps": args.max_steps,
                       "log_every": args.log_every},
        })
    return rows, jobs


def write_index(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_batch(args):
    """Collect the selected episodes of a set; --resume checks the settings and the scenarios and keeps saved
    episodes. Returns the index rows (all saved episodes, sorted)."""
    out_dir = args.out_dir
    check_sealed(out_dir, args.allow_test2)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    config_path = os.path.join(out_dir, "config.json")
    scenarios_path = os.path.join(out_dir, "scenarios.json")
    settings = {key: getattr(args, key) for key in DATASET_ARGS}
    if args.resume and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        changed = [key for key in DATASET_ARGS if saved.get(key) != settings[key]]
        if changed:
            raise SystemExit(f"--resume with different dataset settings than {config_path}: {changed}")
    elif not args.resume and glob.glob(os.path.join(out_dir, "ep_*.npz")):
        # a fresh run must not mix with episodes of an earlier run (a later --resume would keep them)
        raise SystemExit(f"{out_dir} already holds episodes: use --resume, or a new / emptied folder")
    scenarios = (make_no_obstacle_scenarios(args) if args.no_obstacle else
                 make_scenarios(args.seed, args.n_episodes, args.noise_frac))
    if args.resume and os.path.exists(scenarios_path):
        previous, _ = load_scenarios(scenarios_path)
        if [s.to_dict() for s in previous] != [s.to_dict() for s in scenarios]:
            raise SystemExit(f"--resume: {scenarios_path} differs from the scenarios of these settings")
    else:
        save_scenarios(scenarios_path, scenarios, extra={"set_name": args.set_name, "seed": args.seed,
                                                         "noise_frac": args.noise_frac,
                                                         "no_obstacle": args.no_obstacle})
    with open(config_path, "w") as f:
        json.dump({**vars(args), "protocol": PROTOCOL_VERSION, "format": "obstacle_collection_v1"}, f, indent=2)

    rows, jobs = plan_jobs(args, scenarios)
    print(f"Collecting {len(jobs)} episodes ({sum(j['kwargs']['noise_std'] is not None for j in jobs)} noisy) "
          f"of {args.set_name} into {out_dir} with {args.workers} workers; {len(rows)} already saved")
    index_path = os.path.join(out_dir, "index.csv")
    start = time.time()
    # The index is written while collecting, so an interrupted run keeps the rows of finished episodes
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        if jobs:
            # One fresh process per episode (maxtasksperchild=1): each simulation starts from a clean pybullet state
            with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
                for k, row in enumerate(pool.imap_unordered(_worker, jobs), start=1):
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()
                    elapsed = time.time() - start
                    print(f"[{k}/{len(jobs)}] ep {row['episode']:04d} {row['status']:<11} "
                          f"err {row.get('final_error', float('nan')):.4f} m sim {row.get('duration', 0):.1f}s "
                          f"| elapsed {elapsed / 60:.1f} min, ETA {elapsed / k * (len(jobs) - k) / 60:.1f} min",
                          flush=True)
    rows.sort(key=lambda row: row["episode"])
    write_index(index_path, rows)
    statuses = [row["status"] for row in rows]
    print("Summary: " + ", ".join(f"{s} {statuses.count(s)}" for s in sorted(set(statuses))))
    print(f"Collection time: {(time.time() - start) / 60:.1f} min; index written to {index_path}")
    errors = [row for row in rows if row["status"] == "error"]
    if errors:
        print(f"WARNING: {len(errors)} episodes crashed, first: {errors[0]['stop_message']} "
              f"(traceback in {out_dir}/logs/ep_{errors[0]['episode']:04d}.log)")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-name", default="run",
                        help=f"set name; {sorted(SET_DEFAULTS)} give the plan's seed / episodes / noise_frac")
    parser.add_argument("--seed", type=int, help="scenario seed (default: the set's)")
    parser.add_argument("--n-episodes", type=int, help="episodes of the set (default: the set's)")
    parser.add_argument("--noise-frac", type=float, help="fraction of noisy episodes (default: the set's)")
    parser.add_argument("--workers", type=int, default=20, help="parallel simulation processes")
    parser.add_argument("--out-dir", help="set folder (default data/obstacle/<set-name>)")
    parser.add_argument("--resume", action="store_true",
                        help="continue a set in --out-dir: keep saved episodes, run the missing selected ones")
    parser.add_argument("--episodes", help="subset to collect, e.g. '0-99' or '3,7,10' (scenarios.json has all)")
    parser.add_argument("--allow-test2", action="store_true", help="allow building the sealed test2 set")
    parser.add_argument("--no-obstacle", action="store_true", help="A-to-B equivalence mode (gate A2)")
    parser.add_argument("--noise-v", type=float, default=0.02, help="std of base linear speed noise (m/s)")
    parser.add_argument("--noise-w", type=float, default=0.05, help="std of base angular speed noise (rad/s)")
    parser.add_argument("--noise-torque", type=float, default=2.0, help="std of arm torque noise (N*m)")
    parser.add_argument("--noise-hold", type=float, default=0.3, help="time each noise sample is held (s)")
    parser.add_argument("--noise-fade-base", type=float, default=0.5,
                        help="base noise fades out within this distance of the current segment goal (m)")
    parser.add_argument("--noise-fade-arm", type=float, default=0.15,
                        help="arm noise fades out within this EE distance of the target while reaching (m)")
    parser.add_argument("--max-steps", type=int, help=f"step cap (default {STEP_CAP}; 40000 with --no-obstacle)")
    parser.add_argument("--log-every", type=int, default=1, help="save one sample every N simulation steps")
    # A-to-B sampling (--no-obstacle only), collect_data's defaults
    parser.add_argument("--r-range", type=float, nargs=2, default=[0.1, 3.0], help="--no-obstacle only")
    parser.add_argument("--theta-range", type=float, nargs=2, default=[-180.0, 180.0], help="--no-obstacle only")
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.2, 1.2], help="--no-obstacle only")
    parser.add_argument("--init-radius", type=float, default=0.0, help="--no-obstacle only")
    parser.add_argument("--init-yaw-range", type=float, nargs=2, default=[0.0, 0.0], help="--no-obstacle only")
    args = parser.parse_args(argv)
    defaults = SET_DEFAULTS.get(args.set_name, (0, 10, 0.0))
    for key, value in zip(("seed", "n_episodes", "noise_frac"), defaults):
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.out_dir is None:
        args.out_dir = os.path.join("data", "obstacle", args.set_name)
    if args.max_steps is None:
        args.max_steps = 40000 if args.no_obstacle else STEP_CAP
    if args.seed < 0 or args.n_episodes < 1 or args.workers < 1 or args.log_every < 1 or args.max_steps < 0:
        parser.error("seed / max-steps must be nonnegative; n-episodes / workers / log-every must be positive")
    if not 0 <= args.noise_frac <= 1:
        parser.error("noise-frac must lie in [0, 1]")
    if min(args.noise_v, args.noise_w, args.noise_torque, args.noise_hold) < 0:
        parser.error("noise standard deviations and hold time must be nonnegative")
    sealed = args.set_name == SEALED_SET or (args.seed == SEALED_SEED and not args.no_obstacle)
    if sealed and not args.allow_test2:
        parser.error(f"{SEALED_SET} (seed {SEALED_SEED}) is sealed: built only for the final frozen model, "
                     f"pass --allow-test2")
    return args


def main():
    rows = collect_batch(parse_args())
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
