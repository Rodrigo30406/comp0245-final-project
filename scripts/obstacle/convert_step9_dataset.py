"""Convert a step-9 obstacle collection (the group's `obstacle_task/collect.py` sets, expert v2 with one cylinder and a
64-beam lidar) into this project's dataset layout, so the egocentric / GRU / transformer trainers can learn it.

Source set folder (`final/robotnik_arm_task/data/obstacle/<set>/`): scenarios.json, config.json, index.csv and one
ep_XXXX.npz per episode with collect_data.LOG_FIELDS arrays (t, base_pos, base_bearing, q_mes (17), qd_mes (17),
wheel_cmds (4), torque_arm (7), their executed twins, q_des_arm, qd_des_arm, phase, nav_done, ee_pos, segment),
target and a JSON meta (status, collided, nav_timed_out, the full scenario with obstacles and side, noise settings).

Output dataset (`datasets/<name>/`): schema.json, config.json, scenarios.json, dataset.json and episodes/<id>.{npz,json}
in the recorder's field layout (schema.FIELDS: observation (21) = base xyz, bearing, 7 arm q, 7 arm qd, target;
action (11) = 4 wheels + 7 torques, the clean expert command) plus one extra per-row array `lidar` (N, 64): the planar
scan recomputed from the logged base pose and the episode's obstacles with the exact ray-circle caster of
`obstacle_task/lidar.py` (vendored below, checked against the original when it is importable). The episode metadata
carries `success` = expert reached the target, no contact, navigation not timed out (the step-9 learners' keep rule),
`termination` = the collector's status, and the scenario (obstacles, side, ambiguous band, lateral offset, noisy flag).

Then an explicit episode-level split (training / validation by episode, expert failures kept in the split: the trainers
drop them with keep_expert_failures=false) is written with the inventory fingerprint, like runs/splits/*.json.

Usage:
  python scripts/obstacle/convert_step9_dataset.py --source <set dir> --output datasets/<name> \
      --split-out runs/splits/<name>_split.json [--validation-fraction 0.15] [--split-seed 20261104] [--workers 8]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_learning.data.schema import FIELDS, dataset_schema  # noqa: E402
from robot_learning.data.storage import git_info, provenance, read_json, sha256, utc_now, write_json  # noqa: E402
from robot_learning.settings import get_roboenv_root  # noqa: E402
from robot_learning.data.splits import inventory, inventory_fingerprint, split_episodes  # noqa: E402
from robot_learning.simulation.config import resolve_episode_config  # noqa: E402

PHASE_NAMES = {0: "align", 1: "drive", 2: "orient", 3: "reach"}   # collect_data.PHASE_CODES of the group repository
ARM = slice(4, 11)                                                  # arm joints inside the 17-joint vectors
N_BEAMS, MAX_RANGE = 64, 4.0                                        # obstacle_task/lidar.py constants
DT = 0.001


def scan_batch(base_xy, yaw, obstacles, n_beams=N_BEAMS, max_range=MAX_RANGE):
    """Ranges (N, n_beams): vendored from the group's obstacle_task/lidar.py (exact 2D ray-circle casting, beam 0
    forward, counter-clockwise, clipped at max_range, range 0 inside an obstacle, robot body ignored)."""
    base_xy = np.atleast_2d(np.asarray(base_xy, dtype=float))
    yaw = np.atleast_1d(np.asarray(yaw, dtype=float))
    ang = yaw[:, None] + (2.0 * np.pi * np.arange(n_beams) / n_beams)[None, :]
    ux, uy = np.cos(ang), np.sin(ang)
    ranges = np.full(ang.shape, float(max_range))
    for ob in obstacles:
        cx = float(ob["x"]) - base_xy[:, 0]
        cy = float(ob["y"]) - base_xy[:, 1]
        c2 = cx * cx + cy * cy - float(ob["r"]) ** 2
        b = ux * cx[:, None] + uy * cy[:, None]
        disc = b * b - c2[:, None]
        near = b - np.sqrt(np.maximum(disc, 0.0))
        hit = (disc >= 0.0) & (near >= 0.0)
        t = np.where(hit, near, np.inf)
        t = np.where(c2[:, None] < 0.0, 0.0, t)
        ranges = np.minimum(ranges, t)
    return ranges


def check_against_original(source):
    """Compare the vendored caster with the group's module on random poses when that module can be imported."""
    task_dir = Path(source).resolve()
    while task_dir != task_dir.parent and not (task_dir / "obstacle_task" / "lidar.py").exists():
        task_dir = task_dir.parent
    if not (task_dir / "obstacle_task" / "lidar.py").exists():
        return "original lidar module not found; vendored caster unchecked"
    sys.path.insert(0, str(task_dir))
    try:
        from obstacle_task import lidar as original  # noqa: E402
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"original lidar module not importable ({exc!r}); vendored caster unchecked"
    rng = np.random.default_rng(0)
    xy, yaw = rng.uniform(-4, 4, (500, 2)), rng.uniform(-np.pi, np.pi, 500)
    obstacles = [{"x": 0.3, "y": -0.2, "r": 0.3, "h": 1.5}, {"x": -1.5, "y": 2.0, "r": 0.25, "h": 1.5}]
    diff = float(np.max(np.abs(scan_batch(xy, yaw, obstacles) - original.scan_batch(xy, yaw, obstacles))))
    if diff > 1e-12:
        raise RuntimeError(f"vendored lidar caster differs from obstacle_task/lidar.py by {diff}")
    return f"vendored lidar caster equals obstacle_task/lidar.py on 500 poses (max |diff| {diff:.1e})"


def convert_episode(job):
    src, dst_dir, episode_id, scenario_id, index_row = job["src"], Path(job["dst_dir"]), job["episode_id"], job["scenario_id"], job["row"]
    with np.load(src, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files if k != "meta"}
        meta = json.loads(str(d["meta"]))
    n = len(arrays["t"])
    if n < 2 or meta.get("log_every", 1) != 1 or abs(meta["time_step"] - DT) > 1e-12:
        raise ValueError(f"{src}: dense 1 ms recordings only")
    time = np.arange(n, dtype=np.float64) * DT
    if np.max(np.abs(arrays["t"].astype(np.float64) - time)) > 2e-2:
        raise ValueError(f"{src}: recorded clock does not match step * dt")
    target = np.asarray(arrays["target"], dtype=np.float64)
    base_pos = arrays["base_pos"].astype(np.float64)
    bearing = arrays["base_bearing"].astype(np.float64)
    q, qd = arrays["q_mes"].astype(np.float64), arrays["qd_mes"].astype(np.float64)
    observation = np.concatenate([base_pos, bearing[:, None], q[:, ARM], qd[:, ARM], np.broadcast_to(target, (n, 3))], axis=1).astype(np.float32)
    action = np.concatenate([arrays["wheel_cmds"], arrays["torque_arm"]], axis=1).astype(np.float32)
    if not (np.array_equal(action[:, 0], action[:, 3]) and np.array_equal(action[:, 1], action[:, 2])):
        raise ValueError(f"{src}: wheel commands do not keep the same-side identity")
    applied = np.concatenate([arrays["wheel_cmds_exec"], arrays["torque_arm_exec"]], axis=1).astype(np.float32)
    previous = np.vstack([np.zeros((1, 11), dtype=np.float32), applied[:-1]])
    obstacles = meta["scenario"]["obstacles"]
    if "lidar" in arrays:   # moving obstacle (scripts/obstacle/dynamic_expert.py): the logged scan is the only true one
        lidar = arrays["lidar"].astype(np.float32)
    else:
        lidar = scan_batch(base_pos[:, :2], bearing, obstacles).astype(np.float32)
    ee = arrays["ee_pos"].astype(np.float32)
    distance = np.linalg.norm(ee.astype(np.float64) - target, axis=1)
    next_obs = np.vstack([observation[1:], observation[-1:]])
    out = {
        "observation": observation, "next_observation": next_obs,
        "action": action, "applied_action": applied, "previous_action": previous,
        "previous_action_valid": np.r_[False, np.ones(n - 1, dtype=bool)],
        "time": time, "next_time": time + DT, "step": np.arange(n, dtype=np.int64),
        "base_position": base_pos.astype(np.float32), "base_orientation_xyzw": arrays["base_ori"].astype(np.float32),
        "base_bearing": bearing.astype(np.float32),
        "joint_position": q.astype(np.float32), "joint_velocity": qd.astype(np.float32),
        "target": np.broadcast_to(target.astype(np.float32), (n, 3)).copy(),
        "ee_position": ee, "next_ee_position": np.vstack([ee[1:], ee[-1:]]),
        "ee_error": (target - ee.astype(np.float64)).astype(np.float32), "ee_distance": distance.astype(np.float32),
        "expert_phase": np.array([PHASE_NAMES[int(p)] for p in arrays["phase"]], dtype="U8"),
        "base_navigation_done": arrays["nav_done"].astype(bool),
        "expert_q_des_arm": arrays["q_des_arm"].astype(np.float32), "expert_qd_des_arm": arrays["qd_des_arm"].astype(np.float32),
        "expert_segment": arrays["segment"].astype(np.int8),
        "lidar": lidar,
        "terminal_observation": observation[-1], "terminal_ee_position": ee[-1], "terminal_time": np.asarray(time[-1]),
    }
    for name, (width, dtype) in FIELDS.items():
        if name in out and out[name].dtype != np.dtype(dtype) and dtype != "U8":
            out[name] = out[name].astype(dtype)
    episodes = dst_dir / "episodes"
    data_path = episodes / f"{episode_id}.npz"
    tmp = data_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **out)
    tmp.replace(data_path)
    sc = meta["scenario"]
    success = meta["status"] == "reached" and not meta["collided"] and not meta["nav_timed_out"]
    metadata = {
        "episode_id": episode_id, "scenario_id": scenario_id, "status": "complete", "success": bool(success),
        "termination": meta["status"], "termination_message": meta["stop_message"],
        "expert_success_raw": bool(meta["success"]), "collided": bool(meta["collided"]),
        "collision_time_s": meta["collision_time"], "collision_link": meta["collision_link"],
        "navigation_timeout": bool(meta["nav_timed_out"]), "navigation_end_s": meta["nav_done_time"],
        "segment_switch_time_s": meta["segment_switch_time"],
        "target": [float(v) for v in target], "initial_base": {"x": float(sc["init_xy"][0]), "y": float(sc["init_xy"][1]), "yaw": float(sc["init_yaw"])},
        "actual_initial_observation": observation[0].astype(np.float64).tolist(),
        "dt": DT, "sample_period_s": DT, "configured_step_cap": int(index_row.get("max_steps", 60000) or 60000),
        "steps": n, "recorded_transitions": n, "duration_s": float(time[-1]),
        "final_distance_m": float(distance[-1]), "minimum_distance_m": float(distance.min()),
        "data_file": data_path.name, "data_sha256": sha256(data_path),
        "seed": int(meta["seed"]), "noisy": bool(meta["noisy"]),
        "action_noise": {k: meta[k] for k in ("noise_v", "noise_w", "noise_torque", "noise_hold", "noise_fade_base", "noise_fade_arm")},
        "obstacle": {"obstacles": obstacles, "side": int(sc["side"]), "ambiguous": bool(sc["ambiguous"]),
                     "lateral_offset": float(sc["lateral_offset"]), "along": float(sc["along"]),
                     "waypoint_xy": sc["waypoint_xy"], "dock_xy": sc["dock_xy"], "dock_bearing": float(sc["dock_bearing"]),
                     "standoff": float(sc["standoff"])},
        "base_goal_xy": sc["dock_xy"], "base_goal_bearing": float(sc["dock_bearing"]),
        "source_file": str(src), "source_set_episode": int(sc["episode"]), "converted_at": utc_now(),
    }
    write_json(episodes / f"{episode_id}.json", metadata)
    return episode_id, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="step-9 set folder (scenarios.json, index.csv, ep_XXXX.npz)")
    parser.add_argument("--output", required=True, help="dataset folder to create (must not exist)")
    parser.add_argument("--split-out", required=True, help="explicit split JSON to write")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20261104)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"{output} exists; datasets are never overwritten")
    note = check_against_original(source)
    print(note)
    src_config = read_json(source / "config.json")
    with (source / "index.csv").open() as f:
        rows = {int(r["episode"]): r for r in csv.DictReader(f)}
    src_scenarios = read_json(source / "scenarios.json")["scenarios"]
    (output / "episodes").mkdir(parents=True)
    jobs, scenarios = [], []
    for sc in src_scenarios:
        row = rows.get(int(sc["episode"]))
        if row is None or not row.get("file") or row["status"] == "error":
            continue
        initial = {"x": float(sc["init_xy"][0]), "y": float(sc["init_xy"][1]), "yaw": float(sc["init_yaw"])}
        target = [float(v) for v in sc["target"]]
        condition = {"target": target, "initial_base": initial}
        scenario_id = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:20]
        episode_id = f"episode_{int(sc['episode']):06d}"
        scenarios.append({"episode_id": episode_id, "scenario_id": scenario_id, "seed": int(src_config["seed"]) * 100000 + int(sc["episode"]),
                          "split": "development", **condition, "obstacles": sc["obstacles"], "side": int(sc["side"]),
                          "ambiguous": bool(sc["ambiguous"]), "lateral_offset": float(sc["lateral_offset"]), "noisy": bool(sc["noisy"])})
        jobs.append({"src": str(source / row["file"]), "dst_dir": str(output), "episode_id": episode_id,
                     "scenario_id": scenario_id, "row": row})
    print(f"converting {len(jobs)} recorded episodes of {len(src_scenarios)} scenarios")
    episode_cfg = resolve_episode_config({"base_controller": {"nav_timeout": 45.0}, "max_steps": 60000})
    config = {**episode_cfg, "name": f"step9_{src_config.get('set_name', source.name)}", "episodes": len(jobs),
              "seed": int(src_config["seed"]), "initial_base": {"mode": "step9_scenario_sampler"},
              "action_noise": {"model": "collect_data.HeldNoise of the group repository", "noise_frac": src_config.get("noise_frac"),
                               "v_m_s": src_config.get("noise_v"), "w_rad_s": src_config.get("noise_w"),
                               "torque_nm": src_config.get("noise_torque"), "hold_s": src_config.get("noise_hold"),
                               "fade_base_m": src_config.get("noise_fade_base"), "fade_arm_m": src_config.get("noise_fade_arm")},
              "expert": "obstacle expert v2 (two base segments over the unchanged template controllers, nav_timeout 45 s, "
                        "step cap 60 000, contact ends the episode); the group's obstacle_task/expert.py",
              "source": {"set_dir": str(source), "set_config": src_config, "collector": "obstacle_task/collect.py (protocol v2.0)"},
              "converted_by": "scripts/obstacle/convert_step9_dataset.py", "converted_at": utc_now()}
    write_json(output / "config.json", config)
    write_json(output / "config.requested.json", config)
    schema = dataset_schema()
    schema["extra_fields"] = {"lidar": {"shape": ["N", N_BEAMS], "dtype": "float32",
                                        "meaning": "planar lidar ranges (m) in the base frame, beam 0 forward, counter-clockwise, "
                                                   "clipped at 4 m; recomputed from the logged base pose and the episode's obstacles"},
                              "expert_segment": {"shape": ["N"], "dtype": "int8", "meaning": "0 = to the waypoint, 1 = to the dock / reaching"}}
    write_json(output / "schema.json", schema)
    write_json(output / "scenarios.json", scenarios)
    done = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (eid, meta) in enumerate(ex.map(convert_episode, jobs, chunksize=4), start=1):
            done[eid] = meta
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} converted", flush=True)
    successes = sum(m["success"] for m in done.values())
    manifest = {"format": "step9 obstacle collection converted to the project dataset layout", "created_at": utc_now(),
                "source_dir": str(source), "source_index_sha256": sha256(source / "index.csv"),
                "source_scenarios_sha256": sha256(source / "scenarios.json"), "episodes": len(done),
                "expert_successes": successes, "collisions": sum(m["collided"] for m in done.values()),
                "navigation_timeouts": sum(m["navigation_timeout"] for m in done.values()),
                "noisy_episodes": sum(m["noisy"] for m in done.values()), "lidar_check": note,
                "hashes": {name: sha256(output / name) for name in ("config.json", "scenarios.json", "schema.json")}}
    write_json(output / "dataset.json", manifest)
    # Provenance: this project's and RoboEnv's source fingerprint (as the collectors record) plus the group repository's
    # obstacle task sources that produced the recordings (commit and file hashes), hashed by the trainers' run records.
    task_dir = source.parent.parent.parent if (source.parent.parent.parent / "obstacle_task").exists() else None
    step9 = {"task_dir": str(task_dir) if task_dir else None}
    if task_dir:
        step9["git"] = git_info(task_dir)
        step9["files"] = {str(p.relative_to(task_dir)): sha256(p) for p in sorted((task_dir / "obstacle_task").glob("*.py"))}
        for name in ("collect_data.py", "evaluate_policy.py", "robotnik_arm_control.py", "robotnik_arm_task.py", "bc_features.py"):
            if (task_dir / name).exists():
                step9["files"][name] = sha256(task_dir / name)
    write_json(output / "provenance.json", {**provenance(Path(PROJECT_ROOT), Path(get_roboenv_root())), "step9_source": step9})
    records = inventory(output)
    partition = split_episodes(records, args.validation_fraction, args.split_seed)
    split = {"dataset": str(output), "extra_datasets": [], "seed": args.split_seed,
             "unit": "whole episode; uniform random holdout by episode (step-9 scenarios are all distinct)",
             "validation_fraction": args.validation_fraction, "inventory_sha256": inventory_fingerprint(records),
             "expert_failures_in_split": sum(not r["success"] for r in records),
             "note": "expert failures (contact, navigation timeout, not reached) are kept in the split; trainers drop them with keep_expert_failures=false",
             **partition}
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.split_out, split)
    print(f"dataset {output}: {len(done)} episodes, {successes} expert successes; split {args.split_out}: "
          f"{len(partition['training'])} training / {len(partition['validation'])} validation episodes")


if __name__ == "__main__":
    main()
