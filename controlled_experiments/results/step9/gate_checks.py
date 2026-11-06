"""Gates A3-A6 (plan section 7) computed from the outputs of the Track A smoke runs; writes smoke/gates.json.

The runs (from the task folder, 20 workers, each command < 10 min):
  python -m obstacle_task.collect --set-name pilot2  --out-dir results/step9/smoke/pilot2     # 100 clean
  python -m obstacle_task.collect --set-name pilot2n --out-dir results/step9/smoke/pilot2n    # 100 noisy
  python -m obstacle_task.evaluate --policy expert --eval-set results/step9/smoke/pilot2 --workers 20 \\
      --compare-recorded --save-traj --traj-every 10 --out-dir results/step9/smoke/expert_pilot2
  python -m obstacle_task.classify results/step9/smoke/expert_pilot2 --validate-expert
  python results/step9/gate_checks.py
Gates:
  A3  the harness's obs["lidar"] (saved trajectories, float64 poses) == lidar.scan_batch on the logged poses, <= 1e-12
  A4  harness expert on pilot2: success >= 95%, collisions 0, side_taken == scenario_side in every pass,
      docked_clean >= 97%
  A5  noisy expert on pilot2n (collector recordings): collisions <= 3%, success (status reached) >= 90%
  A6  harness expert == collector recordings (--compare-recorded, cmp_identical) on >= 20 pilot2 episodes
"""
import csv
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
from obstacle_task import lidar  # noqa: E402
from obstacle_task.scenario import load_scenarios  # noqa: E402

SMOKE = TASK / "results/step9/smoke"
PILOT, PILOTN, RUN = SMOKE / "pilot2", SMOKE / "pilot2n", SMOKE / "expert_pilot2"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def true(text):
    return str(text) == "True"


def gate_a3():
    scenarios = {sc.episode: sc for sc in load_scenarios(PILOT / "scenarios.json")[0]}
    worst, worst32, n_scans, files = 0.0, 0.0, 0, sorted(glob.glob(str(RUN / "traj" / "ep_*.npz")))
    with open(RUN / "config.json") as f:
        every = json.load(f)["args"]["traj_every"]
    for path in files:
        episode = int(os.path.basename(path)[3:7])
        obstacles = scenarios[episode].obstacles
        with np.load(path) as traj:
            offline = lidar.scan_batch(traj["base_pos"][:, :2], traj["base_bearing"], obstacles)
            worst = max(worst, float(np.max(np.abs(offline - traj["lidar"]))))
            n_scans += len(offline)
            with np.load(PILOT / f"ep_{episode:04d}.npz") as rec:     # the float32 recording of the same steps
                n = min(len(rec["t"][::every]), len(offline))
                rec32 = lidar.scan_batch(rec["base_pos"][::every][:n, :2], rec["base_bearing"][::every][:n],
                                         obstacles)
                worst32 = max(worst32, float(np.max(np.abs(rec32 - traj["lidar"][:n]))))
    return {"passed": bool(files) and worst <= 1e-12, "trajectories": len(files), "scans": n_scans,
            "traj_every": every, "max_abs_diff": worst,
            "info_float32_recorded_poses_max_abs_diff": worst32}


def gate_a4(rows):
    n = len(rows)
    success = sum(true(r["success"]) for r in rows)
    collisions = sum(true(r["collision"]) for r in rows)
    passes = [r for r in rows if true(r["passed_obstacle"])]
    wrong_side = [int(r["episode"]) for r in passes if float(r["side_taken"]) != float(r["scenario_side"])]
    docked_clean = sum(true(r["docked_clean"]) for r in rows)
    clearance = np.array([float(r["min_clearance"]) for r in rows])
    out = {"episodes": n, "success": success, "success_rate": success / n, "collisions": collisions,
           "passes": len(passes), "wrong_side_passes": wrong_side, "docked_clean": docked_clean,
           "docked_clean_rate": docked_clean / n, "failures": [int(r["episode"]) for r in rows if not true(r["success"])],
           "not_docked_clean": [int(r["episode"]) for r in rows if not true(r["docked_clean"])],
           "stop_reasons": _counts(r["stop_reason"] for r in rows),
           "held_10s": sum(true(r["held_10s"]) for r in rows),
           "min_clearance_m": {"min": float(clearance.min()), "p5": float(np.percentile(clearance, 5)),
                               "median": float(np.median(clearance))},
           "t_reach_s": _quantiles([float(r["t_reach"]) for r in rows if r["t_reach"] not in ("", "nan")]),
           "time_to_dock_s": _quantiles([float(r["time_to_dock"]) for r in rows if r["time_to_dock"] not in ("", "nan")])}
    out["passed"] = (success >= 0.95 * n and collisions == 0 and not wrong_side and docked_clean >= 0.97 * n)
    return out


def gate_a5():
    rows = read_csv(PILOTN / "index.csv")
    n = len(rows)
    reached = sum(r["status"] == "reached" for r in rows)
    collisions = sum(true(r["collided"]) for r in rows)
    within = 0                                 # EE < 10 mm at some recorded step (before any contact)
    for r in rows:
        with np.load(PILOTN / r["file"]) as d:
            within += bool(np.min(np.linalg.norm(d["ee_pos"].astype(float) - d["target"], axis=1)) < 0.010)
    out = {"episodes": n, "all_noisy": all(true(r["noisy"]) for r in rows), "reached": reached,
           "success_rate": reached / n, "collisions": collisions, "collision_rate": collisions / n,
           "ee_within_10mm_any_step": within, "statuses": _counts(r["status"] for r in rows),
           "not_reached": {int(r["episode"]): r["status"] for r in rows if r["status"] != "reached"},
           "nav_timed_out": sum(true(r["nav_timed_out"]) for r in rows),
           "duration_s": _quantiles([float(r["duration"]) for r in rows])}
    out["passed"] = collisions <= 0.03 * n and reached >= 0.90 * n
    return out


def gate_a6(rows):
    compared = [r for r in rows if r.get("cmp_identical") not in ("", None)]
    identical = sum(true(r["cmp_identical"]) for r in compared)
    out = {"compared": len(compared), "identical": identical,
           "steps_compared": int(sum(float(r["cmp_steps"]) for r in compared)),
           "max_abs_diff": {k: max(float(r[k]) for r in compared) for k in compared[0] if k.startswith("cmp_max_")}
           if compared else {}}
    out["passed"] = len(compared) >= 20 and identical == len(compared)
    return out


def collector_summary(path):
    rows = read_csv(path / "index.csv")
    return {"episodes": len(rows), "statuses": _counts(r["status"] for r in rows),
            "collisions": sum(true(r["collided"]) for r in rows),
            "ambiguous": sum(true(r["ambiguous"]) for r in rows),
            "left": sum(float(r["side"]) == 1 for r in rows),
            "segment_switch_time_s": _quantiles([float(r["segment_switch_time"]) for r in rows
                                                 if r["segment_switch_time"] not in ("", "nan")]),
            "nav_done_time_s": _quantiles([float(r["nav_done_time"]) for r in rows if r["nav_done_time"] != ""])}


def _counts(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _quantiles(values):
    if not values:
        return {}
    v = np.asarray(values)
    return {"min": float(v.min()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def main():
    rows = read_csv(RUN / "rollouts.csv")
    report = {"A3": gate_a3(), "A4": gate_a4(rows), "A5": gate_a5(), "A6": gate_a6(rows),
              "collector_pilot2": collector_summary(PILOT), "collector_pilot2n": collector_summary(PILOTN)}
    classes = RUN / "classes.csv"
    if classes.exists():
        report["classes_expert_pilot2"] = _counts(r["fclass"] for r in read_csv(classes))
    report["passed"] = {g: report[g]["passed"] for g in ("A3", "A4", "A5", "A6")}
    (SMOKE / "gates.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(report["passed"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
