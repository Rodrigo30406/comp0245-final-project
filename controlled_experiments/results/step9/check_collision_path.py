"""End-to-end check of the contact path (plan section 3): collector and harness both end at the first contact, at the
same time, on the same link, with identical trajectories up to it.

The expert never touches the obstacle on pilot2, so this script moves the obstacle of pilot2 scenarios 0-3 onto its
path (the scenarios' sampler constraints no longer hold; this is a check, not a set):
  ep 0  centre on the start->waypoint segment (head-on: the arm, which leads by 0.73 m, hits it)
  ep 1  beside that segment, 0.25 m + r from it (the chassis / a wheel scrapes it)
  ep 2  unchanged (control: no contact)
  ep 3  0.55 m + r from the waypoint at the mid heading of the waypoint turn (the arm hits it while turning in place)
It records them with obstacle_task.collect.run_episode (one fresh process each) into
results/step9/smoke/collision_check/set, then runs the harness CLI with the expert and --compare-recorded, and compares
status / collision_time / collision_link. Writes results/step9/smoke/collision_check/evidence.json (~1 min).

Run from the task folder: python results/step9/check_collision_path.py
"""
import csv
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
from obstacle_task import collect  # noqa: E402
from obstacle_task.scenario import load_scenarios, save_scenarios  # noqa: E402

OUT = TASK / "results/step9/smoke/collision_check"
SOURCE = TASK / "results/step9/smoke/pilot2"


def moved_scenarios():
    scs = load_scenarios(SOURCE / "scenarios.json")[0][:4]
    out = []
    for sc in scs:
        ob = dict(sc.obstacles[0])
        start, way, dock = (np.asarray(v, dtype=float) for v in (sc.init_xy, sc.waypoint_xy, sc.dock_xy))
        h0 = np.arctan2(*(way - start)[::-1])
        h1 = np.arctan2(*(dock - way)[::-1])
        normal = np.array([-np.sin(h0), np.cos(h0)])
        if sc.episode == 0:
            centre = (start + way) / 2
        elif sc.episode == 1:
            centre = (start + way) / 2 + (ob["r"] + 0.25) * normal
        elif sc.episode == 2:
            centre = np.array([ob["x"], ob["y"]])
        else:
            hm = h0 + np.angle(np.exp(1j * (h1 - h0))) / 2
            centre = way + (ob["r"] + 0.55) * np.array([np.cos(hm), np.sin(hm)])
        ob["x"], ob["y"] = float(centre[0]), float(centre[1])
        out.append(replace(sc, obstacles=[ob], meta={"collision_check": True}))
    return out


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    set_dir = OUT / "set"
    (set_dir / "logs").mkdir(parents=True)
    scenarios = moved_scenarios()
    save_scenarios(str(set_dir / "scenarios.json"), scenarios, extra={"set_name": "collision_check"})
    args = collect.parse_args(["--set-name", "collision_check", "--seed", str(scenarios[0].seed), "--n-episodes",
                               str(len(scenarios)), "--out-dir", str(set_dir), "--workers", "4"])
    with open(set_dir / "config.json", "w") as f:
        json.dump({**vars(args), "protocol": collect.PROTOCOL_VERSION, "format": "obstacle_collection_v1"}, f,
                  indent=2)
    _, jobs = collect.plan_jobs(args, scenarios)
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
        rows = sorted(pool.map(collect._worker, jobs), key=lambda row: row["episode"])
    collect.write_index(str(set_dir / "index.csv"), rows)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    run_dir = OUT / "expert"
    with open(OUT / "evaluate.log", "w") as log:
        subprocess.run([sys.executable, "-m", "obstacle_task.evaluate", "--policy", "expert", "--eval-set",
                        str(set_dir), "--workers", "4", "--compare-recorded", "--out-dir", str(run_dir)],
                       cwd=TASK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=500)
    with open(run_dir / "rollouts.csv", newline="") as f:
        harness = {int(r["episode"]): r for r in csv.DictReader(f)}
    evidence = []
    for row in rows:
        h = harness[row["episode"]]
        with np.load(set_dir / row["file"]) as d:
            n_rec = len(d["t"])
        evidence.append({
            "episode": row["episode"], "collector_status": row["status"], "harness_stop_reason": h["stop_reason"],
            "collector_collision_time": row["collision_time"], "harness_collision_time": h["collision_time"],
            "collector_link": row["collision_link"], "harness_link": h["collision_link"],
            "recorded_steps": n_rec, "cmp_steps": int(float(h["cmp_steps"])), "cmp_identical": h["cmp_identical"],
            "harness_min_clearance": float(h["min_clearance"]), "harness_success": h["success"]})
    ok = []
    for e in evidence:
        if e["collector_status"] == "collision":
            ok.append(e["harness_stop_reason"] == "collision" and e["cmp_identical"] == "True"
                      and abs(float(e["harness_collision_time"]) - e["collector_collision_time"]) < 1e-9
                      and e["harness_link"] == e["collector_link"] and e["cmp_steps"] == e["recorded_steps"])
        else:
            ok.append(e["harness_stop_reason"] != "collision" and e["cmp_identical"] == "True")
    report = {"check": "contact path, collector vs harness", "passed": all(ok) and sum(
        e["collector_status"] == "collision" for e in evidence) == 3, "episodes": evidence}
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
