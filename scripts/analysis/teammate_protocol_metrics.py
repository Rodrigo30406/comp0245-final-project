"""Our evaluation runs read with the teammate's protocol v1.1 definitions (final/robotnik_arm_task/EVALUATION.md on
branch feature/step3c-trigger-copies), so both pipelines can be compared on the same columns:

  success    EE within 10 mm of the target at some time <= 40 s          (from metrics.minimum_distance_m, 1 ms)
  reach_2cm, reach_5cm   same with 20 / 50 mm
  t_reach    first trace sample (100 ms) with EE error <= 10 mm
  settled    EE error < 15 mm at every trace sample of the 1 s after t_reach
  held_10s   EE error < 15 mm at every trace sample of the 10 s after t_reach (t_reach + 10 s <= 40 s)
  arm_extended   EE displacement in the base frame from its position at t = 0 exceeds 0.10 m
  parked     ring distance (|planar base-target distance - standoff(z)|) < 0.20 m and base still (planar speed
             < 0.01 m/s, |yaw rate| < 0.02 rad/s, finite differences at 100 ms) for >= 1.0 s
Traces are sampled every 100 ms, so t_reach, settled, held_10s and parked are approximations of their 1 ms values.

  python3 -I scripts/analysis/teammate_protocol_metrics.py EVAL_DIR [EVAL_DIR ...]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STANDOFF = json.load(open(ROOT / "experiment_specs/expert/default.json"))["standoff"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def episode_metrics(meta, z):
    t = z["time"]; dist = z["distance"]; obs = z["observation"]
    mn = float(meta["metrics"]["minimum_distance_m"])
    out = {"success": mn <= 0.010, "reach_2cm": mn <= 0.020, "reach_5cm": mn <= 0.050}
    hit = np.flatnonzero(dist <= 0.010)
    out["t_reach"] = float(t[hit[0]]) if len(hit) else np.nan
    out["settled"] = out["held_10s"] = False
    if len(hit):
        k = hit[0]
        w1 = (t >= t[k]) & (t <= t[k] + 1.0 + 1e-9); out["settled"] = bool(np.all(dist[w1] < 0.015))
        if t[k] + 10.0 <= t[-1] + 1e-9:
            w10 = (t >= t[k]) & (t <= t[k] + 10.0 + 1e-9); out["held_10s"] = bool(np.all(dist[w10] < 0.015))
    eeb = z["ee_position_base"]; out["arm_extended"] = bool(np.linalg.norm(eeb - eeb[0], axis=1).max() > 0.10)
    target = np.asarray(meta["scenario"]["target"], dtype=float)
    standoff = float(np.interp(target[2], STANDOFF["heights"], STANDOFF["distances"]))
    ring = np.abs(np.hypot(*(obs[:, :2] - target[:2]).T) - standoff)
    dt = np.diff(t); speed = np.linalg.norm(np.diff(obs[:, :2], axis=0), axis=1) / dt; yaw_rate = np.abs(wrap(np.diff(obs[:, 3]))) / dt
    still = (speed < 0.01) & (yaw_rate < 0.02) & (ring[1:] < 0.20)
    run = 0; parked = False
    for s in still:
        run = run + 1 if s else 0
        if run * np.median(dt) >= 1.0 - 1e-9:
            parked = True; break
    out["parked"] = parked
    return out


def main(dirs):
    cols = ["success", "settled", "held_10s", "reach_2cm", "reach_5cm", "arm_extended", "parked"]
    for d in dirs:
        d = ROOT / d if not Path(d).is_absolute() else Path(d)
        per = defaultdict(list)
        for f in sorted((d / "episodes").glob("*.json")):
            label, ep = f.stem.rsplit("_episode_", 1)
            z = np.load(d / "traces" / f.name.replace(".json", ".npz"))
            per[label].append(episode_metrics(json.load(open(f)), z))
        print(f"== {d.name}")
        print(f"{'policy':28s} n   " + " ".join(f"{c:>12s}" for c in cols) + "   median t_reach")
        for label, rows in per.items():
            n = len(rows); tr = [r["t_reach"] for r in rows if np.isfinite(r["t_reach"])]
            print(f"{label:28s} {n:3d} " + " ".join(f"{sum(r[c] for r in rows):5d} {100 * sum(r[c] for r in rows) / n:5.1f}%" for c in cols)
                  + f"   {np.median(tr) if tr else float('nan'):6.1f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
