"""Where a policy's wheels go to zero for good, relative to the expert's stop box.

For every episode of an evaluation directory: whether the base ever entered the expert's stop box (goal distance <
position tolerance), the smallest heading error while inside it (the expert's own stop condition is distance <
0.08 m and |heading error| < 0.05 rad), when the wheel commands fell to zero for the rest of the episode (the
closed-gate / latched-stop proxy), the goal distance at that moment, and whether the arm extended afterwards.
Trace-based (traces every 100 ms) so it runs on any evaluation; policy traces carry no expert phase.

Usage: python3 -I scripts/analysis/gate_closure_probe.py EVAL_DIR [EVAL_DIR ...] [--examples N]
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ZERO = 0.02  # rad/s: below this a wheel command counts as zero (a closed gate leaves ~0.2 % of the head's command)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def closure_index(wheels, has_action):
    """First trace index from which every later wheel command is below ZERO; None if the wheels never settle."""
    moving = (np.abs(wheels).max(axis=1) >= ZERO) & has_action
    if not moving.any():
        return 0
    last = int(np.nonzero(moving)[0][-1])
    return None if last >= len(moving) - 2 else last + 1


def analyse(directory):
    directory = Path(directory); rows = defaultdict(list)
    for p in sorted((directory / "episodes").glob("*.json")):
        e = json.loads(p.read_text())
        if e.get("status") != "complete" or e["policy"] == "expert":
            continue
        b = e["subsystems"]["base_navigation"]
        z = np.load(directory / "traces" / f"{e['job_id']}.npz")
        obs = z["observation"]; t = z["time"]; goal = np.array(b["goal_xy_m"])
        dist = np.linalg.norm(obs[:, :2] - goal, axis=1)
        herr = np.abs(wrap(obs[:, 3] - b["goal_bearing_rad"]))
        inside = dist < b["position_tolerance_m"]
        ci = closure_index(z["action"][:, :4], z["has_action"])
        ee = z["ee_position_base"]
        row = dict(entered=bool(inside.any()), t_enter=float(t[np.argmax(inside)]) if inside.any() else np.nan,
                   min_herr_inside=float(herr[inside].min()) if inside.any() else np.nan,
                   stop_condition_met=bool((inside & (herr < b["bearing_tolerance_rad"])).any()),
                   closed=ci is not None, t_close=float(t[ci]) if ci is not None else np.nan,
                   dist_close=float(dist[ci]) if ci is not None else np.nan,
                   herr_close=float(herr[ci]) if ci is not None else np.nan,
                   arm_after_close=float(np.linalg.norm(ee[-1] - ee[ci])) if ci is not None else np.nan,
                   base_after_close=float(np.linalg.norm(obs[-1, :2] - obs[ci, :2])) if ci is not None else np.nan,
                   min_dist=float(dist.min()), final_dist=float(dist[-1]), episode=e["job_id"])
        rows[e["policy"]].append(row)
    return rows


def med(v):
    v = [x for x in v if x == x]
    return f"{np.median(v):.2f}" if v else "  nan"


def main(argv):
    examples = 0
    if "--examples" in argv:
        i = argv.index("--examples"); examples = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    for d in argv:
        print(f"== {Path(d).name}")
        for pol, rows in analyse(d).items():
            n = len(rows); ent = [r for r in rows if r["entered"]]; met = [r for r in rows if r["stop_condition_met"]]
            closed = [r for r in rows if r["closed"]]
            closed_in = [r for r in closed if r["dist_close"] < 0.08]; closed_out = [r for r in closed if r["dist_close"] >= 0.08]
            ent_closed = [r for r in ent if r["closed"]]; met_closed = [r for r in met if r["closed"]]
            early = [r for r in ent_closed if r["t_close"] < r["t_enter"]]
            print(f"  {pol:26s} n={n:3d}  entered box={len(ent):3d}  heading<0.05 inside={len(met):3d}  "
                  f"wheels zero to end={len(closed):3d} (inside box {len(closed_in):3d}, outside {len(closed_out):3d}, median dist {med([r['dist_close'] for r in closed_out])} m)")
            print(f"  {'':26s} entered & zeroed={len(ent_closed):3d} (zeroed before entering {len(early):3d})  stop-condition met & zeroed={len(met_closed):3d}  "
                  f"min heading err inside (median) {med([r['min_herr_inside'] for r in ent])} rad  "
                  f"arm moved >10 cm after zero={sum(r['arm_after_close'] > 0.10 for r in closed):3d}  base drift after zero (median) {med([r['base_after_close'] for r in closed])} m")
            if examples:
                for r in sorted(ent, key=lambda r: r["t_enter"])[:examples]:
                    zero = f"{r['t_close']:.1f} s at {r['dist_close']:.2f} m" if r["closed"] else "never"
                    print(f"    {r['episode']}: enter {r['t_enter']:.1f} s, min heading err inside {r['min_herr_inside']:.3f} rad, "
                          f"stop cond {'met' if r['stop_condition_met'] else 'NOT met'}, wheels zero from {zero}, "
                          f"min dist {r['min_dist']:.2f} m, final {r['final_dist']:.2f} m")


if __name__ == "__main__":
    main(sys.argv[1:])
