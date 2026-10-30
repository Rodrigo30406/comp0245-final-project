"""Where and how a policy's base ends up: parked in tolerance, parked elsewhere, or still moving at the horizon.

Usage: python scripts/analysis/stop_profile.py EVAL_DIR [EVAL_DIR ...]
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

for d in sys.argv[1:]:
    rows = defaultdict(list)
    for p in sorted(Path(d, "episodes").glob("*.json")):
        e = json.loads(p.read_text())
        if e.get("status") != "complete" or "subsystems" not in e:
            continue
        b = e["subsystems"]["base_navigation"]; t = e["subsystems"]["terminal_motion"]
        rows[e["policy"]].append((b["base_pose_within_tolerance"], t["base_mean_speed_xy_m_s"] < 0.02, b["base_goal_distance_xy_m"],
                                  b["minimum_goal_distance_xy_m"], b["arrival_completion_s"] or np.nan))
    print(f"== {Path(d).name}")
    for pol, r in rows.items():
        r = np.array(r, dtype=float); intol, still, dist, mind, arr = r.T
        stopped_out = (still == 1) & (intol == 0)
        print(f"  {pol:28s} n={len(r):3d} in tolerance at end={int(intol.sum()):3d}  stopped outside={int(stopped_out.sum()):3d} "
              f"(median goal dist {np.median(dist[stopped_out]) if stopped_out.any() else float('nan'):.2f} m, median closest {np.median(mind[stopped_out]) if stopped_out.any() else float('nan'):.2f} m)  "
              f"moving at 40 s={int((still == 0).sum()):3d}  median arrival {np.nanmedian(arr):.1f} s")
