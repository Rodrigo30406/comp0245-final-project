"""What happens at the arrival? Per policy, over the evaluation traces: how many bases came within 10 cm of the expert's
base goal, the mean |wheel command| in the 1 s around the instant of closest end-effector approach, the base speed at that
instant and 2 s / 5 s later, and the lowest 1 s mean |wheel| anywhere in the episode (does the policy ever command a stop?).

Usage: python3 -I scripts/analysis/arrival_profile.py EVAL_DIR [EVAL_DIR ...]
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

for d in sys.argv[1:]:
    d = Path(d); rows = defaultdict(list)
    for p in sorted(d.glob("traces/*.npz")):
        pol = p.stem.rsplit("_episode_", 1)[0]
        with np.load(p, allow_pickle=False) as z:
            t = z["time"]; obs = z["observation"]; raw = z["raw_action"]; dist = z["distance"]
        e = json.loads(Path(str(p).replace("/traces/", "/episodes/")).with_suffix(".json").read_text())
        speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T); i0 = int(np.argmin(dist)); lo, hi = max(0, i0 - 5), min(len(t), i0 + 5)
        w = np.abs(raw[:, :2]).mean(axis=1); k = max(1, int(round(1.0 / np.median(np.diff(t)))))
        rows[pol].append((e["subsystems"]["base_navigation"]["minimum_goal_distance_xy_m"], w[lo:hi].mean(), speed[i0],
                          speed[min(len(t) - 1, i0 + 2 * k)], speed[min(len(t) - 1, i0 + 5 * k)], np.convolve(w, np.ones(k) / k, mode="valid").min()))
    print(f"== {d.name}")
    for pol, r in sorted(rows.items()):
        r = np.array(r); m = np.median(r, axis=0)
        print(f"  {pol:28s} n={len(r):3d}  within 10 cm of the base goal {int((r[:, 0] < 0.1).sum()):3d};  |wheel| around arrival {m[1]:.2f};  "
              f"speed at arrival {m[2]:.2f} m/s, +2 s {m[3]:.2f}, +5 s {m[4]:.2f};  lowest 1 s mean |wheel| in the episode: median {m[5]:.2f}, below 0.1 in {int((r[:, 5] < 0.1).sum()):3d}")
