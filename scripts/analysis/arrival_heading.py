"""At the arrival instant (closest end-effector approach), is the residual wheel command a drive or a rotation, and how
large is the base's heading error to the target? Common mode (right + left)/2 is a drive, differential (right - left)/2
is a rotation in place; the heading error is the bearing from the base to the target minus the base bearing, the quantity
the expert's 'orient' phase zeroes (tolerance 0.05 rad).

Usage: python3 -I scripts/analysis/arrival_heading.py EVAL_DIR [EVAL_DIR ...]
"""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

for d in sys.argv[1:]:
    d = Path(d); rows = defaultdict(list)
    for p in sorted(d.glob("traces/*.npz")):
        pol = p.stem.rsplit("_episode_", 1)[0]
        with np.load(p, allow_pickle=False) as z:
            t = z["time"]; obs = z["observation"]; raw = z["raw_action"]; dist = z["distance"]
        i0 = int(np.argmin(dist)); lo, hi = max(0, i0 - 5), min(len(t), i0 + 5); w = raw[lo:hi, :2]
        common = np.abs((w[:, 0] + w[:, 1]) / 2).mean(); diff = np.abs((w[:, 0] - w[:, 1]) / 2).mean()
        bearing_to_target = np.arctan2(obs[i0, 19] - obs[i0, 1], obs[i0, 18] - obs[i0, 0])
        err = np.abs((bearing_to_target - obs[i0, 3] + np.pi) % (2 * np.pi) - np.pi)
        rows[pol].append((common, diff, err))
    print(f"== {d.name}")
    for pol, r in sorted(rows.items()):
        r = np.array(r); m = np.median(r, axis=0)
        print(f"  {pol:28s} n={len(r):3d}  |wheel| around arrival: drive component {m[0]:.2f}, rotation component {m[1]:.2f};  "
              f"heading error to target at arrival: median {np.degrees(m[2]):.1f} deg, above 2.9 deg (expert tolerance) in {int((r[:, 2] > 0.05).sum()):3d}")
