"""With a wheel dead band at the evaluator output, what makes the base leave an exact stop?

For each episode whose filtered wheel command was exactly zero for >= 0.5 s: length of the longest exact-zero spell,
the raw (pre-band) max |wheel| at the start and at the end of that spell, its slope over the spell, how much the
clock input (observation column 20) and the arm joints (columns 4:11) changed over the spell, whether the base moved
during the spell (it should not: the wheels are commanded to zero), and the value of the raw command on the sample
that first crossed the band after the spell.

Usage: python scripts/analysis/deadband_raw_drift.py EVAL_DIR [EVAL_DIR ...]
"""
from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np

for d in sys.argv[1:]:
    d = Path(d); band = float(json.load(open(d / "protocol.json")).get("wheel_dead_band_rad_s", 0.0)); rows = defaultdict(list)
    for p in sorted(d.glob("traces/*.npz")):
        pol = p.stem.rsplit("_episode_", 1)[0]
        if pol == "expert":
            continue
        with np.load(p, allow_pickle=False) as z:
            t = z["time"]; obs = z["observation"]; act = z["action"]; raw = z["raw_action"]
        zero = np.abs(act[:, :4]).max(axis=1) == 0; rmax = np.abs(raw[:, :4]).max(axis=1)
        # longest exact-zero spell
        best = None; i = 0
        while i < len(t):
            if zero[i]:
                j = i
                while j + 1 < len(t) and zero[j + 1]:
                    j += 1
                if best is None or t[j] - t[i] > t[best[1]] - t[best[0]]:
                    best = (i, j)
                i = j + 1
            else:
                i += 1
        if best is None or t[best[1]] - t[best[0]] < 0.5:
            continue
        a, b = best; dur = t[b] - t[a]
        moved = np.hypot(*(obs[b, :2] - obs[a, :2]))
        slope = np.polyfit(t[a:b + 1], rmax[a:b + 1], 1)[0] if b > a else 0.0
        cross = rmax[b + 1] if b + 1 < len(t) else np.nan
        rows[pol].append((dur, rmax[a], rmax[b], slope, obs[b, 20] - obs[a, 20], np.abs(obs[b, 4:11] - obs[a, 4:11]).max(),
                          moved, cross, t[a], b + 1 >= len(t)))
    print(f"== {d.name} (band {band} rad/s)")
    for pol, r in sorted(rows.items()):
        r = np.array(r, dtype=float); m = np.median(r, axis=0)
        print(f"  {pol:28s} n={len(r):3d}  longest exact-zero spell {m[0]:5.1f} s (starting at {m[8]:4.1f} s), raw max|wheel| {m[1]:.2f} -> {m[2]:.2f} "
              f"(slope {m[3]:+.3f} rad/s per s); clock +{m[4]:.1f}, arm joints moved {m[5]:.3f} rad; base moved {100 * m[6]:.1f} cm during the spell; "
              f"first command over the band {np.nanmedian(r[:, 7]):.2f}; spell lasted to the end in {100 * r[:, 9].mean():.0f}%")
