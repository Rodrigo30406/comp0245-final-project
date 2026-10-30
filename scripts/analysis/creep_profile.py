"""Between parking and restarting, does the base creep, and in which direction?

From evaluation traces, for the episodes where the base was still (speed < 0.02 m/s for >= 0.5 s) and then exceeded
0.05 m/s: time parked, displacement park -> restart, change of planar distance to the target (negative = crept towards it),
heading change, and the mean |wheel| command while parked as a fraction of the policy's own driving level.

Usage: python scripts/analysis/creep_profile.py EVAL_DIR [EVAL_DIR ...]
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
            t = z["time"]; obs = z["observation"]; act = z["action"]
        speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T); still = speed < 0.02
        wheel = np.abs(act[:, :4]).mean(axis=1); drive = wheel[speed > 0.1].mean() if (speed > 0.1).any() else np.nan
        run = 0.0; park = restart = None
        for i in range(1, len(t)):
            run = run + (t[i] - t[i - 1]) if still[i] else 0.0
            if run >= 0.5:
                park = i; later = np.flatnonzero(speed[i:] > 0.05); restart = i + later[0] if len(later) else None
                break
        if park is None or restart is None:
            continue
        tgt = obs[park, 18:20]; dist = lambda k: np.hypot(*(tgt - obs[k, :2]))
        rows[pol].append((t[restart] - t[park], np.hypot(*(obs[restart, :2] - obs[park, :2])), dist(restart) - dist(park),
                          np.degrees(np.abs((obs[restart, 3] - obs[park, 3] + np.pi) % (2 * np.pi) - np.pi)),
                          wheel[park:restart].mean() / drive, dist(park)))
    print(f"== {d.name}")
    for pol, r in sorted(rows.items()):
        r = np.array(r); m = np.median(r, axis=0)
        print(f"  {pol:28s} n={len(r):3d}  parked {m[0]:5.1f} s before restart; crept {100 * m[1]:4.1f} cm, distance to target {100 * m[2]:+5.1f} cm "
              f"(towards it in {100 * (r[:, 2] < 0).mean():3.0f}%), heading {m[3]:4.1f} deg; |wheel| while parked {100 * m[4]:3.0f}% of driving; target distance at park {m[5]:.2f} m")
