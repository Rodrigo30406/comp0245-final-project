"""What makes a parked base leave again? Per episode with a park -> restart event (as in creep_profile.py):
how many park/restart cycles there are, what the wheel command looked like while parked (fraction of samples exactly
zero, size of the first non-zero command), whether the arm was moving in the second before the restart, and where the
base is at the end (distance to the expert's base goal, still or moving).

Usage: python scripts/analysis/restart_trigger.py EVAL_DIR [EVAL_DIR ...]
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
            t = z["time"]; obs = z["observation"]; act = z["action"]; dist_ee = z["distance"]
        speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T); still = speed < 0.02
        wheel = np.abs(act[:, :4]).mean(axis=1)
        # park / restart cycles
        cycles = []; run = 0.0; park = None
        for i in range(1, len(t)):
            run = run + (t[i] - t[i - 1]) if still[i] else 0.0
            if park is None and run >= 0.5:
                park = i
            elif park is not None and speed[i] > 0.05:
                cycles.append((park, i)); park = None; run = 0.0
        if not cycles:
            continue
        park, restart = cycles[0]
        w = wheel[park:restart]; zero_frac = (w == 0).mean()
        first_nz = w[w > 0][0] if (w > 0).any() else 0.0
        # arm motion: joint displacement in the 1 s before the restart vs the 1 s after parking
        k = max(1, int(round(1.0 / np.median(np.diff(t)))))
        arm_before_restart = np.abs(obs[restart, 4:11] - obs[max(park, restart - k), 4:11]).max()
        arm_after_park = np.abs(obs[min(restart, park + k), 4:11] - obs[park, 4:11]).max()
        ee_at_restart = dist_ee[restart]
        rows[pol].append((len(cycles), zero_frac, first_nz, arm_before_restart, arm_after_park, ee_at_restart,
                          still[-1], (speed[restart:] > 0.05).mean()))
    print(f"== {d.name}")
    for pol, r in sorted(rows.items()):
        r = np.array(r, dtype=float); m = np.median(r, axis=0)
        print(f"  {pol:28s} n={len(r):3d}  cycles {m[0]:.0f} (max {r[:, 0].max():.0f});  while parked: samples with wheel exactly 0 = {100 * m[1]:3.0f}%, "
              f"first non-zero |wheel| {m[2]:.2f};  arm moved in the 1 s before restart {m[3]:.3f} rad (1 s after park {m[4]:.3f});  "
              f"EE at restart {100 * m[5]:.0f} cm;  still at end {100 * r[:, 6].mean():3.0f}%;  time moving after first restart {100 * m[7]:3.0f}%")
