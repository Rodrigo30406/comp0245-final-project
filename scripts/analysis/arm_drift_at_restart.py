"""Closed loop: how far has the arm left its hold posture when the base, once parked, starts moving again?

From the evaluation traces: per episode, the rms joint deviation from the initial (hold) posture over time, and, for the
episodes where the base was still (speed < 0.02 m/s for >= 0.5 s) and then restarted (speed > 0.05 m/s), the arm deviation
at the restart. Compare with scripts/analysis/stop_arm_sensitivity.py, which maps how much wheel command each rms deviation
provokes at the stop pose.

Usage: python scripts/analysis/arm_drift_at_restart.py EVAL_DIR [EVAL_DIR ...]
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

MARKS = [5, 10, 20, 39]

for d in sys.argv[1:]:
    d = Path(d); out = defaultdict(lambda: defaultdict(list))
    for p in sorted(d.glob("traces/*.npz")):
        pol = p.stem.rsplit("_episode_", 1)[0]
        with np.load(p, allow_pickle=False) as z:
            t = z["time"]; obs = z["observation"]
        q = obs[:, 4:11]; dev = np.sqrt(((q - q[0]) ** 2).mean(axis=1))
        for m in MARKS:
            k = np.searchsorted(t, m)
            if k < len(t):
                out[pol][f"dev@{m}s"].append(dev[k])
        speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T)
        still = speed < 0.02; restart = None; run = 0
        for i in range(1, len(t)):
            run = run + (t[i] - t[i - 1]) if still[i] else 0.0
            if run >= 0.5:
                later = np.flatnonzero(speed[i:] > 0.05)
                if len(later):
                    restart = i + later[0]
                break
        out[pol]["parked"].append(1.0)
        if restart is not None:
            out[pol]["restarted"].append(1.0); out[pol]["dev@restart"].append(dev[restart]); out[pol]["t_restart"].append(t[restart])
            out[pol]["dev@park"].append(dev[i])
    print(f"== {d.name}")
    for pol, m in sorted(out.items()):
        n = len(m["parked"]); r = len(m.get("restarted", []))
        cells = "  ".join(f"{k} {np.median(v):.3f}" for k, v in m.items() if k.startswith("dev@") and k not in ("dev@restart", "dev@park"))
        extra = (f"  | parked-then-restarted {r}/{n}: arm rms deviation at park {np.median(m['dev@park']):.3f} rad, at restart {np.median(m['dev@restart']):.3f} rad, "
                 f"restart at {np.median(m['t_restart']):.1f} s") if r else f"  | parked-then-restarted 0/{n}"
        print(f"  {pol:28s} median arm rms deviation from hold (rad): {cells}{extra}")
