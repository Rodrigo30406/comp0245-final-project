"""Print one line per policy of a closed-loop evaluation directory (stable arrivals, final/min distances, clipping).

Usage: python3 scripts/evaluation/summarize_eval.py runs/evaluation/<dir>
Not part of the provenance fingerprint (only top-level scripts/*.py are hashed), so it may change while runs are active.
"""
import glob
import json
import sys

import numpy as np

d = sys.argv[1]
s = json.load(open(f'{d}/summary.json'))
for p, v in s.items():
    fin, mn = [], []
    for f in glob.glob(f'{d}/episodes/{p}_*.json'):
        m = json.load(open(f))['metrics']
        if m and m.get('final_distance_m') is not None:
            fin.append(m['final_distance_m']); mn.append(m['minimum_distance_m'])
    fin = np.array(fin); mn = np.array(mn)
    t = v['success_by_tolerance']
    print(f"{p:28s} stable arrival 1/2/5cm = {t['0.01']}/{t['0.02']}/{t['0.05']}  valid {v['valid_full_horizon']}/{v['attempted']}  "
          f"median final {np.median(fin):.3f} m  min<5cm {int((mn < 0.05).sum())}  "
          f"clip wheels {v['mean_final_clipping_fraction_per_command'][0]:.2f} torque {np.mean(v['mean_final_clipping_fraction_per_command'][4:]):.2f}")
