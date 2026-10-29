"""Decompose a closed-loop evaluation into the two halves of the task (docs/BLACKBOX_PLAN_20261027.md §6c-6e).

Per policy: base parked (stable pose arrival at the expert's goal), arm extended (end-effector displacement in the
base frame > 10 cm between the first and the last trace sample), both, drift of the base after parking, held >= 10 s,
and stable arrivals at 5 cm. Usage: python3 scripts/analysis/arm_base_decomposition.py runs/evaluation/<dir> [...]
"""
import collections
import glob
import json
import sys

import numpy as np

for d in sys.argv[1:]:
    rows = collections.defaultdict(lambda: dict(n=0, parked=0, arm=0, both=0, held=0, ok=0, drift=[]))
    for f in sorted(glob.glob(f'{d}/episodes/*.json')):
        e = json.load(open(f)); pol = e['policy']; m = e['metrics']
        if pol == 'expert' or not m or m.get('final_distance_m') is None:
            continue
        t = np.load(f.replace('/episodes/', '/traces/').replace('.json', '.npz'), allow_pickle=True)
        eb = t['ee_position_base']; arm = bool(np.linalg.norm(eb[-1] - eb[0]) > 0.10)
        b = e['subsystems']['base_navigation']; parked = bool(b['stable_pose_arrival'])
        tol = m['by_tolerance']['0.05']
        r = rows[pol]; r['n'] += 1; r['parked'] += parked; r['arm'] += arm; r['both'] += parked and arm
        r['held'] += tol['longest_sampled_dwell_s'] >= 10; r['ok'] += bool(tol.get('stable_arrival', tol.get('success')))
        if parked:
            r['drift'].append(b['maximum_drift_after_stable_arrival_xy_m'])
    print(f'== {d}')
    for pol, r in rows.items():
        drift = np.median(r['drift']) if r['drift'] else float('nan')
        print(f"  {pol:28s} n={r['n']:3d}  base parked={r['parked']:3d}  arm extended={r['arm']:3d}  both={r['both']:3d}  "
              f"drift after parking (median)={drift:.2f} m  held>=10s={r['held']:3d}  stable 5cm={r['ok']:3d}")
