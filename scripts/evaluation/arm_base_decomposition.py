"""Decompose a closed-loop evaluation into the two halves of the task (docs/05_arm_ignition.md, docs/06_base_drift.md).

Per policy: base parked (stable pose arrival at the expert's goal), arm active (the arm leaves its home posture: the
largest joint excursion max_t max_j |q_j(t) - q_j(0)| exceeds 0.1 rad), both, drift of the base after parking,
held >= 10 s, and stable arrivals at 5 cm.

The arm rule was changed on 4 Nov 2026 (experiment records). Until then it was "end-effector displacement in
the base frame > 10 cm between the first and the last sample", which under-counted the expert itself (151 of 205: its
reach from the standoff moves the end effector as little as 2.8 cm) while a sleeping arm jostled by the base moves 3 cm.
In joint space the two populations do not overlap: every expert episode exceeds 0.195 rad, every sleeping arm stays
below 0.08 rad (the DART-trained hold wobbles 0.05 rad). Usage: python3 scripts/evaluation/arm_base_decomposition.py runs/evaluation/<dir> [...]
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
        q = t['observation'][:, 4:11]; arm = bool(np.abs(q - q[0]).max() > 0.10)
        b = e['subsystems']['base_navigation']; parked = bool(b['stable_pose_arrival'])
        tol = m['by_tolerance']['0.05']
        r = rows[pol]; r['n'] += 1; r['parked'] += parked; r['arm'] += arm; r['both'] += parked and arm
        r['held'] += tol['longest_sampled_dwell_s'] >= 10; r['ok'] += bool(tol.get('stable_arrival', tol.get('success')))
        if parked:
            r['drift'].append(b['maximum_drift_after_stable_arrival_xy_m'])
    print(f'== {d}')
    for pol, r in rows.items():
        drift = np.median(r['drift']) if r['drift'] else float('nan')
        print(f"  {pol:28s} n={r['n']:3d}  base parked={r['parked']:3d}  arm active={r['arm']:3d}  both={r['both']:3d}  "
              f"drift after parking (median)={drift:.2f} m  held>=10s={r['held']:3d}  stable 5cm={r['ok']:3d}")
