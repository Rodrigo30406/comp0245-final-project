"""Seed statistics of closed-loop evaluations: mean +- sd and min-max of stable arrivals across seeds.

Usage: python3 scripts/evaluation/seed_stats.py runs/evaluation/<dir> [runs/evaluation/<dir> ...]
Models are grouped by their label with the trailing `_seedN` removed, across every directory given, so seeds evaluated
in separate runs (e.g. seeds 0-2 and seeds 3-4) are pooled. Papers report this (fixed seeds chosen in advance, every
seed listed) rather than the best seed; the report follows that convention.
Not part of the provenance fingerprint (only top-level scripts/*.py are hashed), so it may change while runs are active.
"""
import json
import re
import sys
from collections import defaultdict

import numpy as np

groups = defaultdict(dict)
for d in sys.argv[1:]:
    s = json.load(open(f'{d}/summary.json'))
    for label, v in s.items():
        if label == 'expert':
            continue
        m = re.match(r'(.*?)(?:_seed(\d+))?$', label)
        base, seed = m.group(1), m.group(2) or '0'
        if seed in groups[base]:
            print(f'warning: {base} seed {seed} appears twice, keeping the first', file=sys.stderr)
            continue
        t = v['success_by_tolerance']
        groups[base][seed] = (t['0.01'], t['0.02'], t['0.05'], v['mean_final_distance_valid_only_m'])

print(f"{'policy':34s} n  1 cm (mean+-sd, min-max)   2 cm                     5 cm                     final m")
for base, seeds in sorted(groups.items()):
    a = np.array([seeds[k][:3] for k in sorted(seeds, key=int)], dtype=float)
    fin = np.array([seeds[k][3] for k in sorted(seeds, key=int)])
    cells = [f"{a[:, i].mean():5.1f}+-{a[:, i].std(ddof=1) if len(a) > 1 else 0:4.1f} ({int(a[:, i].min()):3d}-{int(a[:, i].max()):3d})" for i in range(3)]
    print(f"{base:34s} {len(a)}  {'  '.join(cells)}  {fin.mean():.3f}")
    print(f"{'':34s}    seeds {' '.join(sorted(seeds, key=int))}: " + ' | '.join('/'.join(str(int(x)) for x in row) for row in a))
