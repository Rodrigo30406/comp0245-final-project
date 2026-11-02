"""Where are the exact-zero wheel rows in a training split, as a function of the clock and of the episode's outcome?

A categorical wheel head can only learn a stop where the demonstrations put the zero bin. This tabulates, for the
training episodes of a split, the fraction of rows whose demonstrated wheel command is exactly zero in each clock window,
split by episode termination (success / stalled), and the magnitude of the non-zero wheel commands in the expert's
'orient' phase and in the last second before the stop.

Usage: python3 -I scripts/analysis/zero_by_clock.py SPLIT_JSON [--max-episodes 600]
"""
import argparse
import json
from pathlib import Path

import numpy as np

WINDOWS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("--max-episodes", type=int, default=600)
    a = ap.parse_args(); split = json.loads(Path(a.split).read_text())
    recs = split["training"]; rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(a.max_episodes, len(recs)), replace=False))]
    zero = {k: {w: [0, 0] for w in WINDOWS} for k in ("success", "stalled")}; orient, last = [], []
    for r in recs:
        d = Path(r.get("dataset", split["dataset"])); k = r["termination"]
        with np.load(d / "episodes" / f"{r['episode_id']}.npz", allow_pickle=False) as z:
            t = z["time"]; w = np.abs(z["action"][:, :2]).max(axis=1); phase = z["expert_phase"]
        for win in WINDOWS:
            m = (t >= win[0]) & (t < win[1]); zero[k][win][0] += int((w[m] == 0).sum()); zero[k][win][1] += int(m.sum())
        mag = np.abs(z["action"][:, :2]).mean(axis=1) if False else w
        orient += list(w[(phase == "orient") & (w > 0)])
        moving = np.flatnonzero(w > 0)
        if len(moving):
            stop = moving[-1]; m = (t > t[stop] - 1.0) & (t <= t[stop]); last += list(w[m][w[m] > 0])
    print(f"training episodes sampled: {len(recs)} ({sum(r['termination'] == 'stalled' for r in recs)} stalled)")
    for k in ("success", "stalled"):
        cells = "  ".join(f"{w[0]:2d}-{w[1]:2d}s {100 * z[0] / z[1] if z[1] else float('nan'):4.0f}% ({z[1] // 1000:4d}k rows)" for w, z in zero[k].items())
        print(f"  {k:8s} rows with wheels exactly 0: {cells}")
    q = lambda v: " ".join(f"p{p}={np.percentile(v, p):.2f}" for p in (10, 50, 90))
    print(f"  non-zero max|wheel| in 'orient' rows: {q(orient)};  in the last 1 s before the stop: {q(last)}")


if __name__ == "__main__":
    main()
