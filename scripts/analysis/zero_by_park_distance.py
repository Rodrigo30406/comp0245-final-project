"""What does the data say near the goal? For each dataset, successful episodes only: the goal is the expert's own base goal
(`base_goal_xy` in the episode metadata). Rows are binned by planar distance between the base and that park pose, and the table
gives the fraction of rows whose wheel command is exactly zero, the mean |wheel| of the non-zero rows, and the share of
rows in each expert phase, plus the heading error at the start of the episode.

Usage: python3 -I scripts/analysis/zero_by_park_distance.py DATASET_DIR [DATASET_DIR ...] [--episodes 300]
"""
import argparse
import json
from pathlib import Path

import numpy as np

BINS = [(0, 0.08), (0.08, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]
PHASES = ("align", "drive", "orient", "done")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--episodes", type=int, default=300); a = ap.parse_args()
    for d in a.datasets:
        d = Path(d); files = sorted((d / "episodes").glob("*.npz")); rng = np.random.default_rng(0)
        files = [files[i] for i in sorted(rng.choice(len(files), min(a.episodes, len(files)), replace=False))]
        zero = {b: [0, 0] for b in BINS}; mag = {b: [] for b in BINS}; ph = {b: {p: 0 for p in PHASES} for b in BINS}; n_ok = 0; rows_total = 0
        for f in files:
            meta = json.loads(f.with_suffix(".json").read_text())
            if meta.get("termination") != "success":
                continue
            with np.load(f, allow_pickle=False) as z:
                pos = z["base_position"][:, :2]; w = np.abs(z["action"][:, :2]).max(axis=1); phase = z["expert_phase"]
            n_ok += 1; rows_total += len(w); park = np.asarray(meta["base_goal_xy"]); dist = np.hypot(*(pos - park).T)
            for b in BINS:
                m = (dist >= b[0]) & (dist < b[1]); zero[b][0] += int((w[m] == 0).sum()); zero[b][1] += int(m.sum()); mag[b] += list(w[m][w[m] > 0][::50])
                for p in PHASES:
                    ph[b][p] += int((m & (phase == p)).sum())
        print(f"== {d.parent.name}: {n_ok} successful episodes sampled, {rows_total / 1e6:.1f}M rows")
        for b in BINS:
            tot = zero[b][1]
            if not tot:
                continue
            shares = " ".join(f"{p} {100 * ph[b][p] / tot:3.0f}%" for p in PHASES)
            print(f"   {b[0]:4.2f}–{b[1]:4.2f} m from the base goal: {100 * tot / rows_total:4.1f}% of rows, wheels exactly 0 in {100 * zero[b][0] / tot:3.0f}%, "
                  f"non-zero |wheel| median {np.median(mag[b]) if mag[b] else float('nan'):.2f};  phases: {shares}")


if __name__ == "__main__":
    main()
