"""Does a finite-difference base velocity turn the policy into a copycat of its own last command?

Measures, on validation episodes of a split, (1) how much of the expert's wheel commands a linear map recovers from the
three velocity features alone versus from every other feature, and (2) the policy's counterfactual answers: driving rows
with the velocity zeroed ("the base is at rest") and parked rows with the median driving velocity injected.

Usage: python scripts/analysis/velocity_copycat.py RUN_DIR [RUN_DIR ...] [--episodes 60]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import inventory, extra_inventory  # noqa: E402
from robot_learning.learning.egocentric import EgocentricPolicy, VARIANTS, load_group  # noqa: E402


def r2(x, y):
    x1 = np.c_[x, np.ones(len(x))]
    coef, *_ = np.linalg.lstsq(x1, y, rcond=None)
    res = y - x1 @ coef
    return 1 - res.var(axis=0) / y.var(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=60)
    a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = EgocentricPolicy.load(run / "model.npz")
        names = list(VARIANTS[policy.spec["variant"]])
        vel = [i for i, n in enumerate(names) if n.startswith("base_") and n.endswith("_fd")]
        if not vel:
            print(run.name, "has no velocity features"); continue
        split = read_json(run / "split.json")
        records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        x, y, _ = load_group(split["dataset"], records, policy.spec)
        wheels = y[:, :2]
        other = [i for i in range(len(names)) if i not in vel]
        print(f"\n== {run.name}: {len(records)} validation episodes, {len(x)} rows")
        print(f"   linear R^2 of expert wheel commands from velocity only (3 feats): {r2(x[:, vel], wheels).round(3)}")
        print(f"   linear R^2 from everything else ({len(other)} feats):            {r2(x[:, other], wheels).round(3)}")
        pred = policy.predict_features(x)[:, :2]
        drive = np.abs(wheels).max(axis=1) > 0.5 * np.abs(wheels).max()
        parked = (np.abs(wheels).max(axis=1) < 0.02) & (np.hypot(x[:, vel[0]], x[:, vel[1]]) < 0.01)
        x_rest = x.copy(); x_rest[:, vel] = 0.0
        pred_rest = policy.predict_features(x_rest)[:, :2]
        median_drive_v = np.median(x[drive][:, vel], axis=0)
        x_move = x.copy(); x_move[:, vel] = median_drive_v
        pred_move = policy.predict_features(x_move)[:, :2]
        mag = lambda z: np.abs(z).mean()
        print(f"   driving rows ({drive.sum()}): expert |wheel| {mag(wheels[drive]):.3f}, policy {mag(pred[drive]):.3f}, "
              f"policy with velocity zeroed {mag(pred_rest[drive]):.3f}  -> keeps {100 * mag(pred_rest[drive]) / mag(pred[drive]):.0f}% of its command")
        print(f"   parked rows ({parked.sum()}): expert |wheel| {mag(wheels[parked]):.3f}, policy {mag(pred[parked]):.3f}, "
              f"policy with median driving velocity {np.round(median_drive_v, 3)} injected {mag(pred_move[parked]):.3f}")
        print(f"   mean |d wheel / d feature| (standardised), top 6 of {len(names)}:")
        base = policy.predict_features(x)[:, :2]
        sens = []
        for i, n in enumerate(names):
            xp = x.copy(); xp[:, i] += policy.x_scale[i]
            sens.append((np.abs(policy.predict_features(xp)[:, :2] - base).mean(), n))
        for s, n in sorted(sens, reverse=True)[:6]:
            print(f"      {n:32s} {s:.4f}")


if __name__ == "__main__":
    main()
