"""Does the previous wheel command as an input turn the policy into a copycat of itself?

On validation episodes of the run's split, with teacher-forced features (the demonstrated previous command, as in
training): (1) linear R^2 of the expert's wheel commands from the two previous-command features alone versus from every
other feature; (2) counterfactuals: driving rows with the previous command zeroed and parked rows with the median driving
command injected; (3) the residual wheel command after the expert's stop, teacher-forced versus sequential on the policy's
own outputs (what closed loop does), and the fraction of sequential runs that ever reach a stop (|wheel| < 0.05 for 1 s).

Usage: python scripts/analysis/prevwheel_copycat.py RUN_DIR [RUN_DIR ...] [--episodes 40]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import load_episode, record_directory  # noqa: E402
from robot_learning.learning.egocentric import EgocentricPolicy, VARIANTS, load_episode_features  # noqa: E402


def r2(x, y):
    x1 = np.c_[x, np.ones(len(x))]
    coef, *_ = np.linalg.lstsq(x1, y, rcond=None)
    return 1 - (y - x1 @ coef).var(axis=0) / y.var(axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = EgocentricPolicy.load(run / "model.npz"); names = list(VARIANTS[policy.spec["variant"]])
        prev = [i for i, n in enumerate(names) if n.startswith("prev_wheel")]
        if not prev:
            print(run.name, "has no previous-command features"); continue
        split = read_json(run / "split.json"); records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        xs, ys, tf_res, seq_res, seq_stops, drive_levels = [], [], [], [], 0, []
        for rec in records:
            d = record_directory(rec, split["dataset"]); x, y = load_episode_features(d, rec, policy.spec); xs.append(x); ys.append(y)
            obs, _, _ = load_episode(d, rec)
            with np.load(Path(d) / "episodes" / f"{rec['episode_id']}.npz", allow_pickle=False) as data:
                t = data["time"].astype(np.float64)
            wheels = y[:, :2]; moving = np.abs(wheels).max(axis=1) > 0.02
            if not moving.any():
                continue
            stop = int(np.flatnonzero(moving)[-1] + 1); drive = np.abs(wheels[moving]).mean(); drive_levels.append(drive)
            if stop + 10 >= len(x):
                continue
            tf = np.abs(policy.predict_features(x)[:, :2]).mean(axis=1); policy.reset()
            sq = np.abs(np.asarray(policy.predict(obs, t))[:, :2]).mean(axis=1)
            tf_res.append(tf[stop:].mean() / drive); seq_res.append(sq[stop:].mean() / drive)
            dt = t[1] - t[0]; k = int(round(1.0 / dt)); still = np.convolve(sq < 0.05, np.ones(k), mode="valid") >= k
            seq_stops += bool(still.any())
        x = np.vstack(xs); y = np.vstack(ys); wheels = y[:, :2]; other = [i for i in range(len(names)) if i not in prev]
        print(f"\n== {run.name}: {len(records)} validation episodes, {len(x)} rows")
        print(f"   linear R^2 of expert wheel commands from the previous command only (2 feats): {r2(x[:, prev], wheels).round(3)}")
        print(f"   linear R^2 from everything else ({len(other)} feats):                           {r2(x[:, other], wheels).round(3)}")
        pred = policy.predict_features(x)[:, :2]
        driving = np.abs(wheels).max(axis=1) > 0.5 * np.abs(wheels).max(); parked = (np.abs(wheels).max(axis=1) < 0.02) & (np.abs(x[:, prev]).max(axis=1) < 0.02)
        x_rest = x.copy(); x_rest[:, prev] = 0.0; pred_rest = policy.predict_features(x_rest)[:, :2]
        med = np.median(x[driving][:, prev], axis=0); x_move = x.copy(); x_move[:, prev] = med; pred_move = policy.predict_features(x_move)[:, :2]
        mag = lambda z: np.abs(z).mean()
        print(f"   driving rows ({driving.sum()}): expert |wheel| {mag(wheels[driving]):.3f}, policy {mag(pred[driving]):.3f}, "
              f"policy with previous command zeroed {mag(pred_rest[driving]):.3f}  -> keeps {100 * mag(pred_rest[driving]) / mag(pred[driving]):.0f}% of its command")
        print(f"   parked rows ({parked.sum()}): expert |wheel| {mag(wheels[parked]):.3f}, policy {mag(pred[parked]):.3f}, "
              f"policy with the median driving command {np.round(med, 2)} injected {mag(pred_move[parked]):.3f}")
        print(f"   residual |wheel| after the expert's stop, as a fraction of the drive level: teacher-forced {100 * np.mean(tf_res):.0f}%, "
              f"sequential on own outputs {100 * np.mean(seq_res):.0f}%;  sequential runs that ever stop for 1 s: {seq_stops}/{len(seq_res)}")
        base = pred; sens = []
        for i, n in enumerate(names):
            xp = x.copy(); xp[:, i] += policy.x_scale[i]; sens.append((np.abs(policy.predict_features(xp)[:, :2] - base).mean(), n))
        print("   mean |d wheel / d feature| (one standardised unit), top 6: " + ", ".join(f"{n} {s:.3f}" for s, n in sorted(sens, reverse=True)[:6]))


if __name__ == "__main__":
    main()
