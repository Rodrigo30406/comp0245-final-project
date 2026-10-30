"""Open-loop (teacher-forced) check of the stop: on validation demonstrations, does the policy's predicted wheel command
go to zero when the expert's does, and stay there?

For each run: N validation episodes of its own split; the expert's stop is the last row with |wheel| > 0.02.
Reports the drive-phase error and the residual wheel command the policy still outputs 0.5 s, 2 s and 5 s after the stop,
as a percentage of the expert's mean drive-phase |wheel|.

Usage: python scripts/analysis/openloop_phase_error.py RUN_DIR [RUN_DIR ...] [--episodes 40]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import load_episode, record_directory  # noqa: E402


def load_policy(run):
    info = read_json(run / "model_info.json")
    if info.get("kind", "").startswith("gru") or "cell" in info:
        from robot_learning.learning.gru_policy import GRUPolicy
        return GRUPolicy.load(run / "model.npz")
    from robot_learning.learning.egocentric import EgocentricPolicy
    return EgocentricPolicy.load(run / "model.npz")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = load_policy(run); split = read_json(run / "split.json")
        records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        drive_err, resid = [], {0.5: [], 2.0: [], 5.0: []}; late = []
        for rec in records:
            d = record_directory(rec, split["dataset"]); x, y, _ = load_episode(d, rec)
            with np.load(Path(d) / "episodes" / f"{rec['episode_id']}.npz", allow_pickle=False) as data:
                t = data["time"].astype(np.float64)
            policy.reset(); pred = np.asarray(policy.predict(x, t))[:, :2]; wheels = y[:, :2]
            moving = np.abs(wheels).max(axis=1) > 0.02
            if not moving.any():
                continue
            stop = np.flatnonzero(moving)[-1] + 1; drive_level = np.abs(wheels[:stop]).mean()
            drive_err.append(np.abs(pred[:stop] - wheels[:stop]).mean() / drive_level)
            dt = t[1] - t[0]
            for h in resid:
                k = stop + int(round(h / dt))
                if k < len(pred):
                    resid[h].append(np.abs(pred[k - 50:k + 50]).mean() / drive_level)
            # does the prediction stay below 10% of the drive level from 0.5 s after the stop to the end of the demo?
            tail = np.abs(pred[stop + int(round(0.5 / dt)):]).mean(axis=1) if stop + int(round(0.5 / dt)) < len(pred) else np.array([])
            late.append(float((tail > 0.1 * drive_level).mean()) if len(tail) else np.nan)
        print(f"{run.name:48s} n={len(drive_err):2d}  drive-phase |err|/|expert| = {np.mean(drive_err):.2f}   "
              f"residual |wheel| after stop: 0.5 s {100 * np.mean(resid[0.5]):4.0f}%  2 s {100 * np.mean(resid[2.0]):4.0f}%  5 s {100 * np.mean(resid[5.0]):4.0f}%   "
              f"fraction of post-stop time above 10%: {100 * np.nanmean(late):3.0f}%")


if __name__ == "__main__":
    main()
