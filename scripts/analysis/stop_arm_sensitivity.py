"""Do the wheels restart when the ARM is not where the demonstrations had it at the stop?

At the expert's base stop (+0.2 s) on validation demonstrations the base pose is left untouched and the arm joint
positions (or velocities) are perturbed by random offsets of a given size; the policy's wheel command is read
(teacher-forced history for recurrent policies) as a percentage of the expert's drive level.

Usage: python scripts/analysis/stop_arm_sensitivity.py RUN_DIR [RUN_DIR ...] [--episodes 20]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import load_episode, record_directory  # noqa: E402
from openloop_phase_error import load_policy  # noqa: E402

Q_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4]
QD_SIGMAS = [0.2, 1.0]
DIRECTIONS = 8


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = load_policy(run); split = read_json(run / "split.json")
        records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        out = {("q", s): [] for s in Q_SIGMAS}; out.update({("qd", s): [] for s in QD_SIGMAS})
        for rec in records:
            d = record_directory(rec, split["dataset"]); x, y, _ = load_episode(d, rec)
            with np.load(Path(d) / "episodes" / f"{rec['episode_id']}.npz", allow_pickle=False) as data:
                t = data["time"].astype(np.float64)
            wheels = y[:, :2]; moving = np.abs(wheels).max(axis=1) > 0.02; dt = t[1] - t[0]
            k = min(int(np.flatnonzero(moving)[-1] + 1 + round(0.2 / dt)), len(x) - 1); drive = np.abs(wheels[moving]).mean()
            prng = np.random.default_rng(1)
            for key, sig in list(out):
                for _ in range(DIRECTIONS if sig else 1):
                    xp = x[: k + 1].copy()
                    u = prng.normal(size=7); u /= np.linalg.norm(u)
                    if key == "q":
                        xp[k, 4:11] += sig * np.sqrt(7) * u
                    else:
                        xp[k, 11:18] += sig * np.sqrt(7) * u
                    policy.reset(); pred = np.asarray(policy.predict(xp, t[: k + 1]))[-1, :2]
                    out[(key, sig)].append(np.abs(pred).mean() / drive)
        cells = "  ".join(f"{k} {s:g}: {100 * np.mean(v):3.0f}%" for (k, s), v in out.items())
        print(f"{run.name:48s} |wheel| at the stop with the arm perturbed (rms per joint, rad or rad/s): {cells}")


if __name__ == "__main__":
    main()
