"""How wide is the policy's stop around the expert's stop pose?

At the expert's base stop (+0.2 s, arm still in its hold posture) on validation demonstrations, the base pose is displaced
in the base frame (forward/lateral up to 15 cm, yaw up to 0.1 rad) with everything else unchanged and the policy's wheel
command is read (teacher-forced history for recurrent policies). Reports the mean |wheel| as a percentage of the expert's
drive level inside the expert's 8 cm box, just outside it, and with a heading offset beyond the 0.05 rad tolerance.

Usage: python scripts/analysis/stop_basin.py RUN_DIR [RUN_DIR ...] [--episodes 20]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import load_episode, record_directory  # noqa: E402
from openloop_phase_error import load_policy  # noqa: E402

OFFSETS = np.arange(-0.15, 0.151, 0.05)
YAWS = [0.0, -0.1, 0.1]
FAR = [-0.4, -0.3, -0.2, 0.2, 0.3, 0.4]  # forward displacements beyond the grid (positive = towards the target)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = load_policy(run); split = read_json(run / "split.json")
        records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        grid = []  # (forward, lateral, yaw, |wheel|/drive)
        for rec in records:
            d = record_directory(rec, split["dataset"]); x, y, _ = load_episode(d, rec)
            with np.load(Path(d) / "episodes" / f"{rec['episode_id']}.npz", allow_pickle=False) as data:
                t = data["time"].astype(np.float64)
            wheels = y[:, :2]; moving = np.abs(wheels).max(axis=1) > 0.02; dt = t[1] - t[0]
            k = min(int(np.flatnonzero(moving)[-1] + 1 + round(0.2 / dt)), len(x) - 1); drive = np.abs(wheels[moving]).mean()
            yaw = x[k, 3]; c, s = np.cos(yaw), np.sin(yaw)
            for f in FAR:
                xp = x[: k + 1].copy(); xp[k, 0] += c * f; xp[k, 1] += s * f
                policy.reset(); grid.append((f, 0.0, 9.0, np.abs(np.asarray(policy.predict(xp, t[: k + 1]))[-1, :2]).mean() / drive))
            for f in OFFSETS:
                for l in OFFSETS:
                    for dy in YAWS:
                        xp = x[: k + 1].copy(); xp[k, 0] += c * f - s * l; xp[k, 1] += s * f + c * l; xp[k, 3] += dy
                        policy.reset(); pred = np.asarray(policy.predict(xp, t[: k + 1]))[-1, :2]
                        grid.append((f, l, dy, np.abs(pred).mean() / drive))
        g = np.array(grid); r = np.hypot(g[:, 0], g[:, 1]); yaw0 = g[:, 2] == 0; far = g[:, 2] == 9.0
        cell = lambda m: f"{100 * g[m, 3].mean():4.0f}%"
        print(f"{run.name:48s} |wheel| at the stop: inside 8 cm box {cell((r <= 0.08) & yaw0)}  ring 8–15 cm {cell((r > 0.08) & (r <= 0.16) & yaw0)}  "
              f"inside box, yaw ±0.1 rad {cell((r <= 0.08) & ~yaw0)}  | towards target "
              + "  ".join(f"{int(100 * f)} cm {cell(far & (g[:, 0] == f))}" for f in FAR if f > 0) + "  | away "
              + "  ".join(f"{int(-100 * f)} cm {cell(far & (g[:, 0] == f))}" for f in FAR if f < 0))


if __name__ == "__main__":
    main()
