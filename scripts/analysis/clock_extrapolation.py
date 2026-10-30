"""Does the policy keep the base still once the clock runs past the length of any demonstration?

Teacher-forced on validation demonstrations, then the final observation (base parked, arm wherever the demo left it) is
held constant while the clock keeps running to the horizon. Reports the policy's mean |wheel| in windows after the demo
end, as a percentage of the expert's drive-phase |wheel|. A policy that extrapolates the clock badly starts driving again.

Usage: python scripts/analysis/clock_extrapolation.py RUN_DIR [RUN_DIR ...] [--episodes 40] [--horizon 40]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import load_episode, record_directory  # noqa: E402
from openloop_phase_error import load_policy  # noqa: E402

WINDOWS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--horizon", type=float, default=40.0)
    ap.add_argument("--freeze-at", choices=["end", "stop"], default="end",
                    help="'end': hold the demo's final observation (arm at the target); 'stop': hold the observation at the "
                         "expert's base stop (arm still in its hold posture), which is where a policy whose arm never fires sits")
    a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = load_policy(run); split = read_json(run / "split.json")
        records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        res = {w: [] for w in WINDOWS}; lengths = []
        for rec in records:
            d = record_directory(rec, split["dataset"]); x, y, _ = load_episode(d, rec)
            with np.load(Path(d) / "episodes" / f"{rec['episode_id']}.npz", allow_pickle=False) as data:
                t = data["time"].astype(np.float64)
            dt = t[1] - t[0]; n_ext = int(round(a.horizon / dt)) - len(x)
            if n_ext <= 0:
                continue
            wheels = y[:, :2]; moving = np.abs(wheels).max(axis=1) > 0.02
            drive_level = np.abs(wheels[moving]).mean()
            end = len(x) if a.freeze_at == "end" else int(np.flatnonzero(moving)[-1] + 1 + round(0.2 / dt))
            end = min(end, len(x)); n_ext = int(round(a.horizon / dt)) - end
            x_ext = np.vstack([x[:end], np.repeat(x[end - 1:end], n_ext, axis=0)]); t_ext = t[0] + dt * np.arange(len(x_ext))
            policy.reset(); pred = np.abs(np.asarray(policy.predict(x_ext, t_ext))[:, :2]).mean(axis=1)
            lengths.append(t_ext[end - 1])
            for w in WINDOWS:
                i0, i1 = end + int(w[0] / dt), min(end + int(w[1] / dt), len(pred))
                if i1 > i0:
                    res[w].append(pred[i0:i1].mean() / drive_level)
        cells = "  ".join(f"+{w[0]:.0f}..{w[1]:.0f}s {100 * np.mean(v):4.0f}%" for w, v in res.items())
        print(f"{run.name:48s} n={len(lengths):2d} frozen at {np.median(lengths):.1f} s ({a.freeze_at})  |wheel| after freeze: {cells}")


if __name__ == "__main__":
    main()
