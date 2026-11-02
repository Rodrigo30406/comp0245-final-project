"""Which inputs move the wheel command? Mean |d wheel / d feature| for one standardised unit of each feature, on
teacher-forced validation rows, split into driving rows and parked rows (demonstrated wheels zero). Groups the features
into base pose, target geometry, arm position, arm velocity, clock and previous command.

Usage: python scripts/analysis/wheel_sensitivity.py RUN_DIR [RUN_DIR ...] [--episodes 40]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import record_directory  # noqa: E402
from robot_learning.learning.egocentric import EgocentricPolicy, VARIANTS, load_episode_features  # noqa: E402

GROUPS = [("base pose", lambda n: n.startswith("base_")), ("target", lambda n: "target" in n), ("arm q", lambda n: n.startswith("arm_q_")),
          ("arm qd", lambda n: n.startswith("arm_qd_")), ("clock", lambda n: "clock" in n or "time" in n), ("prev cmd", lambda n: n.startswith("prev_wheel"))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=40); a = ap.parse_args()
    for run in a.runs:
        run = Path(run); policy = EgocentricPolicy.load(run / "model.npz"); names = list(VARIANTS[policy.spec["variant"]])
        split = read_json(run / "split.json"); records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        xs, ys = zip(*(load_episode_features(record_directory(r, split["dataset"]), r, policy.spec) for r in records))
        x = np.vstack(xs)[::20]; y = np.vstack(ys)[::20]; wheels = y[:, :2]
        parked = np.abs(wheels).max(axis=1) == 0; driving = np.abs(wheels).max(axis=1) > 1.0
        base = policy.predict_features(x)[:, :2]; sens = np.zeros(len(names)); sens_p = np.zeros(len(names))
        for i in range(len(names)):
            xp = x.copy(); xp[:, i] += policy.x_scale[i]; d = np.abs(policy.predict_features(xp)[:, :2] - base).mean(axis=1)
            sens[i] = d[driving].mean(); sens_p[i] = d[parked].mean()
        tot, tot_p = sens.sum(), sens_p.sum()
        cells = "  ".join(f"{g} {100 * sum(sens[i] for i, n in enumerate(names) if f(n)) / tot:3.0f}%/{100 * sum(sens_p[i] for i, n in enumerate(names) if f(n)) / tot_p:3.0f}%" for g, f in GROUPS)
        print(f"{run.name:44s} rows {len(x)} (driving {driving.sum()}, parked {parked.sum()});  share of total wheel sensitivity, driving/parked: {cells};  total {tot:.2f}/{tot_p:.2f}")
        top = sorted(((sens_p[i], n) for i, n in enumerate(names)), reverse=True)[:5]
        print(f"{'':44s} parked rows, top 5: " + ", ".join(f"{n} {s:.3f}" for s, n in top))


if __name__ == "__main__":
    main()
