"""Can a threshold separate the policy's "zero" from its real commands?

Offline (teacher-forced on validation demonstrations): the predicted |wheel| (max of the two wheels, rad/s) on the rows
where the expert's wheels are exactly zero, against the predicted |wheel| on the rows where the expert moves. The expert
never emits a non-zero command below 0.45 rad/s (min_lin_vel / wheel_radius and kp_ori * ang_tol give the same floor), so
any threshold below that would clip nothing the expert does; the question is whether the policy's zero-row predictions
stay below it. Closed loop (evaluation traces): the raw |wheel| while the base is still (speed < 0.02 m/s) against while
it drives (speed > 0.1 m/s).

Usage: python scripts/analysis/zero_separability.py RUN_DIR[=EVAL_DIR/LABEL] ... [--episodes 40]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloop_phase_error import load_policy  # noqa: E402
from robot_learning.data.storage import read_json  # noqa: E402
from robot_learning.learning.dataset import load_episode, record_directory  # noqa: E402

THRESHOLDS = (0.1, 0.2, 0.3, 0.45)


def pct(v, q): return np.percentile(v, q) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    for spec in a.runs:
        run, _, ev = spec.partition("="); run = Path(run); policy = load_policy(run); split = read_json(run / "split.json")
        records = sorted(split["validation"], key=lambda r: r["episode_id"])
        rng = np.random.default_rng(0); records = [records[i] for i in sorted(rng.choice(len(records), a.episodes, replace=False))]
        on_zero, on_move = [], []
        for rec in records:
            d = record_directory(rec, split["dataset"]); x, y, _ = load_episode(d, rec)
            with np.load(Path(d) / "episodes" / f"{rec['episode_id']}.npz", allow_pickle=False) as data:
                t = data["time"].astype(np.float64)
            policy.reset(); pred = np.abs(np.asarray(policy.predict(x, t))[:, :2]).max(axis=1); expert = np.abs(y[:, :2]).max(axis=1)
            on_zero.append(pred[expert == 0]); on_move.append(pred[expert > 0])
        z = np.concatenate(on_zero); m = np.concatenate(on_move)
        print(f"== {run.name}")
        print(f"  offline, expert zero rows ({len(z)}): predicted |wheel| median {np.median(z):.3f}  p90 {pct(z,90):.3f}  p99 {pct(z,99):.3f}  max {z.max():.3f} rad/s")
        print(f"  offline, expert moving rows ({len(m)}): predicted |wheel| min {m.min():.3f}  p1 {pct(m,1):.3f}  p5 {pct(m,5):.3f}   (expert's own minimum 0.450)")
        print("  threshold -> zero rows that would stay non-zero | moving rows that would be clipped to zero:  " +
              "  ".join(f"{th:.2f}: {100*(z>th).mean():4.1f}% | {100*(m<th).mean():4.1f}%" for th in THRESHOLDS))
        if ev:
            ev_dir, _, label = ev.rpartition("/"); still, drive = [], []
            for p in sorted(Path(ev_dir).glob(f"traces/{label}_episode_*.npz")):
                with np.load(p, allow_pickle=False) as zz:
                    tt = zz["time"]; obs = zz["observation"]; raw = np.abs(zz["raw_action"][:, :2]).max(axis=1)
                speed = np.hypot(*np.gradient(obs[:, :2], tt, axis=0).T)
                still.append(raw[speed < 0.02]); drive.append(raw[speed > 0.1])
            s = np.concatenate(still); dr = np.concatenate(drive)
            print(f"  closed loop ({label}), base still ({len(s)} samples): |wheel| median {np.median(s):.3f}  p90 {pct(s,90):.3f}  p99 {pct(s,99):.3f};  "
                  f"driving ({len(dr)}): min {dr.min():.3f}  p1 {pct(dr,1):.3f}  p5 {pct(dr,5):.3f}")
            print("  threshold -> still samples left non-zero | driving samples clipped:  " +
                  "  ".join(f"{th:.2f}: {100*(s>th).mean():4.1f}% | {100*(dr<th).mean():4.1f}%" for th in THRESHOLDS))


if __name__ == "__main__":
    main()
