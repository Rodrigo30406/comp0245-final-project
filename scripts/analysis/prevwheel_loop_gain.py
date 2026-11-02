"""Closed-loop counterfactual for the previous-command input: at the states the robot actually reached near the goal,
what does the policy output with its own previous command, with the previous command zeroed, and how much does the output
move per unit of previous command (the loop gain of the action feedback)?

For each evaluation trace, the 1 s window around the instant of minimum distance to the base goal. The previous command
is approximated by the recorded command one trace sample (100 ms) earlier. A gain >= 1 means a residual command feeds
itself: the parked state is unstable and the base never settles.

Usage: python scripts/analysis/prevwheel_loop_gain.py EVAL_DIR LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.learning.egocentric import EgocentricPolicy, encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("eval_dir"); ap.add_argument("models", nargs="+"); a = ap.parse_args(); ev = Path(a.eval_dir)
    for m in a.models:
        label, run = m.split("=", 1); policy = EgocentricPolicy.load(Path(run) / "model.npz"); spec = policy.spec; rows = []
        for p in sorted(ev.glob(f"traces/{label}_episode_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                t = z["time"]; obs = z["observation"]; x = z["action_observation"]; tx = z["action_time"]; raw = z["raw_action"]
            ej = Path(str(p).replace("/traces/", "/episodes/")).with_suffix(".json")
            e = json.loads(ej.read_text())
            with np.load(p, allow_pickle=False) as z:
                dist = z["distance"]  # end-effector distance to the target; its minimum marks the arrival instant
            i0 = int(np.argmin(dist)); lo, hi = max(1, i0 - 5), min(len(x), i0 + 5)
            own = raw[lo - 1:hi - 1, :2]
            f_own = encode(x[lo:hi], tx[lo:hi], spec, previous_wheels=own); f_zero = encode(x[lo:hi], tx[lo:hi], spec, previous_wheels=np.zeros_like(own))
            f_plus = encode(x[lo:hi], tx[lo:hi], spec, previous_wheels=own + 0.1)
            y_own = policy.predict_features(f_own)[:, :2]; y_zero = policy.predict_features(f_zero)[:, :2]; y_plus = policy.predict_features(f_plus)[:, :2]
            rows.append((np.abs(raw[lo:hi, :2]).mean(), np.abs(y_own).mean(), np.abs(y_zero).mean(), ((y_plus - y_own) / 0.1).mean(), e["subsystems"]["base_navigation"]["minimum_goal_distance_xy_m"]))
        r = np.array(rows); med = np.median(r, axis=0)
        print(f"  {label:22s} n={len(r)}  around the arrival instant (1 s): recorded |wheel| {med[0]:.2f}; replayed with own previous command {med[1]:.2f}; "
              f"with previous command zeroed {med[2]:.2f}; d(output)/d(previous command) {med[3]:+.2f} (p90 {np.percentile(r[:, 3], 90):+.2f});  min goal distance {med[4]:.2f} m")


if __name__ == "__main__":
    main()
