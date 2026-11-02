"""Why does a parked base leave? Decomposition of the restart on closed-loop traces.

For every episode where the base parked (speed < 0.02 m/s for >= 0.5 s) and later restarted (> 0.05 m/s), the policy is
re-run from the parked instant on three input streams built from the observations it actually saw:
  clock only : the parked observation held constant, the clock running on            (does time alone flip the output?)
  state only : the real observation sequence (arm moving, base drifting), clock frozen at the parking time
  replay     : real observations and real clock (sanity check: must reproduce the recorded restart)
A 'flip' is the first instant the 0.5 s moving average of the mean |wheel command| exceeds THRESH (0.3, two bins of the
categorical head; the regression creep sits at ~0.13); the duty cycle is the fraction of parked samples above THRESH, which
separates a sustained drive from a chattering output that flickers between zero and a drive bin. Reports how often each stream flips before the recorded restart (+1 s slack) and the
median flip delay, so the restart can be attributed to the clock, to the state drift, or to their combination.

Usage: python scripts/analysis/restart_cause.py EVAL_DIR LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--thresh 0.3]
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from openloop_phase_error import load_policy  # noqa: E402


def park_restart(t, obs):
    speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T); still = speed < 0.02; run = 0.0
    for i in range(1, len(t)):
        run = run + (t[i] - t[i - 1]) if still[i] else 0.0
        if run >= 0.5:
            later = np.flatnonzero(speed[i:] > 0.05)
            return (i, i + later[0]) if len(later) else None
    return None


def first_flip(policy, x, t, thresh, window=5):
    policy.reset(); w = np.abs(np.asarray(policy.predict(x, t))[:, :2]).mean(axis=1)
    avg = np.convolve(w, np.ones(window) / window, mode="valid"); k = np.flatnonzero(avg > thresh)
    return ((t[k[0] + window - 1] - t[0]) if len(k) else np.inf), w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("eval_dir"); ap.add_argument("models", nargs="+"); ap.add_argument("--thresh", type=float, default=0.3)
    a = ap.parse_args(); ev = Path(a.eval_dir)
    for m in a.models:
        label, run = m.split("=", 1); policy = load_policy(Path(run)); rows = []
        for p in sorted(ev.glob(f"traces/{label}_episode_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                t = z["time"]; obs = z["observation"]; x = z["action_observation"]; tx = z["action_time"]
            pr = park_restart(t, obs)
            if pr is None:
                continue
            park, restart = pr; delay = t[restart] - t[park]
            held = np.repeat(x[park:park + 1], len(x) - park, axis=0)
            clock, _ = first_flip(policy, held, tx[park:], a.thresh)
            state, _ = first_flip(policy, x[park:], np.full(len(x) - park, tx[park]), a.thresh)
            replay, w = first_flip(policy, x[park:], tx[park:], a.thresh)
            duty = (w[:restart - park] > a.thresh).mean() if restart > park else np.nan
            rows.append((delay, clock, state, replay, duty))
        r = np.array(rows); d = r[:, 0]
        before = lambda c: 100 * np.mean(r[:, c] <= d + 1.0)
        med = lambda c: np.median(r[:, c][np.isfinite(r[:, c])]) if np.isfinite(r[:, c]).any() else np.nan
        print(f"  {label:24s} n={len(r):3d}  recorded restart after {np.median(d):5.1f} s;  flips before it: clock only {before(1):3.0f}% (median {med(1):5.1f} s), "
              f"state only {before(2):3.0f}% (median {med(2):5.1f} s), replay {before(3):3.0f}% (median {med(3):5.1f} s);  never flips: clock {100 * np.mean(~np.isfinite(r[:, 1])):3.0f}%  state {100 * np.mean(~np.isfinite(r[:, 2])):3.0f}%;  "
              f"duty cycle above thresh while parked {100 * np.nanmedian(r[:, 4]):3.0f}%")


if __name__ == "__main__":
    main()
