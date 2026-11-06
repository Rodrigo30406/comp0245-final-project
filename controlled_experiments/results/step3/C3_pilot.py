"""Experiment C3, pilot closed-loop diagnostic (descriptive; report.txt "[Results] Step 3 continued: K calibration of
experiment C3" item 5, reading rule written before the runs). EM cell (expert base + model arm,
--no-early-stop) on data/pilot for the calibration models K 64 / K 256 (seed 0) and the control ego-novw_s0.

  python results/step3/C3_pilot.py      -> results/step3/C3_pilot/C3_pilot.txt
"""
import os
import sys

import numpy as np

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK)
import analyze_rollouts as ar  # noqa: E402

D = os.path.join(ar.STEP3_DIR, "C3_pilot")
RUNS = [("control ego-novw_s0", "B-EM_ego-novw_s0_pilot"), ("K 64 seed 0", "B-EM_ego-novw-trig-k64_s0_pilot"),
        ("K 256 seed 0", "B-EM_ego-novw-trig-k256_s0_pilot")]
TRIGGER_LO, NO_TRIGGER_HI = 0.30, 0.20


def main():
    lines = ["C3 pilot diagnostic: EM cell (expert base, model arm, --no-early-stop) on data/pilot; descriptive only",
             f"reading (written before the runs): arm_extended_toward > {TRIGGER_LO:.0%} = triggers in closed loop, "
             f"< {NO_TRIGGER_HI:.0%} = does not, otherwise uncertain (one seed, 100 episodes; Wilson 95%)"]
    runs = []
    for label, name in RUNS:
        p = os.path.join(D, name)
        if not os.path.exists(os.path.join(p, "rollouts.csv")):
            lines.append(f"{label}: missing {os.path.relpath(p, TASK)}")
            continue
        run = ar.load_prepared(p)
        if run.meta.get("warnings"):
            raise SystemExit(f"{name}: {run.meta['warnings']}")
        runs.append((label, run))
    if not runs:
        print("\n".join(lines))
        return
    eps, _ = ar.paired_episodes([r for _, r in runs])
    lines.append(f"episodes common to all runs: {len(eps)}; integrity: "
                 + ("; ".join(ar.integrity_problems([r for _, r in runs])) or "ok"))
    for label, run in runs:
        lines.append(f"== {label} ({run.name})")
        for metric in ("arm_extended_toward", "arm_extended", "success", "reach_5cm", "reach_2cm", "parked"):
            y = ar.metric_matrix([run], eps, metric)[0]
            k, n = int(y.sum()), len(y)
            lo, hi = ar.wilson(k, n)
            tag = ""
            if metric == "arm_extended_toward":   # the share against 30% / 20%; the Wilson interval shown with it
                tag = ("  -> triggers (> 30%)" if k / n > TRIGGER_LO else
                       "  -> does not trigger (< 20%)" if k / n < NO_TRIGGER_HI else "  -> uncertain")
            lines.append(f"  {metric:<20} {k:3d}/{n} = {100 * k / n:5.1f}% [Wilson {100 * lo:.1f}, {100 * hi:.1f}]{tag}")
        logs = ar.run_logs(run, eps)
        lag, no_nav = [], 0
        for e in eps:
            ext = ar._f(run.by_episode[e], "arm_extend_time")
            if np.isfinite(ext):
                log = logs[e]
                if log is None or log["nav_done"] is None:
                    no_nav += 1
                else:
                    lag.append(ext - log["nav_done"]["t"])
        lag = np.array(lag)
        if lag.size:
            lines.append(f"  arm_extend_time - shadow FSM nav_done: n {lag.size}, median {np.median(lag):.2f} s "
                         f"[p10 {np.percentile(lag, 10):.2f}, p90 {np.percentile(lag, 90):.2f}], min {lag.min():.2f}, "
                         f"max {lag.max():.2f}; before nav_done {int(np.sum(lag < 0))}; no nav_done event {no_nav} "
                         f"(expert 0.33-2.7 s, median 0.47 s)")
        ext_rows = [run.by_episode[e] for e in eps if np.isfinite(ar._f(run.by_episode[e], "arm_extend_time"))]
        lines.append("  classes: all " + str(ar.class_counts([run.by_episode[e] for e in eps]))
                     + "; extended cells " + str(ar.class_counts(ext_rows)))
        lines.append("  min_error of the extended cells (m): " + (
            f"median {np.median([ar._f(r, 'min_error') for r in ext_rows]):.3f}" if ext_rows else "-"))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(D, "C3_pilot.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
