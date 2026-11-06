"""Step 3, experiment A: the pre-registered mechanism predictions (report.txt, frozen 2026-10-29 19:05), evaluated on
the DEV / HARD runs of ego (A) and ego-novw (B), seeds 0-4. Differences are B - A, two-level bootstrap 95% CI."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import analyze_rollouts as ar

PRED_COPYCAT = ("no_start", "no_progress", "left_or_overshot")


def arms(set_name):
    A, _ = ar.arm_runs(ar.RESULTS_DIR, "ego", range(5), set_name)
    B, _ = ar.arm_runs(ar.RESULTS_DIR, "ego-novw", range(5), set_name)
    eps, bad = ar.paired_episodes(A + B)
    assert not bad, bad
    return A, B, eps


def share(runs, eps, classes):
    return ar.matrix(runs, eps, lambda row: float(row["fclass"] in classes))


def ci(A, B):
    d = ar.paired_diff(A, B)
    return f"A {100 * d['mean_a']:.1f}%  B {100 * d['mean_b']:.1f}%  B - A {100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] pp"


def col(runs, eps, key, mask_fn=None):
    """Pooled seed x episode values of a column, optionally only where mask_fn(row)."""
    out = []
    for run in runs:
        for e in eps:
            row = run.by_episode[e]
            if mask_fn is None or mask_fn(row):
                out.append(ar._f(row, key))
    return np.array(out)


lines = []
for set_name in ("dev", "hard"):
    A, B, eps = arms(set_name)
    d0 = np.array([ar._f(A[0].by_episode[e], "d0") for e in eps])
    lines.append(f"== {set_name.upper()} ({len(eps)} episodes, 5 seeds per arm) ==")
    lines.append("H-copycat")
    lines.append("  c1 no_start + no_progress + left_or_overshot (prediction: B - A CI upper < 0): "
                 + ci(share(A, eps, PRED_COPYCAT), share(B, eps, PRED_COPYCAT)))
    for m in ("reach_5cm", "parked"):
        lines.append(f"  c2 {m} (prediction: B higher): " + ci(ar.metric_matrix(A, eps, m), ar.metric_matrix(B, eps, m)))
    lines.append("H-hidden-phase")
    lines.append("  h1 circling + left_or_overshot (prediction: B - A CI lower > 0): "
                 + ci(share(A, eps, ("circling", "left_or_overshot")), share(B, eps, ("circling", "left_or_overshot"))))
    lines.append("     circling alone: " + ci(share(A, eps, ("circling",)), share(B, eps, ("circling",))))
    for name, mask in (("|d0| < 1", np.abs(d0) < 1), ("|d0| >= 1", np.abs(d0) >= 1)):
        a, b = share(A, eps, ("circling", "left_or_overshot"))[:, mask], share(B, eps, ("circling", "left_or_overshot"))[:, mask]
        lines.append(f"  h3 circling + left_or_overshot, {name} (n {mask.sum()}): " + ci(a, b))
        pa, pb = ar.metric_matrix(A, eps, "parked")[:, mask], ar.metric_matrix(B, eps, "parked")[:, mask]
        lines.append(f"     parked, {name}: " + ci(pa, pb))
    lines.append("  h2 parked lower in B: see c2")
    lines.append("  h4 arm_extended (prediction: B lower): " + ci(ar.metric_matrix(A, eps, "arm_extended"), ar.metric_matrix(B, eps, "arm_extended")))
    nr = [int(np.nansum(col(r, eps, "success") == 1)) for r in (A, B)]
    lines.append(f"  h5 hold10_base_disp_max among reached episodes: reached seed x episode A {nr[0]}, B {nr[1]} -> not assessable")
    lines.append("H-goal (parked seed x episode rows; needs >= 30 per arm)")
    expert = ar.load_prepared(os.path.join(ar.RESULTS_DIR, f"expert_v11_{set_name}"))
    parked = lambda row: np.isfinite(ar._f(row, "parked_time"))  # noqa: E731
    stats = {}
    for label, runs in (("A", A), ("B", B), ("expert", [expert])):
        ring = col(runs, eps, "ring_dist_final", parked)
        gap = col(runs, eps, "base_goal_dist_final", parked) - ring
        stats[label] = (len(ring), np.median(ring) if len(ring) else np.nan, np.median(gap) if len(gap) else np.nan)
        lines.append(f"  {label:6s} parked n {len(ring):4d}: ring_dist_final median {100 * stats[label][1]:.1f} cm; "
                     f"base_goal_dist_final - ring_dist_final median {100 * stats[label][2]:.1f} cm")
    ok = stats["A"][0] >= 30 and stats["B"][0] >= 30
    lines.append(f"  ring median difference B - A {100 * (stats['B'][1] - stats['A'][1]):+.1f} cm (prediction |.| < 3 cm); "
                 f"goal-gap difference {100 * (stats['B'][2] - stats['A'][2]):+.1f} cm (prediction |.| < 5 cm; both above "
                 f"the expert's {100 * stats['expert'][2]:.1f} cm); assessable: {ok}")
    lines.append("Rodrigo prior for ego-novw: main failure class parked_no_arm / extended_not_reached (merged base_ok_ee_not)")
    for label, runs in (("A", A), ("B", B)):
        cls = ar.class_matrix(runs, eps)
        counts = {c: float(np.mean(cls == c)) for c in ar.CLASSES if np.any(cls == c)}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        lines.append(f"  {label}: top classes " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in top))
    lines.append("M1 vs M2 (descriptive): parked_no_arm share among parked seed x episode rows")
    modes = {e: ar.ad.mode_of(ar._f(A[0].by_episode[e], "d0")) for e in eps}
    for label, runs in (("A", A), ("B", B)):
        for m in ("M1", "M2"):
            rows = [run.by_episode[e] for run in runs for e in eps if modes[e] == m and parked(run.by_episode[e])]
            k = sum(row["fclass"] == "parked_no_arm" for row in rows)
            lines.append(f"  {label} {m}: parked rows {len(rows)}, parked_no_arm {k} "
                         f"({100 * k / max(len(rows), 1):.0f}%)" + ("" if len(rows) >= 20 else " (< 20 rows: not interpreted)"))
    lines.append("")

lines.append("== Offline lag k (clean validation episodes; k > 0 = the prediction lags the label), per seed ==")
for model in ("ego", "ego-novw"):
    for ev in ("start", "align>drive", "drive>orient"):
        ks = {"v": [], "w": []}
        for s in range(5):
            with open(f"trained/{model}_s{s}.offline.json") as f:
                lag = json.load(f)["lag"][ev]["model"]
            for o in ("v", "w"):
                ks[o].append(lag[o]["k_best"])
        lines.append(f"  {model:9s} {ev:13s} v {ks['v']}  w {ks['w']}")
with open("trained/ego_s0.offline.json") as f:
    cc = json.load(f)["lag"]
lines.append("  copycat (measured speed): " + ", ".join(f"{ev} v {cc[ev]['copycat']['v']['k_best']} w {cc[ev]['copycat']['w']['k_best']}"
                                                          for ev in ("start", "align>drive", "drive>orient")))
text = "\n".join(lines)
print(text)
with open("results/step3/A_mechanisms.txt", "w") as f:
    f.write(text + "\n")
