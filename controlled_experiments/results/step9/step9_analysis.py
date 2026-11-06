"""Step 9 analysis: the pre-registered read-outs of H-MM, H-FACT and H-COPY (report.txt, step-9 pre-registration;
frozen record results/step9/prereg_step9_frozen.txt) on dev2, and the H-MM confirmation on confirm2.

  python results/step9/step9_analysis.py --set dev2                                   # L0 L1 L1-K0 L2, seeds 0-4
  python results/step9/step9_analysis.py --set confirm2 --configs L1 L2 --seeds 100-104

Reads <root>/<config>_s<seed>_<set>/rollouts.csv (failure classes from obstacle_task.classify.classify_run) and the
expert run <root>/expert_<set>; writes <root>/step9_<set>.txt / .json and fig_step9_<set>_{bifurcation,classes,
traj}.{png,pdf}. Statistics exactly as steps 3-7 (analyze_rollouts): episode = unit, configs paired on the same
episodes, two-level bootstrap (seed then episode, 10 000 draws, seed 0) and Welch on per-seed rates, 95% CIs; Holm
over each hypothesis's secondary family with p = max(bootstrap p, Welch p). Nothing here reads data/test or test2.
"""
import argparse
import json
import os
import sys

import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import analyze_rollouts as ar  # noqa: E402
from obstacle_task import classify  # noqa: E402

STEP9_DIR = os.path.join(TASK_DIR, "results", "step9")
CONFIGS = ["L0", "L1", "L1-K0", "L2"]
M1_SHARE = 0.60            # H-MM M1: share of L1's in-band failures that are collision or no_pass
M2_MARGIN = 0.07           # H-MM M2: outside the band both CIs of L2 - L1 within +-7 pp
M4_SHARE = 0.95            # H-MM M4: share of L2's passes on its latched side
COPY_HIGH, COPY_LOW, COPY_SOLVED = 0.50, 0.10, 0.20   # H-COPY thresholds on parked_no_arm
LATERAL_BINS = np.linspace(-0.4, 0.4, 9)             # 0.1 m bins of the lateral offset d
CLASS_GROUPS = ["success", "collision", "no_pass", "crash", "circling", "parked_no_arm", "extended_not_reached"]


def _t(value):
    return str(value) == "True"


def _int(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


METRICS = {
    "docked_clean": lambda r: _t(r["docked_clean"]),
    "success": lambda r: _t(r["success"]),
    "reach_20mm": lambda r: float(r["min_error"]) < 0.020,
    "reach_50mm": lambda r: float(r["min_error"]) < 0.050,
    "collision": lambda r: _t(r["collision"]),
    "no_pass": lambda r: r["fclass"] == "no_pass",
    "crash": lambda r: r["fclass"] == "crash",
    "circling": lambda r: r["fclass"] == "circling",
    "parked_no_arm": lambda r: r["fclass"] == "parked_no_arm",
    "extended_not_reached": lambda r: r["fclass"] == "extended_not_reached",
    "held_10s": lambda r: _t(r["held_10s"]),
}


def load_run(path):
    if not os.path.isfile(os.path.join(path, "summary.txt")):
        raise FileNotFoundError(f"incomplete or missing run: {path}")
    rows, _, _ = classify.classify_run(path)
    return sorted(rows, key=lambda r: int(r["episode"]))


def load_config(root, config, seeds, set_name):
    runs = [load_run(os.path.join(root, f"{config}_s{s}_{set_name}")) for s in seeds]
    eps = [[int(r["episode"]) for r in rows] for rows in runs]
    if any(e != eps[0] for e in eps):
        raise ValueError(f"{config}: runs cover different episodes")
    return runs


def matrices(runs):
    return {m: np.array([[fn(r) for r in rows] for rows in runs], dtype=float) for m, fn in METRICS.items()}


def rate(Y):
    b = ar.boot2(Y)
    return {"mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "per_seed": [float(v) for v in ar.seed_rates(Y)]}


def diff(A, B):
    """B - A on the same episodes: two-level bootstrap and Welch, as analyze_rollouts."""
    pb = ar.paired_diff(A, B)
    w = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B))
    return {"diff": pb["diff"], "boot_lo": pb["lo"], "boot_hi": pb["hi"], "boot_p": pb["p"],
            "welch_lo": w["lo"], "welch_hi": w["hi"], "welch_p": w["p"], "mean_a": pb["mean_a"],
            "mean_b": pb["mean_b"], "p": ar.combined_p(pb["p"], w["p"])}


def holm_family(entries):
    names = list(entries)
    adj, reject = ar.holm([entries[n]["p"] for n in names])
    for n, a, r in zip(names, adj, reject):
        entries[n]["p_holm"], entries[n]["holm_significant"] = float(a), bool(r)
    return entries


def pp(x):
    return f"{100 * x:+.1f}" if np.isfinite(x) else "nan"


def fmt_diff(d):
    return (f"{pp(d['diff'])} pp, bootstrap [{pp(d['boot_lo'])}, {pp(d['boot_hi'])}], Welch [{pp(d['welch_lo'])}, "
            f"{pp(d['welch_hi'])}]")


def fmt_rate(r):
    return f"{100 * r['mean']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def analyse_dev(root, set_name, configs, seeds):
    runs = {c: load_config(root, c, seeds, set_name) for c in configs}
    first = runs[configs[0]][0]
    band = np.array([_t(r["ambiguous"]) for r in first])
    lateral = np.array([float(r["lateral_offset"]) for r in first])
    for c in configs:
        for rows in runs[c]:
            if [_t(r["ambiguous"]) for r in rows] != list(band):
                raise ValueError("ambiguous flags differ between runs")
    Y = {c: matrices(runs[c]) for c in configs}
    out = {"set": set_name, "configs": configs, "seeds": list(seeds), "n_episodes": int(band.size),
           "n_band": int(band.sum()), "overall": {}, "band": {}, "outside": {}}
    for c in configs:
        out["overall"][c] = {m: rate(Y[c][m]) for m in METRICS}
        out["band"][c] = {m: rate(Y[c][m][:, band]) for m in METRICS}
        out["outside"][c] = {m: rate(Y[c][m][:, ~band]) for m in METRICS}
        out["overall"][c]["policy_ms_mean"] = float(np.mean([float(r["policy_ms_mean"]) for rows in runs[c]
                                                             for r in rows]))
    expert_dir = os.path.join(root, f"expert_{set_name}")
    if os.path.isfile(os.path.join(expert_dir, "summary.txt")):
        ex = matrices([load_run(expert_dir)])
        out["expert"] = {m: float(ex[m].mean()) for m in METRICS}
    res = {}
    if {"L1", "L2"} <= set(configs):
        A, B = Y["L1"]["docked_clean"], Y["L2"]["docked_clean"]
        primary = diff(A[:, band], B[:, band])
        fails = [r for rows in runs["L1"] for r, b in zip(rows, band) if b and not _t(r["docked_clean"])]
        m1 = float(np.mean([r["fclass"] in ("collision", "no_pass") for r in fails])) if fails else float("nan")
        outside = diff(A[:, ~band], B[:, ~band])
        m2 = all(abs(outside[k]) <= M2_MARGIN for k in ("boot_lo", "boot_hi", "welch_lo", "welch_hi"))
        m3 = float(np.mean([A[s, band].mean() - A[s, ~band].mean() for s in range(A.shape[0])]))
        passes = [r for rows in runs["L2"] for r in rows if _t(r["passed_obstacle"])]
        on_latched = [_int(r["side_taken"]) == _int(r.get("policy_latched_side")) for r in passes]
        m4 = float(np.mean(on_latched)) if passes else float("nan")
        secondary = holm_family({
            "band_collision": diff(Y["L1"]["collision"][:, band], Y["L2"]["collision"][:, band]),
            "band_no_pass": diff(Y["L1"]["no_pass"][:, band], Y["L2"]["no_pass"][:, band]),
            "band_success": diff(Y["L1"]["success"][:, band], Y["L2"]["success"][:, band]),
            "overall_docked_clean": diff(A, B),
            "overall_success": diff(Y["L1"]["success"], Y["L2"]["success"])})
        res["H-MM"] = {
            "primary": primary,
            "primary_supported": bool(primary["boot_lo"] > 0 and primary["welch_lo"] > 0),
            "M1_share_collision_or_no_pass": m1, "M1_n_failures": len(fails), "M1_holds": bool(m1 >= M1_SHARE),
            "M1_classes": {k: int(sum(r["fclass"] == k for r in fails)) for k in sorted({r["fclass"] for r in fails})},
            "M2_outside": outside, "M2_holds": bool(m2),
            "M3_L1_band_minus_outside": m3, "M3_holds": bool(m3 < 0),
            "M4_share_on_latched_side": m4, "M4_n_passes": len(passes), "M4_holds": bool(m4 >= M4_SHARE),
            "secondary": secondary}
    if {"L0", "L1"} <= set(configs):
        res["H-FACT"] = {"primary": diff(Y["L0"]["docked_clean"], Y["L1"]["docked_clean"]),
                         "secondary": holm_family({m: diff(Y["L0"][m], Y["L1"][m])
                                                   for m in ("success", "collision", "circling", "parked_no_arm")})}
        res["H-FACT"]["prediction_holds"] = bool(res["H-FACT"]["primary"]["diff"] >= 0)
    if {"L1", "L1-K0"} <= set(configs):
        def given_docked(c):
            # per seed: share of the docked units whose class is parked_no_arm (the arm never started)
            shares = []
            for rows in runs[c]:
                docked = [r for r in rows if _t(r["docked"])]
                shares.append(np.mean([r["fclass"] == "parked_no_arm" for r in docked]) if docked else np.nan)
            return {"mean": float(np.nanmean(shares)), "per_seed": [float(v) for v in shares],
                    "n_docked": [int(sum(_t(r["docked"]) for r in rows)) for rows in runs[c]]}
        k0, l1 = given_docked("L1-K0"), given_docked("L1")
        verdict = ("prediction holds: the split alone does not start the arm" if k0["mean"] >= COPY_HIGH and
                   l1["mean"] <= COPY_LOW else "the split alone starts the arm (professor right)"
                   if k0["mean"] < COPY_SOLVED else "in between: neither pre-registered outcome")
        res["H-COPY"] = {"L1-K0_no_start_given_docked": k0, "L1_no_start_given_docked": l1, "verdict": verdict,
                         "L1-K0_parked_no_arm": rate(Y["L1-K0"]["parked_no_arm"]),
                         "L1_parked_no_arm": rate(Y["L1"]["parked_no_arm"]),
                         "parked_no_arm_L1_minus_K0": diff(Y["L1-K0"]["parked_no_arm"], Y["L1"]["parked_no_arm"]),
                         "success_L1_minus_K0": diff(Y["L1-K0"]["success"], Y["L1"]["success"])}
    out["hypotheses"] = res
    out["bins"] = bin_rates(Y, configs, lateral)
    return out, runs, band, lateral


def bin_rates(Y, configs, lateral):
    idx = np.clip(np.digitize(lateral, LATERAL_BINS) - 1, 0, len(LATERAL_BINS) - 2)
    out = {"edges": LATERAL_BINS.tolist(), "n": [int((idx == b).sum()) for b in range(len(LATERAL_BINS) - 1)]}
    for c in configs:
        out[c] = {}
        for m in ("docked_clean", "collision", "no_pass"):
            per_seed = np.array([[Y[c][m][s, idx == b].mean() if (idx == b).any() else np.nan
                                  for b in range(len(LATERAL_BINS) - 1)] for s in range(Y[c][m].shape[0])])
            out[c][m] = {"mean": np.nanmean(per_seed, axis=0).tolist(), "min": np.nanmin(per_seed, axis=0).tolist(),
                         "max": np.nanmax(per_seed, axis=0).tolist()}
    return out


def analyse_confirm(root, set_name, seeds):
    runs = {c: load_config(root, c, seeds, set_name) for c in ("L1", "L2")}
    band = np.array([_t(r["ambiguous"]) for r in runs["L1"][0]])
    A = matrices(runs["L1"])["docked_clean"]
    B = matrices(runs["L2"])["docked_clean"]
    primary = diff(A[:, band], B[:, band])
    return {"set": set_name, "seeds": list(seeds), "n_band": int(band.sum()), "primary": primary,
            "confirmed": bool(primary["diff"] > 0 and primary["boot_lo"] > 0),
            "L1_band": rate(A[:, band]), "L2_band": rate(B[:, band]),
            "L1_overall": rate(A), "L2_overall": rate(B)}


def report_lines(out):
    L = [f"Step 9 analysis, set {out['set']}: {out['n_episodes']} episodes ({out['n_band']} in the ambiguous band), "
         f"configs {out['configs']}, seeds {out['seeds']}", ""]
    L.append("Rates (seed mean, two-level bootstrap 95% CI): overall | band | outside")
    for m in ("docked_clean", "success", "reach_20mm", "reach_50mm", "collision", "no_pass", "crash", "circling",
              "parked_no_arm", "extended_not_reached", "held_10s"):
        L.append(f"  {m}")
        for c in out["configs"]:
            L.append(f"    {c:<6} {fmt_rate(out['overall'][c][m])} | {fmt_rate(out['band'][c][m])} | "
                     f"{fmt_rate(out['outside'][c][m])}")
        if "expert" in out:
            L.append(f"    expert {100 * out['expert'][m]:.1f}%")
    for c in out["configs"]:
        L.append(f"  policy_ms_mean {c}: {out['overall'][c]['policy_ms_mean']:.3f}")
    H = out["hypotheses"]
    if "H-MM" in H:
        h = H["H-MM"]
        L += ["", "H-MM (primary): docked_clean in the band, L2 - L1: " + fmt_diff(h["primary"]) +
              f" -> {'SUPPORTED' if h['primary_supported'] else 'NOT SUPPORTED'}",
              f"  M1: {100 * h['M1_share_collision_or_no_pass']:.1f}% of L1's {h['M1_n_failures']} in-band failures are "
              f"collision or no_pass (>= 60%: {h['M1_holds']}); classes {h['M1_classes']}",
              f"  M2: outside the band L2 - L1 {fmt_diff(h['M2_outside'])} (all within +-7 pp: {h['M2_holds']})",
              f"  M3: L1 band - outside {pp(h['M3_L1_band_minus_outside'])} pp (< 0: {h['M3_holds']})",
              f"  M4: L2 on its latched side in {100 * h['M4_share_on_latched_side']:.1f}% of {h['M4_n_passes']} passes "
              f"(>= 95%: {h['M4_holds']})", "  secondary (Holm):"]
        for n, d in h["secondary"].items():
            L.append(f"    {n}: {fmt_diff(d)}, p_holm {d['p_holm']:.3g}")
    if "H-FACT" in H:
        h = H["H-FACT"]
        L += ["", "H-FACT: overall docked_clean, L1 - L0: " + fmt_diff(h["primary"]) +
              f" (prediction L1 >= L0: {h['prediction_holds']})", "  secondary (Holm):"]
        for n, d in h["secondary"].items():
            L.append(f"    {n}: {fmt_diff(d)}, p_holm {d['p_holm']:.3g}")
    if "H-COPY" in H:
        h = H["H-COPY"]
        k0, l1 = h["L1-K0_no_start_given_docked"], h["L1_no_start_given_docked"]
        L += ["", f"H-COPY: arm never starts given docked: L1-K0 {100 * k0['mean']:.1f}% (per seed "
              f"{[round(100 * v, 1) for v in k0['per_seed']]}, docked {k0['n_docked']}), L1 {100 * l1['mean']:.1f}% "
              f"(per seed {[round(100 * v, 1) for v in l1['per_seed']]}) -> {h['verdict']}",
              f"  unconditional parked_no_arm: L1-K0 {fmt_rate(h['L1-K0_parked_no_arm'])}, L1 {fmt_rate(h['L1_parked_no_arm'])}",
              "  parked_no_arm L1 - L1-K0: " + fmt_diff(h["parked_no_arm_L1_minus_K0"]),
              "  success L1 - L1-K0: " + fmt_diff(h["success_L1_minus_K0"])]
    return L


def figures(out, runs, band, lateral, root, set_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stem = os.path.join(root, f"fig_step9_{set_name}")
    b = out["bins"]
    centers = 0.5 * (np.array(b["edges"][:-1]) + np.array(b["edges"][1:]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, m, title in zip(axes, ("docked_clean", "collision"), ("docked without collision", "collision")):
        ax.axvspan(-0.2, 0.2, color="0.9", zorder=0)
        for k, c in enumerate(out["configs"]):
            mean = np.array(b[c][m]["mean"])
            lo, hi = mean - np.array(b[c][m]["min"]), np.array(b[c][m]["max"]) - mean
            ax.errorbar(centers + (k - 1.5) * 0.008, 100 * mean, yerr=100 * np.vstack([lo, hi]), marker="o", ms=3,
                        capsize=2, lw=1, label=c)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lateral offset d of the obstacle (m)", fontsize=8)
        ax.set_ylim(-3, 103)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("% of episodes (seed mean, min-max)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}_bifurcation.{ext}", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    labels, bottoms = [], None
    groups = [*CLASS_GROUPS, "other"]
    data = []
    for c in out["configs"]:
        for part, mask in (("band", band), ("out", ~band)):
            units = [r["fclass"] for rows in runs[c] for r, keep in zip(rows, mask) if keep]
            counts = [np.mean([u == g for u in units]) for g in CLASS_GROUPS]
            data.append([*counts, 1 - sum(counts)])
            labels.append(f"{c}\n{part}")
    data = np.array(data)
    bottoms = np.zeros(len(labels))
    for j, g in enumerate(groups):
        ax.bar(range(len(labels)), 100 * data[:, j], bottom=100 * bottoms, label=g, width=0.7)
        bottoms += data[:, j]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% of units", fontsize=8)
    ax.legend(fontsize=6, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}_classes.{ext}", dpi=200)
    plt.close(fig)
    trajectory_figure(out, band, root, set_name, stem, plt)


def trajectory_figure(out, band, root, set_name, stem, plt):
    """Top-down paths of the first four in-band episodes (by index, not selected on outcome): expert recording,
    L1 and L2 seed `seeds[0]`."""
    from obstacle_task.scenario import load_scenarios
    data_dir = os.path.join(TASK_DIR, "data", "obstacle", set_name)
    if not os.path.isfile(os.path.join(data_dir, "scenarios.json")):
        print(f"no scenarios.json in {data_dir}: trajectory figure skipped")
        return
    scs, _ = load_scenarios(os.path.join(data_dir, "scenarios.json"))
    picks = [int(e) for e in np.flatnonzero(band)[:4]]
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(7.2, 2.2))
    for ax, ep in zip(np.atleast_1d(axes), picks):
        sc = scs[ep]
        ob = sc.obstacles[0]
        ax.add_patch(plt.Circle((ob["x"], ob["y"]), ob["r"], color="0.4"))
        rec = os.path.join(data_dir, f"ep_{ep:04d}.npz")
        if os.path.isfile(rec):
            xy = np.load(rec)["base_pos"][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], "k--", lw=0.8, label="expert")
        for c in ("L1", "L2"):
            path = os.path.join(root, f"{c}_s{out['seeds'][0]}_{set_name}", "traj", f"ep_{ep:04d}.npz")
            if c in out["configs"] and os.path.isfile(path):
                xy = np.load(path)["base_pos"][:, :2]
                ax.plot(xy[:, 0], xy[:, 1], lw=1, label=c)
        ax.plot(*sc.init_xy, "go", ms=3)
        ax.plot(*sc.dock_xy, "rs", ms=3)
        ax.set_aspect("equal")
        ax.set_title(f"ep {ep}, d = {sc.lateral_offset:+.2f} m", fontsize=7)
        ax.tick_params(labelsize=6)
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}_traj.{ext}", dpi=200)
    plt.close(fig)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k != "draws"}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    return obj


def parse_seeds(text):
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, choices=["dev2", "confirm2", "pilot2"])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--seeds", default=None, help="e.g. 0-4 or 100-104 or 0,1")
    ap.add_argument("--root", default=STEP9_DIR, help="folder holding the run folders (default results/step9)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    if args.set == "confirm2":
        seeds = parse_seeds(args.seeds or "100-104")
        out = analyse_confirm(args.root, args.set, seeds)
        p = out["primary"]
        lines = [f"Step 9 confirmation on confirm2, seeds {seeds}, {out['n_band']} in-band episodes",
                 f"  docked_clean in the band: L1 {fmt_rate(out['L1_band'])}, L2 {fmt_rate(out['L2_band'])}",
                 f"  L2 - L1: {fmt_diff(p)} -> {'CONFIRMED' if out['confirmed'] else 'NOT CONFIRMED'} "
                 "(rule: point estimate > 0 and bootstrap 95% lower bound > 0)",
                 f"  overall docked_clean: L1 {fmt_rate(out['L1_overall'])}, L2 {fmt_rate(out['L2_overall'])}"]
    else:
        configs = args.configs or CONFIGS
        seeds = parse_seeds(args.seeds or "0-4")
        out, runs, band, lateral = analyse_dev(args.root, args.set, configs, seeds)
        lines = report_lines(out)
        if not args.no_figures:
            figures(out, runs, band, lateral, args.root, args.set)
    stem = os.path.join(args.root, f"step9_{args.set}")
    with open(f"{stem}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"{stem}.json", "w") as f:
        json.dump(_clean(out), f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
