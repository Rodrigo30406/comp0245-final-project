"""Step 9: read-out of the ONE sealed test2 opening for the final frozen model (prereg item 7: L2 since H-MM was confirmed
on confirm2 on 2026-11-05) and the expert. Descriptive only: the same rates as step9_analysis.py (same METRICS, two-level
bootstrap, seed Welch), no hypothesis test and no adoption rule. Reuses the frozen analysis module without changing it.
  python results/step9/step9_test2.py                       # test2, L2, seeds 0-4 -> results/step9/step9_test2_final.{txt,json,png}
  python results/step9/step9_test2.py --set confirm2 --configs L1 L2 --seeds 100-104 --out-stem step9_confirm2_wrapcheck
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step9_analysis as sa  # noqa: E402  (frozen; imported, not edited)


def _load_run_allowed(path):
    if not os.path.isfile(os.path.join(path, "summary.txt")):
        raise FileNotFoundError(f"incomplete or missing run: {path}")
    rows, _, _ = sa.classify.classify_run(path, allow_test2=True)
    return sorted(rows, key=lambda r: int(r["episode"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="test2")
    ap.add_argument("--configs", nargs="+", default=["L2"])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--root", default=sa.STEP9_DIR)
    ap.add_argument("--out-stem", default=None)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--allow-test2", action="store_true", help="read the sealed test2 runs (implied by --set test2)")
    args = ap.parse_args()
    if args.set == "test2" or args.allow_test2:
        # the frozen module's load_run opens the run through classify_run, which refuses test2 without this flag
        sa.load_run = lambda path: _load_run_allowed(path)
    seeds = sa.parse_seeds(args.seeds)
    stem = args.out_stem or f"step9_{args.set}_final"

    runs = {c: sa.load_config(args.root, c, seeds, args.set) for c in args.configs}
    first = runs[args.configs[0]][0]
    band = np.array([sa._t(r["ambiguous"]) for r in first])
    lateral = np.array([float(r["lateral_offset"]) for r in first])
    Y = {c: sa.matrices(runs[c]) for c in args.configs}
    out = {"set": args.set, "configs": args.configs, "seeds": seeds, "n_episodes": int(band.size),
           "n_band": int(band.sum()), "overall": {}, "band": {}, "outside": {}, "latched_side": {}, "stop_reasons": {}}
    for c in args.configs:
        out["overall"][c] = {m: sa.rate(Y[c][m]) for m in sa.METRICS}
        out["band"][c] = {m: sa.rate(Y[c][m][:, band]) for m in sa.METRICS}
        out["outside"][c] = {m: sa.rate(Y[c][m][:, ~band]) for m in sa.METRICS}
        out["overall"][c]["policy_ms_mean"] = float(np.mean([float(r["policy_ms_mean"]) for rows in runs[c] for r in rows]))
        passes = [r for rows in runs[c] for r in rows if sa._t(r["passed_obstacle"])]
        lat = ([sa._int(r["side_taken"]) == sa._int(r.get("policy_latched_side")) for r in passes]
               if c.startswith("L2") else [])   # only the side-latched learner has a latched side
        out["latched_side"][c] = {"n_passes": len(passes), "share_on_latched": float(np.mean(lat)) if lat else None}
        classes = {}
        for rows in runs[c]:
            for r in rows:
                classes[r["fclass"]] = classes.get(r["fclass"], 0) + 1
        out["stop_reasons"][c] = {k: classes[k] / (len(runs[c]) * len(first)) for k in sorted(classes)}
    expert_dir = os.path.join(args.root, f"expert_{args.set}")
    if os.path.isfile(os.path.join(expert_dir, "summary.txt")):
        ex = sa.matrices([sa.load_run(expert_dir)])
        out["expert"] = {m: float(ex[m].mean()) for m in sa.METRICS}
        out["expert"]["band_docked_clean"] = float(ex["docked_clean"][:, band].mean())
        with open(os.path.join(expert_dir, "summary.txt")) as f:
            out["expert"]["recorded_comparison"] = [l.strip() for l in f if l.startswith("Recorded comparison")]
    out["bins"] = sa.bin_rates(Y, args.configs, lateral)

    L = [f"Step 9 {args.set} read-out (descriptive, one opening): {out['n_episodes']} episodes ({out['n_band']} in the "
         f"ambiguous band), configs {args.configs}, seeds {seeds}",
         "Rates (seed mean, two-level bootstrap 95% CI): overall | band | outside"]
    for m in sa.METRICS:
        L.append(f"  {m}")
        for c in args.configs:
            L.append(f"    {c:6s} {sa.fmt_rate(out['overall'][c][m])} | {sa.fmt_rate(out['band'][c][m])} | "
                     f"{sa.fmt_rate(out['outside'][c][m])}")
        if "expert" in out:
            L.append(f"    expert {100 * out['expert'][m]:.1f}%")
    for c in args.configs:
        L.append(f"  policy_ms_mean {c}: {out['overall'][c]['policy_ms_mean']:.3f}")
        L.append(f"  per-seed success {c}: {[round(100 * v, 1) for v in out['overall'][c]['success']['per_seed']]}; "
                 f"per-seed docked_clean: {[round(100 * v, 1) for v in out['overall'][c]['docked_clean']['per_seed']]}; "
                 f"per-seed band docked_clean: {[round(100 * v, 1) for v in out['band'][c]['docked_clean']['per_seed']]}")
        ls = out["latched_side"][c]
        if ls["share_on_latched"] is not None:
            L.append(f"  {c} passes on the latched side: {100 * ls['share_on_latched']:.1f}% of {ls['n_passes']}")
        L.append(f"  {c} failure classes (share of all units): " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in
                                                                              out["stop_reasons"][c].items()))
    if "expert" in out:
        L += ["  expert " + s for s in out["expert"].get("recorded_comparison", [])]
    text = "\n".join(L) + "\n"
    print(text, end="")
    with open(os.path.join(args.root, stem + ".txt"), "w") as f:
        f.write(text)
    with open(os.path.join(args.root, stem + ".json"), "w") as f:
        json.dump(sa._clean(out), f, indent=1)

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            centers = 0.5 * (sa.LATERAL_BINS[1:] + sa.LATERAL_BINS[:-1])
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
            for ax, m in zip(axes, ("docked_clean", "collision")):
                for c in args.configs:
                    b = out["bins"][c][m]
                    ax.plot(centers, 100 * np.array(b["mean"]), marker="o", label=f"{c} (seed mean)")
                    ax.fill_between(centers, 100 * np.array(b["min"]), 100 * np.array(b["max"]), alpha=0.2,
                                    label=f"{c} seed range")
                ax.axvspan(-0.2, 0.2, color="0.9", zorder=0)
                ax.set_xlabel("lateral offset d (m)"); ax.set_ylabel(f"{m} (%)"); ax.set_title(f"{args.set}: {m}")
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.root, stem + ".png"), dpi=150)
        except Exception as e:  # the figure is optional
            print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
