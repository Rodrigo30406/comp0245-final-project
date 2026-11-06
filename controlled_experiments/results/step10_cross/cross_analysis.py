"""Cross-evaluation read-out (descriptive, 2026-11-05): both lines of the group on the same DEV / HARD episodes under
protocol v1.1. Arms: expert (results/expert_v11_<set>), C3 = ego-novw-trig s0-4 (this line's main model, early-stop
rows as published), P = ego-novw-trig-pd s0-4, and Rodrigo's v2 policies run through policies/rodrigo_policy.py:
K144 = MLP-Ego-IC (v2_mlp28_trig144 s0-4), GRU40 = GRU-Ego patience 40 (s0-4), CTRL = MLP-Ego control seed 0.
Subsets fixed before any cross run (STEP8_IDEAS_MEMO.md, item 'cross-evaluation'): band = target z in [0.30, 0.80]
(his training height range; DEV n 155), band_box = band and |target x|, |target y| <= 2 m (his target box; DEV n 103).
Statistics as in analyze_rollouts.py: episode = unit, seed-mean rates with the two-level bootstrap 95% CI, paired
differences to C3 with the two-level bootstrap and the per-seed Welch t. No adoption rule: this is a comparison of two
recipes trained on different data, not an experiment with one variable.
"""
import argparse, json, os, sys
import numpy as np

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK)
import analyze_rollouts as ar  # noqa: E402

ARMS = {  # name -> (folder template with {s} and {set}, seeds)
    "expert": ("results/expert_v11_{set}", [None]),
    "C3": ("results/ego-novw-trig_s{s}_{set}", [0, 1, 2, 3, 4]),
    "P": ("results/ego-novw-trig-pd_s{s}_{set}", [0, 1, 2, 3, 4]),
    "K144": ("results/step10_cross/rodrigo-k144_s{s}_{set}", [0, 1, 2, 3, 4]),
    "GRU40": ("results/step10_cross/rodrigo-gru40_s{s}_{set}", [0, 1, 2, 3, 4]),
    "CTRL": ("results/step10_cross/rodrigo-ctrl_s{s}_{set}", [0]),
}
METRICS = ["success", "reach_2cm", "reach_5cm", "held_10s", "stable_1cm", "stable_2cm", "stable_5cm", "tipped"]


def mval(row, m):
    if m in ("success", "reach_2cm", "reach_5cm", "held_10s", "stable_5cm"):
        return ar.metric_value(row, m)
    if m in ("stable_1cm", "stable_2cm"):
        return float(np.isfinite(ar._f(row, f"{m}_time")))
    if m == "tipped":
        return float(row.get("stop_reason") == "tipped")
    raise ValueError(m)


def load_arm(template, seeds, set_name):
    runs = []
    for s in seeds:
        d = os.path.join(TASK, template.format(s=s, set=set_name))
        if not os.path.isfile(os.path.join(d, "summary.txt")):
            return None
        runs.append(ar.load_run(d))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "hard"])
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    arms = {}
    for name, (tpl, seeds) in ARMS.items():
        runs = load_arm(tpl, seeds, args.set)
        if runs is None:
            print(f"[skip] arm {name}: runs missing for set {args.set}")
            continue
        arms[name] = runs
    all_runs = [r for runs in arms.values() for r in runs]
    episodes = ar.common_episodes(all_runs)
    ref_rows = arms["C3"][0].rows if "C3" in arms else all_runs[0].rows
    by_ep = {r["episode"]: r for r in ref_rows}
    z = np.array([ar._f(by_ep[e], "target_z") for e in episodes])
    tx = np.array([ar._f(by_ep[e], "target_x") for e in episodes]); ty = np.array([ar._f(by_ep[e], "target_y") for e in episodes])
    subsets = {"all": np.ones(len(episodes), bool), "band": (z >= 0.30) & (z <= 0.80),
               "band_box": (z >= 0.30) & (z <= 0.80) & (np.abs(tx) <= 2) & (np.abs(ty) <= 2)}
    # failure classes (this task's taxonomy) where the reference is available
    classes = {}
    for name, runs in arms.items():
        try:
            ref = ar.reference_for(runs[0])
            for run in runs:
                ar.classify_run(run, ref)
            classes[name] = True
        except Exception as e:  # noqa: BLE001
            print(f"[warn] classes for {name}: {e!r}")
    lines = [f"Cross-evaluation on {args.set.upper()} ({len(episodes)} common episodes; protocol v1.1; seed-mean rates, "
             f"two-level bootstrap 95% CI, {args.n_boot} draws)", ""]
    result = {"set": args.set, "episodes": len(episodes), "subsets": {k: int(v.sum()) for k, v in subsets.items()}, "arms": {}}
    for sub, mask in subsets.items():
        eps = [e for e, m in zip(episodes, mask) if m]
        lines.append(f"== subset {sub}: n = {len(eps)} ==")
        for name, runs in arms.items():
            res = result["arms"].setdefault(name, {}).setdefault(sub, {})
            parts = []
            for m in METRICS:
                Y = ar.matrix(runs, eps, lambda row, m=m: mval(row, m))
                b = ar.boot2(Y, args.n_boot)
                res[m] = {"mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "per_seed": Y.mean(axis=1).tolist()}
                parts.append(f"{m} {ar.pct_ci(b['mean'], b['lo'], b['hi'])}")
            pm = float(np.mean([ar._f(r.by_episode[e], "policy_ms_mean") for r in runs for e in eps]))
            res["policy_ms_mean"] = pm
            Ys = ar.matrix(runs, eps, lambda row: mval(row, "success"))
            lines.append(f"  {name:7s} success per seed {np.round(100 * Ys.mean(axis=1), 1).tolist()}  policy_ms {pm:.3f}")
            lines.append("          " + "; ".join(parts))
            if classes.get(name):
                cc = ar.class_counts([r.by_episode[e] for r in runs for e in eps])
                res["classes"] = {k: v / (len(runs) * len(eps)) for k, v in cc.items()}
                lines.append("          classes (seed-mean %): " + ", ".join(f"{k} {100 * v / (len(runs) * len(eps)):.1f}" for k, v in sorted(cc.items(), key=lambda kv: -kv[1])))
        if "C3" in arms:
            A = ar.matrix(arms["C3"], eps, lambda row: mval(row, "success"))
            for name in [n for n in arms if n not in ("C3",)]:
                B = ar.matrix(arms[name], eps, lambda row: mval(row, "success"))
                d = ar.paired_diff(A, B, args.n_boot)
                t = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B))
                result["arms"][name][sub]["success_vs_C3"] = {"diff": d["diff"], "boot": [d["lo"], d["hi"]], "p_boot": d["p"],
                                                             "welch": [t["lo"], t["hi"]], "p_welch": t["p"], "test": t["test"]}
                lines.append(f"  success {name} - C3: {ar.pp_ci(d['diff'], d['lo'], d['hi'])} (p_boot {ar.fmt_p(d['p'])}); "
                             f"{t['test']} t [{100 * t['lo']:+.1f}, {100 * t['hi']:+.1f}] pp (p {ar.fmt_p(t['p'])})")
        lines.append("")
    out = os.path.join(TASK, "results", "step10_cross", f"cross_{args.set}")
    with open(out + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(out + ".json", "w") as f:
        json.dump(ar._clean(result), f, indent=1)
    print("\n".join(lines))
    print(f"-> {out}.txt / .json")


if __name__ == "__main__":
    main()
