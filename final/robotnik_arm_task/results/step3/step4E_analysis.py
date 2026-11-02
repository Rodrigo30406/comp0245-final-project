"""Step 4 E, descriptive analysis (report.txt "[Methodology] 第 4 步 E：学习曲线与降采样步长的事先读法"; STEP4_PLAN_PROMPT.md 3.2):
learning curve (training episodes) and downsampling stride (training rows) on the C3 recipe, seeds 0-2, DEV, all runs
without the early stop. The f = 1 / s = 9 point is ego-novw-trig_s0-2 (DEV runs with their stalled episodes replaced by
the --no-early-stop re-runs, as in experiment D).

  python results/step3/step4E_analysis.py   -> results/step3/step4E.txt / .json, fig_step4_learning_curve.*, fig_step4_stride.*
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import D_mechanisms as dm  # noqa: E402  (merged C3 rows, definitions)

cm, ar = dm.cm, dm.ar
SEEDS = [0, 1, 2]
SETTINGS = [  # name, model, training episodes, label
    ("f0125", "ego-novw-trig-f0125", None, "f 1/8"), ("f025", "ego-novw-trig-f025", None, "f 1/4"),
    ("f05", "ego-novw-trig-f05", None, "f 1/2"), ("f1", "ego-novw-trig", None, "f 1 (s 9)"),
    ("s27", "ego-novw-trig-s27", None, "s 27"), ("s81", "ego-novw-trig-s81", None, "s 81"),
]
METRICS = ("success", "arm_extended_toward", "reach_5cm", "held_10s")


def load_setting(model):
    paths = [ar.run_dir_for(ar.RESULTS_DIR, model, s, "dev") for s in SEEDS]
    if not cm.have(paths):
        return None
    runs = cm.load(paths)
    if model == "ego-novw-trig":
        runs, _ = dm.merged_c3(runs)
    else:
        dm.require_no_early_stop(runs, model)
    return runs


def model_meta(model, seed):
    with np.load(os.path.join(ar.TRAINED_DIR, f"{model}_s{seed}.npz")) as z:
        m = json.loads(str(z["meta"]))
    return m


def main():
    lines = ["Step 4 E (descriptive): learning curve and stride on the C3 recipe, DEV, seeds 0-2, no early stop"]
    res, runs_of = {}, {}
    for key, model, _, label in SETTINGS:
        runs = load_setting(model)
        if runs is None:
            lines.append(f"  {label}: missing runs")
            continue
        runs_of[key] = runs
        metas = [model_meta(model, s) for s in SEEDS]
        n_eps = metas[0].get("n_train_episodes")
        rows = metas[0].get("n_train_rows_base", metas[0].get("trigger", {}).get("n_train_rows_base"))
        eps = sorted(runs[0].by_episode)
        r = {"label": label, "n_episodes": n_eps, "n_rows": rows,
             "val_loss": [m.get("val_loss") for m in metas],
             "t_samples_at_best": [m.get("t_samples_at_best") or (m["best_epoch"] * m["n_train_rows"]) for m in metas]}
        for metric in METRICS:
            Y = ar.metric_matrix(runs, eps, metric)
            b = ar.boot2(Y)
            r[metric] = {"per_seed": ar.seed_rates(Y), "mean": b["mean"], "lo": b["lo"], "hi": b["hi"]}
        rr = dm.reach_required(runs, eps)
        F = dm.fine_matrix(runs, rr)
        b = ar.boot2(F)
        r["fine"] = {"per_seed": ar.seed_rates(F), "mean": float(F.mean()), "lo": b["lo"], "hi": b["hi"]}
        cls = ar.class_matrix(runs, eps)
        r["classes"] = {c: float(np.mean(cls == c)) for c in ar.CLASSES if np.any(cls == c)}
        r["class_ci"] = {}
        for c in r["classes"]:
            C = (cls == c).astype(float)
            b = ar.boot2(C)
            r["class_ci"][c] = {"per_seed": ar.seed_rates(C), "lo": b["lo"], "hi": b["hi"]}
        res[key] = r
        lines.append(f"  {label:10s} episodes {n_eps}, rows {rows}: success {cm.pct(r['success']['mean'])}% "
                     f"[{cm.pct(r['success']['lo'])}, {cm.pct(r['success']['hi'])}] (per seed "
                     + "/".join(cm.pct(x) for x in r["success"]["per_seed"]) + f"); arm_extended_toward "
                     f"{cm.pct(r['arm_extended_toward']['mean'])}%; fine {cm.pct(r['fine']['mean'])}%; held_10s "
                     f"{cm.pct(r['held_10s']['mean'])}%; val loss " + "/".join(f"{v:.4f}" for v in r["val_loss"])
                     + "; classes " + ", ".join(f"{c} {cm.pct(v)}" for c, v in sorted(r["classes"].items(), key=lambda kv: -kv[1])[:4]))
    # P-E1: slope of success on log2(training episodes), 12 models (f variants + f 1), t interval
    lc = [k for k in ("f0125", "f025", "f05", "f1") if k in res]
    if len(lc) >= 3:
        x = np.concatenate([[np.log2(res[k]["n_episodes"])] * len(SEEDS) for k in lc])
        y = np.concatenate([res[k]["success"]["per_seed"] for k in lc])
        fit = stats.linregress(x, y)
        q = stats.t.ppf(0.975, len(x) - 2)
        res["P-E1"] = {"slope": fit.slope, "lo": fit.slope - q * fit.stderr, "hi": fit.slope + q * fit.stderr, "n": len(x)}
        lines.append(f"P-E1 success per doubling of the training episodes: {100 * fit.slope:+.1f} pp "
                     f"[{100 * res['P-E1']['lo']:+.1f}, {100 * res['P-E1']['hi']:+.1f}] (t, {len(x)} models)")
    for a, b, name in (("f05", "f1", "P-E1 1/2 -> 1"), ("f0125", "s81", "P-E2 equal rows: 1613 episodes (s 81) - 201 episodes (f 1/8)"),
                       ("f1", "s27", "P-E3 s 27 - s 9"), ("f1", "s81", "P-E3 s 81 - s 9")):
        if a in runs_of and b in runs_of:
            eps = sorted(runs_of[a][0].by_episode)
            A, B = ar.metric_matrix(runs_of[a], eps, "success"), ar.metric_matrix(runs_of[b], eps, "success")
            d, t = cm.diff_both(A, B, ar.Bootstrap([len(SEEDS), len(SEEDS)], len(eps), ar.N_BOOT, ar.BOOT_SEED))
            res[name] = {"diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "welch_lo": t["lo"], "welch_hi": t["hi"]}
            lines.append(f"{name}: success {dm.fmt(d, t)}")
    pts = [(np.mean(res[k]["val_loss"]), res[k]["success"]["mean"]) for k in res if k in dict((s[0], 1) for s in SETTINGS)]
    if len(pts) >= 4:
        allp = [(v, s) for k in res if isinstance(res[k], dict) and "val_loss" in res[k]
                for v, s in zip(res[k]["val_loss"], res[k]["success"]["per_seed"])]
        rho, p = stats.spearmanr([a for a, _ in allp], [b for _, b in allp])
        res["P-E4"] = {"rho": rho, "p": p, "n": len(allp)}
        lines.append(f"P-E4 Spearman(val loss, DEV success) over {len(allp)} models: {rho:+.2f} (p {ar.fmt_p(p)})")
    lines += secondary(res, runs_of)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ar.STEP3_DIR, "step4E.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(ar.STEP3_DIR, "step4E.json"), "w") as f:
        json.dump(ar._clean(res), f, indent=1)
    figures(res)


def secondary(res, runs_of):
    """The other pre-registered quantities (per seed and two-level bootstrap CI of fine precision, held_10s and the
    failure classes), then post-hoc descriptive checks added after the audit of the E results (2026-11-01)."""
    pct = cm.pct
    lines = ["Pre-registered secondary quantities (per seed; two-level bootstrap 95% CI; with 3 seeds the bootstrap CI is"
             " close to the seed range)"]
    for k, r in res.items():
        if not (isinstance(r, dict) and "classes" in r):
            continue
        parts = [f"{m} {pct(r[m]['mean'])} [{pct(r[m]['lo'])}, {pct(r[m]['hi'])}] ("
                 + "/".join(pct(x) for x in r[m]["per_seed"]) + ")" for m in ("fine", "held_10s", "reach_5cm")]
        lines.append(f"  {r['label']:10s} " + "; ".join(parts))
        lines.append("             classes: " + "; ".join(
            f"{c} {pct(v)} [{pct(r['class_ci'][c]['lo'])}, {pct(r['class_ci'][c]['hi'])}] ("
            + "/".join(pct(x) for x in r["class_ci"][c]["per_seed"]) + ")"
            for c, v in sorted(r["classes"].items(), key=lambda kv: -kv[1])))
    lines.append("Post hoc (descriptive, not pre-registered; added after the audit of the E results, 2026-11-01)")
    post = res.setdefault("post_hoc", {})

    def both(a, b, metric):
        eps = sorted(runs_of[a][0].by_episode)
        if metric == "fine":
            eps = dm.reach_required(runs_of[a], eps)
            A, B = dm.fine_matrix(runs_of[a], eps), dm.fine_matrix(runs_of[b], eps)
        elif metric == "crash":
            A = (ar.class_matrix(runs_of[a], eps) == "crash").astype(float)
            B = (ar.class_matrix(runs_of[b], eps) == "crash").astype(float)
        else:
            A, B = ar.metric_matrix(runs_of[a], eps, metric), ar.metric_matrix(runs_of[b], eps, metric)
        d, t = cm.diff_both(A, B, ar.Bootstrap([len(SEEDS), len(SEEDS)], len(eps), ar.N_BOOT, ar.BOOT_SEED))
        post[f"{b} - {a} {metric}"] = {"diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "welch_lo": t["lo"],
                                       "welch_hi": t["hi"]}
        return f"{res[b]['label']} - {res[a]['label']} {metric}: {dm.fmt(d, t)}"
    for a, b, metric in (("f0125", "f1", "success"), ("f0125", "s81", "fine"), ("f0125", "s81", "reach_5cm"),
                         ("f05", "f1", "fine"), ("f05", "f1", "held_10s"), ("f1", "s27", "crash"),
                         ("s27", "s81", "success")):
        if a in runs_of and b in runs_of:
            lines.append("  " + both(a, b, metric))
    # where the success gain along the learning curve comes from: change of each failure class share
    for a, b in (("f0125", "f1"), ("f05", "f1")):
        if a in res and b in res:
            cl = sorted(set(res[a]["classes"]) | set(res[b]["classes"]))
            dd = {c: res[b]["classes"].get(c, 0.0) - res[a]["classes"].get(c, 0.0) for c in cl}
            post[f"classes {b} - {a}"] = dd
            lines.append(f"  class shares {res[b]['label']} - {res[a]['label']}: " + ", ".join(
                f"{c} {100 * v:+.1f}" for c, v in sorted(dd.items(), key=lambda kv: kv[1]) if abs(v) >= 0.005) + " pp")
    # holding and finishing once near the target
    hold, fin = [], []
    for k in ("f0125", "f025", "f05", "f1", "s27", "s81"):
        if k not in runs_of:
            continue
        eps = sorted(runs_of[k][0].by_episode)
        S = ar.metric_matrix(runs_of[k], eps, "success")
        R5 = ar.metric_matrix(runs_of[k], eps, "reach_5cm")
        H = ar.metric_matrix(runs_of[k], eps, "held_10s")
        post[f"{k} held/success"] = float(H.sum() / max(S.sum(), 1))
        post[f"{k} success|reach_5cm"] = float((S * R5).sum() / max(R5.sum(), 1))
        hold.append(f"{res[k]['label']} {pct(post[f'{k} held/success'])}")
        fin.append(f"{res[k]['label']} {pct(post[f'{k} success|reach_5cm'])}")
    lines.append("  held_10s / success (pooled): " + ", ".join(hold))
    lines.append("  success | reach_5cm (pooled): " + ", ".join(fin))
    # P-E1 robustness: leave one setting out
    lc = [k for k in ("f0125", "f025", "f05", "f1") if k in res]
    if len(lc) == 4:
        out = []
        for drop in lc:
            keep = [k for k in lc if k != drop]
            x = np.concatenate([[np.log2(res[k]["n_episodes"])] * len(SEEDS) for k in keep])
            y = np.concatenate([res[k]["success"]["per_seed"] for k in keep])
            fit = stats.linregress(x, y)
            q = stats.t.ppf(0.975, len(x) - 2)
            post[f"P-E1 without {drop}"] = [fit.slope, fit.slope - q * fit.stderr, fit.slope + q * fit.stderr]
            out.append(f"without {res[drop]['label']} {100 * fit.slope:+.1f} [{100 * (fit.slope - q * fit.stderr):+.1f}, "
                       f"{100 * (fit.slope + q * fit.stderr):+.1f}]")
        lines.append("  P-E1 slope, leave one setting out (pp per doubling, t): " + "; ".join(out))
    # P-E4 split by group
    for name, keys in (("learning curve f 1/8..1", ("f0125", "f025", "f05", "f1")),
                       ("1613 episodes, s 9 / 27 / 81", ("f1", "s27", "s81"))):
        keys = [k for k in keys if k in res]
        v = np.concatenate([res[k]["val_loss"] for k in keys])
        s = np.concatenate([res[k]["success"]["per_seed"] for k in keys])
        rho, p = stats.spearmanr(v, s)
        post[f"spearman {name}"] = [rho, p, len(v)]
        lines.append(f"  Spearman(val loss, success), {name}: {rho:+.2f} (p {ar.fmt_p(p)}, {len(v)} models)")
    lines.append("  within a setting (3 seeds): " + ", ".join(
        f"{res[k]['label']} {stats.spearmanr(res[k]['val_loss'], res[k]['success']['per_seed'])[0]:+.1f}"
        for k in ("f0125", "f025", "f05", "f1", "s27", "s81") if k in res))
    return lines


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    # the equal-rows partner of each panel is drawn hollow: s 81 (1613 episodes, 196k rows) on the learning curve,
    # f 1/8 (201 episodes, 217k rows) on the stride panel (P-E2)
    for name, keys, extra, xkey, xlabel in (
            ("fig_step4_learning_curve", ("f0125", "f025", "f05", "f1"), "s81", "n_episodes", "training episodes (log)"),
            ("fig_step4_stride", ("s81", "s27", "f1"), "f0125", "n_rows", "training rows, stride 81 / 27 / 9 (log)")):
        keys = [k for k in keys if k in res]
        if len(keys) < 2:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
        for k in keys + ([extra] if extra in res else []):
            r, x = res[k], res[k][xkey]
            hollow = k == extra
            # keep the partner off a solid point at the same x (s 81 on the learning curve shares f 1's 1613 episodes)
            dx = 1.06 if hollow and x in {res[j][xkey] for j in keys} else 1.0
            col = "#E45756" if hollow else "#4C78A8"
            a1.scatter([x * dx] * len(SEEDS), 100 * np.asarray(r["success"]["per_seed"]), color="#9D9D9D", s=12)
            a1.errorbar(x * dx, 100 * r["success"]["mean"], yerr=[[100 * (r["success"]["mean"] - r["success"]["lo"])],
                                                                  [100 * (r["success"]["hi"] - r["success"]["mean"])]],
                        fmt="o", color=col, mfc="white" if hollow else col, capsize=3)
            a2.scatter([x * dx] * len(SEEDS), r["val_loss"], color=col, facecolors="white" if hollow else col, s=14)
            label = f"{r['label']}\n({r['n_episodes']} ep, {r['n_rows'] / 1000:.0f}k rows)" if hollow else r["label"]
            a1.annotate(label, (x * dx, 100 * r["success"]["mean"]), fontsize=7, xytext=(5, -3),
                        textcoords="offset points", color=col if hollow else "black", va="top" if hollow else "baseline")
        for ax in (a1, a2):
            ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=8)
            xs = sorted({res[k][xkey] for k in keys})
            ax.xaxis.set_major_locator(FixedLocator(xs))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([f"{x / 1000:.0f}k" if x >= 1e5 else str(x) for x in xs], fontsize=7)
            ax.set_xlim(xs[0] / 1.4, xs[-1] * 1.6)
        a1.set_ylabel("DEV success (%)")
        a2.set_ylabel("validation loss (standardised MSE)")
        a1.set_title("seed mean, 2-level bootstrap 95% CI; grey: seeds", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(ar.STEP3_DIR, f"{name}.{ext}"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
