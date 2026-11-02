"""Experiment D, post-hoc descriptive checks (not pre-registered; they do not change the decision of D_mechanisms.py).
Written 2026-11-01 after the D-MM DEV verdict (KEEP C3), for item 4 of report.txt "[Results] 第 4 步：实验 D 结果".

  python results/step3/D_posthoc.py   -> results/step3/D_posthoc.txt
DEV, C3 ego-novw-trig s0-9 (stalled rows replaced by the no-early-stop re-runs, as in D_mechanisms) against
D ego-novw-trig-ee s0-9: success and failure classes by target height, per-seed spread, and the relation between the
early-stopping epoch (and validation loss) and closed-loop success.
"""
import json
import os
import statistics as st
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import D_mechanisms as dm  # noqa: E402

cm, ar = dm.cm, dm.ar
SEEDS = list(range(10))


def main():
    A = cm.load([ar.run_dir_for(ar.RESULTS_DIR, dm.OLD, s, "dev") for s in SEEDS])
    A, _ = dm.merged_c3(A)
    B = cm.load([ar.run_dir_for(ar.RESULTS_DIR, dm.NEW, s, "dev") for s in SEEDS])
    dm.require_no_early_stop(B, dm.NEW)
    eps = sorted(A[0].by_episode)
    z = np.array([ar._f(A[0].by_episode[e], "target_z") for e in eps])
    masks = (("z >= %.2f" % ar.ad.Z_SPLIT, z >= ar.ad.Z_SPLIT), ("z < %.2f" % ar.ad.Z_SPLIT, z < ar.ad.Z_SPLIT))
    lines = ["D post hoc (descriptive, not pre-registered; DEV, C3 = ego-novw-trig merged rows, D = ego-novw-trig-ee)"]
    rates = {}
    for name, runs, model in (("C3", A, dm.OLD), ("D", B, dm.NEW)):
        S = ar.metric_matrix(runs, eps, "success")
        cls = ar.class_matrix(runs, eps)
        per = list(100 * S.mean(axis=1))
        rates[name] = per
        metas = [json.loads(str(np.load(os.path.join(ar.TRAINED_DIR, f"{model}_s{s}.npz"))["meta"])) for s in SEEDS]
        best = [m["best_epoch"] for m in metas]
        val = [m["val_loss"] for m in metas]
        lines.append(f"== {name}: success per seed " + " / ".join(f"{x:.1f}" for x in per)
                     + f"; mean {st.mean(per):.1f}, sd {st.stdev(per):.1f}, median {st.median(per):.1f} pp")
        for lab, mk in masks:
            c = {k: 100 * float(np.mean(cls[:, mk] == k)) for k in ar.CLASSES if np.any(cls[:, mk] == k)}
            lines.append(f"  {lab} ({int(mk.sum())} episodes): " + ", ".join(
                f"{k} {v:.1f}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
        for s in SEEDS:
            c = {k: int(np.sum(cls[s] == k)) for k in ("circling", "crash", "extended_not_reached")}
            lines.append(f"  s{s}: success {per[s]:.1f}%, best epoch {best[s]} of {metas[s]['epochs_run']}, "
                         f"val loss {val[s]:.4f}, circling {c['circling']}, crash {c['crash']}, ENR {c['extended_not_reached']}")
        r1, r2 = stats.spearmanr(best, per), stats.spearmanr(val, per)
        lines.append(f"  Spearman(best epoch, success) {r1.correlation:+.2f} (p {ar.fmt_p(r1.pvalue)}); "
                     f"Spearman(val loss, success) {r2.correlation:+.2f} (p {ar.fmt_p(r2.pvalue)})")
    lines.append(f"D seeds above the C3 mean ({st.mean(rates['C3']):.1f}%): "
                 f"{sum(x > st.mean(rates['C3']) for x in rates['D'])} of {len(SEEDS)}; C3 seeds above their own mean: "
                 f"{sum(x > st.mean(rates['C3']) for x in rates['C3'])}")
    lines += audit_checks(A, B, eps, z, rates)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "D_posthoc.txt"), "w") as f:
        f.write(text + "\n")


def welch_power(s1, s2, n, delta):
    """Welch half-width, 80%-power detectable difference and the power for delta (two-sided 5%, noncentral t)."""
    se = np.sqrt(s1 ** 2 / n + s2 ** 2 / n)
    df = se ** 4 / ((s1 ** 2 / n) ** 2 / (n - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    q = stats.t.ppf(0.975, df)

    def power(d):
        return stats.nct.sf(q, df, d / se) + stats.nct.cdf(-q, df, d / se)
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power(mid) < 0.8 else (lo, mid)
    return q * se, hi, power(delta)


def audit_checks(A, B, eps, z, rates):
    """Checks added after the audit of the D results (2026-11-01): robustness of the post-hoc readings."""
    out = ["Checks added after the audit of the D results (post hoc, descriptive)"]
    seeds = np.array(SEEDS)
    metas = [json.loads(str(np.load(os.path.join(ar.TRAINED_DIR, f"{dm.NEW}_s{s}.npz"))["meta"])) for s in SEEDS]
    best = np.array([m["best_epoch"] for m in metas])
    ran = np.array([m["epochs_run"] for m in metas])
    d = np.array(rates["D"])
    loo = [stats.spearmanr(np.delete(best, k), np.delete(d, k)).correlation for k in SEEDS]
    keep = ~np.isin(seeds, [2, 3])
    out.append(f"  D Spearman(best epoch, success): leave-one-out {min(loo):+.2f} to {max(loo):+.2f}; Pearson "
               f"{stats.pearsonr(best, d)[0]:+.2f}; with epochs run {stats.spearmanr(ran, d).correlation:+.2f}; without "
               f"s2, s3 {stats.spearmanr(best[keep], d[keep]).correlation:+.2f}")
    mc = [json.loads(str(np.load(os.path.join(ar.TRAINED_DIR, f"{dm.OLD}_s{s}.npz"))["meta"]))["best_epoch"] for s in SEEDS]
    c = np.array(rates["C3"])
    loo_c = [stats.spearmanr(np.delete(mc, k), np.delete(c, k)).correlation for k in SEEDS]
    out.append(f"  C3 Spearman(best epoch, success) leave-one-out {min(loo_c):+.2f} to {max(loo_c):+.2f}")
    # base behaviour per seed against the early-stopping epoch
    for name, runs, be in (("D", B, best), ("C3", A, np.array(mc))):
        turn = [np.median([ar._f(r.by_episode[e], "base_turn_total_deg") for e in eps]) for r in runs]
        parked = [np.mean([str(r.by_episode[e].get("parked")) == "True" for e in eps]) for r in runs]
        cls = ar.class_matrix(runs, eps)
        circ = (cls == "circling").sum(axis=1)
        enr = (cls == "extended_not_reached").sum(axis=1)
        crash = (cls == "crash").sum(axis=1)
        k = keep if name == "D" else np.ones(len(SEEDS), dtype=bool)

        def rho(x, mask=None):
            mask = np.ones(len(SEEDS), dtype=bool) if mask is None else mask
            return stats.spearmanr(be[mask], np.asarray(x)[mask]).correlation
        out.append(f"  {name}: Spearman of best epoch with median base turn {rho(turn):+.2f}, parked share {rho(parked):+.2f}, "
                   f"circling count {rho(circ):+.2f} (without s2, s3 {rho(circ, k):+.2f}), ENR count {rho(enr):+.2f} "
                   f"(without s2, s3 {rho(enr, k):+.2f}), crash count {rho(crash):+.2f}")
    # spread
    out.append(f"  D sd without s2, s3: {st.stdev(d[keep]):.1f} pp (mean {st.mean(d[keep]):.1f}); C3 sd {st.stdev(c):.1f}; "
               f"Brown-Forsythe p {stats.levene(d, c, center='median').pvalue:.2f}; F-test p "
               f"{2 * min(stats.f.sf(np.var(d, ddof=1) / np.var(c, ddof=1), 9, 9), stats.f.cdf(np.var(d, ddof=1) / np.var(c, ddof=1), 9, 9)):.3f}; "
               f"Mann-Whitney p {stats.mannwhitneyu(d, c, alternative='two-sided').pvalue:.3f}")
    hw0, det0, pw0 = welch_power(6.7, 6.7, 10, 7.5)
    hw1, det1, pw1 = welch_power(st.stdev(c), st.stdev(d), 10, 7.5)
    out.append(f"  Welch, 10 vs 10 seeds: planned (sd 6.7 / 6.7) half-width {hw0:.1f}, 80%-power detectable {det0:.1f} pp, "
               f"power for +7.5 pp {pw0:.2f}; realised (sd {st.stdev(c):.1f} / {st.stdev(d):.1f}) half-width {hw1:.1f}, "
               f"detectable {det1:.1f} pp, power for +7.5 pp {pw1:.2f}")
    # target height: dependence on D seeds 2 and 3
    hi_m, lo_m = z >= ar.ad.Z_SPLIT, z < ar.ad.Z_SPLIT
    SA, SB = ar.metric_matrix(A, eps, "success"), ar.metric_matrix(B, eps, "success")
    CA, CB = ar.class_matrix(A, eps), ar.class_matrix(B, eps)
    for lab, drop in (("all D seeds", []), ("without D s2", [2]), ("without D s2, s3", [2, 3])):
        kb = ~np.isin(seeds, drop)
        dh = 100 * (SB[kb][:, hi_m].mean() - SA[:, hi_m].mean())
        dl = 100 * (SB[kb][:, lo_m].mean() - SA[:, lo_m].mean())
        ch = 100 * (np.mean(CB[kb][:, hi_m] == "circling") - np.mean(CA[:, hi_m] == "circling"))
        out.append(f"  {lab}: high-z success D {100 * SB[kb][:, hi_m].mean():.1f} vs C3 {100 * SA[:, hi_m].mean():.1f} "
                   f"({dh:+.1f}), low z {dl:+.1f}; high-z circling D {100 * np.mean(CB[kb][:, hi_m] == 'circling'):.1f} vs C3 "
                   f"{100 * np.mean(CA[:, hi_m] == 'circling'):.1f} ({ch:+.1f}); share of the net gain from high z "
                   f"{dh * hi_m.mean() / (dh * hi_m.mean() + dl * lo_m.mean()):.0%}")
    out.append("  high-z circling per D seed: " + " / ".join(f"{100 * np.mean(CB[s][hi_m] == 'circling'):.1f}" for s in SEEDS)
               + f" %; C3 pooled {100 * np.mean(CA[:, hi_m] == 'circling'):.1f}%")

    def near(runs, mk):   # failed with the minimum error in [1, 5) cm, any failure class
        v = [[(str(r.by_episode[e].get("success")) != "True" and 0.01 <= ar._f(r.by_episode[e], "min_error") < 0.05)
              for e in np.array(eps)[mk]] for r in runs]
        return 100 * np.mean(v)
    out.append(f"  failed with min error in [1, 5) cm (any class): high z C3 {near(A, hi_m):.1f} -> D {near(B, hi_m):.1f}; "
               f"low z C3 {near(A, lo_m):.1f} -> D {near(B, lo_m):.1f}")
    # the expert-base diagnostic per seed
    em = cm.load(dm.run_dirs(dm.NEW, "dev", "EM", dm.EM_SEEDS))
    em_eps = sorted(em[0].by_episode)
    S = ar.metric_matrix(em, em_eps, "success")
    E = (ar.class_matrix(em, em_eps) == "extended_not_reached").sum(axis=1)
    out.append("  D-EM (expert base) per seed: success " + " / ".join(f"{100 * x:.1f}" for x in S.mean(axis=1))
               + "; ENR " + " / ".join(str(int(x)) for x in E))
    return out


if __name__ == "__main__":
    main()
