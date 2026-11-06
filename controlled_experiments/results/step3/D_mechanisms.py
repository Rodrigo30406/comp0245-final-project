"""Experiment D (near-field end-effector error features): the pre-registered checks, verdicts, adoption conditions and
the next step of the decision table (report.txt "[Methodology] Step 4 pre-registration: experiment D", frozen before any D run;
STEP4_PLAN_PROMPT.md 2.3-2.5 and 6). Each part is computed when its runs exist; the rest is listed as missing.

  python results/step3/D_mechanisms.py [--allow-confirm] [--out STEM]

Arms (A = C3 = ego-novw-trig, B = D = ego-novw-trig-ee):
  MM   results/<arm>_s<k>_<set>: DEV and HARD seeds 0-9, CONFIRM seeds 100-104. The D runs and the C3 runs of seeds
       5-9 are made with --no-early-stop. The older C3 runs (DEV / HARD seeds 0-4, CONFIRM 100-104) were made with the
       early stop: their stalled episodes are replaced by the --no-early-stop re-runs in results/step3/nostall/<run>
       (every other episode is step-identical without the early stop), so both arms are compared without the early
       stop (the C3 family failed the stall check, EVALUATION.md). Sensitivity: the C3 early-stop rows.
  EM   results/B-EM_<arm>_s<k>_dev, seeds 0-4 (expert base, model arm, --no-early-stop).
Statistics as in step 3 / C3: seed means; two-level bootstrap (10000 draws, rng seed 0) and Welch on the per-seed rates,
both required; the arms are different networks (seeds not paired).
Definitions:
  reach-required episodes: the expert reference navdone_ee_err >= 0.07 m (same episodes in both arms);
  precision pool: failed, arm extended toward the target, base parked (parked_time set), min_error in [1, 5) cm, any class;
  fine precision: min_error < 15 mm, on the reach-required episodes.
Output: results/step3/D_mechanisms.txt / .json and the full comparisons D_MM_compare_<set>.txt, D_EM_compare_dev.txt.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C3_mechanisms as cm  # noqa: E402  (frozen C3 helpers: pct, pp, one_arm_t, diff_both, diff_line, ...)

ar = cm.ar
TASK = cm.TASK
RES = ar.RESULTS_DIR
NOSTALL = os.path.join(ar.STEP3_DIR, "nostall")
OLD, NEW = "ego-novw-trig", "ego-novw-trig-ee"
SEEDS = {"dev": list(range(10)), "hard": list(range(10)), "confirm": [100, 101, 102, 103, 104]}
EM_SEEDS = [0, 1, 2, 3, 4]
REACH_REQUIRED = 0.07          # m, reference navdone_ee_err of a reach-required episode
POOL_LO, POOL_HI = 0.01, 0.05  # m, precision pool min_error range
FINE = 0.015                   # m, fine precision threshold
TRIGGER_MAX_DROP = -0.05       # trigger check (non-inferiority): both CI lower bounds of D-EM - C3-EM arm_extended_toward >= this
TRIGGER_MIN_SEED = 0.70        # ... and every D-EM seed's arm_extended_toward >= this
LABEL_EQUIV = 0.07             # H-label: both 95% CIs of the low-z difference within +-this (user 2026-10-31)
COVERAGE_MAX = 0.03            # H-coverage: point estimate of the precision-pool reduction below this (user 2026-10-31)
FROZEN_INPUTS = ("results/step3/cache/ee/manifest.json", "results/step3/cache/trigger/manifest.json")
SHORTCUT_POOLED, SHORTCUT_SEED, SHORTCUT_SEEDS, SHORTCUT_MIN = 0.10, 0.05, 3, 30
PREMATURE_RING = 0.20
STRATUM_FLOOR = -0.10


def fmt(d, t):
    return f"bootstrap {cm.pp(d['diff'], d['lo'], d['hi'])}, Welch {cm.pp(t['diff'], t['lo'], t['hi'])}"


def run_dirs(model, set_name, cell=None, seeds=None):
    return [ar.run_dir_for(RES, model, s, set_name, cell=cell) for s in (seeds or SEEDS[set_name])]


def require_no_early_stop(runs, what):
    bad = [r.name for r in runs if r.meta.get("no_early_stop") is not True]
    if bad:
        raise SystemExit(f"{what}: runs made with the early stop, expected --no-early-stop: {bad}")
    err = [r.name for r in runs if any(row.get("stop_reason") == "policy_error" for row in r.rows)]
    if err:
        raise SystemExit(f"{what}: runs with policy_error episodes (e.g. a refused reset without the robot model), "
                         f"not policy failures: {err}")


def merged_c3(runs):
    """C3 runs without the early stop: rows of stalled episodes replaced by their --no-early-stop re-runs."""
    out, notes = [], []
    for run in runs:
        stalled = [row["episode"] for row in run.rows if row.get("stop_reason") == "stalled"]
        if run.meta.get("no_early_stop") is True or not stalled:
            out.append(run)
            notes.append(f"{run.name}: as run ({'no early stop' if run.meta.get('no_early_stop') else 'no stalled episode'})")
            continue
        p = os.path.join(NOSTALL, run.name)
        if not os.path.exists(os.path.join(p, "rollouts.csv")):
            raise SystemExit(f"{run.name}: {len(stalled)} stalled episodes but no re-run folder {p}")
        rr = cm.load([p])[0]
        if sorted(r["episode"] for r in rr.rows) != sorted(stalled):
            raise SystemExit(f"{p}: re-run episodes differ from the stalled episodes of {run.name}")
        if rr.meta.get("no_early_stop") is not True:
            raise SystemExit(f"{p}: not a --no-early-stop re-run")
        by = {r["episode"]: r for r in rr.rows}
        rows = [by.get(row["episode"], row) for row in run.rows]
        new = ar.Run(run.path, rows, run.header, run.config, dict(run.meta))
        new.classified = getattr(run, "classified", False)
        out.append(new)
        notes.append(f"{run.name}: {len(stalled)} stalled rows replaced by the re-runs "
                     f"({sum(ar.metric_value(r, 'success') for r in rr.rows):.0f} of them succeed)")
    return out, notes


def pool_fn(row):
    me = ar._f(row, "min_error")
    return (ar.metric_value(row, "success") == 0 and ar.metric_value(row, "arm_extended_toward") == 1
            and np.isfinite(ar._f(row, "parked_time")) and POOL_LO <= me < POOL_HI)


def reach_required(runs, eps):
    return [e for e in eps if ar._f(runs[0].by_episode[e], "ref_navdone_ee_err") >= REACH_REQUIRED]


def fine_matrix(runs, eps):
    return cm.indicator(runs, eps, lambda row: ar._f(row, "min_error") < FINE)


def model_meta_check(lines, out):
    """D and the new C3 models: trigger settings and the code hashes of the D freeze record."""
    frozen = os.path.join(ar.STEP3_DIR, "prereg_D_frozen.txt")
    code = {}
    if os.path.exists(frozen):
        for line in open(frozen):
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                code[parts[1]] = parts[0]
    rows, ok = [], bool(code)
    for model, seeds in ((NEW, SEEDS["dev"] + SEEDS["confirm"]), (OLD, [5, 6, 7, 8, 9])):
        for s in seeds:
            f = os.path.join(ar.TRAINED_DIR, f"{model}_s{s}.npz")
            if not os.path.exists(f):
                continue
            with np.load(f) as z:
                m = json.loads(str(z["meta"]))
            t = m.get("trigger") or {}
            good = (t.get("copies") == 16 and t.get("qd_jitter") == "auto" and t.get("jitter_seed") == 0
                    and m["feature_set"] == ("ego_novw_ee" if model == NEW else "ego_novw")
                    and all(m["code_sha256"].get(k) == code.get(k) for k in ("train_policy.py", "bc_features.py"))
                    and t.get("trigger_cache_manifest_sha256") == code.get(FROZEN_INPUTS[1])
                    and (model != NEW or m.get("ee_cache_manifest_sha256") == code.get(FROZEN_INPUTS[0])))
            ok &= good
            rows.append(f"{model}_s{s} {'ok' if good else 'MISMATCH'}")
    lines.append(f"models checked against the D freeze record ({'found' if code else 'MISSING'}): {len(rows)}; "
                 + ("all ok" if ok and rows else "; ".join(r for r in rows if "MISMATCH" in r) or "none"))
    out["models"] = {"n": len(rows), "ok": bool(ok and rows)}


# ==============================================================================
# D-EM (trigger check and the precision with a perfect base)
# ==============================================================================
def em_part(lines, out):
    a_p, b_p = run_dirs(OLD, "dev", "EM", EM_SEEDS), run_dirs(NEW, "dev", "EM", EM_SEEDS)
    lines.append("== D-EM (DEV, expert base, model arm, --no-early-stop): A = C3-EM, B = D-EM, seeds 0-4 ==")
    if not cm.have(b_p):
        lines.append("  missing: " + ", ".join(os.path.relpath(p, TASK) for p in b_p if not cm.have([p])))
        out["em"] = None
        return
    A, B = cm.load(a_p), cm.load(b_p)
    require_no_early_stop(A + B, "D-EM / C3-EM")
    ar.check_integrity(A + B, "D-EM vs C3-EM")
    eps, _ = ar.paired_episodes(A + B)
    boot = ar.Bootstrap([len(A), len(B)], len(eps), ar.N_BOOT, ar.BOOT_SEED)
    MA, MB = ar.metric_matrix(A, eps, "arm_extended_toward"), ar.metric_matrix(B, eps, "arm_extended_toward")
    d, t = cm.diff_both(MA, MB, boot)
    per_seed = ar.seed_rates(MB)
    ok = bool(d["lo"] >= TRIGGER_MAX_DROP and t["lo"] >= TRIGGER_MAX_DROP and per_seed.min() >= TRIGGER_MIN_SEED)
    lines.append(f"  trigger check: arm_extended_toward A {cm.pct(d['mean_a'])}%, B {cm.pct(d['mean_b'])}% (per seed B "
                 + "/".join(cm.pct(x) for x in per_seed) + f"%); B - A {fmt(d, t)} -> "
                 + ("PASS" if ok else "FAIL: D changes the trigger (D-MM descriptive only)")
                 + f" (non-inferiority: both CI lower bounds >= {100 * TRIGGER_MAX_DROP:+.0f} pp and every seed >= "
                   f"{cm.pct(TRIGGER_MIN_SEED, 0)}%)")
    res = {"trigger_ok": ok, "runs_a": [r.name for r in A], "runs_b": [r.name for r in B], "arm_extended_toward": {"a": d["mean_a"], "b": d["mean_b"], "diff": d["diff"], "lo": d["lo"],
                                                     "hi": d["hi"], "welch_lo": t["lo"], "welch_hi": t["hi"],
                                                     "per_seed_b": per_seed}}
    rr = reach_required(A, eps)
    boot_r = ar.Bootstrap([len(A), len(B)], len(rr), ar.N_BOOT, ar.BOOT_SEED)
    lines.append(f"  reach-required episodes (expert navdone_ee_err >= {REACH_REQUIRED} m): {len(rr)} (descriptive):")
    for name, fn in (("success", lambda runs, e: ar.metric_matrix(runs, e, "success")),
                     ("precision pool", lambda runs, e: cm.indicator(runs, e, pool_fn)),
                     ("fine precision (< 15 mm)", fine_matrix)):
        dd, tt = cm.diff_both(fn(A, rr), fn(B, rr), boot_r)
        res[name] = {"a": dd["mean_a"], "b": dd["mean_b"], "diff": dd["diff"], "lo": dd["lo"], "hi": dd["hi"],
                     "welch_lo": tt["lo"], "welch_hi": tt["hi"]}
        lines.append(f"    {name}: A {cm.pct(dd['mean_a'])}%, B {cm.pct(dd['mean_b'])}%, B - A {fmt(dd, tt)}")
    for label, runs in (("A", A), ("B", B)):
        tow = [(r, e) for r in runs for e in eps if ar.metric_value(r.by_episode[e], "arm_extended_toward") == 1]
        k = sum(ar.metric_value(r.by_episode[e], "success") == 1 for r, e in tow)
        lines.append(f"    success among extended-toward cells {label}: {k}/{len(tow)}"
                     + (f" = {cm.pct(k / len(tow))}%" if tow else "") + " (conditioned on extension; descriptive, "
                                                                          "not pre-registered as a test)")
    e_lines, e_res = ar.compare_arms(A, B, "C3-EM", "D-EM", ("extended_not_reached",),
                                     expert_run=ar.expert_run_for(RES, "dev"))
    ar.write_outputs(ar.STEP3_DIR, "D_EM_compare_dev", e_lines, e_res)
    out["em"] = res


# ==============================================================================
# D-MM
# ==============================================================================
def share(A, B, eps, fn, metric, boot, d):
    """Condition (2): [A - B share of cells failing the metric and in fn] / net difference (B - A)."""
    f = lambda row: ar.metric_value(row, metric) == 0 and fn(row)  # noqa: E731
    IA, IB = cm.indicator(A, eps, f), cm.indicator(B, eps, f)
    dc = ar.paired_diff(IA, IB, boot=boot)
    red = -dc["diff"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = red / d["diff"] if d["diff"] != 0 else np.nan
        lo, hi = ar.percentile_ci(-dc["draws"] / d["draws"])
    return {"a": IA.mean(), "b": IB.mean(), "reduction": red, "share": sh, "lo": lo, "hi": hi}


def mm_part(set_name, lines, out, dev_metric=None):
    a_p, b_p = run_dirs(OLD, set_name), run_dirs(NEW, set_name)
    lines.append(f"== D-MM ({set_name.upper()}): A = C3 ego-novw-trig (no early stop), B = D ego-novw-trig-ee, seeds "
                 f"{SEEDS[set_name]} ==")
    if not cm.have(a_p) or not cm.have(b_p):
        lines.append("  missing: " + ", ".join(os.path.relpath(p, TASK) for p in a_p + b_p if not cm.have([p])))
        out[f"mm_{set_name}"] = None
        return None
    A_raw, B = cm.load(a_p), cm.load(b_p)
    require_no_early_stop(B, "D-MM")
    A, notes = merged_c3(A_raw)
    lines += ["  C3 rows: " + "; ".join(notes)]
    ar.check_integrity(A + B, f"D-MM vs C3 ({set_name})")
    expert = ar.expert_run_for(RES, set_name)
    c_lines, c_res = ar.compare_arms(A, B, f"C3 {set_name}", f"D {set_name}", ("extended_not_reached",), expert_run=expert)
    ar.write_outputs(ar.STEP3_DIR, f"D_MM_compare_{set_name}", c_lines, c_res)
    eps, _ = ar.paired_episodes(A + B)
    boot = ar.Bootstrap([len(A), len(B)], len(eps), ar.N_BOOT, ar.BOOT_SEED)
    YA, YB = ar.metric_matrix(A, eps, "success"), ar.metric_matrix(B, eps, "success")
    M5A, M5B = ar.metric_matrix(A, eps, "reach_5cm"), ar.metric_matrix(B, eps, "reach_5cm")
    metric = cm.decision_metric(YA, YB, M5A, M5B) if set_name == "dev" else (dev_metric or "success")
    MA, MB = ar.metric_matrix(A, eps, metric), ar.metric_matrix(B, eps, metric)
    d, t = cm.diff_both(MA, MB, boot)
    res = {"decision_metric": metric, "diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "welch_lo": t["lo"],
           "welch_hi": t["hi"], "mean_a": d["mean_a"], "mean_b": d["mean_b"], "per_seed_a": ar.seed_rates(MA),
           "per_seed_b": ar.seed_rates(MB), "full_compare": f"results/step3/D_MM_compare_{set_name}.txt"}
    lines.append(f"  decision metric {metric}: A {cm.pct(d['mean_a'])}%  B {cm.pct(d['mean_b'])}%  B - A {fmt(d, t)}")
    lines.append("    per seed A " + "/".join(cm.pct(x) for x in ar.seed_rates(MA)) + ", B "
                 + "/".join(cm.pct(x) for x in ar.seed_rates(MB)) + "%")
    if set_name != "dev":
        res["direction"], res["reverse_direction"] = bool(d["diff"] > 0), bool(d["diff"] < 0)
        lines.append(f"  direction B - A {100 * d['diff']:+.1f} pp: {'> 0' if d['diff'] > 0 else '<= 0'} "
                     f"(condition ({'4' if set_name == 'hard' else '5'}) needs > 0)")
        out[f"mm_{set_name}"] = res
        return res
    c1, r1 = bool(d["lo"] > 0 and t["lo"] > 0), bool(d["hi"] < 0 and t["hi"] < 0)
    sh = share(A, B, eps, pool_fn, metric, boot, d)
    sh_enr = share(A, B, eps, lambda row: row["fclass"] == "extended_not_reached", metric, boot, d)
    c2 = bool(np.isfinite(sh["share"]) and sh["share"] >= 0.5 and d["diff"] > 0)
    r2 = bool(np.isfinite(sh["share"]) and sh["share"] >= 0.5 and d["diff"] < 0)
    lines.append(f"  (1) CI lower bounds: bootstrap {100 * d['lo']:+.1f}, Welch {100 * t['lo']:+.1f} pp -> "
                 f"{'MET' if c1 else 'NOT MET'}")
    lines.append(f"  (2) cells failing {metric} and in the precision pool (share of all cells): A {cm.pct(sh['a'])}%, "
                 f"B {cm.pct(sh['b'])}%, reduction "
                 f"{100 * sh['reduction']:+.1f} pp / net {100 * d['diff']:+.1f} pp = {cm.pct(sh['share'])}% [bootstrap "
                 f"{cm.pct(sh['lo'])}, {cm.pct(sh['hi'])}] -> {'MET' if c2 else 'NOT MET'}; ENR only (reported): "
                 f"{cm.pct(sh_enr['share'])}%")
    d0 = np.array([A[0].by_episode[e]["d0"] for e in eps], dtype=float)
    sm = ar.strata_analysis(MA, MB, d0)
    big = [s_ for s_ in sm["strata"] if s_["n"] >= ar.MIN_STRATUM_N]
    worst = sm["worst_hi_n30"]
    best_lo = max(big, key=lambda s_: s_["lo"]) if big else None
    c3 = bool(worst is not None and worst[1] >= STRATUM_FLOOR)
    r3 = bool(best_lo is not None and best_lo["lo"] <= -STRATUM_FLOOR)
    lines.append(f"  (3) worst n >= 30 stratum CI upper bound: " + (f"{worst[0]} {100 * worst[1]:+.1f} pp" if worst else "none")
                 + f" -> {'MET' if c3 else 'NOT MET'}")
    bnw = bool(d["lo"] > 0 and not t["lo"] > 0)
    if bnw:
        qa, qb = ar.cochran_q(MA), ar.cochran_q(MB)
        lines.append(f"  bootstrap met, Welch not: the effect appears in some seeds only; per-k B_k - A_k (descriptive, "
                     f"seeds not paired) " + "/".join(f"{100 * x:+.1f}" for x in ar.seed_rates(MB) - ar.seed_rates(MA))
                     + f" pp; Cochran Q A {qa[0]:.1f} (p {ar.fmt_p(qa[2])}), B {qb[0]:.1f} (p {ar.fmt_p(qb[2])})")
    lines.append(f"  mirrored: (1) {r1}, (2) {r2}, (3) {r3}")
    # H-precision
    PA, PB = cm.indicator(A, eps, pool_fn), cm.indicator(B, eps, pool_fn)
    dp, tp_ = cm.diff_both(PA, PB, boot)
    rr = reach_required(A, eps)
    boot_r = ar.Bootstrap([len(A), len(B)], len(rr), ar.N_BOOT, ar.BOOT_SEED)
    df, tf = cm.diff_both(fine_matrix(A, rr), fine_matrix(B, rr), boot_r)
    h_prec = bool(dp["hi"] < 0 and tp_["hi"] < 0 and df["lo"] > 0 and tf["lo"] > 0)
    lines.append(f"  H-precision: precision pool A {cm.pct(dp['mean_a'])}%, B {cm.pct(dp['mean_b'])}%, B - A {fmt(dp, tp_)}; "
                 f"fine precision ({len(rr)} reach-required episodes) A {cm.pct(df['mean_a'])}%, B {cm.pct(df['mean_b'])}%, "
                 f"B - A {fmt(df, tf)} -> {'HOLDS' if h_prec else 'does not hold'}")
    # H-label: interaction by target z
    z = np.array([ar._f(A[0].by_episode[e], "target_z") for e in eps])
    hi_m, lo_m = z >= ar.ad.Z_SPLIT, z < ar.ad.Z_SPLIT
    def stratum(Y, arm, m):   # per draw: resampled seeds, then the stratum's mean over the jointly resampled episodes
        v = boot.ws[arm] @ np.asarray(Y, dtype=float)
        w = boot.we[:, m]
        return np.sum(v[:, m] * w, axis=1) / np.maximum(w.sum(axis=1), 1e-12)

    def stratum_diff(m):
        draws = stratum(MB, 1, m) - stratum(MA, 0, m)
        lo_, hi_ = ar.percentile_ci(draws)
        return {"diff": float(MB[:, m].mean() - MA[:, m].mean()), "lo": lo_, "hi": hi_, "draws": draws}
    dh, dl = stratum_diff(hi_m), stratum_diff(lo_m)
    th = ar.seed_t_test(ar.seed_rates(MA[:, hi_m]), ar.seed_rates(MB[:, hi_m]))
    tl = ar.seed_t_test(ar.seed_rates(MA[:, lo_m]), ar.seed_rates(MB[:, lo_m]))
    ilo, ihi = ar.percentile_ci(dh["draws"] - dl["draws"])
    it = ar.seed_t_test(ar.seed_rates(MA[:, hi_m]) - ar.seed_rates(MA[:, lo_m]),
                        ar.seed_rates(MB[:, hi_m]) - ar.seed_rates(MB[:, lo_m]))
    high_sig = bool(dh["lo"] > 0 and th["lo"] > 0)
    low_equiv = bool(min(dl["lo"], tl["lo"]) >= -LABEL_EQUIV and max(dl["hi"], tl["hi"]) <= LABEL_EQUIV)
    h_label = (bool(high_sig and low_equiv) if high_sig else None)
    lines.append(f"  H-label: z >= {ar.ad.Z_SPLIT} ({int(hi_m.sum())} episodes) {fmt(dh, th)}; z < {ar.ad.Z_SPLIT} "
                 f"({int(lo_m.sum())}) {fmt(dl, tl)}; difference high - low bootstrap [{100 * ilo:+.1f}, {100 * ihi:+.1f}] pp, "
                 f"Welch {cm.pp(it['diff'], it['lo'], it['hi'])} -> "
                 + ({True: "HOLDS", False: "does not hold"}[h_label] if h_label is not None else
                    "not decidable (the high-z difference is not significant)"))
    # H-coverage
    h_cov = bool(not h_prec and -dp["diff"] < COVERAGE_MAX)
    lines.append(f"  H-coverage: precision-pool reduction (A - B) {100 * -dp['diff']:+.1f} pp, CI bootstrap "
                 f"[{100 * -dp['hi']:+.1f}, {100 * -dp['lo']:+.1f}], Welch [{100 * -tp_['hi']:+.1f}, {100 * -tp_['lo']:+.1f}] pp "
                 f"(holds when H-precision does not and the point estimate < {100 * COVERAGE_MAX:.0f} pp: 'no reduction "
                 f"seen') -> {'HOLDS' if h_cov else 'does not hold'}")
    # H-shortcut
    ext = [(r, row) for r in B for row in (r.by_episode[e] for e in eps) if np.isfinite(ar._f(row, "arm_extend_time"))]
    far = [x for x in ext if ar._f(x[1], "ring_dist_at_arm_extend") > PREMATURE_RING]
    per_seed_far = []
    for r in B:
        rows = [r.by_episode[e] for e in eps if np.isfinite(ar._f(r.by_episode[e], "arm_extend_time"))]
        per_seed_far.append(np.mean([ar._f(x, "ring_dist_at_arm_extend") > PREMATURE_RING for x in rows]) if rows else 0.0)
    pooled = len(far) / len(ext) if ext else 0.0
    cl = lambda row: row["fclass"] in ("circling", "left_or_overshot")  # noqa: E731
    dcl, tcl = cm.diff_both(cm.indicator(A, eps, cl), cm.indicator(B, eps, cl), boot)
    h_short = bool(len(ext) >= SHORTCUT_MIN and pooled > SHORTCUT_POOLED
                   and sum(x > SHORTCUT_SEED for x in per_seed_far) >= SHORTCUT_SEEDS)
    base_hit = bool(dcl["lo"] > 0 and tcl["lo"] > 0)
    lines.append(f"  H-shortcut: extended cells {len(ext)}, ring_dist_at_arm_extend > {PREMATURE_RING} m pooled "
                 f"{cm.pct(pooled)}% (per seed " + "/".join(cm.pct(x) for x in per_seed_far) + "%) -> "
                 f"{'HOLDS' if h_short else 'does not hold'}; circling + left_or_overshot B - A {fmt(dcl, tcl)} -> base "
                 f"{'affected' if base_hit else 'not affected'}")
    # descriptive: classes, tipping
    for c in ("extended_not_reached", "circling", "crash", "approached_other", "left_or_overshot"):
        dc_, tc_ = cm.diff_both(cm.indicator(A, eps, lambda row, c=c: row["fclass"] == c),
                                cm.indicator(B, eps, lambda row, c=c: row["fclass"] == c), boot)
        lines.append("    " + cm.diff_line(c, dc_, tc_))
    # secondary family of the plan (Holm over bootstrap p and over max(bootstrap p, Welch p))
    fam = {}
    for name, fa, fb, b_ in (
            ("held_10s", ar.metric_matrix(A, eps, "held_10s"), ar.metric_matrix(B, eps, "held_10s"), boot),
            ("crash (tipping)", cm.indicator(A, eps, lambda row: row["fclass"] == "crash"),
             cm.indicator(B, eps, lambda row: row["fclass"] == "crash"), boot),
            ("reach_2cm", ar.metric_matrix(A, eps, "reach_2cm"), ar.metric_matrix(B, eps, "reach_2cm"), boot),
            ("arm_extended_toward", ar.metric_matrix(A, eps, "arm_extended_toward"),
             ar.metric_matrix(B, eps, "arm_extended_toward"), boot),
            ("fine precision", fine_matrix(A, rr), fine_matrix(B, rr), boot_r)):
        d_, t_ = cm.diff_both(fa, fb, b_)
        fam[name] = (d_, t_)
    names = list(fam)
    adj_b, _ = ar.holm([fam[n][0]["p"] for n in names])
    adj_c, _ = ar.holm([ar.combined_p(fam[n][0]["p"], fam[n][1]["p"]) for n in names])
    lines.append("  secondary family (Holm, bootstrap p | max(bootstrap, Welch) p):")
    for n, a_, c_ in zip(names, adj_b, adj_c):
        lines.append(f"    {n}: {fmt(*fam[n])}; Holm {ar.fmt_p(a_)} | {ar.fmt_p(c_)}")
    res_secondary = {n: {"diff": fam[n][0]["diff"], "lo": fam[n][0]["lo"], "hi": fam[n][0]["hi"],
                         "welch_lo": fam[n][1]["lo"], "welch_hi": fam[n][1]["hi"], "p_holm": a_, "p_holm_combined": c_}
                     for n, a_, c_ in zip(names, adj_b, adj_c)}
    # sensitivity: the C3 early-stop rows
    MAr = ar.metric_matrix(A_raw, eps, metric)
    ds, ts = cm.diff_both(MAr, MB, boot)
    shs = share(A_raw, B, eps, pool_fn, metric, boot, ds)
    s1 = bool(ds["lo"] > 0 and ts["lo"] > 0)
    s2 = bool(np.isfinite(shs["share"]) and shs["share"] >= 0.5 and ds["diff"] > 0)
    lines.append(f"  sensitivity (C3 early-stop rows): {metric} B - A {fmt(ds, ts)} -> (1) {'MET' if s1 else 'NOT MET'}; "
                 f"(2) share {cm.pct(shs['share'])}% -> {'MET' if s2 else 'NOT MET'}; differs from the main analysis: "
                 f"{'YES' if (s1, s2) != (c1, c2) else 'no'} (the main analysis without the early stop decides)")
    res.update(cond1=c1, cond2=c2, cond3=c3, share=sh, share_enr=sh_enr, worst_stratum=worst, bootstrap_not_welch=bnw,
               reverse_cond1=r1, reverse_cond2=r2, reverse_cond3=r3, h_precision=h_prec, h_label=h_label,
               h_coverage=h_cov, h_shortcut=h_short, shortcut_base=base_hit, pool={"diff": dp["diff"], "lo": dp["lo"],
               "hi": dp["hi"], "welch_lo": tp_["lo"], "welch_hi": tp_["hi"]},
               fine={"diff": df["diff"], "lo": df["lo"], "hi": df["hi"], "welch_lo": tf["lo"], "welch_hi": tf["hi"]},
               z_high={"diff": dh["diff"], "lo": dh["lo"], "hi": dh["hi"], "welch_lo": th["lo"], "welch_hi": th["hi"]},
               z_low={"diff": dl["diff"], "lo": dl["lo"], "hi": dl["hi"], "welch_lo": tl["lo"], "welch_hi": tl["hi"]},
               sensitivity={"diff": ds["diff"], "lo": ds["lo"], "hi": ds["hi"], "welch_lo": ts["lo"],
                            "welch_hi": ts["hi"], "share": shs["share"], "cond1": s1, "cond2": s2,
                            "differs": (s1, s2) != (c1, c2)},
               secondary=res_secondary,
               half_width=max((d["hi"] - d["lo"]) / 2, (t["hi"] - t["lo"]) / 2))
    out[f"mm_{set_name}"] = res
    return res


def adoption(lines, out):
    em, dev, hard, conf = out.get("em"), out.get("mm_dev"), out.get("mm_hard"), out.get("mm_confirm")
    lines.append("== Adoption and next step (decision table, STEP4_PLAN_PROMPT.md 6; the 'stop' rows take precedence) ==")
    res = {}
    models = out.get("models") or {}
    if models.get("n") and not models.get("ok"):
        v = "STOP: the model check failed (trigger settings, feature set, code or cache hashes differ from the freeze)"
    elif not em:
        v = "NEXT: run D-EM (stage em)"
    elif not em["trigger_ok"]:
        v = "STOP: the trigger check failed (D changes the trigger); D-MM is descriptive only. Report."
    elif not dev:
        v = "NEXT: run D-MM on DEV (stage mm)"
    else:
        c = [em["trigger_ok"], dev["cond1"], dev["cond2"], dev["cond3"], hard["direction"] if hard else None,
             conf["direction"] if conf else None]
        res["conditions_0_to_5"] = c
        lines.append("  conditions (0)-(5): " + ", ".join(f"({i}) {x}" for i, x in enumerate(c)))
        ok_dev = all(c[:4])
        if ok_dev and dev["h_shortcut"]:
            v = ("STOP: DEV (1)-(3) met but H-shortcut also holds (the gain may come from earlier extension): discuss "
                 "with the user; HARD / CONFIRM as part of the proposal")
        elif ok_dev and hard is None:
            v = "NEXT: DEV (0)-(3) met: run HARD (stage hard)"
        elif ok_dev and not hard["direction"]:
            v = "NOT ADOPTED: HARD (4) not met; CONFIRM is not opened. Keep ego-novw-trig."
        elif ok_dev and conf is None:
            v = "NEXT: DEV (0)-(3) and HARD (4) met: run CONFIRM once (stage confirm, --allow-confirm)"
        elif ok_dev:
            v = "ADOPT D: ego-novw-trig-ee replaces ego-novw-trig" if conf["direction"] else "NOT ADOPTED: CONFIRM (5)"
        elif dev["bootstrap_not_welch"]:
            v = "KEEP C3: the effect appears in some seeds only (bootstrap yes, Welch no); not adopted; stop and propose"
        elif dev["reverse_cond1"] and dev["reverse_cond2"] and dev["reverse_cond3"]:
            v = "STOP: mirrored (1)-(3) hold (C3 better); propose, with HARD as part of the proposal"
        elif dev["h_label"]:
            v = "STOP: H-label holds (high z improves, low z unchanged): report the split and propose (low-z labels)"
        elif dev["h_coverage"]:
            v = "STOP: H-coverage holds (precision pool unchanged): report and propose (DART-style reach perturbations)"
        elif dev["lo"] > 0 or dev["hi"] < 0:
            v = f"KEEP C3: the difference {cm.pp(dev['diff'], dev['lo'], dev['hi'])} does not meet the conditions; report"
        else:
            v = (f"KEEP C3: no difference detected at a resolution of about {100 * dev['half_width']:.1f} pp "
                 f"({cm.pp(dev['diff'], dev['lo'], dev['hi'])}); report the reasons")
    lines.append(f"  -> {v}")
    res["verdict"] = v
    out["adoption"] = res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-confirm", action="store_true")
    ap.add_argument("--out", default=os.path.join(ar.STEP3_DIR, "D_mechanisms"))
    args = ap.parse_args(argv)
    ar.ALLOW_CONFIRM = args.allow_confirm
    lines, out = [], {}
    model_meta_check(lines, out)
    lines.append("")
    em_part(lines, out)
    lines.append("")
    dev = mm_part("dev", lines, out)
    lines.append("")
    mm_part("hard", lines, out, dev["decision_metric"] if dev else None)
    lines.append("")
    if args.allow_confirm:
        mm_part("confirm", lines, out, dev["decision_metric"] if dev else None)
    else:
        lines.append("== D-MM (CONFIRM): not opened (no --allow-confirm) ==")
        out["mm_confirm"] = None
    lines.append("")
    adoption(lines, out)
    text = "\n".join(lines)
    print(text)
    with open(args.out + ".txt", "w") as f:
        f.write(text + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump(ar._clean(out), f, indent=1)


if __name__ == "__main__":
    main()
