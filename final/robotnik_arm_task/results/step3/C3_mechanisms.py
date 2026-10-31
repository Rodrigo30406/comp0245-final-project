"""Experiment C3 (trigger-frame copies): the pre-registered verdicts, adoption conditions and mechanism predictions
(report.txt, "[Methodology] 第 3 步续预注册：实验 C3", frozen before any C3 run; STEP3C_PLAN_PROMPT.md 3.3-3.7),
evaluated on explicit run folders. Each part is computed when its runs exist, the others are listed as missing.

  python results/step3/C3_mechanisms.py [--allow-confirm] [--figure]

Run folders (seeds 0-4; CONFIRM 100-104):
  C3-EM  results/B-EM_ego-novw-trig_s<k>_dev   (expert base + C3 arm, --no-early-stop)  vs  B-EM_ego-novw_s<k>_dev
  C3-MM  results/ego-novw-trig_s<k>_<set>      (the C3 model alone, early stop)          vs  MM = ego-novw_s<k>_<set>
  stall re-runs results/step3/nostall/<model>_s<k>_dev (--no-early-stop, the stalled episodes only)
  expert results/expert_v11_<set> (reference on the same episodes)
Statistics as in step 3 (analyze_rollouts.py): seed means; two-level bootstrap (10000 draws, rng seed 0) and the
seed-level t interval (Welch on the per-seed rates for two arms; for one arm the one-sample t interval of its per-seed
rates), both required. C3-MM and MM are different networks: their seeds are not paired (the per-k paired t is
descriptive only).
Output: results/step3/C3_mechanisms.txt and .json (--out stem; with --figure fig_step3_C3_arm.{png,pdf} next to it).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK)
import analyze_rollouts as ar  # noqa: E402

RES = ar.RESULTS_DIR
NOSTALL = os.path.join(ar.STEP3_DIR, "nostall")
OLD, NEW = "ego-novw", "ego-novw-trig"
SEEDS = {"dev": [0, 1, 2, 3, 4], "hard": [0, 1, 2, 3, 4], "confirm": [100, 101, 102, 103, 104]}
PREDICTED = ("parked_no_arm", "circling")          # condition (2): the classes C3 is predicted to reduce
EM_TRIGGER_LO, EM_NO_TRIGGER_HI = 0.30, 0.20      # C3-EM verdict on arm_extended_toward
MIN_CELLS = 30                                    # H-trajectory / H-premature: pooled seed x episode cells needed
H_TRAJ_MAX_SUCCESS = 0.5                          # H-trajectory holds when the extended cells succeed < 50%
PREMATURE_RING, PREMATURE_SHARE = 0.20, 0.30      # H-premature: ring_dist_at_arm_extend > 0.2 m in > 30% of them
PREMATURE_CLASSES = ("extended_not_reached", "left_or_overshot")
FACING_RING, FACING_DEG = 0.20, 6.0               # "extended on the ring, facing the target"
STRATUM_FLOOR = -0.10                             # condition (3): every n >= 30 stratum CI upper bound >= -10 pp
EXPERT_LAG = "expert arm_extend_time - nav_done: 0.33-2.7 s, median 0.47 s (report.txt, protocol v1.1)"


def pct(x, d=1):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def pp(d, lo, hi):
    return f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] pp"


def one_arm_t(rates, level=0.95):
    """t interval of the mean of per-seed rates (S - 1 df); degenerate [m, m] when all rates are equal."""
    r = np.asarray(rates, dtype=float)
    m = float(r.mean())
    if r.size < 2:
        return m, np.nan, np.nan
    se = r.std(ddof=1) / np.sqrt(r.size)
    q = float(stats.t.ppf(0.5 + level / 2, r.size - 1))
    return m, m - q * se, m + q * se


def paired_k_t(rate_a, rate_b):
    """Descriptive only: t interval of the per-k differences rate_b[k] - rate_a[k]."""
    return one_arm_t(np.asarray(rate_b, dtype=float) - np.asarray(rate_a, dtype=float))


def run_dirs(model, set_name, cell=None):
    return [ar.run_dir_for(RES, model, s, set_name, cell=cell) for s in SEEDS[set_name]]


def have(paths):
    return bool(paths) and all(os.path.exists(os.path.join(p, "rollouts.csv")) for p in paths)


def load(paths):
    """Prepared runs; refuses any run whose folder name and config disagree (load_run's warnings: evaluation set,
    policy_arg of the model / hybrid cell), so a run made with the wrong model file cannot pass silently."""
    runs = [ar.load_prepared(p) for p in paths]
    warns = [w for run in runs for w in run.meta.get("warnings", [])]
    if warns:
        raise SystemExit("C3_mechanisms: refused, run folders inconsistent with their config.json:\n  " + "\n  ".join(warns))
    return runs


def indicator(runs, eps, fn):
    return ar.matrix(runs, eps, lambda row: float(bool(fn(row))))


def diff_both(A, B, boot):
    """Paired B - A (bootstrap) and Welch on the per-seed rates."""
    d = ar.paired_diff(A, B, boot=boot)
    t = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B))
    return d, t


def diff_line(name, d, t):
    return (f"{name}: A {pct(d['mean_a'])}%  B {pct(d['mean_b'])}%  B - A bootstrap {pp(d['diff'], d['lo'], d['hi'])}, "
            f"p {ar.fmt_p(d['p'])}; Welch {pp(t['diff'], t['lo'], t['hi'])}, p {ar.fmt_p(t['p'])}")


def lag_stats(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "p5": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)), "min": float(x.min()), "max": float(x.max()),
            "n_before_nav_done": int(np.sum(x < 0)), "n_over_2.7s": int(np.sum(x > 2.7))}


def lag_line(s):
    if s["n"] == 0:
        return "n 0"
    return (f"n {s['n']}, median {s['median']:.3f} s [p5 {s['p5']:.3f}, p25 {s['p25']:.3f}, p75 {s['p75']:.3f}, "
            f"p95 {s['p95']:.3f}], min {s['min']:.3f}, max {s['max']:.3f}; before nav_done {s['n_before_nav_done']}, "
            f"> 2.7 s {s['n_over_2.7s']}")


def facing_extension(row):
    return (np.isfinite(ar._f(row, "arm_extend_time")) and ar.metric_value(row, "arm_extended_toward") == 1
            and ar._f(row, "ring_dist_at_arm_extend") < FACING_RING
            and abs(ar._f(row, "target_bearing_err_at_arm_extend_deg")) < FACING_DEG)


def facing_lines(label, runs, eps):
    rows = [run.by_episode[e] for run in runs for e in eps]
    tow = [r for r in rows if ar.metric_value(r, "arm_extended_toward") == 1]
    k = sum(facing_extension(r) for r in tow)
    return f"  {label}: extended toward the target {len(tow)} seed x episode, of them on the ring and facing " \
           f"(ring_dist_at_arm_extend < {FACING_RING} m, |bearing err| < {FACING_DEG:g} deg) {k}" \
           + (f" ({100 * k / len(tow):.1f}%)" if tow else ""), {"n_toward": len(tow), "n_facing": int(k)}


def model_meta_check(lines, out):
    """Every C3 model file records the same K (the one chosen by C3_K_choice.json, amended rule 1) and the same
    added rows."""
    path = os.path.join(ar.STEP3_DIR, "C3_K_choice.json")
    chosen = json.load(open(path))["chosen_k"] if os.path.exists(path) else None
    seen = {}
    for s in sorted(set(SEEDS["dev"]) | set(SEEDS["confirm"])):
        f = os.path.join(ar.TRAINED_DIR, f"{NEW}_s{s}.npz")
        if os.path.exists(f):
            with np.load(f) as z:
                t = json.loads(str(z["meta"])).get("trigger") or {}
            seen[s] = (t.get("copies"), t.get("n_rows_added"), tuple(np.round(t.get("qd_sigma", []), 12)))
    ok = bool(seen) and len(set(seen.values())) == 1 and chosen is not None and \
        all(v[0] == chosen for v in seen.values())
    lines.append(f"C3 models: {len(seen)} files {sorted(seen)}; K per file {sorted({v[0] for v in seen.values()})}, "
                 f"chosen K (C3_K_choice.json) {chosen}; identical trigger settings: {ok}")
    out["models"] = {"chosen_k": chosen, "files": {str(k): v[:2] for k, v in seen.items()}, "consistent": ok}


# ==============================================================================
# C3-EM (DEV)
# ==============================================================================
def em_part(lines, out):
    new_p, old_p = run_dirs(NEW, "dev", "EM"), run_dirs(OLD, "dev", "EM")
    lines.append("== C3-EM (DEV): expert base + C3 arm, --no-early-stop; A = B-EM ego-novw, B = B-EM ego-novw-trig ==")
    if not have(new_p):
        lines.append("  missing: " + ", ".join(os.path.relpath(p, TASK) for p in new_p if not have([p])))
        out["em"] = None
        return None
    new, old = load(new_p), load(old_p)
    ar.check_integrity(new + old, "C3-EM vs B-EM")
    eps, _ = ar.paired_episodes(new + old)
    boot = ar.Bootstrap([len(old), len(new)], len(eps), ar.N_BOOT, ar.BOOT_SEED)
    Y = ar.metric_matrix(new, eps, "arm_extended_toward")
    b = ar.boot2(Y)
    m, tlo, thi = one_arm_t(ar.seed_rates(Y))
    if b["lo"] > EM_TRIGGER_LO and tlo > EM_TRIGGER_LO:
        verdict = "trigger"
    elif b["hi"] < EM_NO_TRIGGER_HI and thi < EM_NO_TRIGGER_HI:
        verdict = "no_trigger"
    else:
        verdict = "uncertain"
    expert = ar.expert_run_for(RES, "dev")
    xr = ar.metric_matrix([expert], eps, "arm_extended_toward").mean() if expert else np.nan
    lines.append(f"  arm_extended_toward per seed: " + "/".join(pct(x) for x in ar.seed_rates(Y)) + "%")
    lines.append(f"  seed mean {pct(m)}%: two-level bootstrap [{pct(b['lo'])}, {pct(b['hi'])}]%, per-seed t interval "
                 f"[{pct(tlo)}, {pct(thi)}]% (expert {pct(xr)}% on the same episodes)")
    lines.append(f"  VERDICT C3-EM: {verdict.upper()} (trigger: both lower bounds > {pct(EM_TRIGGER_LO, 0)}%; no trigger: "
                 f"both upper bounds < {pct(EM_NO_TRIGGER_HI, 0)}%; otherwise uncertain)")
    res = {"verdict": verdict, "per_seed": ar.seed_rates(Y), "mean": m, "boot": [b["lo"], b["hi"]], "t": [tlo, thi],
           "expert": xr, "diffs": {}}
    lines.append("  descriptive, paired B - A on the same episodes:")
    for metric in ("arm_extended_toward", "arm_extended", "success", "reach_5cm", "reach_2cm"):
        d, t = diff_both(ar.metric_matrix(old, eps, metric), ar.metric_matrix(new, eps, metric), boot)
        res["diffs"][metric] = {"a": d["mean_a"], "b": d["mean_b"], "diff": d["diff"], "lo": d["lo"], "hi": d["hi"],
                                "welch_lo": t["lo"], "welch_hi": t["hi"]}
        lines.append("    " + diff_line(metric, d, t))
    # arm extension time relative to the shadow FSM's nav_done (hybrid logs)
    lags, lags_tow, no_nav = [], [], 0
    for run in new:
        logs = ar.run_logs(run, eps)
        for e in eps:
            row, log = run.by_episode[e], logs[e]
            ext = ar._f(row, "arm_extend_time")
            if not np.isfinite(ext):
                continue
            if log is None or log["nav_done"] is None:
                no_nav += 1
                continue
            lags.append(ext - log["nav_done"]["t"])
            if ar.metric_value(row, "arm_extended_toward") == 1:
                lags_tow.append(ext - log["nav_done"]["t"])
    res["lag"], res["lag_toward"], res["lag_no_nav_done"] = lag_stats(lags), lag_stats(lags_tow), no_nav
    lines.append(f"  arm_extend_time - shadow FSM nav_done (logs), extended cells: {lag_line(res['lag'])}")
    lines.append(f"    ... extended toward the target only: {lag_line(res['lag_toward'])}; extended without a "
                 f"nav_done event {no_nav}")
    lines.append(f"    ({EXPERT_LAG})")
    l_, f_ = facing_lines("C3-EM", new, eps)
    lines.append(l_)
    res["facing"] = f_
    # H-trajectory (only when the verdict is "trigger")
    lines.append("  H-trajectory (triggered but not accurate): assessed only when C3-EM triggers")
    tow = [(run, e) for run in new for e in eps if ar.metric_value(run.by_episode[e], "arm_extended_toward") == 1]
    n_tow, k_succ = len(tow), sum(ar.metric_value(run.by_episode[e], "success") == 1 for run, e in tow)
    classes = {}
    for run, e in tow:
        c = run.by_episode[e]["fclass"]
        classes[c] = classes.get(c, 0) + 1
    xt = [e for e in eps if expert and ar.metric_value(expert.by_episode[e], "arm_extended_toward") == 1]
    xk = sum(ar.metric_value(expert.by_episode[e], "success") == 1 for e in xt)
    if verdict != "trigger":
        h_traj = None
        lines.append(f"    not assessed (verdict {verdict})")
    elif n_tow < MIN_CELLS:
        h_traj = None
        lines.append(f"    not assessable: {n_tow} extended-toward cells (< {MIN_CELLS})")
    else:
        h_traj = bool(k_succ / n_tow < H_TRAJ_MAX_SUCCESS)
        lines.append(f"    extended-toward cells {n_tow}, of them 10 mm success {k_succ} ({pct(k_succ / n_tow)}%): "
                     f"H-trajectory {'HOLDS' if h_traj else 'does not hold'} (holds when < {pct(H_TRAJ_MAX_SUCCESS, 0)}%)")
    lines.append(f"    classes of the extended-toward cells: {dict(sorted(classes.items(), key=lambda kv: -kv[1]))}; "
                 f"expert {xk}/{len(xt)} of its extended-toward episodes succeed")
    res["h_trajectory"] = {"holds": h_traj, "n_toward": n_tow, "n_success": int(k_succ), "classes": classes,
                           "expert": [int(xk), len(xt)]}
    out["em"] = res
    return verdict


# ==============================================================================
# C3-MM (DEV / HARD / CONFIRM)
# ==============================================================================
def decision_metric(YA, YB, MA5, MB5):
    """Floor rule of step 3 (only from whether both arms are on the floor): success; both seed means < 5% ->
    reach_5cm; both reach_5cm < 5% too -> parked."""
    if not (YA.mean() < ar.FLOOR_SUCCESS and YB.mean() < ar.FLOOR_SUCCESS):
        return "success"
    if not (MA5.mean() < ar.FLOOR_SUCCESS and MB5.mean() < ar.FLOOR_SUCCESS):
        return "reach_5cm"
    return "parked"


def share_from_predicted(old, new, eps, metric, boot, d):
    """Condition (2): reduction of the share of seed x episode cells that fail the decision metric AND are in a
    predicted class (A - B), divided by the net difference of the decision metric (B - A); bootstrap CI from the same
    draws (experiment A's rule; for success the failing cells are exactly the failure classes)."""
    fail_pred = lambda row: ar.metric_value(row, metric) == 0 and row["fclass"] in PREDICTED  # noqa: E731
    IA, IB = indicator(old, eps, fail_pred), indicator(new, eps, fail_pred)
    dc = ar.paired_diff(IA, IB, boot=boot)
    red, red_draws = -dc["diff"], -dc["draws"]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = red / d["diff"] if d["diff"] != 0 else np.nan
        lo, hi = ar.percentile_ci(red_draws / d["draws"])
    return {"a": IA.mean(), "b": IB.mean(), "reduction": red, "share": share, "lo": lo, "hi": hi}


def replace_stalled(runs, model, metric, eps):
    """Metric matrix with the rows of the stall re-runs (results/step3/nostall/<run>) in place of the early-stopped
    rows, and the per-run changes of the metrics. Returns (matrix, list of change dicts)."""
    M = ar.metric_matrix(runs, eps, metric)
    changes = []
    col = {e: j for j, e in enumerate(eps)}
    for i, run in enumerate(runs):
        p = os.path.join(NOSTALL, run.name)
        n_stalled = sum(row.get("stop_reason") == "stalled" for row in run.rows)
        if not os.path.exists(os.path.join(p, "rollouts.csv")):
            changes.append({"run": run.name, "rerun": False, "n_stalled": n_stalled})
            continue
        rr = load([p])[0]
        ch = {"run": run.name, "rerun": True, "n": len(rr.rows), "n_stalled": n_stalled,
              "same_episodes": sorted(r["episode"] for r in rr.rows) == sorted(
                  row["episode"] for row in run.rows if row.get("stop_reason") == "stalled")}
        for m in ("success", "reach_5cm", "reach_2cm", "arm_extended", "arm_extended_toward", "parked"):
            ch[m] = int(sum(ar.metric_value(r, m) != ar.metric_value(run.by_episode[r["episode"]], m) for r in rr.rows))
        ch["class_changed"] = int(sum(r["fclass"] != run.by_episode[r["episode"]]["fclass"] for r in rr.rows))
        for r in rr.rows:
            if r["episode"] in col:
                M[i, col[r["episode"]]] = ar.metric_value(r, metric)
        changes.append(ch)
    return M, changes


def mm_part(set_name, lines, out, dev_metric=None):
    """C3-MM vs MM on one set. DEV fixes the decision metric (floor rule) and evaluates conditions (1)-(3) and the
    mechanism predictions; HARD / CONFIRM reuse the DEV decision metric (no new floor rule): direction only."""
    new_p, old_p = run_dirs(NEW, set_name), run_dirs(OLD, set_name)
    lines.append(f"== C3-MM ({set_name.upper()}): A = MM ego-novw, B = C3-MM ego-novw-trig, seeds {SEEDS[set_name]} ==")
    if not have(new_p) or not have(old_p):
        lines.append("  missing: " + ", ".join(os.path.relpath(p, TASK) for p in new_p + old_p if not have([p])))
        out[f"mm_{set_name}"] = None
        return None
    new, old = load(new_p), load(old_p)
    ar.check_integrity(old + new, f"C3-MM vs MM ({set_name})")
    expert = ar.expert_run_for(RES, set_name)
    eps, _ = ar.paired_episodes(old + new)
    boot = ar.Bootstrap([len(old), len(new)], len(eps), ar.N_BOOT, ar.BOOT_SEED)
    YA, YB = (ar.metric_matrix(r, eps, "success") for r in (old, new))
    MA5, MB5 = (ar.metric_matrix(r, eps, "reach_5cm") for r in (old, new))
    if set_name == "dev":
        metric = decision_metric(YA, YB, MA5, MB5)
        rule = "floor rule on DEV"
    else:
        metric = dev_metric or decision_metric(YA, YB, MA5, MB5)
        rule = "the DEV decision metric" if dev_metric else "floor rule on this set (DEV part missing)"
    res = {"decision_metric": metric, "rule": rule,
           "full_compare": f"results/step3/A_{OLD}_vs_{NEW}_{set_name}.txt (analyze_rollouts.py compare)"}
    lines.append(f"  success seed means: A {pct(YA.mean())}%, B {pct(YB.mean())}%; reach_5cm: A {pct(MA5.mean())}%, "
                 f"B {pct(MB5.mean())}% -> decision metric {metric} ({rule})")
    MA, MB = ar.metric_matrix(old, eps, metric), ar.metric_matrix(new, eps, metric)
    d, t = diff_both(MA, MB, boot)
    lines.append("  " + diff_line(f"decision metric {metric}", d, t))
    lines.append("    per seed A " + "/".join(pct(x) for x in ar.seed_rates(MA)) + ", B "
                 + "/".join(pct(x) for x in ar.seed_rates(MB)) + "%")
    pk = paired_k_t(ar.seed_rates(MA), ar.seed_rates(MB))
    lines.append(f"    descriptive only (seeds are different networks): per-k paired t {pp(*pk)}")
    res.update(diff=d["diff"], lo=d["lo"], hi=d["hi"], p=d["p"], welch_lo=t["lo"], welch_hi=t["hi"], welch_p=t["p"],
               per_seed_a=ar.seed_rates(MA), per_seed_b=ar.seed_rates(MB), mean_a=d["mean_a"], mean_b=d["mean_b"])
    if set_name != "dev":
        res["direction"] = bool(d["diff"] > 0)
        res["reverse_direction"] = bool(d["diff"] < 0)
        lines.append(f"  direction B - A {100 * d['diff']:+.1f} pp: {'> 0' if d['diff'] > 0 else '<= 0'} "
                     f"(condition ({'4' if set_name == 'hard' else '5'}) needs > 0)")
        if metric != "success":
            ds, _ = diff_both(YA, YB, boot)
            lines.append(f"  success (the no-worse rule is applied on DEV only; descriptive here): B - A "
                         f"{100 * ds['diff']:+.2f} pp")
        for metric_ in ("arm_extended_toward", "success"):
            dm, tm = diff_both(ar.metric_matrix(old, eps, metric_), ar.metric_matrix(new, eps, metric_), boot)
            lines.append("    descriptive " + diff_line(metric_, dm, tm))
        fac = {}
        for label, runs in (("MM", old), ("C3-MM", new)) + ((("expert", [expert]),) if expert else ()):
            l_, fac[label] = facing_lines(label, runs, eps)
            lines.append(l_)
        res["facing"] = fac
        out[f"mm_{set_name}"] = res
        return res
    c1, r1 = bool(d["lo"] > 0 and t["lo"] > 0), bool(d["hi"] < 0 and t["hi"] < 0)
    sh = share_from_predicted(old, new, eps, metric, boot, d)
    c2 = bool(np.isfinite(sh["share"]) and sh["share"] >= 0.5 and d["diff"] > 0)     # improvement explained
    r2 = bool(np.isfinite(sh["share"]) and sh["share"] >= 0.5 and d["diff"] < 0)     # worsening explained (mirror)
    lines.append(f"  (1) CI lower bounds: bootstrap {100 * d['lo']:+.1f} pp, Welch {100 * t['lo']:+.1f} pp -> "
                 f"{'MET' if c1 else 'NOT MET'} (both > 0)")
    lines.append(f"  (2) cells failing {metric} in {'+'.join(PREDICTED)}: A {pct(sh['a'])}%, B {pct(sh['b'])}%, reduction "
                 f"{100 * sh['reduction']:+.1f} pp / net difference {100 * d['diff']:+.1f} pp = {pct(sh['share'])}% "
                 f"[bootstrap {pct(sh['lo'])}, {pct(sh['hi'])}] -> {'MET' if c2 else 'NOT MET'} (>= 50% of a net "
                 f"improvement)")
    sm = ar.strata_analysis(MA, MB, np.array([old[0].by_episode[e]["d0"] for e in eps], dtype=float))
    worst, best_lo = sm["worst_hi_n30"], None
    big = [s_ for s_ in sm["strata"] if s_["n"] >= ar.MIN_STRATUM_N]
    if big:
        best_lo = max(big, key=lambda s_: s_["lo"])
    c3 = bool(worst is not None and worst[1] >= STRATUM_FLOOR)
    r3 = bool(best_lo is not None and best_lo["lo"] <= -STRATUM_FLOOR)
    lines.append(f"  (3) worst n >= {ar.MIN_STRATUM_N} stratum CI upper bound: "
                 + (f"{worst[0]} {100 * worst[1]:+.1f} pp" if worst else "none") + f" -> {'MET' if c3 else 'NOT MET'} "
                 f"(>= {100 * STRATUM_FLOOR:.0f} pp)")
    for s_ in sm["strata"]:
        if s_["n"]:
            lines.append(f"      {s_['name']:<14} n {s_['n']:>3} A {pct(s_['mean_a'])}%  B {pct(s_['mean_b'])}%  "
                         f"{pp(s_['diff'], s_['lo'], s_['hi'])}" + ("" if s_["in_holm"] else " (descriptive)"))
    no_worse = None
    if metric != "success":
        ds, ts = diff_both(YA, YB, boot)
        no_worse = bool(ds["diff"] >= 0)
        lines.append(f"  no-worse (decision metric is not success): success B - A {100 * ds['diff']:+.2f} pp -> "
                     f"{'MET' if no_worse else 'NOT MET'} (point estimate >= 0)")
    if d["lo"] > 0 and not t["lo"] > 0:
        q_a, q_b = ar.cochran_q(MA), ar.cochran_q(MB)
        lines.append("  bootstrap met but Welch not: the effect appears in some seeds only (not adopted); per-seed "
                     "B_k - A_k " + "/".join(f"{100 * x:+.1f}" for x in ar.seed_rates(MB) - ar.seed_rates(MA))
                     + f" pp; Cochran Q A {q_a[0]:.1f} (p {ar.fmt_p(q_a[2])}), B {q_b[0]:.1f} (p {ar.fmt_p(q_b[2])})")
    res.update(cond1=c1, cond2=c2, share=sh, cond3=c3, worst_stratum=worst, no_worse=no_worse,
               reverse_cond1=r1, reverse_cond2=r2, reverse_cond3=r3, bootstrap_not_welch=bool(d["lo"] > 0 and not t["lo"] > 0),
               reverse_best_stratum_lo=(best_lo["name"], best_lo["lo"]) if best_lo else None,
               strata=[{k: s_.get(k) for k in ("name", "n", "mean_a", "mean_b", "diff", "lo", "hi", "in_holm")}
                       for s_ in sm["strata"]])
    lines.append(f"  reverse direction (MM better): (1) both upper bounds < 0 {'MET' if r1 else 'NOT MET'}; (2) share "
                 f"of the net worsening from increases of {'+'.join(PREDICTED)} >= 50% {'MET' if r2 else 'NOT MET'}; "
                 f"(3) highest n >= 30 stratum CI lower bound <= +10 pp {'MET' if r3 else 'NOT MET'}")
    # mechanism predictions
    lines.append("  Mechanism predictions (B - A, seed-mean shares of all episodes):")
    cls = {}
    for c in ("circling", "parked_no_arm", "extended_not_reached", "left_or_overshot", "approached_other", "crash"):
        IA, IB = indicator(old, eps, lambda row, c=c: row["fclass"] == c), indicator(new, eps, lambda row, c=c: row["fclass"] == c)
        dc, tc = diff_both(IA, IB, boot)
        cls[c] = {"a": dc["mean_a"], "b": dc["mean_b"], "diff": dc["diff"], "lo": dc["lo"], "hi": dc["hi"],
                  "welch_lo": tc["lo"], "welch_hi": tc["hi"]}
        lines.append("    " + diff_line(c, dc, tc))
    coop = bool(cls["circling"]["hi"] < 0 and cls["circling"]["welch_hi"] < 0 and cls["parked_no_arm"]["diff"] < 0)
    lines.append(f"  H-cooperation (the arm extends, then the base stops): circling CI upper bounds < 0 (bootstrap "
                 f"{100 * cls['circling']['hi']:+.1f}, Welch {100 * cls['circling']['welch_hi']:+.1f} pp) and "
                 f"parked_no_arm lower ({100 * cls['parked_no_arm']['diff']:+.1f} pp) -> {'HOLDS' if coop else 'does not hold'}")
    for metric_ in ("arm_extended", "arm_extended_toward", "parked", "reach_5cm", "reach_2cm", "success"):
        dm, tm = diff_both(ar.metric_matrix(old, eps, metric_), ar.metric_matrix(new, eps, metric_), boot)
        lines.append("    " + diff_line(metric_, dm, tm))
    ext = [(run, run.by_episode[e]) for run in new for e in eps if np.isfinite(ar._f(run.by_episode[e], "arm_extend_time"))]
    prem = [(run, row) for run, row in ext if ar._f(row, "ring_dist_at_arm_extend") > PREMATURE_RING]
    pcls = {}
    for _, row in prem:
        if row["fclass"] in ar.FAILURE_CLASSES:
            pcls[row["fclass"]] = pcls.get(row["fclass"], 0) + 1
    n_top = max(pcls.values()) if pcls else 0
    top = sorted(c for c, n in pcls.items() if n == n_top)   # every class with the maximal count (ties reported)
    if len(ext) < MIN_CELLS:
        h_prem = None
        lines.append(f"  H-premature: not assessable ({len(ext)} extended cells < {MIN_CELLS})")
    else:
        share_p = len(prem) / len(ext)
        h_prem = bool(share_p > PREMATURE_SHARE and any(c in PREMATURE_CLASSES for c in top))
        lines.append(f"  H-premature: extended cells {len(ext)}, ring_dist_at_arm_extend > {PREMATURE_RING} m in "
                     f"{len(prem)} ({pct(share_p)}%; needs > {pct(PREMATURE_SHARE, 0)}%); main failure class of those "
                     f"{top}{' (tie)' if len(top) > 1 else ''} {pcls} (needs one of {PREMATURE_CLASSES}; on a tie any of the "
                     f"tied classes) -> {'HOLDS' if h_prem else 'does not hold'}")
    ring_ext = np.array([ar._f(row, "ring_dist_at_arm_extend") for _, row in ext])
    lag_ref = np.array([ar._f(row, "arm_extend_time") - ar._f(row, "ref_navdone_time") for _, row in ext])
    lines.append(f"    descriptive: ring_dist_at_arm_extend median {np.median(ring_ext) if ext else np.nan:.3f} m; "
                 f"arm_extend_time - the expert's recorded nav_done time (another base trajectory): "
                 f"{lag_line(lag_stats(lag_ref[np.isfinite(lag_ref)]))}")
    fa, fa_r = facing_lines("MM", old, eps)
    fb, fb_r = facing_lines("C3-MM", new, eps)
    lines += [fa, fb]
    if expert:
        fx, fx_r = facing_lines("expert", [expert], eps)
        lines.append(fx)
    else:
        fx_r = None
    res.update(classes=cls, h_cooperation=coop, h_premature=h_prem, n_extended=len(ext), n_premature=len(prem),
               premature_classes=pcls, facing={"a": fa_r, "b": fb_r, "expert": fx_r})
    # stall re-runs and the sensitivity of the decision metric
    MAs, ch_a = replace_stalled(old, OLD, metric, eps)
    MBs, ch_b = replace_stalled(new, NEW, metric, eps)
    lines.append("  stall re-runs (--no-early-stop, the stalled episodes): changed outcomes per run "
                 "(success / reach_5cm / reach_2cm / arm_extended / arm_extended_toward / parked / class):")
    for ch in ch_a + ch_b:
        if not ch["rerun"]:
            lines.append(f"    {ch['run']}: " + ("no stalled episode" if ch["n_stalled"] == 0 else
                                                f"WARNING: {ch['n_stalled']} stalled episodes but no re-run folder"))
            continue
        lines.append(f"    {ch['run']}: {ch['n']} re-run ({ch['n_stalled']} stalled; same episodes "
                     f"{ch['same_episodes']}), changes {ch['success']}/{ch['reach_5cm']}/{ch['reach_2cm']}/"
                     f"{ch['arm_extended']}/{ch['arm_extended_toward']}/{ch['parked']}/{ch['class_changed']}")
    ds_, ts_ = diff_both(MAs, MBs, boot)
    lines.append("  sensitivity (descriptive): both arms with the stall re-run rows: " + diff_line(metric, ds_, ts_))
    res.update(stall_changes=ch_a + ch_b, stall_sensitivity={"diff": ds_["diff"], "lo": ds_["lo"], "hi": ds_["hi"],
                                                             "welch_lo": ts_["lo"], "welch_hi": ts_["hi"]})
    out[f"mm_{set_name}"] = res
    return res


def adoption(lines, out):
    """The verdict and the next step by the pre-registered decision table (report.txt C3 item 7; user 2026-10-30:
    CONFIRM is opened only after HARD condition (4) holds)."""
    em, dev, hard, conf = out.get("em"), out.get("mm_dev"), out.get("mm_hard"), out.get("mm_confirm")
    lines.append("== Adoption and next step (C3-MM replaces MM only if all of (1)-(5) and the no-worse rule hold) ==")
    res = {"conditions": None, "reverse": None}
    if not em:
        verdict = "NEXT: run C3-EM (stage em)"
    elif em["verdict"] != "trigger":
        verdict = (f"STOP: C3-EM {em['verdict']}: C3-MM is not run (decision table); "
                   + ("report" if em["verdict"] == "uncertain" else "propose next steps, with the probe r vs closed loop"))
    elif not dev:
        verdict = "NEXT: C3-EM triggered: run C3-MM on DEV (stage mm)"
    else:
        c = [dev["cond1"], dev["cond2"], dev["cond3"], hard["direction"] if hard else None,
             conf["direction"] if conf else None]
        rev = [dev["reverse_cond1"], dev["reverse_cond2"], dev["reverse_cond3"],
               hard["reverse_direction"] if hard else None]
        res.update(conditions=c, reverse=rev)
        if dev["no_worse"] is not None:
            lines.append(f"  no-worse rule (decision metric {dev['decision_metric']}): {dev['no_worse']}")
        lines.append("  conditions (1)-(5): " + ", ".join(f"({i + 1}) {x}" for i, x in enumerate(c)))
        lines.append("  mirrored (1)-(4): " + ", ".join(f"({i + 1}) {x}" for i, x in enumerate(rev)))
        traj = (em.get("h_trajectory") or {}).get("holds") is True
        ok_dev = all(c[:3]) and dev["no_worse"] is not False
        if traj:
            verdict = ("STOP: H-trajectory holds (C3-EM triggers but the reach is not accurate): C3-MM judged above "
                       f"(DEV (1)-(3){' met' if ok_dev else ' not met'}); report the trigger / trajectory "
                       "decomposition and propose next steps (decision table row 3 takes precedence)")
        elif ok_dev and hard is None:
            verdict = "NEXT: DEV (1)-(3) and the no-worse rule met: run HARD (stage hard)"
        elif ok_dev and not hard["direction"]:
            verdict = "NOT ADOPTED: HARD direction (4) not met; CONFIRM is not opened. Keep MM, report, propose"
        elif ok_dev and conf is None:
            verdict = "NEXT: DEV (1)-(3) and HARD (4) met: run CONFIRM once (stage confirm, --allow-confirm)"
        elif ok_dev:
            verdict = "ADOPT C3-MM (all of (1)-(5))" if conf["direction"] else "NOT ADOPTED: CONFIRM direction (5)"
        elif dev["bootstrap_not_welch"]:
            verdict = ("KEEP MM: the effect appears in some seeds only (bootstrap CI lower bound > 0, Welch not); "
                       "not adopted (pre-registered wording); report and propose")
        elif all(rev[:3]):
            verdict = ("REVERSE: evidence that the trigger-frame copies are harmful (mirrored (1)-(4))" if rev[3] else
                       "STOP: mirrored (1)-(3) hold on DEV; mirrored (4) needs HARD, which the decision table does not "
                       "run here: report, and propose HARD as part of the next steps (not yet counted as evidence of "
                       "harm)")
        elif dev["reverse_cond1"]:
            verdict = (f"KEEP MM: C3-MM is worse on {dev['decision_metric']} (both CI upper bounds < 0, "
                       f"{pp(dev['diff'], dev['lo'], dev['hi'])}), but the mirrored conditions (2)-(3) do not all hold: "
                       f"not counted as evidence of harm by the pre-registered rule; report the reasons, propose")
        elif dev["lo"] > 0 or dev["hi"] < 0:
            verdict = (f"KEEP MM: the difference on {dev['decision_metric']} ({pp(dev['diff'], dev['lo'], dev['hi'])}) "
                       f"does not meet the adoption conditions (see (1)-(3)); report the reasons, propose")
        else:
            half = (dev["hi"] - dev["lo"]) / 2
            verdict = (f"KEEP MM: no difference detected at a resolution of about {100 * half:.1f} pp (bootstrap CI "
                       f"half-width; {pp(dev['diff'], dev['lo'], dev['hi'])} on {dev['decision_metric']}); report the "
                       f"reasons, propose")
    lines.append(f"  -> {verdict}")
    res["verdict"] = verdict
    out["adoption"] = res


def figure(out, path_stem):
    """fig_step3_C3_arm: success and arm_extended_toward of B-EM, C3-EM, MM, C3-MM (seed mean, two-level bootstrap
    95% CI) and the expert (single run) on DEV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [("B-EM\nego-novw", run_dirs(OLD, "dev", "EM")), ("C3-EM\nego-novw-trig", run_dirs(NEW, "dev", "EM")),
             ("MM\nego-novw", run_dirs(OLD, "dev")), ("C3-MM\nego-novw-trig", run_dirs(NEW, "dev")),
             ("expert", [ar.run_dir_for(RES, "expert", None, "dev")])]
    cells = [(n, p) for n, p in cells if have(p)]
    runs = {n: load(p) for n, p in cells}
    eps, _ = ar.paired_episodes([r for rs in runs.values() for r in rs])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ("success", "arm_extended_toward")):
        for i, (n, rs) in enumerate(runs.items()):
            Y = ar.metric_matrix(rs, eps, metric)
            if len(rs) > 1:
                b = ar.boot2(Y)
                ax.bar(i, 100 * b["mean"], color="#4C78A8")
                ax.errorbar(i, 100 * b["mean"], yerr=[[100 * (b["mean"] - b["lo"])], [100 * (b["hi"] - b["mean"])]],
                            color="k", capsize=3)
            else:
                ax.bar(i, 100 * Y.mean(), color="#9D9D9D")
            ax.text(i, 100 * Y.mean() + 2, f"{100 * Y.mean():.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(list(runs), fontsize=7)
        ax.set_ylim(0, 108)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"DEV {metric} (seed mean, 2-level bootstrap 95% CI)", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=150)
    plt.close(fig)
    out["figure"] = path_stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-confirm", action="store_true", help="include the CONFIRM part (opened once, before adoption)")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--out", default=os.path.join(ar.STEP3_DIR, "C3_mechanisms"),
                    help="output stem (.txt / .json); the figure goes next to it")
    args = ap.parse_args(argv)
    ar.ALLOW_CONFIRM = args.allow_confirm
    lines, out = [], {}
    model_meta_check(lines, out)
    lines.append("")
    em_part(lines, out)
    lines.append("")
    dev = mm_part("dev", lines, out)
    lines.append("")
    mm_part("hard", lines, out, dev_metric=dev["decision_metric"] if dev else None)
    lines.append("")
    if args.allow_confirm:
        mm_part("confirm", lines, out, dev_metric=dev["decision_metric"] if dev else None)
    else:
        lines.append("== C3-MM (CONFIRM): not opened (no --allow-confirm) ==")
        out["mm_confirm"] = None
    lines.append("")
    adoption(lines, out)
    if args.figure:
        figure(out, os.path.join(os.path.dirname(args.out), "fig_step3_C3_arm"))
    text = "\n".join(lines)
    print(text)
    with open(args.out + ".txt", "w") as f:
        f.write(text + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump(ar._clean(out), f, indent=1)


if __name__ == "__main__":
    main()
