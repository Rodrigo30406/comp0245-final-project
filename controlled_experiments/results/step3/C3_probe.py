"""Experiment C3, choice of K (STEP3C_PLAN_PROMPT.md 3.2; pre-registered in report.txt and frozen before any C3 model
was trained): offline probe of the trigger response on the validation cache (results/step3/cache/val, 1 kHz, 284
episodes). No simulation, no DEV / HARD / CONFIRM data.

Frame sets and the score function are those of results/step3/C0/probe_trigger.py (its frame-set code is executed
unchanged, feats() and score() below are copied from it): k0 = first nav_done step of a validation episode.
Baseline (changed from C0, confirmed by the user 2026-10-30 before the freeze): C0 measured the predicted torque change
from the model's OWN prediction at k0 - 1. The trigger copies make a C3 model fire at k0 - 1 too (it lies 0.02
standardised units from k0, inside the jitter cloud of radius 1.31), so that baseline moves with the thing being
calibrated. Here the hold reference is the RECORDED label instead:
  r = <pred - tau[k0 - 1], tau[k0] - tau[k0 - 1]> / ||tau[k0] - tau[k0 - 1]||^2 over the 7 torques
  (0 = the recorded hold torque at k0 - 1, 1 = the recorded first reach torque at k0; the episode's k0).
C0's model-baseline r is kept as a comparison column for gates (i) and (ii).
Gates (both needed; the smallest K of 16, 64, 256 that passes is chosen; all three must have been tested):
  (i)  EM steady state FROZEN(EM): base pose and target of each POST_STILL frame, arm q = q[k0 - 1], qd_arm = 0,
       v = w = 0 (8,923 frames of 141 episodes): median r over the frames >= 0.5.
  (ii) no false trigger far from the ring: validation frames with nav_done False and ring distance > 0.3 m, each scored
       with its episode's tau[k0 - 1] and tau[k0]: median r < 0.1 and 95th percentile < 0.3.
Reported only (no gate): the contamination index r of pred(k0 - 1); PRE_LOOK frames (nav_done False, ring < 0.2 m,
|beta| < 0.1, arm at hold, base moving) in the trigger direction (the tendency to fire early) and with C0's own-label
score; the recorded k0 frame of the validation episodes (held out) and k0 + 1, 2, 5 ms (direction tau[k0 + j] -
tau[k0 - 1], as C0's probe_time); the recorded k0 frames of the training episodes with k0 > 0 (1,611; unjittered: were
the copies fitted?); the per-episode medians of FROZEN(EM); the validation loss and the (v, w) nRMSE on the navigation
rows (nav_done False), next to ego-novw_s0.

  python results/step3/C3_probe.py [model.npz ...]
default models: trained/ego-novw_s0.npz and trained/ego-novw-trig-k{16,64,256}_s0.npz (those that exist).
Output: results/step3/C3_probe.txt and C3_probe.json (gate values and the chosen K).
"""
import json
import os
import sys

import numpy as np

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SELF_TASK = TASK
C0_PROBE = os.path.join(TASK, "results", "step3", "C0", "probe_trigger.py")
# the frame sets of C0/probe_trigger.py, executed unchanged (defines ep, nav, xy, yaw, q, qd, tgt, tau, vm, wm, ring,
# beta, still, k0_of, episodes, hold, post_still, pre_look, ... and prints its header lines). It sets TASK to its own
# hard-coded task folder: it must be this script's folder (else models and caches would come from another checkout).
exec(open(C0_PROBE).read().split("# Training-row footprint")[0])
assert os.path.samefile(TASK, _SELF_TASK), (TASK, _SELF_TASK)
import train_policy as tp  # noqa: E402

K_GRID = (16, 64, 256)
GATE_FROZEN_MEDIAN = 0.5      # (i)  median r >= this
GATE_FAR_MEDIAN = 0.1         # (ii) median r < this
GATE_FAR_P95 = 0.3            # (ii) 95th percentile of r < this
FAR_RING = 0.3                # m, ring distance of the far-from-ring navigation frames


def feats(fs, idx, v=None, w=None, qa=None, qda=None):   # copied from C0/probe_trigger.py
    sig = {"base_xy": xy[idx], "yaw": yaw[idx], "q_arm": q[idx] if qa is None else qa,
           "qd_arm": qd[idx] if qda is None else qda, "v_mes": vm[idx] if v is None else v,
           "w_mes": wm[idx] if w is None else w, "target": tgt[idx]}
    return bf.build_features(fs, sig)


def score(pred, pred_h, lab, lab_h):   # copied from C0/probe_trigger.py
    dl = lab - lab_h; dp = pred - pred_h
    den = (dl * dl).sum(1)
    ok = den > 1e-6
    r = (dp * dl).sum(1)[ok] / den[ok]
    return r, np.linalg.norm(dp, axis=1)[ok], np.sqrt(den[ok])


def stats_of(r):
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return {"n": 0}
    return {"n": int(r.size), "median": float(np.median(r)), "p25": float(np.percentile(r, 25)),
            "p75": float(np.percentile(r, 75)), "p95": float(np.percentile(r, 95)), "p99": float(np.percentile(r, 99)),
            "max": float(r.max()), "frac_gt_0.5": float(np.mean(r > 0.5))}


def _train_pair(job):
    """Recorded steps k0 - 1 and k0 of one training episode (k0 = first nav_done step) as feature signals + torques."""
    with np.load(job["path"]) as d:
        nd = d["nav_done"].astype(bool)
        k0 = int(np.argmax(nd))
        if k0 == 0:
            return None
        idx = np.array([k0 - 1, k0])
        return {"base_xy": d["base_pos"][idx, :2].astype(float), "yaw": d["base_bearing"][idx].astype(float),
                "q_arm": d["q_mes"][idx, 4:11].astype(float), "qd_arm": d["qd_mes"][idx, 4:11].astype(float),
                "qd_wheels": d["qd_mes"][idx, 0:4].astype(float), "target": np.asarray(d["target"], dtype=float),
                "tau": d["torque_arm"][idx].astype(float), "k0": k0}


def train_k0_pairs():
    split = tp.load_split()
    jobs = [{"path": os.path.join(tp.TRAIN_DIR, split["episodes"][str(e)]["file"])} for e in split["train_episodes"]]
    pairs = [p for p in map(_train_pair, jobs) if p is not None]   # sequential: about 30 s
    sig = {}
    for when, j in (("h", 0), ("k0", 1)):
        v, w = bf.wheels_to_vw(np.array([p["qd_wheels"][j] for p in pairs]))
        sig[when] = {"base_xy": np.array([p["base_xy"][j] for p in pairs]), "yaw": np.array([p["yaw"][j] for p in pairs]),
                     "q_arm": np.array([p["q_arm"][j] for p in pairs]), "qd_arm": np.array([p["qd_arm"][j] for p in pairs]),
                     "v_mes": v, "w_mes": w, "target": np.array([p["target"] for p in pairs])}
    tau_h, tau_k = np.array([p["tau"][0] for p in pairs]), np.array([p["tau"][1] for p in pairs])
    return sig, tau_h, tau_k, len(jobs) - len(pairs)


def main(paths):
    lines = []
    out = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    K0 = np.array([k0 for s, e, k0 in episodes])
    ep_index = np.full(ep.size, -1)
    for i, (s, e, k0) in enumerate(episodes):
        ep_index[s:e] = i
    idx_ps = np.flatnonzero(post_still)
    k0s = k0_of[idx_ps]
    idx_far = np.flatnonzero((~nav) & (ring > FAR_RING) & (k0_of > 0))
    idx_pre = np.flatnonzero(pre_look)
    idx_navrows = np.flatnonzero(~nav)
    out(f"C3 probe: validation episodes with k0 {len(episodes)}; frames POST_STILL/FROZEN {idx_ps.size} (in "
        f"{np.unique(ep_index[idx_ps]).size} episodes), FAR (nav_done False, ring > {FAR_RING} m) {idx_far.size} (in "
        f"{np.unique(ep_index[idx_far]).size} episodes), PRE_LOOK {idx_pre.size}; navigation rows {idx_navrows.size}")
    tsig, ttau_h, ttau_k, n_tr_skip = train_k0_pairs()
    out(f"training episodes with k0 > 0 (recorded k0 - 1, k0 pairs): {len(ttau_k)} (k0 = 0: {n_tr_skip}, left out)")
    zero = np.zeros(idx_ps.size)
    results = {}
    for path in paths:
        m = bf.load_model(path)
        fs = m["feature_set"]
        name = os.path.basename(path)[:-4]
        P = lambda X: bf.forward(m, X)  # noqa: E731
        Pt = lambda X: P(X)[:, 2:]  # noqa: E731
        res = {"model": os.path.relpath(path, TASK), "trigger": m["meta"].get("trigger") and
               {k: m["meta"]["trigger"][k] for k in ("copies", "n_rows_added", "qd_sigma", "jitter_seed")},
               "val_loss": m["meta"].get("val_loss"), "epochs_run": m["meta"].get("epochs_run"),
               "best_epoch": m["meta"].get("best_epoch")}
        ph_ep = Pt(feats(fs, K0 - 1))                     # the model's pred(k0 - 1): C0's baseline (comparison only)
        lab_h, lab_k = tau[K0 - 1], tau[K0]               # recorded hold label at k0 - 1, reach label at k0
        # reported: contamination index, r of the model's own pred(k0 - 1)
        r_a, _, _ = score(ph_ep, lab_h, lab_k, lab_h)
        res["k0m1_contamination"] = stats_of(r_a)
        # (i) FROZEN(EM): C0's synthetic state, recorded-label baseline (gate) and C0's model baseline (comparison)
        Xf = feats(fs, idx_ps, v=zero, w=zero, qa=q[k0s - 1], qda=np.zeros((idx_ps.size, 7)))
        Pf = Pt(Xf)
        r_f, _, _ = score(Pf, tau[k0s - 1], tau[k0s], tau[k0s - 1])
        res["frozen"] = stats_of(r_f)
        res["frozen_c0_baseline"] = stats_of(score(Pf, Pt(feats(fs, k0s - 1)), tau[k0s], tau[k0s - 1])[0])
        epi = ep_index[idx_ps][(((tau[k0s] - tau[k0s - 1]) ** 2).sum(1) > 1e-6)]
        res["frozen_per_episode_median"] = stats_of([np.median(r_f[epi == i]) for i in np.unique(epi)])
        # (ii) far from the ring, trigger direction of the episode, recorded-label baseline
        ei = ep_index[idx_far]
        Pfar = Pt(feats(fs, idx_far))
        r_far, _, _ = score(Pfar, lab_h[ei], lab_k[ei], lab_h[ei])
        res["far"] = stats_of(r_far)
        res["far"]["episodes_with_r_gt_0.5"] = int(np.unique(ei[r_far > 0.5]).size) if r_far.size == ei.size else None
        res["far_c0_baseline"] = stats_of(score(Pfar, ph_ep[ei], lab_k[ei], lab_h[ei])[0])
        del Pfar
        # reported: PRE_LOOK in the trigger direction (recorded-label baseline) and with C0's own-label score
        ep_ = ep_index[idx_pre]
        r_pre, _, _ = score(Pt(feats(fs, idx_pre)), lab_h[ep_], lab_k[ep_], lab_h[ep_])
        res["pre_look_trigger_dir"] = stats_of(r_pre)
        r_pre0, _, _ = score(Pt(feats(fs, idx_pre)), Pt(feats(fs, k0_of[idx_pre] - 1)), tau[idx_pre],
                             tau[k0_of[idx_pre] - 1])
        res["pre_look_own_label_c0"] = stats_of(r_pre0)
        # reported: recorded frames k0 + j of the validation episodes (held out), scored against tau[k0 + j] - tau[k0-1]
        res["val_recorded"] = {}
        for j in (0, 1, 2, 5):
            r_j, _, _ = score(Pt(feats(fs, K0 + j)), lab_h, tau[K0 + j], lab_h)
            res["val_recorded"][str(j)] = stats_of(r_j)
        # reported: the recorded training k0 frames (what the copies were made from, unjittered)
        Xh, Xk = bf.build_features(fs, tsig["h"]), bf.build_features(fs, tsig["k0"])
        r_tr, _, _ = score(Pt(Xk), ttau_h, ttau_k, ttau_h)
        res["train_k0"] = stats_of(r_tr)
        # reported: (v, w) nRMSE on the navigation rows (nav_done False) of the validation cache
        X = feats(fs, idx_navrows)
        pv = np.empty((idx_navrows.size, 2))
        for s in range(0, idx_navrows.size, 262144):
            pv[s:s + 262144] = P(X[s:s + 262144])[:, :2]
        del X
        lv = np.column_stack([np.asarray(L("label_v"))[idx_navrows], np.asarray(L("label_w"))[idx_navrows]])
        ys = np.asarray(m["y_scale"], dtype=float)[:2]
        res["nav_vw_nrmse"] = (np.sqrt(np.mean((pv - lv) ** 2, axis=0)) / ys).tolist()
        g1 = res["frozen"]["median"] >= GATE_FROZEN_MEDIAN
        g2 = res["far"]["median"] < GATE_FAR_MEDIAN and res["far"]["p95"] < GATE_FAR_P95
        res["gate_i"], res["gate_ii"], res["passes"] = bool(g1), bool(g2), bool(g1 and g2)
        results[name] = res
        tr = res["trigger"]
        out()
        out(f"== {name}: K {tr['copies'] if tr else '-'}, rows added {tr['n_rows_added'] if tr else 0}, epochs "
            f"{res['epochs_run']} (best {res['best_epoch']}), val loss {res['val_loss']:.6f}, nav (v, w) nRMSE "
            f"{res['nav_vw_nrmse'][0]:.4f}, {res['nav_vw_nrmse'][1]:.4f}")
        f, fa = res["frozen"], res["far"]
        fc, fac, ca = res["frozen_c0_baseline"], res["far_c0_baseline"], res["k0m1_contamination"]
        out(f"  (i)  FROZEN(EM) n {f['n']}: r median {f['median']:+.3f} [IQR {f['p25']:+.3f}, {f['p75']:+.3f}], "
            f"frac r > 0.5 {f['frac_gt_0.5']:.3f}; per-episode medians (n {res['frozen_per_episode_median']['n']}): "
            f"median {res['frozen_per_episode_median']['median']:+.3f} -> gate (median >= {GATE_FROZEN_MEDIAN}) "
            f"{'PASS' if g1 else 'FAIL'}   [C0 model baseline: median {fc['median']:+.3f}]")
        out(f"  (ii) FAR n {fa['n']}: r median {fa['median']:+.3f}, p95 {fa['p95']:+.3f}, p99 {fa['p99']:+.3f}, max "
            f"{fa['max']:+.3f}, frac r > 0.5 {fa['frac_gt_0.5']:.4f} (episodes with such a frame "
            f"{fa['episodes_with_r_gt_0.5']}) -> gate (median < {GATE_FAR_MEDIAN}, p95 < {GATE_FAR_P95}) "
            f"{'PASS' if g2 else 'FAIL'}   [C0 model baseline: median {fac['median']:+.3f}, p95 {fac['p95']:+.3f}]")
        out(f"  reported: contamination r of pred(k0 - 1) (n {ca['n']}): median {ca['median']:+.3f}, p95 "
            f"{ca['p95']:+.3f}, max {ca['max']:+.3f}, frac > 0.5 {ca['frac_gt_0.5']:.3f}")
        p, p0 = res["pre_look_trigger_dir"], res["pre_look_own_label_c0"]
        out(f"  reported: PRE_LOOK n {p['n']} trigger direction r median {p['median']:+.3f}, p75 {p['p75']:+.3f}, "
            f"p95 {p['p95']:+.3f}, frac > 0.5 {p['frac_gt_0.5']:.3f}; own label (C0) median {p0['median']:+.3f}")
        out("  reported: validation recorded k0 + j ms, r median: " + ", ".join(
            f"+{j} {res['val_recorded'][str(j)]['median']:+.3f} (frac > 0.5 {res['val_recorded'][str(j)]['frac_gt_0.5']:.2f})"
            for j in (0, 1, 2, 5)))
        out(f"  reported: training recorded k0 (n {res['train_k0']['n']}) r median {res['train_k0']['median']:+.3f}, "
            f"frac > 0.5 {res['train_k0']['frac_gt_0.5']:.3f}")
    calib = {n: r for n, r in results.items() if r["trigger"] and r["trigger"]["copies"] in K_GRID
             and n == f"ego-novw-trig-k{r['trigger']['copies']}_s0"}
    passing = sorted(r["trigger"]["copies"] for r in calib.values() if r["passes"])
    tested = sorted(r["trigger"]["copies"] for r in calib.values())
    complete = tested == list(K_GRID)
    chosen = passing[0] if passing and complete else None
    out()
    if not complete:
        out(f"REFUSED: the pre-registered grid is K {list(K_GRID)}; tested {tested}: no K chosen")
    out(f"calibration models tested: K {tested}; passing both gates: {passing} -> "
        + (f"chosen K = {chosen} (smallest passing)" if chosen is not None else
           ("NO K passes: stop and propose (STEP3C_PLAN_PROMPT.md 4)" if complete else "no K chosen")))
    with open(os.path.join(TASK, "results", "step3", "C3_probe.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(TASK, "results", "step3", "C3_probe.json"), "w") as fh:
        json.dump({"grid": list(K_GRID), "tested": tested, "complete": complete, "passing": passing, "chosen_k": chosen,
                   "gates": {"frozen_median_min": GATE_FROZEN_MEDIAN, "far_median_max": GATE_FAR_MEDIAN,
                             "far_p95_max": GATE_FAR_P95, "far_ring": FAR_RING}, "models": results}, fh, indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [os.path.join(TASK, "trained", "ego-novw_s0.npz")] + [
            p for p in (os.path.join(TASK, "trained", f"ego-novw-trig-k{k}_s0.npz") for k in K_GRID) if os.path.exists(p)]
    main([os.path.abspath(a) for a in args])
