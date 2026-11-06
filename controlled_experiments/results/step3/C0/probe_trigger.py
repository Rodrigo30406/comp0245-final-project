"""Offline probe (review only): does the arm head of ego respond to the stillness cue that ego-novw cannot see?

Validation cache (results/step3/cache/val, 1 kHz, 284 episodes), no simulation. For each episode with nav_done:
  k0 = first nav_done step; hold baseline = step k0 - 1 (base rotating at the end of orient, arm at hold).
Frame sets:
  POST_STILL: nav_done, base still (|v|<0.01, |w|<0.02), arm within 0.06 rad of its t=0 posture
  POST_STILL_Q0: POST_STILL and max |qd_arm| < 0.05 rad/s (arm not yet moving: the frame that must *initiate* the reach)
  PRE_LOOK: not nav_done, ring < 0.2, |beta| < 0.1, arm at hold, base moving (the ambiguous look-alike)
  FROZEN (synthetic EM steady state): base pose and target of a POST_STILL frame, arm q = q[k0-1], qd_arm = 0, v=w=0
Response score r = <pred - pred_hold, label - label_hold> / ||label - label_hold||^2 with the episode's hold baseline
(k0 - 1) for pred and label: r ~ 1 the model reproduces the trigger's torque jump, r ~ 0 it keeps holding.
Counterfactual for ego: same frame with (v_mes, w_mes) replaced (POST_STILL -> moving values of k0-1; PRE_LOOK -> 0).
"""
import os
import os
import sys

import numpy as np

TASK = os.path.dirname(os.path.abspath(__file__))  # the task folder; also right when C3_probe.py exec()s this code
while not os.path.exists(os.path.join(TASK, "bc_features.py")) and TASK != os.path.dirname(TASK):
    TASK = os.path.dirname(TASK)
sys.path.insert(0, TASK)
import bc_features as bf  # noqa: E402
from robotnik_arm_task import STANDOFF_POINTS, STANDOFF_Z_POINTS  # noqa: E402

C = os.path.join(TASK, "results/step3/cache/val/")
L = lambda n: np.load(C + n + ".npy", mmap_mode="r")
ep = np.asarray(L("episode")); nav = np.asarray(L("nav_done")).astype(bool)
xy = np.asarray(L("base_xy"), dtype=float); yaw = np.asarray(L("base_bearing"), dtype=float)
q = np.asarray(L("q_arm"), dtype=float); qd = np.asarray(L("qd_arm"), dtype=float)
qdw = np.asarray(L("qd_wheels"), dtype=float); tgt = np.asarray(L("target"), dtype=float)
tau = np.asarray(L("label_tau"), dtype=float); noisy = np.asarray(L("noisy")).astype(bool)
vm, wm = bf.wheels_to_vw(qdw)
dx, dy = bf.ego_target(xy, yaw, tgt[:, :2]); beta = np.arctan2(dy, dx)
ring = np.abs(np.hypot(dx, dy) - np.interp(tgt[:, 2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
still = (np.abs(vm) < 0.01) & (np.abs(wm) < 0.02)

starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]); ends = np.r_[starts[1:], ep.size]
q0 = np.empty_like(q); k0_of = np.full(ep.size, -1)
episodes = []
for s, e in zip(starts, ends):
    q0[s:e] = q[s]
    k = np.flatnonzero(nav[s:e])
    if k.size and k[0] > 0:
        k0 = s + k[0]; k0_of[s:e] = k0; episodes.append((s, e, k0))
qdisp = np.abs(q - q0).max(axis=1)
hold = qdisp < 0.06
post_still = nav & still & hold & (k0_of > 0)
post_still_q0 = post_still & (np.abs(qd).max(axis=1) < 0.05)
pre_look = (~nav) & (ring < 0.2) & (np.abs(beta) < 0.10) & hold & ~still & (k0_of > 0)

# Delay nav_done -> base still, and how long the arm is still at hold & qd ~ 0 after nav_done
d_still, n_ps, n_ps0, lab_jump = [], [], [], []
for s, e, k0 in episodes:
    j = np.flatnonzero(still[k0:e])
    d_still.append(j[0] if j.size else np.nan)
    n_ps.append(post_still[s:e].sum()); n_ps0.append(post_still_q0[s:e].sum())
    lab_jump.append(np.linalg.norm(tau[k0] - tau[k0 - 1]))
d_still, n_ps, n_ps0 = np.array(d_still, float), np.array(n_ps), np.array(n_ps0)
print(f"episodes with nav_done: {len(episodes)}")
print(f"nav_done -> base still: median {np.nanmedian(d_still):.0f} ms, p10 {np.nanpercentile(d_still, 10):.0f}, "
      f"p90 {np.nanpercentile(d_still, 90):.0f}; never still {np.isnan(d_still).sum()}")
print(f"POST_STILL frames {post_still.sum()} (episodes with >= 1: {(n_ps > 0).sum()}; per episode median {np.median(n_ps):.0f}, "
      f"p90 {np.percentile(n_ps, 90):.0f}); with qd_arm ~ 0: {post_still_q0.sum()} (episodes with >= 1: {(n_ps0 > 0).sum()})")
print(f"label torque jump at nav_done ||tau[k0] - tau[k0-1]||: median {np.median(lab_jump):.2f} N m, "
      f"p10 {np.percentile(lab_jump, 10):.2f}, p90 {np.percentile(lab_jump, 90):.2f}")
print(f"PRE_LOOK frames {pre_look.sum()}")

# Training-row footprint (stride-9 rows of the training cache)
CT = os.path.join(TASK, "results/step3/cache/train/")
T = lambda n: np.load(CT + n + ".npy", mmap_mode="r")
tep = np.asarray(T("episode")); tnav = np.asarray(T("nav_done")).astype(bool)
tq = np.asarray(T("q_arm"), dtype=float); tqd = np.asarray(T("qd_arm"), dtype=float)
tv, tw = bf.wheels_to_vw(np.asarray(T("qd_wheels"), dtype=float))
ts = np.flatnonzero(np.r_[True, tep[1:] != tep[:-1]]); te = np.r_[ts[1:], tep.size]
tq0 = np.repeat(tq[ts], te - ts, axis=0)
thold = np.abs(tq - tq0).max(axis=1) < 0.06
tstill = (np.abs(tv) < 0.01) & (np.abs(tw) < 0.02)
m1 = tnav & tstill & thold; m2 = m1 & (np.abs(tqd).max(axis=1) < 0.05)
n_eps = ts.size
print(f"TRAIN rows {tep.size} in {n_eps} episodes: nav_done & still & arm at hold {m1.sum()} ({m1.sum()/n_eps:.2f}/episode, "
      f"{100*m1.mean():.3f}% of rows); ... and qd_arm ~ 0: {m2.sum()} ({m2.sum()/n_eps:.2f}/episode); "
      f"rows still & at hold & NOT nav_done: {(tstill & thold & ~tnav).sum()}; nav_done rows total {tnav.sum()}")
del tq, tqd, tq0


def feats(fs, idx, v=None, w=None, qa=None, qda=None):
    sig = {"base_xy": xy[idx], "yaw": yaw[idx], "q_arm": q[idx] if qa is None else qa,
           "qd_arm": qd[idx] if qda is None else qda, "v_mes": vm[idx] if v is None else v,
           "w_mes": wm[idx] if w is None else w, "target": tgt[idx]}
    return bf.build_features(fs, sig)


def score(pred, pred_h, lab, lab_h):
    dl = lab - lab_h; dp = pred - pred_h
    den = (dl * dl).sum(1)
    ok = den > 1e-6
    r = (dp * dl).sum(1)[ok] / den[ok]
    return r, np.linalg.norm(dp, axis=1)[ok], np.sqrt(den[ok])


idx_ps = np.flatnonzero(post_still); idx_ps0 = np.flatnonzero(post_still_q0); idx_pre = np.flatnonzero(pre_look)
h_ps, h_ps0, h_pre = k0_of[idx_ps] - 1, k0_of[idx_ps0] - 1, k0_of[idx_pre] - 1
# FROZEN: base pose/target of each POST_STILL frame, arm at the hold step's q, qd = 0, v = w = 0
zero = np.zeros(idx_ps.size)
print()
print("model      set              n      r median  r IQR            |dpred| med  |dlabel| med   frac r>0.5")
for name in ["ego_s0", "ego_s1", "ego_s2", "ego_s3", "ego_s4",
             "ego-novw_s0", "ego-novw_s1", "ego-novw_s2", "ego-novw_s3", "ego-novw_s4"]:
    m = bf.load_model(os.path.join(TASK, "trained", name + ".npz")); fs = m["feature_set"]
    P = lambda X: bf.forward(m, X)[:, 2:]
    for label, idx, h in (("POST_STILL", idx_ps, h_ps), ("POST_STILL_qd0", idx_ps0, h_ps0), ("PRE_LOOK", idx_pre, h_pre)):
        if idx.size == 0:
            print(f"{name:11s} {label:15s}      0  (no frames)"); continue
        r, dp, dl = score(P(feats(fs, idx)), P(feats(fs, h)), tau[idx], tau[h])
        print(f"{name:11s} {label:15s} {r.size:6d}  {np.median(r):+8.3f}  [{np.percentile(r,25):+.3f}, {np.percentile(r,75):+.3f}]"
              f"   {np.median(dp):8.2f}    {np.median(dl):8.2f}      {np.mean(r > 0.5):.3f}")
    # FROZEN EM steady state, against the first reach label tau[k0] and the hold label tau[k0-1]
    k0s = k0_of[idx_ps]
    Xf = feats(fs, idx_ps, v=zero, w=zero, qa=q[k0s - 1], qda=np.zeros((idx_ps.size, 7)))
    r, dp, dl = score(P(Xf), P(feats(fs, k0s - 1)), tau[k0s], tau[k0s - 1])
    print(f"{name:11s} {'FROZEN(EM)':15s} {r.size:6d}  {np.median(r):+8.3f}  [{np.percentile(r,25):+.3f}, {np.percentile(r,75):+.3f}]"
          f"   {np.median(dp):8.2f}    {np.median(dl):8.2f}      {np.mean(r > 0.5):.3f}")
    if fs == "ego":
        # counterfactual: POST_STILL with the moving (v, w) of the hold step -> does ego revert to hold?
        Xa = feats(fs, idx_ps); Xb = feats(fs, idx_ps, v=vm[h_ps], w=wm[h_ps])
        r, dp, dl = score(P(Xa), P(Xb), tau[idx_ps], tau[h_ps])
        print(f"{name:11s} {'CF still-moving':15s} {r.size:6d}  {np.median(r):+8.3f}  [{np.percentile(r,25):+.3f}, {np.percentile(r,75):+.3f}]"
              f"   {np.median(dp):8.2f}    {np.median(dl):8.2f}      {np.mean(r > 0.5):.3f}   (pred(actual) - pred(v,w of k0-1))")
        # counterfactual: PRE_LOOK frames with v = w = 0 -> does ego start the reach? (label jump of the episode)
        z = np.zeros(idx_pre.size)
        k0p = k0_of[idx_pre]
        r, dp, dl = score(P(feats(fs, idx_pre, v=z, w=z)), P(feats(fs, idx_pre)), tau[k0p], tau[k0p - 1])
        print(f"{name:11s} {'CF pre v,w=0':15s} {r.size:6d}  {np.median(r):+8.3f}  [{np.percentile(r,25):+.3f}, {np.percentile(r,75):+.3f}]"
              f"   {np.median(dp):8.2f}    {np.median(dl):8.2f}      {np.mean(r > 0.5):.3f}   (pred(v,w=0) - pred(actual) vs episode jump)")
        # FROZEN with the moving (v, w) of k0-1 instead of 0: the stillness effect in the EM steady state
        Xm = feats(fs, idx_ps, v=vm[k0s - 1], w=wm[k0s - 1], qa=q[k0s - 1], qda=np.zeros((idx_ps.size, 7)))
        r, dp, dl = score(P(Xf), P(Xm), tau[k0s], tau[k0s - 1])
        print(f"{name:11s} {'CF FROZEN v,w':15s} {r.size:6d}  {np.median(r):+8.3f}  [{np.percentile(r,25):+.3f}, {np.percentile(r,75):+.3f}]"
              f"   {np.median(dp):8.2f}    {np.median(dl):8.2f}      {np.mean(r > 0.5):.3f}   (frozen v,w=0 minus frozen v,w moving)")
