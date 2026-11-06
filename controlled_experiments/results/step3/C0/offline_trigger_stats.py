"""Offline (validation cache, no simulation): do the ego / ego-novw arms predict 'reach' torques at the
still + parked-looking frames (all reach-labelled)? alpha = projection of (pred - hold) on (label - hold)."""
import sys, os, numpy as np
sys.path.insert(0, os.getcwd())
import bc_features as bf
from robotnik_arm_task import STANDOFF_Z_POINTS, STANDOFF_POINTS
C = "results/step3/cache/val/"
L = lambda n: np.load(C + n + ".npy")
ep, nav = L("episode"), L("nav_done").astype(bool)
q = L("q_arm").astype(float); qd = L("qd_arm").astype(float); tau = L("label_tau").astype(float)
xy, yaw, tgt = L("base_xy").astype(float), L("base_bearing").astype(float), L("target").astype(float)
qw = L("qd_wheels").astype(float)
dx, dy = bf.ego_target(xy, yaw, tgt[:, :2]); beta = np.arctan2(dy, dx)
ring = np.abs(np.hypot(dx, dy) - np.interp(tgt[:, 2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
eps = np.unique(ep)
first = {}
q0 = np.zeros_like(q); hold = np.full_like(tau, np.nan)
starts = np.searchsorted(ep, eps) if np.all(np.diff(ep) >= 0) else None
assert starts is not None
ends = np.r_[starts[1:], ep.size]
for e, a, b in zip(eps, starts, ends):
    q0[a:b] = q[a]
    k = np.flatnonzero(nav[a:b])
    if k.size:
        k0 = a + k[0]
        hold[a:b] = np.median(tau[max(a, k0 - 50):k0], axis=0)
qdisp = np.abs(q - q0).max(axis=1)
vm, wm = bf.wheels_to_vw(qw)
still = (np.abs(vm) < 0.01) & (np.abs(wm) < 0.02)
look = (ring < 0.2) & (np.abs(beta) < 0.10) & (qdisp < 0.06)
sets = {"still+look (reach)": look & still, "moving+look, not reach": look & ~still & ~nav}
sig = {"base_xy": xy, "yaw": yaw, "q_arm": q, "qd_arm": qd, "v_mes": vm, "w_mes": wm, "target": tgt}
for name, m in sets.items():
    idx = np.flatnonzero(m)
    if idx.size > 20000: idx = np.random.default_rng(0).choice(idx, 20000, replace=False)
    s = {k: v[idx] for k, v in sig.items()}
    d = tau[idx] - hold[idx]; nd = np.einsum("ij,ij->i", d, d)
    ok = np.isfinite(nd) & (nd > 0.5 ** 2)
    print(f"== {name}: {m.sum()} frames ({idx.size} used, {ok.sum()} with |label - hold| > 0.5 N m), "
          f"|label - hold| median {np.sqrt(np.nanmedian(nd)):.2f} N m")
    for fam, fs in (("ego", "ego"), ("ego-novw", "ego_novw")):
        al = []
        for k in range(5):
            mdl = bf.load_model(f"trained/{fam}_s{k}.npz")
            y = bf.forward(mdl, bf.build_features(fs, s))[:, 2:9]
            a = np.einsum("ij,ij->i", y - hold[idx], d) / np.where(nd > 0, nd, np.nan)
            al.append(np.nanmedian(a[ok]) if name.startswith("still") else np.nanmedian(np.sqrt(np.einsum('ij,ij->i', y - hold[idx], y - hold[idx]))))
        lab = "median alpha (0 = hold, 1 = reach label)" if name.startswith("still") else "median |pred - hold| N m"
        print(f"   {fam:9s} {lab}: " + " ".join(f"{x:.2f}" for x in al))

print("== split the still+look frames by arm joint speed (max |qd_arm|) ==")
qdmax = np.abs(qd).max(axis=1)
# time since nav_done for each frame
tsn = np.full(ep.size, np.nan)
for e, a, b in zip(eps, starts, ends):
    k = np.flatnonzero(nav[a:b])
    if k.size: tsn[a:b] = (np.arange(b - a) - k[0]) / 1000.0
for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 1e9)):
    m = look & still & (qdmax >= lo) & (qdmax < hi)
    idx = np.flatnonzero(m)
    if idx.size == 0: print(f"  qd in [{lo},{hi}): 0 frames"); continue
    s = {k: v[idx] for k, v in sig.items()}
    d = tau[idx] - hold[idx]; nd = np.einsum("ij,ij->i", d, d)
    out = []
    for fam, fs in (("ego", "ego"), ("ego-novw", "ego_novw")):
        al = []
        for k in range(5):
            y = bf.forward(bf.load_model(f"trained/{fam}_s{k}.npz"), bf.build_features(fs, s))[:, 2:9]
            al.append(np.nanmedian(np.einsum("ij,ij->i", y - hold[idx], d) / nd))
        out.append(f"{fam} alpha " + " ".join(f"{x:.2f}" for x in al))
    print(f"  max|qd_arm| in [{lo}, {hi}) rad/s: {idx.size} frames in {np.unique(ep[idx]).size} episodes; "
          f"time since nav_done median {np.nanmedian(tsn[idx])*1000:.0f} ms; |label-hold| median {np.sqrt(np.median(nd)):.1f} N m; " + "; ".join(out))
# frames at nav_done itself (first reach frame) and first 5 ms, regardless of still
for w in (1, 5, 10):
    m = (tsn >= 0) & (tsn < w / 1000.0)
    idx = np.flatnonzero(m)
    s = {k: v[idx] for k, v in sig.items()}
    d = tau[idx] - hold[idx]; nd = np.einsum("ij,ij->i", d, d)
    out = []
    for fam, fs in (("ego", "ego"), ("ego-novw", "ego_novw")):
        al = []
        for k in range(5):
            y = bf.forward(bf.load_model(f"trained/{fam}_s{k}.npz"), bf.build_features(fs, s))[:, 2:9]
            al.append(np.nanmedian(np.einsum("ij,ij->i", y - hold[idx], d) / nd))
        out.append(f"{fam} alpha " + " ".join(f"{x:.2f}" for x in al))
    print(f"  first {w} ms after nav_done: {idx.size} frames; still share {still[idx].mean()*100:.0f}%; "
          f"max|qd_arm| median {np.median(qdmax[idx]):.3f}; " + "; ".join(out))

print("== the trigger frame (nav_done step) itself ==")
t0 = np.flatnonzero(tsn == 0)
print(f"  n {t0.size}; |v_mes| median {np.median(np.abs(vm[t0])):.4f} p90 {np.percentile(np.abs(vm[t0]),90):.4f}; "
      f"|w_mes| median {np.median(np.abs(wm[t0])):.4f} p10 {np.percentile(np.abs(wm[t0]),10):.4f}; "
      f"|beta| median {np.degrees(np.median(np.abs(beta[t0]))):.2f} deg; ring median {np.median(ring[t0])*100:.1f} cm; "
      f"max|qd_arm| median {np.median(qdmax[t0]):.3f}; look {look[t0].mean()*100:.0f}%")
pre = look & ~nav
print(f"  baseline before nav_done (look frames): max|qd_arm| median {np.median(qdmax[pre]):.3f} p90 {np.percentile(qdmax[pre],90):.3f}")
first_still = []
for e, a, b in zip(eps, starts, ends):
    k = np.flatnonzero(nav[a:b])
    if not k.size: continue
    st = np.flatnonzero(still[a + k[0]: b])
    first_still.append(st[0] / 1000 if st.size else np.nan)
fs = np.array(first_still)
print(f"  base first 'still' (|v|<0.01, |w|<0.02) after nav_done: median {np.nanmedian(fs)*1000:.0f} ms, p10 {np.nanpercentile(fs,10)*1000:.0f}, p90 {np.nanpercentile(fs,90)*1000:.0f}; never {np.isnan(fs).sum()} of {fs.size}")
for name, m in (("look & arm quiet (qd<p90 baseline)", look & (qdmax < np.percentile(qdmax[pre], 90))),
                ("  ... & |v_mes| < 0.01", look & (qdmax < np.percentile(qdmax[pre], 90)) & (np.abs(vm) < 0.01)),
                ("  ... & |v_mes| < 0.01 & |beta| < 0.05", look & (qdmax < np.percentile(qdmax[pre], 90)) & (np.abs(vm) < 0.01) & (np.abs(beta) < 0.05)),
                ("  ... & still (|v|<0.01,|w|<0.02)", look & (qdmax < np.percentile(qdmax[pre], 90)) & still)):
    print(f"  {name}: {m.sum()} frames, reach-labelled {100*nav[m].mean() if m.sum() else float('nan'):.1f}% ({nav[m].sum()} frames)")

print("== counterfactual at the nav_done frame: same state, but v_mes = w_mes = 0 and qd_arm = 0 (the C1 'still' state) ==")
s = {k: v[t0].copy() for k, v in sig.items()}
d = tau[t0] - hold[t0]; nd = np.einsum("ij,ij->i", d, d)
print(f"  |label - hold| at nav_done: median {np.sqrt(np.median(nd)):.1f} N m (n {t0.size})")
for variant in ("as recorded", "v=w=0", "v=w=0, qd_arm=0"):
    s2 = {k: v.copy() for k, v in s.items()}
    if variant != "as recorded":
        s2["v_mes"][:] = 0; s2["w_mes"][:] = 0
    if "qd_arm" in variant:
        s2["qd_arm"][:] = 0
    out = []
    for fam, fs in (("ego", "ego"), ("ego-novw", "ego_novw")):
        al = []
        for k in range(5):
            y = bf.forward(bf.load_model(f"trained/{fam}_s{k}.npz"), bf.build_features(fs, s2))[:, 2:9]
            a = np.einsum("ij,ij->i", y - hold[t0], d) / nd
            al.append(f"{np.median(a):.2f}/{np.mean(a > 0.5)*100:.0f}%")
        out.append(f"{fam} alpha median/share>0.5: " + " ".join(al))
    print(f"  {variant:16s} " + "; ".join(out))

print("== does the switch to reach torques key on the arm's own motion (qd_arm)? frames 3-10 ms after nav_done ==")
idx = np.flatnonzero((tsn >= 0.003) & (tsn < 0.010))
s = {k: v[idx].copy() for k, v in sig.items()}
d = tau[idx] - hold[idx]; nd = np.einsum("ij,ij->i", d, d)
for variant in ("as recorded", "qd_arm = 0", "v=w=0"):
    s2 = {k: v.copy() for k, v in s.items()}
    if variant == "qd_arm = 0": s2["qd_arm"][:] = 0
    if variant == "v=w=0": s2["v_mes"][:] = 0; s2["w_mes"][:] = 0
    out = []
    for fam, fs in (("ego", "ego"), ("ego-novw", "ego_novw")):
        al = []
        for k in range(5):
            y = bf.forward(bf.load_model(f"trained/{fam}_s{k}.npz"), bf.build_features(fs, s2))[:, 2:9]
            al.append(np.median(np.einsum("ij,ij->i", y - hold[idx], d) / nd))
        out.append(f"{fam} alpha " + " ".join(f"{x:.2f}" for x in al))
    print(f"  {variant:12s} ({idx.size} frames): " + "; ".join(out))
