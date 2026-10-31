import numpy as np, sys
TASK = "/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task"
sys.path.insert(0, TASK); import bc_features as bf
CT = TASK + "/results/step3/cache/train/"
T = lambda n: np.load(CT + n + ".npy", mmap_mode="r")
ep = np.asarray(T("episode")); nav = np.asarray(T("nav_done")).astype(bool); step = np.asarray(T("step"))
q = np.asarray(T("q_arm"), dtype=float); qd = np.asarray(T("qd_arm"), dtype=float)
v, w = bf.wheels_to_vw(np.asarray(T("qd_wheels"), dtype=float))
s0 = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]); e0 = np.r_[s0[1:], ep.size]
q0 = np.repeat(q[s0], e0 - s0, axis=0); hold = np.abs(q - q0).max(1) < 0.06
qm = np.abs(qd).max(1)
for thr in (0.1, 0.15, 0.2):
    m = nav & hold & (qm < thr)
    print(f"train rows nav_done & arm at hold & max|qd_arm| < {thr}: {m.sum()} in {len(np.unique(ep[m]))} of {s0.size} episodes "
          f"({100*m.mean():.4f}% of rows); of them base still: {(m & (np.abs(v)<0.01) & (np.abs(w)<0.02)).sum()}; |w| median {np.median(np.abs(w[m])) if m.any() else np.nan:.3f}")
