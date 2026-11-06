"""r (fraction of the trigger torque jump reproduced) vs ms after nav_done on recorded frames, and the 'kick' control:
frozen EM state (base still at a post-done pose, arm q of k0-1, v=w=0) with the recorded arm qd of k0+j."""
import os, sys
import numpy as np
import os as _os
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "probe_trigger.py")).read().split("# Training-row footprint")[0])
def feats(fs, idx, v=None, w=None, qa=None, qda=None):
    sig = {"base_xy": xy[idx], "yaw": yaw[idx], "q_arm": q[idx] if qa is None else qa,
           "qd_arm": qd[idx] if qda is None else qda, "v_mes": vm[idx] if v is None else v,
           "w_mes": wm[idx] if w is None else w, "target": tgt[idx]}
    return bf.build_features(fs, sig)
K0 = np.array([k0 for s, e, k0 in episodes])
# a post-done still pose per episode: first still frame after k0 (median 33 ms)
KS = np.array([k0 + np.flatnonzero(still[k0:e])[0] for s, e, k0 in episodes])
dl = tau[K0] - tau[K0 - 1]; den = (dl * dl).sum(1)
print("ms after nav_done:      " + " ".join(f"{j:>6d}" for j in (0, 1, 2, 3, 5, 10, 20, 33)))
print("|qd_arm| max median:    " + " ".join(f"{np.median(np.abs(qd[K0 + j]).max(1)):6.3f}" for j in (0, 1, 2, 3, 5, 10, 20, 33)))
print("|w_mes| median:         " + " ".join(f"{np.median(np.abs(wm[K0 + j])):6.3f}" for j in (0, 1, 2, 3, 5, 10, 20, 33)))
for name in ["ego_s0", "ego_s1", "ego-novw_s0", "ego-novw_s1"]:
    m = bf.load_model(os.path.join(TASK, "trained", name + ".npz")); fs = m["feature_set"]
    P = lambda X: bf.forward(m, X)[:, 2:]
    ph = P(feats(fs, K0 - 1))
    out = []
    for j in (0, 1, 2, 3, 5, 10, 20, 33):
        # recorded frame k0 + j, scored against the jump of the label at k0+j vs k0-1
        dlj = tau[K0 + j] - tau[K0 - 1]
        r = ((P(feats(fs, K0 + j)) - ph) * dlj).sum(1) / (dlj * dlj).sum(1)
        out.append(np.median(r))
    print(f"{name:11s} recorded r:  " + " ".join(f"{x:+6.2f}" for x in out))
    out = []
    z = np.zeros(K0.size)
    for j in (0, 1, 2, 3, 5, 10, 20, 33):
        X = feats(fs, KS, v=z, w=z, qa=q[K0 - 1], qda=qd[K0 + j])
        r = ((P(X) - ph) * dl).sum(1) / den
        out.append(np.median(r))
    print(f"{name:11s} kick r:      " + " ".join(f"{x:+6.2f}" for x in out) + "   (frozen still pose, arm q of k0-1, qd of k0+j)")
