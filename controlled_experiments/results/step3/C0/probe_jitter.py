import numpy as np
import os as _os
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "probe_trigger.py")).read().split("# Training-row footprint")[0])
qmax = np.abs(qd).max(axis=1)
pre = (~nav) & hold & (k0_of > 0)
# hold-phase jitter of the arm velocity (before nav_done)
print("arm |qd| max during hold before nav_done: median %.3f p90 %.3f p99 %.3f" % tuple(np.percentile(qmax[pre], [50, 90, 99])))
print("arm |qd| max at k0-1: median %.3f, p90 %.3f" % tuple(np.percentile(qmax[[k0-1 for _,_,k0 in episodes]], [50, 90])))
for thr in (0.05, 0.1, 0.15, 0.2, 0.3):
    m = post_still & (qmax < thr)
    print(f"POST_STILL with max|qd_arm| < {thr}: {m.sum()} frames in {len(np.unique(ep[m]))} episodes")
# delay nav_done -> still vs nav_done -> arm qd > 0.2 per episode
d1, d2 = [], []
for s, e, k0 in episodes:
    j = np.flatnonzero(still[k0:e]); d1.append(j[0] if j.size else np.nan)
    j = np.flatnonzero(qmax[k0:e] > 0.2); d2.append(j[0] if j.size else np.nan)
d1, d2 = np.array(d1), np.array(d2)
print(f"per episode: still before arm qd > 0.2 rad/s in {(d1 < d2).sum()} of {len(d1)}; delay still - arm-moving median {np.nanmedian(d1 - d2):.0f} ms")
