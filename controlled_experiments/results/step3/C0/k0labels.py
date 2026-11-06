import numpy as np
import os as _os
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "probe_trigger.py")).read().split("# Training-row footprint")[0])
lv = np.asarray(L("label_v")); lw = np.asarray(L("label_w"))
K0 = np.array([k0 for _, _, k0 in episodes])
print("label v,w at k0: max |v| %.4f, max |w| %.4f; at k0-1: median |v| %.3f, |w| %.3f" % (np.abs(lv[K0]).max(), np.abs(lw[K0]).max(), np.median(np.abs(lv[K0-1])), np.median(np.abs(lw[K0-1]))))
print("ring dist at k0 median %.3f p90 %.3f max %.3f; |beta| at k0 median %.3f p90 %.3f" % (np.median(ring[K0]), np.percentile(ring[K0],90), ring[K0].max(), np.median(np.abs(beta[K0])), np.percentile(np.abs(beta[K0]),90)))
# look-alikes of the k0 frame for ego-novw: ring < 0.2, |beta| < 0.1, arm at hold, not nav_done, per episode
la = (~nav) & (ring < 0.2) & (np.abs(beta) < 0.1) & hold
print("ego-novw look-alike (hold-labelled) frames per episode at 1 kHz: median %.0f, mean %.0f -> at stride 9 about %.0f rows/episode" % (np.median([la[s:e].sum() for s,e,_ in episodes]), np.mean([la[s:e].sum() for s,e,_ in episodes]), np.mean([la[s:e].sum() for s,e,_ in episodes])/9))
