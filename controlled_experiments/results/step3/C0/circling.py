"""Does circling cost success? Cross-tab of the MM (ego-novw) failure class with the same (seed, episode) in the ME cell
(model base + expert arm): if the expert arm, triggered when the model base reaches the expert parking pose, rescues the
episode, the base passed through the parking pose before it started circling."""
import csv, json, re, numpy as np
from collections import Counter, defaultdict
def rows(p): return list(csv.DictReader(open(p)))
tab = defaultdict(Counter); tot = Counter()
for s in range(5):
    cls = {int(r["episode"]): r["fclass"] for r in rows(f"results/step3/classes_ego-novw_s{s}_dev.csv")}
    mm = {int(r["episode"]): r for r in rows(f"results/ego-novw_s{s}_dev/rollouts.csv")}
    me = {int(r["episode"]): r for r in rows(f"results/B-ME_ego-novw_s{s}_dev/rollouts.csv")}
    for e in mm:
        trig, tt = "none", None
        for line in open(f"results/B-ME_ego-novw_s{s}_dev/logs/ep_{e:04d}.log", errors="replace"):
            if line.startswith("HYBRID_EVENT ") and '"nav_done"' in line:
                d = json.loads(line[len("HYBRID_EVENT "):]); trig, tt = d["trigger"], d["t"]; break
        c = "success" if mm[e]["success"] == "True" else cls[e]
        ok = me[e]["success"] == "True"
        tab[c][(trig, ok)] += 1; tot[c] += 1
print("MM class (ego-novw DEV, 5 seeds x 300)  n   | ME same seed x episode: threshold&success  threshold&fail  timeout&success  timeout&fail")
for c, n in tot.most_common():
    t = tab[c]
    print(f"  {c:22s} {n:5d} | {t[('threshold', True)]:6d} ({100*t[('threshold', True)]/n:5.1f}%)  {t[('threshold', False)]:5d}  "
          f"{t[('timeout', True)]:5d}  {t[('timeout', False)]:5d}  none {t[('none', True)] + t[('none', False)]}")
