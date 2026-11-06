"""Step 11: per-seed P(arm never extended | parked) and companions from rollouts.csv (descriptive read-out for S1,
R1 and H-W-onset). Usage: python results/step11/step11_cond.py LABEL=run1,run2,... [LABEL=...]"""
import csv
import math
import sys

def rates(run):
    rows = list(csv.DictReader(open(f"{run}/rollouts.csv")))
    isnan = lambda x: x in ("", "nan", "NaN", "None") or (not x.strip()) or math.isnan(float(x))
    parked = [r for r in rows if r["parked"] in ("True", "1")]
    never = [r for r in parked if isnan(r["arm_extend_time"])]
    ext = sum(not isnan(r["arm_extend_time"]) for r in rows)
    succ = sum(r["success"] in ("True", "1") for r in rows)
    ms = sorted(float(r["policy_ms_mean"]) for r in rows if not isnan(r["policy_ms_mean"]))
    return len(rows), len(parked), len(never), ext, succ, ms[len(ms) // 2] if ms else float("nan")

for arg in sys.argv[1:]:
    label, runs = arg.split("=", 1)
    per = [rates(r) for r in runs.split(",")]
    cond = [100 * nv / pk if pk else float("nan") for n, pk, nv, ex, sc, ms in per]
    print(f"{label:12s} P(never extended | parked) per seed " + " ".join(f"{c:5.1f}" for c in cond)
          + f"  mean {sum(cond) / len(cond):5.1f}%   [parked n " + "/".join(str(p[1]) for p in per)
          + "]  arm_extended " + "/".join(f"{100 * p[3] / p[0]:.0f}" for p in per)
          + "  success " + "/".join(f"{100 * p[4] / p[0]:.1f}" for p in per)
          + "  policy ms " + "/".join(f"{p[5]:.3f}" for p in per))
