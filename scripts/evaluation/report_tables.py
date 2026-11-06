"""Regenerate the reaching tables of docs/results.md from the evaluation summaries (results/evaluations/) and the trained
models (weights/). No simulation and no dataset needed: every cell is recomputed from the summary.json files of the
closed-loop evaluations and the model_info.json files of the training runs.

  python3 scripts/evaluation/report_tables.py [EVALUATIONS_DIR] [WEIGHTS_DIR]     (defaults: results/evaluations weights)

Cells: mean ± sd (min–max) over the seeds found (stable arrivals of 205; sample sd, as seed_stats.py).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

EV = Path(sys.argv[1] if len(sys.argv) > 1 else "results/evaluations")
WE = Path(sys.argv[2] if len(sys.argv) > 2 else "weights")
BANKS = {"nominal": "clean_validation", "random": "e1v2_random_validation"}


def counts(dirs, prefix, tol):
    """{seed: stable arrivals} of the policies prefix_seedN in the given evaluation directories."""
    out = {}
    for d in dirs:
        p = EV / d / "summary.json"
        if not p.exists():
            continue
        for name, v in json.loads(p.read_text()).items():
            m = re.fullmatch(re.escape(prefix) + r"(?:_seed(\d+))?", name)   # single-seed runs carry no suffix: seed 0
            seed = int(m.group(1) or 0) if m else None
            if m and seed not in out:
                out[seed] = v["success_by_tolerance"][tol]
    return out


def cell(values, spread=True):
    v = np.array(list(values.values()), float)
    if not len(v):
        return "—"
    if len(v) == 1:
        return f"{v[0]:.0f}"
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" + (f" ({v.min():.0f}–{v.max():.0f})" if spread else "")


def nrmse(run_prefix):
    vals = [json.loads((d / "model_info.json").read_text())["best_validation_episode_macro_balanced_nrmse"]
            for d in sorted(WE.glob(run_prefix + "_seed*")) if (d / "model_info.json").exists()]
    return f"{np.mean(vals):.3f}" if vals else "—"


def evals(family, bank):
    """Evaluation directories of a family on a bank (seeds 0-2 and the later seeds 3-4)."""
    b = BANKS[bank]
    return sorted(p.name for p in EV.glob(f"v2_{family}_on_{b}_*")) + sorted(p.name for p in EV.glob(f"v2_{family}_s34_on_{b}_*"))


def table(rows, header):
    print("| " + " | ".join(header) + " |"); print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


# 1.1 main table
print("### 1.1 Main table (five seeds, stable arrivals of 205)\n")
main = [("MLP-Base", "mlp22raw_control", "mlp22raw_control"), ("MLP-Ego", "mlp28_control", "mlp28_control"),
        ("MLP-Ego-IC", "mlp28_trig144", "mlp28_trig144"), ("MLP-Ego-IC + DB", "mlp28_trig144_db045", "mlp28_trig144"),
        ("MLP-Ego-Cat", "mlp28_cat41_trig144", "mlp28_cat41_trig144"), ("GRU-Ego", "gru28_patience40", "gru28_patience40"),
        ("TF-Ego", "tf28_p40", "tf28_p40")]
rows = []
for label, family, policy in main:
    n, r = evals(family, "nominal"), evals(family, "random")
    rows.append([label, cell(counts(n, policy, "0.01")), cell(counts(n, policy, "0.02")), cell(counts(n, policy, "0.05")),
                 cell(counts(r, policy, "0.01")), cell(counts(r, policy, "0.05")), nrmse("v2_" + policy)])
table(rows, ["model", "nominal 1 cm", "nominal 2 cm", "nominal 5 cm", "random start 1 cm", "random start 5 cm", "offline NRMSE"])

# 1.2 data controls
print("### 1.2 Data controls\n")
ctrl = [("full mixture, DART (MLP-Ego-IC)", "mlp28_trig144"), ("without DART, MLP-Ego-IC", "mlp28clean_trig144"),
        ("without DART, MLP-Ego", "mlp28clean_control"), ("nominal starts only, MLP-Ego-IC", "mlp28nom_trig144")]
rows = []
for label, family in ctrl:
    n, r = evals(family, "nominal"), evals(family, "random")
    rows.append([label, cell(counts(n, family, "0.01")), cell(counts(n, family, "0.05")), cell(counts(r, family, "0.01")),
                 cell(counts(r, family, "0.05")), nrmse("v2_" + family)])
table(rows, ["training data", "nominal 1 cm", "nominal 5 cm", "random start 1 cm", "random start 5 cm", "offline NRMSE"])

print("Data design validation, balanced coarser grids (seed 0):\n")
rows = []
for n in (128, 256, 512):
    nb = counts(sorted(p.name for p in EV.glob("v2_mlp28_trig144_bal_on_clean_validation_*")), f"mlp28_trig144_bal{n}", "0.01")
    rb = counts(sorted(p.name for p in EV.glob("v2_mlp28_trig144_bal_on_e1v2_random_validation_*")), f"mlp28_trig144_bal{n}", "0.01")
    rows.append([f"{n} targets", cell(nb), cell(rb), nrmse(f"v2_mlp28_trig144_bal{n}")])
table(rows, ["coverage", "nominal 1 cm", "random start 1 cm", "offline NRMSE"])
print("Random subsets of the episodes (seed 0):\n")
rows = []
for c in (400, 800, 1600, 2400):
    nb = counts(sorted(p.name for p in EV.glob("v2_mlp28_trig144_caps_on_clean_validation_*")), f"mlp28_trig144_cap{c}", "0.01")
    rb = counts(sorted(p.name for p in EV.glob("v2_mlp28_trig144_caps_on_e1v2_random_validation_*")), f"mlp28_trig144_cap{c}", "0.01")
    rows.append([f"{c} episodes", cell(nb), cell(rb), nrmse(f"v2_mlp28_trig144_cap{c}")])
table(rows, ["training episodes", "nominal 1 cm", "random start 1 cm", "offline NRMSE"])
print("Ignition copies K (seed 0) and raw inputs with copies:\n")
rows = []
for label, family in (("K 16", "mlp28_trig16"), ("K 512", "mlp28_trig512"), ("MLP-Base + K 144", "mlp22raw_trig144")):
    rows.append([label, cell(counts(evals(family, "nominal"), family, "0.01")), cell(counts(evals(family, "random"), family, "0.01")),
                 nrmse("v2_" + family)])
table(rows, ["model", "nominal 1 cm", "random start 1 cm", "offline NRMSE"])

# 1.3 generalisation, 1.4 robustness: seeds 0 / 1 / 2 per cell
def per_seed(d, prefix):
    out = []
    for tol in ("0.01", "0.05"):
        c = counts([d], prefix, tol)
        out.append(" / ".join(str(c[s]) for s in sorted(c)) if c else "—")
    return "; ".join(out)


def expert(d):
    p = EV / d / "summary.json"
    return str(json.loads(p.read_text())["expert"]["success_by_tolerance"]["0.01"]) if p.exists() else "—"


print("### 1.3 Generalisation (seeds 0 / 1 / 2; 1 cm; 5 cm)\n")
rows = []
for bank in ("far_targets", "target_height", "far_start"):
    r = [bank, expert(f"ood_{bank}_mlp28_trig144_20261103")]
    for group, prefix in (("mlp28_trig144", "mlp28_trig144"), ("mlp28_trig144_db045", "mlp28_trig144"),
                          ("mlp28_cat41_trig144", "mlp28_cat41_trig144"), ("gru28_patience40", "gru28_patience40")):
        r.append(per_seed(f"ood_{bank}_{group}_20261103", prefix))
    rows.append(r)
table(rows, ["bank", "expert 1 cm", "MLP-Ego-IC", "+ DB", "MLP-Ego-Cat", "GRU-Ego"])

print("### 1.4 Robustness (seeds 0 / 1 / 2; 1 cm; 5 cm)\n")
adv = [("left wheels at 95 %", "gain_left095", "20261103"), ("payload 0.1 kg, last link", "payload0p1kg_link7", "20261104"),
       ("payload 0.25 kg", "payload0p25kg_link7", "20261104"), ("payload 0.5 kg", "payload0p5kg_link7", "20261104"),
       ("observation noise 1/10", "obsnoise_tenth", "20261104"), ("observation noise 1/2", "obsnoise_half", "20261104"),
       ("observation noise, full", "obsnoise", "20261103"), ("command delay 2 ms", "delay2ms", "20261104"),
       ("command delay 5 ms", "delay5ms", "20261104"), ("command delay 50 ms", "delay50ms", "20261103")]
rows = []
for label, key, date in adv:
    rows.append([label, expert(f"adv_{key}_mlp28_trig144_{date}"), per_seed(f"adv_{key}_mlp28_trig144_{date}", "mlp28_trig144"),
                 per_seed(f"adv_{key}_gru28_patience40_{date}", "gru28_patience40")])
table(rows, ["adversity", "expert 1 cm", "MLP-Ego-IC", "GRU-Ego"])
