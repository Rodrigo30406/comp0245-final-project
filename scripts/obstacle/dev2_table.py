"""Summarise the obstacle-task (step 9) dev2 evaluations of this project's policies: per family mean +- sd and range
over seeds of the harness metrics (protocol v2.0 rollouts.csv), plus a failure anatomy per family.

  python scripts/obstacle/dev2_table.py [--runs runs/obstacle] [--eval-set dev2]
Families are the run directories runs/obstacle/step9_<tag>_seed<k>_<set>; the expert and other single runs are listed
alone. Nothing is recomputed from the simulation: every number comes from the harness's per-episode rows.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"expert": "expert v2 (recorded clean)", "mlp28lidar_trig144": "MLP-Ego-IC + lidar (K=144)",
          "gru28lidar_patience40": "GRU-Ego + lidar", "tf28lidar_p40": "TF-Ego + lidar (700-episode cap)"}


def pct(x, n=300):
    return 100.0 * x / n


def fmt(values, n=300):
    v = np.asarray([pct(x, n) for x in values], dtype=float)
    if len(v) == 1:
        return f"{v[0]:.1f}"
    lo, hi = v.min(), v.max()
    return f"{v.mean():.0f}$\\pm${v.std(ddof=1):.0f} ({lo:.0f}--{hi:.0f})"


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "1.0")


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metrics(rows):
    n = len(rows)
    ok = [truthy(r["success"]) for r in rows]
    band = [r for r in rows if truthy(r["ambiguous"])]
    out = [r for r in rows if not truthy(r["ambiguous"])]
    stops = {}
    for r in rows:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    fail_err = [1000 * float(r["min_error"]) for r, s in zip(rows, ok) if not s and r["min_error"] not in ("", "nan")]
    return {
        "n": n,
        "success": sum(ok),
        "stable_1cm": sum(r["stable_1cm_time"] not in ("", "nan") for r in rows),
        "stable_2cm": sum(r["stable_2cm_time"] not in ("", "nan") for r in rows),
        "stable_5cm": sum(r["stable_5cm_time"] not in ("", "nan") for r in rows),
        "docked_clean": sum(truthy(r["docked_clean"]) for r in rows),
        "collision": sum(truthy(r["collision"]) for r in rows),
        "same_side": sum(r["side_taken"] == r["scenario_side"] for r in rows),
        "band_success": (sum(truthy(r["success"]) for r in band), len(band)),
        "out_success": (sum(truthy(r["success"]) for r in out), len(out)),
        "stop": stops,
        "min_error_fail_median_mm": float(np.median(fail_err)) if fail_err else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs/obstacle"))
    ap.add_argument("--eval-set", default="dev2")
    args = ap.parse_args()
    families = defaultdict(dict)
    for d in sorted(Path(args.runs).glob(f"step9_*_{args.eval_set}")):
        csv_path = d / "rollouts.csv"
        if not csv_path.exists():
            continue
        tag = d.name[len("step9_"):-len(args.eval_set) - 1]
        m = re.match(r"(.*)_seed(\d+)$", tag)
        fam, seed = (m.group(1), int(m.group(2))) if m else (tag, 0)
        families[fam][seed] = read_rows(csv_path)
    print(f"{'policy':34s} seeds  success  stable1cm  stable2cm  stable5cm  docked_clean  collision  same_side  band/outside")
    for fam, runs in families.items():
        seeds = sorted(runs)
        ms = [metrics(runs[s]) for s in seeds]
        n = ms[0]["n"]
        row = [LABELS.get(fam, fam), ",".join(map(str, seeds))]
        for key in ["success", "stable_1cm", "stable_2cm", "stable_5cm", "docked_clean", "collision", "same_side"]:
            row.append(fmt([m[key] for m in ms], n))
        band = fmt([m["band_success"][0] for m in ms], ms[0]["band_success"][1])
        out = fmt([m["out_success"][0] for m in ms], ms[0]["out_success"][1])
        print(f"{row[0]:34s} {row[1]:6s} " + "  ".join(f"{c:>12s}" for c in row[2:]) + f"  {band} / {out}")
    print("\nper seed (counts of", n, "episodes):")
    for fam, runs in families.items():
        for s in sorted(runs):
            m = metrics(runs[s])
            print(f"  {fam}_seed{s}: success {m['success']}, 1/2/5 cm {m['stable_1cm']}/{m['stable_2cm']}/{m['stable_5cm']}, "
                  f"docked_clean {m['docked_clean']}, collision {m['collision']}, same side {m['same_side']}, "
                  f"stops {m['stop']}, failures' min EE error median {m['min_error_fail_median_mm']:.1f} mm")


if __name__ == "__main__":
    main()
