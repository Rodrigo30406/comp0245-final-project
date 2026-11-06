"""Summarise the obstacle-task (step 9) dev2 evaluations of this project's policies: per family mean +- sd and range
over seeds of the harness metrics (protocol v2.0 rollouts.csv), plus a failure anatomy per family.

  python scripts/obstacle/dev2_table.py [--runs results/obstacle/static_dev2] [--eval-set dev2]
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
LABELS = {"expert": "expert v2 (recorded clean)",
          "mlp28lidar_trig144": "MLP-Ego-IC+lidar, original recordings", "gru28lidar_patience40": "GRU-Ego+lidar, original recordings",
          "tf28lidar_p40": "TF-Ego+lidar, original recordings",
          "mlp28lidarnoclock_trig144": "MLP-Ego-IC+lidar, recorded, no clock",
          "mlp28lidarhold_trig144": "MLP, hold, their DART", "gru28lidarhold_patience40": "GRU, hold, their DART",
          "mlp28lidarholdD_trig144": "MLP, hold, our DART (D)", "gru28lidarholdD_patience40": "GRU, hold, our DART (D)",
          "mlp28lidarholdB_trig144": "MLP-Ego-IC+lidar",
          "gru28lidarholdB_patience40": "GRU-Ego+lidar",
          "mlp28lidarholdF_trig144": "MLP, hold, our DART+base noise+fade (F)",
          "gru28lidarholdF_patience40": "GRU, hold, our DART+base noise+fade (F)"}
# tag suffixes after _seedN that name an evaluation variant of the same model family (appended to the label)
VARIANTS = {"_db045": " +DB 0.45"}


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
        "held_10s": sum(truthy(r["held_10s"]) for r in rows),
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
    ap.add_argument("--runs", default=str(ROOT / "results/obstacle/static_dev2"))
    ap.add_argument("--eval-set", default="dev2")
    args = ap.parse_args()
    families = defaultdict(dict)
    for d in sorted(Path(args.runs).glob(f"step9_*_{args.eval_set}")):
        csv_path = d / "rollouts.csv"
        if not csv_path.exists() or not (d / "summary.txt").exists():   # summary.txt is written last: skip running evaluations
            continue
        tag = d.name[len("step9_"):-len(args.eval_set) - 1]
        m = re.match(r"(.*)_seed(\d+)(.*)$", tag)
        fam, seed = (m.group(1) + m.group(3), int(m.group(2))) if m else (tag, 0)
        families[fam][seed] = read_rows(csv_path)
    print(f"{'policy':44s} seeds  success  held_10s  stable1cm  stable2cm  stable5cm  docked_clean  collision  same_side  band/outside")
    for fam, runs in families.items():
        seeds = sorted(runs)
        ms = [metrics(runs[s]) for s in seeds]
        n = ms[0]["n"]
        base, variant = fam, ""
        for suffix, text in VARIANTS.items():
            if fam.endswith(suffix):
                base, variant = fam[:-len(suffix)], text
        row = [LABELS.get(base, base) + variant, ",".join(map(str, seeds))]
        for key in ["success", "held_10s", "stable_1cm", "stable_2cm", "stable_5cm", "docked_clean", "collision", "same_side"]:
            row.append(fmt([m[key] for m in ms], n))
        band = fmt([m["band_success"][0] for m in ms], ms[0]["band_success"][1])
        out = fmt([m["out_success"][0] for m in ms], ms[0]["out_success"][1])
        print(f"{row[0]:44s} {row[1]:6s} " + "  ".join(f"{c:>12s}" for c in row[2:]) + f"  {band} / {out}")
    print("\nper seed (counts of", n, "episodes):")
    for fam, runs in families.items():
        for s in sorted(runs):
            m = metrics(runs[s])
            print(f"  {fam}_seed{s}: success {m['success']}, held 10 s {m['held_10s']}, 1/2/5 cm {m['stable_1cm']}/{m['stable_2cm']}/{m['stable_5cm']}, "
                  f"docked_clean {m['docked_clean']}, collision {m['collision']}, same side {m['same_side']}, "
                  f"stops {m['stop']}, failures' min EE error median {m['min_error_fail_median_mm']:.1f} mm")


if __name__ == "__main__":
    main()
