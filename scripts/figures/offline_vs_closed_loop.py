"""Offline error vs closed-loop success, one point per trained seed (docs/04_evaluation.md).

x: best validation balanced NRMSE of the training run (model_info.json); y: stable arrivals within 1 cm of the 205
nominal held-out targets (summary.json of the closed-loop evaluation). Families are drawn with their mean.

  python3 scripts/figures/offline_vs_closed_loop.py OUT.png RUNS_DIR "LABEL=RUN_PREFIX:EVAL_DIR[,EVAL_DIR]" [...]
RUN_PREFIX is the training-run name without _seedN (e.g. v2_mlp28_trig144); the policy label inside the evaluation is
RUN_PREFIX without its leading "v2_" plus _seedN.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#bcbd22", "#17becf", "#ff7f0e"]


def main(out, runs, families):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for k, item in enumerate(families):
        label, spec = item.split("=", 1)
        prefix, dirs = spec.split(":", 1)
        arrivals = {}
        for d in dirs.split(","):
            summary = json.loads((Path(d) / "summary.json").read_text())
            for name, v in summary.items():
                if name.startswith(prefix.removeprefix("v2_") + "_seed"):
                    arrivals[int(name.rsplit("seed", 1)[1])] = v["success_by_tolerance"]["0.01"]
        xs, ys = [], []
        for seed, y in sorted(arrivals.items()):
            info = json.loads((Path(runs) / f"{prefix}_seed{seed}" / "model_info.json").read_text())
            xs.append(info["best_validation_episode_macro_balanced_nrmse"]); ys.append(y)
        c = COLOURS[k % len(COLOURS)]
        ax.scatter(xs, ys, color=c, s=22, alpha=.7)
        ax.scatter([np.mean(xs)], [np.mean(ys)], color=c, s=110, marker="D", edgecolor="k", label=f"{label} ({len(xs)} seeds)")
    ax.axhline(205, color="k", lw=.7, ls=":"); ax.text(0.081, 199, "expert 205", fontsize=8)
    ax.set_xlabel("offline error: validation balanced NRMSE (lower = closer to the expert's commands)")
    ax.set_ylabel("closed loop: stable arrivals within 1 cm (of 205)")
    ax.set_title("Offline error does not predict closed-loop success (dots: seeds; diamonds: family means)", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
