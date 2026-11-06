"""How each model copies each phase of the expert, offline (on the expert's own states): per-phase normalised error and
one validation episode in time (docs/04_evaluation.md, docs/05_arm_ignition.md).

The expert's phases come from the recorded diagnostics (never a model input): align (turn in place), drive, orient
(final turn at the dock) and reach (navigation done: wheels exactly zero, the arm reaches). Errors are RMSE / training
std of each output, averaged over the two wheels and over the seven torques separately.

  phase_errors.png   per-phase wheel and arm NRMSE of each model (nominal validation episodes)
  phase_trace.png    one episode: expert command vs each model's prediction, right wheel and arm torque 2

Usage (RoboEnv python; reads the trained model files and the nominal dataset):
  python scripts/figures/phase_errors.py OUT_DIR SPLIT.json LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--episodes 40]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robot_learning.learning.episodes import load_episode  # noqa: E402
from robot_learning.learning.gru import GRUPolicy  # noqa: E402
from robot_learning.learning.mlp import MLPPolicy  # noqa: E402
from robot_learning.learning.transformer import TransformerPolicy  # noqa: E402

PHASES = ["align", "drive", "orient", "reach"]
PHASE_COLOUR = {"align": "#f3d9a4", "drive": "#cfe2f3", "orient": "#f4c7c3", "reach": "#d9ead3"}
COLOURS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
LOADERS = {"egocentric_mlp_regressor": MLPPolicy, "gru_regressor": GRUPolicy, "transformer_regressor": TransformerPolicy}


def phases_of(directory, record):
    with np.load(Path(directory) / "episodes" / f"{record['episode_id']}.npz", allow_pickle=False) as d:
        return np.where(d["base_navigation_done"], "reach", d["expert_phase"]), d["time"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("split"); ap.add_argument("models", nargs="+")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    directory = Path(split["dataset"])
    records = [r for r in split["validation"] if "dataset" not in r and r.get("success") is True][:args.episodes]
    models = []
    for item in args.models:
        label, run = item.split("=", 1)
        kind = json.loads((Path(run) / "model_info.json").read_text())["kind"]
        models.append((label, LOADERS[kind].load(Path(run) / "model.npz")))
    sq = {label: {p: [np.zeros(9), 0] for p in PHASES} for label, _ in models}
    trace = None
    for n, record in enumerate(records):
        x, y, _ = load_episode(directory, record)
        phase, t = phases_of(directory, record)
        preds = {label: m.predict(x, t) for label, m in models}
        for label, m in models:
            e = (preds[label] - y) / m.y_scale
            for p in PHASES:
                on = phase == p
                sq[label][p][0] += (e[on] ** 2).sum(axis=0); sq[label][p][1] += on.sum()
        if n == 0:
            trace = (record["episode_id"], t, y, phase, preds)
        print(f"  {n + 1}/{len(records)} episodes", flush=True)
    # per-phase NRMSE (pooled rows of each phase), wheels and arm
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=False)
    width = 0.8 / len(models)
    for k, (label, _) in enumerate(models):
        wheel = [np.sqrt(sq[label][p][0][:2] / max(sq[label][p][1], 1)).mean() for p in PHASES]
        arm = [np.sqrt(sq[label][p][0][2:] / max(sq[label][p][1], 1)).mean() for p in PHASES]
        for ax, v in zip(axes, (wheel, arm)):
            ax.bar(np.arange(4) + (k - (len(models) - 1) / 2) * width, v, width, color=COLOURS[k], label=label)
    rows = {p: sq[models[0][0]][p][1] for p in PHASES}
    for ax, title in zip(axes, ("wheels (right, left)", "arm (seven torques)")):
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{p}\n{rows[p] / sum(rows.values()) * 100:.0f} % of rows" for p in PHASES])
        ax.set_ylabel("NRMSE (RMSE / training std)"); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Offline error per expert phase, on the expert's own states ({len(records)} nominal validation episodes)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "phase_errors.png", dpi=150); plt.close(fig)
    # one episode in time
    eid, t, y, phase, preds = trace
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    for ax, col, name, unit in ((axes[0], 0, "right wheel command", "rad/s"), (axes[1], 3, "arm torque 2", "N·m")):
        lo, hi = min(y[:, col].min(), min(p[:, col].min() for p in preds.values())), max(y[:, col].max(), max(p[:, col].max() for p in preds.values()))
        for p in PHASES:
            ax.fill_between(t, lo, hi, where=phase == p, color=PHASE_COLOUR[p], step="mid", lw=0, label=p)
        ax.plot(t, y[:, col], color="k", lw=2.2, label="expert (label)")
        for k, (label, _) in enumerate(models):
            ax.plot(t, preds[label][:, col], color=COLOURS[k], lw=1.0, label=label)
        ax.set_ylabel(f"{name} ({unit})"); ax.set_ylim(lo, hi)
    axes[0].legend(fontsize=7, ncol=5, frameon=False, loc="upper right")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Open loop on the expert's states ({eid}): every model copies every phase closely here; the closed loop is "
                 "where they differ", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "phase_trace.png", dpi=150); plt.close(fig)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
