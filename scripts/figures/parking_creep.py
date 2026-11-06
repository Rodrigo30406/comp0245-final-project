"""Where the base parks, how far it creeps and where it restarts driving: expert vs policies, in the goal frame.

Goal frame: origin at the expert's dock (base goal), +x along the required final heading (towards the target), so the
nominal start lies on the negative x axis and the expert's 8 cm stop box is the dashed circle. For each policy and
episode of a nominal-bank evaluation: the first moment the base has been still (< 2 cm/s) for 0.5 s ("parks") and the
first later moment it moves faster than 5 cm/s ("restarts"); grey lines join the two. Green: where the expert's base
stands at the end of each episode.

  python3 scripts/figures/parking_creep.py EVAL_DIR OUT.png LABEL=TRACE_PREFIX [LABEL=TRACE_PREFIX ...]
  e.g. LABEL "MLP-Ego=mlp28_control_seed0" with EVAL_DIR the evaluation that holds traces/mlp28_control_seed0_episode_*.npz
Several evaluation directories: LABEL=EVAL_DIR:TRACE_PREFIX.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HEIGHTS, DISTANCES = [0.1, 0.5, 1.1, 1.25], [0.8, 0.65, 0.65, 0.35]   # expert standoff table (experiment_specs/expert)
BOX = 0.08


def goal_frame(obs):
    target = obs[0, 18:21]
    bearing = float(np.arctan2(target[1], target[0]))
    standoff = float(np.interp(target[2], HEIGHTS, DISTANCES))
    goal = target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
    c, s = np.cos(-bearing), np.sin(-bearing)
    d = obs[:, :2] - goal
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def park_restart(path):
    with np.load(path, allow_pickle=False) as z:
        t, obs = z["time"], z["observation"]
    xy = goal_frame(obs)
    speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T)
    run = 0.0
    for i in range(1, len(t)):
        run = run + (t[i] - t[i - 1]) if speed[i] < 0.02 else 0.0
        if run >= 0.5:
            later = np.flatnonzero(speed[i:] > 0.05)
            return xy[i], (xy[i + later[0]] if len(later) else None), xy[-1]
    return None, None, xy[-1]


def main(argv):
    if len(argv) < 4:
        print(__doc__); return 2
    default_eval, out, specs = Path(argv[1]), Path(argv[2]), argv[3:]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.6 * len(specs), 3.7), sharey=True, squeeze=False)
    for ax, spec in zip(axes[0], specs):
        label, rest = spec.split("=", 1)
        eval_dir, prefix = (Path(rest.rsplit(":", 1)[0]), rest.rsplit(":", 1)[1]) if ":" in rest else (default_eval, rest)
        expert_end = np.array([park_restart(p)[2] for p in sorted(eval_dir.glob("traces/expert_episode_*.npz"))])
        parks, restarts, stays = [], [], []
        for p in sorted(eval_dir.glob(f"traces/{prefix}_episode_*.npz")):
            park, restart, _ = park_restart(p)
            if park is None:
                continue
            (parks.append(park), restarts.append(restart)) if restart is not None else stays.append(park)
        parks, restarts, stays = np.array(parks).reshape(-1, 2), np.array(restarts).reshape(-1, 2), np.array(stays).reshape(-1, 2)
        ax.add_patch(Circle((0, 0), BOX, fill=False, lw=1.2, color="k", ls="--"))
        ax.annotate("", xy=(0.12, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))
        ax.scatter(*expert_end.T, s=6, color="#2e7d32", alpha=.45, label=f"expert stops ({len(expert_end)})")
        if len(stays):
            ax.scatter(*stays.T, s=10, color="#1f77b4", alpha=.35, marker="o", label=f"parks and stays ({len(stays)})")
        for a, b in zip(parks, restarts):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="grey", lw=0.5, alpha=.6)
        if len(parks):
            ax.scatter(*parks.T, s=12, color="#1f77b4", label=f"parks… ({len(parks)})")
            ax.scatter(*restarts.T, s=16, color="#d98c00", marker="x", label="…creeps and restarts")
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-0.25, 0.25); ax.set_ylim(-0.25, 0.25); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlabel("x along the final heading (m)")
        ax.legend(fontsize=6.5, loc="lower left", framealpha=.85)
    axes[0][0].set_ylabel("y (m), origin at the expert's dock")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)
    print("figure:", out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
