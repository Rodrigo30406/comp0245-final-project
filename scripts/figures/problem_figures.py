"""Figures of the two coupled failures of the reaching policies (docs/05_arm_ignition.md, docs/06_base_drift.md).

Read from the closed-loop evaluation traces (100 ms samples over 40 s, nominal validation bank, 205 held-out targets)
of the expert and of the policies, all in the goal frame of the expert: origin at the expert's base goal (dock), +x
along the bearing from the start to the target, so the required final heading is 0 and the target sits on +x.

  1_parking.png         where the base is at the end (40 s): expert vs each policy, with the expert's 8 cm stop box
  2_final_stop.png      one episode in time: wheel command and distance to the dock (expert stops at exactly zero;
                        MLP-Ego parks, creeps and restarts; the arm-igniting policies stay)
  3_after_arrival.png   all episodes, aligned at the first entry into the stop box: base speed and drift from there
  4_arm_error.png       end-effector distance to the target over time (median and IQR), its closest approach and its
                        value at 40 s; 4_arm_error_time.png/.pdf: the first panel alone (report figure)

Usage (any Python with numpy + matplotlib; the evaluation directories hold traces/ and jobs.json):
  python3 scripts/figures/problem_figures.py OUT_DIR LABEL=EVAL_DIR:POLICY_PREFIX [LABEL=...] [--episode=episode_000573]
e.g. "MLP-Ego=runs/evaluation/v2_mlp28_control_on_clean_validation_20261102:mlp28_control"; the expert is read from the
first directory. Seeds: every policy label POLICY_PREFIX_seedN found in the directory is pooled (scatter: seed 0 only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
EXPERT = json.loads((ROOT / "experiment_specs/expert/default.json").read_text())
STANDOFF_H, STANDOFF_D = EXPERT["standoff"]["heights"], EXPERT["standoff"]["distances"]
POS_TOL = EXPERT["base_controller"]["pos_tol"]          # 0.08 m: the expert's parking tolerance
REACH = EXPERT["stop"]["reach_tolerance"]                 # 0.01 m
COLOURS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
EXPERT_COLOUR = "#111111"
PHASE_COLOUR = {"align": "#f3d9a4", "drive": "#cfe2f3", "orient": "#f4c7c3", "done": "#d9ead3"}


def goal_frame(target, start_xy=(0.0, 0.0)):
    """Expert's dock and bearing for a target (bearing from the start position, standoff from the target height)."""
    t = np.asarray(target, float)
    rel = t[:2] - np.asarray(start_xy, float)
    bearing = float(np.arctan2(rel[1], rel[0])) if np.hypot(*rel) > 1e-4 else 0.0
    standoff = float(np.interp(t[2], STANDOFF_H, STANDOFF_D))
    dock = t[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
    return dock, bearing


def to_goal(xy, dock, bearing):
    c, s = np.cos(-bearing), np.sin(-bearing)
    d = np.asarray(xy) - dock
    return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], axis=-1)


def load(eval_dir, policy):
    """{episode_id: trace dict} of one policy of an evaluation directory."""
    eval_dir = Path(eval_dir)
    jobs = [j for j in json.loads((eval_dir / "jobs.json").read_text()) if j["policy"] == policy]
    out = {}
    for job in jobs:
        path = eval_dir / "traces" / f"{job['job_id']}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as d:
            tr = {k: d[k] for k in ("time", "observation", "distance", "action")}
            if "expert_phase" in d.files and d["expert_diagnostics_valid"].any():
                # the phase label stays "orient" after the expert latches done; the navigation flag marks done
                tr["phase"] = np.where(d["expert_navigation_done"], "done", d["expert_phase"])
        start = np.asarray(job["initial_observation"][:2], float)
        tr["dock"], tr["bearing"] = goal_frame(job["scenario"]["target"], start)
        tr["base_goal"] = to_goal(tr["observation"][:, :2], tr["dock"], tr["bearing"])
        tr["dock_distance"] = np.hypot(*tr["base_goal"].T)
        tr["wheel"] = np.abs(tr["action"][:, :2]).mean(axis=1)
        out[job["scenario"]["episode_id"]] = tr
    return out


def seeds(eval_dir, prefix):
    names = json.loads((Path(eval_dir) / "policies.json").read_text())
    return sorted(n for n in names if n.startswith(prefix + "_seed"))


def figure_parking(models, expert, out):
    panels = [("expert", expert)] + [(label, runs[0]) for label, runs in models]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.6), sharex=True, sharey=True)
    for ax, (label, tr) in zip(axes, panels):
        end = np.array([t["base_goal"][-1] for t in tr.values()])
        inside = np.hypot(*end.T) < POS_TOL
        ax.add_patch(Circle((0, 0), POS_TOL, fill=False, ls="--", lw=1.2, color="k"))
        ax.scatter(*end[inside].T, s=7, color="#2ca02c", label=f"inside the box ({inside.sum()})")
        ax.scatter(*end[~inside].T, s=7, color="#d62728", label=f"outside ({(~inside).sum()})")
        ax.annotate("", xy=(0.12, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-0.35, 0.35); ax.set_ylim(-0.35, 0.35); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
        ax.set_xlabel("x in the goal frame (m)")
    axes[0].set_ylabel("y in the goal frame (m)")
    fig.suptitle(f"Base position at 40 s relative to the expert's dock (dashed: the expert's {POS_TOL * 100:.0f} cm stop box; "
                 "arrow: required heading, towards the target); seed 0, 205 nominal targets", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def pick_episode(models, expert):
    """An episode where the first model parks and leaves, and every other model ends inside 15 cm of the dock."""
    first = models[0][1][0]
    for eid in sorted(expert):
        tr = first.get(eid)
        if tr is None:
            continue
        if tr["dock_distance"].min() < 0.10 and tr["dock_distance"][-1] > 0.3 and \
                all(runs[0][eid]["dock_distance"][-1] < 0.15 for _, runs in models[1:] if eid in runs[0]):
            return eid
    return sorted(expert)[0]


def figure_final_stop(models, expert, out, episode=None):
    eid = episode or pick_episode(models, expert)
    rows = [("expert", expert[eid])] + [(label, runs[0][eid]) for label, runs in models if eid in runs[0]]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.9 * len(rows)), sharex=True)
    for ax, (label, tr) in zip(axes, rows):
        t = tr["time"]
        if "phase" in tr:
            for ph, colour in PHASE_COLOUR.items():
                on = tr["phase"] == ph
                ax.fill_between(t, 0, 3.2, where=on, color=colour, step="mid", lw=0, label=ph)
        ax.plot(t, tr["wheel"], color="k", lw=1.1, label="|wheel command| (rad/s)")
        ax.set_ylim(0, 3.2); ax.set_ylabel("rad/s")
        ax2 = ax.twinx()
        ax2.plot(t, tr["dock_distance"], color="#1f77b4", ls="--", lw=1.1, label="distance to the dock (m)")
        ax2.axhline(POS_TOL, color="#1f77b4", lw=.6, ls=":")
        ax2.set_ylim(0, 1.0); ax2.set_ylabel("m", color="#1f77b4")
        ax.set_title(f"{label}: wheel command and distance to the dock, {eid}", fontsize=9)
        if label == "expert":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=6, loc="upper right", frameon=False)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return eid


def figure_after_arrival(models, expert, out, horizon=15.0):
    """Aligned at the first entry into the stop box: median base speed and fraction of episodes outside the box."""
    grid = np.arange(0.0, horizon + 1e-9, 0.1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    series = [("expert", [expert], EXPERT_COLOUR)] + [(label, runs, COLOURS[i]) for i, (label, runs) in enumerate(models)]
    for label, runs, colour in series:
        speed, drift, n = [], [], 0
        for run in runs:
            for tr in run.values():
                k = np.flatnonzero(tr["dock_distance"] < POS_TOL)
                if not len(k):
                    continue
                k = k[0]; t = tr["time"][k:] - tr["time"][k]
                v = np.r_[0.0, np.hypot(*np.diff(tr["observation"][k:, :2], axis=0).T) / np.diff(tr["time"][k:])]
                speed.append(np.interp(grid, t, v, right=np.nan))
                moved = np.hypot(*(tr["observation"][k:, :2] - tr["observation"][k, :2]).T)
                drift.append(np.interp(grid, t, moved, right=np.nan))
                n += 1
        if not n:
            continue
        a1.plot(grid, np.nanmedian(speed, axis=0) * 100, color=colour, lw=1.6, label=f"{label} ({n} arrivals)")
        a2.plot(grid, np.nanmedian(drift, axis=0) * 100, color=colour, lw=1.6, label=label)
    a1.set_ylabel("median base speed (cm/s)"); a1.set_yscale("symlog", linthresh=0.1)
    a2.set_ylabel("median distance from the arrival point (cm)")
    for ax in (a1, a2):
        ax.set_xlabel("time since the base first entered the stop box (s)"); ax.grid(alpha=.3)
    a1.legend(fontsize=8, frameon=False)
    fig.suptitle("After arriving: the expert's wheels are exactly zero; a regressor's 'zero' is a small command that keeps the "
                 "base creeping", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def figure_arm_error(models, expert, out):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 3.9))
    series = [("expert", [expert], EXPERT_COLOUR)] + [(label, runs, COLOURS[i]) for i, (label, runs) in enumerate(models)]
    for label, runs, colour in series:
        curves = np.array([tr["distance"] for run in runs for tr in run.values()])
        t = next(iter(runs[0].values()))["time"]
        med, q1, q3 = np.median(curves, axis=0), np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0)
        a1.plot(t, med, color=colour, lw=1.6, label=label)
        a1.fill_between(t, q1, q3, color=colour, alpha=.12, lw=0)
        for ax, values, what in ((a2, curves.min(axis=1), "closest"), (a3, curves[:, -1], "at 40 s")):
            v = np.sort(values)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, color=colour, lw=1.6,
                    label=f"{label}: {np.mean(v < REACH) * 100:.0f} % < 1 cm")
    a1.set_yscale("log"); a1.set_ylim(1e-3, 5); a1.set_xlabel("time (s)"); a1.set_ylabel("end effector to target (m)")
    a1.axhline(REACH, color="k", lw=.7, ls=":"); a1.set_title("median and interquartile range over the episodes", fontsize=9)
    for ax, title in ((a2, "closest approach during the episode"), (a3, "error at 40 s (after holding)")):
        ax.set_xscale("log"); ax.set_xlim(5e-4, 10); ax.set_xlabel("end effector to target (m)")
        ax.set_ylabel("episodes (%)"); ax.axvline(REACH, color="k", lw=.7, ls=":"); ax.axvline(0.05, color="k", lw=.7, ls="--")
        ax.set_title(f"{title}; cumulative (dotted 1 cm, dashed 5 cm)", fontsize=9)
    for ax in (a1, a2, a3):
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Arm: distance from the end effector to the target, expert vs policies (205 nominal targets, seeds pooled)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def figure_arm_error_time(models, expert, out):
    """Single panel for the report: end-effector distance to the target over time, median and IQR, expert vs policies."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    series = [("expert", [expert], EXPERT_COLOUR)] + [(label, runs, COLOURS[i]) for i, (label, runs) in enumerate(models)]
    for label, runs, colour in series:
        curves = np.array([tr["distance"] for run in runs for tr in run.values()])
        t = next(iter(runs[0].values()))["time"]
        ax.plot(t, np.median(curves, axis=0), color=colour, lw=1.6, label=label)
        ax.fill_between(t, np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0), color=colour, alpha=.12, lw=0)
    ax.set_yscale("log"); ax.set_ylim(5e-5, 5); ax.set_xlim(0, 40)
    ax.axhline(REACH, color="k", lw=.7, ls=":"); ax.text(39.5, REACH * 1.12, "1 cm", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("end effector to target (m)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out = Path(argv[0]); out.mkdir(parents=True, exist_ok=True)
    models, expert = [], None
    forced = [a.split("=", 1)[1] for a in argv[1:] if a.startswith("--episode=")]
    argv = [a for a in argv if not a.startswith("--episode=")]
    for item in argv[1:]:
        label, spec = item.split("=", 1)
        eval_dir, prefix = spec.rsplit(":", 1)
        if expert is None:
            expert = load(eval_dir, "expert")
        models.append((label, [load(eval_dir, p) for p in seeds(eval_dir, prefix)]))
    figure_parking(models, expert, out / "1_parking.png")
    eid = figure_final_stop(models, expert, out / "2_final_stop.png", forced[0] if forced else None)
    figure_after_arrival(models, expert, out / "3_after_arrival.png")
    figure_arm_error(models, expert, out / "4_arm_error.png")
    figure_arm_error_time(models, expert, out / "4_arm_error_time.png")
    print(f"figures in {out} (final-stop episode {eid})")


if __name__ == "__main__":
    main(sys.argv[1:])
