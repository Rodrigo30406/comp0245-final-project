"""Figures for the progress README (3 Nov 2026): the programme's closed-loop results, the offline/closed-loop
inversion, the trajectories of the v2 models, the ignition probe and the parking fix.

Numbers are copied from `scripts/analysis/final_table.py` and the sections of `docs/BLACKBOX_PLAN_20261027.md` named in
each figure; the trajectory figure reads the evaluation traces under `runs/evaluation/` (100 ms, 205 nominal validation
geometries). Output: `docs/figures/progress/*.png`.

Usage: pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/analysis/progress_figures.py
"""
from pathlib import Path
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures/progress"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.facecolor": "white"})


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- 1. programme: stable arrivals per stage (nominal, 205)
STAGES = [  # label, 1 cm per seed, 5 cm per seed, tag
    ("MLP, lecturer's 21 inputs, clean data (27 Oct)", [0], [0], "v1"),
    ("MLP 21, noise-injected data (DART)", [0], [9], "v1"),
    ("MLP 25: + target in base frame + clock", [49, 15, 0, 16, 20], [100, 67, 23, 84, 60], "v1"),
    ("MLP 32: + expert goal geometry (privileged, set aside)", [26, 34, 23, 33, 59], [138, 153, 115, 137, 162], "priv"),
    ("GRU 25, memory 20 ms", [39, 78, 31, 32, 16], [126, 138, 123, 154, 137], "v1"),
    ("MLP 28 pure (black-box inputs), nominal 1024", [9, 0, 0, 0, 6], [129, 38, 33, 66, 57], "v1"),
    ("MLP 28 pure, mixture 4096 (control)", [0, 0, 0], [23, 30, 27], "v1"),
    ("GRU 28 pure, mixture 4096", [6, 19, 4], [32, 102, 27], "v1"),
    ("Transformer 28 pure, attention 20 s", [25, 29, 8], [81, 76, 47], "v1"),
    ("best stop remedy (E7c categorical wheels)", [9, 0, 12], [39, 24, 32], "v1"),
    ("E9: + 144 trigger copies (ignition row), mixture v1", [96, 142, 132], [140, 165, 146], "v1"),
    ("mixture v2, MLP 28 pure control", [1, 0, 2], [41, 31, 39], "v2"),
    ("mixture v2 + 144 trigger copies", [199, 193, 175], [202, 204, 199], "v2"),
    ("v2 + copies + 0.45 rad/s wheel dead band", [198, 195, 190], [201, 204, 197], "v2"),
    ("v2 + copies, categorical wheels", [176, 176, 195], [201, 201, 204], "v2"),
    ("v2 GRU 28 pure, patience 20 (no device)", [128, 60, 150], [191, 177, 205], "v2"),
]


def fig_programme():
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(STAGES))[::-1]
    for ax, col, title in zip(axes, (1, 2), ("Stable arrival within 1 cm", "Stable arrival within 5 cm")):
        for yi, st in zip(y, STAGES):
            v = np.array(st[col]); c = {"v1": MUTED, "priv": MUTED, "v2": BLUE}[st[3]]
            ax.plot([v.min(), v.max()], [yi, yi], color=c, lw=2.2, solid_capstyle="round", alpha=0.45)
            ax.scatter(v, [yi] * len(v), s=22, color=c, zorder=3, edgecolor="white", lw=0.6)
            ax.scatter([np.median(v)], [yi], s=40, color=c, zorder=4, edgecolor="white", lw=0.8)
            ax.text(206, yi, f"{v.min()}–{v.max()}" if v.min() != v.max() else f"{v.min()}", va="center", ha="left",
                    fontsize=7.5, color=INK)
        ax.axvline(205, color=INK, lw=0.8, ls=(0, (3, 3)))
        ax.text(205, len(STAGES) - 0.3, "expert 205", ha="right", va="bottom", fontsize=7.5, color=INK)
        ax.set_xlim(-3, 245); ax.set_xticks([0, 50, 100, 150, 205]); ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("episodes of 205 (one dot per training seed, large dot = median)")
        style(ax)
    axes[0].set_yticks(y); axes[0].set_yticklabels([s[0] for s in STAGES], fontsize=8.2)
    axes[0].tick_params(axis="y", colors=INK)
    for lab, st in zip(axes[0].get_yticklabels(), STAGES):
        lab.set_color(BLUE if st[3] == "v2" else INK)
        if st[3] == "priv":
            lab.set_style("italic")
    fig.suptitle("Closed-loop results across the programme: same 205 held-out geometries, nominal start, 40 s episodes",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Grey: version 1 (mixture 4096, goal from the world origin, expert failures kept). Blue: version 2 "
             "(expert successes only, goal from the start pose). Italic: uses the expert's internal goal construction.",
             fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "01_programme_arrivals.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- 2. offline fit against closed-loop arrivals
ARMS = {  # name: ([(nrmse, 1 cm nominal)], version, label?)
    "MLP 32 mixture (privileged)": ([(0.0961, 22), (0.0943, 34), (0.1015, 15)], "v1", True),
    "MLP 31 (+ base velocity)": ([(0.1082, 0), (0.1069, 3), (0.1059, 0)], "v1", False),
    "GRU 28 v1": ([(0.1356, 6), (0.1526, 19), (0.1451, 4)], "v1", False),
    "MLP 28 control v1": ([(0.1484, 0), (0.1454, 0), (0.1491, 0)], "v1", False),
    "Transformer 28": ([(0.1363, 25), (0.1299, 29), (0.1419, 8)], "v1", True),
    "E7a weights": ([(0.1484, 0), (0.1498, 0), (0.15, 0)], "v1", False),
    "E7b L1": ([(0.1498, 0), (0.15, 0), (0.1495, 0)], "v1", False),
    "E7c categorical": ([(0.1702, 9), (0.162, 0), (0.1608, 12)], "v1", False),
    "E7d previous command": ([(0.1079, 0), (0.1042, 0), (0.1054, 1)], "v1", True),
    "E8b MLP→GRU": ([(0.162, 23), (0.1363, 7), (0.1441, 9)], "v1", False),
    "K 144 copies, v1": ([(0.1655, 96), (0.1634, 142), (0.1691, 132)], "v1", True),
    "MLP 28 control v2": ([(0.1192, 1), (0.1205, 0), (0.1217, 2)], "v2", True),
    "K 144 copies, v2": ([(0.1395, 199), (0.1347, 193), (0.1412, 175)], "v2", True),
    "trigger weight 144, v2": ([(0.188, 150), (0.1743, 181), (0.1857, 183)], "v2", True),
    "action chunk": ([(0.1506, 3), (0.1451, 2), (0.1489, 0)], "v2", False),
    "GRU weight 144": ([(0.1892, 150), (0.197, 133), (0.1952, 102)], "v2", False),
    "GRU control v2": ([(0.134, 78), (0.1289, 78), (0.135, 73)], "v2", True),
    "GRU hidden 256": ([(0.1164, 34), (0.1122, 30), (0.1112, 53)], "v2", True),
    "GRU patience 20": ([(0.1155, 128), (0.1195, 60), (0.1132, 150)], "v2", True),
    "GRU stride 10": ([(0.1241, 36), (0.1242, 57), (0.1241, 39)], "v2", False),
    "GRU lr 3e-4": ([(0.1353, 89), (0.1301, 92), (0.1314, 41)], "v2", False),
    "GRU 4 per batch": ([(0.1254, 105), (0.1345, 56), (0.1296, 66)], "v2", False),
    "categorical + K 144": ([(0.1489, 176), (0.1477, 176), (0.1506, 195)], "v2", True),
    "gated + weight 144": ([(0.2444, 98), (0.259, 78), (0.2546, 53)], "v2", False),
}


OFFSETS = {"E7d previous command": (-4, -11), "MLP 28 control v2": (6, -2), "GRU hidden 256": (-6, -11),
           "MLP 32 mixture (privileged)": (5, 4), "Transformer 28": (6, -3), "GRU control v2": (6, 2), "GRU patience 20": (6, 2)}


def fig_offline_vs_closed():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for name, (pts, ver, lab) in ARMS.items():
        p = np.array(pts); c = BLUE if ver == "v2" else MUTED
        ax.scatter(p[:, 0], p[:, 1], s=26, color=c, edgecolor="white", lw=0.6, zorder=3)
        if lab:
            i = int(np.argmax(p[:, 1])) if "copies" in name or "weight" in name or "GRU" in name else int(np.argmin(p[:, 0]))
            dx, dy = OFFSETS.get(name, (5, 3))
            ax.annotate(name, (p[i, 0], p[i, 1]), xytext=(dx, dy), textcoords="offset points", fontsize=7.2, color=INK)
    ax.axhline(205, color=INK, lw=0.8, ls=(0, (3, 3))); ax.text(0.087, 207, "expert 205", ha="left", fontsize=7.5, color=INK)
    ax.set_xlabel("offline validation error (episode-macro balanced NRMSE, lower = better fit)")
    ax.set_ylabel("stable arrivals within 1 cm, 205 nominal geometries")
    ax.set_title("The offline metric does not rank the policies: the best fits arrive least",
                 loc="left", fontsize=10.5, color=INK)
    ax.set_xlim(0.085, 0.265); ax.set_ylim(-8, 215)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.scatter([], [], color=MUTED, label="version 1 data (mixture 4096)"); ax.scatter([], [], color=BLUE, label="version 2 data")
    ax.legend(loc="center right", frameon=False, fontsize=8)
    fig.savefig(OUT / "02_offline_vs_closed_loop.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- 3. trajectories of the v2 models (traces)
EVALS = {
    "expert": ("v2_mlp28_trig144_on_clean_validation_20261102", "expert", INK),
    "MLP 28 control (v2)": ("v2_mlp28_control_on_clean_validation_20261102", "mlp28_control_seed0", ORANGE),
    "+ 144 trigger copies": ("v2_mlp28_trig144_on_clean_validation_20261102", "mlp28_trig144_seed0", BLUE),
    "+ copies + dead band": ("v2_mlp28_trig144_db045_on_clean_validation_20261103", "mlp28_trig144_seed0", AQUA),
}


def load_traces(eval_dir, label):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "runs/evaluation" / eval_dir / "traces" / f"{label}_episode_*.npz"))):
        ep = int(f.rsplit("_", 1)[1][:-4])
        with np.load(f, allow_pickle=False) as z:
            out[ep] = {"t": z["time"], "d": z["distance"], "xy": z["observation"][:, :2], "target": z["observation"][0, 18:20]}
    return out


def fig_trajectories():
    data = {k: load_traces(*v[:2]) for k, v in EVALS.items()}
    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.42, wspace=0.28)
    for j, (name, (_, _, c)) in enumerate(EVALS.items()):
        ax = fig.add_subplot(gs[0, j])
        eps = data[name]
        D = np.array([eps[e]["d"] for e in sorted(eps)]); t = eps[sorted(eps)[0]]["t"]
        for row in D:
            ax.plot(t, np.maximum(row, 1e-3), color=c, lw=0.5, alpha=0.12)
        ax.plot(t, np.maximum(np.median(D, axis=0), 1e-3), color=c, lw=2.0, label="median")
        ax.axhline(0.01, color=INK, lw=0.8, ls=(0, (3, 3)))
        ax.set_yscale("log"); ax.set_ylim(1e-3, 4); ax.set_xlim(0, 40)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK)
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["1 mm", "1 cm", "10 cm", "1 m"])
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
        if j == 0:
            ax.set_ylabel("end effector to target")
        ax.set_xlabel("time (s)")
        n1 = int((D[:, -1] <= 0.01).sum()); n2 = int((D[:, -1] <= 0.02).sum())
        ax.text(39, 2.4, f"at 40 s: {n1} within 1 cm, {n2} within 2 cm", ha="right", fontsize=7.2, color=INK)
    # one geometry: control fails, copies arrive
    ctrl, k144, band = data["MLP 28 control (v2)"], data["+ 144 trigger copies"], data["+ copies + dead band"]
    cands = [e for e in k144 if e in ctrl and e in band and ctrl[e]["d"][-1] > 0.3 and k144[e]["d"][-1] < 0.01 and band[e]["d"][-1] < 0.01]
    dist = {e: np.hypot(*k144[e]["target"]) for e in cands}
    ep = sorted(cands, key=lambda e: abs(dist[e] - np.median(list(dist.values()))))[0]
    ax = fig.add_subplot(gs[1, :2])
    for name, (_, _, c) in EVALS.items():
        xy = data[name][ep]["xy"]
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1.6 if name != "expert" else 2.2, label=name, alpha=0.95)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=c, s=28, zorder=4, edgecolor="white", lw=0.6)
    tg = k144[ep]["target"]
    ax.scatter(*tg, marker="x", color=INK, s=70, zorder=5, lw=1.6); ax.text(tg[0] + 0.05, tg[1] + 0.05, "target", fontsize=8, color=INK)
    ax.scatter(0, 0, marker="s", color=INK, s=24, zorder=5); ax.text(0.05, -0.1, "start", fontsize=8, color=INK)
    ax.set_aspect("equal"); ax.set_title(f"Base path, validation geometry {ep} (dots: position at 40 s)", loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax = fig.add_subplot(gs[1, 2:])
    for name, (_, _, c) in EVALS.items():
        tr = data[name][ep]; ax.plot(tr["t"], np.maximum(tr["d"], 1e-3), color=c, lw=1.8, label=name)
    ax.axhline(0.01, color=INK, lw=0.8, ls=(0, (3, 3))); ax.text(39.5, 0.0115, "1 cm", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 4); ax.set_xlim(0, 40)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["1 mm", "1 cm", "10 cm", "1 m"])
    ax.set_title(f"End effector to target, geometry {ep}", loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("time (s)"); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.suptitle("Mixture v2 models in closed loop, seed 0, 205 nominal validation geometries (traces at 100 ms)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "03_trajectories_v2.png", bbox_inches="tight")
    plt.close(fig)
    return ep


# ------------------------------------------------------------------- 4. ignition probe (§6z.1)
def fig_probe():
    rows = [("at the policy's own parked state", [0.11, 0.09, 0.03], [0.88, 0.86, 0.70]),
            ("expert's stop row, policy's resting arm", [0.08, 0.08, 0.03], [0.45, 1.24, 1.33]),
            ("expert's stop row, expert's arm state", [1.5, 1.6, 1.7], [None, None, None])]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, ctrl, k) in zip(y, rows):
        ax.scatter(ctrl, [yi + 0.12] * 3, color=ORANGE, s=34, zorder=3, edgecolor="white", lw=0.6)
        if k[0] is not None:
            ax.scatter(k, [yi - 0.12] * 3, color=BLUE, s=34, zorder=3, edgecolor="white", lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.8); ax.axvline(1, color=INK, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.02, -0.42, "expert's reach = 1", ha="left", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5); ax.tick_params(axis="y", colors=INK)
    ax.set_xlim(-0.15, 1.9); ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("predicted arm command on the hold → reach axis (0 = keep holding, 1 = the expert's first reach step)")
    ax.scatter([], [], color=ORANGE, label="MLP 28 control (v2)"); ax.scatter([], [], color=BLUE, label="+ 144 trigger copies")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Ignition probe: the reach is triggered by the arm's own motion; the copies re-weight the one row that starts it",
                 loc="left", fontsize=9.5, color=INK, pad=10)
    style(ax)
    fig.savefig(OUT / "04_ignition_probe.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- 5. parking fix (§6ac.1)
def fig_parking():
    groups = ["v2 + 144 copies", "+ dead band 0.45 rad/s", "categorical wheels + copies"]
    cols = [ORANGE, AQUA, VIOLET]
    panels = [("base inside the expert's\n8 cm box at 40 s", [[30, 3, 52], [139, 118, 125], [97, 84, 132]], 129, "episodes of 205"),
              ("base still moving\nat 40 s", [[15, 129, 36], [3, 25, 5], [1, 10, 0]], 0, "episodes of 205"),
              ("drift after parking\n(median)", [[0.06, 0.25, 0.14], [0.07, 0.07, 0.07], [0.07, 0.06, 0.06]], None, "m"),
              ("held within 5 cm\nfor ≥ 10 s", [[202, 182, 172], [200, 197, 191], [194, 195, 203]], 204, "episodes of 205")]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3), gridspec_kw={"wspace": 0.5})
    for ax, (title, vals, exp, unit) in zip(axes, panels):
        y = np.arange(3)[::-1]
        for yi, v, c in zip(y, vals, cols):
            ax.plot([min(v), max(v)], [yi, yi], color=c, lw=2.2, alpha=0.45, solid_capstyle="round")
            ax.scatter(v, [yi] * 3, color=c, s=30, zorder=3, edgecolor="white", lw=0.6)
        if exp is not None:
            ax.axvline(exp, color=INK, lw=0.8, ls=(0, (3, 3))); ax.text(exp, 2.55, "expert", ha="center", fontsize=7.5, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(groups if ax is axes[0] else [""] * 3, fontsize=8.5); ax.tick_params(axis="y", colors=INK)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK); ax.set_xlabel(unit); ax.set_ylim(-0.6, 2.9)
        if unit != "m":
            ax.set_xlim(-5, 215)
        style(ax)
    fig.suptitle("Parking: with the arm igniting, an exact wheel zero (imposed or learned) gives the expert's parking (nominal, 3 seeds)",
                 x=0.01, y=1.06, ha="left", fontsize=10.5, color=INK)
    fig.savefig(OUT / "05_parking_fix.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- 6. the K sweep (§6y, §6y.1): how 144 was chosen
def fig_ksweep():
    K = ["0 (control)", "16", "144", "512"]; x = np.arange(4)
    nom = {"1 cm": [0, 48, 96, 86], "2 cm": [0, 84, 124, 101], "5 cm": [23, 141, 140, 119]}
    rnd1 = [0, 47, 103, 66]
    held = [0, 22, 107, 79]; drift = [np.nan, 1.22, 0.22, 1.62]; armext = [3, 197, 198, 149]; probe = [-0.02, 0.12, 0.96, 1.27]
    nrmse = [0.145, 0.159, 0.166, 0.177]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    for (lab, v), c, lw in zip(nom.items(), (BLUE, AQUA, MUTED), (2.2, 1.8, 1.6)):
        ax.plot(x, v, color=c, lw=lw, marker="o", ms=5, label=f"nominal, {lab}")
    ax.plot(x, rnd1, color=BLUE, lw=1.8, ls=(0, (4, 2)), marker="o", ms=5, mfc="white", label="random start, 1 cm")
    ax.set_title("stable arrivals (of 205)", loc="left", fontsize=9.5, color=INK)
    ax.set_ylim(-5, 215); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.plot(x, armext, color=MUTED, lw=1.6, marker="o", ms=5, label="arm extended")
    ax.plot(x, held, color=BLUE, lw=2.2, marker="o", ms=5, label="held ≥ 10 s")
    ax.set_ylim(-5, 215); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("ignition and hold (of 205)", loc="left", fontsize=9.5, color=INK)
    ax = axes[2]
    ax.plot(x[1:], drift[1:], color=ORANGE, lw=2.0, marker="s", ms=5)
    ax.set_ylim(0, 2.0); ax.set_title("drift after parking, median (m)", loc="left", fontsize=9.5, color=INK)
    ax.text(0, 0.08, "control: no\nclean stop", ha="center", fontsize=7, color=MUTED)
    ax = axes[3]
    ax.plot(x, probe, color=BLUE, lw=2.2, marker="o", ms=5)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(1, color=INK, lw=0.8, ls=(0, (3, 3)))
    ax.text(0, 1.03, "expert's reach = 1", ha="left", va="bottom", fontsize=7, color=INK)
    ax.set_ylim(-0.2, 1.65); ax.set_yticks([0, 0.5, 1.0, 1.5])
    for xi, n in zip(x, nrmse):
        ax.text(xi, 1.5, f"{n:.3f}", ha="center", fontsize=7, color=MUTED)
    ax.text(1.5, 1.6, "offline validation NRMSE (worse with K):", ha="center", fontsize=7, color=MUTED)
    ax.set_title("arm output at the resting-arm stop row", loc="left", fontsize=9.5, color=INK)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(K); ax.set_xlabel("copies per episode, K"); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.suptitle("Choosing K (2 Nov, mixture v1, one seed per K): every K ignites the arm; 144 holds best and drifts least",
                 x=0.01, y=1.04, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, -0.06, "Pre-registered rule: the smallest K whose model extends the arm in more than half of the nominal episodes; "
             "tie-break on 1 cm arrivals (nominal + random): K 16 = 95, K 144 = 199, K 512 = 152. 144 = the teammate's 16 copies × their 9 ms row stride.",
             fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "06_k_sweep.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_programme(); fig_offline_vs_closed(); ep = fig_trajectories(); fig_probe(); fig_parking(); fig_ksweep()
    print("written", sorted(p.name for p in OUT.glob("*.png")), "trajectory geometry", ep)
