"""Drawing of the egocentric input transforms (docs/02_inputs_and_outputs.md): the target expressed in the base frame.

Left: one robot pose and target in the world frame, with the quantities the transforms compute (d, target_in_base_x/y,
target_distance_xy, heading error). Right: two robot poses facing the same relative situation; their raw 21 inputs
differ (positions, bearing, target), their egocentric features are identical.

  python3 scripts/figures/egocentric_frame.py OUT.png
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrow

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from robot_learning.learning.features import EGO, encode  # noqa: E402

SPEC = {"variant": EGO, "standoff_heights": [0.1, 0.5, 1.1, 1.25], "standoff_distances": [0.8, 0.65, 0.65, 0.35],
        "horizon_s": 40.0, "dt_s": 0.001}


def features(base_xy, yaw, target):
    obs = np.zeros(21); obs[0:2] = base_xy; obs[3] = yaw; obs[18:21] = target
    return encode(obs[None], np.zeros(1), SPEC)[0][21:27]


def robot(ax, xy, yaw, colour, label):
    ax.add_patch(Circle(xy, 0.18, fill=False, color=colour, lw=1.6))
    h = np.array([np.cos(yaw), np.sin(yaw)]); l = np.array([-np.sin(yaw), np.cos(yaw)])
    ax.add_patch(FancyArrow(*xy, *(0.45 * h), width=0.012, color=colour, length_includes_head=True))
    ax.add_patch(FancyArrow(*xy, *(0.3 * l), width=0.008, color=colour, alpha=.6, length_includes_head=True))
    ax.text(*(xy + 0.5 * h), "x_base", color=colour, fontsize=8)
    ax.text(*(xy + 0.34 * l), "y_base", color=colour, fontsize=8)
    ax.text(*(xy - np.array([0.1, 0.32])), label, color=colour, fontsize=9)


def main(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    base, yaw, target = np.array([0.6, -0.4]), np.radians(30), np.array([1.9, 1.1, 0.55])
    f = features(base, yaw, target)
    robot(a1, base, yaw, "#1f77b4", "base (x, y, bearing)")
    a1.plot(*target[:2], "*", ms=16, color="#d62728"); a1.text(target[0] + 0.06, target[1], "target", color="#d62728")
    a1.annotate("", xy=target[:2], xytext=base, arrowprops=dict(arrowstyle="->", lw=1.3, color="k"))
    a1.text(*(base + 0.5 * (target[:2] - base) + [-0.35, 0.08]), f"d = target_xy - base_xy\n|d| = {f[3]:.2f} m", fontsize=9)
    h = np.array([np.cos(yaw), np.sin(yaw)])
    foot = base + f[0] * h
    a1.plot(*np.c_[base, foot], ls="--", color="#1f77b4", lw=1)
    a1.plot(*np.c_[foot, target[:2]], ls="--", color="#1f77b4", lw=1)
    a1.text(*(base + 0.55 * f[0] * h + [0.05, -0.18]), f"target_in_base_x = {f[0]:.2f} m", color="#1f77b4", fontsize=9)
    a1.text(*(foot + 0.5 * (target[:2] - foot) + [0.06, -0.05]), f"target_in_base_y = {f[1]:.2f} m", color="#1f77b4", fontsize=9)
    heading = np.degrees(np.arctan2(f[5], f[4]))
    a1.add_patch(Arc(base, 0.9, 0.9, theta1=np.degrees(yaw), theta2=np.degrees(yaw) + heading, color="#2ca02c", lw=1.5))
    a1.text(*(base + [0.18, 0.42]), f"heading error {heading:.0f}°\n(cos, sin as inputs)", color="#2ca02c", fontsize=9)
    a1.annotate("", xy=(0.45, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="grey"))
    a1.annotate("", xy=(0, 0.45), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="grey"))
    a1.text(0.47, -0.05, "x_world", color="grey", fontsize=8); a1.text(-0.05, 0.48, "y_world", color="grey", fontsize=8)
    a1.set_title("the target expressed in the robot's own frame", fontsize=10)
    # right: same relative situation from two world poses
    for (b, y, colour, name) in ((np.array([-1.5, -1.0]), np.radians(-20), "#1f77b4", "pose A"),
                                 (np.array([1.2, -1.6]), np.radians(115), "#9467bd", "pose B")):
        rot = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
        t = np.r_[b + rot @ np.array([1.2, 0.5]), 0.55]
        robot(a2, b, y, colour, name)
        a2.plot(*t[:2], "*", ms=14, color=colour)
        a2.annotate("", xy=t[:2], xytext=b, arrowprops=dict(arrowstyle="->", lw=1, color=colour))
        g = features(b, y, t)
        a2.text(-2.3, 1.15 if name == "pose A" else -2.55, f"{name}  raw: x={b[0]:.1f}, y={b[1]:.1f}, bearing={np.degrees(y):.0f}°, "
                f"target=({t[0]:.2f}, {t[1]:.2f})\negocentric: ahead {g[0]:.2f} m, left {g[1]:.2f} m, "
                f"dist {g[3]:.2f} m, heading {np.degrees(np.arctan2(g[5], g[4])):.0f}°", color=colour, fontsize=8, va="top")
    a2.set_title("two poses, same task: different raw inputs, identical egocentric features", fontsize=10)
    for ax in (a1, a2):
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    a1.set_xlim(-0.3, 2.5); a1.set_ylim(-0.9, 1.6); a2.set_xlim(-2.4, 2.6); a2.set_ylim(-3.1, 1.5)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
