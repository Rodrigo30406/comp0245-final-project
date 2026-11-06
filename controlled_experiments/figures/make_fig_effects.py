"""Forest plot of the controlled experiments (Fig. 1 of the report).

Each row: difference to the previous main model in percentage points with its
two-level bootstrap 95% CI (P: 97.5%, its Bonferroni bound). All numbers are
copied from the read-outs listed in ../README.md (also in ../report.txt). To add an
experiment, append a row to ROWS and re-run:

    python3 make_fig_effects.py      # writes fig_effects.pdf next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

# verdict -> (colour, marker); reference palette slots 1-3 plus a neutral grey
STYLE = {
    "adopted": ("#2a78d6", "o"),       # adopted / confirmed
    "rejected": ("#8d8b86", "s"),      # not adopted / prediction failed
    "descriptive": ("#eb6834", "D"),   # no adoption rule
}

# (label, delta, lo, hi, verdict, replication set?)  None = group header
ROWS = [
    ("A-to-B task", None),
    ("A  drop wheel speed†", 8.5, 5.5, 11.7, "adopted", False),
    ("HARD", 3.2, 0.6, 6.3, "adopted", True),
    ("CONFIRM", 7.1, 4.3, 10.1, "adopted", True),
    ("C3  ignition copies", 60.1, 53.5, 66.9, "adopted", False),
    ("HARD", 55.9, 49.8, 62.6, "adopted", True),
    ("CONFIRM", 64.0, 57.3, 71.8, "adopted", True),
    ("R  random row (vs C3)", -60.1, -66.9, -53.7, "adopted", False),
    ("S  split, copies vs none", 62.9, 53.5, 72.0, "adopted", False),
    ("split vs C3", 2.8, -7.9, 13.4, "descriptive", False),
    ("W  3×256 MLP (vs C3)", -27.1, -39.3, -15.7, "rejected", False),
    ("D  EE-error features", 7.5, -4.5, 17.7, "rejected", False),
    ("F  one DAgger round", -40.0, -48.2, -31.2, "rejected", False),
    ("P  PD output layer", 8.8, -0.3, 17.1, "rejected", False),
    ("PC  P on HARD+CONFIRM", 2.6, -6.4, 11.1, "rejected", True),
    ("X  final recipe vs C3", -1.5, -10.4, 6.8, "descriptive", False),
    ("height band", 30.8, 23.4, 38.1, "descriptive", False),
    ("HARD, band", 32.6, 21.5, 42.9, "descriptive", True),
    ("Obstacle task", None),
    ("H-MM  side latch‡", 14.6, 9.2, 20.4, "adopted", False),
    ("confirm2", 15.8, 10.0, 21.8, "adopted", True),
    ("H-FACT  base/arm split‡", -2.0, -8.7, 4.8, "rejected", False),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "pdf.fonttype": 42,
})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

fig, ax = plt.subplots(figsize=(3.45, 3.55))
fig.subplots_adjust(left=0.395, right=0.745, top=0.915, bottom=0.095)
n = len(ROWS)
ys = list(range(n - 1, -1, -1))
labels_tf = blended_transform_factory(ax.transAxes, ax.transData)
header_tf = blended_transform_factory(fig.transFigure, ax.transData)

for y, row in zip(ys, ROWS):
    if row[1] is None:  # group header
        ax.text(0.01, y, row[0], transform=header_tf, ha="left", va="center",
                fontweight="bold", color=INK)
        continue
    label, d, lo, hi, verdict, rep = row
    colour, marker = STYLE[verdict]
    ax.plot([lo, hi], [y, y], color=colour, lw=1.1, solid_capstyle="round", zorder=2)
    ax.plot(d, y, marker=marker, ms=4.2 if marker != "D" else 3.6, color=colour,
            mfc="white" if rep else colour, mew=1.0, zorder=3)
    sub = rep or not label[0].isupper() or label in ("HARD", "CONFIRM")
    ax.text(-0.03, y, label, transform=labels_tf, ha="right", va="center",
            color=MUTED if sub and "  " not in label else INK)
    ax.text(1.03, y, f"{d:+.1f} [{lo:.1f}, {hi:.1f}]".replace("-", "\u2212"), transform=labels_tf,
            ha="left", va="center", color=MUTED, fontsize=6.1)

ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
ax.set_xlim(-70, 75)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xticks([-60, -30, 0, 30, 60])
ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
ax.set_yticks([])
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="x", colors=MUTED, labelcolor=INK)
ax.set_xlabel("difference to the previous model (pp)", color=INK, labelpad=2)
ax.text(1.03, n - 0.3, "$\\Delta$ [95% CI]", transform=labels_tf, ha="left",
        va="bottom", color=INK, fontsize=6.5)

handles = [
    Line2D([], [], color=STYLE["adopted"][0], marker="o", lw=1.1, ms=4, label="adopted / confirmed"),
    Line2D([], [], color=STYLE["rejected"][0], marker="s", lw=1.1, ms=4, label="not adopted"),
    Line2D([], [], color=STYLE["descriptive"][0], marker="D", lw=1.1, ms=3.4, label="descriptive"),
    Line2D([], [], color=MUTED, marker="o", mfc="white", lw=0, ms=4, label="replication set"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.3,
           handlelength=1.3, columnspacing=0.9, handletextpad=0.4,
           bbox_to_anchor=(0.5, 1.0))

out = Path(__file__).with_name("fig_effects.pdf")
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=300)
print(out)
