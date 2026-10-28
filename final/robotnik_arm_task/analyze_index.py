"""Analyse the expert episodes listed in the index.csv files written by collect_data.py.

Plots where the expert succeeds or fails as a function of the target position (relative to the robot
start pose), how long the episodes take, and how the targets cover the sampled ranges. Use it to pick
the target sampling range (the expert's working envelope) and to check the variety of a dataset.

Examples:
  python analyze_index.py data/probe
  python analyze_index.py data/run1 data/run2 --out data/analysis_all --show

Figures and a summary.txt are written to <first run folder>/analysis/ unless --out is given.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

# Status roles (fixed status palette) + a marker shape each, so a status never relies on colour alone
STATUS_STYLE = {
    "reached":     {"color": "#0ca30c", "marker": "o", "label": "reached"},
    "unreachable": {"color": "#ec835a", "marker": "^", "label": "unreachable (stopped at closest pose)"},
    "step_cap":    {"color": "#d03b3b", "marker": "X", "label": "step cap"},
    "quit":        {"color": "#898781", "marker": "s", "label": "stopped by user"},
    "error":       {"color": "#52514e", "marker": "D", "label": "crashed"},
}
OTHER_STYLE = {"color": "#898781", "marker": "P", "label": "other"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential ramp
MIN_EPISODE_TIME = 8.0   # StopCriteria.min_time: no episode can stop earlier
NAV_TIMEOUT = 25.0       # BASE_NAV_TIMEOUT in robotnik_arm_task.py
REACH_TOLERANCE = 0.01   # REACH_TOLERANCE in robotnik_arm_task.py

NUMERIC = ["final_error", "duration", "nav_done_time", "target_x", "target_y", "target_z", "r", "theta_deg",
           "init_x", "init_y", "init_yaw_deg", "noise_v", "noise_w", "noise_torque"]


def load_index(run_dirs):
    """Read and concatenate index.csv files into a dict of numpy arrays (one entry per episode)."""
    rows = []
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "index.csv"), newline="") as f:
            for row in csv.DictReader(f):
                row["run"] = os.path.basename(os.path.normpath(run_dir))
                rows.append(row)
    if not rows:
        raise SystemExit("No episodes found in " + ", ".join(run_dirs))

    def number(value):
        return float(value) if value not in (None, "") else np.nan

    data = {
        "status": np.array([row["status"] for row in rows]),
        "run": np.array([row["run"] for row in rows]),
        "nav_timed_out": np.array([row.get("nav_timed_out") == "True" for row in rows]),
    }
    for name in NUMERIC:  # older index files have no init_* columns: they read as NaN
        data[name] = np.array([number(row.get(name)) for row in rows])
    # Index files written before collect_data.py corrected the template's stop-check order report an arm
    # that reached the target early and then waited for min_time as "unreachable"
    on_target = (data["status"] == "unreachable") & (data["final_error"] < REACH_TOLERANCE)
    if on_target.any():
        print(f"Note: {on_target.sum()} 'unreachable' episodes end within {REACH_TOLERANCE} m of the target, "
              f"counted as reached")
        data["status"][on_target] = "reached"
    data["z"] = data["target_z"]
    data["abs_theta"] = np.abs(data["theta_deg"])
    data["noisy"] = (np.nan_to_num(data["noise_v"]) + np.nan_to_num(data["noise_w"])
                     + np.nan_to_num(data["noise_torque"])) > 0
    return data


def style_axes(ax):
    """Recessive chrome: hairline grid, light axes, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def statuses_in(data):
    present = [s for s in STATUS_STYLE if s in set(data["status"])]
    return present + sorted(set(data["status"]) - set(STATUS_STYLE))


def scatter_by_status(ax, data, x, y, mask=None):
    """One marker series per status, with a surface-coloured ring so overlapping markers stay readable."""
    mask = np.ones(len(data["status"]), dtype=bool) if mask is None else mask
    for status in statuses_in(data):
        sel = mask & (data["status"] == status) & np.isfinite(x) & np.isfinite(y)
        if not sel.any():
            continue
        style = STATUS_STYLE.get(status, OTHER_STYLE)
        ax.scatter(x[sel], y[sel], s=42, marker=style["marker"], color=style["color"],
                   edgecolors=SURFACE, linewidths=0.8, label=f"{style['label']} ({sel.sum()})", zorder=3)


def finish(fig, out_dir, name, title, legend_ax=None):
    """Title top-left, legend in a row below the plots (never over the data), save as PNG."""
    bottom = 0.0
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False,
                            fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_2)
        bottom = 0.6 / fig.get_size_inches()[1]  # ~0.6 inch strip for the legend row
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, bottom, 1, 1])
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("saved", path)


def plot_outcome_r_z(plt, data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    scatter_by_status(ax, data, data["r"], data["z"])
    ax.set_xlabel("horizontal distance r from start position (m)")
    ax.set_ylabel("target height z (m)")
    style_axes(ax)
    finish(fig, out_dir, "outcome_r_z.png", "Expert outcome by target distance and height", legend_ax=ax)


def plot_success_heatmap(plt, data, out_dir, r_bins, z_bins):
    from matplotlib.colors import LinearSegmentedColormap
    ok = np.isfinite(data["r"]) & np.isfinite(data["z"])
    r, z, success = data["r"][ok], data["z"][ok], data["status"][ok] == "reached"
    r_edges = np.linspace(r.min(), r.max(), r_bins + 1)
    z_edges = np.linspace(z.min(), z.max(), z_bins + 1)
    n, _, _ = np.histogram2d(r, z, bins=[r_edges, z_edges])
    k, _, _ = np.histogram2d(r[success], z[success], bins=[r_edges, z_edges])
    rate = np.ma.masked_where(n == 0, k / np.maximum(n, 1))

    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)
    mesh = ax.pcolormesh(r_edges, z_edges, rate.T, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i in range(r_bins):
        for j in range(z_bins):
            if n[i, j]:
                ax.text((r_edges[i] + r_edges[i + 1]) / 2, (z_edges[j] + z_edges[j + 1]) / 2,
                        f"{int(k[i, j])}/{int(n[i, j])}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if rate[i, j] > 0.6 else INK)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("success rate (reached / episodes)", color=INK_2)
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.set_xlabel("horizontal distance r from start position (m)")
    ax.set_ylabel("target height z (m)")
    style_axes(ax)
    ax.grid(False)
    finish(fig, out_dir, "success_heatmap.png", "Expert success rate per (r, z) cell")


def plot_outcome_polar(plt, data, out_dir):
    fig = plt.figure(figsize=(6, 6), facecolor=SURFACE)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")   # robot start heading points up, positive theta (robot's left) to the left
    scatter_by_status(ax, data, np.deg2rad(data["theta_deg"]), data["r"])
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.spines["polar"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.set_rlabel_position(135)
    finish(fig, out_dir, "outcome_top_view.png", "Targets seen from above (start heading = up, r in m)",
           legend_ax=ax)


def plot_duration(plt, data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    for ax, key, label in zip(axes, ["abs_theta", "r"],
                              ["|bearing| from start heading (deg)", "horizontal distance r (m)"]):
        scatter_by_status(ax, data, data[key], data["duration"])
        # Many episodes stop exactly at the min length, so that label goes below its line
        for value, text, va in [(MIN_EPISODE_TIME, "min episode length", "top"),
                                (NAV_TIMEOUT, "base nav timeout", "bottom")]:
            ax.axhline(value, color=MUTED, linewidth=1, linestyle="--", zorder=1)
            ax.text(0.01, value, text, transform=ax.get_yaxis_transform(), ha="left", va=va,
                    fontsize=8, color=MUTED, zorder=2)
        ax.set_ylim(bottom=MIN_EPISODE_TIME - 2)
        ax.set_xlabel(label)
        style_axes(ax)
    axes[0].set_ylabel("episode duration (s)")
    finish(fig, out_dir, "duration.png", "Episode duration: large turns and long drives cost time",
           legend_ax=axes[1])


def plot_coverage(plt, data, out_dir):
    reached = data["status"] == "reached"
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, key, label in zip(axes, ["r", "theta_deg", "z"],
                              ["r (m)", "bearing theta (deg)", "z (m)"]):
        values = data[key][np.isfinite(data[key])]
        edges = np.histogram_bin_edges(values, bins=15)
        ax.hist(values, bins=edges, color=BLUES[1], edgecolor=SURFACE, linewidth=2, label="all episodes")
        ax.hist(data[key][reached & np.isfinite(data[key])], bins=edges, histtype="step",
                color=BLUES[5], linewidth=2, label="reached")
        ax.set_xlabel(label)
        style_axes(ax)
    axes[0].set_ylabel("episodes")
    finish(fig, out_dir, "coverage_hist.png", "Target coverage of the sampled ranges", legend_ax=axes[2])


def summary_text(data, r_bins, z_bins):
    n = len(data["status"])
    lines = [f"Runs: {', '.join(sorted(set(data['run'])))}", f"Episodes: {n}"]
    for status in statuses_in(data):
        count = int((data["status"] == status).sum())
        lines.append(f"  {status:<12} {count:4d}  ({100 * count / n:.1f}%)")
    for label, sel in [("clean", ~data["noisy"]), ("noisy", data["noisy"])]:
        if sel.any():
            lines.append(f"Success rate {label}: {100 * (data['status'][sel] == 'reached').mean():.1f}% "
                         f"of {sel.sum()}")
    lines.append(f"Base navigation timed out: {int(data['nav_timed_out'].sum())}")
    reached = data["status"] == "reached"
    if reached.any():
        d = data["duration"][reached]
        lines.append(f"Duration of reached episodes (s): mean {d.mean():.1f}, median {np.median(d):.1f}, "
                     f"max {d.max():.1f}")
    failed = ~reached & np.isfinite(data["final_error"])
    if failed.any():
        lines.append(f"Final error of failed episodes (m): median {np.median(data['final_error'][failed]):.3f}")

    # Success rate per r bin and per z bin (marginals), to read off the expert working range
    for key, bins, unit in [("r", r_bins, "m"), ("z", z_bins, "m")]:
        ok = np.isfinite(data[key])
        edges = np.linspace(data[key][ok].min(), data[key][ok].max(), bins + 1)
        lines.append(f"Success rate by {key}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ok & (data[key] >= lo) & (data[key] <= hi)
            if sel.any():
                lines.append(f"  {key} {lo:5.2f}-{hi:5.2f} {unit}: {100 * reached[sel].mean():5.1f}% of {sel.sum()}")
    return "\n".join(lines)


def check_files(data_runs, max_torque=30.0):
    """Per-sample data quality over all saved episodes: size, finiteness, label structure, phase balance."""
    phase_names = ["align", "drive", "orient", "reach"]
    n_files = n_samples = n_bad_files = n_nonfinite = 0
    phase_counts = np.zeros(4, dtype=np.int64)
    n_saturated = n_wheel_mismatch = 0
    lengths, size = [], 0
    for run_dir in data_runs:
        for name in sorted(os.listdir(run_dir)):
            if not name.endswith(".npz"):
                continue
            path = os.path.join(run_dir, name)
            size += os.path.getsize(path)
            try:
                d = np.load(path)
                arrays = {k: d[k] for k in ["base_pos", "base_bearing", "q_mes", "qd_mes", "wheel_cmds",
                                             "torque_arm", "phase"]}
            except Exception:
                n_bad_files += 1
                continue
            n_files += 1
            T = len(arrays["phase"])
            lengths.append(T)
            n_samples += T
            n_nonfinite += sum(int((~np.isfinite(arrays[k])).any(axis=-1).sum()) for k in
                               ["base_pos", "q_mes", "qd_mes", "wheel_cmds", "torque_arm"])
            phase_counts += np.bincount(arrays["phase"], minlength=4)[:4]
            n_saturated += int((np.abs(arrays["torque_arm"]) >= max_torque - 1e-3).sum())
            wheels = arrays["wheel_cmds"]  # [rw, lw, lw, rw]: left and right pairs must match
            n_wheel_mismatch += int(((np.abs(wheels[:, 0] - wheels[:, 3]) > 1e-4)
                                     | (np.abs(wheels[:, 1] - wheels[:, 2]) > 1e-4)).sum())
    if not n_files:
        return "No readable .npz files found."
    lengths = np.array(lengths)
    lines = [
        "Data quality (all saved episodes):",
        f"  files {n_files} ({size / 1e9:.2f} GB), unreadable {n_bad_files}",
        f"  samples {n_samples:,} (episode length: median {int(np.median(lengths))}, min {lengths.min()}, "
        f"max {lengths.max()})",
        f"  samples with NaN/inf in inputs or labels: {n_nonfinite}",
        f"  wheel labels breaking the [rw, lw, lw, rw] structure: {n_wheel_mismatch}",
        f"  arm torque label entries at the +-{max_torque:.0f} N*m limit: "
        f"{100 * n_saturated / (7 * n_samples):.2f}%",
        "  samples per phase: " + ", ".join(f"{name} {100 * c / n_samples:.1f}%"
                                           for name, c in zip(phase_names, phase_counts)),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run folders containing index.csv")
    parser.add_argument("--out", help="output folder (default: <first run>/analysis)")
    parser.add_argument("--r-bins", type=int, default=8)
    parser.add_argument("--z-bins", type=int, default=8)
    parser.add_argument("--show", action="store_true", help="also open the figures in windows")
    parser.add_argument("--check-files", action="store_true",
                        help="also load every .npz and report per-sample data quality (slower)")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"

    data = load_index(args.runs)
    out_dir = args.out or os.path.join(args.runs[0], "analysis")
    os.makedirs(out_dir, exist_ok=True)

    text = summary_text(data, args.r_bins, args.z_bins)
    if args.check_files:
        text += "\n" + check_files(args.runs)
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")

    plot_outcome_r_z(plt, data, out_dir)
    plot_success_heatmap(plt, data, out_dir, args.r_bins, args.z_bins)
    plot_outcome_polar(plt, data, out_dir)
    plot_duration(plt, data, out_dir)
    plot_coverage(plt, data, out_dir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
