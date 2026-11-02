"""Figures that explain the stop zone: where the expert's demonstrations put the "wheels exactly zero" rows, how the
expert enters its 8 cm / 0.05 rad box from the nominal start and from the E2 ring, and where the learned policy ends up.

Everything is drawn in the goal frame: origin at the expert's base goal, +x along the goal bearing (towards the target,
which sits at +standoff on the x axis), so the required final heading is 0 and the nominal start lies on the negative x axis.

Usage (data-based, standard library + numpy + matplotlib):
  python3 -I scripts/analysis/stop_zone_figures.py NOMINAL_DATASET_DIR E2_DATASET_DIR EVAL_DIR OUT_DIR [--episodes 60]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HEIGHTS, DISTANCES = [0.1, 0.5, 1.1, 1.25], [0.8, 0.65, 0.65, 0.35]   # expert standoff table (experiment_specs/expert)
POS_TOL, ANG_TOL = 0.08, 0.05
PHASE_COLOUR = {"align": "#d98c00", "drive": "#1f77b4", "orient": "#c0392b", "done": "#2e7d32"}

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def goal_of(target):
    target = np.asarray(target, dtype=float); b = float(np.arctan2(target[1], target[0])) if np.hypot(*target[:2]) > 1e-4 else 0.0
    s = float(np.interp(target[2], HEIGHTS, DISTANCES)); return target[:2] - s * np.array([np.cos(b), np.sin(b)]), b, s

def to_goal_frame(xy, yaw, goal, bearing):
    c, s = np.cos(-bearing), np.sin(-bearing); d = np.asarray(xy) - goal
    return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], axis=-1), wrap(np.asarray(yaw) - bearing)

def episodes(root, n):
    out = []
    for f in sorted(glob.glob(str(Path(root) / "**" / "*.npz"), recursive=True)):
        m = Path(f).with_suffix(".json")
        if not m.exists(): continue
        meta = json.load(open(m))
        if "target" not in meta or meta.get("termination") != "success": continue
        with np.load(f, allow_pickle=False) as d:
            goal, b, s = goal_of(meta["target"]); xy, yaw = to_goal_frame(d["base_position"][:, :2], d["base_bearing"], goal, b)
            phase = d["expert_phase"].astype(str); phase[d["base_navigation_done"].astype(bool)] = "done"
            out.append({"xy": xy, "yaw": yaw, "t": d["time"], "wheel": np.abs(d["action"][:, :4]).mean(axis=1), "phase": phase, "standoff": s})
        if len(out) >= n: break
    return out

def draw_box(ax, standoff, lim):
    ax.add_patch(Circle((0, 0), POS_TOL, fill=False, lw=1.5, color="k", ls="--", label="expert stop box (8 cm)"))
    ax.add_patch(Circle((standoff, 0), standoff, fill=False, lw=0.8, color="grey", ls=":", label="standoff circle around the target"))
    ax.plot(standoff, 0, marker="*", ms=14, color="k", ls="none", label="target (planar)")
    L = 0.45 * lim
    ax.annotate("", xy=(L, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="k")); ax.text(0.05 * lim, -0.08 * lim, "required final heading", fontsize=7)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("x in the goal frame (m)"); ax.set_ylabel("y in the goal frame (m)")

def heading_arrows(ax, ep, every_s=1.0, scale=0.12, **kw):
    idx = np.searchsorted(ep["t"], np.arange(ep["t"][0], ep["t"][-1], every_s))
    for i in idx:
        ax.annotate("", xy=ep["xy"][i] + scale * np.array([np.cos(ep["yaw"][i]), np.sin(ep["yaw"][i])]), xytext=ep["xy"][i],
                    arrowprops=dict(arrowstyle="->", lw=0.8, **kw))

def plot_trajectory(ax, ep, lw=1.5, alpha=1.0):
    for ph, col in PHASE_COLOUR.items():
        sel = ep["phase"] == ph
        if sel.any():
            pts = np.where(sel[:, None], ep["xy"], np.nan); ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, alpha=alpha, label=ph)

def box_entry(ep):
    inside = np.hypot(*ep["xy"].T) < POS_TOL; return int(np.argmax(inside)) if inside.any() else None

def fig_single(ep, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]; draw_box(ax, ep["standoff"], 2.4); plot_trajectory(ax, ep, lw=2); heading_arrows(ax, ep, 1.0, 0.15, color="k")
    ax.plot(*ep["xy"][0], marker="o", color="k", ms=6); ax.text(*(ep["xy"][0] + [0.05, 0.05]), "start", fontsize=9)
    k = box_entry(ep)
    if k is not None:
        ax.plot(*ep["xy"][k], marker="s", color="k", ms=6)
    h, l = ax.get_legend_handles_labels(); seen = {}; [seen.setdefault(b, a) for a, b in zip(h, l)]; ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper left")
    ax.set_title(title)
    ax = axes[1]; draw_box(ax, ep["standoff"], 0.3); plot_trajectory(ax, ep, lw=2); heading_arrows(ax, ep, 0.5, 0.06, color="k")
    if k is not None:
        err = np.degrees(abs(ep["yaw"][k])); ax.set_title(f"zoom on the box: enters with a heading error of {err:.0f}°, "
                                                           f"{(ep['phase']=='orient').sum()/1000:.1f} s of orient rotation inside", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def fig_many(nom, e2, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    for ax, eps, name in ((axes[0], nom, "nominal start (origin of the world)"), (axes[1], e2, "E2: start on a ring 0.25–1.3 m around the target")):
        draw_box(ax, np.median([e["standoff"] for e in eps]), 2.4)
        errs = []
        for ep in eps:
            plot_trajectory(ax, ep, lw=0.8, alpha=0.6); k = box_entry(ep)
            if k is not None:
                errs.append(np.degrees(abs(ep["yaw"][k]))); p = ep["xy"][k]
                ax.annotate("", xy=p + 0.2 * np.array([np.cos(ep["yaw"][k]), np.sin(ep["yaw"][k])]), xytext=p, arrowprops=dict(arrowstyle="->", lw=1.2, color="#c0392b"))
        h, l = ax.get_legend_handles_labels(); seen = {}; [seen.setdefault(b, a) for a, b in zip(h, l)]; ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper left")
        ax.set_title(f"{name}\n{len(eps)} episodes; red arrows = heading at box entry, median error {np.median(errs):.0f}°")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def eval_points(eval_dir, policy_prefix, n_max=400):
    """Closed-loop park and restart positions of a policy in the goal frame (creep_profile.py criterion)."""
    parks, restarts, finals = [], [], []
    for p in sorted(Path(eval_dir).glob(f"traces/{policy_prefix}*episode_*.npz"))[:n_max]:
        with np.load(p, allow_pickle=False) as z:
            t = z["time"]; obs = z["observation"]
        goal, b, s = goal_of(obs[0, 18:21]); xy, yaw = to_goal_frame(obs[:, :2], obs[:, 3], goal, b)
        speed = np.hypot(*np.gradient(obs[:, :2], t, axis=0).T); still = speed < 0.02; run = 0.0; park = restart = None
        for i in range(1, len(t)):
            run = run + (t[i] - t[i - 1]) if still[i] else 0.0
            if run >= 0.5:
                park = i; later = np.flatnonzero(speed[i:] > 0.05); restart = i + later[0] if len(later) else None; break
        finals.append(xy[-1])
        if park is not None and restart is not None:
            parks.append(xy[park]); restarts.append(xy[restart])
    return np.array(parks), np.array(restarts), np.array(finals)

def fig_rows(nom, e2, eval_dir, policy_prefix, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    zero_nom = np.concatenate([ep["xy"][ep["wheel"] == 0] for ep in nom]); mov_nom = np.concatenate([ep["xy"][(ep["wheel"] > 0) & (np.hypot(*ep["xy"].T) < POS_TOL)] for ep in nom])
    zero_e2 = np.concatenate([ep["xy"][ep["wheel"] == 0] for ep in e2]); mov_e2 = np.concatenate([ep["xy"][(ep["wheel"] > 0) & (np.hypot(*ep["xy"].T) < POS_TOL)] for ep in e2])
    for ax, zero, mov, name in ((axes[0], zero_nom, mov_nom, "nominal data"), (axes[1], zero_e2, mov_e2, "E2 data")):
        draw_box(ax, np.median([e["standoff"] for e in nom]), 0.22)
        ax.scatter(mov[::5, 0], mov[::5, 1], s=4, color="#c0392b", alpha=.4, label=f"wheels moving inside the box ({len(mov)/1000:.0f} s)")
        ax.scatter(zero[::5, 0], zero[::5, 1], s=4, color="#2e7d32", alpha=.6, label=f"wheels exactly zero ({len(zero)/1000:.0f} s)")
        r = np.hypot(*zero.T); ax.set_title(f"{name}: where the demonstrations say \"zero\"\nparked {np.median(r)*100:.1f} cm from the centre (median), "
                                           f"{(zero[:,0] < 0).mean()*100:.0f}% on the entry side", fontsize=10)
        ax.legend(fontsize=7, loc="lower left", markerscale=3)
    parks, restarts, finals = eval_points(eval_dir, policy_prefix)
    ax = axes[2]; draw_box(ax, np.median([e["standoff"] for e in nom]), 0.22)
    ax.scatter(zero_nom[::5, 0], zero_nom[::5, 1], s=4, color="#2e7d32", alpha=.3, label="demonstrated zero (nominal)")
    ax.scatter(parks[:, 0], parks[:, 1], s=14, color="#1f77b4", label=f"policy parks here ({len(parks)})")
    ax.scatter(restarts[:, 0], restarts[:, 1], s=14, color="#d98c00", marker="x", label=f"…and restarts here ({len(restarts)})")
    for a, b in zip(parks, restarts): ax.plot([a[0], b[0]], [a[1], b[1]], color="grey", lw=0.4, alpha=.5)
    rp, rr = np.hypot(*parks.T), np.hypot(*restarts.T)
    ax.set_title(f"closed loop, {policy_prefix.rstrip('_')}: park → restart\nparks {np.median(rp)*100:.1f} cm, restarts {np.median(rr)*100:.1f} cm from the centre, "
                 f"{((restarts[:,0]-parks[:,0]) > 0).mean()*100:.0f}% crept towards the target", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def fig_time(eval_dir, episode, policies, path):
    fig, axes = plt.subplots(len(policies) + 1, 1, figsize=(12, 2.4 * (len(policies) + 1)), sharex=True)
    for ax, pol in zip(axes, ["expert"] + policies):
        f = Path(eval_dir) / "traces" / f"{pol}_episode_{episode}.npz"
        with np.load(f, allow_pickle=False) as z:
            t = z["time"]; obs = z["observation"]; act = z["action"]; ph = z["expert_phase"].astype(str); done = z["expert_navigation_done"].astype(bool)
        goal, b, s = goal_of(obs[0, 18:21]); xy, _ = to_goal_frame(obs[:, :2], obs[:, 3], goal, b); r = np.hypot(*xy.T)
        w = np.abs(act[:, :4]).mean(axis=1); ax.plot(t, w, color="k", lw=1.2, label="|wheel command| (rad/s)")
        ax2 = ax.twinx(); ax2.plot(t, r, color="#1f77b4", lw=1, ls="--", label="distance to box centre (m)"); ax2.axhline(POS_TOL, color="#1f77b4", lw=0.6, ls=":"); ax2.set_ylim(0, 1.0)
        if pol == "expert":
            ph = ph.copy(); ph[done] = "done"
            for p, col in PHASE_COLOUR.items():
                sel = ph == p
                if sel.any(): ax.fill_between(t, 0, w.max() * 1.05, where=sel, color=col, alpha=.12, label=p)
        ax.set_title(f"{pol}: wheel command (black) and distance to the box centre (blue), episode {episode}", fontsize=10); ax.set_ylim(0, None)
        ax.legend(fontsize=7, loc="upper right"); ax2.legend(fontsize=7, loc="center right")
    axes[-1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("nominal"); ap.add_argument("e2"); ap.add_argument("eval_dir"); ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--policy", default="mlp_polar_seed0"); ap.add_argument("--time-episode", default="000000")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nom = episodes(a.nominal, a.episodes); e2 = episodes(a.e2, a.episodes)
    fig_single(nom[0], "expert from the nominal start: align → drive → orient → done", out / "1_expert_nominal.png")
    inside = [e for e in e2 if np.hypot(*e["xy"][0]) < e["standoff"]] or e2
    fig_single(inside[0], "expert from an E2 start (inside the standoff circle)", out / "2_expert_e2.png")
    fig_many(nom, e2, out / "3_box_entry_nominal_vs_e2.png")
    fig_rows(nom, e2, a.eval_dir, a.policy, out / "4_zero_rows_and_policy_creep.png")
    fig_time(a.eval_dir, a.time_episode, [a.policy], out / "5_wheel_command_vs_time.png")
    print("figures in", out)
