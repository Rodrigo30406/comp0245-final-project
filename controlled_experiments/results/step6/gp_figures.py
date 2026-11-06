"""Descriptive step-6 figures from explicitly named, already completed outputs.

No recorded evaluation data is opened, and a missing episode-reference cache is
never regenerated. ``--root`` is an input task tree (results/ and trained/);
``--output`` is both the G_pilot.json input directory and the output directory.
The chain checks the preregistration before invoking this script as a stage.
"""
import argparse
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gp_common import TASK, OLD, MODELS, ar, pilot_choice, write_json


PARTS = ("pilot", "dev", "tolerance", "offline", "smoothness")
SEGMENTS = ("nav_hold", "reach_move", "reach_hold")
Z_CELLS = ("z<0.45", "z>=0.45")


def is_smoke_path(path):
    return Path(path).resolve().is_relative_to(TASK / "results" / "step6" / "smoke")


def save_figure(fig, output, stem):
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{ext}", dpi=180)
    plt.close(fig)


def write_table(output, stem, lines, result):
    write_json(output / f"{stem}.json", result)
    (output / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def summary(Y, n_boot):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or not Y.size or not np.isfinite(Y).all():
        raise ValueError("descriptive bootstrap needs a nonempty finite seed x episode matrix")
    b = ar.boot2(Y, n_boot=n_boot, seed=ar.BOOT_SEED)
    return {**{key: b[key] for key in ("mean", "lo", "hi")},
            "per_seed": Y.mean(axis=1).tolist(), "n_seeds": len(Y), "n_episodes": Y.shape[1]}


def ci_text(j, percentage=True):
    scale = 100 if percentage else 1
    return f"{scale*j['mean']:.6f} [{scale*j['lo']:.6f}, {scale*j['hi']:.6f}]"


def existing_arm(root, model, seeds):
    paths = [root / "results" / f"{model}_s{seed}_dev" for seed in seeds]
    if not all((path / "summary.txt").is_file() for path in paths):
        return None
    return paths


def matching_trajectory_row(original, repeated):
    if ar._true(original.get("success")) != ar._true(repeated.get("success")):
        return False
    if original.get("stop_reason") != repeated.get("stop_reason"):
        return False
    return all(np.isclose(ar._f(original, key), ar._f(repeated, key), rtol=0, atol=1e-9, equal_nan=True)
               for key in ("min_error", "arm_extend_time"))


def supplemental_trajectory_dirs(root, run, trajectory_dirs):
    """Use only verified repeats of the matching merged C3 rows."""
    folder = root / "results" / "step6" / "smoothness" / run.name
    repeated = None
    selected = []
    for row in run.rows:
        episode = row["episode"]
        filename = f"ep_{episode:04d}.npz"
        if (trajectory_dirs[episode] / filename).is_file() or not (folder / "traj" / filename).is_file():
            continue
        if repeated is None:
            if not (folder / "summary.txt").is_file() or not (folder / "config.json").is_file():
                raise RuntimeError(f"supplemental trajectory run is incomplete: {folder}")
            repeated = ar.load_run(str(folder))
            if (repeated.meta.get("warnings") or repeated.meta.get("no_early_stop") is not True
                    or repeated.meta.get("model") != run.meta.get("model")
                    or repeated.meta.get("seed") != run.meta.get("seed")
                    or len(repeated.rows) != len(repeated.by_episode)
                    or any(r.get("stop_reason") in ("policy_error", "harness_error") or r.get("error_message")
                           or r.get("hold10_stop") == "policy_error" for r in repeated.rows)):
                raise RuntimeError(f"invalid supplemental trajectory run: {folder}")
            args = repeated.config.get("args", {})
            if args.get("save_traj") is not True or args.get("traj_every") != 10:
                raise RuntimeError(f"supplemental trajectory requires --save-traj --traj-every 10: {folder}")
        if episode not in repeated.by_episode or not matching_trajectory_row(row, repeated.by_episode[episode]):
            raise RuntimeError(f"supplemental trajectory differs from merged C3 row: {folder}, episode {episode}")
        trajectory_dirs[episode] = folder / "traj"
        selected.append(episode)
    return selected, folder


def load_arm(root, model, seeds):
    """D/F's exact stalled replacement, including its matching trajectory path."""
    paths = existing_arm(root, model, seeds)
    if paths is None:
        return None
    reference = root / "results" / "step3" / "cache_ref_dev.csv"
    if not reference.is_file():
        raise RuntimeError(f"missing frozen reference; refusing implicit rebuild: {reference}")
    runs, notes = [], []
    for path in paths:
        run = ar.load_run(str(path))
        if run.meta.get("warnings"):
            raise RuntimeError(f"run configuration mismatch: {run.meta['warnings']}")
        trajectory_dirs = {row["episode"]: path / "traj" for row in run.rows}
        stalled = sorted(row["episode"] for row in run.rows if row.get("stop_reason") == "stalled")
        if model == OLD and run.meta.get("no_early_stop") is not True and stalled:
            candidates = [root / "results" / step / "nostall" / run.name
                          for step in ("step3", "step5", "step6")]
            replacement = next((p for p in candidates if (p / "summary.txt").is_file()), None)
            if replacement is None:
                raise RuntimeError(f"missing stalled replacement: {run.name}")
            rr = ar.load_run(str(replacement))
            if rr.meta.get("no_early_stop") is not True or sorted(rr.by_episode) != stalled:
                raise RuntimeError(f"invalid stalled replacement: {replacement}")
            rows = [rr.by_episode.get(row["episode"], row) for row in run.rows]
            run = ar.Run(run.path, rows, run.header, run.config, dict(run.meta))
            trajectory_dirs.update({e: replacement / "traj" for e in stalled})
            notes.append(f"{run.name}: replaced {len(stalled)} stalled rows from {replacement}")
        if model != OLD and run.meta.get("no_early_stop") is not True:
            raise RuntimeError(f"new G/P run requires --no-early-stop: {path}")
        if any(row.get("stop_reason") in ("harness_error", "policy_error") or row.get("error_message")
               or row.get("hold10_stop") == "policy_error" for row in run.rows):
            raise RuntimeError(f"harness_error / policy_error refuses descriptive analysis: {path}")
        ar.prepare(run, out_dir=str(reference.parent))
        if model == OLD:
            supplemented, source = supplemental_trajectory_dirs(root, run, trajectory_dirs)
            if supplemented:
                notes.append(f"{run.name}: verified missing matching trajectories for episodes {supplemented} from {source}")
        run.step6_trajectory_dirs = trajectory_dirs
        runs.append(run)
    ar.check_integrity(runs, f"step6 figures {model} DEV")
    return {"runs": runs, "replacement_notes": notes}


def pilot_figure(output):
    path = output / "G_pilot.json"
    if not path.is_file():
        return {"status": "skipped", "reason": f"missing {path}"}
    j = json.loads(path.read_text())
    rates = {int(delta): np.asarray(values, dtype=float) for delta, values in j["per_seed"].items()}
    baseline = np.asarray(j["c3_per_seed"], dtype=float)
    if set(rates) != {9, 27, 81} or any(v.shape != (2,) for v in [baseline, *rates.values()]):
        raise ValueError("G pilot requires exactly two seed rates for C3 and deltas 9,27,81")
    if any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in [baseline, *rates.values()]):
        raise ValueError("invalid G pilot success rates")
    selected = pilot_choice(rates, baseline)
    if int(j["delta_star"]) != selected["delta_star"] or bool(j["continue"]) != selected["continue"]:
        raise RuntimeError("G_pilot.json does not follow the frozen selection/stop rule")
    means = [baseline.mean(), *[rates[d].mean() for d in (9, 27, 81)]]
    x = [0, 81, 243, 729]
    fig, ax = plt.subplots(figsize=(6.8, 4), constrained_layout=True)
    ax.plot(x[1:], 100*np.asarray(means[1:]), "o-", label="G (2-seed mean)")
    ax.scatter([0], [100*means[0]], marker="s", label="C3 (2-seed mean)")
    ax.axhline(100*means[0], color="grey", linestyle=":", linewidth=.8)
    for duration, values in zip(x, [baseline, *[rates[d] for d in (9, 27, 81)]]):
        ax.scatter([duration]*len(values), 100*values, s=16, alpha=.5, color="black")
    synthetic = bool(j.get("synthetic_gate_fixture"))
    ax.set(xlabel="Chunk horizon (ms); C3 at 0", ylabel="Pilot success (%)", ylim=(-1, 101),
           xticks=x, title=("SYNTHETIC SOFTWARE SMOKE: " if synthetic else "") +
           f"Pilot; selected delta = {j['delta_star']} steps")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save_figure(fig, output, "fig_G_pilot")
    return {"status": "complete", "input": str(path), "synthetic": synthetic, "delta_star": j["delta_star"],
            "continue": j["continue"], "horizon_ms": x, "success_mean": means}


def dev_table(arms, output, n_boot):
    lines = ["DEV descriptive table: seed mean [two-level bootstrap 95% CI]; proportions in %, policy_ms in ms.",
             "The primary verdict uses the 97.5% CI in mechanisms; this descriptive table is not used for verdicts."]
    result = {"set": "dev", "confidence_level": .95, "n_boot": n_boot, "bootstrap_seed": ar.BOOT_SEED,
              "software_smoke": is_smoke_path(output), "models": {}}
    if result["software_smoke"]:
        lines.insert(0, "Software smoke test; contains synthetic results, not official experiment results.")
    for name, arm in arms.items():
        runs = arm["runs"]
        episodes = sorted(runs[0].by_episode)
        metrics = {"success": summary(ar.metric_matrix(runs, episodes, "success"), n_boot)}
        for cls in ar.FAILURE_CLASSES:
            metrics[cls] = summary(ar.matrix(runs, episodes, lambda row, cls=cls: float(row["fclass"] == cls)), n_boot)
        metrics["policy_ms"] = summary(ar.matrix(runs, episodes, lambda row: float(row["policy_ms_mean"])), n_boot)
        result["models"][name] = {"runs": [r.path for r in runs], **arm, "metrics": metrics}
        # Run objects cannot be serialized; retain their explicit paths only.
        result["models"][name]["runs"] = [r.path for r in runs]
        lines += ["", name, *arm["replacement_notes"]]
        lines += [f"  {metric}: {ci_text(value, metric != 'policy_ms')}" for metric, value in metrics.items()]
    write_table(output, "DEV_comparison_table", lines, result)
    return {"status": "complete", "models": list(arms)}


def tolerance_curve(arms, output, n_boot):
    table = np.asarray([5, 7.5, 10, 15, 20, 30, 50, 100], dtype=float)
    grid = np.unique(np.r_[np.geomspace(2, 100, 121), table])
    definition = ("Same method as step5/tolerance_curve.py: P(min_error < tau), strict threshold; "
                  "min_error is frozen at protocol v1 stop, before extended hold. "
                  "DEV C3 uses the exact stalled-replaced controls used for step6 comparisons.")
    result = {"definition": definition, "set": "dev", "confidence_level": .95,
              "bootstrap": {"method": "analyze_rollouts.boot2", "draws": n_boot, "seed": ar.BOOT_SEED},
              "software_smoke": is_smoke_path(output), "models": {}}
    lines = [definition, "Each tolerance is the seed mean [two-level bootstrap 95% CI], in %."]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for name, arm in arms.items():
        runs = arm["runs"]
        episodes = sorted(runs[0].by_episode)
        errors = ar.matrix(runs, episodes, lambda row: ar._f(row, "min_error"))
        if np.isnan(errors).any() or (errors < 0).any():
            raise ValueError(f"invalid min_error in {name}")
        checks = {metric: bool(np.array_equal((errors < tau/1000).astype(float),
                                              ar.metric_matrix(runs, episodes, metric)))
                  for tau, metric in ((10, "success"), (20, "reach_2cm"), (50, "reach_5cm"))}
        if not all(checks.values()):
            raise RuntimeError(f"tolerance/rollout metric inconsistency in {name}: {checks}")
        points = [{"tau_mm": float(tau), **summary((errors < tau/1000).astype(float), n_boot)} for tau in grid]
        result["models"][name] = {"curve": points, "checks": checks, "runs": [r.path for r in runs],
                                   "replacement_notes": arm["replacement_notes"]}
        lines += ["", name, *arm["replacement_notes"]]
        lines += [f"  {p['tau_mm']:g} mm: {ci_text(p)}" for p in points if p["tau_mm"] in table]
        ax.plot(grid, [100*p["mean"] for p in points], label=name)
        ax.fill_between(grid, [100*p["lo"] for p in points], [100*p["hi"] for p in points], alpha=.12)
    ax.set(xscale="log", xlabel="Tolerance (mm)", ylabel="P(min_error < tolerance) (%)", ylim=(0, 101),
           title=("SYNTHETIC SOFTWARE SMOKE: " if result["software_smoke"] else "") + "DEV tolerance curve",
           xticks=[2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.axvspan(5, 20, color="grey", alpha=.05)
    ax.axvline(10, color="grey", linestyle=":", linewidth=.8)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save_figure(fig, output, "fig_tolerance_GP")
    write_table(output, "tolerance_curve_GP", lines, result)
    return {"status": "complete", "models": list(arms)}


def torque_cells(j):
    """Normalize C3/P's shared offline schema and add a correctly pooled z cell."""
    result = {}
    for cell in j["torque_segments"]:
        key = (cell["group"], cell["segment"], cell["z"])
        if key in result:
            raise ValueError(f"duplicate offline cell: {key}")
        n = int(cell["n"])
        value = cell.get("rmse_pooled")
        if n and (value is None or not np.isfinite(value) or value < 0):
            raise ValueError(f"invalid nonempty offline cell: {key}")
        result[key] = {"n": n, "rmse_pooled": value, "rmse": cell.get("rmse")}
    for group in ("all", "clean", "noisy"):
        for segment in SEGMENTS:
            cells = [result[group, segment, z] for z in Z_CELLS]
            n = sum(c["n"] for c in cells)
            value = np.sqrt(sum(c["n"]*c["rmse_pooled"]**2 for c in cells if c["n"])/n) if n else None
            result[group, segment, "all_z"] = {"n": n, "rmse_pooled": value}
    return result


def offline_table(root, output, seeds):
    paths = {name: [root / "trained" / f"{model}_s{seed}.offline.json" for seed in seeds]
             for name, model in (("C3", OLD), ("P", MODELS["P"]))}
    missing = [str(p) for group in paths.values() for p in group if not p.is_file()]
    if missing:
        return {"status": "skipped", "reason": "missing offline inputs", "missing": missing}
    models, per_seed = {}, {}
    for name in ("C3", "P"):
        models[name] = []
        per_seed[name] = []
        expected_model = OLD if name == "C3" else MODELS["P"]
        for seed, path in zip(seeds, paths[name]):
            j = json.loads(path.read_text())
            if j["model"] != expected_model or int(j["seed"]) != seed:
                raise RuntimeError(f"offline model/seed mismatch: {path}")
            cells = torque_cells(j)
            models[name].append(cells)
            per_seed[name].append({"seed": seed, "input": str(path), "n_val_rows": j["n_val_rows"],
                                   "n_val_episodes": j["n_val_episodes"],
                                   "reference_nrmse": j.get("reference_nrmse")})
    rows = []
    for key in models["C3"][0]:
        c3 = [cells[key] for cells in models["C3"]]
        pd = [cells[key] for cells in models["P"]]
        if any(a["n"] != b["n"] for a, b in zip(c3, pd)):
            raise RuntimeError(f"C3/P offline row populations differ: {key}")
        row = dict(zip(("group", "segment", "z"), key))
        row["n_per_seed"] = [c["n"] for c in c3]
        for name, cells in (("C3", c3), ("P", pd)):
            values = [c["rmse_pooled"] for c in cells]
            nonempty = [v for v in values if v is not None]
            row[name] = {"per_seed": values, "seed_mean": float(np.mean(nonempty)) if nonempty else None,
                         "seed_median": float(np.median(nonempty)) if nonempty else None}
        rows.append(row)
    for a, b in zip(per_seed["C3"], per_seed["P"]):
        if any(a[k] != b[k] for k in ("n_val_rows", "n_val_episodes")):
            raise RuntimeError("C3/P offline validation populations differ")
        ref = b["reference_nrmse"]
        if not ref or not all(np.isfinite(ref[k]) and ref[k] >= 0 for k in ("e_mean", "qd_des_mean")):
            raise ValueError("missing/invalid P reference nRMSE")
    hold = next(row for row in rows if (row["group"], row["segment"], row["z"]) == ("all", "reach_hold", "all_z"))
    valid_hold = all(hold[name]["seed_median"] is not None for name in ("C3", "P"))
    result = {"descriptive_only": True, "software_smoke": is_smoke_path(root), "seeds": seeds,
              "definition": "Per seed pooled 7-joint torque RMSE (N m); all_z pools squared errors weighted by row count. "
                            "Tables report mean/median across seeds, without adoption inference.",
              "per_seed_inputs": per_seed, "rows": rows,
              "MP4_reach_hold": {"C3_seed_median": hold["C3"]["seed_median"],
                                  "P_seed_median": hold["P"]["seed_median"],
                                  "P_lower": bool(hold["P"]["seed_median"] < hold["C3"]["seed_median"]) if valid_hold else None}}
    lines = ["P offline comparison (descriptive): torque RMSE, N·m; mean and median are both aggregated per seed.", result["definition"],
             "group / segment / z | C3 mean, median | P mean, median"]
    if result["software_smoke"]:
        lines.insert(0, "Software smoke test on a small training cache; not an official offline result.")
    for row in rows:
        fmt = lambda v: "NA" if v is None else f"{v:.9f}"
        lines.append(f"{row['group']} / {row['segment']} / {row['z']} | " + " | ".join(
            f"{fmt(row[name]['seed_mean'])}, {fmt(row[name]['seed_median'])}" for name in ("C3", "P")))
    lines += ["", "P reference nRMSE: e mean; qd_des mean (per seed)"]
    lines += [f"s{j['seed']}: {j['reference_nrmse']['e_mean']:.9f}; {j['reference_nrmse']['qd_des_mean']:.9f}"
              for j in per_seed["P"]]
    lines.append("MP4 (descriptive): " + json.dumps(result["MP4_reach_hold"], ensure_ascii=False))
    write_table(output, "P_offline_table", lines, result)
    return {"status": "complete", "seeds": seeds}


def trajectory_differences(t, tau, reach_start):
    """Only consecutive saved 10 ms samples, both after detected arm extension."""
    t, tau = np.asarray(t, dtype=float), np.asarray(tau, dtype=float)
    if t.ndim != 1 or tau.shape != (len(t), 7) or not np.isfinite(t).all() or not np.isfinite(tau).all():
        raise ValueError("trajectory must contain finite t[N] and torque_arm[N,7]")
    dt = np.diff(t)
    if np.any(dt <= 0) or (len(dt) and not np.allclose(dt, .010, rtol=0, atol=1e-7)):
        raise ValueError("smoothness requires consecutive --traj-every 10 samples (10 ms)")
    mask = (t[:-1] >= reach_start) & (t[1:] >= reach_start)
    return np.abs(np.diff(tau, axis=0))[mask]


def smoothness(arms, output):
    if "G" not in arms:
        return {"status": "skipped", "reason": "G DEV not completed"}
    result = {"descriptive_only": True, "software_smoke": is_smoke_path(output),
              "definition": "Reach begins at each rollout's arm_extend_time (the common harness detector). "
                            "Both adjacent saved samples must be at/after it. Samples are 10 ms apart; "
                            "median is over absolute differences of all 7 torque commands. Per-seed pooled medians "
                            "are averaged equally for the descriptive G minus C3 point estimate. Episodes without "
                            "detected extension or fewer than 2 reach samples are counted and excluded; no interpolation.",
              "unit": "N m per 10 ms saved-sample difference", "models": {}}
    lines = ["G smoothness (descriptive)", result["definition"]]
    if result["software_smoke"]:
        lines.insert(0, "Software smoke test; contains synthetic trajectories, not official experiment results.")
    for name in ("C3", "G"):
        arm = arms[name]
        all_values, seed_rows = [], []
        for run in arm["runs"]:
            seed_values, episode_rows = [], []
            for row in run.rows:
                episode = row["episode"]
                reach_start = ar._f(row, "arm_extend_time")
                if not np.isfinite(reach_start):
                    episode_rows.append({"episode": episode, "status": "no_detected_extension", "n_pairs": 0})
                    continue
                path = run.step6_trajectory_dirs[episode] / f"ep_{episode:04d}.npz"
                if not path.is_file():
                    raise RuntimeError(f"missing matching smoothness trajectory: {path}")
                with np.load(path, allow_pickle=False) as z:
                    values = trajectory_differences(z["t"], z["torque_arm"], reach_start)
                episode_rows.append({"episode": episode, "trajectory": str(path), "reach_start_s": reach_start,
                                     "status": "ok" if len(values) else "too_few_reach_samples", "n_pairs": len(values),
                                     "median_abs_difference": float(np.median(values)) if len(values) else None})
                if len(values):
                    seed_values.append(values.reshape(-1))
            pooled = np.concatenate(seed_values) if seed_values else np.asarray([], dtype=float)
            seed_rows.append({"seed": run.meta["seed"], "episodes": episode_rows, "n_joint_differences": len(pooled),
                              "median_abs_difference": float(np.median(pooled)) if len(pooled) else None,
                              "n_no_extension": sum(e["status"] == "no_detected_extension" for e in episode_rows),
                              "n_too_few_samples": sum(e["status"] == "too_few_reach_samples" for e in episode_rows)})
            if len(pooled):
                all_values.append(pooled)
        seed_medians = [s["median_abs_difference"] for s in seed_rows if s["median_abs_difference"] is not None]
        result["models"][name] = {"per_seed": seed_rows, "replacement_notes": arm["replacement_notes"],
                                   "seed_mean_of_medians": float(np.mean(seed_medians)) if seed_medians else None,
                                   "pooled_median": float(np.median(np.concatenate(all_values))) if all_values else None}
        lines.append(f"{name}: seed mean of medians {result['models'][name]['seed_mean_of_medians']}; "
                     f"pooled median {result['models'][name]['pooled_median']}")
    a, b = (result["models"][name]["seed_mean_of_medians"] for name in ("C3", "G"))
    result["G_minus_C3"] = b-a if a is not None and b is not None else None
    result["MG4_lower"] = bool(b < a) if a is not None and b is not None else None
    lines.append(f"G minus C3: {result['G_minus_C3']}; MG4 descriptive: {result['MG4_lower']}")
    write_table(output, "G_smoothness", lines, result)
    return {"status": "complete", "G_minus_C3": result["G_minus_C3"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=TASK, help="input task root containing results/ and trained/")
    p.add_argument("--output", type=Path, help="G_pilot.json input and chart output directory; default ROOT/results/step6")
    p.add_argument("--experiment", choices=("G", "P", "both"), default="both")
    p.add_argument("--parts", default="all", help="all or comma-separated pilot,dev,tolerance,offline,smoothness")
    p.add_argument("--pilot-only", action="store_true", help="compatibility alias for --parts pilot")
    p.add_argument("--seeds", default="0-9", help="explicit DEV/offline seeds, e.g. 0-9 or 0,1")
    p.add_argument("--n-boot", type=int, default=ar.N_BOOT)
    p.add_argument("--require-complete", action="store_true", help="fail if any requested part lacks inputs")
    args = p.parse_args(argv)
    if args.n_boot < 1:
        p.error("--n-boot must be positive")
    seeds = ar.parse_seeds(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        p.error("--seeds must be nonempty and unique")
    parts = ["pilot"] if args.pilot_only else list(PARTS) if args.parts == "all" else args.parts.split(",")
    if not parts or set(parts) - set(PARTS):
        p.error("unknown --parts")
    root = args.root.resolve()
    output = (args.output or root / "results" / "step6").resolve()
    output.mkdir(parents=True, exist_ok=True)
    # G and P may finish simultaneously. Serialize their common table/figure
    # writes, including write_json's shared temporary filename.
    with (output / ".gp_figures.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return generate(args, root, output, seeds, parts)


def generate(args, root, output, seeds, parts):
    experiments = ("G", "P") if args.experiment == "both" else (args.experiment,)
    # Omit sections belonging solely to an unrequested experiment.
    parts = [part for part in parts if (part not in ("pilot", "smoothness") or "G" in experiments)
             and (part != "offline" or "P" in experiments)]
    manifest = {"root": str(root), "output": str(output), "experiment": args.experiment,
                "seeds": seeds, "descriptive_only": True, "software_smoke": is_smoke_path(root), "parts": {}}
    arms = {}
    if set(parts) & {"dev", "tolerance", "smoothness"}:
        # Always add every completed DEV arm to the common report figure. Thus
        # the second chain to finish produces the requested C3/G/P table while
        # the first may already report C3 plus its completed experiment.
        for name, model in (("C3", OLD), ("G", MODELS["G"]), ("P", MODELS["P"])):
            arm = load_arm(root, model, seeds)
            if arm is not None:
                arms[name] = arm
        if "C3" not in arms:
            arms.clear()
        if len(arms) > 1:
            ar.check_integrity([run for arm in arms.values() for run in arm["runs"]], "step6 combined DEV figures")
    for part in parts:
        if part == "pilot":
            result = pilot_figure(output)
        elif part == "offline":
            result = offline_table(root, output, seeds)
        elif len(arms) < 2:
            result = {"status": "skipped", "reason": "missing complete C3 and requested experiment DEV runs"}
        elif part == "dev":
            result = dev_table(arms, output, args.n_boot)
        elif part == "tolerance":
            result = tolerance_curve(arms, output, args.n_boot)
        else:
            result = smoothness(arms, output)
        if part in ("dev", "tolerance", "smoothness"):
            result["missing_requested_models"] = [e for e in experiments if e not in arms]
        manifest["parts"][part] = result
        print(f"{part}: {result['status']}", flush=True)
    # Per-experiment manifests avoid collisions when the two authorized chains
    # independently finish their figure stages.
    write_json(output / f"figures_{args.experiment}_manifest.json", manifest)
    if args.require_complete and any(r["status"] != "complete" or r.get("missing_requested_models")
                                     for r in manifest["parts"].values()):
        raise RuntimeError("required figure inputs are incomplete; see figures manifest")
    return manifest


if __name__ == "__main__":
    main()
