"""F3 shadow labels and lossless side cache, including the extended hold."""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from f_common import HERE, TASK, ar, check_freeze, sha, write_json
import bc_features as bcf
import dagger_relabel as dr
import train_policy as tp
from evaluate_policy import load_eval_set


def label_one(job):
    """Replay every step, then subsample; never replay just the training rows."""
    log = job["part"] + ".log"
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    started = time.time()
    with np.load(job["trajectory"]) as z:
        traj = {n: z[n] for n in z.files}
    if any(traj[n].dtype != np.dtype("float64") for n in dr.OBS_FIELDS):
        raise ValueError("F2 trajectories must preserve float64 observations")
    if not np.allclose(np.diff(traj["t"]), .001, rtol=0, atol=1e-12):
        raise ValueError("F3 requires all consecutive 1 kHz steps")
    shadow = dr.replay_expert(traj, job["episode_info"])
    idx = np.arange(job["offset"], len(traj["t"]), 9)
    signals = bcf.measured_signals(traj["base_pos"][idx], traj["base_bearing"][idx],
                                   traj["q_mes"][idx], traj["qd_mes"][idx], job["episode_info"]["target"])
    X = bcf.build_features("ego_novw", signals)
    v, w = bcf.wheels_to_vw(shadow["wheel_cmds"][idx])
    Y = np.column_stack((v, w, shadow["torque_arm"][idx]))
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or np.any(np.abs(Y[:, 2:]) > 30):
        raise ValueError("invalid shadow labels/features")
    np.savez_compressed(job["part"], X=X, Y=Y, nav_done=shadow["nav_done"][idx], step=idx.astype(np.int32))
    return {"episode": job["episode"], "seed": job["seed"], "offset": job["offset"],
            "trajectory": str(Path(job["trajectory"]).relative_to(TASK)),
            "trajectory_sha256": sha(job["trajectory"]), "n_steps": len(traj["t"]), "n_rows": len(idx),
            "trajectory_bytes": os.path.getsize(job["trajectory"]), "label_s": time.time() - started,
            "part": job["part"], "shadow_tau_min": float(Y[:, 2:].min()),
            "shadow_tau_max": float(Y[:, 2:].max())}


def build(selection_path, directory, pilot=False, workers=22):
    check_freeze()
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        tp.load_extra_rows(str(directory), "ego_novw")
        print(f"skip verified cache {directory}", flush=True)
        return
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    selection = json.loads(Path(selection_path).read_text())
    episodes, _ = load_eval_set(str(TASK / "data" / "train"))
    by = {e["episode"]: e for e in episodes}
    jobs = []
    parts = directory / "parts"
    parts.mkdir()
    for r in selection["rollouts"]:
        if pilot and r["episode"] not in selection["pilot_episodes"]:
            continue
        folder = HERE / "pilot" / f"c3_s{r['seed']}_train" if pilot else HERE / "rollouts" / f"c3_s{r['seed']}_train"
        if not (folder / "summary.txt").exists():
            raise RuntimeError(f"incomplete F2 run: {folder}")
        jobs.append({**r, "trajectory": str(folder / "traj" / f"ep_{r['episode']:04d}.npz"),
                     "episode_info": by[r["episode"]],
                     "part": str(parts / f"s{r['seed']}_ep{r['episode']:04d}.npz")})
    expected = 2 if pilot else 400
    if len(jobs) != expected:
        raise RuntimeError(f"expected {expected} trajectories, got {len(jobs)}")
    started = time.time()
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        records = []
        for r in pool.imap(label_one, jobs):
            records.append(r)
            if len(records) % 10 == 0 or len(records) == len(jobs):
                print(f"F3 {len(records)}/{len(jobs)} trajectories labelled ({time.time()-started:.1f}s)", flush=True)
    n = sum(r["n_rows"] for r in records)
    specs = {"X": (np.float64, (20,)), "Y": (np.float64, (9,)), "nav_done": (np.bool_, ()),
             "episode": (np.int32, ()), "seed": (np.int32, ()), "step": (np.int32, ())}
    arrays = {name: np.lib.format.open_memmap(directory / (name + ".npy"), mode="w+", dtype=dt,
                                             shape=(n, *shape)) for name, (dt, shape) in specs.items()}
    start = 0
    for r in records:
        end = start + r["n_rows"]
        with np.load(r["part"]) as z:
            for name in ("X", "Y", "nav_done", "step"):
                arrays[name][start:end] = z[name]
        arrays["episode"][start:end] = r["episode"]
        arrays["seed"][start:end] = r["seed"]
        r["row_start"], r["row_end"] = start, end
        r.pop("part")
        start = end
    for a in arrays.values():
        a.flush()
    arrays.clear()
    manifest = {"format": "bc_dagger_v1", "feature_set": "ego_novw", "n_rows": n,
                "n_trajectories": len(jobs), "split_sha256": sha(tp.SPLIT_PATH), "pilot": pilot,
                "selection_sha256": sha(selection_path), "trajectories": records,
                "code_sha256": {f: sha(TASK / f) for f in ("dagger_relabel.py", "bc_features.py",
                    "evaluate_policy.py", "robotnik_arm_control.py", "results/step5/build_dagger_cache.py")},
                "rule": "Replay every 1 kHz step, including all extended hold steps, before offset::9. "
                        "default_rng(6).integers(0,9) once per trajectory in sorted (seed,episode) order. "
                        "Features: build_features('ego_novw', measured_signals(...)); labels: shadow (v,w,tau). "
                        "All numeric training arrays remain float64; no weighting or reach-based filtering.",
                "arrays": {name: {"sha256": sha(directory / (name + ".npy")), "dtype": str(np.dtype(dt)),
                                  "shape": [n, *shape]} for name, (dt, shape) in specs.items()},
                "elapsed_s": time.time() - started}
    write_json(directory / "manifest.json", manifest)
    tp.load_extra_rows(str(directory), "ego_novw")
    for r in records:
        if sha(TASK / r["trajectory"]) != r["trajectory_sha256"]:
            raise RuntimeError("trajectory changed during cache generation")
    shutil.rmtree(parts)
    print(f"F3 cache validated: {n} rows, {len(records)} trajectories, {time.time()-started:.1f}s", flush=True)


def describe(directory):
    """Write the required descriptive checks before any experiment F DEV run."""
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    tp.load_extra_rows(str(directory), "ego_novw")
    Y = np.load(directory / "Y.npy", mmap_mode="r")
    nav = np.load(directory / "nav_done.npy", mmap_mode="r")
    expert = tp.Cache("train")
    tau = np.asarray(expert.cols["label_tau"], dtype=float)

    def stats(a):
        a = np.asarray(a)
        return {"n_rows": len(a), "clip_fraction": float(np.mean(np.abs(a) >= 30)),
                "abs_tau_p50": float(np.percentile(np.abs(a), 50)),
                "abs_tau_p99": float(np.percentile(np.abs(a), 99)),
                "min": float(a.min()), "max": float(a.max())}

    out = {"n_rows": len(Y), "n_trajectories": m["n_trajectories"],
           "shadow_nav_done_fraction": float(nav.mean()), "dagger_tau": stats(Y[:, 2:]),
           "expert_tau": stats(tau), "cache_manifest_sha256": sha(directory / "manifest.json")}
    ref = {}
    split = tp.load_split()
    for r in m["trajectories"]:
        e = r["episode"]
        ref[e] = ar.episode_reference(str(Path(tp.TRAIN_DIR) / split["episodes"][str(e)]["file"]))
    runs = [ar.load_run(str(HERE / "rollouts" / f"c3_s{s}_train")) for s in range(5)]
    for run in runs:
        ar.classify_run(run, ref)
    rows = [r for run in runs for r in run.rows]
    if len(rows) != 400 or any(r["stop_reason"] in ("harness_error", "policy_error") for r in rows):
        raise RuntimeError("invalid 400-episode F2 background")
    out["rollout_background"] = {"n": len(rows), "success": float(np.mean([ar.metric_value(r, "success") for r in rows])),
                                 "classes": ar.class_counts(rows)}
    lines = ["F3 descriptive check (written before any F-model DEV result)", "Cache rule: " + m["rule"],
             f"Trajectories: {m['n_trajectories']}; training rows: {len(Y)}; shadow nav_done row fraction: {nav.mean():.9f}",
             "Torque clip fraction is computed over all rows × 7 joints; p50/p99 are pooled quantiles of |tau| (N·m).",
             "DAgger: " + json.dumps(out["dagger_tau"], ensure_ascii=False),
             "Original expert stride-9 rows: " + json.dumps(out["expert_tau"], ensure_ascii=False),
             "400 C3 rollouts (background only): " + json.dumps(out["rollout_background"], ensure_ascii=False)]
    write_json(HERE / "F_data.json", out)
    (HERE / "F_data.txt").write_text("\n".join(lines) + "\n")
    expert.release()
    print("F3 descriptive data written before DEV", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--describe", action="store_true")
    a = p.parse_args()
    build(HERE / "F2_selection.json", a.directory, a.pilot, a.workers)
    if a.describe:
        describe(a.directory)


if __name__ == "__main__":
    main()
