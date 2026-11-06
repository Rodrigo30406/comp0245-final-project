"""Resumable, gated step-6 chains.  Only explicit stages touch evaluation sets."""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from gp_common import (HERE, TASK, OLD, MODELS, SEEDS, ar, check_freeze, sha, write_json,
                       dev_pass, hard_pass, pilot_choice, post_hoc)
import train_policy as tp

WORKERS = int(os.environ.get("WORKERS", "22"))
JOBS = int(os.environ.get("JOBS", "1"))
GB_MIB = 1024
LIMIT_MIB = 6 * GB_MIB
CODE_DIR = HERE
RUN_ROOT = TASK
SMOKE_ROOT = None
SMOKE_CASE = "pass"
SMOKE_EPOCHS = 2


def model_path(model, seed):
    return RUN_ROOT / "trained" / f"{model}_s{seed}.npz"


def source_cache():
    return RUN_ROOT / "cache" if SMOKE_ROOT else Path(tp.CACHE_DIR)


def trigger_cache():
    return RUN_ROOT / "trigger" if SMOKE_ROOT else Path(tp.TRIGGER_DIR)


def smoke_args():
    return ["--smoke-root", SMOKE_ROOT] if SMOKE_ROOT else []


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(experiment, stage, status, **fields):
    record = {"time": now(), "experiment": experiment, "stage": stage, "status": status, **fields}
    with open(HERE / f"stage{experiment}_events.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def status(experiment, stage, state, **fields):
    write_json(HERE / f"chain{experiment}_status.json", event(experiment, stage, state, **fields))


def env():
    return {**os.environ, "MPLCONFIGDIR": "/tmp/mplcfg", "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}


def command(args, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        subprocess.run([sys.executable, *map(str, args)], cwd=TASK, env=env(),
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def available_mib():
    return next(int(s.split()[1]) / 1024 for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:"))


def adaptive_jobs(available, peak, requested=5):
    if not np.isfinite(peak) or peak <= 0 or peak > LIMIT_MIB:
        raise RuntimeError(f"Training peak RSS {peak} MiB must be >0 and at most 6 GiB")
    jobs = min(5, int(requested), int((available - 2 * GB_MIB) // peak))
    if jobs < 1:
        raise RuntimeError("Not enough available memory to run one training process after reserving 2 GiB")
    return jobs


def proc_peak_mib(pid):
    try:
        return max(int(s.split()[1]) / 1024 for s in Path(f"/proc/{pid}/status").read_text().splitlines()
                   if s.startswith(("VmHWM:", "VmRSS:")))
    except (OSError, ValueError):
        return 0


def side_cache(experiment, delta=None):
    base = RUN_ROOT / "side_cache" if SMOKE_ROOT else TASK / "results" / "step3" / "cache"
    return base / (f"chunk_d{delta}" if experiment == "G" else "pdref")


def train_args(experiment, seed, out, delta=None):
    args = [TASK / "train_policy.py", "train", "--model", MODELS[experiment], "--seed", seed,
            "--trigger-copies", 16, "--out", out, "--side-cache", side_cache(experiment, delta),
            "--cache", source_cache(), "--trigger-cache", trigger_cache()]
    if experiment == "G":
        args += ["--chunk-k", 10, "--chunk-delta", delta]
    if SMOKE_ROOT:
        args += ["--max-epochs", SMOKE_EPOCHS, "--max-train-episodes", 2, "--max-val-episodes", 1]
    return args


def train_jobs(experiment, jobspecs, delta=None):
    """First model establishes peak; monitor every child and cap jobs by live memory."""
    todo = [(s, Path(p)) for s, p in jobspecs if not Path(p).exists()]
    existing = [Path(p) for _, p in jobspecs if Path(p).exists()]
    peaks = []
    for p in existing:
        with np.load(p) as z:
            peaks.append(float(json.loads(str(z["meta"]))["peak_rss_mib"]))
    running = []
    try:
        while todo or running:
            # Before the very first fit, run a single process; subsequently the
            # observed maximum peak determines the concurrency (at most five).
            limit = adaptive_jobs(available_mib(), max(peaks), JOBS) if peaks else 1
            while todo and len(running) < limit:
                seed, out = todo.pop(0)
                out.parent.mkdir(parents=True, exist_ok=True)
                # Model npz is saved atomically by training. Never --force.
                fh = open(out.with_suffix(".log"), "a")
                proc = subprocess.Popen([sys.executable, *map(str, train_args(experiment, seed, out, delta))],
                                        cwd=TASK, env=env(), stdout=fh, stderr=subprocess.STDOUT)
                running.append((seed, out, proc, fh))
                event(experiment, "train", "start", seed=seed, out=str(out.relative_to(TASK)),
                      adaptive_concurrency=limit, available_mib=available_mib())
            time.sleep(1)
            for item in list(running):
                seed, out, proc, fh = item
                peak = proc_peak_mib(proc.pid)
                if peak > LIMIT_MIB:
                    proc.terminate()
                    proc.wait()
                    raise RuntimeError(f"seed {seed} peak RSS {peak:.0f} MiB > 6 GiB; stopping training")
                if proc.poll() is not None:
                    fh.close()
                    running.remove(item)
                    if proc.returncode:
                        raise RuntimeError(f"Training seed {seed} exit code {proc.returncode}; see {out.with_suffix('.log')}")
                    with np.load(out) as z:
                        meta = json.loads(str(z["meta"]))
                    peak = float(meta["peak_rss_mib"])
                    if peak > LIMIT_MIB:
                        raise RuntimeError(f"seed {seed} peak RSS > 6 GiB")
                    peaks.append(peak)
                    event(experiment, "train", "end", seed=seed, peak_rss_mib=peak,
                          train_s=meta["train_s"], epochs=meta["epochs_run"])
        if peaks:
            write_json(HERE / f"{experiment}_training_memory.json", {"peak_rss_mib": max(peaks),
                        "jobs_requested": JOBS, "jobs_cap": 5, "reserve_mib": 2048, "limit_mib": LIMIT_MIB})
    finally:
        for _, _, proc, fh in running:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            fh.close()


def load_evidence(experiment):
    path = HERE / f"{experiment}_mechanisms.json"
    return json.loads(path.read_text()) if path.exists() else {}


def require_gate(experiment, set_name):
    if set_name == "hard" and not dev_pass(load_evidence(experiment)):
        raise RuntimeError("DEV (1)/(2) gate not passed; HARD not opened")
    if set_name == "confirm" and not hard_pass(load_evidence(experiment)):
        raise RuntimeError("DEV/HARD gate not passed; CONFIRM not opened")


def validate_run(folder, expected, model_arg, trajectories=False):
    r = ar.load_run(str(folder))
    if (sorted(r.by_episode) != sorted(expected) or len(r.rows) != len(expected)
            or r.meta.get("warnings") or r.meta.get("policy_arg") != model_arg
            or r.meta.get("no_early_stop") is not True):
        raise RuntimeError(f"Run configuration/coverage mismatch: {folder}")
    errors = {key: sum(row.get("stop_reason") == key for row in r.rows)
              for key in ("harness_error", "policy_error")}
    errors["extended_hold_policy_error"] = sum(row.get("hold10_stop") == "policy_error" for row in r.rows)
    if any(errors.values()) or any(row.get("error_message") for row in r.rows):
        raise RuntimeError(f"Run errors: {folder}: {errors}")
    if trajectories and any(not (Path(folder) / "traj" / f"ep_{e:04d}.npz").exists() for e in expected):
        raise RuntimeError(f"Missing trajectories: {folder}")
    return r, errors


def evaluate(experiment, model, set_name, folder, episodes=None, save_every=None):
    check_freeze()
    require_gate(experiment, set_name)
    if set_name == "confirm":
        token = HERE / f"{experiment}_confirm_opened.json"
        if not token.exists():
            raise RuntimeError("CONFIRM was not opened through the gate with --allow-confirm")
        ar.ALLOW_CONFIRM = True
    folder = Path(folder)
    model_arg = str(Path(model).relative_to(TASK))
    if episodes is None:
        with open(TASK / "data" / set_name / "config.json") as f:
            episodes = list(range(json.load(f)["n_episodes"]))
    if (folder / "summary.txt").exists():
        _, errors = validate_run(folder, episodes, model_arg, save_every is not None)
        event(experiment, "run", "skip_complete", folder=str(folder.relative_to(TASK)), n=len(episodes), **errors)
        return
    if folder.exists():
        event(experiment, "run", "interrupted_directory_deleted", folder=str(folder.relative_to(TASK)))
        shutil.rmtree(folder)
    args = ["evaluate_policy.py", "--policy", "policies/bc_policy.py:make_policy", "--policy-arg", model_arg,
            "--eval-set", f"data/{set_name}", "--out-dir", folder.relative_to(TASK), "--workers", WORKERS,
            "--no-early-stop", "--episodes", ",".join(map(str, episodes))]
    if save_every is not None:
        args += ["--save-traj", "--traj-every", save_every]
    event(experiment, "run", "start", folder=str(folder.relative_to(TASK)), n=len(episodes))
    command(args, str(folder) + ".log")
    _, errors = validate_run(folder, episodes, model_arg, save_every is not None)
    event(experiment, "run", "end", folder=str(folder.relative_to(TASK)), n=len(episodes), **errors)


def build_caches(experiment):
    if SMOKE_ROOT:
        split = tp.load_split()
        for path, builder in ((source_cache(), lambda: tp.build_cache(cache_dir=str(source_cache()),
                workers=min(WORKERS, 2), train_episodes=split["train_episodes"][:2],
                val_episodes=split["val_episodes"][:1])),
                (trigger_cache(), lambda: tp.build_trigger_cache(trigger_dir=str(trigger_cache()),
                workers=min(WORKERS, 2), train_episodes=split["train_episodes"][:2]))):
            if not (path / "manifest.json").exists():
                if path.exists():
                    shutil.rmtree(path)
                builder()
    for delta in (9, 27, 81) if experiment == "G" else (None,):
        from gp_training import load_side
        path = side_cache(experiment, delta)
        kind = "chunk" if experiment == "G" else "pd_reference"
        if (path / "manifest.json").exists():
            load_side(path, kind, source_cache(), trigger_cache(), tp.SPLIT_PATH,
                      allow_subset=bool(SMOKE_ROOT), k=10, delta=delta or 9)
            event(experiment, "cache", "skip_complete", path=str(path.relative_to(TASK)))
            continue
        if path.exists():
            event(experiment, "cache", "interrupted_directory_deleted", path=str(path.relative_to(TASK)))
            shutil.rmtree(path)
        args = ["train_policy.py", "build-chunk-cache" if experiment == "G" else "build-pdref-cache",
                "--side-cache", path, "--workers", WORKERS,
                "--cache", source_cache(), "--trigger-cache", trigger_cache()]
        if SMOKE_ROOT:
            args += ["--allow-subset"]
        if experiment == "G":
            args += ["--chunk-k", 10, "--chunk-delta", delta]
        command(args, HERE / f"{experiment}_cache_d{delta}.log")


def pilot_g():
    rates = {}
    for delta in (9, 27, 81):
        specs = [(s, HERE / "calibration" / f"chunk_d{delta}_s{s}.npz") for s in (0, 1)]
        train_jobs("G", specs, delta)
        vals = []
        for seed, model in specs:
            folder = HERE / "pilot" / f"chunk_d{delta}_s{seed}_pilot"
            # Calibration folder is a deliberate alias: use the complete model
            # stem in the run folder so analyze_rollouts' mapping stays exact.
            folder = HERE / "pilot" / f"{model.stem}_pilot"
            if SMOKE_ROOT:
                # Real closed loop once, synthetic rates solely exercise G5.
                if delta == 9 and seed == 0:
                    evaluate("G", model, "pilot", folder, episodes=[0, 1])
                vals.append(.1 if SMOKE_CASE == "pilot_g_stop" else .9)
                continue
            evaluate("G", model, "pilot", folder)
            run = ar.load_run(str(folder))
            vals.append(float(np.mean([ar.metric_value(r, "success") for r in run.rows])))
        rates[delta] = vals
    baseline = []
    for seed in (0, 1):
        if SMOKE_ROOT:
            baseline.append(.5)
            continue
        folder = TASK / "results" / f"{OLD}_s{seed}_pilot"
        evaluate("G", model_path(OLD, seed), "pilot", folder)
        r = ar.load_run(str(folder))
        baseline.append(float(np.mean([ar.metric_value(row, "success") for row in r.rows])))
    j = {**pilot_choice(rates, baseline), "per_seed": rates, "c3_per_seed": baseline,
         "synthetic_gate_fixture": bool(SMOKE_ROOT)}
    write_json(HERE / "G_pilot.json", j)
    (HERE / "G_pilot.txt").write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
    command([CODE_DIR / "gp_figures.py", "--pilot-only", "--root", RUN_ROOT, "--output", HERE],
            HERE / "G_pilot_plot.log")


def selected_delta():
    j = json.loads((HERE / "G_pilot.json").read_text())
    if not j["continue"]:
        raise RuntimeError("All three chunk lengths are clearly worse on the pilot; G stops, DEV not opened")
    return int(j["delta_star"])


def promote_and_check_g():
    delta = selected_delta()
    for seed in (0, 1):
        src = HERE / "calibration" / f"chunk_d{delta}_s{seed}.npz"
        dst = model_path(MODELS['G'], seed)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(dst) != sha(src):
            raise RuntimeError("Official G model already exists but differs from the calibration model; refusing to overwrite")
        if not dst.exists():
            shutil.copyfile(src, dst)
    # Refit once using identical settings; never overwrite the pilot model.
    dst = HERE / "calibration" / f"retrain_d{delta}_s0.npz"
    train_jobs("G", [(0, dst)], delta)
    src = HERE / "calibration" / f"chunk_d{delta}_s0.npz"
    with np.load(src) as a, np.load(dst) as b:
        checks = {k: bool(k in b and np.array_equal(a[k], b[k])) for k in a.files if k != "meta"}
    j = {"passed": all(checks.values()), "arrays": checks,
         "rule": "all saved arrays except time/path-dependent meta; promoted model is byte-identical copy",
         "calibration_sha256": sha(src), "promoted_sha256": sha(model_path(MODELS['G'], 0))}
    write_json(HERE / "G_retrain_identity.json", j)
    if not j["passed"]:
        raise RuntimeError("G seed 0 calibration model differs from the retrain with the same parameters")


def pilot_p():
    out = model_path(MODELS['P'], 0)
    train_jobs("P", [(0, out)])
    folder = HERE / "pilot" / f"{MODELS['P']}_s0_pilot"
    evaluate("P", out, "pilot", folder, episodes=[0, 1] if SMOKE_ROOT else None)
    run = ar.load_run(str(folder))
    success = float(np.mean([ar.metric_value(row, "success") for row in run.rows]))
    crash = float(np.mean([row.get("stop_reason") == "tipped" for row in run.rows]))
    errors = sum(row.get("stop_reason") == "policy_error" for row in run.rows)
    j = {"n": len(run.rows), "success": success, "crash": crash, "policy_error": errors,
         "passed": bool(not errors and success >= .10 and crash <= .30),
         "rule": "bug check only, not an experiment decision"}
    if SMOKE_ROOT:
        # A 2-epoch model is deliberately incomplete. Retain its actual evidence
        # and exercise the unchanged production thresholds with a named fixture.
        j = {"real_closed_loop": j, "n": 100, "success": .05 if SMOKE_CASE == "pilot_p_stop" else .9,
             "crash": 0., "policy_error": errors, "synthetic_gate_fixture": True}
        j["passed"] = bool(not errors and j["success"] >= .10 and j["crash"] <= .30)
    write_json(HERE / "P_pilot.json", j)
    (HERE / "P_pilot.txt").write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
    if not j["passed"]:
        raise RuntimeError("P pilot software checks not passed; stop and debug, no scientific verdict is made")


def train_remaining(experiment):
    delta = selected_delta() if experiment == "G" else None
    if experiment == "P" and not json.loads((HERE / "P_pilot.json").read_text())["passed"]:
        raise RuntimeError("P pilot software checks not passed")
    seeds = ([100, 101] if experiment == "G" else [1, 100, 101]) if SMOKE_ROOT else (
        list(range(2 if experiment == "G" else 1, 10)) + list(range(100, 105)))
    train_jobs(experiment, [(s, model_path(MODELS[experiment], s)) for s in seeds], delta)


def model_checks(experiment):
    delta = selected_delta() if experiment == "G" else None
    cache = side_cache(experiment, delta)
    manifest = json.loads((cache / "manifest.json").read_text())
    with np.load(TASK / "trained" / f"{OLD}_s0.npz") as z:
        baseline = json.loads(str(z["meta"]))
        x_scalers = {k: z[k] for k in ("x_mean", "x_scale")}
    expected_rows = (baseline["n_train_rows"], baseline["n_val_rows"])
    expected_sigma = baseline["trigger"]["qd_sigma"]
    expected_epochs = baseline["hyperparameters"]["max_epochs"]
    if SMOKE_ROOT:
        from gp_training import prepare_training_data
        d = prepare_training_data(MODELS[experiment], str(source_cache()), tp.SPLIT_PATH,
                                  str(trigger_cache()), side_dir=cache, chunk_delta=delta or 9,
                                  max_train_episodes=2, max_val_episodes=1, allow_subset=True)
        x_scalers = {"x_mean": d["xm"], "x_scale": d["xs"]}
        expected_rows = (len(d["X"]), len(d["Xv"]))
        expected_sigma = d["sigma"].tolist()
        expected_epochs = SMOKE_EPOCHS
        del d
    records = []
    for seed in ([0, 1, 100, 101] if SMOKE_ROOT else list(range(10)) + list(range(100, 105))):
        path = model_path(MODELS[experiment], seed)
        with np.load(path) as z:
            m = json.loads(str(z["meta"]))
            t = m["trigger"]
            checks = {
                "model_seed": m["model"] == MODELS[experiment] and m["seed"] == seed,
                "features": m["feature_set"] == "ego_novw",
                "trigger": t["copies"] == 16 and t["qd_jitter"] == "auto" and t["jitter_seed"] == 0
                           and t["qd_sigma"] == expected_sigma,
                "rows": (m["n_train_rows"], m["n_val_rows"]) == expected_rows,
                "split": m["split_sha256"] == sha(tp.SPLIT_PATH),
                "cache": m["cache_manifest_sha256"] == sha(source_cache() / "manifest.json")
                         and t["trigger_cache_manifest_sha256"] == sha(trigger_cache() / "manifest.json")
                         and m["side_cache_manifest_sha256"] == sha(cache / "manifest.json"),
                "code": all(m["code_sha256"].get(f) == sha(TASK / f) for f in
                            ("train_policy.py", "bc_features.py", "policies/bc_policy.py", "policies/pd_layer.py",
                             "results/step6/gp_training.py")),
                "x_scalers": all(np.array_equal(z[k], v) for k, v in x_scalers.items()),
                "hyperparameters": all(m["hyperparameters"][k] == baseline["hyperparameters"][k] for k in
                    ("hidden_layer_sizes", "activation", "solver", "learning_rate_init", "alpha", "batch_size", "shuffle", "patience", "rel_tol"))
                    and m["hyperparameters"]["max_epochs"] == expected_epochs,
                "representation": m["output_representation"] == ("chunk" if experiment == "G" else "pd_reference"),
                "memory": float(m["peak_rss_mib"]) <= LIMIT_MIB,
                "scope": bool(m.get("subset")) == bool(SMOKE_ROOT),
            }
            if experiment == "G":
                checks["chunk"] = m["chunk_k"] == 10 and m["chunk_delta"] == delta and m["ensemble_m"] == .11
            records.append({"seed": seed, "checks": checks, "passed": all(checks.values()),
                            "sha256": sha(path), "val_loss": m["val_loss"], "peak_rss_mib": m["peak_rss_mib"]})
    j = {"passed": all(r["passed"] for r in records), "models": records,
         "side_manifest_sha256": sha(cache / "manifest.json"), "side_manifest": manifest}
    write_json(HERE / f"{experiment}_model_checks.json", j)
    if not j["passed"]:
        raise RuntimeError(f"{experiment} model meta checks not passed")


def stalled_controls(experiment, set_name):
    require_gate(experiment, set_name)
    for seed in SEEDS[set_name]:
        folder = TASK / "results" / f"{OLD}_s{seed}_{set_name}"
        r = ar.load_run(str(folder))
        eps = [row["episode"] for row in r.rows if row["stop_reason"] == "stalled"]
        if r.meta.get("no_early_stop") is True or not eps:
            continue
        found = False
        for d in ("step3", "step5", "step6"):
            existing = TASK / "results" / d / "nostall" / folder.name
            if (existing / "summary.txt").exists():
                validate_run(existing, eps, f"trained/{OLD}_s{seed}.npz")
                found = True
                break
        if not found:
            evaluate(experiment, TASK / "trained" / f"{OLD}_s{seed}.npz", set_name,
                     HERE / "nostall" / folder.name, eps)


def ensure_c3_trajectories():
    """New, matching no-stall trajectories where the frozen C3 controls lack them."""
    from gp_common import merged_old, raw_runs
    raw = raw_runs(OLD, "dev", SEEDS["dev"])
    merged, _ = merged_old(raw)
    evidence = []
    for original, run in zip(raw, merged):
        missing = []
        for row in run.rows:
            if not np.isfinite(ar._f(row, "arm_extend_time")):
                continue
            e = row["episode"]
            base = Path(original.path)
            if original.by_episode[e].get("stop_reason") == "stalled" and original.meta.get("no_early_stop") is not True:
                base = next(TASK / "results" / step / "nostall" / original.name for step in
                            ("step3", "step5", "step6") if
                            (TASK / "results" / step / "nostall" / original.name / "summary.txt").exists())
            if not (base / "traj" / f"ep_{e:04d}.npz").exists():
                missing.append(e)
        if not missing:
            continue
        folder = HERE / "smoothness" / run.name
        model = model_path(OLD, run.meta["seed"])
        evaluate("G", model, "dev", folder, episodes=missing, save_every=10)
        rerun = ar.load_run(str(folder))
        for e in missing:
            a, b = run.by_episode[e], rerun.by_episode[e]
            for key in ("success", "stop_reason"):
                if a[key] != b[key]:
                    raise RuntimeError(f"C3 trajectory rerun changed {run.name} ep{e} {key}")
            for key in ("min_error", "arm_extend_time"):
                if not np.isclose(ar._f(a, key), ar._f(b, key), rtol=0, atol=1e-9, equal_nan=True):
                    raise RuntimeError(f"C3 trajectory rerun changed {run.name} ep{e} {key}")
        evidence.append({"run": run.name, "episodes": missing, "folder": str(folder.relative_to(TASK)),
                         "matched_frozen_rows": True})
    write_json(HERE / "C3_smoothness_trajectory_checks.json", {"passed": True, "new_runs": evidence})


def evaluations(experiment, set_name, allow_confirm=False):
    require_gate(experiment, set_name)
    model_checks(experiment)
    if set_name == "confirm":
        if not allow_confirm:
            raise RuntimeError("CONFIRM requires --allow-confirm, and the DEV/HARD gates must have passed")
        path = HERE / f"{experiment}_confirm_opened.json"
        if not path.exists():
            write_json(path, {"time": now(), "gate_evidence_sha256": sha(HERE / f"{experiment}_mechanisms.json"),
                             "rule": "one gated opening per experiment; interrupted runs resume, completed runs skip"})
        ar.ALLOW_CONFIRM = True
    if SMOKE_ROOT:
        from gp_smoke import create_fixture
        create_fixture(SMOKE_ROOT, experiment, case=SMOKE_CASE, set_name=set_name)
    else:
        stalled_controls(experiment, set_name)
        if experiment == "G" and set_name == "dev":
            ensure_c3_trajectories()
    for seed in SEEDS[set_name]:
        model = model_path(MODELS[experiment], seed)
        folder = RUN_ROOT / "results" / f"{MODELS[experiment]}_s{seed}_{set_name}"
        if SMOKE_ROOT:
            _, errors = validate_run(folder, list(range(100)), str(model.relative_to(TASK)), set_name == "dev")
            event(experiment, "run", "synthetic_fixture", folder=str(folder.relative_to(TASK)), n=100, **errors)
        else:
            evaluate(experiment, model, set_name, folder, save_every=10 if set_name == "dev" else None)
    args = [CODE_DIR / f"{experiment}_mechanisms.py", "--through", set_name, *smoke_args()]
    if set_name == "confirm":
        args += ["--allow-confirm"]
    command(args, HERE / f"{experiment}_mechanisms_{set_name}.log")


def offline(experiment):
    for seed in SEEDS["dev"]:
        if SMOKE_ROOT:
            baseline = model_path(OLD, seed)
            if not baseline.with_suffix(".offline.json").exists():
                command(["train_policy.py", "offline", "--model", baseline,
                         "--cache", source_cache(), "--allow-subset"], HERE / f"C3_offline_s{seed}.log")
        model = model_path(MODELS[experiment], seed)
        output = model.with_suffix(".offline.json")
        if not output.exists():
            command(["train_policy.py", "offline", "--model", model, "--cache", source_cache(),
                     *(["--allow-subset"] if SMOKE_ROOT else [])],
                    HERE / f"{experiment}_offline_s{seed}.log")


def stage(experiment, name, fn):
    check_freeze()
    marker = HERE / "completed" / f"{experiment}_{name}.json"
    paths = [*CODE_DIR.glob("*.py"), *CODE_DIR.glob("run_exp*.sh"),
             *(TASK / f for f in ("train_policy.py", "bc_features.py", "policies/bc_policy.py",
                                  "policies/pd_layer.py"))]
    code = {str(p.relative_to(TASK)): sha(p) for p in sorted(paths)}
    settings = {"software_smoke": bool(SMOKE_ROOT), "smoke_case": SMOKE_CASE if SMOKE_ROOT else None,
                "smoke_epochs": SMOKE_EPOCHS if SMOKE_ROOT else None}
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("settings") != settings or (previous.get("implementation_sha256") != code
                                                    and not post_hoc()):
            raise RuntimeError("Implementation hash of a completed stage has changed; refusing to resume silently")
        event(experiment, name, "skip_complete")
        return
    status(experiment, name, "start")
    started = now()
    fn()
    write_json(marker, {"started": started, "finished": now(), "implementation_sha256": code,
                        "settings": settings})
    status(experiment, name, "end")


def main(argv=None):
    global HERE, RUN_ROOT, SMOKE_ROOT, SMOKE_CASE, SMOKE_EPOCHS, SEEDS
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=("G", "P"), required=True)
    p.add_argument("--stage", choices=("all", "cache", "pilot", "promote", "train", "models", "offline", "dev", "hard", "confirm", "figures"), default="all")
    p.add_argument("--allow-confirm", action="store_true")
    p.add_argument("--smoke-root", help="isolated short chain below results/step6/smoke; no DEV/HARD/CONFIRM data")
    p.add_argument("--smoke-case", choices=("pass", "dev_fail", "dev_risk_fail", "hard_fail", "confirm_fail",
                                          "pilot_g_stop", "pilot_p_stop"), default="pass")
    p.add_argument("--smoke-epochs", type=int, choices=(1, 2), default=2)
    a = p.parse_args(argv)
    if WORKERS < 1 or JOBS < 1:
        p.error("WORKERS/JOBS must be positive")
    if a.smoke_root:
        from gp_common import SOURCE_HERE
        root = Path(a.smoke_root).resolve()
        if not root.is_relative_to(SOURCE_HERE / "smoke") or root == SOURCE_HERE / "smoke":
            p.error("--smoke-root must be below results/step6/smoke")
        SMOKE_ROOT = RUN_ROOT = root
        HERE = root / "results" / "step6"
        HERE.mkdir(parents=True, exist_ok=True)
        SMOKE_CASE, SMOKE_EPOCHS = a.smoke_case, a.smoke_epochs
        SEEDS = {"dev": [0, 1], "hard": [0, 1], "confirm": [100, 101]}
        check_freeze()
        for seed in (0, 1, 100, 101):
            dst = model_path(OLD, seed)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = TASK / "trained" / dst.name
            if not dst.exists():
                shutil.copyfile(src, dst)
            elif sha(src) != sha(dst):
                raise RuntimeError("smoke C3 baseline copy differs")
    elif a.smoke_case != "pass" or a.smoke_epochs != 2:
        p.error("smoke options require --smoke-root")
    e, current = a.experiment, a.stage
    os.chdir(TASK)
    funcs = {"cache": lambda: build_caches(e), "pilot": pilot_g if e == "G" else pilot_p,
             "promote": promote_and_check_g if e == "G" else lambda: None,
             "train": lambda: train_remaining(e), "models": lambda: model_checks(e),
             "offline": lambda: offline(e), "dev": lambda: evaluations(e, "dev"),
             "hard": lambda: evaluations(e, "hard"),
             "confirm": lambda: evaluations(e, "confirm", a.allow_confirm),
             "figures": lambda: command([CODE_DIR / "gp_figures.py", "--root", RUN_ROOT, "--output", HERE,
                      "--experiment", e, "--require-complete",
                      *(["--seeds", "0,1", "--n-boot", "200"] if SMOKE_ROOT else [])],
                      HERE / f"{e}_figures.log")}
    try:
        if a.stage != "all":
            stage(e, a.stage, funcs[a.stage])
            return
        for name in ("cache", "pilot", "promote", "train", "models", "offline", "dev"):
            current = name
            stage(e, name, funcs[name])
            if e == "G" and name == "pilot" and not json.loads((HERE / "G_pilot.json").read_text())["continue"]:
                status(e, name, "stopped", reason="All three chunk lengths clearly worse on the pilot", verdict="KEEP C3")
                return
        if dev_pass(load_evidence(e)):
            current = "hard"
            stage(e, current, funcs[current])
            if hard_pass(load_evidence(e)):
                current = "confirm"
                # all is the pre-authorised chain; passing gates supplies the
                # single explicit --allow-confirm to the isolated subprocess.
                stage(e, current, lambda: command([CODE_DIR / "step6_pipeline.py", "--experiment", e,
                      "--stage", "confirm", "--allow-confirm", *smoke_args(),
                      *(["--smoke-case", SMOKE_CASE, "--smoke-epochs", SMOKE_EPOCHS] if SMOKE_ROOT else [])],
                      HERE / f"{e}_confirm_stage.log"))
        current = "figures"
        stage(e, current, funcs[current])
        j = load_evidence(e)
        stopped_after = ("dev" if not dev_pass(j) else "hard" if not hard_pass(j)
                         else "confirm" if not (j.get("confirm") or {}).get("condition_4") else None)
        reason = ({"dev": "DEV condition (1) or (2) not met; HARD/CONFIRM not opened",
                   "hard": "HARD condition (3) not met; CONFIRM not opened",
                   "confirm": "CONFIRM condition (4) not met"}.get(stopped_after))
        status(e, "chain", "complete", verdict=j.get("verdict"),
               stopped_by_gate=stopped_after is not None, stopped_after=stopped_after, reason=reason,
               mechanisms_sha256=sha(HERE / f"{e}_mechanisms.json"), software_smoke=bool(SMOKE_ROOT))
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        status(e, current, "blocked", error=repr(exc), traceback=traceback.format_exc(),
               resume=f"bash results/step6/run_exp{e}.sh all >> results/step6/chain{e}.log 2>&1")
        raise


if __name__ == "__main__":
    main()
