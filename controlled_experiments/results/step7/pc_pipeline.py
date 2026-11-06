"""Step 7 PC: resume controls -> HARD -> CONFIRM -> analysis, without step-6 gates."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "results" / "step6"), str(TASK)]
from gp_common import ar, sha, write_json  # noqa: E402
from step6_pipeline import validate_run  # noqa: E402
from pc_freeze import check_freeze  # noqa: E402

OLD, MODEL = "ego-novw-trig", "ego-novw-trig-pd"
SEEDS = {"hard": list(range(5)), "confirm": list(range(100, 105))}
COUNTS = {"hard": 200, "confirm": 300}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def event(state_dir, stage, status, **fields):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    record = {"time": now(), "experiment": "PC", "stage": stage, "status": status, **fields}
    with (state_dir / "stagePC_events.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def env():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/mplcfg", "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def command(args, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a") as f:
        subprocess.run([sys.executable, *map(str, args)], cwd=TASK, env=env(),
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def _key(folder):
    return str(Path(folder).resolve().relative_to(TASK))


def _allowed_output(folder, model, set_name, state_dir, smoke):
    folder = Path(folder)
    if folder.is_symlink():
        raise RuntimeError("Refusing to use a symbolic link as the run directory")
    folder = folder.resolve()
    if smoke:
        boundary = HERE / "smoke"
        if not folder.is_relative_to(boundary) or folder == boundary:
            raise ValueError("Real smoke runs may only write to a subdirectory of results/step7/smoke")
    else:
        formal = TASK / "results" / f"{Path(model).stem}_{set_name}"
        control = HERE / "nostall" / f"{Path(model).stem}_{set_name}"
        if folder not in (formal, control):
            raise ValueError("Output is not one of the run paths explicitly designated for step 7")
        if folder == formal and not Path(model).stem.startswith(MODEL + "_s"):
            raise ValueError("Formal run directories are only allowed for the new P model")
    if Path(state_dir).resolve() != HERE and not Path(state_dir).resolve().is_relative_to(HERE / "smoke"):
        raise ValueError("Run state may only be written to step7")
    return folder


def confirm_token(state_dir, allow_confirm, create=False):
    """One frozen opening; subsequent invocations only resume that opening."""
    if not allow_confirm:
        raise RuntimeError("CONFIRM requires an explicit --allow-confirm")
    freeze = check_freeze()
    path = Path(state_dir) / "PC_confirm_opened.json"
    if not path.exists():
        if not create:
            raise RuntimeError("P's CONFIRM has not yet been opened once by the confirm stage")
        payload = {"time": now(), "plan_sha256": freeze["plan_sha256"],
                   "frozen_sha256": freeze["frozen_sha256"],
                   "seeds": SEEDS["confirm"], "episodes_per_seed": COUNTS["confirm"],
                   "rule": "one PC opening; completed runs skip, owned interrupted runs resume"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        event(state_dir, "confirm", "opened_once", token=_key(path))
    payload = json.loads(path.read_text())
    if any(payload[k] != freeze[k] for k in ("plan_sha256", "frozen_sha256")):
        raise RuntimeError("freeze mismatch: CONFIRM opening record differs from the current freeze")
    ar.ALLOW_CONFIRM = True
    return payload


def evaluate(model, set_name, folder, episodes, workers=22, state_dir=HERE,
             allow_confirm=False, smoke=False):
    check_freeze()
    if set_name not in COUNTS:
        raise ValueError("PC evaluates only hard or confirm")
    model = Path(model).resolve()
    if model.parent != TASK / "trained" or not model.is_file():
        raise ValueError("Only existing frozen models in trained/ may be used")
    if set_name == "confirm":
        if model.stem.startswith(MODEL + "_s"):
            confirm_token(state_dir, allow_confirm)
        elif not allow_confirm:
            raise RuntimeError("CONFIRM controls require --allow-confirm")
        ar.ALLOW_CONFIRM = True
    episodes = sorted(episodes)
    if not episodes or len(set(episodes)) != len(episodes) or any(e < 0 or e >= COUNTS[set_name] for e in episodes):
        raise ValueError("Episode range/duplicate error")
    folder = _allowed_output(folder, model, set_name, state_dir, smoke)
    model_arg = str(model.relative_to(TASK))
    if (folder / "summary.txt").exists():
        _, errors = validate_run(folder, episodes, model_arg)
        event(state_dir, "run", "skip_complete", folder=_key(folder), n=len(episodes), **errors)
        return
    registry_path = Path(state_dir) / "owned_runs.json"
    owned = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    key = _key(folder)
    spec = {"policy_arg": model_arg, "set": set_name, "episodes": episodes,
            "model_sha256": sha(model)}
    if key in owned and owned[key] != spec:
        raise RuntimeError(f"Existing run registration differs from this request; refusing to overwrite: {key}")
    if folder.exists():
        if key not in owned:
            raise RuntimeError(f"Existing directory does not belong to PC; refusing to delete or rewrite: {key}")
        event(state_dir, "run", "interrupted_directory_deleted", folder=key)
        shutil.rmtree(folder)
    owned[key] = spec
    write_json(registry_path, owned)
    args = [TASK / "evaluate_policy.py", "--policy", "policies/bc_policy.py:make_policy",
            "--policy-arg", model_arg, "--eval-set", f"data/{set_name}",
            "--out-dir", key, "--workers", workers, "--no-early-stop",
            "--episodes", ",".join(map(str, episodes))]
    event(state_dir, "run", "start", folder=key, n=len(episodes), synthetic=False)
    # Logs remain within step7 even when Claude later writes the formal runs.
    log = Path(state_dir) / "logs" / (folder.name + ".log")
    command(args, log)
    _, errors = validate_run(folder, episodes, model_arg)
    event(state_dir, "run", "end", folder=key, n=len(episodes), **errors)


def stalled_controls(set_name, workers=22, allow_confirm=False):
    """Exactly the step-6 replacement logic, with no step-6 require_gate call."""
    if set_name == "confirm":
        if not allow_confirm:
            raise RuntimeError("CONFIRM controls require --allow-confirm")
        ar.ALLOW_CONFIRM = True
    notes = []
    for seed in SEEDS[set_name]:
        folder = TASK / "results" / f"{OLD}_s{seed}_{set_name}"
        run = ar.load_run(str(folder))
        if (run.meta.get("warnings") or sorted(run.by_episode) != list(range(COUNTS[set_name]))
                or len(run.rows) != COUNTS[set_name]
                or run.meta.get("policy_arg") != f"trained/{OLD}_s{seed}.npz"):
            raise RuntimeError(f"C3 original run coverage/model mismatch: {folder}")
        episodes = sorted(row["episode"] for row in run.rows if row.get("stop_reason") == "stalled")
        if run.meta.get("no_early_stop") is True or not episodes:
            notes.append({"run": run.name, "stalled": 0, "source": "original"})
            continue
        found = None
        for step in ("step3", "step5", "step6", "step7"):
            candidate = TASK / "results" / step / "nostall" / folder.name
            if (candidate / "summary.txt").exists():
                validate_run(candidate, episodes, f"trained/{OLD}_s{seed}.npz")
                found = candidate
                break
        if found is None:
            found = HERE / "nostall" / folder.name
            evaluate(TASK / "trained" / f"{OLD}_s{seed}.npz", set_name, found, episodes,
                     workers=workers, allow_confirm=allow_confirm)
        notes.append({"run": run.name, "stalled": len(episodes), "source": _key(found)})
    write_json(HERE / f"PC_controls_{set_name}.json", notes)
    return notes


def evaluations(set_name, allow_confirm=False, smoke_hard=False, workers=22, smoke_episodes=(0, 1)):
    if set_name == "confirm":
        confirm_token(HERE, allow_confirm, create=True)
    if smoke_hard and set_name != "hard":
        raise ValueError("Real smoke runs are only allowed on HARD")
    for seed in ([0] if smoke_hard else SEEDS[set_name]):
        model = TASK / "trained" / f"{MODEL}_s{seed}.npz"
        state_dir = HERE / "smoke" / "real" if smoke_hard else HERE
        folder = (state_dir if smoke_hard else TASK / "results") / f"{MODEL}_s{seed}_{set_name}"
        evaluate(model, set_name, folder, list(smoke_episodes) if smoke_hard else list(range(COUNTS[set_name])),
                 workers=workers, state_dir=state_dir, allow_confirm=allow_confirm, smoke=smoke_hard)


def run_stage(stage, allow_confirm=False, smoke_hard=False, workers=22, smoke_episodes=(0, 1)):
    freeze = check_freeze()
    state_dir = HERE / "smoke" / "real" if smoke_hard else HERE
    event(state_dir, stage, "start", frozen_sha256=freeze["frozen_sha256"])
    try:
        if stage == "controls":
            for set_name in COUNTS:
                stalled_controls(set_name, workers=workers, allow_confirm=allow_confirm)
        elif stage in COUNTS:
            evaluations(stage, allow_confirm, smoke_hard, workers, smoke_episodes)
        elif stage == "analyze":
            confirm_token(HERE, allow_confirm)
            from PC_mechanisms import analyze
            analyze(allow_confirm=allow_confirm)
        else:
            raise ValueError(stage)
    except BaseException as exc:
        event(state_dir, stage, "error", error_type=type(exc).__name__, error=str(exc))
        raise
    write_json(state_dir / "completed" / f"{stage}.json",
               event(state_dir, stage, "complete", frozen_sha256=freeze["frozen_sha256"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("controls", "hard", "confirm", "analyze", "all"), default="all")
    parser.add_argument("--allow-confirm", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "22")))
    parser.add_argument("--smoke-hard", action="store_true", help="only seed 0, 1-2 HARD episodes, output below smoke")
    parser.add_argument("--episodes", default="0,1", help="used only by --smoke-hard")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    episodes = tuple(int(x) for x in args.episodes.split(","))
    if args.smoke_hard and (args.stage != "hard" or not 1 <= len(episodes) <= 2):
        parser.error("--smoke-hard requires --stage hard and 1-2 episodes")
    if args.stage in ("all", "controls", "confirm", "analyze") and not args.allow_confirm:
        parser.error("this stage requires --allow-confirm after the PC freeze")
    for stage in (("controls", "hard", "confirm", "analyze") if args.stage == "all" else (args.stage,)):
        run_stage(stage, args.allow_confirm, args.smoke_hard, args.workers, episodes)


if __name__ == "__main__":
    main()
