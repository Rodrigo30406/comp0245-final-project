"""Bounded episode subprocesses; only the coordinator writes dataset summaries."""

from collections import deque
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

from robot_learning.settings import PROJECT_ROOT
from .storage import provenance, read_json, sha256, summarize, utc_now, write_json


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def validate_workers(workers, config=None):
    if type(workers) is not int or workers < 1:
        raise ValueError("--workers must be a positive integer")
    if config is not None and not config["headless"] and workers > 1:
        raise ValueError("Parallel collection requires headless mode; use --workers 1 with GUI")


def pending_scenarios(directory, scenarios):
    pending = deque()
    for index, scenario in enumerate(scenarios, 1):
        path = directory / "episodes" / f"{scenario['episode_id']}.json"
        if path.exists() and (previous := read_json(path))["status"] == "complete":
            data = directory / "episodes" / (previous["data_file"] or "missing.npz")
            if not data.is_file() or sha256(data) != previous.get("data_sha256"):
                raise ValueError(f"Completed episode data missing or changed: {scenario['episode_id']}")
            print(f"[{index}/{len(scenarios)}] {scenario['episode_id']}: already complete", flush=True)
        else:
            pending.append((index, scenario))
    return pending


def archive_attempt(directory, episode_id):
    paths = (directory / "episodes" / f"{episode_id}.json",
             directory / "episodes" / f"{episode_id}.npz",
             directory / "logs" / f"{episode_id}.log")
    existing = [path for path in paths if path.exists()]
    if existing:
        archive = directory / "attempts" / episode_id / uuid.uuid4().hex[:12]
        archive.mkdir(parents=True)
        for path in existing:
            path.rename(archive / path.name)


def failure_metadata(directory, scenario, termination, returncode, *, force=False):
    path = directory / "episodes" / f"{scenario['episode_id']}.json"
    result = read_json(path) if path.exists() else {
        **scenario, "recorded_transitions": 0, "data_file": None,
    }
    if force or result.get("status") in (None, "running"):
        result.update(status="interrupted" if termination == "interrupted" else "error",
                      success=False, termination=termination, returncode=returncode,
                      finished_at=utc_now())
        write_json(path, result)
    return result


def stop_processes(active):
    """Interrupt all children, allow partial data to save, then reap every child."""
    for process, _, _, _ in active.values():
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 5
    for process, log, _, record in active.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            log.close()
        record.update(finished_at=utc_now(), returncode=process.returncode)


def collect(directory, cfg, scenarios, manifest, robo, *, workers=1):
    validate_workers(workers, cfg)
    lock = directory / ".collection.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"Collection lock exists: {lock}. Check that no collector is active "
                         "before removing a stale lock and resuming.") from error
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"pid={os.getpid()}\nstarted_at={utc_now()}\n")
    active = {}
    started = time.monotonic()
    execution = {"started_at": utc_now(), "status": "running", "workers": workers,
                 "peak_active_workers": 0, "pid": os.getpid(), "hostname": socket.gethostname(),
                 "cpu_count": os.cpu_count(), "argv": sys.argv, "episodes": []}
    executions = directory / "executions"
    execution_path = executions / f"{uuid.uuid4().hex}.json"
    source_changed = False
    try:
        executions.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["ROBOENV_ROOT"] = str(robo)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH")]))
        # Avoid multiplying BLAS threads by worker count; respect explicit user settings.
        for variable in THREAD_VARIABLES:
            env.setdefault(variable, "1")
        execution["thread_environment"] = {key: env[key] for key in THREAD_VARIABLES}
        write_json(execution_path, execution)
        frozen = read_json(directory / "provenance.json")["source_fingerprint"]

        def check_sources():
            nonlocal source_changed
            if provenance(PROJECT_ROOT, robo)["source_fingerprint"] != frozen:
                source_changed = True
                raise ValueError("Sources or environment changed during collection; active episodes marked invalid")

        manifest.update(status="running", updated_at=utc_now())
        write_json(directory / "dataset.json", manifest)
        pending = pending_scenarios(directory, scenarios)
        print(f"Workers: {workers}; pending episodes: {len(pending)}", flush=True)
        while pending or active:
            while pending and len(active) < workers:
                check_sources()
                index, scenario = pending.popleft()
                episode_id = scenario["episode_id"]
                archive_attempt(directory, episode_id)
                log = (directory / "logs" / f"{episode_id}.log").open("w")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "robot_learning.data.worker", str(directory), episode_id],
                        cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record = {"episode_id": episode_id, "pid": process.pid, "started_at": utc_now()}
                active[episode_id] = (process, log, scenario, record)
                execution["episodes"].append(record)
                execution["peak_active_workers"] = max(execution["peak_active_workers"], len(active))
                write_json(execution_path, execution)
                print(f"[{index}/{len(scenarios)}] {episode_id}: target={scenario['target']} pid={process.pid}", flush=True)
            finished = [key for key, (process, _, _, _) in active.items() if process.poll() is not None]
            if not finished:
                time.sleep(.05)
                continue
            for episode_id in finished:
                process, log, scenario, record = active[episode_id]
                process.wait()
                log.close()
                record.update(finished_at=utc_now(), returncode=process.returncode)
                result = failure_metadata(directory, scenario, "worker_error", process.returncode)
                if process.returncode and result["status"] == "complete":
                    result = failure_metadata(directory, scenario, "worker_error", process.returncode, force=True)
                check_sources()
                del active[episode_id]
                write_json(execution_path, execution)
                write_json(directory / "summary.json", summarize(directory, scenarios, cfg))
                print(f"  {episode_id}: {result['status']}: {result.get('termination')}; "
                      f"error={result.get('final_distance_m', 'n/a')} m; "
                      f"samples={result['recorded_transitions']}", flush=True)
        summary = summarize(directory, scenarios, cfg)
        manifest["status"] = "complete" if summary["failed_episodes"] == 0 else "completed_with_errors"
        execution["status"] = manifest["status"]
        write_json(directory / "summary.json", summary)
        return 0 if manifest["status"] == "complete" else 1
    except BaseException:
        manifest["status"] = execution["status"] = "interrupted"
        stop_processes(active)
        for process, _, scenario, _ in active.values():
            failure_metadata(directory, scenario, "source_changed" if source_changed else "interrupted",
                             process.returncode, force=source_changed)
        write_json(directory / "summary.json", summarize(directory, scenarios, cfg))
        raise
    finally:
        execution.update(finished_at=utc_now(), wall_duration_s=time.monotonic() - started)
        manifest["updated_at"] = utc_now()
        try:
            write_json(execution_path, execution)
            write_json(directory / "dataset.json", manifest)
        finally:
            lock.unlink(missing_ok=True)
