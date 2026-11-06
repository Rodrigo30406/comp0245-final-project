"""Atomic dataset metadata, source snapshots, and reproducibility fingerprints."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
from .schema import FIELDS


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "remote": git("remote", "get-url", "origin")}


def source_files(project, robo):
    files = {f"project/{p.relative_to(project).as_posix()}": p
             for p in (project / "src" / "robot_learning").rglob("*.py")}
    for p in (project / "scripts").glob("*.py"):
        files[f"project/{p.relative_to(project).as_posix()}"] = p
    files["project/pyproject.toml"] = project / "pyproject.toml"
    for p in (project / "experiment_specs" / "expert").glob("*.json"):
        files[f"project/{p.relative_to(project).as_posix()}"] = p
    for relative in ("pixi.toml", "pixi.lock", "configs/robotnik_arm.json",
                     "simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py",
                     "simulation_and_control/simulation_and_control/controllers/pin_wrapper.py",
                     "simulation_and_control/simulation_and_control/controllers/servo_motor.py"):
        files[f"roboenv/{relative}"] = robo / relative
    package = robo / "simulation_and_control" / "simulation_and_control"
    for path in package.rglob("*.py"):
        files[f"roboenv/{path.relative_to(robo).as_posix()}"] = path
    return files


def provenance(project, robo):
    files = source_files(project, robo)
    hashes = {name: sha256(path) for name, path in files.items()}
    robot = read_json(robo / "configs/robotnik_arm.json")
    for group in ("robot_pybullet", "robot_pin"):
        for relative in robot[group]["urdf_path"]:
            model_directory = (robo / "models" / relative).parent
            for path in model_directory.rglob("*"):
                if path.is_file():
                    hashes[f"roboenv/{path.relative_to(robo).as_posix()}"] = sha256(path)
    versions = {}
    for package in ("numpy", "pybullet", "pin", "pinocchio", "scipy", "simulation_and_control"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    fingerprint_data = {"files": hashes, "packages": versions, "python": sys.version,
                        "platform": platform.platform()}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    return {"created_at": utc_now(), "source_fingerprint": fingerprint,
            "project": git_info(project), "roboenv": git_info(robo),
            "simulation_and_control": git_info(robo / "simulation_and_control"),
            "roboenv_root": str(robo), "python_executable": sys.executable,
            "cpu_count": os.cpu_count(), **fingerprint_data}


def snapshot_sources(directory, project, robo):
    for name, source in source_files(project, robo).items():
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def summarize(directory, scenarios, config):
    episodes = []
    for scenario in scenarios:
        path = directory / "episodes" / f"{scenario['episode_id']}.json"
        if path.exists():
            episodes.append(read_json(path))
    complete = [e for e in episodes if e["status"] == "complete"]
    reasons = {}
    for episode in episodes:
        reason = episode.get("termination", episode["status"])
        reasons[reason] = reasons.get(reason, 0) + 1
    successes = sum(e["success"] for e in complete)
    return {"updated_at": utc_now(), "requested_episodes": len(scenarios),
            "complete_episodes": len(complete), "failed_episodes": len(episodes) - len(complete),
            "successes": successes, "success_rate_complete": successes / len(complete) if complete else None,
            "success_rate_requested": successes / len(scenarios), "termination_counts": reasons,
            "recorded_transitions": sum(e.get("recorded_transitions", 0) for e in episodes),
            "split": config["split"], "episodes": episodes}


class Recorder:
    def __init__(self, capacity):
        self.arrays = {name: np.empty((capacity,) if width is None else (capacity, width), dtype=dtype)
                       for name, (width, dtype) in FIELDS.items()}
        self.count = 0

    def append(self, row):
        if set(row) != set(self.arrays):
            raise ValueError("Recorded fields do not match the dataset schema")
        if self.count == len(self.arrays["time"]):
            for name, values in self.arrays.items():
                grown = np.empty((max(1, 2 * self.count), *values.shape[1:]), dtype=values.dtype)
                grown[:self.count] = values
                self.arrays[name] = grown
        for name, value in row.items():
            if self.arrays[name].dtype.kind in "f" and not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite values in {name}")
            self.arrays[name][self.count] = value
        self.count += 1

    def save(self, path, terminal):
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{k: v[:self.count] for k, v in self.arrays.items()},
                                terminal_observation=np.asarray(terminal["observation"], dtype=np.float32),
                                terminal_ee_position=np.asarray(terminal["ee_position"], dtype=np.float32),
                                terminal_time=np.asarray(terminal["time"]))
        temporary.replace(path)
