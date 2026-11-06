"""Loading the recorded episodes of a split as model inputs and labels."""
from pathlib import Path

import numpy as np

from robot_learning.data.splits import record_directory
from robot_learning.data.storage import read_json, sha256
from .features import INPUT_NAMES, LIDAR_VARIANTS, encode
from .outputs import independent_actions


def load_episode(directory, record):
    """(observations (N, 21), nine controls (N, 9), expert phases (N,)) of one episode, checked against its record."""
    path = Path(directory) / "episodes" / f"{record['episode_id']}.npz"
    if sha256(path) != record["npz_sha256"]:
        raise ValueError("Episode changed after inventory")
    with np.load(path, allow_pickle=False) as data:
        x = data["observation"].astype(np.float64)
        y = independent_actions(data["action"]).astype(np.float64)
        phases = data["expert_phase"].copy()
    if (x.shape != (record["rows"], 21) or y.shape != (len(x), 9) or phases.shape != (len(x),) or len(x) == 0
            or not np.isfinite(x).all()):
        raise ValueError("Invalid shapes, empty episode or non-finite inputs")
    target = np.asarray(record["target"], dtype=np.float32).astype(np.float64)
    if not np.all(x[:, -3:] == target):
        raise ValueError("Observations must contain the original scenario target")
    return x, y, phases


def validate_episode_sampling(data, metadata, dt_s):
    """Fail closed on sparse logging: the clock input assumes one row per physics step from step zero."""
    if "step" not in data or "time" not in data or "dt" not in metadata:
        raise ValueError("Training requires recorded steps, times and dt")
    step, time = data["step"], data["time"]
    if (not np.isclose(metadata["dt"], dt_s, rtol=0, atol=1e-12) or not np.array_equal(step, np.arange(len(step)))
            or not np.allclose(time, step * dt_s, rtol=0, atol=1e-8)):
        raise ValueError("Training requires dense observations from step zero at the configured dt")


def load_episode_features(directory, record, spec):
    """(features (N, F) of spec["variant"], nine controls (N, 9)) of one episode."""
    directory = record_directory(record, directory)
    x, y, _ = load_episode(directory, record)
    path = Path(directory) / "episodes" / f"{record['episode_id']}.npz"
    with np.load(path, allow_pickle=False) as data:
        validate_episode_sampling(data, read_json(path.with_suffix(".json")), spec["dt_s"])
        times = data["time"].astype(np.float64)
        lidar = None
        if spec["variant"] in LIDAR_VARIANTS:
            if "lidar" not in data.files:
                raise ValueError(f"{path}: {spec['variant']} needs the per-row lidar scan (key 'lidar') of the dataset")
            lidar = data["lidar"].astype(np.float64)
    return encode(x, times, spec, lidar=lidar), y


def first_rows(dataset, record, rows=256):
    """First `rows` raw observations of an episode and, when the dataset stores one, their lidar scans (else None):
    used by the trainers to check that live (step by step) inference equals batch inference."""
    x, _, _ = load_episode(dataset, record)
    path = Path(dataset) / "episodes" / f"{record['episode_id']}.npz"
    with np.load(path, allow_pickle=False) as data:
        lidar = data["lidar"].astype(np.float64)[:rows] if "lidar" in data.files else None
    return x[:rows], lidar


__all__ = ["INPUT_NAMES", "load_episode", "load_episode_features", "validate_episode_sampling", "first_rows"]
