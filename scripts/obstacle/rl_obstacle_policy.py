"""Adapter that runs this project's learned policies (egocentric MLP, GRU, transformer; NumPy inference) inside the
group's obstacle harness (`final/robotnik_arm_task/obstacle_task/evaluate.py`, protocol v2.0).

Harness policy interface: reset(target, info), act(obs) -> (wheel_cmds (4,), torque_arm (7,)), one call per 1 ms step.
The observation is turned into the 21-vector of this project (base x, y, z, bearing, 7 arm joint angles, 7 arm joint
velocities, target x, y, z) and the harness's 64-beam scan obs["lidar"] is passed to the policy's lidar input (the
`raw21_targetpolar_clock_lidar64_v1` variant); the 11 commands are clipped to the expert's envelope exactly as this
project's evaluator does (robot_learning.evaluation.worker.action_limits). Nothing from info is read except the time
step; the scenario (obstacles, waypoint, side) is never seen.

  PYTHONPATH=<project>/src python -m obstacle_task.evaluate \
      --policy <project>/scripts/obstacle/rl_obstacle_policy.py:make_policy --policy-arg runs/training/<run> ...
--policy-arg: a training run directory (model.npz + model_info.json) or a model.npz path next to its model_info.json,
optionally followed by ",db=<rad/s>": the +DB parking variant of this project's evaluator (make_action_filter): when
every wheel command is below the band in magnitude, all four are set to exactly zero (the expert's latched stop).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_learning.evaluation.worker import action_limits  # noqa: E402
from robot_learning.simulation.config import resolve_episode_config  # noqa: E402

ARM = slice(4, 11)
LIMITS = action_limits(resolve_episode_config({}))   # +-5.27 rad/s wheels, +-30 N m torques: the declared envelope


def load_policy(path):
    path = Path(path)
    model_path = path if path.suffix == ".npz" else path / "model.npz"
    info = json.loads((model_path.parent / "model_info.json").read_text())
    kind = info["kind"]
    if kind == "egocentric_mlp_regressor":
        from robot_learning.learning.egocentric import EgocentricPolicy as cls
    elif kind == "gru_regressor":
        from robot_learning.learning.gru_policy import GRUPolicy as cls
    elif kind == "transformer_regressor":
        from robot_learning.learning.transformer_policy import TransformerPolicy as cls
    elif kind == "gated_regressor":
        from robot_learning.learning.gated_policy import GatedPolicy as cls
    else:
        raise ValueError(f"{model_path}: unsupported policy kind {kind!r}")
    return cls.load(model_path), info


class ProjectPolicy:
    """One of this project's policies behind the harness interface."""

    def __init__(self, path, dead_band=0.0):
        self.path = os.path.abspath(path)
        self.dead_band = float(dead_band)
        self.policy, self.info = load_policy(self.path)
        self.uses_lidar = "lidar" in str(self.info.get("variant", self.info.get("input_mode", "")))
        self.dt = None

    def reset(self, target, info):
        self.dt = float(info["time_step"])
        self.policy.reset(self.dt)

    def act(self, obs):
        q, qd = np.asarray(obs["q_mes"], dtype=np.float64), np.asarray(obs["qd_mes"], dtype=np.float64)
        observation = np.concatenate([np.asarray(obs["base_pos"], dtype=np.float64), [float(obs["base_bearing"])],
                                      q[ARM], qd[ARM], np.asarray(obs["target"], dtype=np.float64)])
        lidar = np.asarray(obs["lidar"], dtype=np.float64) if self.uses_lidar else None
        action = np.clip(self.policy.predict_action(observation, lidar=lidar), -LIMITS, LIMITS)
        if self.dead_band > 0 and np.abs(action[:4]).max() < self.dead_band:
            action[:4] = 0.0
        return action[:4], action[4:11]


def make_policy(arg):
    if not arg:
        raise ValueError("--policy-arg must give a training run directory or a model.npz path")
    path, *options = [part.strip() for part in arg.split(",")]
    dead_band = 0.0
    for option in options:
        key, _, value = option.partition("=")
        if key != "db":
            raise ValueError(f"unknown --policy-arg option {option!r} (only db=<rad/s>)")
        dead_band = float(value)
    return ProjectPolicy(path, dead_band)
