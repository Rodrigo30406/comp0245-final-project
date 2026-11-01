"""Configuration validation and deterministic scenario generation."""

import hashlib
import json

import numpy as np
from robot_learning.simulation.config import (
    episode_defaults, interval, resolve_episode_config, vector,
)

DEFAULTS = {
    "name": "pilot_box", "episodes": 64, "seed": 20261023,
    "split": "development",
    "record_every_steps": 1,
    "target_box": {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [0.3, 0.8]},
    "targets": None, "initial_base": {"mode": "nominal"},
    "action_noise": None,
    **episode_defaults(),
}

NOISE_KEYS = {"wheel_std", "torque_std", "hold_steps", "seed"}


def validate_action_noise(noise):
    """DART-style perturbation of the APPLIED expert command; labels stay clean."""
    if noise is None:
        return None
    if not isinstance(noise, dict) or set(noise) != NOISE_KEYS:
        raise ValueError(f"action_noise needs exactly {sorted(NOISE_KEYS)}")
    for key in ("wheel_std", "torque_std"):
        if type(noise[key]) not in (int, float) or not np.isfinite(noise[key]) or noise[key] < 0:
            raise ValueError(f"action_noise.{key} must be a finite nonnegative number")
    if type(noise["hold_steps"]) is not int or noise["hold_steps"] < 1:
        raise ValueError("action_noise.hold_steps must be a positive integer")
    if type(noise["seed"]) is not int or noise["seed"] < 0:
        raise ValueError("action_noise.seed must be a nonnegative integer")
    if noise["wheel_std"] == 0 and noise["torque_std"] == 0:
        raise ValueError("action_noise with zero deviations is not a perturbation; use null")
    return noise


def resolve_config(requested):
    if not isinstance(requested, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(requested) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = json.loads(json.dumps(DEFAULTS))
    for key, value in requested.items():
        if key in ("base_controller", "arm_controller", "stop", "physics", "standoff", "target_box"):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"Invalid fields in {key}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    for key in ("episodes", "max_steps", "record_every_steps"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be nonempty")
    if type(cfg["headless"]) is not bool:
        raise ValueError("headless must be boolean")
    if cfg["split"] not in ("development", "train", "validation", "test"):
        raise ValueError("split must be development, train, validation or test")
    for axis in "xyz":
        interval(cfg["target_box"][axis], f"target_box.{axis}")
    if cfg["targets"] is not None:
        if not isinstance(cfg["targets"], list) or len(cfg["targets"]) != cfg["episodes"]:
            raise ValueError("targets must have exactly one XYZ entry per episode")
        for target in cfg["targets"]:
            vector(target, 3, "targets")
    validate_action_noise(cfg["action_noise"])
    initial = cfg["initial_base"]
    if not isinstance(initial, dict) or set(initial) - {"mode", "x", "y", "yaw", "distance"}:
        raise ValueError("Invalid initial_base fields")
    if initial.get("mode") == "uniform_xy":
        if "distance" in initial:
            raise ValueError("initial_base.distance belongs to the near_target mode")
        for axis in ("x", "y"):
            interval(initial.get(axis), f"initial_base.{axis}")
        initial.setdefault("yaw", [0.0, 0.0])
        interval(initial["yaw"], "initial_base.yaw")
    elif initial.get("mode") == "near_target":
        # Base placed on a ring around the episode's target: planar distance in `distance`, direction from the
        # target uniform on the circle, world yaw uniform in `yaw`. Coverage of the stop region (E2), not a
        # different task: the expert still drives to its own goal from wherever it is put.
        if set(initial) - {"mode", "distance", "yaw"}:
            raise ValueError("near_target takes distance and yaw bounds only")
        interval(initial.get("distance"), "initial_base.distance")
        if initial["distance"][0] <= 0:
            raise ValueError("initial_base.distance must be positive")
        initial.setdefault("yaw", [-np.pi, np.pi])
        interval(initial["yaw"], "initial_base.yaw")
    elif initial != {"mode": "nominal"}:
        raise ValueError("initial_base must be nominal, uniform_xy with x/y/yaw bounds, or near_target with distance/yaw bounds")
    cfg.update(resolve_episode_config({k: cfg[k] for k in episode_defaults()}))
    return cfg


def generate_scenarios(cfg):
    scenarios = []
    for i in range(cfg["episodes"]):
        seed = int(np.random.SeedSequence([cfg["seed"], i]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        target = (cfg["targets"][i] if cfg["targets"] is not None else
                  [float(rng.uniform(*cfg["target_box"][axis])) for axis in "xyz"])
        initial = None
        if cfg["initial_base"]["mode"] == "uniform_xy":
            initial = {axis: float(rng.uniform(*cfg["initial_base"][axis])) for axis in ("x", "y", "yaw")}
        elif cfg["initial_base"]["mode"] == "near_target":
            ring = cfg["initial_base"]
            distance = float(rng.uniform(*ring["distance"])); direction = float(rng.uniform(-np.pi, np.pi))
            initial = {"x": float(target[0] + distance * np.cos(direction)), "y": float(target[1] + distance * np.sin(direction)),
                       "yaw": float(rng.uniform(*ring["yaw"]))}
        condition = {"target": target, "initial_base": initial}
        scenario_id = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:20]
        scenarios.append({"episode_id": f"episode_{i:06d}", "scenario_id": scenario_id,
                          "seed": seed, "split": cfg["split"], **condition})
    return scenarios
