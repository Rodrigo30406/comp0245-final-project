"""Experiment-wide episode rules, shared by the simulator and collection."""

import json
import math
from dataclasses import asdict

from robot_learning.settings import PROJECT_ROOT
from robot_learning.simulation.setup import PhysicsParameters


EXPERT_CONFIG_PATH = PROJECT_ROOT / "experiment_specs/expert/default.json"


def episode_defaults():
    """Read a fresh copy so an experiment cannot mutate another's defaults."""
    rules = json.loads(EXPERT_CONFIG_PATH.read_text())
    physics = json.loads(json.dumps(asdict(PhysicsParameters())))
    return {**rules, "physics": physics, "headless": True, "max_steps": 40000}


def number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{label} has an invalid sign")


def vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    for v in value:
        number(v, label)


def interval(value, label):
    vector(value, 2, label)
    if value[0] > value[1]:
        raise ValueError(f"{label}: lower bound exceeds upper bound")


def resolve_episode_config(requested):
    cfg = episode_defaults()
    if not isinstance(requested, dict) or set(requested) - set(cfg):
        raise ValueError("Unknown episode configuration fields")
    for key, value in requested.items():
        if isinstance(cfg[key], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"Invalid fields in {key}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    if type(cfg["headless"]) is not bool:
        raise ValueError("headless must be boolean")
    if type(cfg["max_steps"]) is not int or cfg["max_steps"] <= 0:
        raise ValueError("max_steps must be a positive integer")
    vector(cfg["fixed_finger_angles"], 6, "fixed_finger_angles")
    if cfg["goal_reference"] not in ("origin", "start"):
        raise ValueError('goal_reference must be "origin" (the lecturer\'s rule: bearing from the world origin) or "start" '
                         '(bearing from the base start position; identical when the robot starts at the origin)')
    for group in ("base_controller", "arm_controller", "stop"):
        for key, value in cfg[group].items():
            number(value, f"{group}.{key}", positive=True)
    arm = cfg["arm_controller"]
    if arm["min_ik_damping"] > arm["max_ik_damping"]:
        raise ValueError("min_ik_damping exceeds max_ik_damping")
    physics = cfg["physics"]
    vector(physics["anisotropic_friction"], 3, "anisotropic_friction")
    for key, value in physics.items():
        for v in value if isinstance(value, list) else [value]:
            number(v, f"physics.{key}", nonnegative=True)
    heights, distances = cfg["standoff"]["heights"], cfg["standoff"]["distances"]
    if not isinstance(heights, list) or len(heights) < 2:
        raise ValueError("standoff.heights requires at least two entries")
    vector(heights, len(heights), "standoff.heights")
    vector(distances, len(heights), "standoff.distances")
    if any(a >= b for a, b in zip(heights, heights[1:])) or any(d <= 0 for d in distances):
        raise ValueError("Standoff heights must increase and distances must be positive")
    return cfg


def same_expert_configuration(a, b):
    """True when two resolved episode configurations describe the same expert and physics. `goal_reference` is not
    compared: it only decides where the base goal is placed for a non-nominal start pose (identical at the nominal
    start), so datasets with different start-pose distributions may legitimately differ in it (the training mixture)."""
    strip = lambda cfg: {k: v for k, v in cfg.items() if k != "goal_reference"}
    return strip(a) == strip(b)
