"""Policy inputs: the brief's 21 observations, their egocentric transforms, the clock and (obstacle tasks) the lidar.

Every feature is a deterministic function of what the robot observes at the current step (the 21 inputs of the
brief), of the controller's own clock, and, for the obstacle tasks, of the 64-beam lidar. No expert memory, expert
phase, expert action, base goal or future information is used. See docs/02_inputs_and_outputs.md.

The 21 observations (`INPUT_NAMES`, world frame):
    base x, y, z (m), base bearing (yaw, rad), 7 arm joint angles (rad), 7 arm joint velocities (rad/s),
    target x, y, z (m).

The egocentric transforms (`TARGET_NAMES` + `POLAR_NAMES`), with d = target_xy - base_xy and yaw the base bearing:
    target_in_base_x          =  cos(yaw) d_x + sin(yaw) d_y     target ahead of the base (m)
    target_in_base_y          = -sin(yaw) d_x + cos(yaw) d_y     target to the base's left (m)
    target_height_above_base  =  target_z - base_z               (m)
    target_distance_xy        =  |d|                             planar distance base -> target (m)
    cos/sin_heading_error_to_target = cos/sin(atan2(target_in_base_y, target_in_base_x))
                                                                 direction of the target relative to the heading
They are a rigid change of frame (rotation by -yaw about the base) and its polar form: no learned parameter, no expert
rule. The clock is the elapsed episode time divided by the evaluation horizon (40 s).

Input variants (names kept as stored in the trained model files):
    raw21_clock_v1                      21 observations + clock                                (22)  MLP-Base
    raw21_targetpolar_clock_v1          21 + 3 target-in-base + 3 polar + clock                (28)  MLP-Ego, GRU-Ego, TF-Ego
    raw21_targetpolar_clock_lidar64_v1  the 28 + 64 lidar ranges (base frame, beam 0 forward,
                                        counter-clockwise, clipped at 4 m)                     (92)  obstacle tasks
"""
import math

import numpy as np

INPUT_NAMES = ([f"base_position_{a}" for a in "xyz"] + ["base_bearing"] +
               [f"arm_q_{i}" for i in range(1, 8)] + [f"arm_qd_{i}" for i in range(1, 8)] +
               [f"target_{a}" for a in "xyz"])
TARGET_NAMES = ["target_in_base_x", "target_in_base_y", "target_height_above_base"]
POLAR_NAMES = ["target_distance_xy", "cos_heading_error_to_target", "sin_heading_error_to_target"]
LIDAR_BEAMS = 64
LIDAR_NAMES = [f"lidar_{k:02d}" for k in range(LIDAR_BEAMS)]
CLOCK_NAME = "episode_clock_fraction"

BASE = "raw21_clock_v1"
EGO = "raw21_targetpolar_clock_v1"
EGO_LIDAR = "raw21_targetpolar_clock_lidar64_v1"
VARIANTS = {BASE: INPUT_NAMES + [CLOCK_NAME],
            EGO: INPUT_NAMES + TARGET_NAMES + POLAR_NAMES + [CLOCK_NAME],
            EGO_LIDAR: INPUT_NAMES + TARGET_NAMES + POLAR_NAMES + LIDAR_NAMES + [CLOCK_NAME]}
LIDAR_VARIANTS = {EGO_LIDAR}


def validate_spec(spec):
    """Feature specification stored with every model: variant, clock horizon and step. The standoff table of the expert
    is stored too, for compatibility with the model files of the report; no kept variant reads it."""
    keys = {"variant", "standoff_heights", "standoff_distances", "horizon_s", "dt_s"}
    if set(spec) != keys or spec["variant"] not in VARIANTS:
        raise ValueError("Invalid feature specification")
    for key in ("horizon_s", "dt_s"):
        if not isinstance(spec[key], (int, float)) or not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} must be positive")


def spec_from_episode_config(cfg, dt_s, horizon_s, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return {"variant": variant, "standoff_heights": list(map(float, cfg["standoff"]["heights"])),
            "standoff_distances": list(map(float, cfg["standoff"]["distances"])),
            "horizon_s": float(horizon_s), "dt_s": float(dt_s)}


def encode(observations, times, spec, lidar=None):
    """(N, 21) observations and (N,) seconds since the episode start -> (N, F) features of spec["variant"].
    `lidar` (N, 64) is required by the lidar variant and ignored by the others."""
    x = np.asarray(observations, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 21 or t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("Expected finite (N, 21) observations and (N,) times")
    variant = spec["variant"]
    clock = (t / spec["horizon_s"])[:, None]
    if variant == BASE:
        features = np.concatenate([x, clock], axis=1)
    else:
        base_xy, base_z, yaw, target = x[:, 0:2], x[:, 2], x[:, 3], x[:, 18:21]
        c, s = np.cos(yaw), np.sin(yaw)
        d = target[:, :2] - base_xy
        target_b = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        heading = np.arctan2(target_b[:, 1], target_b[:, 0])      # already relative to the base heading
        ego = np.concatenate([target_b, (target[:, 2] - base_z)[:, None],
                              np.hypot(target_b[:, 0], target_b[:, 1])[:, None],
                              np.cos(heading)[:, None], np.sin(heading)[:, None]], axis=1)
        parts = [x, ego]
        if variant in LIDAR_VARIANTS:
            if lidar is None:
                raise ValueError(f"{variant} needs the lidar scan of every row")
            scan = np.asarray(lidar, dtype=np.float64)
            if scan.shape != (len(x), LIDAR_BEAMS) or not np.isfinite(scan).all() or (scan < 0).any():
                raise ValueError(f"lidar must be finite non-negative ({len(x)}, {LIDAR_BEAMS}) ranges")
            parts.append(scan)
        features = np.concatenate(parts + [clock], axis=1)
    if features.shape[1] != len(VARIANTS[variant]):
        raise AssertionError("Feature layout drifted from VARIANTS")
    return features
