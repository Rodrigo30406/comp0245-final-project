"""Declared variant: egocentric goal geometry plus an episode clock as policy inputs.

Every feature is a deterministic function of the 21 baseline inputs, the fixed
standoff table of the expert configuration, and the controller's own clock.
No expert memory, expert action or future information is used. The variant
exists to test whether observable surrogates for the expert's navigation phase
(research: docs/PRIVILEGED_ABLATION.md) change supervised and closed-loop
behaviour. Output interface is unchanged: nine controls expanded to eleven.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from robot_learning.data.storage import read_json, write_json, sha256, utc_now, git_info
from robot_learning.settings import PROJECT_ROOT
from robot_learning.simulation.config import episode_defaults, resolve_episode_config, same_expert_configuration
from .dataset import inventory, load_episode, validate_partition, inventory_fingerprint, extra_inventory, record_directory, drop_expert_failures
from .metrics import errors, aggregate
from .model import INPUT_NAMES, OUTPUT_NAMES, expand_actions

FEATURE_NAMES = (["goal_in_base_x", "goal_in_base_y", "goal_distance_xy",
                  "cos_heading_error_to_goal", "sin_heading_error_to_goal",
                  "cos_goal_bearing_error", "sin_goal_bearing_error",
                  "target_in_base_x", "target_in_base_y", "target_height_above_base"] +
                 [f"arm_q_{i}" for i in range(1, 8)] + [f"arm_qd_{i}" for i in range(1, 8)] +
                 ["episode_clock_fraction"])
VARIANT = "egocentric_clock_v1"
GEOMETRY_NAMES = FEATURE_NAMES[:10]
TARGET_NAMES = FEATURE_NAMES[7:10]  # target in the base frame: no expert goal construction involved
POLAR_NAMES = ["target_distance_xy", "cos_heading_error_to_target", "sin_heading_error_to_target"]
VELOCITY_NAMES = ["base_forward_speed_fd", "base_lateral_speed_fd", "base_yaw_rate_fd"]
PREV_WHEEL_NAMES = ["prev_wheel_right_cmd", "prev_wheel_left_cmd"]
CLOCK_NAME = "episode_clock_fraction"
# Input layouts. The lecturer's 21 explicit inputs may be kept and extended (not replaced):
#   egocentric_clock_v1       : 10 relative geometry + 14 joint + clock (25); replaces the 7 absolute pose/target inputs
#   raw21_clock_v1            : the 21 explicit inputs + clock (22)
#   raw21_egocentric_clock_v1 : the 21 explicit inputs + 10 relative geometry + clock (32)
#   raw21_target_clock_v1     : the 21 explicit inputs + 3 target-in-base-frame features + clock (25).
#                               Uses no knowledge of the expert (no standoff table, no goal bearing rule):
#                               only a rigid transform of observed quantities into the robot frame.
#   raw21_targetpolar_clock_v1: raw21_target_clock_v1 plus the same target in polar form (planar distance
#                               base->target and cos/sin of the heading error to the target) + clock (28).
#                               Still no expert knowledge: these are functions of the 3 target-in-base features.
#   raw21_targetpolar_vel_clock_v1: raw21_targetpolar_clock_v1 plus the base velocity by finite difference of two
#                               consecutive observations (forward and lateral speed in the base frame, yaw rate) (31).
#                               The lecturer's inputs carry no base velocity; this is the one-step memory a robot's
#                               odometry provides. Zero on the first step of an episode (the robot starts at rest).
#   raw21_targetpolar_prevwheel_clock_v1: raw21_targetpolar_clock_v1 plus the policy's OWN wheel command of the previous
#                               step (30). In training it is the demonstrated command of the previous row (teacher
#                               forcing), in closed loop the policy's previous output; zero on the first step. No expert
#                               state is read: a controller always knows what it commanded a millisecond ago.
VARIANTS = {VARIANT: list(FEATURE_NAMES),
            "raw21_clock_v1": list(INPUT_NAMES) + [CLOCK_NAME],
            "raw21_egocentric_clock_v1": list(INPUT_NAMES) + list(GEOMETRY_NAMES) + [CLOCK_NAME],
            "raw21_target_clock_v1": list(INPUT_NAMES) + list(TARGET_NAMES) + [CLOCK_NAME],
            "raw21_targetpolar_clock_v1": list(INPUT_NAMES) + list(TARGET_NAMES) + POLAR_NAMES + [CLOCK_NAME],
            "raw21_targetpolar_vel_clock_v1": list(INPUT_NAMES) + list(TARGET_NAMES) + POLAR_NAMES + VELOCITY_NAMES + [CLOCK_NAME],
            "raw21_targetpolar_prevwheel_clock_v1": list(INPUT_NAMES) + list(TARGET_NAMES) + POLAR_NAMES + PREV_WHEEL_NAMES + [CLOCK_NAME]}
ACTION_INPUT_VARIANTS = {"raw21_targetpolar_prevwheel_clock_v1"}  # need the previous wheel command
STATEFUL_VARIANTS = {"raw21_targetpolar_vel_clock_v1", *ACTION_INPUT_VARIANTS}  # live inference keeps a previous step
POLAR_VARIANTS = {"raw21_targetpolar_clock_v1", "raw21_targetpolar_vel_clock_v1", *ACTION_INPUT_VARIANTS}
KIND = "egocentric_mlp_regressor"
# Training options (config block "egocentric_options", overridable with --option KEY=VALUE). Defaults reproduce the
# lecturer's MLP recipe exactly: L2 on all nine outputs, uniform rows, continuous wheel outputs.
DEFAULT_OPTIONS = {"wheel_loss": "l2",            # l2 | l1 on the two wheel outputs (the arm is always L2)
                   "stop_weight": 1.0,            # weight of the wheel loss on rows whose demonstrated wheels are exactly zero
                   "wheel_output": "regression",  # regression | categorical (softmax over wheel_bins velocity bins, argmax decode)
                   "wheel_bins": 41,              # odd, so one bin is exactly zero; range = max |demonstrated wheel command|
                   "prev_wheel_shuffle_p": 0.0,   # action-input variants: fraction of rows whose previous command is swapped within the batch
                   "prev_wheel_noise_std": 0.0,   # action-input variants: N(0, std) noise on the normalized previous command
                   "trigger_copies": 0,           # E9: copies per training episode of its first terminal zero-wheel row (the stop/reach onset)
                   "trigger_qd_jitter": "auto",   # rad/s N(0, s) on the arm velocities of those copies; "auto" = std of arm qd over the moving rows
                   "trigger_weight": 1.0,         # loss weight of the trigger row itself (the no-duplication twin of trigger_copies)
                   "action_chunk": 1,             # predict the commands of this many future steps (chunk), execute the first
                   "action_chunk_stride": 20,     # physics rows between chunk steps (20 = 20 ms)
                   "training_episode_cap": 0}     # data-size series: keep only this many training episodes (after the success filter),
                                                  # a fixed-rng subset shared by every training seed; 0 = all
TRIGGER_JITTER_SEED = 0  # one fixed rng for the trigger-copy jitter, so every training seed sees the same added rows


def resolve_options(config, variant, overrides=None):
    opts = {**DEFAULT_OPTIONS, **config.get("egocentric_options", {}), **(overrides or {})}
    if set(opts) != set(DEFAULT_OPTIONS):
        raise ValueError(f"Unknown egocentric options {sorted(set(opts) - set(DEFAULT_OPTIONS))}")
    if opts["wheel_loss"] not in ("l2", "l1") or opts["wheel_output"] not in ("regression", "categorical"):
        raise ValueError("wheel_loss must be l2|l1 and wheel_output regression|categorical")
    if opts["wheel_output"] == "categorical" and opts["wheel_loss"] != "l2":
        raise ValueError("categorical wheels are trained with cross-entropy; wheel_loss must stay l2 (unused)")
    if type(opts["wheel_bins"]) is not int or opts["wheel_bins"] < 3 or opts["wheel_bins"] % 2 == 0:
        raise ValueError("wheel_bins must be an odd integer >= 3")
    for key, low in (("stop_weight", 1.0), ("prev_wheel_shuffle_p", 0.0), ("prev_wheel_noise_std", 0.0)):
        if not isinstance(opts[key], (int, float)) or isinstance(opts[key], bool) or not math.isfinite(opts[key]) or opts[key] < low:
            raise ValueError(f"{key} must be a finite number >= {low}")
        opts[key] = float(opts[key])
    if opts["prev_wheel_shuffle_p"] > 1:
        raise ValueError("prev_wheel_shuffle_p is a probability")
    if (opts["prev_wheel_shuffle_p"] or opts["prev_wheel_noise_std"]) and variant not in ACTION_INPUT_VARIANTS:
        raise ValueError("prev_wheel options need an action-input variant")
    if type(opts["trigger_copies"]) is not int or opts["trigger_copies"] < 0:
        raise ValueError("trigger_copies must be an integer >= 0")
    jitter = opts["trigger_qd_jitter"]
    if jitter != "auto" and (not isinstance(jitter, (int, float)) or isinstance(jitter, bool) or not math.isfinite(jitter) or jitter < 0):
        raise ValueError('trigger_qd_jitter must be "auto" or a finite number >= 0 (rad/s)')
    if jitter != "auto":
        opts["trigger_qd_jitter"] = float(jitter)
    tw = opts["trigger_weight"]
    if not isinstance(tw, (int, float)) or isinstance(tw, bool) or not math.isfinite(tw) or tw < 1:
        raise ValueError("trigger_weight must be a finite number >= 1")
    opts["trigger_weight"] = float(tw)
    for key in ("action_chunk", "action_chunk_stride"):
        if type(opts[key]) is not int or opts[key] < 1:
            raise ValueError(f"{key} must be an integer >= 1")
    if type(opts["training_episode_cap"]) is not int or opts["training_episode_cap"] < 0:
        raise ValueError("training_episode_cap must be an integer >= 0 (0 = every training episode)")
    if opts["action_chunk"] > 1 and (opts["wheel_output"] != "regression" or opts["wheel_loss"] != "l2" or opts["stop_weight"] != 1.0):
        raise ValueError("action_chunk > 1 needs plain L2 regression (wheel_output regression, wheel_loss l2, stop_weight 1)")
    return opts


def chunk_labels(y, ranges, steps, stride):
    """(N, 9 * steps) labels: for row i of an episode (a, b), the commands at rows i, i + stride, ..., i + (steps - 1) * stride,
    clipped to the episode's last row (the expert keeps its final command). steps = 1 returns y itself."""
    y = np.asarray(y, dtype=np.float64)
    if steps == 1:
        return y
    out = np.empty((len(y), y.shape[1] * steps))
    for a, b in ranges:
        idx = np.arange(a, b)
        for k in range(steps):
            out[a:b, k * y.shape[1]:(k + 1) * y.shape[1]] = y[np.minimum(idx + k * stride, b - 1)]
    return out


def trigger_rows(y, ranges):
    """Index of the trigger row of each episode: the first row of its terminal run of exactly-zero wheel labels (the
    expert's navigation end, where the wheels stop and the reach torque starts). Label-derived only: no expert phase or
    done flag is read. None for an episode whose last row still has a non-zero wheel command."""
    y = np.asarray(y)
    out = []
    for a, b in ranges:
        zero = (y[a:b, :2] == 0).all(axis=1)
        if b <= a or not zero[-1]:
            out.append(None); continue
        k = len(zero) - 1
        while k > 0 and zero[k - 1]:
            k -= 1
        out.append(a + k)
    return out


def trigger_copies(x, y, names, rows, copies, jitter="auto", seed=TRIGGER_JITTER_SEED):
    """`copies` copies of each trigger row (x features and y labels exactly as recorded) with N(0, sigma_j) noise added
    to the seven arm-velocity features only. sigma_j = `jitter` (rad/s), or with "auto" the per-joint std of arm qd over
    the rows whose demonstrated wheels are not both zero (the arm's velocity noise while the base moves). Returns
    (x_copies, y_copies, sigma)."""
    rows = [r for r in rows if r is not None]
    qd_cols = [names.index(f"arm_qd_{i}") for i in range(1, 8)]
    moving = ~(np.asarray(y)[:, :2] == 0).all(axis=1)
    sigma = (np.asarray(x)[moving][:, qd_cols].std(axis=0) if jitter == "auto" else np.full(7, float(jitter)))
    if not rows or copies <= 0:
        return np.zeros((0, len(names))), np.zeros((0, np.asarray(y).shape[1])), sigma
    idx = np.repeat(np.asarray(rows, dtype=np.int64), copies)
    xc = np.asarray(x, dtype=np.float64)[idx].copy(); yc = np.asarray(y, dtype=np.float64)[idx].copy()
    rng = np.random.default_rng(seed)
    xc[:, qd_cols] += rng.normal(0.0, 1.0, (len(idx), 7)) * sigma
    return xc, yc, sigma


def decode_outputs(out, y_mean, y_scale, wheel_bin_centres=None):
    """Network outputs -> nine controls in physical units; categorical wheels take the centre of the argmax bin."""
    out = np.asarray(out, dtype=np.float64)
    if wheel_bin_centres is None:
        return out * y_scale + y_mean
    b = len(wheel_bin_centres)
    wheels = np.stack([wheel_bin_centres[out[:, :b].argmax(axis=1)], wheel_bin_centres[out[:, b:2 * b].argmax(axis=1)]], axis=1)
    return np.concatenate([wheels, out[:, 2 * b:] * y_scale[2:] + y_mean[2:]], axis=1)


def validate_spec(spec):
    keys = {"variant", "standoff_heights", "standoff_distances", "horizon_s", "dt_s"}
    if set(spec) != keys or spec["variant"] not in VARIANTS:
        raise ValueError("Invalid egocentric feature specification")
    h = np.asarray(spec["standoff_heights"], dtype=float); d = np.asarray(spec["standoff_distances"], dtype=float)
    if h.ndim != 1 or h.shape != d.shape or len(h) < 2 or np.any(np.diff(h) <= 0) or not np.isfinite(np.r_[h, d]).all():
        raise ValueError("Standoff table must be increasing heights with matching distances")
    for key in ("horizon_s", "dt_s"):
        if not isinstance(spec[key], (int, float)) or not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} must be positive")


def spec_from_episode_config(cfg, dt_s, horizon_s, variant=VARIANT):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return {"variant": variant, "standoff_heights": list(map(float, cfg["standoff"]["heights"])),
            "standoff_distances": list(map(float, cfg["standoff"]["distances"])),
            "horizon_s": float(horizon_s), "dt_s": float(dt_s)}


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def encode(observations, times, spec, previous=None, previous_wheels=None):
    """(N, 21) raw observations and (N,) seconds since episode start -> (N, F) features of spec["variant"].

    `previous` is the observation before the first row (stateful velocity variant); `previous_wheels` the (N, 2) wheel
    command issued before each row (action-input variants: demonstrated in training, the policy's own in closed loop).

    Reproduces the expert's goal construction from `Episode._initialize`: bearing from the
    world origin to the target, standoff interpolated on target height, goal behind the
    target along that bearing. Relative quantities are expressed in the base frame.
    """
    x = np.asarray(observations, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 21 or t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("Expected finite (N, 21) observations and (N,) times")
    if previous is not None:
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != (21,) or not np.isfinite(previous).all():
            raise ValueError("previous must be one finite 21D observation")
    base_xy, base_z, yaw = x[:, 0:2], x[:, 2], x[:, 3]
    q, qd, target = x[:, 4:11], x[:, 11:18], x[:, 18:21]
    planar = np.hypot(target[:, 0], target[:, 1])
    goal_bearing = np.where(planar > 1e-4, np.arctan2(target[:, 1], target[:, 0]), 0.0)
    standoff = np.interp(target[:, 2], spec["standoff_heights"], spec["standoff_distances"])
    goal = target[:, :2] - standoff[:, None] * np.stack([np.cos(goal_bearing), np.sin(goal_bearing)], axis=1)
    c, s = np.cos(yaw), np.sin(yaw)

    def to_base(delta):
        return np.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], axis=1)

    d_goal = goal - base_xy
    goal_b = to_base(d_goal)
    distance = np.hypot(d_goal[:, 0], d_goal[:, 1])
    heading_error = wrap(np.arctan2(d_goal[:, 1], d_goal[:, 0]) - yaw)
    bearing_error = wrap(goal_bearing - yaw)
    target_b = to_base(target[:, :2] - base_xy)
    clock = t / spec["horizon_s"]
    geometry = np.concatenate([
        goal_b, distance[:, None], np.cos(heading_error)[:, None], np.sin(heading_error)[:, None],
        np.cos(bearing_error)[:, None], np.sin(bearing_error)[:, None],
        target_b, (target[:, 2] - base_z)[:, None]], axis=1)
    variant = spec["variant"]
    if variant == VARIANT:
        features = np.concatenate([geometry, q, qd, clock[:, None]], axis=1)
    elif variant == "raw21_clock_v1":
        features = np.concatenate([x, clock[:, None]], axis=1)
    elif variant == "raw21_egocentric_clock_v1":
        features = np.concatenate([x, geometry, clock[:, None]], axis=1)
    elif variant == "raw21_target_clock_v1":
        features = np.concatenate([x, geometry[:, 7:10], clock[:, None]], axis=1)
    elif variant in POLAR_VARIANTS:
        target_distance = np.hypot(target_b[:, 0], target_b[:, 1])
        heading_to_target = np.arctan2(target_b[:, 1], target_b[:, 0])  # already relative to the base heading
        polar = np.stack([target_distance, np.cos(heading_to_target), np.sin(heading_to_target)], axis=1)
        if variant == "raw21_targetpolar_clock_v1":
            features = np.concatenate([x, geometry[:, 7:10], polar, clock[:, None]], axis=1)
        elif variant in ACTION_INPUT_VARIANTS:
            if previous_wheels is None:
                raise ValueError(f"{variant} needs the previous wheel command of every row")
            pw = np.asarray(previous_wheels, dtype=np.float64)
            if pw.shape != (len(x), 2) or not np.isfinite(pw).all():
                raise ValueError("previous_wheels must be finite (N, 2)")
            features = np.concatenate([x, geometry[:, 7:10], polar, pw, clock[:, None]], axis=1)
        else:
            # Finite-difference base velocity from consecutive observations of ONE episode in order; the row before the
            # first is the first itself (zero velocity: every episode starts at rest) unless `previous` is given.
            prev = np.vstack([x[:1] if previous is None else previous[None], x[:-1]])
            dxy = (x[:, :2] - prev[:, :2]) / spec["dt_s"]
            velocity = np.concatenate([to_base(dxy), (wrap(yaw - prev[:, 3]) / spec["dt_s"])[:, None]], axis=1)
            features = np.concatenate([x, geometry[:, 7:10], polar, velocity, clock[:, None]], axis=1)
    else:
        raise ValueError(f"Unknown input variant {variant!r}")
    if features.shape[1] != len(VARIANTS[variant]):
        raise AssertionError("Feature layout drifted from VARIANTS")
    return features


class EgocentricPolicy:
    """ReLU MLP over egocentric features with an internal episode clock; NumPy inference."""

    def __init__(self, spec, x_mean, x_scale, y_mean, y_scale, weights, biases, wheel_bin_centres=None, action_chunk=1):
        validate_spec(spec)
        self.spec = spec
        if type(action_chunk) is not int or action_chunk < 1 or (action_chunk > 1 and wheel_bin_centres is not None):
            raise ValueError("action_chunk must be an integer >= 1 (regression outputs only)")
        self.action_chunk = action_chunk
        self.x_mean = np.asarray(x_mean); self.x_scale = np.asarray(x_scale)
        self.y_mean = np.asarray(y_mean); self.y_scale = np.asarray(y_scale)
        self.weights = [np.asarray(w) for w in weights]; self.biases = [np.asarray(b) for b in biases]
        self.names = list(VARIANTS[spec["variant"]])
        self.wheel_bin_centres = None if wheel_bin_centres is None else np.asarray(wheel_bin_centres, dtype=np.float64)
        if self.wheel_bin_centres is not None:
            c = self.wheel_bin_centres
            if c.ndim != 1 or len(c) < 3 or not np.isfinite(c).all() or np.any(np.diff(c) <= 0):
                raise ValueError("wheel_bin_centres must be an increasing finite vector")
        n = len(self.names)
        if any(v.shape != (k,) for v, k in [(self.x_mean, n), (self.x_scale, n), (self.y_mean, 9 * action_chunk), (self.y_scale, 9 * action_chunk)]):
            raise ValueError("Invalid normalization shapes")
        if len(self.weights) < 2 or len(self.biases) != len(self.weights):
            raise ValueError("At least one hidden layer with matched biases is required")
        width = n
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError("Invalid layer shapes")
            width = w.shape[1]
        if width != self.output_width:
            raise ValueError("Output width must be nine controls (or 2 x wheel bins + 7 arm torques)")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Scales must be positive")
        if not all(np.isfinite(v).all() for v in self.arrays().values() if v.dtype.kind == "f"):
            raise ValueError("Non-finite parameters")
        self._count = 0; self._dt = spec["dt_s"]; self._prev = None; self._prev_wheels = np.zeros(2)

    @property
    def output_width(self):
        return 9 * self.action_chunk if self.wheel_bin_centres is None else 2 * len(self.wheel_bin_centres) + 7

    @property
    def input_features(self):
        return len(self.names)

    def arrays(self):
        result = {"variant": np.array(self.spec["variant"]), "x_mean": self.x_mean, "x_scale": self.x_scale, "y_mean": self.y_mean, "y_scale": self.y_scale,
                  "standoff_heights": np.asarray(self.spec["standoff_heights"], dtype=np.float64),
                  "standoff_distances": np.asarray(self.spec["standoff_distances"], dtype=np.float64),
                  "clock": np.array([self.spec["horizon_s"], self.spec["dt_s"]], dtype=np.float64)}
        result.update({f"weight_{i}": w for i, w in enumerate(self.weights)})
        result.update({f"bias_{i}": b for i, b in enumerate(self.biases)})
        if self.wheel_bin_centres is not None:
            result["wheel_bin_centres"] = self.wheel_bin_centres
        if self.action_chunk > 1:
            result["action_chunk"] = np.array(self.action_chunk)
        return result

    def predict_features(self, features, block=65536):
        """Row-independent MLP; evaluated in blocks so a 36M-row training set does not allocate 18 GB per layer."""
        features = np.asarray(features, dtype=np.float64)
        if len(features) > block:
            return np.concatenate([self.predict_features(features[s:s + block], block) for s in range(0, len(features), block)])
        hidden = ((features - self.x_mean) / self.x_scale).astype(self.weights[0].dtype)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ w + b
            if i < len(self.weights) - 1:
                np.maximum(hidden, 0, out=hidden)
        return decode_outputs(hidden, self.y_mean, self.y_scale, self.wheel_bin_centres)[:, :9]   # chunk: execute the first step

    def predict_chunk(self, features):
        """(N, 9 * action_chunk) commands of the whole predicted chunk, physical units."""
        features = np.asarray(features, dtype=np.float64)
        hidden = ((features - self.x_mean) / self.x_scale).astype(self.weights[0].dtype)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ w + b
            if i < len(self.weights) - 1:
                np.maximum(hidden, 0, out=hidden)
        return decode_outputs(hidden, self.y_mean, self.y_scale, self.wheel_bin_centres)

    def predict(self, observations, times=None):
        """Batch inference for ONE episode in order; times default to step*dt from zero.

        Action-input variants run sequentially on their own previous outputs (what closed loop does); their
        teacher-forced features come from `load_episode_features`.
        """
        x = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        t = np.arange(len(x)) * self.spec["dt_s"] if times is None else np.asarray(times, dtype=np.float64)
        if self.spec["variant"] not in ACTION_INPUT_VARIANTS:
            return self.predict_features(encode(x, t, self.spec))
        out = np.empty((len(x), 9)); prev = np.zeros((1, 2))
        for i in range(len(x)):
            out[i] = self.predict_features(encode(x[i:i + 1], t[i:i + 1], self.spec, previous_wheels=prev))[0]
            prev = out[i:i + 1, :2]
        return out

    def predict_action(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (21,):
            raise ValueError("Live inference takes one 21D observation")
        t = self._count * self._dt; self._count += 1
        features = encode(x[None], np.array([t]), self.spec, previous=self._prev, previous_wheels=self._prev_wheels[None])
        action = self.predict_features(features)[0]
        self._prev = x; self._prev_wheels = action[:2].copy()
        return expand_actions(action)

    def reset(self, dt=None):
        if dt is not None and not np.isclose(dt, self.spec["dt_s"], rtol=0, atol=1e-12):
            raise ValueError("Policy clock dt differs from simulator dt")
        self._count = 0; self._prev = None; self._prev_wheels = np.zeros(2)

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            count = sum(k.startswith("weight_") for k in data.files)
            spec = {"variant": str(data["variant"]) if "variant" in data.files else VARIANT, "standoff_heights": data["standoff_heights"].tolist(),
                    "standoff_distances": data["standoff_distances"].tolist(),
                    "horizon_s": float(data["clock"][0]), "dt_s": float(data["clock"][1])}
            return cls(spec, data["x_mean"], data["x_scale"], data["y_mean"], data["y_scale"],
                       [data[f"weight_{i}"] for i in range(count)], [data[f"bias_{i}"] for i in range(count)],
                       data["wheel_bin_centres"] if "wheel_bin_centres" in data.files else None,
                       action_chunk=int(data["action_chunk"]) if "action_chunk" in data.files else 1)


def load_episode_features(directory, record, spec):
    """Observations, nine controls and recorded times; fails closed on sparse logging."""
    from .windows import validate_episode_sampling
    directory = record_directory(record, directory)
    x, y, _ = load_episode(directory, record)
    path = Path(directory) / "episodes" / f"{record['episode_id']}.npz"
    with np.load(path, allow_pickle=False) as data:
        validate_episode_sampling(data, read_json(path.with_suffix(".json")), {"dt_s": spec["dt_s"]})
        times = data["time"].astype(np.float64)
    previous_wheels = None
    if spec["variant"] in ACTION_INPUT_VARIANTS:  # the demonstrated command of the previous row; none before the first
        previous_wheels = np.vstack([np.zeros((1, 2)), np.asarray(y, dtype=np.float64)[:-1, :2]])
    return encode(x, times, spec, previous_wheels=previous_wheels), y


def load_group(directory, records, spec):
    xs, ys = [], []
    for record in records:
        f, y = load_episode_features(directory, record, spec); xs.append(f); ys.append(y)
    rows = [len(f) for f in xs]
    return (np.concatenate(xs), np.concatenate(ys),
            list(zip(np.r_[0, np.cumsum(rows)[:-1]].tolist(), np.cumsum(rows).tolist())))


def train(dataset, split_path, config_path, output, horizon_s=40.0, device_name="cuda", variant=VARIANT, options=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant {variant!r}; expected one of {sorted(VARIANTS)}")
    names = list(VARIANTS[variant])
    dataset = Path(dataset).resolve(); output = Path(output).resolve(); split_path = Path(split_path).resolve()
    config = read_json(config_path); hp = config["mlp"]
    if config.get("model") != "mlp" or config.get("backend") != "pytorch":
        raise ValueError("Egocentric variant reuses the PyTorch MLP recipe")
    opts = resolve_options(config, variant, options)
    categorical = opts["wheel_output"] == "categorical"
    augment = variant in ACTION_INPUT_VARIANTS and (opts["prev_wheel_shuffle_p"] > 0 or opts["prev_wheel_noise_std"] > 0)
    saved = read_json(split_path)
    if Path(saved["dataset"]).resolve() != dataset:
        raise ValueError("Split belongs to another dataset")
    records = inventory(dataset) + extra_inventory(saved)
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    keep_failures = bool(config.get("keep_expert_failures", True))
    episode_filter = {"keep_expert_failures": keep_failures}
    if not keep_failures:
        split, counts = drop_expert_failures(split); episode_filter.update(counts)
    cap = opts["training_episode_cap"]
    if cap:
        if cap > len(split["training"]):
            raise ValueError(f"training_episode_cap {cap} exceeds the {len(split['training'])} training episodes available")
        keep = sorted(np.random.RandomState(TRIGGER_JITTER_SEED).permutation(len(split["training"]))[:cap])
        split = {**split, "training": [split["training"][i] for i in keep]}
        episode_filter["training_episode_cap"] = {"cap": cap, "kept": cap, "rng_seed": TRIGGER_JITTER_SEED}
    source_cfg = read_json(dataset / "config.json")
    episode_cfg = resolve_episode_config({k: v for k, v in source_cfg.items() if k in episode_defaults()})
    dt = read_json(dataset / "episodes" / f"{records[0]['episode_id']}.json")["dt"]
    extra_hashes = {}
    for extra in saved.get("extra_datasets", []):
        extra = Path(extra).resolve()
        extra_cfg = resolve_episode_config({k: v for k, v in read_json(extra / "config.json").items() if k in episode_defaults()})
        extra_dt = read_json(extra / "episodes" / f"{inventory(extra)[0]['episode_id']}.json")["dt"]
        if not same_expert_configuration(extra_cfg, episode_cfg) or extra_dt != dt:
            raise ValueError(f"Extra dataset {extra} has a different episode configuration or dt")
        extra_hashes[str(extra)] = {n: sha256(extra / n) for n in ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]}
    spec = spec_from_episode_config(episode_cfg, dt, horizon_s, variant); validate_spec(spec)
    device = torch.device(device_name)
    torch.manual_seed(hp["seed"]); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {**config, "input_features": variant, "clock_horizon_s": horizon_s, "egocentric_options": opts})
    shutil.copy2(split_path, output / "split.json")
    source_paths = sorted((PROJECT_ROOT / "src/robot_learning/learning").glob("*.py"))
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths}
    for p in source_paths:
        dest = output / "source_snapshot" / p.relative_to(PROJECT_ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = {"status": "running", "created_at": utc_now(), "dataset": str(dataset),
                "dataset_hashes": {n: sha256(dataset / n) for n in
                                   ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]},
                "extra_datasets": extra_hashes,
                "source_hashes": source_hashes, "project": git_info(PROJECT_ROOT),
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch"]},
                "python": sys.version, "command": sys.argv, "input_names": names,
                "raw_observation_names": INPUT_NAMES, "output_names": OUTPUT_NAMES,
                "feature_spec": spec, "backend": "pytorch", "egocentric_options": opts, "episode_filter": episode_filter,
                "test_set": "none; development train/validation only",
                "closed_loop_evaluation": "not run by the trainer"}
    write_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        x, y, ranges = load_group(dataset, split["training"], spec)
        vx, vy, vranges = load_group(dataset, split["validation"], spec)
        x_mean, x_scale = x.mean(axis=0), x.std(axis=0)   # scaler from the recorded rows only, copies excluded
        constant_inputs = [names[i] for i in np.flatnonzero(x_scale == 0)]; x_scale[x_scale == 0] = 1.0
        y_mean, y_scale = y.mean(axis=0), y.std(axis=0); y_scale[y_scale == 0] = 1.0
        chunk = opts["action_chunk"]; chunk_stride = opts["action_chunk_stride"]
        ychunk = chunk_labels(y, ranges, chunk, chunk_stride)            # (N, 9 * chunk); == y when chunk == 1
        y_mean_c, y_scale_c = np.tile(y_mean, chunk), np.tile(y_scale, chunk)
        row_weight = np.ones(len(y), dtype=np.float32); trigger_rows_weighted = 0
        if opts["trigger_weight"] != 1.0:
            tr = [r for r in trigger_rows(y, ranges) if r is not None]
            row_weight[tr] = opts["trigger_weight"]; trigger_rows_weighted = len(tr)
        trigger_info = None
        if opts["trigger_copies"] > 0:
            rows = trigger_rows(y, ranges)
            xc, yc, sigma = trigger_copies(x, ychunk, names, rows, opts["trigger_copies"], opts["trigger_qd_jitter"])
            trigger_info = {"copies_per_episode": opts["trigger_copies"], "episodes_with_trigger_row": sum(r is not None for r in rows),
                            "episodes_without": sum(r is None for r in rows), "rows_added": int(len(xc)),
                            "recorded_rows": int(len(x)), "qd_jitter_rad_s": [float(v) for v in sigma],
                            "jitter_rule": opts["trigger_qd_jitter"], "jitter_seed": TRIGGER_JITTER_SEED,
                            "rule": "per training episode, the first row of the terminal run of exactly-zero wheel labels, copied with its "
                                    "recorded features and labels; N(0, sigma_j) added to the seven arm-velocity features only; "
                                    "input/output scaler computed from the recorded rows before the copies were appended"}
            x = np.vstack([x, xc]); ychunk = np.vstack([ychunk, yc]); y = np.vstack([y, yc[:, :9]])
            row_weight = np.concatenate([row_weight, np.ones(len(xc), dtype=np.float32)])
        xn = ((x - x_mean) / x_scale).astype(np.float32); yn = ((ychunk - y_mean_c) / y_scale_c).astype(np.float32)
        vxn = ((vx - x_mean) / x_scale).astype(np.float32)
        tx = torch.from_numpy(xn).to(device); ty = torch.from_numpy(yn).to(device); tvx = torch.from_numpy(vxn).to(device)
        centres = None; tb = None; tw = None
        trw = torch.from_numpy(row_weight).to(device) if opts["trigger_weight"] != 1.0 else None
        if categorical:
            # Symmetric bins over the demonstrated wheel envelope; an odd count puts one bin centre exactly at zero.
            limit = float(np.abs(y[:, :2]).max())
            if limit <= 0:
                raise ValueError("Demonstrated wheel commands are all zero; nothing to bin")
            centres = np.linspace(-limit, limit, opts["wheel_bins"])
            bins = np.rint((y[:, :2] + limit) / (2 * limit) * (opts["wheel_bins"] - 1)).astype(np.int64)
            tb = torch.from_numpy(bins).to(device)
        if opts["stop_weight"] != 1.0:
            stopped = (y[:, :2] == 0).all(axis=1)
            tw = torch.from_numpy(np.where(stopped, opts["stop_weight"], 1.0).astype(np.float32)).to(device)
            stop_rows = int(stopped.sum())
        else:
            stop_rows = int((y[:, :2] == 0).all(axis=1).sum())
        if augment:
            prev_cols = [names.index(n) for n in PREV_WHEEL_NAMES]
            col_mask = torch.zeros(len(names), dtype=torch.bool, device=device); col_mask[prev_cols] = True
        out_width = 9 * chunk if centres is None else 2 * len(centres) + 7
        widths = [len(names), *hp["hidden_layer_sizes"], out_width]
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(torch.nn.ReLU())
        network = torch.nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        rng = np.random.RandomState(hp["seed"])
        with torch.no_grad():
            for layer, a, b in zip(network[::2], widths[:-1], widths[1:]):
                bound = np.sqrt(6.0 / (a + b))
                layer.weight.copy_(torch.from_numpy(rng.uniform(-bound, bound, (a, b)).astype(np.float32).T.copy()))
                layer.bias.copy_(torch.from_numpy(rng.uniform(-bound, bound, b).astype(np.float32)))
        optimizer = torch.optim.Adam(network.parameters(), lr=hp["learning_rate"], betas=(.9, .999), eps=1e-8)

        def export():
            return EgocentricPolicy(spec, x_mean, x_scale, y_mean_c, y_scale_c,
                                    [l.weight.detach().cpu().numpy().T.copy() for l in network[::2]],
                                    [l.bias.detach().cpu().numpy().copy() for l in network[::2]], centres, action_chunk=chunk)

        def score():
            with torch.no_grad():
                pred = np.concatenate([network(tvx[s:s + 16384]).double().cpu().numpy() for s in range(0, len(tvx), 16384)])
            pred = decode_outputs(pred, y_mean_c, y_scale_c, centres)[:, :9]   # the executed (first) step of the chunk
            return float(np.mean([errors(vy[a:b], pred[a:b], y_scale)["balanced_nrmse"] for a, b in vranges]))

        def row_objective(xb, idx):
            """Per-row loss: wheel term (L2 / L1 / cross-entropy, optionally stop-weighted) + half-sum L2 on the arm."""
            out = network(xb)
            if categorical:
                b = len(centres)
                wheel = (torch.nn.functional.cross_entropy(out[:, :b], tb[idx, 0], reduction="none") +
                         torch.nn.functional.cross_entropy(out[:, b:2 * b], tb[idx, 1], reduction="none"))
                arm = out[:, 2 * b:]
            else:
                diff = out[:, :2] - ty[idx, :2]
                wheel = .5 * diff.square().sum(dim=1) if opts["wheel_loss"] == "l2" else diff.abs().sum(dim=1)
                arm = out[:, 2:9]
                if chunk > 1:   # later chunk steps: half-sum L2 on all their columns, added to the wheel term
                    wheel = wheel + .5 * (out[:, 9:] - ty[idx, 9:]).square().sum(dim=1)
            if tw is not None:
                wheel = wheel * tw[idx]
            rows = wheel + .5 * (arm - ty[idx, 2:9]).square().sum(dim=1)
            if trw is not None:
                rows = rows * trw[idx]
            return rows

        def augmented(xb, size):
            """Anti-copycat on the previous-command columns: swap them between rows and add noise, both in normalized units."""
            if opts["prev_wheel_shuffle_p"] > 0:
                swap = torch.from_numpy(rng.random(size) < opts["prev_wheel_shuffle_p"]).to(device)
                perm = torch.from_numpy(rng.permutation(size)).to(device)
                xb = torch.where(swap[:, None] & col_mask[None, :], xb[perm], xb)
            if opts["prev_wheel_noise_std"] > 0:
                noise = np.zeros((size, len(names)), dtype=np.float32)
                noise[:, prev_cols] = rng.normal(0.0, opts["prev_wheel_noise_std"], (size, 2))
                xb = xb + torch.from_numpy(noise).to(device)
            return xb

        best = math.inf; patience_score = math.inf; stale = 0; best_epoch = 0; history = []; epoch = 0; n = len(tx)
        while stale < hp["patience"] and (hp["max_epochs"] is None or epoch < hp["max_epochs"]):
            epoch += 1; t0 = time.perf_counter(); order = torch.from_numpy(rng.permutation(n)).to(device); total = 0.0
            for s in range(0, n, hp["batch_size"]):
                idx = order[s:s + hp["batch_size"]]
                xb = tx[idx]
                if augment:
                    xb = augmented(xb, len(idx))
                rows = row_objective(xb, idx)
                objective = rows.mean() + (.5 * hp["alpha"] / len(idx)) * sum(
                    l.weight.square().sum() for l in network[::2])
                optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
                total += float(rows.detach().sum()) / 9
            current = score()
            if not math.isfinite(current):
                raise ValueError("Non-finite validation score")
            if current < best:
                best = current; best_epoch = epoch; export().save(output / "model.npz")
            if current < patience_score - hp["min_delta"]:
                patience_score = current; stale = 0
            else:
                stale += 1
            history.append({"epoch": epoch, "training_normalized_loss": total / n,
                            "validation_episode_macro_balanced_nrmse": current, "seconds": time.perf_counter() - t0})
            write_json(output / "history.json", history)
            print(f"Epoch {epoch}: loss={total / n:.6f}, validation NRMSE={current:.6f}, best={best:.6f}", flush=True)
        policy = EgocentricPolicy.load(output / "model.npz")
        # Live path (clock counter) must agree with the batch path used for metrics.
        policy.reset(dt); live = np.stack([policy.predict_action(o) for o in x_check(dataset, split["validation"][0])])
        batch = expand_actions(policy.predict(x_check(dataset, split["validation"][0])))
        np.testing.assert_allclose(live, batch, rtol=1e-6, atol=1e-6)
        policy.reset()
        # Free the training tensors before the final metric pass: on 4096-episode datasets the process otherwise
        # peaks above 100 GB RSS (three OOM kills on 27-29 Oct) while the metrics only need one group at a time.
        del x, y, xn, yn, vx, vy, vxn, tx, ty, tvx
        metrics = {}
        for name, group, ranges_ in [("training", split["training"], None), ("validation", split["validation"], vranges)]:
            feats, truth, ranges_ = load_group(dataset, group, spec)
            pred = policy.predict_features(feats)
            per_episode = [errors(truth[a:b], pred[a:b], y_scale) for a, b in ranges_]
            metrics[name] = {"model": aggregate(per_episode),
                             "mean_baseline": aggregate([errors(truth[a:b], np.broadcast_to(y_mean, (b - a, 9)), y_scale) for a, b in ranges_]),
                             "pooled_transitions": {"model": errors(truth, pred, y_scale)}}
        timings = []
        obs = x_check(dataset, split["validation"][0])[0]
        policy.reset(dt)
        for _ in range(100): policy.predict_action(obs)
        for _ in range(2000):
            start = time.perf_counter_ns(); policy.predict_action(obs); timings.append(time.perf_counter_ns() - start)
        policy.reset()
        if any(sha256(PROJECT_ROOT / p) != d for p, d in source_hashes.items()):
            raise ValueError("Training source changed during execution")
        wheel_loss_text = ("cross-entropy over wheel velocity bins (argmax decode)" if categorical else
                           f"{opts['wheel_loss']} on the two wheel outputs")
        if opts["stop_weight"] != 1.0:
            wheel_loss_text += f", weighted x{opts['stop_weight']:g} on the {stop_rows} training rows with zero demonstrated wheels"
        info = {"kind": KIND, "variant": variant, "backend": "pytorch", "training_device": str(device),
                "hidden_layer_sizes": hp["hidden_layer_sizes"], "activation": "relu",
                "output_activation": "identity" if not categorical else "identity (wheel logits) + argmax bin centre",
                "egocentric_options": opts, "wheel_output": opts["wheel_output"],
                "wheel_bin_centres": None if centres is None else centres.tolist(),
                "stop_rows_in_training": stop_rows, "training_rows": int(n), "trigger_copies": trigger_info,
                "trigger_weight": {"weight": opts["trigger_weight"], "rows_weighted": trigger_rows_weighted} if opts["trigger_weight"] != 1.0 else None,
                "action_chunk": {"steps": chunk, "stride_rows": chunk_stride, "horizon_s": chunk * chunk_stride * spec["dt_s"],
                                 "executed": "first step", "outputs": 9 * chunk} if chunk > 1 else None,
                "episode_filter": episode_filter, "training_episodes": len(split["training"]), "validation_episodes": len(split["validation"]),
                "prev_wheel_augmentation": None if not augment else {"shuffle_p": opts["prev_wheel_shuffle_p"], "noise_std_normalized": opts["prev_wheel_noise_std"]},
                "regression_parameters": sum(p.numel() for p in network.parameters()), "weight_dtype": "float32",
                "input_features": len(names), "input_names": names, "independent_outputs": 9,
                "feature_spec": spec,
                "feature_provenance": ("Deterministic functions of the 21 baseline inputs, the expert standoff table and the controller clock; no expert memory or actions"
                                       if variant not in ACTION_INPUT_VARIANTS else
                                       "Deterministic functions of the 21 baseline inputs, the controller clock and the policy's own previous wheel command (demonstrated command in training); no expert state"),
                "epochs_completed": epoch, "best_epoch": best_epoch,
                "best_validation_episode_macro_balanced_nrmse": best,
                "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
                "checkpoint_selection": "minimum validation episode-macro balanced NRMSE",
                "training_loss": f"{wheel_loss_text}; half-sum MSE on the seven arm outputs; alpha/(2*batch_rows) weight L2 (sklearn convention)",
                "constant_training_inputs": constant_inputs,
                "inference_single_observation_us": {"median": float(np.median(timings) / 1000),
                                                    "p95": float(np.percentile(timings, 95) / 1000), "samples": len(timings)},
                "output_limits": "none applied; the evaluator clips to the declared envelope",
                "fit_and_transform_seconds": time.perf_counter() - started}
        write_json(output / "model_info.json", info); write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", finished_at=utc_now(), wall_s=time.perf_counter() - started,
                        model_sha256=sha256(output / "model.npz"), split_sha256=sha256(output / "split.json"),
                        metrics_sha256=sha256(output / "metrics.json"))
        write_json(output / "run.json", manifest)
        return policy, metrics
    except BaseException as exc:
        manifest.update(status="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", manifest)
        raise


def x_check(dataset, record, rows=256):
    x, _, _ = load_episode(dataset, record)
    return x[:rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/mlp_pytorch.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-s", type=float, default=40.0, help="Clock normalization; the evaluation horizon")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs", default=VARIANT, choices=sorted(VARIANTS),
                        help="egocentric_clock_v1 (25) | raw21_clock_v1 (21 explicit + clock) | raw21_egocentric_clock_v1 (21 + 10 + clock) | ... see VARIANTS")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=JSON",
                        help=f"Override an egocentric_options entry, e.g. --option stop_weight=8 --option wheel_output='\"categorical\"'; keys: {sorted(DEFAULT_OPTIONS)}")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.option:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value  # bare strings such as l1 / categorical
    _, metrics = train(args.dataset, args.split, args.config, args.output_dir, args.horizon_s, args.device, args.inputs, overrides)
    for name, result in metrics.items():
        print(f"{name}: episode-macro balanced NRMSE={result['model']['episode_macro']['balanced_nrmse']:.6f}; "
              f"mean baseline={result['mean_baseline']['episode_macro']['balanced_nrmse']:.6f}")
    return 0
