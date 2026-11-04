"""Input features and model file format of the step-9 learners (Track B), shared by training (obstacle_task/train.py)
and deployment (obstacle_task/policies.py).

Training and the closed-loop policies build their inputs with the same function, build_features(), from the same
signals. The ego part is exactly the A-to-B code (bc_features.measured_signals + build_features of the set ego_novw,
imported, never copied); the lidar part is the 64-beam scan of obstacle_task.lidar:
  - deployment (obs_signals): obs["lidar"], put into every observation by the step-9 harness
    (lidar.scan(base_pos[:2], base_bearing, scenario.obstacles));
  - training (logged_signals): recomputed offline with lidar.scan_batch from the logged base pose and the episode's
    obstacles (the recordings hold no scan). On the same recorded step both give the same numbers (gate B1).

Feature sets (columns in this order):
  arm20        the A-to-B ego_novw set: target in the base frame (ego_dx, ego_dy), target_z, rho, cos/sin of its
               bearing, arm q (7), arm qd (7)
  base70       the 6 target columns of ego_novw + lidar (64): no arm state, no wheel speeds, no clock
  single84     arm20 + lidar (64)
  base70_side  base70 + the episode's side (+1 pass left, -1 right): the side-conditioned base regressor of L2. The
               side is not an observation: training uses the scenario side of the episode, deployment the side
               latched from the side classifier at the first step (policies.SideLatchedPolicy).

Model file (.npz, MODEL_FORMAT): an MLP with ReLU hidden layers and a linear output layer, in sklearn's coefs_ layout.
Inputs are standardised with the stored mean and scale; a regressor's outputs are de-standardised with the stored
mean and scale, a classifier's single output unit is a logit turned into p(classes[1]) = p(side +1, pass left) by the
logistic function (sklearn MLPClassifier, binary case). forward() is the numpy forward used at deployment.
"""
import json
import os
import sys

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import bc_features as bcf  # noqa: E402
from obstacle_task import lidar  # noqa: E402

EGO_SET = "ego_novw"                         # bc_features set of the arm20 features (the A-to-B C3 inputs)
EGO_NAMES = list(bcf.FEATURE_SETS[EGO_SET])  # 20 names
N_TARGET = 6                                 # ego_dx, ego_dy, target_z, rho, cos_beta, sin_beta: ego_novw[:6]
N_BEAMS = lidar.N_BEAMS                      # 64
LIDAR_NAMES = [f"lidar{k:02d}" for k in range(N_BEAMS)]
SIDE_NAME = "side"
FEATURE_SETS = {
    "arm20": EGO_NAMES,
    "base70": EGO_NAMES[:N_TARGET] + LIDAR_NAMES,
    "single84": EGO_NAMES + LIDAR_NAMES,
    "base70_side": EGO_NAMES[:N_TARGET] + LIDAR_NAMES + [SIDE_NAME],
}
LIDAR_SETS = ("base70", "single84", "base70_side")   # need the signal lidar
SIDE_SETS = ("base70_side",)                         # need the side input

# Output groups: column names (regressors) and the classifier's single output (probability of pass left)
OUTPUTS = {
    "all": list(bcf.OUTPUT_NAMES),   # v, w, tau1..tau7 (L0)
    "vw": list(bcf.OUTPUT_NAMES[:2]),
    "tau": list(bcf.OUTPUT_NAMES[2:]),
    "side": ["p_left"],
}
CLASSES = (-1, 1)                    # side labels; sklearn sorts them, so column 1 of predict_proba = p(+1, left)
LIDAR_BLOCK = 2048                   # rows per lidar.scan_batch call offline: its (N, 64) temporaries stay in cache
                                     # (~2x faster than one call on 65k rows; row-wise, so the values are identical)

MODEL_FORMAT = "obstacle_mlp_v1"
KINDS = ("regressor", "classifier")


# ==============================================================================
# Signals and features
# ==============================================================================
def logged_lidar(base_xy, yaw, obstacles, episode=None):
    """Training-time lidar (N, N_BEAMS) of recorded base poses, by lidar.scan_batch.

    obstacles: the obstacle list of one episode (all rows from it), or a dict episode id -> obstacle list together
    with `episode` (N,), the episode of each row. scan_batch is called once per run of equal ids (in blocks of at
    most LIDAR_BLOCK rows), so rows of an episode should be contiguous (any order is correct, contiguous is fast)."""
    base_xy = np.atleast_2d(np.asarray(base_xy, dtype=float))
    yaw = np.atleast_1d(np.asarray(yaw, dtype=float))
    n = len(yaw)
    out = np.empty((n, N_BEAMS))
    if n == 0:
        return out
    if episode is None:
        runs = [(0, n, obstacles)]
    else:
        episode = np.asarray(episode)
        cut = np.flatnonzero(episode[1:] != episode[:-1]) + 1
        runs = [(a, b, obstacles[int(episode[a])]) for a, b in zip(np.r_[0, cut], np.r_[cut, n])]
    for a, b, obs in runs:
        for s in range(a, b, LIDAR_BLOCK):
            e = min(b, s + LIDAR_BLOCK)
            out[s:e] = lidar.scan_batch(base_xy[s:e], yaw[s:e], obs)
    return out


def logged_signals(base_pos, base_bearing, q_mes, qd_mes, target, obstacles=None, episode=None):
    """Training: signals of recorded steps (arrays with a leading step axis, the recorded names and layouts:
    base_pos (N, 3), base_bearing (N,), q_mes / qd_mes (N, 17), target (3,) or (N, 3)). The ego signals are
    bc_features.measured_signals; with `obstacles` (see logged_lidar) the signal lidar is added."""
    sig = bcf.measured_signals(base_pos, base_bearing, q_mes, qd_mes, target)
    if obstacles is not None:
        sig["lidar"] = logged_lidar(sig["base_xy"], sig["yaw"], obstacles, episode)
    return sig


def obs_signals(obs):
    """Deployment: the signals of one observation of the step-9 harness (N = 1); the lidar is obs['lidar']."""
    sig = bcf.measured_signals(obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"], obs["target"])
    scan = np.asarray(obs["lidar"], dtype=float)
    if scan.shape != (N_BEAMS,):
        raise ValueError(f"obs['lidar'] has shape {scan.shape}, expected ({N_BEAMS},)")
    sig["lidar"] = scan.reshape(1, N_BEAMS)
    return sig


def build_features(feature_set, signals, side=None):
    """(N, F) float64 feature matrix of one of FEATURE_SETS, columns in the order of FEATURE_SETS[feature_set].

    The ego columns are bc_features.build_features('ego_novw', signals) (its first N_TARGET columns for the base
    sets); side ((N,) or scalar, +-1) is required by, and only accepted for, the SIDE_SETS."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {list(FEATURE_SETS)}")
    if (side is not None) != (feature_set in SIDE_SETS):
        raise ValueError(f"feature set {feature_set}: side is required by, and only by, {SIDE_SETS}")
    ego = bcf.build_features(EGO_SET, signals)
    if feature_set == "arm20":
        return ego
    if "lidar" not in signals:
        raise ValueError(f"feature set {feature_set} needs the signal lidar")
    scan = np.asarray(signals["lidar"], dtype=float)
    if scan.shape != (ego.shape[0], N_BEAMS):
        raise ValueError(f"lidar signal has shape {scan.shape}, expected ({ego.shape[0]}, {N_BEAMS})")
    if feature_set == "single84":
        return np.concatenate([ego, scan], axis=1)
    parts = [ego[:, :N_TARGET], scan]
    if feature_set in SIDE_SETS:
        s = np.broadcast_to(np.asarray(side, dtype=float).reshape(-1), (ego.shape[0],))
        if not np.all(np.abs(s) == 1.0):
            raise ValueError("side must be +1 or -1")
        parts.append(s[:, None])
    return np.concatenate(parts, axis=1)


# ==============================================================================
# Model file and numpy forward
# ==============================================================================
def save_model(path, kind, feature_set, output_names, x_mean, x_scale, y_mean, y_scale, weights, biases, meta):
    """Write a model file. weights[k] has shape (n_in, n_out) (sklearn's coefs_); ReLU after every layer but the
    last. A classifier stores y_mean 0, y_scale 1 (its output is a logit) and output_names ['p_left']."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    arrays = {f"W{k}": np.asarray(w, dtype=float) for k, w in enumerate(weights)}
    arrays.update({f"b{k}": np.asarray(b, dtype=float) for k, b in enumerate(biases)})
    np.savez(path, format=MODEL_FORMAT, kind=kind, feature_set=feature_set,
             feature_names=np.array(FEATURE_SETS[feature_set]), output_names=np.array(output_names),
             classes=np.array(CLASSES if kind == "classifier" else [], dtype=np.int64), n_layers=len(weights),
             x_mean=np.asarray(x_mean, dtype=float), x_scale=np.asarray(x_scale, dtype=float),
             y_mean=np.asarray(y_mean, dtype=float), y_scale=np.asarray(y_scale, dtype=float),
             meta=json.dumps(meta), **arrays)


def load_model(path):
    """Model file -> dict with kind, feature_set, output_names, classes, x_mean, x_scale, y_mean, y_scale, weights,
    biases, meta (checked against FEATURE_SETS and for consistent dimensions)."""
    with np.load(path, allow_pickle=False) as f:
        if str(f["format"]) != MODEL_FORMAT:
            raise ValueError(f"{path}: model format {str(f['format'])!r}, expected {MODEL_FORMAT!r}")
        kind, feature_set = str(f["kind"]), str(f["feature_set"])
        if kind not in KINDS or feature_set not in FEATURE_SETS:
            raise ValueError(f"{path}: unknown kind {kind!r} or feature set {feature_set!r}")
        if list(f["feature_names"]) != FEATURE_SETS[feature_set]:
            raise ValueError(f"{path}: feature names differ from obstacle_task/features.py")
        n = int(f["n_layers"])
        model = {"kind": kind, "feature_set": feature_set, "output_names": [str(s) for s in f["output_names"]],
                 "classes": [int(c) for c in f["classes"]],
                 "x_mean": f["x_mean"], "x_scale": f["x_scale"], "y_mean": f["y_mean"], "y_scale": f["y_scale"],
                 "weights": [f[f"W{k}"] for k in range(n)], "biases": [f[f"b{k}"] for k in range(n)],
                 "meta": json.loads(str(f["meta"]))}
    n_in, n_out = model["weights"][0].shape[0], model["weights"][-1].shape[1]
    if n_in != len(FEATURE_SETS[feature_set]) or not (len(model["x_mean"]) == len(model["x_scale"]) == n_in):
        raise ValueError(f"{path}: input dimensions differ from the feature set")
    if not (len(model["y_mean"]) == len(model["y_scale"]) == len(model["output_names"]) == n_out):
        raise ValueError(f"{path}: output dimensions differ")
    if kind == "classifier" and (tuple(model["classes"]) != CLASSES or n_out != 1):
        raise ValueError(f"{path}: a side classifier has one output unit and classes {CLASSES}")
    return model


def forward(model, X):
    """Numpy forward (N, n_out): a regressor's outputs in physical units, a classifier's p(classes[1]) (= p_left)."""
    out = bcf.forward(model, X)   # standardise, ReLU hidden layers, linear output, de-standardise
    if model["kind"] == "classifier":
        from scipy.special import expit   # sklearn's logistic output activation
        return expit(out)
    return out
