"""Input features, outputs and model file format shared by training (train_policy.py) and deployment (policies/).

Training and the closed-loop policy build their inputs with the same function, build_features(), from the same
measured signals (the names of the recorded datasets and of the policy's obs). Every feature is a pure transform of
the current observation: no time, no expert rule, no start pose.

Feature sets:
  ego       (22) target in the current base frame (dx, dy), target z, planar distance rho, cos/sin of its bearing
                 beta in the base frame, arm q (7), arm qd (7), measured base speeds (v_mes, w_mes)
  ego_novw  (20) ego without (v_mes, w_mes)
  raw       (23) base x, y, sin/cos yaw, arm q (7), arm qd (7), (v_mes, w_mes), target x, y, z (world frame)
  ego_novw_ee (23) ego_novw plus the near-field end-effector error (experiment D): e = target - ee_pos (world frame,
                 ee_pos = forward kinematics of the full base pose and q), clipped to length EE_CLIP
                 (e * min(1, EE_CLIP / |e|)), its horizontal part rotated into the current base frame (ee_dx, ee_dy)
                 and its z part (ee_dz). Needs the signal ee_pos: in training the recorded ee_pos, in deployment the
                 forward kinematics with the robot model of info (policies/bc_policy.py). Unlike the other sets it
                 uses the robot's kinematic model.
(v_mes, w_mes) are the measured wheel speeds converted with the skid-steer model of base_controller.

Outputs (9): base (v, w) and the 7 arm torques. The policy converts (v, w) back to the 4 wheel speeds with
collect_data.wheel_speeds.

Model file (.npz): an MLP with ReLU hidden layers and a linear output layer; a linear model is the same with no hidden
layer. Inputs and outputs are standardised with the training-set mean and scale stored in the file.
"""
import json

import numpy as np

WHEEL_RADIUS = 0.11       # m, BaseControllerParams.wheel_radius
WHEEL_BASE_WIDTH = 0.66   # m, BaseControllerParams.wheel_base_width

ARM = slice(4, 11)        # arm joints in q_mes / qd_mes (wheels 0-3, fingers 11-16)
WHEELS = slice(0, 4)

_Q = [f"q{j}" for j in range(1, 8)]
_QD = [f"qd{j}" for j in range(1, 8)]
FEATURE_SETS = {
    "ego": ["ego_dx", "ego_dy", "target_z", "rho", "cos_beta", "sin_beta", *_Q, *_QD, "v_mes", "w_mes"],
    "ego_novw": ["ego_dx", "ego_dy", "target_z", "rho", "cos_beta", "sin_beta", *_Q, *_QD],
    "raw": ["base_x", "base_y", "sin_yaw", "cos_yaw", *_Q, *_QD, "v_mes", "w_mes", "target_x", "target_y", "target_z"],
    "ego_novw_ee": ["ego_dx", "ego_dy", "target_z", "rho", "cos_beta", "sin_beta", *_Q, *_QD, "ee_dx", "ee_dy", "ee_dz"],
}
FEATURES_NEED_EE = ("ego_novw_ee",)   # feature sets built from the end-effector position (signal ee_pos)
EE_CLIP = 0.10            # m, length at which the end-effector error is clipped (near-field feature)
OUTPUT_NAMES = ["v", "w", *[f"tau{j}" for j in range(1, 8)]]
FORMAT_VERSION = 1


def wheels_to_vw(wheels):
    """Wheel speeds [front_right, front_left, back_left, back_right] (rad/s), shape (..., 4) -> (v, w).

    Inverse of the skid-steer model of base_controller (v = R (rw + lw) / 2, w = R (rw - lw) / W). The two wheels of
    a side are averaged, so measured speeds that differ slightly between front and back are handled too. Exact for
    the expert's labels, which are [rw, lw, lw, rw]."""
    wheels = np.asarray(wheels, dtype=float)
    rw = 0.5 * (wheels[..., 0] + wheels[..., 3])
    lw = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return WHEEL_RADIUS * (rw + lw) / 2, WHEEL_RADIUS * (rw - lw) / WHEEL_BASE_WIDTH


def measured_signals(base_pos, base_bearing, q_mes, qd_mes, target, ee_pos=None):
    """Measured signals the features are built from, as float64 arrays with a leading sample axis.

    Accepts one observation (base_pos (3,), base_bearing scalar, q_mes / qd_mes (17,), target (3,)) or a recorded
    episode (base_pos (T, 3), base_bearing (T,), q_mes / qd_mes (T, 17), target (3,) or (T, 3)). ee_pos (3,) or
    (T, 3): the end-effector position (world frame), only needed by FEATURES_NEED_EE; without it the signal is
    absent."""
    base_pos = np.atleast_2d(np.asarray(base_pos, dtype=float))
    n = base_pos.shape[0]
    qd_mes = np.atleast_2d(np.asarray(qd_mes, dtype=float))
    v_mes, w_mes = wheels_to_vw(qd_mes[:, WHEELS])
    return {
        "base_xy": base_pos[:, :2],
        "yaw": np.atleast_1d(np.asarray(base_bearing, dtype=float)).reshape(n),
        "q_arm": np.atleast_2d(np.asarray(q_mes, dtype=float))[:, ARM],
        "qd_arm": qd_mes[:, ARM],
        "v_mes": v_mes,
        "w_mes": w_mes,
        "target": np.broadcast_to(np.asarray(target, dtype=float), (n, 3)),
        **({} if ee_pos is None else {"ee_pos": np.atleast_2d(np.asarray(ee_pos, dtype=float)).reshape(n, 3)}),
    }


def ego_target(base_xy, yaw, target_xy):
    """Target position in the current base frame (x forward, y left): (dx, dy), each (N,)."""
    delta = np.asarray(target_xy, dtype=float) - np.asarray(base_xy, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]


def clip_ee_error(e):
    """Near-field end-effector error: e * min(1, EE_CLIP / |e|) row by row ((N, 3); a zero vector stays zero)."""
    e = np.asarray(e, dtype=float)
    norm = np.linalg.norm(e, axis=1, keepdims=True)
    return e * np.minimum(1.0, EE_CLIP / np.maximum(norm, 1e-300))


def build_features(feature_set, signals):
    """(N, F) float64 feature matrix of one of FEATURE_SETS, columns in the order of FEATURE_SETS[feature_set]."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {list(FEATURE_SETS)}")
    s = signals
    vw = [s["v_mes"][:, None], s["w_mes"][:, None]]
    if feature_set in ("ego", "ego_novw", "ego_novw_ee"):
        dx, dy = ego_target(s["base_xy"], s["yaw"], s["target"][:, :2])
        beta = np.arctan2(dy, dx)
        parts = [dx[:, None], dy[:, None], s["target"][:, 2:3], np.hypot(dx, dy)[:, None],
                 np.cos(beta)[:, None], np.sin(beta)[:, None], s["q_arm"], s["qd_arm"]]
        if feature_set == "ego":
            parts += vw
        if feature_set == "ego_novw_ee":
            if "ee_pos" not in s:
                raise ValueError("feature set ego_novw_ee needs the signal ee_pos (end-effector position)")
            e = clip_ee_error(s["target"] - s["ee_pos"])
            c, sn = np.cos(s["yaw"]), np.sin(s["yaw"])
            parts += [(c * e[:, 0] + sn * e[:, 1])[:, None], (-sn * e[:, 0] + c * e[:, 1])[:, None], e[:, 2:3]]
    else:
        parts = [s["base_xy"], np.sin(s["yaw"])[:, None], np.cos(s["yaw"])[:, None], s["q_arm"], s["qd_arm"],
                 *vw, s["target"]]
    return np.concatenate(parts, axis=1)


def save_model(path, feature_set, x_mean, x_scale, y_mean, y_scale, weights, biases, meta):
    """Write a model file. weights[k] has shape (n_in, n_out) (sklearn's coefs_ layout); ReLU after every layer but
    the last. A linear model has one layer."""
    arrays = {f"W{k}": np.asarray(w, dtype=float) for k, w in enumerate(weights)}
    arrays.update({f"b{k}": np.asarray(b, dtype=float) for k, b in enumerate(biases)})
    np.savez(path, format_version=FORMAT_VERSION, feature_set=feature_set,
             feature_names=np.array(FEATURE_SETS[feature_set]), output_names=np.array(OUTPUT_NAMES),
             n_layers=len(weights), x_mean=np.asarray(x_mean, dtype=float), x_scale=np.asarray(x_scale, dtype=float),
             y_mean=np.asarray(y_mean, dtype=float), y_scale=np.asarray(y_scale, dtype=float),
             meta=json.dumps(meta), **arrays)


def load_model(path):
    """Model file -> dict with feature_set, x_mean, x_scale, y_mean, y_scale, weights, biases, meta."""
    with np.load(path, allow_pickle=False) as f:
        if int(f["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"{path}: model format {int(f['format_version'])}, expected {FORMAT_VERSION}")
        feature_set = str(f["feature_set"])
        if list(f["feature_names"]) != FEATURE_SETS[feature_set] or list(f["output_names"]) != OUTPUT_NAMES:
            raise ValueError(f"{path}: feature or output names differ from bc_features.py")
        n = int(f["n_layers"])
        return {"feature_set": feature_set, "x_mean": f["x_mean"], "x_scale": f["x_scale"],
                "y_mean": f["y_mean"], "y_scale": f["y_scale"],
                "weights": [f[f"W{k}"] for k in range(n)], "biases": [f[f"b{k}"] for k in range(n)],
                "meta": json.loads(str(f["meta"]))}


def forward(model, X):
    """Model outputs in physical units, (N, 9): standardise, hidden ReLU layers, linear output, de-standardise."""
    h = (np.asarray(X, dtype=float) - model["x_mean"]) / model["x_scale"]
    last = len(model["weights"]) - 1
    for k, (w, b) in enumerate(zip(model["weights"], model["biases"])):
        h = h @ w + b
        if k < last:
            np.maximum(h, 0.0, out=h)
    return h * model["y_scale"] + model["y_mean"]
