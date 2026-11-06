"""Adapter: Rodrigo's PyTorch-trained v2 policies (branch `rodrigo`, weights/v2/*/model.npz) run under this task's
closed-loop harness (evaluate_policy.py, protocol v1.1), so that both lines of the group can be scored on the same
episodes with the same success definition.

  python evaluate_policy.py --policy policies/rodrigo_policy.py:make_policy \
      --policy-arg trained/rodrigo_v2/v2_mlp28_trig144_seed0/model.npz --eval-set data/dev --no-early-stop ...

Two model kinds are supported, both with NumPy inference re-implemented here from his
`src/robot_learning/learning/egocentric.py` (EgocentricPolicy) and `gru_policy.py` (GRUPolicy), commit in
trained/rodrigo_v2/SOURCE_COMMIT.txt:
  * MLP  (keys weight_i / bias_i): ReLU MLP, float32 weights, input variant raw21_targetpolar_clock_v1 (28 features).
  * GRU  (keys w_ih / w_hh / ...): memory updated every `stride` steps (20 = 20 ms), head reacts every step.
Features (his `encode`, variant raw21_targetpolar_clock_v1): the 21 raw observations in his order
(base x, y, z, bearing, arm q1-7, arm qd1-7, target x, y, z), the target in the base frame (x, y) and its height above
the base, the planar distance to the target, cos / sin of the heading error to the target, and the episode clock
t / horizon (t counted by the policy from reset, 1 ms per call, as in his predict_action). Outputs: (wheel_right,
wheel_left, tau1..tau7) -> wheels [fr, fl, bl, br] = [r, l, l, r]; commands are clipped to his evaluator's declared
envelope: wheels +-(max_lin_vel + max_ang_vel * wheel_base_width / 2) / wheel_radius = +-5.27 rad/s, torques +-30 N m
(worker.action_limits / make_action_filter; no dead band).

The policy reads obs["base_pos"], obs["base_bearing"], obs["q_mes"][4:11], obs["qd_mes"][4:11], obs["target"] and
info["time_step"] (to check the clock); nothing else.
"""
import os
import sys

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

from robotnik_arm_control import ArmControllerParams, BaseControllerParams  # noqa: E402

_BP = BaseControllerParams(nav_timeout=25.0)
WHEEL_LIMIT = (_BP.max_lin_vel + _BP.max_ang_vel * _BP.wheel_base_width / 2.0) / _BP.wheel_radius  # 5.27 rad/s
TORQUE_LIMIT = ArmControllerParams.max_torque  # 30 N m (class default)
VARIANT = "raw21_targetpolar_clock_v1"


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def encode_raw21_targetpolar_clock(x, t, horizon_s):
    """His egocentric.encode for the 28-feature variant, one row. x = the 21 raw observations in his order."""
    base_xy, base_z, yaw = x[0:2], x[2], x[3]
    target = x[18:21]
    c, s = np.cos(yaw), np.sin(yaw)
    d = target[:2] - base_xy
    target_b = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    target_distance = np.hypot(target_b[0], target_b[1])
    heading_to_target = np.arctan2(target_b[1], target_b[0])
    return np.concatenate([x, target_b, [target[2] - base_z], [target_distance, np.cos(heading_to_target),
                                                                 np.sin(heading_to_target)], [t / horizon_s]])


class RodrigoPolicy:
    def __init__(self, model_path):
        if not model_path:
            raise ValueError("RodrigoPolicy needs --policy-arg path/to/model.npz")
        self.model_path = os.path.abspath(model_path)
        with np.load(self.model_path, allow_pickle=False) as d:
            files = set(d.files)
            self.x_mean, self.x_scale = d["x_mean"].astype(np.float64), d["x_scale"].astype(np.float64)
            self.y_mean, self.y_scale = d["y_mean"].astype(np.float64), d["y_scale"].astype(np.float64)
            clock = d["clock"].astype(np.float64)
            self.horizon_s, self.dt_s = float(clock[0]), float(clock[1])
            if "weight_0" in files:
                self.kind = "mlp"
                variant = str(d["variant"])
                n = sum(k.startswith("weight_") for k in files)
                self.weights = [np.asarray(d[f"weight_{i}"]) for i in range(n)]   # keep float32, as his inference does
                self.biases = [np.asarray(d[f"bias_{i}"]) for i in range(n)]
                if "wheel_bin_centres" in files or "action_chunk" in files:
                    raise ValueError("categorical-wheel / chunk models are not supported by this adapter")
                self.stride = 1
            elif "w_ih" in files:
                self.kind = "gru"
                variant = str(d["feature_mode"])
                f64 = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
                self.w_ih, self.w_hh, self.b_ih, self.b_hh = f64("w_ih"), f64("w_hh"), f64("b_ih"), f64("b_hh")
                self.w1, self.b1, self.w2, self.b2 = f64("w1"), f64("b1"), f64("w2"), f64("b2")
                self.hidden = self.w_hh.shape[1]
                self.stride = int(clock[2])
            else:
                raise ValueError(f"unknown model layout in {model_path}: {sorted(files)}")
        if variant != VARIANT:
            raise ValueError(f"only the {VARIANT} input variant is implemented, got {variant!r}")
        if self.x_mean.shape != (28,) or self.y_mean.shape != (9,):
            raise ValueError("unexpected normalisation shapes")
        self._count = 0
        self._h = None

    # --- interface of evaluate_policy.py -------------------------------------------------------------------------
    def reset(self, target, info):
        dt = float(info["time_step"]) if info and "time_step" in info else self.dt_s
        if not np.isclose(dt, self.dt_s, rtol=0, atol=1e-12):
            raise ValueError(f"policy clock dt {self.dt_s} differs from simulator dt {dt}")
        self._count = 0
        self._h = np.zeros(self.hidden) if self.kind == "gru" else None

    def raw21(self, obs):
        return np.concatenate([np.asarray(obs["base_pos"], dtype=np.float64)[:3], [float(obs["base_bearing"])],
                               np.asarray(obs["q_mes"], dtype=np.float64)[4:11],
                               np.asarray(obs["qd_mes"], dtype=np.float64)[4:11],
                               np.asarray(obs["target"], dtype=np.float64)[:3]])

    def predict9(self, obs):
        """Nine controls in physical units (wheel_right, wheel_left, tau1..tau7), before clipping."""
        t = self._count * self.dt_s
        f = (encode_raw21_targetpolar_clock(self.raw21(obs), t, self.horizon_s) - self.x_mean) / self.x_scale
        if self.kind == "mlp":
            h = f.astype(self.weights[0].dtype)
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                h = h @ w + b
                if i < len(self.weights) - 1:
                    np.maximum(h, 0, out=h)
            out = h.astype(np.float64) * self.y_scale + self.y_mean
        else:
            if self._count % self.stride == 0:
                self._h = _gru_step(f, self._h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
            hidden = np.maximum(self.w1 @ np.concatenate([self._h, f]) + self.b1, 0.0)
            out = (self.w2 @ hidden + self.b2) * self.y_scale + self.y_mean
        self._count += 1
        return out

    def act(self, obs):
        y = self.predict9(obs)
        wheels = np.clip(y[[0, 1, 1, 0]], -WHEEL_LIMIT, WHEEL_LIMIT)
        torques = np.clip(y[2:9], -TORQUE_LIMIT, TORQUE_LIMIT)
        return wheels, torques


def _gru_step(x, h, w_ih, w_hh, b_ih, b_hh):
    gi = w_ih @ x + b_ih
    gh = w_hh @ h + b_hh
    H = len(h)
    r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
    z = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return (1.0 - z) * n + z * h


def make_policy(arg):
    return RodrigoPolicy(arg)
