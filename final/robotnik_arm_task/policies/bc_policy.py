"""Behaviour-cloning policy: a model file written by bc_features.save_model, run in closed loop by evaluate_policy.py.

  python evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg trained/ego_s3.npz \
      --eval-set data/pilot --workers 4 --out-dir results/ego_s3_pilot

At every step the policy builds the model's features from the observation with exactly the functions used in
training (bc_features.measured_signals + build_features, feature set stored in the model file), predicts the
9 outputs with bc_features.forward, and converts the base command (v, w) to the 4 wheel speeds with
collect_data.wheel_speeds and BaseControllerParams(nav_timeout=25.0) (the expert's skid-steer model).

Inputs: only obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"] and obs["target"]. For the feature
sets of bc_features.FEATURES_NEED_EE (ego_novw_ee, experiment D) also obs["base_ori"] and, from info at reset, the
robot's kinematic model (info["dyn_model"], info["ee_frame_id"]): the end-effector position is computed at every step
with exactly the expert's forward-kinematics call (evaluate_policy.ExpertPolicy.act). The policy uses nothing else from
info (no start pose, no time step, no expert goal) and not obs["t"], so it is a pure function of the current
observation: no internal state, no clock.
"""
import os
import sys

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:  # evaluate_policy.load_policy only adds policies/ to sys.path
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

import bc_features  # noqa: E402
from collect_data import wheel_speeds  # noqa: E402
from robotnik_arm_control import BaseControllerParams  # noqa: E402

# Skid-steer model used to turn the predicted (v, w) into wheel speeds (EVALUATION.md, policy interface)
BASE_PARAMS = BaseControllerParams(nav_timeout=25.0)


class BCPolicy:
    """A learned model mapping the current observation to (v, w, tau1..tau7). Stateless."""

    def __init__(self, model_path):
        if not model_path:
            raise ValueError("BCPolicy needs a model file: --policy-arg path/to/model.npz")
        self.model_path = os.path.abspath(model_path)
        self.model = bc_features.load_model(self.model_path)  # loaded once per process
        self.feature_set = self.model["feature_set"]
        self.base_params = BASE_PARAMS
        self.needs_ee = self.feature_set in bc_features.FEATURES_NEED_EE
        self.kin_model = self.ee_frame_id = None

    def reset(self, target, info):
        """The target is read from obs at every step. info is used only by the feature sets that need the end-effector
        position: the robot model and the end-effector frame (refused when missing; never replaced by zeros)."""
        if self.needs_ee:
            if not info or info.get("dyn_model") is None or info.get("ee_frame_id") is None:
                raise ValueError(f"feature set {self.feature_set} needs info['dyn_model'] and info['ee_frame_id'] "
                                 f"(the robot's kinematic model) at reset")
            self.kin_model, self.ee_frame_id = info["dyn_model"], info["ee_frame_id"]

    def ee_position(self, obs):
        """End-effector position (world frame) by forward kinematics of the full base pose and q_mes: the same calls
        as evaluate_policy.ExpertPolicy.act and collect_data (the recorded ee_pos)."""
        import pinocchio as pin
        if self.kin_model is None:
            raise ValueError(f"feature set {self.feature_set} needs reset(target, info) with the robot model first")
        m = self.kin_model
        q_pin = m.ReoderJoints2PinVec(np.concatenate((obs["base_pos"], obs["base_ori"], obs["q_mes"])), "pos")
        pin.forwardKinematics(m.pin_model, m.pin_data, q_pin)
        pin.updateFramePlacements(m.pin_model, m.pin_data)
        return m.pin_data.oMf[self.ee_frame_id].translation.copy()

    def predict(self, obs):
        """Model outputs (9,) in physical units for one observation: v (m/s), w (rad/s), tau1..tau7 (N m)."""
        ee_pos = self.ee_position(obs) if self.needs_ee else None
        signals = bc_features.measured_signals(obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"],
                                               obs["target"], ee_pos=ee_pos)
        return bc_features.forward(self.model, bc_features.build_features(self.feature_set, signals))[0]

    def act(self, obs):
        y = self.predict(obs)
        return wheel_speeds(y[0], y[1], self.base_params), y[2:9]


class ChunkPolicy(BCPolicy):
    """Experiment G: query every step and ensemble exactly aligned chunks.

    Query t-j*delta contributes segment j to step t. The oldest available
    query receives the highest weight, as fixed in the preregistration.
    """

    def __init__(self, model_path):
        super().__init__(model_path)
        meta = self.model["meta"]
        self.chunk_k = int(meta.get("chunk_k", 1))
        self.chunk_delta = int(meta.get("chunk_delta", 1))
        self.ensemble_m = float(meta.get("ensemble_m", 0.11))
        if self.chunk_k < 1 or self.chunk_delta < 1 or self.ensemble_m != 0.11:
            raise ValueError("invalid chunk metadata (k/delta positive, ensemble_m fixed at 0.11)")
        if len(self.model["y_mean"]) != 9 * self.chunk_k:
            raise ValueError("chunk output width must be 9 * chunk_k")
        self._buffer_size = (self.chunk_k - 1) * self.chunk_delta + 1
        self._chunks = np.empty((self._buffer_size, self.chunk_k, 9), dtype=np.float64)
        self._step = 0

    def reset(self, target, info):
        super().reset(target, info)
        self._step = 0

    def ensemble(self, prediction):
        """Advance one 1 kHz query (clock is call order, never obs['t'])."""
        t = self._step
        self._chunks[t % self._buffer_size] = np.asarray(prediction).reshape(self.chunk_k, 9)
        # Reverse segment order: i=0 denotes the oldest available prediction.
        js = np.arange(min(self.chunk_k - 1, t // self.chunk_delta), -1, -1)
        weights = np.exp(-self.ensemble_m * np.arange(len(js)))
        values = self._chunks[(t - js * self.chunk_delta) % self._buffer_size, js]
        y = np.sum(values * weights[:, None], axis=0) / weights.sum()
        self._step += 1
        return y

    def act(self, obs):
        y = self.ensemble(self.predict(obs))
        return wheel_speeds(y[0], y[1], self.base_params), y[2:9]


class PDPolicy(BCPolicy):
    """Experiment P: learned (v,w,e[7],qd_des[7]), fixed expert output law."""

    def reset(self, target, info):
        super().reset(target, info)
        required = ("dyn_model", "joint_lower", "joint_upper")
        if not info or any(info.get(k) is None for k in required):
            raise ValueError("PDPolicy needs dyn_model and joint_lower/upper at reset")
        if len(self.model["y_mean"]) != 16:
            raise ValueError("PDPolicy output width must be 16")
        self.pd_info = info
        from policies.pd_layer import expert_arm_params
        self.arm_params = expert_arm_params()

    def act(self, obs):
        from policies.pd_layer import pd_layer, q_pin_from_obs
        if not hasattr(self, "pd_info"):
            raise ValueError("PDPolicy needs reset(target, info) before act")
        y = self.predict(obs)
        tau = pd_layer(y[2:9], y[9:16], obs["q_mes"][4:11], obs["qd_mes"][4:11],
                       q_pin_from_obs(obs, self.pd_info), self.pd_info, self.arm_params)
        return wheel_speeds(y[0], y[1], self.base_params), tau


def make_policy(arg):
    """arg: path to a model .npz written by bc_features.save_model (the --policy-arg string)."""
    policy = BCPolicy(arg)
    representation = policy.model["meta"].get("output_representation")
    if representation == "chunk":
        return ChunkPolicy(arg)
    if representation == "pd_reference":
        return PDPolicy(arg)
    return policy
