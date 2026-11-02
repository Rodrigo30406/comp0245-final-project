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


def make_policy(arg):
    """arg: path to a model .npz written by bc_features.save_model (the --policy-arg string)."""
    return BCPolicy(arg)
