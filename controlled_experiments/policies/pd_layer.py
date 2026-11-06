"""Experiment P's fixed expert joint PD law and Pinocchio gravity compensation.

No learned state, simulator truth or expert target is used: only the current
observation and robot model supplied by the evaluation interface.
"""
import numpy as np
import pinocchio as pin

from robotnik_arm_control import ArmControllerParams
from robotnik_arm_task import ARM_REACH_LIMIT


def expert_arm_params():
    """The same parameter construction used by evaluate_policy.ExpertPolicy."""
    return ArmControllerParams(reach_limit=ARM_REACH_LIMIT)


def q_pin_from_obs(obs, info):
    """Full floating-base position vector, exactly as BCPolicy.ee_position."""
    return info["dyn_model"].ReoderJoints2PinVec(
        np.concatenate((obs["base_pos"], obs["base_ori"], obs["q_mes"])), "pos")


def pd_layer(e_hat, qd_des_hat, q_arm, qd_arm, q_pin, info, params=None):
    """Convert seven position leads and seven reference velocities to torque.

    The reference is already the output of the expert's reference generator.
    Do not clip it again: the expert clips references only in its reach branch,
    and the joint-limit clip there can itself exceed the lead limit. Only the
    final torque saturation follows the expert PD/gravity arithmetic.
    """
    params = expert_arm_params() if params is None else params
    q_arm = np.asarray(q_arm, dtype=np.float64)
    qd_arm = np.asarray(qd_arm, dtype=np.float64)
    e_hat = np.asarray(e_hat, dtype=np.float64)
    qd_des_hat = np.asarray(qd_des_hat, dtype=np.float64)
    if any(a.shape != (7,) for a in (q_arm, qd_arm, e_hat, qd_des_hat)):
        raise ValueError("pd_layer expects four seven-joint vectors")
    torque = params.kp * e_hat + params.kd * (qd_des_hat - qd_arm)
    model = info["dyn_model"]
    tau_g = pin.computeGeneralizedGravity(model.pin_model, model.pin_data, q_pin)
    return np.clip(torque + tau_g[6:13], -params.max_torque, params.max_torque)
