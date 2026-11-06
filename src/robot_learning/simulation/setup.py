"""Shared setup extracted from the lecturer's Robotnik task, without retuning."""

from dataclasses import dataclass

import numpy as np
from simulation_and_control import PinWrapper, pb


@dataclass
class PhysicsParameters:
    floor_friction: float = 1.0
    contact_stiffness: float = 30000.0
    contact_damping: float = 1000.0
    anisotropic_friction: tuple = (1.0, 0.01, 1.0)
    servo_position_gain: float = 400.0
    servo_velocity_gain: float = 30.0


def create_simulation(roboenv_root, *, headless, physics=None):
    """Load the same robot and apply the starter's physics/gains in the same order."""
    parameters = PhysicsParameters(**physics) if physics is not None else PhysicsParameters()
    sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=str(roboenv_root), use_gui=not headless)
    sim.SetFloorFriction(parameters.floor_friction)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    for j in range(client.getNumJoints(rid)):
        if "wheel" in client.getJointInfo(rid, j)[1].decode():
            client.changeDynamics(rid, j, contactStiffness=parameters.contact_stiffness,
                                  contactDamping=parameters.contact_damping,
                                  anisotropicFriction=list(parameters.anisotropic_friction))
    sim.bot[0].servo_motor_model.set_motor_gains(parameters.servo_position_gain, parameters.servo_velocity_gain)
    return sim


def create_dynamics(sim, roboenv_root, joint_names=None):
    """Create the Pinocchio wrapper and resolve the task's two reference frames."""
    names = list(sim.getNameActiveJoints()) if joint_names is None else list(joint_names)
    dyn = PinWrapper("robotnik_arm.json", "pybullet", np.expand_dims(np.array(names), axis=0),
                     ["pybullet"], False, 0, str(roboenv_root))
    ee_id = dyn.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_id = dyn.pin_model.getFrameId("robot_j2s7s300_link_2")
    return dyn, ee_id, shoulder_id, names
