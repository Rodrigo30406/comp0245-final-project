"""The lecturer's episode sequence, shared by interactive and recorded runs.

Observe -> expert -> physics -> GUI exit -> stop on the PRE-step measurement.
Logging the post-step state does not change that original decision order.
"""

import time

import numpy as np
import pinocchio as pin
import pybullet as p
from simulation_and_control import MotorCommands

from robot_learning.expert.controllers import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)
from .config import episode_defaults, resolve_episode_config, vector
from .setup import create_dynamics, create_simulation


def base_goal(target, reference_xy, standoff_table):
    """The expert's base goal: standoff(z) behind the target along the bearing from `reference_xy` to the target.
    The lecturer's code takes the bearing from the world origin, where his robot always starts (goal_reference
    "origin"); "start" uses the base start position instead, which is the same point for the nominal start and keeps
    the goal on the near side of the standoff ring for any other start. Returns (goal_bearing, goal_xy)."""
    target = np.asarray(target, dtype=float); reference_xy = np.asarray(reference_xy, dtype=float)
    standoff = float(np.interp(target[2], standoff_table["heights"], standoff_table["distances"]))
    rel = target[:2] - reference_xy
    bearing = float(np.arctan2(rel[1], rel[0])) if np.linalg.norm(rel) > 1e-4 else 0.0
    return bearing, target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])


class Episode:
    """One fresh simulator and controller state; use as a context manager."""

    def __init__(self, target, config, roboenv_root, initial_base=None, *,
                 policy=None, horizon_steps=None, action_filter=None):
        vector(target, 3, "target")
        keys = episode_defaults().keys()
        self.config = resolve_episode_config({k: v for k, v in config.items() if k in keys})
        self.target = np.asarray(target, dtype=float)
        self.roboenv_root = roboenv_root
        self.initial_base = initial_base
        if horizon_steps is not None and (type(horizon_steps) is not int or horizon_steps < 1):
            raise ValueError("horizon_steps must be a positive integer")
        if horizon_steps is None and policy is not None:
            raise ValueError("External policies require an explicit evaluation horizon")
        # An action_filter on the expert (declared perturbation, e.g. DART noise) keeps the
        # lecturer's controller memory and stop rules; 'raw_action' is the clean expert label.
        self.policy = policy
        self.horizon_steps = horizon_steps
        self.action_filter = action_filter
        self.first_lecturer_stop = None
        self.client = None

    def __enter__(self):
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.client is not None:
            self.client.disconnect()
            self.client = None

    def __exit__(self, *_):
        self.close()

    def _initialize(self):
        cfg = self.config
        self.sim = sim = create_simulation(self.roboenv_root, headless=cfg["headless"], physics=cfg["physics"])
        self.client = client = sim.pybullet_client
        rid = sim.bot[0].bot_pybullet
        if self.initial_base is not None:
            position, nominal_orientation = client.getBasePositionAndOrientation(rid)
            linear_velocity, angular_velocity = client.getBaseVelocity(rid)
            roll, pitch, _ = p.getEulerFromQuaternion(nominal_orientation)
            initial = self.initial_base
            orientation = p.getQuaternionFromEuler([roll, pitch, initial["yaw"]])
            client.resetBasePositionAndOrientation(rid, [initial["x"], initial["y"], position[2]], orientation)
            client.resetBaseVelocity(rid, linear_velocity, angular_velocity)
            sim.ReceiveObservation()
        self.finger_angles = configure_gripper_fingers(sim, client, rid, np.asarray(cfg["fixed_finger_angles"]))
        if not cfg["headless"]:
            visual = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, .85])
            client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=self.target.tolist())
            client.addUserDebugPoints(pointPositions=[self.target.tolist()], pointColorsRGB=[[1, 0, 0]], pointSize=10)
        self.dyn, self.ee_id, self.shoulder_id, self.joint_names = create_dynamics(sim, self.roboenv_root)
        if len(self.joint_names) != 17 or not all("wheel" in name for name in self.joint_names[:4]):
            raise ValueError("Expected Robotnik joint order: 4 wheels + 7 arm + 6 fingers")
        self.dt = float(sim.GetTimeStep())
        self.state = self.observe(0.0)
        self.initial_state = self.state
        lower, upper = sim.GetBotJointsLimit()
        self.lower, self.upper = np.asarray(lower[4:11]), np.asarray(upper[4:11])
        self.q_hold = self.state["q"][4:11].copy()
        self.q_des = self.q_hold.copy()
        self.base_params = BaseControllerParams(**cfg["base_controller"])
        self.arm_params = ArmControllerParams(**cfg["arm_controller"])
        self.criteria = StopCriteria(max_steps=cfg["max_steps"], **cfg["stop"])
        self.nav = BaseNavState()
        reference = np.zeros(2) if cfg["goal_reference"] == "origin" else np.asarray(self.state["base"][:2], dtype=float)
        self.goal_reference_xy = reference
        self.goal_bearing, self.goal = base_goal(self.target, reference, cfg["standoff"])
        self.command = MotorCommands()
        self.steps = 0
        self.termination = None
        self.message = None
        self.stop_state = None
        self.navigation_end = None
        self.minimum_distance = self.state["distance"]
        self.error_integral = 0.0
        self.distance_travelled = 0.0

    def observe(self, t):
        sim, dyn = self.sim, self.dyn
        base = np.asarray(sim.GetBasePosition(), dtype=float)
        orientation = np.asarray(sim.GetBaseOrientation(), dtype=float)
        yaw = float(quaternion2bearing(orientation[3], *orientation[:3]))
        q = np.asarray(sim.GetMotorAngles(0), dtype=float)
        qd = np.asarray(sim.GetMotorVelocities(0), dtype=float)
        q_pin = dyn.ReoderJoints2PinVec(np.concatenate((base, orientation, q)), "pos")
        pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
        pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
        ee = dyn.pin_data.oMf[self.ee_id].translation.copy()
        shoulder = dyn.pin_data.oMf[self.shoulder_id].translation.copy()
        observation = np.concatenate((base, [yaw], q[4:11], qd[4:11], self.target))
        if not np.isfinite(observation).all():
            raise FloatingPointError("Nonfinite robot observation")
        return {"time": t, "base": base, "orientation": orientation, "yaw": yaw,
                "q": q, "qd": qd, "q_pin": q_pin, "ee_position": ee, "shoulder": shoulder,
                "velocity": np.asarray(sim.GetBaseLinVelocity()),
                "angular_velocity": np.asarray(sim.GetBaseAngVelocity()),
                "observation": observation, "distance": float(np.linalg.norm(self.target - ee))}

    def transitions(self):
        """Yield every executed step, including the step that triggers termination."""
        while self.termination is None:
            current = self.state
            dyn = self.dyn
            if self.policy is None:
                wheels = base_controller(self.nav, current["base"], current["yaw"], self.goal,
                                         self.goal_bearing, current["time"], self.base_params)
                torque, self.q_des, qd_des = arm_controller(
                    dyn, current["q_pin"], current["q"][4:11], current["qd"][4:11], self.q_des, self.q_hold,
                    self.target, current["ee_position"], self.ee_id, self.shoulder_id,
                    self.lower, self.upper, self.dt, self.arm_params, reach_active=self.nav.done)
                action = np.concatenate((wheels, torque))
                if not np.isfinite(action).all():
                    raise FloatingPointError("Nonfinite expert action")
            else:
                action = np.asarray(self.policy(current["observation"].copy()), dtype=float)
                qd_des = np.zeros(7)
            if action.shape != (11,) or not np.isfinite(action).all():
                raise FloatingPointError("Invalid policy action: expected 11 finite commands")
            raw_action = action.copy()
            if self.action_filter is not None:
                action = np.asarray(self.action_filter(action), dtype=float)
                if action.shape != (11,) or not np.isfinite(action).all():
                    raise FloatingPointError("Invalid filtered action")
            if self.nav.done and self.navigation_end is None:
                self.navigation_end = current["time"]
            self.command.SetControlCmd(np.concatenate((action, np.zeros(6))), ["velocity"] * 4 + ["torque"] * 13)
            self.sim.Step(self.command, "torque")
            if dyn.visualizer:
                for index in range(len(self.sim.bot)):
                    dyn.DisplayModel(self.sim.GetMotorAngles(index))
            # The lecturer accumulates the clock; retain its rounding and stop boundaries.
            self.steps += 1
            self.state = self.observe(current["time"] + self.dt)
            self.minimum_distance = min(self.minimum_distance, self.state["distance"])
            self.error_integral += .5 * (current["distance"] + self.state["distance"]) * self.dt
            self.distance_travelled += float(np.linalg.norm(self.state["base"][:2] - current["base"][:2]))
            keys = self.client.getKeyboardEvents()
            # Keep the starter's keyboard condition as well as its precedence.
            if ord("q") in keys and keys[ord("q")] and self.client.KEY_WAS_TRIGGERED:
                self.termination, self.message = "user_stop", "Stopped with q"
            elif self.horizon_steps is not None:
                # External evaluation observes the original stop decision but continues
                # to a common horizon; expert actions/state updates above are unchanged.
                if self.policy is None and self.first_lecturer_stop is None:
                    message = check_stop(self.criteria, current["time"], self.dt,
                                         current["distance"], self.nav.done)
                    if message:
                        self.first_lecturer_stop = {"message": message, "time_s": current["time"],
                                                   "distance_m": current["distance"],
                                                   "success": message.startswith("Target reached")}
                if self.steps >= self.horizon_steps:
                    self.termination, self.message = "horizon", "External evaluation horizon reached"
            else:
                self.message = check_stop(self.criteria, current["time"], self.dt, current["distance"], self.nav.done)
                if self.message:
                    self.termination = ("success" if self.message.startswith("Target reached") else
                                        "stalled" if self.message.startswith("Target not reachable") else "max_steps")
            if self.termination is not None:
                self.stop_state = current
            yield {"step": self.steps - 1, "current": current, "next": self.state, "action": action,
                   "raw_action": raw_action, "q_des": self.q_des.copy(), "qd_des": qd_des.copy(),
                   "phase": self.nav.phase, "navigation_done": self.nav.done}
            if not self.config["headless"] and self.termination is None:
                time.sleep(self.dt)

    def result(self):
        """Final physical metrics and the earlier measurement used to decide stop."""
        return {"termination": self.termination, "termination_message": self.message,
                "success": self.termination == "success", "steps": self.steps,
                "duration_s": self.state["time"], "final_distance_m": self.state["distance"],
                "minimum_distance_m": self.minimum_distance,
                "mean_distance_m": self.error_integral / self.state["time"],
                "distance_travelled_xy_m": self.distance_travelled,
                "navigation_end_s": self.navigation_end,
                "navigation_timeout": self.navigation_end is not None and self.navigation_end > self.base_params.nav_timeout,
                "final_within_tolerance": self.state["distance"] < self.criteria.reach_tolerance,
                "final_observation": self.state["observation"].tolist(),
                "stop_decision_time_s": self.stop_state["time"],
                "stop_decision_distance_m": self.stop_state["distance"]}
