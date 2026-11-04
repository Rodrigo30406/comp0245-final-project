"""Expert v3 of the obstacle task (plan section 2): waypoint legs over the UNCHANGED template controllers.

TwoSegmentNavigation drives the base with robotnik_arm_control.base_controller, unchanged, along the scenario's route
start -> waypoint 1 -> ... -> waypoint n -> dock (n = 1 in the single-obstacle task, 2-3 in an obstacle course,
obstacle_task/course.py):
  leg i < n: goal = waypoint i (goal bearing dock_bearing, unused: the orient phase never runs there);
  last leg:  goal = scenario.dock_xy, goal bearing = dock_bearing, i.e. exactly the A-to-B navigation.
At every step the controller of the current leg is called. When a waypoint-leg call leaves its state in phase
"orient" (the base reached or passed the waypoint), that call's command is discarded, the next leg starts with a fresh
BaseNavState() and its controller is called in the same step. The navigation timeout (NAV_TIMEOUT = 45 s, A-to-B
25 s) is absolute episode time, so it covers the whole navigation: if it fires in a waypoint leg, every remaining leg
is skipped at once and the dock leg starts with done=True (zero wheel commands from that step on, the arm starts
reaching where the base is).
Without obstacles (gate A2) the waypoint legs are skipped and the A-to-B timeout (25 s) is used: the A-to-B expert
exactly.

`segment` keeps its two-valued meaning for the labels and the learners: 0 = driving to any waypoint, 1 = the dock leg
and reaching. `leg` is the index of the current leg (0 .. n). For one waypoint the behaviour is identical to expert v2
(state0 is the waypoint leg's state, state1 the dock leg's).

ExpertPolicy is the expert of both the collector and the harness: reset(target, info), then act(obs) ->
(wheel_cmds (4,), torque_arm (7,)), the clean expert commands. It reads the standard robot info (dyn_model, frame
ids, joint limits, time step, init_q_mes) plus the PRIVILEGED info["scenario"] (Scenario or its dict: waypoints, dock),
which the harness gives to this exact class only. The arm is robotnik_arm_control.arm_controller unchanged, with
reach_active = (segment == 1 and state1.done). The collector reads its state after act() for the labels and the
noise fade: segment, base_state, current_goal_pos, q_des_arm, qd_des_arm, segment_switch_time, nav_done_time.
"""
import os
import sys

import numpy as np
import pinocchio as pin

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from robotnik_arm_control import (  # noqa: E402
    ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller, base_controller,
)
from robotnik_arm_task import ARM_REACH_LIMIT, BASE_NAV_TIMEOUT  # noqa: E402
from obstacle_task.scenario import Scenario  # noqa: E402

NAV_TIMEOUT = 45.0         # s, whole navigation (all legs); A-to-B: BASE_NAV_TIMEOUT = 25 s


class TwoSegmentNavigation:
    """Waypoints -> dock navigation state over the unchanged base_controller (see the module docstring)."""

    def __init__(self, scenario, params=None):
        self.scenario = Scenario.from_dict(scenario) if isinstance(scenario, dict) else scenario
        self.route = self.scenario.route()                       # (n, 2) waypoints; n = 0 without obstacles
        self.goals = [np.asarray(w, dtype=float) for w in self.route] + \
                     [np.asarray(self.scenario.dock_xy, dtype=float)]
        self.leg = 0 if self.scenario.obstacles else len(self.goals) - 1
        self.params = params or BaseControllerParams(
            nav_timeout=NAV_TIMEOUT if self.scenario.obstacles else BASE_NAV_TIMEOUT)
        self.states = [BaseNavState() for _ in self.goals]
        self.segment_switch_time = None                          # start of the dock leg
        self.nav_done_time = None

    @property
    def segment(self):
        return 0 if self.leg < len(self.goals) - 1 else 1

    @property
    def state(self):
        return self.states[self.leg]

    @property
    def state0(self):
        """State of the first waypoint leg (the only one in the single-obstacle task)."""
        return self.states[0]

    @property
    def state1(self):
        """State of the dock leg."""
        return self.states[-1]

    @property
    def done(self):
        return self.segment == 1 and self.state1.done

    @property
    def current_goal_pos(self):
        return self.goals[self.leg]

    def _call(self, base_pos, base_bearing, current_time):
        return base_controller(self.state, base_pos, base_bearing, self.current_goal_pos,
                               self.scenario.dock_bearing, current_time, self.params)

    def command(self, base_pos, base_bearing, current_time):
        """Wheel commands of this step; advances to the next leg (same step) when a waypoint leg ends."""
        wheels = self._call(base_pos, base_bearing, current_time)
        # A waypoint leg ends in "orient" (waypoint reached or passed) or by the timeout (done in align / drive)
        while self.segment == 0 and (self.state.phase == "orient" or self.state.done):
            timed_out = current_time > self.params.nav_timeout
            if timed_out:
                self.leg = len(self.goals) - 1                   # skip the remaining waypoints
                self.states[-1] = BaseNavState(done=True)
            else:
                self.leg += 1
                self.states[self.leg] = BaseNavState()
            if self.segment == 1:
                self.segment_switch_time = float(current_time)
            wheels = self._call(base_pos, base_bearing, current_time)
        if self.done and self.nav_done_time is None:
            self.nav_done_time = float(current_time)
        return wheels


class ExpertPolicy:
    """The obstacle expert (privileged: reads info["scenario"]); clean commands, state readable after act()."""

    def reset(self, target, info):
        self.target = np.asarray(target, dtype=float).copy()
        self.model = info["dyn_model"]
        self.ee_frame_id, self.shoulder_frame_id = info["ee_frame_id"], info["shoulder_frame_id"]
        self.arm_lower, self.arm_upper = info["joint_lower"][4:11], info["joint_upper"][4:11]
        self.time_step = info["time_step"]
        self.navigation = TwoSegmentNavigation(info["scenario"])
        self.base_params = self.navigation.params
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.base_goal_pos = np.asarray(self.navigation.scenario.dock_xy, dtype=float)
        self.base_goal_bearing = self.navigation.scenario.dock_bearing
        self.q_init_arm = info["init_q_mes"][4:11].copy()
        self.q_des_arm = self.q_init_arm.copy()
        self.qd_des_arm = np.zeros(7)

    @property
    def segment(self):
        return self.navigation.segment

    @property
    def base_state(self):
        return self.navigation.state

    @property
    def current_goal_pos(self):
        return self.navigation.current_goal_pos

    @property
    def segment_switch_time(self):
        return self.navigation.segment_switch_time

    @property
    def nav_done_time(self):
        return self.navigation.nav_done_time

    def act(self, obs):
        q_mes, qd_mes = obs["q_mes"], obs["qd_mes"]
        q_pin = self.model.ReoderJoints2PinVec(np.concatenate((obs["base_pos"], obs["base_ori"], q_mes)), "pos")
        pin.forwardKinematics(self.model.pin_model, self.model.pin_data, q_pin)
        pin.updateFramePlacements(self.model.pin_model, self.model.pin_data)
        ee_pos = self.model.pin_data.oMf[self.ee_frame_id].translation.copy()
        wheels = self.navigation.command(obs["base_pos"], obs["base_bearing"], obs["t"])
        torques, self.q_des_arm, self.qd_des_arm = arm_controller(
            self.model, q_pin, q_mes[4:11], qd_mes[4:11], self.q_des_arm, self.q_init_arm,
            self.target, ee_pos, self.ee_frame_id, self.shoulder_frame_id, self.arm_lower, self.arm_upper,
            self.time_step, self.arm_params, reach_active=self.navigation.done,
        )
        return wheels, torques
