"""Multi-agent extension of the obstacle task (step 9): per-robot controllers and multi-agent expert data collection.

Sticks directly to the obstacle generation method (`obstacle_task/scenario.py`) and the two-segment planner
(`TwoSegmentNavigation` + `ExpertPolicy` in `obstacle_task/expert.py` over `robotnik_arm_control.base_controller` and
`arm_controller`) of the current branch `multi-agent`:
  - Obstacle generation & waypoint planning (`sample_multi_agent_scenario`):
      * Uses `sample_init_poses`, `sample_targets`, `relative_to_world`, `expert_base_goal`, `frame_of`, and
        `constraints_ok` from `obstacle_task/scenario.py` with the exact cylinder obstacle parameters (`OBST_RADIUS`,
        `OBST_HEIGHT`, `ALONG`, `LATERAL`, `AMBIG_BAND`, `WAYPOINT_CLEAR`, `END_CLEAR`, `SEGMENT_CLEAR`, `TARGET_CLEAR`).
      * Places the static cylinder obstacle along the start->dock corridor (`center = init_xy + along * length * t + d * n`)
        and plans each robot's detour waypoint via `waypoint = center + side * (radius + WAYPOINT_CLEAR) * n`, verified
        by `scenario.constraints_ok` for every robot.
      * Multi-agent target & path constraints ensure targets are close enough (`TARGET_SEP_RANGE = (1.65, 2.85) m` and
        converging/crossing start->dock paths) that the robots must avoid collisions with the obstacle and with each
        other, while starts, waypoints, and docks are far enough apart (`START_SEP_MIN`, `WAYPOINT_SEP_MIN`,
        `DOCK_SEP_MIN`) that every robot has a feasible collision-free path to reach its target.
  - Per-robot controller (`RobotNavigationController` + `RobotExpertController`):
      * Each robot i in `0..n_agents-1` has its own `RobotExpertController(agent_id=i)` instance (subclassing
        `ExpertPolicy` from `obstacle_task/expert.py`).
      * Nominal base navigation is driven by `TwoSegmentNavigation` from `obstacle_task/expert.py` (`segment 0` to
        `waypoint_xy`, `segment 1` to `dock_xy` with `dock_bearing` via `robotnik_arm_control.base_controller`).
      * A real-time inter-robot collision avoidance layer checks the 3-circle base+arm footprint clearance against all
        other robots (and static obstacles) at every 1 ms step, steering clear, yielding right-of-way, and guarding
        arm-sweeping in-place turns when another robot is nearby.
      * Arm control uses `robotnik_arm_control.arm_controller` unchanged once `navigation.done` is True.
"""
import copy
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pinocchio as pin
import pybullet as p
from simulation_and_control import MotorCommands, PinWrapper, pb, wrap_angle

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import collect_data as legacy  # noqa: E402
import evaluate_policy as v1  # noqa: E402
from collect_data import relative_to_world, sample_init_poses, sample_targets  # noqa: E402
from evaluate_policy import expert_base_goal  # noqa: E402
from robotnik_arm_control import (  # noqa: E402
    FINGER_JOINT_NAMES, ArmControllerParams, BaseControllerParams, StopCriteria,
    arm_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)
from robotnik_arm_task import ARM_REACH_LIMIT, FIXED_FINGER_ANGLES, REACH_TOLERANCE  # noqa: E402
from obstacle_task import lidar  # noqa: E402
from obstacle_task.expert import NAV_TIMEOUT, ExpertPolicy, TwoSegmentNavigation  # noqa: E402
from obstacle_task.scenario import (  # noqa: E402
    ALONG, AMBIG_BAND, CHASSIS_HALF_WIDTH, END_CLEAR, INIT_RADIUS, INIT_YAW_DEG, LATERAL,
    OBST_HEIGHT, OBST_RADIUS, SEGMENT_CLEAR, TARGET_CLEAR, TARGET_R, TARGET_THETA_DEG,
    TARGET_Z, WAYPOINT_CLEAR, Scenario, constraints_ok, frame_of, lateral_position, segment_distance,
)
from obstacle_task.sim import CLEARANCE_SEARCH, link_name, min_clearance, robot_obstacle_contact, spawn_obstacles  # noqa: E402

MULTI_STREAM = 19                     # RNG stream for multi-agent scenario sampling
DEFAULT_N_AGENTS = 2
STEP_CAP = 60000                      # steps (60 s at 1 kHz)

# Multi-agent inter-robot proximity & feasibility constants (metres)
START_SEP_MIN = 1.90                  # m, min distance between robot start positions (arm sweep ~0.73 m each)
TARGET_SEP_RANGE = (1.65, 2.85)       # m, 2D distance between targets ("close enough" yet "far enough")
DOCK_SEP_MIN = 1.65                   # m, min 2D distance between docking positions
WAYPOINT_SEP_MIN = 1.35               # m, min 2D distance between detour waypoints around shared obstacle
REACH_SEP_MIN = 1.40                  # m, min 2D distance between [dock_i, target_i] and [dock_j, target_j]
START_TO_OTHER_DOCK_MIN = 1.45        # m, min 2D distance from init_xy[i] to dock_xy[j] / target[j]
STRAIGHT_CONFLICT_DIST = 0.95         # m, straight-line start->dock paths pass within this distance in the middle
KINEMATIC_CLEAR_MIN = 0.30            # m, min 3-circle surface clearance along the planned TwoSegmentNavigation paths
KINEMATIC_MAX_NAV_TIME = 40.0         # s, max time for all robots to finish navigation in kinematic preview
MAX_SCENARIO_TRIES = 5000

# 3-circle horizontal footprint of one Robotnik base + folded Kinova arm:
# (forward offset along base_bearing in m, radius in m). Covers chassis (0.31 m half-width) and arm tip (~0.73 m).
ROBOT_FOOTPRINT_CIRCLES = (
    (0.00, 0.36),   # chassis & wheels
    (0.38, 0.22),   # arm mid-links
    (0.68, 0.16),   # wrist & gripper fingers (extends to 0.84 m in front of base centre)
)
LIDAR_ROBOT_CIRCLES = (
    (0.00, 0.35),   # chassis cylinder seen by other robots' 2D lidar
    (0.42, 0.22),   # protruding arm cylinder seen by other robots' 2D lidar
)

# Reactive inter-robot collision-avoidance thresholds
REACT_CLEARANCE = 0.75                # m, 3-circle surface clearance below which reactive avoidance activates
YIELD_CLEARANCE = 0.56                # m, multi-horizon clearance below which lower-priority robot yields before crossing
STOP_CLEARANCE = 0.36                 # m, surface clearance below which yielding/approaching robot pauses on its line
TURN_GUARD_CLEARANCE = 0.40           # m, surface clearance below which arm-approaching in-place turns pause
LOOKAHEAD_DT = 0.50                   # s, short lookahead horizon for immediate clearance prediction
YIELD_HORIZONS = (0.3, 0.6, 1.0, 1.5, 2.0, 2.6)  # s, multi-step lookahead horizons for crossing right-of-way

# Distinct GUI colours for robots and their corresponding targets
AGENT_COLORS = (
    (1.00, 0.18, 0.18, 0.90),  # Agent 0: red
    (0.15, 0.50, 1.00, 0.90),  # Agent 1: blue
    (0.15, 0.80, 0.30, 0.90),  # Agent 2: green
    (0.95, 0.65, 0.10, 0.90),  # Agent 3: orange
    (0.75, 0.25, 0.90, 0.90),  # Agent 4: purple
    (0.10, 0.80, 0.80, 0.90),  # Agent 5: cyan
)


def is_multi_agent_scenario(scenario):
    """True when `scenario` (Scenario or dict) carries multi-agent specifications."""
    meta = scenario.get("meta", {}) if isinstance(scenario, dict) else getattr(scenario, "meta", {})
    return bool(meta and meta.get("multi_agent") and meta.get("agents"))


def get_scenario_agents(scenario):
    """Return the list of per-agent scenario dicts from `scenario`."""
    meta = scenario.get("meta", {}) if isinstance(scenario, dict) else getattr(scenario, "meta", {})
    return list(meta.get("agents", []))


def agent_dict_to_scenario(seed, episode, agent_dict, obstacles, noisy=False):
    """Build a standard `obstacle_task.scenario.Scenario` for one agent so `TwoSegmentNavigation` runs unchanged."""
    return Scenario(
        seed=int(seed),
        episode=int(episode),
        target=[float(v) for v in agent_dict["target"]],
        init_xy=[float(v) for v in agent_dict["init_xy"]],
        init_yaw=float(agent_dict["init_yaw"]),
        dock_xy=[float(v) for v in agent_dict["dock_xy"]],
        dock_bearing=float(agent_dict["dock_bearing"]),
        standoff=float(agent_dict["standoff"]),
        obstacles=[dict(ob) for ob in (obstacles or ())],
        along=float(agent_dict.get("along", 0.5)),
        lateral_offset=float(agent_dict.get("lateral_offset", 0.0)),
        ambiguous=bool(agent_dict.get("ambiguous", False)),
        side=int(agent_dict.get("side", -1)),
        waypoint_xy=[float(v) for v in agent_dict["waypoint_xy"]],
        noisy=bool(noisy),
    )


# ==============================================================================
# 3-Circle Footprint Geometry & Lidar Representation of Other Robots
# ==============================================================================
def footprint_circles(base_xy, yaw):
    """World (x, y, r) of the 3 footprint circles of a robot at (base_xy, yaw)."""
    x, y = float(base_xy[0]), float(base_xy[1])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return [(x + off * c, y + off * s, r) for off, r in ROBOT_FOOTPRINT_CIRCLES]


def robot_pair_clearance(xy_a, yaw_a, xy_b, yaw_b):
    """Minimum surface-to-surface distance (m) and unit vector (from B to A) between two robots' 3-circle footprints."""
    circles_a = footprint_circles(xy_a, yaw_a)
    circles_b = footprint_circles(xy_b, yaw_b)
    best_clear = math.inf
    best_unit = np.array([1.0, 0.0])
    for xa, ya, ra in circles_a:
        for xb, yb, rb in circles_b:
            dx, dy = xa - xb, ya - yb
            dist = math.hypot(dx, dy)
            clear = dist - (ra + rb)
            if clear < best_clear:
                best_clear = clear
                best_unit = np.array([dx / dist, dy / dist]) if dist > 1e-9 else np.array([1.0, 0.0])
    return float(best_clear), best_unit


def robot_obstacle_footprint_clearance(xy, yaw, obstacles):
    """Minimum surface-to-surface distance (m) and unit vector (from obstacle to robot) against static cylinders."""
    if not obstacles:
        return math.inf, np.array([1.0, 0.0])
    circles = footprint_circles(xy, yaw)
    best_clear = math.inf
    best_unit = np.array([1.0, 0.0])
    for xa, ya, ra in circles:
        for ob in obstacles:
            dx, dy = xa - float(ob["x"]), ya - float(ob["y"])
            dist = math.hypot(dx, dy)
            clear = dist - (ra + float(ob["r"]))
            if clear < best_clear:
                best_clear = clear
                best_unit = np.array([dx / dist, dy / dist]) if dist > 1e-9 else np.array([1.0, 0.0])
    return float(best_clear), best_unit


def robot_lidar_obstacles(agents_xy, agents_yaw, exclude_agent, static_obstacles=()):
    """Obstacle list [{"x", "y", "r", "h"}, ...] seen by `exclude_agent`'s 2D lidar (static obstacles + other robots)."""
    obs = [dict(ob) for ob in (static_obstacles or ())]
    for j, (xy, yaw) in enumerate(zip(agents_xy, agents_yaw)):
        if j == exclude_agent:
            continue
        x, y = float(xy[0]), float(xy[1])
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        for off, r in LIDAR_ROBOT_CIRCLES:
            obs.append({"x": x + off * c, "y": y + off * s, "r": r, "h": OBST_HEIGHT})
    return obs


def segment_to_segment_distance(a0, a1, b0, b1):
    """Shortest 2D distance between segment [a0, a1] and segment [b0, b1], plus closest parameters (sa, sb) in [0, 1]."""
    a0, a1, b0, b1 = (np.asarray(v, dtype=float)[:2] for v in (a0, a1, b0, b1))
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    denom = a * c - b * b
    if denom < 1e-12:
        sa = 0.0
        sb = np.clip(e / c, 0.0, 1.0) if c > 1e-12 else 0.0
    else:
        sa = np.clip((b * e - c * d) / denom, 0.0, 1.0)
        sb = np.clip((a * e - b * d) / denom, 0.0, 1.0)
    candidates = [
        (0.0, np.clip(float((a0 - b0) @ v) / c, 0.0, 1.0) if c > 1e-12 else 0.0),
        (1.0, np.clip(float((a1 - b0) @ v) / c, 0.0, 1.0) if c > 1e-12 else 0.0),
        (np.clip(float((b0 - a0) @ u) / a, 0.0, 1.0) if a > 1e-12 else 0.0, 0.0),
        (np.clip(float((b1 - a0) @ u) / a, 0.0, 1.0) if a > 1e-12 else 0.0, 1.0),
        (float(sa), float(sb)),
    ]
    best_dist = math.inf
    best_pair = (0.5, 0.5)
    for s_a, s_b in candidates:
        pa = a0 + s_a * u
        pb = b0 + s_b * v
        dist = float(np.linalg.norm(pa - pb))
        if dist < best_dist:
            best_dist = dist
            best_pair = (float(s_a), float(s_b))
    return best_dist, best_pair[0], best_pair[1]


# ==============================================================================
# Per-Robot Navigation & Arm Controllers (Over `TwoSegmentNavigation` & `ExpertPolicy`)
# ==============================================================================
def _wheel_to_vw(wheels, params):
    """Convert [rw, lw, lw, rw] wheel speeds (rad/s) to body (v, w) in (m/s, rad/s)."""
    rw, lw = float(wheels[0]), float(wheels[1])
    v = params.wheel_radius * (rw + lw) * 0.5
    w = params.wheel_radius * (rw - lw) / params.wheel_base_width
    return v, w


def _vw_to_wheels(v, w, params):
    """Convert body (v, w) to [rw, lw, lw, rw] wheel speeds (rad/s)."""
    lw = (v - w * params.wheel_base_width * 0.5) / params.wheel_radius
    rw = (v + w * params.wheel_base_width * 0.5) / params.wheel_radius
    return np.array([rw, lw, lw, rw], dtype=float)


class RobotNavigationController(TwoSegmentNavigation):
    """Per-robot base controller extending `TwoSegmentNavigation` (`obstacle_task/expert.py`) with inter-robot collision avoidance.

    Uses the unchanged `TwoSegmentNavigation.command(base_pos, base_bearing, current_time)` for the two segments
    (`segment 0: init_xy -> waypoint_xy`, `segment 1: waypoint_xy -> dock_xy` with `dock_bearing`), and applies a
    real-time collision-avoidance filter when another robot comes within `REACT_CLEARANCE`.
    """

    def __init__(self, agent_id, agent_scenario, params=None, reactive=True):
        super().__init__(agent_scenario, params=params)
        self.agent_id = int(agent_id)
        self.reactive = bool(reactive)

    def remaining_segments(self, base_xy):
        """Return list of remaining 2D line segments [(p0, p1), ...] for this robot's TwoSegmentNavigation plan."""
        if self.done:
            return []
        xy = np.asarray(base_xy, dtype=float)[:2]
        wp = np.asarray(self.scenario.waypoint_xy, dtype=float)[:2]
        dock = np.asarray(self.scenario.dock_xy, dtype=float)[:2]
        if self.segment == 0:
            return [(xy, wp), (wp, dock)]
        return [(xy, dock)]

    def _reactive_avoidance(self, base_pos, base_bearing, wheels, other_robots, obstacles):
        """Path-preserving velocity scaling when another robot is within REACT_CLEARANCE along the planned path.

        Scales both left and right wheels by a common factor `scale in [0, 1]` during `"drive"` (or pauses in-place
        turns when another robot is within `TURN_GUARD_CLEARANCE`), keeping the robot strictly on the exact straight
        segments of `TwoSegmentNavigation` (`init_xy -> waypoint_xy -> dock_xy`) so it never deviates into the static
        cylinder obstacle.
        """
        if self.done or not other_robots:
            return wheels
        xy = np.asarray(base_pos, dtype=float)[:2]
        yaw = float(base_bearing)
        v_nom, w_nom = _wheel_to_vw(wheels, self.params)
        heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
        my_goal = self.current_goal_pos

        pred_xy = xy + v_nom * LOOKAHEAD_DT * heading_vec
        pred_yaw = yaw + w_nom * LOOKAHEAD_DT
        scale = 1.0
        block_turn = False

        for other in (other_robots or ()):
            other_id = int(other.get("agent_id", -1))
            if other_id == self.agent_id:
                continue
            other_xy = np.asarray(other["base_pos"], dtype=float)[:2]
            other_yaw = float(other["base_bearing"])
            other_vel = np.asarray(other.get("base_vel", [0.0, 0.0]), dtype=float)[:2]
            other_done = bool(other.get("nav_done", False))
            other_segs = other.get("remaining_segments")
            if other_segs is None:
                other_goal = other.get("current_goal_pos")
                other_segs = [(other_xy, np.asarray(other_goal, dtype=float)[:2])] if (not other_done and other_goal is not None) else []

            clear_now, _ = robot_pair_clearance(xy, yaw, other_xy, other_yaw)
            other_pred_xy = other_xy + other_vel * LOOKAHEAD_DT
            clear_pred, _ = robot_pair_clearance(pred_xy, pred_yaw, other_pred_xy, other_yaw)
            lower_priority = (not other_done) and (self.agent_id > other_id and other_id >= 0)

            # Segment-corridor right-of-way: lower-priority robot waits outside the higher-priority robot's active path
            if lower_priority and self.state.phase == "drive" and v_nom > 1e-4:
                for seg_a, seg_b in other_segs:
                    seg_a = np.asarray(seg_a, dtype=float)[:2]
                    seg_b = np.asarray(seg_b, dtype=float)[:2]
                    dist_seg, s_my, _ = segment_to_segment_distance(xy, my_goal, seg_a, seg_b)
                    if dist_seg < 1.12 and s_my > 0.01:
                        d_now = segment_distance(seg_a, seg_b, xy)
                        d_pred = segment_distance(seg_a, seg_b, pred_xy)
                        if d_pred < d_now - 1e-4 and d_pred < 1.22:
                            if d_now <= 1.25 or clear_now <= STOP_CLEARANCE:
                                scale = 0.0
                            else:
                                scale = min(scale, float(np.clip((d_now - 1.20) / 0.35, 0.0, 1.0)))

            # Immediate proximity protection for any robot
            eff_clear = min(clear_now, clear_pred)
            if self.state.phase == "drive" and v_nom > 1e-4 and eff_clear < REACT_CLEARANCE:
                to_other = other_xy - xy
                dist_centers = float(np.linalg.norm(to_other))
                other_in_front = dist_centers > 1e-6 and float((to_other / dist_centers) @ heading_vec) > 0.25
                if (lower_priority or other_in_front) and clear_pred < clear_now - 0.002:
                    if eff_clear <= STOP_CLEARANCE:
                        scale = 0.0
                    elif eff_clear < YIELD_CLEARANCE:
                        s_now = (eff_clear - STOP_CLEARANCE) / max(1e-6, YIELD_CLEARANCE - STOP_CLEARANCE)
                        scale = min(scale, float(np.clip(s_now, 0.0, 1.0)))

            # Guard in-place turns so the 0.73 m arm does not swing into a nearby robot
            if abs(v_nom) < 1e-4 and abs(w_nom) > 1e-4 and clear_now < TURN_GUARD_CLEARANCE:
                turn_pred_yaw = yaw + math.copysign(0.25, w_nom)
                clear_turn, _ = robot_pair_clearance(xy, turn_pred_yaw, other_xy, other_yaw)
                if clear_turn < clear_now - 0.005:
                    block_turn = True

        if block_turn and abs(v_nom) < 1e-4:
            return np.zeros(4, dtype=float)
        if self.state.phase == "drive" and scale < 1.0:
            return wheels * float(scale)
        return wheels

    def command(self, base_pos, base_bearing, current_time, other_robots=(), obstacles=()):
        """Compute 4 wheel velocity commands (rad/s) using `TwoSegmentNavigation` + inter-robot collision avoidance."""
        wheels = super().command(base_pos, base_bearing, current_time)
        if self.reactive and not self.done and other_robots:
            wheels = self._reactive_avoidance(base_pos, base_bearing, wheels, other_robots, obstacles)
        return wheels


class RobotExpertController(ExpertPolicy):
    """Per-robot controller subclassing `ExpertPolicy` (`obstacle_task/expert.py`) for multi-agent execution."""

    def __init__(self, agent_id=0, reactive=True):
        self.agent_id = int(agent_id)
        self.reactive = bool(reactive)

    def reset(self, target, info):
        self.target = np.asarray(target, dtype=float).copy()
        self.model = info["dyn_model"]
        self.ee_frame_id, self.shoulder_frame_id = info["ee_frame_id"], info["shoulder_frame_id"]
        self.arm_lower, self.arm_upper = info["joint_lower"][4:11], info["joint_upper"][4:11]
        self.time_step = info["time_step"]

        sc_raw = info.get("scenario")
        sc = Scenario.from_dict(sc_raw) if isinstance(sc_raw, dict) else sc_raw
        agent_dict = info.get("agent_scenario")
        if agent_dict is None and sc is not None and is_multi_agent_scenario(sc):
            agents = get_scenario_agents(sc)
            if self.agent_id < len(agents):
                agent_dict = agents[self.agent_id]

        if agent_dict is not None:
            obstacles = sc.obstacles if sc is not None else agent_dict.get("obstacles", [])
            agent_sc = agent_dict_to_scenario(
                sc.seed if sc is not None else 0,
                sc.episode if sc is not None else 0,
                agent_dict,
                obstacles,
                noisy=sc.noisy if sc is not None else False,
            )
        elif sc is not None:
            agent_sc = sc
        else:
            init_pos = np.asarray(info["init_base_pos"], dtype=float)
            init_yaw = float(info["init_base_bearing"])
            dock_xy, dock_bearing, standoff = expert_base_goal(self.target, init_pos, init_yaw)
            agent_sc = Scenario(
                seed=0, episode=0, target=self.target.tolist(), init_xy=init_pos[:2].tolist(),
                init_yaw=init_yaw, dock_xy=dock_xy.tolist(), dock_bearing=float(dock_bearing),
                standoff=float(standoff), obstacles=[], along=0.5, lateral_offset=0.0,
                ambiguous=False, side=-1, waypoint_xy=dock_xy.tolist(),
            )

        self.obstacles = list(agent_sc.obstacles or [])
        self.navigation = RobotNavigationController(self.agent_id, agent_sc, reactive=self.reactive)
        self.base_params = self.navigation.params
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.base_goal_pos = np.asarray(self.navigation.scenario.dock_xy, dtype=float)
        self.base_goal_bearing = self.navigation.scenario.dock_bearing
        self.q_init_arm = info["init_q_mes"][4:11].copy()
        self.q_des_arm = self.q_init_arm.copy()
        self.qd_des_arm = np.zeros(7)

    def act(self, obs):
        q_mes, qd_mes = obs["q_mes"], obs["qd_mes"]
        q_pin = self.model.ReoderJoints2PinVec(np.concatenate((obs["base_pos"], obs["base_ori"], q_mes)), "pos")
        pin.forwardKinematics(self.model.pin_model, self.model.pin_data, q_pin)
        pin.updateFramePlacements(self.model.pin_model, self.model.pin_data)
        ee_pos = self.model.pin_data.oMf[self.ee_frame_id].translation.copy()
        wheels = self.navigation.command(
            obs["base_pos"], obs["base_bearing"], obs["t"],
            other_robots=obs.get("other_robots", ()),
            obstacles=obs.get("obstacles", self.obstacles),
        )
        torques, self.q_des_arm, self.qd_des_arm = arm_controller(
            self.model, q_pin, q_mes[4:11], qd_mes[4:11], self.q_des_arm, self.q_init_arm,
            self.target, ee_pos, self.ee_frame_id, self.shoulder_frame_id, self.arm_lower, self.arm_upper,
            self.time_step, self.arm_params, reach_active=self.navigation.done,
        )
        return wheels, torques


class MultiAgentExpertPolicy:
    """Multi-agent controller holding one independent `RobotExpertController` per robot."""

    def __init__(self, n_agents=DEFAULT_N_AGENTS, reactive=True):
        self.n_agents = int(n_agents)
        self.reactive = bool(reactive)
        self.controllers = [RobotExpertController(i, reactive=reactive) for i in range(self.n_agents)]

    def reset(self, targets, infos):
        self.n_agents = len(targets)
        self.controllers = [RobotExpertController(i, reactive=self.reactive) for i in range(self.n_agents)]
        for ctrl, target, info in zip(self.controllers, targets, infos):
            ctrl.reset(target, info)

    def act(self, obs_list):
        return [ctrl.act(obs) for ctrl, obs in zip(self.controllers, obs_list)]


def simulate_kinematic_multi_agent(agents_spec, obstacles=(), dt=0.05, max_time=42.0, reactive=True):
    """Fast coupled unicycle rollout of `RobotNavigationController` (`TwoSegmentNavigation` + reactive avoidance)."""
    n = len(agents_spec)
    params = BaseControllerParams(nav_timeout=NAV_TIMEOUT)
    navs = [
        RobotNavigationController(
            i, agent_dict_to_scenario(0, 0, spec, obstacles), params=params, reactive=reactive
        )
        for i, spec in enumerate(agents_spec)
    ]
    xy = np.array([spec["init_xy"][:2] for spec in agents_spec], dtype=float)
    yaw = np.array([float(spec["init_yaw"]) for spec in agents_spec], dtype=float)
    vel = np.zeros((n, 2), dtype=float)

    min_pair_clear = math.inf
    min_obst_clear = math.inf
    trajectories = [[xy[i].copy()] for i in range(n)]
    steps = int(round(max_time / dt))

    for step in range(steps):
        t = step * dt
        for i in range(n):
            if obstacles:
                ob_c, _ = robot_obstacle_footprint_clearance(xy[i], yaw[i], obstacles)
                min_obst_clear = min(min_obst_clear, ob_c)
            for j in range(i + 1, n):
                c, _ = robot_pair_clearance(xy[i], yaw[i], xy[j], yaw[j])
                min_pair_clear = min(min_pair_clear, c)

        if all(nav.done for nav in navs):
            break

        other_info = [
            {"agent_id": i, "base_pos": [xy[i, 0], xy[i, 1], 0.0], "base_bearing": yaw[i],
             "base_vel": vel[i].copy(), "nav_done": navs[i].done,
             "current_goal_pos": navs[i].current_goal_pos.copy(),
             "remaining_segments": navs[i].remaining_segments(xy[i])}
            for i in range(n)
        ]
        new_xy = xy.copy()
        new_yaw = yaw.copy()
        for i in range(n):
            wheels = navs[i].command(np.array([xy[i, 0], xy[i, 1], 0.0]), yaw[i], t,
                                     other_robots=other_info, obstacles=obstacles)
            v, w = _wheel_to_vw(wheels, params)
            vel[i] = [v * math.cos(yaw[i]), v * math.sin(yaw[i])]
            new_xy[i, 0] += vel[i, 0] * dt
            new_xy[i, 1] += vel[i, 1] * dt
            new_yaw[i] = wrap_angle(yaw[i] + w * dt)
            trajectories[i].append(new_xy[i].copy())
        xy, yaw = new_xy, new_yaw

    all_done = all(nav.done and (nav.nav_done_time is not None and nav.nav_done_time <= params.nav_timeout)
                   for nav in navs)
    max_done_time = max((nav.nav_done_time or max_time) for nav in navs)
    return all_done, float(min_pair_clear), float(min_obst_clear), float(max_done_time), trajectories


# ==============================================================================
# Multi-Agent Scenario Sampler (Using `scenario.py`'s Exact Obstacle & Waypoint Method)
# ==============================================================================
def _sample_agent_zero(rng):
    """Draw Agent 0 and the static cylinder obstacle using the exact method of `obstacle_task/scenario.py`."""
    xy, yaw = sample_init_poses(1, rng, INIT_RADIUS, INIT_YAW_DEG)
    r, theta, z = sample_targets(1, rng, TARGET_R, TARGET_THETA_DEG, TARGET_Z)
    init_xy, init_yaw = xy[0], float(yaw[0])
    target = relative_to_world(init_xy, init_yaw, float(r[0]), float(theta[0]), float(z[0]))
    dock_xy, dock_bearing, standoff = expert_base_goal(target, np.array([init_xy[0], init_xy[1], 0.0]), init_yaw)
    t, n, length = frame_of(init_xy, dock_xy)
    along = float(rng.uniform(*ALONG))
    d = float(rng.uniform(-LATERAL, LATERAL))
    radius = float(rng.uniform(*OBST_RADIUS))
    coin = float(rng.random())
    center = np.asarray(init_xy) + along * length * t + d * n
    ambiguous = abs(d) < AMBIG_BAND
    side = (1 if coin < 0.5 else -1) if ambiguous else (-1 if d > 0 else 1)
    waypoint = center + side * (radius + WAYPOINT_CLEAR) * n
    ok, _ = constraints_ok(init_xy, dock_xy, target, center, radius, waypoint)
    if not ok:
        return None
    obstacle = {"x": float(center[0]), "y": float(center[1]), "r": radius, "h": OBST_HEIGHT}
    agent0 = {
        "agent_id": 0,
        "init_xy": [float(init_xy[0]), float(init_xy[1])],
        "init_yaw": float(init_yaw),
        "target": [float(v) for v in target],
        "dock_xy": [float(dock_xy[0]), float(dock_xy[1])],
        "dock_bearing": float(dock_bearing),
        "standoff": float(standoff),
        "along": float(along),
        "lateral_offset": float(d),
        "ambiguous": bool(ambiguous),
        "side": int(side),
        "waypoint_xy": [float(waypoint[0]), float(waypoint[1])],
    }
    return agent0, obstacle, center, radius


def _sample_partner_agent(rng, agent_idx, existing_agents, center, radius):
    """Draw Agent `agent_idx` so its start->dock line is blocked by `center` (`scenario.constraints_ok`) and its
    target lies within `TARGET_SEP_RANGE` of Agent 0's target."""
    a0 = existing_agents[0]
    t0, _, _ = frame_of(a0["init_xy"], a0["dock_xy"])
    targ0_2d = np.asarray(a0["target"][:2], dtype=float)
    psi0 = math.atan2(float(t0[1]), float(t0[0]))

    # Sample corridor angle psi1 at 30..62 deg to Agent 0's corridor passing within LATERAL of `center`
    lat_sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    delta_psi = float(np.deg2rad(rng.uniform(30.0, 62.0)))
    psi1 = psi0 + lat_sign * delta_psi
    u_line = np.array([math.cos(psi1), math.sin(psi1)])
    n_line = np.array([-u_line[1], u_line[0]])

    along_est = float(rng.uniform(ALONG[0] + 0.02, ALONG[1] - 0.02))
    len_est = float(rng.uniform(2.85, 3.60))
    d_est = float(rng.uniform(-LATERAL * 0.90, LATERAL * 0.90))
    init_xy = center - (along_est * len_est) * u_line - d_est * n_line
    targ_2d = center + ((1.0 - along_est) * len_est + 0.58) * u_line - d_est * n_line
    z_val = float(rng.uniform(*TARGET_Z))
    init_yaw = float(np.deg2rad(rng.uniform(*INIT_YAW_DEG)))

    # Recompute exact target & dock via scenario.py's relative_to_world and expert_base_goal
    rel_vec = targ_2d - init_xy
    r_exact = float(np.linalg.norm(rel_vec))
    if not (TARGET_R[0] <= r_exact <= TARGET_R[1]):
        return None
    rel_theta_deg = float(np.rad2deg(wrap_angle(math.atan2(float(rel_vec[1]), float(rel_vec[0])) - init_yaw)))
    target = relative_to_world(init_xy, init_yaw, r_exact, rel_theta_deg, z_val)
    dock_xy, dock_bearing, standoff = expert_base_goal(target, np.array([init_xy[0], init_xy[1], 0.0]), init_yaw)
    dock_2d = np.asarray(dock_xy, dtype=float)
    targ_2d = np.asarray(target[:2], dtype=float)

    # Check start & dock separation against all existing agents
    dist_t0 = float(np.linalg.norm(targ_2d - targ0_2d))
    if not (TARGET_SEP_RANGE[0] <= dist_t0 <= TARGET_SEP_RANGE[1]):
        return None

    for prev in existing_agents:
        prev_init = np.asarray(prev["init_xy"], dtype=float)
        prev_dock = np.asarray(prev["dock_xy"], dtype=float)
        prev_targ = np.asarray(prev["target"][:2], dtype=float)
        if np.linalg.norm(init_xy - prev_init) < START_SEP_MIN:
            return None
        if np.linalg.norm(dock_2d - prev_dock) < DOCK_SEP_MIN:
            return None
        if np.linalg.norm(targ_2d - prev_targ) < TARGET_SEP_RANGE[0]:
            return None
        if np.linalg.norm(init_xy - prev_dock) < START_TO_OTHER_DOCK_MIN:
            return None
        if np.linalg.norm(init_xy - prev_targ) < START_TO_OTHER_DOCK_MIN:
            return None
        if np.linalg.norm(dock_2d - prev_init) < START_TO_OTHER_DOCK_MIN:
            return None
        if np.linalg.norm(targ_2d - prev_init) < START_TO_OTHER_DOCK_MIN:
            return None
        reach_dist, _, _ = segment_to_segment_distance(dock_2d, targ_2d, prev_dock, prev_targ)
        if reach_dist < REACH_SEP_MIN:
            return None
        path_dist, _, _ = segment_to_segment_distance(init_xy, dock_2d, prev_init, prev_dock)
        if path_dist > STRAIGHT_CONFLICT_DIST:
            return None

    # Compute `along`, `lateral_offset`, and `waypoint_xy` using `scenario.py`'s exact formula
    t_vec, n_vec, length = frame_of(init_xy, dock_2d)
    along = float(((center - init_xy) @ t_vec) / length)
    if not (ALONG[0] <= along <= ALONG[1]):
        return None
    d = float(lateral_position(center, init_xy, dock_2d))
    if abs(d) > LATERAL:
        return None
    ambiguous = abs(d) < AMBIG_BAND

    preferred_side = (-1 if d > 0 else 1) if not ambiguous else (1 if float(rng.random()) < 0.5 else -1)
    for side in (preferred_side, -preferred_side):
        waypoint = center + side * (radius + WAYPOINT_CLEAR) * n_vec
        ok, _ = constraints_ok(init_xy, dock_2d, target, center, radius, waypoint)
        if not ok:
            continue
        if any(np.linalg.norm(waypoint - np.asarray(prev["waypoint_xy"])) < WAYPOINT_SEP_MIN
               for prev in existing_agents):
            continue
        return {
            "agent_id": int(agent_idx),
            "init_xy": [float(init_xy[0]), float(init_xy[1])],
            "init_yaw": float(init_yaw),
            "target": [float(v) for v in target],
            "dock_xy": [float(dock_xy[0]), float(dock_xy[1])],
            "dock_bearing": float(dock_bearing),
            "standoff": float(standoff),
            "along": float(along),
            "lateral_offset": float(d),
            "ambiguous": bool(ambiguous),
            "side": int(side),
            "waypoint_xy": [float(waypoint[0]), float(waypoint[1])],
        }
    return None


def sample_multi_agent_scenario(seed, episode, n_agents=DEFAULT_N_AGENTS):
    """Draw one multi-agent scenario using `scenario.py`'s obstacle generation and `TwoSegmentNavigation` planner."""
    if n_agents < 2:
        raise ValueError(f"n_agents must be >= 2 for multi-agent scenarios, got {n_agents}")
    rng = np.random.default_rng([int(seed), int(episode), MULTI_STREAM])

    for attempt in range(1, MAX_SCENARIO_TRIES + 1):
        drawn0 = _sample_agent_zero(rng)
        if drawn0 is None:
            continue
        agent0, obstacle, center, radius = drawn0
        obstacles = [obstacle]

        # Try both detour sides for Agent 0 if needed so Agent 0 and Agent 1 can pass on opposite sides of `obstacle`
        t0, n0, _ = frame_of(agent0["init_xy"], agent0["dock_xy"])
        candidate_sides_0 = [agent0["side"], -agent0["side"]]

        found_agents = None
        for side0 in candidate_sides_0:
            wp0 = center + side0 * (radius + WAYPOINT_CLEAR) * n0
            ok0, _ = constraints_ok(agent0["init_xy"], agent0["dock_xy"], agent0["target"], center, radius, wp0)
            if not ok0:
                continue
            a0_try = {**agent0, "side": int(side0), "waypoint_xy": [float(wp0[0]), float(wp0[1])]}
            agents = [a0_try]
            for idx in range(1, n_agents):
                partner = None
                for _ in range(60):
                    partner = _sample_partner_agent(rng, idx, agents, center, radius)
                    if partner is not None:
                        break
                if partner is None:
                    break
                agents.append(partner)
            if len(agents) == n_agents:
                all_done, min_pair, min_obst, done_time, _ = simulate_kinematic_multi_agent(
                    agents, obstacles=obstacles, dt=0.05, max_time=42.0, reactive=True
                )
                if (all_done and min_pair >= KINEMATIC_CLEAR_MIN
                        and min_obst >= 0.20 and done_time <= KINEMATIC_MAX_NAV_TIME):
                    found_agents = (agents, min_pair, min_obst, done_time)
                    break

        if found_agents is None:
            continue

        agents, min_pair, min_obst, done_time = found_agents
        a0 = agents[0]
        target_sep = float(np.linalg.norm(
            np.asarray(agents[0]["target"][:2]) - np.asarray(agents[1]["target"][:2])
        ))
        dock_sep = float(np.linalg.norm(
            np.asarray(agents[0]["dock_xy"]) - np.asarray(agents[1]["dock_xy"])
        ))
        return Scenario(
            seed=int(seed),
            episode=int(episode),
            target=list(a0["target"]),
            init_xy=list(a0["init_xy"]),
            init_yaw=float(a0["init_yaw"]),
            dock_xy=list(a0["dock_xy"]),
            dock_bearing=float(a0["dock_bearing"]),
            standoff=float(a0["standoff"]),
            obstacles=obstacles,
            along=float(a0["along"]),
            lateral_offset=float(a0["lateral_offset"]),
            ambiguous=bool(a0["ambiguous"]),
            side=int(a0["side"]),
            waypoint_xy=list(a0["waypoint_xy"]),
            attempts=attempt,
            meta={
                "multi_agent": True,
                "n_agents": int(n_agents),
                "agents": agents,
                "target_separation_m": target_sep,
                "dock_separation_m": dock_sep,
                "planned_min_pair_clearance_m": float(min_pair),
                "planned_min_obst_clearance_m": float(min_obst),
                "planned_nav_done_time_s": float(done_time),
            },
        )
    raise RuntimeError(f"no valid multi-agent scenario for seed {seed} episode {episode} in {MAX_SCENARIO_TRIES} tries")


def make_multi_agent_scenarios(seed, n_episodes, noise_frac=0.0, n_agents=DEFAULT_N_AGENTS):
    """Draw multi-agent scenarios 0..n_episodes-1; noisy flag uses stream MULTI_STREAM + 1."""
    out = []
    for ep in range(int(n_episodes)):
        sc = sample_multi_agent_scenario(seed, ep, n_agents=n_agents)
        sc.noisy = bool(np.random.default_rng([int(seed), ep, MULTI_STREAM + 1]).random() < noise_frac)
        out.append(sc)
    return out


# ==============================================================================
# Multi-Agent Simulation Setup & Contact / Clearance Queries
# ==============================================================================
def _configure_robot_instance(sim, client, bot_index, init_xy, init_yaw, fixed_finger_angles):
    """Set wheel dynamics, motor gains, base pose and finger joints for sim.bot[bot_index]."""
    rid = sim.bot[bot_index].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0],
        )
    sim.bot[bot_index].servo_motor_model.set_motor_gains(400, 30)

    spawn_pos, _ = client.getBasePositionAndOrientation(rid)
    client.resetBasePositionAndOrientation(
        rid,
        [float(init_xy[0]), float(init_xy[1]), float(spawn_pos[2])],
        client.getQuaternionFromEuler([0.0, 0.0, float(init_yaw)]),
    )
    client.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    init_angles = list(sim.bot[bot_index].init_joint_angles)
    for k, jid in enumerate(sim.bot[bot_index].active_joint_ids):
        client.setJointMotorControl2(
            bodyIndex=rid, jointIndex=jid, controlMode=client.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0
        )
        client.resetJointState(rid, jid, targetValue=float(init_angles[k]), targetVelocity=0.0)

    finger_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if client.getJointInfo(rid, j)[1].decode() in FINGER_JOINT_NAMES
    ]
    for k, jid in enumerate(finger_joint_ids):
        client.resetJointState(rid, jid, targetValue=float(fixed_finger_angles[k]), targetVelocity=0.0)
    return rid


def setup_multi_simulation(agents_spec, gui=False, obstacles=()):
    """Spawn `len(agents_spec)` Robotnik mobile manipulators in one PyBullet world.

    Returns (sim, client, robot_ids, infos, obstacle_ids).
    """
    conf_file_name = "robotnik_arm.json"
    cur_dir = _TASK_DIR
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=gui)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    n_agents = len(agents_spec)

    # Expand the in-memory config lists to length n_agents and spawn additional SimRobot instances
    conf = copy.deepcopy(sim.bot[0].conf)
    for section in ("robot_pybullet", "robot_pin", "sim"):
        if section not in conf:
            continue
        for key, val in list(conf[section].items()):
            if isinstance(val, list) and len(val) == 1:
                conf[section][key] = [copy.deepcopy(val[0]) for _ in range(n_agents)]
    sim.bot[0].conf = conf
    for idx in range(1, n_agents):
        bot = pb.SimRobot(client, conf, idx, cur_dir)
        sim.bot.append(bot)

    target_finger_angles = configure_gripper_fingers(sim, client, sim.bot[0].bot_pybullet, FIXED_FINGER_ANGLES)

    robot_ids = []
    for idx, ag in enumerate(agents_spec):
        rid = _configure_robot_instance(sim, client, idx, ag["init_xy"], ag["init_yaw"], target_finger_angles)
        robot_ids.append(rid)

    sim.ReceiveObservation()
    obstacle_ids = spawn_obstacles(client, obstacles)

    if gui:
        for idx, ag in enumerate(agents_spec):
            color = list(AGENT_COLORS[idx % len(AGENT_COLORS)])
            target = ag.get("target")
            if target is not None:
                vis_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=color)
                client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, basePosition=list(target))
                client.addUserDebugText(
                    f"Target {idx}", [float(target[0]), float(target[1]), float(target[2]) + 0.08],
                    textColorRGB=color[:3], textSize=1.3,
                )
            client.changeVisualShape(robot_ids[idx], -1, rgbaColor=color)
            client.addUserDebugText(
                f"Robot {idx}", [0.0, 0.0, 0.65], textColorRGB=color[:3], textSize=1.3,
                parentObjectUniqueId=robot_ids[idx], parentLinkIndex=-1,
            )

    ext_names = sim.getNameActiveJoints(0)
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    infos = []
    for idx in range(n_agents):
        dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
        lower_limits, upper_limits = sim.GetBotJointsLimit(idx)
        init_base_pos = np.array(sim.GetBasePosition(idx))
        init_base_ori = np.array(sim.GetBaseOrientation(idx))
        infos.append({
            "agent_id": idx,
            "dyn_model": dyn_model,
            "ee_frame_id": dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector"),
            "shoulder_frame_id": dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2"),
            "joint_lower": np.array(lower_limits),
            "joint_upper": np.array(upper_limits),
            "time_step": sim.GetTimeStep(),
            "init_base_pos": init_base_pos,
            "init_base_bearing": quaternion2bearing(
                init_base_ori[3], init_base_ori[0], init_base_ori[1], init_base_ori[2]
            ),
            "init_q_mes": np.array(sim.GetMotorAngles(idx)),
        })
    return sim, client, robot_ids, infos, obstacle_ids


def multi_robot_contact(client, robot_ids, obstacle_ids=()):
    """Check for any robot-robot or robot-obstacle contact after a simulation step.

    Returns (collided: bool, description: str or None).
    """
    n = len(robot_ids)
    for i in range(n):
        for j in range(i + 1, n):
            contacts = client.getContactPoints(bodyA=robot_ids[i], bodyB=robot_ids[j])
            if contacts:
                link_i = link_name(client, robot_ids[i], int(contacts[0][3]))
                link_j = link_name(client, robot_ids[j], int(contacts[0][4]))
                return True, f"robot{i}:{link_i}<->robot{j}:{link_j}"
    for i in range(n):
        hit, link = robot_obstacle_contact(client, robot_ids[i], obstacle_ids)
        if hit:
            return True, f"robot{i}:{link_name(client, robot_ids[i], link)}<->obstacle"
    return False, None


def min_multi_clearance(client, robot_ids, obstacle_ids=(), distance=CLEARANCE_SEARCH):
    """Smallest surface-to-surface distance (m) across all robot-robot pairs and robot-obstacle pairs."""
    closest = math.inf
    n = len(robot_ids)
    for i in range(n):
        for j in range(i + 1, n):
            pts = client.getClosestPoints(bodyA=robot_ids[i], bodyB=robot_ids[j], distance=distance)
            if pts:
                closest = min(closest, min(float(pt[8]) for pt in pts))
        if obstacle_ids:
            closest = min(closest, min_clearance(client, robot_ids[i], obstacle_ids, distance=distance))
    return closest


# ==============================================================================
# Multi-Agent Data Collection & Closed-Loop Evaluation
# ==============================================================================
MULTI_AGENT_ARRAY_FIELDS = {
    "agents_base_pos": np.float32,         # (T, N, 3)
    "agents_base_ori": np.float32,         # (T, N, 4)
    "agents_base_bearing": np.float32,     # (T, N)
    "agents_q_mes": np.float32,            # (T, N, 17)
    "agents_qd_mes": np.float32,           # (T, N, 17)
    "agents_wheel_cmds": np.float32,       # (T, N, 4)
    "agents_torque_arm": np.float32,       # (T, N, 7)
    "agents_wheel_cmds_exec": np.float32,  # (T, N, 4)
    "agents_torque_arm_exec": np.float32,  # (T, N, 7)
    "agents_q_des_arm": np.float32,        # (T, N, 7)
    "agents_qd_des_arm": np.float32,       # (T, N, 7)
    "agents_phase": np.int8,               # (T, N)
    "agents_nav_done": np.bool_,           # (T, N)
    "agents_ee_pos": np.float32,           # (T, N, 3)
    "agents_segment": np.int8,             # (T, N)
    "agents_lidar": np.float32,            # (T, N, 64)
}


def measure_agents_state(sim, infos):
    """Measure base pose, joint angles/velocities, and forward kinematics EE position for every robot."""
    states = []
    for idx, info in enumerate(infos):
        dyn_model = info["dyn_model"]
        ee_frame_id = info["ee_frame_id"]
        base_pos = np.array(sim.GetBasePosition(idx))
        base_ori = np.array(sim.GetBaseOrientation(idx))
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(idx))
        qd_mes = np.array(sim.GetMotorVelocities(idx))
        base_vel = np.array(sim.GetBaseLinVelocity(idx))[:2]
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        states.append({
            "agent_id": idx,
            "base_pos": base_pos,
            "base_ori": base_ori,
            "base_bearing": float(base_bearing),
            "base_vel": base_vel,
            "q_mes": q_mes,
            "qd_mes": qd_mes,
            "ee_pos": ee_pos,
        })
    return states


def run_multi_episode(scenario, out_path, noise_std=None, noise_hold=0.3, noise_fade=(0.5, 0.15), seed=0, gui=False,
                      max_steps=None, log_every=1, verbose=True):
    """Run the multi-agent expert controllers on `scenario` and save all robots' trajectories to `out_path` (.npz)."""
    if isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)
    wall_start = time.time()
    agents_cfg = get_scenario_agents(scenario)
    n_agents = len(agents_cfg)
    targets = [np.asarray(ag["target"], dtype=float) for ag in agents_cfg]
    if max_steps is None:
        max_steps = STEP_CAP
    noise_std = noise_std or {}
    std_v, std_w, std_tau = (float(noise_std.get(k, 0.0)) for k in ("v", "w", "torque"))
    rng = np.random.default_rng(seed)

    sim, client, rids, infos, obstacle_ids = setup_multi_simulation(
        agents_cfg, gui=gui, obstacles=scenario.obstacles
    )
    try:
        controllers = [RobotExpertController(i, reactive=True) for i in range(n_agents)]
        for i, ctrl in enumerate(controllers):
            ctrl.reset(targets[i], {**infos[i], "scenario": scenario, "agent_scenario": agents_cfg[i]})

        time_step = infos[0]["time_step"]
        stop_criteria = [StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps) for _ in range(n_agents)]
        cmds = [MotorCommands() for _ in range(n_agents)]
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6

        hold_steps = round(noise_hold / time_step)
        base_noises = [legacy.HeldNoise(rng, [std_v, std_w], hold_steps) for _ in range(n_agents)]
        arm_noises = [legacy.HeldNoise(rng, [std_tau] * 7, hold_steps) for _ in range(n_agents)]

        log_single = {name: [] for name in (*legacy.LOG_FIELDS, "segment")}
        log_multi = {name: [] for name in MULTI_AGENT_ARRAY_FIELDS}
        step, current_time = 0, 0.0
        nav_done_times = [None] * n_agents
        collision_time = collision_link = None
        collided = False
        min_clear = math.inf

        while True:
            states = measure_agents_state(sim, infos)
            ee_errors = [float(np.linalg.norm(targets[i] - states[i]["ee_pos"])) for i in range(n_agents)]
            all_xy = [st["base_pos"][:2] for st in states]
            all_yaw = [st["base_bearing"] for st in states]

            wheel_cmds_all, torque_arm_all = [], []
            wheel_exec_all, torque_exec_all = [], []
            q_des_all, qd_des_all = [], []
            phases_all, nav_done_all, segments_all, lidars_all = [], [], [], []

            for i in range(n_agents):
                other_robots = [
                    {
                        "agent_id": j,
                        "base_pos": states[j]["base_pos"],
                        "base_bearing": states[j]["base_bearing"],
                        "base_vel": states[j]["base_vel"],
                        "target": targets[j],
                        "nav_done": controllers[j].navigation.done,
                        "current_goal_pos": controllers[j].current_goal_pos.copy(),
                        "remaining_segments": controllers[j].navigation.remaining_segments(states[j]["base_pos"][:2]),
                    }
                    for j in range(n_agents) if j != i
                ]
                lidar_obs = robot_lidar_obstacles(all_xy, all_yaw, exclude_agent=i,
                                                  static_obstacles=scenario.obstacles)
                scan_i = lidar.scan(states[i]["base_pos"][:2], states[i]["base_bearing"], lidar_obs)

                obs_i = {
                    "t": current_time,
                    "base_pos": states[i]["base_pos"],
                    "base_ori": states[i]["base_ori"],
                    "base_bearing": states[i]["base_bearing"],
                    "q_mes": states[i]["q_mes"],
                    "qd_mes": states[i]["qd_mes"],
                    "target": targets[i],
                    "lidar": scan_i,
                    "other_robots": other_robots,
                    "obstacles": scenario.obstacles,
                }
                w_cmd, tau_cmd = controllers[i].act(obs_i)
                phase = controllers[i].base_state.phase
                nav_done = controllers[i].navigation.done
                if nav_done and nav_done_times[i] is None:
                    nav_done_times[i] = current_time

                w_exec = w_cmd.copy()
                if not nav_done:
                    goal_dist = np.linalg.norm(states[i]["base_pos"][:2] - controllers[i].current_goal_pos)
                    dv, dw = base_noises[i].sample() * legacy.fade(goal_dist, noise_fade[0])
                    w_exec += legacy.wheel_speeds(dv, dw, controllers[i].base_params)
                arm_scale = legacy.fade(ee_errors[i], noise_fade[1]) if nav_done else 1.0
                tau_exec = np.clip(
                    tau_cmd + arm_noises[i].sample() * arm_scale,
                    -controllers[i].arm_params.max_torque,
                    controllers[i].arm_params.max_torque,
                )

                wheel_cmds_all.append(w_cmd)
                torque_arm_all.append(tau_cmd)
                wheel_exec_all.append(w_exec)
                torque_exec_all.append(tau_exec)
                q_des_all.append(controllers[i].q_des_arm.copy())
                qd_des_all.append(controllers[i].qd_des_arm.copy())
                phases_all.append(legacy.PHASE_CODES["reach"] if nav_done else legacy.PHASE_CODES[phase])
                nav_done_all.append(nav_done)
                segments_all.append(controllers[i].segment)
                lidars_all.append(scan_i)

                cmds[i].SetControlCmd(np.concatenate((w_exec, tau_exec, np.zeros(6))), control_modes)

            if step % log_every == 0:
                log_single["t"].append(current_time)
                log_single["base_pos"].append(states[0]["base_pos"])
                log_single["base_ori"].append(states[0]["base_ori"])
                log_single["base_bearing"].append(states[0]["base_bearing"])
                log_single["q_mes"].append(states[0]["q_mes"])
                log_single["qd_mes"].append(states[0]["qd_mes"])
                log_single["wheel_cmds"].append(wheel_cmds_all[0])
                log_single["torque_arm"].append(torque_arm_all[0])
                log_single["wheel_cmds_exec"].append(wheel_exec_all[0])
                log_single["torque_arm_exec"].append(torque_exec_all[0])
                log_single["q_des_arm"].append(q_des_all[0])
                log_single["qd_des_arm"].append(qd_des_all[0])
                log_single["phase"].append(phases_all[0])
                log_single["nav_done"].append(nav_done_all[0])
                log_single["ee_pos"].append(states[0]["ee_pos"])
                log_single["segment"].append(segments_all[0])

                log_multi["agents_base_pos"].append([st["base_pos"] for st in states])
                log_multi["agents_base_ori"].append([st["base_ori"] for st in states])
                log_multi["agents_base_bearing"].append([st["base_bearing"] for st in states])
                log_multi["agents_q_mes"].append([st["q_mes"] for st in states])
                log_multi["agents_qd_mes"].append([st["qd_mes"] for st in states])
                log_multi["agents_wheel_cmds"].append(wheel_cmds_all)
                log_multi["agents_torque_arm"].append(torque_arm_all)
                log_multi["agents_wheel_cmds_exec"].append(wheel_exec_all)
                log_multi["agents_torque_arm_exec"].append(torque_exec_all)
                log_multi["agents_q_des_arm"].append(q_des_all)
                log_multi["agents_qd_des_arm"].append(qd_des_all)
                log_multi["agents_phase"].append(phases_all)
                log_multi["agents_nav_done"].append(nav_done_all)
                log_multi["agents_ee_pos"].append([st["ee_pos"] for st in states])
                log_multi["agents_segment"].append(segments_all)
                log_multi["agents_lidar"].append(lidars_all)

            sim.Step(cmds, "torque")

            if step % 50 == 0:
                min_clear = min(min_clear, min_multi_clearance(client, rids, obstacle_ids))
            collided, link_desc = multi_robot_contact(client, rids, obstacle_ids)
            if collided:
                min_clear = min(min_clear, min_multi_clearance(client, rids, obstacle_ids))
                collision_time = (step + 1) * time_step
                collision_link = link_desc
                stop_message = f"Multi-agent contact at t={collision_time:.3f}s ({collision_link})"
                break

            if gui:
                keys = client.getKeyboardEvents()
                if ord("q") in keys and keys[ord("q")] & client.KEY_WAS_TRIGGERED:
                    stop_message = "Stopped by user"
                    break
                time.sleep(0.001)

            agent_msgs = [
                check_stop(stop_criteria[i], current_time, time_step, ee_errors[i], nav_done_all[i])
                for i in range(n_agents)
            ]
            all_reached = (
                all(nav_done_all[i] and ee_errors[i] < REACH_TOLERANCE for i in range(n_agents))
                and current_time > stop_criteria[0].min_time
            )
            if all_reached:
                stop_message = (
                    f"Target reached by all {n_agents} agents within {REACH_TOLERANCE * 1000:.0f}mm "
                    f"at t={current_time:.2f}s (max EE error {max(ee_errors):.4f}m)!"
                )
                break
            if current_time / time_step >= max_steps:
                stop_message = f"Reached step cap {max_steps} at t={current_time:.2f}s"
                break
            if all(msg is not None for msg in agent_msgs):
                stop_message = next(msg for msg in agent_msgs if not msg.startswith("Target reached"))
                break

            current_time += time_step
            step += 1
            if verbose and step % 1000 == 0:
                err_str = ", ".join(f"R{i}:{ee_errors[i]:.4f}m" for i in range(n_agents))
                print(f"t={current_time:.2f}s | EE errors [{err_str}] | nav_done {nav_done_all}")

        max_ee_error = float(max(ee_errors))
        status = "collision" if collided else legacy.stop_status(stop_message)
        if status == "unreachable" and max_ee_error < REACH_TOLERANCE:
            status = "reached"
        nav_done_time = max((t for t in nav_done_times if t is not None), default=None)
        meta = {
            "target": targets[0].tolist(),
            "targets": [t.tolist() for t in targets],
            "init_base_pos": infos[0]["init_base_pos"].tolist(),
            "init_base_yaw": float(infos[0]["init_base_bearing"]),
            "status": status,
            "success": status == "reached",
            "stop_message": stop_message,
            "final_error": max_ee_error,
            "agent_final_errors": [float(e) for e in ee_errors],
            "duration": float(current_time),
            "n_steps": step + 1,
            "log_every": log_every,
            "time_step": time_step,
            "nav_done_time": nav_done_time,
            "agent_nav_done_times": nav_done_times,
            "nav_timed_out": any(t is not None and t > NAV_TIMEOUT for t in nav_done_times),
            "standoff": float(scenario.standoff),
            "base_goal_pos": list(scenario.dock_xy),
            "base_goal_bearing": float(scenario.dock_bearing),
            "noise_v": std_v,
            "noise_w": std_w,
            "noise_torque": std_tau,
            "noise_hold": noise_hold,
            "noise_fade_base": noise_fade[0],
            "noise_fade_arm": noise_fade[1],
            "seed": seed,
            "scenario": scenario.to_dict(),
            "no_obstacle": False,
            "multi_agent": True,
            "n_agents": n_agents,
            "collided": bool(collided),
            "collision_time": collision_time,
            "collision_link": collision_link,
            "min_clearance": float(min_clear),
            "segment_switch_time": controllers[0].segment_switch_time,
            "ambiguous": bool(scenario.ambiguous),
            "side": int(scenario.side),
            "lateral_offset": float(scenario.lateral_offset),
            "obstacle_r": float(scenario.obstacles[0]["r"]) if scenario.obstacles else 0.0,
            "noisy": bool(scenario.noisy),
        }
        arrays = {
            name: np.asarray(log_single[name], dtype=dtype)
            for name, dtype in (*legacy.LOG_FIELDS.items(), ("segment", np.int8))
        }
        arrays.update({
            name: np.asarray(log_multi[name], dtype=dtype)
            for name, dtype in MULTI_AGENT_ARRAY_FIELDS.items()
        })
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        temporary = str(out_path) + ".tmp"
        with open(temporary, "wb") as f:
            np.savez_compressed(
                f, **arrays, target=targets[0], targets=np.asarray(targets, dtype=np.float64), meta=json.dumps(meta)
            )
        os.replace(temporary, out_path)
        if verbose:
            print(f"{stop_message}\nSaved {len(arrays['t'])} samples ({n_agents} agents) to {out_path}")
        return {**meta, "file": os.path.basename(out_path), "wall_time": time.time() - wall_start}
    finally:
        client.disconnect()


def run_multi_policy_episode(policy, episode, gui=False, keep_traj=False, early_stop=True, verbose=True):
    """Evaluate a multi-agent episode under protocol v2.0 (supports --gui and --compare-recorded)."""
    from obstacle_task import evaluate as ev

    wall_start = time.time()
    scenario = episode["scenario"]
    if isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)
    agents_cfg = get_scenario_agents(scenario)
    n_agents = len(agents_cfg)
    targets = [np.asarray(ag["target"], dtype=float) for ag in agents_cfg]

    sim, client, rids, infos, obstacle_ids = setup_multi_simulation(
        agents_cfg, gui=gui, obstacles=scenario.obstacles
    )
    dt = infos[0]["time_step"]

    is_expert = type(policy) is ExpertPolicy or isinstance(policy, (RobotExpertController, MultiAgentExpertPolicy))
    if is_expert:
        controllers = [RobotExpertController(i, reactive=True) for i in range(n_agents)]
        for i, ctrl in enumerate(controllers):
            ctrl.reset(targets[i], {**infos[i], "scenario": scenario, "agent_scenario": agents_cfg[i]})
    else:
        controllers = [copy.deepcopy(policy) for _ in range(n_agents)]
        try:
            for i, ctrl in enumerate(controllers):
                ctrl.reset(targets[i].copy(), ev.sanitize_policy_info(infos[i]))
        except Exception as exc:
            traceback.print_exc()
            client.disconnect()
            return {
                **v1.empty_v11_columns(), **ev.empty_v2_columns(), **ev.scenario_columns(scenario),
                "success": False, "stop_reason": "policy_error", "error_message": repr(exc),
                "wall_time": time.time() - wall_start,
            }, None

    step_cap = round(ev.T_MAX / dt)
    hold_steps = round(v1.HOLD_TIME / dt)
    hold_ext_steps = round(v1.HOLD_EXT_TIME / dt)
    stall_steps = round(v1.STALL_TIME / dt)
    clearance_steps = max(1, round(ev.CLEARANCE_INTERVAL / dt))
    cmds = [MotorCommands() for _ in range(n_agents)]
    control_modes = ["velocity"] * 4 + ["torque"] * 13

    step, current_time = 0, 0.0
    reach_step, hold_errors = None, []
    best_error, last_progress_step, minimum_error = np.inf, 0, np.inf
    prev_xy = [infos[i]["init_base_pos"][:2].copy() for i in range(n_agents)]
    prev_yaw = [infos[i]["init_base_bearing"] for i in range(n_agents)]
    goal_pos = [np.asarray(ag["dock_xy"], dtype=float) for ag in agents_cfg]
    standoff = float(agents_cfg[0]["standoff"])
    q_init_arm = [infos[i]["init_q_mes"][4:11].copy() for i in range(n_agents)]

    path_length, turn_total = 0.0, 0.0
    goal_dist_start = float(max(np.linalg.norm(prev_xy[i] - goal_pos[i]) for i in range(n_agents)))
    goal_dist_min = goal_dist_start
    arm_move_time, goal_dist_at_arm_move = None, np.nan
    saturated_steps, wheel_max = 0, 0.0
    policy_times = []
    traj = {name: [] for name in ev.TRAJ_FIELDS} if keep_traj else None
    stop_reason, error_message, hold_error_message = None, "", ""

    approach = v1.ApproachDiagnostics(targets[0], standoff, dt)
    stable = [v1.StableArrival(tol, dt, horizon=ev.STABLE_HORIZON) for tol in v1.STABLE_TOLS]
    hold10, primary_row = None, None
    collision, collision_time, collision_link = False, np.nan, None
    clearance = np.inf
    hold_completed = False
    passed_inter = False
    side_taken = 0
    obst_t, _, _ = frame_of(scenario.init_xy, scenario.dock_xy)
    obst_along = float((np.array([scenario.obstacles[0]["x"], scenario.obstacles[0]["y"]]) - scenario.init_xy) @ obst_t) if scenario.obstacles else np.nan

    def primary_columns():
        times_ms = np.array(policy_times) * 1000.0 if policy_times else np.array([np.nan])
        hold = np.array(hold_errors) if hold_errors else np.array([np.nan])
        return {
            "success": stop_reason == "reached",
            "t_reach": reach_step * dt if reach_step is not None else np.nan,
            "settled": stop_reason == "reached" and hold_completed and bool(np.max(hold) < v1.SETTLE_TOL),
            "hold_error": float(np.mean(hold)),
            "hold_error_max": float(np.max(hold)),
            "min_error": float(minimum_error),
            "final_error": float(error),
            "stop_reason": stop_reason,
            "duration": current_time,
            "n_steps": step,
            "standoff": standoff,
            "base_goal_dist_start": goal_dist_start,
            "base_goal_dist_min": goal_dist_min,
            "base_goal_dist_final": goal_dist,
            "base_path_length": path_length,
            "base_turn_total_deg": float(np.rad2deg(turn_total)),
            "arm_move_time": arm_move_time if arm_move_time is not None else np.nan,
            "base_goal_dist_at_arm_move": goal_dist_at_arm_move,
            "torque_saturated_frac": saturated_steps / max(1, len(policy_times)),
            "wheel_cmd_max": wheel_max,
            "policy_ms_mean": float(np.mean(times_ms)),
            "policy_ms_p99": float(np.percentile(times_ms, 99)),
            "policy_ms_max": float(np.max(times_ms)),
            "error_message": error_message,
        }

    while True:
        states = measure_agents_state(sim, infos)
        ee_errors = [float(np.linalg.norm(targets[i] - states[i]["ee_pos"])) for i in range(n_agents)]
        error = float(max(ee_errors))

        if primary_row is None:
            minimum_error = min(minimum_error, error)
            for i in range(n_agents):
                path_length += float(np.linalg.norm(states[i]["base_pos"][:2] - prev_xy[i])) / n_agents
                turn_total += abs(wrap_angle(states[i]["base_bearing"] - prev_yaw[i])) / n_agents
                prev_xy[i] = states[i]["base_pos"][:2].copy()
                prev_yaw[i] = states[i]["base_bearing"]
            goal_dist = float(max(np.linalg.norm(states[i]["base_pos"][:2] - goal_pos[i]) for i in range(n_agents)))
            goal_dist_min = min(goal_dist_min, goal_dist)
            if arm_move_time is None and np.max(np.abs(states[0]["q_mes"][4:11] - q_init_arm[0])) > v1.ARM_MOVE_ANGLE:
                arm_move_time, goal_dist_at_arm_move = current_time, goal_dist
            approach.update(
                step, states[0]["ee_pos"], states[0]["base_pos"], states[0]["base_ori"],
                states[0]["base_bearing"], sim.GetBaseLinVelocity(0), sim.GetBaseAngVelocity(0)[2],
            )
            if not passed_inter and scenario.obstacles:
                if float((states[0]["base_pos"][:2] - scenario.init_xy) @ obst_t) > obst_along:
                    passed_inter = True
                    lat = lateral_position(states[0]["base_pos"][:2], scenario.init_xy, scenario.dock_xy)
                    side_taken = 1 if lat >= 0.0 else -1

        for detector in stable:
            detector.update(step, error)

        collision, link_desc = multi_robot_contact(client, rids, obstacle_ids)
        if step % clearance_steps == 0 or collision:
            clearance = min(clearance, min_multi_clearance(client, rids, obstacle_ids))
        if collision:
            collision_time, collision_link = step * dt, link_desc
            if hold10 is not None:
                if primary_row is None:
                    hold_errors.append(error)
                hold10.update(error, states[0]["base_pos"][:2], states[0]["base_bearing"])
                hold10.stop = "collision"
            if primary_row is None:
                stop_reason = "reached" if reach_step is not None else "collision"
            break

        if primary_row is None:
            if reach_step is None and error < v1.REACH_TOL and step <= step_cap:
                reach_step = step
                hold10 = v1.ExtendedHold(states[0]["base_pos"][:2], states[0]["base_bearing"])
            if reach_step is not None:
                hold_errors.append(error)
                hold10.update(error, states[0]["base_pos"][:2], states[0]["base_bearing"])
                if step - reach_step >= hold_steps:
                    hold_completed, stop_reason = True, "reached"
                    primary_row = primary_columns()
            else:
                tipped = any(
                    max(abs(r), abs(p_ang)) > v1.TIP_ANGLE
                    for i in range(n_agents)
                    for r, p_ang, _ in [client.getEulerFromQuaternion(states[i]["base_ori"])]
                )
                if tipped:
                    stop_reason = "tipped"
                    break
                if step >= step_cap:
                    stop_reason = "timeout"
                    break
                if error < best_error - v1.STALL_IMPROVEMENT:
                    best_error, last_progress_step = error, step
                moving = any(
                    np.linalg.norm(np.asarray(sim.GetBaseLinVelocity(i))[:2]) > v1.STILL_LIN
                    or abs(sim.GetBaseAngVelocity(i)[2]) > v1.STILL_ANG
                    or np.max(np.abs(states[i]["qd_mes"][4:11])) > v1.STILL_ARM
                    for i in range(n_agents)
                )
                if moving or error < v1.STALL_MIN_ERROR:
                    last_progress_step = step
                if early_stop and step - last_progress_step >= stall_steps:
                    stop_reason = "stalled"
                    break
        else:
            hold10.update(error, states[0]["base_pos"][:2], states[0]["base_bearing"])
            tipped = any(
                max(abs(r), abs(p_ang)) > v1.TIP_ANGLE
                for i in range(n_agents)
                for r, p_ang, _ in [client.getEulerFromQuaternion(states[i]["base_ori"])]
            )
            if tipped:
                hold10.stop = "tipped"
                break
            if step - reach_step >= hold_ext_steps:
                hold10.stop = "completed"
                break

        all_xy = [st["base_pos"][:2] for st in states]
        all_yaw = [st["base_bearing"] for st in states]
        wheel_cmds_step, torque_arm_step, ranges_0 = [], [], None

        try:
            t0 = time.perf_counter()
            for i in range(n_agents):
                lidar_obs = robot_lidar_obstacles(all_xy, all_yaw, exclude_agent=i,
                                                  static_obstacles=scenario.obstacles)
                obs_i = ev.build_observation(
                    current_time, states[i]["base_pos"], states[i]["base_ori"], states[i]["base_bearing"],
                    states[i]["q_mes"], states[i]["qd_mes"], targets[i], lidar_obs,
                )
                if i == 0 and keep_traj:
                    ranges_0 = obs_i["lidar"].copy()
                if is_expert:
                    obs_i["other_robots"] = [
                        {
                            "agent_id": j,
                            "base_pos": states[j]["base_pos"],
                            "base_bearing": states[j]["base_bearing"],
                            "base_vel": states[j]["base_vel"],
                            "target": targets[j],
                            "nav_done": controllers[j].navigation.done,
                            "current_goal_pos": controllers[j].current_goal_pos.copy(),
                            "remaining_segments": controllers[j].navigation.remaining_segments(states[j]["base_pos"][:2]),
                        }
                        for j in range(n_agents) if j != i
                    ]
                    obs_i["obstacles"] = scenario.obstacles
                w_cmd, tau_cmd = controllers[i].act(obs_i)
                w_cmd = np.array(w_cmd, dtype=float).reshape(4)
                tau_cmd = np.array(tau_cmd, dtype=float).reshape(7)
                wheel_cmds_step.append(w_cmd)
                torque_arm_step.append(tau_cmd)
            policy_times.append(time.perf_counter() - t0)
        except Exception as exc:
            traceback.print_exc()
            if primary_row is None:
                stop_reason, error_message = "policy_error", repr(exc)
            else:
                hold_error_message = f"extended hold: {exc!r}"
            if hold10 is not None:
                hold10.stop = "policy_error"
            break

        if not all(np.all(np.isfinite(w)) and np.all(np.isfinite(t))
                   for w, t in zip(wheel_cmds_step, torque_arm_step)):
            if primary_row is None:
                stop_reason = "invalid_action"
            if hold10 is not None:
                hold10.stop = "invalid_action"
            break

        for i in range(n_agents):
            if np.max(np.abs(torque_arm_step[i])) >= v1.TORQUE_LIMIT:
                saturated_steps += 1
            wheel_max = max(wheel_max, float(np.max(np.abs(wheel_cmds_step[i]))))
            torque_clipped = np.clip(torque_arm_step[i], -v1.TORQUE_LIMIT, v1.TORQUE_LIMIT)
            cmds[i].SetControlCmd(np.concatenate((wheel_cmds_step[i], torque_clipped, np.zeros(6))), control_modes)
            if i == 0:
                torque_arm_step[0] = torque_clipped
                if primary_row is None:
                    approach.command(wheel_cmds_step[0])
                if hold10 is not None:
                    hold10.command(wheel_cmds_step[0])

        if keep_traj:
            traj["t"].append(current_time)
            traj["base_pos"].append(states[0]["base_pos"].copy())
            traj["base_ori"].append(states[0]["base_ori"].copy())
            traj["base_bearing"].append(states[0]["base_bearing"])
            traj["q_mes"].append(states[0]["q_mes"].copy())
            traj["qd_mes"].append(states[0]["qd_mes"].copy())
            traj["ee_pos"].append(states[0]["ee_pos"].copy())
            traj["wheel_cmds"].append(wheel_cmds_step[0].copy())
            traj["torque_arm"].append(torque_arm_step[0].copy())
            traj["lidar"].append(ranges_0)
            traj["segment"].append(int(getattr(controllers[0], "segment", -1)))

        sim.Step(cmds, "torque")
        if gui:
            time.sleep(0.001)
        step += 1
        current_time = step * dt

    client.disconnect()
    if primary_row is None:
        primary_row = primary_columns()
    approach_columns = approach.columns()
    dock_time = approach_columns["parked_time"]
    row = {
        **primary_row,
        **(hold10.columns(dt) if hold10 is not None else v1.ExtendedHold.empty_columns()),
        **v1.stable_columns(stable),
        "run_end_time": current_time,
        **approach_columns,
        "wall_time": time.time() - wall_start,
        "collision": bool(collision),
        "collision_time": float(collision_time) if collision else np.nan,
        "collision_link": collision_link,
        "min_clearance": float(clearance),
        "passed_obstacle": bool(passed_inter),
        "side_taken": int(side_taken),
        "docked": approach_columns["parked"],
        "time_to_dock": dock_time,
        "docked_clean": approach_columns["parked"] and (not collision or dock_time < collision_time),
        **ev.scenario_columns(scenario),
        "policy_latched_side": getattr(controllers[0], "latched_side", None),
        "policy_p_left": getattr(controllers[0], "p_left", None),
    }
    if hold_error_message and not row["error_message"]:
        row["error_message"] = hold_error_message
    if verbose:
        print(f"Stopped: {row['stop_reason']} at t={row['duration']:.2f}s, success {row['success']}, "
              f"min error {row['min_error']:.4f}m, collision {collision}; hold {row['hold10_stop'] or '-'}")
    if keep_traj:
        traj = {name: np.asarray(values, dtype=np.int8 if name == "segment" else float)
                for name, values in traj.items()}
    return row, traj
