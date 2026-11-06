"""Moving-obstacle task: a one-cost (DWA-style) avoidance expert, its training collection and the closed-loop test.

Task (`sample_dynamic_scenario`): start pose and target as the static-obstacle task (target 3.5-5 m away), base goal
("dock") from the reaching expert's standoff rule, no static obstacle. One cylinder (r 0.25-0.35 m, h 1.5 m) moves on a
straight line at a constant speed (OBST_SPEED, 0.10-0.25 m/s, never faster than the base) across the start->dock
segment, at a fraction ALONG of it and an angle CROSS_ANGLE_DEG; it is timed to reach the crossing point when the
undisturbed robot would (+- MEET_JITTER_S), so the two really meet. The crossing point is >= 1 m from the start and the
dock, so the in-place turns at both ends are never threatened.

Expert (`DynamicExpert`, mode "dwa"): goal seeking and avoidance in ONE cost, as open-source local planners do
(PythonRobotics DWA, ir-sim VO/RVO, TEB). Every DWA_PERIOD (50 ms) the candidates of `dwa_rollouts` are integrated for
up to DWA_HORIZON s with the cylinder moving at its current velocity, and the cheapest wins (`dwa_command`): closest
approach to the dock, distance at the end, heading error, clearance between a 3-circle footprint and the cylinder
(over the rollout and at its end), a cost-to-go check (a detour that leaves the way back blocked is no better than
waiting), idle time, turning while driving, command changes, reversing. The template's own align-then-drive law is one
of the candidates, so far from the cylinder the data equal the reaching expert's; the others are wait, slower, reverse,
and in-place turn to a detour heading then straight. The template's state machine still takes over at the dock for the
final orientation and keeps the timeout; the arm is the unchanged arm_controller once navigation is done.
Two properties of the simulator shaped the design: the base slides sideways when it turns while driving (lateral speed
~ v*w, hardly damped: 0.36 m/s for 20 s with the wheels at zero), so every candidate separates turning (in place) from
driving (straight, heading held); and the base keeps rotating after an in-place turn, so a yaw-rate feedback is added.
The expert reads the cylinder's position and velocity from the scenario (privileged); a learner sees them only through
the 64-beam lidar (velocity needs two scans or memory). Mode "blind" is the plain reaching template (control).
Documented in docs/07_obstacles.md, section 7.2.

Modes of use (default pixi env of RoboEnv, from the project root; the group's obstacle task code must be importable,
see OBSTACLE_TASK_DIR below):
  expert, development scenes with GIFs:  python scripts/obstacle/dynamic_obstacle.py --seed 1 --episodes 0-11 --gif --out-dir DIR
  training collection (DART noise):      python scripts/obstacle/dynamic_obstacle.py --seed 3 --episodes 0-1599 --dart --workers 28 --out-dir DIR
  closed-loop test of a learned policy:  python scripts/obstacle/dynamic_obstacle.py --seed 4 --episodes 0-99 --policy RUN_DIR --workers 28 --out-dir DIR
Each run writes one ep_XXXX.npz per episode (the group's LOG_FIELDS + lidar (T, 64), obstacle_xy (T, 2)), index.csv,
scenarios.json and config.json: the set layout scripts/obstacle/convert_obstacle_dataset.py turns into a training dataset.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

# The group's obstacle task code: controlled_experiments/ of this repository (obstacle_task package, collect_data.py,
# evaluate_policy.py; identical to branch feature/step9-obstacle-lidar, folder final/robotnik_arm_task).
TASK_DIR = os.environ.get("OBSTACLE_TASK_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "controlled_experiments"))
if not os.path.isdir(os.path.join(TASK_DIR, "obstacle_task")):
    raise SystemExit(f"obstacle_task not found in {TASK_DIR}: set OBSTACLE_TASK_DIR to a folder holding the group's "
                     "obstacle_task package, collect_data.py and evaluate_policy.py")
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import pinocchio as pin  # noqa: E402
from simulation_and_control import MotorCommands, wrap_angle  # noqa: E402

import collect_data as legacy  # noqa: E402
from evaluate_policy import expert_base_goal, setup_simulation  # noqa: E402
from robotnik_arm_control import (  # noqa: E402
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria, arm_controller, base_controller,
    check_stop, quaternion2bearing,
)
from robotnik_arm_task import ARM_REACH_LIMIT, REACH_TOLERANCE  # noqa: E402
from obstacle_task import lidar  # noqa: E402
from obstacle_task.expert import NAV_TIMEOUT  # noqa: E402
from obstacle_task.scenario import INIT_RADIUS, INIT_YAW_DEG, OBST_HEIGHT, OBST_RADIUS, TARGET_R, TARGET_THETA_DEG, TARGET_Z  # noqa: E402
from obstacle_task.sim import link_name, min_clearance, robot_obstacle_contact, spawn_obstacles  # noqa: E402

STREAM = 23                        # per-episode SeedSequence entry (static-obstacle task: 9)
ALONG = (0.35, 0.65)               # crossing point as a fraction of the start->dock segment
CROSS_ANGLE_DEG = (60.0, 120.0)    # angle between the cylinder's line and the start->dock segment
OBST_SPEED = (0.10, 0.25)          # m/s (<= the base's 0.25 m/s: a faster body coming from behind cannot be escaped)
MEET_JITTER_S = (-2.0, 2.0)        # s, obstacle arrival at the crossing point relative to the robot's nominal one
ROBOT_NOMINAL_SPEED = 0.25         # m/s, base_controller max_lin_vel: nominal arrival time = distance / speed
ALIGN_TIME_S = 1.5                 # s, allowance for the initial in-place turn in the nominal arrival time
END_CLEAR = 1.0                    # m, the cylinder's path (surface) to the start and to the dock
OBST_RUN_IN = 2.5                  # m, the cylinder starts this far from the crossing point at least
STEP_CAP = 75000                   # 75 s at 1 kHz
DYN_NAV_TIMEOUT = 60.0             # s, navigation timeout (static obstacle: 45 s; the avoidance and the recovery take longer)
HOLD_S = 10.0                      # hold tail after the first reach (as our hold sets), 0 = stop at the reach

# --dart: the static-obstacle set F noise for a training collection:
# executed command = expert label + collect_data.HeldNoise, labels stay clean. Base (v, w) noise during navigation,
# faded linearly to zero within DART_FADE_BASE of the dock; once navigation is done, DART_HOLD_WHEEL per wheel side
# whenever the expert's wheel command is exactly zero (the hold, as collect_static_obstacle.py); arm torque noise throughout.
DART_V, DART_W, DART_TORQUE = 0.0389, 0.1178, 1.0   # m/s, rad/s, N*m
DART_HOLD_S = 0.05                 # s per sample
DART_FADE_BASE = 0.5               # m
DART_HOLD_WHEEL = 0.5              # rad/s per wheel side
DART_STREAM = 24                   # per-episode SeedSequence entry of the noise (the scenario uses STREAM)
TORQUE_LIMIT = 30.0                # N*m, the expert's envelope

# The expert: goal seeking and avoidance in ONE cost, as DWA / VO planners do (PythonRobotics DWA, ir-sim RVO, TEB):
# no nominal controller to fall back to, so there is no "recovery" phase after an avoidance.
ROLLOUT_DT = 0.125                 # s, integration step of the candidate rollouts
FOOTPRINT = ((0.00, 0.36), (0.38, 0.22), (0.68, 0.16))   # (forward offset, radius) m: chassis, arm, wrist
DWA_PERIOD = 0.05                  # s, decision period (20 Hz); the command is held in between
DWA_HORIZON = 8.0                  # s, rollout horizon far from the dock (long enough to see that fleeing alongside the cylinder never ends); shrinks to DWA_MIN_HORIZON as the dock nears
DWA_MIN_HORIZON = 3.5              # s (a half turn in place at 1 rad/s must fit: the cost is myopic otherwise)
DWA_MARGIN = 0.25                  # m, footprint-cylinder clearance kept over the rollout (0.40 made the base flee alongside the cylinder)
DWA_W_GOAL = 2.0                   # per m of the closest approach of the rollout to the dock
DWA_W_HEADING = 1.0                # per rad of heading error at the closest approach (to the dock, or the dock bearing within 0.15 m)
DWA_W_END = 0.5                    # per m between the rollout's end and the dock (moving away costs; reaching it slowly is preferred)
DWA_W_FINAL = 0.05                 # per rad of dock-bearing error when the rollout reaches the dock (tie-breaker: the orient phase corrects it; 0.3 stalled the base 0.17 m short of the dock after a detour)
DWA_W_CLEAR = 20.0                 # per m of margin violation (closest approach over the rollout)
DWA_W_CLEAR_END = 20.0             # per m below DWA_MARGIN_END at the end of the rollout (the conflict must be resolved by then: no fleeing alongside the cylinder)
DWA_MARGIN_END = 0.50              # m
DWA_W_BLOCK = 10.0                 # per m of margin violation on the way back to the dock after the rollout (cost-to-go: a detour that leaves the return blocked is no better than waiting)
DWA_BLOCK_HORIZON = 8.0            # s of that straight return (after an in-place turn at DWA_TURN_RATE) that are checked
DWA_HIT = 10.0                     # extra when the rollout comes within DWA_HIT_GAP of the cylinder
DWA_HIT_GAP = 0.10                 # m (the unicycle rollout ignores the base's momentum: keep a reserve)
DWA_W_TURN = 3.0                   # per rad/s of yaw rate while driving (the skid-steer slip builds up when turning while driving)
DWA_W_SMOOTH = 0.1                 # per unit change of (v, w) from the held command
DWA_SLIP_FACTOR = 1.3              # slip counter: measured base speed above this times the commanded speed + 0.08 m/s (v > 0.05 only).
                                   # Zero wheel commands do not stop a sliding base here (the wheels free-wheel and the lateral friction
                                   # is nearly zero: 0.36 m/s sideways for 20 s with the wheels at 0), so sliding is prevented, not cured:
                                   # DWA_W_TURN makes turning while driving expensive (lateral speed builds at about v * w).
DWA_REVERSE = 0.3
DWA_W_WAIT = 0.1                   # per s of idle time at the start of a rollout (otherwise "wait 1 s then go" beats "go" at every decision: the base procrastinates at 0.13 m from the dock)
DWA_HYSTERESIS = 0.02
DWA_TURN_RATE = 0.6                # rad/s, in-place turns (the template uses 1.0; the base keeps rotating after a fast turn here)
DWA_HOLD_GAIN = 1.5                # heading hold while driving straight: w = clip(gain * (yaw_ref - yaw), +-0.3), as the template steers
DWA_YAW_DAMP = 0.8                 # yaw-rate feedback on the executed command: w += gain * (w - measured yaw rate)


def sample_dynamic_scenario(seed, episode):
    """One scenario: start, target, dock (reaching expert) and the cylinder's straight-line motion."""
    rng = np.random.default_rng([int(seed), int(episode), STREAM])
    for attempt in range(1, 10001):
        xy, yaw = legacy.sample_init_poses(1, rng, INIT_RADIUS, INIT_YAW_DEG)
        r, theta, z = legacy.sample_targets(1, rng, TARGET_R, TARGET_THETA_DEG, TARGET_Z)
        init_xy, init_yaw = np.asarray(xy[0], float), float(yaw[0])
        target = np.asarray(legacy.relative_to_world(init_xy, init_yaw, float(r[0]), float(theta[0]), float(z[0])), float)
        dock_xy, dock_bearing, standoff = expert_base_goal(target, np.r_[init_xy, 0.0], init_yaw)
        seg = dock_xy - init_xy
        length = float(np.linalg.norm(seg))
        t = seg / length
        n = np.array([-t[1], t[0]])
        along = float(rng.uniform(*ALONG))
        cross = init_xy + along * length * t
        if min(along * length, (1 - along) * length) < END_CLEAR:
            continue
        ang = math.radians(float(rng.uniform(*CROSS_ANGLE_DEG))) * float(rng.choice([-1.0, 1.0]))
        direction = math.cos(ang) * t + math.sin(ang) * n          # unit direction of motion of the cylinder
        speed = float(rng.uniform(*OBST_SPEED))
        radius = float(rng.uniform(*OBST_RADIUS))
        t_robot = ALIGN_TIME_S + along * length / ROBOT_NOMINAL_SPEED
        t_meet = max(t_robot + float(rng.uniform(*MEET_JITTER_S)), OBST_RUN_IN / speed)
        start = cross - direction * speed * t_meet
        end = cross + direction * speed * 30.0
        path_clear = min(_segment_distance(start, end, init_xy), _segment_distance(start, end, dock_xy)) - radius
        if path_clear < END_CLEAR:                                   # the cylinder never threatens the in-place turns
            continue
        return {
            "seed": int(seed), "episode": int(episode), "target": target.tolist(), "init_xy": init_xy.tolist(),
            "init_yaw": init_yaw, "dock_xy": dock_xy.tolist(), "dock_bearing": float(dock_bearing),
            "standoff": float(standoff), "along": along, "cross_xy": cross.tolist(), "cross_angle_rad": ang,
            "obstacle": {"start_xy": start.tolist(), "dir": direction.tolist(), "speed": speed, "r": radius,
                         "h": OBST_HEIGHT, "t_meet": float(t_meet)},
            "nominal_meet_time_s": float(t_robot), "attempts": attempt,
        }
    raise RuntimeError(f"no valid scenario for seed {seed} episode {episode}")


def _segment_distance(a, b, p):
    ab = b - a
    u = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(a + u * ab - p))


def obstacle_velocity(scenario):
    ob = scenario["obstacle"]
    return np.asarray(ob["dir"], float) * ob["speed"]


def footprint_clearance(xy, yaw, c, r):
    """Surface clearance (m) between the 3-circle footprint at (xy, yaw) and a cylinder (centre c, radius r)."""
    h = np.array([math.cos(yaw), math.sin(yaw)])
    return min(float(np.linalg.norm(xy + off * h - c)) - rad - r for off, rad in FOOTPRINT)


def _template_law(err, dist, params, scale=1.0):
    """The template's align-then-drive law on one rollout state: in-place turn at clip(kp_ori err, +-1) while
    |err| >= ang_tol, then v = scale * clip(kp_pos dist) with a gentle heading correction w = clip(1.5 err, +-0.1)
    (the template allows +-0.3 while driving; here turning while driving is kept minimal, see DWA_W_TURN)."""
    if abs(err) >= params.ang_tol:
        return 0.0, float(np.clip(params.kp_ori * err, -params.max_ang_vel, params.max_ang_vel))
    v = scale * float(np.clip(params.kp_pos * dist, params.min_lin_vel, params.max_lin_vel) * max(0.0, math.cos(err)))
    return v, float(np.clip(1.5 * err, -0.1, 0.1))


def dwa_rollouts(xy, yaw, dock_xy, err, dist, params, steps):
    """[(label, vs, ws)]: the (v, w) sequences of every candidate over `steps` rollout steps of ROLLOUT_DT.

    Every candidate separates turning from driving: this base slides sideways when it turns while driving (the
    lateral speed builds at about v * w and the simulator hardly damps it), so the repertoire has in-place turns
    and straight runs only. Closed-loop candidates (the template's align-then-drive law re-evaluated at every
    rollout step) at full / 0.7 / 0.4 speed and after waiting 1-6 s (let the cylinder pass, then go); open-loop
    candidates: wait, straight ahead at 0.25 / 0.18 / 0.10 m/s,
    reverse straight at 0.12 m/s, and detours (in-place turn at 1 rad/s to a heading +-0.5 / +-1.0 / +-1.5 / +-2.0
    rad off the dock heading, then straight at 0.25 m/s); in-place turns at DWA_TURN_RATE. Far from the cylinder the closed-loop candidate wins, so
    the data equal the reaching expert's there (up to the gentler steering while driving).
    """
    out = []
    closed = [("drive", 1.0, 0.0), ("drive0.7", 0.7, 0.0), ("drive0.4", 0.4, 0.0)]
    closed += [(f"wait{tau:.0f}", 1.0, tau) for tau in (1.0, 2.0, 3.0, 4.0, 6.0)]
    for label, scale, wait in closed:
        p_xy, p_yaw, e, d = xy.copy(), yaw, err, dist
        vs, ws = np.zeros(steps), np.zeros(steps)
        for k in range(steps):
            v, w = _template_law(e, d, params, scale)
            if d < params.pos_tol or (k + 1) * ROLLOUT_DT <= wait:
                v, w = 0.0, 0.0
            vs[k], ws[k] = v, w
            p_yaw += w * ROLLOUT_DT
            p_xy = p_xy + v * ROLLOUT_DT * np.array([math.cos(p_yaw), math.sin(p_yaw)])
            to = dock_xy - p_xy
            d = float(np.linalg.norm(to))
            e = float(wrap_angle(math.atan2(to[1], to[0]) - p_yaw))
        out.append((label, vs, ws))
    for v in (0.0, 0.25, 0.18, 0.10, -0.12):
        out.append((f"straight{v:+.2f}", np.full(steps, v), np.zeros(steps)))
    ts = ROLLOUT_DT * np.arange(1, steps + 1)
    for delta in (0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0):
        turn = wrap_angle(err + delta)
        if abs(turn) > 0.05:
            first = ts <= abs(turn) / DWA_TURN_RATE
            out.append((f"detour{delta:+.1f}", np.where(first, 0.0, 0.25),
                        np.where(first, math.copysign(DWA_TURN_RATE, turn), 0.0)))
    return out


def dwa_command(xy, yaw, dock_xy, dock_bearing, c, v_obs, r, params, previous=None, debug=False):
    """(v, w, min clearance of the chosen rollout, horizon, planned heading of its first straight run): one cost
    for goal seeking and avoidance.

    Each candidate of `dwa_rollouts` is integrated (unicycle) for the horizon with the cylinder at its current
    velocity. Cost = DWA_W_GOAL * closest approach to the dock + DWA_W_END * distance(end, dock) + DWA_W_HEADING *
    |heading error towards the dock at the closest approach| (DWA_W_FINAL * |dock bearing error| instead when the
    rollout reaches the dock, the orient phase corrects it anyway) + DWA_W_CLEAR * max(0, DWA_MARGIN - clearance) +
    DWA_HIT if the footprint touches the cylinder + DWA_W_BLOCK * margin violation of the straight return to the dock
    after the rollout (cost-to-go, see the code) + DWA_W_WAIT * leading idle time + DWA_W_TURN * mean |w| while driving (the skid-steer base slides,
    with growing speed, when it turns while driving for long) + DWA_W_SMOOTH * change of the first command from the
    held one + DWA_REVERSE if reversing. The first command of the best rollout is returned (held for DWA_PERIOD); the
    held command is kept while its cost is within DWA_HYSTERESIS of the best. The horizon is distance / 0.25 m/s
    clipped to [DWA_MIN_HORIZON, DWA_HORIZON].
    """
    to_dock = dock_xy - xy
    dist = float(np.linalg.norm(to_dock))
    err = float(wrap_angle(math.atan2(to_dock[1], to_dock[0]) - yaw))
    horizon = float(np.clip(dist / 0.25, DWA_MIN_HORIZON, DWA_HORIZON))
    steps = max(int(round(horizon / ROLLOUT_DT)), 2)
    ts = ROLLOUT_DT * np.arange(1, steps + 1)
    cs = c[None, :] + v_obs[None, :] * ts[:, None]
    v_prev, w_prev = previous if previous is not None else (0.0, 0.0)
    scored = []
    for label, vs, ws in dwa_rollouts(xy, yaw, dock_xy, err, dist, params, steps):
        yaws = yaw + np.cumsum(ws) * ROLLOUT_DT
        xs = xy[0] + np.cumsum(vs * np.cos(yaws)) * ROLLOUT_DT
        ys = xy[1] + np.cumsum(vs * np.sin(yaws)) * ROLLOUT_DT
        clear, clear_end = np.inf, np.inf
        for off, rad in FOOTPRINT:
            px, py = xs + off * np.cos(yaws), ys + off * np.sin(yaws)
            gaps = np.hypot(px - cs[:, 0], py - cs[:, 1]) - rad - r
            clear, clear_end = min(clear, float(gaps.min())), min(clear_end, float(gaps[-1]))
        # Cost-to-go: from the end of the rollout, turn in place to the dock and drive straight at 0.25 m/s while the
        # cylinder carries on; a candidate whose return is blocked pays for it (waiting until the cylinder has passed
        # leaves a free return; fleeing alongside it does not).
        end_xy, end_yaw = np.array([xs[-1], ys[-1]]), float(yaws[-1])
        back = dock_xy - end_xy
        d_back = float(np.linalg.norm(back))
        if d_back > params.pos_tol:
            back_yaw = math.atan2(back[1], back[0])
            t_turn = abs(wrap_angle(back_yaw - end_yaw)) / DWA_TURN_RATE
            n2 = max(int(round(min(DWA_BLOCK_HORIZON, t_turn + d_back / 0.25) / ROLLOUT_DT)), 1)
            ts2 = horizon + ROLLOUT_DT * np.arange(1, n2 + 1)
            cs2 = c[None, :] + v_obs[None, :] * ts2[:, None]
            moving = np.clip((ts2 - horizon - t_turn), 0.0, None) * 0.25
            xs2 = end_xy[0] + moving * math.cos(back_yaw)
            ys2 = end_xy[1] + moving * math.sin(back_yaw)
            yaws2 = np.where(ts2 - horizon <= t_turn, end_yaw + np.sign(wrap_angle(back_yaw - end_yaw))
                             * DWA_TURN_RATE * (ts2 - horizon), back_yaw)
            clear_back = np.inf
            for off, rad in FOOTPRINT:
                px, py = xs2 + off * np.cos(yaws2), ys2 + off * np.sin(yaws2)
                clear_back = min(clear_back, float(np.min(np.hypot(px - cs2[:, 0], py - cs2[:, 1])) - rad - r))
        else:
            clear_back = np.inf
        d_dock = np.hypot(dock_xy[0] - xs, dock_xy[1] - ys)
        k = int(np.argmin(d_dock))
        d_min, d_end = float(d_dock[k]), float(d_dock[-1])
        if d_min > 0.15:
            heading = DWA_W_HEADING * abs(wrap_angle(math.atan2(dock_xy[1] - ys[k], dock_xy[0] - xs[k]) - yaws[k]))
        else:
            heading = DWA_W_FINAL * abs(wrap_angle(dock_bearing - yaws[k]))
        driving = vs > 0.05
        turn = float(np.mean(np.abs(ws) * driving)) if driving.any() else 0.0
        yaw_ref = float(yaws[int(np.argmax(driving))]) if driving.any() else None   # heading of the first straight run
        active = (np.abs(vs) > 1e-9) | (np.abs(ws) > 1e-9)
        idle = float(ROLLOUT_DT * (int(np.argmax(active)) if active.any() else steps))   # leading idle time
        v, w = float(vs[0]), float(ws[0])
        cost = (DWA_W_GOAL * d_min + DWA_W_END * d_end + heading + DWA_W_CLEAR * max(0.0, DWA_MARGIN - clear)
                + DWA_W_CLEAR_END * max(0.0, DWA_MARGIN_END - clear_end) + (DWA_HIT if clear < DWA_HIT_GAP else 0.0)
                + DWA_W_BLOCK * max(0.0, DWA_MARGIN - clear_back) + DWA_W_TURN * turn + DWA_W_WAIT * idle
                + DWA_W_SMOOTH * (abs(v - v_prev) + abs(w - w_prev)) + (DWA_REVERSE if v < 0 else 0.0))
        scored.append((cost, v, w, clear, label, yaw_ref))
    best = min(scored)
    if debug:
        for row in sorted(scored)[:16]:
            print("   cost %.3f v %+.2f w %+.2f clear %.2f %s" % row[:5])
    if previous is not None:
        for cost, v, w, clear, label, yaw_ref in scored:
            if abs(v - previous[0]) < 1e-9 and abs(w - previous[1]) < 1e-9 and cost <= best[0] + DWA_HYSTERESIS:
                return v, w, clear, horizon, yaw_ref
    return best[1], best[2], best[3], horizon, best[5]


def obstacle_at(scenario, t):
    ob = scenario["obstacle"]
    xy = np.asarray(ob["start_xy"]) + np.asarray(ob["dir"]) * ob["speed"] * t
    return {"x": float(xy[0]), "y": float(xy[1]), "r": ob["r"], "h": ob["h"]}


class DynamicExpert:
    """mode "dwa": the one-cost avoidance replaces the template's align / drive commands until the base is at the dock;
    mode "blind": the plain reaching template (control). The arm_controller reaches once navigation is done."""

    def __init__(self, mode="dwa"):
        if mode not in ("dwa", "blind"):
            raise ValueError(mode)
        self.mode = mode
        self.yaw_ref = None

    def reset(self, target, info, scenario):
        self.target = np.asarray(target, float)
        self.model = info["dyn_model"]
        self.ee_frame_id, self.shoulder_frame_id = info["ee_frame_id"], info["shoulder_frame_id"]
        self.arm_lower, self.arm_upper = info["joint_lower"][4:11], info["joint_upper"][4:11]
        self.time_step = info["time_step"]
        self.params = BaseControllerParams(nav_timeout=DYN_NAV_TIMEOUT,
                                           max_ang_vel=DWA_TURN_RATE if self.mode == "dwa" else 1.0)
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.state = BaseNavState()
        self.dock_xy = np.asarray(scenario["dock_xy"], float)
        self.dock_bearing = float(scenario["dock_bearing"])
        self.q_init_arm = info["init_q_mes"][4:11].copy()
        self.q_des_arm = self.q_init_arm.copy()
        self.qd_des_arm = np.zeros(7)
        self.obstacle_r = float(scenario["obstacle"]["r"])
        self.avoiding_steps = 0
        self.previous_choice = None
        self.held = None                       # dwa: (v, w, decision time)
        self.held_xy = None
        self.slip_steps = 0
        self.w_meas, self.prev_bearing = 0.0, None
        self.last_clear = math.inf
        self.last_avoid = (0.0, 1.0, 0.0)      # (w change, speed scale, repulsion weight | nominal rollout clearance)

    @property
    def nav_done(self):
        return self.state.done

    def act(self, obs):
        q_mes, qd_mes = obs["q_mes"], obs["qd_mes"]
        q_pin = self.model.ReoderJoints2PinVec(np.concatenate((obs["base_pos"], obs["base_ori"], q_mes)), "pos")
        pin.forwardKinematics(self.model.pin_model, self.model.pin_data, q_pin)
        pin.updateFramePlacements(self.model.pin_model, self.model.pin_data)
        ee_pos = self.model.pin_data.oMf[self.ee_frame_id].translation.copy()
        wheels = base_controller(self.state, obs["base_pos"], obs["base_bearing"], self.dock_xy, self.dock_bearing,
                                 obs["t"], self.params)
        if self.prev_bearing is not None:
            instant = wrap_angle(float(obs["base_bearing"]) - self.prev_bearing) / self.time_step
            self.w_meas = 0.98 * self.w_meas + 0.02 * instant              # ~50 ms low-pass of the measured yaw rate
        self.prev_bearing = float(obs["base_bearing"])
        self.last_avoid = (0.0, 1.0, 0.0)
        p = self.params
        if self.mode == "dwa" and not self.state.done and self.state.phase in ("align", "drive"):
            # The template's state machine still runs (it handles the orient phase and the timeout); its align and
            # drive commands are replaced by the DWA choice until the base is at the dock, then handed to "orient".
            xy, bearing = np.asarray(obs["base_pos"][:2], float), float(obs["base_bearing"])
            to_dock = self.dock_xy - xy
            dist = float(np.linalg.norm(to_dock))
            err = wrap_angle(math.atan2(to_dock[1], to_dock[0]) - bearing)
            if dist < p.pos_tol or (dist < 0.15 and math.cos(err) < 0.0):
                self.state.phase = "orient"
                wheels = base_controller(self.state, obs["base_pos"], obs["base_bearing"], self.dock_xy,
                                         self.dock_bearing, obs["t"], self.params)
            else:
                if self.held is None or obs["t"] - self.held[2] >= DWA_PERIOD - 1e-9:
                    c, v_obs = np.asarray(obs["obstacle_xy"], float), np.asarray(obs["obstacle_vel"], float)
                    v, w, clear, horizon, self.yaw_ref = dwa_command(xy, bearing, self.dock_xy, self.dock_bearing, c, v_obs,
                                                                     self.obstacle_r, self.params, previous=self.previous_choice)
                    if self.held is not None:
                        speed = float(np.linalg.norm(xy - self.held_xy)) / max(obs["t"] - self.held[2], 1e-6)
                        if abs(self.held[0]) > 0.05 and speed > DWA_SLIP_FACTOR * abs(self.held[0]) + 0.08:
                            self.slip_steps += int(round(DWA_PERIOD / self.time_step))   # sliding (measured, see below)
                    self.previous_choice = (v, w)
                    self.held = (v, w, obs["t"])
                    self.held_xy = xy.copy()
                    self.last_clear = clear
                    if clear < DWA_MARGIN + 0.1:
                        self.avoiding_steps += int(round(DWA_PERIOD / self.time_step))
                v, w, _ = self.held
                if v > 0.05 and abs(w) < 1e-9 and self.yaw_ref is not None:
                    # Straight run: steer to the heading the rollout planned for it (the base overshoots in-place turns
                    # by up to 0.5 rad: it keeps rotating when the wheels stop), as the template steers while driving.
                    w = float(np.clip(DWA_HOLD_GAIN * wrap_angle(self.yaw_ref - bearing), -0.3, 0.3))
                # Yaw-rate damping: the wheels free-wheel, so a stopped turn command does not stop the rotation;
                # command the difference actively.
                w = float(np.clip(w + DWA_YAW_DAMP * (w - self.w_meas), -p.max_ang_vel, p.max_ang_vel))
                self.last_avoid = (w, v / p.max_lin_vel, self.last_clear)
                lw = (v - w * p.wheel_base_width / 2) / p.wheel_radius
                rw = (v + w * p.wheel_base_width / 2) / p.wheel_radius
                wheels = np.array([rw, lw, lw, rw])
        torques, self.q_des_arm, self.qd_des_arm = arm_controller(
            self.model, q_pin, q_mes[4:11], qd_mes[4:11], self.q_des_arm, self.q_init_arm, self.target, ee_pos,
            self.ee_frame_id, self.shoulder_frame_id, self.arm_lower, self.arm_upper, self.time_step,
            self.arm_params, reach_active=self.state.done)
        return wheels, torques, ee_pos


LOG_FIELDS = {**legacy.LOG_FIELDS, "segment": np.int8, "lidar": np.float32, "obstacle_xy": np.float32,
              "avoid_w": np.float32, "speed_scale": np.float32, "clearance": np.float32}

_POLICIES = {}


def _load_policy(arg):
    """This project's learned policy behind the harness interface (scripts/obstacle/rl_obstacle_policy.py), cached per process."""
    if arg not in _POLICIES:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rl_obstacle_policy import make_policy
        _POLICIES[arg] = make_policy(arg)
    return _POLICIES[arg]


def converter_fields(scenario):
    """Fields that scripts/obstacle/convert_obstacle_dataset.py reads from a static-obstacle scenario; the moving cylinder is given
    at its t = 0 position (the converter uses the logged lidar of a moving-obstacle recording, not this list)."""
    return {"obstacles": [obstacle_at(scenario, 0.0)], "side": 0, "ambiguous": False, "lateral_offset": 0.0,
            "waypoint_xy": None, "moving_obstacle": True}


def run_episode(scenario, out_path, mode="dwa", gui=False, hold_s=HOLD_S, verbose=True, log_every=1, dart=False, policy=None):
    """policy: a training run directory (rl_obstacle_policy.make_policy argument): the learned policy's commands are
    executed and the expert runs alongside as the monitor (labels, end-effector position, navigation state); the episode
    ends at a contact, hold_s after the first reach (end-effector within REACH_TOLERANCE) or at the step cap, and its
    success is "reached, then still within REACH_TOLERANCE at the end of the hold"."""
    scenario = {**scenario, **converter_fields(scenario)}
    target = np.asarray(scenario["target"], float)
    wall = time.time()
    sim, client, rid, info = setup_simulation(scenario["init_xy"], scenario["init_yaw"], gui=gui, target=target)
    try:
        ob0 = obstacle_at(scenario, 0.0)
        obstacle_ids = spawn_obstacles(client, [ob0])
        quat = [0.0, 0.0, 0.0, 1.0]
        expert = DynamicExpert(mode)
        v_obs = obstacle_velocity(scenario)
        expert.reset(target, info, scenario)
        time_step = info["time_step"]
        learner = None
        if policy is not None:
            learner = _load_policy(policy)
            learner.reset(target, info)
        if dart:
            rng = np.random.default_rng([int(scenario["seed"]), int(scenario["episode"]), DART_STREAM])
            hold_steps = round(DART_HOLD_S / time_step)
            base_noise = legacy.HeldNoise(rng, [DART_V, DART_W], hold_steps)
            arm_noise = legacy.HeldNoise(rng, [DART_TORQUE] * 7, hold_steps)
            hold_noise = legacy.HeldNoise(rng, [DART_HOLD_WHEEL] * 2, hold_steps)
        dock = np.asarray(scenario["dock_xy"], float)
        min_ee_err = math.inf
        criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=STEP_CAP)
        cmd = MotorCommands()
        modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        log = {k: [] for k in LOG_FIELDS}
        step, t = 0, 0.0
        nav_done_time = reach_time = collision_time = None
        collision_link = None
        min_clear = math.inf
        stop_message = None
        while True:
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes, qd_mes = np.array(sim.GetMotorAngles(0)), np.array(sim.GetMotorVelocities(0))
            ob = obstacle_at(scenario, t)
            scan = lidar.scan(base_pos[:2], bearing, [ob])
            obs = {"t": t, "base_pos": base_pos, "base_ori": base_ori, "base_bearing": bearing, "q_mes": q_mes,
                   "qd_mes": qd_mes, "target": target, "lidar": scan, "obstacle_xy": [ob["x"], ob["y"]],
                   "obstacle_vel": v_obs}
            wheels, torques, ee_pos = expert.act(obs)
            ee_err = float(np.linalg.norm(target - ee_pos))
            min_ee_err = min(min_ee_err, ee_err)
            nav_done = expert.nav_done
            wheels_exec, torques_exec = wheels, torques
            if learner is not None:
                wheels_exec, torques_exec = (np.asarray(a, float) for a in learner.act(obs))
            elif dart:
                if not nav_done:
                    scale = legacy.fade(float(np.linalg.norm(dock - base_pos[:2])), DART_FADE_BASE)
                    dv, dw = base_noise.sample() * scale
                    wheels_exec = wheels + legacy.wheel_speeds(dv, dw, expert.params)
                elif np.all(wheels == 0.0):
                    right, left = hold_noise.sample()
                    wheels_exec = wheels + np.array([right, left, left, right])
                torques_exec = np.clip(torques + arm_noise.sample(), -TORQUE_LIMIT, TORQUE_LIMIT)
            if nav_done and nav_done_time is None:
                nav_done_time = t
            if step % 50 == 0:
                min_clear = min(min_clear, min_clearance(client, rid, obstacle_ids))
            if step % log_every == 0:
                phase = expert.state.phase
                vals = {"t": t, "base_pos": base_pos, "base_ori": base_ori, "base_bearing": bearing, "q_mes": q_mes,
                        "qd_mes": qd_mes, "wheel_cmds": wheels, "torque_arm": torques, "wheel_cmds_exec": wheels_exec,
                        "torque_arm_exec": torques_exec, "q_des_arm": expert.q_des_arm, "qd_des_arm": expert.qd_des_arm,
                        "phase": legacy.PHASE_CODES["reach"] if nav_done else legacy.PHASE_CODES[phase],
                        "nav_done": nav_done, "segment": int(nav_done), "ee_pos": ee_pos, "lidar": scan,
                        "obstacle_xy": [ob["x"], ob["y"]],
                        "avoid_w": expert.last_avoid[0], "speed_scale": expert.last_avoid[1], "clearance": min_clear}
                for k, v in vals.items():
                    log[k].append(v)
            cmd.SetControlCmd(np.concatenate((wheels_exec, torques_exec, np.zeros(6))), modes)
            sim.Step(cmd, "torque")
            ob_next = obstacle_at(scenario, t + time_step)
            client.resetBasePositionAndOrientation(obstacle_ids[0], [ob_next["x"], ob_next["y"], ob_next["h"] / 2], quat)
            hit, link = robot_obstacle_contact(client, rid, obstacle_ids)
            if hit:
                collision_time, collision_link = (step + 1) * time_step, link_name(client, rid, link)
                stop_message = f"Robot-obstacle contact at t={collision_time:.3f}s ({collision_link})"
                break
            if gui:
                time.sleep(0.001)
            if learner is not None:
                if reach_time is None and ee_err < REACH_TOLERANCE:
                    reach_time = t
                if reach_time is not None and t - reach_time >= hold_s:
                    stop_message = f"Policy hold over at t={t:.3f}s (EE error {ee_err:.4f} m)"
                    break
                if step + 1 >= STEP_CAP:
                    stop_message = f"Step cap at t={t:.3f}s (EE error {ee_err:.4f} m, best {min_ee_err:.4f} m)"
                    break
            else:
                msg = check_stop(criteria, t, time_step, ee_err, nav_done)
                if msg and msg.startswith("Target reached") and reach_time is None:
                    reach_time = t
                if reach_time is not None and hold_s > 0:
                    # Hold (6 Nov, as collect_static_obstacle.py): after the first reach the template's stall rule, which
                    # fires ~3 s into a hold, is ignored until hold_s have elapsed; the step cap still ends the episode.
                    if t - reach_time >= hold_s:
                        stop_message = f"Target reached at t={reach_time:.3f}s, held {hold_s:.0f} s (EE error {ee_err:.4f} m)"
                        break
                    if step + 1 >= STEP_CAP:
                        stop_message = f"Target reached at t={reach_time:.3f}s, step cap during the hold"
                        break
                elif msg and (not msg.startswith("Target reached") or hold_s <= 0):
                    stop_message = msg
                    break
            t += time_step
            step += 1
            if verbose and step % 2000 == 0:
                print(f"t={t:.1f}s base {np.round(base_pos[:2], 2)} obst {np.round([ob['x'], ob['y']], 2)} "
                      f"scan min {scan.min():.2f} m phase {expert.state.phase} avoid_w {expert.last_avoid[0]:+.2f} "
                      f"EE err {ee_err:.3f} m")
        if learner is not None:
            if collision_time is not None:
                status = "collision"
            elif reach_time is None:
                status = "not_reached"
            else:
                status = "reached" if ee_err < REACH_TOLERANCE else "left_target"
        elif collision_time is not None:
            status = "collision"
        elif reach_time is not None:
            status = "reached"          # the expert reached; under --dart the arm noise jitters the final error
        else:
            status = legacy.stop_status(stop_message)
            if status == "unreachable" and ee_err < REACH_TOLERANCE:
                status = "reached"
        nav_timed_out = nav_done_time is not None and nav_done_time >= DYN_NAV_TIMEOUT - 1e-6
        meta = {"scenario": scenario, "status": status, "success": status == "reached" and collision_time is None,
                "seed": int(scenario["seed"]) * 100000 + int(scenario["episode"]), "noisy": bool(dart),
                "noise_v": DART_V if dart else 0.0, "noise_w": DART_W if dart else 0.0,
                "noise_torque": DART_TORQUE if dart else 0.0, "noise_hold": DART_HOLD_S,
                "noise_fade_base": DART_FADE_BASE, "noise_fade_arm": 0.0,
                "hold_wheel_noise": DART_HOLD_WHEEL if dart else 0.0, "nav_timed_out": bool(nav_timed_out),
                "segment_switch_time": None, "policy": policy, "min_ee_error": float(min_ee_err),
                "reach_time": reach_time, "nav_done_time": nav_done_time, "duration": t, "n_steps": step + 1,
                "final_error": ee_err, "collided": collision_time is not None, "collision_time": collision_time,
                "collision_link": collision_link, "min_clearance": None if math.isinf(min_clear) else float(min_clear),
                "mode": mode, "hold_s": hold_s, "avoiding_steps": expert.avoiding_steps, "slip_steps": getattr(expert, "slip_steps", 0), "stop_message": stop_message, "time_step": time_step,
                "log_every": log_every, "wall_time": time.time() - wall}
        arrays = {k: np.asarray(v, dtype=d) for k, (v, d) in ((k, (log[k], LOG_FIELDS[k])) for k in LOG_FIELDS)}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path + ".tmp", "wb") as f:
            np.savez_compressed(f, **arrays, target=target, meta=json.dumps(meta))
        os.replace(out_path + ".tmp", out_path)
        if verbose:
            print(f"{stop_message} | status {status} min clearance {meta['min_clearance']} m")
        return meta
    finally:
        client.disconnect()


def make_gif(npz_path, gif_path, every=100):
    """Top view: planned start->dock line, robot footprint + heading, lidar beams, cylinder, trails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image

    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    sc = meta["scenario"]
    pos, yaw, scan, oxy = d["base_pos"], d["base_bearing"], d["lidar"], d["obstacle_xy"]
    r_ob = sc["obstacle"]["r"]
    pts = np.vstack([pos[:, :2], oxy, [sc["dock_xy"]], [sc["init_xy"]], [sc["target"][:2]]])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    frames = []
    angles = lidar.beam_angles(scan.shape[1])
    for k in range(0, len(pos), every):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.plot([sc["init_xy"][0], sc["dock_xy"][0]], [sc["init_xy"][1], sc["dock_xy"][1]], "k--", lw=0.8, label="start->dock")
        ax.plot(*sc["target"][:2], "g*", ms=12, label="target")
        ax.plot(pos[:k + 1, 0], pos[:k + 1, 1], "b-", lw=1.2, label="base")
        ax.plot(oxy[:k + 1, 0], oxy[:k + 1, 1], "r:", lw=1.0)
        ax.add_patch(Circle(oxy[k], r_ob, color="tab:red", alpha=0.6, label="moving cylinder"))
        ax.add_patch(Circle(pos[k, :2], 0.36, fill=False, color="b"))
        hx, hy = math.cos(yaw[k]), math.sin(yaw[k])
        ax.arrow(pos[k, 0], pos[k, 1], 0.5 * hx, 0.5 * hy, head_width=0.12, color="b")
        hit = scan[k] < lidar.MAX_RANGE - 1e-6
        for a, rng_, h in zip(angles[hit], scan[k][hit], hit[hit]):
            ax.plot([pos[k, 0], pos[k, 0] + rng_ * math.cos(yaw[k] + a)], [pos[k, 1], pos[k, 1] + rng_ * math.sin(yaw[k] + a)],
                    color="orange", lw=0.5, alpha=0.7)
        ax.set_title(f"ep {sc['episode']}  t={d['t'][k]:.1f}s  phase {int(d['phase'][k])}  avoid_w {float(d['avoid_w'][k]):+.2f}  "
                     f"v scale {float(d['speed_scale'][k]):.2f}\nstatus {meta['status']}, min clearance "
                     f"{(meta['min_clearance'] or float('nan')):.2f} m", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=every, loop=0, optimize=True)
    return gif_path


def parse_episodes(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--gif", action="store_true", help="top-view animation per episode")
    ap.add_argument("--mode", default="dwa", choices=["dwa", "blind"],
                    help="dwa: the one-cost expert (default); blind: the plain reaching template (control)")
    ap.add_argument("--hold-s", type=float, default=HOLD_S)
    ap.add_argument("--dart", action="store_true", help="set F noise on the executed command (training collection)")
    ap.add_argument("--policy", default=None, help="run a learned policy (training run dir[,db=..]); the expert only monitors")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    episodes = parse_episodes(args.episodes)
    scenarios = [{**sample_dynamic_scenario(args.seed, ep)} for ep in episodes]
    scenarios = [{**sc, **converter_fields(sc), "noisy": bool(args.dart)} for sc in scenarios]
    with open(os.path.join(args.out_dir, "scenarios.json"), "w") as f:
        json.dump({"scenarios": scenarios}, f)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"set_name": os.path.basename(os.path.normpath(args.out_dir)), "seed": args.seed, "mode": args.mode,
                   "episodes": args.episodes, "hold_s": args.hold_s, "policy": args.policy, "noise_frac": 1.0 if args.dart else 0.0,
                   "noise_v": DART_V if args.dart else 0.0, "noise_w": DART_W if args.dart else 0.0,
                   "noise_torque": DART_TORQUE if args.dart else 0.0, "noise_hold": DART_HOLD_S,
                   "noise_fade_base": DART_FADE_BASE, "noise_fade_arm": 0.0,
                   "hold_wheel_noise": DART_HOLD_WHEEL if args.dart else 0.0, "max_steps": STEP_CAP,
                   "collector": "scripts/obstacle/dynamic_obstacle.py"}, f, indent=1)
    jobs = [(sc, os.path.join(args.out_dir, f"ep_{sc['episode']:04d}.npz")) for sc in scenarios]
    kwargs = {"mode": args.mode, "gui": args.gui, "hold_s": args.hold_s, "dart": args.dart, "policy": args.policy}
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as ex:
            futures = [ex.submit(run_episode, sc, path, verbose=False, **kwargs) for sc, path in jobs]
            metas = []
            for k, fut in enumerate(futures, start=1):
                metas.append(fut.result())
                if k % 50 == 0 or k == len(futures):
                    print(f"  {k}/{len(futures)} episodes, success {sum(m['success'] for m in metas)}", flush=True)
    else:
        metas = []
        for sc, path in jobs:
            print(f"== episode {sc['episode']}: cylinder r {sc['obstacle']['r']:.2f} m at {sc['obstacle']['speed']:.2f} m/s, "
                  f"crossing at {sc['along']:.2f} of the segment, angle {math.degrees(sc['cross_angle_rad']):+.0f} deg, "
                  f"meets the robot at t ~ {sc['obstacle']['t_meet']:.1f} s (nominal {sc['nominal_meet_time_s']:.1f} s)")
            metas.append(run_episode(sc, path, **kwargs))
    rows = []
    for (sc, path), meta in zip(jobs, metas):
        ep = sc["episode"]
        row = {"episode": ep, "status": meta["status"], "success": meta["success"], "reach_time": meta["reach_time"],
               "nav_done_time": meta["nav_done_time"], "duration": meta["duration"], "min_clearance": meta["min_clearance"],
               "collision_link": meta["collision_link"], "avoid_s": meta["avoiding_steps"] / 1000, "slip_s": meta["slip_steps"] / 1000, "obst_speed": sc["obstacle"]["speed"], "obst_r": sc["obstacle"]["r"],
               "along": sc["along"], "cross_angle_deg": math.degrees(sc["cross_angle_rad"]), "t_meet": sc["obstacle"]["t_meet"],
               "final_error": meta["final_error"], "min_ee_error": meta["min_ee_error"],
               "file": os.path.basename(path), "max_steps": STEP_CAP}
        if args.gif:
            row["gif"] = make_gif(path, os.path.join(args.out_dir, f"ep_{ep:04d}.gif"))
        rows.append(row)
    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"\n{n} episodes: success {sum(r['success'] for r in rows)}, collision {sum(r['status'] == 'collision' for r in rows)}, "
          f"other {sum(r['status'] not in ('reached', 'collision') for r in rows)}; min clearance median "
          f"{np.nanmedian([r['min_clearance'] or np.nan for r in rows]):.2f} m")


if __name__ == "__main__":
    main()
