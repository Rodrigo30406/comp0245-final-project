"""Prototype (5 Nov 2026): dynamic-obstacle task and a one-cost (DWA-style) avoidance expert, built on the group's step-9 code.

Why: the group's `multi-agent` branch yields/brakes on a fixed path using the other robot's plan (privileged), so its
demonstrations contain no detour and no action the learner could reproduce from its own observation. This prototype
is the alternative discussed with the group: one scripted cylinder crosses the start -> dock corridor while the robot
drives, and the expert steers around it using the cylinder's position and velocity (observable from two scans) plus the goal geometry
the A-to-B expert already uses. Nothing in the group's code is edited: their simulation setup (`evaluate_policy.
setup_simulation`), base/arm controllers (`robotnik_arm_control`), analytic lidar (`obstacle_task.lidar`) and
obstacle helpers (`obstacle_task.sim`) are imported as they are.

Scenario (`sample_dynamic_scenario`): start pose and target as step 9 (`scenario.py` constants: target 3.5-5 m away),
dock = the A-to-B expert's dock, no static obstacle. One cylinder (r 0.25-0.35 m, h 1.5 m) moves on a straight line
at a constant speed OBST_SPEED (0.10-0.25 m/s; v1-v4 used 0.15-0.45) that crosses the start->dock segment at a fraction ALONG (0.35-0.65),
at an angle CROSS_ANGLE_DEG (60-120 deg to the segment, both directions). Its start is set so that it reaches the
crossing point at the moment the unperturbed robot would (arrival time at 0.25 m/s nominal speed) plus a jitter
MEET_JITTER_S, so the two really meet; after crossing it carries on in a straight line. The crossing point is kept
>= 1.0 m from the start and from the dock so the in-place turns are never threatened.

Expert (`ReactiveExpert`, mode "dwa", the default, v6-v7 of 5 Nov 2026): goal seeking and avoidance in ONE cost,
as the open-source planners do (PythonRobotics DWA, ir-sim VO/RVO, TEB). Every DWA_PERIOD (50 ms) the candidate
commands of `dwa_rollouts` are integrated for up to DWA_HORIZON s with the cylinder moving at its current velocity
and the cheapest wins (`dwa_command`): closest approach to the dock, distance to the dock at the end, heading error
at the closest approach, clearance between the 3-circle footprint (chassis 0.36 m, arm 0.22 m at 0.38 m, wrist
0.16 m at 0.68 m ahead, as the group's multi-agent code) and the cylinder (closest approach and end of the rollout),
turning while driving, command changes, reversing. The template's own align-then-drive law is one of the candidates
(simulated closed loop), so far from the cylinder the data equal the A-to-B expert's; the others are the avoidance
repertoire (wait, slower, reverse, in-place turn to a detour heading then straight). The template's state machine
still runs: it takes over for the final orientation once the base is at the dock (pos_tol) and keeps the timeout.
Two properties of the group's simulation shaped the design (see the DWA_* comments): (1) the base slides sideways
when it turns while driving (lateral speed builds at about v * w) and the simulator hardly damps it (0.36 m/s
sideways for 20 s with the wheels commanded to zero, which free-wheel), so every candidate separates turning (in
place, DWA_TURN_RATE) from driving (straight, with a heading hold); (2) the base keeps rotating after an in-place
turn, so straight runs hold the heading the rollout assumed. The horizon (8 s) is long enough to see that fleeing
alongside a cylinder that comes from behind never ends (the chase trap of the shorter horizons).
The rule needs the cylinder's position and velocity: both are observable from two consecutive scans (the learner
needs a scan history, i.e. a stacked-scan input or a recurrent policy), unlike the other robot's plan in the
multi-agent branch. The expert reads them from the scenario (privileged, like the step-9 expert reads the waypoint).
Mode "vo" (v1-v5) is kept for the record: the unchanged `base_controller` gives the nominal (v, w) and a sampled
velocity-obstacle rule replaces it only when the nominal command would conflict with the cylinder. It avoided well
(9/12 at v5) but could not recover: the template aligns only once, so after a detour the drive phase orbits the
dock (turn radius 0.8 m), and the re-alignment patch alternated with the avoidance (5 of 12 episodes used the whole
60 s). That is the lesson of the open-source planners: with goal seeking and avoidance in one cost there is no
nominal controller to hand back to, hence no recovery phase.
Mode "lidar" is the first attempt kept for the record: single-scan repulsion (turn away from the closest beams,
slow down when they are ahead). It collided in 8/10 episodes, always on a rear wheel: turning away from a crossing
body swings the rear into its path, and one scan cannot tell where the body is going. Mode "blind" is the plain
A-to-B expert (control: the scenario forces an interaction).
The arm is `arm_controller` unchanged once navigation is done. A collision (robot-cylinder contact) ends the episode.

Usage (default pixi env of RoboEnv, from this project's root):
  pixi run --manifest-path ../RoboEnv/pixi.toml python scripts/obstacle/dynamic_expert.py --episodes 0-9 \
      --out-dir runs/obstacle/dynamic_proto [--gui] [--gif] [--mode dwa|vo|lidar|blind]
Writes one .npz per episode (the group's LOG_FIELDS + lidar (T, 64) + obstacle_xy (T, 2)), index.csv, and with
--gif a top-view animation per episode (robot footprint, lidar beams, cylinder, planned line) for review.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

TASK_DIR = os.environ.get("STEP9_TASK_DIR", "/home/labia10/repos/comp0245-step9/final/robotnik_arm_task")
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

STREAM = 23                        # per-episode SeedSequence entry (step 9 uses 9, multi-agent 19)
ALONG = (0.35, 0.65)               # crossing point as a fraction of the start->dock segment
CROSS_ANGLE_DEG = (60.0, 120.0)    # angle between the cylinder's line and the start->dock segment
OBST_SPEED = (0.10, 0.25)          # m/s (<= the base's 0.25 m/s: a faster body coming from behind cannot be escaped; v1-v4 used 0.15-0.45)
MEET_JITTER_S = (-2.0, 2.0)        # s, obstacle arrival at the crossing point relative to the robot's nominal one
ROBOT_NOMINAL_SPEED = 0.25         # m/s, base_controller max_lin_vel: nominal arrival time = distance / speed
ALIGN_TIME_S = 1.5                 # s, allowance for the initial in-place turn in the nominal arrival time
END_CLEAR = 1.0                    # m, the cylinder's path (surface) to the start and to the dock
OBST_RUN_IN = 2.5                  # m, the cylinder starts this far from the crossing point at least
STEP_CAP = 75000                   # 75 s at 1 kHz
DYN_NAV_TIMEOUT = 60.0             # s, navigation timeout (step 9: 45 s; the avoidance and the recovery take longer)
HOLD_S = 10.0                      # hold tail after the first reach (as our hold sets), 0 = stop at the reach

# Reactive layer (all from the scan)
REACT_RANGE = 1.6                  # m
STOP_RANGE = 0.55                  # m, forward range at which the speed scale bottoms out
FRONT_HALF_DEG = 100.0             # beams considered for the repulsion
FRONT_CONE_DEG = 35.0              # beams considered for the speed scale
K_TURN = 0.9                       # rad/s per unit repulsion weight (1/m)
MAX_AVOID_W = 0.6                  # rad/s
MIN_SPEED_SCALE = 0.25

# Velocity-obstacle rule
VO_HORIZON = 5.0                   # s (the cylinder, up to 0.45 m/s, can be faster than the base, 0.25 m/s: react early)
VO_DT = 0.125                      # s
VO_MARGIN = 0.40                   # m, footprint-cylinder clearance kept over the rollout (unicycle model: the skid-steer slips)
REALIGN_ERR = 0.35                 # rad, heading error to the dock after an avoidance beyond which the base re-aligns in place
                                   # (only when the nominal command is conflict-free: never turn in place in the cylinder's way)
VO_SPEEDS = (0.25, 0.18, 0.10, 0.04, 0.0, -0.12)
VO_RATES = (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)
FOOTPRINT = ((0.00, 0.36), (0.38, 0.22), (0.68, 0.16))   # (forward offset, radius) m

# Mode "dwa" (v6): goal seeking and avoidance in ONE cost, as DWA / VO planners do (PythonRobotics DWA, ir-sim RVO,
# TEB): no nominal controller to fall back to, so there is no "recovery" phase after an avoidance.
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
    """One scenario: start, target, dock (A-to-B expert) and the cylinder's straight-line motion."""
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


def velocity_obstacle_command(xy, yaw, v_nom, w_nom, c, v_obs, r, previous=None):
    """(v, w, min clearance of the chosen rollout, nominal's min clearance): sampled velocity-obstacle rule.

    Candidates are constant (v, w) over the horizon plus turn-then-drive manoeuvres (in-place turn at w for tau s,
    then straight at 0.25 m/s): a crossing or chasing cylinder is escaped sideways, which a constant command
    cannot represent. The command of the first phase is returned; `previous` (v, w) is kept while its cost is within
    0.2 of the best (hysteresis against chattering between candidates).
    """
    steps = int(round(VO_HORIZON / VO_DT))
    ts = VO_DT * np.arange(1, steps + 1)
    cs = c[None, :] + v_obs[None, :] * ts[:, None]                        # (K, 2) cylinder centres

    def rollout_clearance(v, w, tau=None):
        if tau is None:
            vs, ws = np.full(steps, v), np.full(steps, w)
        else:
            first = ts <= tau
            vs, ws = np.where(first, v, 0.25), np.where(first, w, 0.0)
        yaws = yaw + np.cumsum(ws) * VO_DT
        xs = xy[0] + np.cumsum(vs * np.cos(yaws)) * VO_DT
        ys = xy[1] + np.cumsum(vs * np.sin(yaws)) * VO_DT
        clear = np.inf
        for off, rad in FOOTPRINT:
            px, py = xs + off * np.cos(yaws), ys + off * np.sin(yaws)
            clear = min(clear, float(np.min(np.hypot(px - cs[:, 0], py - cs[:, 1])) - rad - r))
        return clear

    nominal_clear = rollout_clearance(v_nom, w_nom)
    if nominal_clear >= VO_MARGIN:
        return v_nom, w_nom, nominal_clear, nominal_clear
    candidates = [(v, w, None) for v in (v_nom, *VO_SPEEDS) for w in (w_nom, *VO_RATES)]
    candidates += [(0.0, w, tau) for w in (-1.0, -0.6, 0.6, 1.0) for tau in (1.0, 2.0, 3.0)]
    scored = []
    for v, w, tau in candidates:
        clear = rollout_clearance(v, w, tau)
        cost = 20.0 * max(0.0, VO_MARGIN - clear) + 2.0 * abs(v - v_nom) + 1.0 * abs(w - w_nom) + (0.3 if v < 0 else 0.0)
        scored.append((cost, v, w, clear))
    best = min(scored)
    if previous is not None:
        for cost, v, w, clear in scored:
            if (v, w) == previous and cost <= best[0] + 0.2:
                return v, w, clear, nominal_clear
    return best[1], best[2], best[3], nominal_clear


def _template_law(err, dist, params, scale=1.0):
    """The template's align-then-drive law on one rollout state: in-place turn at clip(kp_ori err, +-1) while
    |err| >= ang_tol, then v = scale * clip(kp_pos dist) with a gentle heading correction w = clip(1.5 err, +-0.1)
    (the template allows +-0.3 while driving; here turning while driving is kept minimal, see DWA_W_TURN)."""
    if abs(err) >= params.ang_tol:
        return 0.0, float(np.clip(params.kp_ori * err, -params.max_ang_vel, params.max_ang_vel))
    v = scale * float(np.clip(params.kp_pos * dist, params.min_lin_vel, params.max_lin_vel) * max(0.0, math.cos(err)))
    return v, float(np.clip(1.5 * err, -0.1, 0.1))


def dwa_rollouts(xy, yaw, dock_xy, err, dist, params, steps):
    """[(label, vs, ws)]: the (v, w) sequences of every candidate over `steps` rollout steps of VO_DT.

    Every candidate separates turning from driving: this base slides sideways when it turns while driving (the
    lateral speed builds at about v * w and the simulator hardly damps it), so the repertoire has in-place turns
    and straight runs only. Closed-loop candidates (the template's align-then-drive law re-evaluated at every
    rollout step) at full / 0.7 / 0.4 speed and after waiting 1-6 s (let the cylinder pass, then go); open-loop
    candidates: wait, straight ahead at 0.25 / 0.18 / 0.10 m/s,
    reverse straight at 0.12 m/s, and detours (in-place turn at 1 rad/s to a heading +-0.5 / +-1.0 / +-1.5 / +-2.0
    rad off the dock heading, then straight at 0.25 m/s); in-place turns at DWA_TURN_RATE. Far from the cylinder the closed-loop candidate wins, so
    the data equal the A-to-B expert's there (up to the gentler steering while driving).
    """
    out = []
    closed = [("drive", 1.0, 0.0), ("drive0.7", 0.7, 0.0), ("drive0.4", 0.4, 0.0)]
    closed += [(f"wait{tau:.0f}", 1.0, tau) for tau in (1.0, 2.0, 3.0, 4.0, 6.0)]
    for label, scale, wait in closed:
        p_xy, p_yaw, e, d = xy.copy(), yaw, err, dist
        vs, ws = np.zeros(steps), np.zeros(steps)
        for k in range(steps):
            v, w = _template_law(e, d, params, scale)
            if d < params.pos_tol or (k + 1) * VO_DT <= wait:
                v, w = 0.0, 0.0
            vs[k], ws[k] = v, w
            p_yaw += w * VO_DT
            p_xy = p_xy + v * VO_DT * np.array([math.cos(p_yaw), math.sin(p_yaw)])
            to = dock_xy - p_xy
            d = float(np.linalg.norm(to))
            e = float(wrap_angle(math.atan2(to[1], to[0]) - p_yaw))
        out.append((label, vs, ws))
    for v in (0.0, 0.25, 0.18, 0.10, -0.12):
        out.append((f"straight{v:+.2f}", np.full(steps, v), np.zeros(steps)))
    ts = VO_DT * np.arange(1, steps + 1)
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
    steps = max(int(round(horizon / VO_DT)), 2)
    ts = VO_DT * np.arange(1, steps + 1)
    cs = c[None, :] + v_obs[None, :] * ts[:, None]
    v_prev, w_prev = previous if previous is not None else (0.0, 0.0)
    scored = []
    for label, vs, ws in dwa_rollouts(xy, yaw, dock_xy, err, dist, params, steps):
        yaws = yaw + np.cumsum(ws) * VO_DT
        xs = xy[0] + np.cumsum(vs * np.cos(yaws)) * VO_DT
        ys = xy[1] + np.cumsum(vs * np.sin(yaws)) * VO_DT
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
            n2 = max(int(round(min(DWA_BLOCK_HORIZON, t_turn + d_back / 0.25) / VO_DT)), 1)
            ts2 = horizon + VO_DT * np.arange(1, n2 + 1)
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
        idle = float(VO_DT * (int(np.argmax(active)) if active.any() else steps))   # leading idle time
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


def reactive_terms(scan, params):
    """(w_avoid, speed_scale, weight) from one scan. Beam 0 is forward, angles counter-clockwise."""
    angles = wrap_angle(lidar.beam_angles(len(scan)))
    half = np.abs(angles) <= math.radians(FRONT_HALF_DEG)
    close = half & (scan < REACT_RANGE)
    if not close.any():
        return 0.0, 1.0, 0.0
    weight = 1.0 / np.maximum(scan[close], 0.05) - 1.0 / REACT_RANGE
    left = float(weight[angles[close] > 0].sum())
    right = float(weight[angles[close] <= 0].sum())
    total = left + right
    side = 1.0 if left >= right else -1.0                           # obstacle mass on the left -> turn right
    w_avoid = float(np.clip(-side * K_TURN * total, -MAX_AVOID_W, MAX_AVOID_W))
    cone = np.abs(angles) <= math.radians(FRONT_CONE_DEG)
    r_front = float(scan[cone].min())
    scale = float(np.clip((r_front - STOP_RANGE) / (REACT_RANGE - STOP_RANGE), MIN_SPEED_SCALE, 1.0))
    return w_avoid, scale, total


class ReactiveExpert:
    """A-to-B base_controller + lidar-driven avoidance in the drive phase; arm_controller once the base is done."""

    def __init__(self, mode="vo"):
        if mode not in ("vo", "dwa", "lidar", "blind"):
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
        self.realigns = 0
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
        if self.mode == "vo" and not self.state.done:
            xy, bearing = np.asarray(obs["base_pos"][:2], float), float(obs["base_bearing"])
            c, v_obs = np.asarray(obs["obstacle_xy"], float), np.asarray(obs["obstacle_vel"], float)
            v_nom = p.wheel_radius * (wheels[0] + wheels[1]) / 2.0
            w_nom = p.wheel_radius * (wheels[0] - wheels[1]) / p.wheel_base_width
            v, w, clear, nominal_clear = velocity_obstacle_command(xy, bearing, v_nom, w_nom, c, v_obs, self.obstacle_r,
                                                                   previous=self.previous_choice)
            self.previous_choice = (v, w) if (v, w) != (v_nom, w_nom) else None
            if (v, w) != (v_nom, w_nom):
                self.avoiding_steps += 1
                self.last_avoid = (w - w_nom, (v / v_nom if abs(v_nom) > 1e-6 else 1.0), nominal_clear)
            elif self.state.phase == "drive" and self.avoiding_steps > 0:
                # After an avoidance the drive phase may face far from the dock (the template aligns only once, at the
                # start): with v > 0 and |w| <= 0.3 rad/s it would orbit the dock (turn radius 0.8 m). Re-run the in-place
                # alignment, but only now that the nominal command is conflict-free.
                to_dock = self.dock_xy - xy
                err = wrap_angle(math.atan2(to_dock[1], to_dock[0]) - bearing)
                if abs(err) > REALIGN_ERR and np.linalg.norm(to_dock) > p.pos_tol:
                    self.state.phase = "align"
                    self.realigns += 1
                    wheels = base_controller(self.state, obs["base_pos"], obs["base_bearing"], self.dock_xy,
                                             self.dock_bearing, obs["t"], self.params)
                    v = p.wheel_radius * (wheels[0] + wheels[1]) / 2.0
                    w = p.wheel_radius * (wheels[0] - wheels[1]) / p.wheel_base_width
            if (v, w) != (v_nom, w_nom):
                lw = (v - w * p.wheel_base_width / 2) / p.wheel_radius
                rw = (v + w * p.wheel_base_width / 2) / p.wheel_radius
                wheels = np.array([rw, lw, lw, rw])
        elif self.mode == "dwa" and not self.state.done and self.state.phase in ("align", "drive"):
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
        elif self.mode == "lidar" and self.state.phase == "drive" and not self.state.done:
            w_avoid, scale, weight = reactive_terms(obs["lidar"], p)
            if weight > 0.0:
                v = p.wheel_radius * (wheels[0] + wheels[1]) / 2.0
                w = p.wheel_radius * (wheels[0] - wheels[1]) / p.wheel_base_width
                v, w = v * scale, float(np.clip(w + w_avoid, -MAX_AVOID_W - 0.3, MAX_AVOID_W + 0.3))
                lw = (v - w * p.wheel_base_width / 2) / p.wheel_radius
                rw = (v + w * p.wheel_base_width / 2) / p.wheel_radius
                wheels = np.array([rw, lw, lw, rw])
                self.last_avoid = (w_avoid, scale, weight)
        torques, self.q_des_arm, self.qd_des_arm = arm_controller(
            self.model, q_pin, q_mes[4:11], qd_mes[4:11], self.q_des_arm, self.q_init_arm, self.target, ee_pos,
            self.ee_frame_id, self.shoulder_frame_id, self.arm_lower, self.arm_upper, self.time_step,
            self.arm_params, reach_active=self.state.done)
        return wheels, torques, ee_pos


LOG_FIELDS = {**legacy.LOG_FIELDS, "lidar": np.float32, "obstacle_xy": np.float32, "avoid_w": np.float32,
              "speed_scale": np.float32, "clearance": np.float32}


def run_episode(scenario, out_path, mode="vo", gui=False, hold_s=HOLD_S, verbose=True, log_every=1):
    target = np.asarray(scenario["target"], float)
    wall = time.time()
    sim, client, rid, info = setup_simulation(scenario["init_xy"], scenario["init_yaw"], gui=gui, target=target)
    try:
        ob0 = obstacle_at(scenario, 0.0)
        obstacle_ids = spawn_obstacles(client, [ob0])
        quat = [0.0, 0.0, 0.0, 1.0]
        expert = ReactiveExpert(mode)
        v_obs = obstacle_velocity(scenario)
        expert.reset(target, info, scenario)
        time_step = info["time_step"]
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
            wheels, torques, ee_pos = expert.act({"t": t, "base_pos": base_pos, "base_ori": base_ori,
                                                  "base_bearing": bearing, "q_mes": q_mes, "qd_mes": qd_mes,
                                                  "target": target, "lidar": scan,
                                                  "obstacle_xy": [ob["x"], ob["y"]], "obstacle_vel": v_obs})
            ee_err = float(np.linalg.norm(target - ee_pos))
            nav_done = expert.nav_done
            if nav_done and nav_done_time is None:
                nav_done_time = t
            if step % 50 == 0:
                min_clear = min(min_clear, min_clearance(client, rid, obstacle_ids))
            if step % log_every == 0:
                phase = expert.state.phase
                vals = {"t": t, "base_pos": base_pos, "base_ori": base_ori, "base_bearing": bearing, "q_mes": q_mes,
                        "qd_mes": qd_mes, "wheel_cmds": wheels, "torque_arm": torques, "wheel_cmds_exec": wheels,
                        "torque_arm_exec": torques, "q_des_arm": expert.q_des_arm, "qd_des_arm": expert.qd_des_arm,
                        "phase": legacy.PHASE_CODES["reach"] if nav_done else legacy.PHASE_CODES[phase],
                        "nav_done": nav_done, "ee_pos": ee_pos, "lidar": scan, "obstacle_xy": [ob["x"], ob["y"]],
                        "avoid_w": expert.last_avoid[0], "speed_scale": expert.last_avoid[1], "clearance": min_clear}
                for k, v in vals.items():
                    log[k].append(v)
            cmd.SetControlCmd(np.concatenate((wheels, torques, np.zeros(6))), modes)
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
            msg = check_stop(criteria, t, time_step, ee_err, nav_done)
            if msg and msg.startswith("Target reached") and reach_time is None:
                reach_time = t
            if msg and (not msg.startswith("Target reached") or hold_s <= 0 or t - reach_time >= hold_s):
                stop_message = msg
                break
            t += time_step
            step += 1
            if verbose and step % 2000 == 0:
                print(f"t={t:.1f}s base {np.round(base_pos[:2], 2)} obst {np.round([ob['x'], ob['y']], 2)} "
                      f"scan min {scan.min():.2f} m phase {expert.state.phase} avoid_w {expert.last_avoid[0]:+.2f} "
                      f"EE err {ee_err:.3f} m")
        status = "collision" if collision_time is not None else legacy.stop_status(stop_message)
        if status == "unreachable" and ee_err < REACH_TOLERANCE:
            status = "reached"
        meta = {"scenario": scenario, "status": status, "success": status == "reached" and collision_time is None,
                "reach_time": reach_time, "nav_done_time": nav_done_time, "duration": t, "n_steps": step + 1,
                "final_error": ee_err, "collided": collision_time is not None, "collision_time": collision_time,
                "collision_link": collision_link, "min_clearance": None if math.isinf(min_clear) else float(min_clear),
                "mode": mode, "hold_s": hold_s, "avoiding_steps": expert.avoiding_steps, "realigns": expert.realigns, "slip_steps": getattr(expert, "slip_steps", 0), "stop_message": stop_message, "time_step": time_step,
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
    ap.add_argument("--mode", default="dwa", choices=["dwa", "vo", "lidar", "blind"],
                    help="dwa: one-cost goal seeking + avoidance (default, v6); vo: nominal controller + velocity-obstacle "
                         "override (v5); lidar: single-scan repulsion; blind: plain A-to-B expert")
    ap.add_argument("--hold-s", type=float, default=HOLD_S)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for ep in parse_episodes(args.episodes):
        sc = sample_dynamic_scenario(args.seed, ep)
        path = os.path.join(args.out_dir, f"ep_{ep:04d}.npz")
        print(f"== episode {ep}: cylinder r {sc['obstacle']['r']:.2f} m at {sc['obstacle']['speed']:.2f} m/s, "
              f"crossing at {sc['along']:.2f} of the segment, angle {math.degrees(sc['cross_angle_rad']):+.0f} deg, "
              f"meets the robot at t ~ {sc['obstacle']['t_meet']:.1f} s (nominal {sc['nominal_meet_time_s']:.1f} s)")
        meta = run_episode(sc, path, mode=args.mode, gui=args.gui, hold_s=args.hold_s)
        row = {"episode": ep, "status": meta["status"], "success": meta["success"], "reach_time": meta["reach_time"],
               "nav_done_time": meta["nav_done_time"], "duration": meta["duration"], "min_clearance": meta["min_clearance"],
               "collision_link": meta["collision_link"], "avoid_s": meta["avoiding_steps"] / 1000, "realigns": meta["realigns"], "slip_s": meta["slip_steps"] / 1000, "obst_speed": sc["obstacle"]["speed"], "obst_r": sc["obstacle"]["r"],
               "along": sc["along"], "cross_angle_deg": math.degrees(sc["cross_angle_rad"]), "t_meet": sc["obstacle"]["t_meet"]}
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
