"""Obstacle courses (harder step-9 scenarios): several obstacles of three kinds, one expert route through all of them.

A course is a Scenario whose `obstacles` hold 2-3 features along the start->dock line and whose `waypoints` are the
expert's route: start -> w1 -> ... -> wn -> dock, driven by the UNCHANGED base_controller leg by leg (expert.py).

Obstacle kinds (all taller than the arm and the lidar plane, so they can be neither crossed nor overflown):
  pillar  one cylinder {"x", "y", "r", "h"} (as the single-obstacle task);
  wall    one thin box {"type": "box", "x", "y", "hx", "hy", "yaw", "h", "r"}, tilted up to WALL_TILT_DEG from the
          path normal; "r" is the bounding-circle radius, so code that reads obstacles[0]["r"] keeps working;
  gate    two walls with a gap of 2 * GATE_HALF_GAP between their inner faces; the route goes through the gap centre.

Why the expert cannot fail on an accepted course (everything is checked in course_violations(), which the sampler and
tests/test_obstacle_course.py both call; nothing here needs the simulator):
  1. start, dock and every waypoint keep END_CLEAR (0.9 m) from every obstacle surface: the in-place turns there
     (arm sweep ~0.73 m) cannot touch anything. For a gate this fixes the gap: 0.9 m on each side of the centre;
  2. every route leg keeps COURSE_SEGMENT_CLEAR (0.5 m) from every surface (chassis half-width 0.31 + drift margin,
     the single-obstacle task uses 0.45);
  3. the arm line dock->target keeps ARM_CLEAR from every obstacle and the target is TARGET_CLEAR away: the reach
     phase cannot touch anything (the base stands still there);
  4. legs are at least MIN_LEG long and turns at waypoints are at most MAX_WAYPOINT_TURN_DEG (no reversals), so the
     controller's switching rule (position tolerance 0.08 m / passed-goal) is unambiguous;
  5. a conservative time estimate of the whole navigation (estimate_nav_time) is within NAV_BUDGET (40 s) so the
     45 s navigation timeout and the 60 s episode limit of the protocol are never the limiting factor;
  6. pillars and walls must block the straight start->dock line (the course is never trivially open).
Sampling is rejection sampling from its own stream np.random.default_rng([seed, episode, COURSE_STREAM]); it is
deterministic in (seed, episode) and independent of every other stream. Scenarios of the single-obstacle sampler are
untouched.
"""
import os
import sys

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from collect_data import relative_to_world, sample_init_poses, sample_targets  # noqa: E402
from evaluate_policy import expert_base_goal  # noqa: E402
from obstacle_task import scenario as sc_mod  # noqa: E402
from obstacle_task.scenario import (  # noqa: E402
    AMBIG_BAND, CHASSIS_HALF_WIDTH, END_CLEAR, INIT_RADIUS, INIT_YAW_DEG, LATERAL, OBST_HEIGHT, OBST_RADIUS,
    TARGET_CLEAR, TARGET_THETA_DEG, TARGET_Z, WAYPOINT_CLEAR, Scenario, frame_of,
)

COURSE_FORMAT = "obstacle_course_v1"
COURSE_STREAM = 20                 # third entry of the per-episode SeedSequence (single obstacle 9, its noisy flag 10)

COURSE_TARGET_R = (3.8, 5.2)       # m, target distance from the start pose (single obstacle: 3.5-5.0)
P_TWO_FEATURES = 0.55              # otherwise three features
FIRST_KINDS = (("pillar", 0.5), ("wall", 0.5))
LATER_KINDS = (("pillar", 0.4), ("wall", 0.3), ("gate", 0.3))
ALONG_JITTER = 0.05                # fraction of the start->dock length, around evenly spaced positions
P_ALTERNATE = 0.7                  # probability that a later feature is passed on the other side than the previous one
LATER_LATERAL = (AMBIG_BAND, LATERAL)   # |d| of later pillars / walls: outside the ambiguous band, side = -sign(d)

WALL_HALF_LEN = (0.5, 0.8)         # m
WALL_HALF_THICK = 0.1              # m
WALL_TILT_DEG = 20.0               # max tilt of a wall from the path normal
GATE_HALF_GAP = END_CLEAR + 0.05   # m, gate centre -> inner face of each wall (centre keeps END_CLEAR for the turn)
GATE_WALL_HALF_LEN = (0.4, 0.6)    # m, half-length of each of the two gate walls
GATE_LATERAL = 0.3                 # m, |d| of the gate centre from the start->dock line
GATE_TILT_DEG = 15.0

COURSE_SEGMENT_CLEAR = 0.5         # m, obstacle surface -> every route leg
ARM_CLEAR = 0.30                   # m, obstacle surface -> the dock->target line
MIN_LEG = 0.9                      # m, shortest route leg
MAX_WAYPOINT_TURN_DEG = 110.0      # largest heading change at a waypoint

# Conservative navigation-time model of base_controller (v <= 0.25 m/s with a slow-down at every goal, yaw <= 1 rad/s
# with a P-law tail): time = sum(len / V_EFF + LEG_OVERHEAD) + sum(turn / TURN_EFF + TURN_OVERHEAD)
V_EFF = 0.22                       # m/s
LEG_OVERHEAD = 2.0                 # s per leg (ramp down to the goal)
TURN_EFF = 0.6                     # rad/s
TURN_OVERHEAD = 1.0                # s per in-place turn (P-law tail)
NAV_BUDGET = 40.0                  # s, largest accepted estimate (navigation timeout 45 s, episode limit 60 s)
MAX_TRIES = 30000
SAMPLE_STEP = 0.02                 # m, polyline sampling step of the leg -> box distance


# ---------------------------------------------------------------------------------------------------------------------
# Obstacle construction and distances
# ---------------------------------------------------------------------------------------------------------------------
def make_box(x, y, hx, hy, yaw, h=OBST_HEIGHT):
    """A wall: box centred at (x, y), half sizes (hx along its axis, hy across), rotated by yaw about z."""
    return {"type": "box", "x": float(x), "y": float(y), "hx": float(hx), "hy": float(hy), "yaw": float(yaw),
            "h": float(h), "r": float(np.hypot(hx, hy))}


def is_box(ob):
    return ob.get("type", "cylinder") == "box"


def box_axes(ob):
    """Unit axis (along hx) and unit normal (along hy) of a box."""
    c, s = np.cos(ob["yaw"]), np.sin(ob["yaw"])
    return np.array([c, s]), np.array([-s, c])


def obstacle_signed_distance(ob, pts):
    """Signed horizontal distance (m) from points (n, 2) to the obstacle surface; negative inside."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    rel = pts - np.array([ob["x"], ob["y"]])
    if not is_box(ob):
        return np.linalg.norm(rel, axis=1) - float(ob["r"])
    u, v = box_axes(ob)
    q = np.stack([rel @ u, rel @ v], axis=1)
    dq = np.abs(q) - np.array([ob["hx"], ob["hy"]])
    outside = np.linalg.norm(np.maximum(dq, 0.0), axis=1)
    inside = np.minimum(np.max(dq, axis=1), 0.0)
    return outside + inside


def segment_obstacle_distance(a, b, ob):
    """Smallest distance from the segment a-b to the obstacle surface (cylinders exact, boxes by sampling)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not is_box(ob):
        return sc_mod.segment_distance(a, b, [ob["x"], ob["y"]]) - float(ob["r"])
    n = int(np.linalg.norm(b - a) / SAMPLE_STEP) + 2
    pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
    return float(np.min(obstacle_signed_distance(ob, pts)))


def point_obstacle_distance(p, obstacles):
    """Smallest distance from point p to any obstacle surface (inf without obstacles)."""
    if not obstacles:
        return np.inf
    return float(min(obstacle_signed_distance(ob, np.asarray(p, dtype=float)[None, :])[0] for ob in obstacles))


# ---------------------------------------------------------------------------------------------------------------------
# Time model and validation
# ---------------------------------------------------------------------------------------------------------------------
def _angle(a, b):
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def estimate_nav_time(init_xy, init_yaw, route, dock_xy, dock_bearing):
    """Conservative duration (s) of the expert's navigation along start -> route -> dock (see the constants)."""
    pts = [np.asarray(init_xy, dtype=float)] + [np.asarray(w, dtype=float) for w in route] + \
          [np.asarray(dock_xy, dtype=float)]
    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
    heads = [float(np.arctan2(b[1] - a[1], b[0] - a[0])) for a, b in zip(pts, pts[1:])]
    turns = [_angle(heads[0], init_yaw)] + [_angle(h1, h0) for h0, h1 in zip(heads, heads[1:])] + \
            [_angle(dock_bearing, heads[-1])]
    return float(sum(l / V_EFF + LEG_OVERHEAD for l in lengths) + sum(t / TURN_EFF + TURN_OVERHEAD for t in turns))


def course_violations(init_xy, init_yaw, dock_xy, dock_bearing, target, obstacles, route, blocking=None,
                      nav_budget=None):
    """Every rule of the module docstring; returns the list of violated rules (empty = the expert cannot fail).

    blocking: indices of the obstacles that must block the straight start->dock line (default: all non-gate ones are
    chosen by the sampler; pass [] to skip the rule). nav_budget: largest accepted navigation-time estimate (s),
    default NAV_BUDGET (the mazes of maze.py pass their own)."""
    nav_budget = NAV_BUDGET if nav_budget is None else nav_budget
    init_xy, dock_xy = np.asarray(init_xy, dtype=float), np.asarray(dock_xy, dtype=float)
    route = [np.asarray(w, dtype=float) for w in route]
    target_xy = np.asarray(target[:2], dtype=float)
    points = [init_xy] + route + [dock_xy]
    bad = []
    for i, ob in enumerate(obstacles):
        tag = f"obstacle {i}"
        for j, p in enumerate(points):
            clear = obstacle_signed_distance(ob, p[None, :])[0]
            if clear < END_CLEAR:
                kind = "start" if j == 0 else ("dock" if j == len(points) - 1 else f"waypoint {j}")
                bad.append(f"{tag}: {kind} too close ({clear:.2f} m)")
        for j, (a, b) in enumerate(zip(points, points[1:])):
            if segment_obstacle_distance(a, b, ob) < COURSE_SEGMENT_CLEAR:
                bad.append(f"{tag}: leg {j} too close")
        if obstacle_signed_distance(ob, target_xy[None, :])[0] < TARGET_CLEAR:
            bad.append(f"{tag}: target too close")
        if segment_obstacle_distance(dock_xy, target_xy, ob) < ARM_CLEAR:
            bad.append(f"{tag}: reach line dock->target too close")
    for j, (a, b) in enumerate(zip(points, points[1:])):
        if np.linalg.norm(b - a) < MIN_LEG:
            bad.append(f"leg {j} shorter than {MIN_LEG} m")
    heads = [float(np.arctan2(b[1] - a[1], b[0] - a[0])) for a, b in zip(points, points[1:])]
    for j, (h0, h1) in enumerate(zip(heads, heads[1:]), start=1):
        if np.rad2deg(_angle(h1, h0)) > MAX_WAYPOINT_TURN_DEG:
            bad.append(f"turn at waypoint {j} above {MAX_WAYPOINT_TURN_DEG} deg")
    for i in (blocking or []):
        if segment_obstacle_distance(init_xy, dock_xy, obstacles[i]) >= CHASSIS_HALF_WIDTH:
            bad.append(f"obstacle {i} does not block the straight path")
    est = estimate_nav_time(init_xy, init_yaw, route, dock_xy, dock_bearing)
    if est > nav_budget:
        bad.append(f"estimated navigation time {est:.1f} s above {nav_budget} s")
    return bad


def verify_course(sc):
    """Violations of a (course) Scenario or its dict; empty list = valid. Blocking is checked for non-gate obstacles
    the same way the sampler does (meta['blocking'])."""
    sc = Scenario.from_dict(sc) if isinstance(sc, dict) else sc
    return course_violations(sc.init_xy, sc.init_yaw, sc.dock_xy, sc.dock_bearing, sc.target, sc.obstacles,
                             sc.route(), blocking=sc.meta.get("blocking", []))


# ---------------------------------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------------------------------
def _choose(rng, kinds):
    u, acc = float(rng.random()), 0.0
    for name, p in kinds:
        acc += p
        if u < acc:
            return name
    return kinds[-1][0]


def _build_feature(kind, centre, n, rng, side):
    """Obstacles and the waypoint of one feature at `centre`; `n` is the left normal of the start->dock line.
    Returns (obstacles, waypoint, info)."""
    centre = np.asarray(centre, dtype=float)
    if kind == "pillar":
        radius = float(rng.uniform(*OBST_RADIUS))
        wp = centre + side * (radius + WAYPOINT_CLEAR) * n
        return [{"x": float(centre[0]), "y": float(centre[1]), "r": radius, "h": OBST_HEIGHT}], wp, {"r": radius}
    tilt = np.deg2rad(rng.uniform(-WALL_TILT_DEG, WALL_TILT_DEG))
    yaw = float(np.arctan2(n[1], n[0]) + tilt)          # wall axis points to the left of the path
    u = np.array([np.cos(yaw), np.sin(yaw)])
    if kind == "wall":
        half = float(rng.uniform(*WALL_HALF_LEN))
        wp = centre + side * (half + WAYPOINT_CLEAR) * u   # WAYPOINT_CLEAR beyond the end face along the axis
        return [make_box(centre[0], centre[1], half, WALL_HALF_THICK, yaw)], wp, {"r": float(np.hypot(half, WALL_HALF_THICK))}
    # gate: two walls, waypoint at the gap centre
    tilt = np.deg2rad(rng.uniform(-GATE_TILT_DEG, GATE_TILT_DEG))
    yaw = float(np.arctan2(n[1], n[0]) + tilt)
    u = np.array([np.cos(yaw), np.sin(yaw)])
    half = float(rng.uniform(*GATE_WALL_HALF_LEN))
    off = GATE_HALF_GAP + half
    walls = [make_box(*(centre + sgn * off * u), half, WALL_HALF_THICK, yaw) for sgn in (-1.0, 1.0)]
    return walls, centre.copy(), {"r": float(np.hypot(half, WALL_HALF_THICK))}


def sample_course(seed, episode):
    """Draw one obstacle course from its own stream; deterministic in (seed, episode)."""
    rng = np.random.default_rng([int(seed), int(episode), COURSE_STREAM])
    for attempt in range(1, MAX_TRIES + 1):
        xy, yaw = sample_init_poses(1, rng, INIT_RADIUS, INIT_YAW_DEG)
        r, theta, z = sample_targets(1, rng, COURSE_TARGET_R, TARGET_THETA_DEG, TARGET_Z)
        init_xy, init_yaw = xy[0], float(yaw[0])
        target = relative_to_world(init_xy, init_yaw, float(r[0]), float(theta[0]), float(z[0]))
        dock_xy, dock_bearing, standoff = expert_base_goal(target, np.array([init_xy[0], init_xy[1], 0.0]), init_yaw)
        t, n, length = frame_of(init_xy, dock_xy)

        n_feat = 2 if rng.random() < P_TWO_FEATURES else 3
        obstacles, waypoints, feats, blocking = [], [], [], []
        prev_side = 0
        first = None
        for k in range(n_feat):
            frac = (k + 1) / (n_feat + 1) + float(rng.uniform(-ALONG_JITTER, ALONG_JITTER))
            kind = _choose(rng, FIRST_KINDS if k == 0 else LATER_KINDS)
            if kind == "gate":
                d, side, ambiguous = float(rng.uniform(-GATE_LATERAL, GATE_LATERAL)), 0, False
                coin = float(rng.random())                       # drawn anyway: fixed stream layout per feature
            elif k == 0:
                d = float(rng.uniform(-LATERAL, LATERAL))
                coin = float(rng.random())
                ambiguous = abs(d) < AMBIG_BAND
                side = (1 if coin < 0.5 else -1) if ambiguous else (-1 if d > 0 else 1)
            else:
                alt = rng.random() < P_ALTERNATE
                side = -prev_side if alt else prev_side
                d = -side * float(rng.uniform(*LATER_LATERAL))
                coin, ambiguous = float(rng.random()), False
            centre = np.asarray(init_xy) + frac * length * t + d * n
            obs, wp, info = _build_feature(kind, centre, n, rng, side)
            if kind != "gate":
                blocking.extend(range(len(obstacles), len(obstacles) + len(obs)))
                prev_side = side
            if first is None:
                first = {"along": frac, "d": d, "ambiguous": ambiguous, "side": side, "r": info["r"]}
            obstacles.extend(obs)
            waypoints.append(wp)
            feats.append({"kind": kind, "along": frac, "lateral_offset": d, "side": side})

        if course_violations(init_xy, init_yaw, dock_xy, dock_bearing, target, obstacles, waypoints, blocking):
            continue
        est = estimate_nav_time(init_xy, init_yaw, waypoints, dock_xy, dock_bearing)
        return Scenario(
            seed=int(seed), episode=int(episode), target=[float(v) for v in target],
            init_xy=[float(v) for v in init_xy], init_yaw=init_yaw,
            dock_xy=[float(v) for v in dock_xy], dock_bearing=float(dock_bearing), standoff=float(standoff),
            obstacles=obstacles, along=first["along"], lateral_offset=first["d"],
            ambiguous=bool(first["ambiguous"]), side=int(first["side"]),
            waypoint_xy=[float(v) for v in waypoints[0]], attempts=attempt,
            waypoints=[[float(v) for v in w] for w in waypoints], course=COURSE_FORMAT,
            meta={"features": feats, "blocking": blocking, "est_nav_time": est})
    raise RuntimeError(f"no valid course for seed {seed} episode {episode} in {MAX_TRIES} tries")


def course_config():
    """The constants that define the course distribution (store them with a set)."""
    names = ["COURSE_FORMAT", "COURSE_STREAM", "COURSE_TARGET_R", "P_TWO_FEATURES", "FIRST_KINDS", "LATER_KINDS",
             "ALONG_JITTER", "P_ALTERNATE", "LATER_LATERAL", "WALL_HALF_LEN", "WALL_HALF_THICK", "WALL_TILT_DEG",
             "GATE_HALF_GAP", "GATE_WALL_HALF_LEN", "GATE_LATERAL", "GATE_TILT_DEG", "COURSE_SEGMENT_CLEAR",
             "ARM_CLEAR", "MIN_LEG", "MAX_WAYPOINT_TURN_DEG", "V_EFF", "LEG_OVERHEAD", "TURN_EFF", "TURN_OVERHEAD",
             "NAV_BUDGET", "MAX_TRIES"]
    g = globals()
    return {k: (list(g[k]) if isinstance(g[k], tuple) else g[k]) for k in names}
