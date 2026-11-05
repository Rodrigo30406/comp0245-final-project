"""Obstacle mazes (harder than the courses of course.py): obstacle layouts the robot must plan a path through.

A maze is a Scenario (course = MAZE_FORMAT) with 2-3 obstacle FEATURES between the start and the dock. Besides the
pillar, short wall and gate of course.py it has
  long_wall  a wall 2-3 m long: going around it is expensive, so the route is long and usually on one specific side;
  door       two collinear long walls with a doorway of 2 * (END_CLEAR + 0.1) = 2 m at a random place: the route must
             go through the door (the walls are too long to go around within the time budget);
  u_trap     three walls forming a U (a cul-de-sac) that opens towards the start or the dock: the straight way leads
             into the pocket, the route has to go around the whole U;
  pylons     two pillars with a doorway between them;
  chicane    two long walls one behind the other that overlap sideways: the route weaves around the free end of the
             first one and then the free end of the second one, on the other side.
There are no hand-made waypoints: planner.plan_route finds the best route (graph search with the turn limit and the
clearances of course.course_violations), the sampler rejects a maze without a route, a maze whose straight start->dock
line is not blocked and a maze whose route has fewer than two waypoints (a single detour is a course, not a maze).
Whatever the planner returns is re-checked by course.course_violations (same rules as for the courses: the expert
cannot fail), with the maze's own navigation budget MAZE_NAV_BUDGET.

side / ambiguous: side = which side of the start->dock line the first waypoint lies on (+1 left); the maze is
ambiguous when the best route that starts on the other side costs at most AMBIG_COST_RATIO times as much, so that
both detours are plausible demonstrations (as the ambiguous band of the single obstacle).
The stream is np.random.default_rng([seed, episode, MAZE_STREAM]): independent of the single-obstacle and course streams.
"""
import os
import sys

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from collect_data import relative_to_world, sample_init_poses, sample_targets  # noqa: E402
from evaluate_policy import expert_base_goal  # noqa: E402
from obstacle_task import course, planner  # noqa: E402
from obstacle_task.scenario import (  # noqa: E402
    CHASSIS_HALF_WIDTH, END_CLEAR, INIT_RADIUS, INIT_YAW_DEG, OBST_HEIGHT, OBST_RADIUS, TARGET_THETA_DEG, TARGET_Z,
    Scenario, frame_of,
)

MAZE_FORMAT = "obstacle_maze_v1"
MAZE_STREAM = 30                   # third entry of the per-episode SeedSequence (single 9/10, courses 20)

MAZE_TARGET_R = (3.2, 4.6)         # m, target distance from the start pose
MAZE_NAV_BUDGET = 60.0             # s, largest accepted estimate. The estimate is very conservative: in the pilot (results/maze/pilot)
#                                    the measured navigation took 0.55-0.65 of it, so 60 s means ~36 s of the 45 s timeout
P_TWO_FEATURES = 0.5               # otherwise three
KINDS = (("pillar", 0.12), ("wall", 0.08), ("long_wall", 0.16), ("gate", 0.10), ("door", 0.16), ("u_trap", 0.14),
         ("pylons", 0.10), ("chicane", 0.14))
ALONG_RANGE = (0.24, 0.76)         # features sit at evenly spaced fractions of the start->dock line inside this range
ALONG_JITTER = 0.04
LATERAL = 0.55                     # m, |lateral offset| of a feature centre from the start->dock line
AMBIG_COST_RATIO = 1.15
MIN_WAYPOINTS = 2
BLOCK_MARGIN = CHASSIS_HALF_WIDTH + 0.05
MAX_TRIES = 3000

WALL_HALF_THICK = 0.1
LONG_WALL_HALF_LEN = (1.0, 1.5)
LONG_WALL_TILT_DEG = 25.0
DOOR_HALF_GAP = END_CLEAR + 0.1    # m, doorway centre -> inner face of each door wall
DOOR_WALL_HALF_LEN = (0.9, 1.4)
DOOR_SHIFT = 0.8                   # m, |doorway centre shift| along the wall axis
DOOR_TILT_DEG = 15.0
U_HALF_WIDTH = (0.65, 0.85)        # m, inner half-width of the pocket
U_DEPTH = (1.0, 1.5)               # m, arm length
U_TILT_DEG = 20.0
P_U_OPENS_TO_START = 0.7
PYLON_TILT_DEG = 20.0
CHICANE_SEP = (1.8, 2.3)           # m, distance between the two walls along the path
CHICANE_HALF_LEN = (1.0, 1.3)
CHICANE_OVERLAP = (0.0, 0.35)      # m, how far the free ends overlap sideways
CHICANE_TILT_DEG = 8.0


def _axis(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)])


def _tilt(rng, deg):
    return float(np.deg2rad(rng.uniform(-deg, deg)))


def _pillar(centre, radius):
    return {"x": float(centre[0]), "y": float(centre[1]), "r": float(radius), "h": OBST_HEIGHT}


def build_feature(kind, centre, t, n, rng):
    """Obstacles of one feature centred at `centre` (t: path tangent, n: its left normal). Returns the obstacle list.
    Walls are boxes (course.make_box); every obstacle carries \"r\" (bounding radius)."""
    centre = np.asarray(centre, dtype=float)
    yaw_n = float(np.arctan2(n[1], n[0]))          # a wall across the path
    yaw_t = float(np.arctan2(t[1], t[0]))          # a wall along the path
    if kind == "pillar":
        return [_pillar(centre, rng.uniform(*OBST_RADIUS))]
    if kind in ("wall", "gate"):
        side = 1                                     # the course builder's waypoint (ignored here) takes a side
        return course._build_feature(kind, centre, n, rng, side)[0]
    if kind == "long_wall":
        yaw = yaw_n + _tilt(rng, LONG_WALL_TILT_DEG)
        return [course.make_box(centre[0], centre[1], rng.uniform(*LONG_WALL_HALF_LEN), WALL_HALF_THICK, yaw)]
    if kind in ("door", "pylons"):
        yaw = yaw_n + _tilt(rng, DOOR_TILT_DEG if kind == "door" else PYLON_TILT_DEG)
        u = _axis(yaw)
        gap_centre = centre + (rng.uniform(-DOOR_SHIFT, DOOR_SHIFT) if kind == "door" else 0.0) * u
        if kind == "door":
            halves = [float(rng.uniform(*DOOR_WALL_HALF_LEN)) for _ in range(2)]
            return [course.make_box(*(gap_centre + sgn * (DOOR_HALF_GAP + h) * u), h, WALL_HALF_THICK, yaw)
                    for sgn, h in zip((-1.0, 1.0), halves)]
        radii = [float(rng.uniform(*OBST_RADIUS)) for _ in range(2)]
        return [_pillar(gap_centre + sgn * (DOOR_HALF_GAP + r) * u, r) for sgn, r in zip((-1.0, 1.0), radii)]
    if kind == "u_trap":
        opens_to_start = rng.random() < P_U_OPENS_TO_START
        yaw_b = yaw_t + _tilt(rng, U_TILT_DEG)       # "forward" of the U: the direction of the closed back wall
        b, a = _axis(yaw_b), _axis(yaw_b + np.pi / 2)
        width, depth = float(rng.uniform(*U_HALF_WIDTH)), float(rng.uniform(*U_DEPTH))
        s = 1.0 if opens_to_start else -1.0          # back wall on the dock side (opening towards the start) or reversed
        back = centre + s * (depth / 2) * b
        return [course.make_box(back[0], back[1], width + WALL_HALF_THICK, WALL_HALF_THICK, yaw_b + np.pi / 2),
                *(course.make_box(*(centre + sgn * width * a), depth / 2 + WALL_HALF_THICK, WALL_HALF_THICK, yaw_b)
                  for sgn in (-1.0, 1.0))]
    if kind == "chicane":
        sep = float(rng.uniform(*CHICANE_SEP))
        side = 1.0 if rng.random() < 0.5 else -1.0   # side of the FIRST wall's free end
        overlap = [float(rng.uniform(*CHICANE_OVERLAP)) for _ in range(2)]
        halves = [float(rng.uniform(*CHICANE_HALF_LEN)) for _ in range(2)]
        # wall 1 (nearer the start): free end at lateral side * overlap[0], extends towards -side;
        # wall 2: free end at lateral -side * overlap[1], extends towards +side. The robot passes wall 1 on the +side
        # side of its free end and wall 2 on the other one.
        lateral = (side * (overlap[0] - halves[0]), side * (halves[1] - overlap[1]))
        out = []
        for k, along in enumerate((-sep / 2, sep / 2)):
            pos = centre + along * t + lateral[k] * n
            out.append(course.make_box(pos[0], pos[1], halves[k], WALL_HALF_THICK,
                                       yaw_n + _tilt(rng, CHICANE_TILT_DEG)))
        return out
    raise ValueError(f"unknown feature kind {kind!r}")


def _choose(rng):
    u, acc = float(rng.random()), 0.0
    for name, p in KINDS:
        acc += p
        if u < acc:
            return name
    return KINDS[-1][0]


def sample_maze(seed, episode):
    """Draw one maze from its own stream; deterministic in (seed, episode)."""
    rng = np.random.default_rng([int(seed), int(episode), MAZE_STREAM])
    for attempt in range(1, MAX_TRIES + 1):
        xy, yaw = sample_init_poses(1, rng, INIT_RADIUS, INIT_YAW_DEG)
        r, theta, z = sample_targets(1, rng, MAZE_TARGET_R, TARGET_THETA_DEG, TARGET_Z)
        init_xy, init_yaw = xy[0], float(yaw[0])
        target = relative_to_world(init_xy, init_yaw, float(r[0]), float(theta[0]), float(z[0]))
        dock_xy, dock_bearing, standoff = expert_base_goal(target, np.array([init_xy[0], init_xy[1], 0.0]), init_yaw)
        t, n, length = frame_of(init_xy, dock_xy)

        n_feat = 2 if rng.random() < P_TWO_FEATURES else 3
        obstacles, feats = [], []
        lo, hi = ALONG_RANGE
        for k in range(n_feat):
            frac = lo + (hi - lo) * (k + 0.5) / n_feat + float(rng.uniform(-ALONG_JITTER, ALONG_JITTER))
            kind = _choose(rng)
            d = float(rng.uniform(-LATERAL, LATERAL))
            centre = np.asarray(init_xy) + frac * length * t + d * n
            obs = build_feature(kind, centre, t, n, rng)
            obstacles.extend(obs)
            feats.append({"kind": kind, "along": frac, "lateral_offset": d, "n_obstacles": len(obs)})

        if not any(course.segment_obstacle_distance(init_xy, dock_xy, ob) < BLOCK_MARGIN for ob in obstacles):
            continue                                  # the straight line is open: nothing to plan
        planned = planner.plan_with_alternative(init_xy, init_yaw, dock_xy, dock_bearing, obstacles)
        if planned is None:
            continue
        waypoints, cost, alt_cost = planned
        if len(waypoints) < MIN_WAYPOINTS:
            continue
        if course.course_violations(init_xy, init_yaw, dock_xy, dock_bearing, target, obstacles, waypoints, [],
                                    nav_budget=MAZE_NAV_BUDGET):
            continue
        side = 1 if float((np.asarray(waypoints[0]) - init_xy) @ n) >= 0.0 else -1
        ambiguous = bool(alt_cost <= AMBIG_COST_RATIO * cost)
        est = course.estimate_nav_time(init_xy, init_yaw, waypoints, dock_xy, dock_bearing)
        return Scenario(
            seed=int(seed), episode=int(episode), target=[float(v) for v in target],
            init_xy=[float(v) for v in init_xy], init_yaw=init_yaw,
            dock_xy=[float(v) for v in dock_xy], dock_bearing=float(dock_bearing), standoff=float(standoff),
            obstacles=obstacles, along=feats[0]["along"], lateral_offset=float(
                (np.asarray(waypoints[0]) - init_xy) @ n),
            ambiguous=ambiguous, side=side, waypoint_xy=[float(v) for v in waypoints[0]], attempts=attempt,
            waypoints=[[float(v) for v in w] for w in waypoints], course=MAZE_FORMAT,
            meta={"features": feats, "blocking": [], "est_nav_time": est, "plan_cost": cost,
                  "alt_cost": None if not np.isfinite(alt_cost) else alt_cost})
    raise RuntimeError(f"no valid maze for seed {seed} episode {episode} in {MAX_TRIES} tries")


def verify_maze(sc):
    """Violations of a maze Scenario (or its dict): the course rules with the maze budget, plus the maze rules (the
    straight line is blocked, at least MIN_WAYPOINTS waypoints). Empty list = valid."""
    sc = Scenario.from_dict(sc) if isinstance(sc, dict) else sc
    bad = course.course_violations(sc.init_xy, sc.init_yaw, sc.dock_xy, sc.dock_bearing, sc.target, sc.obstacles,
                                   sc.route(), [], nav_budget=MAZE_NAV_BUDGET)
    if not any(course.segment_obstacle_distance(sc.init_xy, sc.dock_xy, ob) < BLOCK_MARGIN for ob in sc.obstacles):
        bad.append("the straight start->dock line is open")
    if len(sc.route()) < MIN_WAYPOINTS:
        bad.append(f"fewer than {MIN_WAYPOINTS} waypoints")
    return bad


def maze_config():
    """The constants that define the maze distribution (store them with a set)."""
    names = ["MAZE_FORMAT", "MAZE_STREAM", "MAZE_TARGET_R", "MAZE_NAV_BUDGET", "P_TWO_FEATURES", "KINDS", "ALONG_RANGE",
             "ALONG_JITTER", "LATERAL", "AMBIG_COST_RATIO", "MIN_WAYPOINTS", "MAX_TRIES", "LONG_WALL_HALF_LEN",
             "LONG_WALL_TILT_DEG", "DOOR_HALF_GAP", "DOOR_WALL_HALF_LEN", "DOOR_SHIFT", "DOOR_TILT_DEG", "U_HALF_WIDTH",
             "U_DEPTH", "U_TILT_DEG", "P_U_OPENS_TO_START", "PYLON_TILT_DEG", "CHICANE_SEP", "CHICANE_HALF_LEN",
             "CHICANE_OVERLAP", "CHICANE_TILT_DEG"]
    g = globals()
    def plain(v):
        return [plain(x) for x in v] if isinstance(v, (tuple, list)) else v

    cfg = {k: plain(g[k]) for k in names}          # JSON-friendly
    cfg.update({f"planner.{k}": getattr(planner, k) for k in
                ("PLAN_SEG_CLEAR", "NODE_CLEAR", "GRID_STEP", "GRID_LATERAL", "RING_MARGIN", "MAX_EDGE")})
    return cfg
