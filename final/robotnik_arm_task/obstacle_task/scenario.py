"""Scenarios of the obstacle task (step 9, B1 "fork in the road").

One scenario is everything needed to replay an episode: the start pose, the target, one static cylinder that
blocks the straight path from the start to the docking point, and the expert's detour side.

Geometry (all in the world frame, 2D unless stated):
  - start pose and target are drawn like the A-to-B task (start uniform in a disc, target relative to the start),
    but farther away (TARGET_R) so that a detour fits between the start and the docking point;
  - the docking point and its bearing are exactly the A-to-B expert's (evaluate_policy.expert_base_goal);
  - the obstacle centre sits at a fraction `along` of the start->dock segment, shifted by `lateral_offset` d
    along the left normal n of that segment; |d| <= LATERAL < r + chassis half-width, so the straight
    path always hits it;
  - side +1 = pass on the left of the obstacle (waypoint at centre + (r + WAYPOINT_CLEAR) n), -1 = right.
    Inside the ambiguous band |d| < AMBIG_BAND the side is a fair coin (genuinely multimodal demonstrations);
    outside it the expert takes the shorter detour, side = -sign(d);
  - rejection constraints keep the in-place turns at the start, the waypoint and the dock clear of the
    obstacle (the arm sticks out ~0.73 m in front of the base centre) and both straight segments clear of it.

Randomness: every episode uses its own stream np.random.default_rng([seed, episode, STREAM]), independent of the
A-to-B collectors (they use other seeds and orders), so step 9 can never change an existing data or eval set.
The noisy flag of an episode uses a second stream [seed, episode, STREAM + 1].
"""
import json
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from collect_data import relative_to_world, sample_init_poses, sample_targets  # noqa: E402
from evaluate_policy import expert_base_goal  # noqa: E402

FORMAT = "obstacle_scenarios_v1"
STREAM = 9                    # third entry of the per-episode SeedSequence

TARGET_R = (3.5, 5.0)         # m, target distance from the start pose (A-to-B: 0.1-3.0)
TARGET_THETA_DEG = (-180.0, 180.0)
TARGET_Z = (0.2, 1.2)         # m, as A-to-B
INIT_RADIUS = 2.0             # m, start position uniform in this disc around the origin, as A-to-B
INIT_YAW_DEG = (-180.0, 180.0)

OBST_RADIUS = (0.25, 0.35)    # m, cylinder radius
OBST_HEIGHT = 1.5             # m, taller than the arm (EE start height ~0.76 m), so the arm cannot pass over it
ALONG = (0.3, 0.7)            # obstacle centre as a fraction of the start->dock segment
LATERAL = 0.4                 # m, |d| <= LATERAL; the straight path is blocked because d < r_min + 0.31
AMBIG_BAND = 0.2              # m, |d| < AMBIG_BAND: the side is a fair coin
CHASSIS_HALF_WIDTH = 0.31     # m, half-width of the robot including the wheels (grounding measurement)
WAYPOINT_CLEAR = 0.9          # m, obstacle surface -> waypoint (in-place turn there: arm sweep ~0.73 m + margin)
END_CLEAR = 0.9               # m, obstacle surface -> start position and -> dock (in-place turns there too)
SEGMENT_CLEAR = 0.45          # m, obstacle surface -> both straight segments (chassis half-width 0.31 + margin)
TARGET_CLEAR = 0.30           # m, obstacle surface -> target (horizontal), keeps the reach away from the obstacle
MAX_TRIES = 10000


@dataclass
class Scenario:
    seed: int
    episode: int
    target: list                 # [x, y, z] world
    init_xy: list                # [x, y] world
    init_yaw: float              # rad
    dock_xy: list                # [x, y] world, the A-to-B expert's docking point
    dock_bearing: float          # rad, bearing the base faces at the dock (towards the target)
    standoff: float              # m
    obstacles: list              # [{"x", "y", "r", "h"}], cylinders standing on the floor
    along: float                 # fraction of the start->dock segment
    lateral_offset: float        # m, signed offset d along the left normal of start->dock
    ambiguous: bool              # |d| < AMBIG_BAND
    side: int                    # +1 pass left of the obstacle, -1 right
    waypoint_xy: list            # [x, y] world
    noisy: bool = False          # DART-style command noise in data collection (decided by make_scenarios)
    attempts: int = 1            # rejection-sampling attempts used
    meta: dict = field(default_factory=dict)
    waypoints: list = field(default_factory=list)   # [[x, y], ...] all waypoints in driving order (obstacle courses);
    #                              empty = the single-obstacle scenario, whose only waypoint is waypoint_xy
    course: str = ""             # "" = single obstacle (fork in the road); course.COURSE_FORMAT = multi-obstacle course

    def route(self):
        """Waypoints (n, 2) the expert drives through between the start and the dock, in order."""
        if self.waypoints:
            return np.asarray(self.waypoints, dtype=float).reshape(-1, 2)
        if self.obstacles:
            return np.asarray(self.waypoint_xy, dtype=float).reshape(1, 2)
        return np.zeros((0, 2))

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Scenario(**d)


def segment_distance(a, b, p):
    """Distance from point p to the segment a-b (2D)."""
    a, b, p = (np.asarray(v, dtype=float) for v in (a, b, p))
    ab = b - a
    denom = float(ab @ ab)
    s = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + s * ab - p))


def frame_of(init_xy, dock_xy):
    """Unit tangent t (start -> dock), left normal n and the segment length."""
    seg = np.asarray(dock_xy, dtype=float) - np.asarray(init_xy, dtype=float)
    length = float(np.linalg.norm(seg))
    t = seg / length
    return t, np.array([-t[1], t[0]]), length


def lateral_position(xy, init_xy, dock_xy):
    """Signed distance of xy from the start->dock line (positive on the left)."""
    t, n, _ = frame_of(init_xy, dock_xy)
    return float((np.asarray(xy, dtype=float) - np.asarray(init_xy, dtype=float)) @ n)


def constraints_ok(init_xy, dock_xy, target, center, radius, waypoint):
    """The rejection constraints of the sampler; returns (ok, reasons)."""
    c = np.asarray(center, dtype=float)
    reasons = []
    if np.linalg.norm(np.asarray(init_xy) - c) - radius < END_CLEAR:
        reasons.append("start too close")
    if np.linalg.norm(np.asarray(dock_xy) - c) - radius < END_CLEAR:
        reasons.append("dock too close")
    if segment_distance(init_xy, waypoint, c) - radius < SEGMENT_CLEAR:
        reasons.append("segment start->waypoint too close")
    if segment_distance(waypoint, dock_xy, c) - radius < SEGMENT_CLEAR:
        reasons.append("segment waypoint->dock too close")
    if np.linalg.norm(np.asarray(target[:2]) - c) - radius < TARGET_CLEAR:
        reasons.append("target too close")
    if segment_distance(init_xy, dock_xy, c) - radius >= CHASSIS_HALF_WIDTH:
        reasons.append("straight path not blocked")
    return not reasons, reasons


def sample_scenario(seed, episode):
    """Draw one scenario from its own stream; deterministic in (seed, episode)."""
    rng = np.random.default_rng([int(seed), int(episode), STREAM])
    for attempt in range(1, MAX_TRIES + 1):
        xy, yaw = sample_init_poses(1, rng, INIT_RADIUS, INIT_YAW_DEG)
        r, theta, z = sample_targets(1, rng, TARGET_R, TARGET_THETA_DEG, TARGET_Z)
        init_xy, init_yaw = xy[0], float(yaw[0])
        target = relative_to_world(init_xy, init_yaw, float(r[0]), float(theta[0]), float(z[0]))
        dock_xy, dock_bearing, standoff = expert_base_goal(target, np.array([init_xy[0], init_xy[1], 0.0]), init_yaw)
        t, n, length = frame_of(init_xy, dock_xy)
        along = float(rng.uniform(*ALONG))
        d = float(rng.uniform(-LATERAL, LATERAL))
        radius = float(rng.uniform(*OBST_RADIUS))
        coin = float(rng.random())                     # always drawn: the stream does not depend on d
        center = np.asarray(init_xy) + along * length * t + d * n
        ambiguous = abs(d) < AMBIG_BAND
        side = (1 if coin < 0.5 else -1) if ambiguous else (-1 if d > 0 else 1)
        waypoint = center + side * (radius + WAYPOINT_CLEAR) * n
        ok, _ = constraints_ok(init_xy, dock_xy, target, center, radius, waypoint)
        if ok:
            return Scenario(
                seed=int(seed), episode=int(episode), target=[float(v) for v in target],
                init_xy=[float(v) for v in init_xy], init_yaw=init_yaw,
                dock_xy=[float(v) for v in dock_xy], dock_bearing=float(dock_bearing), standoff=float(standoff),
                obstacles=[{"x": float(center[0]), "y": float(center[1]), "r": radius, "h": OBST_HEIGHT}],
                along=along, lateral_offset=d, ambiguous=bool(ambiguous), side=int(side),
                waypoint_xy=[float(v) for v in waypoint], attempts=attempt)
    raise RuntimeError(f"no valid scenario for seed {seed} episode {episode} in {MAX_TRIES} tries")


def make_scenarios(seed, n_episodes, noise_frac=0.0, course=False):
    """Scenarios 0..n-1 of a set; the noisy flag of each comes from its own stream.
    course=True draws multi-obstacle courses (obstacle_task/course.py) instead of the single obstacle."""
    if course:
        from obstacle_task.course import sample_course     # lazy: course.py imports this module
        draw = sample_course
    else:
        draw = sample_scenario
    # (every scenario gets its noisy flag from its own stream)
    out = []
    for ep in range(int(n_episodes)):
        sc = draw(seed, ep)
        sc.noisy = bool(np.random.default_rng([int(seed), ep, STREAM + 1]).random() < noise_frac)
        out.append(sc)
    return out


def sampler_config():
    """The constants that define the scenario distribution (stored with every set)."""
    names = ["FORMAT", "STREAM", "TARGET_R", "TARGET_THETA_DEG", "TARGET_Z", "INIT_RADIUS", "INIT_YAW_DEG",
             "OBST_RADIUS", "OBST_HEIGHT", "ALONG", "LATERAL", "AMBIG_BAND", "CHASSIS_HALF_WIDTH",
             "WAYPOINT_CLEAR", "END_CLEAR", "SEGMENT_CLEAR", "TARGET_CLEAR", "MAX_TRIES"]
    g = globals()
    return {k: (list(g[k]) if isinstance(g[k], tuple) else g[k]) for k in names}


def save_scenarios(path, scenarios, extra=None):
    payload = {"format": FORMAT, "sampler": sampler_config(), "extra": extra or {},
               "scenarios": [s.to_dict() for s in scenarios]}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def load_scenarios(path):
    with open(path) as f:
        payload = json.load(f)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: format {payload.get('format')!r}, expected {FORMAT!r}")
    return [Scenario.from_dict(d) for d in payload["scenarios"]], payload
