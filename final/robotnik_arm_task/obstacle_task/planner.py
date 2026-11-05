"""Route planner of the obstacle mazes (obstacle_task/maze.py): shortest-time waypoint route through arbitrary obstacles.

The expert drives start -> w1 -> ... -> wn -> dock leg by leg with the UNCHANGED base_controller (expert.py). What it
needs from a planner is therefore a short list of waypoints that obeys the same rules course.course_violations() checks:
  * every waypoint (and the start / dock) keeps END_CLEAR from every obstacle surface (the in-place turns there),
  * every leg keeps a clearance from every surface (planning uses PLAN_SEG_CLEAR, a margin above the validator's
    COURSE_SEGMENT_CLEAR, because legs are sampled coarsely here),
  * legs are at least MIN_LEG long, turns between legs are at most MAX_WAYPOINT_TURN_DEG (no reversals).
and it should be fast: the cost of a route is course.estimate_nav_time (leg length / V_EFF + LEG_OVERHEAD per leg,
turn / TURN_EFF + TURN_OVERHEAD per in-place turn, including the first turn from the start yaw and the last one to
the dock bearing), so the planner prefers few legs and gentle turns, not just short ones.

Graph: nodes = start, dock, a grid in the start->dock frame (GRID_STEP, +-GRID_LATERAL around the line) and points on a
ring around every obstacle (rounded-rectangle corners for boxes), all filtered by END_CLEAR. Edges = node pairs whose
straight segment keeps PLAN_SEG_CLEAR. Search = A* over (previous node, node) states, which is what makes the
turn limit and the turn cost exact. `first_side` forces the first waypoint to the left (+1) or the right (-1) of the
start->dock line, which gives the best route of the other homotopy class (used to decide whether a maze is ambiguous).
Nothing here needs the simulator.
"""
import heapq

import numpy as np

from obstacle_task import course
from obstacle_task.scenario import END_CLEAR, frame_of

PLAN_SEG_CLEAR = 0.60          # m, obstacle surface -> every leg (validator: course.COURSE_SEGMENT_CLEAR = 0.5)
NODE_CLEAR = END_CLEAR + 0.02  # m, obstacle surface -> every node
GRID_STEP = 0.5                # m
GRID_LATERAL = 2.8             # m, grid half-width around the start->dock line
RING_MARGIN = END_CLEAR + 0.08 # m, ring nodes around obstacles
MAX_EDGE = 3.4                 # m, longest leg considered
EDGE_SAMPLES = 35              # samples per leg: spacing <= MAX_EDGE / 34 = 0.1 m, so the distance error is <= 0.05 m
SIDE_MIN_LATERAL = 0.15        # m, a first waypoint must be this far from the start->dock line to count as a side


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ring_points(ob):
    """Nodes around one obstacle at RING_MARGIN from its surface (rounded-rectangle corners and sides for boxes)."""
    cx, cy = float(ob["x"]), float(ob["y"])
    if not course.is_box(ob):
        radius = float(ob["r"]) + RING_MARGIN
        ang = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)
    u, v = course.box_axes(ob)
    hx, hy = float(ob["hx"]) + RING_MARGIN, float(ob["hy"]) + RING_MARGIN
    local = [(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
    local += [(0.0, hy), (0.0, -hy), (hx, 0.0), (-hx, 0.0)]
    local += [(sx * hx * 0.5, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]      # long walls: more nodes along the side
    return np.array([[cx, cy] + a * u + b * v for a, b in local])


def build_nodes(init_xy, dock_xy, obstacles):
    """(n, 2) node array: row 0 = start, row 1 = dock, then the filtered grid and ring points."""
    init_xy, dock_xy = np.asarray(init_xy, dtype=float), np.asarray(dock_xy, dtype=float)
    t, nrm, length = frame_of(init_xy, dock_xy)
    along = np.arange(0.0, length + 1e-9, GRID_STEP)
    lateral = np.arange(-GRID_LATERAL, GRID_LATERAL + 1e-9, GRID_STEP)
    aa, ll = np.meshgrid(along, lateral, indexing="ij")
    cand = init_xy[None, :] + aa.reshape(-1, 1) * t[None, :] + ll.reshape(-1, 1) * nrm[None, :]
    if obstacles:
        cand = np.vstack([cand] + [_ring_points(ob) for ob in obstacles])
        clear = np.min([course.obstacle_signed_distance(ob, cand) for ob in obstacles], axis=0)
        cand = cand[clear >= NODE_CLEAR]
    far_from_ends = (np.linalg.norm(cand - init_xy, axis=1) >= course.MIN_LEG * 0.5) & \
                    (np.linalg.norm(cand - dock_xy, axis=1) >= course.MIN_LEG * 0.5)
    return np.vstack([init_xy[None, :], dock_xy[None, :], cand[far_from_ends]])


def build_edges(nodes, obstacles):
    """Boolean (n, n) matrix: the straight leg between two nodes is allowed (length in [MIN_LEG, MAX_EDGE], clearance
    >= PLAN_SEG_CLEAR from every obstacle). Symmetric; no edges from the dock node."""
    n = len(nodes)
    ii, jj = np.triu_indices(n, k=1)
    length = np.linalg.norm(nodes[jj] - nodes[ii], axis=1)
    keep = (length >= course.MIN_LEG) & (length <= MAX_EDGE)
    ii, jj, length = ii[keep], jj[keep], length[keep]
    ok = np.ones(len(ii), dtype=bool)
    a, b = nodes[ii], nodes[jj]
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    s = np.linspace(0.0, 1.0, EDGE_SAMPLES)
    for ob in obstacles:
        centre = np.array([ob["x"], ob["y"]], dtype=float)
        s0 = np.clip(np.einsum("ij,ij->i", centre - a, ab) / ab2, 0.0, 1.0)
        near = np.linalg.norm(a + s0[:, None] * ab - centre, axis=1) < float(ob["r"]) + PLAN_SEG_CLEAR + 0.15
        idx = np.flatnonzero(ok & near)          # only the legs that pass close to this obstacle need the exact test
        if idx.size == 0:
            continue
        pts = a[idx, None, :] + s[None, :, None] * ab[idx, None, :]
        dist = course.obstacle_signed_distance(ob, pts.reshape(-1, 2)).reshape(len(idx), EDGE_SAMPLES).min(axis=1)
        ok[idx[dist < PLAN_SEG_CLEAR]] = False
    valid = np.zeros((n, n), dtype=bool)
    valid[ii[ok], jj[ok]] = True
    valid = valid | valid.T
    valid[1, :] = False                          # the route ends at the dock
    return valid


def search(nodes, valid, init_yaw, dock_bearing, first_side=None):
    """A* over (previous node, node) states. Returns (list of node indices start..dock, cost in seconds) or None."""
    n = len(nodes)
    delta = nodes[None, :, :] - nodes[:, None, :]
    dist = np.linalg.norm(delta, axis=2)
    head = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    leg_cost = dist / course.V_EFF + course.LEG_OVERHEAD
    max_turn = np.deg2rad(course.MAX_WAYPOINT_TURN_DEG)
    t, nrm, _ = frame_of(nodes[0], nodes[1])
    lateral = (nodes - nodes[0][None, :]) @ nrm
    heuristic = np.linalg.norm(nodes - nodes[1][None, :], axis=1) / course.V_EFF
    adj = [np.flatnonzero(valid[i]) for i in range(n)]

    def turn_cost(angle):
        return np.abs(angle) / course.TURN_EFF + course.TURN_OVERHEAD

    best = {}
    parent = {}
    heap = []
    start_state = (-1, 0)
    best[start_state] = 0.0
    heapq.heappush(heap, (heuristic[0], 0.0, -1, 0))
    while heap:
        _, g, p, c = heapq.heappop(heap)
        if g > best.get((p, c), np.inf) + 1e-12:
            continue
        if c == 1:
            path, state = [], (p, c)
            while state[1] != 0:
                path.append(state[1])
                state = parent[state]
            path.append(0)
            return path[::-1], g
        nxt = adj[c]
        if p < 0:
            if first_side is not None:
                nxt = nxt[(nxt != 1) & (first_side * lateral[nxt] > SIDE_MIN_LATERAL)]
            turn = _wrap(head[c, nxt] - init_yaw)
            ok = np.ones(len(nxt), dtype=bool)
        else:
            turn = _wrap(head[c, nxt] - head[p, c])
            ok = (np.abs(turn) <= max_turn) & (nxt != p)
        for m, tn in zip(nxt[ok], turn[ok]):
            g2 = g + leg_cost[c, m] + float(turn_cost(tn))
            if m == 1:
                g2 += float(turn_cost(_wrap(dock_bearing - head[c, m])))
                f = g2
            else:
                f = g2 + heuristic[m]
            if g2 < best.get((c, m), np.inf) - 1e-12:
                best[(c, m)] = g2
                parent[(c, m)] = (p, c)
                heapq.heappush(heap, (f, g2, c, int(m)))
    return None


def plan_route(init_xy, init_yaw, dock_xy, dock_bearing, obstacles, first_side=None, graph=None):
    """Waypoints (list of [x, y], without start and dock) of the best route and its estimated duration (s), or None.

    graph: optional (nodes, valid) from build_nodes / build_edges to reuse between calls of the same maze."""
    nodes, valid = graph if graph is not None else (None, None)
    if nodes is None:
        nodes = build_nodes(init_xy, dock_xy, obstacles)
        valid = build_edges(nodes, obstacles)
    found = search(nodes, valid, init_yaw, dock_bearing, first_side)
    if found is None:
        return None
    path, cost = found
    return [nodes[k].tolist() for k in path[1:-1]], float(cost)


def plan_with_alternative(init_xy, init_yaw, dock_xy, dock_bearing, obstacles):
    """Best route plus the best route that leaves the start->dock line on the other side of the first waypoint.

    Returns (route, cost, alt_cost) with alt_cost = inf when no route of the other side exists, or None if the maze
    has no route at all."""
    nodes = build_nodes(init_xy, dock_xy, obstacles)
    valid = build_edges(nodes, obstacles)
    best = plan_route(init_xy, init_yaw, dock_xy, dock_bearing, obstacles, graph=(nodes, valid))
    if best is None or not best[0]:
        return None
    route, cost = best
    _, nrm, _ = frame_of(init_xy, dock_xy)
    side = 1 if float((np.asarray(route[0]) - np.asarray(init_xy)) @ nrm) >= 0.0 else -1
    other = plan_route(init_xy, init_yaw, dock_xy, dock_bearing, obstacles, first_side=-side, graph=(nodes, valid))
    return route, cost, (other[1] if other is not None else float("inf"))
