"""Tests of the obstacle mazes and their planner (obstacle_task/maze.py, planner.py). No simulator needed."""
import os
import sys
import unittest

import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from obstacle_task import course, maze, planner, scenario  # noqa: E402


class MazeSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scs = scenario.make_scenarios(11, 30, noise_frac=0.0, course="maze")

    def test_every_maze_is_valid(self):
        for sc in self.scs:
            self.assertEqual(maze.verify_maze(sc), [], f"episode {sc.episode}")
            self.assertEqual(sc.course, maze.MAZE_FORMAT)

    def test_deterministic(self):
        again = scenario.make_scenarios(11, 5, course="maze")
        self.assertEqual([s.to_dict() for s in again], [s.to_dict() for s in self.scs[:5]])

    def test_needs_planning_and_has_new_kinds(self):
        self.assertTrue(all(len(sc.route()) >= maze.MIN_WAYPOINTS for sc in self.scs))
        kinds = {f["kind"] for sc in self.scs for f in sc.meta["features"]}
        self.assertTrue(kinds & {"long_wall", "door", "u_trap", "pylons", "chicane"})

    def test_compatible_with_single_obstacle_code(self):
        for sc in self.scs:
            self.assertTrue(all("r" in ob for ob in sc.obstacles))
            self.assertIn(sc.side, (-1, 1))
            self.assertEqual(sc.waypoint_xy, sc.waypoints[0])

    def test_every_feature_kind_builds(self):
        rng = np.random.default_rng(0)
        t, n = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        for kind, _ in maze.KINDS:
            obs = maze.build_feature(kind, [2.0, 0.0], t, n, rng)
            self.assertTrue(obs and all("r" in ob for ob in obs), kind)


class Planner(unittest.TestCase):
    def test_open_field_has_no_waypoints(self):
        route, cost = planner.plan_route([0, 0], 0.0, [3, 0], 0.0, [])
        self.assertEqual(route, [])

    def test_goes_around_a_long_wall_and_keeps_clearance(self):
        wall = course.make_box(1.5, 0.0, 1.2, 0.1, np.pi / 2)        # across the straight line
        route, cost = planner.plan_route([0, 0], 0.0, [3, 0], 0.0, [wall])
        self.assertGreaterEqual(len(route), 2)
        pts = [np.array([0.0, 0.0])] + [np.array(w) for w in route] + [np.array([3.0, 0.0])]
        for a, b in zip(pts, pts[1:]):
            self.assertGreaterEqual(course.segment_obstacle_distance(a, b, wall), course.COURSE_SEGMENT_CLEAR)

    def test_door_is_used_when_walls_are_long(self):
        # two long walls along y (yaw = pi/2) centred at y = +-2.4 leave the doorway |y| < 1.0 on the straight line
        walls = [course.make_box(1.5, sgn * 2.4, 1.4, 0.1, np.pi / 2) for sgn in (-1, 1)]
        route, _ = planner.plan_route([0, 0], 0.0, [3, 0], 0.0, walls)
        self.assertEqual(route, [])                                   # straight through the doorway

    def test_alternative_side_exists_for_a_symmetric_pillar(self):
        pillar = {"x": 1.5, "y": 0.0, "r": 0.3, "h": 1.5}
        out = planner.plan_with_alternative([0, 0], 0.0, [3, 0], 0.0, [pillar])
        self.assertIsNotNone(out)
        self.assertTrue(np.isfinite(out[2]))


if __name__ == "__main__":
    unittest.main()
