"""Tests of the obstacle courses (obstacle_task/course.py): validity of every sampled course, determinism, the box
lidar and the multi-waypoint navigation. No simulator needed except for imports of the task modules."""
import os
import sys
import unittest

import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from obstacle_task import course, lidar, scenario  # noqa: E402
from obstacle_task.expert import TwoSegmentNavigation  # noqa: E402
from simulation_and_control import wrap_angle  # noqa: E402


class CourseSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scs = scenario.make_scenarios(7, 60, noise_frac=0.0, course=True)

    def test_every_course_is_valid(self):
        for sc in self.scs:
            self.assertEqual(course.verify_course(sc), [], f"episode {sc.episode}")

    def test_deterministic(self):
        again = scenario.make_scenarios(7, 10, course=True)
        self.assertEqual([s.to_dict() for s in again], [s.to_dict() for s in self.scs[:10]])

    def test_harder_than_single_obstacle(self):
        self.assertTrue(all(len(sc.obstacles) >= 2 and len(sc.route()) >= 2 for sc in self.scs))
        kinds = {f["kind"] for sc in self.scs for f in sc.meta["features"]}
        self.assertEqual(kinds, {"pillar", "wall", "gate"})

    def test_first_obstacle_compatible_with_single_obstacle_code(self):
        for sc in self.scs:
            ob = sc.obstacles[0]
            self.assertIn("r", ob)
            self.assertEqual(sc.waypoint_xy, sc.waypoints[0])
            self.assertIn(sc.side, (-1, 1))

    def test_scenario_roundtrip_and_legacy_dict(self):
        sc = self.scs[0]
        self.assertEqual(scenario.Scenario.from_dict(sc.to_dict()).to_dict(), sc.to_dict())
        legacy = scenario.sample_scenario(3000, 0).to_dict()
        legacy.pop("waypoints"), legacy.pop("course")
        self.assertEqual(len(scenario.Scenario.from_dict(legacy).route()), 1)

    def test_validator_rejects_a_blocked_route(self):
        sc = self.scs[0]
        bad = course.course_violations(sc.init_xy, sc.init_yaw, sc.dock_xy, sc.dock_bearing, sc.target,
                                       sc.obstacles, [], sc.meta["blocking"])     # straight through the obstacles
        self.assertTrue(bad)


class BoxLidar(unittest.TestCase):
    def test_matches_ray_march(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ob = course.make_box(*rng.uniform(-1, 1, 2), rng.uniform(0.3, 0.9), 0.1, rng.uniform(-3, 3))
            xy, yaw = rng.uniform(-3, 3, 2), float(rng.uniform(-3, 3))
            got = lidar.scan(xy, yaw, [ob])
            for k, ang in enumerate(yaw + lidar.beam_angles()):
                ds = np.arange(0.0, lidar.MAX_RANGE, 0.002)
                pts = xy[None, :] + ds[:, None] * np.array([np.cos(ang), np.sin(ang)])[None, :]
                inside = course.obstacle_signed_distance(ob, pts) <= 0.0
                want = ds[np.argmax(inside)] if inside.any() else lidar.MAX_RANGE
                self.assertAlmostEqual(got[k], want, delta=0.01)


class MultiWaypointNavigation(unittest.TestCase):
    def test_visits_every_waypoint_then_docks(self):
        sc = scenario.make_scenarios(7, 1, course=True)[0]
        nav = TwoSegmentNavigation(sc)
        params = nav.params
        pos, yaw, dt, t = np.array(sc.init_xy, dtype=float), float(sc.init_yaw), 0.001, 0.0
        visited = []
        for _ in range(int(params.nav_timeout / dt)):
            wheels = nav.command(np.array([pos[0], pos[1], 0.2]), yaw, t)
            lw, rw = wheels[1], wheels[0]
            v = (lw + rw) / 2 * params.wheel_radius
            w = (rw - lw) * params.wheel_radius / params.wheel_base_width
            yaw = float(wrap_angle(yaw + w * dt))
            pos = pos + v * dt * np.array([np.cos(yaw), np.sin(yaw)])
            t += dt
            if nav.leg > len(visited):
                visited.append(pos.copy())
            if nav.done:
                break
        self.assertTrue(nav.done)
        self.assertEqual(len(visited), len(sc.route()))
        for p, w in zip(visited, sc.route()):
            self.assertLess(np.linalg.norm(p - w), 0.25)
        self.assertLess(nav.nav_done_time, params.nav_timeout)
        self.assertLess(nav.nav_done_time, course.NAV_BUDGET + 5.0)


if __name__ == "__main__":
    unittest.main()
