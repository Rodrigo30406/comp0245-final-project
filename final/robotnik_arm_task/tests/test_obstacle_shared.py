"""Tests of the step-9 shared modules: obstacle_task.scenario and obstacle_task.lidar."""
import os
import sys
import tempfile
import unittest

import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from evaluate_policy import expert_base_goal  # noqa: E402
from obstacle_task import lidar, scenario  # noqa: E402


class TestLidar(unittest.TestCase):
    def test_straight_ahead_and_behind(self):
        ob = [{"x": 2.0, "y": 0.0, "r": 0.3}]
        r = lidar.scan([0.0, 0.0], 0.0, ob)
        self.assertAlmostEqual(r[0], 1.7, places=12)
        self.assertEqual(r[lidar.N_BEAMS // 2], lidar.MAX_RANGE)       # beam pointing backwards

    def test_rotation_and_translation_invariance(self):
        ob = [{"x": 1.0, "y": 3.0, "r": 0.25}]
        a = lidar.scan([1.0, 1.0], np.pi / 2, ob)                       # obstacle 2 m straight ahead
        b = lidar.scan([0.0, 0.0], 0.0, [{"x": 2.0, "y": 0.0, "r": 0.25}])
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_left_beam_and_tangent_miss(self):
        ob = [{"x": 0.0, "y": 1.5, "r": 0.3}]
        r = lidar.scan([0.0, 0.0], 0.0, ob)
        self.assertAlmostEqual(r[lidar.N_BEAMS // 4], 1.2, places=12)   # beam at +90 deg
        far = lidar.scan([0.0, 0.0], 0.0, [{"x": 3.0, "y": 0.31, "r": 0.3}])
        self.assertEqual(far[0], lidar.MAX_RANGE)                        # passes 1 cm beside the circle

    def test_inside_and_out_of_range(self):
        self.assertTrue(np.all(lidar.scan([0, 0], 0.3, [{"x": 0.05, "y": 0.0, "r": 0.3}]) == 0.0))
        self.assertTrue(np.all(lidar.scan([0, 0], 0.0, [{"x": 9.0, "y": 0.0, "r": 0.3}]) == lidar.MAX_RANGE))

    def test_batch_equals_single(self):
        rng = np.random.default_rng(0)
        obs = [{"x": 1.0, "y": -0.5, "r": 0.3}, {"x": -2.0, "y": 1.0, "r": 0.35}]
        xy = rng.uniform(-3, 3, size=(50, 2))
        yaw = rng.uniform(-np.pi, np.pi, size=50)
        batch = lidar.scan_batch(xy, yaw, obs)
        for i in range(50):
            np.testing.assert_array_equal(batch[i], lidar.scan(xy[i], yaw[i], obs))


class TestScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scs = [scenario.sample_scenario(5, ep) for ep in range(400)]

    def test_deterministic(self):
        a = scenario.sample_scenario(5, 17).to_dict()
        b = scenario.sample_scenario(5, 17).to_dict()
        self.assertEqual(a, b)
        self.assertNotEqual(a, scenario.sample_scenario(6, 17).to_dict())

    def test_constraints_and_geometry(self):
        for sc in self.scs:
            ob = sc.obstacles[0]
            ok, why = scenario.constraints_ok(sc.init_xy, sc.dock_xy, sc.target, [ob["x"], ob["y"]], ob["r"],
                                              sc.waypoint_xy)
            self.assertTrue(ok, why)
            dock, bearing, standoff = expert_base_goal(np.asarray(sc.target),
                                                       np.array([*sc.init_xy, 0.0]), sc.init_yaw)
            np.testing.assert_allclose(dock, sc.dock_xy, atol=1e-12)
            self.assertAlmostEqual(bearing, sc.dock_bearing, places=12)
            d = scenario.lateral_position([ob["x"], ob["y"]], sc.init_xy, sc.dock_xy)
            self.assertAlmostEqual(d, sc.lateral_offset, places=9)
            self.assertLessEqual(abs(d), scenario.LATERAL)
            self.assertEqual(sc.ambiguous, abs(d) < scenario.AMBIG_BAND)
            if not sc.ambiguous:
                self.assertEqual(sc.side, -1 if d > 0 else 1)
            w = scenario.lateral_position(sc.waypoint_xy, sc.init_xy, sc.dock_xy)
            self.assertAlmostEqual(w - d, sc.side * (ob["r"] + scenario.WAYPOINT_CLEAR), places=9)

    def test_side_balance(self):
        amb = [sc for sc in self.scs if sc.ambiguous]
        frac_amb = len(amb) / len(self.scs)
        self.assertGreater(frac_amb, 0.35)
        self.assertLess(frac_amb, 0.65)
        left = np.mean([sc.side == 1 for sc in amb])
        self.assertGreater(left, 0.38)
        self.assertLess(left, 0.62)

    def test_round_trip_and_noise_flag(self):
        scs = scenario.make_scenarios(5, 30, noise_frac=0.5)
        self.assertEqual([s.to_dict() for s in scs[:3]][0]["target"], self.scs[0].target)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scenarios.json")
            scenario.save_scenarios(p, scs, extra={"set": "unit"})
            back, payload = scenario.load_scenarios(p)
        self.assertEqual([s.to_dict() for s in back], [s.to_dict() for s in scs])
        self.assertEqual(payload["sampler"]["AMBIG_BAND"], scenario.AMBIG_BAND)
        self.assertTrue(0 < sum(s.noisy for s in scs) < 30)


if __name__ == "__main__":
    unittest.main()
