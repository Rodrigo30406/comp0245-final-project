"""Tests for the multi-agent obstacle task extension (`obstacle_task.multi_agent`, `collect`, `evaluate`)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from evaluate_policy import expert_base_goal  # noqa: E402
from obstacle_task import collect, evaluate, multi_agent as ma, scenario as sc_mod  # noqa: E402


class TestMultiAgentScenarioAndPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = [ma.sample_multi_agent_scenario(42, ep, n_agents=2) for ep in range(15)]

    def test_deterministic_and_json_roundtrip(self):
        a = ma.sample_multi_agent_scenario(42, 3, n_agents=2).to_dict()
        b = ma.sample_multi_agent_scenario(42, 3, n_agents=2).to_dict()
        self.assertEqual(a, b)
        self.assertNotEqual(a, ma.sample_multi_agent_scenario(43, 3, n_agents=2).to_dict())

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scenarios.json")
            sc_mod.save_scenarios(p, self.scenarios[:3])
            back, _ = sc_mod.load_scenarios(p)
        self.assertEqual([s.to_dict() for s in back], [s.to_dict() for s in self.scenarios[:3]])
        self.assertTrue(all(ma.is_multi_agent_scenario(s) for s in back))

    def test_uses_scenario_py_obstacle_constraints_and_close_targets(self):
        for sc in self.scenarios:
            self.assertTrue(ma.is_multi_agent_scenario(sc))
            self.assertEqual(len(sc.obstacles), 1)
            ob = sc.obstacles[0]
            agents = ma.get_scenario_agents(sc)
            self.assertEqual(len(agents), 2)

            # Every robot satisfies scenario.py's exact constraints_ok and waypoint formula
            for ag in agents:
                ok, why = sc_mod.constraints_ok(
                    ag["init_xy"], ag["dock_xy"], ag["target"],
                    [ob["x"], ob["y"]], ob["r"], ag["waypoint_xy"]
                )
                self.assertTrue(ok, f"Agent {ag['agent_id']} failed constraints_ok: {why}")
                dock, bearing, _ = expert_base_goal(
                    np.asarray(ag["target"]), np.array([*ag["init_xy"], 0.0]), ag["init_yaw"]
                )
                np.testing.assert_allclose(dock, ag["dock_xy"], atol=1e-12)
                self.assertAlmostEqual(bearing, ag["dock_bearing"], places=12)
                d = sc_mod.lateral_position([ob["x"], ob["y"]], ag["init_xy"], ag["dock_xy"])
                self.assertAlmostEqual(d, ag["lateral_offset"], places=9)
                self.assertLessEqual(abs(d), sc_mod.LATERAL)
                w = sc_mod.lateral_position(ag["waypoint_xy"], ag["init_xy"], ag["dock_xy"])
                self.assertAlmostEqual(w - d, ag["side"] * (ob["r"] + sc_mod.WAYPOINT_CLEAR), places=9)

            # Targets are close enough to require interaction and far enough to reach
            t0, t1 = np.asarray(agents[0]["target"][:2]), np.asarray(agents[1]["target"][:2])
            d0, d1 = np.asarray(agents[0]["dock_xy"]), np.asarray(agents[1]["dock_xy"])
            i0, i1 = np.asarray(agents[0]["init_xy"]), np.asarray(agents[1]["init_xy"])
            w0, w1 = np.asarray(agents[0]["waypoint_xy"]), np.asarray(agents[1]["waypoint_xy"])
            self.assertGreaterEqual(np.linalg.norm(t0 - t1), ma.TARGET_SEP_RANGE[0])
            self.assertLessEqual(np.linalg.norm(t0 - t1), ma.TARGET_SEP_RANGE[1])
            self.assertGreaterEqual(np.linalg.norm(d0 - d1), ma.DOCK_SEP_MIN)
            self.assertGreaterEqual(np.linalg.norm(i0 - i1), ma.START_SEP_MIN)
            self.assertGreaterEqual(np.linalg.norm(w0 - w1), ma.WAYPOINT_SEP_MIN)


class TestMultiAgentPyBulletCollectAndEvaluate(unittest.TestCase):
    def test_collect_and_evaluate_one_multi_agent_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "ma_data")
            eval_dir = os.path.join(tmp, "ma_eval")

            # Collect 1 multi-agent trajectory via obstacle_task.collect
            args_collect = collect.parse_args([
                "--set-name", "ma_smoke",
                "--multi-agent",
                "--n-agents", "2",
                "--seed", "42",
                "--n-episodes", "1",
                "--workers", "1",
                "--out-dir", data_dir,
            ])
            rows = collect.collect_batch(args_collect)
            self.assertEqual(len(rows), 1)
            row0 = rows[0]
            self.assertEqual(row0["status"], "reached", row0.get("stop_message"))
            self.assertFalse(row0["collided"])
            self.assertLess(row0["final_error"], 0.01)

            npz_path = os.path.join(data_dir, "ep_0000.npz")
            self.assertTrue(os.path.exists(npz_path))
            with np.load(npz_path) as data:
                meta = json.loads(str(data["meta"]))
                self.assertTrue(meta["multi_agent"])
                self.assertEqual(meta["n_agents"], 2)
                self.assertEqual(data["agents_base_pos"].shape[1:], (2, 3))
                self.assertEqual(data["agents_wheel_cmds"].shape[1:], (2, 4))
                self.assertEqual(data["agents_torque_arm"].shape[1:], (2, 7))
                self.assertEqual(data["agents_lidar"].shape[1:], (2, 64))
                self.assertTrue(all(err < 0.01 for err in meta["agent_final_errors"]))

            # Replay and verify via obstacle_task.evaluate
            args_eval = evaluate.parse_args([
                "--policy", "expert",
                "--eval-set", data_dir,
                "--workers", "1",
                "--compare-recorded",
                "--out-dir", eval_dir,
            ])
            eval_rows = evaluate.run_batch(args_eval)
            self.assertEqual(len(eval_rows), 1)
            erow = eval_rows[0]
            self.assertTrue(erow["success"], erow)
            self.assertFalse(erow["collision"], erow)
            self.assertEqual(erow["stop_reason"], "reached")
            self.assertGreater(erow["min_clearance"], 0.02)


if __name__ == "__main__":
    unittest.main()
