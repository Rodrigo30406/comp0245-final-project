"""Regression against captured lecturer episodes and both public consumers."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning.data.config import generate_scenarios, resolve_config
from robot_learning.data.storage import read_json, write_json
from robot_learning.data.worker import run_episode
from robot_learning.settings import get_roboenv_root
from robot_learning.simulation.config import episode_defaults, resolve_episode_config, same_expert_configuration
from robot_learning.simulation.episode import Episode, base_goal
from robot_learning.simulation.interactive import run_interactive


class EpisodeTests(unittest.TestCase):
    def test_lecturer_actions_states_and_stop_boundaries(self):
        fixture = read_json(ROOT / "tests/fixtures/lecturer_episode.json")
        for case in fixture["cases"]:
            with self.subTest(target=case["target"]):
                config = resolve_episode_config({"max_steps": case["max_steps"]})
                checkpoints = {sample["step"]: sample for sample in case["samples"]}
                with Episode(case["target"], config, get_roboenv_root()) as episode:
                    for transition in episode.transitions():
                        if transition["step"] in checkpoints:
                            expected = checkpoints.pop(transition["step"])
                            np.testing.assert_allclose(transition["current"]["observation"], expected["observation"], rtol=1e-11, atol=1e-12)
                            np.testing.assert_allclose(transition["action"], expected["action"], rtol=1e-11, atol=1e-12)
                    self.assertFalse(checkpoints)
                    result = episode.result()
                    self.assertEqual(result["steps"], case["steps"])
                    self.assertEqual(result["termination_message"], case["stop"]["message"])
                    self.assertEqual(result["stop_decision_time_s"], case["stop"]["decision_time"])
                    self.assertAlmostEqual(result["stop_decision_distance_m"], case["stop"]["distance"], places=12)
                    self.assertAlmostEqual(result["duration_s"] - result["stop_decision_time_s"], episode.dt)
                    if case["target"][0] == 20:
                        self.assertTrue(result["navigation_timeout"])

    def test_interactive_and_collector_share_rules_and_final_result(self):
        overrides = {"headless": True, "max_steps": 25, "base_controller": {"max_ang_vel": .7}}
        config = resolve_config({**overrides, "episodes": 1, "targets": [[0, 1.1, .5]]})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "episodes").mkdir()
            spec = directory / "rules.json"
            write_json(spec, overrides)
            with contextlib.redirect_stdout(io.StringIO()):
                displayed = run_interactive(["--config", str(spec), "--target", "0", "1.1", ".5"])
            recorded = run_episode(directory, generate_scenarios(config)[0], config, get_roboenv_root())
            self.assertEqual(recorded["status"], "complete")
            for key, value in displayed.items():
                self.assertEqual(recorded[key], value, key)
            with np.load(directory / "episodes/episode_000000.npz") as data:
                self.assertAlmostEqual(data["time"][-1], recorded["stop_decision_time_s"])
                self.assertAlmostEqual(float(data["ee_distance"][-1]), recorded["stop_decision_distance_m"], places=6)
                np.testing.assert_array_equal(data["terminal_observation"], data["next_observation"][-1])

    def test_defaults_are_shared_isolated_and_overrides_are_explicit(self):
        baseline = episode_defaults()
        self.assertEqual(resolve_config({})["standoff"], baseline["standoff"])
        self.assertEqual(baseline["base_controller"]["nav_timeout"], 25)
        self.assertEqual(baseline["arm_controller"]["reach_limit"], .85)
        self.assertEqual(baseline["stop"]["reach_tolerance"], .01)
        changed = resolve_episode_config({"stop": {"reach_tolerance": .02}})
        self.assertEqual(changed["stop"]["reach_tolerance"], .02)
        self.assertEqual(episode_defaults(), baseline)
        for invalid in ({"target": [0, 0, 1]}, {"stop": {"best_ee_error": 0}},
                        {"standoff": {"heights": [1, .5, 1.1, 1.25]}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_episode_config(invalid)


if __name__ == "__main__":
    unittest.main()


class GoalReferenceTests(unittest.TestCase):
    """goal_reference: "origin" is the lecturer's rule (bearing from the world origin); "start" takes the bearing from
    the base start position and is identical for the nominal start at the origin."""

    def test_default_is_origin_and_values_are_validated(self):
        self.assertEqual(episode_defaults()["goal_reference"], "origin")
        self.assertEqual(resolve_episode_config({"goal_reference": "start"})["goal_reference"], "start")
        with self.assertRaises(ValueError):
            resolve_episode_config({"goal_reference": "nearest"})

    def test_start_reference_equals_origin_rule_at_the_nominal_start_and_moves_the_goal_otherwise(self):
        table = episode_defaults()["standoff"]
        target = np.array([1.5, -0.8, 0.5])
        b0, g0 = base_goal(target, [0.0, 0.0], table)
        self.assertAlmostEqual(b0, float(np.arctan2(-0.8, 1.5)))
        standoff = float(np.interp(0.5, table["heights"], table["distances"]))
        np.testing.assert_allclose(g0, target[:2] - standoff * np.array([np.cos(b0), np.sin(b0)]))
        np.testing.assert_allclose(np.linalg.norm(g0 - target[:2]), standoff)
        # a start on the far side of the target: the origin rule parks the base facing away from the start, the
        # start rule parks it on the near side of the ring, at the same standoff distance
        start = np.array([3.0, -1.6])
        b1, g1 = base_goal(target, start, table)
        np.testing.assert_allclose(np.linalg.norm(g1 - target[:2]), standoff)
        self.assertLess(np.linalg.norm(g1 - start), np.linalg.norm(g0 - start))
        self.assertAlmostEqual(b1, float(np.arctan2(target[1] - start[1], target[0] - start[0])))
        # degenerate: target above the reference point -> bearing 0, as the lecturer's code
        self.assertEqual(base_goal([0.0, 0.0, 0.5], [0.0, 0.0], table)[0], 0.0)


class ExpertConfigurationComparisonTests(unittest.TestCase):
    def test_goal_reference_is_the_only_tolerated_difference(self):
        a = resolve_episode_config({}); b = resolve_episode_config({"goal_reference": "start"})
        self.assertTrue(same_expert_configuration(a, b)); self.assertTrue(same_expert_configuration(a, a))
        c = resolve_episode_config({"base_controller": {"pos_tol": 0.09}})
        self.assertFalse(same_expert_configuration(a, c))
