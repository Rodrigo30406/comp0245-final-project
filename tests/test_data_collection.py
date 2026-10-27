"""Meaningful data/config invariants; run with RoboEnv Python and unittest."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from robot_learning.data.config import generate_scenarios, resolve_config
from robot_learning.data.storage import read_json, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_invalid_config_rejected_before_collection(self):
        for change in ({"episodes": 0}, {"seed": -1}, {"max_steps": True},
                       {"target_box": {"z": [1, 0]}}, {"max_step": 20},
                       {"targets": [[0, 1, float("nan")]]},
                       {"initial_base": {"mode": "uniform_xy", "x": [-1, 1]}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve_config(change)

    def test_reproducible_targets_and_prefix_when_dataset_grows(self):
        small = generate_scenarios(resolve_config({"episodes": 8, "seed": 12}))
        large = generate_scenarios(resolve_config({"episodes": 16, "seed": 12}))
        self.assertEqual(small, large[:8])
        self.assertNotEqual(small, generate_scenarios(resolve_config({"episodes": 8, "seed": 13})))
        for scenario in small:
            x, y, z = scenario["target"]
            self.assertTrue(-2 <= x <= 2 and -2 <= y <= 2 and .3 <= z <= .8)
            self.assertIsNone(scenario["initial_base"])

    def test_random_start_does_not_change_sampled_targets(self):
        base = resolve_config({"episodes": 8})
        randomized = resolve_config({"episodes": 8, "initial_base": {
            "mode": "uniform_xy", "x": [-2, 2], "y": [-2, 2], "yaw": [0, 0]}})
        a, b = generate_scenarios(base), generate_scenarios(randomized)
        self.assertEqual([s["target"] for s in a], [s["target"] for s in b])
        self.assertTrue(all(-2 <= s["initial_base"]["x"] <= 2 for s in b))


class CollectionIntegrationTests(unittest.TestCase):
    def invoke(self, *args, expected=0):
        process = subprocess.run([sys.executable, str(ROOT / "scripts/collect_dataset.py"), *map(str, args)],
                                 cwd=ROOT.parent, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return process

    def test_expert_matches_pre_refactor_behaviour(self):
        reference = read_json(ROOT / "tests/fixtures/expert_reference.json")
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            spec = temporary / "reference.json"
            write_json(spec, reference["config"])
            self.invoke("--config", spec, "--output-dir", temporary / "datasets")
            directory = next((temporary / "datasets").iterdir())
            metadata = read_json(directory / "episodes/episode_000000.json")
            self.assertTrue(metadata["success"])
            self.assertAlmostEqual(metadata["duration_s"], reference["duration_s"])
            with np.load(directory / "episodes/episode_000000.npz", allow_pickle=False) as data:
                self.assertEqual(len(data["time"]), reference["transitions"])
                indices = reference["samples"]["step"]
                for key in ("observation", "action"):
                    np.testing.assert_allclose(data[key][indices], reference["samples"][key], rtol=1e-6, atol=1e-7)
                for key in ("expert_phase", "base_navigation_done"):
                    np.testing.assert_array_equal(data[key][indices], reference["samples"][key])

    def test_alignment_terminal_state_isolation_and_resume_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            spec = temporary / "spec.json"
            write_json(spec, {"name": "integration", "episodes": 2, "max_steps": 25,
                              "targets": [[0, 1.1, .5], [0, 1.1, .5]]})
            self.invoke("--config", spec, "--output-dir", temporary / "datasets", "--plan-only")
            directory = next((temporary / "datasets").iterdir())
            self.assertEqual(read_json(directory / "dataset.json")["status"], "planned")
            self.invoke("--resume", directory)
            summary = read_json(directory / "summary.json")
            self.assertEqual(summary["complete_episodes"], 2)
            self.assertEqual(summary["recorded_transitions"], 52)
            self.assertEqual(summary["termination_counts"], {"max_steps": 2})
            data_files = sorted((directory / "episodes").glob("*.npz"))
            for data_file in data_files:
                with np.load(data_file, allow_pickle=False) as data:
                    self.assertEqual(data["observation"].shape, (26, 21))
                    self.assertEqual(data["action"].shape, (26, 11))
                    np.testing.assert_array_equal(data["observation"][1:], data["next_observation"][:-1])
                    np.testing.assert_array_equal(data["previous_action"][1:], data["action"][:-1])
                    np.testing.assert_array_equal(data["terminal_observation"], data["next_observation"][-1])
                    np.testing.assert_allclose(data["next_time"] - data["time"], .001)
                    np.testing.assert_allclose(data["terminal_time"], .026)
                    np.testing.assert_allclose(data["ee_distance"],
                                               np.linalg.norm(data["target"] - data["ee_position"], axis=1), atol=1e-6)
                meta = read_json(data_file.with_suffix(".json"))
                self.assertEqual(meta["data_sha256"], sha256(data_file))
                self.assertFalse(meta["success"])
            with np.load(data_files[0]) as first, np.load(data_files[1]) as second:
                np.testing.assert_array_equal(first["observation"], second["observation"])
                np.testing.assert_array_equal(first["action"], second["action"])
            hashes = [sha256(p) for p in data_files]
            self.invoke("--resume", directory)
            self.assertEqual(hashes, [sha256(p) for p in data_files])
            metadata = read_json(data_files[0].with_suffix(".json"))
            metadata["status"] = "error"
            write_json(data_files[0].with_suffix(".json"), metadata)
            self.invoke("--resume", directory)
            archived = list((directory / "attempts/episode_000000").glob("*/episode_000000.npz"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(read_json(data_files[0].with_suffix(".json"))["status"], "complete")
            original = data_files[0].read_bytes()
            data_files[0].write_bytes(original + b"changed")
            corrupted = self.invoke("--resume", directory, expected=2)
            self.assertIn("data missing or changed", corrupted.stderr)
            data_files[0].write_bytes(original)
            cfg = read_json(directory / "config.json")
            cfg["max_steps"] = 99
            write_json(directory / "config.json", cfg)
            result = self.invoke("--resume", directory, expected=2)
            self.assertIn("changed", result.stderr)

    def test_random_initial_position_and_transition_subsampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            spec = temporary / "spec.json"
            write_json(spec, {"name": "random_start", "episodes": 1, "max_steps": 12,
                              "record_every_steps": 5, "initial_base": {
                                  "mode": "uniform_xy", "x": [1, 1], "y": [-1, -1], "yaw": [.2, .2]}})
            self.invoke("--config", spec, "--output-dir", temporary / "datasets")
            directory = next((temporary / "datasets").iterdir())
            meta = read_json(directory / "episodes/episode_000000.json")
            np.testing.assert_allclose(meta["actual_initial_observation"][:2], [1, -1])
            self.assertAlmostEqual(meta["actual_initial_observation"][3], .2)
            with np.load(directory / "episodes/episode_000000.npz") as data:
                np.testing.assert_array_equal(data["step"], [0, 5, 10])
                np.testing.assert_allclose(data["next_time"] - data["time"], .001)
                np.testing.assert_allclose(data["terminal_time"], .013)


if __name__ == "__main__":
    unittest.main()
