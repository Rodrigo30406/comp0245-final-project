"""Declared DART-style perturbation: validation, determinism, clean labels, untouched default loop."""
# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
import tempfile
import unittest

import numpy as np
from robot_learning.data.config import resolve_config, validate_action_noise, generate_scenarios
from robot_learning.data.worker import make_perturbation, run_episode
from robot_learning.data.storage import write_json, read_json
from robot_learning.data.schema import dataset_schema
from robot_learning.settings import get_roboenv_root

NOISE = {"wheel_std": 1.0, "torque_std": 2.0, "hold_steps": 5, "seed": 7}


class NoiseConfigTests(unittest.TestCase):
    def test_default_is_null_and_invalid_specs_are_rejected(self):
        cfg = resolve_config({"name": "x", "episodes": 1, "targets": [[0.5, 0.5, 0.5]]})
        self.assertIsNone(cfg["action_noise"])
        for bad in [{"wheel_std": 1.0}, {**NOISE, "hold_steps": 0}, {**NOISE, "wheel_std": -1},
                    {**NOISE, "wheel_std": 0, "torque_std": 0}, {**NOISE, "extra": 1}]:
            with self.assertRaises(ValueError):
                validate_action_noise(bad)
        self.assertEqual(resolve_config({"name": "x", "episodes": 1, "targets": [[0.5, 0.5, 0.5]],
                                         "action_noise": NOISE})["action_noise"], NOISE)

    def test_perturbation_is_deterministic_held_and_pair_consistent(self):
        a = make_perturbation(NOISE, 123); b = make_perturbation(NOISE, 123); c = make_perturbation(NOISE, 124)
        action = np.zeros(11)
        outs_a = np.stack([a(action) for _ in range(12)]); outs_b = np.stack([b(action) for _ in range(12)])
        outs_c = np.stack([c(action) for _ in range(12)])
        np.testing.assert_array_equal(outs_a, outs_b)
        self.assertFalse(np.allclose(outs_a, outs_c))
        # Held for hold_steps calls, then redrawn.
        np.testing.assert_array_equal(outs_a[0], outs_a[4]); self.assertFalse(np.allclose(outs_a[4], outs_a[5]))
        # Skid-steer pair identity of the applied wheel command.
        np.testing.assert_array_equal(outs_a[:, 0], outs_a[:, 3]); np.testing.assert_array_equal(outs_a[:, 1], outs_a[:, 2])
        self.assertIsNone(make_perturbation(None, 1))


class NoisyCollectionTests(unittest.TestCase):
    def collect(self, root, noise, steps=40):
        cfg = resolve_config({"name": "noise_test", "episodes": 1, "targets": [[1.0, 0.5, 0.6]], "headless": True,
                              "max_steps": steps, "action_noise": noise})
        scenarios = generate_scenarios(cfg)
        (root / "episodes").mkdir(parents=True)
        write_json(root / "config.json", cfg); write_json(root / "scenarios.json", scenarios)
        write_json(root / "schema.json", dataset_schema())
        meta = run_episode(root, scenarios[0], cfg, Path(get_roboenv_root()))
        self.assertEqual(meta["status"], "complete", meta.get("error_message"))
        with np.load(root / "episodes" / meta["data_file"], allow_pickle=False) as data:
            return meta, {k: data[k] for k in ("action", "applied_action", "previous_action", "observation")}

    def test_clean_labels_noisy_execution_and_untouched_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_meta, clean = self.collect(Path(tmp) / "clean", None)
            noisy_meta, noisy = self.collect(Path(tmp) / "noisy", NOISE)
        # Without noise the applied command is the label: the lecturer's loop is unchanged.
        np.testing.assert_array_equal(clean["applied_action"], clean["action"])
        self.assertIsNone(clean_meta["action_noise"]); self.assertEqual(noisy_meta["action_noise"], NOISE)
        # With noise: same first state, same clean expert label, different applied command.
        np.testing.assert_array_equal(noisy["observation"][0], clean["observation"][0])
        np.testing.assert_allclose(noisy["action"][0], clean["action"][0], rtol=0, atol=1e-6)
        self.assertFalse(np.allclose(noisy["applied_action"][0], noisy["action"][0]))
        # Labels stay within the expert's envelope while applied commands may leave it.
        self.assertLessEqual(np.abs(noisy["action"][:, 4:]).max(), 30.0 + 1e-6)
        # previous_action is the previously APPLIED command.
        np.testing.assert_array_equal(noisy["previous_action"][1:], noisy["applied_action"][:-1])
        # Trajectories diverge once perturbed commands act on the physics.
        self.assertFalse(np.allclose(noisy["observation"][-1], clean["observation"][-1]))


if __name__ == "__main__":
    unittest.main()
