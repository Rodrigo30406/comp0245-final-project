"""Egocentric-plus-clock variant: feature geometry, live/batch agreement, portable export, training run layout."""
# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.data.schema import dataset_schema
from robot_learning.data.storage import write_json, sha256
from robot_learning.learning.model import expand_actions
from robot_learning.learning.egocentric import (
    encode, FEATURE_NAMES, VARIANTS, EgocentricPolicy, VARIANT, validate_spec, spec_from_episode_config)
from robot_learning.learning.model import INPUT_NAMES

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
SPEC = {"variant": VARIANT, "standoff_heights": [0.1, 0.5, 1.1, 1.25],
        "standoff_distances": [0.8, 0.65, 0.65, 0.35], "horizon_s": 40.0, "dt_s": 0.001}


def observation(base_xy, yaw, target, q=None, qd=None):
    q = np.zeros(7) if q is None else q; qd = np.zeros(7) if qd is None else qd
    return np.r_[base_xy[0], base_xy[1], 0.0, yaw, q, qd, target]


class FeatureGeometryTests(unittest.TestCase):
    def test_goal_reconstruction_matches_expert_rule_and_vanishes_at_goal_pose(self):
        target = np.array([1.5, 1.0, 0.7])
        bearing = np.arctan2(1.0, 1.5); standoff = np.interp(0.7, SPEC["standoff_heights"], SPEC["standoff_distances"])
        goal = target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
        f = encode(observation(goal, bearing, target)[None], np.array([0.0]), SPEC)[0]
        names = dict(zip(FEATURE_NAMES, f))
        self.assertAlmostEqual(names["goal_distance_xy"], 0.0, places=12)
        self.assertAlmostEqual(names["cos_goal_bearing_error"], 1.0, places=12)
        self.assertAlmostEqual(names["sin_goal_bearing_error"], 0.0, places=12)
        # Target straight ahead at the standoff distance, expressed in the base frame.
        self.assertAlmostEqual(names["target_in_base_x"], standoff, places=12)
        self.assertAlmostEqual(names["target_in_base_y"], 0.0, places=12)
        self.assertAlmostEqual(names["target_height_above_base"], 0.7, places=12)
        self.assertEqual(names["episode_clock_fraction"], 0.0)

    def test_base_frame_rotation_and_clock(self):
        target = np.array([0.0, 2.0, 0.5])
        obs = observation([0.0, 0.0], np.pi / 2, target)  # facing +y, target straight ahead
        f = dict(zip(FEATURE_NAMES, encode(obs[None], np.array([20.0]), SPEC)[0]))
        self.assertAlmostEqual(f["target_in_base_x"], 2.0, places=12)
        self.assertAlmostEqual(f["target_in_base_y"], 0.0, places=12)
        self.assertAlmostEqual(f["sin_heading_error_to_goal"], 0.0, places=12)
        self.assertAlmostEqual(f["episode_clock_fraction"], 0.5, places=12)
        # Rotating the robot by pi flips the goal behind it; yaw wrap must not break the features.
        obs2 = observation([0.0, 0.0], np.pi / 2 + np.pi + 1e-9, target)
        g = dict(zip(FEATURE_NAMES, encode(obs2[None], np.array([0.0]), SPEC)[0]))
        self.assertAlmostEqual(g["cos_heading_error_to_goal"], -1.0, places=6)
        self.assertAlmostEqual(g["target_in_base_x"], -2.0, places=6)

    def test_feature_count_and_validation(self):
        x = np.random.default_rng(0).normal(size=(5, 21))
        self.assertEqual(encode(x, np.zeros(5), SPEC).shape, (5, len(FEATURE_NAMES)))
        with self.assertRaises(ValueError):
            encode(x[:, :20], np.zeros(5), SPEC)
        with self.assertRaises(ValueError):
            validate_spec({**SPEC, "standoff_heights": [1.0, 0.5, 1.1, 1.25]})
        cfg = {"standoff": {"heights": SPEC["standoff_heights"], "distances": SPEC["standoff_distances"]}}
        validate_spec(spec_from_episode_config(cfg, 0.001, 40.0))


class ExtendedVariantTests(unittest.TestCase):
    """The lecturer's 21 explicit inputs kept and extended, never replaced."""

    def test_extended_variants_are_raw_inputs_plus_derived_columns(self):
        rng = np.random.default_rng(0)
        x = np.stack([observation(rng.normal(size=2), rng.uniform(-np.pi, np.pi), rng.uniform(.5, 2, 3),
                                  rng.normal(size=7), rng.normal(size=7)) for _ in range(6)])
        t = rng.uniform(0, 40, 6)
        ego = encode(x, t, SPEC)
        plus = encode(x, t, {**SPEC, "variant": "raw21_egocentric_clock_v1"})
        clock = encode(x, t, {**SPEC, "variant": "raw21_clock_v1"})
        self.assertEqual(plus.shape, (6, 32)); self.assertEqual(clock.shape, (6, 22))
        self.assertEqual(VARIANTS["raw21_egocentric_clock_v1"][:21], list(INPUT_NAMES))
        np.testing.assert_array_equal(plus[:, :21], x); np.testing.assert_array_equal(clock[:, :21], x)
        np.testing.assert_array_equal(plus[:, 21:31], ego[:, :10])  # the 10 relative geometry columns
        np.testing.assert_array_equal(plus[:, 31], ego[:, 24]); np.testing.assert_array_equal(clock[:, 21], ego[:, 24])
        with self.assertRaises(ValueError):
            validate_spec({**SPEC, "variant": "raw21"})

    def test_extended_variant_policy_roundtrip_keeps_variant(self):
        spec = {**SPEC, "variant": "raw21_egocentric_clock_v1"}
        rng = np.random.default_rng(1)
        policy = EgocentricPolicy(spec, np.zeros(32), np.ones(32), np.zeros(9), np.ones(9),
                                  [rng.normal(size=(32, 5)), rng.normal(size=(5, 9))], [np.zeros(5), np.zeros(9)])
        self.assertEqual(policy.input_features, 32)
        x = np.stack([observation(rng.normal(size=2), 0.3, [1.0, 1.0, 0.6]) for _ in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            policy.save(Path(tmp) / "m.npz"); loaded = EgocentricPolicy.load(Path(tmp) / "m.npz")
        self.assertEqual(loaded.spec["variant"], "raw21_egocentric_clock_v1")
        np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))
        with self.assertRaises(ValueError):  # 25-wide weights do not fit a 32-input variant
            EgocentricPolicy(spec, np.zeros(25), np.ones(25), np.zeros(9), np.ones(9),
                             [rng.normal(size=(25, 5)), rng.normal(size=(5, 9))], [np.zeros(5), np.zeros(9)])


class PolicyTests(unittest.TestCase):
    def policy(self):
        rng = np.random.default_rng(3); n = len(FEATURE_NAMES)
        return EgocentricPolicy(SPEC, np.zeros(n), np.ones(n), np.zeros(9), np.ones(9),
                                [rng.normal(size=(n, 16)).astype(np.float32), rng.normal(size=(16, 9)).astype(np.float32)],
                                [np.zeros(16, np.float32), np.zeros(9, np.float32)])

    def test_live_clock_matches_batch_and_resets(self):
        policy = self.policy(); rng = np.random.default_rng(4)
        x = rng.normal(size=(50, 21))
        policy.reset(0.001)
        live = np.stack([policy.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-6, atol=1e-6)
        first_again = policy.predict_action(x[0])
        self.assertFalse(np.allclose(first_again, live[0]), "clock must advance between calls")
        policy.reset(0.001)
        np.testing.assert_allclose(policy.predict_action(x[0]), live[0])
        with self.assertRaisesRegex(ValueError, "dt"):
            policy.reset(0.002)

    def test_save_load_roundtrip(self):
        policy = self.policy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"; policy.save(path)
            loaded = EgocentricPolicy.load(path)
            self.assertEqual(loaded.spec, policy.spec)
            x = np.random.default_rng(5).normal(size=(7, 21))
            np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))


def make_dense_dataset(root, episodes=5, rows=80, dt=0.001):
    (root / 'episodes').mkdir()
    rng = np.random.default_rng(21)
    scenarios = []
    for i in range(episodes):
        eid = f'episode_{i:06d}'
        target = [float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(.3, .8))]
        x = rng.normal(scale=.3, size=(rows, 21)).astype(np.float32); x[:, -3:] = target
        y = (encode(x, np.arange(rows) * dt, SPEC)[:, :9] * .5).astype(np.float32)
        scenario = {'episode_id': eid, 'scenario_id': str(i), 'target': target}
        scenarios.append(scenario)
        npz = root / 'episodes' / f'{eid}.npz'
        np.savez_compressed(npz, observation=x, action=expand_actions(y), expert_phase=np.full(rows, 'drive'),
                            step=np.arange(rows), time=np.arange(rows) * dt)
        write_json(root / 'episodes' / f'{eid}.json', {
            **scenario, 'status': 'complete', 'success': True, 'termination': 'success', 'dt': dt,
            'actual_initial_observation': x[0].tolist(), 'recorded_transitions': rows,
            'data_file': npz.name, 'data_sha256': sha256(npz)})
    write_json(root / 'scenarios.json', scenarios)
    write_json(root / 'schema.json', dataset_schema())
    write_json(root / 'config.json', {'standoff': {'heights': SPEC['standoff_heights'], 'distances': SPEC['standoff_distances']}})
    for name in ['dataset.json', 'provenance.json']:
        write_json(root / name, {'synthetic_test': True})


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TrainingRunTests(unittest.TestCase):
    def test_training_produces_evaluator_compatible_run(self):
        from robot_learning.learning.egocentric import train, KIND
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        from robot_learning.data.storage import read_json
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data)
            records = inventory(data)
            write_json(root / 'split.json', {'dataset': str(data), 'training': records[:4], 'validation': records[4:]})
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(hidden_layer_sizes=[16, 16], batch_size=16, max_epochs=5, patience=100)
            cfg['torch']['device'] = 'cpu'; write_json(root / 'cfg.json', cfg)
            policy, metrics = train(data, root / 'split.json', root / 'cfg.json', root / 'run', 40.0, 'cpu')
            manifest = read_json(root / 'run/run.json'); info = read_json(root / 'run/model_info.json')
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(sha256(root / 'run/model.npz'), manifest['model_sha256'])
            self.assertEqual(sha256(root / 'run/split.json'), manifest['split_sha256'])
            self.assertEqual(set(manifest['dataset_hashes']), {'dataset.json', 'config.json', 'scenarios.json', 'schema.json', 'provenance.json'})
            self.assertEqual(info['kind'], KIND); self.assertEqual(info['input_features'], len(FEATURE_NAMES))
            self.assertIn('validation', metrics)
            loaded = EgocentricPolicy.load(root / 'run/model.npz')
            x = np.random.default_rng(9).normal(size=(3, 21))
            np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))


if __name__ == '__main__':
    unittest.main()
