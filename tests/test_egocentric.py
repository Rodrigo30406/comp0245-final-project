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
    encode, FEATURE_NAMES, VARIANTS, EgocentricPolicy, VARIANT, validate_spec, spec_from_episode_config,
    PREV_WHEEL_NAMES, ACTION_INPUT_VARIANTS, STATEFUL_VARIANTS, resolve_options, decode_outputs, DEFAULT_OPTIONS)
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

    def test_target_variant_is_independent_of_the_expert_goal_construction(self):
        rng = np.random.default_rng(4)
        x = np.stack([observation(rng.normal(size=2), rng.uniform(-np.pi, np.pi), rng.uniform(.5, 2, 3),
                                  rng.normal(size=7), rng.normal(size=7)) for _ in range(6)])
        t = rng.uniform(0, 40, 6)
        spec = {**SPEC, "variant": "raw21_target_clock_v1"}
        feats = encode(x, t, spec)
        self.assertEqual(feats.shape, (6, 25)); self.assertEqual(len(VARIANTS["raw21_target_clock_v1"]), 25)
        np.testing.assert_array_equal(feats[:, :21], x)
        np.testing.assert_array_equal(feats[:, 21:24], encode(x, t, SPEC)[:, 7:10])
        np.testing.assert_array_equal(feats[:, 24], t / SPEC["horizon_s"])
        # A different standoff table (expert knowledge) must not change a single column.
        other = {**spec, "standoff_heights": [0.1, 2.0], "standoff_distances": [0.2, 1.7]}
        np.testing.assert_array_equal(encode(x, t, other), feats)
        # And the three columns are the rigid transform of the target into the base frame.
        yaw = x[:, 3]; d = x[:, 18:20] - x[:, 0:2]
        np.testing.assert_allclose(feats[:, 21], np.cos(yaw) * d[:, 0] + np.sin(yaw) * d[:, 1])
        np.testing.assert_allclose(feats[:, 22], -np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1])
        np.testing.assert_allclose(feats[:, 23], x[:, 20] - x[:, 2])

    def test_targetpolar_variant_is_pure_transform_of_the_target_in_base(self):
        rng = np.random.default_rng(5)
        x = np.stack([observation(rng.normal(size=2), rng.uniform(-np.pi, np.pi), rng.uniform(.5, 2, 3),
                                  rng.normal(size=7), rng.normal(size=7)) for _ in range(6)])
        t = rng.uniform(0, 40, 6)
        spec = {**SPEC, "variant": "raw21_targetpolar_clock_v1"}
        feats = encode(x, t, spec)
        self.assertEqual(feats.shape, (6, 28)); self.assertEqual(len(VARIANTS["raw21_targetpolar_clock_v1"]), 28)
        np.testing.assert_array_equal(feats[:, :24], encode(x, t, {**SPEC, "variant": "raw21_target_clock_v1"})[:, :24])
        np.testing.assert_array_equal(feats[:, 27], t / SPEC["horizon_s"])
        # The polar columns are functions of the target-in-base columns only.
        np.testing.assert_allclose(feats[:, 24], np.hypot(feats[:, 21], feats[:, 22]))
        np.testing.assert_allclose(feats[:, 25], np.cos(np.arctan2(feats[:, 22], feats[:, 21])))
        np.testing.assert_allclose(feats[:, 26], np.sin(np.arctan2(feats[:, 22], feats[:, 21])))
        # A different standoff table (expert knowledge) must not change a single column.
        other = {**spec, "standoff_heights": [0.1, 2.0], "standoff_distances": [0.2, 1.7]}
        np.testing.assert_array_equal(encode(x, t, other), feats)

    def test_velocity_variant_is_a_finite_difference_and_live_matches_batch(self):
        rng = np.random.default_rng(6)
        # A smooth trajectory: the base drives and turns, so consecutive observations differ.
        n = 40; tt = np.arange(n) * SPEC["dt_s"]
        x = np.stack([observation([0.3 * t, 0.1 * np.sin(t)], 0.2 * t, [1.5, 0.5, 0.6], rng.normal(size=7), rng.normal(size=7)) for t in tt])
        spec = {**SPEC, "variant": "raw21_targetpolar_vel_clock_v1"}
        feats = encode(x, tt, spec)
        self.assertEqual(feats.shape, (n, 31)); self.assertEqual(len(VARIANTS["raw21_targetpolar_vel_clock_v1"]), 31)
        np.testing.assert_array_equal(feats[:, :27], encode(x, tt, {**SPEC, "variant": "raw21_targetpolar_clock_v1"})[:, :27])
        np.testing.assert_array_equal(feats[:, 30], tt / SPEC["horizon_s"])
        np.testing.assert_array_equal(feats[0, 27:30], 0.0)  # at rest before the first observation
        dt = SPEC["dt_s"]; yaw = x[1:, 3]; d = (x[1:, :2] - x[:-1, :2]) / dt
        np.testing.assert_allclose(feats[1:, 27], np.cos(yaw) * d[:, 0] + np.sin(yaw) * d[:, 1])
        np.testing.assert_allclose(feats[1:, 28], -np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1])
        np.testing.assert_allclose(feats[1:, 29], (x[1:, 3] - x[:-1, 3]) / dt)
        # Expert knowledge changes nothing.
        other = {**spec, "standoff_heights": [0.1, 2.0], "standoff_distances": [0.2, 1.7]}
        np.testing.assert_array_equal(encode(x, tt, other), feats)
        # `previous` continues an episode: encoding row k alone with row k-1 as previous equals the batch row.
        np.testing.assert_allclose(encode(x[5:6], tt[5:6], spec, previous=x[4])[0], feats[5])
        # Live inference keeps the previous observation and resets it.
        policy = EgocentricPolicy(spec, np.zeros(31), np.ones(31), np.zeros(9), np.ones(9),
                                  [rng.normal(size=(31, 5)), rng.normal(size=(5, 9))], [np.zeros(5), np.zeros(9)])
        policy.reset(dt); live = np.stack([policy.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-9, atol=1e-9)
        policy.reset(dt)
        np.testing.assert_allclose(policy.predict_action(x[0]), live[0])
        # The GRU has no previous-observation state, so it must refuse the stateful variant.
        from robot_learning.learning.gru_policy import MODES
        self.assertNotIn("raw21_targetpolar_vel_clock_v1", MODES)

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


class PreviousWheelVariantTests(unittest.TestCase):
    """30 inputs = 28 pure transforms + the policy's own previous wheel command; a latch through the input."""
    V = "raw21_targetpolar_prevwheel_clock_v1"

    def test_layout_requires_and_places_the_previous_command(self):
        spec = {**SPEC, "variant": self.V}
        self.assertEqual(len(VARIANTS[self.V]), 30)
        self.assertEqual(VARIANTS[self.V][27:29], PREV_WHEEL_NAMES)
        self.assertIn(self.V, ACTION_INPUT_VARIANTS); self.assertIn(self.V, STATEFUL_VARIANTS)
        rng = np.random.default_rng(11)
        x = np.stack([observation(rng.normal(size=2), 0.4, [1.0, -0.5, 0.6]) for _ in range(5)]); t = np.arange(5) * 1e-3
        with self.assertRaisesRegex(ValueError, "previous wheel"):
            encode(x, t, spec)
        pw = rng.normal(size=(5, 2))
        f = encode(x, t, spec, previous_wheels=pw)
        np.testing.assert_array_equal(f[:, 27:29], pw)
        np.testing.assert_array_equal(f[:, :27], encode(x, t, {**SPEC, "variant": "raw21_targetpolar_clock_v1"})[:, :27])
        with self.assertRaises(ValueError):
            encode(x, t, spec, previous_wheels=pw[:3])
        from robot_learning.learning.gru_policy import MODES
        self.assertNotIn(self.V, MODES)

    def test_live_runs_on_its_own_previous_output_and_matches_sequential_batch(self):
        spec = {**SPEC, "variant": self.V}; rng = np.random.default_rng(12)
        policy = EgocentricPolicy(spec, np.zeros(30), np.ones(30), np.zeros(9), np.ones(9),
                                  [rng.normal(size=(30, 6)), rng.normal(size=(6, 9))], [np.zeros(6), np.zeros(9)])
        x = np.stack([observation(rng.normal(size=2), 0.1 * i, [1.0, 1.0, 0.5]) for i in range(6)])
        policy.reset(1e-3); live = np.stack([policy.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-9, atol=1e-9)
        # The first step sees a zero previous command; the second sees the first output.
        first = policy.predict_features(encode(x[:1], np.array([0.0]), spec, previous_wheels=np.zeros((1, 2))))[0]
        np.testing.assert_allclose(expand_actions(first), live[0])
        second = policy.predict_features(encode(x[1:2], np.array([1e-3]), spec, previous_wheels=first[None, :2]))[0]
        np.testing.assert_allclose(expand_actions(second), live[1])
        self.assertFalse(np.allclose(policy.predict_features(encode(x[1:2], np.array([1e-3]), spec, previous_wheels=np.zeros((1, 2))))[0], second))
        policy.reset(1e-3)
        np.testing.assert_allclose(policy.predict_action(x[0]), live[0])

    def test_training_features_use_the_demonstrated_previous_command(self):
        from robot_learning.learning.egocentric import load_episode_features
        from robot_learning.learning.dataset import inventory
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "d"; data.mkdir(); make_dense_dataset(data, episodes=2, rows=20)
            record = inventory(data)[0]
            f, y = load_episode_features(data, record, {**SPEC, "variant": self.V})
            np.testing.assert_array_equal(f[0, 27:29], 0.0)
            np.testing.assert_allclose(f[1:, 27:29], y[:-1, :2])


class CategoricalWheelTests(unittest.TestCase):
    def test_decode_takes_bin_centres_and_regresses_the_arm(self):
        centres = np.linspace(-3, 3, 5); y_mean = np.arange(9.0); y_scale = np.full(9, 2.0)
        out = np.zeros((2, 2 * 5 + 7)); out[0, 4] = 1; out[0, 5 + 0] = 1; out[1, 2] = 1; out[1, 5 + 2] = 1; out[:, 10:] = 1.0
        dec = decode_outputs(out, y_mean, y_scale, centres)
        np.testing.assert_array_equal(dec[:, :2], [[3.0, -3.0], [0.0, 0.0]])
        np.testing.assert_array_equal(dec[:, 2:], np.broadcast_to(1.0 * 2.0 + np.arange(2.0, 9.0), (2, 7)))
        np.testing.assert_array_equal(decode_outputs(out[:, :9], y_mean, y_scale), out[:, :9] * 2.0 + y_mean)

    def test_policy_with_bins_outputs_centres_and_roundtrips(self):
        rng = np.random.default_rng(13); n = len(FEATURE_NAMES); centres = np.linspace(-3, 3, 7)
        policy = EgocentricPolicy(SPEC, np.zeros(n), np.ones(n), np.zeros(9), np.ones(9),
                                  [rng.normal(size=(n, 8)), rng.normal(size=(8, 21))], [np.zeros(8), np.zeros(21)], centres)
        self.assertEqual(policy.output_width, 21)
        x = rng.normal(size=(12, 21)); pred = policy.predict(x)
        self.assertEqual(pred.shape, (12, 9))
        self.assertTrue(all(w in centres for w in pred[:, :2].ravel()))
        with tempfile.TemporaryDirectory() as tmp:
            policy.save(Path(tmp) / "m.npz"); loaded = EgocentricPolicy.load(Path(tmp) / "m.npz")
        np.testing.assert_array_equal(loaded.wheel_bin_centres, centres)
        np.testing.assert_array_equal(loaded.predict(x), pred)
        policy.reset(1e-3)
        np.testing.assert_array_equal(np.stack([policy.predict_action(o) for o in x]), expand_actions(pred))
        with self.assertRaises(ValueError):  # nine outputs do not fit seven bins per wheel
            EgocentricPolicy(SPEC, np.zeros(n), np.ones(n), np.zeros(9), np.ones(9),
                             [rng.normal(size=(n, 8)), rng.normal(size=(8, 9))], [np.zeros(8), np.zeros(9)], centres)
        with self.assertRaises(ValueError):
            EgocentricPolicy(SPEC, np.zeros(n), np.ones(n), np.zeros(9), np.ones(9),
                             [rng.normal(size=(n, 8)), rng.normal(size=(8, 21))], [np.zeros(8), np.zeros(21)], centres[::-1])


class OptionTests(unittest.TestCase):
    def test_defaults_and_rejections(self):
        self.assertEqual(resolve_options({}, VARIANT), DEFAULT_OPTIONS)
        self.assertEqual(resolve_options({"egocentric_options": {"stop_weight": 8}}, VARIANT)["stop_weight"], 8.0)
        self.assertEqual(resolve_options({}, VARIANT, {"wheel_loss": "l1"})["wheel_loss"], "l1")
        for bad in ({"wheel_loss": "huber"}, {"stop_weight": 0.5}, {"wheel_bins": 40}, {"wheel_bins": 2}, {"nonsense": 1},
                    {"wheel_output": "categorical", "wheel_loss": "l1"}, {"prev_wheel_shuffle_p": 1.5}, {"prev_wheel_noise_std": -1},
                    {"prev_wheel_shuffle_p": 0.2}):  # last one: not an action-input variant
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_options({}, VARIANT, bad)
        ok = resolve_options({}, "raw21_targetpolar_prevwheel_clock_v1", {"prev_wheel_shuffle_p": 0.25, "prev_wheel_noise_std": 0.1})
        self.assertEqual((ok["prev_wheel_shuffle_p"], ok["prev_wheel_noise_std"]), (0.25, 0.1))


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


def make_dense_dataset(root, episodes=5, rows=80, dt=0.001, stop_tail=0, failures=0):
    """Synthetic dense episodes; `stop_tail` rows at the end of each episode have exactly zero wheel commands."""
    (root / 'episodes').mkdir()
    rng = np.random.default_rng(21)
    scenarios = []
    for i in range(episodes):
        eid = f'episode_{i:06d}'
        target = [float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(.3, .8))]
        x = rng.normal(scale=.3, size=(rows, 21)).astype(np.float32); x[:, -3:] = target
        y = (encode(x, np.arange(rows) * dt, SPEC)[:, :9] * .5).astype(np.float32)
        if stop_tail:
            y[rows - stop_tail:, :2] = 0.0
        scenario = {'episode_id': eid, 'scenario_id': str(i), 'target': target}
        scenarios.append(scenario)
        npz = root / 'episodes' / f'{eid}.npz'
        np.savez_compressed(npz, observation=x, action=expand_actions(y), expert_phase=np.full(rows, 'drive'),
                            step=np.arange(rows), time=np.arange(rows) * dt)
        failed = i >= episodes - failures   # the last `failures` episodes are expert failures (navigation timeout)
        write_json(root / 'episodes' / f'{eid}.json', {
            **scenario, 'status': 'complete', 'success': not failed, 'termination': 'stalled' if failed else 'success', 'dt': dt,
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


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TrainingOptionTests(unittest.TestCase):
    """Stop-weighted / L1 / categorical wheels and the previous-command input train end to end and export loadable policies."""

    def run_training(self, variant, options, tmp):
        from robot_learning.learning.egocentric import train
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        from robot_learning.data.storage import read_json
        root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data, stop_tail=20)
        records = inventory(data)
        write_json(root / 'split.json', {'dataset': str(data), 'training': records[:4], 'validation': records[4:]})
        cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
        cfg['mlp'].update(hidden_layer_sizes=[16, 16], batch_size=16, max_epochs=3, patience=100)
        cfg['torch']['device'] = 'cpu'; write_json(root / 'cfg.json', cfg)
        policy, metrics = train(data, root / 'split.json', root / 'cfg.json', root / 'run', 40.0, 'cpu', variant, options)
        info = read_json(root / 'run/model_info.json'); manifest = read_json(root / 'run/run.json')
        self.assertEqual(manifest['status'], 'complete'); self.assertEqual(manifest['egocentric_options'], info['egocentric_options'])
        self.assertEqual(info['stop_rows_in_training'], 4 * 20)
        loaded = EgocentricPolicy.load(root / 'run/model.npz')
        return policy, loaded, info

    def test_stop_weight_and_l1(self):
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            policy, loaded, info = self.run_training("raw21_targetpolar_clock_v1", {"stop_weight": 8, "wheel_loss": "l1"}, tmp)
            self.assertEqual(info['egocentric_options']['stop_weight'], 8.0); self.assertIn('l1', info['training_loss']); self.assertIn('x8', info['training_loss'])
            x = np.random.default_rng(9).normal(size=(3, 21)); np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))

    def test_categorical_wheels(self):
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            policy, loaded, info = self.run_training("raw21_targetpolar_clock_v1", {"wheel_output": "categorical", "wheel_bins": 11}, tmp)
            centres = np.asarray(info['wheel_bin_centres']); self.assertEqual(len(centres), 11); self.assertEqual(centres[5], 0.0)
            self.assertEqual(policy.output_width, 2 * 11 + 7); np.testing.assert_array_equal(loaded.wheel_bin_centres, centres)
            x = np.random.default_rng(9).normal(size=(30, 21)); pred = loaded.predict(x)
            np.testing.assert_array_equal(pred, policy.predict(x)); self.assertTrue(all(w in centres for w in pred[:, :2].ravel()))
            self.assertIn('cross-entropy', info['training_loss'])

    def test_previous_command_with_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            policy, loaded, info = self.run_training("raw21_targetpolar_prevwheel_clock_v1", {"prev_wheel_shuffle_p": 0.25, "prev_wheel_noise_std": 0.1}, tmp)
            self.assertEqual(info['input_features'], 30); self.assertEqual(info['prev_wheel_augmentation'], {"shuffle_p": 0.25, "noise_std_normalized": 0.1})
            self.assertIn('previous wheel command', info['feature_provenance'])
            x = np.random.default_rng(9).normal(size=(5, 21)); loaded.reset(1e-3)
            np.testing.assert_allclose(np.stack([loaded.predict_action(o) for o in x]), expand_actions(policy.predict(x)), rtol=1e-6, atol=1e-6)


class TriggerRowTests(unittest.TestCase):
    """E9 trigger copies: the trigger row is the first row of the terminal zero-wheel run; copies keep labels and jitter only arm qd."""

    def test_trigger_row_is_the_first_row_of_the_terminal_zero_wheel_run(self):
        from robot_learning.learning.egocentric import trigger_rows
        y = np.ones((12, 9))
        y[3:5, :2] = 0.0          # a zero spell in the middle does not count
        y[8:, :2] = 0.0           # terminal run: rows 8..11 -> trigger 8
        self.assertEqual(trigger_rows(y, [(0, 12)]), [8])
        y2 = np.ones((4, 9)); y2[:, :2] = 0.0     # zero from the first row: trigger 0
        y3 = np.ones((4, 9))                       # never stops: None
        y4 = np.ones((4, 9)); y4[3, 0] = 0.0       # one wheel zero only: None
        yy = np.vstack([y, y2, y3, y4])
        self.assertEqual(trigger_rows(yy, [(0, 12), (12, 16), (16, 20), (20, 24)]), [8, 12, None, None])

    def test_copies_keep_labels_and_jitter_only_the_arm_velocities(self):
        from robot_learning.learning.egocentric import trigger_copies, trigger_rows, VARIANTS
        names = VARIANTS["raw21_targetpolar_clock_v1"]
        rng = np.random.default_rng(3); x = rng.normal(size=(30, len(names))); y = rng.normal(size=(30, 9)); y[20:, :2] = 0.0
        rows = trigger_rows(y, [(0, 30)]); self.assertEqual(rows, [20])
        xc, yc, sigma = trigger_copies(x, y, names, rows, 5, "auto")
        self.assertEqual(xc.shape, (5, len(names))); np.testing.assert_array_equal(yc, np.repeat(y[[20]], 5, axis=0))
        qd = [names.index(f"arm_qd_{i}") for i in range(1, 8)]; other = [i for i in range(len(names)) if i not in qd]
        np.testing.assert_array_equal(xc[:, other], np.repeat(x[[20]], 5, axis=0)[:, other])
        self.assertTrue((xc[:, qd] != x[20, qd]).all())
        np.testing.assert_allclose(sigma, x[:20][:, qd].std(axis=0))           # moving rows only
        xc2, _, _ = trigger_copies(x, y, names, rows, 5, "auto"); np.testing.assert_array_equal(xc, xc2)   # fixed rng
        xc3, _, s3 = trigger_copies(x, y, names, rows, 5, 0.0); np.testing.assert_array_equal(xc3, np.repeat(x[[20]], 5, axis=0))
        np.testing.assert_array_equal(s3, 0.0)
        xe, ye, _ = trigger_copies(x, y, names, [None], 5, "auto"); self.assertEqual((len(xe), len(ye)), (0, 0))

    def test_option_validation(self):
        from robot_learning.learning.egocentric import resolve_options
        self.assertEqual(resolve_options({}, "raw21_targetpolar_clock_v1", {"trigger_copies": 16})["trigger_copies"], 16)
        self.assertEqual(resolve_options({}, "raw21_targetpolar_clock_v1", {"trigger_qd_jitter": 0.2})["trigger_qd_jitter"], 0.2)
        for bad in ({"trigger_copies": -1}, {"trigger_copies": 2.0}, {"trigger_copies": True}, {"trigger_qd_jitter": -0.1},
                    {"trigger_qd_jitter": "std"}):
            with self.assertRaises(ValueError):
                resolve_options({}, "raw21_targetpolar_clock_v1", bad)


class ExpertFailureFilterTests(unittest.TestCase):
    def test_drop_expert_failures_keeps_only_successes_and_counts(self):
        from robot_learning.learning.dataset import drop_expert_failures
        part = {'training': [{'episode_id': 'a', 'success': True}, {'episode_id': 'b', 'success': False}, {'episode_id': 'c', 'success': True}],
                'validation': [{'episode_id': 'd', 'success': True}, {'episode_id': 'e', 'success': False}]}
        kept, counts = drop_expert_failures(part)
        self.assertEqual([r['episode_id'] for r in kept['training']], ['a', 'c']); self.assertEqual([r['episode_id'] for r in kept['validation']], ['d'])
        self.assertEqual(counts, {'training': {'kept': 2, 'dropped_expert_failures': 1}, 'validation': {'kept': 1, 'dropped_expert_failures': 1}})
        with self.assertRaises(ValueError):
            drop_expert_failures({'training': part['training'], 'validation': [{'episode_id': 'e', 'success': False}]})


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class ExpertFailureTrainingTests(unittest.TestCase):
    def test_config_keep_expert_failures_false_drops_failed_episodes_from_training_and_validation(self):
        from robot_learning.learning.egocentric import train
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        from robot_learning.data.storage import read_json
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data, episodes=6, failures=2)
            records = inventory(data)   # episodes 4 and 5 are expert failures
            write_json(root / 'split.json', {'dataset': str(data), 'training': records[:3] + records[4:5], 'validation': records[3:4] + records[5:6]})
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch_successes.json')
            self.assertFalse(cfg['keep_expert_failures'])
            cfg['mlp'].update(hidden_layer_sizes=[16, 16], batch_size=16, max_epochs=2, patience=100); cfg['torch']['device'] = 'cpu'
            write_json(root / 'cfg.json', cfg)
            train(data, root / 'split.json', root / 'cfg.json', root / 'run', 40.0, 'cpu', "raw21_targetpolar_clock_v1")
            info = read_json(root / 'run/model_info.json'); manifest = read_json(root / 'run/run.json')
            self.assertEqual(info['episode_filter'], {'keep_expert_failures': False, 'training': {'kept': 3, 'dropped_expert_failures': 1}, 'validation': {'kept': 1, 'dropped_expert_failures': 1}})
            self.assertEqual((info['training_episodes'], info['validation_episodes'], info['training_rows']), (3, 1, 3 * 80))
            self.assertEqual(manifest['episode_filter'], info['episode_filter'])
            # default config keeps them, as every run before 2 Nov 2026
            cfg['keep_expert_failures'] = True; write_json(root / 'cfg2.json', cfg)
            train(data, root / 'split.json', root / 'cfg2.json', root / 'run2', 40.0, 'cpu', "raw21_targetpolar_clock_v1")
            info2 = read_json(root / 'run2/model_info.json')
            self.assertEqual((info2['episode_filter'], info2['training_rows']), ({'keep_expert_failures': True}, 4 * 80))


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TriggerCopyTrainingTests(unittest.TestCase):
    def test_training_appends_the_copies_after_fitting_the_scaler(self):
        from robot_learning.learning.egocentric import train, load_group, spec_from_episode_config
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        from robot_learning.data.storage import read_json
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data, stop_tail=20)
            records = inventory(data)
            write_json(root / 'split.json', {'dataset': str(data), 'training': records[:4], 'validation': records[4:]})
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(hidden_layer_sizes=[16, 16], batch_size=16, max_epochs=2, patience=100); cfg['torch']['device'] = 'cpu'
            cfg['egocentric_options'] = {"trigger_copies": 16}; write_json(root / 'cfg.json', cfg)
            policy, metrics = train(data, root / 'split.json', root / 'cfg.json', root / 'run', 40.0, 'cpu', "raw21_targetpolar_clock_v1")
            info = read_json(root / 'run/model_info.json')
            t = info['trigger_copies']
            self.assertEqual((t['copies_per_episode'], t['episodes_with_trigger_row'], t['episodes_without'], t['rows_added']), (16, 4, 0, 64))
            self.assertEqual(info['training_rows'], 4 * 80 + 64); self.assertEqual(info['stop_rows_in_training'], 4 * 20 + 64)
            self.assertEqual(len(t['qd_jitter_rad_s']), 7); self.assertTrue(all(v > 0 for v in t['qd_jitter_rad_s']))
            x, _, _ = load_group(data, records[:4], policy.spec)
            np.testing.assert_allclose(policy.x_mean, x.mean(axis=0), rtol=1e-6, atol=1e-9)   # scaler from the recorded rows only
            loaded = EgocentricPolicy.load(root / 'run/model.npz')
            xx = np.random.default_rng(9).normal(size=(3, 21)); np.testing.assert_array_equal(loaded.predict(xx), policy.predict(xx))


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class ExtraDatasetTests(unittest.TestCase):
    """A split may add whole datasets on top of its primary one; records are tagged, never merged by ID."""

    def test_training_uses_primary_and_extra_datasets(self):
        from robot_learning.learning.egocentric import train
        from robot_learning.learning.dataset import inventory, extra_inventory, validate_partition
        from robot_learning.learning.train import DEFAULT_CONFIG
        from robot_learning.data.storage import read_json
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); a = root / 'a'; b = root / 'b'; a.mkdir(); b.mkdir()
            make_dense_dataset(a); make_dense_dataset(b, episodes=4, rows=60)
            saved = {'dataset': str(a), 'extra_datasets': [str(b)]}
            ra = inventory(a); rb = extra_inventory(saved)
            self.assertEqual([r['dataset'] for r in rb], [str(b.resolve())] * 4)
            self.assertEqual(ra[0]['episode_id'], rb[0]['episode_id'])  # same IDs in both: must not collide
            saved.update(training=ra[:4] + rb[:3], validation=ra[4:] + rb[3:])
            validate_partition(ra + rb, {k: saved[k] for k in ['training', 'validation']})
            with self.assertRaises(ValueError):  # an extra record stripped of its tag is foreign
                validate_partition(ra + rb, {'training': ra[:4] + [{k: v for k, v in rb[0].items() if k != 'dataset'}] + rb[1:3], 'validation': ra[4:] + rb[3:]})
            write_json(root / 'split.json', saved)
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(hidden_layer_sizes=[16, 16], batch_size=16, max_epochs=2, patience=100)
            cfg['torch']['device'] = 'cpu'; write_json(root / 'cfg.json', cfg)
            policy, metrics = train(a, root / 'split.json', root / 'cfg.json', root / 'run', 40.0, 'cpu')
            manifest = read_json(root / 'run/run.json')
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(list(manifest['extra_datasets']), [str(b.resolve())])
            saved_split = read_json(root / 'run/split.json')
            self.assertEqual([len(saved_split['training']), len(saved_split['validation'])], [7, 2])
            self.assertEqual(sum('dataset' in r for r in saved_split['training'] + saved_split['validation']), 4)


if __name__ == '__main__':
    unittest.main()
