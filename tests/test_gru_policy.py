"""GRU policy: NumPy cell equals torch.nn.GRU, live state equals batch path, export roundtrip, training layout."""
# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.data.storage import write_json, read_json, sha256
from robot_learning.learning.model import expand_actions
from robot_learning.learning.egocentric import FEATURE_NAMES
from test_egocentric import SPEC, make_dense_dataset

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def random_policy(hidden=8, seed=0, mode='egocentric_clock_v1', stride=1, width=6):
    from robot_learning.learning.gru_policy import GRUPolicy, input_names
    rng = np.random.default_rng(seed); F = len(input_names(mode))
    spec = SPEC if mode in ('raw21', SPEC['variant']) else {**SPEC, 'variant': mode}
    return GRUPolicy(spec, mode, np.zeros(F), np.ones(F), np.zeros(9), np.ones(9),
                     rng.normal(scale=.3, size=(3 * hidden, F)), rng.normal(scale=.3, size=(3 * hidden, hidden)),
                     rng.normal(scale=.1, size=3 * hidden), rng.normal(scale=.1, size=3 * hidden),
                     rng.normal(scale=.3, size=(width, hidden + F)), np.zeros(width),
                     rng.normal(scale=.3, size=(9, width)), np.zeros(9), stride_steps=stride)


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class CellEquivalenceTests(unittest.TestCase):
    def test_numpy_policy_matches_torch_network_for_both_strides(self):
        import torch
        from robot_learning.learning.gru_policy import build_network
        for stride in (1, 5):
            policy = random_policy(hidden=8, stride=stride)
            net = build_network(torch, len(FEATURE_NAMES), 8, 6).double()
            with torch.no_grad():
                net.gru.weight_ih_l0.copy_(torch.from_numpy(policy.w_ih)); net.gru.weight_hh_l0.copy_(torch.from_numpy(policy.w_hh))
                net.gru.bias_ih_l0.copy_(torch.from_numpy(policy.b_ih)); net.gru.bias_hh_l0.copy_(torch.from_numpy(policy.b_hh))
                net.head1.weight.copy_(torch.from_numpy(policy.w1)); net.head1.bias.copy_(torch.from_numpy(policy.b1))
                net.head2.weight.copy_(torch.from_numpy(policy.w2)); net.head2.bias.copy_(torch.from_numpy(policy.b2))
                x = np.random.default_rng(1).normal(size=(43, len(FEATURE_NAMES)))
                ref, _ = net(torch.from_numpy(x)[None], stride)
            np.testing.assert_allclose(policy.predict_features(x), ref[0].numpy(), rtol=1e-9, atol=1e-9)


class PolicyStateTests(unittest.TestCase):
    def test_live_state_matches_batch_and_reset_restarts(self):
        policy = random_policy()
        x = np.random.default_rng(2).normal(size=(30, 21))
        policy.reset(0.001)
        live = np.stack([policy.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-9, atol=1e-9)
        again = policy.predict_action(x[0])
        self.assertFalse(np.allclose(again, live[0]), "hidden state and clock must carry across calls")
        policy.reset(0.001)
        np.testing.assert_allclose(policy.predict_action(x[0]), live[0])

    def test_save_load_roundtrip_both_input_modes(self):
        from robot_learning.learning.gru_policy import GRUPolicy
        for mode, width in [('egocentric_clock_v1', len(FEATURE_NAMES)), ('raw21', 21),
                            ('raw21_egocentric_clock_v1', 32), ('raw21_clock_v1', 22)]:
            policy = random_policy(mode=mode)
            self.assertEqual(policy.input_features, width)
            with tempfile.TemporaryDirectory() as tmp:
                policy.save(Path(tmp) / 'm.npz'); loaded = GRUPolicy.load(Path(tmp) / 'm.npz')
                self.assertEqual(loaded.mode, mode)
                x = np.random.default_rng(3).normal(size=(5, 21))
                np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))

    def test_strided_memory_reacts_every_step_and_matches_batch(self):
        from robot_learning.learning.gru_policy import GRUPolicy
        policy = random_policy(stride=5)
        x = np.random.default_rng(8).normal(size=(23, 21))
        policy.reset(0.001)
        live = np.stack([policy.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-9, atol=1e-9)
        # The head reacts to every observation even between memory updates.
        self.assertFalse(np.allclose(live[1], live[2]))
        # Between memory updates only the current features change (raw21 has no clock):
        # the same observation gives the same action until the next memory update.
        raw = random_policy(stride=5, mode='raw21'); raw.reset(0.001)
        raw.predict_action(x[0]); a1 = raw.predict_action(x[1]); a2 = raw.predict_action(x[1])
        np.testing.assert_allclose(a1, a2)
        raw.predict_action(x[1]); raw.predict_action(x[1]); a3 = raw.predict_action(x[1])  # 6th call: memory updated
        self.assertFalse(np.allclose(a1, a3))
        with tempfile.TemporaryDirectory() as tmp:
            policy.save(Path(tmp) / 'm.npz'); self.assertEqual(GRUPolicy.load(Path(tmp) / 'm.npz').stride, 5)

    def test_spec_variant_must_match_mode(self):
        from robot_learning.learning.gru_policy import GRUPolicy
        with self.assertRaises(ValueError):
            GRUPolicy(SPEC, 'raw21_egocentric_clock_v1', np.zeros(32), np.ones(32), np.zeros(9), np.ones(9),
                      np.zeros((24, 32)), np.zeros((24, 8)), np.zeros(24), np.zeros(24), np.zeros((6, 40)), np.zeros(6),
                      np.zeros((9, 6)), np.zeros(9))

    def test_raw21_mode_feeds_observations_unchanged(self):
        policy = random_policy(mode='raw21')
        x = np.random.default_rng(6).normal(size=(4, 21))
        np.testing.assert_array_equal(policy.predict(x), policy.predict_features(x))


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TrainingRunTests(unittest.TestCase):
    def test_training_produces_evaluator_compatible_run(self):
        from robot_learning.learning.gru_policy import train, KIND, GRUPolicy
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data)
            records = inventory(data)
            write_json(root / 'split.json', {'dataset': str(data), 'training': records[:4], 'validation': records[4:]})
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(max_epochs=3, patience=100); cfg['torch']['device'] = 'cpu'
            write_json(root / 'cfg.json', cfg)
            for mode, stride in [('egocentric_clock_v1', 1), ('raw21', 4)]:
                policy, metrics = train(data, root / 'split.json', root / 'cfg.json', root / f'run_{mode}', 40.0, 'cpu',
                                        hidden=8, chunk_steps=16, episodes_per_batch=2, mode=mode, stride=stride, head_width=6)
                manifest = read_json(root / f'run_{mode}/run.json'); info = read_json(root / f'run_{mode}/model_info.json')
                self.assertEqual(manifest['status'], 'complete')
                self.assertEqual(sha256(root / f'run_{mode}/model.npz'), manifest['model_sha256'])
                self.assertEqual(info['kind'], KIND); self.assertEqual(info['hidden'], 8); self.assertEqual(info['input_mode'], mode)
                self.assertEqual(info['memory_stride_steps'], stride)
                loaded = GRUPolicy.load(root / f'run_{mode}/model.npz')
                x = np.random.default_rng(9).normal(size=(3, 21))
                np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))


if __name__ == '__main__':
    unittest.main()
