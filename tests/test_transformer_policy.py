"""Transformer policy: NumPy cached inference equals the torch network, live state equals batch path, window, roundtrip, training."""
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
from test_egocentric import SPEC, make_dense_dataset

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
D, LAYERS, HEADS, FFN, WIDTH = 8, 2, 2, 12, 6


def random_params(F, seed=0):
    rng = np.random.default_rng(seed); n = lambda *s: rng.normal(scale=.3, size=s)
    p = {"embed_w": n(D, F), "embed_b": n(D), "lnf_g": 1 + .1 * n(D), "lnf_b": .1 * n(D),
         "head1_w": n(WIDTH, D + F), "head1_b": np.zeros(WIDTH), "head2_w": n(9, WIDTH), "head2_b": np.zeros(9)}
    for i in range(LAYERS):
        p.update({f"l{i}_ln1_g": 1 + .1 * n(D), f"l{i}_ln1_b": .1 * n(D), f"l{i}_ln2_g": 1 + .1 * n(D), f"l{i}_ln2_b": .1 * n(D),
                  f"l{i}_ff1_w": n(FFN, D), f"l{i}_ff1_b": .1 * n(FFN), f"l{i}_ff2_w": n(D, FFN), f"l{i}_ff2_b": .1 * n(D)})
        for k in "qkvo":
            p[f"l{i}_w{k}"] = n(D, D); p[f"l{i}_b{k}"] = .1 * n(D)
    return p


def random_policy(mode='raw21_targetpolar_clock_v1', stride=1, window=1000, seed=0):
    from robot_learning.learning.transformer_policy import TransformerPolicy, input_names
    F = len(input_names(mode)); spec = SPEC if mode in ('raw21', SPEC['variant']) else {**SPEC, 'variant': mode}
    return TransformerPolicy(spec, mode, np.zeros(F), np.ones(F), np.zeros(9), np.ones(9), random_params(F, seed),
                             stride_steps=stride, window=window, heads=HEADS)


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class NetworkEquivalenceTests(unittest.TestCase):
    def test_cached_numpy_inference_matches_torch_for_strides_and_windows(self):
        import torch
        from robot_learning.learning.transformer_policy import build_network, load_params
        for stride, window in [(1, 1000), (5, 1000), (1, 7)]:
            policy = random_policy(stride=stride, window=window); F = policy.input_features
            net = build_network(torch, F, D, LAYERS, HEADS, FFN, WIDTH, window).double(); load_params(torch, net, policy.p)
            x = np.random.default_rng(1).normal(size=(43, F))
            with torch.no_grad():
                ref = net(torch.from_numpy(x)[None], stride)[0].numpy()
            np.testing.assert_allclose(policy.predict_features(x), ref, rtol=1e-9, atol=1e-9)


class PolicyStateTests(unittest.TestCase):
    def test_live_state_matches_batch_and_reset_restarts(self):
        policy = random_policy(stride=5)
        x = np.random.default_rng(2).normal(size=(30, 21))
        policy.reset(0.001)
        live = np.stack([policy.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-9, atol=1e-9)
        self.assertFalse(np.allclose(live[1], live[2]), "the head reacts between tokens")
        again = policy.predict_action(x[0])
        self.assertFalse(np.allclose(again, live[0]), "memory and clock must carry across calls")
        policy.reset(0.001)
        np.testing.assert_allclose(policy.predict_action(x[0]), live[0])

    def test_window_bounds_the_memory(self):
        policy = random_policy(window=4)
        x = np.random.default_rng(5).normal(size=(12, 21))
        full = policy.predict(x); tail = policy.predict(x[-4:])
        # Receptive field of L layers with a W-token window is (W - 1) * L + 1 tokens: 7 here. raw21 mode has no clock,
        # so the last output must depend on the last 7 observations only.
        raw = random_policy(mode='raw21', window=4)
        np.testing.assert_allclose(raw.predict(x)[-1], raw.predict(x[-7:])[-1], rtol=1e-9, atol=1e-9)
        self.assertFalse(np.allclose(raw.predict(x)[-1], raw.predict(x[-6:])[-1]))
        self.assertEqual(full.shape, (12, 9)); self.assertEqual(tail.shape, (4, 9))

    def test_save_load_roundtrip(self):
        from robot_learning.learning.transformer_policy import TransformerPolicy
        for mode in ['raw21_targetpolar_clock_v1', 'raw21', 'raw21_egocentric_clock_v1']:
            policy = random_policy(mode=mode, stride=20, window=50)
            with tempfile.TemporaryDirectory() as tmp:
                policy.save(Path(tmp) / 'm.npz'); loaded = TransformerPolicy.load(Path(tmp) / 'm.npz')
                self.assertEqual((loaded.mode, loaded.stride, loaded.window, loaded.heads), (mode, 20, 50, HEADS))
                x = np.random.default_rng(3).normal(size=(45, 21))
                np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))

    def test_invalid_parameters_are_rejected(self):
        from robot_learning.learning.transformer_policy import TransformerPolicy, input_names
        F = len(input_names('raw21')); p = random_params(F); p['embed_w'] = np.zeros((D, F + 1))
        with self.assertRaises(ValueError):
            TransformerPolicy(SPEC, 'raw21', np.zeros(F), np.ones(F), np.zeros(9), np.ones(9), p, heads=HEADS)
        with self.assertRaises(ValueError):
            TransformerPolicy(SPEC, 'raw21_egocentric_clock_v1', np.zeros(32), np.ones(32), np.zeros(9), np.ones(9), random_params(32), heads=HEADS)


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TrainingRunTests(unittest.TestCase):
    def test_training_produces_evaluator_compatible_run(self):
        from robot_learning.learning.transformer_policy import train, KIND, TransformerPolicy
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data)
            records = inventory(data)
            write_json(root / 'split.json', {'dataset': str(data), 'training': records[:4], 'validation': records[4:]})
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(max_epochs=3, patience=100); cfg['torch']['device'] = 'cpu'
            write_json(root / 'cfg.json', cfg)
            policy, metrics = train(data, root / 'split.json', root / 'cfg.json', root / 'run', 40.0, 'cpu', mode='raw21_targetpolar_clock_v1',
                                    stride=4, window=10, d_model=D, layers=LAYERS, heads=HEADS, ffn=FFN, head_width=WIDTH, episodes_per_batch=2)
            manifest = read_json(root / 'run/run.json'); info = read_json(root / 'run/model_info.json')
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(sha256(root / 'run/model.npz'), manifest['model_sha256'])
            self.assertEqual((info['kind'], info['d_model'], info['memory_stride_steps'], info['window_tokens']), (KIND, D, 4, 10))
            loaded = TransformerPolicy.load(root / 'run/model.npz')
            x = np.random.default_rng(9).normal(size=(3, 21))
            np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))


if __name__ == '__main__':
    unittest.main()
