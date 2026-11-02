"""Gated-stop (E8a) and MLP->GRU (E8b) recurrent variants: NumPy inference against the torch twin, the gate's
contract (starts open, never reopens, zeroes the wheels exactly, is seen by the arm head), save/load, and a
training run the evaluator can load."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src')); _sys.path.insert(0, str(_Path(__file__).resolve().parent))

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
MODE = 'raw21_targetpolar_clock_v1'


def random_policy(architecture, hidden=8, seed=0, mode=MODE, stride=1, width=6, encoder=5, y_mean=None, y_scale=None):
    from robot_learning.learning.gated_policy import GatedPolicy, input_names
    rng = np.random.default_rng(seed); F = len(input_names(mode))
    spec = SPEC if mode in ('raw21', SPEC['variant']) else {**SPEC, 'variant': mode}
    G = F if architecture == 'gated' else encoder
    p = {'w_ih': rng.normal(scale=.3, size=(3 * hidden, G)), 'w_hh': rng.normal(scale=.3, size=(3 * hidden, hidden)),
         'b_ih': rng.normal(scale=.1, size=3 * hidden), 'b_hh': rng.normal(scale=.1, size=3 * hidden),
         'w1': rng.normal(scale=.3, size=(width, F + 1 if architecture == 'gated' else hidden + F)), 'b1': np.zeros(width),
         'w2': rng.normal(scale=.3, size=(9, width)), 'b2': rng.normal(scale=.2, size=9)}
    if architecture == 'gated':
        p.update(w_s=rng.normal(scale=1.0, size=(1, hidden)), b_s=np.array([-1.0]))
    else:
        p.update(w_e=rng.normal(scale=.3, size=(encoder, F)), b_e=np.zeros(encoder))
    ym = np.zeros(9) if y_mean is None else y_mean; ys = np.ones(9) if y_scale is None else y_scale
    return GatedPolicy(spec, mode, architecture, np.zeros(F), np.ones(F), ym, ys, p, stride_steps=stride)


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TwinEquivalenceTests(unittest.TestCase):
    def test_numpy_policy_matches_torch_network_for_both_architectures_and_strides(self):
        import torch
        from robot_learning.learning.gated_policy import build_network, load_into, input_names
        F = len(input_names(MODE)); y_mean = np.linspace(-1, 1, 9); y_scale = np.linspace(.5, 2, 9)
        for arch in ('gated', 'mlp_gru'):
            for stride in (1, 5):
                policy = random_policy(arch, hidden=8, stride=stride, y_mean=y_mean, y_scale=y_scale)
                net = build_network(torch, arch, F, 8, 6, y_mean, y_scale, encoder_width=5).double(); load_into(net, policy, torch)
                x = np.random.default_rng(1).normal(size=(43, F))
                with torch.no_grad():
                    ref, _ = net(torch.from_numpy(x)[None], stride)
                ref = ref[0].numpy() * y_scale + y_mean
                np.testing.assert_allclose(policy.predict_features(x), ref, rtol=1e-9, atol=1e-9)
                # the torch network carries its state across chunks exactly like one long sequence
                with torch.no_grad():
                    a, state = net(torch.from_numpy(x[:20])[None], stride); b, _ = net(torch.from_numpy(x[20:])[None], stride, state)
                np.testing.assert_allclose(torch.cat([a, b], dim=1)[0].numpy() * y_scale + y_mean, ref, rtol=1e-9, atol=1e-9)


class GateContractTests(unittest.TestCase):
    def test_gate_starts_open_never_reopens_and_zeroes_the_wheels_exactly(self):
        y_mean = np.full(9, 3.0); y_scale = np.full(9, 2.0)
        policy = random_policy('gated', hidden=8, stride=3, y_mean=y_mean, y_scale=y_scale)
        x = np.random.default_rng(4).normal(size=(60, policy.input_features))
        out, gates = policy.predict_features(x, return_gate=True)
        self.assertTrue(np.all(np.diff(gates) <= 1e-15), "the gate never reopens")
        self.assertTrue(np.all(gates <= 1.0) and np.all(gates >= 0.0))
        self.assertTrue(gates[0] < 1.0, "a stop score is read at the first memory update")
        # with the stop score forced to saturate, the wheels are exactly zero whatever the head says
        policy.p['b_s'][:] = 60.0
        out, gates = policy.predict_features(x, return_gate=True)
        np.testing.assert_array_equal(out[:, :2], 0.0); np.testing.assert_array_equal(gates, 0.0)
        self.assertTrue(np.all(np.isfinite(out[:, 2:])) and np.any(out[:, 2:] != 0.0), "the arm torques are not gated")
        # with the stop score forced to zero the gate is 1 and the wheels are the head's own prediction
        policy.p['b_s'][:] = -60.0
        out, gates = policy.predict_features(x, return_gate=True)
        np.testing.assert_array_equal(gates, 1.0)
        f = (x - policy.x_mean) / policy.x_scale
        hidden = np.maximum(f @ policy.p['w1'][:, :-1].T + policy.p['w1'][:, -1] + policy.p['b1'], 0.0)
        np.testing.assert_allclose(out, (hidden @ policy.p['w2'].T + policy.p['b2']) * y_scale + y_mean, rtol=1e-12, atol=1e-12)

    def test_arm_head_sees_the_gate(self):
        policy = random_policy('gated', hidden=8)
        x = np.random.default_rng(5).normal(size=(1, policy.input_features))
        policy.p['b_s'][:] = -60.0; open_arm = policy.predict_features(x)[0, 2:]
        policy.p['b_s'][:] = 60.0; closed_arm = policy.predict_features(x)[0, 2:]
        self.assertFalse(np.allclose(open_arm, closed_arm), "the same features give a different arm torque once the gate is closed")

    def test_mlp_gru_has_no_gate_and_live_matches_batch_for_both(self):
        for arch in ('gated', 'mlp_gru'):
            policy = random_policy(arch, stride=4)
            x = np.random.default_rng(6).normal(size=(27, 21))
            policy.reset(0.001)
            live = np.stack([policy.predict_action(o) for o in x])
            np.testing.assert_allclose(live, expand_actions(policy.predict(x)), rtol=1e-9, atol=1e-9)
            self.assertEqual(live.shape, (27, 11)); np.testing.assert_array_equal(live[:, 1], live[:, 2])  # same-side wheels
            again = policy.predict_action(x[0])
            self.assertFalse(np.allclose(again, live[0]), "state and clock carry across calls")
            policy.reset(0.001); np.testing.assert_allclose(policy.predict_action(x[0]), live[0])
            if arch == 'mlp_gru':
                self.assertEqual(policy.gate(), 1.0)
            with self.assertRaises(ValueError):
                policy.reset(0.002)

    def test_save_load_roundtrip_and_shape_checks(self):
        from robot_learning.learning.gated_policy import GatedPolicy
        for arch in ('gated', 'mlp_gru'):
            policy = random_policy(arch, stride=2)
            with tempfile.TemporaryDirectory() as tmp:
                policy.save(Path(tmp) / 'm.npz'); loaded = GatedPolicy.load(Path(tmp) / 'm.npz')
                self.assertEqual((loaded.arch, loaded.mode, loaded.stride), (arch, MODE, 2))
                x = np.random.default_rng(3).normal(size=(5, 21))
                np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))
        bad = random_policy('gated'); params = dict(bad.p); params['w_s'] = np.zeros((2, 8))
        with self.assertRaises(ValueError):
            GatedPolicy(bad.spec, MODE, 'gated', bad.x_mean, bad.x_scale, bad.y_mean, bad.y_scale, params)
        with self.assertRaises(ValueError):
            GatedPolicy(bad.spec, MODE, 'mlp_gru', bad.x_mean, bad.x_scale, bad.y_mean, bad.y_scale, bad.p)
        with self.assertRaises(ValueError):
            GatedPolicy(bad.spec, MODE, 'lstm', bad.x_mean, bad.x_scale, bad.y_mean, bad.y_scale, bad.p)


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TrainingRunTests(unittest.TestCase):
    def test_training_produces_evaluator_compatible_run_for_both_architectures(self):
        from robot_learning.learning.gated_policy import train, KIND, GatedPolicy
        from robot_learning.learning.dataset import inventory
        from robot_learning.learning.train import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_dense_dataset(data)
            records = inventory(data)
            write_json(root / 'split.json', {'dataset': str(data), 'training': records[:4], 'validation': records[4:]})
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(max_epochs=3, patience=100); cfg['torch']['device'] = 'cpu'
            write_json(root / 'cfg.json', cfg)
            for arch, stride in [('gated', 4), ('mlp_gru', 1)]:
                policy, metrics = train(data, root / 'split.json', root / 'cfg.json', root / f'run_{arch}', 40.0, 'cpu', arch,
                                        hidden=8, chunk_steps=16, episodes_per_batch=2, mode=MODE, stride=stride, head_width=6,
                                        encoder_width=5)
                manifest = read_json(root / f'run_{arch}/run.json'); info = read_json(root / f'run_{arch}/model_info.json')
                self.assertEqual(manifest['status'], 'complete')
                self.assertEqual(sha256(root / f'run_{arch}/model.npz'), manifest['model_sha256'])
                self.assertEqual((info['kind'], info['architecture'], info['memory_stride_steps']), (KIND, arch, stride))
                loaded = GatedPolicy.load(root / f'run_{arch}/model.npz')
                x = np.random.default_rng(9).normal(size=(3, 21))
                np.testing.assert_array_equal(loaded.predict(x), policy.predict(x))
                if arch == 'gated':
                    self.assertIn('gate', metrics['validation']); self.assertIn('final_gate_median', metrics['validation']['gate'])


if __name__ == '__main__':
    unittest.main()
