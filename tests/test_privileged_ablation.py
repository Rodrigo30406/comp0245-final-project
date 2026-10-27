"""Oracle ablation: aligned privileged columns, hidden-state benefit, no exported policy."""
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
from robot_learning.data.storage import write_json, read_json, sha256
from robot_learning.learning.model import expand_actions
from robot_learning.learning.dataset import inventory

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def make_hidden_state_dataset(root, episodes=6, rows=60):
    """Labels depend on an unobservable done flag and reference lead; inputs do not."""
    (root / 'episodes').mkdir()
    rng = np.random.default_rng(11)
    coef = rng.normal(size=(21, 9)) * .1
    scenarios = []
    for i in range(episodes):
        eid = f'episode_{i:06d}'
        x = rng.normal(size=(rows, 21)).astype(np.float32)
        target = [float(i), .3, .5]
        x[:, -3:] = target
        half = rows // 2
        phase = np.array(['drive'] * half + ['orient'] * (rows - half), dtype='U8')
        done = np.r_[np.zeros(half, bool), np.ones(rows - half, bool)]
        lead = rng.normal(scale=.1, size=(rows, 7)).astype(np.float32)
        q_des = x[:, 4:11] + lead
        y = x.astype(float) @ coef
        y[:, :2] += np.where(done, 0., 3.)[:, None]          # wheels stop once navigation is done
        y[:, 2:] += 100. * lead                              # PD term on the hidden reference
        scenario = {'episode_id': eid, 'scenario_id': str(i), 'target': target}
        scenarios.append(scenario)
        npz = root / 'episodes' / f'{eid}.npz'
        np.savez_compressed(npz, observation=x, action=expand_actions(y).astype(np.float32),
                            expert_phase=phase, base_navigation_done=done, expert_q_des_arm=q_des)
        write_json(root / 'episodes' / f'{eid}.json', {
            **scenario, 'status': 'complete', 'success': True, 'termination': 'success',
            'actual_initial_observation': x[0].tolist(), 'recorded_transitions': rows,
            'data_file': npz.name, 'data_sha256': sha256(npz)})
    write_json(root / 'scenarios.json', scenarios)
    write_json(root / 'schema.json', dataset_schema())
    for name in ['dataset.json', 'config.json', 'provenance.json']:
        write_json(root / name, {'synthetic_test': True})


class PrivilegedColumnTests(unittest.TestCase):
    def test_columns_align_and_unknown_phase_is_rejected(self):
        from robot_learning.learning.privileged_ablation import privileged_columns, PHASES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_hidden_state_dataset(root)
            records = inventory(root)
            phase_block, lead = privileged_columns(root, records[0])
            self.assertEqual(phase_block.shape, (60, len(PHASES) + 1))
            self.assertEqual(lead.shape, (60, 7))
            np.testing.assert_array_equal(phase_block[:, :3].sum(axis=1), 1)
            np.testing.assert_array_equal(phase_block[:30, 3], 0); np.testing.assert_array_equal(phase_block[30:, 3], 1)
            with np.load(root / 'episodes' / 'episode_000000.npz') as data:
                bad = {k: data[k] for k in data.files}
            bad['expert_phase'] = np.full(60, 'reach', dtype='U8')
            np.savez_compressed(root / 'episodes' / 'episode_000000.npz', **bad)
            with self.assertRaisesRegex(ValueError, 'Unexpected expert phases'):
                privileged_columns(root, records[0])


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class PrivilegedAblationRunTests(unittest.TestCase):
    def test_hidden_state_arms_beat_baseline_and_nothing_is_exported(self):
        from robot_learning.learning.privileged_ablation import run, ARMS
        from robot_learning.learning.train import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root = Path(tmp); data = root / 'data'; data.mkdir(); make_hidden_state_dataset(data)
            records = inventory(data)
            split = {'dataset': str(data), 'training': records[:4], 'validation': records[4:]}
            write_json(root / 'split.json', split)
            cfg = read_json(DEFAULT_CONFIG.parent / 'mlp_pytorch.json')
            cfg['mlp'].update(hidden_layer_sizes=[16, 16], batch_size=16, max_epochs=40, patience=100)
            cfg['torch']['device'] = 'cpu'
            write_json(root / 'cfg.json', cfg)
            results = run(data, root / 'split.json', root / 'cfg.json', root / 'out', list(ARMS), 'cpu')
            self.assertEqual(set(results), set(ARMS))
            self.assertEqual(results['baseline']['input_features'], 21)
            self.assertEqual(results['phase']['input_features'], 25)
            self.assertEqual(results['reference']['input_features'], 28)
            self.assertEqual(results['full']['input_features'], 32)
            score = {k: v['best_validation_episode_macro_balanced_nrmse'] for k, v in results.items()}
            self.assertLess(score['full'], score['baseline'])
            self.assertLess(score['reference'], score['baseline'])
            self.assertLess(score['phase'], score['baseline'])
            self.assertFalse(list((root / 'out').glob('**/model.npz')), 'oracle models must not be exported')
            self.assertTrue((root / 'out' / 'summary.md').exists())
            run_meta = json.loads((root / 'out' / 'run.json').read_text())
            self.assertEqual(run_meta['status'], 'complete')
            by_done = results['baseline']['validation_by_navigation_done_pooled']
            self.assertIsNotNone(by_done['navigation_done']); self.assertIsNotNone(by_done['navigating'])


if __name__ == '__main__':
    unittest.main()
