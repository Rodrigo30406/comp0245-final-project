"""Guard leakage boundaries, control symmetry and physical-unit inference."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from robot_learning.data.schema import dataset_schema
from robot_learning.data.storage import write_json, sha256, read_json
from robot_learning.learning.dataset import inventory, split_episodes
from robot_learning.learning.model import independent_actions, expand_actions, LinearPolicy, MLPPolicy
from robot_learning.learning.metrics import errors, aggregate
from robot_learning.learning.train import train, DEFAULT_CONFIG


def make_dataset(root):
    (root / 'episodes').mkdir()
    rng = np.random.default_rng(8)
    scenarios = []
    coef = rng.normal(size=(21, 9))
    for i in range(5):
        eid = f'episode_{i:06d}'
        x = rng.normal(size=(18 + i, 21)).astype(np.float32)
        target = [float(i), .2, .5]
        x[:, -3:] = target
        # A held-out-only large offset tests that normalization never fits all rows.
        x[:, 0] += i * 100
        y = x.astype(float) @ coef + np.arange(9)
        scenario = {'episode_id': eid, 'scenario_id': str(i), 'target': target}
        scenarios.append(scenario)
        npz = root / 'episodes' / f'{eid}.npz'
        np.savez_compressed(npz, observation=x, action=expand_actions(y),
                            expert_phase=np.full(len(x), 'reach'))
        write_json(root / 'episodes' / f'{eid}.json', {
            **scenario, 'status': 'complete', 'success': i != 0, 'termination': 'success' if i else 'stalled',
            'actual_initial_observation': x[0].tolist(), 'recorded_transitions': len(x),
            'data_file': npz.name, 'data_sha256': sha256(npz)})
    write_json(root / 'scenarios.json', scenarios)
    write_json(root / 'schema.json', dataset_schema())
    for name in ['dataset.json', 'config.json', 'provenance.json']:
        write_json(root / name, {'synthetic_test': True})


class LearningTests(unittest.TestCase):
    def test_control_symmetry_roundtrip_and_rejection(self):
        reduced = np.arange(36).reshape(4, 9).astype(float)
        full = expand_actions(reduced)
        np.testing.assert_array_equal(independent_actions(full), reduced)
        full[0, 3] += .1
        with self.assertRaisesRegex(ValueError, 'same-side'):
            independent_actions(full)

    def test_episode_macro_not_dominated_by_long_episode(self):
        a = errors(np.zeros((1, 9)), np.ones((1, 9)), np.ones(9))
        b = errors(np.zeros((100, 9)), np.full((100, 9), 10), np.ones(9))
        self.assertEqual(aggregate([a, b])['episode_macro']['balanced_nrmse'], 5.5)
        self.assertEqual(aggregate([a, b])['per_output_episode_macro']['arm_torque_1']['mae'], 5.5)

    def test_inventory_refuses_corruption_and_duplicate_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_dataset(root)
            records = inventory(root)
            split = split_episodes(records, .2, 42)
            self.assertFalse(set(r['episode_id'] for r in split['training']) &
                             set(r['episode_id'] for r in split['validation']))
            meta_path = root / 'episodes/episode_000001.json'
            meta = read_json(meta_path)
            original = read_json(root / 'episodes/episode_000000.json')
            scenarios = read_json(root / 'scenarios.json')
            meta['target'] = original['target']; meta['actual_initial_observation'] = original['actual_initial_observation']
            scenarios[1]['target'] = original['target']
            write_json(meta_path, meta); write_json(root / 'scenarios.json', scenarios)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                inventory(root)
            scenarios[1]['target'] = [1., .2, .5]; meta['target'] = scenarios[1]['target']
            meta['actual_initial_observation'][0] += 1
            write_json(meta_path, meta); write_json(root / 'scenarios.json', scenarios)
            (root / 'episodes/episode_000001.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                inventory(root)

    def test_mlp_checkpoint_split_and_numpy_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'; root.mkdir(); make_dataset(root)
            config = read_json(DEFAULT_CONFIG.parent / 'mlp_baseline.json')
            config['mlp'].update(hidden_layer_sizes=[8, 8], max_epochs=3, patience=2, batch_size=16)
            out = Path(tmp) / 'mlp'
            policy, metrics = train(root, config, out)
            history = read_json(out / 'history.json')
            info = read_json(out / 'model_info.json')
            self.assertEqual(info['best_epoch'], min(history, key=lambda r: r['validation_episode_macro_balanced_nrmse'])['epoch'])
            self.assertAlmostEqual(info['best_validation_episode_macro_balanced_nrmse'],
                                   metrics['validation']['model']['episode_macro']['balanced_nrmse'])
            self.assertEqual(info['regression_parameters'], 21*8+8+8*8+8+8*9+9)
            split = read_json(out / 'split.json')
            linear_split = split_episodes(inventory(root), config['validation_fraction'], config['split_seed'])
            self.assertEqual(split['training'], linear_split['training'])
            with np.load(root / 'episodes' / f"{split['validation'][0]['episode_id']}.npz") as data:
                x = data['observation']
            restored = MLPPolicy.load(out / 'model.npz')
            np.testing.assert_array_equal(policy.predict(x), restored.predict(x))
            action = restored.predict_action(x[0])
            self.assertEqual(action.shape, (11,))
            self.assertEqual(action[0], action[3]); self.assertEqual(action[1], action[2])
            batched = restored.predict(x)
            np.testing.assert_allclose(restored.predict(x[0]), batched[0], rtol=1e-5, atol=1e-5)

    def test_training_only_scaling_and_portable_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'; root.mkdir(); make_dataset(root)
            out = Path(tmp) / 'trained'
            config = read_json(DEFAULT_CONFIG)
            policy, metrics = train(root, config, out)
            split = read_json(out / 'split.json')
            train_x = []
            for record in split['training']:
                with np.load(root / 'episodes' / f"{record['episode_id']}.npz") as data:
                    train_x.append(data['observation'].astype(float))
            expected = np.concatenate(train_x).mean(axis=0)
            np.testing.assert_allclose(policy.x_mean, expected, atol=1e-12)
            self.assertEqual(read_json(out / 'run.json')['status'], 'complete')
            with (out / 'per_output_metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2 * 2 * 9)
            joint = next(r for r in rows if r['split'] == 'validation' and
                         r['aggregation'] == 'episode_macro' and r['output'] == 'arm_torque_2')
            self.assertAlmostEqual(float(joint['model_mae']),
                metrics['validation']['model']['per_output_episode_macro']['arm_torque_2']['mae'])
            restored = LinearPolicy.load(out / 'model.npz')
            np.testing.assert_array_equal(policy.predict(train_x[0]), restored.predict(train_x[0]))
            self.assertLess(metrics['training']['pooled_transitions']['model']['balanced_nrmse'], 1e-10)
            self.assertEqual(sum(r['rows'] for r in split['training']) + sum(r['rows'] for r in split['validation']), 100)
            with self.assertRaises(FileExistsError):
                train(root, config, out)


if __name__ == '__main__':
    unittest.main()
