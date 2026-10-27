"""Nested learning curves must preserve holdouts and reject cross-model leakage."""
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from test_learning import make_dataset
from robot_learning.learning.dataset import inventory, split_episodes, nested_partition, validate_partition
from robot_learning.evaluation.cli import validate_common_validation


class NestedSplitTests(unittest.TestCase):
    def test_nested_holdouts_and_explicit_partition_rejection(self):
        records=[{'episode_id':str(i),'scenario_id':str(i),'target':[i,0,0]} for i in range(512)]
        old=split_episodes(records[:64],.2,20261025)
        for n,val in [(128,26),(256,52),(512,103)]:
            new=nested_partition(records[:n],old,.2,20261025)
            self.assertEqual(len(new['validation']),val)
            for side in ['training','validation']:
                self.assertTrue({r['episode_id'] for r in old[side]} <= {r['episode_id'] for r in new[side]})
            self.assertEqual(new,nested_partition(records[:n],old,.2,20261025))
            old=new
        bad=deepcopy(old);bad['validation'].append(old['training'][0])
        with self.assertRaisesRegex(ValueError,'overlapping'):validate_partition(records,bad)
        bad=deepcopy(old);bad['training'][0]['target']=[99,99,99]
        with self.assertRaisesRegex(ValueError,'stale'):validate_partition(records,bad)

    def test_common_evaluation_rejects_training_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_dataset(root)
            partition=split_episodes(inventory(root),.2,42)
            manifest={'dataset':str(root),**partition}
            validate_common_validation(root,partition['validation'],manifest)
            with self.assertRaisesRegex(ValueError,'leakage'):
                validate_common_validation(root,partition['training'][:1],manifest)
