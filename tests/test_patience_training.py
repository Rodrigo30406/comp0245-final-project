"""Check bounded-memory metrics, patience-only stopping and exact Adam resume."""
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from test_learning import make_dataset
from robot_learning.learning.train import train,validate_config,DEFAULT_CONFIG,evaluate,evaluate_streaming
from robot_learning.data.storage import read_json
from robot_learning.learning.dataset import inventory,split_episodes


class PatienceTrainingTests(unittest.TestCase):
    def config(self):
        cfg=read_json(DEFAULT_CONFIG.parent/'mlp_baseline.json')
        cfg['mlp'].update(hidden_layer_sizes=[8,8],batch_size=16,max_epochs=6,patience=100)
        return cfg

    def test_resume_preserves_optimizer_shuffle_and_frozen_scalers(self):
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            cfg=self.config();reference,_=train(data,cfg,root/'reference')
            short=deepcopy(cfg);short['mlp']['max_epochs']=3
            train(data,short,root/'short')
            restored,_=train(data,cfg,root/'resumed',mapped=True,preprocessing_from=root/'short',
                             resume_checkpoint=root/'short/optimizer_latest.json')
            for k,a in reference.arrays().items():np.testing.assert_array_equal(a,restored.arrays()[k])
            history=read_json(root/'resumed/history.json');original=read_json(root/'reference/history.json')
            self.assertEqual([h['validation_episode_macro_balanced_nrmse'] for h in history],
                             [h['validation_episode_macro_balanced_nrmse'] for h in original])
            self.assertFalse((root/'resumed/work').exists())
            bad=deepcopy(cfg);bad['mlp']['learning_rate']=.002
            with self.assertRaisesRegex(ValueError,'mismatch'):
                train(data,bad,root/'bad',mapped=True,preprocessing_from=root/'short',
                      resume_checkpoint=root/'short/optimizer_latest.json')

    def test_patience_only_and_streaming_metrics(self):
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            cfg=self.config();cfg['mlp'].update(max_epochs=None,patience=2,min_delta=100.)
            policy,_=train(data,cfg,root/'mapped',mapped=True)
            info=read_json(root/'mapped/model_info.json')
            self.assertEqual(info['stop_reason'],'patience');self.assertEqual(info['epochs_completed'],3)
            records=inventory(data);split=split_episodes(records,cfg['validation_fraction'],cfg['split_seed'])
            for side in ['training','validation']:
                a=evaluate(data,split[side],policy);b=evaluate_streaming(data,split[side],policy)
                self.assertEqual(a['model'],b['model']);self.assertEqual(a['per_episode'],b['per_episode'])
                for name in ['model','mean_baseline']:
                    for output,v in a['pooled_transitions'][name]['per_output'].items():
                        for metric in ['mae','rmse','nrmse_train_std']:
                            self.assertAlmostEqual(v[metric],b['pooled_transitions'][name]['per_output'][output][metric],places=10)
            expected=np.concatenate([__import__('robot_learning.learning.dataset',fromlist=['load_episode']).load_episode(data,r)[0] for r in split['training']]).mean(axis=0)
            np.testing.assert_allclose(policy.x_mean,expected,rtol=1e-12,atol=1e-12)
            cfg['mlp']['min_delta']=0
            with self.assertRaises(ValueError):validate_config(cfg)
