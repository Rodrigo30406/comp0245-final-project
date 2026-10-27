"""Cross-framework math, portable export, checkpoint recovery and CUDA fitting."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from robot_learning.data.storage import read_json
from robot_learning.learning.dataset import inventory, split_episodes
from robot_learning.learning.train import train, validate_config, DEFAULT_CONFIG
from robot_learning.learning.model import MLPPolicy
from test_learning import make_dataset

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional PyTorch environment is not installed')
class TorchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def config(self, device='cpu'):
        cfg = read_json(DEFAULT_CONFIG.parent/'mlp_pytorch.json')
        cfg['mlp'].update(hidden_layer_sizes=[8,8], batch_size=16, max_epochs=4, patience=100)
        cfg['torch']['device'] = device
        return cfg

    def test_actual_sklearn_first_adam_update_and_reported_loss(self):
        import torch
        from robot_learning.learning.torch_fitting import build_network, batch_objective, adam_step
        cfg = self.config(); hp = cfg['mlp']; hp['batch_size'] = 32
        rng = np.random.default_rng(821)
        x = rng.normal(size=(32,21)).astype(np.float32); y = rng.normal(size=(32,9)).astype(np.float32)
        with threadpool_limits(limits=1):
            ref = MLPRegressor(hidden_layer_sizes=(8,8), activation='relu', solver='adam',
                alpha=hp['alpha'], learning_rate_init=hp['learning_rate'], batch_size=32,
                random_state=np.random.RandomState(hp['seed']), shuffle=False)
            ref.partial_fit(x,y)
            model,_ = build_network(hp,torch.device('cpu'))
            optimizer = torch.optim.Adam(model.parameters(),lr=hp['learning_rate'],betas=(.9,.999),eps=1e-8)
            objective,reported = batch_objective(model,torch.from_numpy(x),torch.from_numpy(y),hp['alpha'])
            self.assertAlmostEqual(float(reported),float(ref.loss_),places=6)
            objective.backward();adam_step(optimizer,1)
            for i,layer in enumerate(model[::2]):
                np.testing.assert_allclose(layer.weight.detach().numpy().T,ref.coefs_[i],rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(layer.bias.detach().numpy(),ref.intercepts_[i],rtol=2e-5,atol=2e-6)

    def test_resume_restores_adam_shuffle_and_export_exactly(self):
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            cfg=self.config();full,_=train(data,cfg,root/'full',mapped=True)
            short=deepcopy(cfg);short['mlp']['max_epochs']=2
            train(data,short,root/'short',mapped=True)
            resumed,_=train(data,cfg,root/'resume',mapped=True,preprocessing_from=root/'short',
                            resume_checkpoint=root/'short/optimizer_latest.json')
            for key,value in full.arrays().items():np.testing.assert_array_equal(value,resumed.arrays()[key])
            a=read_json(root/'full/history.json');b=read_json(root/'resume/history.json')
            self.assertEqual([x['validation_episode_macro_balanced_nrmse'] for x in a],
                             [x['validation_episode_macro_balanced_nrmse'] for x in b])
            info=read_json(root/'resume/model_info.json');self.assertEqual(info['resumed_from_epoch'],2)
            self.assertFalse((root/'resume/work').exists());self.assertFalse((root/'resume/torch_validation').exists())
            bad=deepcopy(cfg);bad['mlp']['alpha']=.01
            with self.assertRaisesRegex(ValueError,'mismatch'):
                train(data,bad,root/'bad',mapped=True,preprocessing_from=root/'short',
                      resume_checkpoint=root/'short/optimizer_latest.json')

    def test_patience_and_streaming_placement_use_same_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            cfg=self.config();cfg['mlp'].update(max_epochs=None,patience=2,min_delta=100.)
            a,metric=train(data,cfg,root/'resident',mapped=True)
            cfg['torch']['data_placement']='stream'
            b,_=train(data,cfg,root/'stream',mapped=True,preprocessing_from=root/'resident')
            for k,v in a.arrays().items():np.testing.assert_array_equal(v,b.arrays()[k])
            info=read_json(root/'resident/model_info.json')
            self.assertEqual(info['epochs_completed'],3);self.assertEqual(info['stop_reason'],'patience')
            self.assertEqual(info['best_epoch'],min(read_json(root/'resident/history.json'),key=lambda h:h['validation_episode_macro_balanced_nrmse'])['epoch'])
            self.assertAlmostEqual(info['best_validation_episode_macro_balanced_nrmse'],metric['validation']['model']['episode_macro']['balanced_nrmse'],places=5)
            restored=MLPPolicy.load(root/'resident/model.npz');self.assertEqual(restored.predict_action(np.zeros(21)).shape,(11,))
            cfg['torch']['device']='invalid'
            with self.assertRaises(ValueError):validate_config(cfg)

    def test_cuda_forward_export_and_training_when_available(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA device unavailable')
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            cfg=self.config('cuda');cfg['mlp'].update(max_epochs=None,patience=2,min_delta=100.)
            policy,metric=train(data,cfg,root/'gpu',mapped=True)
            info=read_json(root/'gpu/model_info.json')
            self.assertEqual(info['training_device'],'cuda');self.assertGreater(info['gpu_peak_allocated_bytes'],0)
            self.assertEqual(info['stop_reason'],'patience');self.assertEqual(info['epochs_completed'],3)
            self.assertAlmostEqual(info['best_validation_episode_macro_balanced_nrmse'],metric['validation']['model']['episode_macro']['balanced_nrmse'],places=5)
            resumed,_=train(data,cfg,root/'already_done',mapped=True,preprocessing_from=root/'gpu',
                            resume_checkpoint=root/'gpu/optimizer_latest.json')
            for k,v in policy.arrays().items():np.testing.assert_array_equal(v,resumed.arrays()[k])
            self.assertEqual(len(read_json(root/'already_done/history.json')),3)

            # Recover an interrupted CUDA optimizer and continue actual updates.
            cfg['mlp'].update(max_epochs=4,patience=100,min_delta=1e-4)
            full,_=train(data,cfg,root/'cuda_full',mapped=True)
            short=deepcopy(cfg);short['mlp']['max_epochs']=2
            train(data,short,root/'cuda_short',mapped=True)
            continued,_=train(data,cfg,root/'cuda_resumed',mapped=True,preprocessing_from=root/'cuda_short',
                              resume_checkpoint=root/'cuda_short/optimizer_latest.json')
            for k,v in full.arrays().items():np.testing.assert_array_equal(v,continued.arrays()[k])
