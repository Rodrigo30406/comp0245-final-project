"""Variable depth: portable export, linear output, temporal execution and recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.data.storage import read_json,write_json,sha256
from robot_learning.learning.model import MLPPolicy,expand_actions
from robot_learning.learning.train import train,DEFAULT_CONFIG,validate_config
from test_learning import make_dataset


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Optional PyTorch environment required')
class DeepMLPTests(unittest.TestCase):
    def test_deep_export_linear_output_and_temporal_live_batch_agreement(self):
        import torch
        from robot_learning.learning.torch_fitting import build_network,export_policy
        torch.set_num_threads(1)
        hp=read_json(DEFAULT_CONFIG.parent/'mlp_pytorch.json')['mlp']
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            for hidden in [[32,64,128],[32,64,128,64,32]]:
                for window in [None,dict(snapshots=11,stride_steps=20,dt_s=.001,padding='repeat_initial_state')]:
                    hp['hidden_layer_sizes']=hidden;inputs=21 if window is None else 201
                    network,_=build_network(hp,torch.device('cpu'),input_features=inputs)
                    with torch.no_grad():
                        network[-1].weight.zero_();network[-1].bias.copy_(torch.arange(-9,0,dtype=torch.float32))
                    policy=export_policy(network,(np.zeros(21),np.ones(21),np.zeros(9),np.ones(9)),window=window)
                    x=np.random.default_rng(3).normal(size=(430,21))
                    expected=np.broadcast_to(np.arange(-9.,0.),(len(x),9))
                    np.testing.assert_array_equal(policy.predict(x),expected)
                    self.assertEqual(len(policy.weights),len(hidden)+1)
                    path=Path(tmp)/'model.npz';policy.save(path);restored=MLPPolicy.load(path)
                    np.testing.assert_array_equal(restored.predict(x),expected)
                    if window is not None:
                        np.testing.assert_array_equal(np.array([restored.predict_action(row) for row in x]),expand_actions(expected))
                    # Nonconstant weights check every hidden layer against Torch.
                    actual_network,_=build_network(hp,torch.device('cpu'),input_features=inputs)
                    p=export_policy(actual_network,(np.zeros(21),np.ones(21),np.zeros(9),np.ones(9)),window=window)
                    features=np.random.default_rng(4).normal(size=(30,inputs)).astype(np.float32)
                    with torch.inference_mode():pred=actual_network(torch.from_numpy(features)).numpy()
                    np.testing.assert_allclose(p.predict_encoded(features),pred,rtol=2e-5,atol=2e-6)
                    if window is not None:
                        np.testing.assert_allclose(np.array([p.predict_action(row) for row in x]),expand_actions(p.predict(x)),rtol=2e-5,atol=2e-6)
            bad=policy.arrays();bad['weight_8']=bad.pop('weight_0');np.savez(Path(tmp)/'bad.npz',**bad)
            with self.assertRaises(ValueError):MLPPolicy.load(Path(tmp)/'bad.npz')

    def test_deep_torch_and_sklearn_training_resume_export_and_same_scalers(self):
        import torch
        torch.set_num_threads(1)
        cfg=read_json(DEFAULT_CONFIG.parent/'mlp_pytorch.json');cfg['torch']['device']='cpu'
        cfg['mlp'].update(hidden_layer_sizes=[4,6,8,6,4],batch_size=16,max_epochs=3,patience=100)
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            full,metrics=train(data,cfg,root/'full',mapped=True)
            info=read_json(root/'full/model_info.json');self.assertEqual(len(full.weights),6)
            self.assertAlmostEqual(info['best_validation_episode_macro_balanced_nrmse'],metrics['validation']['model']['episode_macro']['balanced_nrmse'],places=5)
            short=deepcopy(cfg);short['mlp']['max_epochs']=1
            train(data,short,root/'short',mapped=True)
            resumed,_=train(data,cfg,root/'resume',mapped=True,preprocessing_from=root/'short',resume_checkpoint=root/'short/optimizer_latest.json')
            for k,v in full.arrays().items():np.testing.assert_array_equal(v,resumed.arrays()[k])
            sklearn=deepcopy(cfg);sklearn.pop('torch');sklearn['backend']='sklearn'
            sk,_=train(data,sklearn,root/'sk',mapped=True,preprocessing_from=root/'full')
            train(data,{**sklearn,'mlp':{**sklearn['mlp'],'max_epochs':1}},root/'skshort',mapped=True)
            skr,_=train(data,sklearn,root/'skresume',mapped=True,preprocessing_from=root/'skshort',resume_checkpoint=root/'skshort/optimizer_latest.json')
            for k,v in sk.arrays().items():np.testing.assert_array_equal(v,skr.arrays()[k])
            np.testing.assert_array_equal(sk.x_mean,full.x_mean)
            with np.load(root/'full/model.npz') as z:self.assertIn('weight_5',z.files)
            for invalid in [[],[32,0],[True,32]]:
                bad=deepcopy(cfg);bad['mlp']['hidden_layer_sizes']=invalid
                with self.assertRaises(ValueError):validate_config(bad)

    def test_actual_cuda_deep_200ms_training_when_available(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        torch.set_num_threads(1)
        cfg=read_json(DEFAULT_CONFIG.parent/'mlp_pytorch.json')
        cfg['mlp'].update(hidden_layer_sizes=[4,6,8,6,4],batch_size=16,max_epochs=None,patience=2,min_delta=100.)
        cfg['observation_window']=dict(snapshots=11,stride_steps=20,dt_s=.001,padding='repeat_initial_state')
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            for path in (data/'episodes').glob('*.npz'):
                with np.load(path) as z:arrays={k:z[k] for k in z.files}
                n=len(arrays['observation']);arrays.update(step=np.arange(n),time=np.arange(n)*.001)
                np.savez_compressed(path,**arrays);meta=read_json(path.with_suffix('.json'))
                meta.update(dt=.001,data_sha256=sha256(path));write_json(path.with_suffix('.json'),meta)
            policy,metrics=train(data,cfg,root/'gpu',mapped=True)
            info=read_json(root/'gpu/model_info.json')
            self.assertEqual(info['training_device'],'cuda');self.assertEqual(info['stop_reason'],'patience')
            self.assertEqual(policy.input_features,201);self.assertEqual(len(policy.weights),6)
            self.assertAlmostEqual(info['best_validation_episode_macro_balanced_nrmse'],metrics['validation']['model']['episode_macro']['balanced_nrmse'],places=5)
