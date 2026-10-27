"""Causality, episode isolation, dense timing and portable temporal deployment."""
# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.data.storage import read_json, write_json, sha256
from robot_learning.learning.windows import encode_numpy, OnlineHistory, validate_episode_sampling
from robot_learning.learning.model import MLPPolicy, expand_actions
from robot_learning.learning.train import train, validate_config, DEFAULT_CONFIG
from test_learning import make_dataset

SPEC = dict(snapshots=6, stride_steps=20, dt_s=.001, padding='repeat_initial_state')


class WindowTests(unittest.TestCase):
    def test_agreed_200ms_design_and_ring_match_offline(self):
        from robot_learning.learning.temporal_comparison import window_variants
        variants,width=window_variants(200,20);spec=variants['window200ms']
        self.assertEqual(spec['snapshots'],11);self.assertEqual(width,193)
        self.assertEqual(201*128+128+128*128+128+128*9+9,43529)
        self.assertEqual(width*width+32*width+9,43434)
        base=np.arange(440*21,dtype=float).reshape(440,21)
        history=OnlineHistory(spec)
        live=np.array([history.append(row) for row in base])
        expected=encode_numpy(base,np.arange(len(base)),[0],spec)
        self.assertEqual(live.shape,(440,201))
        np.testing.assert_array_equal(live,expected)
        np.testing.assert_array_equal(live[200,:18],base[0,:18])
        np.testing.assert_array_equal(live[200,-21:-3],base[200,:18])
        with self.assertRaises(ValueError):window_variants(200,30)

    def test_causal_padding_episode_boundaries_and_current_goal(self):
        base = np.arange(260*21, dtype=float).reshape(260,21)
        indices = np.array([0, 19, 20, 99, 100, 129, 130, 131, 259])
        actual = encode_numpy(base, indices, [0,130], SPEC)
        for k, i in enumerate(indices):
            origin = 0 if i < 130 else 130
            expected = np.r_[base[[max(i-lag,origin) for lag in [100,80,60,40,20,0]],:18].ravel(),base[i,18:]]
            np.testing.assert_array_equal(actual[k],expected)
        future = base.copy(); future[101:] = -1e12
        np.testing.assert_array_equal(encode_numpy(base,np.arange(101),[0],SPEC),
                                      encode_numpy(future,np.arange(101),[0],SPEC))
        online = OnlineHistory(SPEC)
        np.testing.assert_array_equal(np.array([online.append(x) for x in base[:130]]),
                                      encode_numpy(base[:130],np.arange(130),[0],SPEC))

    def test_online_ring_export_reset_and_sampling_rejection(self):
        rng=np.random.default_rng(18);x=rng.normal(size=(4300,21))
        policy=MLPPolicy(np.zeros(21),np.ones(21),np.zeros(9),np.ones(9),
            [rng.normal(size=(111,8)).astype('float32'),rng.normal(size=(8,8)).astype('float32'),rng.normal(size=(8,9)).astype('float32')],
            [np.zeros(8),np.zeros(8),np.zeros(9)],window=SPEC)
        batch=policy.predict(x)
        live=np.array([policy.predict_action(row) for row in x])
        np.testing.assert_allclose(live,expand_actions(batch),rtol=1e-10,atol=1e-10)
        policy.reset(.001)
        np.testing.assert_allclose(policy.predict_action(x[0]),expand_actions(batch[0]))
        with self.assertRaises(ValueError):policy.reset(.002)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';policy.save(path);restored=MLPPolicy.load(path)
            self.assertEqual(restored.window,SPEC)
            np.testing.assert_array_equal(restored.predict(x),batch)
        for step in [np.array([0,2,4]),np.array([1,2,3])]:
            with self.assertRaises(ValueError):validate_episode_sampling({'step':step,'time':step*.001},{'dt':.001},SPEC)
        validate_episode_sampling({'step':np.arange(4),'time':np.arange(4)*.001},{'dt':.001},SPEC)

    def test_torch_gather_training_export_resume_and_no_validation_scaler_leakage(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional PyTorch environment is not installed')
        from robot_learning.learning.windows import encode_torch, lags
        torch.set_num_threads(1)
        base=np.arange(130*21,dtype=np.float32).reshape(130,21);indices=np.array([0,30,69,70,71,129])
        expected=encode_numpy(base,indices,[0,70],SPEC)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            actual=encode_torch(torch.tensor(base,device=device),torch.tensor(indices,device=device),
                torch.tensor([0,70],device=device),torch.tensor(lags(SPEC),device=device))
            np.testing.assert_array_equal(actual.cpu().numpy(),expected)
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=Path(tmp);data=root/'data';data.mkdir();make_dataset(data)
            for p in (data/'episodes').glob('*.npz'):
                with np.load(p) as z:arrays={k:z[k] for k in z.files}
                arrays.update(step=np.arange(len(arrays['observation'])),time=np.arange(len(arrays['observation']))*.001)
                np.savez_compressed(p,**arrays)
                meta=read_json(p.with_suffix('.json'));meta.update(dt=.001,data_sha256=sha256(p));write_json(p.with_suffix('.json'),meta)
            cfg=read_json(DEFAULT_CONFIG.parent/'mlp_pytorch.json');cfg['torch']['device']='cpu'
            cfg['mlp'].update(hidden_layer_sizes=[8,8],batch_size=16,max_epochs=4,patience=100)
            cfg['observation_window']={**SPEC,'stride_steps':2}
            policy,metrics=train(data,cfg,root/'full',mapped=True)
            info=read_json(root/'full/model_info.json');self.assertEqual(info['input_features'],111)
            self.assertEqual(info['regression_parameters'],111*8+8+8*8+8+8*9+9)
            self.assertAlmostEqual(info['best_validation_episode_macro_balanced_nrmse'],metrics['validation']['model']['episode_macro']['balanced_nrmse'],places=5)
            split=read_json(root/'full/split.json')
            raw=[]
            for r in split['training']:
                with np.load(data/'episodes'/f"{r['episode_id']}.npz") as z:raw.append(z['observation'])
            np.testing.assert_allclose(policy.x_mean,np.concatenate(raw).mean(axis=0,dtype=np.float64),atol=1e-10)
            short=deepcopy(cfg);short['mlp']['max_epochs']=2
            train(data,short,root/'short',mapped=True)
            resumed,_=train(data,cfg,root/'resumed',mapped=True,preprocessing_from=root/'short',resume_checkpoint=root/'short/optimizer_latest.json')
            for k,v in policy.arrays().items():np.testing.assert_array_equal(v,resumed.arrays()[k])
            stream=deepcopy(cfg);stream['torch']['data_placement']='stream'
            streamed,_=train(data,stream,root/'stream',mapped=True,preprocessing_from=root/'short')
            for k,v in policy.arrays().items():np.testing.assert_array_equal(v,streamed.arrays()[k])
            bad=deepcopy(cfg);bad['backend']='sklearn';bad.pop('torch')
            with self.assertRaises(ValueError):validate_config(bad)
