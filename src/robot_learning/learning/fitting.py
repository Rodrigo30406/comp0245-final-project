"""Estimator-specific fitting; datasets, scaling and final evaluation stay shared."""
import time
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from robot_learning.data.storage import write_json,read_json,sha256
from .dataset import load_episode
from .model import LinearPolicy, MLPPolicy
from .metrics import errors


def fit_estimator(x, y, x_scaler, y_scaler, config, dataset, validation_records, output,
                  *, mapped=False, checkpoint_context=None, resume_checkpoint=None, training_ranges=None):
    if config.get('backend','sklearn')=='pytorch':
        from .torch_fitting import fit_torch
        return fit_torch(x,y,x_scaler,y_scaler,config,dataset,validation_records,output,
                         mapped=mapped,checkpoint_context=checkpoint_context,resume_checkpoint=resume_checkpoint,
                         training_ranges=training_ranges)
    started = time.perf_counter()
    normalizers = (x_scaler.mean_, x_scaler.scale_, y_scaler.mean_, y_scaler.scale_)
    if config['model'] == 'linear_regression':
        if mapped:raise ValueError('Mapped estimator path is currently MLP-only')
        xn = x_scaler.transform(x);yn = y_scaler.transform(y)
        estimator = LinearRegression(fit_intercept=True).fit(xn, yn)
        policy = LinearPolicy(*normalizers, estimator.coef_, estimator.intercept_)
        np.testing.assert_allclose(policy.predict(x[:128]),
            y_scaler.inverse_transform(estimator.predict(xn[:128])), rtol=1e-12, atol=1e-12)
        info = {'kind': 'ordinary_least_squares', 'fit_intercept': True,
                'regression_parameters': int(estimator.coef_.size + estimator.intercept_.size),
                'training_design_rank': int(estimator.rank_),
                'training_singular_values': estimator.singular_.tolist()}
    else:
        hp = config['mlp']
        if mapped:
            from .workspace import normalized_maps
            xn,yn=normalized_maps(x,y,x_scaler,y_scaler,output)
        else:
            xn=x_scaler.transform(x).astype(np.float32)
            yn=y_scaler.transform(y).astype(np.float32)
        validation = [load_episode(dataset, record)[:2] for record in validation_records]
        estimator = MLPRegressor(hidden_layer_sizes=tuple(hp['hidden_layer_sizes']), activation='relu',
            solver='adam', alpha=hp['alpha'], batch_size=hp['batch_size'],
            learning_rate_init=hp['learning_rate'], random_state=np.random.RandomState(hp['seed']),
            early_stopping=False, shuffle=True)
        best_score = np.inf; patience_score = np.inf; stale = 0; history = [];start_epoch=1
        best_arrays=None
        signature=json.loads(json.dumps(config));signature['mlp'].pop('max_epochs');signature.pop('name')
        checkpoint_context=checkpoint_context or {}
        if resume_checkpoint is not None:
            source=Path(resume_checkpoint)
            if source.suffix=='.json':
                header=read_json(source);source=source.parent/header['file']
            else:header=read_json(source.with_suffix('.json'))
            if sha256(source)!=header['sha256'] or header['context']!=checkpoint_context or header['config_without_cap']!=signature:
                raise ValueError('Checkpoint integrity/configuration/data mismatch')
            saved=joblib.load(source)  # Only hash-verified checkpoints produced by this trainer.
            for a,b in zip(normalizers,saved['normalizers']):np.testing.assert_array_equal(a,b)
            estimator=saved['estimator'];best_score=saved['best_score'];best_epoch=saved['best_epoch']
            patience_score=saved['patience_score'];stale=saved['stale'];history=saved['history']
            best_arrays=saved['best_arrays'];start_epoch=saved['epoch']+1
            layer_count=len(hp['hidden_layer_sizes'])+1
            MLPPolicy(*normalizers,[best_arrays[f'weight_{i}'] for i in range(layer_count)],
                      [best_arrays[f'bias_{i}'] for i in range(layer_count)]).save(output/'model.npz')
        epoch=start_epoch-1
        for epoch in itertools.count(start_epoch):
            if stale>=hp['patience'] or (hp['max_epochs'] is not None and epoch>hp['max_epochs']):
                epoch-=1;break
            epoch_start = time.perf_counter()
            estimator.partial_fit(xn, yn)  # Exactly one epoch; Adam state retained.
            current = MLPPolicy(*normalizers, estimator.coefs_, estimator.intercepts_)
            score = float(np.mean([errors(vy, current.predict(vx), y_scaler.scale_)['balanced_nrmse']
                                   for vx, vy in validation]))
            if not np.isfinite(score) or not np.isfinite(estimator.loss_):
                raise ValueError('Non-finite MLP training or validation error')
            if score < best_score:
                best_score = score; best_epoch = epoch
                current.save(output / 'model.npz')
                best_arrays={k:v.copy() for k,v in current.arrays().items()}
            if score < patience_score - hp['min_delta']:
                patience_score = score; stale = 0
            else:
                stale += 1
            history.append({'epoch': epoch, 'training_normalized_loss': float(estimator.loss_),
                            'validation_episode_macro_balanced_nrmse': score,
                            'seconds': time.perf_counter()-epoch_start})
            write_json(output / 'history.json', history)
            folder=output/'optimizer_checkpoints';folder.mkdir(exist_ok=True)
            checkpoint=folder/f'epoch_{epoch:06d}.joblib';temporary=checkpoint.with_suffix('.tmp')
            joblib.dump({'schema_version':1,'estimator':estimator,'normalizers':normalizers,
                'epoch':epoch,'best_epoch':best_epoch,'best_score':best_score,'patience_score':patience_score,
                'stale':stale,'history':history,'best_arrays':best_arrays},temporary,compress=3)
            temporary.replace(checkpoint)
            header={'schema_version':1,'sha256':sha256(checkpoint),'file':checkpoint.name,
                    'context':checkpoint_context,'config_without_cap':signature,'epoch':epoch}
            write_json(checkpoint.with_suffix('.json'),header)
            write_json(output/'optimizer_latest.json',{**header,'file':str(checkpoint.relative_to(output))})
            print(f'Epoch {epoch}: loss={estimator.loss_:.6f}, validation NRMSE={score:.6f}, best={best_score:.6f}', flush=True)
            if stale >= hp['patience']:
                break
        policy = MLPPolicy.load(output / 'model.npz')
        # Check NumPy inference against sklearn at the final epoch before selecting best weights.
        current=MLPPolicy(*normalizers,estimator.coefs_,estimator.intercepts_)
        np.testing.assert_allclose(current.predict(x[:128]),
            y_scaler.inverse_transform(estimator.predict(xn[:128])).astype(np.float64), rtol=3e-5, atol=3e-5)
        info = {'kind': 'mlp_regressor', 'hidden_layer_sizes': hp['hidden_layer_sizes'],
                'activation': 'relu', 'output_activation': 'identity', 'optimizer': 'adam',
                'regression_parameters': sum(w.size + b.size for w,b in zip(policy.weights,policy.biases)),
                'weight_dtype': str(policy.weights[0].dtype),
                'epochs_completed': epoch, 'best_epoch': best_epoch,
                'best_validation_episode_macro_balanced_nrmse': best_score,
                'stop_reason': 'patience' if stale >= hp['patience'] else 'max_epochs',
                'checkpoint_selection': 'minimum validation episode-macro balanced NRMSE',
                'training_loss': 'uniform-channel normalized squared error plus sklearn L2 penalty; transitions uniformly weighted',
                'training_updates': epoch * int(np.ceil(len(x)/hp['batch_size'])),
                'training_examples_seen': epoch * len(x),
                'resumed_from_epoch':start_epoch-1,'epochs_executed_this_run':epoch-start_epoch+1,
                'cumulative_fit_validation_seconds':sum(h['seconds'] for h in history),
                'optimizer_checkpoint':'optimizer_latest.json; full Adam and shuffle state, immutable epoch generations',
                'validation_for_stopping': 'explicit whole held-out episodes; sklearn automatic random row holdout disabled'}
    info['fit_and_transform_seconds'] = time.perf_counter() - started
    return policy, info
