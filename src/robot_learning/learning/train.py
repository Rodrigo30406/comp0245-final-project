"""Reproducible ETL, episode split, normalized regression and offline validation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info
from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json, write_json, sha256, git_info, utc_now
from .dataset import inventory, split_episodes, load_episode, inventory_fingerprint, validate_partition
from .model import INPUT_NAMES, OUTPUT_NAMES
from .fitting import fit_estimator
from .metrics import errors, aggregate, write_channel_report, ErrorMoments

DEFAULT_CONFIG = PROJECT_ROOT / 'experiment_specs/training/linear_baseline.json'


def validate_config(config):
    expected = {'name', 'model', 'split_seed', 'validation_fraction', 'normalization',
                'sample_weighting', 'primary_metric', 'keep_expert_failures'}
    if config.get('model') == 'mlp':
        expected.add('mlp')
    if 'backend' in config:
        expected.add('backend')
    backend=config.get('backend','sklearn')
    if 'observation_window' in config:
        from .windows import validate_window
        expected.add('observation_window')
        validate_window(config['observation_window'])
        if backend != 'pytorch' or config.get('model') != 'mlp':
            raise ValueError('Observation windows currently require the PyTorch MLP backend')
    if backend not in ('sklearn','pytorch') or (backend=='pytorch' and config.get('model')!='mlp'):
        raise ValueError('PyTorch backend requires model=mlp')
    if backend=='pytorch':
        expected.add('torch')
    if set(config) != expected:
        raise ValueError("Unknown or missing training configuration keys")
    if config['model'] not in ('linear_regression', 'mlp'):
        raise ValueError('Unsupported model')
    required = {'normalization': 'standard_scaler_training_only',
                'sample_weighting': 'uniform_transition',
                'primary_metric': 'episode_macro_balanced_nrmse', 'keep_expert_failures': True}
    if any(config[k] != v for k, v in required.items()):
        raise ValueError("Unsupported baseline setting")
    if type(config['split_seed']) is not int or config['split_seed'] < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    if type(config['validation_fraction']) not in (float, int) or not 0 < config['validation_fraction'] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if config['model'] == 'mlp':
        hp = config['mlp']
        if set(hp) != {'hidden_layer_sizes', 'learning_rate', 'alpha', 'batch_size',
                        'seed', 'max_epochs', 'patience', 'min_delta'}:
            raise ValueError('Invalid MLP hyperparameters')
        if (not isinstance(hp['hidden_layer_sizes'],list) or not hp['hidden_layer_sizes'] or
                any(type(n) is not int or n < 1 for n in hp['hidden_layer_sizes'])):
            raise ValueError('Specify a nonempty list of positive hidden layer widths')
        for key in ['batch_size', 'patience', 'seed']:
            if type(hp[key]) is not int or hp[key] < (0 if key == 'seed' else 1):
                raise ValueError(f'Invalid MLP {key}')
        if hp['max_epochs'] is not None and (type(hp['max_epochs']) is not int or hp['max_epochs']<1):
            raise ValueError('max_epochs must be positive or null for patience-only training')
        for key in ['learning_rate', 'alpha', 'min_delta']:
            if type(hp[key]) not in (int, float) or not np.isfinite(hp[key]) or hp[key] < 0:
                raise ValueError(f'Invalid MLP {key}')
        if hp['learning_rate'] == 0:
            raise ValueError('learning_rate must be positive')
        if hp['max_epochs'] is None and hp['min_delta']<=0:
            raise ValueError('Patience-only training requires positive min_delta')
    if backend=='pytorch':
        from .torch_fitting import validate_torch_config
        validate_torch_config(config['torch'])



def evaluate(directory, records, policy):
    episodes = []; all_y = []; all_pred = []; all_phase = []
    for record in records:
        x, y, phase = load_episode(directory, record, window=getattr(policy, 'window', None))
        prediction = policy.predict(x)
        episodes.append({'episode_id': record['episode_id'], 'expert_success': record['success'],
                         'model': errors(y, prediction, policy.y_scale),
                         'mean_baseline': errors(y, np.broadcast_to(policy.y_mean, y.shape), policy.y_scale)})
        all_y.append(y); all_pred.append(prediction); all_phase.append(phase)
    y = np.concatenate(all_y); pred = np.concatenate(all_pred); phase = np.concatenate(all_phase)
    mean = np.broadcast_to(policy.y_mean, y.shape)
    return {
        'model': aggregate([e['model'] for e in episodes]),
        'mean_baseline': aggregate([e['mean_baseline'] for e in episodes]),
        'pooled_transitions': {'model': errors(y, pred, policy.y_scale),
                               'mean_baseline': errors(y, mean, policy.y_scale)},
        'by_expert_phase_pooled': {str(label): {'model': errors(y[phase == label], pred[phase == label], policy.y_scale),
                                                'mean_baseline': errors(y[phase == label], mean[phase == label], policy.y_scale)}
                                   for label in np.unique(phase)},
        'per_episode': episodes,
    }


def evaluate_streaming(directory, records, policy):
    """Same episode scores and pooled moments without concatenating all rows."""
    episodes=[];moments={k:ErrorMoments() for k in ['model','mean_baseline']};phases={}
    for record in records:
        x,y,phase=load_episode(directory,record,window=getattr(policy,'window',None));pred=policy.predict(x)
        mean=np.broadcast_to(policy.y_mean,y.shape)
        episodes.append({'episode_id':record['episode_id'],'expert_success':record['success'],
                         'model':errors(y,pred,policy.y_scale),'mean_baseline':errors(y,mean,policy.y_scale)})
        for name,p in [('model',pred),('mean_baseline',mean)]:moments[name].add(y,p)
        for label in np.unique(phase):
            key=str(label);phases.setdefault(key,{k:ErrorMoments() for k in ['model','mean_baseline']})
            mask=phase==label
            for name,p in [('model',pred),('mean_baseline',mean)]:phases[key][name].add(y[mask],p[mask])
    return {'model':aggregate([e['model'] for e in episodes]),
            'mean_baseline':aggregate([e['mean_baseline'] for e in episodes]),
            'pooled_transitions':{k:v.result(policy.y_scale) for k,v in moments.items()},
            'by_expert_phase_pooled':{label:{k:v.result(policy.y_scale) for k,v in group.items()}
                                     for label,group in sorted(phases.items())},'per_episode':episodes}


def train(dataset, config, output, *, explicit_split=None, mapped=False, preprocessing_from=None, resume_checkpoint=None):
    validate_config(config)
    if config.get('observation_window') is not None and not mapped:
        raise ValueError('Temporal fitting requires mapped=True for validated per-episode data')
    if mapped and config['model']!='mlp':
        raise ValueError('Mapped training currently supports MLP only')
    if config.get('backend')=='pytorch':
        try:
            import torch  # Optional dependency; the simulator environment stays unchanged.
        except ImportError as exc:
            raise RuntimeError('PyTorch backend needs the optional ML environment; see docs/PYTORCH_TRAINING.md') from exc
        if config['torch']['device']=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; select device=cpu/auto explicitly')
    dataset = Path(dataset).resolve(); output = Path(output).resolve()
    # Fail closed on partial datasets. Choose exactly one collection, never benchmark replicas.
    records = inventory(dataset)
    split = (split_episodes(records, config['validation_fraction'], config['split_seed'])
             if explicit_split is None else validate_partition(records, explicit_split))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    split_manifest = {'dataset': str(dataset), 'inventory_sha256': inventory_fingerprint(records),
                      'seed': config['split_seed'], 'validation_fraction': config['validation_fraction'],
                      'unit': 'whole episode; duplicate physical scenarios rejected', **split}
    write_json(output / 'split.json', split_manifest)
    source_paths = sorted((PROJECT_ROOT / 'src/robot_learning/learning').glob('*.py'))
    source_paths += [PROJECT_ROOT / 'src/robot_learning/data/storage.py',
                     PROJECT_ROOT / 'src/robot_learning/settings.py', PROJECT_ROOT / 'scripts/train_model.py']
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(PROJECT_ROOT)
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = sha256(path)
    input_names = INPUT_NAMES
    if config.get('observation_window') is not None:
        from .windows import feature_names
        input_names = feature_names(config['observation_window'], INPUT_NAMES)
    metadata = {'status': 'running', 'created_at': utc_now(), 'dataset': str(dataset),
                'dataset_hashes': {n: sha256(dataset / n) for n in
                                   ['dataset.json', 'config.json', 'scenarios.json', 'schema.json', 'provenance.json']},
                'source_hashes': source_hashes, 'project': git_info(PROJECT_ROOT),
                'versions': {name: importlib.metadata.version(name) for name in
                             ['numpy', 'scipy', 'scikit-learn', 'threadpoolctl', 'joblib']},
                'python': sys.version, 'command': sys.argv,
                'input_names': input_names, 'raw_observation_names': INPUT_NAMES, 'output_names': OUTPUT_NAMES,
                'action_expansion_indices': [0, 1, 1, 0, 2, 3, 4, 5, 6, 7, 8],
                'test_set': 'none; development train/validation only',
                'closed_loop_evaluation': 'not run; offline errors do not establish robot success'}
    if preprocessing_from is not None and not mapped:
        raise ValueError('Frozen normalizers require the explicit mapped training path')
    metadata.update(memory_mode='mapped' if mapped else 'ram',
                    preprocessing_from=str(preprocessing_from) if preprocessing_from else None,
                    resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    metadata['backend']=config.get('backend','sklearn')
    if metadata['backend']=='pytorch':
        metadata['versions']['torch']=importlib.metadata.version('torch')
    write_json(output / 'run.json', metadata)
    started = time.perf_counter()
    try:
        if mapped:
            from .workspace import mapped_training
            x,y,x_scaler,y_scaler,workspace_info=mapped_training(dataset,split['training'],output,preprocessing_from,
                window=config.get('observation_window'))
        else:
            training_data = [load_episode(dataset, r)[:2] for r in split['training']]
            x = np.concatenate([pair[0] for pair in training_data])
            y = np.concatenate([pair[1] for pair in training_data]); del training_data
            x_scaler = StandardScaler().fit(x)
            y_scaler = StandardScaler().fit(y)
            workspace_info={'mode':'ram','normalization':'StandardScaler.fit on training only'}
        checkpoint_context={'training_inventory':inventory_fingerprint(split['training']),
                            'validation_inventory':inventory_fingerprint(split['validation']),
                            'versions':metadata['versions'],
                            'implementation':{p:d for p,d in source_hashes.items()
                                if Path(p).name in ('fitting.py','torch_fitting.py','workspace.py','model.py','metrics.py','windows.py','dataset.py')}}
        boundaries = np.r_[0, np.cumsum([r['rows'] for r in split['training']])]
        training_ranges = list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))
        policy, fit_info = fit_estimator(x, y, x_scaler, y_scaler, config,
                                         dataset, split['validation'], output,mapped=mapped,
                                         checkpoint_context=checkpoint_context,resume_checkpoint=resume_checkpoint,
                                         training_ranges=training_ranges)
        policy.save(output / 'model.npz')
        restored = type(policy).load(output / 'model.npz')
        np.testing.assert_array_equal(restored.predict(x[:128]), policy.predict(x[:128]))
        model_info = {**fit_info,
                      'train_rows': len(x), 'input_features': len(input_names), 'independent_outputs': 9,
                      'preprocessing_values': 60, 'serialized_model_bytes': (output / 'model.npz').stat().st_size,
                      'constant_training_inputs': [INPUT_NAMES[i] for i in np.flatnonzero(x_scaler.var_ == 0)],
                      'constant_training_outputs': [OUTPUT_NAMES[i] for i in np.flatnonzero(y_scaler.var_ == 0)],
                      'constant_scale_rule': 'StandardScaler uses scale=1 for constant features',
                      'numeric_threads': threadpool_info(),
                      'training_weighting': 'uniform transitions; long episodes contribute more to fitting',
                      'primary_evaluation': 'equal episodes; within each, 0.5 mean wheel NRMSE + 0.5 mean arm NRMSE; scales from training only',
                      'output_limits': 'none applied; physical-unit predictions may exceed expert limits'}
        model_info['training_workspace']=workspace_info
        model_info['observation_window']=config.get('observation_window')
        if config.get('observation_window') is not None:
            workspace_info['window_storage']='21D observations only; causal windows gathered per minibatch'
            model_info['window_normalization']='Frozen current-state training scaler reused at every lag; goal normalized once'
        evaluator=evaluate_streaming if mapped else evaluate
        metrics = {name: evaluator(dataset, group, policy) for name, group in split.items()}
        observation = x[0].copy(); del x, y
        if hasattr(policy,'reset'):policy.reset()
        for _ in range(100):
            policy.predict_action(observation)
        timings = []
        for _ in range(2000):
            start = time.perf_counter_ns(); policy.predict_action(observation)
            timings.append(time.perf_counter_ns() - start)
        model_info['inference_single_observation_us'] = {
            'median': float(np.median(timings) / 1000), 'p95': float(np.percentile(timings, 95) / 1000),
            'samples': len(timings), 'includes': 'history update if configured, input standardization, network prediction, inverse scaling and 9-to-11 expansion; no simulator or state reading'}
        if hasattr(policy,'reset'):policy.reset()
        if any(sha256(PROJECT_ROOT / p) != digest for p, digest in source_hashes.items()):
            raise ValueError("Training source changed during execution")
        if inventory(dataset) != records:
            raise ValueError("Dataset changed during execution")
        if any(sha256(dataset / p) != digest for p, digest in metadata['dataset_hashes'].items()):
            raise ValueError("Dataset metadata changed during execution")
        write_json(output / 'model_info.json', model_info)
        write_json(output / 'metrics.json', metrics)
        write_channel_report(output / 'per_output_metrics.csv', metrics)
        metadata.update(status='complete', finished_at=utc_now(), wall_s=time.perf_counter() - started,
                        model_sha256=sha256(output / 'model.npz'), split_sha256=sha256(output / 'split.json'),
                        metrics_sha256=sha256(output / 'metrics.json'),
                        per_output_metrics_sha256=sha256(output / 'per_output_metrics.csv'))
        write_json(output / 'run.json', metadata)
        if mapped:shutil.rmtree(output/'work')
        return policy, metrics
    except BaseException as exc:
        metadata.update(status='failed', finished_at=utc_now(), error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'run.json', metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='One complete dataset directory')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--split', type=Path, help='Explicit frozen split manifest for nested learning curves')
    parser.add_argument('--output-dir', type=Path, help='New run directory; must not exist')
    parser.add_argument('--mapped',action='store_true',help='Disk-backed MLP training and streaming final metrics')
    parser.add_argument('--preprocessing-from',type=Path,help='Reuse exact normalizers from an identical frozen training inventory')
    parser.add_argument('--resume-checkpoint',type=Path,help='Hash-verified optimizer checkpoint from the same backend; same data/config except epoch cap')
    args = parser.parse_args(argv)
    config = read_json(args.config)
    output = args.output_dir or PROJECT_ROOT / 'runs/training' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + config['model'] + '_' + uuid.uuid4().hex[:8])
    explicit_split = None
    if args.split:
        saved = read_json(args.split)
        if Path(saved['dataset']).resolve() != args.dataset.resolve():
            raise ValueError('Explicit split belongs to another dataset')
        explicit_split = {k: saved[k] for k in ['training', 'validation']}
    with threadpool_limits(limits=1):
        _, metrics = train(args.dataset, config, output, explicit_split=explicit_split,mapped=args.mapped or config.get('backend')=='pytorch',
                           preprocessing_from=args.preprocessing_from,resume_checkpoint=args.resume_checkpoint)
    print(f'Run: {output.resolve()}')
    for split, result in metrics.items():
        model = result['model']['episode_macro']['balanced_nrmse']
        baseline = result['mean_baseline']['episode_macro']['balanced_nrmse']
        print(f'{split}: episode-macro balanced NRMSE={model:.6f}; mean baseline={baseline:.6f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
