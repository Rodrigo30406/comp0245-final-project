"""Compare the frozen uniform pilot with a spatially balanced 64-target pilot."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json, write_json, sha256, utc_now
from robot_learning.data.sampling import stratified_pilot
from .dataset import inventory, load_episode
from .model import LinearPolicy, MLPPolicy
from .metrics import errors, aggregate
from .fidelity import relative_command_fidelity
from .scaling import command


def offline_comparison(runs, banks, output):
    reference = LinearPolicy.load(runs['linear_uniform64']/'model.npz')
    data = {name: [load_episode(split['dataset'], r)[:2] for r in split['validation']]
            for name, split in banks.items()}
    data['common26'] = [pair for name in banks for pair in data[name]]
    results = {}
    for label, run in runs.items():
        info = read_json(run/'model_info.json'); manifest = read_json(run/'run.json')
        if (manifest['status'] != 'complete' or sha256(run/'model.npz') != manifest['model_sha256']
                or sha256(run/'split.json') != manifest['split_sha256']):
            raise ValueError('Frozen model/split changed')
        cls = LinearPolicy if info['kind'] == 'ordinary_least_squares' else MLPPolicy
        model = cls.load(run/'model.npz'); by_bank = {}
        for bank, episodes in data.items():
            truth = []; predictions = []; metrics = []
            for x, y in episodes:
                p = model.predict(x); truth.append(y); predictions.append(p)
                metrics.append(errors(y, p, reference.y_scale))
            by_bank[bank] = {'episodes': len(episodes), 'errors': aggregate(metrics),
                'relative_and_stop_pooled': relative_command_fidelity(
                    np.concatenate(truth), np.concatenate(predictions), reference.y_scale)}
        results[label] = {'model_sha256': manifest['model_sha256'], 'model_info': info,
                         'native_metrics': read_json(run/'metrics.json'), 'banks': by_bank}
    write_json(output/'offline_comparison.json', {
        'scope': 'Same two held-out banks for every model, and their 26-condition union',
        'normalization': 'Fixed original uniform64 training output standard deviations',
        'primary_fidelity': '5% of absolute nonzero reference; zeros separate, no absolute floor',
        'results': results})


def run(output, workers):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    recipe = read_json(PROJECT_ROOT/'experiment_specs/collection/pilot_box.json')
    plan = stratified_pilot(recipe['target_box'], 20261026)
    write_json(output/'sampling_plan.json', plan)
    recipe.update(name='stratified_box64_nominal', episodes=64, targets=plan['targets'])
    write_json(output/'collection.json', recipe)
    spec = {'counts': [64, 128, 256, 512], 'executed_pilot_count': 64,
            'training_episodes': 51, 'validation_episodes': 13,
            'sampling_seed': 20261026, 'physics_seed': recipe['seed'],
            'changes': 'target layout and geometry-only validation selection',
            'unchanged': 'expert, domain, initial state, MLP/OLS recipe, limits, horizon',
            'scope': 'One realization per sampling design; developmental comparison, not final test',
            'training_budget': 'Original max60/patience8/min_delta0.0001; report cap hits separately'}
    write_json(output/'protocol.json', spec)
    manifest = {'status': 'running', 'created_at': utc_now(), 'workers': workers}
    write_json(output/'run.json', manifest)
    try:
        print('Collecting 64 stratified episodes', flush=True)
        command(['scripts/collect_dataset.py', '--config', output/'collection.json',
                 '--workers', workers, '--output-dir', output/'datasets'], output/'logs/collection.log')
        candidates = list((output/'datasets').glob('*/dataset.json'))
        if len(candidates) != 1 or read_json(candidates[0])['status'] != 'complete':
            raise ValueError('Expected one complete stratified collection')
        dataset = candidates[0].parent; records = inventory(dataset)
        val_ids = {f'episode_{i:06d}' for i in plan['validation_indices']}
        split = {'dataset': str(dataset), 'training': [r for r in records if r['episode_id'] not in val_ids],
                 'validation': [r for r in records if r['episode_id'] in val_ids]}
        write_json(output/'split.json', split)
        manifest['dataset'] = str(dataset); write_json(output/'run.json', manifest)
        runs = {'linear_uniform64': PROJECT_ROOT/'runs/training/linear_pilot_20261025',
                'mlp_uniform64': PROJECT_ROOT/'runs/training/mlp_pilot_20261025'}
        for kind in ('linear', 'mlp'):
            label = f'{kind}_stratified64'; directory = output/'training'/label
            print(f'Training {label}', flush=True)
            command(['scripts/train_model.py', '--dataset', dataset, '--config',
                     PROJECT_ROOT/f'experiment_specs/training/{kind}_baseline.json',
                     '--split', output/'split.json', '--output-dir', directory], output/'logs'/f'{label}.log')
            runs[label] = directory
        bank_splits = {'uniform13': read_json(runs['linear_uniform64']/'split.json'),
                       'stratified13': read_json(runs['linear_stratified64']/'split.json')}
        # Reuse the exact guards used by autonomous evaluation for offline banks.
        from robot_learning.evaluation.cli import validate_common_validation
        for run_path in runs.values():
            model_split = read_json(run_path/'split.json')
            for bank in bank_splits.values():
                validate_common_validation(bank['dataset'], bank['validation'], model_split, allow_external=True)
        with threadpool_limits(limits=1): offline_comparison(runs, bank_splits, output)
        for bank, selected_run in [('uniform13', runs['linear_uniform64']), ('stratified13', runs['linear_stratified64'])]:
            print(f'Evaluating all four policies and expert on {bank}', flush=True)
            args = ['scripts/evaluate_policy.py', '--split', selected_run/'split.json',
                    '--allow-external-holdout', '--config', PROJECT_ROOT/'experiment_specs/evaluation/trajectory_diagnostic.json',
                    '--workers', workers, '--output-dir', output/'evaluation'/bank]
            for label, path in runs.items(): args.extend(['--model', f'{label}={path}'])
            command(args, output/'logs'/f'evaluation_{bank}.log')
        manifest.update(status='complete', finished_at=utc_now(), training_runs={k:str(v) for k,v in runs.items()})
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=utc_now())
        raise
    finally:
        write_json(output/'run.json', manifest)
    print(f'Completed sampling comparison: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(); run(args.output_dir, args.workers)
