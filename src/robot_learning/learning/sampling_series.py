"""Full power-of-two sampling comparison through the existing frozen-run CLIs."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json, write_json, sha256, utc_now
from robot_learning.data.sampling import stratified_grid
from robot_learning.evaluation.cli import validate_common_validation
from .dataset import inventory, nested_partition, load_episode
from .model import LinearPolicy, MLPPolicy
from .metrics import errors, aggregate
from .fidelity import command_fidelity, relative_command_fidelity
from .scaling import command

PILOT = PROJECT_ROOT/'runs/sampling_comparison/stratified64_20261026'
UNIFORM = PROJECT_ROOT/'runs/learning_curves/episode_counts_20261025'
NATIVE = PROJECT_ROOT/'runs/learning_curves/full_validation_20261026'
GRIDS = {64: [4,4,4], 128: [8,4,4], 256: [8,8,4], 512: [8,8,8]}


def checked_run(directory):
    directory = Path(directory)
    manifest = read_json(directory/'run.json')
    if (manifest['status'] != 'complete' or sha256(directory/'model.npz') != manifest['model_sha256']
            or sha256(directory/'split.json') != manifest['split_sha256']):
        raise ValueError(f'Incomplete or changed reference model: {directory}')
    return directory


def offline_count(n, runs, native_splits, anchors, output):
    scale = LinearPolicy.load(PROJECT_ROOT/'runs/training/linear_pilot_20261025/model.npz').y_scale
    splits = {**native_splits, **anchors}
    cache = {}
    keys = {}
    for bank, split in splits.items():
        keys[bank] = []
        for record in split['validation']:
            key = (split['dataset'], record['episode_id'])
            if key not in cache:
                cache[key] = load_episode(split['dataset'], record)[:2]
            keys[bank].append(key)
    keys['native_union'] = keys['uniform_validation']+keys['stratified_validation']
    keys['anchor26'] = keys['anchor_uniform13']+keys['anchor_stratified13']
    results = {}
    for label, directory in runs.items():
        checked_run(directory)
        model_split = read_json(directory/'split.json')
        for split in splits.values():
            validate_common_validation(split['dataset'], split['validation'], model_split, allow_external=True)
        info = read_json(directory/'model_info.json')
        cls = LinearPolicy if info['kind'] == 'ordinary_least_squares' else MLPPolicy
        model = cls.load(directory/'model.npz')
        predicted = {key: model.predict(x) for key, (x, _) in cache.items()}
        per_bank = {}
        for bank, selected in keys.items():
            truth = np.concatenate([cache[key][1] for key in selected])
            prediction = np.concatenate([predicted[key] for key in selected])
            episode_metrics = [errors(cache[key][1], predicted[key], scale) for key in selected]
            per_bank[bank] = {'episodes': len(selected), 'rows': len(truth),
                'errors': aggregate(episode_metrics),
                'relative_and_stop_pooled': relative_command_fidelity(truth, prediction, scale),
                'absolute_fidelity_pooled': command_fidelity(truth, prediction)}
        results[label] = {'model_sha256': sha256(directory/'model.npz'), 'model_info': info,
                         'native_training_metrics': read_json(directory/'metrics.json'), 'banks': per_bank}
    write_json(output/f'offline_{n}.json', {'normalization': 'Fixed original uniform64 training output std',
        'scope': 'Full native banks and identical 26-condition anchor, development only', 'results': results})


def run(output, workers):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    for sub in ['logs', 'plans', 'configs', 'splits', 'offline']:
        (output/sub).mkdir()
    pilot = read_json(PILOT/'run.json')
    if pilot['status'] != 'complete':
        raise ValueError('Need the completed 64-episode pilot')
    runs = {label: checked_run(path) for label, path in pilot['training_runs'].items()}
    for n in [128, 512]:
        for kind in ['linear', 'mlp']:
            runs[f'{kind}_uniform{n}'] = checked_run(UNIFORM/'training'/f'{kind}{n}')
    protocol = {'counts': list(GRIDS), 'grids': GRIDS, 'sampling_seed': 20261026,
        'validation_counts': [13,26,52,103], 'workers': workers,
        'training_recipe': 'Original OLS and MLP, max60/patience8; cap hits not convergence',
        'sampling': 'Independent jittered stratified designs; frozen pilot retained; geometry-only balanced holdout',
        'uniform256_split': 'Extend frozen uniform128 partition to 204/52; not assumed nested within frozen uniform512 split',
        'primary_comparison': 'Same original uniform13 plus stratified-pilot13 anchor, absent from every training set',
        'native_comparison': 'Both full native banks per size; targets differ between sizes',
        'scope': 'One realization per design and development validation, not final test or fixed-update causal isolation'}
    write_json(output/'protocol.json', protocol)
    manifest = {'status': 'running', 'created_at': utc_now(), 'workers': workers,
                'datasets': {'stratified64': pilot['dataset']},
                'training_runs': {k:str(v) for k,v in runs.items()}, 'evaluations': {}, 'completed_counts': [64]}
    for n in [64,128,512]:
        manifest['datasets'][f'uniform{n}'] = read_json(runs[f'linear_uniform{n}']/'split.json')['dataset']
    write_json(output/'run.json', manifest)
    def collect(name, recipe):
        write_json(output/'configs'/f'{name}.json', recipe)
        print(f'Collecting {name}', flush=True)
        command(['scripts/collect_dataset.py','--config',output/'configs'/f'{name}.json',
                 '--workers',workers,'--output-dir',output/'datasets'/name], output/'logs'/f'collect_{name}.log')
        found = list((output/'datasets'/name).glob('*/dataset.json'))
        if len(found) != 1 or read_json(found[0])['status'] != 'complete':
            raise ValueError('Expected exactly one complete new dataset')
        directory = found[0].parent
        manifest['datasets'][name] = str(directory); write_json(output/'run.json', manifest)
        return directory
    def train(name, dataset, split):
        write_json(output/'splits'/f'{name}.json', split)
        for kind in ['linear', 'mlp']:
            label = f'{kind}_{name}'; directory = output/'training'/label
            print(f'Training {label}', flush=True)
            command(['scripts/train_model.py','--dataset',dataset,'--config',
                     PROJECT_ROOT/f'experiment_specs/training/{kind}_baseline.json','--split',output/'splits'/f'{name}.json',
                     '--output-dir',directory], output/'logs'/f'train_{label}.log')
            runs[label] = checked_run(directory)
            manifest['training_runs'][label] = str(directory); write_json(output/'run.json', manifest)
    def evaluate(name, split_path, selected):
        directory = output/'evaluation'/name
        print(f'Evaluating {name}: {len(read_json(split_path)["validation"])} goals, {len(selected)} models plus expert', flush=True)
        args = ['scripts/evaluate_policy.py','--split',split_path,'--allow-external-holdout',
                '--config',PROJECT_ROOT/'experiment_specs/evaluation/trajectory_diagnostic.json',
                '--workers',workers,'--output-dir',directory]
        for label in selected: args.extend(['--model', f'{label}={runs[label]}'])
        command(args, output/'logs'/f'evaluate_{name}.log')
        if read_json(directory/'run.json')['status'] != 'complete':
            raise ValueError('Evaluation did not complete')
        manifest['evaluations'][name] = str(directory); write_json(output/'run.json', manifest)
    try:
        base = read_json(PROJECT_ROOT/'experiment_specs/collection/pilot_box.json')
        # A real corrected-size collection, never a relabelled historical run.
        uniform = collect('uniform256', {**base, 'name':'uniform256_corrected', 'episodes':256})
        records = inventory(uniform)
        prior = read_json(runs['linear_uniform128']/'split.json')
        lookup = {r['episode_id']: r for r in records}
        for old in prior['training']+prior['validation']:
            new = lookup[old['episode_id']]
            if any(new[k] != old[k] for k in ['scenario_id','target','npz_sha256']):
                raise ValueError('Uniform prefix no longer reproduces frozen data')
        partition = nested_partition(records, prior, .2, 20261025)
        uniform_split = {'dataset': str(uniform), **partition}
        # Freeze all larger geometry before observing their outcomes.
        for n in [128,256,512]:
            plan = stratified_grid(base['target_box'], GRIDS[n], 20261026)
            write_json(output/'plans'/f'stratified{n}.json', plan)
        for n in [128,256,512]:
            plan = read_json(output/'plans'/f'stratified{n}.json')
            dataset = collect(f'stratified{n}', {**base, 'name':f'stratified{n}_nominal',
                                              'episodes':n, 'targets':plan['targets']})
            records = inventory(dataset); val = {f'episode_{i:06d}' for i in plan['validation_indices']}
            split = {'dataset': str(dataset),
                     'training': [r for r in records if r['episode_id'] not in val],
                     'validation': [r for r in records if r['episode_id'] in val]}
            train(f'stratified{n}', dataset, split)
            if n == 256:
                train('uniform256', uniform, uniform_split)
            manifest['completed_counts'].append(n); write_json(output/'run.json', manifest)
        anchors = {'anchor_uniform13': read_json(runs['linear_uniform64']/'split.json'),
                   'anchor_stratified13': read_json(runs['linear_stratified64']/'split.json')}
        for n in [128,256,512]:
            selected = {label: runs[label] for kind in ['linear','mlp']
                        for label in [f'{kind}_uniform{n}', f'{kind}_stratified{n}']}
            native = {'uniform_validation': read_json(runs[f'linear_uniform{n}']/'split.json'),
                      'stratified_validation': read_json(runs[f'linear_stratified{n}']/'split.json')}
            print(f'Offline paired comparison {n}', flush=True)
            with threadpool_limits(limits=1): offline_count(n, selected, native, anchors, output/'offline')
        for n in [128,256,512]:
            labels = [f'{kind}_{design}{n}' for kind in ['linear','mlp'] for design in ['uniform','stratified']]
            # Existing uniform native outcomes are reused at 128/512. Expert is
            # still included by the evaluator, providing a repeat integrity check.
            uniform_models = labels if n == 256 else [f'{kind}_stratified{n}' for kind in ['linear','mlp']]
            evaluate(f'uniform_bank{n}', runs[f'linear_uniform{n}']/'split.json', uniform_models)
            evaluate(f'stratified_bank{n}', runs[f'linear_stratified{n}']/'split.json', labels)
            evaluate(f'anchor_stratified13_count{n}', runs['linear_stratified64']/'split.json', labels)
        manifest.update(status='complete', finished_at=utc_now())
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=utc_now())
        raise
    finally:
        write_json(output/'run.json', manifest)
    print(f'Completed full sampling series: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(); run(args.output_dir, args.workers)
