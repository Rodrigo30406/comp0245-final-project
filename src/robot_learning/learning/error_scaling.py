"""Compare supervised errors on full validation banks and a fixed anchor."""
import argparse
import csv
from pathlib import Path
import shutil

import numpy as np
from threadpoolctl import threadpool_limits

from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json, write_json, sha256, utc_now
from .dataset import load_episode
from .metrics import errors, aggregate
from .model import LinearPolicy, OUTPUT_NAMES
from .sampling_series import checked_run

COUNTS = [64, 128, 256, 512, 1024, 2048]
DESIGNS = ['uniform', 'stratified']


def improvement(previous, current):
    """Positive means error reduction; keep physical and relative changes separate."""
    return {'absolute_reduction': float(previous-current),
            'relative_reduction_percent': float(100*(previous-current)/previous) if previous else None}


def write_csv(path, rows):
    with Path(path).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def bank_description(bank):
    return {
        'anchor26': 'Same 26 goals across all sizes',
        'own_validation': 'Full own validation: 13 / 26 / 52 / 103 / 205 / 410 goals per model',
        'native_union': 'Full paired validation: 26 / 52 / 104 / 206 / 410 / 820 goals',
    }[bank]


def validation_context(splits, fixed):
    episode_errors=[];sources=[];absolute=np.zeros(9);rows=0
    for split in splits:
        for record in split['validation']:
            _, y, _ = load_episode(split['dataset'], record)
            episode_errors.append(errors(y,np.broadcast_to(fixed.y_mean,y.shape),fixed.y_scale))
            absolute+=np.abs(y).sum(axis=0);rows+=len(y)
            sources.append({'dataset':split['dataset'],'episode_id':record['episode_id'],'npz_sha256':record['npz_sha256']})
    return {'episode_errors':episode_errors,'rows':rows,'absolute_sum':absolute,'sources':sources}


def plot_report(groups, channels, output, bank):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'uniform': '#2563eb', 'stratified': '#ea580c'}

    def style(ax, title, ylabel):
        ax.set(title=title, xlabel='Collected episodes', ylabel=ylabel)
        ax.set_xscale('log', base=2); ax.set_xticks(COUNTS, labels=COUNTS, rotation=25)
        ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(fontsize=8)

    def save(fig, name):
        fig.savefig(output/f'{name}.png', dpi=160)
        fig.savefig(output/f'{name}.pdf'); plt.close(fig)

    for metric in ['rmse', 'mae']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        for ax, group, title, unit in zip(axes, ['base', 'arm'], ['Base wheel commands', 'Arm torque commands'], ['rad/s', 'Nm']):
            for design in DESIGNS:
                selected = [r for r in groups if r['design']==design and r['group']==group]
                ax.plot([r['episodes'] for r in selected], [r[metric] for r in selected], marker='o', color=colors[design], label=design)
            style(ax, title, f'Mean per-output {metric.upper()} ({unit})')
        fig.suptitle(bank_description(bank)+'\nEqual episode weights; CPU sklearn MLPs; expert-state command errors')
        save(fig, f'base_arm_{metric}')
    for metric in ['rmse', 'nrmse_train_std']:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout='constrained')
        for joint, ax in enumerate(axes.flat, 1):
            if joint == 8:
                ax.axis('off')
                ax.text(.02,.65,('Fixed 26-goal anchor' if bank=='anchor26' else 'Full validation at each size')+'\nEqual episode weighting\nLinear torque outputs\nOriginal 64 training scales\nNot an online position score\nPyTorch pilot kept separate',fontsize=11)
                continue
            for design in DESIGNS:
                selected = [r for r in channels if r['design']==design and r['output']==f'arm_torque_{joint}']
                ax.plot([r['episodes'] for r in selected], [r[metric] for r in selected], marker='o', label=design, color=colors[design])
            style(ax, f'Joint {joint}', 'RMSE (Nm)' if metric=='rmse' else 'NRMSE (fixed training std)')
        fig.suptitle('Arm: each joint separately; '+bank_description(bank)+'\nAverages do not establish convergence of every output')
        save(fig, f'joint_{metric}')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, name in zip(axes, ['wheel_right', 'wheel_left']):
        for design in DESIGNS:
            selected = [r for r in channels if r['design']==design and r['output']==name]
            ax.plot([r['episodes'] for r in selected], [r['rmse'] for r in selected], marker='o', label=design, color=colors[design])
        style(ax, name.replace('_', ' ').capitalize(), 'Command RMSE (rad/s)')
    fig.suptitle('Two independent wheel commands; '+bank_description(bank)+'\nDuplicated wheels are not counted twice')
    save(fig, 'wheel_rmse')


def report(series, reference_series, output, *, bank="native_union", contexts=None):
    series=Path(series).resolve(); reference_series=Path(reference_series).resolve()
    output=Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    reference=read_json(reference_series/'series_summary.json')
    manifest=read_json(series/'run.json')
    scaler_path=PROJECT_ROOT/'runs/training/linear_pilot_20261025/model.npz'
    fixed=LinearPolicy.load(scaler_path)
    if bank not in ('anchor26','own_validation','native_union'):
        raise ValueError('Unsupported validation bank')
    contexts={} if contexts is None else contexts
    def get_context(n,design):
        count=64 if bank=='anchor26' else n
        def one(d):
            key=(count,d)
            if key not in contexts:
                split=read_json(Path(manifest['training_runs'][f'mlp_{d}{count}'])/'split.json')
                contexts[key]=validation_context([split],fixed)
            return contexts[key]
        selected=[one(design)] if bank=='own_validation' else [one(d) for d in DESIGNS]
        return {'episode_errors':[e for c in selected for e in c['episode_errors']],
                'rows':sum(c['rows'] for c in selected),
                'absolute_sum':sum((c['absolute_sum'] for c in selected),start=np.zeros(9)),
                'sources':[r for c in selected for r in c['sources']]}
    mean_baselines={};profiles={};validation_sources={}
    groups=[]; channels=[]; sources=[]
    for n in COUNTS:
        source=(reference_series/'series_summary.json' if n<512 else series/'offline'/f'offline_{n}.json')
        results=reference['offline'][str(n)] if n<512 else read_json(source)['results']
        for design in DESIGNS:
            label=f'mlp_{design}{n}'; directory=checked_run(manifest['training_runs'][label])
            saved=results[label]
            selected_bank=f'{design}_validation' if bank=='own_validation' else bank
            selected=saved['banks'][selected_bank]
            expected=26 if bank=='anchor26' else int(np.ceil(.2*n))*(2 if bank=='native_union' else 1)
            assert selected['episodes']==expected
            context=get_context(n,design)
            assert len(context['episode_errors'])==expected
            mean_baseline=aggregate(context['episode_errors']);mean_baselines[label]=mean_baseline
            profile={name:{'reference_training_mean':float(fixed.y_mean[j]),
                'reference_training_std':float(fixed.y_scale[j]),
                'validation_mean_abs_command':float(context['absolute_sum'][j]/context['rows'])}
                for j,name in enumerate(OUTPUT_NAMES)}
            profiles[label]=profile;validation_sources[label]=context['sources']
            coverage={'evaluation_bank':selected_bank,'validation_episodes':expected,'validation_rows':context['rows']}
            assert saved['model_sha256']==sha256(directory/'model.npz')
            scores=selected['errors']['per_output_episode_macro']; macro=selected['errors']['episode_macro']
            for names,group,unit,key in [(OUTPUT_NAMES[:2],'base','rad/s','wheels'),(OUTPUT_NAMES[2:],'arm','N*m','arm')]:
                mae=float(np.mean([scores[name]['mae'] for name in names]));rmse=float(np.mean([scores[name]['rmse'] for name in names]))
                suffix='rad_s' if group=='base' else 'nm'
                np.testing.assert_allclose(rmse,macro[f'{key}_macro_rmse_{suffix}'],rtol=1e-12,atol=1e-12)
                groups.append({'episodes':n,'design':design,'group':group,'unit':unit,'mae':mae,'rmse':rmse,**coverage,
                    'mean_predictor_rmse':float(np.mean([mean_baseline['per_output_episode_macro'][name]['rmse'] for name in names]))})
            for name in OUTPUT_NAMES:
                score=scores[name];j=OUTPUT_NAMES.index(name)
                np.testing.assert_allclose(score['rmse']/fixed.y_scale[j],score['nrmse_train_std'],rtol=1e-12,atol=1e-12)
                f=selected['relative_and_stop_pooled']['per_output'][name]
                channels.append({'episodes':n,'design':design,'output':name,**coverage,**score,
                    **profile[name],'fixed_mean_predictor_rmse':mean_baseline['per_output_episode_macro'][name]['rmse'],
                    'within_5pct_nonzero_fraction':f['nonzero_relative_within_fraction'][0]})
            sources.append({'model':label,'model_run':str(directory),'model_sha256':saved['model_sha256'],
                            'source_report':str(source),'source_report_sha256':sha256(source),**coverage})
    changes=[]
    for design in DESIGNS:
        for previous,current in zip(COUNTS[:-1],COUNTS[1:]):
            for group in ['base','arm']:
                a=next(r for r in groups if r['design']==design and r['episodes']==previous and r['group']==group)
                b=next(r for r in groups if r['design']==design and r['episodes']==current and r['group']==group)
                changes.append({'design':design,'previous_episodes':previous,'episodes':current,'output':group,
                    'unit':b['unit'],'previous_rmse':a['rmse'],'rmse':b['rmse'],**improvement(a['rmse'],b['rmse'])})
            for name in OUTPUT_NAMES:
                a=next(r for r in channels if r['design']==design and r['episodes']==previous and r['output']==name)
                b=next(r for r in channels if r['design']==design and r['episodes']==current and r['output']==name)
                changes.append({'design':design,'previous_episodes':previous,'episodes':current,'output':name,
                    'unit':b['unit'],'previous_rmse':a['rmse'],'rmse':b['rmse'],**improvement(a['rmse'],b['rmse'])})
    result={'created_at_utc':utc_now(),'scope':bank_description(bank)+'; expert-state commands, CPU sklearn MLPs; not pose/online errors',
        'selected_bank':bank,'cross_size_conditions':'Fixed for anchor26; native conditions change across sizes',
        'averaging':'Average within-episode channel error, then equal episodes; groups average2wheel or7torque outputs',
        'reference_scale_model':str(scaler_path),'reference_scale_sha256':sha256(scaler_path),
        'validation_sources':validation_sources,'sources':sources,'fixed_mean_baseline_by_model':mean_baselines,
        'command_profile_by_model':profiles,'groups':groups,'channels':channels,'incremental_changes':changes,
        'limitations':'One realization per design; selected epochs/updates differ; no declared plateau tolerance or final test set'}
    write_json(output/'metrics.json',result)
    write_csv(output/'base_arm.csv',groups);write_csv(output/'per_output.csv',channels);write_csv(output/'incremental_changes.csv',changes)
    plot_report(groups,channels,output,bank)
    shutil.copy2(__file__,output/'source_report.py')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series-run',type=Path,default=PROJECT_ROOT/'runs/sampling_comparison/patience_series_20261026')
    parser.add_argument('--reference-series',type=Path,default=PROJECT_ROOT/'runs/sampling_comparison/full_series_20261026')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bank',choices=['native_union','own_validation','anchor26','all'],default='native_union',
                        help='Full paired union by default; own full holdout or fixed26 anchor also available')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.bank=='all':
            args.output_dir.mkdir(parents=True,exist_ok=False);contexts={}
            for bank in ['own_validation','native_union','anchor26']:
                print(f'Reporting {bank}',flush=True)
                report(args.series_run,args.reference_series,args.output_dir/bank,bank=bank,contexts=contexts)
        else:
            report(args.series_run,args.reference_series,args.output_dir,bank=args.bank)
    print(args.output_dir.resolve())
