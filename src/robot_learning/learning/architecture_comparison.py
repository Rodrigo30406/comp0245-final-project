"""Depth/shape ablation with instantaneous and agreed 200ms measured inputs."""
import argparse
from pathlib import Path
import time
from robot_learning.data.storage import read_json,write_json,sha256,utc_now
from robot_learning.settings import PROJECT_ROOT
from .temporal_comparison import command,launch_frozen

ARCHITECTURES={'pyramid':[32,64,128],'expand_contract':[32,64,128,64,32]}


def parameter_count(inputs,hidden):
    widths=[inputs,*hidden,9]
    return sum(a*b+b for a,b in zip(widths[:-1],widths[1:]))


def campaign(reference_campaign,output,workers):
    previous=read_json(reference_campaign/'run.json')
    if previous['window_ms']!=200 or previous['spacing_ms']!=20:
        raise ValueError('Reuse only the agreed200ms/20ms reference campaign')
    reference=Path(previous['reference_run']);external=Path(previous['external_split'])
    split=read_json(reference/'split.json')
    if len(split['training'])!=819 or len(split['validation'])!=205:raise ValueError('Expected frozen819/205 split')
    baselines={label:previous['training_runs'][label] for label in ['instant128','window200ms']}
    for label,path in baselines.items():
        p=Path(path);r=read_json(p/'run.json')
        if (r['status']!='complete' or sha256(p/'model.npz')!=r['model_sha256'] or
                sha256(p/'split.json')!=r['split_sha256']):raise ValueError('Baseline changed or incomplete')
    for name in ['configs','logs','training','evaluation','reports']:(output/name).mkdir()
    variants={}
    for shape,hidden in ARCHITECTURES.items():
        for temporal in [False,True]:
            label=f'{"window200ms" if temporal else "instant"}_{shape}'
            cfg=read_json(PROJECT_ROOT/'experiment_specs/training/mlp_pytorch.json')
            cfg['name']=f'stratified1024_{label}';cfg['mlp']['hidden_layer_sizes']=hidden
            if temporal:cfg['observation_window']=dict(snapshots=11,stride_steps=20,dt_s=.001,padding='repeat_initial_state')
            write_json(output/'configs'/f'{label}.json',cfg)
            variants[label]=dict(hidden=hidden,input_features=201 if temporal else 21,
                parameters=parameter_count(201 if temporal else 21,hidden))
    state=dict(status='running',stage='training',created_at=utc_now(),reference_campaign=str(reference_campaign),
        reference_run=str(reference),external_split=str(external),comparison_baselines=baselines,baseline_evaluations={},
        training_runs={},evaluations={},variants=variants,workers=workers,
        controls='Same819/205 split, scalers,seed20261025,Adam,patience8/min_delta1e-4,no cap;200ms/20ms window unchanged',
        scope='One-seed development depth/shape screen, not pure depth isolation: widths and parameter budgets differ')
    write_json(output/'run.json',state)
    try:
        for label in variants:
            print(f'Training {label}: {variants[label]}',flush=True)
            target=output/'training'/label
            command(['scripts/train_model.py','--dataset',split['dataset'],'--split',reference/'split.json',
                '--mapped','--preprocessing-from',reference,'--config',output/'configs'/f'{label}.json','--output-dir',target],
                output/'logs'/f'train_{label}.log')
            manifest=read_json(target/'run.json');info=read_json(target/'model_info.json')
            if manifest['status']!='complete' or info['stop_reason']!='patience':raise ValueError('Fit must complete by patience')
            state['training_runs'][label]=str(target);write_json(output/'run.json',state)
        args=['scripts/report_command_fidelity.py','--output-dir',output/'reports/command_fidelity']
        for label,path in {**baselines,**state['training_runs']}.items():args+=['--run',f'{label}={path}']
        command(args,output/'logs/command_fidelity.log')
        state.update(stage='waiting_for_reference_evaluation');write_json(output/'run.json',state)
        print('Waiting for the temporal campaign to finish before starting additional physics workers',flush=True)
        while True:
            prior=read_json(reference_campaign/'run.json')
            if prior['status']=='complete':break
            if prior['status'] not in ['running']:raise ValueError('Reference campaign interrupted/failed; inspect before continuing')
            time.sleep(5)
        state.update(baseline_evaluations=prior['evaluations'],stage='evaluation');write_json(output/'run.json',state)
        for bank,path in [('own_validation',reference/'split.json'),('external_uniform_validation',external)]:
            target=output/'evaluation'/bank
            args=['scripts/evaluate_policy.py','--split',path,'--allow-external-holdout','--workers',workers,
                '--config',PROJECT_ROOT/'experiment_specs/evaluation/trajectory_diagnostic.json','--output-dir',target]
            for label,run in state['training_runs'].items():args+=['--model',f'{label}={run}']
            print(f'Evaluating {bank}:205goals,expert plus four new architectures',flush=True)
            command(args,output/'logs'/f'evaluate_{bank}.log')
            if read_json(target/'run.json')['status']!='complete':raise ValueError('Evaluation incomplete')
            state['evaluations'][bank]=str(target);write_json(output/'run.json',state)
            command(['scripts/report_subsystems.py','--evaluation-run',target,'--output-dir',output/'reports'/bank,'--no-plots'],
                    output/'logs'/f'report_{bank}.log')
        state.update(stage='reporting');write_json(output/'run.json',state)
        command(['scripts/report_temporal_windows.py','--campaign',output,'--output-dir',output/'reports/paired_control','--workers',4],
                output/'logs/paired_control.log')
        state.update(status='complete',stage='complete',finished_at=utc_now());write_json(output/'run.json',state)
        print(f'Depth/shape ablation complete: {output}',flush=True)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}',finished_at=utc_now())
        write_json(output/'run.json',state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    output=args.output_dir.resolve();reference=args.reference_campaign.resolve()
    if args.frozen:campaign(reference,output,args.workers)
    else:launch_frozen(output,'scripts/compare_mlp_architectures.py',[
        '--reference-campaign',reference,'--output-dir',output,'--workers',args.workers])


if __name__=='__main__':main()
