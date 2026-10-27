"""Frozen-source, three-model temporal ablation on an existing 1024-episode split."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from robot_learning.data.storage import read_json, write_json, sha256, utc_now
from robot_learning.settings import PROJECT_ROOT, get_roboenv_root


def command(arguments, log):
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=PROJECT_ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed; inspect {log}')


def window_variants(window_ms, spacing_ms):
    """Explicit temporal design plus nearest two-layer instantaneous capacity."""
    if (type(window_ms) is not int or type(spacing_ms) is not int or
            window_ms < 1 or spacing_ms < 1 or window_ms % spacing_ms):
        raise ValueError('Window and spacing must be positive whole milliseconds, with exact divisibility')
    snapshots=window_ms//spacing_ms+1
    if snapshots < 2:raise ValueError('Window must include at least two snapshots')
    inputs=18*snapshots+3
    parameters=inputs*128+128+128*128+128+128*9+9
    capacity=round(math.sqrt(parameters+247)-16)
    return {'instant128':None,f'window{window_ms}ms':dict(snapshots=snapshots,stride_steps=spacing_ms,dt_s=.001,padding='repeat_initial_state'),
            f'instant{capacity}':None},capacity


def campaign(reference, external_split, output, workers, window_ms, spacing_ms, baseline_checkpoint=None):
    reference=reference.resolve();output=output.resolve()
    source=read_json(reference/'run.json');split=read_json(reference/'split.json')
    if (source['status']!='complete' or sha256(reference/'model.npz')!=source['model_sha256'] or
            sha256(reference/'split.json')!=source['split_sha256']):
        raise ValueError('Reference artifact changed or incomplete')
    if len(split['training'])!=819 or len(split['validation'])!=205:
        raise ValueError('This experiment requires the frozen 1024 split (819/205)')
    for folder in ['logs','configs','training','evaluation','reports']:(output/folder).mkdir()
    variants,capacity=window_variants(window_ms,spacing_ms)
    state=dict(status='running',created_at=utc_now(),reference_run=str(reference),external_split=str(external_split.resolve()),
        seed=20261025,workers=workers,training_runs={},evaluations={},window_ms=window_ms,spacing_ms=spacing_ms,
        baseline_checkpoint=str(baseline_checkpoint.resolve()) if baseline_checkpoint else None,
        scope='Development ablation; same stratified1024 data/split/scalers, PyTorch/CUDA, one seed; no new collection',
        controls='Same 1ms physics, expert rules, software clipping, 40s horizon and patience8; full own and paired-union validation')
    write_json(output/'run.json',state)
    try:
        for label,window in variants.items():
            cfg=read_json(PROJECT_ROOT/'experiment_specs/training/mlp_pytorch.json')
            cfg['name']=f'stratified1024_{label}_pytorch'
            if label==f'instant{capacity}':cfg['mlp']['hidden_layer_sizes']=[capacity,capacity]
            if window is not None:cfg['observation_window']=window
            config=output/'configs'/f'{label}.json';write_json(config,cfg)
            target=output/'training'/label
            print(f'Training {label}',flush=True)
            args=['scripts/train_model.py','--dataset',split['dataset'],'--split',reference/'split.json',
                '--mapped','--preprocessing-from',reference,'--config',config,'--output-dir',target]
            if label=='instant128' and baseline_checkpoint is not None:
                args+=['--resume-checkpoint',baseline_checkpoint.resolve()]
            command(args,output/'logs'/f'train_{label}.log')
            manifest=read_json(target/'run.json');info=read_json(target/'model_info.json')
            if manifest['status']!='complete' or info['stop_reason']!='patience':
                raise ValueError('Training did not finish by patience')
            state['training_runs'][label]=str(target);write_json(output/'run.json',state)
        print('Reporting full held-out command fidelity',flush=True)
        args=['scripts/report_command_fidelity.py','--output-dir',output/'reports/command_fidelity']
        for label,path in state['training_runs'].items():args+=['--run',f'{label}={path}']
        command(args,output/'logs/command_fidelity.log')
        for bank,path in [('own_validation',reference/'split.json'),('external_uniform_validation',external_split)]:
            target=output/'evaluation'/bank
            args=['scripts/evaluate_policy.py','--split',path,'--allow-external-holdout',
                  '--config',PROJECT_ROOT/'experiment_specs/evaluation/trajectory_diagnostic.json',
                  '--workers',workers,'--output-dir',target]
            for label,run in state['training_runs'].items():args+=['--model',f'{label}={run}']
            print(f'Evaluating {bank}: all205 goals, expert and three models',flush=True)
            command(args,output/'logs'/f'evaluate_{bank}.log')
            if read_json(target/'run.json')['status']!='complete':raise ValueError('Incomplete evaluation')
            state['evaluations'][bank]=str(target);write_json(output/'run.json',state)
            command(['scripts/report_subsystems.py','--evaluation-run',target,'--output-dir',output/'reports'/bank,'--no-plots'],
                    output/'logs'/f'report_{bank}.log')
        print('Reporting robust control curves on all410 paired goals',flush=True)
        command(['scripts/report_temporal_windows.py','--campaign',output,'--output-dir',output/'reports/paired_control','--workers',4],
                output/'logs/paired_control.log')
        state.update(status='complete',finished_at=utc_now());write_json(output/'run.json',state)
        print(f'Temporal ablation complete: {output}',flush=True)
    except BaseException as exc:
        state.update(status='failed',finished_at=utc_now(),error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run.json',state);raise


def launch_frozen(output, script, arguments):
    """Run an experiment from an immutable code copy shared by campaign launchers."""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    code=output/'frozen_project';code.mkdir()
    for folder in ['src','scripts','experiment_specs']:
        shutil.copytree(PROJECT_ROOT/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(PROJECT_ROOT/'pyproject.toml',code/'pyproject.toml')
    hashes={str(p.relative_to(code)):sha256(p) for p in code.rglob('*') if p.is_file()}
    write_json(output/'frozen_source.json',dict(created_at=utc_now(),source=str(PROJECT_ROOT),files=hashes))
    env=os.environ.copy();env['PYTHONPATH']=str(code/'src');env['ROBOENV_ROOT']=get_roboenv_root()
    from robot_learning.data.scheduler import THREAD_VARIABLES
    for name in THREAD_VARIABLES:env.setdefault(name,'1')
    result=subprocess.run([sys.executable,str(code/script),'--frozen',*map(str,arguments)],cwd=code,env=env)
    if result.returncode:raise SystemExit(result.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--external-split',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--window-ms',type=int,required=True,help='Explicit history duration in milliseconds')
    parser.add_argument('--spacing-ms',type=int,required=True,help='Explicit spacing between snapshots in milliseconds')
    parser.add_argument('--baseline-checkpoint',type=Path,help='Optional verified instantaneous128 optimizer checkpoint')
    parser.add_argument('--frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    window_variants(args.window_ms,args.spacing_ms)
    output=args.output_dir.resolve()
    if args.frozen:
        campaign(args.reference_run,args.external_split,output,args.workers,args.window_ms,args.spacing_ms,args.baseline_checkpoint)
        return
    command_args=['--reference-run',str(args.reference_run.resolve()),'--external-split',str(args.external_split.resolve()),
        '--output-dir',str(output),'--workers',str(args.workers),'--window-ms',str(args.window_ms),'--spacing-ms',str(args.spacing_ms)]
    if args.baseline_checkpoint:command_args+=['--baseline-checkpoint',str(args.baseline_checkpoint.resolve())]
    launch_frozen(output,'scripts/compare_temporal_windows.py',command_args)


if __name__=='__main__':main()
