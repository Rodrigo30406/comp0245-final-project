"""Extend the frozen sampling study with patience-only MLPs and larger cohorts."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from threadpoolctl import threadpool_limits
import numpy as np
from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json,write_json,utc_now
from robot_learning.data.sampling import stratified_grid
from .dataset import inventory,nested_partition
from .sampling_series import checked_run
from .scaling import command

PREVIOUS=PROJECT_ROOT/'runs/sampling_comparison/full_series_20261026'
GRIDS={1024:[16,8,8],2048:[16,16,8]}


def run(output,workers):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ['logs','plans','configs','splits','offline']:(output/name).mkdir()
    old=read_json(PREVIOUS/'run.json')
    if old['status']!='complete':raise ValueError('Previous series is incomplete')
    runs={k:checked_run(v) for k,v in old['training_runs'].items() if k.startswith('mlp_')}
    for label,path in runs.items():
        if label!='mlp_uniform512' and read_json(path/'model_info.json')['stop_reason']!='patience':
            raise ValueError('Cannot reuse a capped MLP as patience-converged')
    state={'status':'running','created_at':utc_now(),'datasets':old['datasets'].copy(),
           'training_runs':{k:str(v) for k,v in runs.items()},'evaluations':{},'workers':workers,
           'historical_capped_uniform512':str(runs['mlp_uniform512']),'reused_patience_models':7}
    write_json(output/'run.json',state)
    write_json(output/'protocol.json',{'counts':[64,128,256,512,1024,2048],
        'grids':GRIDS,'new_splits':{'1024':[819,205],'2048':[1638,410]},
        'training_recipe':'experiment_specs/training/mlp_patience.json',
        'stopping':'No epoch cap; patience8, min_delta1e-4; restore minimum-validation checkpoint',
        'reference':str(PREVIOUS),'models':'MLP; older OLS retained only as historical reference',
        'invariants':'Same architecture/features/outputs/loss/Adam/expert/box and simulation limits',
        'uniform':'Verified deterministic prefixes; frozen512 split extended without moving old memberships',
        'stratified':'Independent jittered grids and geometry-only balanced holdouts, not nested',
        'evaluation':'Both full native banks plus identical26-goal anchor; all conditions excluded from training',
        'scope':'Development validation, one realization; no final test or fixed-update causal isolation'})
    children=[]
    def save():write_json(output/'run.json',state)
    def train_args(label,dataset,split,preprocessing=None):
        args=['scripts/train_model.py','--mapped','--dataset',dataset,'--split',split,
              '--config',PROJECT_ROOT/'experiment_specs/training/mlp_patience.json',
              '--output-dir',output/'training'/label]
        if preprocessing:args+=['--preprocessing-from',preprocessing]
        return args
    def register(label):
        path=checked_run(output/'training'/label)
        if read_json(path/'model_info.json')['stop_reason']!='patience':raise ValueError('MLP did not stop by patience')
        runs[label]=path;state['training_runs'][label]=str(path);save()
    def evaluate(name,split,labels):
        path=output/'evaluation'/name
        args=['scripts/evaluate_policy.py','--split',split,'--allow-external-holdout','--config',
              PROJECT_ROOT/'experiment_specs/evaluation/trajectory_diagnostic.json','--workers',workers,'--output-dir',path]
        for label in labels:args+=['--model',f'{label}={runs[label]}']
        print(f'Evaluating {name}',flush=True);command(args,output/'logs'/f'evaluate_{name}.log')
        if read_json(path/'run.json')['status']!='complete':raise ValueError('Incomplete evaluation')
        state['evaluations'][name]=str(path);save()
    try:
        base=read_json(PROJECT_ROOT/'experiment_specs/collection/pilot_box.json')
        # All stratified geometry is committed before collecting any new outcome.
        for n,grid in GRIDS.items():
            write_json(output/'plans'/f'stratified{n}.json',stratified_grid(base['target_box'],grid,20261026))
        old512=runs['mlp_uniform512'];prior=read_json(old512/'split.json')
        log=(output/'logs/train_mlp_uniform512.log').open('w')
        proc=subprocess.Popen([sys.executable,*map(str,train_args('mlp_uniform512',prior['dataset'],old512/'split.json',old512))],
                               cwd=PROJECT_ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append((proc,log));print('Training uniform512 beyond original cap alongside collection',flush=True)
        for design in ['uniform','stratified']:
            for n in GRIDS:
                name=f'{design}{n}';recipe={**base,'name':f'{name}_patience_series','episodes':n}
                if design=='stratified':recipe['targets']=read_json(output/'plans'/f'{name}.json')['targets']
                config=output/'configs'/f'{name}.json';write_json(config,recipe)
                print(f'Collecting {name}',flush=True)
                command(['scripts/collect_dataset.py','--config',config,'--workers',workers,
                         '--output-dir',output/'datasets'/name],output/'logs'/f'collect_{name}.log')
                found=list((output/'datasets'/name).glob('*/dataset.json'))
                if len(found)!=1 or read_json(found[0])['status']!='complete':raise ValueError('Incomplete dataset')
                dataset=found[0].parent;records=inventory(dataset)
                if design=='uniform':
                    lookup={r['episode_id']:r for r in records}
                    for record in prior['training']+prior['validation']:
                        if any(lookup[record['episode_id']][k]!=record[k] for k in ['scenario_id','target','npz_sha256']):
                            raise ValueError('Uniform prefix differs from frozen trajectories')
                    split={'dataset':str(dataset),**nested_partition(records,prior,.2,20261025)};prior=split
                else:
                    val={f'episode_{i:06d}' for i in read_json(output/'plans'/f'{name}.json')['validation_indices']}
                    split={'dataset':str(dataset),'training':[r for r in records if r['episode_id'] not in val],
                           'validation':[r for r in records if r['episode_id'] in val]}
                write_json(output/'splits'/f'{name}.json',split)
                state['datasets'][name]=str(dataset);save()
        proc.wait();log.close()
        if proc.returncode:raise RuntimeError('uniform512 training failed; inspect log')
        register('mlp_uniform512')
        # At most two large fits; collection has finished before launching them.
        for n in GRIDS:
            active=[]
            for design in ['uniform','stratified']:
                name=f'{design}{n}';label=f'mlp_{name}';log=(output/'logs'/f'train_{label}.log').open('w')
                proc=subprocess.Popen([sys.executable,*map(str,train_args(label,state['datasets'][name],output/'splits'/f'{name}.json'))],
                    cwd=PROJECT_ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((proc,log));active.append((label,proc,log));print(f'Training {label}',flush=True)
            for label,proc,log in active:
                proc.wait();log.close()
                if proc.returncode:raise RuntimeError(f'{label} failed; inspect log')
                register(label)
        state['training_status']='complete';save()
        # Paired offline analysis streams episodes into temporary disk-backed banks.
        from .streaming_comparison import offline_count
        anchors={f'anchor_{d}13':read_json(runs[f'mlp_{d}64']/'split.json') for d in ['uniform','stratified']}
        for n in [512,1024,2048]:
            selected={f'mlp_{d}{n}':runs[f'mlp_{d}{n}'] for d in ['uniform','stratified']}
            if n==512:selected['mlp_uniform512_capped']=old512
            native={f'{d}_validation':read_json(runs[f'mlp_{d}{n}']/'split.json') for d in ['uniform','stratified']}
            print(f'Offline paired comparison {n}',flush=True)
            with threadpool_limits(limits=1):offline_count(n,selected,native,anchors,output/'offline')
        for n in [512,1024,2048]:
            labels=[f'mlp_{d}{n}' for d in ['uniform','stratified']] if n!=512 else ['mlp_uniform512']
            for design in ['uniform','stratified']:
                evaluate(f'{design}_bank{n}',runs[f'mlp_{design}{n}']/'split.json',labels)
            evaluate(f'anchor_stratified13_count{n}',runs['mlp_stratified64']/'split.json',labels)
        state.update(status='complete',finished_at=utc_now());save()
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=utc_now());save();raise
    finally:
        for proc,log in children:
            if proc.poll() is None:
                import signal
                os.killpg(proc.pid,signal.SIGTERM);proc.wait()
            if not log.closed:log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);a=p.parse_args();run(a.output_dir,a.workers)
