"""Evaluate frozen policies on paired held-out scenarios in autonomous simulation."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
from robot_learning.settings import PROJECT_ROOT, get_roboenv_root
from robot_learning.data.storage import read_json, write_json, sha256, provenance, snapshot_sources, utc_now
from robot_learning.data.scheduler import THREAD_VARIABLES, stop_processes, validate_workers
from robot_learning.learning.dataset import inventory, validate_partition
from robot_learning.simulation.config import resolve_episode_config, episode_defaults

DEFAULT_CONFIG = PROJECT_ROOT/'experiment_specs/evaluation/validation.json'


def validate_protocol(cfg):
    required={'name','horizon_s','dwell_s','tolerances_m','trace_every_steps','limits','failure_distance_penalty_m'}
    if not required <= set(cfg) or set(cfg)-required-{'dense_trace_until_s'}:
        raise ValueError('Unknown or missing evaluation configuration')
    for k in ['horizon_s','dwell_s','failure_distance_penalty_m']:
        if type(cfg[k]) not in (int,float) or not np.isfinite(cfg[k]) or cfg[k]<=0:
            raise ValueError(f'Invalid {k}')
    dense=cfg.get('dense_trace_until_s',0.)
    if type(dense) not in (float,int) or not np.isfinite(dense) or not 0<=dense<=cfg['horizon_s']:
        raise ValueError('Invalid dense_trace_until_s')
    if cfg['dwell_s']>cfg['horizon_s']:
        raise ValueError('Dwell exceeds horizon')
    if type(cfg['trace_every_steps']) is not int or cfg['trace_every_steps']<1:
        raise ValueError('Invalid trace interval')
    if not isinstance(cfg['tolerances_m'],list) or not cfg['tolerances_m']:
        raise ValueError('Expected nonempty tolerances')
    for t in cfg['tolerances_m']:
        if type(t) not in (float,int) or not np.isfinite(t) or t<=0:
            raise ValueError('Invalid tolerance')
    if sorted(set(cfg['tolerances_m']))!=cfg['tolerances_m']:
        raise ValueError('Tolerances must increase without duplicates')
    if cfg['limits']!='expert_command_envelope':
        raise ValueError('Unsupported actuator limit protocol')


def validate_common_validation(directory, selected, model_split, *, allow_external=False):
    """Cross-dataset comparison only on physical scenarios held out by every model."""
    model_directory = Path(model_split['dataset'])
    validate_partition(inventory(model_directory), {k:model_split[k] for k in ['training','validation']})
    def condition(root, record):
        meta = read_json(Path(root)/'episodes'/f"{record['episode_id']}.json")
        return json.dumps([record['target'], meta['actual_initial_observation'][:18]], sort_keys=True)
    selected_conditions = {condition(directory, r) for r in selected}
    training_conditions = {condition(model_directory, r) for r in model_split['training']}
    validation_conditions = {condition(model_directory, r) for r in model_split['validation']}
    if selected_conditions & training_conditions or (not allow_external and not selected_conditions <= validation_conditions):
        raise ValueError('Common evaluation scenarios must be held out by every model; training leakage rejected')


def prepare(split_path, model_arguments, protocol, output, limit=None, *, allow_cross_split=False, allow_external_holdout=False):
    validate_protocol(protocol)
    split_path=Path(split_path).resolve();split=read_json(split_path)
    dataset=Path(split['dataset']);records=inventory(dataset)
    lookup={r['episode_id']:r for r in records}
    selected=split['validation']
    if not selected or any(lookup.get(r['episode_id'])!=r for r in split['training']+selected):
        raise ValueError('Split differs from frozen source dataset')
    train_ids={r['episode_id'] for r in split['training']}
    val_ids={r['episode_id'] for r in selected}
    if train_ids & val_ids or len(val_ids)!=len(selected) or len(train_ids)+len(val_ids)!=len(records):
        raise ValueError('Overlapping, duplicated or incomplete split')
    if limit is not None:
        if type(limit) is not int or not 1<=limit<=len(selected):raise ValueError('Invalid scenario limit')
        selected=selected[:limit]
    scenarios={s['episode_id']:s for s in read_json(dataset/'scenarios.json')}
    source_cfg=read_json(dataset/'config.json')
    cfg=resolve_episode_config({k:v for k,v in source_cfg.items() if k in episode_defaults()})
    cfg['headless']=True
    policies={'expert':{'kind':'expert'}};model_sources={}
    for argument in model_arguments:
        label,separator,raw=argument.partition('=')
        if not separator or not label.replace('_','').isalnum() or label in policies:
            raise ValueError('Use unique LABEL=RUN_DIRECTORY; expert is reserved')
        run=Path(raw).resolve();manifest=read_json(run/'run.json');info=read_json(run/'model_info.json')
        if manifest['status']!='complete' or sha256(run/'model.npz')!=manifest['model_sha256']:
            raise ValueError('Model run incomplete or artifact changed')
        model_split = read_json(run/'split.json')
        if sha256(run/'split.json') != manifest['split_sha256']:
            raise ValueError('Model split artifact changed')
        model_dataset = Path(model_split['dataset'])
        if model_split != split:
            if not (allow_cross_split or allow_external_holdout):
                raise ValueError('Policies must share this exact source split')
            validate_common_validation(dataset, selected, model_split, allow_external=allow_external_holdout)
            model_cfg_source=read_json(model_dataset/'config.json')
            model_cfg=resolve_episode_config({k:v for k,v in model_cfg_source.items() if k in episode_defaults()})
            model_cfg['headless']=True
            if model_cfg!=cfg:
                raise ValueError('Different physical/expert configurations across learning-curve datasets')
            model_provenance=read_json(model_dataset/'provenance.json')
            source_provenance=read_json(dataset/'provenance.json')
            for name,digest in source_provenance['files'].items():
                if name.startswith('roboenv/') or name in ('project/src/robot_learning/expert/controllers.py',
                                                           'project/src/robot_learning/simulation/setup.py'):
                    if model_provenance['files'].get(name)!=digest:
                        raise ValueError('Different physics/assets/expert across model datasets')
        if any(sha256(model_dataset/name)!=digest for name,digest in manifest['dataset_hashes'].items()):
            raise ValueError('Training dataset metadata changed')
        if info['kind'] not in ('ordinary_least_squares','mlp_regressor'):
            raise ValueError('Unsupported policy')
        policies[label]={'kind':info['kind'],'artifact':f'models/{label}.npz',
                         'sha256':manifest['model_sha256'],'training_run':str(run)}
        model_sources[label]=run
    current=provenance(PROJECT_ROOT,Path(get_roboenv_root()))
    historical=read_json(dataset/'provenance.json')
    for name,digest in historical['files'].items():
        if name.startswith('roboenv/') or name in ('project/src/robot_learning/expert/controllers.py',
                                                  'project/src/robot_learning/simulation/setup.py'):
            if current['files'].get(name)!=digest:
                raise ValueError(f'Physics/assets/expert differ from source dataset: {name}')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ['episodes','logs','traces','models']:(output/name).mkdir()
    write_json(output/'protocol.json',protocol);write_json(output/'episode_config.json',cfg)
    write_json(output/'source_split.json',split);write_json(output/'policies.json',policies)
    write_json(output/'provenance.json',current)
    snapshot_sources(output,PROJECT_ROOT,Path(get_roboenv_root()))
    for label,run in model_sources.items():
        shutil.copy2(run/'model.npz',output/policies[label]['artifact'])
        (output/'models'/label).mkdir()
        for filename in ['run.json','model_info.json','config.json']:
            shutil.copy2(run/filename,output/'models'/label/filename)
    jobs=[]
    for r in selected:
        scenario=scenarios[r['episode_id']]
        metadata=read_json(dataset/'episodes'/f"{r['episode_id']}.json")
        for label in policies:
            jobs.append({'job_id':f"{label}_{r['episode_id']}",'policy':label,'scenario':scenario,
                         'initial_observation':metadata['actual_initial_observation']})
    write_json(output/'jobs.json',jobs)
    return jobs,current


def summarize(directory, jobs, protocol):
    results=[]
    for job in jobs:
        p=directory/'episodes'/f"{job['job_id']}.json"
        r=read_json(p) if p.exists() else {**job,'status':'missing','valid':False,'metrics':None}
        results.append(r)
    summary={}
    for policy in sorted({job['policy'] for job in jobs}):
        group=[r for r in results if r['policy']==policy];n=len(group)
        valid=[r for r in group if r['valid']]
        summary[policy]={'attempted':n,'valid_full_horizon':len(valid),'invalid_or_missing':n-len(valid),
            'success_by_tolerance':{str(t):sum(bool(r.get('metrics') and r['metrics']['by_tolerance'][str(t)]['success']) for r in group) for t in protocol['tolerances_m']},
            'mean_final_distance_valid_only_m':float(np.mean([r['metrics']['final_distance_m'] for r in valid])) if valid else None,
            'mean_penalized_final_distance_m':float(np.mean([r['metrics']['penalized_final_distance_m'] if r.get('metrics') else protocol['failure_distance_penalty_m'] for r in group])),
            'mean_penalized_time_average_distance_m':float(np.mean([r['metrics']['penalized_mean_distance_m'] if r.get('metrics') else protocol['failure_distance_penalty_m'] for r in group])),
            'mean_final_clipping_fraction_per_command':np.mean([r.get('clipping_fraction_per_command',[0]*11) for r in group],axis=0).tolist(),
            'arm_joint_limit_violation_steps':sum(r.get('arm_joint_limit_violation_steps',0) for r in group)}
    write_json(directory/'summary.json',summary)
    return summary


def verify_frozen_files(directory, hashes):
    for name, digest in hashes.items():
        if sha256(directory/name) != digest:
            raise ValueError(f'Frozen evaluation artifact changed: {name}')


def execute(directory,jobs,workers,source):
    active={};pending=deque(jobs);events=[];started=time.monotonic()
    frozen_names=['protocol.json','episode_config.json','source_split.json','policies.json','jobs.json','provenance.json']
    frozen_names += [str(p.relative_to(directory)) for p in (directory/'models').glob('*.npz')]
    frozen_hashes={name:sha256(directory/name) for name in frozen_names}
    manifest={'frozen_file_hashes':frozen_hashes,'status':'running','started_at':utc_now(),'workers':workers,'jobs':len(jobs),
              'source_fingerprint':source['source_fingerprint'],'events':events,
              'protocol_sha256':sha256(directory/'protocol.json'),
              'jobs_sha256':sha256(directory/'jobs.json')}
    write_json(directory/'run.json',manifest)
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(filter(None,[str(PROJECT_ROOT/'src'),env.get('PYTHONPATH')]))
    for variable in THREAD_VARIABLES:env.setdefault(variable,'1')
    env['ROBOENV_ROOT']=get_roboenv_root()
    try:
        while pending or active:
            while pending and len(active)<workers:
                verify_frozen_files(directory,frozen_hashes)
                job=pending.popleft();jid=job['job_id'];log=(directory/'logs'/f'{jid}.log').open('w')
                try:
                    process=subprocess.Popen([sys.executable,'-m','robot_learning.evaluation.worker',str(directory),jid],
                                             stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                except BaseException:
                    log.close();raise
                event={'job_id':jid,'pid':process.pid,'started_at':utc_now()};events.append(event)
                active[jid]=(process,log,job,event)
            for jid,(process,log,job,event) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();event.update(returncode=code,finished_at=utc_now());del active[jid]
                print(f"[{sum('returncode' in e for e in events)}/{len(jobs)}] {jid}: exit {code}",flush=True)
                write_json(directory/'run.json',manifest)
            if active:time.sleep(.02)
        verify_frozen_files(directory,frozen_hashes)
        if provenance(PROJECT_ROOT,Path(get_roboenv_root()))['source_fingerprint']!=source['source_fingerprint']:
            raise ValueError('Source/assets changed during evaluation; results invalid')
        manifest.update(status='complete',finished_at=utc_now(),wall_s=time.monotonic()-started)
    except BaseException as exc:
        stop_processes(active)
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}',finished_at=utc_now(),wall_s=time.monotonic()-started)
        raise
    finally:
        write_json(directory/'run.json',manifest)
        summarize(directory,jobs,read_json(directory/'protocol.json'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',type=Path,required=True,help='Frozen training split.json; evaluates validation only')
    parser.add_argument('--model',action='append',default=[],help='LABEL=TRAINING_RUN_DIRECTORY; expert always included')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--allow-cross-split',action='store_true',help='Compare different training sizes only on validation conditions held out by every model')
    parser.add_argument('--allow-external-holdout',action='store_true',help='Explicit independent target-bank evaluation; reject any physical condition used in model training')
    parser.add_argument('--limit',type=int,help='First N validation scenarios, for integration smoke checks')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv);validate_workers(args.workers)
    output=args.output_dir or PROJECT_ROOT/'runs/evaluation'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    jobs,source=prepare(args.split,args.model,read_json(args.config),output,args.limit,allow_cross_split=args.allow_cross_split,allow_external_holdout=args.allow_external_holdout)
    execute(output.resolve(),jobs,args.workers,source)
    print(f'Evaluation: {output.resolve()}')
    return 0


if __name__=='__main__':
    sys.exit(main())
