"""Exact full-bank offline metrics without retaining all observation matrices."""
import shutil
import numpy as np
from robot_learning.data.storage import read_json,write_json,sha256
from robot_learning.evaluation.cli import validate_common_validation
from .dataset import load_episode
from .model import MLPPolicy,LinearPolicy,OUTPUT_NAMES
from .metrics import errors,aggregate
from .fidelity import command_fidelity,relative_command_fidelity
from .sampling_series import checked_run
from robot_learning.settings import PROJECT_ROOT


def bank_metrics(selected,model,scale,workspace):
    workspace.mkdir();rows=sum(r['rows'] for _,r in selected)
    truth=np.lib.format.open_memmap(workspace/'truth.npy',mode='w+',dtype='float64',shape=(rows,9))
    prediction=np.lib.format.open_memmap(workspace/'prediction.npy',mode='w+',dtype='float64',shape=(rows,9))
    episodes=[];means=[];offset=0
    for dataset,record in selected:
        x,y,_=load_episode(dataset,record);p=model.predict(x);end=offset+len(y)
        truth[offset:end]=y;prediction[offset:end]=p;offset=end
        episodes.append({'dataset':dataset,'episode_id':record['episode_id'],**errors(y,p,scale)})
        means.append(errors(y,np.broadcast_to(model.y_mean,y.shape),scale))
    truth.flush();prediction.flush()
    fidelity=relative_command_fidelity(truth,prediction,scale)
    tails={}
    for j,name in enumerate(OUTPUT_NAMES):
        y=np.asarray(truth[:,j]);p=np.asarray(prediction[:,j]);active=np.abs(y)>1e-6
        delta=np.abs(p-y);failed=active&(delta>.05*np.abs(y));a=delta[failed];r=a/np.abs(y[failed])
        tails[name]={'nonzero_rows':int(active.sum()),'outside_5pct_rows':int(failed.sum()),
            'absolute_error_mean':float(a.mean()) if len(a) else None,
            'absolute_error_median':float(np.median(a)) if len(a) else None,
            'absolute_error_p95':float(np.percentile(a,95)) if len(a) else None,
            'relative_error_mean':float(r.mean()) if len(r) else None,
            'relative_error_median':float(np.median(r)) if len(r) else None,
            'relative_error_p95':float(np.percentile(r,95)) if len(r) else None}
    result={'episodes':len(selected),'rows':rows,'errors':aggregate(episodes),
            'mean_baseline':aggregate(means),'relative_and_stop_pooled':fidelity,
            'absolute_fidelity_pooled':command_fidelity(truth,prediction),
            'outside_5pct_tails':tails,'per_episode':episodes}
    del truth,prediction;shutil.rmtree(workspace)
    return result


def offline_count(n,runs,native_splits,anchors,output):
    scale=LinearPolicy.load(PROJECT_ROOT/'runs/training/linear_pilot_20261025/model.npz').y_scale
    splits={**native_splits,**anchors}
    banks={name:[(split['dataset'],r) for r in split['validation']] for name,split in splits.items()}
    banks['native_union']=banks['uniform_validation']+banks['stratified_validation']
    banks['anchor26']=banks['anchor_uniform13']+banks['anchor_stratified13']
    result={}
    for label,directory in runs.items():
        checked_run(directory);model_split=read_json(directory/'split.json')
        for split in splits.values():validate_common_validation(split['dataset'],split['validation'],model_split,allow_external=True)
        policy=MLPPolicy.load(directory/'model.npz');metrics={}
        for name,selected in banks.items():
            print(f'Offline {label} {name}: {len(selected)} episodes',flush=True)
            metrics[name]=bank_metrics(selected,policy,scale,output/f'work_{label}_{name}')
        result[label]={'model_sha256':sha256(directory/'model.npz'),'model_info':read_json(directory/'model_info.json'),
                       'native_training_metrics':read_json(directory/'metrics.json'),'banks':metrics}
        write_json(output/f'offline_{n}.json',{'normalization':'Fixed original uniform64 training output std',
            'scope':'Full native banks and same26 anchor; exact disk-backed pooled metrics; development only',
            'results':result})
