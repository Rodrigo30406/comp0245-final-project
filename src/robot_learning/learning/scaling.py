"""Run nested episode-count comparisons through the existing collection/training CLIs."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json,write_json,sha256,utc_now
from .dataset import inventory,nested_partition,load_episode
from .model import LinearPolicy,MLPPolicy
from .metrics import errors,aggregate


def command(arguments,log):
    with log.open('w') as handle:
        subprocess.run([sys.executable,*map(str,arguments)],cwd=PROJECT_ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)


def common_offline(runs,reference,output):
    split=read_json(reference/'split.json');scale=LinearPolicy.load(reference/'model.npz').y_scale
    results={}
    for label,path in runs.items():
        info=read_json(path/'model_info.json');cls=LinearPolicy if info['kind']=='ordinary_least_squares' else MLPPolicy
        policy=cls.load(path/'model.npz');episodes=[];mean=[]
        for record in split['validation']:
            x,y,_=load_episode(split['dataset'],record)
            episodes.append({'episode_id':record['episode_id'],**errors(y,policy.predict(x),scale)})
            mean.append(errors(y,np.broadcast_to(policy.y_mean,y.shape),scale))
        native=read_json(path/'metrics.json')
        results[label]={'training_run':str(path),'model_sha256':sha256(path/'model.npz'),
            'common_validation_fixed_scale':aggregate(episodes),'common_mean_predictor_fixed_scale':aggregate(mean),
            'native_training':native['training']['model'],'native_validation':native['validation']['model'],
            'per_common_episode':episodes,'model_info':info}
    write_json(output/'offline_comparison.json',{'common_split':str(reference/'split.json'),
        'normalization':'same original training standard deviations for every model; physical per-output metrics also retained',
        'results':results})
    return results


def plots(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for kind in ['linear','mlp']:
        labels=sorted((k for k in results if k.startswith(kind)),key=lambda x:int(x[len(kind):]))
        sizes=[int(k[len(kind):]) for k in labels]
        for ax,key,title in zip(axes,['balanced_nrmse','wheels_macro_rmse_rad_s','arm_macro_rmse_nm'],
            ['Balanced NRMSE (fixed 64-episode training scales)','Wheel RMSE (rad/s)','Mean joint-torque RMSE (Nm)']):
            ax.plot(sizes,[results[k]['common_validation_fixed_scale']['episode_macro'][key] for k in labels],marker='o',label=kind)
            ax.set(xlabel='Total collected episodes',title=title);ax.set_xticks(sizes);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Data scaling: identical 13 held-out scenarios, equal episode weights; one seed')
    fig.savefig(output/'offline_learning_curve.png',dpi=150);fig.savefig(output/'offline_learning_curve.pdf');plt.close(fig)


def run(spec_path,output,workers):
    spec=read_json(spec_path);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',spec);(output/'logs').mkdir();(output/'splits').mkdir()
    manifest={'status':'running','created_at':utc_now(),'protocol_sha256':sha256(output/'protocol.json'),
              'datasets':{},'training_runs':{}}
    write_json(output/'run.json',manifest)
    reference=PROJECT_ROOT/spec['reference_runs']['linear'];previous=read_json(reference/'split.json')
    runs={f'{kind}64':PROJECT_ROOT/path for kind,path in spec['reference_runs'].items()}
    previous_records=inventory(previous['dataset'])
    manifest['datasets']['64']=previous['dataset']
    try:
        # Each requested size is a real independent collection invocation. Overlapping
        # prefixes are repeated measurements, never concatenated as independent data.
        for n in spec['episode_counts'][1:]:
            print(f'Collecting {n} episodes with {workers} workers',flush=True)
            parent=output/'datasets'/str(n)
            command(['scripts/collect_dataset.py','--config',PROJECT_ROOT/spec['collection_recipe'],
                     '--episodes',n,'--workers',workers,'--output-dir',parent],output/'logs'/f'collect_{n}.log')
            candidates=list(parent.glob('*/dataset.json'))
            if len(candidates)!=1:raise ValueError('Expected exactly one new dataset')
            directory=candidates[0].parent
            if read_json(directory/'dataset.json')['status']!='complete':raise ValueError('Collection incomplete')
            records=inventory(directory);lookup={r['episode_id']:r for r in records}
            for old in previous_records:
                current=lookup[old['episode_id']]
                if any(current[k]!=old[k] for k in ['scenario_id','target','npz_sha256']):
                    raise ValueError('Repeated prefix differs from previous dataset; stop comparison')
            partition=nested_partition(records,previous,spec['validation_fraction'],spec['split_seed'])
            saved={'dataset':str(directory),'training':partition['training'],'validation':partition['validation']}
            write_json(output/'splits'/f'{n}.json',saved)
            manifest['datasets'][str(n)]=str(directory);write_json(output/'run.json',manifest)
            print(f'Collected {n}; train={len(partition["training"])}, validation={len(partition["validation"])}; prefix hashes identical',flush=True)
            previous=saved;previous_records=records
        for n in spec['episode_counts'][1:]:
            for kind,recipe in spec['training_recipes'].items():
                label=f'{kind}{n}';path=output/'training'/label
                print(f'Training {label}',flush=True)
                command(['scripts/train_model.py','--dataset',manifest['datasets'][str(n)],'--config',PROJECT_ROOT/recipe,
                         '--split',output/'splits'/f'{n}.json','--output-dir',path],output/'logs'/f'train_{label}.log')
                runs[label]=path;manifest['training_runs'][label]=str(path);write_json(output/'run.json',manifest)
                metric=read_json(path/'metrics.json')['validation']['model']['episode_macro']['balanced_nrmse']
                print(f'{label}: native validation NRMSE {metric:.6f}',flush=True)
        with threadpool_limits(limits=1):results=common_offline(runs,reference,output)
        plots(results,output)
        print('Evaluating all eight policies and expert on the same 13 held-out scenarios',flush=True)
        arguments=['scripts/evaluate_policy.py','--split',reference/'split.json','--allow-cross-split',
                   '--workers',workers,'--config',PROJECT_ROOT/spec['autonomous_protocol'],'--output-dir',output/'autonomous']
        for label,path in runs.items():arguments.extend(['--model',f'{label}={path}'])
        command(arguments,output/'logs'/'autonomous.log')
        manifest.update(status='complete',finished_at=utc_now())
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=utc_now());raise
    finally:write_json(output/'run.json',manifest)
    print(f'Completed comparison: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=PROJECT_ROOT/'experiment_specs/learning_curves/episode_counts.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();run(args.spec,args.output_dir,args.workers)
