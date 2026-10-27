"""Full-bank robust route distributions, optionally reusing frozen baselines."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from robot_learning.data.storage import read_json,write_json,sha256
from .control_pattern_report import process_pair, METRICS
from .control_quality import robust_summary
from .size_report import write_csv


def report(campaign, output, workers=4):
    campaign=Path(campaign).resolve();output=Path(output).resolve()
    state=read_json(campaign/'run.json');runs={**state.get('comparison_baselines',{}),**state['training_runs']}
    if not runs or len(state['evaluations'])!=2:raise ValueError('Need completed models and both evaluation banks')
    reference=read_json(Path(state['reference_run'])/'split.json')
    tree=cKDTree(np.array([r['target'] for r in reference['training']])/[4.,4.,.5])
    jobs=[];source_hashes={};seen=set()
    banks=list(state['evaluations'].items())+list(state.get('baseline_evaluations',{}).items())
    for bank,directory in banks:
        directory=Path(directory)
        if read_json(directory/'run.json')['status']!='complete':raise ValueError('Evaluation incomplete')
        scheduled=read_json(directory/'jobs.json')
        experts={j['scenario']['scenario_id']:directory/'episodes'/f"{j['job_id']}.json" for j in scheduled if j['policy']=='expert'}
        for j in scheduled:
            if j['policy'] not in runs:continue
            sid=j['scenario']['scenario_id'];path=directory/'episodes'/f"{j['job_id']}.json"
            key=(j['policy'],sid)
            if key in seen:raise ValueError('Duplicate scenario for a comparison policy')
            seen.add(key)
            digest=sha256(path);source_hashes[str(path)]=digest
            jobs.append(dict(episodes=1024,design=j['policy'],policy=j['policy'],bank=bank,scenario_id=sid,
                model_record=str(path),expert_record=str(experts[sid]),expected_result_sha256=digest,
                nearest_training_target_normalized=float(tree.query(np.asarray(j['scenario']['target'])/[4.,4.,.5])[0])))
    if len(jobs)!=410*len(runs):raise ValueError('Expected 410 goals for each comparison policy')
    goal_sets=[{j['scenario_id'] for j in jobs if j['policy']==label} for label in runs]
    if any(goals!=goal_sets[0] for goals in goal_sets):raise ValueError('Comparison policies must share exactly the same goals')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'jobs.json',jobs)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(process_pair,jobs,chunksize=4))
    summaries=[]
    for label in runs:
        selected=[r for r in rows if r['policy']==label]
        if len({r['scenario_id'] for r in selected})!=410:raise ValueError('Duplicate or missing paired goals')
        for metric in METRICS+['yaw_time_rmse_rad']:
            summaries.append(dict(policy=label,metric=metric,validation_episodes=len(selected),
                invalid_episodes=sum(not r['valid'] for r in selected),task_successes=sum(r['task_success_1cm'] for r in selected),
                **robust_summary(r.get(metric) for r in selected)))
    write_csv(output/'per_episode.csv',rows);write_csv(output/'robust_summary.csv',summaries)
    write_json(output/'metrics.json',dict(scope='Complete paired native union, 410 goals per policy; no metric outlier removal',
        summary=summaries,source_record_hashes=source_hashes,per_episode_sha256=sha256(output/'per_episode.csv')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette=['#2563eb','#ea580c','#059669','#dc2626','#9333ea','#92400e','#0891b2']
    colors={label:palette[i%len(palette)] for i,label in enumerate(runs)}
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    for ax,metric,title in zip(axes,['frechet_xy_m','base_xy_time_rmse_m','path_length_ratio'],
        ['Frechet XY (m)','Timed XY RMSE (m)','Path length / expert path length']):
        for label in runs:
            selected=[r for r in rows if r['policy']==label]
            values=np.sort([r[metric] for r in selected if r['valid'] and r.get(metric) is not None])
            ax.step(values,np.arange(1,len(values)+1)/len(selected),where='post',label=label,color=colors[label])
        ax.set(xlabel=title,ylabel='Fraction of all 410 evaluation goals',ylim=(0,1))
        ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Policy comparison: all held-out goals; lower error is better; invalid cases remain in denominator')
    fig.savefig(output/'control_cdfs.png',dpi=150);fig.savefig(output/'control_cdfs.pdf');plt.close(fig)
    shutil.copy2(__file__,output/Path(__file__).name)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    report(args.campaign,args.output_dir,args.workers)


if __name__=='__main__':main()
