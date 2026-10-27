"""Robust full-bank route distributions and exploratory goal-difficulty maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json,write_json,sha256,utc_now
from .control_quality import discrete_frechet,robust_summary
from .trajectory import compare_traces
from .size_report import write_csv
from .subsystems import wrapped_angle

METRICS=['frechet_xy_m','base_xy_time_rmse_m','final_base_goal_xy_m','final_ee_error_m',
         'path_length_ratio','excess_path_length_m','fullstep_wheel_clip_fraction',
         'sampled_normalized_clip_excess_wheels','sampled_SO3_rotation_travel_lower_bound_rad']


def process_pair(job):
    """Read one matched model/expert trajectory; preserve hashes and full horizon."""
    model_path=Path(job['model_record']);expert_path=Path(job['expert_record'])
    r=read_json(model_path);er=read_json(expert_path)
    row={k:v for k,v in job.items() if k not in ['expected_result_sha256','nearest_training_target_normalized']}
    if sha256(model_path)!=job['expected_result_sha256']:raise ValueError('Episode record changed')
    row.update(valid=bool(r['valid'] and er['valid']),task_success_1cm=bool(r.get('metrics') and r['metrics']['by_tolerance']['0.01']['success']),
               target_x_m=r['scenario']['target'][0],target_y_m=r['scenario']['target'][1],target_z_m=r['scenario']['target'][2])
    if not row['valid']:return row
    initial=np.asarray(r['actual_initial_observation']);target=np.asarray(r['scenario']['target']);goal=np.asarray(er['subsystems']['base_navigation']['goal_xy_m'])
    nav_delta=goal-initial[:2];direction=float(wrapped_angle(np.arctan2(nav_delta[1],nav_delta[0])-initial[3]))
    row.update(initial_target_radius_xy_m=float(np.linalg.norm(target[:2]-initial[:2])),
               initial_nav_distance_xy_m=float(np.linalg.norm(nav_delta)),initial_nav_direction_rad=direction,
               abs_initial_nav_turn_rad=abs(direction),nearest_training_target_normalized=job['nearest_training_target_normalized'])
    if not row['valid']:return row
    traces=[]
    for path,record in [(model_path,r),(expert_path,er)]:
        trace=path.parent.parent/'traces'/f'{path.stem}.npz'
        if sha256(trace)!=record['trace_sha256']:raise ValueError('Trace changed')
        with np.load(trace) as z:traces.append({k:z[k] for k in z.files})
    m,e=traces;np.testing.assert_array_equal(m['step'],e['step'])
    stride=round(.1/r['dt'])
    if abs(stride*r['dt']-.1)>1e-10:raise ValueError('100ms not representable at source control period')
    take=m['step']%stride==0
    if take.sum()!=401:raise ValueError('Expected complete40s/100ms geometry')
    compared,_=compare_traces(m,e,r['dt']);crossing=compared['first_observed_base_xy_crossings']['0.1']
    lm=r['subsystems']['base_navigation']['path_length_xy_m'];le=er['subsystems']['base_navigation']['path_length_xy_m']
    valid=m['has_action'];times=m['action_time'][valid];indices=[0,1,4,5,6,7,8,9,10]
    limit=np.asarray(r['limits'])[indices];excess=np.abs(m['raw_action'][valid][:,indices]-m['action'][valid][:,indices])/limit
    sampled=np.trapezoid(excess,times,axis=0)/(times[-1]-times[0])
    quat=m['base_orientation_xyzw'];quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
    angular=2*np.arccos(np.clip(np.abs(np.sum(quat[1:]*quat[:-1],axis=1)),0,1))
    row.update(frechet_xy_m=discrete_frechet(m['observation'][take,:2],e['observation'][take,:2]),
       base_xy_time_rmse_m=compared['base_xy_separation_m']['time_rmse'],
       yaw_time_rmse_rad=compared['base_yaw_separation_rad']['time_rmse'],
       final_base_goal_xy_m=r['subsystems']['base_navigation']['base_goal_distance_xy_m'],
       final_ee_error_m=r['metrics']['final_distance_m'],path_length_ratio=lm/le if le>1e-12 else None,
       model_path_length_m=lm,expert_path_length_m=le,excess_path_length_m=lm-le,
       fullstep_wheel_clip_fraction=float(np.mean(r['clipping_fraction_per_command'][:4])),
       sampled_normalized_clip_excess_wheels=float(sampled[:2].mean()),
       sampled_SO3_rotation_travel_lower_bound_rad=float(angular.sum()),
       first_observed_xy_10cm_departure_s=crossing['time_s'] if crossing else None,
       no_observed_10cm_departure=bool(crossing is None),max_trace_time_gap_s=float(np.diff(m['time']).max()))
    return row


def build_jobs(campaign,counts):
    import csv
    manifest=read_json(campaign/'run.json');assert manifest['status']=='complete'
    registry={}
    for r in csv.DictReader((campaign/'online_per_episode.csv').open()):
        n=int(r['dataset_episodes']);bank=r['evaluation_bank']
        if n in counts and bank in ['uniform_validation','stratified_validation']:
            key=(n,bank,r['scenario_id'],r['policy_label'])
            if key in registry:raise ValueError('Duplicate evaluation record')
            registry[key]=r['source_result']
    trace_sources=read_json(campaign/'full_trajectory_metrics.json');jobs=[]
    for n in counts:
        expected=2*int(np.ceil(.2*n))
        for design in ['uniform','stratified']:
            label=f'mlp_{design}{n}';split=read_json(Path(manifest['training_runs'][label])/'split.json')
            tree=cKDTree(np.array([r['target'] for r in split['training']])/[4.,4.,.5])
            selected=[(key,path) for key,path in registry.items() if key[0]==n and key[3]==label]
            if len(selected)!=expected:raise ValueError('Full paired validation coverage missing')
            for (n,bank,sid,_),path in selected:
                record=read_json(path);distance=float(tree.query(np.asarray(record['scenario']['target'])/[4.,4.,.5])[0])
                jobs.append({'episodes':n,'design':design,'policy':label,'bank':bank,'scenario_id':sid,
                    'model_record':path,'expert_record':registry[(n,bank,sid,'expert')],
                    'expected_result_sha256':trace_sources[f'{label}_{sid}']['result_sha256'],
                    'nearest_training_target_normalized':distance})
    return jobs


def subgroup_report(rows):
    # Geometric edges are fixed from the configured box before inspecting errors.
    specs={'initial_target_radius_xy_m':[0.,1.,2.,3.],
           'target_z_m':[.3,.425,.55,.675,.8],
           'initial_nav_direction_rad':np.linspace(-np.pi,np.pi,9).tolist()}
    groups=[];correlations=[]
    for n,design in sorted({(r['episodes'],r['design']) for r in rows}):
        selected=[r for r in rows if r['episodes']==n and r['design']==design and r['valid']]
        for feature,edges in specs.items():
            assigned=np.clip(np.digitize([r[feature] for r in selected],edges)-1,0,len(edges)-2)
            for b in range(len(edges)-1):
                members=[r for i,r in enumerate(selected) if assigned[i]==b]
                stats=robust_summary(r['frechet_xy_m'] for r in members)
                groups.append({'episodes':n,'design':design,'feature':feature,'lower':float(edges[b]),'upper':float(edges[b+1]),
                    'goals':len(members),'at_least20_goals':len(members)>=20,**stats,
                    'task_successes':sum(r['task_success_1cm'] for r in members)})
        for feature in ['initial_target_radius_xy_m','initial_nav_distance_xy_m','abs_initial_nav_turn_rad',
                        'target_z_m','nearest_training_target_normalized']:
            rho=float(spearmanr([r[feature] for r in selected],[r['frechet_xy_m'] for r in selected]).statistic)
            correlations.append({'episodes':n,'design':design,'feature':feature,'goals':len(selected),
                                 'spearman_rho':rho if np.isfinite(rho) else None,'scope':'Exploratory association, not causal importance'})
    return groups,correlations


def plots(rows,groups,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    colors={'uniform':'#2563eb','stratified':'#ea580c'};counts=sorted({r['episodes'] for r in rows})
    def save(fig,name):fig.savefig(output/f'{name}.png',dpi=150);fig.savefig(output/f'{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(len(counts),3,figsize=(16,4.5*len(counts)),squeeze=False,layout='constrained')
    for i,n in enumerate(counts):
        for ax,metric,title in zip(axes[i],['frechet_xy_m','base_xy_time_rmse_m','path_length_ratio'],['Frechet XY(m)','Timed XY RMSE(m)','Path ratio to expert']):
            for d in colors:
                a=np.sort([r[metric] for r in rows if r['episodes']==n and r['design']==d and r['valid'] and r.get(metric) is not None])
                ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=d,color=colors[d])
            ax.set(title=f'{n} episodes: {title}',xlabel=title,ylabel='Fraction of validation episodes',ylim=(0,1));ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Complete paired validation distributions; every episode retained; logarithmic display only')
    save(fig,'episode_error_cdfs')
    for feature,unit,name in [('initial_target_radius_xy_m','Initial target radius(m)','radius_groups'),
                              ('target_z_m','Target Z(m)','height_groups'),
                              ('initial_nav_direction_rad','Initial navigation direction(rad)','bearing_groups')]:
        fig,axes=plt.subplots(1,len(counts),figsize=(7*len(counts),4.8),squeeze=False,layout='constrained')
        for ax,n in zip(axes.flat,counts):
            for d in colors:
                selected=sorted([g for g in groups if g['episodes']==n and g['design']==d and g['feature']==feature and g['goals']],key=lambda x:x['lower'])
                x=np.array([.5*(g['lower']+g['upper']) for g in selected]);med=np.array([g['median'] for g in selected])
                ax.plot(x,med,marker='o',label=d,color=colors[d]);ax.fill_between(x,[g['q25'] for g in selected],[g['q75'] for g in selected],alpha=.15,color=colors[d])
                for a,b,g in zip(x,med,selected):ax.annotate(f'n={g["goals"]}',(a,b),xytext=(0,8 if d=='uniform' else -14),textcoords='offset points',fontsize=7,ha='center')
            ax.set(title=f'{n}: median and IQR',xlabel=unit,ylabel='Discrete Frechet XY(m)');ax.set_yscale('symlog',linthresh=.01);ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Exploratory geometric subgroups; fixed bins, all goals; association is not causation')
        save(fig,name)
    fig=plt.figure(figsize=(14,6*len(counts)),layout='constrained')
    maximum=max(r['frechet_xy_m'] for r in rows if r['valid']);norm=SymLogNorm(linthresh=.1,vmin=0,vmax=maximum)
    for i,n in enumerate(counts):
        for j,d in enumerate(colors):
            ax=fig.add_subplot(len(counts),2,i*2+j+1,projection='3d');a=[r for r in rows if r['episodes']==n and r['design']==d and r['valid']]
            scatter=ax.scatter([r['target_x_m'] for r in a],[r['target_y_m'] for r in a],[r['target_z_m'] for r in a],c=[r['frechet_xy_m'] for r in a],norm=norm,cmap='viridis',s=12,depthshade=False)
            ax.set(title=f'{d}{n}: all{len(a)}paired validation goals',xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)',xlim=(-2,2),ylim=(-2,2),zlim=(.3,.8));ax.set_box_aspect([1,1,1])
    fig.colorbar(scatter,ax=fig.axes,label='Frechet XY(m), shared symlog scale',shrink=.5)
    fig.suptitle('Where routes are difficult: goal position coloured by route error; no points removed')
    save(fig,'goal_difficulty_maps')


def report(campaign,output,counts,workers):
    campaign=Path(campaign).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    jobs=build_jobs(campaign,counts);write_json(output/'jobs.json',jobs)
    run={'status':'running','created_at_utc':utc_now(),'counts':counts,'workers':workers,'pairs':len(jobs),
         'scope':'Complete paired native holdouts; no training, simulation or expert changes'};write_json(output/'run.json',run)
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=[]
            for i,r in enumerate(pool.map(process_pair,jobs,chunksize=4),1):
                rows.append(r)
                if i%100==0:print(f'Analysed {i}/{len(jobs)}paired trajectories',flush=True)
        summaries=[]
        for n,design in sorted({(r['episodes'],r['design']) for r in rows}):
            selected=[r for r in rows if r['episodes']==n and r['design']==design]
            for metric in METRICS:
                summaries.append({'episodes':n,'design':design,'metric':metric,'validation_episodes':len(selected),'invalid_episodes':sum(not r['valid'] for r in selected),
                    'task_successes':sum(r['task_success_1cm'] for r in selected),**robust_summary(r.get(metric) for r in selected)})
        groups,correlations=subgroup_report(rows)
        write_csv(output/'per_episode.csv',rows);write_csv(output/'robust_summary.csv',summaries)
        write_csv(output/'subgroups.csv',groups);write_csv(output/'geometry_associations.csv',correlations)
        write_json(output/'metrics.json',{'scope':run['scope'],'counts':counts,'summary':summaries,'groups':groups,'correlations':correlations,
          'aggregation':'Each episode equal; median/IQR for typical behavior, P90/P95/max and counts for tail; no clipping/removal of metric values',
          'sampling':'Frechet uses401points at100ms; timed errors use time weighting; fullstep path/clip counters; rotation lower bound and clip severity sampled only',
          'pattern_limits':'Fixed geometry bins and minimum20goals for interpretation; exploratory associations, not causal findings or training-seed uncertainty'})
        plots(rows,groups,output)
        for module in [Path(__file__),Path(__file__).with_name('control_quality.py')]:shutil.copy2(module,output/module.name)
        run.update(status='complete',finished_at_utc=utc_now(),per_episode_sha256=sha256(output/'per_episode.csv'));write_json(output/'run.json',run)
    except BaseException as exc:
        run.update(status='failed',error=f'{type(exc).__name__}: {exc}');write_json(output/'run.json',run);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=PROJECT_ROOT/'runs/sampling_comparison/patience_series_20261026')
    parser.add_argument('--counts',type=int,nargs='+',default=[1024,2048]);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    if args.workers<1 or not set(args.counts)<={512,1024,2048}:raise ValueError('Positive workers and completed extension sizes512/1024/2048 required')
    with threadpool_limits(limits=1):report(args.campaign,args.output_dir,args.counts,args.workers)
