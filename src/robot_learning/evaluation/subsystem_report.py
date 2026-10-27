"""Base and whole-system evaluation with paired trajectory plots; no arm-only score."""
import argparse
from pathlib import Path
import sys
import numpy as np
from robot_learning.data.storage import read_json,write_json,sha256
from .trajectory import compare_traces,saturation_diagnostics,POSITION_THRESHOLDS_M,YAW_THRESHOLDS_DEG


def load_evaluation(directory):
    directory=Path(directory).resolve();run=read_json(directory/'run.json')
    if run['status']!='complete':raise ValueError('Need a complete evaluation run')
    grouped={};hashes={}
    for job in read_json(directory/'jobs.json'):
        path=directory/'episodes'/f"{job['job_id']}.json";record=read_json(path)
        if not record['valid'] or 'subsystems' not in record:
            raise ValueError('Trajectory report requires full valid traces; inspect summary/failures instead of silently filtering')
        trace_path=directory/'traces'/f"{job['job_id']}.npz"
        if sha256(trace_path)!=record['trace_sha256']:raise ValueError('Trace checksum mismatch')
        hashes[str(path.relative_to(directory))]=sha256(path)
        hashes[str(trace_path.relative_to(directory))]=record['trace_sha256']
        with np.load(trace_path,allow_pickle=False) as data:trace={k:data[k] for k in data.files}
        grouped.setdefault(job['scenario']['episode_id'],{})[job['policy']]=(record,trace)
    if not grouped:raise ValueError('Empty evaluation')
    names=set(next(iter(grouped.values())))
    if names=={'expert'} or 'expert' not in names or any(set(group)!=names for group in grouped.values()):
        raise ValueError('Need expert and identical policies for every scenario')
    return grouped,hashes


def report(directory,output,*,plots=True,zoom_s=2.):
    if not np.isfinite(zoom_s) or zoom_s<=0:raise ValueError('Invalid early-time zoom')
    directory=Path(directory).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    grouped,hashes=load_evaluation(directory);comparisons=[];curves={}
    for eid,group in grouped.items():
        reference=group['expert'][1];curves[eid]={}
        for label,(record,trace) in group.items():
            metrics,curve=compare_traces(trace,reference,record['dt']);curves[eid][label]=curve
            base=record['subsystems']['base_navigation'];terminal=record['subsystems']['terminal_motion']
            base_only={k:v for k,v in base.items() if k not in ('shoulder_target_distance_m','within_expert_reach_rule')}
            comparisons.append({'policy':label,'episode_id':eid,'scenario_id':record['scenario']['scenario_id'],
                'base_navigation':base_only,'system_task':record['metrics'],
                'stop_motion':{k:v for k,v in terminal.items() if not k.startswith('arm_')},
                'trajectory_vs_expert':metrics,
                'saturation':({**saturation_diagnostics(trace,record['limits'],record['dt']),
                    'full_step_clipping_fraction_per_command':record.get('clipping_fraction_per_command')}
                    if 'limits' in record else None)})
    summary={}
    for label in sorted(next(iter(grouped.values()))):
        rows=[r for r in comparisons if r['policy']==label]
        threshold_times={}
        for threshold in POSITION_THRESHOLDS_M:
            entries=[r['trajectory_vs_expert']['first_observed_base_xy_crossings'][str(threshold)] for r in rows]
            values=[e['time_s'] for e in entries if e is not None]
            threshold_times[str(threshold)]={'crossed_episodes':len(values),'not_observed_crossed':len(rows)-len(values),
                'median_first_observed_s_among_crossed':float(np.median(values)) if values else None}
        summary[label]={'episodes':len(rows),
            'base_stable_arrivals':sum(r['base_navigation']['stable_pose_arrival'] for r in rows),
            'system_stable_arrivals_1cm':sum(r['system_task']['by_tolerance']['0.01']['success'] for r in rows),
            'base_final_goal_distance_xy_median_m':float(np.median([r['base_navigation']['base_goal_distance_xy_m'] for r in rows])),
            'system_final_distance_median_m':float(np.median([r['system_task']['final_distance_m'] for r in rows])),
            'base_time_weighted_xy_rmse_median_m':float(np.median([r['trajectory_vs_expert']['base_xy_separation_m']['time_rmse'] for r in rows])),
            'system_time_average_distance_median_m':float(np.median([r['system_task']['mean_distance_m'] for r in rows])),
            'base_terminal_speed_xy_median_m_s':float(np.median([r['stop_motion']['base_mean_speed_xy_m_s'] for r in rows])),
            'base_departure_observations':threshold_times}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'trajectory_analysis.json',{'schema_version':3,'evaluation_run':str(directory),
        'source_run_sha256':sha256(directory/'run.json'),'source_result_trace_hashes':hashes,
        'scope':'Base and whole-system task metrics; time-aligned imitation diagnostics; no isolated arm score',
        'departure_thresholds':{'base_xy_m':POSITION_THRESHOLDS_M,'base_yaw_deg':YAW_THRESHOLDS_DEG,
                                'ee_trajectory_m':POSITION_THRESHOLDS_M,'purpose':'descriptive onset thresholds, not stability limits'},
        'time_weighting':'trajectory integrals use sample times so dense initial logging does not overweight the start',
        'summary':summary,'per_episode':comparisons})
    for filename in ['subsystem_report.py','trajectory.py','trajectory_plots.py']:
        (output/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    if plots:
        from .trajectory_plots import plot_case,plot_overview
        for eid,group in grouped.items():plot_case(output,eid,group,curves[eid],zoom_s)
        plot_overview(output,grouped,curves,summary,zoom_s)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--zoom-until-s',type=float,default=2.)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    print(report(args.evaluation_run,args.output_dir,plots=not args.no_plots,zoom_s=args.zoom_until_s));return 0


if __name__=='__main__':sys.exit(main())
