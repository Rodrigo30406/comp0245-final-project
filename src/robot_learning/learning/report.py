"""Compare frozen policies' offline command fidelity without fitting new models."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import uuid
import numpy as np
from threadpoolctl import threadpool_limits
from robot_learning.settings import PROJECT_ROOT
from robot_learning.data.storage import read_json,write_json,sha256
from .dataset import load_episode
from .model import LinearPolicy,MLPPolicy,OUTPUT_NAMES
from .fidelity import command_fidelity,relative_command_fidelity,DEFAULT_TOLERANCES


def report(arguments,output):
    policies={};reference=None;source_hashes={}
    for argument in arguments:
        label,sep,path=argument.partition('=');run=Path(path).resolve()
        if not sep or not label or label=='mean_baseline' or label in policies:
            raise ValueError('Expected unique LABEL=RUN_DIRECTORY; mean_baseline reserved')
        manifest=read_json(run/'run.json');info=read_json(run/'model_info.json');split=read_json(run/'split.json')
        if (manifest['status']!='complete' or sha256(run/'model.npz')!=manifest['model_sha256']
            or sha256(run/'split.json')!=manifest['split_sha256']):
            raise ValueError('Model artifact changed or incomplete')
        if reference is None:reference=split
        if split!=reference:raise ValueError('Different source dataset or split')
        kind=info['kind']
        if kind not in ('ordinary_least_squares','mlp_regressor'):raise ValueError('Unsupported policy')
        cls=LinearPolicy if kind=='ordinary_least_squares' else MLPPolicy
        policies[label]=cls.load(run/'model.npz')
        source_hashes[label]={'run':str(run),'model_sha256':manifest['model_sha256'],'split_sha256':sha256(run/'split.json')}
    if not policies:raise ValueError('At least one frozen run required')
    first=next(iter(policies.values()))
    for policy in policies.values():
        for key in ['x_mean','x_scale','y_mean','y_scale']:
            np.testing.assert_array_equal(getattr(first,key),getattr(policy,key))
    truth=[];predictions={name:[] for name in [*policies,'mean_baseline']};episode_reports={name:[] for name in predictions}
    for record in reference['validation']:
        x,y,_=load_episode(reference['dataset'],record);truth.append(y)
        predicted={name:policy.predict(x) for name,policy in policies.items()}
        predicted['mean_baseline']=np.broadcast_to(first.y_mean,y.shape)
        for name,pred in predicted.items():
            predictions[name].append(pred)
            episode_reports[name].append({'episode_id':record['episode_id'],**command_fidelity(y,pred)})
    y=np.concatenate(truth);results={}
    for name,group in predictions.items():
        pooled=command_fidelity(y,np.concatenate(group))
        macro={key:{'unit':pooled['per_output'][key]['unit'],
                    'within_tolerance_fraction':np.mean([e['per_output'][key]['within_tolerance_fraction'] for e in episode_reports[name]],axis=0).tolist()}
               for key in OUTPUT_NAMES}
        results[name]={'pooled':pooled,
                       'relative_and_stop_pooled':relative_command_fidelity(y,np.concatenate(group),first.y_scale),'episode_macro_per_output':macro,
                       'episode_macro_all_nine_within_tolerance_fraction':np.mean([e['all_nine_within_tolerance_fraction'] for e in episode_reports[name]],axis=0).tolist(),
                       'per_episode':episode_reports[name]}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'command_fidelity.json',{'scope':'Post-hoc offline diagnostic, original validation expert states, unclipped predictions; not a stability metric',
        'tolerances':DEFAULT_TOLERANCES,'threshold_units':'rad/s per wheel side; N*m per torque; same numeric threshold across outputs',
        'no_safety_threshold_selected':True,'validation_episodes':len(reference['validation']),
        'validation_rows':len(y),'sources':source_hashes,'results':results})
    for path in [Path(__file__),Path(__file__).with_name('fidelity.py')]:
        (output/path.name).write_bytes(path.read_bytes())
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',required=True,help='LABEL=TRAINING_RUN_DIRECTORY')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    output=args.output_dir or PROJECT_ROOT/'runs/diagnostics'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_fidelity_'+uuid.uuid4().hex[:8])
    with threadpool_limits(limits=1):report(args.run,output)
    print(output.resolve());return 0


if __name__=='__main__':sys.exit(main())
