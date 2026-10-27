"""Time-aligned trajectory diagnostics with explicit sampling and action alignment."""
import numpy as np
from .subsystems import wrapped_angle

POSITION_THRESHOLDS_M=(.001,.01,.1)
YAW_THRESHOLDS_DEG=(.1,1.,5.)


def validate_time(time):
    t=np.asarray(time,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Need finite strictly increasing sample times')
    return t


def time_weighted_errors(time,error):
    t=validate_time(time);e=np.asarray(error,dtype=float)
    if e.shape!=t.shape or not np.isfinite(e).all():raise ValueError('Invalid error curve')
    duration=t[-1]-t[0]
    return {'time_mean_absolute':float(np.trapezoid(np.abs(e),t)/duration),
            'time_rmse':float(np.sqrt(np.trapezoid(e**2,t)/duration)),
            'maximum':float(np.max(np.abs(e))),'final':float(e[-1])}


def first_observed_crossing(time,error,threshold):
    t=validate_time(time);e=np.asarray(error,dtype=float)
    if e.shape!=t.shape or not np.isfinite(e).all() or not np.isfinite(threshold) or threshold<0:
        raise ValueError('Invalid crossing inputs')
    indices=np.flatnonzero(e>=threshold)
    if not len(indices):return None
    i=int(indices[0]);previous=max(0,i-1)
    return {'sample_index':i,'time_s':float(t[i]),'previous_sample_time_s':float(t[previous]),
            'sample_gap_s':float(t[i]-t[previous]),'observed_error':float(e[i]),'threshold':float(threshold)}


def compare_traces(trace,reference,dt):
    np.testing.assert_array_equal(trace['step'],reference['step'])
    np.testing.assert_array_equal(trace['time'],reference['time'])
    t=validate_time(trace['time'])
    xy=np.linalg.norm(trace['observation'][:,:2]-reference['observation'][:,:2],axis=1)
    yaw=np.abs(wrapped_angle(trace['observation'][:,3]-reference['observation'][:,3]))
    ee=np.linalg.norm(trace['ee_position_world']-reference['ee_position_world'],axis=1)
    z=np.abs(trace['observation'][:,2]-reference['observation'][:,2])
    changes={}
    for name,cols in [('wheel_commands',slice(0,4)),('all_commands',slice(0,11))]:
        valid=trace.get('has_action',trace['step']>0)&reference.get('has_action',reference['step']>0)
        delta=np.max(np.abs(trace['raw_action'][:,cols]-reference['raw_action'][:,cols]),axis=1)
        indices=np.flatnonzero(valid&(delta>1e-9))
        if len(indices):
            i=int(indices[0]);action_time=trace['action_time'][i] if 'action_time' in trace else t[i]-dt
            changes[name]={'time_s':float(action_time),'trace_step':int(trace['step'][i]),
                           'maximum_absolute_numeric_difference':float(delta[i]),
                           'definition':'raw command discrepancy between separate live runs; same input only guaranteed initially'}
        else:changes[name]=None
    initial_indices=np.flatnonzero(trace.get('has_action',trace['step']>0))
    initial=None
    if len(initial_indices):
        i=int(initial_indices[0]);same=None
        if 'action_observation' in trace and 'action_observation' in reference:
            same=bool(np.array_equal(trace['action_observation'][i],reference['action_observation'][i]))
        initial={'time_s':float(trace['action_time'][i] if 'action_time' in trace else t[i]-dt),
                 'identical_input':same,'expert_raw_action':reference['raw_action'][i].tolist(),
                 'model_raw_action':trace['raw_action'][i].tolist(),
                 'raw_delta':(trace['raw_action'][i]-reference['raw_action'][i]).tolist(),
                 'expert_applied_action':reference['action'][i].tolist(),
                 'model_applied_action':trace['action'][i].tolist()}
    metrics={'duration_s':float(t[-1]-t[0]),'sample_count':len(t),
             'sampling_gaps_s':{'min':float(np.diff(t).min()),'max':float(np.diff(t).max())},
             'base_xy_separation_m':time_weighted_errors(t,xy),
             'base_yaw_separation_rad':time_weighted_errors(t,yaw),
             'base_z_separation_m':time_weighted_errors(t,z),
             'whole_system_ee_trajectory_separation_m':time_weighted_errors(t,ee),
             'first_observed_base_xy_crossings':{str(v):first_observed_crossing(t,xy,v) for v in POSITION_THRESHOLDS_M},
             'first_observed_base_yaw_crossings_deg':{str(v):first_observed_crossing(t,np.rad2deg(yaw),v) for v in YAW_THRESHOLDS_DEG},
             'first_observed_ee_crossings':{str(v):first_observed_crossing(t,ee,v) for v in POSITION_THRESHOLDS_M},
             'first_command_discrepancy':changes,'first_recorded_control':initial}
    return metrics,{'time':t,'base_xy':xy,'base_yaw_deg':np.rad2deg(yaw),'ee_world':ee,'base_z':z}


def saturation_diagnostics(trace, limits, dt):
    """Clipping is raw != applied, distinct from merely touching a bound.

    First observations use PRE-step time. Sparse records cannot establish exact
    onset between records; full-step counts live in the evaluation record.
    """
    limits=np.asarray(limits,dtype=float)
    if limits.shape!=(11,) or not np.isfinite(limits).all() or np.any(limits<=0):
        raise ValueError('Invalid command limits')
    valid=np.asarray(trace.get('has_action',trace['step']>0),dtype=bool)
    times=np.asarray(trace.get('action_time',trace['time']-dt))[valid]
    raw=np.asarray(trace['raw_action'])[valid];applied=np.asarray(trace['action'])[valid]
    if not len(times) or raw.shape!=(len(times),11) or not np.isfinite(raw).all() or not np.isfinite(applied).all():
        raise ValueError('Invalid saturation trace')
    clipped=np.abs(raw-applied)>1e-9
    names=['right','left','left_duplicate','right_duplicate']+[f'torque_{j}' for j in range(1,8)]
    def onset(mask):
        indices=np.flatnonzero(mask)
        if not len(indices):return None
        i=int(indices[0])
        return {'action_time_s':float(times[i]),'previous_recorded_action_time_s':float(times[max(0,i-1)]),
                'recording_gap_s':float(times[i]-times[max(0,i-1)]),
                'exact_from_episode_start':bool(i==0 or np.all(np.diff(times[:i+1])<=dt*1.001))}
    return {'first_observed_clipping':{'wheels':onset(clipped[:,:4].any(axis=1)),
                                     'torques':onset(clipped[:,4:].any(axis=1)),
                                     'per_command':{name:onset(clipped[:,j]) for j,name in enumerate(names)}},
            'first_observed_at_boundary':{'wheels':onset((np.abs(applied[:,:4])>=limits[:4]-1e-9).any(axis=1)),
                                          'torques':onset((np.abs(applied[:,4:])>=limits[4:]-1e-9).any(axis=1))},
            'sampling_note':'First recorded PRE-step clipping; exact only with continuous logging from start. Full-step fractions are stored separately.'}
