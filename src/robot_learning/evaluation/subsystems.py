"""Base navigation, arm geometry and terminal motion diagnostics, not policy inputs."""
from collections import deque
import numpy as np


def wrapped_angle(error):
    return np.arctan2(np.sin(error),np.cos(error))


def world_to_base(points,base_position,orientation_xyzw):
    """Full rigid rotation: do not discard roll/pitch when a rollout diverges."""
    x,y,z,w=np.asarray(orientation_xyzw,dtype=float)
    norm=np.sqrt(x*x+y*y+z*z+w*w)
    if not np.isfinite(norm) or norm<1e-12:raise ValueError('Invalid base quaternion')
    x,y,z,w=np.array([x,y,z,w])/norm
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return (np.asarray(points)-base_position)@rotation


class SubsystemMetrics:
    def __init__(self,initial,goal_xy,goal_yaw,cfg,dt,dwell_s):
        self.goal=np.asarray(goal_xy);self.yaw=goal_yaw
        self.pos_tol=cfg['base_controller']['pos_tol'];self.yaw_tol=cfg['base_controller']['ang_tol']
        self.reach=cfg['arm_controller']['reach_limit'];self.target=np.asarray(initial['observation'][-3:])
        self.dt=dt;self.steps=0;self.start=None;self.arrival=None
        self.required=int(np.ceil(dwell_s/dt-1e-10))
        self.tail=deque(maxlen=max(1,self.required+1))
        self.initial_base=np.asarray(initial['base']).copy();self.last_base=self.initial_base.copy()
        self.path_length=0.;self.integral_xy=0.;self.within_count=0;self.reach_count=0
        self.min_xy=np.inf;self.min_shoulder=np.inf;self.post_arrival_max_drift=0.
        self.arrival_base=None
        self.last=self._geometry(initial)
        self.min_xy=self.last['base_goal_distance_xy_m'];self.min_shoulder=self.last['shoulder_target_distance_m']
        if self.last['base_pose_within_tolerance']:self.start=0
        self._terminal(initial)

    def _geometry(self,state):
        xy=float(np.linalg.norm(np.asarray(state['base'])[:2]-self.goal))
        yaw=float(abs(wrapped_angle(state['yaw']-self.yaw)))
        shoulder=float(np.linalg.norm(self.target-state['shoulder']))
        return {'base_goal_distance_xy_m':xy,'base_goal_bearing_error_rad':yaw,
                'base_pose_within_tolerance':bool(xy<=self.pos_tol and yaw<=self.yaw_tol),
                'shoulder_target_distance_m':shoulder,
                'within_expert_reach_rule':bool(shoulder<=self.reach)}

    def _terminal(self,state):
        self.tail.append((np.asarray(state['base']).copy(),state['yaw'],
                          np.asarray(state['ee_position']).copy(),
                          float(np.linalg.norm(state['velocity'][:2])),abs(float(state['angular_velocity'][2])),
                          np.asarray(state['qd'][4:11]).copy()))

    def update(self,state):
        self.steps+=1;current=self._geometry(state)
        self.integral_xy+=.5*(self.last['base_goal_distance_xy_m']+current['base_goal_distance_xy_m'])*self.dt
        self.min_xy=min(self.min_xy,current['base_goal_distance_xy_m'])
        self.min_shoulder=min(self.min_shoulder,current['shoulder_target_distance_m'])
        self.within_count+=current['base_pose_within_tolerance'];self.reach_count+=current['within_expert_reach_rule']
        if current['base_pose_within_tolerance']:
            if self.start is None:self.start=self.steps
            if self.steps-self.start>=self.required and self.arrival is None:
                self.arrival=self.steps*self.dt;self.arrival_base=np.asarray(state['base']).copy()
        else:self.start=None
        if self.arrival_base is not None:
            self.post_arrival_max_drift=max(self.post_arrival_max_drift,
                float(np.linalg.norm(np.asarray(state['base'])[:2]-self.arrival_base[:2])))
        self.path_length+=float(np.linalg.norm(np.asarray(state['base'])[:2]-self.last_base[:2]))
        self.last_base=np.asarray(state['base']).copy();self.last=current;self._terminal(state)

    def result(self,valid):
        tail=list(self.tail);base=np.array([r[0] for r in tail]);ee=np.array([r[2] for r in tail])
        return {'status':'full_horizon' if valid else 'partial_invalid',
                'base_navigation':{**self.last,'goal_xy_m':self.goal.tolist(),'goal_bearing_rad':self.yaw,
                    'position_tolerance_m':self.pos_tol,'bearing_tolerance_rad':self.yaw_tol,
                    'stable_pose_arrival':bool(valid and self.arrival is not None),
                    'arrival_completion_s':self.arrival if valid else None,
                    'minimum_goal_distance_xy_m':self.min_xy,'path_length_xy_m':self.path_length,
                    'time_average_goal_distance_xy_m':self.integral_xy/(max(1,self.steps)*self.dt),
                    'fraction_steps_pose_within_tolerance':self.within_count/max(1,self.steps),
                    'maximum_drift_after_stable_arrival_xy_m':self.post_arrival_max_drift if self.arrival is not None else None},
                'arm_geometry':{'expert_reach_rule_m':self.reach,
                    'final_shoulder_target_distance_m':self.last['shoulder_target_distance_m'],
                    'minimum_shoulder_target_distance_m':self.min_shoulder,
                    'fraction_steps_within_expert_reach_rule':self.reach_count/max(1,self.steps),
                    'reach_rule_is_not_physical_reachability':True},
                'terminal_motion':{'window_samples':len(tail),'window_elapsed_s':(len(tail)-1)*self.dt,
                    'base_mean_speed_xy_m_s':float(np.mean([r[3] for r in tail])),
                    'base_p95_speed_xy_m_s':float(np.percentile([r[3] for r in tail],95)),
                    'base_mean_abs_yaw_rate_rad_s':float(np.mean([r[4] for r in tail])),
                    'base_max_xy_displacement_from_window_start_m':float(np.max(np.linalg.norm(base[:,:2]-base[0,:2],axis=1))),
                    'ee_max_displacement_from_window_start_m':float(np.max(np.linalg.norm(ee-ee[0],axis=1))),
                    'arm_mean_abs_joint_velocity_rad_s':np.mean(np.abs([r[5] for r in tail]),axis=0).tolist()}}
