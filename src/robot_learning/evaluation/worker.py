"""One closed-loop rollout with fresh physics, model and controller state."""
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from robot_learning.data.storage import read_json, sha256, write_json
from robot_learning.learning.model import LinearPolicy, MLPPolicy
from robot_learning.settings import get_roboenv_root
from robot_learning.simulation.episode import Episode
from .metrics import TaskMetrics
from .subsystems import SubsystemMetrics, world_to_base


def action_limits(episode_config):
    base = episode_config['base_controller']
    wheel = (base['max_lin_vel'] + base['max_ang_vel']*base['wheel_base_width']/2)/base['wheel_radius']
    return np.array([wheel]*4 + [episode_config['arm_controller']['max_torque']]*7)


def joint_limit_violations(q, lower, upper):
    # PyBullet uses lower > upper for continuous joints: these have no angular bound.
    bounded = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    return bounded & ((q < lower-1e-6) | (q > upper+1e-6))


def run_rollout(directory, job):
    directory = Path(directory); protocol = read_json(directory/'protocol.json')
    cfg = read_json(directory/'episode_config.json')
    spec = read_json(directory/'policies.json')[job['policy']]
    scenario = job['scenario']; output = directory/'episodes'/f"{job['job_id']}.json"
    policy = None; metrics = None; subsystem = None; episode = None; trace = []
    started = time.monotonic(); limits = action_limits(cfg)
    clipped = np.zeros(11, dtype=np.int64); peak_raw = np.zeros(11)
    prediction_ns = 0; count = 0; violations = 0; joint_counts = np.zeros(7,dtype=np.int64)
    record = {**job, 'status': 'running', 'limits': limits.tolist(),
              'limits_meaning': 'expert command envelope, not a hardware certification',
              'valid': False, 'metrics': None}
    write_json(output, record)
    try:
        if spec['kind'] != 'expert':
            path = directory/spec['artifact']
            if sha256(path) != spec['sha256']:
                raise ValueError('Changed model artifact')
            if spec['kind'] == 'egocentric_mlp_regressor':
                from robot_learning.learning.egocentric import EgocentricPolicy
                model = EgocentricPolicy.load(path)
            elif spec['kind'] == 'gru_regressor':
                from robot_learning.learning.gru_policy import GRUPolicy
                model = GRUPolicy.load(path)
            else:
                model = (MLPPolicy if spec['kind'] == 'mlp_regressor' else LinearPolicy).load(path)
            def policy(observation):
                nonlocal prediction_ns, count
                start = time.perf_counter_ns()
                action = model.predict_action(observation)
                prediction_ns += time.perf_counter_ns()-start; count += 1
                return action
        np.random.seed(scenario['seed'])
        # Effective dt is checked after setup; horizon is integer physical steps.
        with Episode(scenario['target'], cfg, get_roboenv_root(), scenario['initial_base'],
                     policy=policy, horizon_steps=1,
                     action_filter=lambda action: np.clip(action, -limits, limits)) as episode:
            if policy is not None and hasattr(model, 'reset'):
                model.reset(episode.dt)
            steps = int(round(protocol['horizon_s']/episode.dt))
            if steps < 1 or not np.isclose(steps*episode.dt, protocol['horizon_s'],rtol=0,atol=1e-9):
                raise ValueError('Horizon must be an integer multiple of simulation dt')
            episode.horizon_steps = steps
            if not np.allclose(episode.initial_state['observation'],job['initial_observation'],rtol=0,atol=1e-10):
                raise ValueError('Initial state differs from paired source scenario')
            metrics = TaskMetrics(episode.state['distance'],episode.dt,steps,
                                  protocol['tolerances_m'],protocol['dwell_s'],
                                  protocol['failure_distance_penalty_m'])
            subsystem = SubsystemMetrics(episode.state,episode.goal,episode.goal_bearing,cfg,episode.dt,protocol['dwell_s'])
            record.update(dt=episode.dt, actual_initial_observation=episode.state['observation'].tolist(),
                          initially_outside_joint_limits=joint_limit_violations(episode.state['q'][4:11],episode.lower,episode.upper).tolist())
            dense_steps=int(round(protocol.get("dense_trace_until_s",0.)/episode.dt))
            def capture(step, action, raw, transition=None):
                trace.append((step, episode.state['time'],episode.state['observation'].copy(),
                              episode.state['distance'],action.copy(),raw.copy(),
                              episode.state['ee_position'].copy(),episode.state['shoulder'].copy(),
                              episode.state['orientation'].copy(),
                              world_to_base(episode.state['ee_position'],episode.state['base'],episode.state['orientation']),
                              transition['current']['time'] if transition is not None else 0.,
                              transition['current']['observation'].copy() if transition is not None else episode.state['observation'].copy(),
                              transition['phase'] if transition is not None and policy is None else 'unavailable',
                              bool(transition['navigation_done']) if transition is not None and policy is None else False,
                              transition is not None))
            capture(0,np.zeros(11),np.zeros(11))
            for transition in episode.transitions():
                metrics.update(transition['next']['distance'])
                subsystem.update(transition['next'])
                action,raw = transition['action'],transition['raw_action']
                clipped += np.abs(action-raw)>1e-9
                peak_raw = np.maximum(peak_raw,np.abs(raw))
                q = transition['next']['q'][4:11]
                outside = joint_limit_violations(q,episode.lower,episode.upper)
                violations += int(np.any(outside)); joint_counts += outside
                if episode.steps<=dense_steps or episode.steps % protocol['trace_every_steps']==0 or episode.termination is not None:
                    capture(episode.steps,action,raw,transition)
            valid = episode.termination == 'horizon' and episode.steps == steps
            record.update(status='complete' if valid else 'invalid',valid=valid,
                          termination=episode.termination,
                          first_lecturer_stop=episode.first_lecturer_stop,
                          metrics=metrics.result(valid))
    except KeyboardInterrupt:
        record.update(status='interrupted',error='Interrupted')
        if metrics is not None:record['metrics']=metrics.result(False)
        raise
    except Exception:
        record.update(status='invalid',error=traceback.format_exc())
        if metrics is not None:record['metrics']=metrics.result(False)
    finally:
        if subsystem is not None:
            record['subsystems']=subsystem.result(bool(record['valid']))
        executed = metrics.steps if metrics is not None else 0
        record.update(wall_s=time.monotonic()-started,
                      clipping_count_per_command=clipped.tolist(),
                      clipping_fraction_per_command=(clipped/max(1,executed)).tolist(),
                      raw_peak_abs_per_command=peak_raw.tolist(),
                      arm_joint_limit_violation_steps=violations,
                      arm_joint_limit_violation_steps_per_joint=joint_counts.tolist(),
                      prediction_calls=count,
                      mean_online_prediction_us=prediction_ns/count/1000 if count else None)
        if trace:
            np.savez_compressed(directory/'traces'/f"{job['job_id']}.npz",
                step=np.array([r[0] for r in trace]),time=np.array([r[1] for r in trace]),
                observation=np.stack([r[2] for r in trace]),distance=np.array([r[3] for r in trace]),
                action=np.stack([r[4] for r in trace]),raw_action=np.stack([r[5] for r in trace]),
                ee_position_world=np.stack([r[6] for r in trace]),
                shoulder_position_world=np.stack([r[7] for r in trace]),
                base_orientation_xyzw=np.stack([r[8] for r in trace]),
                ee_position_base=np.stack([r[9] for r in trace]),
                action_time=np.array([r[10] for r in trace]),
                action_observation=np.stack([r[11] for r in trace]),
                expert_phase=np.array([r[12] for r in trace],dtype='U16'),
                expert_navigation_done=np.array([r[13] for r in trace]),
                has_action=np.array([r[14] for r in trace]),
                expert_diagnostics_valid=np.full(len(trace),policy is None,dtype=bool))
            record['trace_sha256']=sha256(directory/'traces'/f"{job['job_id']}.npz")
        write_json(output, record)
    return record


def main():
    directory=Path(sys.argv[1]);job_id=sys.argv[2]
    job=next(job for job in read_json(directory/'jobs.json') if job['job_id']==job_id)
    run_rollout(directory,job)


if __name__ == '__main__':
    main()
