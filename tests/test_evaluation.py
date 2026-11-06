"""Metrics correctness and shared-engine autonomous/legacy regression tests."""
# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from robot_learning.evaluation.metrics import TaskMetrics
from robot_learning.evaluation.worker import action_limits, joint_limit_violations, make_action_filter
from robot_learning.evaluation.cli import validate_protocol, verify_frozen_files
from robot_learning.data.storage import read_json, sha256
from robot_learning.settings import PROJECT_ROOT, get_roboenv_root
from robot_learning.simulation.config import resolve_episode_config
from robot_learning.simulation.episode import Episode


class EvaluationMetricTests(unittest.TestCase):
    def test_dwell_requires_continuous_time_and_keeps_common_horizon(self):
        metric=TaskMetrics(.02,.1,10,[.01],.3,10.)
        for d in [.005,.005,.02,.005,.005,.005]:metric.update(d)
        self.assertFalse(metric.result(True)['by_tolerance']['0.01']['success'])
        metric.update(.005)
        self.assertAlmostEqual(metric.result(True)['by_tolerance']['0.01']['stable_arrival_completion_s'],.7)
        for d in [.02,.03,.04]:metric.update(d)
        result=metric.result(True)
        self.assertTrue(result['by_tolerance']['0.01']['success'])
        self.assertFalse(result['by_tolerance']['0.01']['final_within_tolerance'])
        self.assertEqual(result['executed_steps'],10)
        self.assertAlmostEqual(result['mean_distance_m'],np.trapezoid([.02,.005,.005,.02,.005,.005,.005,.005,.02,.03,.04],dx=.1))

    def test_invalid_rollout_cannot_claim_success_and_uses_failure_penalty(self):
        metric=TaskMetrics(0.,.1,10,[.01],.2,10.)
        for _ in range(3):metric.update(0.)
        result=metric.result(False)
        self.assertFalse(result['by_tolerance']['0.01']['success'])
        self.assertIsNone(result['final_distance_m'])
        self.assertEqual(result['penalized_final_distance_m'],10.)
        self.assertAlmostEqual(result['penalized_mean_distance_m'],7.)

    def test_continuous_joints_are_excluded_from_position_limit_checks(self):
        np.testing.assert_array_equal(joint_limit_violations(np.array([300.,.2]),np.array([0.,.5]),np.array([-1.,1.])),[False,True])

    def test_changed_frozen_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);p=directory/'protocol.json';p.write_text('{"horizon_s":40}')
            hashes={'protocol.json':sha256(p)}
            verify_frozen_files(directory,hashes)
            p.write_text('{"horizon_s":1}')
            with self.assertRaisesRegex(ValueError,'changed'):
                verify_frozen_files(directory,hashes)

    def test_protocol_and_declared_limits(self):
        cfg=read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json')
        validate_protocol(cfg)
        for replacement in [{'dwell_s':0},{'horizon_s':float('nan')},{'dense_trace_until_s':-1},{'tolerances_m':[.05,.01]}]:
            with self.assertRaises(ValueError):validate_protocol({**cfg,**replacement})
        limits=action_limits(resolve_episode_config({}))
        self.assertAlmostEqual(limits[0],(.25+1*.66/2)/.11)
        np.testing.assert_array_equal(limits[4:],[30]*7)

    def test_wheel_dead_band_is_optional_validated_and_zeroes_only_small_wheel_commands(self):
        cfg=read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json')
        validate_protocol({**cfg,'wheel_dead_band_rad_s':0.45})
        for bad in [-0.1,float('nan'),'0.45']:
            with self.assertRaises(ValueError):validate_protocol({**cfg,'wheel_dead_band_rad_s':bad})
        limits=action_limits(resolve_episode_config({}))
        plain=make_action_filter(limits); band=make_action_filter(limits,0.45)
        small=np.array([0.2,-0.3,-0.3,0.2]+[1.0]*7); large=np.array([0.2,-0.5,-0.5,0.2]+[1.0]*7); huge=np.array([9.0]*4+[50.0]*7)
        np.testing.assert_array_equal(plain(small),small)
        np.testing.assert_array_equal(band(small),np.array([0.0]*4+[1.0]*7))   # both wheels below the band -> exact zero, torques kept
        np.testing.assert_array_equal(band(large),large)                       # one wheel at or above the band -> untouched
        np.testing.assert_array_equal(band(huge),np.concatenate([[limits[0]]*4,[30.0]*7]))   # clipping still applies
        self.assertIsNot(band(small),small)                                     # the caller's array is not modified


class AdversityTests(unittest.TestCase):
    def test_protocol_adversity_keys_are_validated(self):
        from robot_learning.evaluation.adversity import validate_adversities
        cfg=read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json')
        validate_protocol({**cfg,'action_delay_steps':50,'payload_mass_kg':1.0,
                           'base_push':{'time_s':4.0,'duration_s':0.5,'force_n':100.0},
                           'observation_noise':{'base_m':0.02,'yaw_rad':0.02,'q_rad':0.01,'qd_rad_s':0.05}})
        for bad in [{'action_delay_steps':-1},{'action_delay_steps':2.0},{'payload_mass_kg':-1.0},{'payload_mass_kg':'1'},
                    {'base_push':{'time_s':39.8,'duration_s':0.5,'force_n':100.0}},{'base_push':{'time_s':4.0,'force_n':100.0}},
                    {'base_push':{'time_s':4.0,'duration_s':0.0,'force_n':100.0}},
                    {'observation_noise':{}},{'observation_noise':{'base_m':0.0}},{'observation_noise':{'target_m':0.1}},{'observation_noise':{'q_rad':-0.1}},
                    {'unknown_adversity':1}]:
            with self.assertRaises(ValueError,msg=str(bad)):validate_protocol({**cfg,**bad})
        validate_adversities({**cfg,'action_delay_steps':0})

    def test_delay_filter_executes_earlier_commands_and_zeros_before_the_first(self):
        limits=action_limits(resolve_episode_config({}))
        delayed=make_action_filter(limits,0.0,2)
        a=[np.full(11,float(i+1)) for i in range(4)]
        np.testing.assert_array_equal(delayed(a[0]),np.zeros(11)); np.testing.assert_array_equal(delayed(a[1]),np.zeros(11))
        np.testing.assert_array_equal(delayed(a[2]),np.clip(a[0],-limits,limits)); np.testing.assert_array_equal(delayed(a[3]),np.clip(a[1],-limits,limits))
        with self.assertRaises(ValueError):make_action_filter(limits,0.0,-1)
        from robot_learning.evaluation.adversity import Adversity
        cfg=read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json')
        adv=Adversity({**cfg,'actuator_gain':{'wheel_left':0.85}},1,is_expert=True); self.assertTrue(adv.active)
        gained=make_action_filter(limits,0.0,0,adv.gain); out=gained(np.array([2.0,2.0,2.0,2.0]+[1.0]*7))
        np.testing.assert_allclose(out,[2.0,1.7,1.7,2.0]+[1.0]*7)   # wheels 1 and 2 are the left side (worker.action_limits order)
        for bad in [{'actuator_gain':{}},{'actuator_gain':{'wheel_left':1.0}},{'actuator_gain':{'wheel_left':0.0}},{'actuator_gain':{'front':0.5}}]:
            with self.assertRaises(ValueError):validate_protocol({**cfg,**bad})

    def test_observation_noise_touches_the_policy_state_only_and_is_seeded(self):
        from robot_learning.evaluation.adversity import Adversity
        cfg={**read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json'),'observation_noise':{'base_m':0.02,'yaw_rad':0.02,'q_rad':0.01,'qd_rad_s':0.05}}
        obs=np.arange(21,dtype=float)
        expert=Adversity(cfg,5,is_expert=True); np.testing.assert_array_equal(expert.observe(obs),obs)
        self.assertIn('expert reads',expert.summary()['note'])
        a=Adversity(cfg,5,is_expert=False); b=Adversity(cfg,5,is_expert=False); c=Adversity(cfg,6,is_expert=False)
        na=a.observe(obs); np.testing.assert_array_equal(na,b.observe(obs)); self.assertFalse(np.array_equal(na,c.observe(obs)))
        np.testing.assert_array_equal(na[[2,18,19,20]],obs[[2,18,19,20]])   # base z and the target are never noised
        self.assertTrue(np.all(na[[0,1,3]]!=obs[[0,1,3]])); self.assertTrue(np.all(na[4:18]!=obs[4:18]))
        self.assertTrue(a.active); self.assertFalse(Adversity(read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json'),5,is_expert=False).active)

    def test_push_and_payload_act_on_the_simulated_robot(self):
        from robot_learning.evaluation.adversity import Adversity, PAYLOAD_LINK
        base=read_json(PROJECT_ROOT/'experiment_specs/evaluation/validation.json'); cfg=resolve_episode_config({})
        def run(protocol,steps=300):
            adv=Adversity(protocol,3,is_expert=True)
            with Episode([1,0,.5],cfg,get_roboenv_root(),horizon_steps=steps,perturbation=adv if adv.active else None) as ep:
                if protocol.get('payload_mass_kg'):
                    link=next(j for j in range(ep.client.getNumJoints(ep.robot_id)) if ep.client.getJointInfo(ep.robot_id,j)[12].decode()==PAYLOAD_LINK)
                    mass=ep.client.getDynamicsInfo(ep.robot_id,link)[0]
                else: mass=None
                last=None
                for tr in ep.transitions(): last=tr
                return last['next'],adv.summary(),mass
        plain,_,_=run(base)
        pushed,realised,_=run({**base,'base_push':{'time_s':0.05,'duration_s':0.1,'force_n':200.0}})
        self.assertIn(realised['base_push']['sense'],('from_behind','head_on')); self.assertAlmostEqual(realised['base_push']['start_time_s'],0.05,places=6)
        self.assertGreater(np.linalg.norm(pushed['base'][:2]-plain['base'][:2]),1e-4,'the push moves the base')
        loaded,realised_p,mass=run({**base,'payload_mass_kg':1.5},steps=50)
        self.assertAlmostEqual(mass,realised_p['payload_mass_kg']['link_mass_kg']); self.assertAlmostEqual(realised_p['payload_mass_kg']['added_kg'],1.5)
        self.assertGreater(mass,1.5)


class AutonomousEngineTests(unittest.TestCase):
    def test_extended_expert_matches_original_prefix_and_first_stop(self):
        cfg=resolve_episode_config({'max_steps':5})
        with Episode([0,1.1,.5],cfg,get_roboenv_root()) as legacy:
            expected=list(legacy.transitions());result=legacy.result()
        with Episode([0,1.1,.5],cfg,get_roboenv_root(),horizon_steps=12) as extended:
            actual=list(extended.transitions())
            self.assertEqual(len(actual),12)
            self.assertEqual(extended.termination,'horizon')
            self.assertEqual(extended.first_lecturer_stop['message'],result['termination_message'])
            self.assertEqual(extended.first_lecturer_stop['time_s'],result['stop_decision_time_s'])
            for a,b in zip(expected,actual):
                np.testing.assert_array_equal(a['action'],b['action'])
                np.testing.assert_array_equal(a['next']['observation'],b['next']['observation'])

    def test_policy_uses_own_live_states_without_expert_calls(self):
        observed=[]
        def policy(observation):
            observed.append(observation.copy());return np.zeros(11)
        with patch('robot_learning.simulation.episode.base_controller',side_effect=AssertionError('Expert invoked')),\
             patch('robot_learning.simulation.episode.arm_controller',side_effect=AssertionError('Expert invoked')):
            with Episode([1,0,.5],resolve_episode_config({}),get_roboenv_root(),policy=policy,horizon_steps=5) as ep:
                transitions=list(ep.transitions())
                self.assertEqual(len(observed),5)
                for obs,transition in zip(observed,transitions):
                    np.testing.assert_array_equal(obs,transition['current']['observation'])
                np.testing.assert_array_equal(observed[1],transitions[0]['next']['observation'])
                self.assertFalse(np.array_equal(observed[0],observed[-1]))
                self.assertIsNone(ep.first_lecturer_stop)

    def test_invalid_model_output_is_rejected_before_physics(self):
        with self.assertRaises(ValueError):
            Episode([0,0,.5],resolve_episode_config({}),get_roboenv_root(),policy=lambda x:np.zeros(11))
        with Episode([0,0,.5],resolve_episode_config({}),get_roboenv_root(),
                     policy=lambda x:np.full(11,np.nan),horizon_steps=5) as ep:
            with self.assertRaises(FloatingPointError):next(ep.transitions())
            self.assertEqual(ep.steps,0)


if __name__=='__main__':unittest.main()
