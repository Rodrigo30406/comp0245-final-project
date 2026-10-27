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
from robot_learning.evaluation.worker import action_limits, joint_limit_violations
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
