# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import unittest
import tempfile
from pathlib import Path
import numpy as np
from robot_learning.data.storage import write_json,read_json,sha256
from robot_learning.evaluation.subsystem_report import report,load_evaluation
from robot_learning.evaluation.trajectory import time_weighted_errors,first_observed_crossing,compare_traces,saturation_diagnostics


class TrajectoryTests(unittest.TestCase):
    def test_nonuniform_time_is_weighted_and_crossing_has_sampling_gap(self):
        r=time_weighted_errors([0,1,10],[0,2,2])
        self.assertAlmostEqual(r['time_mean_absolute'],1.9)
        self.assertAlmostEqual(r['time_rmse'],np.sqrt(3.8))
        crossing=first_observed_crossing([0,.1,.2],[0,.0005,.001],.001)
        self.assertAlmostEqual(crossing['time_s'],.2)
        self.assertAlmostEqual(crossing['sample_gap_s'],.1)
        self.assertIsNone(first_observed_crossing([0,.1],[0,0],.001))

    def test_first_control_time_is_prestep_and_placeholder_is_ignored(self):
        base={'step':np.arange(3),'time':np.array([0,.001,.002]),'observation':np.zeros((3,21)),
              'ee_position_world':np.zeros((3,3)),'action':np.zeros((3,11)),
              'raw_action':np.zeros((3,11)),'action_time':np.array([0,0,.001]),
              'action_observation':np.zeros((3,21)),'has_action':np.array([False,True,True])}
        other={k:v.copy() for k,v in base.items()}
        other['raw_action'][0]=100 # Initial placeholder is never an actual command.
        other['raw_action'][1:,0]=1
        r,_=compare_traces(other,base,.001)
        self.assertEqual(r['first_command_discrepancy']['wheel_commands']['time_s'],0)
        self.assertTrue(r['first_recorded_control']['identical_input'])
        self.assertEqual(r['base_xy_separation_m']['time_rmse'],0)

    def test_report_uses_base_and_system_only_and_rejects_changed_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'evaluation';(directory/'episodes').mkdir(parents=True);(directory/'traces').mkdir()
            write_json(directory/'run.json',{'status':'complete'})
            jobs=[]
            for label in ['expert','mlp']:
                jid=f'{label}_episode_000000';jobs.append({'job_id':jid,'policy':label,'scenario':{'episode_id':'episode_000000'}})
                trace=directory/'traces'/f'{jid}.npz'
                np.savez_compressed(trace,step=np.arange(3),time=np.array([0,.001,.002]),
                    observation=np.zeros((3,21)),ee_position_world=np.zeros((3,3)),
                    action=np.zeros((3,11)),raw_action=np.zeros((3,11)))
                write_json(directory/'episodes'/f'{jid}.json',{'valid':True,'dt':.001,'trace_sha256':sha256(trace),
                    'scenario':{'scenario_id':'paired'},
                    'metrics':{'by_tolerance':{'0.01':{'success':False}},'final_distance_m':.2,'mean_distance_m':.3},
                    'subsystems':{'base_navigation':{'stable_pose_arrival':False,'base_goal_distance_xy_m':.1,
                      'shoulder_target_distance_m':.7,'within_expert_reach_rule':True},
                      'terminal_motion':{'base_mean_speed_xy_m_s':0.,'arm_mean_abs_joint_velocity_rad_s':[0]*7}}})
            write_json(directory/'jobs.json',jobs)
            output=Path(tmp)/'report';summary=report(directory,output,plots=False)
            result=read_json(output/'trajectory_analysis.json')
            self.assertEqual(summary['mlp']['episodes'],1)
            self.assertNotIn('arm_base_frame_ee_rmse_median_m',summary['mlp'])
            self.assertNotIn('arm_mean_abs_joint_velocity_rad_s',result['per_episode'][0]['stop_motion'])
            self.assertNotIn('shoulder_target_distance_m',result['per_episode'][0]['base_navigation'])
            with self.assertRaises(FileExistsError):report(directory,output,plots=False)
            (directory/'traces/mlp_episode_000000.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):load_evaluation(directory)

    def test_clipping_onset_distinguishes_boundary_and_preserves_sparse_uncertainty(self):
        trace={'step':np.array([0,1,2,101]),'time':np.array([0,.001,.002,.101]),
               'action_time':np.array([0,0,.001,.1]),'has_action':np.array([False,True,True,True]),
               'raw_action':np.zeros((4,11)),'action':np.zeros((4,11))}
        trace['raw_action'][0]=100 # Ignore placeholder.
        trace['raw_action'][1,0]=trace['action'][1,0]=1 # At bound, no clip.
        trace['raw_action'][2,1]=2;trace['action'][2,1]=1
        trace['raw_action'][3,4]=2;trace['action'][3,4]=1
        r=saturation_diagnostics(trace,np.ones(11),.001)
        self.assertEqual(r['first_observed_at_boundary']['wheels']['action_time_s'],0)
        wheel=r['first_observed_clipping']['wheels'];arm=r['first_observed_clipping']['torques']
        self.assertEqual(wheel['action_time_s'],.001)
        self.assertTrue(wheel['exact_from_episode_start'])
        self.assertFalse(arm['exact_from_episode_start'])
        self.assertAlmostEqual(arm['recording_gap_s'],.099)
        self.assertIsNone(r['first_observed_clipping']['per_command']['right'])
        with self.assertRaises(ValueError):saturation_diagnostics(trace,np.zeros(11),.001)

    def test_nonfinite_and_misaligned_curves_rejected(self):
        for t,e in [([0,0],[0,1]),([0,1],[0,np.nan]),([0,1],[0])]:
            with self.assertRaises(ValueError):time_weighted_errors(t,e)


if __name__=='__main__':unittest.main()
