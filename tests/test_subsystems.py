# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
from robot_learning.evaluation.subsystems import world_to_base,wrapped_angle,SubsystemMetrics


def state(base=(0,0,0),yaw=0,ee=(.5,0,0)):
    return {'base':np.array(base,dtype=float),'yaw':yaw,'ee_position':np.array(ee,dtype=float),
            'shoulder':np.array(base,dtype=float),'observation':np.r_[np.zeros(18),[.5,0,0]],
            'velocity':np.zeros(3),'angular_velocity':np.zeros(3),'qd':np.zeros(17)}


class SubsystemTests(unittest.TestCase):
    def test_full_rotation_and_translation(self):
        # 90-degree pitch: local +X becomes world -Z.
        q=np.array([0,np.sin(np.pi/4),0,np.cos(np.pi/4)])
        np.testing.assert_allclose(world_to_base(np.array([1,2,2]),np.array([1,2,3]),q),[1,0,0],atol=1e-12)
        self.assertAlmostEqual(wrapped_angle(2*np.pi-.1),-.1)

    def test_base_requires_both_position_and_bearing_and_dwell(self):
        cfg={'base_controller':{'pos_tol':.08,'ang_tol':.05},'arm_controller':{'reach_limit':.85}}
        m=SubsystemMetrics(state(yaw=1),[0,0],0,cfg,.1,.2)
        m.update(state(yaw=1));self.assertFalse(m.result(True)['base_navigation']['stable_pose_arrival'])
        for _ in range(3):m.update(state())
        r=m.result(True)
        self.assertTrue(r['base_navigation']['stable_pose_arrival'])
        self.assertEqual(r['arm_geometry']['fraction_steps_within_expert_reach_rule'],1.)
        m.update(state(base=(1,0,0),ee=(1.5,0,0)))
        self.assertAlmostEqual(m.result(True)['base_navigation']['maximum_drift_after_stable_arrival_xy_m'],1)
        self.assertFalse(m.result(False)['base_navigation']['stable_pose_arrival'])


if __name__=='__main__':unittest.main()
