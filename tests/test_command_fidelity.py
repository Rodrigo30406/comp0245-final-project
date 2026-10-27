# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
from robot_learning.learning.fidelity import command_fidelity, relative_command_fidelity


class CommandFidelityTests(unittest.TestCase):
    def test_channel_and_joint_tolerances_do_not_hide_a_bad_channel(self):
        y=np.zeros((10,9));p=y.copy();p[:,2]=.2
        r=command_fidelity(y,p,[0,.1,.2])
        self.assertEqual(r['per_output']['wheel_right']['within_tolerance_fraction'],[1.,1.,1.])
        self.assertEqual(r['per_output']['arm_torque_1']['within_tolerance_fraction'],[0.,0.,1.])
        self.assertEqual(r['all_nine_within_tolerance_fraction'],[0.,0.,1.])
        self.assertAlmostEqual(r['per_output']['arm_torque_1']['p95_absolute_error'],.2)

    def test_relative_tolerance_scales_and_zero_command_is_separate(self):
        y=np.ones((3,9));p=y.copy();y[:,0]=[1.,10.,0.];p[:,0]=[1.0625,10.625,.2]
        y[2,1]=0.;p[2,1]=.3
        r=relative_command_fidelity(y,p,np.ones(9),relative_tolerances=[.125],floor_factors=[0.,.05])
        channel=r['per_output']['wheel_right']
        self.assertEqual(channel['nonzero_relative_within_fraction'],[1.])
        self.assertEqual(channel['zero_rows'],1)
        self.assertEqual(channel['zero_command_prediction_mean_abs'],.2)
        self.assertEqual(r['base_stop_command_rows'],1)
        self.assertEqual(r['base_stop_prediction_mean_abs_by_side'],[.2,.3])
        self.assertAlmostEqual(r['mixed_profiles'][0]['all_nine_within_fraction'],2/3)

    def test_nonfinite_misaligned_and_invalid_thresholds_rejected(self):
        y=np.zeros((10,9))
        for p,t in [(np.full_like(y,np.nan),[.1]),(np.zeros((9,9)),[.1]),(y,[.2,.1]),(y,[-1]),(y,[])]:
            with self.assertRaises(ValueError):command_fidelity(y,p,t)


if __name__=='__main__':unittest.main()
