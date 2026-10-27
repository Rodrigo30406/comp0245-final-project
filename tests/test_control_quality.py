"""Ordered route comparison and resistance to a few catastrophic episodes."""
# Make src/ importable regardless of discovery order or installation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
from robot_learning.evaluation.control_quality import discrete_frechet,robust_summary


class ControlQualityTests(unittest.TestCase):
    def test_geometry_pauses_and_order(self):
        line=np.array([[0.,0.],[1.,0.],[2.,0.]])
        self.assertEqual(discrete_frechet(line,line),0.)
        self.assertEqual(discrete_frechet(line,line[[0,0,1,2,2]]),0.)
        self.assertEqual(discrete_frechet(line,line+[0,1]),1.)
        self.assertEqual(discrete_frechet(line,line[[0,2,1,2]]),1.)
        a=line[[0,2,1,2]];self.assertEqual(discrete_frechet(a,line),discrete_frechet(line,a))
        with self.assertRaises(ValueError):discrete_frechet([],line)

    def test_robust_distribution_keeps_catastrophic_tail_visible(self):
        result=robust_summary([.1]*98+[1000.,10000.])
        self.assertEqual(result['count'],100)
        self.assertNotIn('episodes',result)
        self.assertEqual(result['median'],.1);self.assertEqual(result['iqr'],0.)
        self.assertEqual(result['p90'],.1);self.assertEqual(result['maximum'],10000.)
        self.assertGreater(result['mean'],100.)
        self.assertGreater(result['top_two_episodes_share_of_sum'],.99)
        missing=robust_summary([None,np.nan,1.]);self.assertEqual(missing['missing_or_nonfinite'],2)
        self.assertEqual(missing['median'],1.)

    def test_subgroup_counts_cannot_replace_dataset_size(self):
        from robot_learning.evaluation.control_pattern_report import subgroup_report
        rows=[]
        for n in [1024,2048]:
            for i in range(32):
                rows.append({'episodes':n,'design':'uniform','valid':True,'task_success_1cm':False,
                    'frechet_xy_m':float(i+1),'initial_target_radius_xy_m':i/12.,
                    'target_z_m':.31+i*.014,'initial_nav_direction_rad':-3.+i*.18,
                    'initial_nav_distance_xy_m':i/15.,'abs_initial_nav_turn_rad':abs(-3.+i*.18),
                    'nearest_training_target_normalized':.02+i*.001})
        groups,correlations=subgroup_report(rows)
        self.assertEqual({g['episodes'] for g in groups},{1024,2048})
        self.assertTrue(all(g['count']==g['goals'] for g in groups))
        self.assertTrue(all(g['episodes'] in [1024,2048] for g in correlations))
