"""Stratified holdout geometry and explicit external evaluation leakage guards."""
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
import numpy as np
from robot_learning.data.sampling import stratified_pilot, stratified_grid
from robot_learning.learning.dataset import inventory, split_episodes
from robot_learning.evaluation.cli import validate_common_validation
from test_learning import make_dataset
from robot_learning.data.storage import read_json, write_json


class SpatialSamplingTests(unittest.TestCase):
    def test_larger_grids_preserve_count_margins_octants_and_training_coverage(self):
        box={'x':[-2,2],'y':[-2,2],'z':[.3,.8]}
        for grid,n,nval in [([8,4,4],128,26),([8,8,4],256,52),([8,8,8],512,103),([16,8,8],1024,205),([16,16,8],2048,410)]:
            plan=stratified_grid(box,grid,20261026,candidates=2)
            self.assertEqual(plan,stratified_grid(box,grid,20261026,candidates=2))
            cells=np.array(plan['cells']);v=cells[plan['validation_indices']]
            self.assertEqual(len(cells),n);self.assertEqual(len(v),nval)
            for axis,size in enumerate(grid):
                self.assertLessEqual(np.ptp(np.bincount(v[:,axis],minlength=size)),1)
            self.assertEqual(len(np.unique(v//(np.array(grid)//2),axis=0)),8)
            train=cells[plan['training_indices']].copy();train[:,2]//=2
            self.assertEqual(len(np.unique(train,axis=0)),n//2)
            recovered=np.floor((np.array(plan['targets'])-[-2,-2,.3])/[4,4,.5]*grid).astype(int)
            np.testing.assert_array_equal(cells,recovered)

    def test_jitter_and_holdout_are_reproducible_and_balanced(self):
        box={'x':[-2,2],'y':[-2,2],'z':[.3,.8]}
        plan=stratified_pilot(box,20261026)
        self.assertEqual(plan,stratified_pilot(box,20261026))
        points=np.asarray(plan['targets']);cells=np.asarray(plan['cells'])
        recovered=np.floor((points-[-2,-2,.3])/[4,4,.5]*4).astype(int)
        np.testing.assert_array_equal(recovered,cells)
        val=cells[plan['validation_indices']];train=cells[plan['training_indices']]
        self.assertEqual(len(val),13);self.assertEqual(len(train),51)
        for j in range(3):self.assertLessEqual(np.ptp(np.bincount(val[:,j],minlength=4)),1)
        self.assertEqual(len(np.unique(val//2,axis=0)),8)
        train[:,2]//=2
        self.assertEqual(len(np.unique(train,axis=0)),32)

    def test_external_conditions_require_opt_in_and_never_allow_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'original';root.mkdir();make_dataset(root)
            partition=split_episodes(inventory(root),.2,42)
            manifest={'dataset':str(root),**partition}
            selected=deepcopy(partition['validation'][:1])
            selected[0]['target']=[123.,456.,.5]
            with self.assertRaises(ValueError):validate_common_validation(root,selected,manifest)
            validate_common_validation(root,selected,manifest,allow_external=True)
            with self.assertRaises(ValueError):validate_common_validation(root,partition['training'][:1],manifest,allow_external=True)
