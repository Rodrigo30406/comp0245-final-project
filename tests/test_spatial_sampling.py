"""Stratified target plan: balanced geometry-only holdout, and the frozen plan of the report."""
import json
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_learning.data.sampling import stratified_grid  # noqa: E402

BOX = {"x": [-2, 2], "y": [-2, 2], "z": [.3, .8]}


class SpatialSamplingTests(unittest.TestCase):
    def test_grids_preserve_count_margins_octants_and_training_coverage(self):
        for grid, n, nval in [([8, 4, 4], 128, 26), ([8, 8, 4], 256, 52), ([16, 8, 8], 1024, 205)]:
            plan = stratified_grid(BOX, grid, 20261026, candidates=2)
            self.assertEqual(plan, stratified_grid(BOX, grid, 20261026, candidates=2))
            cells = np.array(plan["cells"])
            v = cells[plan["validation_indices"]]
            self.assertEqual(len(cells), n)
            self.assertEqual(len(v), nval)
            for axis, size in enumerate(grid):        # every level of every axis holds out the same number of cells (+-1)
                self.assertLessEqual(np.ptp(np.bincount(v[:, axis], minlength=size)), 1)
            self.assertEqual(len(np.unique(v // (np.array(grid) // 2), axis=0)), 8)     # all octants
            train = cells[plan["training_indices"]].copy()
            train[:, 2] //= 2
            self.assertEqual(len(np.unique(train, axis=0)), n // 2)   # every vertical pair keeps a training cell
            recovered = np.floor((np.array(plan["targets"]) - [-2, -2, .3]) / [4, 4, .5] * grid).astype(int)
            np.testing.assert_array_equal(cells, recovered)     # each jittered target lies in its own cell

    def test_frozen_plan_is_consistent(self):
        plan = json.loads((ROOT / "experiment_specs/collection/stratified1024_plan.json").read_text())
        cells = np.array(plan["cells"])
        self.assertEqual(plan["grid"], [16, 8, 8])
        self.assertEqual(len(plan["targets"]), 1024)
        self.assertEqual(len(plan["validation_indices"]), 205)
        self.assertEqual(sorted(plan["validation_indices"] + plan["training_indices"]), list(range(1024)))
        recovered = np.floor((np.array(plan["targets"]) - [-2, -2, .3]) / [4, 4, .5] * [16, 8, 8]).astype(int)
        np.testing.assert_array_equal(cells, recovered)
        for name in ("dart_stratified1024_low", "stratified1024_nominal"):   # nominal collections demonstrate the plan
            spec = json.loads((ROOT / f"experiment_specs/collection/{name}.json").read_text())
            self.assertEqual(spec["targets"], plan["targets"])
        spec = json.loads((ROOT / "experiment_specs/collection/e3_multistart3x_dart_low_v2_startgoal.json").read_text())
        self.assertEqual(spec["targets"], plan["targets"] * 3)               # random starts: each target three times


if __name__ == "__main__":
    unittest.main()
