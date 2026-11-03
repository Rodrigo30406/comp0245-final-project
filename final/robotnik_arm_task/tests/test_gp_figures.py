"""Analytic checks for the descriptive G/P metrics (no evaluation data)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results" / "step6"))
from gp_figures import SEGMENTS, Z_CELLS, torque_cells, trajectory_differences


class TestGPFigures(unittest.TestCase):
    def test_reach_filter_excludes_difference_crossing_extension(self):
        # The jump before reach must not contribute to the reach statistic.
        t = np.asarray([0., .01, .02, .03])
        tau = np.tile(np.asarray([0., 100., 103., 107.])[:, None], (1, 7))
        differences = trajectory_differences(t, tau, .015)
        np.testing.assert_array_equal(differences, np.full((1, 7), 4.))
        self.assertEqual(trajectory_differences(t, tau, .04).shape, (0, 7))

    def test_smoothness_rejects_sample_gap_and_wrong_frequency(self):
        tau = np.zeros((3, 7))
        for times in ([0., .01, .03], [0., .02, .04], [0., .01, .01]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                trajectory_differences(times, tau, 0.)

    def test_z_pooling_uses_squared_error_and_row_count(self):
        # One error-1 row and three error-3 rows pool to sqrt(7), not 2.
        cells = [{"group": group, "segment": segment, "z": z,
                  "n": 1 if z == Z_CELLS[0] else 3,
                  "rmse_pooled": 1. if z == Z_CELLS[0] else 3.}
                 for group in ("all", "clean", "noisy") for segment in SEGMENTS for z in Z_CELLS]
        pooled = torque_cells({"torque_segments": cells})
        for group in ("all", "clean", "noisy"):
            for segment in SEGMENTS:
                self.assertEqual(pooled[group, segment, "all_z"]["n"], 4)
                self.assertAlmostEqual(pooled[group, segment, "all_z"]["rmse_pooled"], np.sqrt(7))

    def test_empty_z_cell_does_not_change_pooled_rmse(self):
        cells = [{"group": group, "segment": segment, "z": z,
                  "n": 0 if z == Z_CELLS[0] else 3,
                  "rmse_pooled": None if z == Z_CELLS[0] else 2.}
                 for group in ("all", "clean", "noisy") for segment in SEGMENTS for z in Z_CELLS]
        pooled = torque_cells({"torque_segments": cells})
        self.assertEqual(pooled["all", "reach_hold", "all_z"]["rmse_pooled"], 2.)
        self.assertEqual(pooled["all", "reach_hold", "all_z"]["n"], 3)


if __name__ == "__main__":
    unittest.main(failfast=True)
