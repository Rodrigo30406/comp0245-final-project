"""Experiment F integration: extra rows preserve C3 scaling and expert validation."""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_policy as tp
import test_training as tt


class TestExtraRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tt.TestTriggerCopies.setUpClass.__func__(cls)
        cls.directory = os.path.join(cls.tmp, "extra")
        os.makedirs(cls.directory)
        rng = np.random.default_rng(42)
        cls.Xe, cls.Ye = rng.normal(10, 2, (11, 20)), rng.normal(5, 3, (11, 9))
        for name, a in (("X", cls.Xe), ("Y", cls.Ye)):
            np.save(os.path.join(cls.directory, name + ".npy"), a)
        cls.manifest = {"format": "bc_dagger_v1", "feature_set": "ego_novw", "n_rows": 11,
                        "split_sha256": tp.sha256_file(cls.split_path),
                        "arrays": {n: {"sha256": tp.sha256_file(os.path.join(cls.directory, n + ".npy"))}
                                   for n in ("X", "Y")}}
        cls.manifest_path = os.path.join(cls.directory, "manifest.json")
        tp.write_json_atomic(cls.manifest_path, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        tt.TestTriggerCopies.tearDownClass.__func__(cls)

    def test_append_preserves_scaling_trigger_rows_and_validation(self):
        captured = []
        real = tp.train_mlp

        def capture(X, Y, Xv, Yv, *a, **kw):
            captured.append(tuple(x.copy() for x in (X, Y, Xv, Yv)))
            return real(X, Y, Xv, Yv, *a, **kw)

        metas = []
        with mock.patch.object(tp, "train_mlp", side_effect=capture):
            for name, extra in (("ego-novw-trig", None), ("ego-novw-trig-dag1", self.directory)):
                _, m = tp.run_train(name, seed=7, out=os.path.join(self.tmp, name + ".npz"),
                                    cache_dir=self.cache_dir, split_path=self.split_path,
                                    max_epochs=1, max_train_episodes=3, trigger_copies_k=16,
                                    trigger_dir=self.trig_dir, extra_rows=extra, log=lambda s: None)
                metas.append(m)
        X0, Y0, Xv0, Yv0 = captured[0]
        X1, Y1, Xv1, Yv1 = captured[1]
        for a, b in ((X0, X1[:-11]), (Y0, Y1[:-11]), (Xv0, Xv1), (Yv0, Yv1)):
            np.testing.assert_array_equal(a, b)
        raw = tp.load_training_data("ego_novw", self.cache_dir, self.split_path,
                                    max_train_episodes=3, allow_subset=True)
        xm, xs = tp.standardiser(raw[0])
        ym, ys = tp.standardiser(raw[1])
        np.testing.assert_array_equal(X1[-11:], (self.Xe - xm) / xs)
        np.testing.assert_array_equal(Y1[-11:], (self.Ye - ym) / ys)
        self.assertEqual(metas[1]["extra_rows"]["manifest_sha256"], tp.sha256_file(self.manifest_path))
        self.assertNotIn("extra_rows", metas[0])
        self.assertEqual(metas[1]["n_val_rows"], metas[0]["n_val_rows"])

    def test_refuses_wrong_split_and_tampered_arrays(self):
        with self.assertRaises(ValueError):
            tp.load_extra_rows(self.directory, "raw", self.split_path)
        a = self.manifest.copy()
        a["split_sha256"] = "0" * 64
        tp.write_json_atomic(self.manifest_path, a)
        try:
            with self.assertRaises(ValueError):
                tp.load_extra_rows(self.directory, "ego_novw", self.split_path)
        finally:
            tp.write_json_atomic(self.manifest_path, self.manifest)
        path = os.path.join(self.directory, "X.npy")
        np.save(path, self.Xe + 1)
        try:
            with self.assertRaises(ValueError):
                tp.load_extra_rows(self.directory, "ego_novw", self.split_path)
        finally:
            np.save(path, self.Xe)

    def test_train_many_forwards_extra_rows(self):
        with mock.patch.object(tp, "run_train_many") as many:
            tp.main(["train-many", "--model", "ego-novw-trig-dag1", "--seeds", "0,1",
                     "--trigger-copies", "16", "--extra-rows", self.directory])
        forwarded = many.call_args.kwargs["extra_args"]
        self.assertEqual(forwarded[forwarded.index("--extra-rows") + 1], self.directory)


if __name__ == "__main__":
    unittest.main()
