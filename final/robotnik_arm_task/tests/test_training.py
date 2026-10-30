"""Unit tests of the training pipeline (train_policy.py): split, cache, partial_fit epochs, forward = predict, best
epoch kept, the two-sided lag search of the offline metrics.

Run from the task directory:
  python -m unittest tests.test_training -v
Uses data/train (index.csv and a handful of episode files) and builds a small cache in a temporary directory.
The checks against the full cache in results/step3/cache are skipped when it has not been built.
"""
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

import numpy as np  # noqa: E402

import bc_features as bcf  # noqa: E402
import train_policy as tp  # noqa: E402

N_MINI_TRAIN, N_MINI_VAL = 3, 1


def npz_rows(path, steps):
    """The recorded values of the cached columns at the given steps, read directly from an episode file."""
    with np.load(path) as d:
        a = {k: d[k] for k in d.files}
    target = a["target"].astype(np.float64)
    v, w = bcf.wheels_to_vw(a["wheel_cmds"][steps])
    return {
        "step": steps, "t": a["t"][steps], "base_xy": a["base_pos"][steps, :2],
        "base_bearing": a["base_bearing"][steps], "q_arm": a["q_mes"][steps, 4:11],
        "qd_arm": a["qd_mes"][steps, 4:11], "qd_wheels": a["qd_mes"][steps, 0:4],
        "target": np.broadcast_to(target, (len(steps), 3)), "label_v": v, "label_w": w,
        "label_tau": a["torque_arm"][steps], "phase": a["phase"][steps], "nav_done": a["nav_done"][steps],
        "ee_err": np.linalg.norm(a["ee_pos"][steps].astype(np.float64) - target, axis=1),
    }, a


class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = tp.make_split()
        cls.split2 = tp.make_split()

    def test_deterministic(self):
        self.assertEqual(tp.split_core(self.split), tp.split_core(self.split2))

    def test_matches_saved_split_file(self):
        if not os.path.exists(tp.SPLIT_PATH):
            self.skipTest("trained/data_split.json not created yet")
        saved = tp.load_split(tp.SPLIT_PATH)
        self.assertEqual(tp.split_core(saved), json.loads(json.dumps(tp.split_core(self.split))))

    def test_counts_and_disjoint(self):
        s = self.split
        self.assertEqual(s["n_kept"], 1897)
        self.assertEqual(len(s["kept_episodes"]), 1897)
        tr, va = set(s["train_episodes"]), set(s["val_episodes"])
        self.assertFalse(tr & va)
        self.assertEqual(tr | va, set(s["kept_episodes"]))
        self.assertEqual(len(tr) + len(va), 1897)
        self.assertEqual(s["train_episodes"], sorted(tr))
        self.assertEqual(s["val_episodes"], sorted(va))

    def test_filter(self):
        import csv
        with open(os.path.join(tp.TRAIN_DIR, "index.csv")) as f:
            rows = list(csv.DictReader(f))
        kept = sorted(int(r["episode"]) for r in rows if r["status"] == "reached" and r["nav_timed_out"] == "False")
        self.assertEqual(self.split["kept_episodes"], kept)

    def test_stratified(self):
        s = self.split
        val = set(s["val_episodes"])
        n_strata = 0
        for noisy in (False, True):
            for mode in tp.MODES:
                eps = [int(e) for e, info in s["episodes"].items() if info["noisy"] == noisy and info["mode"] == mode]
                n_val = sum(e in val for e in eps)
                self.assertEqual(n_val, round(0.15 * len(eps)), (noisy, mode))
                n_strata += 1
        self.assertEqual(n_strata, 6)
        self.assertEqual(sum(v["n"] for v in s["strata"].values()), 1897)

    def test_modes_agree_with_step2_analysis(self):
        path = os.path.join(TASK_DIR, "results", "data_analysis", "episodes_train.csv")
        if not os.path.exists(path):
            self.skipTest("step-2 episode table missing")
        import csv
        with open(path) as f:
            ref = {int(r["episode"]): r for r in csv.DictReader(f)}
        for e, info in self.split["episodes"].items():
            self.assertEqual(info["mode"], ref[int(e)]["mode"])
            self.assertEqual(str(info["noisy"]), ref[int(e)]["noisy"])

    def test_offsets(self):
        s = self.split
        off = np.array(s["train_offsets"])
        self.assertEqual(len(off), len(s["train_episodes"]))
        self.assertTrue(((off >= 0) & (off < 9)).all())
        np.testing.assert_array_equal(off, np.random.default_rng(0).integers(0, 9, size=len(off)))
        self.assertEqual(len(set(off.tolist())), 9)  # all offsets occur
        self.assertEqual(s["stride"], 9)

    def test_seed_is_used(self):
        other = tp.make_split(split_seed=1)
        self.assertNotEqual(other["val_episodes"], self.split["val_episodes"])

    def test_refuses_overwrite(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "split.json")
            tp.write_split(path)
            with self.assertRaises(SystemExit):
                tp.write_split(path)
            tp.write_split(path, force=True)
        finally:
            shutil.rmtree(tmp)


class TestCache(unittest.TestCase):
    """A small cache (3 training and 1 validation episodes of the real split), built in a temporary directory."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.split_path = os.path.join(cls.tmp, "split.json")
        cls.split = tp.write_split(cls.split_path)
        cls.cache_dir = os.path.join(cls.tmp, "cache")
        cls.train_eps = cls.split["train_episodes"][:N_MINI_TRAIN]
        cls.val_eps = cls.split["val_episodes"][:N_MINI_VAL]
        tp.build_cache(cls.split_path, cls.cache_dir, workers=2, train_episodes=cls.train_eps,
                       val_episodes=cls.val_eps, verbose=False)
        cls.tr = tp.Cache("train", cls.cache_dir, cls.split_path, allow_subset=True)
        cls.va = tp.Cache("val", cls.cache_dir, cls.split_path, allow_subset=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def path_of(self, e):
        return os.path.join(tp.TRAIN_DIR, self.split["episodes"][str(e)]["file"])

    def test_refuses_subset_cache(self):
        with self.assertRaises(RuntimeError):
            tp.Cache("train", self.cache_dir, self.split_path)

    def test_refuses_other_split(self):
        other = os.path.join(self.tmp, "other_split.json")
        tp.write_split(other, split_seed=1)
        with self.assertRaises(RuntimeError):
            tp.Cache("train", self.cache_dir, other, allow_subset=True)

    def test_train_rows_are_offset_stride(self):
        offsets = dict(zip(self.split["train_episodes"], self.split["train_offsets"]))
        for i, e in enumerate(self.tr.episodes):
            sl = self.tr.episode_slice(i)
            n_steps = self.split["episodes"][str(e)]["n_steps"]
            np.testing.assert_array_equal(self.tr.cols["step"][sl], np.arange(offsets[e], n_steps, 9))
            self.assertTrue((self.tr.cols["episode"][sl] == e).all())
        for i, e in enumerate(self.va.episodes):
            sl = self.va.episode_slice(i)
            np.testing.assert_array_equal(self.va.cols["step"][sl],
                                          np.arange(self.split["episodes"][str(e)]["n_steps"]))

    def test_cached_values_equal_npz(self):
        rng = np.random.default_rng(1)
        for cache in (self.tr, self.va):
            for i, e in enumerate(cache.episodes):
                sl = cache.episode_slice(i)
                rows = np.sort(rng.choice(np.arange(sl.start, sl.stop), size=25, replace=False))
                steps = np.asarray(cache.cols["step"][rows])
                ref, _ = npz_rows(self.path_of(e), steps)
                for name, value in ref.items():
                    got = np.asarray(cache.cols[name][rows])
                    self.assertEqual(got.dtype, np.dtype(tp.CACHE_COLUMNS[name][0]), name)
                    np.testing.assert_array_equal(got, np.asarray(value, dtype=got.dtype), err_msg=name)
                    if got.dtype == np.float32:  # float32 columns are the recorded float32 values, unchanged
                        self.assertEqual(np.asarray(value).dtype, np.float32, name)
                self.assertTrue((np.asarray(cache.cols["noisy"][rows]) ==
                                 self.split["episodes"][str(e)]["noisy"]).all())

    def test_features_equal_direct_build(self):
        """Cached signals -> features equals bc_features on the recorded arrays (all feature sets)."""
        i, e = 0, self.va.episodes[0]
        sl = self.va.episode_slice(i)
        _, a = npz_rows(self.path_of(e), np.arange(1))
        sig = bcf.measured_signals(a["base_pos"], a["base_bearing"], a["q_mes"], a["qd_mes"], a["target"])
        for fs in bcf.FEATURE_SETS:
            np.testing.assert_array_equal(tp.cache_features(self.va.cols, fs, sl, chunk=1000),
                                          bcf.build_features(fs, sig), err_msg=fs)
        Y = tp.cache_labels(self.va.cols, sl)
        v, w = bcf.wheels_to_vw(a["wheel_cmds"])
        np.testing.assert_array_equal(Y, np.column_stack([v, w, a["torque_arm"].astype(float)]))

    def test_full_cache(self):
        manifest = os.path.join(tp.CACHE_DIR, "manifest.json")
        if not (os.path.exists(manifest) and os.path.exists(tp.SPLIT_PATH)):
            self.skipTest("full cache not built")
        split = tp.load_split(tp.SPLIT_PATH)
        rng = np.random.default_rng(2)
        for part in ("train", "val"):
            cache = tp.Cache(part, tp.CACHE_DIR, tp.SPLIT_PATH)
            self.assertEqual(cache.episodes, split[f"{part}_episodes"])
            for i in rng.choice(len(cache.episodes), size=3, replace=False):
                e = cache.episodes[i]
                sl = cache.episode_slice(i)
                rows = np.sort(rng.choice(np.arange(sl.start, sl.stop), size=10, replace=False))
                steps = np.asarray(cache.cols["step"][rows])
                if part == "train":
                    off = dict(zip(split["train_episodes"], split["train_offsets"]))[e]
                    np.testing.assert_array_equal(cache.cols["step"][sl],
                                                  np.arange(off, split["episodes"][str(e)]["n_steps"], 9))
                ref, _ = npz_rows(os.path.join(tp.TRAIN_DIR, split["episodes"][str(e)]["file"]), steps)
                for name, value in ref.items():
                    got = np.asarray(cache.cols[name][rows])
                    np.testing.assert_array_equal(got, np.asarray(value, dtype=got.dtype), err_msg=name)


class TestForwardEqualsPredict(unittest.TestCase):
    """(a) The saved .npz model run by bc_features.forward reproduces the trained model to 1e-9."""

    @classmethod
    def setUpClass(cls):
        TestCache.setUpClass.__func__(cls)
        cls.X, cls.Y, cls.Xv, cls.Yv, _, _ = tp.load_training_data(
            "ego", cls.cache_dir, cls.split_path, allow_subset=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def standardised(self):
        xm, xs = tp.standardiser(self.X)
        ym, ys = tp.standardiser(self.Y)
        return xm, xs, ym, ys, (self.X - xm) / xs, (self.Y - ym) / ys, (self.Xv - xm) / xs, (self.Yv - ym) / ys

    def test_sizes(self):
        self.assertGreater(self.X.shape[0], 2000)
        self.assertGreater(self.Xv.shape[0], 3000)
        self.assertEqual(self.Y.shape[1], 9)
        self.assertEqual(self.X.dtype, np.float64)

    def test_mlp(self):
        xm, xs, ym, ys, Xs, Ys, Xvs, Yvs = self.standardised()
        weights, biases, hist, model = tp.train_mlp(Xs, Ys, Xvs, Yvs, seed=0, max_epochs=3)
        self.assertEqual(hist["epochs_run"], 3)
        model.coefs_, model.intercepts_ = weights, biases  # the saved (best-epoch) weights
        path = os.path.join(self.tmp, "mlp.npz")
        tp.save_trained(path, "ego", xm, xs, ym, ys, weights, biases, {"test": True})
        loaded = bcf.load_model(path)
        self.assertEqual(len(loaded["weights"]), 3)
        ref = model.predict(Xvs) * ys + ym
        got = bcf.forward(loaded, self.Xv)
        self.assertLess(np.max(np.abs(got - ref)), 1e-9)
        # the chunked validation loss is the MSE of predict in standardised units
        self.assertAlmostEqual(tp.std_mse(weights, biases, Xvs, Yvs, chunk=777),
                               float(np.mean((model.predict(Xvs) - Yvs) ** 2)), places=12)
        self.assertAlmostEqual(tp.std_mse(weights, biases, Xvs, Yvs), hist["best_val_loss"], places=14)

    def test_ols(self):
        xm, xs, ym, ys, Xs, Ys, Xvs, Yvs = self.standardised()
        W, b, _ = tp.fit_ols(Xs, Ys)
        path = os.path.join(self.tmp, "ols.npz")
        tp.save_trained(path, "ego", xm, xs, ym, ys, [W], [b], {"test": True})
        loaded = bcf.load_model(path)
        self.assertEqual(len(loaded["weights"]), 1)
        A = np.column_stack([Xs, np.ones(len(Xs))])
        beta = np.linalg.lstsq(A, Ys, rcond=None)[0]
        ref = np.column_stack([Xvs, np.ones(len(Xvs))]) @ beta * ys + ym
        self.assertLess(np.max(np.abs(bcf.forward(loaded, self.Xv) - ref)), 1e-9)

    def test_offline_metrics_lag_report(self):
        """offline_metrics on the small cache (an OLS model): every lag entry keeps the old keys and has the new
        ones, the JSON documents them (lag_definition) and print_offline prints the signed lags."""
        xm, xs, ym, ys, Xs, Ys, _, _ = self.standardised()
        W, b, _ = tp.fit_ols(Xs, Ys)
        path = os.path.join(self.tmp, "ols_offline.npz")
        tp.save_trained(path, "ego", xm, xs, ym, ys, [W], [b], {"test": True})
        res = tp.offline_metrics(path, self.cache_dir, self.split_path, allow_subset=True)
        self.assertIsInstance(res["lag_definition"], str)
        lags = list(range(-tp.LAG_MAX, tp.LAG_MAX + 1))
        clean = not self.split["episodes"][str(self.val_eps[0])]["noisy"]
        for name in tp.EVENT_TYPES:
            x = res["lag"][name]
            self.assertEqual(x["n_events"], int(clean) * res["counts"]["events"][name])
            self.assertEqual(x["n_pairs_lag0"], x["n_pairs"])
            self.assertEqual(list(x["lags"]), lags)
            for src in ("model", "copycat"):
                for o in ("v", "w"):
                    y = x[src][o]
                    self.assertEqual(set(y), {"k_best", "mse", "mse_best", "mse_k0", "rel_range", "flat",
                                              "boundary"})
                    self.assertEqual(len(y["mse"]), len(lags))
                    if x["n_events"] and not y["flat"]:
                        self.assertEqual(y["mse_best"], y["mse"][lags.index(y["k_best"])])
        self.assertEqual(json.loads(json.dumps(tp.to_json(res)))["lag"]["start"]["lags"], lags)
        lines = []
        tp.print_offline(res, log=lines.append)
        self.assertTrue(all(any(line.strip().startswith(name + " ") for line in lines) for name in tp.EVENT_TYPES))

    def test_run_train_end_to_end(self):
        """The CLI training path: saved meta and the val loss recomputed from the saved file agree."""
        for model_name, epochs in (("ego", 2), ("ego-ols", tp.MAX_EPOCHS)):
            out = os.path.join(self.tmp, f"{model_name}.npz")
            _, meta = tp.run_train(model_name, seed=1, out=out, cache_dir=self.cache_dir, split_path=self.split_path,
                                   max_epochs=epochs, max_train_episodes=N_MINI_TRAIN, log=lambda s: None)
            m = bcf.load_model(out)
            self.assertEqual(m["meta"]["model"], model_name)
            self.assertEqual(m["meta"]["split_sha256"], tp.sha256_file(self.split_path))
            self.assertEqual(m["meta"]["n_train_rows"], self.X.shape[0])
            self.assertEqual(m["meta"]["n_val_rows"], self.Xv.shape[0])
            val = float(np.mean(((bcf.forward(m, self.Xv) - self.Yv) / m["y_scale"]) ** 2))
            self.assertAlmostEqual(val, meta["val_loss"], delta=1e-9 * max(1.0, val))
            for key in ("git_commit", "sklearn_version", "train_s", "hyperparameters", "stride", "epochs_run"):
                self.assertIn(key, m["meta"])
            if model_name == "ego":
                self.assertEqual(m["meta"]["epochs_run"], 2)
                self.assertEqual(len(m["meta"]["history"]["val_loss"]), 2)
                self.assertEqual(m["meta"]["history"]["val_loss"][m["meta"]["best_epoch"] - 1], meta["val_loss"])
            with self.assertRaises(SystemExit):  # no silent overwrite
                tp.run_train(model_name, seed=1, out=out, cache_dir=self.cache_dir, split_path=self.split_path,
                             max_epochs=epochs, max_train_episodes=N_MINI_TRAIN, log=lambda s: None)
        with self.assertRaises(SystemExit):  # smoke settings never take an official trained/ name
            tp.run_train("ego", seed=1, cache_dir=self.cache_dir, split_path=self.split_path, max_epochs=1,
                         log=lambda s: None)

    def test_train_many(self):
        """Parallel seeds as separate processes (output folder redirected to the temporary directory)."""
        out_dir = os.path.join(self.tmp, "many")
        os.makedirs(out_dir)
        extra = ["--max-epochs", "1", "--max-train-episodes", str(N_MINI_TRAIN)]
        with mock.patch.object(tp, "TRAINED_DIR", out_dir), mock.patch("builtins.print"):
            rc = tp.run_train_many("ego", [3, 4], jobs=2, cache_dir=self.cache_dir, split_path=self.split_path,
                                   extra_args=extra)
            self.assertEqual(rc, {3: 0, 4: 0})
            models = {s: bcf.load_model(os.path.join(out_dir, f"ego_s{s}.npz")) for s in (3, 4)}
            self.assertEqual([models[s]["meta"]["seed"] for s in (3, 4)], [3, 4])
            self.assertFalse(np.array_equal(models[3]["weights"][0], models[4]["weights"][0]))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "ego_s3.log")))
            self.assertEqual(tp.run_train_many("ego", [3, 4], jobs=2, cache_dir=self.cache_dir,
                                               split_path=self.split_path, extra_args=extra), {})  # skipped
        # the same seed in a separate process gives the same weights as in this process
        Xs_mean, Xs_scale = tp.standardiser(self.X)
        ym, ys = tp.standardiser(self.Y)
        w, b, _, _ = tp.train_mlp((self.X - Xs_mean) / Xs_scale, (self.Y - ym) / ys, (self.Xv - Xs_mean) / Xs_scale,
                                  (self.Yv - ym) / ys, seed=3, max_epochs=1)
        for a, c in zip(w + b, models[3]["weights"] + models[3]["biases"]):
            np.testing.assert_array_equal(a, c)


class TestPartialFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.normal(size=(1000, 6))
        cls.Y = np.column_stack([np.sin(cls.X[:, 0]), cls.X[:, 1] * cls.X[:, 2], cls.X[:, 3]])

    def test_sklearn_source_breaks_when_incremental(self):
        import sklearn
        from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
        src = [line.strip() for line in inspect.getsource(BaseMultilayerPerceptron._fit_stochastic).splitlines()]
        src = [line for line in src if line and not line.startswith("#")]
        i = src.index("if incremental:")
        self.assertEqual(src[i + 1], "break")
        self.assertEqual(sklearn.__version__, "1.9.1")  # the version the plan was checked against

    def test_one_epoch_per_call_and_adam_state_kept(self):
        model = tp.make_mlp(0)
        n = len(self.X)
        n_batches = int(np.ceil(n / tp.MLP_PARAMS["batch_size"]))
        opt = None
        for k in range(1, 4):
            model.partial_fit(self.X, self.Y)
            self.assertEqual(len(model.loss_curve_), k)
            self.assertEqual(model.t_, k * n)
            self.assertEqual(model.n_iter_, 1)
            if opt is None:
                opt = model._optimizer
            self.assertIs(model._optimizer, opt)       # optimiser not rebuilt
            self.assertEqual(model._optimizer.t, k * n_batches)  # Adam step count continues

    def record_shuffles(self, random_state, epochs=2):
        from sklearn.neural_network import MLPRegressor
        from sklearn.neural_network import _multilayer_perceptron as mlp_mod
        seen = []
        real = mlp_mod.shuffle

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            seen.append(np.array(out))
            return out

        model = MLPRegressor(**tp.MLP_PARAMS, random_state=random_state)
        with mock.patch.object(mlp_mod, "shuffle", spy):
            for _ in range(epochs):
                model.partial_fit(self.X, self.Y)
        self.assertEqual(len(seen), epochs)
        return seen

    def test_shuffle_differs_between_epochs(self):
        perms = self.record_shuffles(np.random.RandomState(0), epochs=3)
        self.assertFalse(np.array_equal(perms[0], perms[1]))
        self.assertFalse(np.array_equal(perms[1], perms[2]))
        # the pitfall the plan avoids: an int random_state rebuilds RandomState(0) in every partial_fit call, so
        # every epoch after the first (whose draws follow the weight initialisation) replays the same shuffle
        perms_int = self.record_shuffles(0, epochs=3)
        self.assertTrue(np.array_equal(perms_int[1], perms_int[2]))
        # make_mlp passes a RandomState instance
        self.assertIsInstance(tp.make_mlp(3).random_state, np.random.RandomState)

    def test_same_seed_identical_weights(self):
        Xv, Yv = self.X[:200], self.Y[:200]
        runs = [tp.train_mlp(self.X, self.Y, Xv, Yv, seed=s, max_epochs=2) for s in (5, 5, 6)]
        for a, b in zip(runs[0][0] + runs[0][1], runs[1][0] + runs[1][1]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(runs[0][2]["val_loss"], runs[1][2]["val_loss"])
        self.assertFalse(np.array_equal(runs[0][0][0], runs[2][0][0]))

    def test_best_weights_kept(self):
        """The returned weights are those of the best epoch, not of the last one. The validation labels are half the
        training labels, so the validation loss falls while the fit approaches them and rises after (history
        0.117, 0.070, 0.072, 0.116 with seed 0): the best epoch is not the last. std_mse is wrapped to record the
        weights it is called with, one snapshot per epoch."""
        Xv, Yv = self.X[:200], 0.5 * self.Y[:200]
        real, snapshots = tp.std_mse, []

        def spy(weights, biases, *args, **kwargs):
            snapshots.append([a.copy() for a in weights + biases])
            return real(weights, biases, *args, **kwargs)

        with mock.patch.object(tp, "std_mse", spy):
            w, b, hist, model = tp.train_mlp(self.X, self.Y, Xv, Yv, seed=0, max_epochs=4)
        self.assertEqual(hist["epochs_run"], 4)
        self.assertEqual(len(snapshots), 4)
        self.assertLess(hist["best_epoch"], hist["epochs_run"])          # precondition: the last epoch is not best
        self.assertEqual(hist["best_epoch"], int(np.argmin(hist["val_loss"])) + 1)
        self.assertEqual(hist["best_val_loss"], min(hist["val_loss"]))
        for a, s in zip(w + b, snapshots[hist["best_epoch"] - 1]):     # exactly the best epoch's snapshot ...
            np.testing.assert_array_equal(a, s)
        self.assertFalse(all(np.array_equal(a, c) for a, c in zip(w + b, model.coefs_ + model.intercepts_)))
        self.assertAlmostEqual(tp.std_mse(w, b, Xv, Yv), hist["best_val_loss"], places=14)   # ... not the last one
        self.assertGreater(tp.std_mse(model.coefs_, model.intercepts_, Xv, Yv), hist["best_val_loss"])

    def run_stopper(self, losses):
        es = tp.EarlyStopping(patience=5, rel_tol=1e-3)
        for epoch, loss in enumerate(losses, start=1):
            _, stop = es.update(epoch, loss)
            if stop:
                return epoch, es.best_epoch
        return None, es.best_epoch

    def test_early_stopping_rule(self):
        # plateau after epoch 2: epochs 3-7 do not improve -> stop at 7, best epoch 2
        self.assertEqual(self.run_stopper([1.0, 0.5, 0.6, 0.55, 0.5001, 0.52, 0.51, 0.1]), (7, 2))
        # a decrease smaller than 1e-3 relative updates the kept weights but not the patience counter
        self.assertEqual(self.run_stopper([1.0, 0.5, 0.4999, 0.7, 0.7, 0.7, 0.7, 0.1]), (7, 3))
        # slow steady decrease (0.04% per epoch): cumulatively > 1e-3 every 3 epochs -> never stops
        self.assertEqual(self.run_stopper([1.0, 0.5] + [0.5 - 0.0002 * k for k in range(1, 40)]), (None, 41))
        # non-finite loss never counts as an improvement
        self.assertEqual(self.run_stopper([1.0, np.nan, np.nan, np.nan, np.nan, np.nan]), (6, 1))

    def test_single_thread(self):
        from threadpoolctl import threadpool_info
        with tp.single_thread():
            for pool in threadpool_info():
                if pool.get("user_api") == "blas":
                    self.assertEqual(pool["num_threads"], 1)


class FakeVal:
    """The parts of a validation Cache that lag_analysis uses: episodes of the given lengths, contiguous rows."""
    time_step = 0.001

    def __init__(self, lengths):
        self.n_rows_episode = np.asarray(lengths)
        self.row_start = np.r_[0, np.cumsum(lengths)[:-1]]
        self.episodes = list(range(len(lengths)))

    def episode_slice(self, i):
        s = int(self.row_start[i])
        return slice(s, s + int(self.n_rows_episode[i]))


def shifted(label, d):
    """label[t - d] for every step t of one episode, with the padding of lag_analysis: 0 before step 0, the last
    value after the end. d > 0 delays the signal (a prediction that lags), d < 0 advances it (one that leads)."""
    src = np.arange(len(label)) - d
    out = label[np.clip(src, 0, len(label) - 1)].copy()
    out[src < 0] = 0.0
    return out


class TestLagAnalysis(unittest.TestCase):
    """Two-sided lag search on synthetic event windows (k > 0: the prediction lags the label)."""
    T = 1000

    @classmethod
    def setUpClass(cls):
        from analyze_dataset import CALIBRATION
        cls.width = {e: round(CALIBRATION["settle"][e] / FakeVal.time_step) for e in tp.EVENT_TYPES}
        t = np.arange(cls.T)
        ramp = np.clip((t - 400) / 5.0, 0.0, 1.0)                      # 5-step ramp at step 400
        cls.step_label = np.column_stack([0.25 * ramp, -0.5 * ramp])  # (v, w) labels switching at an event

    def run_lag(self, pred, label, events, clean=None, lengths=None):
        lengths = lengths or [len(label)]
        clean = np.ones(len(lengths), dtype=bool) if clean is None else np.asarray(clean)
        return tp.lag_analysis(pred, label, FakeVal(lengths), {"events": events, "width": self.width}, clean)

    def k_best(self, res, name):
        return [s["k_best"] for s in res[name]["summary"]]

    def test_delayed_and_advanced(self):
        j = tp.EVENT_TYPES.index("align>drive")
        rng = np.random.default_rng(0)
        lab = self.step_label
        for d in (0, 1, 3, 7, -1, -3, -7):
            pred = shifted(lab, d) + rng.normal(scale=1e-3, size=lab.shape)
            res = self.run_lag(pred, lab, [(j, 0, 400)])
            x = res["align>drive"]
            self.assertEqual(self.k_best(res, "align>drive"), [d, d], d)
            np.testing.assert_array_equal(x["lags"], np.arange(-tp.LAG_MAX, tp.LAG_MAX + 1))
            self.assertEqual(x["mse"].shape, (2 * tp.LAG_MAX + 1, 2))
            self.assertEqual((x["n_events"], x["n_pairs"]), (1, 20 + self.width["align>drive"] + 1))
            for s in x["summary"]:
                self.assertFalse(s["flat"] or s["boundary"])
                self.assertGreater(s["rel_range"], tp.LAG_FLAT_TOL)
                self.assertLessEqual(s["mse_best"], s["mse_k0"])
                self.assertEqual(s["mse_best"] == s["mse_k0"], d == 0)
            # the other event types have no events here
            self.assertIsNone(res["start"]["summary"][0]["k_best"])

    def test_start_first_order_response(self):
        """Constant command from step 0; the measured response is first order with time constant tau (y[0] = 0, at
        rest). The label before step 0 is 0, so every k uses the same steps and the lag is small and positive
        (left-out pairs drove it to LAG_MAX)."""
        j = tp.EVENT_TYPES.index("start")
        t = np.arange(self.T)
        u = np.array([0.25, -0.4])
        lab = np.tile(u, (self.T, 1))
        for tau in (1.0, 3.0, 10.0):
            pred = u * (1.0 - np.exp(-t / tau))[:, None]
            res = self.run_lag(pred, lab, [(j, 0, 0)])
            self.assertEqual(res["start"]["n_pairs"], self.width["start"] + 1)   # no steps before step 0
            for s in res["start"]["summary"]:
                self.assertGreaterEqual(s["k_best"], 1, tau)
                self.assertLessEqual(s["k_best"], tau, tau)
                self.assertFalse(s["boundary"] or s["flat"])
        # a pure 2-step delay, and a zero-lag prediction with a decaying offset (k = LAG_MAX before the fix)
        res = self.run_lag(shifted(lab, 2), lab, [(j, 0, 0)])
        self.assertEqual(self.k_best(res, "start"), [2, 2])
        res = self.run_lag(lab + 0.01 * np.exp(-t / 20.0)[:, None], lab, [(j, 0, 0)])
        self.assertEqual(self.k_best(res, "start"), [0, 0])

    def test_flat_curve(self):
        """A label constant across the window (v at orient>reach is 0 on both sides) gives the same MSE for every
        k: no lag is reported."""
        j = tp.EVENT_TYPES.index("orient>reach")
        rng = np.random.default_rng(1)
        lab = np.column_stack([np.zeros(self.T), np.full(self.T, 0.3)])
        pred = lab + 1e-3 + rng.normal(scale=1e-4, size=lab.shape)
        res = self.run_lag(pred, lab, [(j, 0, 300)])
        for s in res["orient>reach"]["summary"]:
            self.assertTrue(s["flat"])
            self.assertIsNone(s["k_best"])
            self.assertFalse(s["boundary"])
            self.assertLess(s["rel_range"], tp.LAG_FLAT_TOL)
            self.assertAlmostEqual(s["mse_best"], s["mse_k0"], places=15)

    def test_boundary(self):
        """A shift beyond the searched range: the minimum is on the boundary, flagged (100-step ramp, so the shifted
        ramp overlaps the window on both sides)."""
        j = tp.EVENT_TYPES.index("align>drive")
        ramp = np.clip((np.arange(self.T) - 400) / 100.0, 0.0, 1.0)
        lab = np.column_stack([0.25 * ramp, -0.5 * ramp])
        for d in (40, -40):
            res = self.run_lag(shifted(lab, d), lab, [(j, 0, 400)])
            for s in res["align>drive"]["summary"]:
                self.assertEqual(s["k_best"], int(np.sign(d)) * tp.LAG_MAX)
                self.assertTrue(s["boundary"])

    def test_clean_episodes_and_episode_end(self):
        """Only clean episodes are pooled; a lead at an event near the episode end uses the last label value."""
        j = tp.EVENT_TYPES.index("align>drive")
        n0, n1 = self.T, 450                                          # episode 1 ends 49 steps after its event
        lab = np.r_[self.step_label, self.step_label[:n1]]
        pred = np.r_[shifted(self.step_label, 3), shifted(self.step_label[:n1], -5)]
        res = self.run_lag(pred, lab, [(j, 0, 400), (j, 1, 400)], clean=[True, False], lengths=[n0, n1])
        self.assertEqual((res["align>drive"]["n_events"], self.k_best(res, "align>drive")), (1, [3, 3]))
        res = self.run_lag(pred, lab, [(j, 0, 400), (j, 1, 400)], clean=[False, True], lengths=[n0, n1])
        x = res["align>drive"]
        self.assertEqual((x["n_events"], self.k_best(res, "align>drive")), (1, [-5, -5]))
        self.assertEqual(x["n_pairs"], n1 - (400 - 20))                # window clipped at the episode end

    def test_curve_equals_direct_sum(self):
        """MSE(k) equals a direct loop over the window steps with the padding rule (0 before step 0, last value
        after the end), with the same steps for every k."""
        rng = np.random.default_rng(2)
        lengths = [300, 120]
        n = sum(lengths)
        lab, pred = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
        events = [(0, 0, 0), (1, 0, 250), (1, 1, 90), (0, 1, 0)]
        res = self.run_lag(pred, lab, events, lengths=lengths)
        va = FakeVal(lengths)
        for j, name in enumerate(tp.EVENT_TYPES[:2]):
            ref = np.zeros((2 * tp.LAG_MAX + 1, 2))
            n_t = 0
            for (jj, i, k) in events:
                if jj != j:
                    continue
                sl = va.episode_slice(i)
                L = sl.stop - sl.start
                for t in range(max(0, k - 20), min(L, k + self.width[name] + 1)):
                    n_t += 1
                    for a, kk in enumerate(range(-tp.LAG_MAX, tp.LAG_MAX + 1)):
                        s = t - kk
                        y = np.zeros(2) if s < 0 else lab[sl.start + min(s, L - 1)]
                        ref[a] += (pred[sl.start + t] - y) ** 2
            self.assertEqual(res[name]["n_pairs"], n_t)
            np.testing.assert_allclose(res[name]["mse"], ref / n_t, rtol=1e-12)

    def test_summary_rules(self):
        lags = np.arange(-tp.LAG_MAX, tp.LAG_MAX + 1)
        curve = (lags.astype(float) ** 2 - 4.0) ** 2 + 1.0      # equal minima at k = -2 and k = +2
        s = tp.lag_curve_summary(lags, curve)
        self.assertEqual((s["k_best"], s["mse_best"], s["mse_k0"]), (-2, 1.0, 17.0))
        self.assertAlmostEqual(s["rel_range"], float(curve.max()) - 1.0)
        curve = np.where(lags == 0, 1.0, 2.0)
        curve[lags == 3] = 1.0                                    # ties between 0 and 3: the smaller |k|
        self.assertEqual(tp.lag_curve_summary(lags, curve)["k_best"], 0)
        s = tp.lag_curve_summary(lags, np.full(len(lags), 2.0) + 1e-4 * (lags == 5))
        self.assertEqual((s["k_best"], s["flat"]), (None, True))   # relative range 5e-5 < LAG_FLAT_TOL
        s = tp.lag_curve_summary(lags, np.full(len(lags), np.nan))  # no events
        self.assertEqual((s["k_best"], s["mse_best"], s["flat"], s["boundary"]), (None, None, False, False))


class TestLabelConventions(unittest.TestCase):
    """Plan 3.1 (b)-(d): wheel round trip, ego invariance, sign conventions of the labels."""

    def test_wheel_round_trip(self):
        from collect_data import wheel_speeds
        from robotnik_arm_control import BaseControllerParams
        params = BaseControllerParams(nav_timeout=25.0)
        self.assertEqual((params.wheel_radius, params.wheel_base_width), (bcf.WHEEL_RADIUS, bcf.WHEEL_BASE_WIDTH))
        rng = np.random.default_rng(0)
        for v, w in rng.uniform(-1, 1, size=(50, 2)):
            v2, w2 = bcf.wheels_to_vw(wheel_speeds(v, w, params))
            self.assertAlmostEqual(float(v2), v, places=12)
            self.assertAlmostEqual(float(w2), w, places=12)

    def test_ego_invariance(self):
        rng = np.random.default_rng(1)
        n = 100
        base = rng.uniform(-3, 3, (n, 3))
        yaw = rng.uniform(-np.pi, np.pi, n)
        q, qd = rng.normal(size=(n, 17)), rng.normal(size=(n, 17))
        target = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0.2, 1.2, n)])
        ref = bcf.build_features("ego", bcf.measured_signals(base, yaw, q, qd, target))
        for phi, shift in ((0.7, (1.0, -2.0)), (-2.9, (-0.3, 0.4))):
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            b2, t2 = base.copy(), target.copy()
            b2[:, :2] = base[:, :2] @ R.T + shift
            t2[:, :2] = target[:, :2] @ R.T + shift
            yaw2 = (yaw + phi + np.pi) % (2 * np.pi) - np.pi
            got = bcf.build_features("ego", bcf.measured_signals(b2, yaw2, q, qd, t2))
            np.testing.assert_allclose(got, ref, atol=1e-9)

    def test_sign_conventions(self):
        """w > 0 turns the base to larger yaw; the first w label has the sign of the bearing beta of the turn
        the expert makes first (towards the target for M0 / M2, away from it for M1)."""
        split = tp.make_split()
        eps = split["train_episodes"][:60]
        checked, n_turns = 0, {1: 0, -1: 0}
        for e in eps:
            info = split["episodes"][str(e)]
            with np.load(os.path.join(tp.TRAIN_DIR, info["file"])) as d:
                pos, yaw, qd, wheels, target = (d[k] for k in ("base_pos", "base_bearing", "qd_mes", "wheel_cmds",
                                                               "target"))
            sig = bcf.measured_signals(pos[:1], yaw[:1], np.zeros((1, 17)), qd[:1], target)
            f = bcf.build_features("ego", sig)[0]
            beta = np.arctan2(f[5], f[4])
            if info["mode"] == "M1":  # base goal behind the start: the first turn is away from the target
                beta = (beta + 2 * np.pi) % (2 * np.pi) - np.pi
            if not (0.1 < abs(beta) < np.pi - 0.1):
                continue
            _, w0 = bcf.wheels_to_vw(wheels[0])
            self.assertEqual(np.sign(w0), np.sign(beta), e)
            # measured turning: every turn segment (>= 100 steps with |w_mes| > 0.2 rad/s, one sign) changes the
            # yaw in the direction of w (single 1 ms steps disagree at turn onsets, while the wheels spin up)
            _, w_mes = bcf.wheels_to_vw(qd[:, :4])
            dyaw = (np.diff(yaw.astype(float)) + np.pi) % (2 * np.pi) - np.pi
            for sign in (1, -1):
                turning = np.r_[False, sign * w_mes[:-1] > 0.2, False].astype(int)
                starts, stops = np.flatnonzero(np.diff(turning) == 1), np.flatnonzero(np.diff(turning) == -1)
                for a, b in zip(starts, stops):
                    if b - a >= 100:
                        self.assertGreater(sign * dyaw[a:b].sum(), 0.0, (e, a, b))
                        n_turns[sign] += 1
            checked += 1
        self.assertGreater(checked, 30)
        self.assertGreater(min(n_turns.values()), 5)


if __name__ == "__main__":
    unittest.main()
