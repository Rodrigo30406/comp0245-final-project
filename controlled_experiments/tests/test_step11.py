"""Unit tests of step 11 (results/step11/step11.py, policies/split_policy.py).

  (a) placebo row selection: the three rules, at least 1 s before the latch, deterministic per episode
  (b) the placebo cache (when built): one row per training episode, the chosen steps, nav_done False but for k0 = 0
  (c) SplitPolicy == the two networks' forward passes on the ego_novw features, (v, w) -> wheels as BCPolicy;
      wrong network roles are refused
  (d) the W models (when trained): four layers of (256, 256, 256), loaded and run by BCPolicy

Run from the task directory:
  python -m unittest tests/test_step11.py -v
"""
import json
import os
import sys
import tempfile
import unittest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (TASK_DIR, os.path.join(TASK_DIR, "policies"), os.path.join(TASK_DIR, "results", "step11")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

import bc_features as bcf  # noqa: E402
import step11  # noqa: E402
from bc_policy import BASE_PARAMS, BCPolicy  # noqa: E402
from collect_data import wheel_speeds  # noqa: E402
from split_policy import SplitPolicy, make_policy  # noqa: E402


def random_model(path, feature_set, out_names, rng, hidden=(8, 8)):
    n_in = len(bcf.FEATURE_SETS[feature_set])
    sizes = [n_in, *hidden, len(out_names)]
    W = [rng.standard_normal((a, b)) * 0.3 for a, b in zip(sizes[:-1], sizes[1:])]
    b = [rng.standard_normal(s) * 0.1 for s in sizes[1:]]
    bcf.save_model(path, feature_set, rng.standard_normal(n_in), rng.uniform(0.5, 2, n_in),
                   rng.standard_normal(len(out_names)), rng.uniform(0.5, 2, len(out_names)), W, b,
                   {"output_names": list(out_names)})


def random_obs(rng):
    return {"t": 1.0, "base_pos": np.r_[rng.uniform(-2, 2, 2), 0.1], "base_ori": np.array([0, 0, 0, 1.0]),
            "base_bearing": rng.uniform(-np.pi, np.pi), "q_mes": rng.uniform(-1, 1, 17),
            "qd_mes": rng.uniform(-0.5, 0.5, 17), "target": np.r_[rng.uniform(-3, 3, 2), rng.uniform(0.2, 1.2)]}


class TestPlacebo(unittest.TestCase):
    def test_rules(self):
        nav = np.zeros(5000, bool)
        nav[3000:] = True
        for e in range(50):
            k, k0, rule = step11.placebo_step(nav, e)
            self.assertEqual((k0, rule), (3000, "moving>=1s"))
            self.assertTrue(0 <= k <= k0 - step11.PLACEBO_GAP and not nav[k])
        self.assertEqual(step11.placebo_step(nav, 7), step11.placebo_step(nav, 7))    # deterministic per episode
        self.assertGreater(len({step11.placebo_step(nav, e)[0] for e in range(50)}), 40)
        short = np.zeros(1500, bool)
        short[500:] = True
        k, k0, rule = step11.placebo_step(short, 3)
        self.assertEqual((k0, rule), (500, "before_k0"))
        self.assertLess(k, 500)
        k, k0, rule = step11.placebo_step(np.ones(100, bool), 747)
        self.assertEqual((k0, rule), (0, "k0_zero_any"))
        self.assertGreaterEqual(k, 1)

    @unittest.skipUnless(os.path.exists(os.path.join(step11.PLACEBO_DIR, "manifest.json")), "placebo cache not built")
    def test_cache(self):
        with open(os.path.join(step11.PLACEBO_DIR, "manifest.json")) as f:
            m = json.load(f)
        cols, _ = step11.load_placebo_rows(m["episodes"])
        self.assertEqual(len(cols["episode"]), 1613)
        np.testing.assert_array_equal(cols["step"], m["k0_steps"])
        nav_eps = {int(e) for e, nd in zip(cols["episode"], cols["nav_done"]) if nd}
        self.assertLessEqual(nav_eps, set(m["k0_zero_episodes"]))
        gaps = [k0 - k for k, k0, r in zip(m["k0_steps"], m["onset_k0_steps"], m["placebo_rule_per_episode"])
                if r == "moving>=1s"]
        self.assertGreaterEqual(min(gaps), step11.PLACEBO_GAP)


class TestSplitPolicy(unittest.TestCase):
    def test_matches_forward(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            base, arm = os.path.join(d, "base.npz"), os.path.join(d, "arm.npz")
            random_model(base, "ego_target", ["v", "w"], rng)
            random_model(arm, "ego_novw", bcf.OUTPUT_NAMES[2:], rng)
            pol = make_policy(f"base={base};arm={arm}")
            mb, ma = bcf.load_model(base), bcf.load_model(arm)
            for _ in range(20):
                obs = random_obs(rng)
                wheels, tau = pol.act(obs)
                sig = bcf.measured_signals(obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"],
                                           obs["target"])
                F = bcf.build_features("ego_novw", sig)
                vw = bcf.forward(mb, F[:, :6])[0]
                np.testing.assert_allclose(wheels, wheel_speeds(vw[0], vw[1], BASE_PARAMS), rtol=0, atol=1e-12)
                np.testing.assert_allclose(tau, bcf.forward(ma, F)[0], rtol=0, atol=1e-12)
            with self.assertRaises(ValueError):
                SplitPolicy(arm, base)          # roles swapped
            with self.assertRaises(ValueError):
                make_policy(f"base={base}")     # arm missing

    def test_target_features_are_the_first_six(self):
        self.assertEqual(bcf.FEATURE_SETS["ego_target"], bcf.FEATURE_SETS["ego_novw"][:6])
        self.assertFalse(any(n.startswith("q") for n in bcf.FEATURE_SETS["ego_target"]))
        rng = np.random.default_rng(2)
        obs = random_obs(rng)
        sig = bcf.measured_signals(obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"], obs["target"])
        np.testing.assert_array_equal(bcf.build_features("ego_target", sig), bcf.build_features("ego_novw", sig)[:, :6])

    @unittest.skipUnless(os.path.exists(os.path.join(step11.TRAINED_DIR, "S-base_s0.npz")), "S models not trained")
    def test_trained_split_models(self):
        pol = make_policy(f"base={step11.TRAINED_DIR}/S-base_s0.npz;arm={step11.TRAINED_DIR}/S-arm-K16_s0.npz")
        wheels, tau = pol.act(random_obs(np.random.default_rng(3)))
        self.assertEqual((wheels.shape, tau.shape), ((4,), (7,)))
        self.assertEqual(pol.base["meta"]["n_train_rows"] - pol.base["meta"]["n_train_rows_base"], 16 * 1613)


class TestWModels(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(os.path.join(step11.TRAINED_DIR, "W-K16_s0.npz")), "W models not trained")
    def test_w_loads_in_bcpolicy(self):
        rng = np.random.default_rng(1)
        for v in ("K0", "K16"):
            path = os.path.join(step11.TRAINED_DIR, f"W-{v}_s0.npz")
            m = bcf.load_model(path)
            self.assertEqual([w.shape[1] for w in m["weights"]], [256, 256, 256, 9])
            self.assertEqual(m["meta"]["hyperparameters"]["hidden_layer_sizes"], [256, 256, 256])
            wheels, tau = BCPolicy(path).act(random_obs(rng))
            self.assertEqual((wheels.shape, tau.shape), ((4,), (7,)))


if __name__ == "__main__":
    unittest.main()
