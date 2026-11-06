"""Unit tests of analyze_rollouts.py (module D): statistics against known answers (and the null calibration of the
seed-level t test at S = 5), the failure classification and its precedence, the merged classes, run loading, the
integrity checks of paired runs, hybrid-log parsing, the experiment-A / B / descriptive reports on a synthetic results
tree, and figure smoke tests. No simulation and no dataset is opened (synthetic data only)."""
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_rollouts as ar  # noqa: E402

NAN = float("nan")
REAL_OFFLINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained",
                            "ego-ols.offline.json")


# ------------------------------------------------------------------------------
# synthetic rows
# ------------------------------------------------------------------------------
def make_row(episode=0, **kw):
    """A row with every v1 / v1.1 column the analysis uses; default = a clean success."""
    row = {
        "episode": episode, "success": True, "settled": True, "t_reach": 9.0, "stop_reason": "reached",
        "min_error": 0.004, "hold_error": 0.006, "hold_error_max": 0.009, "final_error": 0.005, "duration": 10.0,
        "n_steps": 10000, "r": 1.5, "theta_deg": 30.0, "target_z": 0.6, "d0": 0.85, "standoff": 0.65,
        "base_path_length": 1.0, "base_turn_total_deg": 40.0, "policy_ms_mean": 0.01, "policy_ms_p99": 0.02,
        "hold10_stop": "completed", "held_10s": True, "stable_5cm_time": 8.0, "arm_extend_time": 7.0,
        "ee_base_disp_max": 0.3, "ring_dist_start": 0.85, "ring_dist_min": 0.02, "ring_dist_final": 0.03,
        "parked_time": 6.5, "wheel_over_envelope_frac": 0.0, "error_message": "",
    }
    row.update(kw)
    return row


REF = {"align_req_deg": 30.0, "orient_req_deg": 0.0, "navdone_ee_err": 0.30}


def failure(**kw):
    base = dict(success=False, stop_reason="timeout", min_error=0.20, held_10s=False, stable_5cm_time=NAN,
                parked_time=NAN)
    base.update(kw)
    return make_row(**base)


V11_HEADER = list(make_row().keys())


def write_run(root, name, rows, eval_set="data/dev", policy="x.py:make_policy", policy_arg=None, protocol="v1.1",
              logs=None, n_episodes="rows", no_early_stop=None):
    """A run folder as evaluate_policy writes it. n_episodes: the full size of the evaluation set in config.json
    ('rows' = the number of rows, None = left out); no_early_stop: default True for a hybrid policy_arg."""
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    header = list(rows[0].keys())
    with open(os.path.join(path, "rollouts.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})
    if no_early_stop is None:
        no_early_stop = "base=" in str(policy_arg)
    config = {"args": {"policy": policy, "policy_arg": policy_arg, "eval_set": eval_set, "no_early_stop": no_early_stop},
              "protocol": {"PROTOCOL_VERSION": protocol}}
    if n_episodes is not None:
        config["eval_set_config"] = {"n_episodes": len(rows) if n_episodes == "rows" else n_episodes}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    for e, text in (logs or {}).items():
        with open(os.path.join(path, "logs", f"ep_{e:04d}.log"), "w") as f:
            f.write(text)
    return path


def write_ref(out_dir, set_name, d0, theta, navdone_ee_err=0.30, navdone_time=6.0):
    rows = []
    for e, (d, th) in enumerate(zip(d0, theta)):
        align, orient = ar.ad.required_turns(np.deg2rad(th), d)
        rows.append({"episode": e, "noisy": False, "expert_status": "reached", "r": d + 0.65, "theta_deg": th, "z": 0.6,
                     "standoff": 0.65, "d0": d, "mode": ar.ad.mode_of(d), "align_req_deg": np.rad2deg(align),
                     "orient_req_deg": np.rad2deg(orient), "navdone_step": 6000, "navdone_time": navdone_time,
                     "navdone_time_meta": navdone_time, "navdone_ee_err": navdone_ee_err, "init_x": 0, "init_y": 0,
                     "init_yaw_deg": 0, "target_x": 1, "target_y": 0, "cov_pct_world": (e % 10) / 10 + 0.05,
                     "cov_nn_world": 0.5, "cov_pct_rel": ((e * 3) % 10) / 10 + 0.05, "cov_nn_rel": 0.1})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"cache_ref_{set_name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ar.REF_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------------------
class TestStatistics(unittest.TestCase):
    def test_wilson_known_answers(self):
        lo, hi = ar.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 0.27753, places=4)
        lo, hi = ar.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.23659, places=4)
        self.assertAlmostEqual(hi, 0.76341, places=4)
        lo, hi = ar.wilson(298, 300)  # EVALUATION.md: 99.3% [97.6, 99.8]
        self.assertAlmostEqual(100 * lo, 97.6, places=1)
        self.assertAlmostEqual(100 * hi, 99.8, places=1)
        self.assertTrue(all(np.isnan(ar.wilson(0, 0))))

    def test_mcnemar_matches_binomtest(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(1, 120))
            a = rng.random(n) < rng.random()
            b = rng.random(n) < rng.random()
            n01, n10, p = ar.mcnemar_exact(a, b)
            self.assertEqual(n01, int(np.sum(~a & b)))
            self.assertEqual(n10, int(np.sum(a & ~b)))
            ref = 1.0 if n01 + n10 == 0 else stats.binomtest(n01, n01 + n10, 0.5).pvalue
            self.assertAlmostEqual(p, ref, places=10)
        self.assertEqual(ar.mcnemar_exact([1, 0, 1], [1, 0, 1])[2], 1.0)

    def test_cochran_q_hand_example(self):
        # seeds (treatments) x episodes (blocks); column totals 3, 2, 1; row totals 2, 1, 3, 0:
        # Q = (k-1)(k sum C^2 - N^2) / (k N - sum R^2) = 2 (3*14 - 36) / (18 - 14) = 3, df 2, p = exp(-1.5)
        Y = np.array([[1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]])
        q, df, p = ar.cochran_q(Y)
        self.assertAlmostEqual(q, 3.0, places=12)
        self.assertEqual(df, 2)
        self.assertAlmostEqual(p, math.exp(-1.5), places=12)
        # every episode decided the same way by all seeds: no evidence of a seed difference
        q, df, p = ar.cochran_q(np.array([[1, 0, 1], [1, 0, 1]]))
        self.assertEqual((q, p), (0.0, 1.0))
        # two seeds: Q equals the McNemar chi-square (b - c)^2 / (b + c) without continuity correction
        rng = np.random.default_rng(0)
        Y2 = (rng.random((2, 80)) < 0.5).astype(int)
        b = np.sum((Y2[0] == 1) & (Y2[1] == 0))
        c = np.sum((Y2[0] == 0) & (Y2[1] == 1))
        self.assertAlmostEqual(ar.cochran_q(Y2)[0], (b - c) ** 2 / (b + c), places=10)
        self.assertTrue(np.isnan(ar.cochran_q(np.ones((1, 5)))[0]))

    def test_holm_textbook(self):
        adj, rej = ar.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(list(rej), [True, False, False, True])
        adj, rej = ar.holm([0.5, np.nan, 0.01])  # NaN left out of the family (m = 2)
        np.testing.assert_allclose(adj[[0, 2]], [0.5, 0.02])
        self.assertTrue(np.isnan(adj[1]))
        adj, _ = ar.holm([0.6, 0.7])
        np.testing.assert_allclose(adj, [1.0, 1.0])

    def test_holm_non_strict(self):
        # textbook Holm rejects at p_(i) <= alpha / (m - i + 1): a tie at exactly 0.05 is rejected
        adj, rej = ar.holm([0.025, 0.3, 0.01])
        np.testing.assert_allclose(adj, [0.05, 0.3, 0.03])
        self.assertEqual(list(rej), [True, False, True])
        self.assertTrue(ar.holm([0.05])[1][0])
        self.assertFalse(ar.holm([0.0500001])[1][0])

    def test_seed_t_test_matches_scipy(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            na, nb = int(rng.integers(2, 9)), int(rng.integers(2, 9))
            ra, rb = rng.uniform(0, 1, na), rng.uniform(0, 1, nb)
            t = ar.seed_t_test(ra, rb)
            ref = stats.ttest_ind(rb, ra, equal_var=False)
            ci = ref.confidence_interval(0.95)
            self.assertEqual(t["test"], "welch")
            self.assertAlmostEqual(t["diff"], rb.mean() - ra.mean(), places=12)
            self.assertAlmostEqual(t["p"], ref.pvalue, places=10)
            self.assertAlmostEqual(t["df"], ref.df, places=8)
            self.assertAlmostEqual(t["lo"], ci.low, places=10)
            self.assertAlmostEqual(t["hi"], ci.high, places=10)
        # a single deterministic run (expert / EE cell): one-sample t on the per-seed differences, S - 1 df
        rb = np.array([0.62, 0.55, 0.70, 0.48, 0.66])
        t = ar.seed_t_test([0.9], rb)
        ref = stats.ttest_1samp(rb - 0.9, 0.0)
        self.assertEqual((t["test"], t["df"]), ("one-sample", 4.0))
        self.assertAlmostEqual(t["p"], ref.pvalue, places=10)
        self.assertAlmostEqual(t["lo"], ref.confidence_interval(0.95).low, places=10)
        self.assertAlmostEqual(ar.seed_t_test(rb, [0.9])["diff"], 0.9 - rb.mean())  # sign B - A either way
        # degenerate cases
        self.assertTrue(np.isnan(ar.seed_t_test([0.5], [0.6])["p"]))  # one run each: undefined
        z = ar.seed_t_test([0.5, 0.5], [0.5, 0.5])
        self.assertEqual((z["lo"], z["hi"], z["p"]), (0.0, 0.0, 1.0))
        z = ar.seed_t_test([0.5, 0.5], [0.6, 0.6])
        self.assertAlmostEqual(z["lo"], 0.1)
        self.assertEqual(z["p"], 0.0)

    def test_combined_p(self):
        self.assertEqual(ar.combined_p(0.01, 0.04), 0.04)
        self.assertEqual(ar.combined_p(0.2, 0.04), 0.2)
        self.assertTrue(np.isnan(ar.combined_p(0.01, np.nan)))  # t undefined: out of the Holm family

    def test_null_calibration_s5_large_seed_effect(self):
        """S = 5 seeds per arm with a large seed effect (logistic, seed SD of success about 20 pp) and a shared
        episode effect, true difference 0. The Welch interval on the per-seed rates rejects near the nominal 5%; the
        two-level percentile bootstrap rejects about twice as often (step-3 review: 0.09-0.12 at E = 300), which is
        why compare / split2x2 report the t test next to it. Only loose bounds are asserted (600 simulations)."""
        rng = np.random.default_rng(20261029)
        S, E, n_sim = 5, 100, 600
        rej_t = rej_boot = lo_t = lo_boot = 0
        for i in range(n_sim):
            b = rng.normal(0, 2.0, E)
            pa = 1 / (1 + np.exp(-(rng.normal(0, 1.6, S)[:, None] + b[None, :])))
            pb = 1 / (1 + np.exp(-(rng.normal(0, 1.6, S)[:, None] + b[None, :])))
            A = (rng.random(pa.shape) < pa).astype(float)
            B = (rng.random(pb.shape) < pb).astype(float)
            t = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B))
            d = ar.paired_diff(A, B, n_boot=400, seed=i)
            rej_t += t["p"] < 0.05
            lo_t += t["lo"] > 0
            rej_boot += d["p"] < 0.05
            lo_boot += d["lo"] > 0
        rate_t, rate_boot = rej_t / n_sim, rej_boot / n_sim
        print(f"\nS=5 null: Welch p<0.05 {rate_t:.3f}, CI lo>0 {lo_t / n_sim:.3f}; bootstrap p<0.05 {rate_boot:.3f}, "
              f"CI lo>0 {lo_boot / n_sim:.3f} (nominal 0.05 / 0.025)")
        self.assertGreater(rate_t, 0.02)
        self.assertLess(rate_t, 0.08)
        self.assertLess(lo_t / n_sim, 0.05)
        self.assertGreater(rate_boot, 0.05)   # the bootstrap's documented anti-conservativeness ...
        self.assertLess(rate_boot, 0.20)
        self.assertGreater(rate_boot, rate_t)  # ... relative to the calibrated companion

    def test_psi_edge_cases(self):
        Y = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])  # all seeds agree, mixed outcomes
        r = ar.seed_lottery(Y)
        self.assertEqual(r["psi_raw"], 0.0)
        self.assertEqual(r["psi"], 0.0)
        self.assertAlmostEqual(r["psi_indep"], 1 - 0.5 ** 3 - 0.5 ** 3)
        self.assertTrue(np.isnan(ar.seed_lottery(np.ones((3, 5)))["psi"]))  # psi_indep = 0
        self.assertTrue(np.isnan(ar.seed_lottery(np.ones((1, 5)))["psi"]))
        rng = np.random.default_rng(7)
        p = np.array([0.3, 0.5, 0.6, 0.8])
        Y = rng.random((4, 20000)) < p[:, None]  # seeds independent of the episode
        self.assertAlmostEqual(ar.seed_lottery(Y)["psi"], 1.0, delta=0.03)

    def test_bootstrap_degenerate(self):
        r = ar.boot2(np.ones((4, 50)), n_boot=500)
        self.assertEqual((r["mean"], r["lo"], r["hi"]), (1.0, 1.0, 1.0))
        r = ar.boot2(np.zeros((1, 50)), n_boot=500)
        self.assertEqual((r["lo"], r["hi"]), (0.0, 0.0))
        rng = np.random.default_rng(1)
        A = (rng.random((5, 120)) < 0.4).astype(float)
        d = ar.paired_diff(A, A.copy(), n_boot=2000)
        self.assertEqual(d["diff"], 0.0)
        self.assertLessEqual(d["lo"], 0.0)
        self.assertGreaterEqual(d["hi"], 0.0)
        d1 = ar.paired_diff(A[:1], A[:1].copy(), n_boot=500)  # one seed each: every draw is exactly 0
        self.assertEqual((d1["lo"], d1["hi"], d1["p"]), (0.0, 0.0, 1.0))

    def test_bootstrap_direction_and_p(self):
        rng = np.random.default_rng(2)
        A = (rng.random((4, 200)) < 0.3).astype(float)
        B = (rng.random((4, 200)) < 0.7).astype(float)
        d = ar.paired_diff(A, B, n_boot=2000)
        self.assertGreater(d["lo"], 0.0)
        self.assertAlmostEqual(d["p"], 1 / 2000)  # floored at 1 / n_boot
        self.assertAlmostEqual(d["diff"], B.mean() - A.mean())
        self.assertEqual(ar.boot_p(np.array([-1.0, 1.0, 2.0, 3.0])), 0.5)
        # fixed rng seed: reproducible
        self.assertEqual(ar.boot2(A, n_boot=300, seed=5)["lo"], ar.boot2(A, n_boot=300, seed=5)["lo"])

    def test_bootstrap_coverage(self):
        """Two-level bootstrap CI covers the population mean (0.5 by symmetry) about 95% of the time when the
        outcome depends on the episode (Beta(0.5, 0.5) probabilities) and on the seed (shift U(-0.15, 0.15))."""
        rng = np.random.default_rng(1)
        hits, n_sim = 0, 300
        for i in range(n_sim):
            pe = rng.beta(0.5, 0.5, 100)
            a = rng.uniform(-0.15, 0.15, 8)
            Y = (rng.random((8, 100)) < np.clip(pe[None, :] + a[:, None], 0, 1)).astype(float)
            r = ar.boot2(Y, n_boot=1000, seed=i)
            hits += r["lo"] <= 0.5 <= r["hi"]
        self.assertGreater(hits / n_sim, 0.90)
        self.assertLess(hits / n_sim, 0.99)

    def test_strata_holm_family(self):
        # the Holm family is the strata with n >= 30; the others are descriptive (p shown, no Holm)
        rng = np.random.default_rng(8)
        d0 = np.r_[np.full(10, 0.0), np.full(40, -0.5), np.full(35, 0.3), np.full(12, 0.7), np.full(50, 1.5)]
        A = (rng.random((3, d0.size)) < 0.4).astype(float)
        B = (rng.random((3, d0.size)) < 0.6).astype(float)
        st = ar.strata_analysis(A, B, d0, n_boot=500)
        by = {s_["name"]: s_ for s_ in st["strata"]}
        names = ("M0", "M1", "M2 0.08-0.5", "M2 0.5-1.0", "M2 >= 1.0")
        self.assertEqual([by[n]["n"] for n in names], [10, 40, 35, 12, 50])
        self.assertEqual([by[n]["in_holm"] for n in names], [False, True, True, False, True])
        self.assertEqual(st["n_holm"], 3)
        for n in ("M0", "M2 0.5-1.0"):
            self.assertTrue(np.isfinite(by[n]["p"]))
            self.assertTrue(np.isnan(by[n]["p_holm"]))
            self.assertFalse(by[n]["reject"])
        big = [by[n] for n in ("M1", "M2 0.08-0.5", "M2 >= 1.0")]
        adj, _ = ar.holm([s_["p"] for s_ in big])
        np.testing.assert_allclose([s_["p_holm"] for s_ in big], adj)
        self.assertEqual(st["worst_hi_n30"][0], min(big, key=lambda s_: s_["hi"])["name"])

    def test_logistic_slope(self):
        rng = np.random.default_rng(4)
        x = rng.random(400)
        p = 1 / (1 + np.exp(-(-1.0 + 3.0 * x)))
        Y = (rng.random((3, 400)) < p[None, :]).astype(float)
        r = ar.logistic_slope(Y, x, n_boot=200)
        self.assertLess(r["lo"], 3.0)
        self.assertGreater(r["hi"], 3.0)
        self.assertTrue(np.isnan(ar.logistic_slope(np.ones((2, 50)), x[:50], n_boot=20)["slope"]))

    def test_metrics(self):
        row = make_row(min_error=0.015)
        self.assertEqual(ar.metric_value(row, "reach_2cm"), 1.0)
        self.assertEqual(ar.metric_value(make_row(min_error=0.03), "reach_2cm"), 0.0)
        self.assertEqual(ar.metric_value(make_row(min_error=0.03), "reach_5cm"), 1.0)
        self.assertEqual(ar.metric_value(make_row(parked_time=NAN), "parked"), 0.0)
        self.assertEqual(ar.metric_value(make_row(arm_extend_time=NAN), "arm_extended"), 0.0)
        self.assertEqual(ar.metric_value(make_row(held_10s=False), "held_10s"), 0.0)
        self.assertEqual(ar.metric_value(make_row(stable_5cm_time=3.0), "stable_5cm"), 1.0)
        self.assertEqual(ar.metric_value(make_row(success=False), "success"), 0.0)
        # arm_extended_toward: extended AND min_error < navdone_ee_err - 0.02 (reaching, not a collapsing arm)
        toward = dict(arm_extend_time=1.0, ref_navdone_ee_err=0.30)
        self.assertEqual(ar.metric_value(make_row(min_error=0.27, **toward), "arm_extended_toward"), 1.0)
        self.assertEqual(ar.metric_value(make_row(min_error=0.29, **toward), "arm_extended_toward"), 0.0)
        self.assertEqual(ar.metric_value(make_row(min_error=0.29, **toward), "arm_extended"), 1.0)
        self.assertEqual(ar.metric_value(make_row(min_error=0.01, arm_extend_time=NAN, ref_navdone_ee_err=0.30),
                                         "arm_extended_toward"), 0.0)
        self.assertEqual(ar.metric_value(make_row(min_error=0.01, arm_extend_time=1.0), "arm_extended_toward"), 0.0)


# ------------------------------------------------------------------------------
class TestClassification(unittest.TestCase):
    def cls(self, row, ref=REF):
        return ar.classify_row(row, ref)

    def test_every_class(self):
        self.assertEqual(self.cls(make_row()), "success")
        for reason in ("tipped", "invalid_action", "policy_error"):
            self.assertEqual(self.cls(failure(stop_reason=reason)), "crash")
        self.assertEqual(self.cls(failure(stop_reason="harness_error")), "harness_error")
        never = dict(ring_dist_start=1.0, ring_dist_min=0.5)
        self.assertEqual(self.cls(failure(base_path_length=0.01, base_turn_total_deg=2.0, ring_dist_final=1.0, **never)),
                         "no_start")
        self.assertEqual(self.cls(failure(ring_dist_final=2.5, **never)), "runaway")
        self.assertEqual(self.cls(failure(ring_dist_final=1.2, **never)), "never_other")
        appr = dict(ring_dist_start=1.0, ring_dist_min=0.05)
        self.assertEqual(self.cls(failure(base_turn_total_deg=30 + 361, ring_dist_final=0.05, **appr)), "circling")
        self.assertEqual(self.cls(failure(ring_dist_final=0.5, **appr)), "left_or_overshot")
        self.assertEqual(self.cls(failure(ring_dist_final=0.05, arm_extend_time=NAN, min_error=0.31, **appr)),
                         "parked_no_arm")
        self.assertEqual(self.cls(failure(ring_dist_final=0.05, min_error=0.03, **appr)), "extended_not_reached")
        self.assertEqual(self.cls(failure(ring_dist_final=0.05, min_error=0.2, **appr)), "approached_other")

    def test_precedence(self):
        # crash before everything (even a no-start pattern)
        self.assertEqual(self.cls(failure(stop_reason="tipped", ring_dist_min=0.5, base_path_length=0.0,
                                          base_turn_total_deg=0.0)), "crash")
        # no_start before runaway
        self.assertEqual(self.cls(failure(ring_dist_start=1.0, ring_dist_min=0.5, ring_dist_final=3.0,
                                          base_path_length=0.0, base_turn_total_deg=0.0)), "no_start")
        # never approached: circling / parked predicates are not used
        self.assertEqual(self.cls(failure(ring_dist_start=1.0, ring_dist_min=0.31, ring_dist_final=1.0,
                                          base_turn_total_deg=1000.0, arm_extend_time=NAN, min_error=0.30)),
                         "never_other")
        appr = dict(ring_dist_start=1.0, ring_dist_min=0.30)  # <= 0.3: approached
        # circling before left_or_overshot before parked_no_arm before extended_not_reached
        self.assertEqual(self.cls(failure(base_turn_total_deg=400.0, ring_dist_final=1.0, **appr)), "circling")
        self.assertEqual(self.cls(failure(base_turn_total_deg=390.0, ring_dist_final=1.0, **appr)),
                         "left_or_overshot")  # excess 360 is not > 360
        self.assertEqual(self.cls(failure(ring_dist_final=0.31, arm_extend_time=NAN, min_error=0.30, **appr)),
                         "left_or_overshot")
        self.assertEqual(self.cls(failure(ring_dist_final=0.30, arm_extend_time=NAN, min_error=0.30, **appr)),
                         "parked_no_arm")
        # short-reach episode (navdone_ee_err < 0.07): parked_no_arm cannot fire, the near miss is extended_not_reached
        self.assertEqual(self.cls(failure(ring_dist_final=0.1, arm_extend_time=NAN, min_error=0.03),
                                  {**REF, "navdone_ee_err": 0.04}), "extended_not_reached")
        # parked_no_arm needs the arm never extended and the min error close to the expert's parking error
        self.assertEqual(self.cls(failure(ring_dist_final=0.1, arm_extend_time=5.0, min_error=0.30)),
                         "approached_other")
        self.assertEqual(self.cls(failure(ring_dist_final=0.1, arm_extend_time=NAN, min_error=0.25)),
                         "approached_other")
        # extended_not_reached bounds: [0.01, 0.05)
        self.assertEqual(self.cls(failure(ring_dist_final=0.1, min_error=0.01)), "extended_not_reached")
        self.assertEqual(self.cls(failure(ring_dist_final=0.1, min_error=0.05)), "approached_other")
        # required turns of the episode are subtracted (M1: align + 180 deg orient, the orient turn only once parked)
        m1 = {"align_req_deg": 150.0, "orient_req_deg": 180.0, "navdone_ee_err": 0.3}
        self.assertEqual(self.cls(failure(base_turn_total_deg=600.0, ring_dist_final=0.05, parked_time=12.0), m1),
                         "approached_other")
        self.assertEqual(self.cls(failure(base_turn_total_deg=700.0, ring_dist_final=0.05, parked_time=12.0), m1),
                         "circling")
        # an M1 base circling its goal without parking (the expert's M1 failures): only the align turn is owed
        self.assertEqual(self.cls(failure(base_turn_total_deg=520.0, ring_dist_final=0.05), m1), "circling")
        self.assertEqual(self.cls(failure(base_turn_total_deg=500.0, ring_dist_final=0.05), m1), "approached_other")

    def test_no_start_before_approach_test(self):
        # review decision: no_start (motion test) comes before the approach test for every start >= 0.15 m from the
        # ring, so a still robot starting 0.15-0.30 m from the ring (approached by construction) is no_start
        def still(start, **kw):
            return failure(**{**dict(ring_dist_start=start, ring_dist_min=start, ring_dist_final=start,
                                     base_path_length=0.0, base_turn_total_deg=0.0, min_error=0.4), **kw})
        self.assertEqual(self.cls(still(0.2)), "no_start")
        self.assertEqual(self.cls(still(0.15)), "no_start")  # boundary: >= START_ON_RING
        self.assertEqual(self.cls(still(0.25, arm_extend_time=NAN, min_error=0.30)), "no_start")  # not parked_no_arm
        # within 0.15 m (M0-like): never no_start, classified on the approached branch and flagged
        p = ar.failure_predicates(still(0.1), REF)
        self.assertTrue(p["no_motion"])
        self.assertFalse(p["no_start"])
        self.assertEqual(self.cls(still(0.149)), "approached_other")
        # motion thresholds: path 0.15 m (above the zero policy's 0.108 m drift) or turn 10 deg is motion
        self.assertEqual(self.cls(failure(ring_dist_start=0.2, ring_dist_min=0.2, ring_dist_final=0.2,
                                          base_path_length=0.12, base_turn_total_deg=0.0, min_error=0.4)), "no_start")
        self.assertEqual(self.cls(failure(ring_dist_start=0.2, ring_dist_min=0.2, ring_dist_final=0.2,
                                          base_path_length=0.15, base_turn_total_deg=0.0, min_error=0.4)),
                         "approached_other")
        self.assertEqual(self.cls(failure(ring_dist_start=0.2, ring_dist_min=0.2, ring_dist_final=0.2,
                                          base_path_length=0.0, base_turn_total_deg=10.0, min_error=0.4)),
                         "approached_other")
        # crash still comes first
        self.assertEqual(self.cls(still(0.2, stop_reason="tipped")), "crash")
        # a near-ring start that drives away is left_or_overshot (runaway stays in the never-approached branch)
        self.assertEqual(self.cls(failure(ring_dist_start=0.2, ring_dist_min=0.2, ring_dist_final=2.5,
                                          base_path_length=3.0)), "left_or_overshot")
        self.assertEqual(self.cls(failure(ring_dist_start=1.0, ring_dist_min=0.9, ring_dist_final=2.5,
                                          base_path_length=3.0)), "runaway")

    def test_no_progress(self):
        # never approached, turned in place (not no_start) and never got NO_PROGRESS closer to the ring than at the start
        def turned(start, ring_min, final, **kw):
            return failure(**{**dict(ring_dist_start=start, ring_dist_min=ring_min, ring_dist_final=final,
                                     base_path_length=0.12, base_turn_total_deg=95.0, min_error=0.8), **kw})
        self.assertEqual(self.cls(turned(1.0, 0.95, 1.0)), "no_progress")
        self.assertEqual(self.cls(turned(1.0, 0.901, 1.2)), "no_progress")
        self.assertEqual(self.cls(turned(1.0, 0.89, 1.2)), "never_other")   # got 11 cm closer: some progress
        self.assertEqual(self.cls(turned(1.0, 0.95, 2.1)), "runaway")       # runaway comes first
        self.assertEqual(self.cls(turned(0.25, 0.25, 0.25)), "approached_other")   # approached by construction
        p = ar.failure_predicates(make_row(), REF)                           # a success: never fires
        self.assertFalse(p["no_progress"])

    def test_parked_no_arm_needs_long_reach(self):
        # parked_no_arm needs navdone_ee_err >= EXTENDED_HI + PARKED_NO_ARM_TOL = 0.07 m
        self.assertAlmostEqual(ar.PARKED_NO_ARM_MIN_NAVDONE, 0.07)
        parked = dict(ring_dist_start=1.0, ring_dist_min=0.05, ring_dist_final=0.05, arm_extend_time=NAN)
        self.assertEqual(self.cls(failure(min_error=0.06, **parked), {**REF, "navdone_ee_err": 0.07}), "parked_no_arm")
        self.assertEqual(self.cls(failure(min_error=0.06, **parked), {**REF, "navdone_ee_err": 0.0699}),
                         "approached_other")
        self.assertEqual(self.cls(failure(min_error=0.02, **parked), {**REF, "navdone_ee_err": 0.03}),
                         "extended_not_reached")
        # an unknown reference never gives parked_no_arm
        self.assertEqual(self.cls(failure(min_error=0.3, **parked), {**REF, "navdone_ee_err": NAN}), "approached_other")

    def test_flags_and_class_tables(self):
        tmp = tempfile.mkdtemp()
        try:
            rows = [make_row(0),
                    failure(episode=1, ring_dist_start=0.1, ring_dist_min=0.1, ring_dist_final=0.1, base_path_length=0.0,
                            base_turn_total_deg=0.0),                                       # M0-like, did not move
                    failure(episode=2, ring_dist_start=0.1, ring_dist_min=0.05, ring_dist_final=0.05),   # M0-like, moved
                    failure(episode=3, ring_dist_min=0.05, ring_dist_final=0.05, min_error=0.03, arm_extend_time=NAN),
                    failure(episode=4, ring_dist_start=0.2, ring_dist_min=0.2, ring_dist_final=0.2, base_path_length=0.0,
                            base_turn_total_deg=0.0)]
            run = ar.load_run(write_run(tmp, "ego_s0_dev", rows))
            ref = {e: dict(REF) for e in range(5)}
            ref[3]["navdone_ee_err"] = 0.04  # short-reach episode
            ar.classify_run(run, ref)
            by = run.by_episode
            self.assertEqual([by[e]["fclass"] for e in range(5)],
                             ["success", "approached_other", "approached_other", "extended_not_reached", "no_start"])
            self.assertEqual([by[e]["no_motion_near_ring"] for e in range(5)], [False, True, False, False, False])
            self.assertEqual([by[e]["short_reach"] for e in range(5)], [False, False, False, True, False])
            text = "\n".join(ar.class_summary_lines(run))
            self.assertIn("no_motion_near_ring (base did not move) 1", text)
            self.assertIn("failures on short-reach episodes", text)
            self.assertIn("1 (of 1 short-reach episodes): extended_not_reached 1", text)
        finally:
            shutil.rmtree(tmp)

    def test_merged_mapping(self):
        expected = {"crash": "crash", "no_start": "base_not", "runaway": "base_not", "no_progress": "base_not",
                    "never_other": "base_not",
                    "circling": "base_not", "left_or_overshot": "base_not", "parked_no_arm": "base_ok_ee_not",
                    "extended_not_reached": "base_ok_ee_not", "approached_other": "base_ok_ee_not"}
        for c in ar.FAILURE_CLASSES:
            self.assertEqual(ar.merged_class(c, False), expected[c])
        self.assertEqual(set(expected), set(ar.FAILURE_CLASSES))
        self.assertEqual(ar.merged_class("success", True), "arrive_stay")
        self.assertEqual(ar.merged_class("success", False), "arrive_leave")
        self.assertEqual(ar.merged_class("success", NAN), "arrive_leave")
        self.assertEqual(ar.merged_class("success", None), "arrive")
        self.assertEqual(set(ar.MERGED_CLASSES), set(expected.values()) | {"arrive_stay", "arrive_leave"})

    def test_classify_run_and_missing_columns(self):
        tmp = tempfile.mkdtemp()
        try:
            rows = [make_row(0), failure(episode=1, ring_dist_min=0.05, ring_dist_final=0.5)]
            run = ar.load_run(write_run(tmp, "ego_s0_dev", rows))
            ar.classify_run(run, {0: REF, 1: REF})
            self.assertEqual([r["fclass"] for r in run.rows], ["success", "left_or_overshot"])
            self.assertEqual([r["mclass"] for r in run.rows], ["arrive_stay", "base_not"])
            v1 = [{k: v for k, v in r.items() if k not in ("ring_dist_start", "ring_dist_min", "ring_dist_final",
                                                            "arm_extend_time", "held_10s")} for r in rows]
            run = ar.load_run(write_run(tmp, "expert_dev", v1, policy="expert", protocol="v1"))
            ar.classify_run(run, {0: REF, 1: REF})
            self.assertEqual([r["fclass"] for r in run.rows], ["success", "unclassified"])
            self.assertEqual(run.rows[0]["mclass"], "arrive")
        finally:
            shutil.rmtree(tmp)

    def test_majority_merged(self):
        M = np.array([["arrive_stay", "base_not", "crash"], ["arrive_stay", "base_ok_ee_not", "arrive_stay"],
                      ["base_not", "base_ok_ee_not", "crash"]], dtype=object)
        cls, share = ar.majority_merged(M)
        self.assertEqual(list(cls), ["arrive_stay", "base_ok_ee_not", "crash"])
        np.testing.assert_allclose(share, [2 / 3, 2 / 3, 2 / 3])
        cls, _ = ar.majority_merged(np.array([["arrive_stay"], ["base_not"]], dtype=object))
        self.assertEqual(cls[0], "base_not")  # tie -> the worse class


# ------------------------------------------------------------------------------
class TestLoading(unittest.TestCase):
    def test_parse_run_name(self):
        P = ar.parse_run_name
        self.assertEqual((P("ego_s3_dev")["model"], P("ego_s3_dev")["seed"], P("ego_s3_dev")["set"]), ("ego", 3, "dev"))
        self.assertEqual((P("ego-novw_s12_hard")["model"], P("ego-novw_s12_hard")["seed"]), ("ego-novw", 12))
        self.assertEqual((P("ego-ols_dev")["model"], P("ego-ols_dev")["seed"], P("ego-ols_dev")["kind"]),
                         ("ego-ols", None, "model"))
        self.assertEqual((P("raw-ols_pilot")["model"], P("raw-ols_pilot")["set"]), ("raw-ols", "pilot"))
        self.assertEqual((P("expert_v11_hard")["kind"], P("expert_v11_hard")["protocol"], P("expert_v11_hard")["set"]),
                         ("expert", "v1.1", "hard"))
        self.assertEqual((P("expert_dev")["kind"], P("expert_dev")["set"]), ("expert", "dev"))
        self.assertEqual((P("zero_v11_dev")["kind"], P("zero_v11_dev")["set"]), ("zero", "dev"))
        b = P("B-ME_ego_s4_dev")
        self.assertEqual((b["kind"], b["cell"], b["model"], b["seed"], b["set"]), ("hybrid", "ME", "ego", 4, "dev"))
        b = P("B-EM_ego-novw_s0_dev")
        self.assertEqual((b["cell"], b["model"]), ("EM", "ego-novw"))
        self.assertEqual((P("B-EE_dev")["cell"], P("B-EE_dev")["set"]), ("EE", "dev"))
        self.assertEqual(P("ego_s0_dev")["cell"], "MM")
        self.assertEqual(P("something odd")["kind"], "other")

    def test_read_rows_types_and_load_run(self):
        tmp = tempfile.mkdtemp()
        try:
            rows = [make_row(1, arm_extend_time=NAN), failure(episode=0, stop_reason="stalled")]
            path = write_run(tmp, "ego_s2_dev", rows, policy_arg="trained/ego_s2.npz")
            run = ar.load_run(path)
            self.assertEqual([r["episode"] for r in run.rows], [0, 1])  # sorted
            r = run.by_episode[1]
            self.assertIs(r["success"], True)
            self.assertIsInstance(r["min_error"], float)
            self.assertTrue(math.isnan(r["arm_extend_time"]))
            self.assertEqual(r["error_message"], "")
            self.assertEqual(run.by_episode[0]["stop_reason"], "stalled")
            self.assertEqual((run.meta["model"], run.meta["seed"], run.meta["set"], run.meta["protocol"]),
                             ("ego", 2, "dev", "v1.1"))
            self.assertEqual(run.meta["warnings"], [])
            bad = ar.load_run(write_run(tmp, "ego_s3_dev", rows, policy_arg="trained/ego_s2.npz", eval_set="data/hard"))
            self.assertEqual(len(bad.meta["warnings"]), 2)
            self.assertEqual(bad.meta["set"], "hard")  # config.json wins
        finally:
            shutil.rmtree(tmp)

    def test_sealed_set_refused(self):
        with self.assertRaises(SystemExit):
            ar.check_set_allowed("test")
        with self.assertRaises(SystemExit):
            ar.load_reference("test", out_dir=tempfile.gettempdir())
        tmp = tempfile.mkdtemp()
        try:
            write_run(tmp, "ego_s0_test", [make_row()], eval_set="data/test")
            with self.assertRaises(SystemExit):
                ar.load_run(os.path.join(tmp, "ego_s0_test"))
            self.assertEqual(ar.discover_runs(tmp), [])
        finally:
            shutil.rmtree(tmp)

    def test_confirm_needs_flag(self):
        self.assertFalse(ar.ALLOW_CONFIRM)
        with self.assertRaises(SystemExit):
            ar.build_reference("confirm", data_dir="/nonexistent", out_dir=tempfile.gettempdir())

    def test_parse_seeds(self):
        self.assertEqual(ar.parse_seeds("0-3,7"), [0, 1, 2, 3, 7])
        self.assertEqual(ar.parse_seeds("5"), [5])
        self.assertEqual(ar.parse_seeds(""), [])

    def test_read_offline_format(self):
        tmp = tempfile.mkdtemp()
        try:
            data = offline_json(0.2)
            path = os.path.join(tmp, "x.offline.json")
            with open(path, "w") as f:
                json.dump(data, f)
            off = ar.read_offline(path)
            cell = next(c for c in data["cells"] if (c["group"], c["class"], c["window"]) == ("all", "all", "all"))
            self.assertEqual(off["val_loss"], 0.2)
            self.assertEqual(set(off["rmse"]), {"v", "w", "tau_mean"})
            self.assertEqual(set(off["nrmse"]), {"v", "w", "tau_mean"})
            self.assertAlmostEqual(off["rmse"]["v"], cell["rmse"]["v"])  # the overall cell, not clean / inside
            self.assertAlmostEqual(off["rmse"]["w"], cell["rmse"]["w"])
            self.assertAlmostEqual(off["rmse"]["tau_mean"], np.mean([cell["rmse"][t] for t in TAUS]))
            self.assertAlmostEqual(off["nrmse"]["v"], cell["nrmse"]["v"])
            self.assertAlmostEqual(off["nrmse"]["tau_mean"], cell["nrmse_tau_mean"])
            with open(path, "w") as f:
                json.dump({"val_loss": 0.3}, f)
            self.assertEqual(ar.read_offline(path), {"val_loss": 0.3, "rmse": {}, "nrmse": {}})
        finally:
            shutil.rmtree(tmp)

    @unittest.skipUnless(os.path.exists(REAL_OFFLINE), "no trained/ego-ols.offline.json")
    def test_read_offline_real_file(self):
        """The real format written by train_policy.py offline (trained/ego-ols.offline.json; no value printed)."""
        off = ar.read_offline(REAL_OFFLINE)
        self.assertTrue(np.isfinite(off["val_loss"]))
        for key in ("rmse", "nrmse"):
            self.assertEqual(set(off[key]), {"v", "w", "tau_mean"})
            self.assertTrue(all(np.isfinite(v) and v > 0 for v in off[key].values()))
        with open(REAL_OFFLINE) as f:
            cells = json.load(f)["cells"]
        cell = next(c for c in cells if (c["group"], c["class"], c["window"]) == ("all", "all", "all"))
        self.assertAlmostEqual(off["nrmse"]["tau_mean"], cell["nrmse_tau_mean"], places=12)


# ------------------------------------------------------------------------------
HYBRID_LOG = """pybullet build time: Oct 21 2025
HYBRID_EVENT {"event": "config", "base": "trained/ego_s0.npz", "arm": "expert", "trigger": "fsm"}
HYBRID_EVENT {"event": "phase", "t": 0.0, "phase": "align"}
t=1.00s | base [0. 0.] | EE error 1.2m
HYBRID_EVENT {"event": "phase", "t": 1.234, "phase": "drive"}
HYBRID_EVENT {"event": "phase", "t": 7.5, "phase": "orient"}
HYBRID_EVENT {"event": "nav_done", "t": 25.001, "trigger": "timeout"}
HYBRID_EVENT {"event": "arm_trigger", "t": 25.001, "source": "fsm"}
HYBRID_EVENT {"event": "nav_done", "t": 26.0, "trigger": "threshold"}
HYBRID_EVENT {broken json
Traceback (most recent call last):
Stopped: timeout at t=40.00s
"""


class TestHybridLog(unittest.TestCase):
    def test_parse(self):
        log = ar.parse_hybrid_log(HYBRID_LOG.splitlines())
        self.assertEqual(log["config"]["base"], "trained/ego_s0.npz")
        self.assertEqual(log["config"]["trigger"], "fsm")
        self.assertEqual(log["phases"], [(0.0, "align"), (1.234, "drive"), (7.5, "orient")])
        self.assertEqual(log["nav_done"], {"t": 25.001, "trigger": "timeout"})  # the first one
        self.assertEqual(log["arm_trigger"], {"t": 25.001, "source": "fsm"})
        self.assertEqual(log["n_bad"], 1)
        self.assertEqual(len(log["events"]), 7)

    def test_no_events_and_missing_file(self):
        log = ar.parse_hybrid_log(["nothing here", "t=1.00s"])
        self.assertIsNone(log["nav_done"])
        self.assertIsNone(log["config"])
        self.assertEqual(log["phases"], [])
        self.assertIsNone(ar.parse_hybrid_log("/nonexistent/ep_0000.log"))

    def test_file(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "ep_0003.log")
            with open(path, "w") as f:
                f.write(HYBRID_LOG)
            self.assertEqual(ar.parse_hybrid_log(path)["nav_done"]["trigger"], "timeout")
        finally:
            shutil.rmtree(tmp)


# ------------------------------------------------------------------------------
TAUS = [f"tau{j}" for j in range(1, 8)]


def offline_json(val_loss, rmse_v=0.011, rmse_w=0.031, rmse_tau=1.5):
    """The structure of trained/<name>.offline.json as train_policy.py offline writes it (cells per group x class x
    window, event windows, torque segments; made-up values)."""
    def block(v, w, tau, scale):
        return {"v": v / scale, "w": w / scale, **{t: (tau + 0.1 * j) / scale for j, t in enumerate(TAUS)}}
    cells = [{"group": g, "class": c, "window": wn, "n": 1000,
              "rmse": block(rmse_v * (1 + k), rmse_w, rmse_tau, 1.0), "nrmse": block(rmse_v * (1 + k), rmse_w, rmse_tau, 2.0),
              "nrmse_tau_mean": None, "copycat_rmse": {"v": 0.02, "w": 0.04}, "copycat_nrmse": {"v": 0.2, "w": 0.1}}
             for k, (g, c, wn) in enumerate((("clean", "all", "all"), ("all", "drive", "inside"), ("all", "all", "all"),
                                              ("all", "all", "inside")))]
    for c in cells:
        c["nrmse_tau_mean"] = float(np.mean([c["nrmse"][t] for t in TAUS]))
    return {"model_file": "trained/x.npz", "model": "x", "kind": "mlp", "seed": 0, "feature_set": "ego",
            "n_val_episodes": 284, "n_val_rows": 2802594, "val_loss": val_loss, "val_loss_meta": val_loss,
            "label_std_train": {"v": 0.117, "w": 0.43}, "definitions": {"val_loss": "standardised MSE"},
            "settle_s": {}, "counts": {}, "cells": cells,
            "event_windows": [{"event": "start", "group": "all", "n_events": 284, "n": 36352,
                               "rmse": {"v": 9.0, "w": 9.0}, "nrmse": {"v": 9.0, "w": 9.0}}],
            "torque_segments": [{"group": "all", "segment": "nav_hold", "z": "z<0.45", "n": 10, "rmse": {"tau1": 9.0},
                                 "nrmse": {"tau1": 9.0}, "rmse_pooled": 9.0}],
            "lag": {}, "seconds": 1.0, "created": "2026-10-29"}


def synthetic_tree(root, n_ep=60, seeds=(0, 1, 2)):
    """A results tree on a synthetic 'dev' set: expert_v11 / zero_v11 / ego / ego-novw / B cells / OLS runs."""
    results, out, trained = os.path.join(root, "results"), os.path.join(root, "step3"), os.path.join(root, "trained")
    os.makedirs(results)
    rng = np.random.default_rng(11)
    d0 = np.linspace(-1.0, 2.0, n_ep)
    theta = rng.uniform(-180, 180, n_ep)
    write_ref(out, "dev", d0, theta)
    ref = ar.load_reference("dev", out_dir=out, data_dir="/nonexistent")

    def rows_for(p_success, fail_kind, extra=None):
        rows = []
        for e in range(n_ep):
            align = ref[e]["align_req_deg"]
            common = dict(d0=d0[e], theta_deg=theta[e], ring_dist_start=abs(d0[e]) + 0.4)
            if rng.random() < p_success:
                rows.append(make_row(e, base_turn_total_deg=align + 10, held_10s=bool(rng.random() < 0.7), **common))
            elif fail_kind == "circling":
                rows.append(failure(episode=e, base_turn_total_deg=align + 400 + ref[e]["orient_req_deg"],
                                    ring_dist_min=0.0, ring_dist_final=0.05, **common))
            elif fail_kind == "no_start":
                rows.append(failure(episode=e, base_path_length=0.0, base_turn_total_deg=0.0, ring_dist_min=0.9,
                                    ring_dist_final=0.9, arm_extend_time=0.3, **{**common, "ring_dist_start": 0.9}))
            else:
                kind = rng.choice(["park", "left", "ext"])
                # 'left': the arm extended but the EE ended no closer than the parking hold posture (0.30 - 0.29 <
                # 0.02): arm_extended, not arm_extended_toward
                kw = {"park": dict(arm_extend_time=NAN, min_error=0.31, ring_dist_final=0.05),
                      "left": dict(ring_dist_final=0.8, min_error=0.29),
                      "ext": dict(min_error=0.03, ring_dist_final=0.05)}[kind]
                rows.append(failure(episode=e, ring_dist_min=0.05, **kw, **common))
        return rows

    write_run(results, "expert_v11_dev", rows_for(0.97, "circling"), policy="expert")
    write_run(results, "B-EE_dev", rows_for(0.97, "circling"), policy="policies/hybrid.py:make_policy",
              policy_arg="base=expert;arm=expert")
    write_run(results, "zero_v11_dev", rows_for(0.0, "no_start"), policy="zero")
    for s in seeds:
        write_run(results, f"ego_s{s}_dev", rows_for(0.5, "mixed"), policy_arg=f"trained/ego_s{s}.npz")
        write_run(results, f"ego-novw_s{s}_dev", rows_for(0.6, "mixed"), policy_arg=f"trained/ego-novw_s{s}.npz")
        me_logs = {e: HYBRID_LOG if e % 3 == 0 else HYBRID_LOG.replace('"timeout"', '"threshold"') for e in range(n_ep)}
        write_run(results, f"B-ME_ego_s{s}_dev", rows_for(0.7, "mixed"), logs=me_logs,
                  policy_arg=f"base=trained/ego_s{s}.npz;arm=expert")
        em_logs = {e: 'HYBRID_EVENT {"event": "nav_done", "t": 6.5, "trigger": "threshold"}\n' for e in range(n_ep)}
        write_run(results, f"B-EM_ego_s{s}_dev", rows_for(0.8, "mixed"), logs=em_logs,
                  policy_arg=f"base=expert;arm=trained/ego_s{s}.npz")
        os.makedirs(trained, exist_ok=True)
        for m in ("ego", "ego-novw"):
            with open(os.path.join(trained, f"{m}_s{s}.offline.json"), "w") as f:
                json.dump(offline_json(0.1 + 0.01 * s), f)
    write_run(results, "ego-ols_dev", rows_for(0.1, "mixed"), policy_arg="trained/ego-ols.npz")
    return results, out, trained


class TestReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.results, cls.out, cls.trained = synthetic_tree(cls.tmp)
        ar._REF_CACHE.clear()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def load(self, name):
        return ar.load_prepared(os.path.join(self.results, name), out_dir=self.out, data_dir="/nonexistent")

    def test_compare(self):
        ra = [self.load(f"ego_s{s}_dev") for s in range(3)]
        rb = [self.load(f"ego-novw_s{s}_dev") for s in range(3)]
        lines, res = ar.compare_arms(ra, rb, "ego", "ego-novw", ("parked_no_arm", "left_or_overshot"), n_boot=500)
        self.assertEqual(res["n_episodes"], 60)
        self.assertEqual(len(res["seed_pairs"]), 9)
        self.assertAlmostEqual(res["diff"]["diff"], res["seed_mean"]["ego-novw"]["mean"] - res["seed_mean"]["ego"]["mean"])
        # class shares sum to 1 per arm, and the net difference is the sum of the class reductions
        self.assertAlmostEqual(sum(v["a"] for v in res["classes"].values()), 1.0)
        self.assertAlmostEqual(sum(v["b"] for v in res["classes"].values()), 1.0)
        red = sum(v["a"] - v["b"] for c, v in res["classes"].items() if c != "success")
        self.assertAlmostEqual(red, res["diff"]["diff"])
        self.assertEqual(len(res["strata"]), 5)
        self.assertIn("held_10s", res["secondary"])
        self.assertIn("ci_lower_b_minus_a", res["adoption_inputs"])
        self.assertTrue(any("Adoption-rule inputs" in line for line in lines))
        # identical arms: zero difference
        _, same = ar.compare_arms(ra, ra, "ego", "ego", n_boot=300)
        self.assertEqual(same["diff"]["diff"], 0.0)
        # the seed-level Welch t next to the bootstrap, and adoption condition (1) = both lower bounds > 0
        def rate(runs):
            return [np.mean([ar.metric_value(r.by_episode[e], "success") for e in range(60)]) for r in runs]
        ref = stats.ttest_ind(rate(rb), rate(ra), equal_var=False)
        self.assertEqual(res["welch"]["test"], "welch")
        self.assertAlmostEqual(res["welch"]["p"], ref.pvalue, places=10)
        self.assertAlmostEqual(res["welch"]["lo"], ref.confidence_interval(0.95).low, places=10)
        ai = res["adoption_inputs"]
        self.assertEqual(ai["welch_ci_lower_b_minus_a"], res["welch"]["lo"])
        self.assertEqual(ai["cond1"], ai["cond1_bootstrap"] and ai["cond1_welch"])
        self.assertEqual(ai["cond1_bootstrap"], res["diff"]["lo"] > 0)
        text = "\n".join(lines)
        self.assertIn("condition (1) (both > 0)", text)
        self.assertIn("Welch t (per-seed rates)", text)
        self.assertIn("welch", res["secondary"]["held_10s"])
        # class-table flags: near-ring no-motion failures and failures on short-reach episodes
        self.assertIn("short_reach_failures", res["flags"])
        self.assertIn("failures on short-reach episodes", text)
        # arm_extended_toward excludes the extended arms that did not get closer than the parking posture
        sec = res["secondary"]
        self.assertLess(sec["arm_extended_toward"]["mean_a"], sec["arm_extended"]["mean_a"])
        # the expert's rate on the same episodes is the reference of every metric
        ex = self.load("expert_v11_dev")
        lx, rx = ar.compare_arms(ra, rb, "ego", "ego-novw", n_boot=300, expert_run=ex)
        for metric in ("arm_extended", "arm_extended_toward", "success"):
            manual = np.mean([ar.metric_value(ex.by_episode[e], metric) for e in range(60)])
            self.assertAlmostEqual(rx["expert_reference"]["rates"][metric], manual)
        self.assertAlmostEqual(rx["secondary"]["arm_extended_toward"]["expert"],
                               rx["expert_reference"]["rates"]["arm_extended_toward"])
        self.assertTrue(any("expert" in line and "% on the same episodes" in line for line in lx))

    def test_compare_one_run_arm(self):
        # an arm with a single run (OLS): one-sample t of the other arm's per-seed rates against it
        ra = [self.load("ego-ols_dev")]
        rb = [self.load(f"ego_s{s}_dev") for s in range(3)]
        _, res = ar.compare_arms(ra, rb, "ego-ols", "ego", n_boot=300)
        rate_a = np.mean([ar.metric_value(ra[0].by_episode[e], "success") for e in range(60)])
        rate_b = [np.mean([ar.metric_value(r.by_episode[e], "success") for e in range(60)]) for r in rb]
        self.assertEqual(res["welch"]["test"], "one-sample")
        self.assertAlmostEqual(res["welch"]["p"], stats.ttest_1samp(np.array(rate_b) - rate_a, 0.0).pvalue, places=10)

    def test_split2x2(self):
        ee = self.load("B-EE_dev")
        em = [self.load(f"B-EM_ego_s{s}_dev") for s in range(3)]
        me = [self.load(f"B-ME_ego_s{s}_dev") for s in range(3)]
        mm = [self.load(f"ego_s{s}_dev") for s in range(3)]
        lines, res = ar.split2x2(ee, em, me, mm, n_boot=500)
        att = res["attribution"]
        self.assertEqual(sum(att[k]["count"] for k in ("base_only", "arm_only", "both", "interaction")),
                         att["n_mm_failures"])
        self.assertIn(res["verdict"], ("only base", "only arm", "both", "neither significant"))
        self.assertEqual(res["me_trigger"]["timeout"]["n"] + res["me_trigger"]["threshold"]["n"], 3 * 60)
        self.assertEqual(res["me_trigger"]["timeout"]["n"], 3 * 20)
        lag_n = res["em_lag"]["n"] + res["em_lag"]["never_extended"]
        self.assertEqual(lag_n, 3 * 60)
        self.assertAlmostEqual(res["em_lag"]["median"], 7.0 - 6.5)
        # manual attribution count
        f = np.array([[ar.metric_value(r.by_episode[e], "success") == 0 for e in range(60)] for r in mm])
        self.assertEqual(int(f.sum()), att["n_mm_failures"])
        # ME cell by nav_done trigger: successes and failures (timeout logs on episodes e % 3 == 0)
        trig = res["me_trigger"]
        self.assertEqual(sum(v["success"] + v["failure"] for v in trig.values()), 3 * 60)
        self.assertEqual(trig["timeout"]["success"], int(sum(ar.metric_value(r.by_episode[e], "success")
                                                             for r in me for e in range(0, 60, 3))))
        self.assertEqual(trig["timeout"]["failure"], 3 * 20 - trig["timeout"]["success"])
        self.assertTrue(any("timeout-triggered ME success does not show the model base parked" in line for line in lines))
        # attribution without the M0-like episodes (|d0| < 0.15 m)
        keep = [e for e in range(60) if abs(ee.by_episode[e]["d0"]) >= 0.15]
        self.assertLess(len(keep), 60)
        far = res["attribution_no_m0like"]
        self.assertEqual(far["n_episodes"], len(keep))
        self.assertEqual(far["n_mm_failures"], int(f[:, keep].sum()))
        self.assertEqual(sum(far[k]["count"] for k in ("base_only", "arm_only", "both", "interaction")),
                         far["n_mm_failures"])
        # contrasts against EE: one-sample t on the per-seed differences; Holm on max(bootstrap p, t p)
        rate_ee = np.mean([ar.metric_value(ee.by_episode[e], "success") for e in range(60)])
        for c, runs in (("ME", me), ("EM", em), ("MM", mm)):
            dd = res["diffs"][c]
            rates = np.array([np.mean([ar.metric_value(r.by_episode[e], "success") for e in range(60)]) for r in runs])
            self.assertEqual(dd["t"]["test"], "one-sample")
            self.assertAlmostEqual(dd["t"]["p"], stats.ttest_1samp(rates - rate_ee, 0.0).pvalue, places=10)
            self.assertEqual(dd["p_used"], max(dd["p"], dd["t"]["p"]))
        adj, rej = ar.holm([res["diffs"][c]["p_used"] for c in ("ME", "EM", "MM")])
        np.testing.assert_allclose([res["diffs"][c]["p_holm"] for c in ("ME", "EM", "MM")], adj)
        self.assertEqual([res["diffs"][c]["reject"] for c in ("ME", "EM", "MM")], list(rej))

    def test_validate(self):
        expert, zero = [self.load("expert_v11_dev")], [self.load("zero_v11_dev")]
        lines, checks = ar.validate_classifier(expert, zero)
        by = {c["check"].split(":")[0]: c["passed"] for c in checks}
        self.assertTrue(by["(i) expert failures classify as circling"])
        self.assertTrue(by["(ii) zero_v11_dev"])
        self.assertTrue(by["(iii) expert successes"])
        self.assertTrue(by["(iv) expert arm_extend_time >= recorded nav_done time"])
        self.assertTrue(any("PASS" in line for line in lines))

    def test_describe_and_report(self):
        runs = ar.discover_runs(self.results)
        runs = [ar.prepare(r, self.out, "/nonexistent") for r in runs]
        lines, res = ar.describe(runs, trained_dir=self.trained, n_boot=300)
        cons = {(c["model"], c["set"]): c for c in res["consistency"]}
        self.assertEqual(cons[("ego", "dev")]["S"], 3)
        self.assertEqual(len(res["offline"]), 6)
        self.assertEqual(set(res["offline"][0]["rmse"]), {"v", "w", "tau_mean"})
        self.assertEqual(set(res["offline"][0]["nrmse"]), {"v", "w", "tau_mean"})
        self.assertTrue(any("nRMSE v" in line for line in lines))
        self.assertTrue(res["m1m2"])
        report = ar.write_report(self.results, self.out, "/nonexistent", self.trained, n_boot=300)
        self.assertTrue(os.path.exists(os.path.join(self.out, "summary.txt")))
        text = "\n".join(report)
        self.assertIn("Experiment A", text)
        self.assertIn("Experiment B", text)
        for fig in ("fig_step3_classes_d0_theta", "fig_step3_split2x2", "fig_step3_success_coverage",
                    "fig_step3_offline_vs_closed"):
            self.assertTrue(os.path.exists(os.path.join(self.out, fig + ".png")), fig)
            self.assertTrue(os.path.exists(os.path.join(self.out, fig + ".pdf")), fig)

    def test_cli_compare(self):
        ar.main(["--results", self.results, "--out", self.out, "--data", "/nonexistent", "--n-boot", "200",
                 "compare", "--a", "ego", "--b", "ego-novw", "--set", "dev", "--seeds", "0-2",
                 "--predicted-classes", "parked_no_arm"])
        with open(os.path.join(self.out, "A_ego_vs_ego-novw_dev.json")) as f:
            res = json.load(f)
        self.assertEqual(res["n_episodes"], 60)


    def test_cli_subcommands(self):
        common = ["--results", self.results, "--out", self.out, "--data", "/nonexistent", "--trained", self.trained,
                  "--n-boot", "200"]
        ar.main(common + ["split2x2", "--model", "ego", "--set", "dev", "--seeds", "0-2"])
        with open(os.path.join(self.out, "B_ego_dev.json")) as f:
            res = json.load(f)
        self.assertEqual(res["n_seeds"], 3)
        self.assertEqual(res["ee_vs_expert"]["n"], 60)
        ar.main(common + ["validate", "--expert-runs", os.path.join(self.results, "expert_v11_dev"),
                          "--zero-runs", os.path.join(self.results, "zero_v11_dev")])
        self.assertTrue(os.path.exists(os.path.join(self.out, "validate.json")))
        ar.main(common + ["classify", os.path.join(self.results, "ego_s0_dev")])
        self.assertTrue(os.path.exists(os.path.join(self.out, "classes_ego_s0_dev.csv")))
        ar.main(common + ["describe", "--sets", "dev"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "describe.json")))
        ar.main(common + ["figures", "--set", "dev"])
        # explicit run folders and a seedless (OLS) arm
        ar.main(common + ["compare", "--a", "ego-ols", "--b", "ego", "--set", "dev", "--seeds", "0-2"])
        with open(os.path.join(self.out, "A_ego-ols_vs_ego_dev.json")) as f:
            self.assertEqual(len(json.load(f)["runs_a"]), 1)
        ar.main(common + ["compare", "--a-runs", os.path.join(self.results, "ego_s0_dev"),
                          "--b-runs", os.path.join(self.results, "ego-novw_s0_dev"), "--set", "dev"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "A_A_vs_B_dev.json")))


class TestIntegrity(unittest.TestCase):
    """Paired runs must come from the same set, cover its full episode set, have no harness_error, and a hybrid run
    needs --no-early-stop; otherwise compare / split2x2 / describe refuse unless --allow-incomplete."""
    N = 20

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.results, self.out = os.path.join(self.tmp, "results"), os.path.join(self.tmp, "step3")
        write_ref(self.out, "dev", np.linspace(-0.5, 1.5, self.N), np.zeros(self.N))
        self.addCleanup(shutil.rmtree, self.tmp)

    def make(self, name, rows=None, prepare=True, **kw):
        rows = rows if rows is not None else [make_row(e, d0=float(np.linspace(-0.5, 1.5, self.N)[e]))
                                               for e in range(self.N)]
        run = ar.load_run(write_run(self.results, name, rows, **kw))
        return ar.prepare(run, self.out, "/nonexistent") if prepare else run

    def test_complete_runs_pass(self):
        runs = [self.make("ego_s0_dev"), self.make("ego_s1_dev"),
                self.make("B-ME_ego_s0_dev", policy="policies/hybrid.py:make_policy",
                          policy_arg="base=trained/ego_s0.npz;arm=expert")]
        self.assertEqual(ar.integrity_problems(runs), [])
        self.assertEqual(ar.check_integrity(runs, "test"), [])

    def test_each_problem_refused(self):
        ok = self.make("ego_s0_dev")
        rows = [make_row(e) for e in range(self.N)]
        cases = {
            "different (or unknown) evaluation sets": self.make("ego_s1_hard", rows, prepare=False,
                                                                 eval_set="data/hard"),
            "missing 5": self.make("ego_s2_dev", [r for r in rows if r["episode"] != 5], n_episodes=self.N),
            "full size of the evaluation set unknown": self.make("ego_s3_dev", rows, n_episodes=None),
            "harness_error in episodes 3": self.make("ego_s4_dev", [failure(episode=3, stop_reason="harness_error")
                                                                    if r["episode"] == 3 else r for r in rows]),
            "duplicate episode rows": self.make("ego_s5_dev", rows + [make_row(4)], n_episodes=self.N),
            "without --no-early-stop": self.make("B-ME_ego_s1_dev", rows, policy="policies/hybrid.py:make_policy",
                                                 policy_arg="base=trained/ego_s1.npz;arm=expert", no_early_stop=False),
        }
        for text, bad in cases.items():
            problems = ar.integrity_problems([ok, bad])
            self.assertTrue(any(text in p for p in problems), (text, problems))
            with self.assertRaises(SystemExit) as cm:
                ar.check_integrity([ok, bad], "test", allow_incomplete=False)
            self.assertIn(text, str(cm.exception))
            self.assertIn("--allow-incomplete", str(cm.exception))
            self.assertTrue(ar.check_integrity([ok, bad], "test", allow_incomplete=True))  # warnings only
        # hybrid detected by the policy file too
        hyb = self.make("other_dev", rows, prepare=False, policy="policies/hybrid.py:make_policy", policy_arg=None)
        self.assertTrue(ar.is_hybrid(hyb))
        self.assertFalse(ar.is_hybrid(ok))

    def test_compare_refuses_and_allow_incomplete(self):
        a = [self.make("ego_s0_dev"), self.make("ego_s1_dev")]
        rows = [make_row(e, d0=float(np.linspace(-0.5, 1.5, self.N)[e])) for e in range(self.N)]
        b = [self.make("ego-novw_s0_dev", [failure(episode=3, stop_reason="harness_error", d0=rows[3]["d0"])
                                           if r["episode"] == 3 else r for r in rows]),
             self.make("ego-novw_s1_dev", [r for r in rows if r["episode"] != 7], n_episodes=self.N)]
        with self.assertRaises(SystemExit):
            ar.compare_arms(a, b, n_boot=200)
        lines, res = ar.compare_arms(a, b, n_boot=200, allow_incomplete=True)
        self.assertEqual(res["n_episodes"], self.N - 2)  # episode 7 missing, episode 3 a harness_error
        self.assertEqual(res["harness_error_excluded"], [3])
        self.assertTrue(any("WARNING (--allow-incomplete)" in line for line in lines))
        self.assertNotIn("harness_error", res["classes"])
        # the module flag (CLI --allow-incomplete) has the same effect
        try:
            ar.ALLOW_INCOMPLETE = True
            self.assertEqual(ar.compare_arms(a, b, n_boot=200)[1]["n_episodes"], self.N - 2)
        finally:
            ar.ALLOW_INCOMPLETE = False
        # CLI: refused without the flag, analysed with it
        common = ["--results", self.results, "--out", self.out, "--data", "/nonexistent", "--n-boot", "100"]
        runs = ["compare", "--a-runs", *[r.path for r in a], "--b-runs", *[r.path for r in b]]
        with self.assertRaises(SystemExit):
            ar.main(common + runs)
        try:
            ar.main(common[:6] + ["--allow-incomplete"] + common[6:] + runs)
            self.assertTrue(os.path.exists(os.path.join(self.out, "A_A_vs_B_dev.json")))
        finally:
            ar.ALLOW_INCOMPLETE = False

    def test_split2x2_refuses_hybrid_without_no_early_stop(self):
        hy = "policies/hybrid.py:make_policy"
        ee = self.make("B-EE_dev", policy=hy, policy_arg="base=expert;arm=expert")
        em = [self.make("B-EM_ego_s0_dev", policy=hy, policy_arg="base=expert;arm=trained/ego_s0.npz")]
        me = [self.make("B-ME_ego_s0_dev", policy=hy, policy_arg="base=trained/ego_s0.npz;arm=expert",
                        no_early_stop=False)]
        mm = [self.make("ego_s0_dev")]
        with self.assertRaises(SystemExit) as cm:
            ar.split2x2(ee, em, me, mm, n_boot=100)
        self.assertIn("B-ME_ego_s0_dev: hybrid run made without --no-early-stop", str(cm.exception))
        lines, res = ar.split2x2(ee, em, me, mm, n_boot=100, allow_incomplete=True)
        self.assertEqual(len(res["integrity_warnings"]), 1)
        # one seed: the t test is undefined, so the contrasts stay out of the Holm family (not significant)
        self.assertTrue(all(np.isnan(res["diffs"][c]["p_used"]) for c in ("ME", "EM", "MM")))
        self.assertEqual(res["verdict"], "neither significant")

    def test_describe_refuses_incomplete_group(self):
        rows = [make_row(e) for e in range(self.N)]
        runs = [self.make("ego_s0_dev"), self.make("ego_s1_dev", rows[:-1], n_episodes=self.N)]
        with self.assertRaises(SystemExit) as cm:
            ar.describe(runs, trained_dir=self.tmp, n_boot=100)
        self.assertIn("ego dev: ego_s1_dev: 19 episodes", str(cm.exception))
        lines, res = ar.describe(runs, trained_dir=self.tmp, n_boot=100, allow_incomplete=True)
        self.assertTrue(res["integrity_warnings"])
        self.assertEqual(res["consistency"][0]["E"], self.N - 1)


class TestValidation(unittest.TestCase):
    """Classifier validation (iii) with its expected exceptions, and the CONFIRM guard of the validate command."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.results, self.out = os.path.join(self.tmp, "results"), os.path.join(self.tmp, "step3")
        self.addCleanup(shutil.rmtree, self.tmp)

    def expert(self, set_name, n, circling=()):
        """Expert successes on a synthetic set; the episodes in circling circled > 360 deg before a late success."""
        d0 = np.linspace(0.3, 2.0, n)
        write_ref(self.out, set_name, d0, np.zeros(n))
        rows = [make_row(e, d0=float(d0[e]), ring_dist_start=float(d0[e]),
                         **(dict(base_turn_total_deg=1000.0, parked_time=26.2) if e in circling else {}))
                for e in range(n)]
        path = write_run(self.results, f"expert_v11_{set_name}", rows, eval_set=f"data/{set_name}", policy="expert")
        return ar.load_prepared(path, self.out, "/nonexistent")

    def check(self, checks, prefix):
        return next(c for c in checks if c["check"].startswith(prefix))

    def test_expected_exceptions(self):
        hard = self.expert("hard", 160, circling=(19, 158))
        lines, checks = ar.validate_classifier([hard], [])
        c3 = self.check(checks, "(iii)")
        self.assertTrue(c3["passed"])
        self.assertIn("expected exceptions 2, unexpected 0", c3["detail"])
        text = "\n".join(lines)
        self.assertIn("expert_v11_hard ep 19 circling", text)
        self.assertIn("expert_v11_hard ep 158 circling", text)
        # another circling success is not an expected exception, nor is episode 19 of another set
        hard = self.expert("hard", 160, circling=(5, 19, 158))
        lines, checks = ar.validate_classifier([hard], [])
        self.assertFalse(self.check(checks, "(iii)")["passed"])
        self.assertIn("UNEXPECTED: expert_v11_hard ep 5 circling", "\n".join(lines))
        dev = self.expert("dev", 30, circling=(19,))
        self.assertFalse(self.check(ar.validate_classifier([dev], [])[1], "(iii)")["passed"])
        # an expected exception that did not fire is reported, not failed
        lines, checks = ar.validate_classifier([self.expert("hard", 160, circling=(19,))], [])
        self.assertTrue(self.check(checks, "(iii)")["passed"])
        self.assertIn("did not fire (informational): HARD 158 circling", "\n".join(lines))

    def test_validate_skips_confirm_without_flag(self):
        self.expert("dev", 30)
        self.expert("confirm", 30)  # writes results/expert_v11_confirm (and its synthetic reference cache)
        common = ["--results", self.results, "--out", self.out, "--data", "/nonexistent"]
        try:
            ar.main(common + ["validate"])  # default runs: dev, hard (missing), confirm (skipped): no abort
            with open(os.path.join(self.out, "validate.json")) as f:
                res = json.load(f)
            self.assertEqual([os.path.basename(p) for p in res["skipped_confirm"]], ["expert_v11_confirm"])
            with open(os.path.join(self.out, "validate.txt")) as f:
                self.assertIn("Runs: expert expert_v11_dev;", f.read())
            ar.main(common[:6] + ["--allow-confirm"] + ["validate"])
            with open(os.path.join(self.out, "validate.json")) as f:
                self.assertEqual(json.load(f)["skipped_confirm"], [])
            with open(os.path.join(self.out, "validate.txt")) as f:
                self.assertIn("Runs: expert expert_v11_dev, expert_v11_confirm;", f.read())
        finally:
            ar.ALLOW_CONFIRM = False


class TestFigures(unittest.TestCase):
    def test_figures_on_synthetic_data(self):
        tmp = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(0)
            panels = [{"title": f"model {k}", "d0": rng.uniform(-1, 2.5, 80), "theta": rng.uniform(-180, 180, 80),
                       "classes": rng.choice(list(ar.MERGED_CLASSES), 80), "share": np.ones(80)} for k in range(2)]
            paths = [ar.plot_failure_d0_theta(panels, tmp),
                     ar.plot_failure_d0_theta(panels[:1], tmp, name="one_panel.png"),
                     ar.plot_split2x2({"EE": (0.99, 0.97, 1.0), "EM": (0.6, 0.5, 0.7), "ME": (0.4, 0.3, 0.5),
                                       "MM": (0.2, 0.1, 0.3)},
                                      {"base_only": 0.4, "arm_only": 0.3, "both": 0.2, "interaction": 0.1}, tmp)]
            bins = [{"bin": b, "n": 10, "centre": 0.1 + 0.2 * b, "mean": 0.5, "lo": 0.4, "hi": 0.6} for b in range(5)]
            paths.append(ar.plot_success_coverage([{"label": "ego dev", "rep": "world", "bins": bins},
                                                   {"label": "ego dev", "rep": "rel", "bins": bins},
                                                   {"label": "raw dev", "rep": "world", "bins": bins}], tmp))
            paths.append(ar.plot_offline_vs_closed([{"model": "ego", "val_loss": 0.1, "success": 0.3, "lo": 0.25,
                                                     "hi": 0.35},
                                                    {"model": "raw", "val_loss": 0.2, "success": 0.0, "lo": 0.0,
                                                     "hi": 0.02}], tmp))
            for path in paths:
                self.assertTrue(os.path.getsize(path) > 0, path)
                self.assertTrue(os.path.exists(os.path.splitext(path)[0] + ".pdf"), path)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
