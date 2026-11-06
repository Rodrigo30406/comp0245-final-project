"""P2 checks using only data/train; evidence is written under step6/smoke.

Revision 1 leaves expert references untouched, including navigation holds below
joint limits and reach references whose lead exceeds max_q_lead.
"""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

import numpy as np
import pinocchio as pin

import bc_features as bcf
import train_policy as tp
from collect_data import wheel_speeds
from evaluate_policy import ExpertPolicy, setup_simulation
from policies.bc_policy import BASE_PARAMS, PDPolicy, make_policy
from policies.pd_layer import expert_arm_params, pd_layer, q_pin_from_obs
from robotnik_arm_control import arm_controller

SMOKE = TASK_DIR / "results" / "step6" / "smoke"


def recorded_obs(a, k):
    return {"base_pos": a["base_pos"][k].astype(float), "base_ori": a["base_ori"][k].astype(float),
            "base_bearing": float(a["base_bearing"][k]), "q_mes": a["q_mes"][k].astype(float),
            "qd_mes": a["qd_mes"][k].astype(float), "target": a["target"].astype(float)}


class TestPDLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SMOKE.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(io.StringIO()):
            _, client, _, cls.info = setup_simulation(np.zeros(2), 0.0)
        client.disconnect()
        cls.params = expert_arm_params()
        cls.evidence = {"source": "data/train only", "exact_states": 0, "exact_max_error": None,
                        "recorded": [], "recorded_max_error": None,
                        "revision": 1,
                        "reference_rule": "No lead or joint-limit clipping; final torque saturation only"}
        split = tp.load_split()
        eps = split["train_episodes"]
        cls.selected = ([e for e in eps if not split["episodes"][str(e)]["noisy"]][:5]
                        + [e for e in eps if split["episodes"][str(e)]["noisy"]][:5])
        cls.episode_paths = [Path(tp.TRAIN_DIR) / split["episodes"][str(e)]["file"] for e in cls.selected]

    @classmethod
    def tearDownClass(cls):
        (SMOKE / "P2_revision1_verification.json").write_text(json.dumps(cls.evidence, indent=2) + "\n")

    def controller(self, obs, q_des, q_hold, target, reach_active):
        m = self.info["dyn_model"]
        q_pin = q_pin_from_obs(obs, self.info)
        pin.forwardKinematics(m.pin_model, m.pin_data, q_pin)
        pin.updateFramePlacements(m.pin_model, m.pin_data)
        tau, q_des, qd_des = arm_controller(
            m, q_pin, obs["q_mes"][4:11], obs["qd_mes"][4:11], q_des, q_hold, target,
            m.pin_data.oMf[self.info["ee_frame_id"]].translation.copy(),
            self.info["ee_frame_id"], self.info["shoulder_frame_id"],
            self.info["joint_lower"][4:11], self.info["joint_upper"][4:11],
            self.info["time_step"], self.params, reach_active)
        return tau, q_des, qd_des, q_pin

    def test_a_exact_200_states(self):
        rng = np.random.default_rng(20261102)
        lo, hi = self.info["joint_lower"][4:11], self.info["joint_upper"][4:11]
        finite = lo < hi
        lower = np.where(finite, lo + 2 * self.params.limit_margin, -np.pi)
        upper = np.where(finite, hi - 2 * self.params.limit_margin, np.pi)
        max_error = 0.0
        reach_counts = {"true": 0, "false": 0}
        below_limit_counts = {"true": 0, "false": 0}
        unbounded_input_counts = {"q_hold": 0, "q_des": 0}
        for k in range(200):
            q = self.info["init_q_mes"].copy()
            q[4:11] = rng.uniform(lower, upper)
            # Include the actual initial joint-6 condition, on both branches.
            if k % 4 < 2:
                q[9] = self.info["init_q_mes"][9]
            qd = rng.uniform(-1.0, 1.0, len(q))
            yaw = rng.uniform(-np.pi, np.pi)
            obs = {"q_mes": q, "qd_mes": qd, "base_pos": np.r_[rng.uniform(-1, 1, 2), 0.15],
                   "base_ori": np.array([0., 0., np.sin(yaw / 2), np.cos(yaw / 2)])}
            hold = q[4:11] + rng.uniform(-0.7, 0.7, 7)
            desired = q[4:11] + rng.uniform(-0.7, 0.7, 7)
            if k % 4 < 2:
                hold[5] = desired[5] = self.info["init_q_mes"][9]
            branch = "true" if k % 2 else "false"
            reach_counts[branch] += 1
            below_limit_counts[branch] += int(hold[5] < lo[5] and desired[5] < lo[5])
            for name, reference in (("q_hold", hold), ("q_des", desired)):
                unbounded_input_counts[name] += int(
                    np.any(np.abs(reference - q[4:11]) > self.params.max_q_lead)
                    or np.any(finite & ((reference < lo) | (reference > hi))))
            target = rng.uniform([-1., -1., .2], [1., 1., 1.2])
            tau, q_des, qd_des, q_pin = self.controller(
                obs, desired, hold, target, bool(k % 2))
            actual = pd_layer(q_des - q[4:11], qd_des, q[4:11], qd[4:11], q_pin, self.info)
            max_error = max(max_error, float(np.max(np.abs(tau - actual))))
        self.evidence.update(exact_states=200, exact_max_error=max_error,
                             reach_active_counts=reach_counts,
                             joint6_below_limit_input_counts=below_limit_counts,
                             unclipped_input_counts=unbounded_input_counts)
        self.assertEqual(reach_counts, {"true": 100, "false": 100})
        self.assertTrue(all(n > 0 for n in below_limit_counts.values()))
        self.assertTrue(all(n > 0 for n in unbounded_input_counts.values()))
        self.assertLessEqual(max_error, 1e-9)
        expert = ExpertPolicy()
        expert.reset(np.array([1., 0., .5]), self.info)
        self.assertEqual(vars(self.params), vars(expert.arm_params))

    def test_b_recorded_float32_10_train_episodes(self):
        maximum = 0.0
        nav_total, nav_exceed, total, exceed = 0, 0, 0, 0
        for e, path in zip(self.selected, self.episode_paths):
            with np.load(path) as d:
                a = {key: d[key] for key in ("base_pos", "base_ori", "q_mes", "qd_mes", "q_des_arm",
                                             "qd_des_arm", "torque_arm", "nav_done")}
            q = a["q_mes"][:, 4:11].astype(float)
            e_ref = a["q_des_arm"].astype(float) - q
            nav = ~a["nav_done"].astype(bool)
            bad_nav = nav & np.any(np.abs(e_ref) > self.params.max_q_lead, axis=1)
            nav_total += int(nav.sum())
            nav_exceed += int(bad_nav.sum())
            total += len(q)
            exceed += int(np.any(np.abs(e_ref) > self.params.max_q_lead, axis=1).sum())
            self.evidence.update(navigation_steps=nav_total, navigation_exceed_steps=nav_exceed,
                                 navigation_exceed_fraction=nav_exceed / max(1, nav_total),
                                 recorded_steps=total, lead_exceed_steps=exceed,
                                 lead_exceed_fraction=exceed / max(1, total))
            worst, worst_step = 0.0, None
            for k in range(len(q)):
                obs = {key: a[key][k].astype(float) for key in ("base_pos", "base_ori", "q_mes")}
                tau = pd_layer(e_ref[k], a["qd_des_arm"][k], q[k], a["qd_mes"][k, 4:11],
                               q_pin_from_obs(obs, self.info), self.info)
                err = float(np.max(np.abs(tau - a["torque_arm"][k].astype(float))))
                if err > worst:
                    worst, worst_step = err, k
            maximum = max(maximum, worst)
            self.evidence["recorded"].append({"episode": e, "steps": len(q), "max_error": worst,
                                                "worst_step": worst_step})
            self.evidence["recorded_max_error"] = maximum
            self.assertLessEqual(worst, 1e-3, f"recorded torque mismatch in episode {e}, step {worst_step}")
        self.assertEqual(len(self.evidence["recorded"]), 10)

    def test_c_deployment_matches_offline_on_recorded_episode(self):
        with np.load(self.episode_paths[0]) as d:
            a = {key: d[key] for key in ("base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "target")}
        X = bcf.build_features("ego_novw", bcf.measured_signals(
            a["base_pos"], a["base_bearing"], a["q_mes"], a["qd_mes"], a["target"]))
        rng = np.random.default_rng(19)
        W = rng.normal(0, .01, (20, 16))
        b = rng.normal(0, .005, 16)
        meta = {"output_representation": "pd_reference",
                "output_names": ["v", "w"] + [f"e{j}" for j in range(1, 8)]
                                + [f"qd_des{j}" for j in range(1, 8)]}
        with tempfile.TemporaryDirectory(dir=SMOKE) as tmp:
            path = os.path.join(tmp, "pd_model.npz")
            # Arm scale sqrt(2) is also checked through the shared de-standardisation path.
            bcf.save_model(path, "ego_novw", X.mean(0), np.maximum(X.std(0), 1e-6), np.zeros(16),
                           np.r_[np.ones(2), np.repeat(np.sqrt(2), 14)], [W], [b], meta)
            policy = make_policy(path)
            self.assertIsInstance(policy, PDPolicy)
            policy.reset(None, self.info)
            predictions = bcf.forward(policy.model, X)
            maximum = 0.0
            for k, y in enumerate(predictions):
                obs = recorded_obs(a, k)
                offline_tau = pd_layer(y[2:9], y[9:16], obs["q_mes"][4:11], obs["qd_mes"][4:11],
                                       q_pin_from_obs(obs, self.info), self.info)
                wheels, tau = policy.act(obs)
                maximum = max(maximum, float(np.max(np.abs(tau - offline_tau))),
                              float(np.max(np.abs(wheels - wheel_speeds(y[0], y[1], BASE_PARAMS)))))
            self.evidence.update(deploy_steps=len(X), deploy_max_error=maximum)
            self.assertLess(maximum, 1e-9)

    def test_revision1_navigation_reference_is_not_clipped(self):
        q = self.info["init_q_mes"].copy()
        hold = q[4:11].copy()
        q[4] += .2
        obs = {"q_mes": q, "qd_mes": np.zeros_like(q), "base_pos": np.array([0., 0., .15]),
               "base_ori": np.array([0., 0., 0., 1.])}
        tau, q_des, qd_des, q_pin = self.controller(obs, hold, hold, np.array([1., 0., .5]), False)
        actual = pd_layer(q_des - q[4:11], qd_des, q[4:11], np.zeros(7), q_pin, self.info)
        difference = float(np.max(np.abs(actual - tau)))
        self.evidence["navigation_unclipped_reference_max_error"] = difference
        self.assertGreater(abs(q_des[0] - q[4]), self.params.max_q_lead)
        self.assertLess(q_des[5], self.info["joint_lower"][9])
        self.assertLessEqual(difference, 1e-9)


if __name__ == "__main__":
    unittest.main()
