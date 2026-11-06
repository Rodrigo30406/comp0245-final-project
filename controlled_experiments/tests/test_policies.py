"""Unit tests of the behaviour-cloning policy (policies/bc_policy.py) and of the conventions it relies on.

  (b) (v, w) -> collect_data.wheel_speeds -> bc_features.wheels_to_vw round trip
  (c) ego / ego_novw features invariant under a planar rigid motion of base and target; raw features are not
  (d) sign conventions: (d1) w > 0 turns the base to the left (yaw increases) in pybullet, v > 0 drives along the
      heading; (d2) on clean data/pilot M2 episodes the first non-zero w label has the sign of beta at t = 0;
      (d3) the measured (v_mes, w_mes) features follow the commanded (v, w) with the same sign
  (e) BCPolicy.act on single observations == the vectorised training path, for every feature set

Run from the task directory:
  python -m unittest tests/test_policies.py -v
The test models (random weights, and an OLS fitted on a few data/pilot episodes) are written to
results/tmp_policies/ by make_test_models(); they carry no information about any trained model.
"""
import csv
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(TASK_DIR, "policies")
for _d in (TASK_DIR, POLICY_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

import bc_features  # noqa: E402
from bc_policy import BASE_PARAMS, BCPolicy, make_policy  # noqa: E402
from collect_data import wheel_speeds  # noqa: E402

PILOT_DIR = os.path.join(TASK_DIR, "data", "pilot")
TMP_DIR = os.path.join(TASK_DIR, "results", "tmp_policies")
# Random-weight MLPs (2 x 128, the architecture of the trained models) for each feature set, and a linear model
TEST_MODELS = {
    "ego": os.path.join(TMP_DIR, "test_ego_mlp.npz"),
    "ego_novw": os.path.join(TMP_DIR, "test_ego_novw_mlp.npz"),
    "raw": os.path.join(TMP_DIR, "test_raw_mlp.npz"),
    "ego_ols": os.path.join(TMP_DIR, "test_ego_ols.npz"),
}
STATS_EPISODES = range(10)  # data/pilot episodes used for the scalers of the test models and for the OLS fit


def pilot_index():
    with open(os.path.join(PILOT_DIR, "index.csv")) as f:
        return {int(row["episode"]): row for row in csv.DictReader(f)}


def is_clean(row):
    return all(float(row[k]) == 0.0 for k in ("noise_v", "noise_w", "noise_torque"))


def load_episode(i):
    with np.load(os.path.join(PILOT_DIR, f"ep_{i:04d}.npz")) as d:
        ep = {k: d[k] for k in ("t", "base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "wheel_cmds",
                                "torque_arm", "phase", "nav_done", "target")}
        ep["meta"] = json.loads(str(d["meta"]))
    return ep


def episode_signals(ep, idx=slice(None)):
    return bc_features.measured_signals(ep["base_pos"][idx], ep["base_bearing"][idx], ep["q_mes"][idx],
                                        ep["qd_mes"][idx], ep["target"])


def episode_labels(ep, idx=slice(None)):
    v, w = bc_features.wheels_to_vw(ep["wheel_cmds"][idx])
    return np.column_stack([v, w, np.asarray(ep["torque_arm"][idx], dtype=float)])


def obs_at(ep, k, t=None):
    """The observation of recorded step k, as evaluate_policy passes it (float64 arrays, float scalars)."""
    return {"t": float(ep["t"][k]) if t is None else t, "base_pos": ep["base_pos"][k].astype(float),
            "base_ori": ep["base_ori"][k].astype(float), "base_bearing": float(ep["base_bearing"][k]),
            "q_mes": ep["q_mes"][k].astype(float), "qd_mes": ep["qd_mes"][k].astype(float),
            "target": np.asarray(ep["target"], dtype=float).copy()}


def _atomic_save(path, *args):
    tmp = path + f".tmp{os.getpid()}.npz"
    bc_features.save_model(tmp, *args)
    os.replace(tmp, path)


def make_test_models(force=False):
    """Write the test models to results/tmp_policies/ (deterministic). Returns TEST_MODELS.

    Scalers: mean / std of the features and labels of STATS_EPISODES (so that the random MLPs produce outputs of
    a realistic size). ego_ols: least squares fit of the standardised labels on the standardised ego features of the
    same episodes (every 9th step), only for smoke tests of the closed-loop path."""
    if not force and all(os.path.exists(p) for p in TEST_MODELS.values()):
        return TEST_MODELS
    os.makedirs(TMP_DIR, exist_ok=True)
    episodes = [load_episode(i) for i in STATS_EPISODES]
    Y = np.concatenate([episode_labels(ep, slice(None, None, 9)) for ep in episodes])
    y_mean, y_scale = Y.mean(axis=0), np.maximum(Y.std(axis=0), 1e-6)
    rng = np.random.default_rng(12345)
    for name, fs in (("ego", "ego"), ("ego_novw", "ego_novw"), ("raw", "raw"), ("ego_ols", "ego")):
        X = np.concatenate([bc_features.build_features(fs, episode_signals(ep, slice(None, None, 9)))
                            for ep in episodes])
        x_mean, x_scale = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-6)
        if name == "ego_ols":
            Xs = np.column_stack([(X - x_mean) / x_scale, np.ones(len(X))])
            coef = np.linalg.lstsq(Xs, (Y - y_mean) / y_scale, rcond=None)[0]
            weights, biases = [coef[:-1]], [coef[-1]]
            meta = {"kind": "test model: OLS on data/pilot episodes 0-9 (every 9th step), smoke tests only"}
        else:
            sizes = [X.shape[1], 128, 128, len(bc_features.OUTPUT_NAMES)]
            weights = [rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_in, n_out)) for n_in, n_out in zip(sizes, sizes[1:])]
            biases = [rng.normal(0.0, 0.1, n_out) for n_out in sizes[1:]]
            meta = {"kind": "test model: random weights, scalers from data/pilot episodes 0-9"}
        _atomic_save(TEST_MODELS[name], fs, x_mean, x_scale, y_mean, y_scale, weights, biases, meta)
    return TEST_MODELS


NEEDS_DATA = unittest.skipUnless(  # recorded demonstrations are not in git
    all(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", d, "index.csv"))
        for d in ("train", "pilot")),
    "needs data/train and data/pilot (collect_data.py, see DATA_COLLECTION.md)")


class TestWheelRoundTrip(unittest.TestCase):
    """(b) (v, w) -> wheel speeds -> (v, w)."""

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        v = rng.uniform(-1.0, 1.0, 10000)
        w = rng.uniform(-3.0, 3.0, 10000)
        wheels = wheel_speeds(v, w, BASE_PARAMS).T  # (N, 4) [rw, lw, lw, rw]
        v2, w2 = bc_features.wheels_to_vw(wheels)
        self.assertLess(max(np.max(np.abs(v2 - v)), np.max(np.abs(w2 - w))), 1e-12)
        # scalar path, as used by BCPolicy.act
        for k in range(100):
            v1, w1 = bc_features.wheels_to_vw(wheel_speeds(v[k], w[k], BASE_PARAMS))
            self.assertLess(max(abs(v1 - v[k]), abs(w1 - w[k])), 1e-12)

    def test_constants_match_expert(self):
        self.assertEqual(bc_features.WHEEL_RADIUS, BASE_PARAMS.wheel_radius)
        self.assertEqual(bc_features.WHEEL_BASE_WIDTH, BASE_PARAMS.wheel_base_width)


class TestInvariance(unittest.TestCase):
    """(c) A planar rotation + translation applied jointly to the base pose and the target leaves ego unchanged."""

    def test_rigid_motion(self):
        rng = np.random.default_rng(1)
        n = 5000
        base_pos = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0.0, 0.2, n)])
        yaw = rng.uniform(-np.pi, np.pi, n)
        target = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0.2, 1.2, n)])
        keep = np.hypot(*(target[:, :2] - base_pos[:, :2]).T) > 0.01  # the bearing is undefined at the target
        base_pos, yaw, target = base_pos[keep], yaw[keep], target[keep]
        n = len(yaw)
        q_mes, qd_mes = rng.normal(0, 1, (n, 17)), rng.normal(0, 1, (n, 17))
        signals = bc_features.measured_signals(base_pos, yaw, q_mes, qd_mes, target)
        for k in range(10):
            phi, shift = rng.uniform(-np.pi, np.pi), rng.uniform(-5, 5, 2)
            rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            base2, target2 = base_pos.copy(), target.copy()
            base2[:, :2] = base_pos[:, :2] @ rot.T + shift
            target2[:, :2] = target[:, :2] @ rot.T + shift
            yaw2 = (yaw + phi + np.pi) % (2 * np.pi) - np.pi  # the measured bearing wraps at +-pi
            signals2 = bc_features.measured_signals(base2, yaw2, q_mes, qd_mes, target2)
            for fs in ("ego", "ego_novw"):
                diff = np.abs(bc_features.build_features(fs, signals) - bc_features.build_features(fs, signals2))
                self.assertLess(diff.max(), 1e-9, f"{fs} transform {k}")
            raw_diff = np.abs(bc_features.build_features("raw", signals) - bc_features.build_features("raw", signals2))
            self.assertGreater(raw_diff.max(), 0.1, "raw features should change under a rigid motion")


class TestSignConventions(unittest.TestCase):
    """(d) The sign of w and of beta, and the forward direction of v."""

    @staticmethod
    def run_base_command(v, w, init_yaw, duration=1.0):
        """Command wheel_speeds(v, w) for `duration` s from (0, 0, init_yaw), arm held by the expert's hold law.
        Returns (start xy, start yaw, end xy, end yaw)."""
        import pinocchio  # noqa: F401  (imported by evaluate_policy anyway)
        from simulation_and_control import MotorCommands
        from evaluate_policy import setup_simulation
        from robotnik_arm_control import ArmControllerParams, arm_controller, quaternion2bearing
        from robotnik_arm_task import ARM_REACH_LIMIT

        with redirect_stdout(io.StringIO()):
            sim, client, rid, info = setup_simulation(np.zeros(2), init_yaw)
        try:
            model, params = info["dyn_model"], ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
            q_hold = info["init_q_mes"][4:11].copy()
            lower, upper = info["joint_lower"][4:11], info["joint_upper"][4:11]
            cmd = MotorCommands()
            modes = ["velocity"] * 4 + ["torque"] * 13
            wheels = wheel_speeds(v, w, BASE_PARAMS)

            def pose():
                ori = np.array(sim.GetBaseOrientation())
                return np.array(sim.GetBasePosition())[:2], quaternion2bearing(ori[3], ori[0], ori[1], ori[2])

            xy0, yaw0 = pose()
            for _ in range(round(duration / info["time_step"])):
                q_mes, qd_mes = np.array(sim.GetMotorAngles(0)), np.array(sim.GetMotorVelocities(0))
                q_pin = model.ReoderJoints2PinVec(np.concatenate((sim.GetBasePosition(), sim.GetBaseOrientation(),
                                                                  q_mes)), "pos")
                tau, _, _ = arm_controller(model, q_pin, q_mes[4:11], qd_mes[4:11], q_hold, q_hold, None, None,
                                           None, None, lower, upper, info["time_step"], params, reach_active=False)
                cmd.SetControlCmd(np.concatenate((wheels, tau, np.zeros(6))), modes)
                sim.Step(cmd, "torque")
            xy1, yaw1 = pose()
        finally:
            client.disconnect()
        return xy0, yaw0, xy1, yaw1

    def test_d1_turn_sign(self):
        for w, sign in ((0.5, 1.0), (-0.5, -1.0)):
            _, yaw0, _, yaw1 = self.run_base_command(0.0, w, init_yaw=0.3)
            dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
            self.assertGreater(sign * dyaw, 0.1, f"w = {w}: yaw change {dyaw:.3f} rad in 1 s")

    def test_d1_forward_direction(self):
        """v > 0 moves the base along its heading (x forward in the ego frame)."""
        init_yaw = 1.0
        xy0, _, xy1, _ = self.run_base_command(0.2, 0.0, init_yaw=init_yaw)
        step = xy1 - xy0
        heading = np.array([np.cos(init_yaw), np.sin(init_yaw)])
        self.assertGreater(step @ heading, 0.05)
        self.assertLess(abs(step @ np.array([-heading[1], heading[0]])), 0.3 * (step @ heading))

    @NEEDS_DATA
    def test_d2_beta_sign(self):
        """beta > 0 (target to the left) -> the expert's first non-zero w label is positive (turn left)."""
        from analyze_dataset import mode_of
        from robotnik_arm_task import STANDOFF_POINTS, STANDOFF_Z_POINTS

        checked = 0
        for i, row in sorted(pilot_index().items()):
            if not is_clean(row):
                continue
            d0 = float(row["r"]) - float(np.interp(float(row["target_z"]), STANDOFF_Z_POINTS, STANDOFF_POINTS))
            if mode_of(d0) != "M2":
                continue
            ep = load_episode(i)
            x0 = bc_features.build_features("ego", episode_signals(ep, slice(0, 1)))[0]
            names = bc_features.FEATURE_SETS["ego"]
            beta = np.arctan2(x0[names.index("sin_beta")], x0[names.index("cos_beta")])
            if not np.deg2rad(10) < abs(beta) < np.deg2rad(170):
                continue
            _, w = bc_features.wheels_to_vw(ep["wheel_cmds"])
            first = np.flatnonzero(np.abs(w) > 1e-6)
            self.assertTrue(first.size, f"episode {i}: no non-zero w label")
            self.assertEqual(np.sign(w[first[0]]), np.sign(beta),
                             f"episode {i}: beta {np.rad2deg(beta):.1f} deg, first w {w[first[0]]:.4f}")
            checked += 1
        self.assertGreaterEqual(checked, 15, "too few qualifying pilot episodes for a meaningful check")

    @NEEDS_DATA
    def test_d3_measured_speeds_follow_labels(self):
        """The measured (v_mes, w_mes) features follow the commanded (v, w) with the same sign and scale (wheel
        order of qd_mes = that of the commands), pooled over 5 clean pilot episodes, one step later."""
        labels, measured = [], []
        for i in [i for i, row in sorted(pilot_index().items()) if is_clean(row)][:5]:
            ep = load_episode(i)
            labels.append(np.column_stack(bc_features.wheels_to_vw(ep["wheel_cmds"]))[:-1])
            measured.append(np.column_stack(bc_features.wheels_to_vw(ep["qd_mes"][:, bc_features.WHEELS]))[1:])
        labels, measured = np.concatenate(labels), np.concatenate(measured)
        for j, name in enumerate(("v", "w")):
            moving = np.abs(labels[:, j]) > 1e-3
            a, b = labels[moving, j], measured[moving, j]
            self.assertGreater(moving.sum(), 1000)
            self.assertGreater(np.corrcoef(a, b)[0, 1], 0.9, name)
            slope = (a @ b) / (a @ a)
            self.assertTrue(0.8 < slope < 1.2, f"{name}: measured / commanded slope {slope:.3f}")


@NEEDS_DATA
class TestTrainDeployConsistency(unittest.TestCase):
    """(e) BCPolicy.act(obs) == build_features(measured_signals(arrays)) -> forward -> wheel_speeds."""

    @classmethod
    def setUpClass(cls):
        make_test_models()
        index = pilot_index()
        clean = next(i for i, row in sorted(index.items()) if is_clean(row) and row["status"] == "reached")
        noisy = next(i for i, row in sorted(index.items()) if not is_clean(row) and row["status"] == "reached")
        cls.episodes = [load_episode(clean), load_episode(noisy)]

    def test_act_matches_vectorised_path(self):
        for name, path in TEST_MODELS.items():
            policy = make_policy(path)
            policy.reset(None, None)  # uses neither the reset target nor info
            model = bc_features.load_model(path)
            n_steps = 0
            for ep in self.episodes:
                idx = np.unique(np.linspace(0, len(ep["t"]) - 1, 150).round().astype(int))
                Y = bc_features.forward(model, bc_features.build_features(model["feature_set"],
                                                                          episode_signals(ep, idx)))
                wheels_ref = wheel_speeds(Y[:, 0], Y[:, 1], BASE_PARAMS).T
                for row, k in enumerate(idx):
                    wheels, tau = policy.act(obs_at(ep, k))
                    self.assertEqual(np.shape(wheels), (4,))
                    self.assertEqual(np.shape(tau), (7,))
                    self.assertLess(np.max(np.abs(wheels - wheels_ref[row])), 1e-9, f"{name} step {k}")
                    self.assertLess(np.max(np.abs(tau - Y[row, 2:])), 1e-9, f"{name} step {k}")
                    n_steps += 1
            self.assertEqual(n_steps, 300)

    def test_load_policy_from_harness(self):
        """The harness's loader (file path + factory) builds the same policy."""
        from evaluate_policy import load_policy
        policy = load_policy(os.path.join(POLICY_DIR, "bc_policy.py") + ":make_policy", TEST_MODELS["ego"])
        obs = obs_at(self.episodes[0], 100)
        ref = BCPolicy(TEST_MODELS["ego"]).act(obs)
        out = policy.act(obs)
        self.assertTrue(np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1]))

    def test_policy_does_not_change_obs(self):
        obs = obs_at(self.episodes[0], 200)
        copy = {k: np.copy(v) for k, v in obs.items()}
        BCPolicy(TEST_MODELS["raw"]).act(obs)
        for k in obs:
            self.assertTrue(np.array_equal(obs[k], copy[k]), k)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            make_policy(None)
        with self.assertRaises(OSError):
            make_policy(os.path.join(TMP_DIR, "does_not_exist.npz"))


if __name__ == "__main__":
    unittest.main()
