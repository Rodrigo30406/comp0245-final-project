"""Unit tests of experiment D (STEP4_PLAN_PROMPT.md 2.1): the near-field end-effector error features ego_novw_ee, the
forward kinematics of the deployed BCPolicy, the ee side cache and the training path of ego-novw-trig-ee.

Run from the task directory:
  python -m unittest tests.test_ee_features -v
The test model (random weights, scalers from data/pilot episodes 0-9) is written to results/tmp_policies/.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(TASK_DIR, "policies")
for _d in (TASK_DIR, POLICY_DIR, os.path.join(TASK_DIR, "tests")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

import bc_features as bcf  # noqa: E402
import train_policy as tp  # noqa: E402
from bc_policy import BASE_PARAMS, BCPolicy  # noqa: E402
from collect_data import wheel_speeds  # noqa: E402
from test_policies import PILOT_DIR, TMP_DIR, STATS_EPISODES, episode_labels, is_clean, obs_at, pilot_index  # noqa: E402
import test_training as tt  # noqa: E402

EE_TEST_MODEL = os.path.join(TMP_DIR, "test_ego_novw_ee_mlp.npz")


def load_episode_ee(path):
    with np.load(path) as d:
        ep = {k: d[k] for k in ("t", "base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "wheel_cmds",
                                "torque_arm", "nav_done", "ee_pos", "target")}
        ep["meta"] = json.loads(str(d["meta"]))
    return ep


def ee_signals(ep, idx, ee_pos=None):
    return bcf.measured_signals(ep["base_pos"][idx], ep["base_bearing"][idx], ep["q_mes"][idx], ep["qd_mes"][idx],
                                ep["target"], ee_pos=ep["ee_pos"][idx] if ee_pos is None else ee_pos)


def make_ee_test_model():
    """Random-weight 2 x 128 MLP on ego_novw_ee, scalers from data/pilot episodes 0-9 (every 9th step)."""
    if os.path.exists(EE_TEST_MODEL):
        return EE_TEST_MODEL
    os.makedirs(TMP_DIR, exist_ok=True)
    eps = [load_episode_ee(os.path.join(PILOT_DIR, f"ep_{i:04d}.npz")) for i in STATS_EPISODES]
    X = np.concatenate([bcf.build_features("ego_novw_ee", ee_signals(ep, slice(None, None, 9))) for ep in eps])
    Y = np.concatenate([episode_labels(ep, slice(None, None, 9)) for ep in eps])
    rng = np.random.default_rng(54321)
    sizes = [X.shape[1], 128, 128, 9]
    weights = [rng.normal(0.0, 1.0 / np.sqrt(a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    biases = [rng.normal(0.0, 0.1, b) for b in sizes[1:]]
    tmp = EE_TEST_MODEL + f".tmp{os.getpid()}.npz"
    bcf.save_model(tmp, "ego_novw_ee", X.mean(0), np.maximum(X.std(0), 1e-6), Y.mean(0), np.maximum(Y.std(0), 1e-6),
                   weights, biases, {"kind": "test model: random weights, ego_novw_ee"})
    os.replace(tmp, EE_TEST_MODEL)
    return EE_TEST_MODEL


def sim_info():
    from evaluate_policy import setup_simulation
    with redirect_stdout(io.StringIO()):
        _, client, _, info = setup_simulation(np.zeros(2), 0.0)
    client.disconnect()
    return info


def train_episode_paths(n_clean=3, n_noisy=3, n_far=2):
    """Some data/train episodes: clean, noisy, and those whose base got farthest from the origin (largest |base xy| in
    the training cache)."""
    split = tp.load_split()
    eps = split["train_episodes"]
    info = split["episodes"]
    clean = [e for e in eps if not info[str(e)]["noisy"]][:n_clean]
    noisy = [e for e in eps if info[str(e)]["noisy"]][:n_noisy]
    far = []
    if n_far:
        c = tp.Cache("train")
        r = np.linalg.norm(np.asarray(c.cols["base_xy"], dtype=float), axis=1)
        ep_col = np.asarray(c.cols["episode"])
        order = np.argsort(-r)
        for i in order:
            e = int(ep_col[i])
            if e not in far:
                far.append(e)
            if len(far) == n_far:
                break
        c.release()
    return [os.path.join(tp.TRAIN_DIR, info[str(e)]["file"]) for e in clean + noisy + far]


class TestClipAndFeatures(unittest.TestCase):
    def test_clip(self):
        rng = np.random.default_rng(0)
        e = rng.normal(0, 0.2, (5000, 3))
        c = bcf.clip_ee_error(e)
        n, nc = np.linalg.norm(e, axis=1), np.linalg.norm(c, axis=1)
        inside = n < bcf.EE_CLIP
        np.testing.assert_array_equal(c[inside], e[inside])
        np.testing.assert_allclose(nc[~inside], bcf.EE_CLIP, rtol=1e-12)
        np.testing.assert_allclose(c[~inside] / nc[~inside, None], e[~inside] / n[~inside, None], atol=1e-12)
        np.testing.assert_array_equal(bcf.clip_ee_error(np.zeros((1, 3))), np.zeros((1, 3)))

    def test_feature_definition(self):
        """ee_dx, ee_dy = the clipped error rotated into the base frame (as ego_target), ee_dz = its z part; the first
        20 columns are ego_novw."""
        rng = np.random.default_rng(1)
        n = 2000
        base = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0, 0.2, n)])
        yaw = rng.uniform(-np.pi, np.pi, n)
        target = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0.2, 1.2, n)])
        ee = target + rng.normal(0, 0.1, (n, 3))
        q, qd = rng.normal(size=(n, 17)), rng.normal(size=(n, 17))
        sig = bcf.measured_signals(base, yaw, q, qd, target, ee_pos=ee)
        X = bcf.build_features("ego_novw_ee", sig)
        self.assertEqual(X.shape, (n, 23))
        np.testing.assert_array_equal(X[:, :20], bcf.build_features("ego_novw", sig))
        c = bcf.clip_ee_error(target - ee)
        dx, dy = bcf.ego_target(np.zeros((n, 2)), yaw, c[:, :2])
        np.testing.assert_allclose(X[:, 20], dx, atol=1e-12)
        np.testing.assert_allclose(X[:, 21], dy, atol=1e-12)
        np.testing.assert_array_equal(X[:, 22], c[:, 2])
        with self.assertRaises(ValueError):   # no silent zeros without ee_pos
            bcf.build_features("ego_novw_ee", bcf.measured_signals(base, yaw, q, qd, target))
        # the other feature sets ignore ee_pos
        sig0 = bcf.measured_signals(base, yaw, q, qd, target)
        for fs in ("ego", "ego_novw", "raw"):
            np.testing.assert_array_equal(bcf.build_features(fs, sig), bcf.build_features(fs, sig0))

    def test_rigid_motion_invariance(self):
        rng = np.random.default_rng(2)
        n = 3000
        base = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0, 0.2, n)])
        yaw = rng.uniform(-np.pi, np.pi, n)
        target = np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0.2, 1.2, n)])
        keep = np.hypot(*(target[:, :2] - base[:, :2]).T) > 0.01
        base, yaw, target = base[keep], yaw[keep], target[keep]
        n = len(yaw)
        ee = target + rng.normal(0, 0.08, (n, 3))
        q, qd = rng.normal(size=(n, 17)), rng.normal(size=(n, 17))
        ref = bcf.build_features("ego_novw_ee", bcf.measured_signals(base, yaw, q, qd, target, ee_pos=ee))
        for k in range(5):
            phi, shift = rng.uniform(-np.pi, np.pi), rng.uniform(-5, 5, 2)
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            b2, t2, e2 = base.copy(), target.copy(), ee.copy()
            b2[:, :2] = base[:, :2] @ R.T + shift
            t2[:, :2] = target[:, :2] @ R.T + shift
            e2[:, :2] = ee[:, :2] @ R.T + shift
            yaw2 = (yaw + phi + np.pi) % (2 * np.pi) - np.pi
            got = bcf.build_features("ego_novw_ee", bcf.measured_signals(b2, yaw2, q, qd, t2, ee_pos=e2))
            self.assertLess(np.abs(got - ref).max(), 1e-9, k)


@unittest.skipUnless(os.path.exists(os.path.join(TASK_DIR, "results", "step3", "cache", "manifest.json"))
                     and os.path.exists(os.path.join(TASK_DIR, "results", "step3", "cache", "trigger", "manifest.json")),
                     "needs the step-3 feature caches (train_policy.py build-cache, build-trigger-cache)")
class TestDeployment(unittest.TestCase):
    """Forward kinematics of BCPolicy vs the recorded ee_pos; train / deploy consistency; refusals."""

    @classmethod
    def setUpClass(cls):
        cls.info = sim_info()
        cls.model_path = make_ee_test_model()
        cls.paths = train_episode_paths()

    def policy(self):
        p = BCPolicy(self.model_path)
        p.reset(None, self.info)
        return p

    def test_fk_reproduces_recorded_ee(self):
        p = self.policy()
        worst, n = 0.0, 0
        for path in self.paths:
            ep = load_episode_ee(path)
            for k in np.unique(np.linspace(0, len(ep["t"]) - 1, 300 // len(self.paths) + 1).round().astype(int)):
                worst = max(worst, float(np.abs(p.ee_position(obs_at(ep, k)) - ep["ee_pos"][k]).max()))
                n += 1
        self.assertGreaterEqual(n, 300)
        self.assertLess(worst, 1e-6)

    def test_train_deploy_consistency(self):
        """(i) features from the recorded ee_pos vs from the deployed FK: ee columns < 1e-6, others < 1e-9;
        (ii) outputs: the vectorised path fed with the FK ee_pos equals BCPolicy.act to 1e-9."""
        p = self.policy()
        model = bcf.load_model(self.model_path)
        index = pilot_index()
        clean = next(i for i, r in sorted(index.items()) if is_clean(r) and r["status"] == "reached")
        noisy = next(i for i, r in sorted(index.items()) if not is_clean(r) and r["status"] == "reached")
        n = 0
        for i in (clean, noisy):
            ep = load_episode_ee(os.path.join(PILOT_DIR, f"ep_{i:04d}.npz"))
            idx = np.unique(np.linspace(0, len(ep["t"]) - 1, 150).round().astype(int))
            fk = np.array([p.ee_position(obs_at(ep, k)) for k in idx])
            X_rec = bcf.build_features("ego_novw_ee", ee_signals(ep, idx))
            X_fk = bcf.build_features("ego_novw_ee", ee_signals(ep, idx, ee_pos=fk))
            self.assertLess(np.abs(X_rec[:, 20:] - X_fk[:, 20:]).max(), 1e-6)
            self.assertLess(np.abs(X_rec[:, :20] - X_fk[:, :20]).max(), 1e-9)
            Y = bcf.forward(model, X_fk)
            wheels_ref = wheel_speeds(Y[:, 0], Y[:, 1], BASE_PARAMS).T
            for row, k in enumerate(idx):
                wheels, tau = p.act(obs_at(ep, k))
                self.assertLess(np.abs(wheels - wheels_ref[row]).max(), 1e-9)
                self.assertLess(np.abs(tau - Y[row, 2:]).max(), 1e-9)
                n += 1
        self.assertEqual(n, 300)

    def test_refusals(self):
        p = BCPolicy(self.model_path)
        for bad in (None, {}, {"dyn_model": None, "ee_frame_id": 3}):
            with self.assertRaises(ValueError):
                p.reset(None, bad)
        ep = load_episode_ee(self.paths[0])
        with self.assertRaises(ValueError):   # predict before a reset with the robot model
            BCPolicy(self.model_path).act(obs_at(ep, 10))
        # an old feature set still accepts reset(None, None) and never touches info
        from test_policies import make_test_models
        q = BCPolicy(make_test_models()["ego_novw"])
        q.reset(None, None)
        q.act(obs_at(ep, 10))

    def test_hybrid_em_cell(self):
        """hybrid base=expert;arm=<ee model>: the arm torque is BCPolicy.act's (the model gets info at reset); without
        the robot model in info the reset is refused."""
        from hybrid import HybridPolicy
        from test_hybrid import pilot_eval_episodes, sim_info as hybrid_sim_info
        from test_policies import load_episode
        episode = pilot_eval_episodes()[0]
        info = hybrid_sim_info(episode)
        ep = load_episode(episode["episode"])
        hyb = HybridPolicy(f"base=expert;arm={self.model_path}")
        with redirect_stdout(io.StringIO()):
            hyb.reset(np.asarray(episode["target"], dtype=float), info)
        ref = BCPolicy(self.model_path)
        ref.reset(None, info)
        for k in range(0, 400, 40):
            obs = obs_at(ep, k, t=0.001 * k)
            with redirect_stdout(io.StringIO()):
                _, tau = hyb.act({key: np.copy(v) for key, v in obs.items()})
            np.testing.assert_array_equal(tau, ref.act(obs)[1])
        bad = {k: v for k, v in info.items() if k not in ("dyn_model", "ee_frame_id")}
        with self.assertRaises((ValueError, KeyError)), redirect_stdout(io.StringIO()):
            HybridPolicy(f"base=expert;arm={self.model_path}").reset(np.asarray(episode["target"], dtype=float), bad)


class TestOldModelsUnchanged(unittest.TestCase):
    """The deployed outputs of an existing model are bit-identical with the committed (HEAD) bc_features.py and
    bc_policy.py: the C3 control runs stay valid for the current code."""

    def test_bitwise_against_head(self):
        model = os.path.join(TASK_DIR, "trained", "ego-novw-trig_s0.npz")
        if not os.path.exists(model):
            self.skipTest("trained/ego-novw-trig_s0.npz missing")
        tmp = tempfile.mkdtemp()
        try:
            root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=TASK_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
            rel = os.path.relpath(TASK_DIR, root)
            for f in ("bc_features.py", "policies/bc_policy.py"):
                src = subprocess.run(["git", "show", f"HEAD:{rel}/{f}"], cwd=TASK_DIR, capture_output=True,
                                     text=True, check=True).stdout
                with open(os.path.join(tmp, os.path.basename(f)), "w") as fh:
                    fh.write(src)
            ep_path = train_episode_paths(1, 1, 0)
            script = (
                "import sys, numpy as np\n"
                f"sys.path[:0] = [{tmp!r}, {TASK_DIR!r}]\n"
                "import bc_features, bc_policy\n"
                f"assert bc_features.__file__.startswith({tmp!r}) and bc_policy.__file__.startswith({tmp!r})\n"
                f"sys.path.insert(0, {os.path.join(TASK_DIR, 'tests')!r})\n"
                "from test_policies import obs_at\n"
                f"p = bc_policy.BCPolicy({model!r}); p.reset(None, None)\n"
                "out = []\n"
                f"for path in {ep_path!r}:\n"
                "    d = dict(np.load(path))\n"
                "    for k in np.linspace(0, len(d['t']) - 1, 150).round().astype(int):\n"
                "        w, t = p.act(obs_at(d, k)); out.append(np.concatenate([w, t]))\n"
                f"np.save({os.path.join(tmp, 'old.npy')!r}, np.array(out))\n")
            subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
            old = np.load(os.path.join(tmp, "old.npy"))
            p = BCPolicy(model)
            p.reset(None, None)
            new = []
            for path in ep_path:
                d = dict(np.load(path))
                for k in np.linspace(0, len(d["t"]) - 1, 150).round().astype(int):
                    w, t = p.act(obs_at(d, k))
                    new.append(np.concatenate([w, t]))
            self.assertEqual(old.shape, (300, 11))
            np.testing.assert_array_equal(old, np.array(new))
        finally:
            shutil.rmtree(tmp)


class TestEeCacheAndTraining(unittest.TestCase):
    """The ee side cache of the small test cache (3 training, 1 validation episode) and ego-novw-trig-ee training."""

    @classmethod
    def setUpClass(cls):
        tt.TestCache.setUpClass.__func__(cls)
        cls.trig_dir = os.path.join(cls.cache_dir, "trigger")
        tp.build_trigger_cache(cls.split_path, cls.trig_dir, workers=2, train_episodes=cls.train_eps, verbose=False)
        cls.manifest = tp.build_ee_cache(cls.cache_dir, cls.trig_dir, cls.split_path, workers=2, allow_subset=True,
                                         verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_ee_rows_equal_recording(self):
        for part, cache in (("train", self.tr), ("val", self.va)):
            ee, _ = tp.load_ee(part, self.cache_dir, self.split_path, allow_subset=True)
            self.assertEqual(ee.shape, (cache.n_rows, 3))
            for i, e in enumerate(cache.episodes):
                sl = cache.episode_slice(i)
                with np.load(os.path.join(tp.TRAIN_DIR, self.split["episodes"][str(e)]["file"])) as d:
                    ref = d["ee_pos"][np.asarray(cache.cols["step"][sl])]
                np.testing.assert_array_equal(np.asarray(ee[sl]), ref)
        trig, _ = tp.load_trigger_rows(self.train_eps, self.trig_dir, self.split_path, allow_subset=True)
        ee_t, _ = tp.ee_trigger_rows(self.train_eps, self.cache_dir, self.split_path, self.trig_dir, allow_subset=True)
        for i, e in enumerate(self.train_eps):
            with np.load(os.path.join(tp.TRAIN_DIR, self.split["episodes"][str(e)]["file"])) as d:
                np.testing.assert_array_equal(ee_t[i], d["ee_pos"][int(trig["step"][i])])

    @unittest.skipUnless(os.path.exists(os.path.join(TASK_DIR, "results", "step3", "cache", "manifest.json"))
                         and os.path.exists(os.path.join(TASK_DIR, "results", "step3", "cache", "trigger", "manifest.json")),
                         "needs the step-3 feature caches (train_policy.py build-cache, build-trigger-cache)")
    def test_refusals(self):
        with self.assertRaises(RuntimeError):   # a subset side cache never feeds an official model
            tp.load_ee("train", self.cache_dir, self.split_path)
        with self.assertRaises(RuntimeError):   # built for another trigger cache
            tp.load_ee("trigger", self.cache_dir, self.split_path, trigger_dir=tp.TRIGGER_DIR, allow_subset=True)
        other = os.path.join(self.tmp, "cache_other")
        shutil.copytree(self.cache_dir, other)
        with open(os.path.join(other, "manifest.json")) as f:
            m = json.load(f)
        m["created"] = "changed"
        with open(os.path.join(other, "manifest.json"), "w") as f:
            json.dump(m, f)
        with self.assertRaises(RuntimeError):   # the main cache changed after the side cache was built
            tp.load_ee("train", other, self.split_path, allow_subset=True)

    def test_training_uses_recorded_ee(self):
        X, _, Xv, _, _, _ = tp.load_training_data("ego_novw_ee", self.cache_dir, self.split_path, allow_subset=True)
        X0, _, Xv0, _, _, _ = tp.load_training_data("ego_novw", self.cache_dir, self.split_path, allow_subset=True)
        self.assertEqual(X.shape[1], 23)
        np.testing.assert_array_equal(X[:, :20], X0)
        np.testing.assert_array_equal(Xv[:, :20], Xv0)
        i, e = 0, self.va.episodes[0]
        sl = self.va.episode_slice(i)
        ep = load_episode_ee(os.path.join(tp.TRAIN_DIR, self.split["episodes"][str(e)]["file"]))
        np.testing.assert_array_equal(Xv[sl], bcf.build_features("ego_novw_ee", ee_signals(ep, slice(None))))
        out = os.path.join(self.tmp, "ee_model.npz")
        _, meta = tp.run_train("ego-novw-trig-ee", seed=1, out=out, cache_dir=self.cache_dir,
                               split_path=self.split_path, max_epochs=1, max_train_episodes=tt.N_MINI_TRAIN,
                               log=lambda s: None, trigger_copies_k=3, trigger_dir=self.trig_dir)
        m = bcf.load_model(out)
        self.assertEqual((m["feature_set"], len(m["x_mean"])), ("ego_novw_ee", 23))
        self.assertEqual(m["meta"]["trigger"]["n_rows_added"], 3 * tt.N_MINI_TRAIN)
        self.assertIn("ee_cache_manifest_sha256", m["meta"])
        self.assertEqual(m["meta"]["ee_clip_m"], bcf.EE_CLIP)
        # the trigger copies carry the recorded k0 end-effector position
        trig, _ = tp.load_trigger_rows(self.train_eps, self.trig_dir, self.split_path, allow_subset=True)
        trig["ee_pos"], _ = tp.ee_trigger_rows(self.train_eps, self.cache_dir, self.split_path, self.trig_dir,
                                               allow_subset=True)
        Xt, _, _ = tp.trigger_copies(trig, "ego_novw_ee", 2, tp.hold_qd_sigma(self.tr.cols))
        ref = bcf.build_features("ego_novw_ee", tp.signals_from_cache(trig, slice(0, tt.N_MINI_TRAIN)))
        np.testing.assert_array_equal(Xt[:, 20:], np.repeat(ref[:, 20:], 2, axis=0))


if __name__ == "__main__":
    unittest.main()
