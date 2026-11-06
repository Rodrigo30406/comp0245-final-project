"""Unit tests of the step-9 learners (Track B): obstacle_task/features.py, train.py, policies.py.

Gates of STEP9_PLAN_PROMPT.md section 7:
  B1  training-time features == deployment features (<= 1e-9) on recorded steps, every feature set
      (TestFeatureParity on synthetic episodes; TestPilot2Parity on Track A's recordings when they exist)
  B2  numpy forward == sklearn predict / predict_proba (<= 1e-9)                     (TestForwardEqualsSklearn)
  B3  copies: K per training episode, jitter only on the arm-qd columns, labels = the source row (TestTriggerCopies)
plus the split (determinism, filter, stratification), the cache, the epoch loop (chunked partial_fit = one call),
early stopping, determinism, resumability, the policies' act() shapes, side latching and the factories.

Run from the task directory:
  python -m unittest tests.test_obstacle_learning -v
Until Track A's recordings exist the data are synthetic episodes in the section-4 format (collect_data.LOG_FIELDS +
segment int8 + target + meta JSON with the scenario dict, scenarios.json, config.json, index.csv), written to a
temporary directory with real scenarios (obstacle_task.scenario) and the real lidar.
"""
import csv
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
from obstacle_task import features as obf  # noqa: E402
from obstacle_task import lidar  # noqa: E402
from obstacle_task import policies as obp  # noqa: E402
from obstacle_task import train as obt  # noqa: E402

PILOT2_DIR = os.path.join(TASK_DIR, "results", "step9", "smoke", "pilot2")
N_EPISODES = 40
COLLIDED_EPS = (3, 17)        # status collision, collided True: excluded
STEP_CAP_EPS = (8,)           # status step_cap: excluded
NAV_TIMEOUT_EPS = (12,)       # status reached with nav_timed_out True: kept by the plan's filter
TOL = 1e-9                    # gates B1, B2
FEATURE_OUTPUTS = {"arm20": "tau", "base70": "vw", "single84": "all", "base70_side": "vw"}
STATE = {}                    # the synthetic set, its split and cache (setUpModule)


# ==============================================================================
# Synthetic step-9 set (section-4 format)
# ==============================================================================
def write_synthetic_set(out_dir, n_episodes=N_EPISODES, seed=11, noise_frac=0.5, rng_seed=0):
    """n_episodes synthetic episodes of real scenarios: the base moves start -> waypoint -> dock (piecewise linear,
    yaw wiggling), arm signals and labels are random, nav_done from 75% of the episode (wheel commands 0 from then
    on, as the expert's), segment 1 from 35%. Episodes COLLIDED_EPS / STEP_CAP_EPS fail."""
    from collect_data import INDEX_COLUMNS, LOG_FIELDS, wheel_speeds
    from obstacle_task import scenario
    from robotnik_arm_control import BaseControllerParams
    params = BaseControllerParams(nav_timeout=45.0)
    os.makedirs(out_dir, exist_ok=True)
    scs = scenario.make_scenarios(seed, n_episodes, noise_frac)
    scenario.save_scenarios(os.path.join(out_dir, "scenarios.json"), scs, extra={"set": "synthetic"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"set": "synthetic", "seed": seed, "n_episodes": n_episodes, "noise_frac": noise_frac}, f)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for sc in scs:
        n = int(rng.integers(700, 1700))
        k_switch, k_nav = int(0.35 * n), int(0.75 * n)
        steps = np.arange(n)
        s = np.clip(steps / k_nav, 0.0, 1.0)
        p0, p1, p2 = (np.asarray(p, dtype=float) for p in (sc.init_xy, sc.waypoint_xy, sc.dock_xy))
        a = np.clip(s / 0.5, 0, 1)[:, None]
        b = np.clip((s - 0.5) / 0.5, 0, 1)[:, None]
        xy = np.where(s[:, None] <= 0.5, p0 + a * (p1 - p0), p1 + b * (p2 - p1)) + rng.normal(0, 0.01, (n, 2))
        yaw = np.angle(np.exp(1j * (sc.init_yaw + 0.8 * np.sin(steps / 300.0) + rng.normal(0, 0.01, n))))
        nav_done = steps >= k_nav
        v = np.where(nav_done, 0.0, rng.uniform(0.0, 0.25, n))
        w = np.where(nav_done, 0.0, rng.uniform(-0.3, 0.3, n))
        noisy = bool(sc.noisy)
        arrays = {
            "t": steps * 0.001, "base_pos": np.column_stack([xy, np.full(n, 0.12)]),
            "base_ori": np.column_stack([np.zeros((n, 2)), np.sin(yaw / 2), np.cos(yaw / 2)]),
            "base_bearing": yaw,
            "q_mes": np.column_stack([rng.normal(0, 1, (n, 4)), 0.5 + np.cumsum(rng.normal(0, 0.002, (n, 7)), axis=0),
                                      np.zeros((n, 6))]),
            "qd_mes": np.column_stack([rng.normal(0, 2, (n, 4)), rng.normal(0, 0.05, (n, 7)), np.zeros((n, 6))]),
            "wheel_cmds": np.array([wheel_speeds(v[k], w[k], params) for k in range(n)]),
            "torque_arm": rng.normal(0, 3, (n, 7)),
            "q_des_arm": np.zeros((n, 7)), "qd_des_arm": np.zeros((n, 7)),
            "phase": np.where(nav_done, 3, np.where(steps < 50, 0, 1)), "nav_done": nav_done,
            "ee_pos": np.asarray(sc.target) + rng.normal(0, 0.1, (n, 3)),
        }
        arrays["wheel_cmds_exec"], arrays["torque_arm_exec"] = arrays["wheel_cmds"], arrays["torque_arm"]
        log = {name: np.asarray(arrays[name], dtype=dt) for name, dt in LOG_FIELDS.items()}
        log["segment"] = (steps >= k_switch).astype(np.int8)
        collided = sc.episode in COLLIDED_EPS
        status = "collision" if collided else ("step_cap" if sc.episode in STEP_CAP_EPS else "reached")
        noise = {"noise_v": 0.02, "noise_w": 0.05, "noise_torque": 2.0} if noisy else \
            {"noise_v": 0.0, "noise_w": 0.0, "noise_torque": 0.0}
        timed_out = sc.episode in NAV_TIMEOUT_EPS
        meta = {"scenario": sc.to_dict(), "target": list(sc.target), "status": status, "success": status == "reached",
                "collided": collided, "collision_time": 0.5 if collided else None,
                "collision_link": "base_link" if collided else None, "n_steps": n, "log_every": 1,
                "time_step": 0.001, "nav_done_time": k_nav * 0.001, "nav_timed_out": timed_out, **noise}
        name = f"ep_{sc.episode:04d}.npz"
        np.savez_compressed(os.path.join(out_dir, name), **log, target=np.asarray(sc.target, dtype=float),
                            meta=json.dumps(meta))
        rows.append({"episode": sc.episode, "file": name, "status": status, "success": status == "reached",
                     "final_error": 0.005, "duration": (n - 1) * 0.001, "n_steps": n, "nav_done_time": k_nav * 0.001,
                     "nav_timed_out": timed_out, "target_x": sc.target[0], "target_y": sc.target[1],
                     "target_z": sc.target[2], **noise, "seed": seed, "ambiguous": sc.ambiguous, "side": sc.side,
                     "lateral_offset": sc.lateral_offset, "obstacle_r": sc.obstacles[0]["r"], "collided": collided,
                     "segment_switch_time": k_switch * 0.001})
    columns = INDEX_COLUMNS + ["ambiguous", "side", "lateral_offset", "obstacle_r", "collided", "segment_switch_time"]
    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return scs


_RECORDED = {}


def recorded(data_dir, split, episode):
    """The recorded arrays of one episode (from its npz, memoised) and its obstacles (from the split)."""
    path = os.path.join(data_dir, split["episodes"][str(episode)]["file"])
    if path not in _RECORDED:
        with np.load(path) as d:
            _RECORDED[path] = {k: d[k] for k in d.files}
    return _RECORDED[path], split["episodes"][str(episode)]["obstacles"]


def obs_of(a, k, obstacles):
    """The observation the step-9 harness gives at recorded step k: the recorded values (as float64) and
    obs['lidar'] = lidar.scan(base_pos[:2], base_bearing, obstacles), as the harness computes it."""
    base_pos = a["base_pos"][k].astype(float)
    bearing = float(a["base_bearing"][k])
    return {"t": float(a["t"][k]), "base_pos": base_pos, "base_ori": a["base_ori"][k].astype(float),
            "base_bearing": bearing, "q_mes": a["q_mes"][k].astype(float), "qd_mes": a["qd_mes"][k].astype(float),
            "target": np.asarray(a["target"], dtype=float).copy(),
            "lidar": lidar.scan(base_pos[:2], bearing, obstacles)}


def deploy_features(feature_set, obs, side=None):
    return obf.build_features(feature_set, obf.obs_signals(obs), side=side)


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="test_obstacle_learning_")
    data_dir = os.path.join(tmp, "set")
    write_synthetic_set(data_dir)
    split_path = os.path.join(tmp, "split.json")
    split = obt.write_split(split_path, data_dir=data_dir)
    cache_dir = os.path.join(tmp, "cache")
    manifest = obt.build_cache(split_path, cache_dir, workers=2, verbose=False)
    STATE.update(tmp=tmp, data_dir=data_dir, split_path=split_path, split=split, cache_dir=cache_dir,
                 manifest=manifest, models={})


def tearDownModule():
    shutil.rmtree(STATE["tmp"], ignore_errors=True)


def part(name):
    return obt.CachePart(name, STATE["cache_dir"], STATE["split_path"])


def trained(config, seed=0, epochs=2):
    """A network trained by run_train on the synthetic cache (memoised): (npz path, meta, sklearn net)."""
    key = (config, seed, epochs)
    if key not in STATE["models"]:
        out_dir = os.path.join(STATE["tmp"], f"models_s{seed}_e{epochs}")
        STATE["models"][key] = obt.run_train(config, seed, out_dir, STATE["cache_dir"], STATE["split_path"],
                                             max_epochs=epochs, log=lambda s: None)
    return STATE["models"][key]


# ==============================================================================
# Split and cache
# ==============================================================================
class TestSplit(unittest.TestCase):
    def test_deterministic(self):
        a, b = obt.make_split(STATE["data_dir"]), obt.make_split(STATE["data_dir"])
        self.assertEqual(a, b)
        core = {k: v for k, v in STATE["split"].items() if k != "created"}
        self.assertEqual(json.loads(json.dumps(a)), core)

    def test_filter(self):
        s = STATE["split"]
        failed = set(COLLIDED_EPS) | set(STEP_CAP_EPS)
        self.assertEqual(s["kept_episodes"], [e for e in range(N_EPISODES) if e not in failed])
        self.assertEqual(s["excluded"], {"collided": list(COLLIDED_EPS), "status step_cap": list(STEP_CAP_EPS)})
        tr, va = set(s["train_episodes"]), set(s["val_episodes"])
        self.assertFalse(tr & va)
        self.assertEqual(tr | va, set(s["kept_episodes"]))
        self.assertEqual(s["train_episodes"], sorted(tr))
        self.assertEqual(s["val_episodes"], sorted(va))
        # the plan's filter keeps a reached episode whose navigation timed out (counted); the option drops it
        self.assertEqual(s["n_kept_nav_timed_out"], len(NAV_TIMEOUT_EPS))
        strict = obt.make_split(STATE["data_dir"], exclude_nav_timeout=True)
        self.assertEqual(strict["excluded"]["nav_timed_out"], list(NAV_TIMEOUT_EPS))
        self.assertEqual(strict["kept_episodes"], [e for e in s["kept_episodes"] if e not in NAV_TIMEOUT_EPS])
        self.assertTrue(strict["filter"].endswith("nav_timed_out False"))

    def test_stratified(self):
        s = STATE["split"]
        from obstacle_task import scenario
        scs = {sc.episode: sc for sc in scenario.load_scenarios(os.path.join(STATE["data_dir"], "scenarios.json"))[0]}
        val = set(s["val_episodes"])
        sizes = []
        for noisy in (False, True):
            for amb in (False, True):
                eps = [e for e in s["kept_episodes"] if scs[e].noisy == noisy and scs[e].ambiguous == amb]
                self.assertEqual(sum(e in val for e in eps), round(0.15 * len(eps)), (noisy, amb))
                self.assertEqual(s["strata"][obt.stratum_key(noisy, amb)]["n"], len(eps))
                sizes.append(len(eps))
        self.assertTrue(all(n >= 4 for n in sizes), sizes)   # every stratum populated, so the draw is exercised
        for e in s["kept_episodes"]:
            info = s["episodes"][str(e)]
            self.assertEqual((info["noisy"], info["ambiguous"], info["side"]),
                             (scs[e].noisy, scs[e].ambiguous, scs[e].side))
            self.assertEqual(info["obstacles"], scs[e].obstacles)

    def test_draw_and_offsets(self):
        """The validation draw of train_policy.make_split (one rng over the strata in fixed order), offsets from
        default_rng(0)."""
        s = STATE["split"]
        rng = np.random.default_rng(0)
        val = []
        for key in ("clean/unambiguous", "clean/ambiguous", "noisy/unambiguous", "noisy/ambiguous"):
            noisy, amb = key.startswith("noisy"), key.endswith("/ambiguous")
            eps = sorted(e for e in s["kept_episodes"]
                         if s["episodes"][str(e)]["noisy"] == noisy and s["episodes"][str(e)]["ambiguous"] == amb)
            n_val = round(0.15 * len(eps))
            val += sorted(int(e) for e in rng.choice(np.array(eps), size=n_val, replace=False)) if n_val else []
        self.assertEqual(sorted(val), s["val_episodes"])
        off = np.array(s["train_offsets"])
        np.testing.assert_array_equal(off, np.random.default_rng(0).integers(0, 9, size=len(s["train_episodes"])))
        self.assertEqual((s["stride"], s["split_seed"], s["data_seed"]), (9, 0, 0))

    def test_seed_is_used_and_refuses_overwrite(self):
        self.assertNotEqual(obt.make_split(STATE["data_dir"], split_seed=1)["val_episodes"],
                            STATE["split"]["val_episodes"])
        with self.assertRaises(SystemExit):
            obt.write_split(STATE["split_path"], data_dir=STATE["data_dir"])

    def test_index_must_agree_with_scenarios(self):
        bad = os.path.join(STATE["tmp"], "bad_set")
        shutil.copytree(STATE["data_dir"], bad)
        path = os.path.join(bad, "index.csv")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["ambiguous"] = str(not obt.parse_bool(rows[0]["ambiguous"]))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            obt.make_split(bad)


class TestCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.va, cls.trig = part("train"), part("val"), part("trigger")

    def test_rows_are_offset_stride_and_all_steps(self):
        s = STATE["split"]
        offsets = dict(zip(s["train_episodes"], s["train_offsets"]))
        for i, e in enumerate(self.tr.episodes):
            sl = self.tr.episode_slice(i)
            np.testing.assert_array_equal(self.tr.cols["step"][sl], np.arange(offsets[e], s["episodes"][str(e)]["n_steps"], 9))
            self.assertTrue((self.tr.cols["episode"][sl] == e).all())
        for i, e in enumerate(self.va.episodes):
            np.testing.assert_array_equal(self.va.cols["step"][self.va.episode_slice(i)],
                                          np.arange(s["episodes"][str(e)]["n_steps"]))
        self.assertEqual(self.tr.episodes, s["train_episodes"])
        self.assertEqual(self.va.episodes, s["val_episodes"])
        self.assertEqual(self.trig.episodes, s["train_episodes"])

    def test_values_equal_recording(self):
        rng = np.random.default_rng(1)
        for cache in (self.tr, self.va):
            for i, e in enumerate(cache.episodes):
                a, _ = recorded(STATE["data_dir"], STATE["split"], e)
                sl = cache.episode_slice(i)
                rows = np.sort(rng.choice(np.arange(sl.start, sl.stop), size=10, replace=False))
                k = np.asarray(cache.cols["step"][rows])
                v, w = bcf.wheels_to_vw(a["wheel_cmds"][k])
                ref = {"t": a["t"][k], "base_xy": a["base_pos"][k, :2], "base_bearing": a["base_bearing"][k],
                       "q_arm": a["q_mes"][k, 4:11], "qd_arm": a["qd_mes"][k, 4:11], "label_v": v, "label_w": w,
                       "label_tau": a["torque_arm"][k], "nav_done": a["nav_done"][k],
                       "target": np.broadcast_to(a["target"], (len(k), 3)),
                       "side": np.full(len(k), STATE["split"]["episodes"][str(e)]["side"])}
                for name, value in ref.items():
                    got = np.asarray(cache.cols[name][rows])
                    self.assertEqual(got.dtype, np.dtype(obt.CACHE_COLUMNS[name][0]), name)
                    np.testing.assert_array_equal(got, np.asarray(value, dtype=got.dtype), err_msg=name)
                    if got.dtype == np.float32:   # recorded float32 values, unchanged
                        self.assertEqual(np.asarray(value).dtype, np.float32, name)

    def test_trigger_rows_are_the_first_nav_done_step(self):
        for i, e in enumerate(self.trig.episodes):
            a, _ = recorded(STATE["data_dir"], STATE["split"], e)
            k0 = int(np.argmax(a["nav_done"]))
            self.assertTrue(a["nav_done"][k0] and not a["nav_done"][k0 - 1])
            self.assertEqual(int(self.trig.cols["step"][i]), k0)
            self.assertEqual(STATE["manifest"]["trigger"]["k0_steps"][i], k0)
            self.assertEqual(STATE["manifest"]["trigger"]["k0_segment"][i], int(a["segment"][k0]))
            np.testing.assert_array_equal(self.trig.cols["qd_arm"][i], a["qd_mes"][k0, 4:11])
            np.testing.assert_array_equal(self.trig.cols["label_tau"][i], a["torque_arm"][k0])
            self.assertEqual((float(self.trig.cols["label_v"][i]), float(self.trig.cols["label_w"][i])), (0.0, 0.0))

    def test_refusals(self):
        other = os.path.join(STATE["tmp"], "other_split.json")
        obt.write_split(other, force=True, data_dir=STATE["data_dir"], split_seed=1)
        with self.assertRaises(RuntimeError):     # built from another split file
            obt.CachePart("train", STATE["cache_dir"], other)
        sub_dir = os.path.join(STATE["tmp"], "cache_subset")
        obt.build_cache(STATE["split_path"], sub_dir, workers=1, train_episodes=STATE["split"]["train_episodes"][:2],
                        val_episodes=STATE["split"]["val_episodes"][:1], verbose=False)
        with self.assertRaises(RuntimeError):     # a subset cache never feeds an official model
            obt.CachePart("train", sub_dir, STATE["split_path"])
        self.assertEqual(obt.CachePart("trigger", sub_dir, STATE["split_path"], allow_subset=True).n_rows, 2)
        with self.assertRaises(SystemExit):       # smoke settings never take an official trained/step9 name
            obt.run_train("L1arm", 0, cache_dir=sub_dir, split_path=STATE["split_path"], max_epochs=1,
                          log=lambda s: None)


# ==============================================================================
# B1: training-time features == deployment features
# ==============================================================================
class TestFeatureParity(unittest.TestCase):
    """Gate B1 on the synthetic recordings: the features train.py fits on (cache -> raw_rows / the standardised
    chunks / the trigger copies) equal the features the policies compute from the harness's observation of the same
    recorded step (features.obs_signals with obs['lidar'] = lidar.scan of the recorded pose), <= 1e-9."""

    @classmethod
    def setUpClass(cls):
        cls.tr, cls.va = part("train"), part("val")
        rng = np.random.default_rng(2)
        cls.train_rows = rng.permutation(cls.tr.n_rows)[:300]     # shuffled, as in a training chunk
        cls.val_rows = np.arange(cls.va.episode_slice(0).start, cls.va.episode_slice(1).stop)
        cls.max_err = {}

    def deployed(self, cache, rows, feature_set):
        out = []
        for r in rows:
            e, k = int(cache.cols["episode"][r]), int(cache.cols["step"][r])
            a, obstacles = recorded(STATE["data_dir"], STATE["split"], e)
            side = STATE["split"]["episodes"][str(e)]["side"] if feature_set in obf.SIDE_SETS else None
            out.append(deploy_features(feature_set, obs_of(a, k, obstacles), side=side)[0])
        return np.array(out)

    def test_every_feature_set(self):
        for fs, outputs in FEATURE_OUTPUTS.items():
            for cache, rows in ((self.tr, self.train_rows), (self.va, self.val_rows[::7])):
                X, _ = obt.raw_rows(cache, rows, fs, outputs)
                ref = self.deployed(cache, rows, fs)
                self.assertEqual(X.shape, (len(rows), len(obf.FEATURE_SETS[fs])))
                err = float(np.max(np.abs(X - ref)))
                self.max_err[(fs, cache.part)] = err
                self.assertLessEqual(err, TOL, (fs, cache.part))

    def test_columns(self):
        """arm20 = ego_novw; base70 = ego_novw[:6] + lidar; single84 = arm20 + lidar; base70_side = base70 + side."""
        rows = self.train_rows[:50]
        X = {fs: obt.raw_rows(self.tr, rows, fs, out)[0] for fs, out in FEATURE_OUTPUTS.items()}
        sig = obt.cache_signals(self.tr, np.sort(rows), with_lidar=False)
        order = np.argsort(np.argsort(rows))
        np.testing.assert_array_equal(X["arm20"], bcf.build_features("ego_novw", sig)[order])
        np.testing.assert_array_equal(X["single84"][:, :20], X["arm20"])
        np.testing.assert_array_equal(X["base70"][:, :6], X["arm20"][:, :6])
        np.testing.assert_array_equal(X["base70"][:, 6:], X["single84"][:, 20:])
        np.testing.assert_array_equal(X["base70_side"][:, :70], X["base70"])
        sides = [STATE["split"]["episodes"][str(int(self.tr.cols["episode"][r]))]["side"] for r in rows]
        np.testing.assert_array_equal(X["base70_side"][:, 70], sides)
        self.assertEqual(obf.FEATURE_SETS["arm20"], bcf.FEATURE_SETS["ego_novw"])
        self.assertEqual([len(obf.FEATURE_SETS[k]) for k in ("arm20", "base70", "single84", "base70_side")],
                         [20, 70, 84, 71])
        self.assertTrue(np.all(X["base70"][:, 6:] <= lidar.MAX_RANGE) and np.any(X["base70"][:, 6:] < lidar.MAX_RANGE))

    def test_lidar_is_scan_batch_of_the_logged_pose(self):
        e = self.va.episodes[0]
        a, obstacles = recorded(STATE["data_dir"], STATE["split"], e)
        X, _ = obt.raw_rows(self.va, np.arange(self.va.episode_slice(0).stop), "base70", "vw")
        np.testing.assert_array_equal(X[:, 6:], lidar.scan_batch(a["base_pos"][:, :2], a["base_bearing"], obstacles))
        # grouped by episode (any order) = one scan_batch per episode
        rows = self.train_rows
        X, _ = obt.raw_rows(self.tr, rows, "base70", "vw")
        for r, x in zip(rows[:40], X[:40]):
            e = int(self.tr.cols["episode"][r])
            ref = lidar.scan_batch(np.asarray(self.tr.cols["base_xy"][r:r + 1], float),
                                   np.asarray(self.tr.cols["base_bearing"][r:r + 1], float),
                                   STATE["split"]["episodes"][str(e)]["obstacles"])
            np.testing.assert_allclose(x[6:], ref[0], rtol=0, atol=1e-12)

    def test_training_chunks_and_copies(self):
        """The standardised chunks fed to partial_fit are the standardised B1 features; the copies' unjittered part
        equals the deployment features of the recorded trigger step."""
        for config in ("L0", "L1base", "L1arm", "L2base"):
            d = obt.NetData(config, STATE["cache_dir"], STATE["split_path"], allow_subset=True)
            idx = np.random.default_rng(3).permutation(d.n_train)[:400]
            X, Y = d.train_chunk(idx)
            base = idx < d.n_base
            Xr, Yr = obt.raw_rows(d.tr, d.train_rows[idx[base]], d.feature_set, d.outputs)
            np.testing.assert_array_equal(X[base], (Xr - d.xm) / d.xs)
            np.testing.assert_array_equal(Y[base], (Yr - d.ym) / d.ys)
            np.testing.assert_array_equal(X[~base], d.Xc[idx[~base] - d.n_base])
            trig = part("trigger")
            X0, _, _ = obt.trigger_copies(trig, d.feature_set, d.outputs, 2, np.zeros(7))
            for i, e in enumerate(trig.episodes[:5]):
                a, obstacles = recorded(STATE["data_dir"], STATE["split"], e)
                k0 = int(np.argmax(a["nav_done"]))
                side = STATE["split"]["episodes"][str(e)]["side"] if d.feature_set in obf.SIDE_SETS else None
                ref = deploy_features(d.feature_set, obs_of(a, k0, obstacles), side=side)[0]
                self.assertLessEqual(float(np.max(np.abs(X0[2 * i] - ref))), TOL, config)

    @classmethod
    def tearDownClass(cls):
        if cls.max_err:
            print("\nB1 max |train - deploy| per (feature set, part): "
                  + ", ".join(f"{k[0]}/{k[1]} {v:.1e}" for k, v in sorted(cls.max_err.items())))


def complete_set(path):
    """True when a step-9 set has its index.csv with one row per episode of its config.json (collection done)."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            n = int(json.load(f)["n_episodes"])
        with open(os.path.join(path, "index.csv"), newline="") as f:
            return len(list(csv.DictReader(f))) == n and os.path.exists(os.path.join(path, "scenarios.json"))
    except (OSError, ValueError, KeyError):
        return False


@unittest.skipUnless(complete_set(PILOT2_DIR), "Track A's pilot2 recordings missing or still being collected")
class TestPilot2Parity(unittest.TestCase):
    """Gate B1 on Track A's recordings (results/step9/smoke/pilot2): split and a small subset cache in a temporary
    directory; training features of sampled rows == deployment features of the same recorded steps."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_obstacle_pilot2_")
        cls.split_path = os.path.join(cls.tmp, "split.json")
        cls.split = obt.write_split(cls.split_path, data_dir=PILOT2_DIR)
        cls.cache_dir = os.path.join(cls.tmp, "cache")
        obt.build_cache(cls.split_path, cls.cache_dir, workers=3, train_episodes=cls.split["train_episodes"][:3],
                        val_episodes=cls.split["val_episodes"][:2], verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_parity(self):
        rng = np.random.default_rng(4)
        worst = {}
        for name in ("train", "val", "trigger"):
            cache = obt.CachePart(name, self.cache_dir, self.split_path, allow_subset=True)
            rows = np.sort(rng.choice(cache.n_rows, size=min(cache.n_rows, 150), replace=False))
            for fs, outputs in FEATURE_OUTPUTS.items():
                X, _ = obt.raw_rows(cache, rows, fs, outputs)
                for r, x in zip(rows, X):
                    e, k = int(cache.cols["episode"][r]), int(cache.cols["step"][r])
                    a, obstacles = recorded(PILOT2_DIR, self.split, e)
                    side = self.split["episodes"][str(e)]["side"] if fs in obf.SIDE_SETS else None
                    err = float(np.max(np.abs(x - deploy_features(fs, obs_of(a, k, obstacles), side=side)[0])))
                    worst[(fs, name)] = max(worst.get((fs, name), 0.0), err)
        print("\nB1 pilot2 max |train - deploy|: " + ", ".join(f"{k[0]}/{k[1]} {v:.1e}" for k, v in sorted(worst.items())))
        self.assertTrue(all(v <= TOL for v in worst.values()), worst)


# ==============================================================================
# B3: trigger copies
# ==============================================================================
class TestTriggerCopies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.trig = part("train"), part("trigger")
        cls.sigma = tp.hold_qd_sigma(cls.tr.cols)
        cls.n = cls.trig.n_rows

    def test_count_and_place(self):
        """K = 16 copies per training episode, appended after the cached rows; none for L1armK0 / L2cls."""
        for config, spec in obt.CONFIGS.items():
            d = obt.NetData(config, STATE["cache_dir"], STATE["split_path"], allow_subset=True)
            self.assertEqual(d.n_copies, spec["copies"] * len(STATE["split"]["train_episodes"]), config)
            self.assertEqual(d.n_train, d.n_base + d.n_copies)
            if spec["copies"]:
                self.assertEqual(spec["copies"], 16)
                t = d.trigger_meta
                self.assertEqual((t["copies"], t["n_episodes"], t["n_rows_added"], t["jitter_seed"]),
                                 (16, self.n, 16 * self.n, 0))
                self.assertEqual(t["k0_steps"], STATE["manifest"]["trigger"]["k0_steps"])
                np.testing.assert_array_equal(t["qd_sigma"], self.sigma)
            else:
                self.assertIsNone(d.trigger_meta)

    def test_jitter_only_on_arm_qd(self):
        for fs, outputs in FEATURE_OUTPUTS.items():
            X, _, noise = obt.trigger_copies(self.trig, fs, outputs, 16, self.sigma)
            X0, _, noise0 = obt.trigger_copies(self.trig, fs, outputs, 16, np.zeros(7))
            self.assertFalse(noise0.any())
            names = obf.FEATURE_SETS[fs]
            qd = [names.index(f"qd{j}") for j in range(1, 8)] if "qd1" in names else []
            other = [c for c in range(len(names)) if c not in qd]
            np.testing.assert_array_equal(X[:, other], X0[:, other], err_msg=fs)
            if qd:
                np.testing.assert_array_equal(X[:, qd], X0[:, qd] + noise, err_msg=fs)
                self.assertTrue(np.all(noise != 0))
            else:   # base sets: no arm-qd column, the K copies of a row are identical
                self.assertTrue(np.array_equal(X, X0) and np.array_equal(X, np.repeat(X[::16], 16, axis=0)), fs)
            # unjittered copies = the trigger rows' features, each repeated K times (episode-major)
            ref, _ = obt.raw_rows(self.trig, np.arange(self.n), fs, outputs)
            np.testing.assert_array_equal(X0, np.repeat(ref, 16, axis=0), err_msg=fs)

    def test_noise_and_sigma_as_c3(self):
        """noise = default_rng(0).standard_normal((n K, 7)) * sigma_hold; sigma_hold = per-joint std of arm qd over
        the training rows with nav_done False, from the recordings; arm20 copies = train_policy.trigger_copies."""
        _, _, noise = obt.trigger_copies(self.trig, "arm20", "tau", 16, self.sigma)
        np.testing.assert_array_equal(noise, np.random.default_rng(0).standard_normal((16 * self.n, 7)) * self.sigma)
        offsets = dict(zip(STATE["split"]["train_episodes"], STATE["split"]["train_offsets"]))
        qd = []
        for e in self.tr.episodes:
            a, _ = recorded(STATE["data_dir"], STATE["split"], e)
            steps = np.arange(offsets[e], len(a["t"]), 9)
            qd.append(a["qd_mes"][steps[~a["nav_done"][steps]], 4:11].astype(np.float64))
        np.testing.assert_allclose(self.sigma, np.concatenate(qd).std(axis=0), rtol=1e-12)
        # bit for bit the C3 implementation, fed the same rows in train_policy's cache layout
        cols = {k: np.asarray(v) for k, v in self.trig.cols.items()}
        cols["qd_wheels"] = np.zeros((self.n, 4), dtype=np.float32)
        Xc3, Yc3, noise_c3 = tp.trigger_copies(cols, "ego_novw", 16, self.sigma)
        X, Y, _ = obt.trigger_copies(self.trig, "arm20", "all", 16, self.sigma)
        np.testing.assert_array_equal(X, Xc3)
        np.testing.assert_array_equal(Y, Yc3)
        np.testing.assert_array_equal(noise, noise_c3)

    def test_labels_equal_the_source_row(self):
        for outputs, cols in (("all", slice(0, 9)), ("vw", slice(0, 2)), ("tau", slice(2, 9))):
            _, Y, _ = obt.trigger_copies(self.trig, "single84", outputs, 16, self.sigma)
            for i, e in enumerate(self.trig.episodes):
                a, _ = recorded(STATE["data_dir"], STATE["split"], e)
                k0 = int(np.argmax(a["nav_done"]))
                v, w = bcf.wheels_to_vw(a["wheel_cmds"][k0])
                ref = np.concatenate([[v, w], a["torque_arm"][k0].astype(np.float64)])[cols]
                for c in range(16):
                    np.testing.assert_array_equal(Y[16 * i + c], ref)

    def test_same_copies_for_every_seed_and_config_and_base_standardisation(self):
        Xa, Ya, na = obt.trigger_copies(self.trig, "arm20", "tau", 16, self.sigma)
        Xb, Yb, nb = obt.trigger_copies(self.trig, "single84", "all", 16, self.sigma)
        Xv, Yv, nv = obt.trigger_copies(self.trig, "base70", "vw", 16, self.sigma)
        np.testing.assert_array_equal(Xa, Xb[:, :20])        # the same rows, each network its own columns
        np.testing.assert_array_equal(Xv, Xb[:, list(range(6)) + list(range(20, 84))])
        np.testing.assert_array_equal(Ya, Yb[:, 2:])
        np.testing.assert_array_equal(Yv, Yb[:, :2])
        np.testing.assert_array_equal(na, nb)
        a = obt.NetData("L1arm", STATE["cache_dir"], STATE["split_path"], allow_subset=True)
        np.testing.assert_array_equal(a.Xc, (Xa - a.xm) / a.xs)   # standardised with the base rows' scalers
        np.testing.assert_array_equal(a.Yc, (Ya - a.ym) / a.ys)
        k0 = obt.NetData("L1armK0", STATE["cache_dir"], STATE["split_path"], allow_subset=True)
        for key in ("xm", "xs", "ym", "ys"):   # from the original rows only: equal with and without copies
            np.testing.assert_array_equal(getattr(a, key), getattr(k0, key), err_msg=key)
        X, Y = obt.raw_rows(a.tr, a.train_rows, "arm20", "tau")
        np.testing.assert_allclose(a.xm, X.mean(axis=0), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(a.xs, X.std(axis=0), rtol=1e-10)
        np.testing.assert_allclose(a.ym, Y.mean(axis=0), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(a.ys, Y.std(axis=0), rtol=1e-10)
        m1, m2 = trained("L1arm", seed=0)[1], trained("L1arm", seed=1)[1]
        self.assertEqual(m1["trigger"], m2["trigger"])


# ==============================================================================
# Epoch loop, early stopping, determinism, resumability
# ==============================================================================
class FakeData:
    """The parts of NetData that fit_epoch / train_net use, on in-memory arrays."""

    def __init__(self, X, Y, kind="regressor", val_losses=None):
        self.X, self.Y, self.kind, self.n_train = X, Y, kind, len(X)
        self.val_losses, self.snapshots = list(val_losses or []), []

    def train_chunk(self, idx):
        return self.X[idx], self.Y[idx]

    def val_loss(self, weights, biases):
        self.snapshots.append([a.copy() for a in weights + biases])
        return self.val_losses.pop(0) if self.val_losses else 1.0


class TestEpochLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.normal(size=(1050, 6))
        cls.Y = np.column_stack([np.sin(cls.X[:, 0]), cls.X[:, 1] * cls.X[:, 2]])

    def test_chunked_partial_fit_equals_one_call(self):
        """partial_fit over chunks (multiples of the batch size, sklearn shuffle off) of a permutation gives exactly
        the weights and the epoch loss of one partial_fit call over the permuted rows; also for the classifier."""
        for kind, Y in (("regressor", self.Y), ("classifier", np.where(self.X[:, 0] > 0, 1, -1))):
            a, b = obt.make_net(kind, 3), obt.make_net(kind, 3)
            data = FakeData(self.X, Y, kind)
            rng = np.random.default_rng(5)
            for _ in range(2):
                perm = rng.permutation(len(self.X))
                if kind == "classifier":
                    a.partial_fit(self.X[perm], Y[perm], classes=[-1, 1])
                else:
                    a.partial_fit(self.X[perm], Y[perm])
                loss = obt.fit_epoch(b, data, perm, chunk=400)
                self.assertAlmostEqual(loss, a.loss_, places=12)
            for wa, wb in zip(a.coefs_ + a.intercepts_, b.coefs_ + b.intercepts_):
                np.testing.assert_array_equal(wa, wb)
            self.assertEqual((a.t_, a._optimizer.t), (b.t_, b._optimizer.t))
            self.assertEqual(b._optimizer.t, 2 * 6)   # ceil(1050 / 200) batches per epoch
            self.assertFalse(b.shuffle)

    def test_shuffle_differs_between_epochs_and_seeds(self):
        perms = []
        real = obt.fit_epoch

        def spy(net, data, perm, chunk=obt.TRAIN_CHUNK):
            perms.append(np.array(perm))
            return real(net, data, perm, chunk)

        with mock.patch.object(obt, "fit_epoch", spy):
            for seed in (0, 0, 1):
                obt.train_net(FakeData(self.X, self.Y, val_losses=[1.0, 0.9, 0.8]), seed, max_epochs=3)
        self.assertFalse(np.array_equal(perms[0], perms[1]))          # a new permutation every epoch
        np.testing.assert_array_equal(perms[0], perms[3])              # deterministic for a seed
        self.assertFalse(np.array_equal(perms[0], perms[6]))           # another seed, another order
        np.testing.assert_array_equal(perms[0], np.random.default_rng([0, obt.SHUFFLE_STREAM]).permutation(1050))

    def test_early_stopping_patience_8_and_best_weights(self):
        losses = [1.0, 0.5, 0.6, 0.55, 0.5001, 0.52, 0.51, 0.53, 0.54, 0.56, 0.3]   # epochs 3-10 do not improve
        data = FakeData(self.X, self.Y, val_losses=losses)
        w, b, hist, net = obt.train_net(data, 0, max_epochs=150)
        self.assertEqual((hist["epochs_run"], hist["best_epoch"]), (10, 2))
        self.assertEqual(hist["val_loss"], losses[:10])
        for x, s in zip(w + b, data.snapshots[1]):     # exactly epoch 2's weights ...
            np.testing.assert_array_equal(x, s)
        self.assertFalse(all(np.array_equal(x, c) for x, c in zip(w + b, net.coefs_ + net.intercepts_)))  # ... not the last
        self.assertIn("8 consecutive epochs", hist["stop_reason"])
        # slow steady decrease keeps going until max_epochs
        steady = [1.0 - 0.002 * k for k in range(20)]
        hist = obt.train_net(FakeData(self.X, self.Y, val_losses=steady), 0, max_epochs=20)[2]
        self.assertEqual((hist["epochs_run"], hist["best_epoch"], hist["stop_reason"]), (20, 20, "max_epochs"))
        self.assertEqual((obt.PATIENCE, obt.MAX_EPOCHS, obt.REL_TOL), (8, 150, 1e-3))

    def test_single_thread(self):
        from threadpoolctl import threadpool_info
        seen = []
        real = obt.fit_epoch

        def spy(*args, **kwargs):
            seen.extend(p["num_threads"] for p in threadpool_info() if p.get("user_api") == "blas")
            return real(*args, **kwargs)

        with mock.patch.object(obt, "fit_epoch", spy):
            obt.train_net(FakeData(self.X, self.Y), 0, max_epochs=1)
        self.assertTrue(seen and all(n == 1 for n in seen), seen)

    def test_hyperparameters(self):
        r, c = obt.make_net("regressor", 0), obt.make_net("classifier", 0)
        self.assertEqual((r.hidden_layer_sizes, c.hidden_layer_sizes), ((128, 128), (64, 64)))
        for net in (r, c):
            self.assertEqual((net.activation, net.solver, net.learning_rate_init, net.alpha, net.batch_size,
                              net.shuffle), ("relu", "adam", 1e-3, 1e-4, 200, False))
            self.assertIsInstance(net.random_state, np.random.RandomState)


class TestRunTrain(unittest.TestCase):
    def test_outputs_and_meta(self):
        for config, spec in obt.CONFIGS.items():
            npz, meta, net = trained(config)
            js = npz[:-4] + ".json"
            self.assertTrue(os.path.exists(npz) and os.path.exists(js))
            with open(js) as f:
                side = json.load(f)
            self.assertTrue(side["complete"])
            self.assertEqual(side["model_sha256"], tp.sha256_file(npz))
            m = obf.load_model(npz)
            self.assertEqual((m["kind"], m["feature_set"], m["output_names"]),
                             (spec["kind"], spec["features"], obf.OUTPUTS[spec["outputs"]]))
            self.assertEqual(m["meta"]["epochs_run"], 2)
            self.assertEqual(len(m["meta"]["history"]["val_loss"]), 2)
            self.assertEqual(m["meta"]["val_loss"], min(m["meta"]["history"]["val_loss"]))
            self.assertEqual(m["meta"]["split_sha256"], tp.sha256_file(STATE["split_path"]))
            self.assertEqual(set(m["meta"]["code_sha256"]), set(obt.CODE_FILES))
            self.assertEqual(m["meta"]["code_sha256"]["obstacle_task/train.py"],
                             tp.sha256_file(os.path.join(TASK_DIR, "obstacle_task", "train.py")))
            self.assertEqual(m["meta"]["n_train_rows"], m["meta"]["n_train_rows_base"]
                             + spec["copies"] * len(STATE["split"]["train_episodes"]))
            if spec["rows"] == "t<1":
                va = part("val")
                self.assertEqual(m["meta"]["n_val_rows"], int(np.sum(np.asarray(va.cols["t"]) < 1.0)))
                self.assertLess(m["meta"]["n_val_rows"], va.n_rows)
            else:
                self.assertEqual(m["meta"]["n_val_rows"], part("val").n_rows)
            self.assertEqual(m["meta"], {k: v for k, v in side.items() if k not in ("model_file", "model_sha256",
                                                                                    "complete")})

    def test_val_loss_definitions(self):
        """The recorded validation loss equals the definition recomputed directly on all validation rows."""
        from scipy.special import expit
        for config in ("L1base", "L2cls"):
            npz, meta, net = trained(config)
            d = obt.NetData(config, STATE["cache_dir"], STATE["split_path"], allow_subset=True)
            m = obf.load_model(npz)
            X, Y = obt.raw_rows(d.va, d.val_rows, d.feature_set, d.outputs)
            if d.kind == "regressor":
                ref = float(np.mean(((obf.forward(m, X) - Y) / m["y_scale"]) ** 2))
            else:
                p = np.clip(expit(bcf.forward(m, X)[:, 0]), np.finfo(float).eps, 1 - np.finfo(float).eps)
                ref = float(np.mean(-np.where(Y == 1, np.log(p), np.log(1 - p))))
            self.assertAlmostEqual(meta["val_loss"], ref, places=10)

    def test_deterministic(self):
        a = obf.load_model(trained("L2base", seed=0)[0])
        out_dir = os.path.join(STATE["tmp"], "again")
        b = obf.load_model(obt.run_train("L2base", 0, out_dir, STATE["cache_dir"], STATE["split_path"], max_epochs=2,
                                         log=lambda s: None)[0])
        for x, y in zip(a["weights"] + a["biases"], b["weights"] + b["biases"]):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(a["meta"]["history"]["val_loss"], b["meta"]["history"]["val_loss"])
        c = obf.load_model(trained("L2base", seed=1)[0])
        self.assertFalse(np.array_equal(a["weights"][0], c["weights"][0]))

    def test_resumable(self):
        out_dir = os.path.join(STATE["tmp"], "resume")
        args = ("L2cls", 4, out_dir, STATE["cache_dir"], STATE["split_path"])
        npz, _, net = obt.run_train(*args, max_epochs=1, log=lambda s: None)
        self.assertIsNotNone(net)
        stamp = os.path.getmtime(npz)
        logs = []
        npz2, meta, net2 = obt.run_train(*args, max_epochs=1, log=logs.append)
        self.assertIsNone(net2)
        self.assertEqual((npz2, os.path.getmtime(npz)), (npz, stamp))
        self.assertTrue(logs and logs[0].startswith("skip"))
        self.assertTrue(meta["complete"])
        os.remove(npz[:-4] + ".json")     # an unfinished model (no sidecar) is trained again
        self.assertFalse(obt.is_finished("L2cls", 4, out_dir))
        self.assertIsNotNone(obt.run_train(*args, max_epochs=1, log=lambda s: None)[2])
        self.assertIsNotNone(obt.run_train(*args, max_epochs=1, force=True, log=lambda s: None)[2])

    def test_cli(self):
        out_dir = os.path.join(STATE["tmp"], "cli")
        with mock.patch("builtins.print"):
            obt.main(["train", "--config", "L1armK0,L2cls", "--seed", "5-6", "--split", STATE["split_path"],
                      "--cache", STATE["cache_dir"], "--out-dir", out_dir, "--max-epochs", "1"])
        self.assertEqual(sorted(os.listdir(out_dir)), sorted(f"{c}_s{s}.{x}" for c in ("L1armK0", "L2cls")
                                                             for s in (5, 6) for x in ("npz", "json")))
        with self.assertRaises(SystemExit):
            obt.main(["train", "--config", "L9", "--seed", "0"])


# ==============================================================================
# B2: numpy forward == sklearn
# ==============================================================================
class TestForwardEqualsSklearn(unittest.TestCase):
    def test_regressors(self):
        for config in ("L0", "L1base", "L1arm", "L1armK0", "L2base"):
            npz, _, net = trained(config)
            m = obf.load_model(npz)
            d = obt.NetData(config, STATE["cache_dir"], STATE["split_path"], allow_subset=True)
            X, _ = obt.raw_rows(d.va, d.val_rows, d.feature_set, d.outputs)
            ref = net.predict((X - m["x_mean"]) / m["x_scale"]) * m["y_scale"] + m["y_mean"]
            err = float(np.max(np.abs(obf.forward(m, X) - ref)))
            self.assertLessEqual(err, TOL, config)

    def test_classifier(self):
        npz, _, net = trained("L2cls")
        m = obf.load_model(npz)
        self.assertEqual(list(net.classes_), [-1, 1])
        self.assertEqual(net.out_activation_, "logistic")
        d = obt.NetData("L2cls", STATE["cache_dir"], STATE["split_path"], allow_subset=True)
        X, _ = obt.raw_rows(d.va, d.val_rows, "base70", "side")
        Xs = (X - m["x_mean"]) / m["x_scale"]
        p = obf.forward(m, X)
        self.assertLessEqual(float(np.max(np.abs(p[:, 0] - net.predict_proba(Xs)[:, 1]))), TOL)
        np.testing.assert_array_equal(np.where(p[:, 0] >= 0.5, 1, -1)[p[:, 0] != 0.5], net.predict(Xs)[p[:, 0] != 0.5])

    def test_model_file_checks(self):
        npz, _, _ = trained("L1arm")
        with self.assertRaises(ValueError):    # an arm network is not a base network
            obp.load_net(npz, "regressor", "base70", "vw")
        bad = os.path.join(STATE["tmp"], "bad_model.npz")
        with np.load(npz) as f:
            arrays = {k: f[k] for k in f.files}
        arrays["feature_names"] = np.array(obf.FEATURE_SETS["base70"][:20])
        np.savez(bad, **arrays)
        with self.assertRaises(ValueError):
            obf.load_model(bad)


# ==============================================================================
# Policies
# ==============================================================================
def craft_classifier(path, weight, bias):
    """A side classifier with one hidden unit: logit = relu(weight * x_std[ego_dx]) + bias (standardisation
    identity), so p_left can be steered with the target's position in the base frame."""
    n = len(obf.FEATURE_SETS["base70"])
    w0 = np.zeros((n, 1))
    w0[0, 0] = weight
    obf.save_model(path, "classifier", "base70", ["p_left"], np.zeros(n), np.ones(n), np.zeros(1), np.ones(1),
                   [w0, np.ones((1, 1))], [np.zeros(1), np.array([bias])], {"crafted": True})
    return path


class TestPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {config: trained(config)[0] for config in obt.CONFIGS}
        cls.va = part("val")
        e = cls.va.episodes[0]
        cls.a, cls.obstacles = recorded(STATE["data_dir"], STATE["split"], e)
        cls.side = STATE["split"]["episodes"][str(e)]["side"]
        cls.obs = [obs_of(cls.a, k, cls.obstacles) for k in (0, 5, 400, len(cls.a["t"]) - 1)]

    def policies(self):
        p = self.paths
        return {"L0": obp.make_single(f"model={p['L0']}"),
                "L1": obp.make_two_net(f"base={p['L1base']},arm={p['L1arm']}"),
                "L1-K0": obp.make_two_net(f"base={p['L1base']},arm={p['L1armK0']}"),
                "L2": obp.make_side_latched(f"cls={p['L2cls']},base={p['L2base']},arm={p['L1arm']}")}

    def test_act_shapes_and_conversion(self):
        from collect_data import wheel_speeds
        from robotnik_arm_control import BaseControllerParams
        params = BaseControllerParams(nav_timeout=25.0)
        for name, pol in self.policies().items():
            pol.reset(self.obs[0]["target"], {"time_step": 0.001})
            for obs in self.obs:
                wheels, tau = pol.act(obs)
                self.assertEqual((np.shape(wheels), np.shape(tau)), ((4,), (7,)), name)
                self.assertTrue(np.all(np.isfinite(wheels)) and np.all(np.isfinite(tau)), name)
                y = pol.predict(obs)
                np.testing.assert_array_equal(wheels, wheel_speeds(y[0], y[1], params))
                np.testing.assert_array_equal(tau, y[2:9])
            self.assertEqual(pol.base_params.nav_timeout, 25.0)

    def test_policy_outputs_equal_training_features_through_the_nets(self):
        """End to end: act() on the harness observation of a recorded step == the networks run on the training-time
        features of the same cached row (L2 with its latched side)."""
        rows = [0, 5, 400, self.va.n_rows_episode[0] - 1]
        for name, pol in self.policies().items():
            pol.reset(None, None)
            for r, obs in zip(rows, self.obs):
                y = pol.predict(obs)
                if name == "L0":
                    ref = obf.forward(obf.load_model(self.paths["L0"]), obt.raw_rows(self.va, [r], "single84", "all")[0])[0]
                else:
                    arm = self.paths["L1armK0" if name == "L1-K0" else "L1arm"]
                    ref_tau = obf.forward(obf.load_model(arm), obt.raw_rows(self.va, [r], "arm20", "tau")[0])[0]
                    if name == "L2":
                        X = obt.raw_rows(self.va, [r], "base70_side", "vw")[0]
                        X[0, -1] = pol.latched_side      # the cache holds the scenario side, the policy its latch
                        ref_vw = obf.forward(obf.load_model(self.paths["L2base"]), X)[0]
                    else:
                        ref_vw = obf.forward(obf.load_model(self.paths["L1base"]),
                                             obt.raw_rows(self.va, [r], "base70", "vw")[0])[0]
                    ref = np.concatenate([ref_vw, ref_tau])
                self.assertLessEqual(float(np.max(np.abs(y - ref))), TOL, name)

    def test_side_latching(self):
        p = self.paths
        far = dict(self.obs[0], target=self.obs[0]["base_pos"] + np.array([3.0, 0.0, 0.5]))
        obs_ahead = dict(far, base_bearing=0.0)            # target 3 m ahead: ego_dx = 3 -> p_left > 0.5
        obs_behind = dict(far, base_bearing=np.pi)         # target 3 m behind: ego_dx = -3 -> relu 0 -> p < 0.5
        cls = craft_classifier(os.path.join(STATE["tmp"], "cls_dx.npz"), 1.0, -0.5)
        pol = obp.SideLatchedPolicy(cls, p["L2base"], p["L1arm"])
        pol.reset(None, None)
        self.assertIsNone(pol.latched_side)
        pol.act(obs_ahead)
        self.assertEqual(pol.latched_side, 1)
        self.assertGreater(pol.p_left, 0.5)
        for obs in (obs_behind, obs_ahead, obs_behind):      # never re-evaluated within the episode
            pol.act(obs)
            self.assertEqual(pol.latched_side, 1)
            self.assertEqual(pol.features(obs)["base"][0, -1], 1.0)   # the base network gets the latched side
        self.assertEqual(pol.episode_log(), {"latched_side": 1, "p_left": pol.p_left})
        pol.reset(None, None)                                 # a new episode latches again
        self.assertIsNone(pol.latched_side)
        pol.act(obs_behind)
        self.assertEqual(pol.latched_side, -1)
        self.assertEqual(pol.features(obs_ahead)["base"][0, -1], -1.0)
        with mock.patch.object(obf, "forward", wraps=obf.forward) as fwd:   # the classifier runs once per episode
            pol.reset(None, None)
            for obs in self.obs:
                pol.act(obs)
            self.assertEqual(sum(1 for c in fwd.call_args_list if c.args[0] is pol.cls), 1)
        # threshold: p_left = 0.5 exactly -> +1, just below -> -1
        for bias, side in ((0.0, 1), (-1e-9, -1)):
            pol = obp.SideLatchedPolicy(craft_classifier(os.path.join(STATE["tmp"], "cls_c.npz"), 0.0, bias),
                                        p["L2base"], p["L1arm"])
            pol.reset(None, None)
            pol.act(obs_ahead)
            self.assertEqual(pol.latched_side, side, bias)

    def test_factories(self):
        p = self.paths
        self.assertIsInstance(obp.make_single(p["L0"]), obp.SinglePolicy)
        model_dir = os.path.dirname(p["L0"])
        self.assertEqual(obp.config_paths("L2", 0, model_dir),
                         {"cls": p["L2cls"], "base": p["L2base"], "arm": p["L1arm"]})
        for name, cls in (("L0", obp.SinglePolicy), ("L1", obp.TwoNetPolicy), ("L1-K0", obp.TwoNetPolicy),
                          ("L2", obp.SideLatchedPolicy)):
            pol = obp.make_config(f"{name},seed=0,dir={model_dir}")
            self.assertIsInstance(pol, cls)
        self.assertEqual(obp.make_config(f"L1-K0,seed=0,dir={model_dir}").arm_path, p["L1armK0"])
        for bad in (None, "", f"base={p['L1base']}", f"base={p['L1base']},arm={p['L1arm']},x=1", "base"):
            with self.assertRaises(ValueError):
                obp.make_two_net(bad)
        with self.assertRaises(ValueError):    # roles swapped: wrong network types
            obp.make_two_net(f"base={p['L1arm']},arm={p['L1base']}")
        with self.assertRaises(ValueError):
            obp.make_config("L3,seed=0")

    def test_loaded_like_the_harness(self):
        """evaluate_policy.load_policy imports the file as module 'policies'; the factory works from there."""
        import importlib.util
        path = os.path.join(TASK_DIR, "obstacle_task", "policies.py")
        spec = importlib.util.spec_from_file_location("policies_harness_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pol = module.make_two_net(f"base={self.paths['L1base']},arm={self.paths['L1arm']}")
        wheels, tau = pol.act(self.obs[1])
        self.assertEqual((wheels.shape, tau.shape), ((4,), (7,)))

    def test_lidar_required(self):
        pol = self.policies()["L1"]
        obs = dict(self.obs[0])
        del obs["lidar"]
        with self.assertRaises(KeyError):
            pol.act(obs)
        with self.assertRaises(ValueError):
            pol.act(dict(self.obs[0], lidar=np.zeros(32)))


if __name__ == "__main__":
    unittest.main()
