"""G4: clean chunk labels, C3-compatible matrices and deployed ensembling.

Only approved data/train recordings are read. New evidence gets a unique name
under results/step6/smoke; temporary models and caches are removed after tests.
Run: python -m unittest discover -s tests -p test_chunk.py -v -f
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "tests"))
sys.path.insert(0, str(TASK_DIR / "results" / "step6"))

import numpy as np

import bc_features as bcf
import gp_training as gp
import test_training as tt
import train_policy as tp
from collect_data import wheel_speeds
from policies.bc_policy import BASE_PARAMS, BCPolicy, ChunkPolicy, make_policy

SMOKE = TASK_DIR / "results" / "step6" / "smoke"
EVIDENCE = {"source": "data/train only", "G4": {}}


def setUpModule():
    SMOKE.mkdir(parents=True, exist_ok=True)


def tearDownModule():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = SMOKE / f"G4_verification_revision1_{stamp}_{os.getpid()}.json"
    path.write_text(json.dumps(EVIDENCE, indent=2) + "\n")
    print(f"G4 evidence: {path}", flush=True)


def recorded_episode(episode, split):
    path = Path(tp.TRAIN_DIR) / split["episodes"][str(episode)]["file"]
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("base_pos", "base_bearing", "q_mes", "qd_mes", "target")}


def recorded_obs(a, step):
    return {key: (a[key].astype(float) if key == "target" else
                  float(a[key][step]) if key == "base_bearing" else a[key][step].astype(float))
            for key in a}


def offline_ensemble(chunks, delta):
    """Independent batched gather: all queries for a whole recorded episode.

    Rows are time, columns are segment j (oldest to newest); unavailable queries
    have zero weight. This does not use ChunkPolicy's ring buffer or ensemble.
    """
    n, k, _ = chunks.shape
    times = np.arange(n)[:, None]
    js = np.arange(k - 1, -1, -1)[None, :]
    queries = times - js * delta
    available = queries >= 0
    oldest_j = np.minimum(k - 1, times // delta)
    weights = np.exp(-0.11 * (oldest_j - js)) * available
    values = chunks[np.maximum(queries, 0), np.broadcast_to(js, queries.shape)]
    return (values * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None]


def save_linear(path, k, delta, weights=None, bias=None):
    width = 9 * k
    if weights is None:
        weights = np.zeros((20, width))
    if bias is None:
        bias = np.zeros(width)
    meta = {"output_representation": "chunk", "chunk_k": k, "chunk_delta": delta,
            "ensemble_m": 0.11, "output_names": gp.output_names("chunk", k)}
    bcf.save_model(str(path), "ego_novw", np.zeros(20), np.ones(20), np.zeros(width),
                   np.ones(width), [weights], [bias], meta)


NEEDS_DATA = unittest.skipUnless(  # recorded demonstrations are not in git
    all(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", d, "index.csv"))
        for d in ("train", "pilot")),
    "needs data/train and data/pilot (collect_data.py, see DATA_COLLECTION.md)")


@unittest.skipUnless((TASK_DIR / "results" / "step3" / "cache" / "manifest.json").exists()
                     and (TASK_DIR / "results" / "step3" / "cache" / "trigger" / "manifest.json").exists(),
                     "needs the step-3 feature caches (train_policy.py build-cache, build-trigger-cache)")
class TestChunkRecordedLabels(unittest.TestCase):
    def test_a_1000_random_training_rows_all_segments_and_terminal_padding(self):
        """Sample actual C3 fitted rows, including trigger-frame source copies."""
        split = tp.load_split()
        cache = tp.Cache("train")
        trigger, _ = tp.load_trigger_rows(cache.episodes)
        rng = np.random.default_rng(2026110204)
        # C3 adds 16 copies of each trigger source frame after its stride rows.
        n_base, n_trigger = cache.n_rows, len(cache.episodes) * 16
        rows = rng.choice(n_base + n_trigger, size=1000, replace=False)
        by_episode = {}
        for row in rows:
            if row < n_base:
                episode, step = int(cache.cols["episode"][row]), int(cache.cols["step"][row])
                label = tp.cache_labels(cache.cols, slice(int(row), int(row) + 1))[0]
                source = "stride"
            else:
                index = int((row - n_base) // 16)
                episode, step = int(trigger["episode"][index]), int(trigger["step"][index])
                label = tp.cache_labels(trigger, slice(index, index + 1))[0]
                source = "trigger"
            by_episode.setdefault(episode, []).append((step, label, source))
        counts = {"stride": 0, "trigger": 0}
        padded = {str(delta): 0 for delta in (9, 27, 81)}
        extra_terminal_checks = 0
        for episode, source_rows in by_episode.items():
            path = Path(tp.TRAIN_DIR) / split["episodes"][str(episode)]["file"]
            with np.load(path, allow_pickle=False) as data:
                a = {key: data[key] for key in ("t", "wheel_cmds", "torque_arm")}
            steps = np.array([r[0] for r in source_rows])
            c3 = np.stack([r[1] for r in source_rows])
            for _, _, source in source_rows:
                counts[source] += 1
            for delta in (9, 27, 81):
                block = gp.chunk_labels(a, steps, k=10, delta=delta).reshape(-1, 10, 9)
                np.testing.assert_array_equal(block[:, 0], c3, err_msg=f"j0/C3 episode {episode}")
                for j in range(10):
                    at = np.minimum(steps + j * delta, len(a["t"]) - 1)
                    v, w = bcf.wheels_to_vw(a["wheel_cmds"][at])
                    expected = np.column_stack([v, w, a["torque_arm"][at].astype(np.float64)])
                    np.testing.assert_array_equal(block[:, j], expected,
                                                  err_msg=f"episode {episode}, delta {delta}, j {j}")
                padded[str(delta)] += int((steps + 9 * delta >= len(a["t"])).sum())
            # Force padding at the final source row even if the random sample
            # happens not to contain a padded row for one of the deltas.
            terminal = gp.chunk_labels(a, [len(a["t"]) - 1], k=10, delta=81).reshape(10, 9)
            np.testing.assert_array_equal(terminal, np.repeat(terminal[:1], 10, axis=0))
            extra_terminal_checks += 1
        cache.release()
        self.assertEqual(sum(counts.values()), 1000)
        self.assertGreater(counts["trigger"], 0)
        EVIDENCE["G4"]["a_b_labels"] = {"random_rows": 1000, "episodes": len(by_episode),
            "source_rows": counts, "deltas": [9, 27, 81], "segments": 10,
            "exact_j0_and_all_segments": True, "padded_random_rows": padded,
            "forced_terminal_checks": extra_terminal_checks}


@NEEDS_DATA
class TestChunkTrainingMatrices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tt.TestTriggerCopies.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.tr.release()
        cls.va.release()
        tt.TestTriggerCopies.tearDownClass.__func__(cls)

    def test_b_k1_training_matrices_sha256_match_actual_c3_trainer(self):
        captured = {}
        real_train = tp.train_mlp

        def capture(X, Y, Xv, Yv, *args, **kwargs):
            for name, value in (("X", X), ("Y", Y), ("Xv", Xv), ("Yv", Yv)):
                captured[name] = value.copy()
            return real_train(X, Y, Xv, Yv, *args, **kwargs)

        with mock.patch.object(tp, "train_mlp", side_effect=capture):
            tp.run_train("ego-novw-trig", seed=0, out=os.path.join(self.tmp, "c3_capture.npz"),
                         cache_dir=self.cache_dir, split_path=self.split_path, trigger_dir=self.trig_dir,
                         trigger_copies_k=16, trigger_qd_jitter="auto", max_epochs=1,
                         max_train_episodes=3, max_val_episodes=1, log=lambda s: None)
        chunk = gp.prepare_training_data(gp.G_MODEL, cache_dir=self.cache_dir, split_path=self.split_path,
                        trigger_dir=self.trig_dir, trigger_copies_k=16, trigger_qd_jitter="auto",
                        chunk_k=1, chunk_delta=9, max_train_episodes=3, max_val_episodes=1, allow_subset=True)
        hashes = {}
        for name in ("X", "Y", "Xv", "Yv"):
            np.testing.assert_array_equal(captured[name], chunk[name], err_msg=name)
            hashes[name] = {"C3": gp.matrix_sha(captured[name]), "chunk_k1": gp.matrix_sha(chunk[name]),
                            "shape": list(chunk[name].shape)}
            self.assertEqual(hashes[name]["C3"], hashes[name]["chunk_k1"])
        for key, model_key in (("xm", "x_mean"), ("xs", "x_scale"), ("ym", "y_mean"), ("ys", "y_scale")):
            np.testing.assert_array_equal(chunk[key], bcf.load_model(os.path.join(self.tmp, "c3_capture.npz"))[model_key])
        EVIDENCE["G4"]["b_k1_matrices"] = {"scope": "3 training + 1 validation episodes; actual C3 train_mlp input",
                                           "trigger_copies": 16, "sha256": hashes}


@NEEDS_DATA
class TestChunkTemporalEnsembling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="g4_models_", dir=SMOKE)
        cls.split = tp.load_split()
        cls.episode = cls.split["train_episodes"][0]
        cls.a = recorded_episode(cls.episode, cls.split)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def path(self, name):
        return Path(self.tmp.name) / name

    def test_c_constant_model_and_reset(self):
        constant = np.array([.2, -.4, 1., -2., 3., -4., 5., -6., 7.])
        maximum = 0.
        obs = recorded_obs(self.a, 0)
        for delta in (9, 27, 81):
            path = self.path(f"constant_{delta}.npz")
            save_linear(path, 10, delta, bias=np.tile(constant, 10))
            policy = make_policy(str(path))
            self.assertIsInstance(policy, ChunkPolicy)
            policy.reset(None, None)
            for _ in range(2 * policy._buffer_size + 1):
                wheels, tau = policy.act(obs)
                maximum = max(maximum, float(np.max(np.abs(tau - constant[2:]))),
                              float(np.max(np.abs(wheels - wheel_speeds(*constant[:2], BASE_PARAMS)))))
            # Deliberately make the old buffer incompatible with a new episode.
            policy._chunks.fill(1e6)
            policy.reset(None, None)
            wheels, tau = policy.act(obs)
            np.testing.assert_array_equal(tau, constant[2:])
            np.testing.assert_array_equal(wheels, wheel_speeds(*constant[:2], BASE_PARAMS))
        self.assertLess(maximum, 1e-9)
        EVIDENCE["G4"]["c_constant_reset"] = {"deltas": [9, 27, 81], "max_error": maximum}

    def test_c_known_sequence_weights_alignment_and_start_normalisation(self):
        k, delta, n = 4, 3, 43
        path = self.path("known_sequence.npz")
        save_linear(path, k, delta)
        policy = make_policy(str(path))
        policy.reset(None, None)
        maximum = 0.
        for t in range(n):
            current = t * .01 + np.arange(k)[:, None] * .1 + np.arange(9)[None, :] * .001
            got = policy.ensemble(current)
            js = list(range(min(k - 1, t // delta), -1, -1))
            expected = np.zeros(9)
            denominator = 0.
            for rank, j in enumerate(js):
                query = t - j * delta
                weight = np.exp(-.11 * rank)
                expected += weight * (query * .01 + j * .1 + np.arange(9) * .001)
                denominator += weight
            expected /= denominator
            maximum = max(maximum, float(np.max(np.abs(got - expected))))
            if t < delta:
                np.testing.assert_array_equal(got, current[0])
        self.assertLess(maximum, 1e-9)
        # At t=9, j=3 queried at t=0 is oldest and gets weight 1;
        # the current query's j=0 gets exp(-0.33), not the reverse.
        self.assertGreater(np.exp(-.11 * 0), np.exp(-.11 * (k - 1)))
        EVIDENCE["G4"]["c_known_sequence"] = {"steps": n, "k": k, "delta": delta,
            "max_error": maximum, "oldest_weight": 1., "newest_weight": float(np.exp(-.33))}

    def test_c_k1_matches_bcpolicy_on_300_recorded_steps(self):
        path = self.path("k1.npz")
        rng = np.random.default_rng(106)
        save_linear(path, 1, 9, rng.normal(0, .03, (20, 9)), rng.normal(0, .01, 9))
        chunk, c3 = make_policy(str(path)), BCPolicy(str(path))
        chunk.reset(None, None)
        c3.reset(None, None)
        maximum = 0.
        for step in range(300):
            obs = recorded_obs(self.a, step)
            for got, expected in zip(chunk.act(obs), c3.act(obs)):
                maximum = max(maximum, float(np.max(np.abs(got - expected))))
        self.assertLess(maximum, 1e-9)
        EVIDENCE["G4"]["c_k1_BCPolicy"] = {"episode": self.episode, "steps": 300, "max_error": maximum}

    def test_d_recorded_episode_act_matches_offline_vectorised_ensembling(self):
        X = bcf.build_features("ego_novw", bcf.measured_signals(
            self.a["base_pos"], self.a["base_bearing"], self.a["q_mes"], self.a["qd_mes"], self.a["target"]))
        rng = np.random.default_rng(109)
        k, maxima = 10, {}
        # Use non-constant real-observation predictions and all registered deltas.
        for delta in (9, 27, 81):
            path = self.path(f"recorded_{delta}.npz")
            save_linear(path, k, delta, rng.normal(0, .015, (20, 9 * k)), rng.normal(0, .01, 9 * k))
            policy = make_policy(str(path))
            policy.reset(None, None)
            predictions = bcf.forward(policy.model, X).reshape(len(X), k, 9)
            offline = offline_ensemble(predictions, delta)
            maximum = 0.
            with tp.single_thread():
                for step, expected in enumerate(offline):
                    wheels, tau = policy.act(recorded_obs(self.a, step))
                    maximum = max(maximum, float(np.max(np.abs(tau - expected[2:]))),
                                  float(np.max(np.abs(wheels - wheel_speeds(*expected[:2], BASE_PARAMS)))))
            self.assertLess(maximum, 1e-9, f"delta {delta}")
            maxima[str(delta)] = maximum
        EVIDENCE["G4"]["d_train_deploy"] = {"episode": self.episode, "steps_per_delta": len(X),
            "deltas": [9, 27, 81], "max_errors": maxima, "max_error": max(maxima.values())}


if __name__ == "__main__":
    unittest.main(failfast=True)
