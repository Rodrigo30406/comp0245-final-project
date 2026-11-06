"""Inputs, outputs and the three policies: transforms, live == batch inference, save/load, NumPy == torch, training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_learning.data.schema import dataset_schema  # noqa: E402
from robot_learning.data.storage import sha256, write_json  # noqa: E402
from robot_learning.learning.features import BASE, EGO, EGO_LIDAR, VARIANTS, encode  # noqa: E402
from robot_learning.learning.mlp import MLPPolicy, decode_outputs, trigger_copies, trigger_rows  # noqa: E402
from robot_learning.learning.outputs import expand_actions, independent_actions  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def spec(variant=EGO):
    return {"variant": variant, "standoff_heights": [0.1, 0.5, 1.1, 1.25], "standoff_distances": [0.8, 0.65, 0.65, 0.35],
            "horizon_s": 40.0, "dt_s": 0.001}


def observation(base_xy, yaw, target, q=None):
    x = np.zeros(21); x[0:2] = base_xy; x[2] = 0.1; x[3] = yaw; x[18:21] = target
    if q is not None:
        x[4:18] = q
    return x


class FeatureTests(unittest.TestCase):
    def test_target_in_base_frame_formula(self):
        # target 2 m straight ahead of a base facing +y, 0.4 m above it
        f = encode(observation([1.0, -1.0], np.pi / 2, [1.0, 1.0, 0.5])[None], np.zeros(1), spec())[0]
        np.testing.assert_allclose(f[21:27], [2.0, 0.0, 0.4, 2.0, 1.0, 0.0], atol=1e-12)
        # target to the left: heading error +90 degrees
        f = encode(observation([0, 0], 0.0, [0.0, 1.5, 0.3])[None], np.zeros(1), spec())[0]
        np.testing.assert_allclose(f[21:27], [0.0, 1.5, 0.2, 1.5, 0.0, 1.0], atol=1e-12)

    def test_same_relative_situation_gives_identical_egocentric_features(self):
        rel = np.array([1.2, 0.5])
        rows = []
        for base, yaw in (([-1.5, -1.0], -0.35), ([1.2, -1.6], 2.0), ([0.3, 1.9], -2.9)):
            r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rows.append(encode(observation(base, yaw, [*(np.array(base) + r @ rel), 0.6])[None], np.zeros(1), spec())[0])
        for f in rows[1:]:
            np.testing.assert_allclose(f[21:27], rows[0][21:27], atol=1e-12)
            self.assertFalse(np.allclose(f[:21], rows[0][:21]))

    def test_variants_keep_the_raw_inputs_and_append_the_clock(self):
        x = np.random.default_rng(0).normal(size=(5, 21)); t = np.arange(5) * 0.5
        for v in (BASE, EGO):
            f = encode(x, t, spec(v))
            self.assertEqual(f.shape, (5, len(VARIANTS[v])))
            np.testing.assert_array_equal(f[:, :21], x)
            np.testing.assert_allclose(f[:, -1], t / 40.0)
        scan = np.full((5, 64), 4.0)
        f = encode(x, t, spec(EGO_LIDAR), lidar=scan)
        np.testing.assert_array_equal(f[:, 27:91], scan)
        with self.assertRaises(ValueError):
            encode(x, t, spec(EGO_LIDAR))


class OutputTests(unittest.TestCase):
    def test_nine_outputs_expand_to_eleven_and_pairs_are_required(self):
        reduced = np.arange(36.0).reshape(4, 9)
        full = expand_actions(reduced)
        np.testing.assert_array_equal(independent_actions(full), reduced)
        full[0, 3] += 0.1
        with self.assertRaisesRegex(ValueError, "same-side"):
            independent_actions(full)


def random_mlp(variant=EGO, bins=None, seed=0):
    rng = np.random.default_rng(seed); n = len(VARIANTS[variant]); out = 9 if bins is None else 2 * bins + 7
    w = [rng.normal(0, .3, (n, 16)), rng.normal(0, .3, (16, out))]; b = [rng.normal(0, .1, 16), rng.normal(0, .1, out)]
    centres = None if bins is None else np.linspace(-3, 3, bins)
    return MLPPolicy(spec(variant), rng.normal(size=n), np.ones(n), np.zeros(9), np.ones(9), w, b, centres)


class MLPTests(unittest.TestCase):
    def test_live_inference_matches_batch_and_reset_restarts_the_clock(self):
        p = random_mlp(); x = np.random.default_rng(1).normal(size=(30, 21))
        batch = expand_actions(p.predict(x))
        p.reset(0.001); live = np.stack([p.predict_action(o) for o in x])
        np.testing.assert_allclose(live, batch, atol=1e-12)
        p.reset(0.001); np.testing.assert_allclose(p.predict_action(x[0]), batch[0], atol=1e-12)

    def test_save_load_roundtrip_regression_and_categorical(self):
        x = np.random.default_rng(2).normal(size=(10, 21))
        with tempfile.TemporaryDirectory() as tmp:
            for p in (random_mlp(), random_mlp(bins=41)):
                p.save(Path(tmp) / "m.npz"); q = MLPPolicy.load(Path(tmp) / "m.npz")
                np.testing.assert_array_equal(q.predict(x), p.predict(x))

    def test_categorical_wheels_output_bin_centres_with_an_exact_zero(self):
        centres = np.linspace(-3, 3, 41)
        self.assertEqual(centres[20], 0.0)
        out = np.zeros((1, 2 * 41 + 7)); out[0, 20] = 5; out[0, 41 + 3] = 5
        d = decode_outputs(out, np.zeros(9), np.ones(9), centres)
        self.assertEqual(d[0, 0], 0.0); self.assertEqual(d[0, 1], centres[3])

    def test_trigger_row_and_copies(self):
        y = np.zeros((10, 9)); y[:6, :2] = 1.0; y[:, 2:] = np.arange(7)   # wheels stop at row 6
        x = np.random.default_rng(3).normal(size=(10, len(VARIANTS[EGO])))
        rows = trigger_rows(y, [(0, 10)])
        self.assertEqual(rows, [6])
        xc, yc, sigma = trigger_copies(x, y, list(VARIANTS[EGO]), rows, 5, jitter=0.1)
        self.assertEqual(xc.shape[0], 5)
        np.testing.assert_array_equal(yc, np.repeat(y[6:7], 5, axis=0))     # labels untouched
        qd = [VARIANTS[EGO].index(f"arm_qd_{i}") for i in range(1, 8)]
        other = [i for i in range(x.shape[1]) if i not in qd]
        np.testing.assert_array_equal(xc[:, other], np.repeat(x[6:7, other], 5, axis=0))   # only the arm velocities move
        self.assertTrue(np.all(xc[:, qd] != x[6, qd]))


@unittest.skipUnless(HAVE_TORCH, "torch not installed (RoboEnv ml environment)")
class SequencePolicyTests(unittest.TestCase):
    def test_gru_numpy_matches_torch_and_live_matches_batch(self):
        import torch
        from robot_learning.learning.gru import build_network, export_policy
        torch.manual_seed(0); n = len(VARIANTS[EGO]); net = build_network(torch, n, 8, 8).double()
        stats = (np.zeros(n), np.ones(n), np.zeros(9), np.ones(9))
        x = np.random.default_rng(4).normal(size=(45, 21)); t = np.arange(45) * 0.001
        f = encode(x, t, spec())
        for stride in (1, 20):
            p = export_policy(net, spec(), EGO, stats, stride)
            with torch.no_grad():
                ref, _ = net(torch.from_numpy(f[None]), stride)
            np.testing.assert_allclose(p.predict(x), ref[0].numpy(), atol=1e-9)
            p.reset(0.001); live = np.stack([p.predict_action(o) for o in x])
            np.testing.assert_allclose(live, expand_actions(p.predict(x)), atol=1e-9)

    def test_transformer_numpy_matches_torch_and_live_matches_batch(self):
        import torch
        from robot_learning.learning.transformer import build_network, export_policy
        torch.manual_seed(0); n = len(VARIANTS[EGO]); net = build_network(torch, n, 8, 1, 2, 16, 8, 5).double().eval()
        stats = (np.zeros(n), np.ones(n), np.zeros(9), np.ones(9))
        x = np.random.default_rng(5).normal(size=(60, 21)); t = np.arange(60) * 0.001
        f = encode(x, t, spec())
        p = export_policy(net, spec(), EGO, stats, 4, 5, 2)
        with torch.no_grad():
            ref = net(torch.from_numpy(f[None]), 4)
        np.testing.assert_allclose(p.predict(x), ref[0].numpy(), atol=1e-8)
        p.reset(0.001); live = np.stack([p.predict_action(o) for o in x])
        np.testing.assert_allclose(live, expand_actions(p.predict(x)), atol=1e-8)


def make_dataset(root, episodes=6, rows=40):
    """A tiny dense dataset in the collector's layout (1 ms steps), enough for the trainers' checks."""
    (root / "episodes").mkdir(parents=True)
    rng = np.random.default_rng(8); scenarios = []
    for i in range(episodes):
        eid = f"episode_{i:06d}"; target = [float(i), 0.2, 0.5]
        x = rng.normal(size=(rows, 21)).astype(np.float32); x[:, -3:] = target
        y = rng.normal(size=(rows, 9)); y[-5:, :2] = 0.0     # a terminal zero-wheel run: a trigger row per episode
        npz = root / "episodes" / f"{eid}.npz"
        np.savez_compressed(npz, observation=x, action=expand_actions(y), expert_phase=np.full(rows, "drive"),
                            step=np.arange(rows), time=np.arange(rows) * 0.001)
        scenario = {"episode_id": eid, "scenario_id": str(i), "target": target}
        scenarios.append(scenario)
        write_json(root / "episodes" / f"{eid}.json", {**scenario, "status": "complete", "success": True, "termination": "success",
                   "actual_initial_observation": x[0].tolist(), "recorded_transitions": rows, "dt": 0.001,
                   "data_file": npz.name, "data_sha256": sha256(npz)})
    write_json(root / "scenarios.json", scenarios); write_json(root / "schema.json", dataset_schema())
    write_json(root / "config.json", {"episodes": episodes})
    for name in ("dataset.json", "provenance.json"):
        write_json(root / name, {"synthetic_test": True})


@unittest.skipUnless(HAVE_TORCH, "torch not installed (RoboEnv ml environment)")
class TrainingTests(unittest.TestCase):
    def test_mlp_training_with_copies_produces_an_evaluator_compatible_run(self):
        from robot_learning.data.splits import inventory, split_episodes
        from robot_learning.evaluation.worker import load_policy
        from robot_learning.learning.mlp import train
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); ds = tmp / "ds"; make_dataset(ds)
            split = {"dataset": str(ds), "extra_datasets": [], **split_episodes(inventory(ds), 0.34, 1)}
            write_json(tmp / "split.json", split)
            cfg = json.loads((ROOT / "experiment_specs/training/mlp.json").read_text())
            cfg["training"].update(hidden_layer_sizes=[8, 8], batch_size=16, max_epochs=3)
            write_json(tmp / "cfg.json", cfg)
            train(ds, tmp / "split.json", tmp / "cfg.json", tmp / "run", device_name="cpu",
                  options={"trigger_copies": 4}, seed=3)
            info = json.loads((tmp / "run" / "model_info.json").read_text())
            self.assertEqual(info["trigger_copies"]["episodes_with_trigger_row"], len(split["training"]))
            self.assertEqual(json.loads((tmp / "run" / "run.json").read_text())["status"], "complete")
            p = load_policy(info["kind"], tmp / "run" / "model.npz")
            self.assertEqual(p.predict_action(np.zeros(21)).shape, (11,))


if __name__ == "__main__":
    unittest.main()
