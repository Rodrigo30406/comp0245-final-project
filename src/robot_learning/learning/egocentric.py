"""Declared variant: egocentric goal geometry plus an episode clock as policy inputs.

Every feature is a deterministic function of the 21 baseline inputs, the fixed
standoff table of the expert configuration, and the controller's own clock.
No expert memory, expert action or future information is used. The variant
exists to test whether observable surrogates for the expert's navigation phase
(research: docs/PRIVILEGED_ABLATION.md) change supervised and closed-loop
behaviour. Output interface is unchanged: nine controls expanded to eleven.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from robot_learning.data.storage import read_json, write_json, sha256, utc_now, git_info
from robot_learning.settings import PROJECT_ROOT
from robot_learning.simulation.config import episode_defaults, resolve_episode_config
from .dataset import inventory, load_episode, validate_partition, inventory_fingerprint, extra_inventory, record_directory
from .metrics import errors, aggregate
from .model import INPUT_NAMES, OUTPUT_NAMES, expand_actions

FEATURE_NAMES = (["goal_in_base_x", "goal_in_base_y", "goal_distance_xy",
                  "cos_heading_error_to_goal", "sin_heading_error_to_goal",
                  "cos_goal_bearing_error", "sin_goal_bearing_error",
                  "target_in_base_x", "target_in_base_y", "target_height_above_base"] +
                 [f"arm_q_{i}" for i in range(1, 8)] + [f"arm_qd_{i}" for i in range(1, 8)] +
                 ["episode_clock_fraction"])
VARIANT = "egocentric_clock_v1"
GEOMETRY_NAMES = FEATURE_NAMES[:10]
TARGET_NAMES = FEATURE_NAMES[7:10]  # target in the base frame: no expert goal construction involved
CLOCK_NAME = "episode_clock_fraction"
# Input layouts. The lecturer's 21 explicit inputs may be kept and extended (not replaced):
#   egocentric_clock_v1       : 10 relative geometry + 14 joint + clock (25); replaces the 7 absolute pose/target inputs
#   raw21_clock_v1            : the 21 explicit inputs + clock (22)
#   raw21_egocentric_clock_v1 : the 21 explicit inputs + 10 relative geometry + clock (32)
#   raw21_target_clock_v1     : the 21 explicit inputs + 3 target-in-base-frame features + clock (25).
#                               Uses no knowledge of the expert (no standoff table, no goal bearing rule):
#                               only a rigid transform of observed quantities into the robot frame.
VARIANTS = {VARIANT: list(FEATURE_NAMES),
            "raw21_clock_v1": list(INPUT_NAMES) + [CLOCK_NAME],
            "raw21_egocentric_clock_v1": list(INPUT_NAMES) + list(GEOMETRY_NAMES) + [CLOCK_NAME],
            "raw21_target_clock_v1": list(INPUT_NAMES) + list(TARGET_NAMES) + [CLOCK_NAME]}
KIND = "egocentric_mlp_regressor"


def validate_spec(spec):
    keys = {"variant", "standoff_heights", "standoff_distances", "horizon_s", "dt_s"}
    if set(spec) != keys or spec["variant"] not in VARIANTS:
        raise ValueError("Invalid egocentric feature specification")
    h = np.asarray(spec["standoff_heights"], dtype=float); d = np.asarray(spec["standoff_distances"], dtype=float)
    if h.ndim != 1 or h.shape != d.shape or len(h) < 2 or np.any(np.diff(h) <= 0) or not np.isfinite(np.r_[h, d]).all():
        raise ValueError("Standoff table must be increasing heights with matching distances")
    for key in ("horizon_s", "dt_s"):
        if not isinstance(spec[key], (int, float)) or not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} must be positive")


def spec_from_episode_config(cfg, dt_s, horizon_s, variant=VARIANT):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return {"variant": variant, "standoff_heights": list(map(float, cfg["standoff"]["heights"])),
            "standoff_distances": list(map(float, cfg["standoff"]["distances"])),
            "horizon_s": float(horizon_s), "dt_s": float(dt_s)}


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def encode(observations, times, spec):
    """(N, 21) raw observations and (N,) seconds since episode start -> (N, F) features of spec["variant"].

    Reproduces the expert's goal construction from `Episode._initialize`: bearing from the
    world origin to the target, standoff interpolated on target height, goal behind the
    target along that bearing. Relative quantities are expressed in the base frame.
    """
    x = np.asarray(observations, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 21 or t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("Expected finite (N, 21) observations and (N,) times")
    base_xy, base_z, yaw = x[:, 0:2], x[:, 2], x[:, 3]
    q, qd, target = x[:, 4:11], x[:, 11:18], x[:, 18:21]
    planar = np.hypot(target[:, 0], target[:, 1])
    goal_bearing = np.where(planar > 1e-4, np.arctan2(target[:, 1], target[:, 0]), 0.0)
    standoff = np.interp(target[:, 2], spec["standoff_heights"], spec["standoff_distances"])
    goal = target[:, :2] - standoff[:, None] * np.stack([np.cos(goal_bearing), np.sin(goal_bearing)], axis=1)
    c, s = np.cos(yaw), np.sin(yaw)

    def to_base(delta):
        return np.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], axis=1)

    d_goal = goal - base_xy
    goal_b = to_base(d_goal)
    distance = np.hypot(d_goal[:, 0], d_goal[:, 1])
    heading_error = wrap(np.arctan2(d_goal[:, 1], d_goal[:, 0]) - yaw)
    bearing_error = wrap(goal_bearing - yaw)
    target_b = to_base(target[:, :2] - base_xy)
    clock = t / spec["horizon_s"]
    geometry = np.concatenate([
        goal_b, distance[:, None], np.cos(heading_error)[:, None], np.sin(heading_error)[:, None],
        np.cos(bearing_error)[:, None], np.sin(bearing_error)[:, None],
        target_b, (target[:, 2] - base_z)[:, None]], axis=1)
    variant = spec["variant"]
    if variant == VARIANT:
        features = np.concatenate([geometry, q, qd, clock[:, None]], axis=1)
    elif variant == "raw21_clock_v1":
        features = np.concatenate([x, clock[:, None]], axis=1)
    elif variant == "raw21_egocentric_clock_v1":
        features = np.concatenate([x, geometry, clock[:, None]], axis=1)
    elif variant == "raw21_target_clock_v1":
        features = np.concatenate([x, geometry[:, 7:10], clock[:, None]], axis=1)
    else:
        raise ValueError(f"Unknown input variant {variant!r}")
    if features.shape[1] != len(VARIANTS[variant]):
        raise AssertionError("Feature layout drifted from VARIANTS")
    return features


class EgocentricPolicy:
    """ReLU MLP over egocentric features with an internal episode clock; NumPy inference."""

    def __init__(self, spec, x_mean, x_scale, y_mean, y_scale, weights, biases):
        validate_spec(spec)
        self.spec = spec
        self.x_mean = np.asarray(x_mean); self.x_scale = np.asarray(x_scale)
        self.y_mean = np.asarray(y_mean); self.y_scale = np.asarray(y_scale)
        self.weights = [np.asarray(w) for w in weights]; self.biases = [np.asarray(b) for b in biases]
        self.names = list(VARIANTS[spec["variant"]])
        n = len(self.names)
        if any(v.shape != (k,) for v, k in [(self.x_mean, n), (self.x_scale, n), (self.y_mean, 9), (self.y_scale, 9)]):
            raise ValueError("Invalid normalization shapes")
        if len(self.weights) < 2 or len(self.biases) != len(self.weights):
            raise ValueError("At least one hidden layer with matched biases is required")
        width = n
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError("Invalid layer shapes")
            width = w.shape[1]
        if width != 9:
            raise ValueError("Output width must be nine controls")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Scales must be positive")
        if not all(np.isfinite(v).all() for v in self.arrays().values() if v.dtype.kind == "f"):
            raise ValueError("Non-finite parameters")
        self._count = 0; self._dt = spec["dt_s"]

    @property
    def input_features(self):
        return len(self.names)

    def arrays(self):
        result = {"variant": np.array(self.spec["variant"]), "x_mean": self.x_mean, "x_scale": self.x_scale, "y_mean": self.y_mean, "y_scale": self.y_scale,
                  "standoff_heights": np.asarray(self.spec["standoff_heights"], dtype=np.float64),
                  "standoff_distances": np.asarray(self.spec["standoff_distances"], dtype=np.float64),
                  "clock": np.array([self.spec["horizon_s"], self.spec["dt_s"]], dtype=np.float64)}
        result.update({f"weight_{i}": w for i, w in enumerate(self.weights)})
        result.update({f"bias_{i}": b for i, b in enumerate(self.biases)})
        return result

    def predict_features(self, features):
        hidden = ((np.asarray(features, dtype=np.float64) - self.x_mean) / self.x_scale).astype(self.weights[0].dtype)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ w + b
            if i < len(self.weights) - 1:
                np.maximum(hidden, 0, out=hidden)
        return hidden * self.y_scale + self.y_mean

    def predict(self, observations, times=None):
        """Batch inference for ONE episode in order; times default to step*dt from zero."""
        x = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        t = np.arange(len(x)) * self.spec["dt_s"] if times is None else np.asarray(times, dtype=np.float64)
        return self.predict_features(encode(x, t, self.spec))

    def predict_action(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (21,):
            raise ValueError("Live inference takes one 21D observation")
        t = self._count * self._dt; self._count += 1
        return expand_actions(self.predict_features(encode(x[None], np.array([t]), self.spec))[0])

    def reset(self, dt=None):
        if dt is not None and not np.isclose(dt, self.spec["dt_s"], rtol=0, atol=1e-12):
            raise ValueError("Policy clock dt differs from simulator dt")
        self._count = 0

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            count = sum(k.startswith("weight_") for k in data.files)
            spec = {"variant": str(data["variant"]) if "variant" in data.files else VARIANT, "standoff_heights": data["standoff_heights"].tolist(),
                    "standoff_distances": data["standoff_distances"].tolist(),
                    "horizon_s": float(data["clock"][0]), "dt_s": float(data["clock"][1])}
            return cls(spec, data["x_mean"], data["x_scale"], data["y_mean"], data["y_scale"],
                       [data[f"weight_{i}"] for i in range(count)], [data[f"bias_{i}"] for i in range(count)])


def load_episode_features(directory, record, spec):
    """Observations, nine controls and recorded times; fails closed on sparse logging."""
    from .windows import validate_episode_sampling
    directory = record_directory(record, directory)
    x, y, _ = load_episode(directory, record)
    path = Path(directory) / "episodes" / f"{record['episode_id']}.npz"
    with np.load(path, allow_pickle=False) as data:
        validate_episode_sampling(data, read_json(path.with_suffix(".json")), {"dt_s": spec["dt_s"]})
        times = data["time"].astype(np.float64)
    return encode(x, times, spec), y


def load_group(directory, records, spec):
    xs, ys = [], []
    for record in records:
        f, y = load_episode_features(directory, record, spec); xs.append(f); ys.append(y)
    rows = [len(f) for f in xs]
    return (np.concatenate(xs), np.concatenate(ys),
            list(zip(np.r_[0, np.cumsum(rows)[:-1]].tolist(), np.cumsum(rows).tolist())))


def train(dataset, split_path, config_path, output, horizon_s=40.0, device_name="cuda", variant=VARIANT):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant {variant!r}; expected one of {sorted(VARIANTS)}")
    names = list(VARIANTS[variant])
    dataset = Path(dataset).resolve(); output = Path(output).resolve(); split_path = Path(split_path).resolve()
    config = read_json(config_path); hp = config["mlp"]
    if config.get("model") != "mlp" or config.get("backend") != "pytorch":
        raise ValueError("Egocentric variant reuses the PyTorch MLP recipe")
    saved = read_json(split_path)
    if Path(saved["dataset"]).resolve() != dataset:
        raise ValueError("Split belongs to another dataset")
    records = inventory(dataset) + extra_inventory(saved)
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    source_cfg = read_json(dataset / "config.json")
    episode_cfg = resolve_episode_config({k: v for k, v in source_cfg.items() if k in episode_defaults()})
    dt = read_json(dataset / "episodes" / f"{records[0]['episode_id']}.json")["dt"]
    extra_hashes = {}
    for extra in saved.get("extra_datasets", []):
        extra = Path(extra).resolve()
        extra_cfg = resolve_episode_config({k: v for k, v in read_json(extra / "config.json").items() if k in episode_defaults()})
        extra_dt = read_json(extra / "episodes" / f"{inventory(extra)[0]['episode_id']}.json")["dt"]
        if extra_cfg != episode_cfg or extra_dt != dt:
            raise ValueError(f"Extra dataset {extra} has a different episode configuration or dt")
        extra_hashes[str(extra)] = {n: sha256(extra / n) for n in ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]}
    spec = spec_from_episode_config(episode_cfg, dt, horizon_s, variant); validate_spec(spec)
    device = torch.device(device_name)
    torch.manual_seed(hp["seed"]); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {**config, "input_features": variant, "clock_horizon_s": horizon_s})
    shutil.copy2(split_path, output / "split.json")
    source_paths = sorted((PROJECT_ROOT / "src/robot_learning/learning").glob("*.py"))
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths}
    for p in source_paths:
        dest = output / "source_snapshot" / p.relative_to(PROJECT_ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = {"status": "running", "created_at": utc_now(), "dataset": str(dataset),
                "dataset_hashes": {n: sha256(dataset / n) for n in
                                   ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]},
                "extra_datasets": extra_hashes,
                "source_hashes": source_hashes, "project": git_info(PROJECT_ROOT),
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch"]},
                "python": sys.version, "command": sys.argv, "input_names": names,
                "raw_observation_names": INPUT_NAMES, "output_names": OUTPUT_NAMES,
                "feature_spec": spec, "backend": "pytorch",
                "test_set": "none; development train/validation only",
                "closed_loop_evaluation": "not run by the trainer"}
    write_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        x, y, _ = load_group(dataset, split["training"], spec)
        vx, vy, vranges = load_group(dataset, split["validation"], spec)
        x_mean, x_scale = x.mean(axis=0), x.std(axis=0); x_scale[x_scale == 0] = 1.0
        y_mean, y_scale = y.mean(axis=0), y.std(axis=0); y_scale[y_scale == 0] = 1.0
        xn = ((x - x_mean) / x_scale).astype(np.float32); yn = ((y - y_mean) / y_scale).astype(np.float32)
        vxn = ((vx - x_mean) / x_scale).astype(np.float32)
        tx = torch.from_numpy(xn).to(device); ty = torch.from_numpy(yn).to(device); tvx = torch.from_numpy(vxn).to(device)
        widths = [len(names), *hp["hidden_layer_sizes"], 9]
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(torch.nn.ReLU())
        network = torch.nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        rng = np.random.RandomState(hp["seed"])
        with torch.no_grad():
            for layer, a, b in zip(network[::2], widths[:-1], widths[1:]):
                bound = np.sqrt(6.0 / (a + b))
                layer.weight.copy_(torch.from_numpy(rng.uniform(-bound, bound, (a, b)).astype(np.float32).T.copy()))
                layer.bias.copy_(torch.from_numpy(rng.uniform(-bound, bound, b).astype(np.float32)))
        optimizer = torch.optim.Adam(network.parameters(), lr=hp["learning_rate"], betas=(.9, .999), eps=1e-8)

        def export():
            return EgocentricPolicy(spec, x_mean, x_scale, y_mean, y_scale,
                                    [l.weight.detach().cpu().numpy().T.copy() for l in network[::2]],
                                    [l.bias.detach().cpu().numpy().copy() for l in network[::2]])

        def score():
            with torch.no_grad():
                pred = np.concatenate([network(tvx[s:s + 16384]).double().cpu().numpy() for s in range(0, len(tvx), 16384)])
            pred = pred * y_scale + y_mean
            return float(np.mean([errors(vy[a:b], pred[a:b], y_scale)["balanced_nrmse"] for a, b in vranges]))

        best = math.inf; patience_score = math.inf; stale = 0; best_epoch = 0; history = []; epoch = 0; n = len(tx)
        while stale < hp["patience"] and (hp["max_epochs"] is None or epoch < hp["max_epochs"]):
            epoch += 1; t0 = time.perf_counter(); order = torch.from_numpy(rng.permutation(n)).to(device); total = 0.0
            for s in range(0, n, hp["batch_size"]):
                idx = order[s:s + hp["batch_size"]]
                squared = (network(tx[idx]) - ty[idx]).square()
                objective = .5 * squared.sum(dim=1).mean() + (.5 * hp["alpha"] / len(idx)) * sum(
                    l.weight.square().sum() for l in network[::2])
                optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
                total += float(.5 * squared.detach().mean()) * len(idx)
            current = score()
            if not math.isfinite(current):
                raise ValueError("Non-finite validation score")
            if current < best:
                best = current; best_epoch = epoch; export().save(output / "model.npz")
            if current < patience_score - hp["min_delta"]:
                patience_score = current; stale = 0
            else:
                stale += 1
            history.append({"epoch": epoch, "training_normalized_loss": total / n,
                            "validation_episode_macro_balanced_nrmse": current, "seconds": time.perf_counter() - t0})
            write_json(output / "history.json", history)
            print(f"Epoch {epoch}: loss={total / n:.6f}, validation NRMSE={current:.6f}, best={best:.6f}", flush=True)
        policy = EgocentricPolicy.load(output / "model.npz")
        # Live path (clock counter) must agree with the batch path used for metrics.
        policy.reset(dt); live = np.stack([policy.predict_action(o) for o in x_check(dataset, split["validation"][0])])
        batch = expand_actions(policy.predict(x_check(dataset, split["validation"][0])))
        np.testing.assert_allclose(live, batch, rtol=1e-6, atol=1e-6)
        policy.reset()
        metrics = {}
        for name, group, ranges_ in [("training", split["training"], None), ("validation", split["validation"], vranges)]:
            feats, truth, ranges_ = load_group(dataset, group, spec)
            pred = policy.predict_features(feats)
            per_episode = [errors(truth[a:b], pred[a:b], y_scale) for a, b in ranges_]
            metrics[name] = {"model": aggregate(per_episode),
                             "mean_baseline": aggregate([errors(truth[a:b], np.broadcast_to(y_mean, (b - a, 9)), y_scale) for a, b in ranges_]),
                             "pooled_transitions": {"model": errors(truth, pred, y_scale)}}
        timings = []
        obs = x_check(dataset, split["validation"][0])[0]
        policy.reset(dt)
        for _ in range(100): policy.predict_action(obs)
        for _ in range(2000):
            start = time.perf_counter_ns(); policy.predict_action(obs); timings.append(time.perf_counter_ns() - start)
        policy.reset()
        if any(sha256(PROJECT_ROOT / p) != d for p, d in source_hashes.items()):
            raise ValueError("Training source changed during execution")
        info = {"kind": KIND, "variant": variant, "backend": "pytorch", "training_device": str(device),
                "hidden_layer_sizes": hp["hidden_layer_sizes"], "activation": "relu", "output_activation": "identity",
                "regression_parameters": sum(p.numel() for p in network.parameters()), "weight_dtype": "float32",
                "input_features": len(names), "input_names": names, "independent_outputs": 9,
                "feature_spec": spec,
                "feature_provenance": "Deterministic functions of the 21 baseline inputs, the expert standoff table and the controller clock; no expert memory or actions",
                "epochs_completed": epoch, "best_epoch": best_epoch,
                "best_validation_episode_macro_balanced_nrmse": best,
                "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
                "checkpoint_selection": "minimum validation episode-macro balanced NRMSE",
                "training_loss": "half output-sum MSE gradients with alpha/(2*batch_rows) weight L2; sklearn convention",
                "constant_training_inputs": [names[i] for i in np.flatnonzero(x.std(axis=0) == 0)],
                "inference_single_observation_us": {"median": float(np.median(timings) / 1000),
                                                    "p95": float(np.percentile(timings, 95) / 1000), "samples": len(timings)},
                "output_limits": "none applied; the evaluator clips to the declared envelope",
                "fit_and_transform_seconds": time.perf_counter() - started}
        write_json(output / "model_info.json", info); write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", finished_at=utc_now(), wall_s=time.perf_counter() - started,
                        model_sha256=sha256(output / "model.npz"), split_sha256=sha256(output / "split.json"),
                        metrics_sha256=sha256(output / "metrics.json"))
        write_json(output / "run.json", manifest)
        return policy, metrics
    except BaseException as exc:
        manifest.update(status="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", manifest)
        raise


def x_check(dataset, record, rows=256):
    x, _, _ = load_episode(dataset, record)
    return x[:rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/mlp_pytorch.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-s", type=float, default=40.0, help="Clock normalization; the evaluation horizon")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs", default=VARIANT, choices=sorted(VARIANTS),
                        help="egocentric_clock_v1 (25) | raw21_clock_v1 (21 explicit + clock) | raw21_egocentric_clock_v1 (21 + 10 + clock)")
    args = parser.parse_args(argv)
    _, metrics = train(args.dataset, args.split, args.config, args.output_dir, args.horizon_s, args.device, args.inputs)
    for name, result in metrics.items():
        print(f"{name}: episode-macro balanced NRMSE={result['model']['episode_macro']['balanced_nrmse']:.6f}; "
              f"mean baseline={result['mean_baseline']['episode_macro']['balanced_nrmse']:.6f}")
    return 0
