"""MLP policy (MLP-Base, MLP-Ego, MLP-Ego-IC, MLP-Ego-Cat): one ReLU MLP from the current features to the nine controls.

Recipe (experiment_specs/training/mlp.json): two hidden layers of 128 ReLU units, Adam (lr 1e-3), batches of 1024 rows
drawn from all training episodes, half-sum squared error on the normalised outputs plus a small L2 penalty on the
weights, early stopping on the validation balanced NRMSE (patience 8), the checkpoint with the lowest validation error
is kept. Inputs and outputs are standardised with the training rows' mean and standard deviation.

Training options (config block "options", or --option KEY=VALUE):
  trigger_copies         ignition copies (MLP-Ego-IC): per training episode, K copies of its trigger row (the first
                         row of the terminal run of exactly-zero wheel labels: the instant the base has stopped and the
                         arm starts to reach), with N(0, sigma) noise on the seven arm-velocity inputs only. The row is
                         found from the labels alone, no expert phase is read. See docs/05_arm_ignition.md.
  trigger_qd_jitter      sigma of that noise: "auto" (per-joint std of the arm velocities over the rows where the base
                         moves) or a value in rad/s.
  wheel_output           "regression" (default) or "categorical" (MLP-Ego-Cat): each wheel is a softmax over
                         `wheel_bins` velocity bins with one bin exactly at zero, trained with cross-entropy and decoded
                         by the argmax bin centre. See docs/06_base_drift.md.
  wheel_bins             odd number of bins (41).
  training_episode_cap   data-size curve: keep this many training episodes (fixed random subset).
  training_target_cells  data-size curve with balanced coverage: JSON {"target_grid", "cells"}; keep the training
                         episodes whose target is one of these grid cells.
Inference is NumPy (no torch needed in simulation); the model file is portable (.npz, no pickle).
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

from robot_learning.data.splits import drop_expert_failures, extra_inventory, inventory, resolve_dataset, validate_partition
from robot_learning.data.storage import git_info, read_json, sha256, utc_now, write_json
from robot_learning.settings import PROJECT_ROOT
from robot_learning.simulation.config import episode_defaults, resolve_episode_config, same_expert_configuration
from .episodes import first_rows, load_episode_features
from .features import EGO, INPUT_NAMES, VARIANTS, encode, spec_from_episode_config, validate_spec
from .metrics import aggregate, errors
from .outputs import OUTPUT_NAMES, expand_actions

KIND = "egocentric_mlp_regressor"   # stored in model_info.json; the evaluator loads by kind
DEFAULT_OPTIONS = {"trigger_copies": 0, "trigger_qd_jitter": "auto", "wheel_output": "regression", "wheel_bins": 41,
                   "training_episode_cap": 0, "training_target_cells": ""}
SUBSET_SEED = 0      # one fixed rng for the trigger-copy noise and the episode cap: every training seed sees the same rows


def resolve_options(config, overrides=None):
    opts = {**DEFAULT_OPTIONS, **config.get("options", {}), **(overrides or {})}
    if set(opts) != set(DEFAULT_OPTIONS):
        raise ValueError(f"Unknown options {sorted(set(opts) - set(DEFAULT_OPTIONS))}")
    if opts["wheel_output"] not in ("regression", "categorical"):
        raise ValueError("wheel_output must be regression or categorical")
    if type(opts["wheel_bins"]) is not int or opts["wheel_bins"] < 3 or opts["wheel_bins"] % 2 == 0:
        raise ValueError("wheel_bins must be an odd integer >= 3")
    if type(opts["trigger_copies"]) is not int or opts["trigger_copies"] < 0:
        raise ValueError("trigger_copies must be an integer >= 0")
    jitter = opts["trigger_qd_jitter"]
    if jitter != "auto" and (not isinstance(jitter, (int, float)) or isinstance(jitter, bool) or not math.isfinite(jitter) or jitter < 0):
        raise ValueError('trigger_qd_jitter must be "auto" or a finite number >= 0 (rad/s)')
    if jitter != "auto":
        opts["trigger_qd_jitter"] = float(jitter)
    if type(opts["training_episode_cap"]) is not int or opts["training_episode_cap"] < 0:
        raise ValueError("training_episode_cap must be an integer >= 0 (0 = every training episode)")
    if not isinstance(opts["training_target_cells"], str):
        raise ValueError("training_target_cells must be a path string ('' = every training episode)")
    if opts["training_target_cells"] and opts["training_episode_cap"]:
        raise ValueError("training_target_cells and training_episode_cap are two different subsets; use one")
    return opts


def trigger_rows(y, ranges):
    """Index of the trigger row of each episode: the first row of its terminal run of exactly-zero wheel labels (the
    expert's navigation end, where the wheels stop and the reach torque starts). Label-derived only. None for an
    episode whose last row still has a non-zero wheel command."""
    y = np.asarray(y)
    out = []
    for a, b in ranges:
        zero = (y[a:b, :2] == 0).all(axis=1)
        if b <= a or not zero[-1]:
            out.append(None)
            continue
        k = len(zero) - 1
        while k > 0 and zero[k - 1]:
            k -= 1
        out.append(a + k)
    return out


def trigger_copies(x, y, names, rows, copies, jitter="auto", seed=SUBSET_SEED):
    """`copies` copies of each trigger row (features and labels as recorded) with N(0, sigma_j) noise on the seven
    arm-velocity features only. Returns (x_copies, y_copies, sigma)."""
    rows = [r for r in rows if r is not None]
    qd_cols = [names.index(f"arm_qd_{i}") for i in range(1, 8)]
    moving = ~(np.asarray(y)[:, :2] == 0).all(axis=1)
    sigma = (np.asarray(x)[moving][:, qd_cols].std(axis=0) if jitter == "auto" else np.full(7, float(jitter)))
    if not rows or copies <= 0:
        return np.zeros((0, len(names))), np.zeros((0, np.asarray(y).shape[1])), sigma
    idx = np.repeat(np.asarray(rows, dtype=np.int64), copies)
    xc = np.asarray(x, dtype=np.float64)[idx].copy()
    yc = np.asarray(y, dtype=np.float64)[idx].copy()
    rng = np.random.default_rng(seed)
    xc[:, qd_cols] += rng.normal(0.0, 1.0, (len(idx), 7)) * sigma
    return xc, yc, sigma


def decode_outputs(out, y_mean, y_scale, wheel_bin_centres=None):
    """Network outputs -> nine controls in physical units; categorical wheels take the centre of the argmax bin."""
    out = np.asarray(out, dtype=np.float64)
    if wheel_bin_centres is None:
        return out * y_scale + y_mean
    b = len(wheel_bin_centres)
    wheels = np.stack([wheel_bin_centres[out[:, :b].argmax(axis=1)], wheel_bin_centres[out[:, b:2 * b].argmax(axis=1)]], axis=1)
    return np.concatenate([wheels, out[:, 2 * b:] * y_scale[2:] + y_mean[2:]], axis=1)


class MLPPolicy:
    """ReLU MLP over the features of one input variant, with an internal episode clock; NumPy inference."""

    def __init__(self, spec, x_mean, x_scale, y_mean, y_scale, weights, biases, wheel_bin_centres=None):
        validate_spec(spec)
        self.spec = spec
        self.x_mean, self.x_scale = np.asarray(x_mean), np.asarray(x_scale)
        self.y_mean, self.y_scale = np.asarray(y_mean), np.asarray(y_scale)
        self.weights = [np.asarray(w) for w in weights]
        self.biases = [np.asarray(b) for b in biases]
        self.names = list(VARIANTS[spec["variant"]])
        self.wheel_bin_centres = None if wheel_bin_centres is None else np.asarray(wheel_bin_centres, dtype=np.float64)
        if self.wheel_bin_centres is not None:
            c = self.wheel_bin_centres
            if c.ndim != 1 or len(c) < 3 or not np.isfinite(c).all() or np.any(np.diff(c) <= 0):
                raise ValueError("wheel_bin_centres must be an increasing finite vector")
        n = len(self.names)
        if any(v.shape != (k,) for v, k in [(self.x_mean, n), (self.x_scale, n), (self.y_mean, 9), (self.y_scale, 9)]):
            raise ValueError("Invalid normalization shapes")
        if len(self.weights) < 2 or len(self.biases) != len(self.weights):
            raise ValueError("At least one hidden layer with matched biases is required")
        width = n
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError("Invalid layer shapes")
            width = w.shape[1]
        if width != self.output_width:
            raise ValueError("Output width must be nine controls (or 2 x wheel bins + 7 arm torques)")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Scales must be positive")
        if not all(np.isfinite(v).all() for v in self.arrays().values() if v.dtype.kind == "f"):
            raise ValueError("Non-finite parameters")
        self._count = 0
        self._dt = spec["dt_s"]

    @property
    def output_width(self):
        return 9 if self.wheel_bin_centres is None else 2 * len(self.wheel_bin_centres) + 7

    @property
    def input_features(self):
        return len(self.names)

    def arrays(self):
        result = {"variant": np.array(self.spec["variant"]), "x_mean": self.x_mean, "x_scale": self.x_scale,
                  "y_mean": self.y_mean, "y_scale": self.y_scale,
                  "standoff_heights": np.asarray(self.spec["standoff_heights"], dtype=np.float64),
                  "standoff_distances": np.asarray(self.spec["standoff_distances"], dtype=np.float64),
                  "clock": np.array([self.spec["horizon_s"], self.spec["dt_s"]], dtype=np.float64)}
        result.update({f"weight_{i}": w for i, w in enumerate(self.weights)})
        result.update({f"bias_{i}": b for i, b in enumerate(self.biases)})
        if self.wheel_bin_centres is not None:
            result["wheel_bin_centres"] = self.wheel_bin_centres
        return result

    def predict_features(self, features, block=65536):
        """Row-independent MLP, evaluated in blocks (a 36 M-row training set would otherwise allocate 18 GB per layer)."""
        features = np.asarray(features, dtype=np.float64)
        if len(features) > block:
            return np.concatenate([self.predict_features(features[s:s + block], block) for s in range(0, len(features), block)])
        hidden = ((features - self.x_mean) / self.x_scale).astype(self.weights[0].dtype)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ w + b
            if i < len(self.weights) - 1:
                np.maximum(hidden, 0, out=hidden)
        return decode_outputs(hidden, self.y_mean, self.y_scale, self.wheel_bin_centres)[:, :9]

    def predict(self, observations, times=None, lidar=None):
        """Batch inference for one episode in order; times default to step * dt from zero."""
        x = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        t = np.arange(len(x)) * self.spec["dt_s"] if times is None else np.asarray(times, dtype=np.float64)
        return self.predict_features(encode(x, t, self.spec, lidar=lidar))

    def predict_action(self, observation, lidar=None):
        """Live inference: one 21D observation per physics step; returns the eleven commands."""
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (21,):
            raise ValueError("Live inference takes one 21D observation")
        t = self._count * self._dt
        self._count += 1
        features = encode(x[None], np.array([t]), self.spec,
                          lidar=None if lidar is None else np.asarray(lidar, dtype=np.float64)[None])
        return expand_actions(self.predict_features(features)[0])

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
            spec = {"variant": str(data["variant"]), "standoff_heights": data["standoff_heights"].tolist(),
                    "standoff_distances": data["standoff_distances"].tolist(),
                    "horizon_s": float(data["clock"][0]), "dt_s": float(data["clock"][1])}
            return cls(spec, data["x_mean"], data["x_scale"], data["y_mean"], data["y_scale"],
                       [data[f"weight_{i}"] for i in range(count)], [data[f"bias_{i}"] for i in range(count)],
                       data["wheel_bin_centres"] if "wheel_bin_centres" in data.files else None)


def load_group(directory, records, spec):
    """Stacked features, nine controls and per-episode row ranges, written episode by episode into one preallocated
    array (35 M rows x 28 inputs in float64 must never exist twice)."""
    total = int(sum(int(r["rows"]) for r in records))
    x, ys, ranges, k = None, [], [], 0
    for record in records:
        f, y = load_episode_features(directory, record, spec)
        if len(f) != int(record["rows"]):
            raise ValueError(f"{record['episode_id']}: {len(f)} rows, the split says {record['rows']}")
        if x is None:
            x = np.empty((total, f.shape[1]), dtype=np.float64)
        x[k:k + len(f)] = f
        ys.append(np.asarray(y, dtype=np.float64))
        ranges.append((k, k + len(f)))
        k += len(f)
    return x, np.concatenate(ys), ranges


def standardised_float32(blocks, mean, scale, chunk=1 << 20):
    """((block - mean) / scale).astype(float32) of the stacked blocks, chunk by chunk into one preallocated array."""
    out = np.empty((sum(len(b) for b in blocks), len(mean)), dtype=np.float32)
    k = 0
    for b in blocks:
        for s in range(0, len(b), chunk):
            e = min(s + chunk, len(b))
            out[k + s:k + e] = (b[s:e] - mean) / scale
        k += len(b)
    return out


def select_training(split, opts):
    """Apply the data-size options to the training side (validation untouched); returns (split, record of the filter)."""
    info = {}
    cap = opts["training_episode_cap"]
    if cap:
        if cap > len(split["training"]):
            raise ValueError(f"training_episode_cap {cap} exceeds the {len(split['training'])} training episodes available")
        keep = sorted(np.random.RandomState(SUBSET_SEED).permutation(len(split["training"]))[:cap])
        split = {**split, "training": [split["training"][i] for i in keep]}
        info["training_episode_cap"] = {"cap": cap, "kept": cap, "rng_seed": SUBSET_SEED}
    if opts["training_target_cells"]:
        cells_path = Path(opts["training_target_cells"]).resolve()
        chosen = read_json(cells_path)
        grid = read_json(PROJECT_ROOT / chosen["target_grid"])
        index = {tuple(np.round(t, 9)): i for i, t in enumerate(grid["targets"])}
        cell_of = lambda r: index[tuple(np.round(np.asarray(r["target"], dtype=float), 9))]
        cells = set(chosen["cells"])
        if cells & set(grid["validation_indices"]):
            raise ValueError("training_target_cells contains held-out validation cells")
        kept = [r for r in split["training"] if cell_of(r) in cells]
        if not kept:
            raise ValueError("training_target_cells kept no training episode")
        split = {**split, "training": kept}
        info["training_target_cells"] = {"file": str(cells_path), "sha256": sha256(cells_path), "cells": len(cells),
                                         "kept_episodes": len(kept), "cells_with_episodes": len({cell_of(r) for r in kept})}
    return split, info


def train(dataset, split_path, config_path, output, horizon_s=40.0, device_name="cuda", variant=EGO, options=None, seed=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant {variant!r}; expected one of {sorted(VARIANTS)}")
    names = list(VARIANTS[variant])
    dataset, output, split_path = Path(dataset).resolve(), Path(output).resolve(), Path(split_path).resolve()
    config = read_json(config_path)
    if seed is not None:
        config["training"]["seed"] = int(seed)
    hp = config["training"]
    opts = resolve_options(config, options)
    categorical = opts["wheel_output"] == "categorical"
    saved = read_json(split_path)
    if resolve_dataset(saved["dataset"]) != dataset:
        raise ValueError("Split belongs to another dataset")
    records = inventory(dataset) + extra_inventory(saved)
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    keep_failures = bool(config.get("keep_expert_failures", False))
    episode_filter = {"keep_expert_failures": keep_failures}
    if not keep_failures:
        split, counts = drop_expert_failures(split)
        episode_filter.update(counts)
    split, subset = select_training(split, opts)
    episode_filter.update(subset)
    source_cfg = read_json(dataset / "config.json")
    episode_cfg = resolve_episode_config({k: v for k, v in source_cfg.items() if k in episode_defaults()})
    dt = read_json(dataset / "episodes" / f"{records[0]['episode_id']}.json")["dt"]
    extra_hashes = {}
    for extra in saved.get("extra_datasets", []):
        extra = resolve_dataset(extra)
        extra_cfg = resolve_episode_config({k: v for k, v in read_json(extra / "config.json").items() if k in episode_defaults()})
        extra_dt = read_json(extra / "episodes" / f"{inventory(extra)[0]['episode_id']}.json")["dt"]
        if not same_expert_configuration(extra_cfg, episode_cfg) or extra_dt != dt:
            raise ValueError(f"Extra dataset {extra} has a different episode configuration or dt")
        extra_hashes[str(extra)] = {n: sha256(extra / n) for n in ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]}
    spec = spec_from_episode_config(episode_cfg, dt, horizon_s, variant)
    device = torch.device(device_name)
    torch.manual_seed(hp["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {**config, "input_features": variant, "clock_horizon_s": horizon_s, "options": opts})
    shutil.copy2(split_path, output / "split.json")
    source_paths = sorted((PROJECT_ROOT / "src/robot_learning/learning").glob("*.py"))
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths}
    for p in source_paths:
        dest = output / "source_snapshot" / p.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = {"status": "running", "created_at": utc_now(), "dataset": str(dataset),
                "dataset_hashes": {n: sha256(dataset / n) for n in ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]},
                "extra_datasets": extra_hashes, "source_hashes": source_hashes, "project": git_info(PROJECT_ROOT),
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch"]},
                "python": sys.version, "command": sys.argv, "input_names": names, "raw_observation_names": INPUT_NAMES,
                "output_names": OUTPUT_NAMES, "feature_spec": spec, "backend": "pytorch", "options": opts,
                "episode_filter": episode_filter, "closed_loop_evaluation": "not run by the trainer"}
    write_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        x, y, ranges = load_group(dataset, split["training"], spec)
        vx, vy, vranges = load_group(dataset, split["validation"], spec)
        x_mean, x_scale = x.mean(axis=0), x.std(axis=0)   # scaler from the recorded rows only, copies excluded
        constant_inputs = [names[i] for i in np.flatnonzero(x_scale == 0)]
        x_scale[x_scale == 0] = 1.0
        y_mean, y_scale = y.mean(axis=0), y.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        trigger_info = None
        if opts["trigger_copies"] > 0:
            rows = trigger_rows(y, ranges)
            xc, yc, sigma = trigger_copies(x, y, names, rows, opts["trigger_copies"], opts["trigger_qd_jitter"])
            trigger_info = {"copies_per_episode": opts["trigger_copies"], "episodes_with_trigger_row": sum(r is not None for r in rows),
                            "episodes_without": sum(r is None for r in rows), "rows_added": int(len(xc)),
                            "recorded_rows": int(len(x)), "qd_jitter_rad_s": [float(v) for v in sigma],
                            "jitter_rule": opts["trigger_qd_jitter"], "jitter_seed": SUBSET_SEED}
            y = np.vstack([y, yc])
        else:
            xc = np.zeros((0, x.shape[1]))
        xn = standardised_float32([x, xc], x_mean, x_scale)   # the recorded rows, then the copies
        del x, xc
        yn = ((y - y_mean) / y_scale).astype(np.float32)
        vxn = standardised_float32([vx], x_mean, x_scale)
        tx, ty, tvx = (torch.from_numpy(a).to(device) for a in (xn, yn, vxn))
        centres, tb = None, None
        if categorical:
            # Symmetric bins over the demonstrated wheel envelope; an odd count puts one bin centre exactly at zero.
            limit = float(np.abs(y[:, :2]).max())
            if limit <= 0:
                raise ValueError("Demonstrated wheel commands are all zero; nothing to bin")
            centres = np.linspace(-limit, limit, opts["wheel_bins"])
            bins = np.rint((y[:, :2] + limit) / (2 * limit) * (opts["wheel_bins"] - 1)).astype(np.int64)
            tb = torch.from_numpy(bins).to(device)
        stop_rows = int((y[:, :2] == 0).all(axis=1).sum())
        out_width = 9 if centres is None else 2 * len(centres) + 7
        widths = [len(names), *hp["hidden_layer_sizes"], out_width]
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(torch.nn.ReLU())
        network = torch.nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        rng = np.random.RandomState(hp["seed"])
        with torch.no_grad():   # Glorot-uniform initialisation from the recipe's seed
            for layer, a, b in zip(network[::2], widths[:-1], widths[1:]):
                bound = np.sqrt(6.0 / (a + b))
                layer.weight.copy_(torch.from_numpy(rng.uniform(-bound, bound, (a, b)).astype(np.float32).T.copy()))
                layer.bias.copy_(torch.from_numpy(rng.uniform(-bound, bound, b).astype(np.float32)))
        optimizer = torch.optim.Adam(network.parameters(), lr=hp["learning_rate"], betas=(.9, .999), eps=1e-8)

        def export():
            return MLPPolicy(spec, x_mean, x_scale, y_mean, y_scale,
                             [l.weight.detach().cpu().numpy().T.copy() for l in network[::2]],
                             [l.bias.detach().cpu().numpy().copy() for l in network[::2]], centres)

        def score():
            with torch.no_grad():
                pred = np.concatenate([network(tvx[s:s + 16384]).double().cpu().numpy() for s in range(0, len(tvx), 16384)])
            pred = decode_outputs(pred, y_mean, y_scale, centres)
            return float(np.mean([errors(vy[a:b], pred[a:b], y_scale)["balanced_nrmse"] for a, b in vranges]))

        def row_objective(xb, idx):
            """Per-row loss: wheel term (half-sum L2, or cross-entropy for categorical wheels) + half-sum L2 on the arm."""
            out = network(xb)
            if categorical:
                b = len(centres)
                wheel = (torch.nn.functional.cross_entropy(out[:, :b], tb[idx, 0], reduction="none") +
                         torch.nn.functional.cross_entropy(out[:, b:2 * b], tb[idx, 1], reduction="none"))
                arm = out[:, 2 * b:]
            else:
                wheel = .5 * (out[:, :2] - ty[idx, :2]).square().sum(dim=1)
                arm = out[:, 2:9]
            return wheel + .5 * (arm - ty[idx, 2:9]).square().sum(dim=1)

        best, patience_score, stale, best_epoch, history, epoch, n = math.inf, math.inf, 0, 0, [], 0, len(tx)
        while stale < hp["patience"] and (hp["max_epochs"] is None or epoch < hp["max_epochs"]):
            epoch += 1
            t0 = time.perf_counter()
            order = torch.from_numpy(rng.permutation(n)).to(device)
            total = 0.0
            for s in range(0, n, hp["batch_size"]):
                idx = order[s:s + hp["batch_size"]]
                rows = row_objective(tx[idx], idx)
                objective = rows.mean() + (.5 * hp["alpha"] / len(idx)) * sum(l.weight.square().sum() for l in network[::2])
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                total += float(rows.detach().sum()) / 9
            current = score()
            if not math.isfinite(current):
                raise ValueError("Non-finite validation score")
            if current < best:
                best, best_epoch = current, epoch
                export().save(output / "model.npz")
            if current < patience_score - hp["min_delta"]:
                patience_score, stale = current, 0
            else:
                stale += 1
            history.append({"epoch": epoch, "training_normalized_loss": total / n,
                            "validation_episode_macro_balanced_nrmse": current, "seconds": time.perf_counter() - t0})
            write_json(output / "history.json", history)
            print(f"Epoch {epoch}: loss={total / n:.6f}, validation NRMSE={current:.6f}, best={best:.6f}", flush=True)
        policy = MLPPolicy.load(output / "model.npz")
        # The live path (one observation per step, internal clock) must agree with the batch path used for the metrics.
        xr, lr = first_rows(dataset, split["validation"][0])
        policy.reset(dt)
        live = np.stack([policy.predict_action(o, lidar=None if lr is None else lr[i]) for i, o in enumerate(xr)])
        np.testing.assert_allclose(live, expand_actions(policy.predict(xr, lidar=lr)), rtol=1e-6, atol=1e-6)
        policy.reset()
        del y, xn, yn, vx, vy, vxn, tx, ty, tvx   # free the training tensors before the final metric pass
        metrics = {}
        for name, group in [("training", split["training"]), ("validation", split["validation"])]:
            feats, truth, ranges_ = load_group(dataset, group, spec)
            pred = policy.predict_features(feats)
            metrics[name] = {"model": aggregate([errors(truth[a:b], pred[a:b], y_scale) for a, b in ranges_]),
                             "mean_baseline": aggregate([errors(truth[a:b], np.broadcast_to(y_mean, (b - a, 9)), y_scale) for a, b in ranges_]),
                             "pooled_transitions": {"model": errors(truth, pred, y_scale)}}
        timings = []
        policy.reset(dt)
        for _ in range(100):
            policy.predict_action(xr[0], lidar=None if lr is None else lr[0])
        for _ in range(2000):
            start = time.perf_counter_ns()
            policy.predict_action(xr[0], lidar=None if lr is None else lr[0])
            timings.append(time.perf_counter_ns() - start)
        policy.reset()
        if any(sha256(PROJECT_ROOT / p) != d for p, d in source_hashes.items()):
            raise ValueError("Training source changed during execution")
        info = {"kind": KIND, "variant": variant, "backend": "pytorch", "training_device": str(device),
                "hidden_layer_sizes": hp["hidden_layer_sizes"], "activation": "relu",
                "output_activation": "identity" if not categorical else "identity (wheel logits) + argmax bin centre",
                "options": opts, "wheel_output": opts["wheel_output"],
                "wheel_bin_centres": None if centres is None else centres.tolist(),
                "stop_rows_in_training": stop_rows, "training_rows": int(n), "trigger_copies": trigger_info,
                "episode_filter": episode_filter, "training_episodes": len(split["training"]),
                "validation_episodes": len(split["validation"]),
                "regression_parameters": sum(p.numel() for p in network.parameters()), "weight_dtype": "float32",
                "input_features": len(names), "input_names": names, "independent_outputs": 9, "feature_spec": spec,
                "epochs_completed": epoch, "best_epoch": best_epoch, "best_validation_episode_macro_balanced_nrmse": best,
                "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
                "checkpoint_selection": "minimum validation episode-macro balanced NRMSE",
                "training_loss": ("cross-entropy over wheel velocity bins" if categorical else "half-sum L2 on the two wheel outputs")
                                 + "; half-sum L2 on the seven arm outputs; alpha/(2*batch_rows) weight L2",
                "constant_training_inputs": constant_inputs,
                "inference_single_observation_us": {"median": float(np.median(timings) / 1000),
                                                    "p95": float(np.percentile(timings, 95) / 1000), "samples": len(timings)},
                "output_limits": "none applied; the evaluator clips to the declared envelope",
                "fit_and_transform_seconds": time.perf_counter() - started}
        write_json(output / "model_info.json", info)
        write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", finished_at=utc_now(), wall_s=time.perf_counter() - started,
                        model_sha256=sha256(output / "model.npz"), split_sha256=sha256(output / "split.json"),
                        metrics_sha256=sha256(output / "metrics.json"))
        write_json(output / "run.json", manifest)
        return policy, metrics
    except BaseException as exc:
        manifest.update(status="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="primary dataset directory of the split")
    parser.add_argument("--split", required=True, help="split JSON (scripts/data/make_train_val_split.py)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/mlp.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-s", type=float, default=40.0, help="clock normalisation (the evaluation horizon)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs", default=EGO, choices=sorted(VARIANTS), help="input variant (learning/features.py)")
    parser.add_argument("--seed", type=int, help="training seed (overrides the config; seed 0 of the report = 20261025)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help=f"override a training option, e.g. --option trigger_copies=144; keys: {sorted(DEFAULT_OPTIONS)}")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.option:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value   # bare strings such as categorical
    _, metrics = train(args.dataset, args.split, args.config, args.output_dir, args.horizon_s, args.device, args.inputs, overrides, args.seed)
    for name, result in metrics.items():
        print(f"{name}: episode-macro balanced NRMSE={result['model']['episode_macro']['balanced_nrmse']:.6f}; "
              f"mean baseline={result['mean_baseline']['episode_macro']['balanced_nrmse']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
