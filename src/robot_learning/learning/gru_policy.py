"""Recurrent (GRU) policy: slow memory, fast reaction.

Inputs are either the assignment's 21 explicit observations (`raw21`) or the 25
egocentric-plus-clock features of `learning/egocentric.py`. A single-layer GRU
updates its hidden state every `stride` physics steps (stride 1 = every 1 ms
step; stride 20 = every 20 ms). The output head reacts at EVERY physics step: it
receives the current features concatenated with the latest hidden state and
emits the nine controls. Holding the whole action for 20 ms is not an option
here: the expert itself fails when its command is held
(runs/diagnostics/expert_hold20_validation_20261027), so the control period
stays at 1 ms and only the memory update is strided.

Training uses whole episodes on the GPU with truncated backpropagation through
time: the hidden state is carried across consecutive chunks of the same episode
and detached between them, and reset at every episode boundary. The loss covers
every physics step. Inference is NumPy; the live cell is verified against
torch.nn.GRU by test.
"""
import argparse
import importlib.metadata
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
from .dataset import inventory, load_episode, validate_partition, drop_expert_failures, inventory_fingerprint, extra_inventory, record_directory
from .egocentric import (FEATURE_NAMES, VARIANTS, STATEFUL_VARIANTS, encode, validate_spec, spec_from_episode_config,
                         load_episode_features, VARIANT, x_check)
from .metrics import errors, aggregate
from .model import INPUT_NAMES, OUTPUT_NAMES, expand_actions

KIND = "gru_regressor"
MODES = {"raw21": list(INPUT_NAMES), **{variant: list(names) for variant, names in VARIANTS.items()
                                        if variant not in STATEFUL_VARIANTS}}  # the GRU live path keeps no previous observation


def input_names(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown input mode {mode!r}; expected one of {sorted(MODES)}")
    return MODES[mode]


def features_for(mode, observations, times, spec, lidar=None):
    """raw21: the slide's explicit interface unchanged; any other mode: a variant of egocentric.py (`lidar` (N, 64) for
    the lidar variants)."""
    x = np.asarray(observations, dtype=np.float64)
    if mode == "raw21":
        if x.ndim != 2 or x.shape[1] != 21 or not np.isfinite(x).all():
            raise ValueError("Expected finite (N, 21) observations")
        return x
    return encode(x, times, spec, lidar=lidar)


def gru_step(x, h, w_ih, w_hh, b_ih, b_hh):
    """One torch.nn.GRU cell update in NumPy; x (F,), h (H,); weights in torch layout."""
    gi = w_ih @ x + b_ih
    gh = w_hh @ h + b_hh
    H = len(h)
    r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
    z = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return (1.0 - z) * n + z * h


class GRUPolicy:
    """features -> GRU memory (updated every `stride` steps) -> head(memory, current features) each step."""

    def __init__(self, spec, mode, x_mean, x_scale, y_mean, y_scale, w_ih, w_hh, b_ih, b_hh,
                 w1, b1, w2, b2, stride_steps=1):
        validate_spec(spec)
        if type(stride_steps) is not int or stride_steps < 1:
            raise ValueError("stride_steps must be a positive integer")
        self.spec = spec; self.mode = mode; self.names = input_names(mode); self.stride = stride_steps
        if mode != "raw21" and spec["variant"] != mode:
            raise ValueError("Feature specification variant must match the input mode")
        f64 = lambda a: np.asarray(a, dtype=np.float64)
        self.x_mean, self.x_scale, self.y_mean, self.y_scale = f64(x_mean), f64(x_scale), f64(y_mean), f64(y_scale)
        self.w_ih, self.w_hh, self.b_ih, self.b_hh = f64(w_ih), f64(w_hh), f64(b_ih), f64(b_hh)
        self.w1, self.b1, self.w2, self.b2 = f64(w1), f64(b1), f64(w2), f64(b2)
        F = len(self.names); H = self.w_hh.shape[1]; W = self.w1.shape[0]
        if (self.x_mean.shape != (F,) or self.x_scale.shape != (F,) or self.y_mean.shape != (9,) or self.y_scale.shape != (9,)
                or self.w_ih.shape != (3 * H, F) or self.w_hh.shape != (3 * H, H) or self.b_ih.shape != (3 * H,)
                or self.b_hh.shape != (3 * H,) or self.w1.shape != (W, H + F) or self.b1.shape != (W,)
                or self.w2.shape != (9, W) or self.b2.shape != (9,)):
            raise ValueError("Invalid GRU policy shapes")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Scales must be positive")
        if not all(np.isfinite(v).all() for v in self.arrays().values() if v.dtype.kind == "f"):
            raise ValueError("Non-finite parameters")
        self.hidden = H
        self._h = np.zeros(H); self._count = 0; self._dt = spec["dt_s"]

    @property
    def input_features(self):
        return len(self.names)

    def arrays(self):
        return {"feature_mode": np.array(self.mode), "x_mean": self.x_mean, "x_scale": self.x_scale,
                "y_mean": self.y_mean, "y_scale": self.y_scale,
                "w_ih": self.w_ih, "w_hh": self.w_hh, "b_ih": self.b_ih, "b_hh": self.b_hh,
                "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2,
                "standoff_heights": np.asarray(self.spec["standoff_heights"], dtype=np.float64),
                "standoff_distances": np.asarray(self.spec["standoff_distances"], dtype=np.float64),
                "clock": np.array([self.spec["horizon_s"], self.spec["dt_s"], self.stride], dtype=np.float64)}

    def _head(self, h, f):
        hidden = np.maximum(self.w1 @ np.concatenate([h, f]) + self.b1, 0.0)
        return (self.w2 @ hidden + self.b2) * self.y_scale + self.y_mean

    def predict_features(self, features):
        """One episode's physics-step feature rows, in order, from a zero memory."""
        f = (np.asarray(features, dtype=np.float64) - self.x_mean) / self.x_scale
        h = np.zeros(self.hidden); out = np.empty((len(f), 9))
        for i, row in enumerate(f):
            if i % self.stride == 0:
                h = gru_step(row, h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
            out[i] = self._head(h, row)
        return out

    def predict(self, observations, times=None, lidar=None):
        x = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        t = np.arange(len(x)) * self.spec["dt_s"] if times is None else np.asarray(times, dtype=np.float64)
        return self.predict_features(features_for(self.mode, x, t, self.spec, lidar=lidar))

    def predict_action(self, observation, lidar=None):
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (21,):
            raise ValueError("Live inference takes one 21D observation")
        t = self._count * self._dt
        scan = None if lidar is None else np.asarray(lidar, dtype=np.float64)[None]
        f = (features_for(self.mode, x[None], np.array([t]), self.spec, lidar=scan)[0] - self.x_mean) / self.x_scale
        if self._count % self.stride == 0:
            self._h = gru_step(f, self._h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
        self._count += 1
        return expand_actions(self._head(self._h, f))

    def reset(self, dt=None):
        if dt is not None and not np.isclose(dt, self.spec["dt_s"], rtol=0, atol=1e-12):
            raise ValueError("Policy clock dt differs from simulator dt")
        self._h = np.zeros(self.hidden); self._count = 0

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            mode = str(d["feature_mode"])
            spec = {"variant": mode if mode != "raw21" else VARIANT, "standoff_heights": d["standoff_heights"].tolist(),
                    "standoff_distances": d["standoff_distances"].tolist(),
                    "horizon_s": float(d["clock"][0]), "dt_s": float(d["clock"][1])}
            return cls(spec, mode, d["x_mean"], d["x_scale"], d["y_mean"], d["y_scale"],
                       d["w_ih"], d["w_hh"], d["b_ih"], d["b_hh"], d["w1"], d["b1"], d["w2"], d["b2"],
                       stride_steps=int(d["clock"][2]))


def build_network(torch, n_features, hidden, head_width):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
            self.head1 = torch.nn.Linear(hidden + n_features, head_width)
            self.head2 = torch.nn.Linear(head_width, 9)

        def memory(self, x_strided, h=None):
            out, h = self.gru(x_strided, h)
            return out, h

        def react(self, h_rows, x_rows):
            return self.head2(torch.relu(self.head1(torch.cat([h_rows, x_rows], dim=-1))))

        def forward(self, x_rows, stride, h=None):
            """x_rows (B, T, F) physics rows; memory from rows 0, stride, 2*stride, ..."""
            out, h = self.memory(x_rows[:, ::stride], h)
            h_rows = out.repeat_interleave(stride, dim=1)[:, :x_rows.shape[1]]
            return self.react(h_rows, x_rows), h
    return Net()


def export_policy(network, spec, mode, stats, stride):
    g = network.gru
    cpu = lambda p: p.detach().cpu().numpy()
    return GRUPolicy(spec, mode, *stats, cpu(g.weight_ih_l0), cpu(g.weight_hh_l0), cpu(g.bias_ih_l0), cpu(g.bias_hh_l0),
                     cpu(network.head1.weight), cpu(network.head1.bias), cpu(network.head2.weight), cpu(network.head2.bias),
                     stride_steps=stride)


def episode_group(directory, records, spec, mode):
    """Per-episode feature and label arrays (float64) in record order, with their lengths."""
    feats, labels = [], []
    for record in records:
        if mode == "raw21":
            f, y, _ = load_episode(record_directory(record, directory), record)
        else:
            f, y = load_episode_features(directory, record, spec)
        feats.append(np.asarray(f, dtype=np.float64)); labels.append(np.asarray(y, dtype=np.float64))
    return feats, labels, np.array([len(f) for f in feats])


def _pad(arrays, length, transform, fill):
    out = np.empty((len(arrays), length, arrays[0].shape[1]), dtype=fill.dtype); out[...] = fill
    for i, a in enumerate(arrays):
        out[i, :len(a)] = transform(a)
    return out


def padded_group(directory, records, spec, mode):
    """Per-episode physics-step rows padded to the longest episode; returns lengths too."""
    feats, labels, lengths = episode_group(directory, records, spec, mode); T = int(lengths.max())
    same = lambda a: a
    return _pad(feats, T, same, np.zeros(feats[0].shape[1])), _pad(labels, T, same, np.zeros(9)), lengths


def group_statistics(feats, labels):
    """Mean and standard deviation of the valid rows (what `padded_group` under its mask gives); a zero scale becomes 1."""
    # Two passes over the episode list (mean, then squared deviations) in float64: no concatenated copy of the
    # features (42 GB for 57 M rows x 92), the same statistics up to summation order.
    def stats(arrays):
        n = sum(len(a) for a in arrays)
        mean = sum(a.sum(axis=0) for a in arrays) / n
        scale = np.sqrt(sum(((a - mean) ** 2).sum(axis=0) for a in arrays) / n); scale[scale == 0] = 1.0
        return mean, scale
    x_mean, x_scale = stats(feats); y_mean, y_scale = stats(labels)
    return x_mean, x_scale, y_mean, y_scale



TRIGGER_JITTER_SEED = 0  # one fixed rng for the trigger-step jitter (same draws for every training seed)


def sequence_trigger_rows(Y, L, y_mean, y_scale):
    """Per episode, the index of its trigger row (first row of the terminal run of exactly-zero wheel labels, found in
    normalised units), or -1 when the episode does not end with zero wheels. (n,) int64."""
    n = Y.shape[0]
    zero = ((0.0 - np.asarray(y_mean)[:2]) / np.asarray(y_scale)[:2]).astype(np.float32)
    rows = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if L[i] == 0:
            continue
        wheels = np.asarray(Y[i, :L[i], :2]); z = np.all(np.abs(wheels - zero) <= 1e-5, axis=1)
        if not z[-1]:
            continue
        k = L[i] - 1
        while k > 0 and z[k - 1]:
            k -= 1
        rows[i] = k
    return rows


def sequence_trigger_weights(Y, L, y_mean, y_scale, weight):
    """(n, T) float32 loss weights for padded, normalised label sequences: 1 on valid rows, 0 on padding, `weight` on each
    episode's trigger row (`sequence_trigger_rows`). Returns (weights, number of weighted rows)."""
    n, T = Y.shape[:2]
    w = (np.arange(T)[None, :] < np.asarray(L)[:, None]).astype(np.float32)
    count = 0
    if weight != 1.0:
        for i, k in enumerate(sequence_trigger_rows(Y, L, y_mean, y_scale)):
            if k >= 0:
                w[i, k] = weight; count += 1
    return w, count


def trigger_jitter_sigma(X, Y, L, names, x_scale, y_mean, y_scale, jitter):
    """Per-joint N(0, sigma) scale, in normalised feature units, for the seven arm-velocity features at the trigger step.
    `jitter` = "auto" (std of the recorded arm qd over the rows whose demonstrated wheels are not both zero, as the MLP
    trainer's trigger copies) or a value in rad/s. Returns (sigma_normalised (7,), sigma_rad_s (7,), qd_cols)."""
    qd_cols = [names.index(f"arm_qd_{i}") for i in range(1, 8)]
    zero = ((0.0 - np.asarray(y_mean)[:2]) / np.asarray(y_scale)[:2]).astype(np.float32)
    if jitter == "auto":
        sq = np.zeros(7); s1 = np.zeros(7); m = 0
        for i in range(X.shape[0]):
            moving = ~np.all(np.abs(np.asarray(Y[i, :L[i], :2]) - zero) <= 1e-5, axis=1)
            v = np.asarray(X[i, :L[i]][moving][:, qd_cols], dtype=np.float64) * np.asarray(x_scale)[qd_cols]
            sq += (v ** 2).sum(axis=0); s1 += v.sum(axis=0); m += len(v)
        sigma_rad = np.sqrt(np.maximum(sq / max(m, 1) - (s1 / max(m, 1)) ** 2, 0.0))
    else:
        sigma_rad = np.full(7, float(jitter))
    return (sigma_rad / np.asarray(x_scale)[qd_cols]).astype(np.float32), sigma_rad, qd_cols


def normalised_group(directory, records, spec, mode, stats=None):
    """`padded_group` normalised and cast to float32 one episode at a time, so the padded float64 arrays and their
    normalised copies never exist (4914 episodes padded to 28 s cost ~110 GB that way and ~20 GB this way).
    For given statistics it is element for element identical to ((padded - mean) / scale).astype(float32), padding
    rows included; the statistics computed here are those of the same valid rows (summation order can differ from the
    masked padded array by an ulp). Returns the normalised inputs and labels, the lengths, the statistics (from the
    valid rows when not given) and the raw per-episode labels for scoring."""
    feats, labels, lengths = episode_group(directory, records, spec, mode); T = int(lengths.max())
    stats = group_statistics(feats, labels) if stats is None else stats
    x_mean, x_scale, y_mean, y_scale = stats
    nx = lambda a: ((a - x_mean) / x_scale).astype(np.float32); ny = lambda a: ((a - y_mean) / y_scale).astype(np.float32)
    X = _pad(feats, T, nx, nx(np.zeros((1, len(x_mean))))[0]); del feats
    Y = _pad(labels, T, ny, ny(np.zeros((1, 9)))[0])
    return X, Y, lengths, stats, labels


def train(dataset, split_path, config_path, output, horizon_s=40.0, device_name="cuda",
          hidden=128, chunk_steps=1000, episodes_per_batch=8, mode="egocentric_clock_v1", stride=1,
          head_width=128, validation_episodes_per_batch=8, trigger_weight=1.0, trigger_jitter=0.0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    names = input_names(mode)
    if type(stride) is not int or stride < 1 or type(chunk_steps) is not int or chunk_steps < 1:
        raise ValueError("stride and chunk_steps must be positive integers")
    dataset = Path(dataset).resolve(); output = Path(output).resolve(); split_path = Path(split_path).resolve()
    config = read_json(config_path); hp = config["mlp"]
    saved = read_json(split_path)
    if Path(saved["dataset"]).resolve() != dataset:
        raise ValueError("Split belongs to another dataset")
    records = inventory(dataset) + extra_inventory(saved)
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    keep_failures = bool(config.get("keep_expert_failures", True)); episode_filter = {"keep_expert_failures": keep_failures}
    if not keep_failures:
        split, counts = drop_expert_failures(split); episode_filter.update(counts)
    if not isinstance(trigger_weight, (int, float)) or isinstance(trigger_weight, bool) or not math.isfinite(trigger_weight) or trigger_weight < 1:
        raise ValueError("trigger_weight must be a finite number >= 1")
    trigger_weight = float(trigger_weight)
    if trigger_jitter != "auto" and (not isinstance(trigger_jitter, (int, float)) or isinstance(trigger_jitter, bool)
                                     or not math.isfinite(trigger_jitter) or trigger_jitter < 0):
        raise ValueError('trigger_jitter must be "auto" or a finite number >= 0 (rad/s)')
    if trigger_jitter != "auto":
        trigger_jitter = float(trigger_jitter)
    source_cfg = read_json(dataset / "config.json")
    episode_cfg = resolve_episode_config({k: v for k, v in source_cfg.items() if k in episode_defaults()})
    dt = read_json(dataset / "episodes" / f"{records[0]['episode_id']}.json")["dt"]
    spec = spec_from_episode_config(episode_cfg, dt, horizon_s, mode if mode != "raw21" else VARIANT)
    device = torch.device(device_name); torch.manual_seed(hp["seed"])
    output.mkdir(parents=True, exist_ok=False)
    chunk_rows = chunk_steps * stride  # physics rows per TBPTT chunk
    write_json(output / "config.json", {**config, "input_features": mode, "clock_horizon_s": horizon_s,
                                        "recurrent": {"cell": "gru", "hidden": hidden, "layers": 1, "head_width": head_width,
                                                      "memory_stride_steps": stride, "reaction_period_steps": 1,
                                                      "tbptt_chunk_decisions": chunk_steps, "tbptt_chunk_rows": chunk_rows,
                                                      "episodes_per_batch": episodes_per_batch,
                                                      "state_reset": "every episode; carried and detached across chunks"}})
    shutil.copy2(split_path, output / "split.json")
    source_paths = sorted((PROJECT_ROOT / "src/robot_learning/learning").glob("*.py"))
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths}
    for p in source_paths:
        dest = output / "source_snapshot" / p.relative_to(PROJECT_ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = {"status": "running", "created_at": utc_now(), "dataset": str(dataset),
                "dataset_hashes": {n: sha256(dataset / n) for n in
                                   ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]},
                "source_hashes": source_hashes, "project": git_info(PROJECT_ROOT),
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch"]},
                "python": sys.version, "command": sys.argv, "input_names": names,
                "raw_observation_names": INPUT_NAMES, "output_names": OUTPUT_NAMES, "feature_spec": spec,
                "input_mode": mode, "backend": "pytorch", "test_set": "none; development train/validation only",
                "closed_loop_evaluation": "not run by the trainer"}
    write_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        X, Y, L, stats, _ = normalised_group(dataset, split["training"], spec, mode)
        x_mean, x_scale, y_mean, y_scale = stats
        mask = np.arange(X.shape[1])[None, :] < L[:, None]
        weights, trigger_rows_weighted = sequence_trigger_weights(Y, L, y_mean, y_scale, trigger_weight)
        tw = torch.from_numpy(weights).to(device); del weights
        jitter_on = trigger_jitter == "auto" or trigger_jitter > 0.0
        trig_rows = sequence_trigger_rows(Y, L, y_mean, y_scale) if jitter_on else None
        if jitter_on:
            sigma_n, sigma_rad, qd_cols = trigger_jitter_sigma(X, Y, L, names, x_scale, y_mean, y_scale, trigger_jitter)
            tsigma = torch.from_numpy(sigma_n).to(device); tqd = torch.tensor(qd_cols, device=device)
            jitter_rng = np.random.default_rng(TRIGGER_JITTER_SEED); jitter_rows_total = 0
        tx = torch.from_numpy(X).to(device); ty = torch.from_numpy(Y).to(device); tmask = torch.from_numpy(mask).to(device)
        del X, Y
        VX, _, VL, _, VYS = normalised_group(dataset, split["validation"], spec, mode, stats)  # training statistics
        tvx = torch.from_numpy(VX).to(device); T_val = VX.shape[1]; del VX
        network = build_network(torch, len(names), hidden, head_width).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=hp["learning_rate"], betas=(.9, .999), eps=1e-8)
        rng = np.random.RandomState(hp["seed"]); n = len(tx); T = tx.shape[1]

        def validation_score():
            network.eval()
            with torch.no_grad():
                preds = []
                for s in range(0, len(tvx), validation_episodes_per_batch):
                    out, _ = network(tvx[s:s + validation_episodes_per_batch], stride)
                    preds.append(out.double().cpu().numpy())
            network.train()
            pred = np.concatenate(preds) * y_scale + y_mean
            per = [errors(VYS[i], pred[i, :VL[i]], y_scale) for i in range(len(VL))]
            return float(np.mean([e["balanced_nrmse"] for e in per])), pred, per

        best = math.inf; patience_score = math.inf; stale = 0; best_epoch = 0; history = []; epoch = 0; updates = 0
        while stale < hp["patience"] and (hp["max_epochs"] is None or epoch < hp["max_epochs"]):
            epoch += 1; t0 = time.perf_counter(); order = rng.permutation(n); total = 0.0; rows = 0
            for s in range(0, n, episodes_per_batch):
                idx = torch.from_numpy(order[s:s + episodes_per_batch]).to(device)
                h = None
                for c in range(0, T, chunk_rows):
                    bm = tmask[idx, c:c + chunk_rows]
                    if not bool(bm.any()):
                        break
                    bx = tx[idx, c:c + chunk_rows]; by = ty[idx, c:c + chunk_rows]
                    if jitter_on:  # fresh N(0, sigma) on the arm velocities of each episode's trigger step, this chunk only
                        rows_np = trig_rows[order[s:s + episodes_per_batch]]
                        for j in np.flatnonzero((rows_np >= c) & (rows_np < c + chunk_rows)):
                            noise = torch.from_numpy(jitter_rng.normal(0.0, 1.0, 7).astype(np.float32)).to(device)
                            bx[j, rows_np[j] - c, tqd] += noise * tsigma; jitter_rows_total += 1
                    out, h = network(bx, stride, h)
                    count = bm.sum()
                    bw = tw[idx, c:c + chunk_rows]
                    squared = ((out - by).square().sum(dim=2) * bw).sum() / count
                    objective = .5 * squared + (.5 * hp["alpha"] / max(1, int(count))) * sum(
                        p.square().sum() for name, p in network.named_parameters() if "weight" in name)
                    optimizer.zero_grad(set_to_none=True); objective.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                    optimizer.step(); updates += 1; h = h.detach()
                    total += float(squared.detach()) * int(count); rows += int(count)
            score, _, _ = validation_score()
            if not math.isfinite(score):
                raise ValueError("Non-finite validation score")
            if score < best:
                best = score; best_epoch = epoch; export_policy(network, spec, mode, stats, stride).save(output / "model.npz")
            if score < patience_score - hp["min_delta"]:
                patience_score = score; stale = 0
            else:
                stale += 1
            history.append({"epoch": epoch, "training_normalized_loss": .5 * total / max(rows, 1) / 9,
                            "validation_episode_macro_balanced_nrmse": score, "seconds": time.perf_counter() - t0,
                            "updates": updates})
            write_json(output / "history.json", history)
            print(f"Epoch {epoch}: loss={history[-1]['training_normalized_loss']:.6f}, validation NRMSE={score:.6f}, best={best:.6f}", flush=True)
        policy = GRUPolicy.load(output / "model.npz")
        # NumPy inference must reproduce the torch network (compared in float64) on a real validation episode.
        check = build_network(torch, len(names), hidden, head_width).double()
        with torch.no_grad():
            check.gru.weight_ih_l0.copy_(torch.from_numpy(policy.w_ih)); check.gru.weight_hh_l0.copy_(torch.from_numpy(policy.w_hh))
            check.gru.bias_ih_l0.copy_(torch.from_numpy(policy.b_ih)); check.gru.bias_hh_l0.copy_(torch.from_numpy(policy.b_hh))
            check.head1.weight.copy_(torch.from_numpy(policy.w1)); check.head1.bias.copy_(torch.from_numpy(policy.b1))
            check.head2.weight.copy_(torch.from_numpy(policy.w2)); check.head2.bias.copy_(torch.from_numpy(policy.b2))
            v0 = episode_group(dataset, split["validation"][:1], spec, mode)[0][0][:400]
            ref, _ = check(torch.from_numpy(((v0 - x_mean) / x_scale)[None]), stride)
        ref = ref[0].numpy() * y_scale + y_mean
        np.testing.assert_allclose(policy.predict_features(v0), ref, rtol=1e-8, atol=1e-8)
        raw, raw_lidar = x_check(dataset, split["validation"][0], 400)
        policy.reset(dt); live = np.stack([policy.predict_action(o, lidar=None if raw_lidar is None else raw_lidar[i]) for i, o in enumerate(raw)]); policy.reset()
        np.testing.assert_allclose(live, expand_actions(policy.predict(raw, lidar=raw_lidar)), rtol=1e-8, atol=1e-8)
        score, pred, per = validation_score()
        vmask = np.arange(T_val)[None, :] < VL[:, None]
        metrics = {"validation": {"model": aggregate(per),
                                  "mean_baseline": aggregate([errors(VYS[i], np.broadcast_to(y_mean, (VL[i], 9)), y_scale) for i in range(len(VL))]),
                                  "pooled_transitions": {"model": errors(np.concatenate(VYS), pred[vmask], y_scale)}}}
        timings = []; obs, obs_lidar = raw[0], (None if raw_lidar is None else raw_lidar[0]); policy.reset(dt)
        for _ in range(100): policy.predict_action(obs, lidar=obs_lidar)
        for _ in range(2000):
            t1 = time.perf_counter_ns(); policy.predict_action(obs, lidar=obs_lidar); timings.append(time.perf_counter_ns() - t1)
        policy.reset()
        if any(sha256(PROJECT_ROOT / p) != d for p, d in source_hashes.items()):
            raise ValueError("Training source changed during execution")
        info = {"kind": KIND, "variant": mode, "input_mode": mode, "backend": "pytorch", "training_device": str(device),
                "cell": "gru", "hidden": hidden, "layers": 1, "head_width": head_width,
                "memory_stride_steps": stride, "reaction_period_steps": 1,
                "tbptt_chunk_decisions": chunk_steps, "episodes_per_batch": episodes_per_batch, "gradient_clip_norm": 10.0,
                "episode_filter": episode_filter, "training_episodes": len(split["training"]), "validation_episodes": len(split["validation"]),
                "trigger_weight": {"weight": trigger_weight, "rows_weighted": trigger_rows_weighted} if trigger_weight != 1.0 else None,
                "trigger_jitter": ({"rule": trigger_jitter, "qd_sigma_rad_s": [float(v) for v in sigma_rad], "seed": TRIGGER_JITTER_SEED,
                                    "rows_jittered_total": int(jitter_rows_total), "resampled": "every pass over the trigger step",
                                    "applies_to": "the seven arm-velocity input features at each episode's trigger step; labels untouched"}
                                   if jitter_on else None),
                "regression_parameters": sum(p.numel() for p in network.parameters()), "weight_dtype": "float32",
                "input_features": len(names), "input_names": names, "independent_outputs": 9, "feature_spec": spec,
                "control_period": f"head reacts every physics step (1 ms); GRU memory updated every {stride} step(s); state reset per episode",
                "epochs_completed": epoch, "best_epoch": best_epoch, "training_updates": updates,
                "best_validation_episode_macro_balanced_nrmse": best,
                "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
                "checkpoint_selection": "minimum validation episode-macro balanced NRMSE over all physics steps",
                "training_loss": "masked half output-sum MSE per chunk with weight L2; Adam; grad-norm clip 10",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/mlp_pytorch.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-s", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--chunk-steps", type=int, default=1000, help="TBPTT chunk length in memory updates")
    parser.add_argument("--episodes-per-batch", type=int, default=8)
    parser.add_argument("--inputs", default="egocentric_clock_v1", choices=sorted(MODES),
                        help="raw21: the assignment's explicit interface; otherwise an input variant of egocentric.py "
                             "(egocentric_clock_v1, raw21_clock_v1, raw21_egocentric_clock_v1)")
    parser.add_argument("--trigger-weight", type=float, default=1.0, help="Loss weight of each episode's trigger row (first terminal zero-wheel row)")
    parser.add_argument("--trigger-jitter", default="0", help='N(0, sigma) on the arm-velocity inputs at the trigger step: "auto" (std of arm qd on moving rows) or rad/s; 0 = off')
    parser.add_argument("--stride", type=int, default=1, help="Memory update period in physics steps; the head reacts every step")
    args = parser.parse_args(argv)
    _, metrics = train(args.dataset, args.split, args.config, args.output_dir, args.horizon_s, args.device,
                       args.hidden, args.chunk_steps, args.episodes_per_batch, args.inputs, args.stride, args.head_width,
                       trigger_weight=args.trigger_weight,
                       trigger_jitter="auto" if args.trigger_jitter == "auto" else float(args.trigger_jitter))
    print(f"validation: episode-macro balanced NRMSE={metrics['validation']['model']['episode_macro']['balanced_nrmse']:.6f}")
    return 0
