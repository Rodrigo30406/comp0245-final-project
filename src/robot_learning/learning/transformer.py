"""Causal transformer policy: attention over the episode's own past input states, fast reaction.

Same two-rate structure as `gru_policy.py` so the comparison is clean: every `stride` physics steps the
current features become a token and a causal transformer attends over the last `window` tokens (20 s at
stride 20, window 1000); the output head reacts at EVERY physics step from the latest memory vector and the
current features. No positional encoding: order comes from the causal mask and time from the clock input,
so nothing extrapolates when evaluation episodes (40 s) outlast the demonstrations (<= 29 s).

Training uses whole padded episodes on the GPU in one forward pass (the window bounds the receptive field,
no truncated backpropagation is needed). Inference is NumPy with a per-layer key/value cache; the live path
is verified against the torch network by test and at the end of every training run.
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
from robot_learning.data.splits import drop_expert_failures, extra_inventory, inventory, resolve_dataset, validate_partition
from .episodes import first_rows as x_check
from .features import EGO, INPUT_NAMES, spec_from_episode_config, validate_spec
from .gru import MODES, input_names, features_for, normalised_group, episode_group
from .metrics import errors, aggregate
from .outputs import OUTPUT_NAMES, expand_actions

KIND = "transformer_regressor"
LN_EPS = 1e-5


def layer_norm(x, g, b):
    mean = x.mean(axis=-1, keepdims=True); var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + LN_EPS) * g + b


class TransformerPolicy:
    """features -> tokens every `stride` steps -> causal windowed attention -> head(memory, current features) each step."""

    def __init__(self, spec, mode, x_mean, x_scale, y_mean, y_scale, params, stride_steps=1, window=1000, heads=4):
        validate_spec(spec)
        if type(stride_steps) is not int or stride_steps < 1 or type(window) is not int or window < 1 or type(heads) is not int or heads < 1:
            raise ValueError("stride_steps, window and heads must be positive integers")
        self.spec = spec; self.mode = mode; self.names = input_names(mode); self.stride = stride_steps
        self.window = window; self.heads = heads
        if spec["variant"] != mode:
            raise ValueError("Feature specification variant must match the input mode")
        f64 = lambda a: np.asarray(a, dtype=np.float64)
        self.x_mean, self.x_scale, self.y_mean, self.y_scale = f64(x_mean), f64(x_scale), f64(y_mean), f64(y_scale)
        self.p = {k: f64(v) for k, v in params.items()}
        F = len(self.names); d = self.p["embed_w"].shape[0]; W = self.p["head1_w"].shape[0]
        self.layers = sum(1 for k in self.p if k.endswith("_wq")); self.d = d
        if d % heads:
            raise ValueError("d_model must be a multiple of heads")
        ff = self.p["l0_ff1_w"].shape[0] if self.layers else None
        expected = {"embed_w": (d, F), "embed_b": (d,), "lnf_g": (d,), "lnf_b": (d,), "head1_w": (W, d + F), "head1_b": (W,),
                    "head2_w": (9, W), "head2_b": (9,)}
        for i in range(self.layers):
            expected.update({f"l{i}_ln1_g": (d,), f"l{i}_ln1_b": (d,), f"l{i}_ln2_g": (d,), f"l{i}_ln2_b": (d,),
                             f"l{i}_ff1_w": (ff, d), f"l{i}_ff1_b": (ff,), f"l{i}_ff2_w": (d, ff), f"l{i}_ff2_b": (d,)})
            for n in "qkvo":
                expected[f"l{i}_w{n}"] = (d, d); expected[f"l{i}_b{n}"] = (d,)
        if set(expected) != set(self.p) or any(self.p[k].shape != s for k, s in expected.items()):
            raise ValueError("Invalid transformer policy parameters")
        if (self.x_mean.shape != (F,) or self.x_scale.shape != (F,) or self.y_mean.shape != (9,) or self.y_scale.shape != (9,)
                or np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0)):
            raise ValueError("Invalid normalisation")
        if not all(np.isfinite(v).all() for v in self.p.values()):
            raise ValueError("Non-finite parameters")
        self._dt = spec["dt_s"]; self.reset()

    @property
    def input_features(self):
        return len(self.names)

    def arrays(self):
        return {"feature_mode": np.array(self.mode), "x_mean": self.x_mean, "x_scale": self.x_scale,
                "y_mean": self.y_mean, "y_scale": self.y_scale, **self.p,
                "standoff_heights": np.asarray(self.spec["standoff_heights"], dtype=np.float64),
                "standoff_distances": np.asarray(self.spec["standoff_distances"], dtype=np.float64),
                "clock": np.array([self.spec["horizon_s"], self.spec["dt_s"], self.stride, self.window, self.heads], dtype=np.float64)}

    def _token(self, f):
        """Append one token (normalised features) to the memory; returns the memory vector for it."""
        p = self.p; H = self.heads; dh = self.d // H
        h = p["embed_w"] @ f + p["embed_b"]
        for i in range(self.layers):
            u = layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
            q = p[f"l{i}_wq"] @ u + p[f"l{i}_bq"]; k = p[f"l{i}_wk"] @ u + p[f"l{i}_bk"]; v = p[f"l{i}_wv"] @ u + p[f"l{i}_bv"]
            if self._n == self.window:
                self._k[i][:-1] = self._k[i][1:]; self._v[i][:-1] = self._v[i][1:]; pos = self.window - 1
            else:
                pos = self._n
            self._k[i][pos] = k; self._v[i][pos] = v
            K = self._k[i][:pos + 1].reshape(pos + 1, H, dh); V = self._v[i][:pos + 1].reshape(pos + 1, H, dh)
            scores = np.einsum("hd,nhd->hn", q.reshape(H, dh), K) / math.sqrt(dh)
            scores -= scores.max(axis=1, keepdims=True); w = np.exp(scores); w /= w.sum(axis=1, keepdims=True)
            att = np.einsum("hn,nhd->hd", w, V).reshape(self.d)
            h = h + p[f"l{i}_wo"] @ att + p[f"l{i}_bo"]
            u = layer_norm(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
            h = h + p[f"l{i}_ff2_w"] @ np.maximum(p[f"l{i}_ff1_w"] @ u + p[f"l{i}_ff1_b"], 0.0) + p[f"l{i}_ff2_b"]
        self._n = min(self._n + 1, self.window)
        return layer_norm(h, p["lnf_g"], p["lnf_b"])

    def _head(self, m, f):
        hidden = np.maximum(self.p["head1_w"] @ np.concatenate([m, f]) + self.p["head1_b"], 0.0)
        return (self.p["head2_w"] @ hidden + self.p["head2_b"]) * self.y_scale + self.y_mean

    def predict_features(self, features):
        """One episode's physics-step feature rows, in order, from an empty memory (the live state is left untouched)."""
        saved = (self._k, self._v, self._n, self._m, self._count); self.reset()
        f = (np.asarray(features, dtype=np.float64) - self.x_mean) / self.x_scale
        out = np.empty((len(f), 9))
        for i, row in enumerate(f):
            if i % self.stride == 0:
                self._m = self._token(row)
            out[i] = self._head(self._m, row)
        self._k, self._v, self._n, self._m, self._count = saved
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
            self._m = self._token(f)
        self._count += 1
        return expand_actions(self._head(self._m, f))

    def reset(self, dt=None):
        if dt is not None and not np.isclose(dt, self.spec["dt_s"], rtol=0, atol=1e-12):
            raise ValueError("Policy clock dt differs from simulator dt")
        self._k = [np.zeros((self.window, self.d)) for _ in range(self.layers)]
        self._v = [np.zeros((self.window, self.d)) for _ in range(self.layers)]
        self._n = 0; self._m = np.zeros(self.d); self._count = 0

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            mode = str(d["feature_mode"])
            spec = {"variant": mode, "standoff_heights": d["standoff_heights"].tolist(),
                    "standoff_distances": d["standoff_distances"].tolist(),
                    "horizon_s": float(d["clock"][0]), "dt_s": float(d["clock"][1])}
            skip = {"feature_mode", "x_mean", "x_scale", "y_mean", "y_scale", "standoff_heights", "standoff_distances", "clock"}
            params = {k: d[k] for k in d.files if k not in skip}
            return cls(spec, mode, d["x_mean"], d["x_scale"], d["y_mean"], d["y_scale"], params,
                       stride_steps=int(d["clock"][2]), window=int(d["clock"][3]), heads=int(d["clock"][4]))


def build_network(torch, n_features, d_model, layers, heads, ffn, head_width, window):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model, eps=LN_EPS); self.ln2 = nn.LayerNorm(d_model, eps=LN_EPS)
            self.wq = nn.Linear(d_model, d_model); self.wk = nn.Linear(d_model, d_model)
            self.wv = nn.Linear(d_model, d_model); self.wo = nn.Linear(d_model, d_model)
            self.ff1 = nn.Linear(d_model, ffn); self.ff2 = nn.Linear(ffn, d_model)

        def forward(self, h, allowed):
            B, T, _ = h.shape; dh = d_model // heads
            u = self.ln1(h)
            q = self.wq(u).view(B, T, heads, dh).transpose(1, 2); k = self.wk(u).view(B, T, heads, dh).transpose(1, 2)
            v = self.wv(u).view(B, T, heads, dh).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            scores = scores.masked_fill(~allowed[:T, :T], float("-inf"))
            att = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, d_model)
            h = h + self.wo(att)
            return h + self.ff2(torch.relu(self.ff1(self.ln2(h))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(n_features, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.lnf = nn.LayerNorm(d_model, eps=LN_EPS)
            self.head1 = nn.Linear(d_model + n_features, head_width); self.head2 = nn.Linear(head_width, 9)
            self.window = window; self.heads = heads

        def allowed(self, T, device):
            i = torch.arange(T, device=device)
            return (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)

        def memory(self, tokens):
            h = self.embed(tokens); allowed = self.allowed(tokens.shape[1], tokens.device)
            for block in self.blocks:
                h = block(h, allowed)
            return self.lnf(h)

        def react(self, m_rows, x_rows):
            return self.head2(torch.relu(self.head1(torch.cat([m_rows, x_rows], dim=-1))))

        def forward(self, x_rows, stride):
            """x_rows (B, T, F) physics rows; tokens from rows 0, stride, 2*stride, ..."""
            m = self.memory(x_rows[:, ::stride])
            return self.react(m.repeat_interleave(stride, dim=1)[:, :x_rows.shape[1]], x_rows)
    return Net()


def network_params(network):
    cpu = lambda p: p.detach().cpu().numpy().astype(np.float64)
    params = {"embed_w": cpu(network.embed.weight), "embed_b": cpu(network.embed.bias),
              "lnf_g": cpu(network.lnf.weight), "lnf_b": cpu(network.lnf.bias),
              "head1_w": cpu(network.head1.weight), "head1_b": cpu(network.head1.bias),
              "head2_w": cpu(network.head2.weight), "head2_b": cpu(network.head2.bias)}
    for i, b in enumerate(network.blocks):
        params.update({f"l{i}_ln1_g": cpu(b.ln1.weight), f"l{i}_ln1_b": cpu(b.ln1.bias), f"l{i}_ln2_g": cpu(b.ln2.weight), f"l{i}_ln2_b": cpu(b.ln2.bias),
                       f"l{i}_ff1_w": cpu(b.ff1.weight), f"l{i}_ff1_b": cpu(b.ff1.bias), f"l{i}_ff2_w": cpu(b.ff2.weight), f"l{i}_ff2_b": cpu(b.ff2.bias)})
        for n in "qkvo":
            lin = getattr(b, f"w{n}"); params[f"l{i}_w{n}"] = cpu(lin.weight); params[f"l{i}_b{n}"] = cpu(lin.bias)
    return params


def load_params(torch, network, params):
    with torch.no_grad():
        network.embed.weight.copy_(torch.from_numpy(params["embed_w"])); network.embed.bias.copy_(torch.from_numpy(params["embed_b"]))
        network.lnf.weight.copy_(torch.from_numpy(params["lnf_g"])); network.lnf.bias.copy_(torch.from_numpy(params["lnf_b"]))
        network.head1.weight.copy_(torch.from_numpy(params["head1_w"])); network.head1.bias.copy_(torch.from_numpy(params["head1_b"]))
        network.head2.weight.copy_(torch.from_numpy(params["head2_w"])); network.head2.bias.copy_(torch.from_numpy(params["head2_b"]))
        for i, b in enumerate(network.blocks):
            for name, mod in [("ln1", b.ln1), ("ln2", b.ln2)]:
                mod.weight.copy_(torch.from_numpy(params[f"l{i}_{name}_g"])); mod.bias.copy_(torch.from_numpy(params[f"l{i}_{name}_b"]))
            for name, mod in [("ff1", b.ff1), ("ff2", b.ff2)]:
                mod.weight.copy_(torch.from_numpy(params[f"l{i}_{name}_w"])); mod.bias.copy_(torch.from_numpy(params[f"l{i}_{name}_b"]))
            for n in "qkvo":
                lin = getattr(b, f"w{n}"); lin.weight.copy_(torch.from_numpy(params[f"l{i}_w{n}"])); lin.bias.copy_(torch.from_numpy(params[f"l{i}_b{n}"]))


def export_policy(network, spec, mode, stats, stride, window, heads):
    return TransformerPolicy(spec, mode, *stats, network_params(network), stride_steps=stride, window=window, heads=heads)


def train(dataset, split_path, config_path, output, horizon_s=40.0, device_name="cuda", mode=EGO,
          stride=20, window=1000, d_model=64, layers=2, heads=4, ffn=128, head_width=128, episodes_per_batch=8,
          validation_episodes_per_batch=8, seed=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    names = input_names(mode)
    for v in (stride, window, d_model, layers, heads, ffn, head_width, episodes_per_batch):
        if type(v) is not int or v < 1:
            raise ValueError("Architecture arguments must be positive integers")
    dataset = Path(dataset).resolve(); output = Path(output).resolve(); split_path = Path(split_path).resolve()
    config = read_json(config_path)
    if seed is not None:
        config["training"]["seed"] = int(seed)
    hp = config["training"]
    saved = read_json(split_path)
    if resolve_dataset(saved["dataset"]) != dataset:
        raise ValueError("Split belongs to another dataset")
    records = inventory(dataset) + extra_inventory(saved)
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    keep_failures = bool(config.get("keep_expert_failures", False)); episode_filter = {"keep_expert_failures": keep_failures}
    if not keep_failures:  # rule of 2 Nov: train and select on expert successes only (same as the MLP and GRU trainers)
        split, counts = drop_expert_failures(split); episode_filter.update(counts)
    source_cfg = read_json(dataset / "config.json")
    episode_cfg = resolve_episode_config({k: v for k, v in source_cfg.items() if k in episode_defaults()})
    dt = read_json(dataset / "episodes" / f"{records[0]['episode_id']}.json")["dt"]
    spec = spec_from_episode_config(episode_cfg, dt, horizon_s, mode)
    device = torch.device(device_name); torch.manual_seed(hp["seed"])
    output.mkdir(parents=True, exist_ok=False)
    arch = {"cell": "causal_transformer", "d_model": d_model, "layers": layers, "heads": heads, "ffn": ffn, "head_width": head_width,
            "memory_stride_steps": stride, "reaction_period_steps": 1, "window_tokens": window, "window_s": window * stride * dt,
            "positional_encoding": "none (causal mask orders tokens, the clock input carries time)",
            "episodes_per_batch": episodes_per_batch, "state_reset": "every episode"}
    write_json(output / "config.json", {**config, "input_features": mode, "clock_horizon_s": horizon_s, "attention": arch})
    shutil.copy2(split_path, output / "split.json")
    source_paths = sorted((PROJECT_ROOT / "src/robot_learning/learning").glob("*.py"))
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths}
    for p in source_paths:
        dest = output / "source_snapshot" / p.relative_to(PROJECT_ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = {"status": "running", "created_at": utc_now(), "dataset": str(dataset),
                "dataset_hashes": {n: sha256(dataset / n) for n in ["dataset.json", "config.json", "scenarios.json", "schema.json", "provenance.json"]},
                "source_hashes": source_hashes, "project": git_info(PROJECT_ROOT),
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch"]},
                "python": sys.version, "command": sys.argv, "input_names": names,
                "raw_observation_names": INPUT_NAMES, "output_names": OUTPUT_NAMES, "feature_spec": spec,
                "input_mode": mode, "backend": "pytorch", "test_set": "none; development train/validation only",
                "closed_loop_evaluation": "not run by the trainer"}
    write_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        # Normalised float32 episodes padded one at a time (gru_policy.normalised_group): the padded float64 arrays and
        # their normalised copies never exist (1700 obstacle episodes x 29 021 steps x 92 inputs cost 36 GB that way).
        X, Y, L, stats, _ = normalised_group(dataset, split["training"], spec, mode)
        x_mean, x_scale, y_mean, y_scale = stats
        mask = np.arange(X.shape[1])[None, :] < L[:, None]
        tx = torch.from_numpy(X).to(device); ty = torch.from_numpy(Y).to(device); tmask = torch.from_numpy(mask).to(device)
        del X, Y
        VX, _, VL, _, VYS = normalised_group(dataset, split["validation"], spec, mode, stats)  # training statistics
        tvx = torch.from_numpy(VX).to(device); T_val = VX.shape[1]; del VX
        network = build_network(torch, len(names), d_model, layers, heads, ffn, head_width, window).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=hp["learning_rate"], betas=(.9, .999), eps=1e-8)
        rng = np.random.RandomState(hp["seed"]); n = len(tx)

        def validation_score():
            network.eval()
            with torch.no_grad():
                preds = [network(tvx[s:s + validation_episodes_per_batch], stride).double().cpu().numpy()
                         for s in range(0, len(tvx), validation_episodes_per_batch)]
            network.train()
            pred = np.concatenate(preds) * y_scale + y_mean
            per = [errors(VYS[i], pred[i, :VL[i]], y_scale) for i in range(len(VL))]
            return float(np.mean([e["balanced_nrmse"] for e in per])), pred, per

        best = math.inf; patience_score = math.inf; stale = 0; best_epoch = 0; history = []; epoch = 0; updates = 0
        while stale < hp["patience"] and (hp["max_epochs"] is None or epoch < hp["max_epochs"]):
            epoch += 1; t0 = time.perf_counter(); order = rng.permutation(n); total = 0.0; rows = 0
            for s in range(0, n, episodes_per_batch):
                idx = torch.from_numpy(order[s:s + episodes_per_batch]).to(device)
                bm = tmask[idx]; T = int(bm.sum(dim=1).max())  # trim the batch to its longest episode
                bx = tx[idx, :T]; by = ty[idx, :T]; bm = bm[:, :T]
                out = network(bx, stride)
                count = bm.sum()
                squared = ((out - by).square().sum(dim=2) * bm).sum() / count
                objective = .5 * squared + (.5 * hp["alpha"] / max(1, int(count))) * sum(
                    p.square().sum() for name, p in network.named_parameters() if "weight" in name and "ln" not in name)
                optimizer.zero_grad(set_to_none=True); objective.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                optimizer.step(); updates += 1
                total += float(squared.detach()) * int(count); rows += int(count)
            score, _, _ = validation_score()
            if not math.isfinite(score):
                raise ValueError("Non-finite validation score")
            if score < best:
                best = score; best_epoch = epoch; export_policy(network, spec, mode, stats, stride, window, heads).save(output / "model.npz")
            if score < patience_score - hp["min_delta"]:
                patience_score = score; stale = 0
            else:
                stale += 1
            history.append({"epoch": epoch, "training_normalized_loss": .5 * total / max(rows, 1) / 9,
                            "validation_episode_macro_balanced_nrmse": score, "seconds": time.perf_counter() - t0, "updates": updates})
            write_json(output / "history.json", history)
            print(f"Epoch {epoch}: loss={history[-1]['training_normalized_loss']:.6f}, validation NRMSE={score:.6f}, best={best:.6f}", flush=True)
        policy = TransformerPolicy.load(output / "model.npz")
        # NumPy inference (key/value cache) must reproduce the torch network (compared in float64) on a real validation episode.
        check = build_network(torch, len(names), d_model, layers, heads, ffn, head_width, window).double()
        load_params(torch, check, policy.p)
        rows_check = min(400, int(VL[0]))
        vx0 = episode_group(dataset, split["validation"][:1], spec, mode)[0][0][:rows_check]  # raw float64 features
        with torch.no_grad():
            ref = check(torch.from_numpy(((vx0[None] - x_mean) / x_scale)), stride)
        ref = ref[0].numpy() * y_scale + y_mean
        np.testing.assert_allclose(policy.predict_features(vx0), ref, rtol=1e-7, atol=1e-7)
        raw, raw_lidar = x_check(dataset, split["validation"][0], rows_check)
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
        info = {"kind": KIND, "variant": mode, "input_mode": mode, "backend": "pytorch", "training_device": str(device), **arch,
                "gradient_clip_norm": 10.0, "regression_parameters": sum(p.numel() for p in network.parameters()), "weight_dtype": "float32",
                "input_features": len(names), "input_names": names, "independent_outputs": 9, "feature_spec": spec,
                "control_period": f"head reacts every physics step (1 ms); one attention token every {stride} step(s) over the last {window} tokens; memory reset per episode",
                "episode_filter": episode_filter, "training_episodes": len(split["training"]), "validation_episodes": len(split["validation"]),
                "epochs_completed": epoch, "best_epoch": best_epoch, "training_updates": updates,
                "best_validation_episode_macro_balanced_nrmse": best,
                "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
                "checkpoint_selection": "minimum validation episode-macro balanced NRMSE over all physics steps",
                "training_loss": "masked half output-sum MSE per whole episode with weight L2 (LayerNorm gains excluded); Adam; grad-norm clip 10",
                "inference_single_observation_us": {"median": float(np.median(timings) / 1000), "p95": float(np.percentile(timings, 95) / 1000), "samples": len(timings)},
                "output_limits": "none applied; the evaluator clips to the declared envelope",
                "fit_and_transform_seconds": time.perf_counter() - started}
        write_json(output / "model_info.json", info); write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", finished_at=utc_now(), wall_s=time.perf_counter() - started,
                        model_sha256=sha256(output / "model.npz"), split_sha256=sha256(output / "split.json"), metrics_sha256=sha256(output / "metrics.json"))
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
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/transformer.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-s", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs", default=EGO, choices=sorted(MODES), help="input variant (learning/features.py)")
    parser.add_argument("--stride", type=int, default=20, help="Token period in physics steps; the head reacts every step")
    parser.add_argument("--window", type=int, default=1000, help="Causal attention window in tokens (1000 x 20 ms = 20 s)")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=128)
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--episodes-per-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, help="training seed (overrides the config; seed 0 of the report = 20261025)")
    args = parser.parse_args(argv)
    _, metrics = train(args.dataset, args.split, args.config, args.output_dir, args.horizon_s, args.device, args.inputs, args.stride,
                       args.window, args.d_model, args.layers, args.heads, args.ffn, args.head_width, args.episodes_per_batch, seed=args.seed)
    print(f"validation: episode-macro balanced NRMSE={metrics['validation']['model']['episode_macro']['balanced_nrmse']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
