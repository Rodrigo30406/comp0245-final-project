"""Recurrent variants of the GRU policy for the stop problem (E8a, E8b).

Both share the input pipeline, the strided memory and the training loop of `gru_policy.py` and keep the
assignment's two outputs (wheel commands, arm torques). They differ in where the memory enters:

`gated`  (E8a, the learned stop state). A stateless head reads the current features and a scalar gate
    g_t in [0, 1]: u_t = head([x_t, g_t]). A small GRU reads the feature history and emits a stop score
    s_t = sigmoid(w_s . h_t + b_s) at every memory update; the gate is the running complement
    g_t = 1 - max_{tau <= t} s_tau, so it starts at 1, can only decrease, and stays wherever it was driven.
    The wheel commands are the head's wheel commands multiplied by g_t in physical units (so g_t = 0 is an
    exact zero whatever the head says), and the arm torques are the head's arm torques, which see g_t as an
    input (the base-stopped signal the arm never had, §6t.1). Trained end to end from the wheel and arm
    labels only; nothing tells the gate when to close except the labels going to zero.

`mlp_gru` (E8b). An encoder MLP e_t = relu(W_e x_t + b_e) feeds the GRU instead of the raw features; the
    head reads [h_t, x_t] as in the base GRU. The memory therefore sees a learned view of the inputs.

Inference is NumPy; it is verified against the torch network by test and at the end of every training run.
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
from .dataset import inventory, load_episode, validate_partition, extra_inventory
from .egocentric import validate_spec, spec_from_episode_config, VARIANT
from .gru_policy import (MODES, input_names, features_for, gru_step, episode_group, normalised_group)
from .metrics import errors, aggregate
from .model import INPUT_NAMES, OUTPUT_NAMES, expand_actions

KIND = "gru_gated_regressor"
ARCHITECTURES = ("gated", "mlp_gru")
GATE_BIAS_INIT = -4.0  # initial stop score sigmoid(-4) = 0.018: the gate starts open


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


class GatedPolicy:
    """NumPy inference for both architectures; `params` holds the torch-layout weights (see `PARAMS`)."""

    PARAMS = {"gated": ["w_ih", "w_hh", "b_ih", "b_hh", "w_s", "b_s", "w1", "b1", "w2", "b2"],
              "mlp_gru": ["w_e", "b_e", "w_ih", "w_hh", "b_ih", "b_hh", "w1", "b1", "w2", "b2"]}

    def __init__(self, spec, mode, architecture, x_mean, x_scale, y_mean, y_scale, params, stride_steps=1):
        validate_spec(spec)
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")
        if type(stride_steps) is not int or stride_steps < 1:
            raise ValueError("stride_steps must be a positive integer")
        self.spec = spec; self.mode = mode; self.arch = architecture; self.names = input_names(mode); self.stride = stride_steps
        if mode != "raw21" and spec["variant"] != mode:
            raise ValueError("Feature specification variant must match the input mode")
        f64 = lambda a: np.asarray(a, dtype=np.float64)
        self.x_mean, self.x_scale, self.y_mean, self.y_scale = f64(x_mean), f64(x_scale), f64(y_mean), f64(y_scale)
        if set(params) != set(self.PARAMS[architecture]):
            raise ValueError(f"Parameters for {architecture!r} must be exactly {self.PARAMS[architecture]}")
        self.p = {k: f64(v) for k, v in params.items()}
        p = self.p; F = len(self.names); H = p["w_hh"].shape[1]; W = p["w1"].shape[0]
        G = p["w_ih"].shape[1]  # what the GRU reads: F features (gated) or E encoder units (mlp_gru)
        head_in = F + 1 if architecture == "gated" else H + F
        ok = (self.x_mean.shape == (F,) and self.x_scale.shape == (F,) and self.y_mean.shape == (9,) and self.y_scale.shape == (9,)
              and p["w_ih"].shape == (3 * H, G) and p["w_hh"].shape == (3 * H, H) and p["b_ih"].shape == (3 * H,)
              and p["b_hh"].shape == (3 * H,) and p["w1"].shape == (W, head_in) and p["b1"].shape == (W,)
              and p["w2"].shape == (9, W) and p["b2"].shape == (9,))
        if architecture == "gated":
            ok = ok and G == F and p["w_s"].shape == (1, H) and p["b_s"].shape == (1,)
        else:
            ok = ok and p["w_e"].shape == (G, F) and p["b_e"].shape == (G,)
        if not ok:
            raise ValueError("Invalid policy shapes")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Scales must be positive")
        if not all(np.isfinite(v).all() for v in self.arrays().values() if v.dtype.kind == "f"):
            raise ValueError("Non-finite parameters")
        self.hidden = H
        self.reset()

    @property
    def input_features(self):
        return len(self.names)

    def arrays(self):
        return {"feature_mode": np.array(self.mode), "architecture": np.array(self.arch),
                "x_mean": self.x_mean, "x_scale": self.x_scale, "y_mean": self.y_mean, "y_scale": self.y_scale, **self.p,
                "standoff_heights": np.asarray(self.spec["standoff_heights"], dtype=np.float64),
                "standoff_distances": np.asarray(self.spec["standoff_distances"], dtype=np.float64),
                "clock": np.array([self.spec["horizon_s"], self.spec["dt_s"], self.stride], dtype=np.float64)}

    # --- one memory update and one head evaluation, shared by batch and live inference ---
    def _memory(self, f, state):
        p = self.p; h, m = state
        if self.arch == "mlp_gru":
            f = np.maximum(p["w_e"] @ f + p["b_e"], 0.0)
        h = gru_step(f, h, p["w_ih"], p["w_hh"], p["b_ih"], p["b_hh"])
        if self.arch == "gated":
            m = max(m, float(_sigmoid(p["w_s"] @ h + p["b_s"])[0]))
        return h, m

    def _head(self, f, state):
        p = self.p; h, m = state
        if self.arch == "gated":
            g = 1.0 - m
            hidden = np.maximum(p["w1"] @ np.concatenate([f, [g]]) + p["b1"], 0.0)
            raw = (p["w2"] @ hidden + p["b2"]) * self.y_scale + self.y_mean
            raw[:2] *= g
            return raw
        hidden = np.maximum(p["w1"] @ np.concatenate([h, f]) + p["b1"], 0.0)
        return (p["w2"] @ hidden + p["b2"]) * self.y_scale + self.y_mean

    def gate(self):
        """Current gate value (1 = wheels pass through, 0 = wheels exactly zero); 1 for `mlp_gru`."""
        return 1.0 - self._state[1]

    def predict_features(self, features, return_gate=False):
        """One episode's physics-step feature rows, in order, from a fresh state."""
        f = (np.asarray(features, dtype=np.float64) - self.x_mean) / self.x_scale
        state = (np.zeros(self.hidden), 0.0); out = np.empty((len(f), 9)); gates = np.empty(len(f))
        for i, row in enumerate(f):
            if i % self.stride == 0:
                state = self._memory(row, state)
            out[i] = self._head(row, state); gates[i] = 1.0 - state[1]
        return (out, gates) if return_gate else out

    def predict(self, observations, times=None):
        x = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        t = np.arange(len(x)) * self.spec["dt_s"] if times is None else np.asarray(times, dtype=np.float64)
        return self.predict_features(features_for(self.mode, x, t, self.spec))

    def predict_action(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (21,):
            raise ValueError("Live inference takes one 21D observation")
        t = self._count * self._dt
        f = (features_for(self.mode, x[None], np.array([t]), self.spec)[0] - self.x_mean) / self.x_scale
        if self._count % self.stride == 0:
            self._state = self._memory(f, self._state)
        self._count += 1
        return expand_actions(self._head(f, self._state))

    def reset(self, dt=None):
        if dt is not None and not np.isclose(dt, self.spec["dt_s"], rtol=0, atol=1e-12):
            raise ValueError("Policy clock dt differs from simulator dt")
        self._state = (np.zeros(self.hidden), 0.0); self._count = 0; self._dt = self.spec["dt_s"]

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            mode = str(d["feature_mode"]); arch = str(d["architecture"])
            spec = {"variant": mode if mode != "raw21" else VARIANT, "standoff_heights": d["standoff_heights"].tolist(),
                    "standoff_distances": d["standoff_distances"].tolist(),
                    "horizon_s": float(d["clock"][0]), "dt_s": float(d["clock"][1])}
            params = {k: d[k] for k in cls.PARAMS[arch]}
            return cls(spec, mode, arch, d["x_mean"], d["x_scale"], d["y_mean"], d["y_scale"], params,
                       stride_steps=int(d["clock"][2]))


def build_network(torch, architecture, n_features, hidden, head_width, y_mean, y_scale, encoder_width=64):
    """Torch twin of `GatedPolicy`. Outputs are normalised (as the labels); the gate acts in physical units inside."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = architecture
            if architecture == "mlp_gru":
                self.encoder = torch.nn.Linear(n_features, encoder_width)
                self.gru = torch.nn.GRU(encoder_width, hidden, num_layers=1, batch_first=True)
                self.head1 = torch.nn.Linear(hidden + n_features, head_width)
            else:
                self.gru = torch.nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
                self.stop = torch.nn.Linear(hidden, 1)
                with torch.no_grad():
                    self.stop.bias.fill_(GATE_BIAS_INIT)
                self.head1 = torch.nn.Linear(n_features + 1, head_width)
                self.register_buffer("y_mean_wheels", torch.as_tensor(np.asarray(y_mean[:2], dtype=np.float64)))
                self.register_buffer("y_scale_wheels", torch.as_tensor(np.asarray(y_scale[:2], dtype=np.float64)))
            self.head2 = torch.nn.Linear(head_width, 9)

        def initial_state(self, batch, device, dtype):
            h = torch.zeros(1, batch, hidden, device=device, dtype=dtype)
            m = torch.zeros(batch, 1, device=device, dtype=dtype)
            return (h, m)

        def forward(self, x_rows, stride, state=None):
            """x_rows (B, T, F) normalised physics rows; memory from rows 0, stride, 2*stride, ...
            Returns normalised outputs (B, T, 9) and the state (h, running max stop score) after the last memory step."""
            B, T, _ = x_rows.shape
            if state is None:
                state = self.initial_state(B, x_rows.device, x_rows.dtype)
            h, m = state; xs = x_rows[:, ::stride]
            if self.arch == "mlp_gru":
                out, h = self.gru(torch.relu(self.encoder(xs)), h)
                h_rows = out.repeat_interleave(stride, dim=1)[:, :T]
                return self.head2(torch.relu(self.head1(torch.cat([h_rows, x_rows], dim=-1)))), (h, m)
            out, h = self.gru(xs, h)
            s = torch.sigmoid(self.stop(out))                                   # (B, Tm, 1)
            m_run = torch.cummax(torch.cat([m[:, None, :], s], dim=1), dim=1).values[:, 1:]  # running max incl. carried state
            g_rows = (1.0 - m_run).repeat_interleave(stride, dim=1)[:, :T]        # (B, T, 1)
            u = self.head2(torch.relu(self.head1(torch.cat([x_rows, g_rows], dim=-1))))
            ym = self.y_mean_wheels.to(u.dtype); ys = self.y_scale_wheels.to(u.dtype)
            wheels = ((u[..., :2] * ys + ym) * g_rows - ym) / ys                # gate in physical units, back to normalised
            return torch.cat([wheels, u[..., 2:]], dim=-1), (h, m_run[:, -1])
    return Net()


def export_policy(network, spec, mode, architecture, stats, stride):
    g = network.gru
    cpu = lambda p: p.detach().cpu().numpy()
    params = {"w_ih": cpu(g.weight_ih_l0), "w_hh": cpu(g.weight_hh_l0), "b_ih": cpu(g.bias_ih_l0), "b_hh": cpu(g.bias_hh_l0),
              "w1": cpu(network.head1.weight), "b1": cpu(network.head1.bias), "w2": cpu(network.head2.weight), "b2": cpu(network.head2.bias)}
    if architecture == "gated":
        params.update(w_s=cpu(network.stop.weight), b_s=cpu(network.stop.bias))
    else:
        params.update(w_e=cpu(network.encoder.weight), b_e=cpu(network.encoder.bias))
    return GatedPolicy(spec, mode, architecture, *stats, params, stride_steps=stride)


def load_into(network, policy, torch):
    """Copy a policy's parameters into a torch twin (used by the post-training check and the tests)."""
    p = policy.p
    with torch.no_grad():
        network.gru.weight_ih_l0.copy_(torch.from_numpy(p["w_ih"])); network.gru.weight_hh_l0.copy_(torch.from_numpy(p["w_hh"]))
        network.gru.bias_ih_l0.copy_(torch.from_numpy(p["b_ih"])); network.gru.bias_hh_l0.copy_(torch.from_numpy(p["b_hh"]))
        network.head1.weight.copy_(torch.from_numpy(p["w1"])); network.head1.bias.copy_(torch.from_numpy(p["b1"]))
        network.head2.weight.copy_(torch.from_numpy(p["w2"])); network.head2.bias.copy_(torch.from_numpy(p["b2"]))
        if policy.arch == "gated":
            network.stop.weight.copy_(torch.from_numpy(p["w_s"])); network.stop.bias.copy_(torch.from_numpy(p["b_s"]))
        else:
            network.encoder.weight.copy_(torch.from_numpy(p["w_e"])); network.encoder.bias.copy_(torch.from_numpy(p["b_e"]))


def train(dataset, split_path, config_path, output, horizon_s=40.0, device_name="cuda", architecture="gated",
          hidden=32, chunk_steps=1000, episodes_per_batch=8, mode="raw21_targetpolar_clock_v1", stride=20,
          head_width=128, encoder_width=64, validation_episodes_per_batch=8):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    names = input_names(mode)
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}")
    if type(stride) is not int or stride < 1 or type(chunk_steps) is not int or chunk_steps < 1:
        raise ValueError("stride and chunk_steps must be positive integers")
    dataset = Path(dataset).resolve(); output = Path(output).resolve(); split_path = Path(split_path).resolve()
    config = read_json(config_path); hp = config["mlp"]
    saved = read_json(split_path)
    if Path(saved["dataset"]).resolve() != dataset:
        raise ValueError("Split belongs to another dataset")
    records = inventory(dataset) + extra_inventory(saved)
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    source_cfg = read_json(dataset / "config.json")
    episode_cfg = resolve_episode_config({k: v for k, v in source_cfg.items() if k in episode_defaults()})
    dt = read_json(dataset / "episodes" / f"{records[0]['episode_id']}.json")["dt"]
    spec = spec_from_episode_config(episode_cfg, dt, horizon_s, mode if mode != "raw21" else VARIANT)
    device = torch.device(device_name); torch.manual_seed(hp["seed"])
    output.mkdir(parents=True, exist_ok=False)
    chunk_rows = chunk_steps * stride
    arch_doc = {"gated": "head([features, gate]); gate = 1 - running max of sigmoid(stop(GRU)); wheels = gate * head wheels (physical units)",
                "mlp_gru": "GRU reads relu(encoder(features)); head([memory, features])"}[architecture]
    write_json(output / "config.json", {**config, "input_features": mode, "clock_horizon_s": horizon_s,
                                        "recurrent": {"cell": "gru", "architecture": architecture, "architecture_doc": arch_doc,
                                                      "hidden": hidden, "layers": 1, "head_width": head_width,
                                                      "encoder_width": encoder_width if architecture == "mlp_gru" else None,
                                                      "gate_bias_init": GATE_BIAS_INIT if architecture == "gated" else None,
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
                "input_mode": mode, "architecture": architecture, "backend": "pytorch",
                "test_set": "none; development train/validation only", "closed_loop_evaluation": "not run by the trainer"}
    write_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        X, Y, L, stats, _ = normalised_group(dataset, split["training"], spec, mode)
        x_mean, x_scale, y_mean, y_scale = stats
        mask = np.arange(X.shape[1])[None, :] < L[:, None]
        tx = torch.from_numpy(X).to(device); ty = torch.from_numpy(Y).to(device); tmask = torch.from_numpy(mask).to(device)
        del X, Y
        VX, _, VL, _, VYS = normalised_group(dataset, split["validation"], spec, mode, stats)
        tvx = torch.from_numpy(VX).to(device); T_val = VX.shape[1]; del VX
        make = lambda: build_network(torch, architecture, len(names), hidden, head_width, y_mean, y_scale, encoder_width)
        network = make().to(device)
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
                state = None
                for c in range(0, T, chunk_rows):
                    bm = tmask[idx, c:c + chunk_rows]
                    if not bool(bm.any()):
                        break
                    bx = tx[idx, c:c + chunk_rows]; by = ty[idx, c:c + chunk_rows]
                    out, state = network(bx, stride, state)
                    count = bm.sum()
                    squared = ((out - by).square().sum(dim=2) * bm).sum() / count
                    objective = .5 * squared + (.5 * hp["alpha"] / max(1, int(count))) * sum(
                        p.square().sum() for name, p in network.named_parameters() if "weight" in name)
                    optimizer.zero_grad(set_to_none=True); objective.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                    optimizer.step(); updates += 1; state = tuple(s_.detach() for s_ in state)
                    total += float(squared.detach()) * int(count); rows += int(count)
            score, _, _ = validation_score()
            if not math.isfinite(score):
                raise ValueError("Non-finite validation score")
            if score < best:
                best = score; best_epoch = epoch; export_policy(network, spec, mode, architecture, stats, stride).save(output / "model.npz")
            if score < patience_score - hp["min_delta"]:
                patience_score = score; stale = 0
            else:
                stale += 1
            history.append({"epoch": epoch, "training_normalized_loss": .5 * total / max(rows, 1) / 9,
                            "validation_episode_macro_balanced_nrmse": score, "seconds": time.perf_counter() - t0,
                            "updates": updates})
            write_json(output / "history.json", history)
            print(f"Epoch {epoch}: loss={history[-1]['training_normalized_loss']:.6f}, validation NRMSE={score:.6f}, best={best:.6f}", flush=True)
        policy = GatedPolicy.load(output / "model.npz")
        # NumPy inference must reproduce the torch network (compared in float64) on a real validation episode.
        check = make().double(); load_into(check, policy, torch)
        with torch.no_grad():
            v0 = episode_group(dataset, split["validation"][:1], spec, mode)[0][0][:400]
            ref, _ = check(torch.from_numpy(((v0 - x_mean) / x_scale)[None]), stride)
        ref = ref[0].numpy() * y_scale + y_mean
        np.testing.assert_allclose(policy.predict_features(v0), ref, rtol=1e-8, atol=1e-8)
        raw = load_episode(dataset, split["validation"][0])[0][:400]
        policy.reset(dt); live = np.stack([policy.predict_action(o) for o in raw]); policy.reset()
        np.testing.assert_allclose(live, expand_actions(policy.predict(raw)), rtol=1e-8, atol=1e-8)
        score, pred, per = validation_score()
        vmask = np.arange(T_val)[None, :] < VL[:, None]
        metrics = {"validation": {"model": aggregate(per),
                                  "mean_baseline": aggregate([errors(VYS[i], np.broadcast_to(y_mean, (VL[i], 9)), y_scale) for i in range(len(VL))]),
                                  "pooled_transitions": {"model": errors(np.concatenate(VYS), pred[vmask], y_scale)}}}
        if architecture == "gated":  # where does the learned gate close on the validation episodes?
            closes = []
            for i, rec in enumerate(split["validation"][:min(64, len(VL))]):
                feats = episode_group(dataset, [rec], spec, mode)[0][0]
                _, gates = policy.predict_features(feats, return_gate=True)
                below = np.flatnonzero(gates < 0.5)
                closes.append({"episode_id": rec["episode_id"], "gate_final": float(gates[-1]), "gate_min": float(gates.min()),
                               "first_below_half_s": float(below[0] * dt) if len(below) else None,
                               "label_zero_from_s": float(np.flatnonzero(np.abs(VYS[i][:, :2]).max(axis=1) > 0)[-1] * dt + dt)
                               if (np.abs(VYS[i][:, :2]).max(axis=1) > 0).any() else 0.0})
            metrics["validation"]["gate"] = {"episodes": closes,
                                             "final_gate_median": float(np.median([c["gate_final"] for c in closes])),
                                             "closed_below_half": int(sum(c["first_below_half_s"] is not None for c in closes))}
        timings = []; obs = raw[0]; policy.reset(dt)
        for _ in range(100): policy.predict_action(obs)
        for _ in range(2000):
            t1 = time.perf_counter_ns(); policy.predict_action(obs); timings.append(time.perf_counter_ns() - t1)
        policy.reset()
        if any(sha256(PROJECT_ROOT / p) != d for p, d in source_hashes.items()):
            raise ValueError("Training source changed during execution")
        info = {"kind": KIND, "variant": mode, "input_mode": mode, "architecture": architecture, "architecture_doc": arch_doc,
                "backend": "pytorch", "training_device": str(device),
                "cell": "gru", "hidden": hidden, "layers": 1, "head_width": head_width,
                "encoder_width": encoder_width if architecture == "mlp_gru" else None,
                "memory_stride_steps": stride, "reaction_period_steps": 1,
                "tbptt_chunk_decisions": chunk_steps, "episodes_per_batch": episodes_per_batch, "gradient_clip_norm": 10.0,
                "regression_parameters": sum(p.numel() for p in network.parameters()), "weight_dtype": "float32",
                "input_features": len(names), "input_names": names, "independent_outputs": 9, "feature_spec": spec,
                "control_period": f"head reacts every physics step (1 ms); GRU memory updated every {stride} step(s); state reset per episode",
                "epochs_completed": epoch, "best_epoch": best_epoch, "training_updates": updates,
                "best_validation_episode_macro_balanced_nrmse": best,
                "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
                "checkpoint_selection": "minimum validation episode-macro balanced NRMSE over all physics steps",
                "training_loss": "masked half output-sum MSE per chunk with weight L2; Adam; grad-norm clip 10; wheel and arm labels only",
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/mlp_pytorch.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", default="gated", choices=ARCHITECTURES)
    parser.add_argument("--horizon-s", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=32, help="GRU width (the gate's memory is small by design)")
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--encoder-width", type=int, default=64, help="mlp_gru only: encoder output width")
    parser.add_argument("--chunk-steps", type=int, default=1000, help="TBPTT chunk length in memory updates")
    parser.add_argument("--episodes-per-batch", type=int, default=8)
    parser.add_argument("--inputs", default="raw21_targetpolar_clock_v1", choices=sorted(MODES))
    parser.add_argument("--stride", type=int, default=20, help="Memory update period in physics steps; the head reacts every step")
    args = parser.parse_args(argv)
    _, metrics = train(args.dataset, args.split, args.config, args.output_dir, args.horizon_s, args.device, args.architecture,
                       args.hidden, args.chunk_steps, args.episodes_per_batch, args.inputs, args.stride, args.head_width,
                       args.encoder_width)
    print(f"validation: episode-macro balanced NRMSE={metrics['validation']['model']['episode_macro']['balanced_nrmse']:.6f}")
    return 0
