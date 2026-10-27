"""Diagnostic ablation: does the expert's hidden controller state explain the label variance?

Trains the reference MLP under identical data, split, scalers for the 21 baseline
inputs, architecture, optimizer, seed and patience, varying only the extra input
columns taken from the expert's own memory as recorded in the episode NPZ files:

    baseline       21 inputs (control arm)
    phase          + navigation phase one-hot + nav.done flag
    reference      + arm reference lead (expert_q_des_arm - measured q), 7 values
    full           + both

The resulting models are oracles: they consume signals a deployed policy cannot
observe, so they are never exported as MLPPolicy and cannot be loaded by the
autonomous evaluator. The output is a validation-error comparison only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from robot_learning.data.storage import read_json, write_json, sha256, utc_now, git_info
from robot_learning.settings import PROJECT_ROOT
from .dataset import inventory, load_episode, validate_partition, inventory_fingerprint
from .metrics import errors, aggregate
from .model import INPUT_NAMES, OUTPUT_NAMES

PHASES = ("align", "drive", "orient")
ARMS = {
    "baseline": {"phase": False, "reference": False},
    "phase": {"phase": True, "reference": False},
    "reference": {"phase": False, "reference": True},
    "full": {"phase": True, "reference": True},
}


def privileged_columns(directory, record):
    """Return (phase one-hot + done, reference lead) for one validated episode."""
    path = Path(directory) / "episodes" / f"{record['episode_id']}.npz"
    with np.load(path, allow_pickle=False) as data:
        phase = data["expert_phase"]
        done = data["base_navigation_done"].astype(np.float64)
        q_des = data["expert_q_des_arm"].astype(np.float64)
        q = data["observation"][:, 4:11].astype(np.float64)
    unknown = set(np.unique(phase)) - set(PHASES)
    if unknown:
        raise ValueError(f"Unexpected expert phases {sorted(unknown)} in {record['episode_id']}")
    one_hot = np.stack([(phase == name).astype(np.float64) for name in PHASES], axis=1)
    phase_block = np.concatenate([one_hot, done[:, None]], axis=1)
    lead = q_des - q
    return phase_block, lead


def load_group(directory, records):
    xs, ys, phases, leads, names, labels = [], [], [], [], [], []
    for record in records:
        x, y, phase_labels = load_episode(directory, record)
        phase_block, lead = privileged_columns(directory, record)
        if len(phase_block) != len(x) or len(lead) != len(x):
            raise ValueError("Privileged columns misaligned with observations")
        xs.append(x); ys.append(y); phases.append(phase_block); leads.append(lead)
        names.append(record["episode_id"]); labels.append(phase_labels)
    rows = [len(x) for x in xs]
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "phase": np.concatenate(phases),
            "lead": np.concatenate(leads), "labels": np.concatenate(labels),
            "ranges": list(zip(np.r_[0, np.cumsum(rows)[:-1]].tolist(), np.cumsum(rows).tolist())),
            "episodes": names}


def assemble(group, arm, stats):
    """Standardize each block with training statistics; one-hot/done stay raw 0/1."""
    blocks = [(group["x"] - stats["x_mean"]) / stats["x_scale"]]
    names = list(INPUT_NAMES)
    if ARMS[arm]["phase"]:
        blocks.append(group["phase"])
        names += [f"expert_phase_{p}" for p in PHASES] + ["expert_navigation_done"]
    if ARMS[arm]["reference"]:
        blocks.append((group["lead"] - stats["lead_mean"]) / stats["lead_scale"])
        names += [f"expert_reference_lead_{i}" for i in range(1, 8)]
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def scale_of(values):
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return scale


def build_network(widths, seed, device):
    import torch
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(fan_in, fan_out))
        if i < len(widths) - 2:
            layers.append(torch.nn.ReLU())
    network = torch.nn.Sequential(*layers).to(device=device, dtype=torch.float32)
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        for layer, fan_in, fan_out in zip(network[::2], widths[:-1], widths[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            layer.weight.copy_(torch.from_numpy(rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32).T.copy()))
            layer.bias.copy_(torch.from_numpy(rng.uniform(-bound, bound, fan_out).astype(np.float32)))
    return network, rng


def episode_scores(network, vx, vy_phys, ranges, y_mean, y_scale, device, batch):
    """Episode-macro balanced NRMSE with the project's own error definition."""
    import torch
    predictions = np.empty_like(vy_phys)
    with torch.no_grad():
        for start in range(0, len(vx), batch):
            chunk = torch.from_numpy(vx[start:start + batch]).to(device)
            predictions[start:start + batch] = network(chunk).double().cpu().numpy() * y_scale + y_mean
    per_episode = [errors(vy_phys[a:b], predictions[a:b], y_scale) for a, b in ranges]
    return per_episode, predictions


def fit_arm(arm, train, valid, stats, hp, device, log):
    import torch
    torch.manual_seed(hp["seed"])
    x_train, names = assemble(train, arm, stats)
    x_valid, _ = assemble(valid, arm, stats)
    y_train = ((train["y"] - stats["y_mean"]) / stats["y_scale"]).astype(np.float32)
    tx = torch.from_numpy(x_train).to(device); ty = torch.from_numpy(y_train).to(device)
    network, rng = build_network([x_train.shape[1], *hp["hidden_layer_sizes"], 9], hp["seed"], device)
    optimizer = torch.optim.Adam(network.parameters(), lr=hp["learning_rate"], betas=(.9, .999), eps=1e-8)
    alpha = hp["alpha"]; batch = hp["batch_size"]; n = len(tx)
    best = math.inf; patience_score = math.inf; stale = 0; best_epoch = 0; best_state = None; history = []
    epoch = 0; started = time.perf_counter()
    while stale < hp["patience"] and (hp["max_epochs"] is None or epoch < hp["max_epochs"]):
        epoch += 1; t0 = time.perf_counter()
        order = torch.from_numpy(rng.permutation(n)).to(device)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            pred = network(tx[idx])
            squared = (pred - ty[idx]).square()
            # sklearn convention: gradients from half output-sum MSE, L2 on weights scaled by alpha/(2*rows)
            objective = .5 * squared.sum(dim=1).mean() + (.5 * alpha / len(idx)) * sum(
                layer.weight.square().sum() for layer in network[::2])
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            total += float(.5 * squared.detach().mean()) * len(idx)
        per_episode, _ = episode_scores(network, x_valid, valid["y"], valid["ranges"],
                                        stats["y_mean"], stats["y_scale"], device, 16384)
        score = float(np.mean([e["balanced_nrmse"] for e in per_episode]))
        if not math.isfinite(score):
            raise ValueError(f"Non-finite validation score in arm {arm}")
        if score < best:
            best = score; best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        if score < patience_score - hp["min_delta"]:
            patience_score = score; stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "training_normalized_loss": total / n,
                        "validation_episode_macro_balanced_nrmse": score, "seconds": time.perf_counter() - t0})
        log(f"[{arm}] epoch {epoch}: loss={total / n:.6f} val NRMSE={score:.6f} best={best:.6f}")
    network.load_state_dict(best_state)
    per_episode, predictions = episode_scores(network, x_valid, valid["y"], valid["ranges"],
                                              stats["y_mean"], stats["y_scale"], device, 16384)
    def masked(mask):
        return errors(valid["y"][mask], predictions[mask], stats["y_scale"]) if mask.any() else None
    by_phase = {str(label): masked(valid["labels"] == label) for label in np.unique(valid["labels"])}
    done = valid["phase"][:, 3] > 0.5
    by_done = {"navigation_done": masked(done), "navigating": masked(~done)}
    return {"arm": arm, "input_names": names, "input_features": len(names),
            "parameters": sum(p.numel() for p in network.parameters()),
            "epochs_completed": epoch, "best_epoch": best_epoch,
            "stop_reason": "patience" if stale >= hp["patience"] else "max_epochs",
            "best_validation_episode_macro_balanced_nrmse": best,
            "validation_episode_macro": aggregate(per_episode),
            "validation_pooled": errors(valid["y"], predictions, stats["y_scale"]),
            "validation_by_expert_phase_pooled": by_phase,
            "validation_by_navigation_done_pooled": by_done,
            "fit_seconds": time.perf_counter() - started, "history": history}


def run(dataset, split_path, config_path, output, arms, device_name):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    dataset = Path(dataset).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(config_path); hp = config["mlp"]
    records = inventory(dataset)
    saved = read_json(split_path)
    if Path(saved["dataset"]).resolve() != dataset:
        raise ValueError("Split belongs to another dataset")
    split = validate_partition(records, {k: saved[k] for k in ["training", "validation"]})
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    log_path = output / "log.txt"

    def log(message):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    meta = {"status": "running", "created_at": utc_now(), "dataset": str(dataset), "split": str(split_path),
            "inventory_sha256": inventory_fingerprint(records), "training_config": config,
            "config_sha256": sha256(config_path), "project": git_info(PROJECT_ROOT),
            "source_sha256": sha256(Path(__file__)), "device": str(device), "arms": arms,
            "purpose": "Diagnostic oracle ablation of expert hidden state; models are not deployable policies and are not exported",
            "privileged_inputs": {"phase": "expert_phase one-hot over align/drive/orient plus base_navigation_done",
                                   "reference": "expert_q_des_arm minus measured arm q (PD reference lead), standardized on training"}}
    write_json(output / "run.json", meta)
    started = time.perf_counter()
    try:
        log(f"Loading {len(split['training'])} training and {len(split['validation'])} validation episodes")
        train = load_group(dataset, split["training"]); valid = load_group(dataset, split["validation"])
        stats = {"x_mean": train["x"].mean(axis=0), "x_scale": scale_of(train["x"]),
                 "y_mean": train["y"].mean(axis=0), "y_scale": scale_of(train["y"]),
                 "lead_mean": train["lead"].mean(axis=0), "lead_scale": scale_of(train["lead"])}
        log(f"Training rows {len(train['x'])}, validation rows {len(valid['x'])}")
        phase_share = {name: float(train["phase"][:, i].mean()) for i, name in enumerate(PHASES)}
        phase_share["navigation_done"] = float(train["phase"][:, 3].mean())
        results = {}
        for arm in arms:
            results[arm] = fit_arm(arm, train, valid, stats, hp, device, log)
            write_json(output / "results.json", {"training_phase_row_share": phase_share, "arms": results})
        meta.update(status="complete", finished_at=utc_now(), wall_s=time.perf_counter() - started)
        write_json(output / "run.json", meta)
        write_summary(output, results, phase_share)
        return results
    except BaseException as exc:
        meta.update(status="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", meta)
        raise


def fmt(block):
    return f"{block['balanced_nrmse']:.4f}" if block else "n/a"


def write_summary(output, results, phase_share):
    lines = ["# Privileged-input ablation (diagnostic oracle)", "",
             f"Training rows by expert state: " + ", ".join(f"{k} {v:.3f}" for k, v in phase_share.items()), "",
             "| Arm | Inputs | Params | Best epoch | Val NRMSE (episode macro) | Wheels RMSE rad/s | Arm RMSE N·m | NRMSE navigating | NRMSE nav done |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, r in results.items():
        pooled = r["validation_pooled"]; done = r["validation_by_navigation_done_pooled"]
        lines.append(f"| {arm} | {r['input_features']} | {r['parameters']} | {r['best_epoch']} | "
                     f"{r['best_validation_episode_macro_balanced_nrmse']:.4f} | {pooled['wheels_macro_rmse_rad_s']:.4f} | "
                     f"{pooled['arm_macro_rmse_nm']:.4f} | {fmt(done['navigating'])} | {fmt(done['navigation_done'])} |")
    lines += ["", "Per-output validation RMSE (pooled rows):", "",
              "| Arm | " + " | ".join(OUTPUT_NAMES) + " |", "|---|" + "---:|" * len(OUTPUT_NAMES)]
    for arm, r in results.items():
        per = r["validation_pooled"]["per_output"]
        lines.append(f"| {arm} | " + " | ".join(f"{per[n]['rmse']:.3f}" for n in OUTPUT_NAMES) + " |")
    lines += ["", "Oracle models consume expert memory unavailable to a deployed policy; they were not exported.",
              "Same split, scalers for the 21 baseline inputs, architecture, optimizer, seed and patience across arms."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, help="split.json of the reference training run")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiment_specs/training/mlp_pytorch.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(args.dataset, args.split, args.config, args.output_dir, args.arms, args.device)
    print((Path(args.output_dir) / "summary.md").read_text(encoding="utf-8"))
    return 0
