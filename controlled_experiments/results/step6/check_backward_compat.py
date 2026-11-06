"""Short, isolated C3 retraining and G k=1 matrix compatibility evidence.

This command reads approved train/validation caches and the original C3 model.
It only creates new outputs below step6/smoke and refuses existing artifacts.
Stages run in separate processes so C3's training matrices are released before
the full k=1 matrices are materialised. Use the approved pixi Python launcher.
"""
import os

for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

TASK_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

import numpy as np

import train_policy as tp

SHARED_FILES = ("train_policy.py", "bc_features.py", "policies/bc_policy.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def code_hashes():
    return {name: tp.sha256_file(TASK_DIR / name) for name in SHARED_FILES}


def matrix_descriptor(a):
    if a.dtype != np.dtype("float64"):
        raise AssertionError(f"expected float64 training matrix; got {a.dtype}")
    digest = hashlib.sha256()
    for start in range(0, len(a), 65536):
        digest.update(np.ascontiguousarray(a[start:start + 65536]).tobytes())
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": digest.hexdigest()}


def compare_models(reference, actual):
    """Compare every NPZ member byte-for-byte except JSON provenance metadata."""
    comparisons = {}
    with np.load(reference, allow_pickle=False) as old, np.load(actual, allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise AssertionError("C3 model NPZ members changed")
        for name in old.files:
            if name == "meta":
                continue
            left, right = old[name], new[name]
            identical = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
            comparisons[name] = {"dtype": str(left.dtype), "shape": list(left.shape), "bitwise_identical": identical,
                                 "reference_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                                 "retrained_sha256": hashlib.sha256(right.tobytes()).hexdigest()}
            if not identical:
                raise AssertionError(f"C3 model member {name} is not bitwise identical")
        old_meta, new_meta = json.loads(str(old["meta"])), json.loads(str(new["meta"]))
    stable = {}
    for key in ("epochs_run", "best_epoch", "val_loss", "stop_reason", "n_train_rows", "n_val_rows"):
        stable[key] = old_meta[key] == new_meta[key]
        if not stable[key]:
            raise AssertionError(f"C3 deterministic metadata changed: {key}")
    for key in ("epoch", "train_loss", "val_loss"):
        stable[f"history.{key}"] = old_meta["history"][key] == new_meta["history"][key]
        if not stable[f"history.{key}"]:
            raise AssertionError(f"C3 deterministic training history changed: {key}")
    return comparisons, stable


def c3_stage(root, evidence):
    reference = TASK_DIR / "trained" / "ego-novw-trig_s0.npz"
    output = root / "c3_seed0.npz"
    if output.exists():
        raise RuntimeError(f"refusing existing retraining model: {output}")
    evidence.update(reference=str(reference.relative_to(TASK_DIR)), output=str(output.relative_to(TASK_DIR)),
                    reference_sha256_before=tp.sha256_file(reference),
                    recipe={"model": "ego-novw-trig", "seed": 0, "trigger_copies": 16,
                            "trigger_qd_jitter": "auto", "max_epochs": tp.MAX_EPOCHS,
                            "max_train_episodes": None, "max_val_episodes": None,
                            "split_sha256": tp.sha256_file(tp.SPLIT_PATH)})
    real_train = tp.train_mlp

    def capture(X, Y, Xv, Yv, *args, **kwargs):
        evidence["C3_actual_train_mlp_matrices"] = {
            name: matrix_descriptor(value) for name, value in (("X", X), ("Y", Y), ("Xv", Xv), ("Yv", Yv))}
        print(json.dumps(evidence["C3_actual_train_mlp_matrices"], indent=2), flush=True)
        return real_train(X, Y, Xv, Yv, *args, **kwargs)

    tp.train_mlp = capture
    try:
        _, meta = tp.run_train("ego-novw-trig", seed=0, out=str(output), trigger_copies_k=16,
                               trigger_qd_jitter="auto", log=lambda line: print(line, flush=True))
    finally:
        tp.train_mlp = real_train
    comparisons, stable = compare_models(reference, output)
    evidence.update(model_members=comparisons, deterministic_metadata=stable,
                    all_non_meta_members_bitwise_identical=True,
                    json_meta_comparison="Timestamp, elapsed times, peak RSS, git/code provenance and output location "
                                         "may differ; full NPZ container bytes are not required to be identical.",
                    epochs_run=meta["epochs_run"], best_epoch=meta["best_epoch"], val_loss=meta["val_loss"],
                    peak_rss_mib=meta["peak_rss_mib"], load_s=meta["load_s"], train_s=meta["train_s"],
                    reference_sha256_after=tp.sha256_file(reference))
    if evidence["reference_sha256_before"] != evidence["reference_sha256_after"]:
        raise AssertionError("original trained C3 model changed")


def k1_stage(root, evidence):
    # Import only after C3 has completed and this fresh process has started.
    from results.step6 import gp_training as gp
    reference = root / "backward_compat_c3.json"
    original = json.loads(reference.read_text())
    if original.get("status") != "passed":
        raise RuntimeError("k=1 verification requires a passed C3 retraining stage")
    if original["shared_code_sha256_after"] != code_hashes():
        raise RuntimeError("shared code changed after C3 retraining; regenerate fresh compatibility evidence")
    evidence.update(C3_evidence_sha256=tp.sha256_file(reference),
                    gp_training_sha256=tp.sha256_file(TASK_DIR / "results" / "step6" / "gp_training.py"),
                    recipe={"model": gp.G_MODEL, "chunk_k": 1, "chunk_delta": 9,
                            "trigger_copies": 16, "trigger_qd_jitter": "auto", "full_split": True})
    data = gp.prepare_training_data(gp.G_MODEL, trigger_copies_k=16, trigger_qd_jitter="auto",
                                    chunk_k=1, chunk_delta=9)
    matrices = {}
    for name in ("X", "Y", "Xv", "Yv"):
        actual, expected = matrix_descriptor(data[name]), original["C3_actual_train_mlp_matrices"][name]
        matrices[name] = {"C3": expected, "chunk_k1": actual, "identical": actual == expected}
        if actual != expected:
            raise AssertionError(f"k=1 full training matrix {name} differs from actual C3 trainer input")
    evidence.update(matrices=matrices, all_four_full_matrices_identical=True,
                    peak_rss_mib=tp.peak_rss_mb())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("c3", "k1"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.out_dir).resolve()
    smoke = TASK_DIR / "results" / "step6" / "smoke"
    if not root.is_relative_to(smoke.resolve()):
        raise SystemExit("compatibility outputs must stay below results/step6/smoke")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"backward_compat_{args.stage}.json"
    if path.exists():
        raise SystemExit(f"refusing existing evidence {path}")
    start = time.monotonic()
    evidence = {"stage": args.stage, "status": "running", "started": now(),
                "data_access": "existing approved data/train cache and its validation split; no test access",
                "shared_code_sha256_before": code_hashes()}
    # Reserve the new evidence name before any expensive operation.
    with path.open("x") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    try:
        (c3_stage if args.stage == "c3" else k1_stage)(root, evidence)
        evidence["shared_code_sha256_after"] = code_hashes()
        if evidence["shared_code_sha256_before"] != evidence["shared_code_sha256_after"]:
            raise AssertionError("shared code changed during compatibility verification")
        evidence["status"] = "passed"
    except BaseException:
        evidence.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        evidence.update(finished=now(), wall_seconds=time.monotonic() - start)
        tp.write_json_atomic(str(path), evidence)
        print(f"compatibility {args.stage}: {evidence['status']}; evidence {path}", flush=True)


if __name__ == "__main__":
    main()
