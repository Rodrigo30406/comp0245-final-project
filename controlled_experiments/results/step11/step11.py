"""Step 11 (2026-11-06): three supplementary checks on the experimental line, pre-registered in
results/step11/prereg_step11_draft.md (frozen copy: prereg_step11_frozen.txt).

  R  placebo copies: the K = 16 jittered copies come from a random base-moving row instead of the onset row k0
  S  two networks on the A-to-B task: a base net without arm state (v, w) and an arm net (tau1..7), arm with / without
     the copies (the step-9 L1 / L1-K0 design on the main task)
  W  capacity: hidden layers (256, 256, 256) instead of (128, 128), with (K16) and without (K0) the copies

No existing file is edited: this module imports train_policy / bc_features and only adds the feature set
"ego_target" (the first six ego_novw columns, the target geometry) to bc_features.FEATURE_SETS at import.

  python results/step11/step11.py build-placebo-cache
  python train_policy.py train --model ego-novw-trig --trigger-copies 16 \
      --trigger-cache results/step11/cache/placebo --seed S --out trained/step11/R_sS.npz          # R
  python results/step11/step11.py train-s --net base|arm-K0|arm-K16 --seed S                         # S
  python results/step11/step11.py train-w --variant K0|K16 --seed S                                  # W
"""
import argparse
import json
import os
import sys
import time

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(os.path.dirname(STEP_DIR))
for _d in (TASK_DIR, os.path.join(TASK_DIR, "policies")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

import bc_features as bcf  # noqa: E402
import train_policy as tp  # noqa: E402
from split_policy import register_ego_target  # noqa: E402

register_ego_target()   # feature set ego_target = the first six ego_novw columns (registry + build_features branch)
TARGET_FEATURES = bcf.FEATURE_SETS["ego_target"]
assert TARGET_FEATURES == ["ego_dx", "ego_dy", "target_z", "rho", "cos_beta", "sin_beta"], TARGET_FEATURES

TRAINED_DIR = os.path.join(TASK_DIR, "trained", "step11")
PLACEBO_DIR = os.path.join(STEP_DIR, "cache", "placebo")
K = 16
PLACEBO_GAP = 1000        # steps (1 s at 1 kHz): the placebo row is at least this far before the latch k0
PLACEBO_RNG = 11          # numpy default_rng([PLACEBO_RNG, episode])
W_HIDDEN = (256, 256, 256)

# S networks: (feature set, feature columns of ego_novw, output columns of the 9 labels, output names, copies)
S_NETS = {
    "base": ("ego_target", slice(0, 6), [0, 1], ["v", "w"], K),
    "arm-K16": ("ego_novw", slice(0, 20), list(range(2, 9)), bcf.OUTPUT_NAMES[2:], K),
    "arm-K0": ("ego_novw", slice(0, 20), list(range(2, 9)), bcf.OUTPUT_NAMES[2:], 0),
}


def log(s):
    print(s, flush=True)


# ------------------------------------------------------------------------------------------------------------------
# R: the placebo cache (same format as the trigger cache, so train_policy's C3 path reads it unchanged)
# ------------------------------------------------------------------------------------------------------------------
def placebo_step(nav_done, episode):
    """(chosen step, onset k0, rule used) for one 1 kHz recording."""
    nav = np.asarray(nav_done, dtype=bool)
    if not nav.any():
        raise ValueError("no nav_done step")
    k0 = int(np.argmax(nav))
    rng = np.random.default_rng([PLACEBO_RNG, int(episode)])
    cand = np.flatnonzero(~nav[:max(0, k0 - PLACEBO_GAP + 1)])
    rule = "moving>=1s"
    if cand.size == 0 and k0 > 0:
        cand, rule = np.arange(k0), "before_k0"
    if cand.size == 0:                       # k0 = 0: the episode starts at its parking pose
        cand, rule = np.arange(1, nav.size), "k0_zero_any"
    return int(rng.choice(cand)), k0, rule


def build_placebo_cache(out_dir=PLACEBO_DIR, split_path=tp.SPLIT_PATH):
    split = tp.load_split(split_path)
    train_dir = os.path.join(TASK_DIR, split["train_dir"])
    eps = list(split["train_episodes"])
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    t0 = time.time()
    arrays = {name: np.lib.format.open_memmap(os.path.join(out_dir, name + ".npy"), mode="w+", dtype=dt,
                                              shape=(len(eps), *shape)) for name, (dt, shape) in tp.CACHE_COLUMNS.items()}
    steps, k0s, rules, dts = [], [], [], set()
    for i, e in enumerate(eps):
        job = {"path": os.path.join(train_dir, split["episodes"][str(e)]["file"]), "episode": int(e), "part": "trigger",
               "offset": 0, "stride": 1, "n_steps": int(split["episodes"][str(e)]["n_steps"]),
               "noisy": bool(split["episodes"][str(e)]["noisy"])}
        a, meta = tp._load_episode(job)
        k, k0, rule = placebo_step(a["nav_done"], e)
        out = tp._episode_rows(a, np.array([k]), job)
        for name, arr in out.items():
            arrays[name][i:i + 1] = arr
        steps.append(k), k0s.append(k0), rules.append(rule), dts.add(float(meta["time_step"]))
        if (i + 1) % 400 == 0:
            log(f"  {i + 1}/{len(eps)} episodes ({time.time() - t0:.0f} s)")
    nav_chosen = np.asarray(arrays["nav_done"]).tolist()
    label_v = np.asarray(arrays["label_v"], dtype=float)
    for arr in arrays.values():
        arr.flush()
    arrays.clear()
    if len(dts) != 1:
        raise ValueError(f"episodes with different time steps: {dts}")
    counts = {r: rules.count(r) for r in sorted(set(rules))}
    manifest = {
        "format": tp.TRIGGER_FORMAT, "created": tp.now_iso(),
        "split_file": os.path.relpath(split_path, TASK_DIR), "split_sha256": tp.sha256_file(split_path),
        "time_step": dts.pop(), "subset": False, "episodes": eps, "k0_steps": steps, "n_rows": len(eps),
        "onset_k0_steps": k0s, "placebo_rule_per_episode": rules, "placebo_rule_counts": counts,
        "k0_zero_episodes": [e for e, k0 in zip(eps, k0s) if k0 == 0],
        "rule": f"step 11 R PLACEBO cache (not the trigger frame): one row per training episode, the recorded step k "
                f"drawn uniformly by numpy default_rng([{PLACEBO_RNG}, episode]) from {{k : nav_done False, "
                f"k <= k0 - {PLACEBO_GAP}}} (k0 = first nav_done step); if empty and k0 > 0 from {{k < k0}}; if k0 = 0 "
                f"from {{k >= 1}}. 'k0_steps' holds the CHOSEN steps (field name kept for the trigger-cache reader); "
                f"'onset_k0_steps' the true onsets. Columns and labels as recorded.",
        "columns": {name: {"dtype": np.dtype(dt).name, "shape": list(shape)} for name, (dt, shape) in
                    tp.CACHE_COLUMNS.items()},
        "checks": {"n_chosen_nav_done_true": int(sum(bool(x) for x in nav_chosen)),
                   "min_gap_to_k0_moving_rule": int(min((k0 - k) for k, k0, r in zip(steps, k0s, rules)
                                                        if r == "moving>=1s")),
                   "frac_label_v_nonzero": float(np.mean(np.abs(label_v) > 0))},
        "build_seconds": time.time() - t0,
    }
    tp.write_json_atomic(manifest_path, manifest)
    log(f"placebo cache {out_dir}: {len(eps)} rows, rules {counts}, checks {manifest['checks']}, "
        f"{manifest['build_seconds']:.0f} s")
    return manifest


def load_placebo_rows(episodes, trigger_dir=PLACEBO_DIR, split_path=tp.SPLIT_PATH, allow_subset=False):
    """train_policy.load_trigger_rows for the placebo cache: the same checks, except that the rows must be the
    manifest's chosen steps with nav_done False (only the two k0 = 0 episodes may be nav_done) instead of nav_done."""
    manifest_path = os.path.join(trigger_dir, "manifest.json")
    with open(manifest_path) as f:
        m = json.load(f)
    split = tp.load_split(split_path)
    if m.get("format") != tp.TRIGGER_FORMAT or "placebo_rule_counts" not in m:
        raise RuntimeError(f"{manifest_path}: not a step-11 placebo cache")
    if m["split_sha256"] != tp.sha256_file(split_path) or list(m["episodes"]) != list(split["train_episodes"]):
        raise RuntimeError(f"placebo cache {trigger_dir}: built from another split")
    row_of = {int(e): i for i, e in enumerate(m["episodes"])}
    rows = np.array([row_of[int(e)] for e in episodes], dtype=np.int64)
    cols = {name: np.asarray(np.load(os.path.join(trigger_dir, name + ".npy"), mmap_mode="r")[rows])
            for name in tp.CACHE_COLUMNS}
    zero = set(m["k0_zero_episodes"])
    ok_nav = all((not nd) or int(e) in zero for e, nd in zip(cols["episode"], cols["nav_done"]))
    if (not np.array_equal(cols["episode"], np.asarray(episodes, dtype=np.int32))
            or not np.array_equal(cols["step"], np.asarray(m["k0_steps"])[rows]) or not ok_nav):
        raise RuntimeError(f"placebo cache {trigger_dir}: rows do not match the episodes / chosen steps")
    return cols, tp.sha256_file(manifest_path)


def train_r(seed, out=None, force=False):
    """C3's training path (train_policy.run_train, model ego-novw-trig, K = 16, jitter auto) with the placebo rows in
    place of the trigger rows: only the row loader differs."""
    out = out or os.path.join(TRAINED_DIR, f"R_s{seed}.npz")
    tp.load_trigger_rows = load_placebo_rows
    return tp.run_train("ego-novw-trig", seed, out, force=force, log=log, trigger_copies_k=K,
                        trigger_dir=PLACEBO_DIR)


# ------------------------------------------------------------------------------------------------------------------
# S: one network of the split (feature / output subsets of the ego_novw training data)
# ------------------------------------------------------------------------------------------------------------------
def train_s(net, seed, out=None, force=False):
    feature_set, fi, oi, out_names, copies = S_NETS[net]
    out = out or os.path.join(TRAINED_DIR, f"S-{net}_s{seed}.npz")
    if os.path.exists(out) and not force:
        raise SystemExit(f"{out} exists (pass --force to overwrite)")
    import sklearn
    t_start = time.time()
    X, Y, Xv, Yv, tr, va = tp.load_training_data("ego_novw")
    xm, xs = tp.standardiser(X)      # column-wise: slicing afterwards equals standardising the sliced columns
    ym, ys = tp.standardiser(Y)
    xm, xs, ym, ys = xm[fi].copy(), xs[fi].copy(), ym[oi].copy(), ys[oi].copy()
    X, Xv = np.ascontiguousarray(X[:, fi]), np.ascontiguousarray(Xv[:, fi])
    Y, Yv = np.ascontiguousarray(Y[:, oi]), np.ascontiguousarray(Yv[:, oi])
    for A, m_, s_ in ((X, xm, xs), (Y, ym, ys), (Xv, xm, xs), (Yv, ym, ys)):
        tp.standardise_inplace(A, m_, s_)
    n_base = int(X.shape[0])
    trig = None
    if copies > 0:
        tr_cols = tp.Cache("train", tp.CACHE_DIR, tp.SPLIT_PATH)
        sigma = tp.hold_qd_sigma(tr_cols.cols, tr.rows_of_first(None))
        tr_cols.release()
        train_eps = list(tr.episodes)
        trig_cols, trig_sha = tp.load_trigger_rows(train_eps, tp.TRIGGER_DIR, tp.SPLIT_PATH)
        Xt, Yt, _ = tp.trigger_copies(trig_cols, "ego_novw", copies, sigma)   # bit-identical to C3's copies
        Xt, Yt = np.ascontiguousarray(Xt[:, fi]), np.ascontiguousarray(Yt[:, oi])
        tp.standardise_inplace(Xt, xm, xs)
        tp.standardise_inplace(Yt, ym, ys)
        X, Y = np.concatenate([X, Xt]), np.concatenate([Y, Yt])
        trig = {"copies": copies, "n_rows_added": int(Xt.shape[0]), "qd_sigma": np.asarray(sigma).tolist(),
                "jitter_seed": tp.TRIGGER_JITTER_SEED, "trigger_cache_manifest_sha256": trig_sha}
        del Xt, Yt
    t_load = time.time() - t_start
    log(f"S-{net} seed {seed}: features {bcf.FEATURE_SETS[feature_set]}, outputs {out_names}, train {X.shape[0]} rows "
        f"({n_base} + copies {X.shape[0] - n_base}), val {Xv.shape[0]} rows, loaded in {t_load:.1f} s")
    t_fit = time.time()
    weights, biases, hist, _ = tp.train_mlp(X, Y, Xv, Yv, seed, log=log)
    check = tp.std_mse(weights, biases, Xv, Yv)
    assert abs(check - hist["best_val_loss"]) <= 1e-12 * max(1.0, check), "best weights do not match"
    commit, dirty = tp.git_state()
    meta = {
        "model": f"S-{net}", "experiment": "step11 S (two networks on A-to-B)", "kind": "mlp",
        "feature_set": feature_set, "feature_names": bcf.FEATURE_SETS[feature_set], "output_names": out_names,
        "seed": seed, "split_sha256": tp.sha256_file(tp.SPLIT_PATH), "cache_manifest_sha256": tr.manifest_sha256,
        "n_train_rows": int(X.shape[0]), "n_train_rows_base": n_base, "n_val_rows": int(Xv.shape[0]),
        **({"trigger": trig} if trig else {}),
        "standardisation": "X and Y columns with the mean and std of the original stride-9 training rows only",
        "val_loss_definition": f"standardised MSE over all validation rows (1 kHz) and the {len(oi)} outputs",
        "hyperparameters": {**tp.MLP_PARAMS, "hidden_layer_sizes": list(tp.MLP_PARAMS["hidden_layer_sizes"]),
                            "random_state": f"np.random.RandomState({seed})", "patience": tp.PATIENCE,
                            "rel_tol": tp.REL_TOL},
        "epochs_run": hist["epochs_run"], "best_epoch": hist["best_epoch"], "val_loss": hist["best_val_loss"],
        "stop_reason": hist["stop_reason"],
        "history": {k: hist[k] for k in ("epoch", "train_loss", "val_loss", "fit_s", "val_s")},
        "n_params": int(sum(w.size + b.size for w, b in zip(weights, biases))),
        "git_commit": commit, "git_dirty": dirty,
        "code_sha256": {f: tp.sha256_file(os.path.join(TASK_DIR, f))
                        for f in ("train_policy.py", "bc_features.py", "results/step11/step11.py")},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "created": tp.now_iso(),
        "load_s": t_load, "train_s": time.time() - t_fit,
    }
    tp.save_trained(out, feature_set, xm, xs, ym, ys, weights, biases, meta)
    log(f"saved {out}: {hist['epochs_run']} epochs, best {hist['best_epoch']}, val loss {hist['best_val_loss']:.6f}, "
        f"train {meta['train_s']:.0f} s")
    return out


# ------------------------------------------------------------------------------------------------------------------
# W: the C3 / ego-novw recipes with a wider and deeper MLP
# ------------------------------------------------------------------------------------------------------------------
def train_w(variant, seed, out=None, force=False):
    tp.MLP_PARAMS["hidden_layer_sizes"] = W_HIDDEN      # read by make_mlp and by run_train's metadata at call time
    out = out or os.path.join(TRAINED_DIR, f"W-{variant}_s{seed}.npz")
    if variant == "K0":
        return tp.run_train("ego-novw", seed, out, force=force, log=log)
    if variant == "K16":
        return tp.run_train("ego-novw-trig", seed, out, force=force, log=log, trigger_copies_k=K)
    raise SystemExit(f"unknown W variant {variant!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-placebo-cache")
    r = sub.add_parser("train-r")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("train-s")
    s.add_argument("--net", required=True, choices=sorted(S_NETS))
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--force", action="store_true")
    w = sub.add_parser("train-w")
    w.add_argument("--variant", required=True, choices=("K0", "K16"))
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "build-placebo-cache":
        build_placebo_cache()
    elif a.cmd == "train-r":
        train_r(a.seed, force=a.force)
    elif a.cmd == "train-s":
        train_s(a.net, a.seed, force=a.force)
    elif a.cmd == "train-w":
        train_w(a.variant, a.seed, force=a.force)


if __name__ == "__main__":
    main()
