"""Opt-in G/P labels, immutable side caches and training (step 6).

The original C3 trainer remains the default path in train_policy.py.  This module
is imported only for the two new models / commands.  No test-set discovery is
performed here; every recording comes from the approved training split.
"""
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import bc_features as bcf
import train_policy as tp

G_MODEL = "ego-novw-trig-chunk"
P_MODEL = "ego-novw-trig-pd"
MODELS = (G_MODEL, P_MODEL)
FORMAT = "bc_step6_labels_v1"
PD_NAMES = ["v", "w"] + [f"e{j}" for j in range(1, 8)] + [f"qd_des{j}" for j in range(1, 8)]
MAX_RSS_MIB = 6144
CODE_FILES = ("train_policy.py", "bc_features.py", "policies/bc_policy.py",
              "policies/pd_layer.py", "results/step6/gp_training.py")


def code_hashes():
    return {p: tp.sha256_file(os.path.join(tp.TASK_DIR, p)) for p in CODE_FILES}


def output_names(kind, k=10):
    return ([f"{o}_j{j}" for j in range(k) for o in bcf.OUTPUT_NAMES]
            if kind == "chunk" else list(PD_NAMES))


def chunk_labels(a, steps, k=10, delta=9):
    """Clean recorded [a(s), a(s+delta), ...], terminal padding, float64.

    Conversion of wheel commands and the torque cast deliberately match
    train_policy._episode_rows + cache_labels, including floating-point order.
    """
    if int(k) < 1 or int(delta) < 1:
        raise ValueError("chunk k and delta must be positive integers")
    steps = np.asarray(steps, dtype=np.int64)
    idx = np.minimum(steps[:, None] + np.arange(int(k))[None, :] * int(delta), len(a["t"]) - 1)
    v, w = bcf.wheels_to_vw(a["wheel_cmds"][idx.reshape(-1)])
    label = np.column_stack([v, w, np.asarray(a["torque_arm"][idx.reshape(-1)], dtype=np.float64)])
    return label.reshape(len(steps), int(k) * 9)


def pdref_labels(a, steps):
    """Recorded v,w and (q_des-q, qd_des), computed in float64."""
    steps = np.asarray(steps, dtype=np.int64)
    v, w = bcf.wheels_to_vw(a["wheel_cmds"][steps])
    e = np.asarray(a["q_des_arm"][steps], dtype=np.float64) - np.asarray(a["q_mes"][steps, bcf.ARM], dtype=np.float64)
    return np.column_stack([v, w, e, np.asarray(a["qd_des_arm"][steps], dtype=np.float64)])


def default_side(cache_dir, kind, delta=9):
    return os.path.join(cache_dir, f"chunk_d{delta}" if kind == "chunk" else "pdref")


def _extract_side(job):
    """One recording per worker, never a label derived from a downsampled row."""
    with np.load(job["path"], allow_pickle=False) as z:
        keys = ["t", "wheel_cmds", "torque_arm"]
        if job["kind"] == "pd_reference":
            keys += ["q_des_arm", "qd_des_arm", "q_mes", "nav_done"]
        a = {key: z[key] for key in keys}
    if len(a["t"]) != job["n_steps"]:
        raise RuntimeError(f"recording length changed: {job['path']}")
    nav_count = nav_exceeded = 0
    if job["kind"] == "pd_reference":
        from policies.pd_layer import expert_arm_params
        e = np.asarray(a["q_des_arm"], dtype=np.float64) - np.asarray(a["q_mes"][:, bcf.ARM], dtype=np.float64)
        nav = ~np.asarray(a["nav_done"], dtype=bool)
        nav_count = int(nav.sum())
        nav_exceeded = int((nav & (np.abs(e) > expert_arm_params().max_q_lead).any(axis=1)).sum())
    arrays, padded = {}, {}
    for part, steps in job["steps"].items():
        steps = np.asarray(steps, dtype=np.int64)
        arrays[part] = (chunk_labels(a, steps, job["k"], job["delta"]) if job["kind"] == "chunk"
                        else pdref_labels(a, steps))
        padded[part] = int((steps + (job["k"] - 1) * job["delta"] >= len(a["t"])).sum())
    return job["episode"], arrays, padded, nav_count, nav_exceeded, tp.sha256_file(job["path"])


def build_side_cache(kind, cache_dir=tp.CACHE_DIR, trigger_dir=tp.TRIGGER_DIR, split_path=tp.SPLIT_PATH,
                     side_dir=None, k=10, delta=9, workers=12, allow_subset=False):
    """Build only a new directory; complete existing artifacts are never overwritten.

    G stores stride-9 training, trigger and stride-9 validation blocks. Its 1 kHz
    validation j=0 labels continue to come from the unchanged original cache.
    P stores the 16 references at every original validation row.
    """
    if kind not in ("chunk", "pd_reference"):
        raise ValueError(kind)
    if kind == "chunk" and (int(k) < 1 or int(delta) not in (9, 27, 81)):
        raise ValueError("G needs k>=1 and delta in {9,27,81}")
    side_dir = side_dir or default_side(cache_dir, kind, delta)
    root = Path(side_dir)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite side cache {root}; remove an interrupted NEW side cache explicitly")
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    split = tp.load_split(split_path)
    train_dir = Path(tp.TASK_DIR) / split["train_dir"]
    tr = tp.Cache("train", cache_dir, split_path, allow_subset=allow_subset)
    va = tp.Cache("val", cache_dir, split_path, allow_subset=allow_subset)
    with open(os.path.join(trigger_dir, "manifest.json")) as f:
        tm = json.load(f)
    # This validates the complete trigger layout, not merely the manifest fields.
    tc, _ = tp.load_trigger_rows(tr.episodes, trigger_dir, split_path, allow_subset=allow_subset)
    layouts, by_episode = {}, {}
    for part, cache in (("train", tr), ("val_stride" if kind == "chunk" else "val", va)):
        ep_rows, starts, nrows = [], [], 0
        for i, e in enumerate(cache.episodes):
            steps = np.asarray(cache.cols["step"][cache.episode_slice(i)])
            if part == "val_stride":
                steps = steps[::tp.STRIDE]
            starts.append(nrows)
            ep_rows.append(len(steps))
            nrows += len(steps)
            by_episode.setdefault(int(e), {})[part] = steps
        layouts[part] = {"episodes": list(cache.episodes), "row_start": starts,
                         "n_rows_episode": ep_rows, "n_rows": nrows}
    layouts["trigger"] = {"episodes": list(tr.episodes), "row_start": list(range(len(tr.episodes))),
                           "n_rows_episode": [1] * len(tr.episodes), "n_rows": len(tr.episodes)}
    for e, step in zip(tc["episode"], tc["step"]):
        by_episode[int(e)]["trigger"] = np.array([int(step)])
    width = int(k) * 9 if kind == "chunk" else 16
    arrays = {}
    for part, lay in layouts.items():
        (root / part).mkdir()
        arrays[part] = np.lib.format.open_memmap(root / part / "Y.npy", mode="w+", dtype=np.float64,
                                                shape=(lay["n_rows"], width))
    rows = {(part, int(e)): start for part, lay in layouts.items()
            for e, start in zip(lay["episodes"], lay["row_start"])}
    jobs = [{"episode": e, "steps": steps, "path": str(train_dir / split["episodes"][str(e)]["file"]),
             "n_steps": split["episodes"][str(e)]["n_steps"], "kind": kind, "k": int(k), "delta": int(delta)}
            for e, steps in sorted(by_episode.items())]
    padded = {part: 0 for part in layouts}
    source_files, nav_count, nav_exceeded = {}, 0, 0
    workers = max(1, min(int(workers), 12))
    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers) as pool:
        for e, values, pad, nav, exceeded, source_sha in pool.imap(_extract_side, jobs, chunksize=2):
            for part, value in values.items():
                start = rows[(part, int(e))]
                arrays[part][start:start + len(value)] = value
                padded[part] += pad[part]
            nav_count += nav
            nav_exceeded += exceeded
            source_files[str(e)] = {"file": split["episodes"][str(e)]["file"], "sha256": source_sha}
    for arr in arrays.values():
        arr.flush()
    arrays.clear()
    # Revision 1: reference lead is descriptive only. Recorded references already
    # include reach-branch limits; the navigation hold must remain unrestricted.
    manifest = {"format": FORMAT, "kind": kind, "chunk_k": int(k) if kind == "chunk" else None,
                "chunk_delta": int(delta) if kind == "chunk" else None,
                "split_sha256": tp.sha256_file(split_path),
                "cache_manifest_sha256": tr.manifest_sha256,
                "trigger_manifest_sha256": tp.sha256_file(os.path.join(trigger_dir, "manifest.json")),
                "source_files": source_files, "code_sha256": code_hashes(), "parts": layouts,
                "arrays": {p: {"sha256": tp.sha256_file(root / p / "Y.npy"), "shape": [lay["n_rows"], width],
                                "dtype": "float64"} for p, lay in layouts.items()},
                "padded_rows": padded,
                "padded_fraction": {p: padded[p] / lay["n_rows"] if lay["n_rows"] else 0 for p, lay in layouts.items()},
                "navigation_lead": {"n_steps": nav_count, "exceeded_steps": nav_exceeded,
                                    "fraction": nav_exceeded / nav_count if nav_count else 0},
                "subset": bool(tr.manifest["subset"] or va.manifest["subset"] or tm["subset"]),
                "rules": ("Each segment uses the clean recorded wheel_cmds/torque_arm at s+j*delta; terminal label "
                          "padding. j0 matches original cache. Validation early stopping uses original 1kHz j0; "
                          "val_stride uses each episode steps 0::9, description only." if kind == "chunk" else
                          "v,w from clean wheel_cmds; e=float64(q_des_arm)-float64(q_mes[:,4:11]); "
                          "qd_des_arm recorded at the same step. Navigation lead checked on ALL steps of selected "
                          "train and validation recordings, not only stride rows."),
                "created": tp.now_iso(), "build_seconds": time.time() - t0, "workers": workers}
    tp.write_json_atomic(str(root / "manifest.json"), manifest)
    print(f"{kind} side cache {root}: " + ", ".join(f"{p} {v['n_rows']}" for p, v in layouts.items())
          + f"; {manifest['build_seconds']:.1f}s", flush=True)
    return manifest


def load_side(side_dir, kind, cache_dir, trigger_dir, split_path, allow_subset=False, k=10, delta=9):
    with open(os.path.join(side_dir, "manifest.json")) as f:
        m = json.load(f)
    checks = {"format": FORMAT, "kind": kind, "split_sha256": tp.sha256_file(split_path),
              "cache_manifest_sha256": tp.sha256_file(os.path.join(cache_dir, "manifest.json")),
              "trigger_manifest_sha256": tp.sha256_file(os.path.join(trigger_dir, "manifest.json")),
              "code_sha256": code_hashes()}
    if kind == "chunk":
        checks.update(chunk_k=int(k), chunk_delta=int(delta))
    for key, value in checks.items():
        if m.get(key) != value:
            raise RuntimeError(f"side cache {side_dir}: {key} mismatch; rebuild a new cache")
    if m["subset"] and not allow_subset:
        raise RuntimeError("formal training refuses a subset side cache")
    out = {}
    for part, desc in m["arrays"].items():
        p = os.path.join(side_dir, part, "Y.npy")
        if tp.sha256_file(p) != desc["sha256"]:
            raise RuntimeError(f"side cache array hash mismatch: {p}")
        a = np.load(p, mmap_mode="r")
        if list(a.shape) != desc["shape"] or a.dtype != np.dtype("float64"):
            raise RuntimeError(f"side cache array shape/dtype mismatch: {p}")
        out[part] = a
    return out, m, tp.sha256_file(os.path.join(side_dir, "manifest.json"))


def matrix_sha(a):
    h = hashlib.sha256()
    for s in range(0, len(a), 65536):
        h.update(np.ascontiguousarray(a[s:s + 65536]).tobytes())
    return h.hexdigest()


def prepare_training_data(model_name, cache_dir=tp.CACHE_DIR, split_path=tp.SPLIT_PATH,
                          trigger_dir=tp.TRIGGER_DIR, trigger_copies_k=16, trigger_qd_jitter="auto",
                          chunk_k=10, chunk_delta=9, side_dir=None, max_train_episodes=None,
                          max_val_episodes=None, allow_subset=False):
    """Actual float64 matrices sent to sklearn; k=1 follows C3 operations exactly."""
    kind = "chunk" if model_name == G_MODEL else "pd_reference"
    if model_name not in MODELS:
        raise ValueError(model_name)
    tr = tp.Cache("train", cache_dir, split_path, allow_subset=allow_subset)
    va = tp.Cache("val", cache_dir, split_path, allow_subset=allow_subset)
    sl, vl = tr.rows_of_first(max_train_episodes), va.rows_of_first(max_val_episodes)
    nt = len(tr.episodes) if max_train_episodes is None else min(max_train_episodes, len(tr.episodes))
    nv = len(va.episodes) if max_val_episodes is None else min(max_val_episodes, len(va.episodes))
    train_eps = tr.episodes[:nt]
    X, Xv = tp.cache_features(tr.cols, "ego_novw", sl), tp.cache_features(va.cols, "ego_novw", vl)
    side, sm, side_sha = None, None, None
    if kind == "chunk" and int(chunk_k) == 1:
        Y, Yv = tp.cache_labels(tr.cols, sl), tp.cache_labels(va.cols, vl)
    else:
        side_dir = side_dir or default_side(cache_dir, kind, chunk_delta)
        side, sm, side_sha = load_side(side_dir, kind, cache_dir, trigger_dir, split_path, allow_subset,
                                       chunk_k, chunk_delta)
        Y = np.array(side["train"][sl])
        Yv = tp.cache_labels(va.cols, vl) if kind == "chunk" else np.array(side["val"][vl])
    sigma_hold = tp.hold_qd_sigma(tr.cols, sl)
    fixed = tp.parse_qd_jitter(trigger_qd_jitter)
    sigma = sigma_hold if fixed is None else fixed
    tc, trigger_sha = tp.load_trigger_rows(train_eps, trigger_dir, split_path, allow_subset=allow_subset)
    xm, xs = tp.standardiser(X)
    ym, raw_ys = tp.standardiser(Y)
    weights = np.ones(Y.shape[1])
    if kind == "pd_reference":
        weights[2:] = 1 / np.sqrt(2.)
    ys = raw_ys / weights
    # j0 uses exactly C3's own mean/std, not a pooled chunk scalar.
    for a, mean, scale in ((X, xm, xs), (Xv, xm, xs), (Y, ym, ys),
                            (Yv, ym[:9] if kind == "chunk" else ym, ys[:9] if kind == "chunk" else ys)):
        tp.standardise_inplace(a, mean, scale)
    n_base = len(X)
    if trigger_copies_k:
        Xt, Yt, _ = tp.trigger_copies(tc, "ego_novw", int(trigger_copies_k), sigma)
        if side is not None:
            row_of = {int(e): i for i, e in enumerate(sm["parts"]["trigger"]["episodes"])}
            source = np.asarray(side["trigger"][[row_of[int(e)] for e in train_eps]])
            Yt = np.repeat(source, int(trigger_copies_k), axis=0)
        tp.standardise_inplace(Xt, xm, xs)
        tp.standardise_inplace(Yt, ym, ys)
        X, Y = np.concatenate([X, Xt]), np.concatenate([Y, Yt])
    # stride validation features are subsets of the already-standardised 1kHz matrix.
    Xvc = Yvc = None
    if kind == "chunk" and side is not None:
        indices = np.concatenate([np.arange(va.episode_slice(i).start, va.episode_slice(i).stop, tp.STRIDE)
                                  for i in range(nv)])
        Xvc = Xv[indices]
        n_chunk_val = sum(sm["parts"]["val_stride"]["n_rows_episode"][:nv])
        Yvc = np.array(side["val_stride"][:n_chunk_val])
        tp.standardise_inplace(Yvc, ym, ys)
    tr.release()
    va.release()
    return dict(X=X, Y=Y, Xv=Xv, Yv=Yv, Xvc=Xvc, Yvc=Yvc, xm=xm, xs=xs, ym=ym, ys=ys,
                raw_ys=raw_ys, loss_weights=weights, tr=tr, va=va, n_train_episodes=nt,
                n_val_episodes=nv, n_base_rows=n_base, trigger_cache_manifest_sha256=trigger_sha,
                side_cache_manifest_sha256=side_sha, side_manifest=sm, side_dir=side_dir,
                sigma_hold=sigma_hold, sigma=sigma, k0_steps=np.asarray(tc["step"]).tolist())


def sliced_mse(weights, biases, X, Y, width=None):
    net = dict(x_mean=0., x_scale=1., y_mean=0., y_scale=1., weights=weights, biases=biases)
    sse = 0.
    for start in range(0, len(X), tp.FWD_CHUNK):
        pred = bcf.forward(net, X[start:start + tp.FWD_CHUNK])
        if width is not None:
            pred = pred[:, :width]
        err = pred - Y[start:start + tp.FWD_CHUNK]
        sse += float(np.einsum("ij,ij->", err, err))
    return sse / Y.size


def check_memory():
    peak = tp.peak_rss_mb()
    if peak > MAX_RSS_MIB:
        raise RuntimeError(f"G/P peak RSS {peak:.1f} MiB exceeds {MAX_RSS_MIB} MiB: STOP and report")
    return peak


def run_training(model_name, seed=None, out=None, cache_dir=tp.CACHE_DIR, split_path=tp.SPLIT_PATH,
                 max_epochs=tp.MAX_EPOCHS, max_train_episodes=None, max_val_episodes=None, force=False,
                 hide_losses=False, log=print, trigger_copies_k=None, trigger_qd_jitter="auto",
                 trigger_dir=None, chunk_k=None, chunk_delta=None, side_dir=None):
    import sklearn
    seed = 0 if seed is None else int(seed)
    trigger_dir = trigger_dir or tp.TRIGGER_DIR
    kind = "chunk" if model_name == G_MODEL else "pd_reference"
    if trigger_copies_k is None or int(trigger_copies_k) < 0:
        raise SystemExit("G/P training needs --trigger-copies K (formal recipe K=16)")
    k, delta = (10 if chunk_k is None else int(chunk_k)), (9 if chunk_delta is None else int(chunk_delta))
    if kind == "pd_reference" and (chunk_k is not None or chunk_delta is not None):
        raise SystemExit("chunk parameters are only for the chunk model")
    if kind == "chunk" and (k < 1 or delta not in (9, 27, 81)):
        raise SystemExit("G needs k>=1 and delta in {9,27,81}")
    custom = max_epochs != tp.MAX_EPOCHS or max_train_episodes is not None or max_val_episodes is not None
    if custom and out is None:
        raise SystemExit("smoke settings need an explicit --out")
    out = out or tp.default_out(model_name, seed)
    if not str(out).endswith(".npz") or (os.path.exists(out) and not force):
        raise SystemExit(f"invalid or existing output: {out}")
    t0 = time.time()
    d = prepare_training_data(model_name, cache_dir, split_path, trigger_dir, trigger_copies_k,
                              trigger_qd_jitter, k, delta, side_dir, max_train_episodes,
                              max_val_episodes, allow_subset=custom)
    check_memory()
    load_s = time.time() - t0
    log(f"{model_name} s{seed}: train {len(d['X'])} rows, val {len(d['Xv'])}; loaded {load_s:.1f}s")
    model = tp.make_mlp(seed)
    stopper = tp.EarlyStopping()
    hist = {key: [] for key in ("epoch", "train_loss", "val_loss", "val_chunk_loss", "fit_s", "val_s")}
    best_w = best_b = None
    reason = "max_epochs"
    fit0 = time.time()
    with tp.single_thread():
        for epoch in range(1, int(max_epochs) + 1):
            ts = time.time()
            model.partial_fit(d["X"], d["Y"])
            tf = time.time()
            if len(model.loss_curve_) != epoch or model.t_ != epoch * len(d["X"]):
                raise RuntimeError("partial_fit did not run exactly one epoch")
            val = sliced_mse(model.coefs_, model.intercepts_, d["Xv"], d["Yv"], 9 if kind == "chunk" else None)
            vc = (sliced_mse(model.coefs_, model.intercepts_, d["Xvc"], d["Yvc"])
                  if d["Xvc"] is not None else None)
            tv = time.time()
            improved, stop = stopper.update(epoch, val)
            if improved:
                best_w = [w.copy() for w in model.coefs_]
                best_b = [b.copy() for b in model.intercepts_]
            for key, value in (("epoch", epoch), ("train_loss", float(model.loss_)), ("val_loss", val),
                                ("val_chunk_loss", vc), ("fit_s", tf - ts), ("val_s", tv - tf)):
                hist[key].append(value)
            peak = check_memory()
            loss_text = "" if hide_losses else f" train_loss {model.loss_:.6f} val_loss {val:.6f} chunk {vc}"
            log(f"epoch {epoch:3d}{loss_text} fit {tf-ts:.1f}s val {tv-tf:.1f}s peak RSS {peak:.0f} MiB")
            if not np.isfinite(val):
                raise RuntimeError("non-finite G/P validation loss")
            if stop:
                reason = f"no relative improvement > {tp.REL_TOL:g} for {tp.PATIENCE} consecutive epochs"
                break
    check = sliced_mse(best_w, best_b, d["Xv"], d["Yv"], 9 if kind == "chunk" else None)
    if abs(check - stopper.best) > 1e-12 * max(1., check):
        raise RuntimeError("best weights do not match stopping loss")
    meta = {"model": model_name, "kind": "mlp", "feature_set": "ego_novw", "feature_names": bcf.FEATURE_SETS["ego_novw"],
            "output_names": output_names(kind, k), "output_representation": kind, "seed": seed,
            "split_sha256": tp.sha256_file(split_path), "cache_manifest_sha256": d["tr"].manifest_sha256,
            "side_cache_manifest_sha256": d["side_cache_manifest_sha256"],
            "side_cache": os.path.relpath(os.path.abspath(d["side_dir"]), tp.TASK_DIR) if d["side_dir"] else None,
            "stride": int(d["tr"].split["stride"]), "subset": bool(custom), "smoke": bool(custom),
            "n_train_rows": len(d["X"]), "n_val_rows": len(d["Xv"]),
            "n_train_episodes": d["n_train_episodes"], "n_val_episodes": d["n_val_episodes"],
            "trigger": {"copies": int(trigger_copies_k), "n_episodes": d["n_train_episodes"],
                        "n_rows_added": int(trigger_copies_k) * d["n_train_episodes"],
                        "n_train_rows_base": d["n_base_rows"], "n_train_rows_total": len(d["X"]),
                        "qd_sigma": d["sigma"].tolist(), "qd_sigma_hold": d["sigma_hold"].tolist(),
                        "qd_jitter": trigger_qd_jitter, "jitter_seed": tp.TRIGGER_JITTER_SEED,
                        "k0_steps": d["k0_steps"], "trigger_cache_manifest_sha256": d["trigger_cache_manifest_sha256"]},
            "hyperparameters": {**tp.MLP_PARAMS, "hidden_layer_sizes": list(tp.MLP_PARAMS["hidden_layer_sizes"]),
                                "max_epochs": int(max_epochs), "patience": tp.PATIENCE, "rel_tol": tp.REL_TOL,
                                "random_state": f"np.random.RandomState({seed})", "dtype": "float64"},
            "standardisation": "original stride-9 rows only; independent per-output mean/std; trigger copies appended afterwards",
            "label_std_train": d["raw_ys"].tolist(), "loss_weights": d["loss_weights"].tolist(),
            "val_loss_definition": ("j=0 standardised MSE on all 1kHz validation steps, exactly C3's 9-output quantity"
                                    if kind == "chunk" else "standardised 16-output MSE; arm dimensions weight 1/sqrt(2)"),
            "epochs_run": len(hist["epoch"]), "best_epoch": stopper.best_epoch, "val_loss": stopper.best,
            "stop_reason": reason, "history": hist, "code_sha256": code_hashes(),
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "created": tp.now_iso(), "load_s": load_s, "train_s": time.time() - fit0,
            "peak_rss_mib": check_memory(), "memory_end_mib": tp.mem_status(),
            "n_params": int(sum(w.size + b.size for w, b in zip(best_w, best_b)))}
    if kind == "chunk":
        meta.update(chunk_k=k, chunk_delta=delta, ensemble_m=.11,
                    val_chunk_loss=(sliced_mse(best_w, best_b, d["Xvc"], d["Yvc"]) if d["Xvc"] is not None else check),
                    chunk_padding=d["side_manifest"]["padded_fraction"] if d["side_manifest"] else None)
    tp.save_trained(out, "ego_novw", d["xm"], d["xs"], d["ym"], d["ys"], best_w, best_b, meta)
    log(f"saved {out}; {meta['epochs_run']} epochs, best {meta['best_epoch']}, peak {meta['peak_rss_mib']:.0f}MiB")
    return out, meta


def offline_metrics(model_path, cache_dir=tp.CACHE_DIR, split_path=tp.SPLIT_PATH, allow_subset=False, log=print):
    """G current-action validation; P fixed-PD torque RMSE and reference nRMSE.

    P converts every recorded validation step through the very same deployment
    layer.  The simulator is created once to obtain model and limits, and is not
    stepped.  All observations and labels are recorded training data.
    """
    t0 = time.time()
    model = bcf.load_model(model_path)
    meta = model["meta"]
    kind = meta["output_representation"]
    if meta["split_sha256"] != tp.sha256_file(split_path):
        raise RuntimeError("model/split mismatch")
    va = tp.Cache("val", cache_dir, split_path, allow_subset=allow_subset)
    X = tp.cache_features(va.cols, model["feature_set"])
    pred = np.empty((len(X), len(model["y_mean"])))
    with tp.single_thread():
        for start in range(0, len(X), tp.FWD_CHUNK):
            pred[start:start + tp.FWD_CHUNK] = bcf.forward(model, X[start:start + tp.FWD_CHUNK])
    del X
    action = tp.cache_labels(va.cols)
    ref_sq = np.zeros(16)
    ref_n = 0
    sim = client = None
    if kind == "chunk":
        action_pred = pred[:, :9]
    else:
        from evaluate_policy import setup_simulation
        from policies.pd_layer import pd_layer, q_pin_from_obs
        sim, client, _, info = setup_simulation(np.zeros(2), 0., gui=False)
        action_pred = np.empty_like(action)
        action_pred[:, :2] = pred[:, :2]
        split = tp.load_split(split_path)
        train_dir = Path(tp.TASK_DIR) / split["train_dir"]
        try:
            with tp.single_thread():
                for i, e in enumerate(va.episodes):
                    sl = va.episode_slice(i)
                    with np.load(train_dir / split["episodes"][str(e)]["file"]) as z:
                        a = {key: z[key] for key in ("t", "base_pos", "base_ori", "q_mes", "qd_mes",
                                                     "wheel_cmds", "q_des_arm", "qd_des_arm")}
                    steps = np.asarray(va.cols["step"][sl], dtype=int)
                    label = pdref_labels(a, steps)
                    ref_sq += np.sum((pred[sl] - label) ** 2, axis=0)
                    ref_n += len(steps)
                    for row, step in zip(range(sl.start, sl.stop), steps):
                        obs = {key: a[key][step] for key in ("base_pos", "base_ori", "q_mes", "qd_mes")}
                        qpin = q_pin_from_obs(obs, info)
                        action_pred[row, 2:] = pd_layer(pred[row, 2:9], pred[row, 9:16],
                                                        obs["q_mes"][bcf.ARM], obs["qd_mes"][bcf.ARM], qpin, info)
        finally:
            if client is not None:
                client.disconnect()
    err2 = (action_pred - action) ** 2
    ann = tp.annotate_val(va)
    noisy = np.asarray(va.cols["noisy"], dtype=bool)
    high = np.asarray(va.cols["target"][:, 2]) >= .45
    torque = []
    for group, group_mask in (("all", np.ones(va.n_rows, dtype=bool)), ("clean", ~noisy), ("noisy", noisy)):
        for si, segment in enumerate(tp.SEGMENTS):
            for zname, zval in (("z<0.45", False), ("z>=0.45", True)):
                mask = group_mask & (ann["seg"] == si) & (high == zval)
                n = int(mask.sum())
                sm = err2[mask, 2:].sum(axis=0)
                torque.append({"group": group, "segment": segment, "z": zname, "n": n,
                               "rmse": dict(zip(bcf.OUTPUT_NAMES[2:], np.sqrt(sm / n))) if n else None,
                               "rmse_pooled": float(np.sqrt(sm.sum() / (7 * n))) if n else None})
    result = {"model": meta["model"], "seed": meta["seed"], "feature_set": model["feature_set"],
              "output_representation": kind, "model_sha256": tp.sha256_file(model_path),
              "n_val_rows": va.n_rows, "n_val_episodes": len(va.episodes), "torque_segments": torque,
              "val_loss_meta": meta["val_loss"], "seconds": time.time() - t0, "created": tp.now_iso()}
    if kind == "chunk":
        result["val_loss"] = float(np.mean(err2 / np.asarray(model["y_scale"][:9]) ** 2))
        result["val_chunk_loss"] = meta["val_chunk_loss"]
    else:
        raw_std = np.asarray(meta["label_std_train"])
        nrmse = np.sqrt(ref_sq / ref_n) / raw_std
        result["reference_nrmse"] = {"per_output": dict(zip(PD_NAMES, nrmse)),
                                      "e_mean": float(nrmse[2:9].mean()),
                                      "qd_des_mean": float(nrmse[9:16].mean())}
        result["val_loss"] = float(np.mean(ref_sq / (ref_n * np.asarray(model["y_scale"]) ** 2)))
    log(f"{meta['model']} s{meta['seed']}: offline {va.n_rows} rows, {result['seconds']:.1f}s")
    return result
