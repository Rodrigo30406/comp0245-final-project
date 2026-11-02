"""Step 4 E (descriptive; STEP4_PLAN_PROMPT.md 3): learning curve and downsampling stride on the C3 recipe
(ego_novw features, K trigger copies per training episode). Kept apart from train_policy.py, which experiment D froze;
this module only imports it.

Variants (seed 0-2 each; the f = 1, s = 9 point is the existing ego-novw-trig_s0-2):
  ego-novw-trig-f0125 / -f025 / -f05   nested stratified subsets of the training episodes (1/8, 1/4, 1/2), stride 9, K 16
  ego-novw-trig-s27 / -s81             all training episodes, stride 27 / 81 (own split variant and caches), K 5 / 2
                                       (K = round(16 * 9 / s): the copies stay about 1.5% of the training rows)
Training: the C3 network and early stopping, except that the validation check comes after a fixed number of training
samples (about one stride-9 epoch of the full training set, 1.79M): f 1/8, 1/4, 1/2 call partial_fit 8 / 4 / 2 times
per check, s 27 / 81 3 / 9 times. Patience 5 checks, at most 200 checks; the weights of the best check are kept.
So the curves change the amount / density of data, not the number of optimisation steps between checks.

  python train_step4e.py prepare                       # split variants, their caches and trigger caches, subsets
  python train_step4e.py train --model ego-novw-trig-f025 --seed 0
  python train_step4e.py train-many --models all --seeds 0-2 --jobs 8
"""
import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse
import json
import subprocess
import sys
import time

import numpy as np

import bc_features as bcf
import train_policy as tp

TASK_DIR = tp.TASK_DIR
SUBSETS_PATH = os.path.join(tp.TRAINED_DIR, "data_subsets.json")
SUBSET_SEED = 1
FRACTIONS = {"f0125": 1 / 8, "f025": 1 / 4, "f05": 1 / 2}
STRIDES = (27, 81)
SAMPLES_PER_CHECK_REF = 1764684 + 16 * 1613   # rows of one ego-novw-trig (stride 9, all episodes) epoch
MAX_CHECKS, PATIENCE = 200, 5


def split_path_of(stride):
    return tp.SPLIT_PATH if stride == 9 else os.path.join(tp.TRAINED_DIR, f"data_split_s{stride}.json")


def cache_dir_of(stride):
    return tp.CACHE_DIR if stride == 9 else os.path.join(TASK_DIR, "results", "step3", f"cache_s{stride}")


def spec_of(model):
    """Data specification of an E variant: split, cache, trigger cache, K, subset fraction, partial_fit calls per check."""
    tag = model.rsplit("-", 1)[-1]
    if tag in FRACTIONS:
        f = FRACTIONS[tag]
        return {"stride": 9, "fraction": f, "copies": 16, "calls": int(round(1 / f)), "tag": tag}
    if tag.startswith("s") and int(tag[1:]) in STRIDES:
        s = int(tag[1:])
        return {"stride": s, "fraction": 1.0, "copies": int(round(16 * 9 / s)), "calls": int(round(s / 9)), "tag": tag}
    raise SystemExit(f"unknown E variant {model}")


ALL_MODELS = [f"ego-novw-trig-{t}" for t in FRACTIONS] + [f"ego-novw-trig-s{s}" for s in STRIDES]


def make_subsets(split):
    """Nested stratified subsets of the training episodes: per stratum (noisy x mode, fixed order) one permutation
    from numpy default_rng(SUBSET_SEED); the subset of fraction f takes round(f n) episodes of each stratum's prefix."""
    rng = np.random.default_rng(SUBSET_SEED)
    train = set(split["train_episodes"])
    perms = {}
    for noisy in (False, True):
        for mode in tp.MODES:
            key = f"{'noisy' if noisy else 'clean'}/{mode}"
            eps = sorted(e for e in train if split["episodes"][str(e)]["noisy"] == noisy
                         and split["episodes"][str(e)]["mode"] == mode)
            perms[key] = [int(e) for e in rng.permutation(np.array(eps, dtype=int))]
    subsets = {}
    for tag, f in FRACTIONS.items():
        subsets[tag] = sorted(e for key, p in perms.items() for e in p[:int(round(f * len(p)))])
    return {"format": "bc_subsets_v1", "split_sha256": tp.sha256_file(tp.SPLIT_PATH), "seed": SUBSET_SEED,
            "rule": "per stratum (clean M0, M1, M2, noisy M0, M1, M2) one numpy default_rng(1) permutation of the "
                    "training episodes; the subset of fraction f is the union of the first round(f n) episodes",
            "fractions": FRACTIONS, "subsets": subsets}


def prepare(workers=12):
    split = tp.load_split()
    if not os.path.exists(SUBSETS_PATH):
        tp.write_json_atomic(SUBSETS_PATH, make_subsets(split))
    sub = json.load(open(SUBSETS_PATH))
    for a, b in zip(["f0125", "f025"], ["f025", "f05"]):
        assert set(sub["subsets"][a]) <= set(sub["subsets"][b]), "subsets must be nested"
    print("subsets: " + ", ".join(f"{t} {len(v)} episodes" for t, v in sub["subsets"].items()))
    for s in STRIDES:
        path = split_path_of(s)
        if not os.path.exists(path):
            tp.write_split(path, stride=s)
        v = tp.load_split(path)
        assert v["train_episodes"] == split["train_episodes"] and v["val_episodes"] == split["val_episodes"]
        cdir = cache_dir_of(s)
        if not os.path.exists(os.path.join(cdir, "manifest.json")):
            tp.build_cache(path, cdir, workers=workers)
        tdir = os.path.join(cdir, "trigger")
        if not os.path.exists(os.path.join(tdir, "manifest.json")):
            tp.build_trigger_cache(path, tdir, workers=workers)
        # the trigger rows come from the 1 kHz recording: identical to the stride-9 trigger cache
        a, _ = tp.load_trigger_rows(split["train_episodes"], tdir, path)
        b, _ = tp.load_trigger_rows(split["train_episodes"], tp.TRIGGER_DIR, tp.SPLIT_PATH)
        same = all(np.array_equal(a[k], b[k]) for k in tp.CACHE_COLUMNS)
        print(f"stride {s}: split {os.path.relpath(path, TASK_DIR)}, cache {os.path.relpath(cdir, TASK_DIR)}, trigger "
              f"rows identical to stride 9: {same}")
        if not same:
            raise SystemExit("trigger caches differ")


def train(model_name, seed, out=None, force=False, log=print):
    sp = spec_of(model_name)
    out = out or os.path.join(tp.TRAINED_DIR, f"{model_name}_s{seed}.npz")
    if os.path.exists(out) and not force:
        raise SystemExit(f"{out} exists")
    split_path, cache_dir = split_path_of(sp["stride"]), cache_dir_of(sp["stride"])
    trigger_dir = os.path.join(cache_dir, "trigger")
    t0 = time.time()
    X, Y, Xv, Yv, tr, va = tp.load_training_data("ego_novw", cache_dir, split_path)
    tr_cols = tp.Cache("train", cache_dir, split_path)
    ep_col = np.asarray(tr_cols.cols["episode"])
    if sp["fraction"] < 1:
        sub = json.load(open(SUBSETS_PATH))
        if sub["split_sha256"] != tp.sha256_file(tp.SPLIT_PATH):
            raise SystemExit("data_subsets.json belongs to another split")
        train_eps = sub["subsets"][sp["tag"]]
        mask = np.isin(ep_col, train_eps)
        X, Y = X[mask], Y[mask]
    else:
        train_eps = list(tr.episodes)
        mask = np.ones(len(ep_col), dtype=bool)
    nav = np.asarray(tr_cols.cols["nav_done"], dtype=bool)
    qd = np.asarray(tr_cols.cols["qd_arm"], dtype=np.float64)
    sigma = qd[mask & ~nav].std(axis=0)          # sigma_hold of this variant's own training rows
    tr_cols.release()
    xm, xs = tp.standardiser(X)
    ym, ys = tp.standardiser(Y)
    for A, m_, s_ in ((X, xm, xs), (Y, ym, ys), (Xv, xm, xs), (Yv, ym, ys)):
        tp.standardise_inplace(A, m_, s_)
    n_base = X.shape[0]
    trig, trig_sha = tp.load_trigger_rows(train_eps, trigger_dir, split_path)
    Xt, Yt, _ = tp.trigger_copies(trig, "ego_novw", sp["copies"], sigma)
    tp.standardise_inplace(Xt, xm, xs)
    tp.standardise_inplace(Yt, ym, ys)
    X, Y = np.concatenate([X, Xt]), np.concatenate([Y, Yt])
    log(f"{model_name} seed {seed}: {len(train_eps)} episodes, {n_base} rows + {len(Xt)} copies (K {sp['copies']}), "
        f"{sp['calls']} partial_fit calls per check ({sp['calls'] * X.shape[0]} samples), loaded in {time.time() - t0:.1f} s")
    model = tp.make_mlp(seed)
    stopper = tp.EarlyStopping(PATIENCE, tp.REL_TOL)
    best_w = best_b = None
    hist = {"check": [], "val_loss": [], "train_loss": [], "t_samples": []}
    stop_reason = "max_checks"
    calls = 0
    with tp.single_thread():
        for check in range(1, MAX_CHECKS + 1):
            for _ in range(sp["calls"]):
                model.partial_fit(X, Y)
                calls += 1
                if model.t_ != calls * X.shape[0]:
                    raise RuntimeError("partial_fit did not run exactly one pass")
            val = tp.std_mse(model.coefs_, model.intercepts_, Xv, Yv)
            new_best, stop = stopper.update(check, val)
            if new_best:
                best_w, best_b = [c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_]
            for k, v in (("check", check), ("val_loss", val), ("train_loss", float(model.loss_)),
                         ("t_samples", int(model.t_))):
                hist[k].append(v)
            log(f"check {check:3d} samples {model.t_} val_loss {val:.6f} (best {stopper.best:.6f} @ {stopper.best_epoch})")
            if stop:
                stop_reason = f"no relative improvement > {tp.REL_TOL:g} for {PATIENCE} consecutive checks"
                break
    best = stopper.best_epoch
    meta = {"model": model_name, "kind": "mlp", "feature_set": "ego_novw", "seed": seed, "step4e": sp,
            "split_file": os.path.relpath(split_path, TASK_DIR), "split_sha256": tp.sha256_file(split_path),
            "subsets_sha256": tp.sha256_file(SUBSETS_PATH) if sp["fraction"] < 1 else None,
            "n_train_episodes": len(train_eps), "n_train_rows_base": n_base, "n_train_rows": int(X.shape[0]),
            "n_val_rows": int(Xv.shape[0]), "trigger": {"copies": sp["copies"], "n_rows_added": int(len(Xt)),
                                                        "qd_sigma": sigma.tolist(), "jitter_seed": tp.TRIGGER_JITTER_SEED,
                                                        "trigger_cache_manifest_sha256": trig_sha},
            "standardisation": "X and Y with the mean and std of this variant's own stride / subset training rows",
            "hyperparameters": {**tp.MLP_PARAMS, "hidden_layer_sizes": list(tp.MLP_PARAMS["hidden_layer_sizes"]),
                                "random_state": f"np.random.RandomState({seed})", "patience_checks": PATIENCE,
                                "max_checks": MAX_CHECKS, "partial_fit_calls_per_check": sp["calls"]},
            "checks_run": len(hist["check"]), "best_check": best, "epochs_run": calls, "best_epoch": best,
            "t_samples_at_best": hist["t_samples"][best - 1] if best else None,
            "val_loss": stopper.best, "stop_reason": stop_reason, "history": hist,
            "code_sha256": {f: tp.sha256_file(os.path.join(TASK_DIR, f))
                            for f in ("train_step4e.py", "train_policy.py", "bc_features.py")},
            "created": tp.now_iso(), "train_s": time.time() - t0}
    tp.save_trained(out, "ego_novw", xm, xs, ym, ys, best_w, best_b, meta)
    log(f"saved {out}: {len(hist['check'])} checks, best check {best}, val loss {stopper.best:.6f}, "
        f"{time.time() - t0:.0f} s")
    return out


def train_many(models, seeds, jobs):
    todo = [(m, s) for m in models for s in seeds
            if not os.path.exists(os.path.join(tp.TRAINED_DIR, f"{m}_s{s}.npz"))]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    running, failed = [], []
    while todo or running:
        while todo and len(running) < jobs:
            m, s = todo.pop(0)
            log_path = os.path.join(tp.TRAINED_DIR, f"{m}_s{s}.log")
            fh = open(log_path, "w")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "train", "--model", m, "--seed", str(s)],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=TASK_DIR)
            running.append((m, s, p, fh))
            print(f"started {m} seed {s}", flush=True)
        time.sleep(2.0)
        for item in list(running):
            m, s, p, fh = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                print(f"{m} seed {s} finished with code {p.returncode}", flush=True)
                if p.returncode:
                    failed.append((m, s))
    if failed:
        raise SystemExit(f"failed: {failed}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("train")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("train-many")
    p.add_argument("--models", default="all")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.workers)
    elif args.cmd == "train":
        train(args.model, args.seed, args.out, args.force, log=lambda s: print(s, flush=True))
    else:
        models = ALL_MODELS if args.models == "all" else args.models.split(",")
        train_many(models, tp.parse_seeds(args.seeds), args.jobs)


if __name__ == "__main__":
    main()
