"""Training pipeline of the behaviour-cloning policies (step 3): episode split, signal cache, MLP / OLS training and
offline metrics. Features and the model file format come from bc_features.py (shared with the deployed policies).

Subcommands:
  split        create trained/data_split.json once: the kept data/train episodes, the stratified episode split
               (85% train / 15% validation, noisy x mode) and one downsampling offset per training episode
  build-cache  extract the measured signals and labels of the training rows (stride 9, per-episode offset) and of
               all 1 kHz validation steps into results/step3/cache/ (one memory-mappable .npy per signal)
  build-trigger-cache
               extract the trigger frame k0 (first nav_done step, 1 kHz recording) of every training episode into
               results/step3/cache/trigger/ (experiment C3)
  build-ee-cache
               side cache of the recorded end-effector position for every row of the training / validation cache
               and of the trigger cache -> results/step3/cache/ee/ (experiment D; the other caches are not touched)
  train        train one model: ego, ego-novw, raw (MLP 2x128, early stopping on the validation set),
               ego-ols, raw-ols (least squares), ego-novw-trig (ego-novw plus K jittered copies of each training
               episode's trigger frame, --trigger-copies K; experiment C3) or ego-novw-trig-ee (the same with the
               near-field end-effector error features ego_novw_ee; experiment D; needs the ee side cache)
  train-many   train one MLP model for several seeds, one single-threaded process per seed
  offline      offline metrics of a saved model on the validation rows -> trained/<name>.offline.json

Examples:
  python train_policy.py split
  python train_policy.py build-cache --workers 12
  python train_policy.py build-trigger-cache --workers 12
  python train_policy.py train --model ego --seed 0                  # -> trained/ego_s0.npz
  python train_policy.py train-many --model ego --seeds 0-7 --jobs 8 # -> trained/ego_s{0..7}.npz (+ .log)
  python train_policy.py train --model ego-ols                       # -> trained/ego-ols.npz
  python train_policy.py train --model ego-novw-trig --trigger-copies 64 --seed 0 --out trained/ego-novw-trig-k64_s0.npz
  python train_policy.py offline --model trained/ego_s0.npz          # -> trained/ego_s0.offline.json

Every model and every training seed uses exactly the same training rows (the split file); the cache refuses to
load when its manifest does not match the split file. Training runs single-threaded (BLAS threads = 1), so several
seeds can run in parallel.

Trigger-frame copies (experiment C3, STEP3C_PLAN_PROMPT.md 3.1): the only change to the training rows. The
standardisation parameters are computed from the original stride-9 rows only, then K copies of each training
episode's recorded trigger frame k0 are appended (measured signals and labels exactly as recorded: v = w = 0 and the
torque of the expert's first reach step). Each copy's arm qd gets N(0, sigma_hold_j) per joint, sigma_hold = the
per-joint std of arm qd over the training rows with nav_done False; the noise comes from one fixed numpy
default_rng(TRIGGER_JITTER_SEED), drawn once, so every training seed sees the same added rows. With K = 0 the
training is exactly ego-novw's (same rows, same arrays, bit-identical weights for the same seed).
"""
import os

if __name__ == "__main__":
    # single-threaded BLAS: must be set before numpy is imported (threadpool_limits below also guards it at runtime)
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse
import csv
import datetime
import hashlib
import json
import multiprocessing as mp
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

import bc_features as bcf

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(TASK_DIR, "data", "train")
TRAINED_DIR = os.path.join(TASK_DIR, "trained")
SPLIT_PATH = os.path.join(TRAINED_DIR, "data_split.json")
CACHE_DIR = os.path.join(TASK_DIR, "results", "step3", "cache")
TRIGGER_DIR = os.path.join(CACHE_DIR, "trigger")

# ==============================================================================
# Settings of the approved step-3 plan (STEP3_PLAN_PROMPT.md section 3.1)
# ==============================================================================
N_KEPT = 1897            # data/train episodes with status == reached and nav_timed_out == False
VAL_FRAC = 0.15          # validation share per stratum (noisy x mode)
SPLIT_SEED = 0           # numpy default_rng seed of the validation draw
DATA_SEED = 0            # numpy default_rng seed of the per-episode downsampling offsets
STRIDE = 9               # odd on purpose: the low-z torque chatter alternates sign every step
MODES = ("M0", "M1", "M2")

MODELS = {  # model name -> (kind, feature set)
    "ego": ("mlp", "ego"),
    "ego-novw": ("mlp", "ego_novw"),
    "raw": ("mlp", "raw"),
    "ego-ols": ("ols", "ego"),
    "raw-ols": ("ols", "raw"),
    "ego-novw-trig": ("mlp", "ego_novw"),   # experiment C3: ego-novw + K copies of the trigger frame
    "ego-novw-trig-ee": ("mlp", "ego_novw_ee"),   # experiment D: C3 + the near-field end-effector error features
    "ego-novw-trig-dag1": ("mlp", "ego_novw"),   # experiment F: C3 + one round of shadow-expert rows
    "ego-novw-trig-chunk": ("mlp", "ego_novw"),  # opt-in experiment G: action blocks
    "ego-novw-trig-pd": ("mlp", "ego_novw"),     # opt-in experiment P: joint references + fixed PD
}
TRIGGER_MODELS = ("ego-novw-trig", "ego-novw-trig-ee", "ego-novw-trig-dag1",
                  "ego-novw-trig-chunk", "ego-novw-trig-pd")
TRIGGER_JITTER_SEED = 0  # numpy default_rng seed of the arm-qd jitter of the trigger copies (shared by all seeds)
MLP_PARAMS = {"hidden_layer_sizes": (128, 128), "activation": "relu", "solver": "adam", "learning_rate_init": 1e-3,
              "alpha": 1e-4, "batch_size": 200, "shuffle": True}
MAX_EPOCHS = 200
PATIENCE = 5             # epochs without a relative improvement of REL_TOL of the best validation loss
REL_TOL = 1e-3
SCALE_EPS = 1e-12        # standard deviation below this -> scale 1
CHUNK = 262144           # rows per chunk of the feature build
FWD_CHUNK = 4096         # rows per chunk of the forward passes (cache-friendly: ~1.5x faster than 262144 rows)

# Offline metrics
EVENT_TYPES = ["start", "align>drive", "drive>orient", "drive>reach", "orient>reach"]
NAV_NAMES = ["align", "drive", "orient", "reach"]                     # phase code, 3 = nav_done
CLASSES = ["align", "drive", "orient", "reach_move", "reach_hold"]
SEGMENTS = ["nav_hold", "reach_move", "reach_hold"]                   # torque segments
LAG_MAX = 30             # steps searched on each side: k in [-LAG_MAX, LAG_MAX]
LAG_PRE = 0.020          # s before the event included in the lag windows
LAG_FLAT_TOL = 1e-3      # relative range (max - min) / min of an MSE(k) curve below this -> flat, no k_best

# Cache columns: name -> (dtype, trailing shape). Raw signals keep the recorded dtype (float32 / float64 target),
# so the cache is lossless; the (v, w) labels are computed in float64 from the recorded wheel commands.
CACHE_COLUMNS = {
    "episode": (np.int32, ()), "step": (np.int32, ()), "t": (np.float32, ()),
    "base_xy": (np.float32, (2,)), "base_bearing": (np.float32, ()),
    "q_arm": (np.float32, (7,)), "qd_arm": (np.float32, (7,)), "qd_wheels": (np.float32, (4,)),
    "target": (np.float64, (3,)),
    "label_v": (np.float64, ()), "label_w": (np.float64, ()), "label_tau": (np.float32, (7,)),
    "phase": (np.int8, ()), "nav_done": (np.bool_, ()), "ee_err": (np.float64, ()), "noisy": (np.bool_, ()),
}
CACHE_FORMAT = "bc_cache_v1"
SPLIT_FORMAT = "bc_split_v1"
TRIGGER_FORMAT = "bc_trigger_v1"
EE_FORMAT = "bc_ee_v1"       # side cache <cache_dir>/ee/{train,val,trigger}/ee_pos.npy (experiment D)


# ==============================================================================
# Small helpers
# ==============================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: kB


def mem_status():
    """Current memory of this process in MiB from /proc/self/status: peak RSS (VmHWM), RSS, anonymous (private)
    and file-backed (memory-mapped cache, shared page cache) parts. Empty dict where /proc is missing."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key = line.split(":")[0]
                if key in ("VmHWM", "VmRSS", "RssAnon", "RssFile"):
                    out[key] = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return out


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TASK_DIR, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "train_policy.py", "bc_features.py"],
                               cwd=TASK_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
        return commit or None, bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return None, None


def to_json(obj):
    """Recursively convert numpy scalars / arrays to plain Python for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def write_json_atomic(path, obj, indent=1):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_json(obj), f, indent=indent)
        f.write("\n")
    os.replace(tmp, path)


def single_thread():
    """Context manager limiting BLAS / OpenMP to one thread (no-op if threadpoolctl is missing)."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


def parse_seeds(text):
    """'0-7' / '0,2,5' / '0-3,8' -> sorted list of ints."""
    seeds = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.update(range(int(a), int(b) + 1))
        elif part:
            seeds.add(int(part))
    return sorted(seeds)


def default_out(model, seed):
    kind, _ = MODELS[model]
    name = f"{model}_s{seed}" if kind == "mlp" else model
    return os.path.join(TRAINED_DIR, name + ".npz")


# ==============================================================================
# Split
# ==============================================================================
def standoff_of(z):
    from robotnik_arm_task import STANDOFF_POINTS, STANDOFF_Z_POINTS
    return float(np.interp(z, STANDOFF_Z_POINTS, STANDOFF_POINTS))


def read_kept_episodes(index_path):
    """Kept episodes of a dataset index (status == reached and nav_timed_out == False), with the per-episode
    information the split needs: noisy, mode (analyze_dataset.mode_of of d0 = r - standoff(z)), d0, z, n_steps."""
    from analyze_dataset import mode_of
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))
    kept = {}
    for r in rows:
        if r["status"] != "reached" or r["nav_timed_out"] != "False":
            continue
        z = float(r["target_z"])
        d0 = float(r["r"]) - standoff_of(z)
        noisy = float(r["noise_v"]) > 0 or float(r["noise_w"]) > 0 or float(r["noise_torque"]) > 0
        kept[int(r["episode"])] = {"file": r["file"], "n_steps": int(r["n_steps"]), "noisy": bool(noisy),
                                   "mode": mode_of(d0), "d0": d0, "z": z}
    return kept


def stratum_key(info):
    return f"{'noisy' if info['noisy'] else 'clean'}/{info['mode']}"


def make_split(train_dir=TRAIN_DIR, split_seed=SPLIT_SEED, data_seed=DATA_SEED, stride=STRIDE, val_frac=VAL_FRAC,
               expected=N_KEPT):
    """Episode split as a dict (deterministic given the index and the seeds)."""
    index_path = os.path.join(train_dir, "index.csv")
    kept = read_kept_episodes(index_path)
    if expected is not None and len(kept) != expected:
        raise AssertionError(f"{index_path}: {len(kept)} kept episodes, expected {expected}")
    rng = np.random.default_rng(split_seed)
    strata, val = {}, []
    for noisy in (False, True):        # fixed stratum order, so the draw is reproducible
        for mode in MODES:
            key = f"{'noisy' if noisy else 'clean'}/{mode}"
            eps = sorted(e for e, info in kept.items() if info["noisy"] == noisy and info["mode"] == mode)
            n_val = round(val_frac * len(eps))
            drawn = sorted(int(e) for e in rng.choice(np.array(eps, dtype=int), size=n_val, replace=False)) \
                if n_val else []
            val += drawn
            strata[key] = {"n": len(eps), "n_val": n_val, "n_train": len(eps) - n_val}
    val = sorted(val)
    train = sorted(set(kept) - set(val))
    offsets = np.random.default_rng(data_seed).integers(0, stride, size=len(train))
    return {
        "format": SPLIT_FORMAT,
        "train_dir": os.path.relpath(train_dir, TASK_DIR),
        "index_sha256": sha256_file(index_path),
        "filter": "status == reached and nav_timed_out == False",
        "n_kept": len(kept),
        "kept_episodes": sorted(kept),
        "split_seed": split_seed, "data_seed": data_seed, "stride": stride, "val_frac": val_frac,
        "stratify": "noisy x mode; mode = analyze_dataset.mode_of(d0), d0 = r - standoff(z) "
                    "(np.interp over robotnik_arm_task.STANDOFF_Z_POINTS / STANDOFF_POINTS)",
        "split_rule": "per stratum (clean M0, M1, M2, noisy M0, M1, M2 in this order), n_val = round(val_frac * n) "
                      "episodes drawn without replacement with one numpy default_rng(split_seed)",
        "offset_rule": "one offset in [0, stride) per training episode (ascending episode order), "
                       "numpy default_rng(data_seed).integers(0, stride); training rows = steps offset::stride",
        "strata": strata,
        "train_episodes": train,
        "val_episodes": val,
        "train_offsets": [int(o) for o in offsets],
        "episodes": {str(e): kept[e] for e in sorted(kept)},
    }


def split_core(split):
    """The parts of a split that define the data (everything except the creation time)."""
    return {k: v for k, v in split.items() if k != "created"}


def write_split(path=SPLIT_PATH, force=False, **kwargs):
    if os.path.exists(path) and not force:
        raise SystemExit(f"{path} exists; every model reuses it. Pass --force to overwrite (invalidates the cache "
                         f"and all models trained on it).")
    split = make_split(**kwargs)
    split["created"] = now_iso()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json_atomic(path, split)
    return split


def load_split(path=SPLIT_PATH):
    with open(path) as f:
        split = json.load(f)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"{path}: not a {SPLIT_FORMAT} split file")
    return split


# ==============================================================================
# Cache
# ==============================================================================
def _load_episode(job):
    """The recorded arrays of one kept episode (checked against the split's n_steps and filter) and its meta."""
    with np.load(job["path"]) as d:
        a = {k: d[k] for k in ("t", "base_pos", "base_bearing", "q_mes", "qd_mes", "wheel_cmds", "torque_arm",
                               "phase", "nav_done", "ee_pos", "target", "meta")}
    meta = json.loads(str(a["meta"]))
    n = len(a["t"])
    if n != job["n_steps"] or meta["log_every"] != 1 or meta["status"] != "reached" or meta["nav_timed_out"]:
        raise ValueError(f"{job['path']}: unexpected episode (n {n} vs {job['n_steps']}, log_every "
                         f"{meta['log_every']}, status {meta['status']}, nav_timed_out {meta['nav_timed_out']})")
    return a, meta


def _extract_episode(job):
    """Worker: the cached columns of one episode (training rows offset::stride, or every step)."""
    a, meta = _load_episode(job)
    idx = np.arange(job["offset"], len(a["t"]), job["stride"])
    return job["part"], job["episode"], _episode_rows(a, idx, job), float(meta["time_step"])


def _episode_rows(a, idx, job):
    """The CACHE_COLUMNS of the recorded steps idx of one episode (recorded dtypes; (v, w) labels in float64)."""
    target = np.asarray(a["target"], dtype=np.float64)
    v, w = bcf.wheels_to_vw(a["wheel_cmds"][idx])
    return {
        "episode": np.full(len(idx), job["episode"], dtype=np.int32), "step": idx.astype(np.int32),
        "t": a["t"][idx], "base_xy": a["base_pos"][idx, :2], "base_bearing": a["base_bearing"][idx],
        "q_arm": a["q_mes"][idx, bcf.ARM], "qd_arm": a["qd_mes"][idx, bcf.ARM],
        "qd_wheels": a["qd_mes"][idx, bcf.WHEELS], "target": np.broadcast_to(target, (len(idx), 3)),
        "label_v": v, "label_w": w, "label_tau": a["torque_arm"][idx],
        "phase": a["phase"][idx], "nav_done": a["nav_done"][idx],
        "ee_err": np.linalg.norm(a["ee_pos"][idx].astype(np.float64) - target, axis=1),
        "noisy": np.full(len(idx), job["noisy"], dtype=np.bool_),
    }


def n_rows_of(n_steps, offset, stride):
    return len(range(offset, n_steps, stride))


def _extract_trigger(job):
    """Worker: the CACHE_COLUMNS of one training episode's trigger frame k0 = its first nav_done step (1 kHz
    recording). k0 = 0 happens when the episode starts at its parking pose (nav_done from the first step: training
    episodes 747 and 880, both M0); that step is then the trigger frame."""
    a, meta = _load_episode(job)
    nav = np.asarray(a["nav_done"], dtype=bool)
    if not nav.any():
        raise ValueError(f"{job['path']}: no nav_done step")
    k0 = int(np.argmax(nav))
    return job["episode"], _episode_rows(a, np.array([k0]), job), float(meta["time_step"])


def build_trigger_cache(split_path=SPLIT_PATH, trigger_dir=TRIGGER_DIR, workers=12, train_episodes=None,
                        verbose=True):
    """One row per training episode of the split (or of a subset, for tests): the CACHE_COLUMNS of its trigger frame
    k0 (first nav_done step) read from the 1 kHz recording, in the order of split['train_episodes'] -> trigger_dir
    (one .npy per column + manifest.json). Returns the manifest."""
    split = load_split(split_path)
    train_dir = os.path.join(TASK_DIR, split["train_dir"])
    eps = list(split["train_episodes"])
    subset = train_episodes is not None
    if subset:
        assert set(train_episodes) <= set(eps), "train subset not in the split"
        eps = sorted(train_episodes)
    os.makedirs(trigger_dir, exist_ok=True)
    manifest_path = os.path.join(trigger_dir, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)  # a half-written cache must never look valid
    t_start = time.time()
    jobs = [{"path": os.path.join(train_dir, split["episodes"][str(e)]["file"]), "episode": int(e), "part": "trigger",
             "offset": 0, "stride": 1, "n_steps": int(split["episodes"][str(e)]["n_steps"]),
             "noisy": bool(split["episodes"][str(e)]["noisy"])} for e in eps]
    arrays = {name: np.lib.format.open_memmap(os.path.join(trigger_dir, name + ".npy"), mode="w+", dtype=dt,
                                              shape=(len(eps), *shape)) for name, (dt, shape) in CACHE_COLUMNS.items()}
    pos = {e: i for i, e in enumerate(eps)}
    time_steps = set()
    workers = max(1, min(int(workers), 12))
    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers) as pool:
        for i, (e, out, dt) in enumerate(pool.imap(_extract_trigger, jobs, chunksize=2)):
            for name, arr in out.items():
                arrays[name][pos[e]:pos[e] + 1] = arr
            time_steps.add(dt)
            if verbose and (i + 1) % 400 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes ({time.time() - t_start:.0f} s)", flush=True)
    for arr in arrays.values():
        arr.flush()
    steps = np.asarray(arrays["step"]).tolist()
    arrays.clear()
    if len(time_steps) != 1:
        raise ValueError(f"episodes with different time steps: {time_steps}")
    manifest = {
        "format": TRIGGER_FORMAT, "created": now_iso(),
        "split_file": os.path.relpath(split_path, TASK_DIR), "split_sha256": sha256_file(split_path),
        "time_step": time_steps.pop(), "subset": subset, "episodes": eps, "k0_steps": steps, "n_rows": len(eps),
        "k0_zero_episodes": [e for e, k in zip(eps, steps) if k == 0],
        "rule": "one row per training episode (order of split train_episodes): the recorded step k0 = first step with "
                "nav_done (k0 = 0 when nav_done holds from the first step); columns and dtypes as the training cache "
                "(CACHE_COLUMNS), labels = the recorded clean commands of step k0",
        "columns": {name: {"dtype": np.dtype(dt).name, "shape": list(shape)} for name, (dt, shape) in
                    CACHE_COLUMNS.items()},
        "build_seconds": time.time() - t_start, "workers": workers,
    }
    write_json_atomic(manifest_path, manifest)
    if verbose:
        print(f"trigger cache {trigger_dir}: {len(eps)} episodes, k0 step median {np.median(steps):.0f}, "
              f"{manifest['build_seconds']:.1f} s with {workers} workers")
    return manifest


def load_trigger_rows(episodes, trigger_dir=TRIGGER_DIR, split_path=SPLIT_PATH, allow_subset=False):
    """The trigger-frame rows (dict column -> array, in the given episode order) of the training episodes `episodes`
    from the trigger cache, checked against the split file. Returns (columns, manifest sha256)."""
    manifest_path = os.path.join(trigger_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"no trigger cache manifest in {trigger_dir}: run 'train_policy.py build-trigger-cache'")
    with open(manifest_path) as f:
        m = json.load(f)
    split = load_split(split_path)
    if m.get("format") != TRIGGER_FORMAT:
        raise RuntimeError(f"{manifest_path}: not a {TRIGGER_FORMAT} cache")
    if m["split_sha256"] != sha256_file(split_path):
        raise RuntimeError(f"trigger cache {trigger_dir} was built from another split file than {split_path}")
    if m["subset"] and not allow_subset:
        raise RuntimeError(f"trigger cache {trigger_dir} holds a subset of the split (test cache)")
    if not m["subset"] and list(m["episodes"]) != list(split["train_episodes"]):
        raise RuntimeError(f"trigger cache {trigger_dir}: episodes differ from the split file's training episodes")
    if set(m["columns"]) != set(CACHE_COLUMNS):
        raise RuntimeError(f"trigger cache {trigger_dir}: columns differ from CACHE_COLUMNS")
    row_of = {int(e): i for i, e in enumerate(m["episodes"])}
    missing = [e for e in episodes if int(e) not in row_of]
    if missing:
        raise RuntimeError(f"trigger cache {trigger_dir}: no row for training episodes {missing[:5]}")
    rows = np.array([row_of[int(e)] for e in episodes], dtype=np.int64)
    cols = {}
    for name in CACHE_COLUMNS:
        arr = np.load(os.path.join(trigger_dir, name + ".npy"), mmap_mode="r")
        if arr.shape[0] != m["n_rows"]:
            raise RuntimeError(f"trigger cache {trigger_dir}/{name}.npy: {arr.shape[0]} rows, manifest {m['n_rows']}")
        cols[name] = np.asarray(arr[rows])
    if not np.array_equal(cols["episode"], np.asarray(episodes, dtype=np.int32)) or not cols["nav_done"].all():
        raise RuntimeError(f"trigger cache {trigger_dir}: rows do not match the episodes or are not nav_done")
    return cols, sha256_file(manifest_path)


def build_cache(split_path=SPLIT_PATH, cache_dir=CACHE_DIR, workers=12, train_episodes=None, val_episodes=None,
                verbose=True):
    """Extract the cache for the split (or for a subset of its episodes, for tests). Returns the manifest."""
    split = load_split(split_path)
    train_dir = os.path.join(TASK_DIR, split["train_dir"])
    stride = int(split["stride"])
    offsets = dict(zip(split["train_episodes"], split["train_offsets"]))
    parts = {"train": list(split["train_episodes"]), "val": list(split["val_episodes"])}
    subset = train_episodes is not None or val_episodes is not None
    if train_episodes is not None:
        assert set(train_episodes) <= set(parts["train"]), "train subset not in the split"
        parts["train"] = sorted(train_episodes)
    if val_episodes is not None:
        assert set(val_episodes) <= set(parts["val"]), "val subset not in the split"
        parts["val"] = sorted(val_episodes)
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)  # a half-written cache must never look valid
    t_start = time.time()
    jobs, layout, arrays = [], {}, {}
    for part, eps in parts.items():
        rows, starts, total = [], [], 0
        for e in eps:
            info = split["episodes"][str(e)]
            off, st = (offsets[e], stride) if part == "train" else (0, 1)
            k = n_rows_of(info["n_steps"], off, st)
            starts.append(total)
            rows.append(k)
            total += k
            jobs.append({"path": os.path.join(train_dir, info["file"]), "episode": int(e), "part": part,
                         "offset": int(off), "stride": int(st), "n_steps": int(info["n_steps"]),
                         "noisy": bool(info["noisy"])})
        layout[part] = {"episodes": eps, "row_start": starts, "n_rows_episode": rows, "n_rows": total}
        os.makedirs(os.path.join(cache_dir, part), exist_ok=True)
        arrays[part] = {name: np.lib.format.open_memmap(os.path.join(cache_dir, part, name + ".npy"), mode="w+",
                                                        dtype=dt, shape=(total, *shape))
                        for name, (dt, shape) in CACHE_COLUMNS.items()}
    pos = {part: dict(zip(lay["episodes"], lay["row_start"])) for part, lay in layout.items()}
    time_steps = set()
    workers = max(1, min(int(workers), 12))
    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers) as pool:
        for i, (part, e, out, dt) in enumerate(pool.imap(_extract_episode, jobs, chunksize=2)):
            s = pos[part][e]
            for name, arr in out.items():
                arrays[part][name][s:s + len(arr)] = arr
            time_steps.add(dt)
            if verbose and (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes ({time.time() - t_start:.0f} s)", flush=True)
    for cols in arrays.values():
        for arr in cols.values():
            arr.flush()
    arrays.clear()
    if len(time_steps) != 1:
        raise ValueError(f"episodes with different time steps: {time_steps}")
    size = sum(os.path.getsize(os.path.join(cache_dir, part, name + ".npy"))
               for part in layout for name in CACHE_COLUMNS)
    manifest = {
        "format": CACHE_FORMAT, "created": now_iso(),
        "split_file": os.path.relpath(split_path, TASK_DIR), "split_sha256": sha256_file(split_path),
        "stride": stride, "time_step": time_steps.pop(), "subset": subset,
        "columns": {name: {"dtype": np.dtype(dt).name, "shape": list(shape)} for name, (dt, shape) in
                    CACHE_COLUMNS.items()},
        "parts": layout, "n_rows": {part: lay["n_rows"] for part, lay in layout.items()},
        "per_row_episode_id": "column 'episode' (int32); rows of an episode are contiguous, in step order "
                              "(column 'step'); parts.<part>.row_start / n_rows_episode give the ranges",
        "target_z": "column target[:, 2]",
        "build_seconds": time.time() - t_start, "workers": workers, "size_bytes": size,
    }
    write_json_atomic(manifest_path, manifest)
    if verbose:
        print(f"cache {cache_dir}: train {layout['train']['n_rows']} rows ({len(parts['train'])} episodes), "
              f"val {layout['val']['n_rows']} rows ({len(parts['val'])} episodes), {size / 2**20:.0f} MiB, "
              f"{manifest['build_seconds']:.1f} s with {workers} workers")
    return manifest


class Cache:
    """Memory-mapped cache of one part ('train' or 'val'), checked against the split file."""

    def __init__(self, part, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, allow_subset=False):
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise RuntimeError(f"no cache manifest in {cache_dir}: run 'train_policy.py build-cache'")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        split = load_split(split_path)
        m = self.manifest
        if m.get("format") != CACHE_FORMAT:
            raise RuntimeError(f"{manifest_path}: not a {CACHE_FORMAT} cache")
        if m["split_sha256"] != sha256_file(split_path):
            raise RuntimeError(f"cache {cache_dir} was built from another split file than {split_path}: "
                               f"rebuild it with 'train_policy.py build-cache'")
        if int(m["stride"]) != int(split["stride"]):
            raise RuntimeError("cache stride differs from the split file")
        if m["subset"] and not allow_subset:
            raise RuntimeError(f"cache {cache_dir} holds a subset of the split (test cache)")
        expected = split["train_episodes"] if part == "train" else split["val_episodes"]
        lay = m["parts"][part]
        if not m["subset"] and list(lay["episodes"]) != list(expected):
            raise RuntimeError(f"cache {cache_dir}: {part} episodes differ from the split file")
        if set(m["columns"]) != set(CACHE_COLUMNS):
            raise RuntimeError(f"cache {cache_dir}: columns differ from CACHE_COLUMNS")
        self.part, self.cache_dir, self.split = part, cache_dir, split
        self.episodes = list(lay["episodes"])
        self.row_start = np.asarray(lay["row_start"], dtype=np.int64)
        self.n_rows_episode = np.asarray(lay["n_rows_episode"], dtype=np.int64)
        self.n_rows = int(lay["n_rows"])
        self.time_step = float(m["time_step"])
        self.manifest_sha256 = sha256_file(manifest_path)
        self.cols = {}
        for name in CACHE_COLUMNS:
            arr = np.load(os.path.join(cache_dir, part, name + ".npy"), mmap_mode="r")
            if arr.shape[0] != self.n_rows:
                raise RuntimeError(f"cache {cache_dir}/{part}/{name}.npy: {arr.shape[0]} rows, manifest {self.n_rows}")
            self.cols[name] = arr

    def release(self):
        """Drop the memory maps (their pages stop counting in this process's RSS)."""
        self.cols = {}

    def rows_of_first(self, n_episodes):
        """Row slice of the first n_episodes episodes (None: all)."""
        if n_episodes is None or n_episodes >= len(self.episodes):
            return slice(0, self.n_rows)
        return slice(0, int(self.row_start[n_episodes]))

    def episode_slice(self, i):
        s = int(self.row_start[i])
        return slice(s, s + int(self.n_rows_episode[i]))


def signals_from_cache(cols, sl):
    """bc_features.measured_signals of cached rows (the cached arm / wheel columns are placed back into the
    recorded q_mes / qd_mes layout, so the features are built by exactly the deployment code)."""
    base_xy = np.asarray(cols["base_xy"][sl])
    n = base_xy.shape[0]
    base_pos = np.zeros((n, 3))
    base_pos[:, :2] = base_xy
    q = np.zeros((n, 17))
    q[:, bcf.ARM] = cols["q_arm"][sl]
    qd = np.zeros((n, 17))
    qd[:, bcf.ARM] = cols["qd_arm"][sl]
    qd[:, bcf.WHEELS] = cols["qd_wheels"][sl]
    ee_pos = np.asarray(cols["ee_pos"][sl]) if "ee_pos" in cols else None   # only for FEATURES_NEED_EE (side cache)
    return bcf.measured_signals(base_pos, np.asarray(cols["base_bearing"][sl]), q, qd, np.asarray(cols["target"][sl]),
                                ee_pos=ee_pos)


def cache_features(cols, feature_set, sl=None, chunk=CHUNK):
    """(N, F) float64 features of the cached rows sl, built with bc_features.build_features in chunks."""
    sl = sl or slice(0, cols["episode"].shape[0])
    start, stop = sl.start or 0, sl.stop
    X = np.empty((stop - start, len(bcf.FEATURE_SETS[feature_set])))
    for s in range(start, stop, chunk):
        e = min(stop, s + chunk)
        X[s - start:e - start] = bcf.build_features(feature_set, signals_from_cache(cols, slice(s, e)))
    return X


def cache_labels(cols, sl=None):
    """(N, 9) float64 labels (v, w, tau1..tau7) of the cached rows sl."""
    sl = sl or slice(0, cols["episode"].shape[0])
    return np.column_stack([cols["label_v"][sl], cols["label_w"][sl],
                            np.asarray(cols["label_tau"][sl], dtype=np.float64)])


def _extract_ee(job):
    """Worker: the recorded ee_pos (float32) of one episode at the given steps."""
    with np.load(job["path"]) as d:
        n = len(d["t"])
        if n != job["n_steps"]:
            raise ValueError(f"{job['path']}: {n} steps, split says {job['n_steps']}")
        return job["part"], job["episode"], d["ee_pos"][np.asarray(job["steps"])]


def ee_dir_of(cache_dir):
    return os.path.join(cache_dir, "ee")


def build_ee_cache(cache_dir=CACHE_DIR, trigger_dir=TRIGGER_DIR, split_path=SPLIT_PATH, workers=12, allow_subset=False,
                   verbose=True):
    """Side cache of the recorded end-effector position (float32, as recorded) for every row of the training and
    validation parts of the cache in cache_dir and of the trigger cache in trigger_dir, row-aligned with them, in
    <cache_dir>/ee/{train,val,trigger}/ee_pos.npy. The existing caches are only read. The manifest records the
    sha256 of the manifests it belongs to (load_ee refuses another cache). Returns the manifest."""
    split = load_split(split_path)
    train_dir = os.path.join(TASK_DIR, split["train_dir"])
    ee_dir = ee_dir_of(cache_dir)
    os.makedirs(ee_dir, exist_ok=True)
    manifest_path = os.path.join(ee_dir, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    t_start = time.time()
    trig_manifest_path = os.path.join(trigger_dir, "manifest.json")
    with open(trig_manifest_path) as f:
        trig_m = json.load(f)
    parts, jobs, subset = {}, [], False
    for part in ("train", "val"):
        cache = Cache(part, cache_dir, split_path, allow_subset=allow_subset)
        subset = subset or bool(cache.manifest["subset"])
        steps = np.asarray(cache.cols["step"])
        parts[part] = {"episodes": cache.episodes, "n_rows": cache.n_rows}
        for i, e in enumerate(cache.episodes):
            sl = cache.episode_slice(i)
            jobs.append({"part": part, "episode": int(e), "row": sl.start, "steps": steps[sl].tolist(),
                         "path": os.path.join(train_dir, split["episodes"][str(e)]["file"]),
                         "n_steps": int(split["episodes"][str(e)]["n_steps"])})
        cache.release()
    if trig_m["split_sha256"] != sha256_file(split_path) or (trig_m["subset"] and not allow_subset):
        raise RuntimeError(f"trigger cache {trigger_dir} does not belong to {split_path}")
    parts["trigger"] = {"episodes": list(trig_m["episodes"]), "n_rows": int(trig_m["n_rows"])}
    for i, (e, k0) in enumerate(zip(trig_m["episodes"], trig_m["k0_steps"])):
        jobs.append({"part": "trigger", "episode": int(e), "row": i, "steps": [int(k0)],
                     "path": os.path.join(train_dir, split["episodes"][str(e)]["file"]),
                     "n_steps": int(split["episodes"][str(e)]["n_steps"])})
    arrays = {}
    for part, lay in parts.items():
        os.makedirs(os.path.join(ee_dir, part), exist_ok=True)
        arrays[part] = np.lib.format.open_memmap(os.path.join(ee_dir, part, "ee_pos.npy"), mode="w+",
                                                 dtype=np.float32, shape=(lay["n_rows"], 3))
    row_of = {(j["part"], j["episode"]): j["row"] for j in jobs}
    with mp.get_context("forkserver").Pool(max(1, min(int(workers), 12))) as pool:
        for part, e, ee in pool.imap(_extract_ee, jobs, chunksize=4):
            r = row_of[(part, e)]
            arrays[part][r:r + len(ee)] = ee
    for arr in arrays.values():
        arr.flush()
    arrays.clear()
    manifest = {
        "format": EE_FORMAT, "created": now_iso(), "split_sha256": sha256_file(split_path),
        "cache_manifest_sha256": sha256_file(os.path.join(cache_dir, "manifest.json")),
        "trigger_manifest_sha256": sha256_file(trig_manifest_path),
        "parts": {part: {"n_rows": lay["n_rows"], "episodes": lay["episodes"]} for part, lay in parts.items()},
        "subset": subset,
        "rule": "ee_pos (float32, the recorded value: the expert's pinocchio forward kinematics of the full base pose "
                "and q_mes at the same step) for every row of the cache parts train / val (their step column) and of "
                "the trigger cache (its k0 step), in the same row order",
        "build_seconds": time.time() - t_start,
    }
    write_json_atomic(manifest_path, manifest)
    if verbose:
        print(f"ee cache {ee_dir}: " + ", ".join(f"{p} {lay['n_rows']} rows" for p, lay in parts.items())
              + f", {manifest['build_seconds']:.1f} s")
    return manifest


def load_ee(part, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, trigger_dir=None, allow_subset=False):
    """The ee_pos side cache of one part ('train', 'val' or 'trigger'), memory-mapped, checked against the cache (and
    trigger cache) it was built for. Returns (array, ee manifest sha256)."""
    ee_dir = ee_dir_of(cache_dir)
    manifest_path = os.path.join(ee_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"no ee side cache in {ee_dir}: run 'train_policy.py build-ee-cache'")
    with open(manifest_path) as f:
        m = json.load(f)
    if m.get("format") != EE_FORMAT:
        raise RuntimeError(f"{manifest_path}: not a {EE_FORMAT} cache")
    if m["split_sha256"] != sha256_file(split_path):
        raise RuntimeError(f"ee cache {ee_dir} was built for another split file")
    if m["cache_manifest_sha256"] != sha256_file(os.path.join(cache_dir, "manifest.json")):
        raise RuntimeError(f"ee cache {ee_dir} was built for another version of the cache in {cache_dir}")
    if m["subset"] and not allow_subset:
        raise RuntimeError(f"ee cache {ee_dir} belongs to a subset cache (test cache)")
    if part == "trigger":
        trigger_dir = trigger_dir or TRIGGER_DIR
        if m["trigger_manifest_sha256"] != sha256_file(os.path.join(trigger_dir, "manifest.json")):
            raise RuntimeError(f"ee cache {ee_dir} was built for another trigger cache than {trigger_dir}")
    arr = np.load(os.path.join(ee_dir, part, "ee_pos.npy"), mmap_mode="r")
    if arr.shape != (m["parts"][part]["n_rows"], 3):
        raise RuntimeError(f"ee cache {ee_dir}/{part}: shape {arr.shape}, manifest {m['parts'][part]['n_rows']} rows")
    return arr, sha256_file(manifest_path)


def ee_trigger_rows(episodes, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, trigger_dir=None, allow_subset=False):
    """ee_pos of the trigger rows of the given training episodes (the order of load_trigger_rows)."""
    arr, sha = load_ee("trigger", cache_dir, split_path, trigger_dir, allow_subset)
    with open(os.path.join(ee_dir_of(cache_dir), "manifest.json")) as f:
        eps = json.load(f)["parts"]["trigger"]["episodes"]
    row_of = {int(e): i for i, e in enumerate(eps)}
    return np.asarray(arr[[row_of[int(e)] for e in episodes]]), sha


# ==============================================================================
# Models
# ==============================================================================
def standardiser(A):
    """Column mean and scale (std; std < SCALE_EPS -> 1) of a float64 matrix."""
    mean = A.mean(axis=0)
    scale = A.std(axis=0)
    scale[scale < SCALE_EPS] = 1.0
    return mean, scale


def forward_std(weights, biases, Xs, chunk=FWD_CHUNK):
    """Network output in standardised units, computed in chunks by bc_features.forward (identity scalers)."""
    net = {"x_mean": 0.0, "x_scale": 1.0, "y_mean": 0.0, "y_scale": 1.0, "weights": weights, "biases": biases}
    out = np.empty((Xs.shape[0], weights[-1].shape[1]))
    for s in range(0, Xs.shape[0], chunk):
        out[s:s + chunk] = bcf.forward(net, Xs[s:s + chunk])
    return out


def std_mse(weights, biases, Xs, Ys, chunk=FWD_CHUNK):
    """Validation loss: standardised MSE averaged over all rows and the 9 outputs (chunked)."""
    net = {"x_mean": 0.0, "x_scale": 1.0, "y_mean": 0.0, "y_scale": 1.0, "weights": weights, "biases": biases}
    sse = 0.0
    for s in range(0, Xs.shape[0], chunk):
        d = bcf.forward(net, Xs[s:s + chunk]) - Ys[s:s + chunk]
        sse += float(np.einsum("ij,ij->", d, d))
    return sse / Ys.size


class EarlyStopping:
    """Early stopping on the validation loss.

    Patience: an epoch counts as an improvement when its loss is below ref * (1 - rel_tol), where ref is the loss
    of the last epoch that counted as an improvement (Keras min_delta semantics, relative). Training stops after
    `patience` consecutive epochs without one, so a slow but steady decrease (cumulatively > rel_tol) keeps going.
    Kept weights: those of the epoch with the lowest validation loss (any decrease updates them)."""

    def __init__(self, patience=PATIENCE, rel_tol=REL_TOL):
        self.patience, self.rel_tol = patience, rel_tol
        self.ref = np.inf
        self.best, self.best_epoch, self.bad = np.inf, None, 0

    def update(self, epoch, loss):
        """-> (is_new_best, stop)."""
        if np.isfinite(loss) and (not np.isfinite(self.ref) or loss < self.ref * (1.0 - self.rel_tol)):
            self.ref, self.bad = loss, 0
        else:
            self.bad += 1
        new_best = bool(loss < self.best)
        if new_best:
            self.best, self.best_epoch = loss, epoch
        return new_best, self.bad >= self.patience


def make_mlp(seed, params=None):
    from sklearn.neural_network import MLPRegressor
    # a RandomState instance (not an int): check_random_state returns the same instance on every partial_fit call,
    # so the shuffle differs between epochs; an int would rebuild the same RandomState and replay one shuffle
    return MLPRegressor(**(params or MLP_PARAMS), random_state=np.random.RandomState(seed))


def train_mlp(Xs, Ys, Xvs, Yvs, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE, rel_tol=REL_TOL, params=None,
              log=None, hide_losses=False):
    """MLPRegressor trained with one partial_fit call per epoch (Adam state kept between calls), early stopping on
    the standardised validation MSE. Xs, Ys, Xvs, Yvs are already standardised float64 matrices.
    Returns (best coefs, best intercepts, history dict, model)."""
    model = make_mlp(seed, params)
    stopper = EarlyStopping(patience, rel_tol)
    best_w = best_b = None
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "fit_s": [], "val_s": [], "t_samples": []}
    stop_reason = "max_epochs"
    n = Xs.shape[0]
    with single_thread():
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.partial_fit(Xs, Ys)
            t1 = time.time()
            # one partial_fit call = exactly one pass over the data (sklearn breaks out of its loop when incremental)
            if len(model.loss_curve_) != epoch or model.t_ != epoch * n:
                raise RuntimeError(f"partial_fit did not run exactly one epoch (loss_curve_ {len(model.loss_curve_)},"
                                   f" t_ {model.t_}, epoch {epoch}, n {n})")
            val = std_mse(model.coefs_, model.intercepts_, Xvs, Yvs)
            t2 = time.time()
            new_best, stop = stopper.update(epoch, val)
            if new_best:
                best_w = [c.copy() for c in model.coefs_]
                best_b = [b.copy() for b in model.intercepts_]
            for k, v in (("epoch", epoch), ("train_loss", float(model.loss_)), ("val_loss", val),
                         ("fit_s", t1 - t0), ("val_s", t2 - t1), ("t_samples", int(model.t_))):
                hist[k].append(v)
            if log:
                losses = "" if hide_losses else (f" train_loss {model.loss_:.6f} val_loss {val:.6f} "
                                                 f"(best {stopper.best:.6f} @ {stopper.best_epoch})")
                mem = mem_status()
                log(f"epoch {epoch:3d}{losses} fit {t1 - t0:.1f} s val {t2 - t1:.1f} s, MiB: peak RSS "
                    f"{mem.get('VmHWM', peak_rss_mb()):.0f}, anon {mem.get('RssAnon', float('nan')):.0f}, "
                    f"file {mem.get('RssFile', float('nan')):.0f}")
            if not np.isfinite(val):
                stop_reason = "non_finite_val_loss"
                break
            if stop:
                stop_reason = f"no relative improvement > {rel_tol:g} for {patience} consecutive epochs"
                break
    hist.update(best_epoch=stopper.best_epoch, best_val_loss=stopper.best, epochs_run=len(hist["epoch"]),
                stop_reason=stop_reason)
    return best_w, best_b, hist, model


def fit_ols(Xs, Ys):
    """Least squares on standardised features with an intercept -> (W (F, 9), b (9,))."""
    A = np.empty((Xs.shape[0], Xs.shape[1] + 1))
    A[:, :-1] = Xs
    A[:, -1] = 1.0
    beta, _, rank, sv = np.linalg.lstsq(A, Ys, rcond=None)
    return beta[:-1], beta[-1], {"rank": int(rank), "singular_values": sv}


def standardise_inplace(A, mean, scale):
    A -= mean
    A /= scale
    return A


def load_training_data(feature_set, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, max_train_episodes=None,
                       max_val_episodes=None, allow_subset=False):
    """Raw float64 (X, Y) of the training rows and of the validation rows, plus the two Cache objects."""
    tr = Cache("train", cache_dir, split_path, allow_subset=allow_subset)
    va = Cache("val", cache_dir, split_path, allow_subset=allow_subset)
    if feature_set in bcf.FEATURES_NEED_EE:   # the recorded end-effector position from the side cache
        tr.cols["ee_pos"], tr.ee_manifest_sha256 = load_ee("train", cache_dir, split_path, allow_subset=allow_subset)
        va.cols["ee_pos"], _ = load_ee("val", cache_dir, split_path, allow_subset=allow_subset)
    s_tr, s_va = tr.rows_of_first(max_train_episodes), va.rows_of_first(max_val_episodes)
    X = cache_features(tr.cols, feature_set, s_tr)
    Y = cache_labels(tr.cols, s_tr)
    Xv = cache_features(va.cols, feature_set, s_va)
    Yv = cache_labels(va.cols, s_va)
    tr.release()
    va.release()
    return X, Y, Xv, Yv, tr, va


def hold_qd_sigma(cols, sl=None):
    """sigma_hold (7,): per-joint standard deviation (float64, ddof 0) of the arm qd over the cached rows sl with
    nav_done False (the arm at its hold posture during navigation)."""
    sl = sl or slice(0, cols["episode"].shape[0])
    nav = np.asarray(cols["nav_done"][sl], dtype=bool)
    qd = np.asarray(cols["qd_arm"][sl], dtype=np.float64)[~nav]
    if qd.shape[0] == 0:
        raise ValueError("no rows with nav_done False")
    return qd.std(axis=0)


def trigger_copies(trig_cols, feature_set, copies, qd_sigma, jitter_seed=TRIGGER_JITTER_SEED):
    """Raw (not standardised) features and labels of `copies` copies of every trigger row (episode-major: the copies
    of the first episode, then of the second, ...). Every copy keeps the recorded signals and labels; only its arm qd
    gets N(0, qd_sigma_j) per joint, drawn once from numpy default_rng(jitter_seed) as one (n_rows * copies, 7)
    standard-normal array times qd_sigma. Returns (X, Y, noise)."""
    n = trig_cols["episode"].shape[0]
    k = int(copies)
    if k < 0:
        raise ValueError("copies must be >= 0")
    rep = np.repeat(np.arange(n), k)
    sig = signals_from_cache(trig_cols, slice(0, n))
    sig = {key: np.asarray(val)[rep] for key, val in sig.items()}
    noise = np.random.default_rng(jitter_seed).standard_normal((n * k, 7)) * np.asarray(qd_sigma, dtype=float)
    sig["qd_arm"] = sig["qd_arm"] + noise
    X = bcf.build_features(feature_set, sig)
    Y = cache_labels(trig_cols, slice(0, n))[rep]
    return X, Y, noise


def parse_qd_jitter(text):
    """--trigger-qd-jitter: 'auto' (sigma_hold of the training rows) or 7 comma-separated values / one value for all
    joints (rad/s; 0 = no jitter). Returns None for 'auto', else a (7,) array."""
    if text is None or str(text).strip().lower() == "auto":
        return None
    vals = [float(x) for x in str(text).split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * 7
    if len(vals) != 7 or any(not np.isfinite(v) or v < 0 for v in vals):
        raise SystemExit(f"--trigger-qd-jitter: expected 'auto', one value or 7 values >= 0, got {text!r}")
    return np.array(vals, dtype=float)


def save_trained(path, feature_set, xm, xs, ym, ys, weights, biases, meta):
    """bc_features.save_model, written to a temporary file first so a partial file never looks finished."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = os.path.join(os.path.dirname(os.path.abspath(path)), "." + os.path.basename(path)[:-4] + ".tmp.npz")
    bcf.save_model(tmp, feature_set, xm, xs, ym, ys, weights, biases, to_json(meta))
    os.replace(tmp, path)


def load_extra_rows(directory, feature_set, split_path=SPLIT_PATH):
    """Validate a side cache and return raw float64 features, labels and metadata.

    Extra rows never affect the standardisers or validation set. A manifest is
    written last by the cache builder, so an interrupted cache is refused.
    """
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path) as f:
        m = json.load(f)
    if (m.get("format") != "bc_dagger_v1" or m.get("feature_set") != feature_set
            or m.get("split_sha256") != sha256_file(split_path)):
        raise ValueError("extra-row manifest format, features or split mismatch")
    n = int(m["n_rows"])
    arrays = []
    for name, width in (("X", len(bcf.FEATURE_SETS[feature_set])), ("Y", len(bcf.OUTPUT_NAMES))):
        path = os.path.join(directory, name + ".npy")
        if sha256_file(path) != m["arrays"][name]["sha256"]:
            raise ValueError(f"extra-row array hash mismatch: {name}")
        a = np.load(path, mmap_mode="r")
        if a.shape != (n, width) or a.dtype != np.dtype("float64") or not np.isfinite(a).all():
            raise ValueError(f"invalid extra-row array: {name}")
        arrays.append(np.array(a))
    meta = {"directory": os.path.relpath(os.path.abspath(directory), TASK_DIR),
            "manifest_sha256": sha256_file(manifest_path), "n_rows": n,
            "rule": "unweighted rows appended after trigger copies; original stride-9 standardisers; "
                    "expert validation unchanged"}
    return *arrays, meta


def run_train(model_name, seed=None, out=None, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, max_epochs=MAX_EPOCHS,
              max_train_episodes=None, max_val_episodes=None, force=False, hide_losses=False, log=print,
              trigger_copies_k=None, trigger_qd_jitter="auto", trigger_dir=None, extra_rows=None,
              chunk_k=None, chunk_delta=None, side_dir=None):
    if model_name in ("ego-novw-trig-chunk", "ego-novw-trig-pd"):
        if extra_rows is not None:
            raise SystemExit("G/P do not accept experiment F's --extra-rows")
        from results.step6.gp_training import run_training
        return run_training(model_name, seed, out, cache_dir, split_path, max_epochs,
                            max_train_episodes, max_val_episodes, force, hide_losses, log,
                            trigger_copies_k, trigger_qd_jitter, trigger_dir, chunk_k, chunk_delta, side_dir)
    if chunk_k is not None or chunk_delta is not None or side_dir is not None:
        raise SystemExit("--chunk-k / --chunk-delta / --side-cache are opt-in G/P arguments")
    kind, feature_set = MODELS[model_name]
    if model_name == "ego-novw-trig-dag1" and extra_rows is None:
        raise SystemExit("ego-novw-trig-dag1 needs --extra-rows")
    custom = max_epochs != MAX_EPOCHS or max_train_episodes is not None or max_val_episodes is not None
    if custom and out is None:
        raise SystemExit("non-default training settings (--max-epochs / --max-*-episodes) need an explicit --out, "
                         "so a smoke-test model never takes an official trained/ name")
    if model_name in TRIGGER_MODELS:
        if trigger_copies_k is None or int(trigger_copies_k) < 0:
            raise SystemExit(f"{model_name} needs --trigger-copies K (K >= 0)")
        trigger_copies_k = int(trigger_copies_k)
        jitter_fixed = parse_qd_jitter(trigger_qd_jitter)
        trigger_dir = trigger_dir or TRIGGER_DIR
    elif trigger_copies_k is not None or parse_qd_jitter(trigger_qd_jitter) is not None:
        raise SystemExit(f"--trigger-copies / --trigger-qd-jitter are only for the models {TRIGGER_MODELS}, "
                         f"not {model_name}")
    if kind == "mlp" and seed is None:
        seed = 0
    if kind == "ols":
        if seed is not None:
            log(f"{model_name} is deterministic: --seed {seed} ignored")
        seed = None
    out = out or default_out(model_name, seed)
    if not out.endswith(".npz"):
        raise SystemExit("--out must end with .npz")
    if os.path.exists(out) and not force:
        raise SystemExit(f"{out} exists (pass --force to overwrite)")
    import sklearn
    t_start = time.time()
    X, Y, Xv, Yv, tr, va = load_training_data(feature_set, cache_dir, split_path, max_train_episodes,
                                              max_val_episodes, allow_subset=custom)
    t_load = time.time() - t_start
    xm, xs = standardiser(X)
    ym, ys = standardiser(Y)
    for A, m_, s_ in ((X, xm, xs), (Y, ym, ys), (Xv, xm, xs), (Yv, ym, ys)):
        standardise_inplace(A, m_, s_)
    n_tr_eps = len(tr.episodes) if max_train_episodes is None else min(max_train_episodes, len(tr.episodes))
    n_va_eps = len(va.episodes) if max_val_episodes is None else min(max_val_episodes, len(va.episodes))
    n_base_rows = int(X.shape[0])
    trigger_meta = None
    if model_name in TRIGGER_MODELS:
        # the standardisation (xm, xs, ym, ys) above comes from the original stride-9 rows only; the copies are
        # standardised with it and appended (K = 0: nothing appended, the arrays are ego-novw's)
        tr_cols = Cache("train", cache_dir, split_path, allow_subset=custom)
        sigma_hold = hold_qd_sigma(tr_cols.cols, tr.rows_of_first(max_train_episodes))
        tr_cols.release()
        sigma = sigma_hold if jitter_fixed is None else jitter_fixed
        train_eps = list(tr.episodes[:n_tr_eps])
        trig_cols, trig_sha = load_trigger_rows(train_eps, trigger_dir, split_path, allow_subset=custom)
        if feature_set in bcf.FEATURES_NEED_EE:   # the recorded k0 end-effector position (the jitter leaves q alone)
            trig_cols["ee_pos"], _ = ee_trigger_rows(train_eps, cache_dir, split_path, trigger_dir, allow_subset=custom)
        n_added = trigger_copies_k * len(train_eps)
        if trigger_copies_k > 0:
            Xt, Yt, _ = trigger_copies(trig_cols, feature_set, trigger_copies_k, sigma)
            standardise_inplace(Xt, xm, xs)
            standardise_inplace(Yt, ym, ys)
            X, Y = np.concatenate([X, Xt]), np.concatenate([Y, Yt])
            del Xt, Yt
        assert X.shape[0] == n_base_rows + n_added
        trigger_meta = {
            "copies": trigger_copies_k, "n_episodes": len(train_eps), "n_rows_added": n_added,
            "n_train_rows_base": n_base_rows, "n_train_rows_total": int(X.shape[0]),
            "k0_steps": np.asarray(trig_cols["step"]).tolist(),
            "qd_jitter": "auto" if jitter_fixed is None else jitter_fixed.tolist(),
            "qd_sigma": np.asarray(sigma).tolist(), "qd_sigma_hold": sigma_hold.tolist(),
            "jitter_seed": TRIGGER_JITTER_SEED,
            "trigger_cache": os.path.relpath(os.path.abspath(trigger_dir), TASK_DIR),
            "trigger_cache_manifest_sha256": trig_sha,
            "rule": "K copies of each training episode's trigger frame k0 (first nav_done step, 1 kHz recording), "
                    "signals and labels as recorded; arm qd + N(0, qd_sigma_j) per joint, one numpy "
                    "default_rng(jitter_seed).standard_normal((n_episodes * K, 7)) * qd_sigma, episode-major copies; "
                    "qd_sigma_hold = per-joint std of arm qd over the training rows with nav_done False; "
                    "standardisation from the original training rows only; validation rows unchanged",
        }
        log(f"trigger copies: K {trigger_copies_k} x {len(train_eps)} episodes = {n_added} rows added to "
            f"{n_base_rows}; qd sigma " + ", ".join(f"{s:.4f}" for s in sigma)
            + ("" if jitter_fixed is None else " (fixed)") + f", jitter seed {TRIGGER_JITTER_SEED}")
    extra_meta = None
    if extra_rows is not None:
        Xe, Ye, extra_meta = load_extra_rows(extra_rows, feature_set, split_path)
        standardise_inplace(Xe, xm, xs)
        standardise_inplace(Ye, ym, ys)
        X, Y = np.concatenate([X, Xe]), np.concatenate([Y, Ye])
        del Xe, Ye
        log(f"extra rows: {extra_meta['n_rows']} appended from {extra_rows}")
    log(f"{model_name} ({kind}, features {feature_set}, seed {seed}): train {X.shape[0]} rows "
        f"({n_tr_eps} episodes), val {Xv.shape[0]} rows ({n_va_eps} episodes), loaded in {t_load:.1f} s")
    commit, dirty = git_state()
    meta = {
        "model": model_name, "kind": kind, "feature_set": feature_set, "feature_names": bcf.FEATURE_SETS[feature_set],
        "output_names": bcf.OUTPUT_NAMES, "seed": seed,
        "split_file": os.path.relpath(os.path.abspath(split_path), TASK_DIR),
        "split_sha256": sha256_file(split_path), "cache_manifest_sha256": tr.manifest_sha256,
        "stride": int(tr.split["stride"]), "n_train_rows": int(X.shape[0]), "n_val_rows": int(Xv.shape[0]),
        "n_train_episodes": n_tr_eps, "n_val_episodes": n_va_eps,
        **({"trigger": trigger_meta} if trigger_meta is not None else {}),
        **({"extra_rows": extra_meta} if extra_meta is not None else {}),
        **({"ee_cache_manifest_sha256": tr.ee_manifest_sha256, "ee_clip_m": bcf.EE_CLIP,
            "ee_features": "near-field end-effector error e * min(1, ee_clip / |e|), e = target - recorded ee_pos, "
                           "horizontal part in the base frame"}
           if feature_set in bcf.FEATURES_NEED_EE else {}),
        "subset": bool(custom) and (max_train_episodes is not None or max_val_episodes is not None),
        "standardisation": "X and Y with the training rows' mean and std (std < 1e-12 -> 1)" if trigger_meta is None
                           else "X and Y with the mean and std (std < 1e-12 -> 1) of the original stride-9 training "
                                "rows only (trigger.n_train_rows_base); the trigger copies are standardised with them",
        "val_loss_definition": "standardised MSE averaged over all validation rows (1 kHz) and the 9 outputs",
        "git_commit": commit, "git_dirty": dirty,
        "code_sha256": {f: sha256_file(os.path.join(TASK_DIR, f)) for f in ("train_policy.py", "bc_features.py")},
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "created": now_iso(), "load_s": t_load,
    }
    t_fit = time.time()
    if kind == "mlp":
        hp = {**MLP_PARAMS, "hidden_layer_sizes": list(MLP_PARAMS["hidden_layer_sizes"]),
              "random_state": f"np.random.RandomState({seed})", "max_epochs": max_epochs, "patience": PATIENCE,
              "rel_tol": REL_TOL, "epoch": "one partial_fit call", "dtype": "float64"}
        weights, biases, hist, model = train_mlp(X, Y, Xv, Yv, seed, max_epochs=max_epochs, log=log,
                                                 hide_losses=hide_losses)
        check = std_mse(weights, biases, Xv, Yv)
        assert abs(check - hist["best_val_loss"]) <= 1e-12 * max(1.0, check), "best weights do not match"
        meta.update(hyperparameters=hp, epochs_run=hist["epochs_run"], best_epoch=hist["best_epoch"],
                    val_loss=hist["best_val_loss"], stop_reason=hist["stop_reason"],
                    history={k: hist[k] for k in ("epoch", "train_loss", "val_loss", "fit_s", "val_s")},
                    train_loss_definition="sklearn MLPRegressor.loss_ of the epoch: 0.5 * mean squared standardised "
                                          "error over the epoch's batches + the L2 penalty (alpha)",
                    n_params=int(sum(w.size + b.size for w, b in zip(weights, biases))))
    else:
        hp = {"method": "numpy.linalg.lstsq on [standardised X, 1] -> standardised Y", "rcond": None}
        W, b, info = fit_ols(X, Y)
        weights, biases = [W], [b]
        val = std_mse(weights, biases, Xv, Yv)
        meta.update(hyperparameters=hp, epochs_run=None, best_epoch=None, val_loss=val, history=None,
                    lstsq_rank=info["rank"], lstsq_singular_values=info["singular_values"],
                    n_params=int(W.size + b.size))
    meta["train_s"] = time.time() - t_fit
    meta["peak_rss_mib"] = peak_rss_mb()
    meta["memory_end_mib"] = mem_status()
    save_trained(out, feature_set, xm, xs, ym, ys, weights, biases, meta)
    summary = f"saved {out}: {meta['epochs_run'] or '-'} epochs, best epoch {meta['best_epoch'] or '-'}"
    if not hide_losses:
        summary += f", val loss {meta['val_loss']:.6f}"
    log(summary + f", train {meta['train_s']:.1f} s, peak RSS {meta['peak_rss_mib']:.0f} MiB")
    return out, meta


def run_train_many(model_name, seeds, jobs, force=False, cache_dir=CACHE_DIR, split_path=SPLIT_PATH,
                   extra_args=()):
    kind, _ = MODELS[model_name]
    if kind != "mlp":
        raise SystemExit("train-many is for the MLP models (OLS is deterministic)")
    todo = []
    for s in seeds:
        out = default_out(model_name, s)
        if os.path.exists(out) and not force:
            print(f"skip seed {s}: {out} exists")
            continue
        todo.append((s, out))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    running, results = [], {}
    t0 = time.time()
    while todo or running:
        while todo and len(running) < jobs:
            s, out = todo.pop(0)
            log_path = out[:-4] + ".log"
            cmd = [sys.executable, os.path.abspath(__file__), "train", "--model", model_name, "--seed", str(s),
                   "--out", out, "--cache", cache_dir, "--split", split_path, "--force", *extra_args]
            fh = open(log_path, "w")
            running.append((s, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=TASK_DIR), fh))
            print(f"started seed {s} -> {out} (log {log_path})", flush=True)
        time.sleep(2.0)
        for item in list(running):
            s, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                results[s] = proc.returncode
                print(f"seed {s} finished with code {proc.returncode} ({time.time() - t0:.0f} s)", flush=True)
    failed = [s for s, rc in results.items() if rc != 0]
    if failed:
        raise SystemExit(f"failed seeds: {failed}")
    return results


# ==============================================================================
# Offline metrics
# ==============================================================================
def annotate_val(va):
    """Per-row phase class, torque segment and event windows of the validation rows (all steps of each episode).

    class: align / drive / orient (phase code before nav_done), reach_move (nav_done up to and including the first
    step with EE error < 10 mm, as in analyze_dataset), reach_hold (after it). Events: 'start' (step 0) and the
    changes of the phase code with nav_done as 'reach'; windows [event, event + settle] with the settle times of
    analyze_dataset.CALIBRATION."""
    from analyze_dataset import CALIBRATION
    from robotnik_arm_task import REACH_TOLERANCE
    settle = CALIBRATION["settle"]
    dt = va.time_step
    width = {e: round(settle[e] / dt) for e in EVENT_TYPES}
    n = va.n_rows
    cls = np.empty(n, dtype=np.int8)
    inside = np.zeros(n, dtype=bool)
    in_type = np.zeros((len(EVENT_TYPES), n), dtype=bool)
    events = []          # (type index, episode index, event step)
    counts = {e: 0 for e in EVENT_TYPES}
    other, mismatch = {}, 0
    c = va.cols
    for i in range(len(va.episodes)):
        sl = va.episode_slice(i)
        steps = np.asarray(c["step"][sl])
        if not np.array_equal(steps, np.arange(len(steps))):
            raise RuntimeError("validation rows must hold every step of the episode")
        ph = np.asarray(c["phase"][sl]).astype(int)
        nd = np.asarray(c["nav_done"][sl])
        err = np.asarray(c["ee_err"][sl])
        mismatch += int(np.sum((ph == 3) != nd))
        nav = np.where(nd, 3, np.minimum(ph, 2))   # 0 align, 1 drive, 2 orient, 3 reach (= nav_done)
        ce = nav.copy()
        i_reach = int(np.argmax(nd)) if nd.any() else len(nd)
        hits = np.flatnonzero(err[i_reach:] < REACH_TOLERANCE)
        if hits.size:
            ce[i_reach + int(hits[0]) + 1:] = 4
        cls[sl] = ce
        changes = np.flatnonzero(nav[1:] != nav[:-1]) + 1
        for name, k in [("start", 0)] + [(f"{NAV_NAMES[nav[k - 1]]}>{NAV_NAMES[nav[k]]}", int(k)) for k in changes]:
            if name not in width:
                other[name] = other.get(name, 0) + 1
                continue
            j = EVENT_TYPES.index(name)
            a, b = sl.start + k, sl.start + min(len(nav), k + width[name] + 1)
            inside[a:b] = True
            in_type[j, a:b] = True
            events.append((j, i, k))
            counts[name] += 1
    seg = np.where(cls <= 2, 0, cls - 2).astype(np.int8)   # nav_hold (before nav_done), reach_move, reach_hold
    return {"cls": cls, "inside": inside, "in_type": in_type, "events": events, "counts": counts,
            "other_events": other, "phase_navdone_mismatch": mismatch, "seg": seg, "width": width}


def _rmse_block(sq_sum, n, scale, names):
    if n == 0:
        return {k: None for k in names}, {k: None for k in names}
    rmse = np.sqrt(sq_sum / n)
    return dict(zip(names, rmse)), dict(zip(names, rmse / scale))


def lag_curve_summary(lags, mse, lag_max=LAG_MAX, flat_tol=LAG_FLAT_TOL):
    """Summary of one pooled MSE(k) curve (k = lags, signed): k_best (the minimising k; exact ties -> smallest |k|,
    then the lead), mse_best (the curve minimum), mse_k0 (k = 0), rel_range = (max - min) / min (inf when
    min = 0 < max), flat (rel_range < flat_tol: no identifiable lag, k_best None) and boundary (|k_best| = lag_max:
    the minimum may lie outside the searched range). A curve without pairs (NaN) gives None and both flags False."""
    lags = np.asarray(lags)
    mse = np.asarray(mse, dtype=float)
    i0 = int(np.flatnonzero(lags == 0)[0])
    if mse.size == 0 or not np.isfinite(mse).all():
        return {"k_best": None, "mse_best": None, "mse_k0": None, "rel_range": None, "flat": False,
                "boundary": False}
    lo, hi = float(mse.min()), float(mse.max())
    rel = (hi - lo) / lo if lo > 0 else (0.0 if hi == lo else np.inf)
    order = np.lexsort((lags, np.abs(lags)))       # by |k|, then k: argmin keeps the first minimiser of this order
    i_best = int(order[np.argmin(mse[order])])
    flat = bool(rel < flat_tol)
    k_best = None if flat else int(lags[i_best])
    return {"k_best": k_best, "mse_best": lo, "mse_k0": float(mse[i0]), "rel_range": rel, "flat": flat,
            "boundary": k_best is not None and abs(k_best) == lag_max}


def lag_analysis(pred, label, va, ann, clean, lag_max=LAG_MAX, flat_tol=LAG_FLAT_TOL):
    """Per event type: pooled MSE(k) between pred[t] and label[t - k] for k = -lag_max..lag_max (k > 0: the
    prediction lags the label, pred[t] ~ label[t - k]; k < 0: it leads), over the [event - LAG_PRE, event + settle]
    windows (clipped to the episode) of the clean episodes.

    Every k averages over the same steps t: where t - k falls outside the episode the label is padded, with 0
    before step 0 (the robot starts at rest, as analyze_dataset's ve_prev = 0) and with its value at the last step
    after the end. Returns per event type n_events, n_pairs (= number of t, the same for every k), lags, the
    (n_lag, n_out) curve mse and per output the lag_curve_summary."""
    pre = round(LAG_PRE / va.time_step)
    lags = np.arange(-lag_max, lag_max + 1)
    n_out = pred.shape[1]
    out = {}
    for j, name in enumerate(EVENT_TYPES):
        sse = np.zeros((len(lags), n_out))
        n_ev = n_pairs = 0
        for (jj, i, k) in ann["events"]:
            if jj != j or not clean[i]:
                continue
            n_ev += 1
            sl = va.episode_slice(i)
            T = sl.stop - sl.start
            t_idx = np.arange(max(0, k - pre), min(T, k + ann["width"][name] + 1))
            src = t_idx[:, None] - lags[None, :]                 # (n_t, n_lag) label step paired with t
            p = pred[sl.start + t_idx]                           # (n_t, n_out)
            lab = label[sl.start + np.clip(src, 0, T - 1)]       # (n_t, n_lag, n_out), a copy: nearest valid step
            lab[src < 0] = 0.0                                   # before step 0: at rest
            sse += ((p[:, None, :] - lab) ** 2).sum(axis=0)
            n_pairs += len(t_idx)
        mse = sse / n_pairs if n_pairs else np.full_like(sse, np.nan)
        out[name] = {"n_events": n_ev, "n_pairs": n_pairs, "lags": lags, "mse": mse,
                     "summary": [lag_curve_summary(lags, mse[:, o], lag_max, flat_tol) for o in range(n_out)]}
    return out


def offline_metrics(model_path, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, allow_subset=False, log=print):
    from analyze_dataset import CALIBRATION, Z_SPLIT
    t0 = time.time()
    model = bcf.load_model(model_path)
    meta = model["meta"]
    if meta.get("output_representation") in ("chunk", "pd_reference"):
        from results.step6.gp_training import offline_metrics as step6_offline
        return step6_offline(model_path, cache_dir, split_path, allow_subset, log)
    va = Cache("val", cache_dir, split_path, allow_subset=allow_subset)
    if meta.get("split_sha256") and meta["split_sha256"] != sha256_file(split_path):
        raise RuntimeError(f"{model_path} was trained on another split file")
    fs = model["feature_set"]
    names = bcf.OUTPUT_NAMES
    if fs in bcf.FEATURES_NEED_EE:
        va.cols["ee_pos"], _ = load_ee("val", cache_dir, split_path, allow_subset=allow_subset)
    with single_thread():
        X = cache_features(va.cols, fs)
        Y = cache_labels(va.cols)
        pred = np.empty_like(Y)
        for s in range(0, X.shape[0], FWD_CHUNK):
            pred[s:s + FWD_CHUNK] = bcf.forward(model, X[s:s + FWD_CHUNK])
    del X
    v_mes, w_mes = bcf.wheels_to_vw(va.cols["qd_wheels"])
    cc = np.column_stack([v_mes, w_mes])
    ys = np.asarray(model["y_scale"], dtype=float)
    err2 = (pred - Y) ** 2
    cerr2 = (cc - Y[:, :2]) ** 2
    val_loss = float(np.mean(err2 / ys ** 2))
    ann = annotate_val(va)
    noisy_row = np.asarray(va.cols["noisy"], dtype=bool)
    ep_noisy = np.array([bool(va.cols["noisy"][va.row_start[i]]) for i in range(len(va.episodes))])
    z_high = np.asarray(va.cols["target"][:, 2]) >= Z_SPLIT

    # sums per (noisy, class, inside) combination, then pooled into the cells
    code = (noisy_row.astype(np.int64) * len(CLASSES) + ann["cls"]) * 2 + ann["inside"]
    n_codes = 2 * len(CLASSES) * 2
    cnt = np.bincount(code, minlength=n_codes).reshape(2, len(CLASSES), 2)
    s_m = np.stack([np.bincount(code, weights=err2[:, j], minlength=n_codes) for j in range(9)], axis=1)
    s_m = s_m.reshape(2, len(CLASSES), 2, 9)
    s_c = np.stack([np.bincount(code, weights=cerr2[:, j], minlength=n_codes) for j in range(2)], axis=1)
    s_c = s_c.reshape(2, len(CLASSES), 2, 2)
    groups = {"all": [0, 1], "clean": [0], "noisy": [1]}
    class_sel = {"all": list(range(len(CLASSES))), **{c: [i] for i, c in enumerate(CLASSES)}}
    win_sel = {"all": [0, 1], "inside": [1], "outside": [0]}
    cells = []
    for g, gi in groups.items():
        for cname, ci in class_sel.items():
            for wname, wi in win_sel.items():
                ix = np.ix_(gi, ci, wi)
                n = int(cnt[ix].sum())
                sm = s_m[ix].reshape(-1, 9).sum(axis=0)
                sc = s_c[ix].reshape(-1, 2).sum(axis=0)
                rmse, nrmse = _rmse_block(sm, n, ys, names)
                crmse, cnrmse = _rmse_block(sc, n, ys[:2], ["v", "w"])
                cells.append({"group": g, "class": cname, "window": wname, "n": n, "rmse": rmse, "nrmse": nrmse,
                              "nrmse_tau_mean": None if n == 0 else float(np.mean(np.sqrt(sm[2:] / n) / ys[2:])),
                              "copycat_rmse": crmse, "copycat_nrmse": cnrmse})
    # inside the windows of each event type
    by_event = []
    for j, name in enumerate(EVENT_TYPES):
        for g, gi in groups.items():
            m = ann["in_type"][j] & np.isin(noisy_row.astype(int), gi)
            n = int(m.sum())
            rmse, nrmse = _rmse_block(err2[m].sum(axis=0), n, ys, names)
            crmse, cnrmse = _rmse_block(cerr2[m].sum(axis=0), n, ys[:2], ["v", "w"])
            n_ev = sum(1 for (jj, i, _) in ann["events"] if jj == j and int(ep_noisy[i]) in gi)
            by_event.append({"event": name, "group": g, "n_events": n_ev, "n": n, "rmse": rmse, "nrmse": nrmse,
                             "copycat_rmse": crmse, "copycat_nrmse": cnrmse})
    # torque RMSE by segment x target z
    tau_names = names[2:]
    torque = []
    for g, gi in groups.items():
        for si, seg in enumerate(SEGMENTS):
            for zname, zval in (("z<0.45", False), ("z>=0.45", True)):
                m = (ann["seg"] == si) & (z_high == zval) & np.isin(noisy_row.astype(int), gi)
                n = int(m.sum())
                sm = err2[m][:, 2:].sum(axis=0)
                rmse, nrmse = _rmse_block(sm, n, ys[2:], tau_names)
                torque.append({"group": g, "segment": seg, "z": zname, "n": n, "rmse": rmse, "nrmse": nrmse,
                               "rmse_pooled": None if n == 0 else float(np.sqrt(sm.sum() / (7 * n)))})
    # lag of the base outputs around events (clean episodes)
    clean = ~ep_noisy
    lag_m = lag_analysis(pred[:, :2], Y[:, :2], va, ann, clean)
    lag_c = lag_analysis(cc, Y[:, :2], va, ann, clean)
    lag = {}
    for name in EVENT_TYPES:
        x = lag_m[name]
        lag[name] = {"n_events": x["n_events"], "n_pairs": x["n_pairs"], "n_pairs_lag0": x["n_pairs"],
                     "lags": x["lags"]}
        for src, res in (("model", lag_m), ("copycat", lag_c)):
            lag[name][src] = {o: {**res[name]["summary"][j], "mse": res[name]["mse"][:, j]}
                              for j, o in enumerate(("v", "w"))}
    lag_definition = (
        "lag.<event>: n_events = events of that type in the clean validation episodes; lags = the searched k, "
        f"-{LAG_MAX}..{LAG_MAX} steps ({va.time_step * 1000:g} ms each); n_pairs = number of steps t pooled, the "
        "same for every k (n_pairs_lag0 = n_pairs, kept for older readers); t runs over the [event - "
        f"{LAG_PRE * 1000:.0f} ms, event + settle] windows clipped to the episode. <model|copycat>.<v|w>: mse[i] = "
        "pooled MSE between pred[t] and label[t - lags[i]], the label padded with 0 before step 0 (robot at rest) "
        "and with its last value after the episode end; k_best = the minimising k, signed: k > 0 the prediction "
        "lags the label (pred[t] ~ label[t - k]), k < 0 it leads (exact ties: smallest |k|, then k < 0); "
        "mse_best = minimum of the curve; mse_k0 = MSE at k = 0; rel_range = (max - min) / min of the curve "
        f"(null when min = 0 < max); flat = rel_range < {LAG_FLAT_TOL:g}: no identifiable lag, k_best null; "
        f"boundary = |k_best| = {LAG_MAX}: the true minimum may lie outside the searched range; without events "
        "k_best, mse_best, mse_k0 and rel_range are null and both flags false")
    result = {
        "model_file": os.path.relpath(os.path.abspath(model_path), TASK_DIR), "model_sha256": sha256_file(model_path),
        "model": meta.get("model"), "kind": meta.get("kind"), "seed": meta.get("seed"), "feature_set": fs,
        "split_sha256": sha256_file(split_path), "cache_manifest_sha256": va.manifest_sha256,
        "n_val_episodes": len(va.episodes), "n_val_episodes_noisy": int(ep_noisy.sum()), "n_val_rows": va.n_rows,
        "val_loss": val_loss, "val_loss_meta": meta.get("val_loss"),
        "label_std_train": dict(zip(names, ys)),
        "definitions": {
            "val_loss": "standardised MSE averaged over all validation rows (1 kHz) and the 9 outputs "
                        "(the early-stopping loss)",
            "nrmse": "RMSE / pooled training-row label std (the model's y_scale)",
            "classes": "align, drive, orient = phase code before nav_done; reach_move = nav_done up to and "
                       "including the first step with EE error < 10 mm; reach_hold = after it",
            "events": "start (step 0) and changes of the phase code (nav_done = reach): " + ", ".join(EVENT_TYPES)
                      + "; other transitions are counted in counts.other_events and get no window",
            "windows": "[event, event + settle(type)] steps, settle = analyze_dataset.CALIBRATION['settle']",
            "copycat": "measured (v_mes, w_mes) = bc_features.wheels_to_vw(qd_mes[0:4]) as the prediction of (v, w)",
            "torque_segments": "nav_hold = before nav_done, reach_move, reach_hold; z split at analyze_dataset.Z_SPLIT",
            "lag": f"signed k in -{LAG_MAX}..{LAG_MAX} minimising the pooled squared error between pred[t] and "
                   f"label[t - k] (k > 0: the prediction lags) over the same steps t for every k, the [event - "
                   f"{LAG_PRE * 1000:.0f} ms, event + settle] windows of the clean episodes; see lag_definition",
        },
        "settle_s": {e: CALIBRATION["settle"][e] for e in EVENT_TYPES},
        "counts": {"events": ann["counts"], "other_events": ann["other_events"],
                   "phase_navdone_mismatch_steps": ann["phase_navdone_mismatch"],
                   "rows_inside_windows": int(ann["inside"].sum()),
                   "rows_by_class": dict(zip(CLASSES, np.bincount(ann["cls"], minlength=len(CLASSES))))},
        "cells": cells, "event_windows": by_event, "torque_segments": torque, "lag": lag,
        "lag_definition": lag_definition,
        "seconds": time.time() - t0, "created": now_iso(),
    }
    return result


def print_offline(res, log=print):
    def cell(g, c, w):
        return next(x for x in res["cells"] if x["group"] == g and x["class"] == c and x["window"] == w)

    def f(x, d=4):
        return "   -  " if x is None else f"{x:.{d}f}"

    def fe(x):
        return "-" if x is None else f"{x:.3e}"

    def lag_k(x):
        if x["k_best"] is None:
            return "flat" if x["flat"] else "-"
        return f"{x['k_best']:+d}" + (" (boundary)" if x["boundary"] else "")

    log(f"{res['model_file']}: {res['n_val_rows']} validation rows ({res['n_val_episodes']} episodes), "
        f"val loss (standardised MSE) {res['val_loss']:.6f}")
    a = cell("all", "all", "all")
    log("overall RMSE: " + ", ".join(f"{k} {f(v)}" for k, v in a["rmse"].items()))
    log("overall nRMSE: " + ", ".join(f"{k} {f(v, 3)}" for k, v in a["nrmse"].items()))
    log(f"{'cell':28s} {'n':>8s} {'v':>7s} {'w':>7s} {'cc v':>7s} {'cc w':>7s} {'nRMSE tau':>9s}")
    rows = [("all", c, "all") for c in ["all", *CLASSES]] + [("all", "all", "inside"), ("all", "all", "outside")]
    rows += [(g, "all", "all") for g in ("clean", "noisy")]
    for g, c, w in rows:
        x = cell(g, c, w)
        log(f"{g + ' ' + c + ' ' + w:28s} {x['n']:8d} {f(x['rmse']['v'])} {f(x['rmse']['w'])} "
            f"{f(x['copycat_rmse']['v'])} {f(x['copycat_rmse']['w'])} {f(x['nrmse_tau_mean'], 3):>9s}")
    log("torque RMSE pooled over joints (N m), all episodes: " + ", ".join(
        f"{t['segment']} {t['z']} {f(t['rmse_pooled'], 3)}" for t in res["torque_segments"] if t["group"] == "all"))
    log("lag k (steps, signed, k > 0 = the prediction lags) minimising |pred[t] - label[t-k]|^2, clean episodes; "
        "per output: k_best, MSE at k_best / at k = 0, relative range (max - min) / min of MSE(k):")
    for e, x in res["lag"].items():
        for src in ("model", "copycat"):
            y = x[src]
            log(f"  {e:13s} {x['n_events']:4d} events {src:7s} " + "  ".join(
                f"{o} {lag_k(y[o]):>4s} {fe(y[o]['mse_best'])} / {fe(y[o]['mse_k0'])} range {fe(y[o]['rel_range'])}"
                for o in ("v", "w")))
    log(f"({res['seconds']:.1f} s)")


# ==============================================================================
# CLI
# ==============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="create the episode split file (once)")
    p.add_argument("--out", default=SPLIT_PATH)
    p.add_argument("--force", action="store_true", help="overwrite an existing split file")

    p = sub.add_parser("build-cache", help="extract the training / validation rows into the cache")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--workers", type=int, default=12, help="pool size (at most 12)")

    p = sub.add_parser("build-ee-cache", help="side cache of the recorded end-effector position (experiment D)")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--trigger-cache", default=TRIGGER_DIR)
    p.add_argument("--workers", type=int, default=12, help="pool size (at most 12)")

    p = sub.add_parser("build-trigger-cache", help="extract the trigger frame k0 of every training episode (C3)")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--trigger-cache", default=TRIGGER_DIR)
    p.add_argument("--workers", type=int, default=12, help="pool size (at most 12)")

    for name in ("build-chunk-cache", "build-pdref-cache"):
        p = sub.add_parser(name, help="new immutable side labels (step 6; original caches remain read-only)")
        p.add_argument("--split", default=SPLIT_PATH)
        p.add_argument("--cache", default=CACHE_DIR)
        p.add_argument("--trigger-cache", default=TRIGGER_DIR)
        p.add_argument("--side-cache", default=None)
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--allow-subset", action="store_true", help="smoke only: allow subset source caches")
        if name == "build-chunk-cache":
            p.add_argument("--chunk-k", type=int, default=10)
            p.add_argument("--chunk-delta", type=int, default=9, choices=(9, 27, 81))

    for name in ("train", "train-many"):
        p = sub.add_parser(name, help="train one model" if name == "train" else "train several seeds in parallel")
        p.add_argument("--model", required=True, choices=sorted(MODELS))
        if name == "train":
            p.add_argument("--seed", type=int, default=None, help="MLP seed (default 0; ignored by OLS)")
            p.add_argument("--out", default=None, help="default trained/<model>_s<seed>.npz, OLS trained/<model>.npz")
            p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
            p.add_argument("--max-train-episodes", type=int, default=None, help="smoke test: first N train episodes")
            p.add_argument("--max-val-episodes", type=int, default=None, help="smoke test: first N val episodes")
            p.add_argument("--hide-losses", action="store_true", help="print no loss values (blind runs)")
        else:
            p.add_argument("--seeds", required=True, help="e.g. 0-7 or 0,2,5")
            p.add_argument("--jobs", type=int, default=8)
            p.add_argument("--hide-losses", action="store_true", help="the seed logs show no loss values")
        p.add_argument("--split", default=SPLIT_PATH)
        p.add_argument("--cache", default=CACHE_DIR)
        p.add_argument("--force", action="store_true", help="overwrite existing outputs")
        p.add_argument("--trigger-copies", type=int, default=None,
                       help=f"K copies of each training episode's trigger frame (only and required for {TRIGGER_MODELS})")
        p.add_argument("--trigger-qd-jitter", default="auto",
                       help="arm-qd jitter sigma of the copies: 'auto' (per-joint std of arm qd over the training rows "
                            "with nav_done False), one value or 7 comma-separated values (rad/s; 0 = none)")
        p.add_argument("--trigger-cache", default=TRIGGER_DIR)
        p.add_argument("--extra-rows", default=None, help="append validated, unweighted side-cache rows (experiment F)")
        p.add_argument("--chunk-k", type=int, default=None, help="opt-in G action block length (G default 10)")
        p.add_argument("--chunk-delta", type=int, default=None, help="opt-in G action spacing in steps (G default 9)")
        p.add_argument("--side-cache", default=None, help="opt-in G/P immutable label cache")

    p = sub.add_parser("offline", help="offline metrics of a saved model on the validation rows")
    p.add_argument("--model", required=True, help="trained/<name>.npz")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--out", default=None, help="default: next to the model, <name>.offline.json")
    p.add_argument("--allow-subset", action="store_true", help="smoke only: allow subset validation cache")

    args = ap.parse_args(argv)
    if args.cmd == "split":
        split = write_split(args.out, force=args.force)
        print(f"wrote {args.out}: {split['n_kept']} kept episodes, train {len(split['train_episodes'])}, "
              f"val {len(split['val_episodes'])}; strata " +
              ", ".join(f"{k} {v['n_train']}/{v['n_val']}" for k, v in split["strata"].items()))
    elif args.cmd == "build-cache":
        build_cache(args.split, args.cache, workers=args.workers)
    elif args.cmd == "build-ee-cache":
        build_ee_cache(args.cache, args.trigger_cache, args.split, workers=args.workers)
    elif args.cmd == "build-trigger-cache":
        build_trigger_cache(args.split, args.trigger_cache, workers=args.workers)
    elif args.cmd in ("build-chunk-cache", "build-pdref-cache"):
        from results.step6.gp_training import build_side_cache
        build_side_cache("chunk" if args.cmd == "build-chunk-cache" else "pd_reference",
                         args.cache, args.trigger_cache, args.split, args.side_cache,
                         getattr(args, "chunk_k", 10), getattr(args, "chunk_delta", 9),
                         args.workers, args.allow_subset)
    elif args.cmd == "train":
        run_train(args.model, args.seed, args.out, args.cache, args.split, args.max_epochs, args.max_train_episodes,
                  args.max_val_episodes, force=args.force, hide_losses=args.hide_losses,
                  log=lambda s: print(s, flush=True), trigger_copies_k=args.trigger_copies,
                  trigger_qd_jitter=args.trigger_qd_jitter, trigger_dir=args.trigger_cache,
                  extra_rows=args.extra_rows, chunk_k=args.chunk_k, chunk_delta=args.chunk_delta,
                  side_dir=args.side_cache)
    elif args.cmd == "train-many":
        if (args.model in TRIGGER_MODELS) != (args.trigger_copies is not None):
            raise SystemExit(f"--trigger-copies is required for, and only for, the models {TRIGGER_MODELS}")
        extra = ["--hide-losses"] if args.hide_losses else []
        if args.extra_rows is not None:
            extra += ["--extra-rows", args.extra_rows]
        for flag, value in (("--chunk-k", args.chunk_k), ("--chunk-delta", args.chunk_delta),
                            ("--side-cache", args.side_cache)):
            if value is not None:
                extra += [flag, str(value)]
        if args.trigger_copies is not None:
            extra += ["--trigger-copies", str(args.trigger_copies), "--trigger-qd-jitter", str(args.trigger_qd_jitter),
                      "--trigger-cache", args.trigger_cache]
        run_train_many(args.model, parse_seeds(args.seeds), args.jobs, force=args.force, cache_dir=args.cache,
                       split_path=args.split, extra_args=extra)
    elif args.cmd == "offline":
        res = offline_metrics(args.model, args.cache, args.split, allow_subset=args.allow_subset)
        out = args.out or args.model[:-4] + ".offline.json"
        write_json_atomic(out, res)
        if res.get("output_representation") in ("chunk", "pd_reference"):
            print(json.dumps(to_json(res), indent=1))
        else:
            print_offline(res)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
