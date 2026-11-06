"""Training of the step-9 learners (Track B, STEP9_PLAN_PROMPT.md section 5): episode split, signal cache and the six
networks of the configs L0 / L1 / L1-K0 / L2. Features and the model file format come from obstacle_task/features.py
(shared with the deployed policies, obstacle_task/policies.py).

Subcommands:
  split        create trained/step9/data_split.json once from a step-9 set (default data/obstacle/train2): the kept
               episodes (expert status reached, no collision), the stratified episode split (85% train / 15%
               validation, noisy x ambiguous, split seed 0) and one downsampling offset per training episode
  build-cache  extract the measured signals and labels of the training rows (stride 9, per-episode offset), of all
               1 kHz steps of the validation episodes and of each training episode's trigger row (its first
               nav_done step) into results/step9/train_cache/ (one memory-mappable .npy per column and part)
  train        fit networks: --config (comma list of CONFIGS names, or 'all') x --seed (e.g. 0-4) ->
               trained/step9/<config>_s<seed>.npz + .json; a finished model is skipped (resumable)

Examples (from the task directory, PY = pixi run --manifest-path $ROBOENV_DIR/pixi.toml python):
  PY obstacle_task/train.py split --data data/obstacle/train2
  PY obstacle_task/train.py build-cache --workers 12
  PY obstacle_task/train.py train --config L1arm --seed 0          # -> trained/step9/L1arm_s0.npz + .json

Networks (CONFIGS; MLPs of sklearn, Adam 1e-3, alpha 1e-4, batch 200, ReLU, standardised inputs, regressor outputs
standardised; policies.POLICY_MODELS maps the configs onto the policies):
  L0       single84 -> v, w, tau1..7 (9 outputs, equal weight), trigger copies   (the C3 recipe + lidar)
  L1base   base70 -> v, w, trigger copies          L1arm    arm20 -> tau1..7, trigger copies
  L1armK0  arm20 -> tau1..7, no copies (the L1-K0 policy reuses L1base)
  L2cls    MLPClassifier 2x64, base70 rows with t < 1.0 s -> scenario side (+1 left / -1 right), no copies
  L2base   base70_side (base70 + scenario side) -> v, w, trigger copies (the L2 policy reuses L1arm)
Regressors are 2x128. Each network stops early on its own validation loss (regressors: standardised MSE averaged
over the validation rows and its outputs; classifier: mean binary log-loss), patience 8 epochs with
train_policy.EarlyStopping's relative rule (1e-3), at most 150 epochs, and keeps the best epoch's weights.

Training rows, epochs and memory. One epoch is one pass over the training rows in a fresh random order (numpy
default_rng([seed, SHUFFLE_STREAM]).permutation, one per epoch), fed to sklearn's partial_fit in chunks of TRAIN_CHUNK
rows with sklearn's own shuffle off. TRAIN_CHUNK is a multiple of the batch size and partial_fit keeps the Adam state
between calls, so the mini-batches and updates are exactly those of one partial_fit call over the permuted rows
(train_policy.py: one partial_fit call per epoch with sklearn's shuffle; tests/test_obstacle_learning.py checks the
equivalence bit for bit). The chunks exist because the lidar sets do not fit in memory: ~4.4M training rows x 84
float64 features are ~3 GB and the ~7M validation rows (1 kHz) another ~4.7 GB. The features of every chunk are
therefore rebuilt from the cached signals by features.build_features (the lidar by lidar.scan_batch per run of rows
of one episode, features.logged_lidar), and the validation loss is accumulated over sequential chunks of all
validation rows. Measured at train2 scale (4.44M / 6.92M rows): about 0.2-0.3 GB anonymous memory per fit plus the
memory-mapped cache pages (file-backed, shared by parallel fits), peak RSS 1.6-1.9 GB.

Trigger copies (C3, replicated from train_policy.py, STEP3C_PLAN_PROMPT.md 3.1): K = TRIGGER_COPIES copies of each
training episode's trigger row k0 (first nav_done step of the 1 kHz recording), episode-major, appended after the
original rows. Signals and labels are those recorded at k0; each copy's arm qd gets N(0, sigma_hold_j) per joint
from one numpy default_rng(TRIGGER_JITTER_SEED = 0).standard_normal((n_episodes * K, 7)) * sigma_hold, sigma_hold =
train_policy.hold_qd_sigma (per-joint std, ddof 0, of arm qd over the cached training rows with nav_done False).
The same added rows for every seed and config; each network takes its own feature columns and outputs of them (the
base sets have no arm-qd column, so their copies are K identical rows). The standardisation comes from the original
training rows only; the copies are standardised with it; the validation rows are never copied.

Everything runs single-threaded (BLAS threads = 1, as train_policy.single_thread) and is deterministic for a seed.
"""
import os

if __name__ == "__main__":
    # single-threaded BLAS: must be set before numpy is imported (train_policy.single_thread also guards it)
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import bc_features as bcf  # noqa: E402
import train_policy as tp  # noqa: E402
from obstacle_task import features as obf  # noqa: E402

TASK_DIR = _TASK_DIR
DATA_DIR = os.path.join(TASK_DIR, "data", "obstacle", "train2")
TRAINED_DIR = os.path.join(TASK_DIR, "trained", "step9")
SPLIT_PATH = os.path.join(TRAINED_DIR, "data_split.json")
CACHE_DIR = os.path.join(TASK_DIR, "results", "step9", "train_cache")

# ==============================================================================
# Settings of the step-9 plan (STEP9_PLAN_PROMPT.md section 5)
# ==============================================================================
KEEP_STATUS = "reached"   # kept episodes: expert status reached and no collision
VAL_FRAC = 0.15           # validation share per stratum (noisy x ambiguous)
SPLIT_SEED = 0            # numpy default_rng seed of the validation draw
DATA_SEED = 0             # numpy default_rng seed of the per-episode downsampling offsets
STRIDE = 9                # training rows offset::9 (odd: the low-z torque chatter alternates sign every step)
TRIGGER_COPIES = 16       # K of the configs with copies (C3)
TRIGGER_JITTER_SEED = tp.TRIGGER_JITTER_SEED   # 0, the C3 jitter stream
CLS_T_MAX = 1.0           # s, rows of the side classifier: recorded t < CLS_T_MAX
MLP_PARAMS = {**tp.MLP_PARAMS, "shuffle": False}      # 2x128 C3 regressor; the epoch permutation is drawn here
CLS_PARAMS = {**MLP_PARAMS, "hidden_layer_sizes": (64, 64)}
MAX_EPOCHS = 150
PATIENCE = 8              # epochs without a relative improvement of REL_TOL of the reference validation loss
REL_TOL = tp.REL_TOL      # 1e-3, train_policy.EarlyStopping semantics
SHUFFLE_STREAM = 1        # second entry of the epoch-permutation seed sequence [seed, SHUFFLE_STREAM]
TRAIN_CHUNK = 200 * 512   # rows per partial_fit call (a multiple of the batch size 200)
VAL_CHUNK = 65536         # rows per feature build of the validation pass and of the standardisation pass
FWD_CHUNK = tp.FWD_CHUNK  # rows per forward pass inside a chunk

CONFIGS = {  # network name -> kind, feature set, output group (features.OUTPUTS), copies K, rows
    "L0": {"kind": "regressor", "features": "single84", "outputs": "all", "copies": TRIGGER_COPIES, "rows": "all"},
    "L1base": {"kind": "regressor", "features": "base70", "outputs": "vw", "copies": TRIGGER_COPIES, "rows": "all"},
    "L1arm": {"kind": "regressor", "features": "arm20", "outputs": "tau", "copies": TRIGGER_COPIES, "rows": "all"},
    "L1armK0": {"kind": "regressor", "features": "arm20", "outputs": "tau", "copies": 0, "rows": "all"},
    "L2cls": {"kind": "classifier", "features": "base70", "outputs": "side", "copies": 0, "rows": "t<1"},
    "L2base": {"kind": "regressor", "features": "base70_side", "outputs": "vw", "copies": TRIGGER_COPIES,
               "rows": "all"},
}
ROW_RULES = {"all": "every cached row of the part",
             "t<1": f"the cached rows with recorded t < {CLS_T_MAX} s (float32 column t)"}

# Cache columns: name -> (dtype, trailing shape). Raw signals keep the recorded dtype (collect_data.LOG_FIELDS:
# float32; target float64), so the cache is lossless; the (v, w) labels are computed in float64 from the recorded
# clean wheel commands; side is the scenario side of the row's episode.
CACHE_COLUMNS = {
    "episode": (np.int32, ()), "step": (np.int32, ()), "t": (np.float32, ()),
    "base_xy": (np.float32, (2,)), "base_bearing": (np.float32, ()),
    "q_arm": (np.float32, (7,)), "qd_arm": (np.float32, (7,)),
    "target": (np.float64, (3,)),
    "label_v": (np.float64, ()), "label_w": (np.float64, ()), "label_tau": (np.float32, (7,)),
    "nav_done": (np.bool_, ()), "side": (np.int8, ()),
}
NPZ_FIELDS = {"t": np.float32, "base_pos": np.float32, "base_bearing": np.float32, "q_mes": np.float32,
              "qd_mes": np.float32, "wheel_cmds": np.float32, "torque_arm": np.float32, "nav_done": np.bool_}
PARTS = ("train", "val", "trigger")
SPLIT_FORMAT = "obstacle_split_v1"
CACHE_FORMAT = "obstacle_cache_v1"
CODE_FILES = ("obstacle_task/train.py", "obstacle_task/features.py", "obstacle_task/lidar.py", "bc_features.py",
              "train_policy.py")
TARGET_TOL = 1e-9         # m, recorded target vs the scenario's


# ==============================================================================
# Split
# ==============================================================================
def parse_bool(text):
    s = str(text).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", ""):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def stratum_key(noisy, ambiguous):
    return f"{'noisy' if noisy else 'clean'}/{'ambiguous' if ambiguous else 'unambiguous'}"


def data_dir_of(split):
    return os.path.normpath(os.path.join(TASK_DIR, split["data_dir"]))


def read_kept_episodes(data_dir, exclude_nav_timeout=False):
    """Kept episodes of a step-9 set: index.csv status == reached and collided False (the column is optional), and
    with exclude_nav_timeout also nav_timed_out False (the A-to-B filter; off by default, the plan's filter), with
    the per-episode information the split and the cache need, taken from the set's scenarios.json (the noisy flag,
    the ambiguous flag, the side and the obstacles; the index's ambiguous / side / noisy / noise columns must agree
    with it when present). Returns (kept dict episode -> info, excluded dict reason -> episodes, n index rows)."""
    from obstacle_task import scenario   # imported here: it pulls in the simulator modules
    index_path = os.path.join(data_dir, "index.csv")
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))
    scenarios = {int(sc.episode): sc for sc in scenario.load_scenarios(os.path.join(data_dir, "scenarios.json"))[0]}
    missing_cols = {"episode", "file", "status", "n_steps"} - set(rows[0] if rows else {})
    if missing_cols:
        raise ValueError(f"{index_path}: missing columns {sorted(missing_cols)}")
    kept, excluded, seen = {}, {}, set()
    for r in rows:
        e = int(r["episode"])
        if e in seen:
            raise ValueError(f"{index_path}: episode {e} listed twice")
        seen.add(e)
        if e not in scenarios:
            raise ValueError(f"{index_path}: episode {e} has no scenario in scenarios.json")
        sc = scenarios[e]
        collided = parse_bool(r["collided"]) if "collided" in r else False
        timed_out = parse_bool(r["nav_timed_out"]) if "nav_timed_out" in r else None
        if exclude_nav_timeout and timed_out is None:
            raise ValueError(f"{index_path}: no nav_timed_out column to exclude on")
        if r["status"] != KEEP_STATUS or collided or (exclude_nav_timeout and timed_out):
            reason = "collided" if collided else (f"status {r['status']}" if r["status"] != KEEP_STATUS
                                                  else "nav_timed_out")
            excluded.setdefault(reason, []).append(e)
            continue
        if "ambiguous" in r and parse_bool(r["ambiguous"]) != bool(sc.ambiguous):
            raise ValueError(f"{index_path}: episode {e} ambiguous {r['ambiguous']} differs from scenarios.json")
        if "side" in r and int(float(r["side"])) != int(sc.side):
            raise ValueError(f"{index_path}: episode {e} side {r['side']} differs from scenarios.json")
        noise_cols = [c for c in ("noise_v", "noise_w", "noise_torque") if c in r]
        if (noise_cols and any(float(r[c]) > 0 for c in noise_cols) != bool(sc.noisy)) or \
                ("noisy" in r and parse_bool(r["noisy"]) != bool(sc.noisy)):
            raise ValueError(f"{index_path}: episode {e} noisy / noise columns differ from the scenario's noisy flag")
        if not os.path.exists(os.path.join(data_dir, r["file"])):
            raise ValueError(f"{data_dir}: file {r['file']} of episode {e} missing")
        kept[e] = {"file": r["file"], "n_steps": int(r["n_steps"]), "noisy": bool(sc.noisy),
                   "ambiguous": bool(sc.ambiguous), "side": int(sc.side), "lateral_offset": float(sc.lateral_offset),
                   "target": [float(v) for v in sc.target], "obstacles": sc.obstacles, "nav_timed_out": timed_out}
    return kept, {k: sorted(v) for k, v in sorted(excluded.items())}, len(rows)


def make_split(data_dir=DATA_DIR, split_seed=SPLIT_SEED, data_seed=DATA_SEED, stride=STRIDE, val_frac=VAL_FRAC,
               exclude_nav_timeout=False):
    """Episode split as a dict (deterministic given index.csv, scenarios.json and the seeds); the rule of
    train_policy.make_split with the strata noisy x ambiguous."""
    data_dir = os.path.abspath(data_dir)
    kept, excluded, n_index = read_kept_episodes(data_dir, exclude_nav_timeout)
    if not kept:
        raise ValueError(f"{data_dir}: no kept episodes")
    rng = np.random.default_rng(split_seed)
    strata, val = {}, []
    for noisy in (False, True):        # fixed stratum order, so the draw is reproducible
        for ambiguous in (False, True):
            key = stratum_key(noisy, ambiguous)
            eps = sorted(e for e, info in kept.items() if info["noisy"] == noisy and info["ambiguous"] == ambiguous)
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
        "data_dir": os.path.relpath(data_dir, TASK_DIR),
        "index_sha256": tp.sha256_file(os.path.join(data_dir, "index.csv")),
        "scenarios_sha256": tp.sha256_file(os.path.join(data_dir, "scenarios.json")),
        "filter": f"index.csv status == {KEEP_STATUS} and collided False"
                  + (" and nav_timed_out False" if exclude_nav_timeout else ""),
        "n_index_rows": n_index, "n_kept": len(kept), "kept_episodes": sorted(kept), "excluded": excluded,
        "n_kept_nav_timed_out": sum(bool(info["nav_timed_out"]) for info in kept.values()),
        "split_seed": split_seed, "data_seed": data_seed, "stride": stride, "val_frac": val_frac,
        "stratify": "noisy x ambiguous, both from the set's scenarios.json (Scenario.noisy, Scenario.ambiguous)",
        "split_rule": "per stratum (clean unambiguous, clean ambiguous, noisy unambiguous, noisy ambiguous in this "
                      "order), n_val = round(val_frac * n) episodes drawn without replacement with one numpy "
                      "default_rng(split_seed) (train_policy.make_split's rule)",
        "offset_rule": "one offset in [0, stride) per training episode (ascending episode order), "
                       "numpy default_rng(data_seed).integers(0, stride); training rows = steps offset::stride",
        "strata": strata,
        "train_episodes": train,
        "val_episodes": val,
        "train_offsets": [int(o) for o in offsets],
        "episodes": {str(e): kept[e] for e in sorted(kept)},
    }


def write_split(path=SPLIT_PATH, force=False, **kwargs):
    if os.path.exists(path) and not force:
        raise SystemExit(f"{path} exists; every model reuses it. Pass --force to overwrite (invalidates the cache "
                         f"and all models trained on it).")
    split = make_split(**kwargs)
    split["created"] = tp.now_iso()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tp.write_json_atomic(path, split)
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
    """The recorded arrays of one kept episode, checked against the split and the section-4 format."""
    with np.load(job["path"]) as d:
        missing = [k for k in (*NPZ_FIELDS, "target", "meta") if k not in d.files]
        if missing:
            raise ValueError(f"{job['path']}: missing fields {missing}")
        a = {k: d[k] for k in (*NPZ_FIELDS, "target", "meta")}
        a["segment"] = d["segment"] if "segment" in d.files else None
    for k, dt in NPZ_FIELDS.items():
        if a[k].dtype != np.dtype(dt):
            raise ValueError(f"{job['path']}: field {k} has dtype {a[k].dtype}, collect_data.LOG_FIELDS says "
                             f"{np.dtype(dt)}")
    meta = json.loads(str(a["meta"]))
    n = len(a["t"])
    if (n != job["n_steps"] or meta.get("log_every", 1) != 1 or meta.get("status") != KEEP_STATUS
            or bool(meta.get("collided", False))):
        raise ValueError(f"{job['path']}: unexpected episode (n {n} vs {job['n_steps']}, log_every "
                         f"{meta.get('log_every')}, status {meta.get('status')}, collided {meta.get('collided')})")
    if np.max(np.abs(np.asarray(a["target"], dtype=float) - np.asarray(job["target"]))) > TARGET_TOL:
        raise ValueError(f"{job['path']}: recorded target differs from scenarios.json")
    sc = meta.get("scenario")
    if sc is not None and sc.get("obstacles") != job["obstacles"]:
        raise ValueError(f"{job['path']}: obstacles in the episode meta differ from scenarios.json")
    return a, meta


def _episode_rows(a, idx, job):
    """The CACHE_COLUMNS of the recorded steps idx of one episode (recorded dtypes; (v, w) labels in float64)."""
    target = np.asarray(a["target"], dtype=np.float64)
    v, w = bcf.wheels_to_vw(a["wheel_cmds"][idx])
    return {
        "episode": np.full(len(idx), job["episode"], dtype=np.int32), "step": idx.astype(np.int32),
        "t": a["t"][idx], "base_xy": a["base_pos"][idx, :2], "base_bearing": a["base_bearing"][idx],
        "q_arm": a["q_mes"][idx, bcf.ARM], "qd_arm": a["qd_mes"][idx, bcf.ARM],
        "target": np.broadcast_to(target, (len(idx), 3)),
        "label_v": v, "label_w": w, "label_tau": a["torque_arm"][idx], "nav_done": a["nav_done"][idx],
        "side": np.full(len(idx), job["side"], dtype=np.int8),
    }


def _extract_episode(job):
    """Worker: the cached rows of one episode (training rows offset::stride plus its trigger row, or every step)."""
    a, meta = _load_episode(job)
    idx = np.arange(job["offset"], len(a["t"]), job["stride"])
    rows, trig, info = _episode_rows(a, idx, job), None, {}
    if job["part"] == "train":
        nav = np.asarray(a["nav_done"], dtype=bool)
        if not nav.any():
            raise ValueError(f"{job['path']}: no nav_done step")
        k0 = int(np.argmax(nav))
        trig = _episode_rows(a, np.array([k0]), job)
        info = {"k0": k0, "k0_t": float(a["t"][k0]),
                "k0_segment": None if a["segment"] is None else int(a["segment"][k0])}
    return job["part"], job["episode"], rows, trig, info, float(meta["time_step"])


def build_cache(split_path=SPLIT_PATH, cache_dir=CACHE_DIR, workers=12, train_episodes=None, val_episodes=None,
                verbose=True):
    """Extract the cache of the split (or of a subset of its episodes, for tests and smoke runs): parts train
    (stride-9 rows), val (every step) and trigger (one row per training episode, its first nav_done step), each
    <cache_dir>/<part>/<column>.npy, plus manifest.json written last. Returns the manifest."""
    split = load_split(split_path)
    data_dir = data_dir_of(split)
    stride = int(split["stride"])
    offsets = dict(zip(split["train_episodes"], split["train_offsets"]))
    parts = {"train": list(split["train_episodes"]), "val": list(split["val_episodes"])}
    subset = train_episodes is not None or val_episodes is not None
    if train_episodes is not None:
        assert set(train_episodes) <= set(parts["train"]), "train subset not in the split"
        parts["train"] = sorted(int(e) for e in train_episodes)
    if val_episodes is not None:
        assert set(val_episodes) <= set(parts["val"]), "val subset not in the split"
        parts["val"] = sorted(int(e) for e in val_episodes)
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)  # a half-written cache must never look valid
    t_start = time.time()
    jobs, layout = [], {}
    for part, eps in parts.items():
        rows, starts, total = [], [], 0
        for e in eps:
            info = split["episodes"][str(e)]
            off, st = (offsets[e], stride) if part == "train" else (0, 1)
            k = tp.n_rows_of(info["n_steps"], off, st)
            starts.append(total)
            rows.append(k)
            total += k
            jobs.append({"path": os.path.join(data_dir, info["file"]), "episode": int(e), "part": part,
                         "offset": int(off), "stride": int(st), "n_steps": int(info["n_steps"]),
                         "side": int(info["side"]), "target": info["target"], "obstacles": info["obstacles"]})
        layout[part] = {"episodes": eps, "row_start": starts, "n_rows_episode": rows, "n_rows": total}
    n_tr = len(parts["train"])
    layout["trigger"] = {"episodes": parts["train"], "row_start": list(range(n_tr)), "n_rows_episode": [1] * n_tr,
                         "n_rows": n_tr}
    arrays = {}
    for part, lay in layout.items():
        os.makedirs(os.path.join(cache_dir, part), exist_ok=True)
        arrays[part] = {name: np.lib.format.open_memmap(os.path.join(cache_dir, part, name + ".npy"), mode="w+",
                                                        dtype=dt, shape=(lay["n_rows"], *shape))
                        for name, (dt, shape) in CACHE_COLUMNS.items()}
    pos = {part: dict(zip(lay["episodes"], lay["row_start"])) for part, lay in layout.items()}
    time_steps, trig_info = set(), {}
    workers = max(1, min(int(workers), 12))
    with mp.get_context("forkserver").Pool(workers) as pool:
        for i, (part, e, out, trig, info, dt) in enumerate(pool.imap(_extract_episode, jobs, chunksize=2)):
            s = pos[part][e]
            for name, arr in out.items():
                arrays[part][name][s:s + len(arr)] = arr
            if trig is not None:
                r = pos["trigger"][e]
                for name, arr in trig.items():
                    arrays["trigger"][name][r:r + 1] = arr
                trig_info[e] = info
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
    k0 = [trig_info[e]["k0"] for e in parts["train"]]
    manifest = {
        "format": CACHE_FORMAT, "created": tp.now_iso(),
        "split_file": os.path.relpath(os.path.abspath(split_path), TASK_DIR),
        "split_sha256": tp.sha256_file(split_path),
        "stride": stride, "time_step": time_steps.pop(), "subset": subset,
        "columns": {name: {"dtype": np.dtype(dt).name, "shape": list(shape)} for name, (dt, shape) in
                    CACHE_COLUMNS.items()},
        "parts": layout, "n_rows": {part: lay["n_rows"] for part, lay in layout.items()},
        "trigger": {"k0_steps": k0, "k0_t": [trig_info[e]["k0_t"] for e in parts["train"]],
                    "k0_segment": [trig_info[e]["k0_segment"] for e in parts["train"]],
                    "rule": "one row per training episode (order of the train part): the recorded step k0 = first "
                            "step with nav_done, columns as the other parts, labels = the recorded clean commands"},
        "per_row_episode_id": "column 'episode' (int32); rows of an episode are contiguous, in step order "
                              "(column 'step'); parts.<part>.row_start / n_rows_episode give the ranges",
        "build_seconds": time.time() - t_start, "workers": workers, "size_bytes": size,
    }
    tp.write_json_atomic(manifest_path, manifest)
    if verbose:
        print(f"cache {cache_dir}: train {layout['train']['n_rows']} rows ({n_tr} episodes), val "
              f"{layout['val']['n_rows']} rows ({len(parts['val'])} episodes), trigger {n_tr} rows (k0 median "
              f"{np.median(k0):.0f}), {size / 2**20:.0f} MiB, {manifest['build_seconds']:.1f} s with {workers} "
              f"workers")
    return manifest


class CachePart:
    """Memory-mapped cache of one part ('train', 'val' or 'trigger'), checked against the split file."""

    def __init__(self, part, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, allow_subset=False):
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}")
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise RuntimeError(f"no cache manifest in {cache_dir}: run 'obstacle_task/train.py build-cache'")
        with open(manifest_path) as f:
            m = self.manifest = json.load(f)
        split = load_split(split_path)
        if m.get("format") != CACHE_FORMAT:
            raise RuntimeError(f"{manifest_path}: not a {CACHE_FORMAT} cache")
        if m["split_sha256"] != tp.sha256_file(split_path):
            raise RuntimeError(f"cache {cache_dir} was built from another split file than {split_path}: rebuild it "
                               f"with 'obstacle_task/train.py build-cache'")
        if int(m["stride"]) != int(split["stride"]):
            raise RuntimeError("cache stride differs from the split file")
        if m["subset"] and not allow_subset:
            raise RuntimeError(f"cache {cache_dir} holds a subset of the split (test / smoke cache)")
        lay = m["parts"][part]
        expected = split["val_episodes"] if part == "val" else split["train_episodes"]
        if not m["subset"] and list(lay["episodes"]) != list(expected):
            raise RuntimeError(f"cache {cache_dir}: {part} episodes differ from the split file")
        if m["columns"] != {n: {"dtype": np.dtype(dt).name, "shape": list(s)} for n, (dt, s) in CACHE_COLUMNS.items()}:
            raise RuntimeError(f"cache {cache_dir}: columns differ from CACHE_COLUMNS")
        self.part, self.cache_dir, self.split = part, cache_dir, split
        self.episodes = [int(e) for e in lay["episodes"]]
        self.row_start = np.asarray(lay["row_start"], dtype=np.int64)
        self.n_rows_episode = np.asarray(lay["n_rows_episode"], dtype=np.int64)
        self.n_rows = int(lay["n_rows"])
        self.time_step = float(m["time_step"])
        self.manifest_sha256 = tp.sha256_file(manifest_path)
        self.obstacles = {e: split["episodes"][str(e)]["obstacles"] for e in self.episodes}
        self.cols = {}
        for name in CACHE_COLUMNS:
            arr = np.load(os.path.join(cache_dir, part, name + ".npy"), mmap_mode="r")
            if arr.shape[0] != self.n_rows:
                raise RuntimeError(f"cache {cache_dir}/{part}/{name}.npy: {arr.shape[0]} rows, manifest {self.n_rows}")
            self.cols[name] = arr

    def episode_slice(self, i):
        s = int(self.row_start[i])
        return slice(s, s + int(self.n_rows_episode[i]))


def select_rows(part, rule):
    """Cache rows (ascending int64 indices) of a part used by a network (ROW_RULES)."""
    if rule == "all":
        return np.arange(part.n_rows, dtype=np.int64)
    if rule == "t<1":
        return np.flatnonzero(np.asarray(part.cols["t"]) < CLS_T_MAX).astype(np.int64)
    raise ValueError(f"unknown row rule {rule!r}")


def cache_signals(part, rows, with_lidar):
    """features.logged_signals of the cached rows (ascending indices): the cached arm columns are placed back into
    the recorded q_mes / qd_mes layout (as train_policy.signals_from_cache), the lidar is recomputed per episode with
    lidar.scan_batch from the cached base pose and the episode's obstacles."""
    c = part.cols
    n = len(rows)
    base_pos = np.zeros((n, 3))
    base_pos[:, :2] = c["base_xy"][rows]
    q = np.zeros((n, 17))
    q[:, bcf.ARM] = c["q_arm"][rows]
    qd = np.zeros((n, 17))
    qd[:, bcf.ARM] = c["qd_arm"][rows]
    return obf.logged_signals(base_pos, np.asarray(c["base_bearing"][rows]), q, qd, np.asarray(c["target"][rows]),
                              obstacles=part.obstacles if with_lidar else None,
                              episode=np.asarray(c["episode"][rows]) if with_lidar else None)


def cache_labels(cols, rows, outputs):
    """Labels of the cached rows: float64 (N, n_out) of the output group, or the side (N,) int64 for 'side'."""
    if outputs == "side":
        return np.asarray(cols["side"][rows], dtype=np.int64)
    parts = {"all": ["label_v", "label_w", "label_tau"], "vw": ["label_v", "label_w"], "tau": ["label_tau"]}[outputs]
    return np.column_stack([np.asarray(cols[name][rows], dtype=np.float64) for name in parts])


def raw_rows(part, rows, feature_set, outputs):
    """Raw (not standardised) features (N, F) and labels of the cached rows `rows` of a part, in the given order.
    The rows are built in ascending order (one lidar.scan_batch call per run of one episode) and put back."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable") if rows.size > 1 and np.any(rows[1:] < rows[:-1]) else None
    srt = rows if order is None else rows[order]
    sig = cache_signals(part, srt, feature_set in obf.LIDAR_SETS)
    side = np.asarray(part.cols["side"][srt]) if feature_set in obf.SIDE_SETS else None
    X = obf.build_features(feature_set, sig, side=side)
    Y = cache_labels(part.cols, srt, outputs)
    if order is not None:
        Xo, Yo = np.empty_like(X), np.empty_like(Y)
        Xo[order], Yo[order] = X, Y
        X, Y = Xo, Yo
    return X, Y


def trigger_copies(trig, feature_set, outputs, copies, qd_sigma, jitter_seed=TRIGGER_JITTER_SEED):
    """Raw features and labels of `copies` copies of every trigger row (episode-major), train_policy.trigger_copies
    with the step-9 feature sets: every copy keeps the recorded signals (lidar from the recorded k0 pose) and labels;
    only its arm qd gets N(0, qd_sigma_j) per joint, drawn once from numpy default_rng(jitter_seed) as one
    (n_rows * copies, 7) standard-normal array times qd_sigma. Returns (X, Y, noise)."""
    n = trig.n_rows
    k = int(copies)
    if k < 0:
        raise ValueError("copies must be >= 0")
    rows = np.arange(n, dtype=np.int64)
    rep = np.repeat(np.arange(n), k)
    sig = cache_signals(trig, rows, feature_set in obf.LIDAR_SETS)
    sig = {key: np.asarray(val)[rep] for key, val in sig.items()}
    noise = np.random.default_rng(jitter_seed).standard_normal((n * k, 7)) * np.asarray(qd_sigma, dtype=float)
    sig["qd_arm"] = sig["qd_arm"] + noise
    side = np.asarray(trig.cols["side"])[rep] if feature_set in obf.SIDE_SETS else None
    X = obf.build_features(feature_set, sig, side=side)
    Y = cache_labels(trig.cols, rows, outputs)[rep]
    return X, Y, noise


def column_moments(chunks):
    """Column mean and scale (std, ddof 0; std < SCALE_EPS -> 1) over a sequence of row chunks, merged with Chan's
    pairwise update (numerically stable, one pass, deterministic)."""
    n, mean, m2 = 0, None, None
    for A in chunks:
        nb = A.shape[0]
        if nb == 0:
            continue
        mb = A.mean(axis=0)
        m2b = ((A - mb) ** 2).sum(axis=0)
        if mean is None:
            n, mean, m2 = nb, mb, m2b
        else:
            delta, tot = mb - mean, n + nb
            mean = mean + delta * (nb / tot)
            m2 = m2 + m2b + delta ** 2 * (n * nb / tot)
            n = tot
    if mean is None:
        raise ValueError("no rows")
    scale = np.sqrt(m2 / n)
    scale[scale < tp.SCALE_EPS] = 1.0
    return mean, scale


# ==============================================================================
# One network's data: rows, standardisation, copies, chunks
# ==============================================================================
class NetData:
    """The training rows (cached rows + trigger copies) and validation rows of one network of CONFIGS, served as
    standardised chunks; features are rebuilt from the cache for every chunk (see the module docstring)."""

    def __init__(self, config, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, allow_subset=False, log=None):
        spec = self.spec = CONFIGS[config]
        self.config, self.kind = config, spec["kind"]
        self.feature_set, self.outputs = spec["features"], spec["outputs"]
        self.tr = CachePart("train", cache_dir, split_path, allow_subset)
        self.va = CachePart("val", cache_dir, split_path, allow_subset)
        self.train_rows = select_rows(self.tr, spec["rows"])
        self.val_rows = select_rows(self.va, spec["rows"])
        if len(self.train_rows) == 0 or len(self.val_rows) == 0:
            raise RuntimeError(f"{config}: no training or validation rows")
        t0 = time.time()
        if self.kind == "regressor":
            # one pass for X and Y together: Y columns are appended to the moments' chunks
            mean, scale = column_moments(np.column_stack(self.raw(self.tr, r)) for r in self.chunks(self.train_rows))
            n_x = len(obf.FEATURE_SETS[self.feature_set])
            self.xm, self.xs, self.ym, self.ys = mean[:n_x], scale[:n_x], mean[n_x:], scale[n_x:]
        else:
            self.xm, self.xs = column_moments(self.raw(self.tr, r)[0] for r in self.chunks(self.train_rows))
            self.ym, self.ys = np.zeros(1), np.ones(1)   # a classifier's logit: stored for the model file
        self.std_s = time.time() - t0
        self.n_base = len(self.train_rows)
        self.n_copies, self.Xc, self.Yc, self.trigger_meta = 0, None, None, None
        if spec["copies"]:
            self._add_copies(int(spec["copies"]), cache_dir, split_path, allow_subset)
        self.n_train = self.n_base + self.n_copies
        if log:
            log(f"{config} ({self.kind}, features {self.feature_set}, outputs {self.outputs}, rows "
                f"{spec['rows']}): train {self.n_base} rows ({len(self.tr.episodes)} episodes) + {self.n_copies} "
                f"copies, val {len(self.val_rows)} rows ({len(self.va.episodes)} episodes); standardisation "
                f"{self.std_s:.1f} s")

    def _add_copies(self, k, cache_dir, split_path, allow_subset):
        trig = CachePart("trigger", cache_dir, split_path, allow_subset)
        if trig.episodes != self.tr.episodes:
            raise RuntimeError("trigger part and train part hold different episodes")
        sigma = tp.hold_qd_sigma(self.tr.cols)   # per-joint std of arm qd over the training rows with nav_done False
        X, Y, _ = trigger_copies(trig, self.feature_set, self.outputs, k, sigma)
        self.Xc = tp.standardise_inplace(X, self.xm, self.xs)
        self.Yc = tp.standardise_inplace(Y, self.ym, self.ys)
        self.n_copies = X.shape[0]
        self.trigger_meta = {
            "copies": k, "n_episodes": trig.n_rows, "n_rows_added": self.n_copies, "n_train_rows_base": self.n_base,
            "n_train_rows_total": self.n_base + self.n_copies,
            "row_share": self.n_copies / (self.n_base + self.n_copies),
            "k0_steps": np.asarray(trig.cols["step"]).tolist(), "qd_jitter": "auto", "qd_sigma": sigma.tolist(),
            "qd_sigma_hold": sigma.tolist(), "jitter_seed": TRIGGER_JITTER_SEED,
            "rule": "K copies of each training episode's trigger row k0 (first nav_done step, 1 kHz recording), "
                    "signals and labels as recorded; arm qd + N(0, qd_sigma_j) per joint, one numpy "
                    "default_rng(jitter_seed).standard_normal((n_episodes * K, 7)) * qd_sigma, episode-major copies; "
                    "qd_sigma = qd_sigma_hold = per-joint std of arm qd over the training rows with nav_done False "
                    "(train_policy.hold_qd_sigma); standardisation from the original training rows only; "
                    "validation rows unchanged (train_policy.py C3 rule)",
        }

    @staticmethod
    def chunks(rows, size=VAL_CHUNK):
        return (rows[s:s + size] for s in range(0, len(rows), size))

    def raw(self, part, rows):
        return raw_rows(part, rows, self.feature_set, self.outputs)

    def standardised(self, part, rows):
        X, Y = self.raw(part, rows)
        tp.standardise_inplace(X, self.xm, self.xs)
        if self.kind == "regressor":
            tp.standardise_inplace(Y, self.ym, self.ys)
        return X, Y

    def train_chunk(self, idx):
        """Standardised (X, Y) of the training positions idx (0..n_base-1: cached rows, n_base..: copies), in order."""
        idx = np.asarray(idx, dtype=np.int64)
        is_copy = idx >= self.n_base
        if not is_copy.any():
            return self.standardised(self.tr, self.train_rows[idx])
        n_x = len(obf.FEATURE_SETS[self.feature_set])
        X = np.empty((len(idx), n_x))
        Y = np.empty((len(idx), self.Yc.shape[1]))
        base = np.flatnonzero(~is_copy)
        if base.size:
            X[base], Y[base] = self.standardised(self.tr, self.train_rows[idx[base]])
        cp = np.flatnonzero(is_copy)
        X[cp], Y[cp] = self.Xc[idx[cp] - self.n_base], self.Yc[idx[cp] - self.n_base]
        return X, Y

    def val_chunks(self):
        for rows in self.chunks(self.val_rows):
            yield self.standardised(self.va, rows)

    def val_loss(self, weights, biases):
        """Regressor: standardised MSE averaged over all validation rows and outputs. Classifier: mean binary
        log-loss, probabilities clipped to [eps, 1 - eps] as sklearn's binary_log_loss."""
        from scipy.special import expit   # sklearn's logistic output activation
        net = {"x_mean": 0.0, "x_scale": 1.0, "y_mean": 0.0, "y_scale": 1.0, "weights": weights, "biases": biases}
        eps = np.finfo(np.float64).eps
        total, n = 0.0, 0
        for X, Y in self.val_chunks():
            for s in range(0, X.shape[0], FWD_CHUNK):
                out = bcf.forward(net, X[s:s + FWD_CHUNK])
                if self.kind == "regressor":
                    d = out - Y[s:s + FWD_CHUNK]
                    total += float(np.einsum("ij,ij->", d, d))
                    n += d.size
                else:
                    p = np.clip(expit(out[:, 0]), eps, 1 - eps)
                    left = Y[s:s + FWD_CHUNK] == obf.CLASSES[1]
                    total += float(-np.sum(np.where(left, np.log(p), np.log1p(-p))))
                    n += len(p)
        return total / n


# ==============================================================================
# Fitting
# ==============================================================================
def make_net(kind, seed):
    """sklearn MLP of a network kind; a RandomState instance (not an int), as train_policy.make_mlp."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    cls, params = (MLPClassifier, CLS_PARAMS) if kind == "classifier" else (MLPRegressor, MLP_PARAMS)
    return cls(**params, random_state=np.random.RandomState(seed))


def fit_epoch(net, data, perm, chunk=TRAIN_CHUNK):
    """One epoch: partial_fit on consecutive chunks of the permutation perm. chunk is a multiple of the batch size, so
    the batches (and the Adam updates) are those of one partial_fit call over the permuted rows. Returns the epoch's
    training loss with sklearn's loss_ definition (mean over the epoch's rows of the batch losses incl. L2 term)."""
    assert chunk % MLP_PARAMS["batch_size"] == 0
    loss_sum = 0.0
    for s in range(0, len(perm), chunk):
        X, Y = data.train_chunk(perm[s:s + chunk])
        with warnings.catch_warnings():   # a last chunk smaller than a batch: sklearn warns and clips the batch
            warnings.filterwarnings("ignore", message="Got `batch_size` less than 1 or larger than sample size")
            if data.kind == "classifier":
                net.partial_fit(X, Y, classes=list(obf.CLASSES))
            else:
                net.partial_fit(X, Y)
        loss_sum += float(net.loss_) * X.shape[0]
    return loss_sum / len(perm)


def train_net(data, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE, rel_tol=REL_TOL, log=None):
    """Fit one network with early stopping on its own validation loss. Returns (best coefs, best intercepts, history,
    the sklearn net with its last-epoch weights)."""
    net = make_net(data.kind, seed)
    stopper = tp.EarlyStopping(patience, rel_tol)
    rng = np.random.default_rng([int(seed), SHUFFLE_STREAM])
    best_w = best_b = None
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "fit_s": [], "val_s": []}
    stop_reason = "max_epochs"
    batches = -(-data.n_train // MLP_PARAMS["batch_size"])
    with tp.single_thread():
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = fit_epoch(net, data, rng.permutation(data.n_train))
            t1 = time.time()
            # exactly one pass per epoch, with the batches of one partial_fit call (Adam step count)
            if net.t_ != epoch * data.n_train or net._optimizer.t != epoch * batches:
                raise RuntimeError(f"epoch {epoch}: t_ {net.t_}, Adam steps {net._optimizer.t}, expected "
                                   f"{epoch * data.n_train} and {epoch * batches}")
            val = data.val_loss(net.coefs_, net.intercepts_)
            t2 = time.time()
            new_best, stop = stopper.update(epoch, val)
            if new_best:
                best_w = [c.copy() for c in net.coefs_]
                best_b = [b.copy() for b in net.intercepts_]
            for key, v in (("epoch", epoch), ("train_loss", train_loss), ("val_loss", val), ("fit_s", t1 - t0),
                           ("val_s", t2 - t1)):
                hist[key].append(v)
            if log:
                mem = tp.mem_status()
                log(f"epoch {epoch:3d} train_loss {train_loss:.6f} val_loss {val:.6f} (best {stopper.best:.6f} @ "
                    f"{stopper.best_epoch}) fit {t1 - t0:.1f} s val {t2 - t1:.1f} s, MiB: peak RSS "
                    f"{mem.get('VmHWM', tp.peak_rss_mb()):.0f}, anon {mem.get('RssAnon', float('nan')):.0f}, "
                    f"file {mem.get('RssFile', float('nan')):.0f}")
            if not np.isfinite(val):
                stop_reason = "non_finite_val_loss"
                break
            if stop:
                stop_reason = f"no relative improvement > {rel_tol:g} for {patience} consecutive epochs"
                break
    if best_w is None:
        raise RuntimeError("no finite validation loss")
    hist.update(best_epoch=stopper.best_epoch, best_val_loss=stopper.best, epochs_run=len(hist["epoch"]),
                stop_reason=stop_reason)
    return best_w, best_b, hist, net


def model_paths(config, seed, out_dir=TRAINED_DIR):
    base = os.path.join(out_dir, f"{config}_s{seed}")
    return base + ".npz", base + ".json"


def is_finished(config, seed, out_dir=TRAINED_DIR):
    npz, js = model_paths(config, seed, out_dir)
    if not (os.path.exists(npz) and os.path.exists(js)):
        return False
    try:
        with open(js) as f:
            return bool(json.load(f).get("complete"))
    except (OSError, ValueError):
        return False


def run_train(config, seed, out_dir=TRAINED_DIR, cache_dir=CACHE_DIR, split_path=SPLIT_PATH, max_epochs=MAX_EPOCHS,
              force=False, log=print):
    """Train network `config` with `seed` -> <out_dir>/<config>_s<seed>.npz (features.save_model) + .json (its meta,
    written last: 'complete' marks a finished model). A finished model is skipped unless force. Returns
    (npz path, meta, sklearn net with the best weights; None when skipped)."""
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}, expected one of {list(CONFIGS)}")
    npz, js = model_paths(config, seed, out_dir)
    official = os.path.abspath(out_dir) == os.path.abspath(TRAINED_DIR)
    if official and (max_epochs != MAX_EPOCHS or os.path.abspath(split_path) != os.path.abspath(SPLIT_PATH)
                     or os.path.abspath(cache_dir) != os.path.abspath(CACHE_DIR)):
        raise SystemExit("non-default settings (--max-epochs / --split / --cache) need an explicit --out-dir, so a "
                         "smoke model never takes an official trained/step9 name")
    if is_finished(config, seed, out_dir) and not force:
        log(f"skip {config} seed {seed}: {npz} finished")
        with open(js) as f:
            return npz, json.load(f), None
    import sklearn
    spec = CONFIGS[config]
    t_start = time.time()
    with tp.single_thread():   # the whole fit (data, epochs, the final check) with one BLAS thread
        data = NetData(config, cache_dir, split_path, allow_subset=not official, log=log)
        t_load = time.time() - t_start
        w, b, hist, net = train_net(data, seed, max_epochs=max_epochs, log=log)
        check = data.val_loss(w, b)
    assert abs(check - hist["best_val_loss"]) <= 1e-12 * max(1.0, abs(check)), "best weights do not match"
    net.coefs_, net.intercepts_ = w, b   # the returned net carries the saved (best-epoch) weights
    params = CLS_PARAMS if data.kind == "classifier" else MLP_PARAMS
    meta = {
        "config": config, "kind": data.kind, "feature_set": data.feature_set,
        "feature_names": obf.FEATURE_SETS[data.feature_set], "output_group": data.outputs,
        "output_names": obf.OUTPUTS[data.outputs], "seed": int(seed), "rows": spec["rows"],
        "rows_rule": ROW_RULES[spec["rows"]], "classes": list(obf.CLASSES) if data.kind == "classifier" else None,
        "split_file": os.path.relpath(os.path.abspath(split_path), TASK_DIR),
        "split_sha256": tp.sha256_file(split_path), "cache_dir": os.path.relpath(os.path.abspath(cache_dir), TASK_DIR),
        "cache_manifest_sha256": data.tr.manifest_sha256, "data_dir": data.tr.split["data_dir"],
        "stride": int(data.tr.split["stride"]), "n_train_rows": data.n_train, "n_train_rows_base": data.n_base,
        "n_val_rows": int(len(data.val_rows)), "n_train_episodes": len(data.tr.episodes),
        "n_val_episodes": len(data.va.episodes), "subset": bool(data.tr.manifest["subset"]),
        **({"trigger": data.trigger_meta} if data.trigger_meta is not None else {"trigger": None}),
        "standardisation": ("X with the mean and std (ddof 0, std < 1e-12 -> 1) of the original training rows (no "
                            "copies), " + ("Y likewise; the trigger copies are standardised with them"
                                           if data.kind == "regressor" else "Y not standardised (class labels)")),
        "val_loss_definition": ("standardised MSE averaged over all validation rows (1 kHz) and the outputs"
                                if data.kind == "regressor" else
                                "mean binary log-loss over the validation rows, p clipped to [eps, 1 - eps]"),
        "train_loss_definition": "sklearn loss_ of the epoch's partial_fit calls, row-weighted: 0.5 * mean squared "
                                 "standardised error (regressor) or binary log-loss (classifier) over the epoch's "
                                 "batches + the L2 penalty (alpha)",
        "hyperparameters": {**params, "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
                            "random_state": f"np.random.RandomState({seed})", "max_epochs": max_epochs,
                            "patience": PATIENCE, "rel_tol": REL_TOL, "dtype": "float64",
                            "epoch": f"one pass over a numpy default_rng([seed, {SHUFFLE_STREAM}]).permutation of "
                                     f"the training rows (one per epoch), partial_fit in chunks of {TRAIN_CHUNK} rows "
                                     f"(sklearn shuffle off; same batches as one partial_fit call)"},
        "epochs_run": hist["epochs_run"], "best_epoch": hist["best_epoch"], "val_loss": hist["best_val_loss"],
        "stop_reason": hist["stop_reason"], "history": {k: hist[k] for k in ("epoch", "train_loss", "val_loss",
                                                                            "fit_s", "val_s")},
        "n_params": int(sum(x.size + y.size for x, y in zip(w, b))),
        "code_sha256": {f: tp.sha256_file(os.path.join(TASK_DIR, f)) for f in CODE_FILES},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "created": tp.now_iso(),
        "load_s": t_load, "train_s": time.time() - t_start - t_load,
    }
    meta["peak_rss_mib"] = tp.peak_rss_mb()
    meta["memory_end_mib"] = tp.mem_status()
    meta = tp.to_json(meta)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f".{config}_s{seed}.tmp.npz")
    obf.save_model(tmp, data.kind, data.feature_set, obf.OUTPUTS[data.outputs], data.xm, data.xs, data.ym, data.ys,
                   w, b, meta)
    os.replace(tmp, npz)
    tp.write_json_atomic(js, {**meta, "model_file": os.path.basename(npz), "model_sha256": tp.sha256_file(npz),
                              "complete": True})
    log(f"saved {npz}: {meta['epochs_run']} epochs, best epoch {meta['best_epoch']}, val loss {meta['val_loss']:.6f}, "
        f"train {meta['train_s']:.0f} s, peak RSS {meta['peak_rss_mib']:.0f} MiB")
    return npz, meta, net


def parse_configs(text):
    names = list(CONFIGS) if text.strip() == "all" else [s.strip() for s in text.split(",") if s.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}, expected names of {list(CONFIGS)} or 'all'")
    return names


# ==============================================================================
# Command line
# ==============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="create the episode split file (once)")
    p.add_argument("--data", default=DATA_DIR, help="step-9 set (scenarios.json, index.csv, ep_XXXX.npz)")
    p.add_argument("--out", default=SPLIT_PATH)
    p.add_argument("--force", action="store_true", help="overwrite an existing split file")
    p.add_argument("--exclude-nav-timeout", action="store_true",
                   help="also drop episodes with nav_timed_out True (the A-to-B filter; the plan's filter keeps them)")

    p = sub.add_parser("build-cache", help="extract the training / validation / trigger rows into the cache")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--workers", type=int, default=12, help="pool size (at most 12)")

    p = sub.add_parser("train", help="fit networks (a finished model is skipped)")
    p.add_argument("--config", required=True, help=f"comma list of {list(CONFIGS)} or 'all'")
    p.add_argument("--seed", required=True, help="e.g. 0, 0-4 or 100,101 (one model per config and seed, in order)")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--out-dir", default=TRAINED_DIR)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="smoke tests only (needs --out-dir)")
    p.add_argument("--force", action="store_true", help="retrain finished models")

    args = ap.parse_args(argv)
    if args.cmd == "split":
        split = write_split(args.out, force=args.force, data_dir=args.data,
                            exclude_nav_timeout=args.exclude_nav_timeout)
        print(f"wrote {args.out}: {split['n_kept']} kept of {split['n_index_rows']} episodes (excluded "
              + (", ".join(f"{k}: {len(v)}" for k, v in split["excluded"].items()) or "none")
              + f"), train {len(split['train_episodes'])}, val {len(split['val_episodes'])}; strata "
              + ", ".join(f"{k} {v['n_train']}/{v['n_val']}" for k, v in split["strata"].items())
              + f"; kept with nav_timed_out: {split['n_kept_nav_timed_out']}")
    elif args.cmd == "build-cache":
        build_cache(args.split, args.cache, workers=args.workers)
    elif args.cmd == "train":
        for config in parse_configs(args.config):
            for seed in tp.parse_seeds(args.seed):
                run_train(config, seed, args.out_dir, args.cache, args.split, args.max_epochs, force=args.force,
                          log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
