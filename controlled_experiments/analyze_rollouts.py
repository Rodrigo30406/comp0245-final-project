"""Closed-loop result analysis of step 3 (module D): failure classes, paired multi-seed statistics, experiment A
(two models compared on the same episodes), experiment B (base / arm 2x2 split with the hybrid policy), descriptive
analyses and figures. Reads the run folders written by evaluate_policy.py (results/<run>/rollouts.csv, config.json,
logs/) and never runs a simulation.

Run folder names (parsed into model / seed / cell / set; see parse_run_name):
  results/<model>_s<seed>_<set>          MLP runs (model ego, ego-novw, raw)
  results/<model>_<set>                  OLS runs (model ego-ols, raw-ols)
  results/expert_v11_<set>, results/zero_v11_<set>   expert / zero-policy baselines (protocol v1.1)
  results/expert_<set>                   expert baselines of protocol v1 (no v1.1 columns)
  results/B-<EM|ME>_<model>_s<seed>_<set>, results/B-EE_<set>   experiment-B hybrid cells
Per-episode references (expert nav_done time and EE error there, mode, required turns, coverage percentiles) are
computed from the recorded expert episodes of the evaluation set and cached in results/step3/cache_ref_<set>.csv.
data/test is sealed: every entry point refuses it.

Integrity of paired analyses (compare, split2x2, describe): the runs must come from the same evaluation set
(config.json eval_set), cover exactly its full episode set (config.json eval_set_config n_episodes), contain no
harness_error episode (a harness failure to re-run, not a policy failure), and a hybrid run must have been made with
--no-early-stop. Otherwise the analysis refuses, unless --allow-incomplete is given (then the problems are printed as
warnings and harness_error episodes are left out of every arm).

Statistics: every paired difference is reported with the pre-registered two-level percentile bootstrap AND a
seed-level calibrated companion (seed_t_test: Welch t on the per-seed rates of two arms; one-sample t on the per-seed
differences against a single deterministic run, e.g. the expert cell EE). With S = 5 seeds and a large seed variance
the bootstrap is anti-conservative (step-3 review: P(CI lower bound > 0) about 0.05-0.07 under the null, nominal
0.025). Experiment A's adoption condition (1) needs both lower bounds > 0; experiment B's Holm uses max(bootstrap p,
t p). Holm rejects at p_adj <= alpha; the strata family is the d0 strata with n >= MIN_STRATUM_N.

Examples:
  python analyze_rollouts.py refs --sets dev hard                    # build + cross-check the episode references
  python analyze_rollouts.py classify results/expert_v11_dev         # failure classes of runs
  python analyze_rollouts.py validate                                # classifier validation (CONFIRM: --allow-confirm)
  python analyze_rollouts.py compare --a ego --b ego-novw --set dev --seeds 0-7 --predicted-classes no_start,left_or_overshot
  python analyze_rollouts.py --allow-incomplete compare --a-runs <run A> --b-runs <run B>   # subsets: warnings only
  python analyze_rollouts.py split2x2 --model ego --set dev --seeds 0-4
  python analyze_rollouts.py describe --sets dev hard
  python analyze_rollouts.py report                                  # results/step3/summary.txt with all available parts
  python analyze_rollouts.py precheck-recorded --sets dev hard       # development check on the recorded expert episodes
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np
from scipy import stats

import analyze_dataset as ad

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
STEP3_DIR = os.path.join(RESULTS_DIR, "step3")
DATA_DIR = os.path.join(HERE, "data")
TRAINED_DIR = os.path.join(HERE, "trained")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "data_analysis")

SEALED_SETS = ("test",)            # never opened, listed, loaded or evaluated
# data/confirm: its recorded expert episodes are read (for the references) only when this is set, by the
# --allow-confirm flag of the CLI (the confirmation set is looked at once, after a decision)
ALLOW_CONFIRM = False
# --allow-incomplete: integrity problems of paired runs (check_integrity) become warnings instead of a refusal
ALLOW_INCOMPLETE = False
KNOWN_SETS = ("dev", "hard", "pilot", "confirm")
N_BOOT = 10000
BOOT_SEED = 0
ALPHA = 0.05

# ==============================================================================
# Failure classification (pre-registered in STEP3_PLAN_PROMPT.md 3.2, with the step-3 review decisions).
# Precedence for a failed, non-harness-error episode:
#   crash -> no_start (motion test, only for starts >= START_ON_RING from the ring; tested before the approach test)
#   -> never approached (ring_dist_min > RING_APPROACH): runaway, no_progress, else never_other
#   -> approached: circling, left_or_overshot, parked_no_arm, extended_not_reached, else approached_other
# A failure that starts within START_ON_RING of the ring (M0-like) is never no_start; when its base does not move it
# carries the flag no_motion_near_ring (reported separately). A near-ring start that drives away is approached by
# construction (ring_dist_min <= ring_dist_start <= RING_APPROACH), so it is left_or_overshot, not runaway.
# ==============================================================================
CRASH_REASONS = ("tipped", "invalid_action", "policy_error")
RUN_ERROR_REASONS = ("harness_error",)   # an error of the run (harness), not of the policy: flagged, not classified
RING_APPROACH = 0.30        # m, the base "approached" the parking ring: ring_dist_min <= this
NO_START_PATH = 0.15        # m, no_start: base path shorter than this ... (above the zero policy's drift: its collapsing
                            # arm jolts the base up to 0.108 m on DEV; the plan's 0.05 m missed 25/280 of them)
NO_START_TURN_DEG = 10.0    # deg, ... and total turn below this ...
START_ON_RING = 0.15        # m, ... for episodes that start at least this far from the ring (closer: M0-like, flagged
                            # no_motion_near_ring instead)
RUNAWAY_EXTRA = 1.0         # m, runaway: final ring distance more than this above the start ring distance
NO_PROGRESS = 0.10          # m, no_progress: never approached and the ring distance never dropped this far below its
                            # start value (turns, but never drives towards the ring; added after the ego pilot)
CIRCLING_EXTRA_DEG = 360.0  # deg, circling: total turn minus the required turns above this; the required turns are
                            # align + orient, but the orient turn only counts once the base has parked (an M1
                            # base circling its goal never owes the 180 deg orient turn; expert validation)
LEFT_RING = 0.30            # m, left_or_overshot: final ring distance above this
PARKED_NO_ARM_TOL = 0.02    # m, parked_no_arm: |min_error - navdone_ee_err| below this (and the arm never extended) ...
EXTENDED_LO, EXTENDED_HI = 0.01, 0.05   # m, extended_not_reached: min_error in [lo, hi)
PARKED_NO_ARM_MIN_NAVDONE = EXTENDED_HI + PARKED_NO_ARM_TOL   # 0.07 m, ... and navdone_ee_err >= this: when the
                            # expert's hold posture at parking is already this close to the target ("short-reach"
                            # episode), "arm not extended" and "extended but missed" cannot be told apart by min_error;
                            # such failures fall through to extended_not_reached / approached_other
ARM_TOWARD_MARGIN = 0.02    # m, arm_extended_toward: arm extended and min_error < navdone_ee_err - this (the EE got
                            # closer than the parking hold posture: reaching, not a collapsing arm)
NAV_LAG_MAX = 0.2           # s, the plan's bound of (expert arm_extend_time - recorded expert nav_done time); it cannot
                            # hold (0.10 m at the 0.25 m/s Cartesian speed limit takes >= 0.4 s): reported, not checked
BASE_NAV_TIMEOUT = ad.BASE_NAV_TIMEOUT   # s (25), the expert FSM's navigation timeout (robotnik_arm_task)
# Classifier validation (iii): failure predicates expected to fire on expert successes, {(set, episode): predicate}.
# Genuine circling before a late success: the expert circled until the 25 s navigation timeout, parked after it and
# still reached (HARD 19: turn excess 397 deg, HARD 158: 596 deg, CONFIRM 62: 472 deg, CONFIRM 69: 586 deg).
# A success without driving: DEV 11 starts close enough for the arm to reach without the base moving (t_reach 1.57 s).
EXPECTED_SUCCESS_FIRINGS = {("hard", 19): "circling", ("hard", 158): "circling", ("confirm", 62): "circling",
                            ("confirm", 69): "circling", ("dev", 11): "no_start"}

NEVER_CLASSES = ("no_start", "runaway", "no_progress", "never_other")
APPROACHED_CLASSES = ("circling", "left_or_overshot", "parked_no_arm", "extended_not_reached", "approached_other")
FAILURE_CLASSES = ("crash", *NEVER_CLASSES, *APPROACHED_CLASSES)   # = the precedence (runaway / never_other only
                                                                   # when never approached)
CLASSES = ("success", *FAILURE_CLASSES)
# Merged classes (figures; comparison with teammate Rodrigo's 4 classes), defined over ALL episodes:
#   arrive_stay    = success and held_10s
#   arrive_leave   = success and not held_10s
#   base_ok_ee_not = failure in parked_no_arm, extended_not_reached, approached_other
#   base_not       = failure in no_start, runaway, no_progress, never_other, circling, left_or_overshot
#   crash          = crash
# Runs without the v1.1 column held_10s (protocol v1) give "arrive" (hold unknown) for a success.
MERGED_OF = {
    "crash": "crash",
    **{c: "base_not" for c in (*NEVER_CLASSES, "circling", "left_or_overshot")},
    **{c: "base_ok_ee_not" for c in ("parked_no_arm", "extended_not_reached", "approached_other")},
}
MERGED_CLASSES = ("arrive_stay", "arrive_leave", "base_ok_ee_not", "base_not", "crash")
MERGED_WORST_FIRST = ("crash", "base_not", "base_ok_ee_not", "arrive_leave", "arrive", "arrive_stay")
MERGED_DOC = ("merged classes: arrive_stay = success and held_10s; arrive_leave = success and not held_10s; "
              "base_ok_ee_not = parked_no_arm + extended_not_reached + approached_other; "
              "base_not = no_start + runaway + no_progress + never_other + circling + left_or_overshot; crash = crash")
# Columns the failure classifier needs (protocol v1.1); a run without them is not classified
CLASSIFIER_COLUMNS = ("ring_dist_start", "ring_dist_min", "ring_dist_final", "arm_extend_time")

# Per-episode binary metrics (any of them can be compared between arms); the column a metric needs.
# arm_extended_toward = arm_extended and min_error < navdone_ee_err - ARM_TOWARD_MARGIN (needs the episode reference,
# joined by prepare): it separates reaching from a collapsing arm, which arm_extended alone counts as extended
METRIC_COLUMNS = {"success": "success", "reach_2cm": "min_error", "reach_5cm": "min_error", "parked": "parked_time",
                  "arm_extended": "arm_extend_time", "arm_extended_toward": "arm_extend_time", "held_10s": "held_10s",
                  "stable_5cm": "stable_5cm_time"}
SECONDARY_METRICS = ("reach_2cm", "reach_5cm", "parked", "arm_extended", "arm_extended_toward", "held_10s",
                     "stable_5cm")
FLOOR_SUCCESS = 0.05        # floor rule: both arms' seed-mean success below this -> decide on the secondary metrics

# d0 strata (analyze_dataset.section_modes); the boundary band overlaps them and is descriptive only
STRATA = (("M0", lambda d: np.abs(d) < 0.08), ("M1", lambda d: d <= -0.08),
          ("M2 0.08-0.5", lambda d: (d >= 0.08) & (d < 0.5)), ("M2 0.5-1.0", lambda d: (d >= 0.5) & (d < 1.0)),
          ("M2 >= 1.0", lambda d: d >= 1.0))
BOUNDARY = ("boundary 0.08-0.15", lambda d: (np.abs(d) >= 0.08) & (np.abs(d) < 0.15))
MIN_STRATUM_N = 30          # strata with fewer episodes are descriptive only (report.txt, step 2): the Holm family and
                            # adoption condition (3) use the strata with n >= this in the analysed set (DEV: M1 and
                            # the three M2 strata; M0 and the boundary band are shown without Holm)

STRING_COLUMNS = {"stop_reason", "error_message", "expert_status_recorded", "hold10_stop"}


# ==============================================================================
# Loading
# ==============================================================================
def check_set_allowed(set_name):
    if set_name in SEALED_SETS:
        raise SystemExit(f"data/{set_name} is sealed: it is never opened or analysed in step 3")


def _parse_value(text, column):
    if column in STRING_COLUMNS:
        return text
    if text == "":
        return np.nan
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return float(text)
    except ValueError:
        return text


def read_rows(path):
    """rollouts.csv (or any csv written by csv.DictWriter) -> (rows with typed values, header)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = []
        for raw in reader:
            row = {key: _parse_value(value if value is not None else "", key) for key, value in raw.items()}
            if "episode" in row:
                row["episode"] = int(row["episode"])
            rows.append(row)
    return rows, header


_NAME_PATTERNS = [
    (re.compile(r"^expert_v11_(?P<set>[a-z]+)$"), {"kind": "expert", "model": "expert", "protocol": "v1.1"}),
    (re.compile(r"^zero_v11_(?P<set>[a-z]+)$"), {"kind": "zero", "model": "zero", "protocol": "v1.1"}),
    (re.compile(r"^B-EE_(?P<set>[a-z]+)$"), {"kind": "hybrid", "model": "expert", "cell": "EE"}),
    (re.compile(r"^B-(?P<cell>EM|ME)_(?P<model>[a-z][a-z0-9-]*)_s(?P<seed>\d+)_(?P<set>[a-z]+)$"), {"kind": "hybrid"}),
    (re.compile(r"^expert_(?P<set>[a-z]+)$"), {"kind": "expert", "model": "expert"}),
    (re.compile(r"^zero_(?P<set>[a-z]+)$"), {"kind": "zero", "model": "zero"}),
    (re.compile(r"^(?P<model>[a-z][a-z0-9-]*)_s(?P<seed>\d+)_(?P<set>[a-z]+)$"), {"kind": "model"}),
    (re.compile(r"^(?P<model>[a-z][a-z0-9-]*-ols)_(?P<set>[a-z]+)$"), {"kind": "model"}),
]


def parse_run_name(name):
    """Run folder name -> {kind, model, seed, cell, set} following the shared naming conventions; kind 'other'
    (model = the folder name) when no convention matches."""
    for pattern, fixed in _NAME_PATTERNS:
        m = pattern.match(name)
        if m:
            out = {"kind": None, "model": None, "seed": None, "cell": None, "set": None, "protocol": None}
            out.update(fixed)
            out.update({k: v for k, v in m.groupdict().items() if v is not None})
            if out["seed"] is not None:
                out["seed"] = int(out["seed"])
            if out["kind"] == "model":
                out["cell"] = "MM"
            return out
    return {"kind": "other", "model": name, "seed": None, "cell": None, "set": None, "protocol": None}


class Run:
    """One run folder: typed rows (sorted by episode), header, config.json, metadata."""

    def __init__(self, path, rows, header, config, meta):
        self.path, self.rows, self.header, self.config, self.meta = path, rows, header, config, meta
        self.by_episode = {row["episode"]: row for row in rows}

    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.path))

    def has(self, column):
        return column in self.header

    def __repr__(self):
        return f"Run({self.name}, {len(self.rows)} episodes, {self.meta.get('protocol')})"


def load_run(run_dir):
    """Run folder -> Run. Metadata: policy, policy_arg, eval_set, set (eval set name), protocol, no_early_stop,
    n_expected (the full size of the evaluation set, config.json eval_set_config.n_episodes; None if unknown), and the
    model / seed / cell parsed from the folder name; warnings for inconsistencies (stored in meta['warnings'])."""
    rows, header = read_rows(os.path.join(run_dir, "rollouts.csv"))
    rows.sort(key=lambda row: row["episode"])
    config = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    name = os.path.basename(os.path.normpath(run_dir))
    meta = parse_run_name(name)
    warns = []
    args = config.get("args", {})
    eval_set = args.get("eval_set")
    set_from_config = os.path.basename(os.path.normpath(eval_set)) if eval_set else None
    if set_from_config and meta["set"] and set_from_config != meta["set"]:
        warns.append(f"{name}: folder name says set {meta['set']}, config.json says {eval_set}")
    meta["set"] = set_from_config or meta["set"]
    meta["protocol"] = config.get("protocol", {}).get("PROTOCOL_VERSION") or meta["protocol"]
    n_expected = config.get("eval_set_config", {}).get("n_episodes")
    meta.update(policy=args.get("policy"), policy_arg=args.get("policy_arg"), eval_set=eval_set,
                no_early_stop=args.get("no_early_stop"), n_expected=int(n_expected) if n_expected is not None else None,
                warnings=warns)
    if meta["kind"] == "model" and meta["policy_arg"]:
        stem = os.path.splitext(os.path.basename(str(meta["policy_arg"])))[0]
        expected = f"{meta['model']}_s{meta['seed']}" if meta["seed"] is not None else meta["model"]
        if stem != expected:
            warns.append(f"{name}: policy_arg {meta['policy_arg']} does not match the folder name ({expected})")
    if meta["kind"] == "hybrid" and meta["policy_arg"]:
        parts = dict(item.split("=", 1) for item in str(meta["policy_arg"]).split(";") if "=" in item)
        stem = {k: os.path.splitext(os.path.basename(v))[0] if v != "expert" else "expert" for k, v in parts.items()}
        model_stem = f"{meta['model']}_s{meta['seed']}" if meta["seed"] is not None else meta["model"]
        expected = {"EE": ("expert", "expert"), "EM": ("expert", model_stem), "ME": (model_stem, "expert")}.get(meta["cell"])
        if expected and (stem.get("base"), stem.get("arm")) != expected:
            warns.append(f"{name}: policy_arg {meta['policy_arg']} does not match cell {meta['cell']} "
                         f"(expected base={expected[0]}, arm={expected[1]})")
    if meta["set"]:
        check_set_allowed(meta["set"])
    return Run(run_dir, rows, header, config, meta)


def discover_runs(results_dir=RESULTS_DIR, include_tmp=False):
    """All run folders (with a rollouts.csv) directly under results_dir, as Run objects; runs on a sealed set are
    skipped with a warning."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "rollouts.csv"))):
        run_dir = os.path.dirname(path)
        name = os.path.basename(run_dir)
        if name in ("step3", "data_analysis") or (name.startswith("tmp") and not include_tmp):
            continue
        try:
            runs.append(load_run(run_dir))
        except SystemExit as e:
            print(f"WARNING: skipped {run_dir}: {e}")
    return runs


def is_confirm_name(run_dir):
    """A run folder whose name says it is on the CONFIRM set (decided without opening the folder)."""
    return parse_run_name(os.path.basename(os.path.normpath(run_dir)))["set"] == "confirm"


def run_dir_for(results_dir, model, seed, set_name, cell=None):
    """Folder of a run by the naming conventions (model 'expert' -> expert_v11_<set>, 'zero' -> zero_v11_<set>)."""
    if cell == "EE":
        return os.path.join(results_dir, f"B-EE_{set_name}")
    if cell in ("EM", "ME"):
        return os.path.join(results_dir, f"B-{cell}_{model}_s{seed}_{set_name}")
    if model in ("expert", "zero"):
        return os.path.join(results_dir, f"{model}_v11_{set_name}")
    if seed is None:
        return os.path.join(results_dir, f"{model}_{set_name}")
    return os.path.join(results_dir, f"{model}_s{seed}_{set_name}")


def is_seedless(model):
    return model in ("expert", "zero") or model.endswith("-ols")


def parse_seeds(text):
    """'0-7', '0,2,5' or '3' -> sorted list of ints."""
    if text is None or text == "":
        return []
    seeds = set()
    for part in str(text).split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    return sorted(seeds)


# ==============================================================================
# Per-episode references from the recorded expert episodes
# ==============================================================================
REF_COLUMNS = ["episode", "noisy", "expert_status", "r", "theta_deg", "z", "standoff", "d0", "mode",
               "align_req_deg", "orient_req_deg", "navdone_step", "navdone_time", "navdone_time_meta",
               "navdone_ee_err", "init_x", "init_y", "init_yaw_deg", "target_x", "target_y",
               "cov_pct_world", "cov_nn_world", "cov_pct_rel", "cov_nn_rel"]
CHECKED_COLUMNS = ("align_req_deg", "orient_req_deg", "navdone_ee_err", "d0", "cov_pct_world", "cov_pct_rel")


def episode_reference(path):
    """Reference values of one recorded expert episode (same formulas as analyze_dataset.scan_episode):
    task (r, theta, d0, mode, required turns), the expert's nav_done (first step with nav_done, its time and the EE
    error ||target - ee_pos|| there = the hold-posture error at parking)."""
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        t = d["t"]
        nav_done = d["nav_done"].astype(bool)
        target = d["target"].astype(float)
        i = int(np.argmax(nav_done)) if nav_done.any() else None
        ee = d["ee_pos"][i].astype(float) if i is not None else None
    init = np.array(meta["init_base_pos"][:2], dtype=float)
    yaw0 = float(meta["init_base_yaw"])
    standoff = float(meta["standoff"])
    rel = target[:2] - init
    r = float(np.linalg.norm(rel))
    theta = float(ad.wrap(np.arctan2(rel[1], rel[0]) - yaw0))
    d0 = r - standoff
    align_req, orient_req = ad.required_turns(theta, d0)
    noisy = float(meta["noise_v"]) > 0 or float(meta["noise_w"]) > 0 or float(meta["noise_torque"]) > 0
    return {
        "noisy": noisy, "expert_status": meta["status"], "r": r, "theta_deg": float(np.rad2deg(theta)),
        "z": float(target[2]), "standoff": standoff, "d0": d0, "mode": ad.mode_of(d0),
        "align_req_deg": float(np.rad2deg(align_req)), "orient_req_deg": float(np.rad2deg(orient_req)),
        "navdone_step": i if i is not None else np.nan,
        "navdone_time": float(t[i]) if i is not None else np.nan,
        "navdone_time_meta": float(meta["nav_done_time"]) if meta.get("nav_done_time") is not None else np.nan,
        "navdone_ee_err": float(np.linalg.norm(ee - target)) if i is not None else np.nan,
        "init_x": float(init[0]), "init_y": float(init[1]), "init_yaw_deg": float(np.rad2deg(yaw0)),
        "target_x": float(target[0]), "target_y": float(target[1]),
    }


def _coverage_embed(rows):
    """Same embeddings as analyze_dataset.section_coverage: world 7-D and relative 3-D."""
    def col(key):
        return np.array([float(row[key]) for row in rows])
    yaw0 = np.deg2rad(col("init_yaw_deg"))
    world = np.column_stack([col("init_x"), col("init_y"), np.cos(yaw0), np.sin(yaw0), col("target_x"),
                             col("target_y"), col("z")])
    th = np.deg2rad(col("theta_deg"))
    rel = np.column_stack([col("r") * np.cos(th), col("r") * np.sin(th), col("z")])
    return {"world": world, "rel": rel}


def coverage_percentiles(rows, train_csv=os.path.join(ANALYSIS_DIR, "episodes_train.csv")):
    """Adds cov_pct_world / cov_nn_world / cov_pct_rel / cov_nn_rel to rows (in place): distance to the nearest kept
    train episode and its percentile among the train leave-one-out distances (analyze_dataset.section_coverage)."""
    from scipy.spatial import cKDTree
    train, _ = read_rows(train_csv)
    train = [row for row in train if row["keep"] is True]
    ref = _coverage_embed(train)
    emb = _coverage_embed(rows)
    for rep in ("world", "rel"):
        tree = cKDTree(ref[rep])
        loo = np.sort(tree.query(ref[rep], k=2)[0][:, 1])
        d = tree.query(emb[rep], k=1)[0]
        pct = np.searchsorted(loo, d, side="right") / len(loo)
        for row, value, dist in zip(rows, pct, d):
            row[f"cov_pct_{rep}"] = float(value)
            row[f"cov_nn_{rep}"] = float(dist)
    return len(train)


def build_reference(set_name, data_dir=DATA_DIR, out_dir=STEP3_DIR, train_csv=None):
    """Computes the reference of every episode of data/<set_name> from the recorded expert .npz and writes
    <out_dir>/cache_ref_<set_name>.csv."""
    check_set_allowed(set_name)
    if set_name == "confirm" and not ALLOW_CONFIRM:
        raise SystemExit("the references of data/confirm are built from its recorded episodes: pass --allow-confirm "
                         "(analyze_rollouts.ALLOW_CONFIRM = True) once CONFIRM is due")
    set_dir = os.path.join(data_dir, set_name)
    with open(os.path.join(set_dir, "index.csv")) as f:
        index = list(csv.DictReader(f))
    rows = []
    for item in index:
        ref = episode_reference(os.path.join(set_dir, item["file"]))
        ref["episode"] = int(item["episode"])
        rows.append(ref)
    rows.sort(key=lambda row: row["episode"])
    coverage_percentiles(rows, **({"train_csv": train_csv} if train_csv else {}))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cache_ref_{set_name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REF_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_reference(set_name, out_dir=STEP3_DIR, data_dir=DATA_DIR, rebuild=False):
    """Reference rows of a set by episode (dict episode -> row), from the cache or built from the recordings."""
    check_set_allowed(set_name)
    path = os.path.join(out_dir, f"cache_ref_{set_name}.csv")
    if rebuild or not os.path.exists(path):
        rows = build_reference(set_name, data_dir=data_dir, out_dir=out_dir)
    else:
        rows, _ = read_rows(path)
    for row in rows:
        row["episode"] = int(row["episode"])
        row["navdone_step"] = row["navdone_step"]
    return {row["episode"]: row for row in rows}


def cross_check_reference(set_name, ref, analysis_dir=ANALYSIS_DIR, data_dir=DATA_DIR):
    """Compares the reference with results/data_analysis/episodes_<set>.csv (where it exists) and the recorded
    nav_done_time of index.csv. Returns (lines, dict of max |difference| per column)."""
    lines, diffs = [], {}
    path = os.path.join(analysis_dir, f"episodes_{set_name}.csv")
    if os.path.exists(path):
        rows, _ = read_rows(path)
        common = [row for row in rows if row["episode"] in ref]
        mode_mismatch = sum(row["mode"] != ref[row["episode"]]["mode"] for row in common)
        for key in CHECKED_COLUMNS:
            a = np.array([float(row[key]) for row in common])
            b = np.array([float(ref[row["episode"]][key]) for row in common])
            both_nan = np.isnan(a) & np.isnan(b)
            one_nan = np.isnan(a) ^ np.isnan(b)
            d = np.abs(a - b)[~(both_nan | one_nan)]
            diffs[key] = float(d.max()) if d.size else 0.0
            if one_nan.any():
                diffs[key + "_nan_mismatch"] = int(one_nan.sum())
        diffs["mode_mismatch"] = mode_mismatch
        lines.append(f"{set_name}: {len(common)}/{len(ref)} episodes in {os.path.relpath(path, HERE)}; mode mismatches "
                     f"{mode_mismatch}; max |difference| " + ", ".join(
                         f"{k} {v:.3g}" for k, v in diffs.items() if k != "mode_mismatch"))
    else:
        lines.append(f"{set_name}: no {os.path.relpath(path, HERE)} (computed from the recordings only)")
    index_path = os.path.join(data_dir, set_name, "index.csv")
    if os.path.exists(index_path):
        index, _ = read_rows(index_path)
        a = np.array([float(row["nav_done_time"]) if row["nav_done_time"] == row["nav_done_time"] else np.nan
                      for row in index if row["episode"] in ref])
        b = np.array([ref[row["episode"]]["navdone_time"] for row in index if row["episode"] in ref])
        ok = np.isfinite(a) & np.isfinite(b)
        diffs["navdone_time_vs_index"] = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else np.nan
        lines.append(f"{set_name}: recorded nav_done time (first nav_done step of the .npz) vs index.csv nav_done_time: "
                     f"max |difference| {diffs['navdone_time_vs_index']:.3g} s over {ok.sum()} episodes "
                     f"({int((np.isfinite(a) != np.isfinite(b)).sum())} with only one defined)")
    return lines, diffs


# ==============================================================================
# Failure classification
# ==============================================================================
def _f(row, key):
    value = row.get(key, np.nan)
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _true(value):
    return value is True or (isinstance(value, (float, int, np.floating)) and not isinstance(value, bool)
                             and np.isfinite(value) and value == 1)


def required_turn_deg(row, ref):
    """In-place turns the episode needs (deg): the align turn, plus the orient turn once the base has parked."""
    orient = _f(ref, "orient_req_deg") if np.isfinite(_f(row, "parked_time")) else 0.0  # parked = parked_time set
    return _f(ref, "align_req_deg") + orient


def failure_predicates(row, ref):
    """Every failure predicate of the classification, evaluated on one row (also on successes, for the validation).
    ref: the episode's reference (navdone_ee_err, align_req_deg, orient_req_deg).
    no_motion: the base did not move (path < NO_START_PATH and turn < NO_START_TURN_DEG); no_start = no_motion for a
    start at least START_ON_RING from the ring. parked_no_arm needs a reference navdone_ee_err of at least
    PARKED_NO_ARM_MIN_NAVDONE (it never fires on short-reach episodes)."""
    ring_start, ring_min, ring_final = _f(row, "ring_dist_start"), _f(row, "ring_dist_min"), _f(row, "ring_dist_final")
    path, turn = _f(row, "base_path_length"), _f(row, "base_turn_total_deg")
    min_error, arm_ext = _f(row, "min_error"), _f(row, "arm_extend_time")
    required = required_turn_deg(row, ref)
    navdone_err = _f(ref, "navdone_ee_err")
    no_motion = path < NO_START_PATH and turn < NO_START_TURN_DEG
    return {
        "crash": row.get("stop_reason") in CRASH_REASONS,
        "never_approached": ring_min > RING_APPROACH,
        "no_motion": no_motion,
        "no_start": no_motion and ring_start >= START_ON_RING,
        "runaway": ring_final > ring_start + RUNAWAY_EXTRA,
        "no_progress": ring_min > RING_APPROACH and ring_min > ring_start - NO_PROGRESS,
        "circling": turn - required > CIRCLING_EXTRA_DEG,
        "left_or_overshot": ring_final > LEFT_RING,
        "parked_no_arm": bool(navdone_err >= PARKED_NO_ARM_MIN_NAVDONE and np.isnan(arm_ext)
                              and abs(min_error - navdone_err) < PARKED_NO_ARM_TOL),
        "extended_not_reached": EXTENDED_LO <= min_error < EXTENDED_HI,
    }


def classify_row(row, ref):
    """One class per episode: 'success', a failure class, or 'harness_error' (an error of the run, flagged
    separately). Precedence (see the block comment above the classification constants): crash; no_start (starts
    >= START_ON_RING from the ring, before the approach test); never approached: runaway, no_progress, never_other;
    approached:
    circling, left_or_overshot, parked_no_arm, extended_not_reached, approached_other."""
    if _true(row.get("success")):
        return "success"
    if row.get("stop_reason") in RUN_ERROR_REASONS:
        return "harness_error"
    p = failure_predicates(row, ref)
    if p["crash"]:
        return "crash"
    if p["no_start"]:
        return "no_start"
    if p["never_approached"]:
        for name in ("runaway", "no_progress"):
            if p[name]:
                return name
        return "never_other"
    for name in ("circling", "left_or_overshot", "parked_no_arm", "extended_not_reached"):
        if p[name]:
            return name
    return "approached_other"


def merged_class(cls, held_10s=None):
    """Merged class of an episode (MERGED_DOC). held_10s None = unknown (v1 run): a success is then 'arrive'."""
    if cls == "success":
        if held_10s is None:
            return "arrive"
        return "arrive_stay" if _true(held_10s) else "arrive_leave"
    return MERGED_OF.get(cls, cls)


def classify_run(run, ref):
    """Adds 'fclass' (failure class), 'mclass' (merged class) and flags to every row of the run (in place).
    A run without the v1.1 classifier columns gets 'unclassified' for its failures. Returns the run.
    Flags: start_on_ring (ring_dist_start < START_ON_RING, M0-like), start_near_ring (<= RING_APPROACH), no_motion,
    no_motion_near_ring (a failed, non-crash episode starting on the ring whose base does not move: reported
    separately, never no_start), short_reach (reference navdone_ee_err < PARKED_NO_ARM_MIN_NAVDONE: parked_no_arm
    cannot fire on the episode)."""
    classifiable = all(run.has(c) for c in CLASSIFIER_COLUMNS)
    has_held = run.has("held_10s")
    for row in run.rows:
        r = ref.get(row["episode"], {})
        if _true(row.get("success")):
            cls = "success"
        elif row.get("stop_reason") in RUN_ERROR_REASONS:
            cls = "harness_error"
        else:
            cls = classify_row(row, r) if classifiable else "unclassified"
        row["fclass"] = cls
        row["mclass"] = merged_class(cls, row.get("held_10s") if has_held else None)
        row["start_on_ring"] = _f(row, "ring_dist_start") < START_ON_RING
        row["start_near_ring"] = _f(row, "ring_dist_start") <= RING_APPROACH
        row["no_motion"] = _f(row, "base_path_length") < NO_START_PATH and _f(row, "base_turn_total_deg") < NO_START_TURN_DEG
        row["no_motion_near_ring"] = bool(cls in FAILURE_CLASSES and cls != "crash" and row["start_on_ring"]
                                          and row["no_motion"])
        row["short_reach"] = _f(r, "navdone_ee_err") < PARKED_NO_ARM_MIN_NAVDONE
    run.classified = classifiable
    return run


def class_counts(rows, key="fclass"):
    counts = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


# ==============================================================================
# Statistics
# ==============================================================================
def wilson(k, n, z=1.96):
    """95% Wilson score interval of k / n (analyze_dataset.wilson = evaluate_policy.wilson_interval), clipped to
    [0, 1] (the formula gives -1e-17 for k = 0)."""
    lo, hi = ad.wilson(k, n, z)
    return (lo, hi) if n == 0 else (max(0.0, lo), min(1.0, hi))


def _weights(rng, n, n_boot):
    """Resampling n items with replacement, n_boot times, as weights (counts / n), shape (n_boot, n)."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n


class Bootstrap:
    """Two-level bootstrap draws for one or more arms evaluated on the same episodes: per draw, the seeds of each
    arm are resampled with replacement independently, and ONE episode resample is shared by all arms.
    A draw's mean of an (S x E) matrix is the mean of the resampled seeds x resampled episodes."""

    def __init__(self, n_seeds, n_episodes, n_boot=N_BOOT, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        self.n_boot = n_boot
        self.ws = [_weights(rng, s, n_boot) for s in n_seeds]
        self.we = _weights(rng, n_episodes, n_boot)

    def means(self, Y, arm=0):
        Y = np.asarray(Y, dtype=float)
        return np.sum((self.ws[arm] @ Y) * self.we, axis=1)


def percentile_ci(draws, level=0.95):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(draws, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def boot_p(draws):
    """Two-sided bootstrap p-value of a difference: 2 min(P(d <= 0), P(d >= 0)), floored at 1 / n_boot."""
    draws = np.asarray(draws, dtype=float)
    n = len(draws)
    p = 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))
    return float(min(1.0, max(p, 1.0 / n)))


def boot2(Y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Two-level bootstrap of a seed x episode matrix Y (S x E): resample seeds, then episodes, with replacement;
    mean -> percentile 95% CI. Returns dict(mean, lo, hi, draws)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    b = Bootstrap([Y.shape[0]], Y.shape[1], n_boot, seed)
    draws = b.means(Y)
    lo, hi = percentile_ci(draws)
    return {"mean": float(Y.mean()), "lo": lo, "hi": hi, "draws": draws}


def paired_diff(A, B, n_boot=N_BOOT, seed=BOOT_SEED, boot=None):
    """Paired difference mean(B) - mean(A) of two arms on the same episodes, A (S_A x E) and B (S_B x E): per draw
    the seeds of each arm are resampled independently and one shared episode resample is used. Returns dict(diff,
    lo, hi, p, draws, mean_a, mean_b). boot: an existing Bootstrap with arms [S_A, S_B] (to reuse draws)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[1] != B.shape[1]:
        raise ValueError("paired arms need the same episodes")
    boot = boot or Bootstrap([A.shape[0], B.shape[0]], A.shape[1], n_boot, seed)
    draws = boot.means(B, 1) - boot.means(A, 0)
    lo, hi = percentile_ci(draws)
    return {"diff": float(B.mean() - A.mean()), "lo": lo, "hi": hi, "p": boot_p(draws), "draws": draws,
            "mean_a": float(A.mean()), "mean_b": float(B.mean())}


def seed_t_test(rate_a, rate_b, level=0.95):
    """Seed-level calibrated companion of the two-level bootstrap: t test and CI of mean(rate_b) - mean(rate_a) on
    the per-seed rates of two arms (each rate over the same episodes, so the shared episode effects cancel in the
    difference; the seeds are the replicates).
    Both arms with >= 2 seeds: Welch (unequal variances, Welch-Satterthwaite df). One arm with a single run (a
    deterministic arm: the expert / EE cell, an OLS model): one-sample t on the other arm's per-seed differences to it
    (S - 1 df). Fewer than 2 seeds in both arms: undefined (NaN). A zero standard error gives the degenerate CI
    [diff, diff] and p = 1 (diff = 0) or 0. Returns dict(diff, lo, hi, p, df, se, test)."""
    ra = np.asarray(rate_a, dtype=float).ravel()
    rb = np.asarray(rate_b, dtype=float).ravel()
    na, nb = ra.size, rb.size
    diff = float(rb.mean() - ra.mean()) if na and nb else np.nan
    out = {"diff": diff, "lo": np.nan, "hi": np.nan, "p": np.nan, "df": np.nan, "se": np.nan, "test": "undefined"}
    if na < 1 or nb < 1 or (na < 2 and nb < 2):
        return out
    va = ra.var(ddof=1) / na if na >= 2 else 0.0
    vb = rb.var(ddof=1) / nb if nb >= 2 else 0.0
    if na >= 2 and nb >= 2:
        out["test"] = "welch"
        den = (va ** 2 / (na - 1) if va > 0 else 0.0) + (vb ** 2 / (nb - 1) if vb > 0 else 0.0)
        df = (va + vb) ** 2 / den if den > 0 else np.nan
    else:
        out["test"] = "one-sample"
        df = float(max(na, nb) - 1)
    se = float(np.sqrt(va + vb))
    out.update(df=float(df), se=se)
    if se == 0:
        out.update(lo=diff, hi=diff, p=1.0 if diff == 0 else 0.0)
        return out
    q = float(stats.t.ppf(0.5 + level / 2, df))
    out.update(lo=diff - q * se, hi=diff + q * se, p=float(2 * stats.t.sf(abs(diff) / se, df)))
    return out


def seed_rates(Y):
    """Per-seed rates (row means) of a seed x episode matrix."""
    return np.atleast_2d(np.asarray(Y, dtype=float)).mean(axis=1)


def combined_p(p_boot, p_t):
    """The p used in the experiment-B Holm family: max(bootstrap p, t p); NaN when either is undefined (the
    contrast then stays out of the family, i.e. is not significant)."""
    if p_boot is None or p_t is None or not (np.isfinite(p_boot) and np.isfinite(p_t)):
        return np.nan
    return float(max(p_boot, p_t))


def mcnemar_exact(a, b):
    """Exact McNemar test of two binary outcome vectors on the same episodes (binomial test of the discordant
    episodes, p = 0.5, two-sided). Returns (n_ab = a fails & b succeeds, n_ba = a succeeds & b fails, p)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return n01, n10, p


def cochran_q(Y):
    """Cochran's Q test that the seeds (rows of Y, S x E binary) have the same success probability, the episodes
    being the blocks. Returns (Q, df, p); Q = 0, p = 1 when every episode has the same outcome for all seeds."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    k = Y.shape[0]
    if k < 2:
        return np.nan, 0, np.nan
    col, row, total = Y.sum(axis=1), Y.sum(axis=0), Y.sum()
    den = k * total - np.sum(row ** 2)
    if den == 0:
        return 0.0, k - 1, 1.0
    q = (k - 1) * (k * np.sum(col ** 2) - total ** 2) / den
    return float(q), k - 1, float(stats.chi2.sf(q, k - 1))


def seed_lottery(Y):
    """Normalised seed lottery of a seed x episode binary matrix: psi_raw = share of episodes on which the seeds
    disagree, psi_indep = 1 - prod p_s - prod (1 - p_s) (its value if seeds were independent of the episode),
    psi = psi_raw / psi_indep (0: the outcome is fixed by the episode, 1: pure seed lottery; NaN if psi_indep = 0)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=bool))
    if Y.shape[0] < 2:
        return {"psi_raw": np.nan, "psi_indep": np.nan, "psi": np.nan}
    p = Y.mean(axis=1)
    raw = float(np.mean(Y.any(axis=0) & ~Y.all(axis=0)))
    indep = float(1 - np.prod(p) - np.prod(1 - p))
    return {"psi_raw": raw, "psi_indep": indep, "psi": raw / indep if indep > 1e-12 else np.nan}


def holm(pvalues, alpha=ALPHA):
    """Holm step-down adjustment of a family of p-values (NaN entries are left out of the family).
    Returns (adjusted p-values, reject = adjusted p <= alpha, the textbook non-strict rule) in the input order."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj, np.where(np.isfinite(adj), adj <= alpha, False)


def metric_value(row, metric):
    """Per-episode binary metric (METRIC_COLUMNS) of a row as 0/1. arm_extended_toward needs the reference joined by
    prepare (ref_navdone_ee_err); without it the metric is 0."""
    if metric == "success":
        return float(_true(row.get("success")))
    if metric == "reach_2cm":
        return float(_f(row, "min_error") < 0.02)
    if metric == "reach_5cm":
        return float(_f(row, "min_error") < 0.05)
    if metric == "parked":
        return float(np.isfinite(_f(row, "parked_time")))
    if metric == "arm_extended":
        return float(np.isfinite(_f(row, "arm_extend_time")))
    if metric == "arm_extended_toward":
        return float(np.isfinite(_f(row, "arm_extend_time"))
                     and _f(row, "min_error") < _f(row, "ref_navdone_ee_err") - ARM_TOWARD_MARGIN)
    if metric == "held_10s":
        return float(_true(row.get("held_10s")))
    if metric == "stable_5cm":
        return float(np.isfinite(_f(row, "stable_5cm_time")))
    raise ValueError(f"unknown metric {metric!r}")


def common_episodes(runs):
    """Sorted episodes present in every run."""
    sets = [set(run.by_episode) for run in runs]
    return sorted(set.intersection(*sets)) if sets else []


def matrix(runs, episodes, fn):
    """(len(runs) x len(episodes)) float matrix of fn(row)."""
    return np.array([[fn(run.by_episode[e]) for e in episodes] for run in runs], dtype=float)


def metric_matrix(runs, episodes, metric):
    """Metric matrix, or None when a run lacks the metric's column."""
    if not all(run.has(METRIC_COLUMNS[metric]) for run in runs):
        return None
    return matrix(runs, episodes, lambda row: metric_value(row, metric))


def class_matrix(runs, episodes, key="fclass"):
    return np.array([[run.by_episode[e][key] for e in episodes] for run in runs], dtype=object)


def strata_masks(d0):
    d0 = np.asarray(d0, dtype=float)
    return [(name, fn(d0)) for name, fn in STRATA], (BOUNDARY[0], BOUNDARY[1](d0))


def logistic_slope(Y, x, n_boot=1000, seed=BOOT_SEED):
    """Logistic regression of success on x (one value per episode), fitted to the episode seed-mean success (a
    fractional response; the same point estimate as pooling the seed x episode rows). CI of the slope from the
    two-level bootstrap. Returns dict(slope, lo, hi) (NaN when the fit separates or y is constant)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    Y, x = Y[:, ok], x[ok]

    def fit(xv, y):
        if y.size == 0 or y.min() == y.max():
            return np.nan
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                beta, _, _ = ad.logistic_fit(xv[:, None], y, n_iter=50)
        except np.linalg.LinAlgError:
            return np.nan
        return float(beta[1]) if np.all(np.isfinite(beta)) and abs(beta[1]) < 50 else np.nan

    slope = fit(x, Y.mean(axis=0))
    rng = np.random.default_rng(seed)
    S, E = Y.shape
    draws = []
    for _ in range(n_boot if E else 0):
        si, ei = rng.integers(0, S, S), rng.integers(0, E, E)
        draws.append(fit(x[ei], Y[si][:, ei].mean(axis=0)))
    lo, hi = percentile_ci(draws)
    return {"slope": slope, "lo": lo, "hi": hi, "n_failed_draws": int(np.sum(~np.isfinite(draws)))}


# ==============================================================================
# Formatting
# ==============================================================================
def pct(x, digits=1):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}"


def pct_ci(point, lo, hi, digits=1):
    return f"{pct(point, digits)} [{pct(lo, digits)}, {pct(hi, digits)}]"


def pp_ci(point, lo, hi, digits=1):
    """Difference in percentage points with sign."""
    def s(x):
        return "nan" if not np.isfinite(x) else f"{100 * x:+.{digits}f}"
    return f"{s(point)} [{s(lo)}, {s(hi)}] pp"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "nan"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def _clean(obj):
    """JSON-safe copy (NaN -> None, numpy -> python, bootstrap draws dropped)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if k != "draws"}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_outputs(out_dir, stem, lines, result=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if result is not None:
        with open(os.path.join(out_dir, stem + ".json"), "w") as f:
            json.dump(_clean(result), f, indent=1)


# ==============================================================================
# Run sets with references and classes
# ==============================================================================
_REF_CACHE = {}


def reference_for(run, out_dir=STEP3_DIR, data_dir=DATA_DIR):
    set_name = run.meta["set"]
    if set_name is None:
        raise SystemExit(f"{run.path}: unknown evaluation set (no config.json and no set in the folder name)")
    key = (os.path.abspath(out_dir), set_name)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = load_reference(set_name, out_dir=out_dir, data_dir=data_dir)
    return _REF_CACHE[key]


def prepare(run, out_dir=STEP3_DIR, data_dir=DATA_DIR):
    """Joins the episode references to the rows (ref_* columns) and classifies the run."""
    ref = reference_for(run, out_dir, data_dir)
    missing = [row["episode"] for row in run.rows if row["episode"] not in ref]
    if missing:
        raise SystemExit(f"{run.path}: episodes {missing[:5]} missing from the reference of set {run.meta['set']}")
    for row in run.rows:
        r = ref[row["episode"]]
        for key in ("mode", "align_req_deg", "orient_req_deg", "navdone_ee_err", "navdone_time", "cov_pct_world",
                    "cov_pct_rel", "noisy"):
            row["ref_" + key] = r.get(key)
    classify_run(run, ref)
    return run


def load_prepared(path, out_dir=STEP3_DIR, data_dir=DATA_DIR):
    return prepare(load_run(path), out_dir, data_dir)


def harness_errors(runs):
    return {run.name: sum(row.get("stop_reason") in RUN_ERROR_REASONS for row in run.rows) for run in runs
            if any(row.get("stop_reason") in RUN_ERROR_REASONS for row in run.rows)}


# ==============================================================================
# Integrity of paired analyses (compare, split2x2, describe)
# ==============================================================================
def _ids(ids, k=10):
    ids = sorted(ids)
    return ", ".join(str(i) for i in ids[:k]) + (f", ... ({len(ids)} in all)" if len(ids) > k else "")


def is_hybrid(run):
    """A run of the hybrid policy (policies/hybrid.py, policy_arg 'base=...;arm=...')."""
    return ("base=" in str(run.meta.get("policy_arg") or "")
            or "hybrid" in os.path.basename(str(run.meta.get("policy") or "")))


def integrity_problems(runs):
    """Reasons why runs cannot be paired as they are (empty list = fine): different evaluation sets (config.json
    eval_set); an episode set that is not exactly the full set 0..n-1 of the evaluation set (config.json
    eval_set_config.n_episodes; also unknown size, duplicate rows, or episode sets that differ between the runs);
    harness_error episodes (a harness failure to re-run, not a policy failure); a hybrid run made without
    --no-early-stop (the stall rule ends an ME episode before the shadow FSM's 25 s timeout: biased ME cell)."""
    problems = []
    sets = {}
    for run in runs:
        sets.setdefault(run.meta.get("set"), []).append(run.name)
    if len(sets) > 1 or None in sets:
        problems.append("runs from different (or unknown) evaluation sets (config.json eval_set): "
                        + "; ".join(f"{s}: {', '.join(names)}" for s, names in sets.items()))
    episode_sets = {}
    for run in runs:
        eps = set(run.by_episode)
        episode_sets[run.name] = eps
        if len(run.rows) != len(eps):
            problems.append(f"{run.name}: {len(run.rows) - len(eps)} duplicate episode rows")
        n_exp = run.meta.get("n_expected")
        if n_exp is None:
            problems.append(f"{run.name}: full size of the evaluation set unknown (no config.json "
                            f"eval_set_config.n_episodes)")
        elif eps != set(range(n_exp)):
            missing, extra = set(range(n_exp)) - eps, eps - set(range(n_exp))
            problems.append(f"{run.name}: {len(eps)} episodes, the evaluation set {run.meta.get('set')} has {n_exp}"
                            + (f"; missing {_ids(missing)}" if missing else "")
                            + (f"; not in the set {_ids(extra)}" if extra else ""))
        errors = [row["episode"] for row in run.rows if row.get("stop_reason") in RUN_ERROR_REASONS]
        if errors:
            problems.append(f"{run.name}: harness_error in episodes {_ids(errors)} (a harness failure, not a policy "
                            f"failure: re-run these episodes)")
        if is_hybrid(run) and run.meta.get("no_early_stop") is not True:
            problems.append(f"{run.name}: hybrid run made without --no-early-stop (config.json args.no_early_stop = "
                            f"{run.meta.get('no_early_stop')}): the stall rule would bias the ME cell")
    if len({frozenset(e) for e in episode_sets.values()}) > 1:
        problems.append("the runs do not have the same episode set: " + "; ".join(
            f"{name} {len(e)}" for name, e in episode_sets.items()))
    return problems


def check_integrity(runs, what, allow_incomplete=None):
    """Refuses (SystemExit with every problem listed) when the runs fail integrity_problems, unless allow_incomplete
    (default: the module flag ALLOW_INCOMPLETE, CLI --allow-incomplete); then returns the problems as WARNING lines."""
    allow = ALLOW_INCOMPLETE if allow_incomplete is None else allow_incomplete
    problems = integrity_problems(runs)
    if problems and not allow:
        raise SystemExit(f"{what}: refused, the paired runs fail the integrity checks:\n  " + "\n  ".join(problems)
                         + "\n(re-run / complete the runs, or pass --allow-incomplete to analyse the common episodes "
                           "anyway, harness_error episodes left out of every arm)")
    return [f"  WARNING (--allow-incomplete): {p}" for p in problems]


def paired_episodes(runs):
    """Episodes present in every run, without those with a harness_error in any run. Returns (episodes, excluded
    harness_error episodes)."""
    common = common_episodes(runs)
    bad = {e for run in runs for e in common if run.by_episode[e].get("stop_reason") in RUN_ERROR_REASONS}
    return [e for e in common if e not in bad], sorted(bad)


def short_reach_line(rows, indent="  "):
    """Class-table line: failures on short-reach episodes (reference navdone_ee_err < PARKED_NO_ARM_MIN_NAVDONE),
    where parked_no_arm cannot fire (their arm-not-extended failures fall to extended_not_reached /
    approached_other)."""
    short = [row for row in rows if row.get("fclass") in FAILURE_CLASSES and row.get("short_reach")]
    n_ep = sum(bool(row.get("short_reach")) for row in rows)
    return (f"{indent}failures on short-reach episodes (expert hold posture at parking < {PARKED_NO_ARM_MIN_NAVDONE:.2f} m "
            f"from the target: parked_no_arm cannot fire): {len(short)} (of {n_ep} short-reach episodes)"
            + (": " + ", ".join(f"{c} {v}" for c, v in class_counts(short).items()) if short else ""))


def class_summary_lines(run):
    """Per-run class counts (failure classes and merged), with the M0-like / start-near-ring cases, the
    no_motion_near_ring flag and the failures on short-reach episodes."""
    n = len(run.rows)
    fc, mc = class_counts(run.rows, "fclass"), class_counts(run.rows, "mclass")
    k = fc.get("success", 0)
    lo, hi = wilson(k, n)
    lines = [f"{run.name} ({run.meta.get('protocol')}, {n} episodes): success {k}/{n} = {pct(k / max(n, 1))}% "
             f"[{pct(lo)}, {pct(hi)}]"]
    lines.append("  failure classes: " + ", ".join(f"{c} {fc[c]}" for c in (*FAILURE_CLASSES, "harness_error",
                                                                              "unclassified") if fc.get(c)))
    lines.append("  merged: " + ", ".join(f"{c} {mc[c]}" for c in (*MERGED_CLASSES, "arrive", "harness_error",
                                                                     "unclassified") if mc.get(c)))
    fails = [row for row in run.rows if row["fclass"] != "success"]
    on_ring = [row for row in fails if row["start_on_ring"]]
    if on_ring:
        lines.append(f"  failures starting within {START_ON_RING} m of the ring (M0-like, reported separately, never "
                     f"no_start): {len(on_ring)}: " + ", ".join(f"{c} {v}" for c, v in class_counts(on_ring).items())
                     + f"; of them no_motion_near_ring (base did not move) "
                       f"{sum(bool(row.get('no_motion_near_ring')) for row in on_ring)}")
    if getattr(run, "classified", False):
        lines.append(short_reach_line(run.rows))
    errors = fc.get("harness_error", 0)
    if errors:
        lines.append(f"  WARNING: {errors} harness_error episodes: an error of the run, not of the policy")
    return lines


# ==============================================================================
# Experiment A: two arms on the same episodes
# ==============================================================================
def arm_runs(results_dir, model, seeds, set_name, out_dir=STEP3_DIR, data_dir=DATA_DIR, explicit=None):
    """Prepared runs of one arm: explicit run folders, or <model>_s<seed>_<set> for the seeds (OLS / expert: one run).
    Returns (runs, missing folders)."""
    if explicit:
        paths = list(explicit)
    elif is_seedless(model):
        paths = [run_dir_for(results_dir, model, None, set_name)]
    else:
        paths = [run_dir_for(results_dir, model, s, set_name) for s in seeds]
    runs, missing = [], []
    for path in paths:
        if os.path.exists(os.path.join(path, "rollouts.csv")):
            runs.append(load_prepared(path, out_dir, data_dir))
        else:
            missing.append(path)
    return runs, missing


def strata_analysis(YA, YB, d0, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired B - A difference in each d0 stratum (two-level bootstrap within the stratum's episodes) and in the
    boundary band (descriptive). The Holm family is the strata with n >= MIN_STRATUM_N in the analysed set (the
    others and the boundary band are descriptive: p shown, p_holm NaN, in_holm False). Returns dict(strata, boundary,
    worst_hi_all, worst_hi_n30, n_holm)."""
    masks, (_, bmask) = strata_masks(d0)
    strata, pvals = [], []
    for k, (name, m) in enumerate(masks):
        idx = np.flatnonzero(m)
        if idx.size == 0:
            strata.append({"name": name, "n": 0, "p": np.nan, "in_holm": False})
            pvals.append(np.nan)
            continue
        ds = paired_diff(YA[:, idx], YB[:, idx], n_boot=n_boot, seed=seed + 1 + k)
        in_holm = bool(idx.size >= MIN_STRATUM_N)
        strata.append({"name": name, "n": int(idx.size), "in_holm": in_holm, **ds})
        pvals.append(ds["p"] if in_holm else np.nan)
    adj, rej = holm(pvals)
    for s_, a, r in zip(strata, adj, rej):
        s_["p_holm"], s_["reject"] = a, bool(r)
    bidx = np.flatnonzero(bmask)
    boundary = {"n": int(bidx.size), **paired_diff(YA[:, bidx], YB[:, bidx], n_boot=n_boot, seed=seed + 99)} \
        if bidx.size else None
    full = [s_ for s_ in strata if s_["n"] > 0]
    big = [s_ for s_ in full if s_["n"] >= MIN_STRATUM_N]
    w_all = min(full, key=lambda s_: s_["hi"]) if full else None
    w_big = min(big, key=lambda s_: s_["hi"]) if big else None
    return {"strata": strata, "boundary": boundary,
            "worst_hi_all": (w_all["name"], w_all["hi"]) if w_all else None,
            "worst_hi_n30": (w_big["name"], w_big["hi"]) if w_big else None, "n_holm": len(big)}


def _t_name(t):
    return {"welch": "Welch t (per-seed rates)", "one-sample": "one-sample t (per-seed differences to the single run)",
            "undefined": "t test undefined (< 2 seeds)"}[t["test"]]


def _df(t):
    return "nan" if not np.isfinite(t["df"]) else f"{t['df']:.1f}"


def _yes(flag):
    return "YES" if flag else "NO"


def expert_reference(expert_run, episodes, set_name):
    """The expert's rate of every metric on the given episodes (a single deterministic run: the reference / ceiling
    of the arms). Returns (dict metric -> rate or None, note line or None)."""
    if expert_run is None:
        return {}, None
    if expert_run.meta.get("set") != set_name:
        return {}, f"  expert reference {expert_run.name}: evaluation set {expert_run.meta.get('set')}, not {set_name}: not used"
    eps = [e for e in episodes if e in expert_run.by_episode]
    out = {}
    for metric in ("success", *SECONDARY_METRICS):
        M = metric_matrix([expert_run], eps, metric) if eps else None
        out[metric] = float(M.mean()) if M is not None else None
    note = None if len(eps) == len(episodes) else (f"  expert reference {expert_run.name}: on {len(eps)} of the "
                                                    f"{len(episodes)} episodes only")
    return out, note


def compare_arms(runs_a, runs_b, label_a="A", label_b="B", predicted_classes=(), n_boot=N_BOOT, seed=BOOT_SEED,
                 expert_run=None, allow_incomplete=None):
    """Experiment A report of arm B against arm A on their common episodes. Returns (lines, result dict).
    Differences are B - A; 'reduction' of a failure class = share in A - share in B. The runs must pass
    check_integrity (refused otherwise, unless allow_incomplete; then harness_error episodes are left out of every
    arm). Every difference is shown with the two-level bootstrap and the seed-level t test (seed_t_test).
    expert_run (optional, prepared, same set): the expert's rate of each metric on the same episodes, as the
    reference (ceiling) of the arms."""
    warn = check_integrity(runs_a + runs_b, f"experiment A ({label_b} vs {label_a})", allow_incomplete)
    episodes, excluded = paired_episodes(runs_a + runs_b)
    E, Sa, Sb = len(episodes), len(runs_a), len(runs_b)
    lines = [f"== Experiment A: {label_b} (B) vs {label_a} (A), {E} common episodes, seeds A {Sa}, B {Sb} ==",
             "Differences are B - A (seed-mean over training seeds; two-level bootstrap: seeds resampled within each "
             f"arm, one shared episode resample; {n_boot} draws, rng seed {seed}; percentile 95% CI). Calibrated "
             "companion: Welch t on the per-seed rates (one-sample t when an arm is a single run), 95% CI."]
    lines += warn
    res = {"label_a": label_a, "label_b": label_b, "n_episodes": E, "runs_a": [r.name for r in runs_a],
           "runs_b": [r.name for r in runs_b], "integrity_warnings": warn, "harness_error_excluded": excluded}
    if excluded:
        lines.append(f"  harness_error episodes left out of every arm: {_ids(excluded)}")
    for r in runs_a + runs_b:
        if len(r.rows) != E:
            lines.append(f"  note: {r.name} has {len(r.rows)} episodes, {E} are common to all runs")
    if E == 0 or Sa == 0 or Sb == 0:
        lines.append("  no common episodes / no runs: nothing to compare")
        return lines, res
    set_name = runs_a[0].meta.get("set")
    xref, xnote = expert_reference(expert_run, episodes, set_name)
    res["expert_reference"] = {"run": expert_run.name if expert_run is not None else None, "rates": xref}
    if xnote:
        lines.append(xnote)
    boot = Bootstrap([Sa, Sb], E, n_boot, seed)
    YA = matrix(runs_a, episodes, lambda row: metric_value(row, "success"))
    YB = matrix(runs_b, episodes, lambda row: metric_value(row, "success"))

    # per-seed success with Wilson
    lines.append("Per-seed success (Wilson 95%):")
    res["per_seed"] = {}
    for label, runs, Y in ((label_a, runs_a, YA), (label_b, runs_b, YB)):
        parts = []
        for run, y in zip(runs, Y):
            k = int(y.sum())
            lo, hi = wilson(k, E)
            parts.append(f"{run.meta.get('seed') if run.meta.get('seed') is not None else run.name}: "
                         f"{k}/{E} {pct(k / E)} [{pct(lo)}, {pct(hi)}]")
            res["per_seed"].setdefault(label, []).append({"run": run.name, "k": k, "n": E, "lo": lo, "hi": hi})
        lines.append(f"  {label}: " + "; ".join(parts))
    # seed means and the paired difference (bootstrap and the seed-level t test)
    ma, mb = boot.means(YA, 0), boot.means(YB, 1)
    res["seed_mean"] = {label_a: {"mean": YA.mean(), "ci": percentile_ci(ma)},
                        label_b: {"mean": YB.mean(), "ci": percentile_ci(mb)}}
    d = paired_diff(YA, YB, boot=boot)
    t = seed_t_test(seed_rates(YA), seed_rates(YB))
    res["diff"], res["welch"] = d, t
    lines.append(f"Seed-mean success: {label_a} {pct_ci(YA.mean(), *percentile_ci(ma))}%, "
                 f"{label_b} {pct_ci(YB.mean(), *percentile_ci(mb))}%"
                 + (f"; expert {pct(xref['success'])}% on the same episodes" if xref.get("success") is not None else ""))
    lines.append(f"Paired difference B - A: bootstrap {pp_ci(d['diff'], d['lo'], d['hi'])}, p {fmt_p(d['p'])}; "
                 f"{_t_name(t)} {pp_ci(t['diff'], t['lo'], t['hi'])}, df {_df(t)}, p {fmt_p(t['p'])}")
    floor = YA.mean() < FLOOR_SUCCESS and YB.mean() < FLOOR_SUCCESS
    res["floor_rule_active"] = bool(floor)
    lines.append(f"Floor rule (both seed-mean successes < {pct(FLOOR_SUCCESS, 0)}%): "
                 + ("ACTIVE: the decision moves to the secondary metrics" if floor else "not active"))

    # per seed pair: McNemar and the difference table
    lines.append("Seed x seed (all pairs; seeds of different arms are not paired): success difference B - A in pp, "
                 "exact McNemar p (descriptive) [n A fails & B succeeds / n A succeeds & B fails]:")
    res["seed_pairs"] = []
    header = "  A \\ B " + "".join(f"{str(r.meta.get('seed') if r.meta.get('seed') is not None else r.name):>22}"
                                    for r in runs_b)
    lines.append(header)
    for ra, ya in zip(runs_a, YA):
        cells = []
        for rb, yb in zip(runs_b, YB):
            n01, n10, p = mcnemar_exact(ya, yb)
            diff = yb.mean() - ya.mean()
            cells.append(f"{100 * diff:+6.1f} p {fmt_p(p):>7} [{n01}/{n10}]")
            res["seed_pairs"].append({"a": ra.name, "b": rb.name, "diff": diff, "n01": n01, "n10": n10, "p": p})
        lines.append(f"  {str(ra.meta.get('seed') if ra.meta.get('seed') is not None else ra.name):<6}"
                     + "".join(f"{c:>22}" for c in cells))
    all_pairs = np.array([sp["diff"] for sp in res["seed_pairs"]])
    lines.append(f"  all {len(all_pairs)} seed-pair differences: min {100 * all_pairs.min():+.1f}, median "
                 f"{100 * np.median(all_pairs):+.1f}, max {100 * all_pairs.max():+.1f} pp; "
                 f"{int(np.sum(all_pairs > 0))} > 0, {int(np.sum(all_pairs < 0))} < 0")

    # failure classes as seed-mean shares ("for each episode the share of seeds in each class, averaged over episodes")
    classified = all(getattr(r, "classified", False) for r in runs_a + runs_b)
    res["classes"] = {}
    if classified:
        CA, CB = class_matrix(runs_a, episodes), class_matrix(runs_b, episodes)
        lines.append("Failure classes, seed-mean share of the episodes (A, B, B - A with CI):")
        reductions = {}
        for c in (*CLASSES, "harness_error"):
            IA, IB = (CA == c).astype(float), (CB == c).astype(float)
            if IA.sum() + IB.sum() == 0:
                continue
            dc = paired_diff(IA, IB, boot=boot)
            reductions[c] = -dc["draws"]
            res["classes"][c] = {"a": IA.mean(), "b": IB.mean(), "diff": dc["diff"], "lo": dc["lo"], "hi": dc["hi"]}
            lines.append(f"  {c:<22} A {pct(IA.mean()):>5}%  B {pct(IB.mean()):>5}%  B - A {pp_ci(dc['diff'], dc['lo'], dc['hi'])}")
        # flags of the class table: near-ring failures whose base did not move, failures on short-reach episodes
        NA = matrix(runs_a, episodes, lambda row: float(bool(row.get("no_motion_near_ring"))))
        NB = matrix(runs_b, episodes, lambda row: float(bool(row.get("no_motion_near_ring"))))
        short = np.array([bool(runs_a[0].by_episode[e].get("short_reach")) for e in episodes])
        SA, SB = np.isin(CA, FAILURE_CLASSES) & short[None, :], np.isin(CB, FAILURE_CLASSES) & short[None, :]
        by_a = {c: int(np.sum(SA & (CA == c))) for c in FAILURE_CLASSES if np.sum(SA & (CA == c))}
        by_b = {c: int(np.sum(SB & (CB == c))) for c in FAILURE_CLASSES if np.sum(SB & (CB == c))}
        res["flags"] = {"no_motion_near_ring": {"a": NA.mean(), "b": NB.mean()},
                        "short_reach_failures": {"n_short_episodes": int(short.sum()), "a": SA.mean(), "b": SB.mean(),
                                                 "classes_a": by_a, "classes_b": by_b}}
        lines.append(f"  flag no_motion_near_ring (failure starting within {START_ON_RING} m of the ring, base did not "
                     f"move; never no_start): A {pct(NA.mean())}%  B {pct(NB.mean())}%")
        lines.append(f"  failures on short-reach episodes ({int(short.sum())} of {E} episodes with the expert's hold "
                     f"posture at parking < {PARKED_NO_ARM_MIN_NAVDONE:.2f} m from the target: parked_no_arm cannot "
                     f"fire): A {pct(SA.mean())}%  B {pct(SB.mean())}% of the episodes; seed x episode counts A "
                     f"{by_a or '{}'}, B {by_b or '{}'}")
        MA, MB = class_matrix(runs_a, episodes, "mclass"), class_matrix(runs_b, episodes, "mclass")
        lines.append("Merged classes, seed-mean share (A, B, B - A): " + "; ".join(
            f"{c} {pct((MA == c).mean())} / {pct((MB == c).mean())} / {100 * ((MB == c).mean() - (MA == c).mean()):+.1f}"
            for c in (*MERGED_CLASSES, "arrive") if (MA == c).any() or (MB == c).any()))
        res["merged"] = {c: {"a": (MA == c).mean(), "b": (MB == c).mean()} for c in (*MERGED_CLASSES, "arrive")}
        # share of the net difference from the predicted classes
        if predicted_classes:
            unknown = [c for c in predicted_classes if c not in FAILURE_CLASSES]
            if unknown:
                raise SystemExit(f"unknown failure classes {unknown}; expected some of {FAILURE_CLASSES}")
            red_point = sum(res["classes"].get(c, {"a": 0.0, "b": 0.0})["a"] - res["classes"].get(c, {"a": 0.0, "b": 0.0})["b"]
                            for c in predicted_classes)
            red_draws = sum((reductions[c] for c in predicted_classes if c in reductions), np.zeros(n_boot))
            with np.errstate(divide="ignore", invalid="ignore"):
                share_draws = red_draws / d["draws"]
                share = red_point / d["diff"] if d["diff"] != 0 else np.nan
            lo, hi = percentile_ci(share_draws)
            res["predicted_share"] = {"classes": list(predicted_classes), "reduction": red_point, "share": share,
                                      "lo": lo, "hi": hi}
            lines.append(f"Share of the net difference from {'+'.join(predicted_classes)}: reduction of their share "
                         f"{100 * red_point:+.1f} pp / net difference {100 * d['diff']:+.1f} pp = "
                         f"{pct(share)}% [bootstrap {pct(lo)}, {pct(hi)}] (unstable when the net difference is near 0)")
    else:
        lines.append("Failure classes: not available (a run lacks the v1.1 classifier columns)")

    # d0 strata
    d0 = np.array([runs_a[0].by_episode[e]["d0"] for e in episodes], dtype=float)
    st = strata_analysis(YA, YB, d0, n_boot, seed)
    lines.append(f"d0 strata (Holm over the {st['n_holm']} strata with n >= {MIN_STRATUM_N}, reject at p_adj <= "
                 f"{ALPHA}; the strata with fewer episodes are descriptive, without Holm):")
    res["strata"], res["boundary"] = st["strata"], st["boundary"]
    for s_ in st["strata"]:
        if s_["n"] == 0:
            lines.append(f"  {s_['name']:<14} n   0")
            continue
        lines.append(f"  {s_['name']:<14} n {s_['n']:>3}{'              ' if s_['in_holm'] else ' (descriptive)'}"
                     f" A {pct(s_['mean_a']):>5}%  B {pct(s_['mean_b']):>5}%  B - A {pp_ci(s_['diff'], s_['lo'], s_['hi'])}, "
                     f"p {fmt_p(s_['p'])}, " + (f"Holm {fmt_p(s_['p_holm'])}{' (significant)' if s_['reject'] else ''}"
                                                if s_["in_holm"] else "no Holm"))
    if st["boundary"]:
        ds = st["boundary"]
        lines.append(f"  {BOUNDARY[0]} (descriptive, overlaps the strata): n {ds['n']}, A {pct(ds['mean_a'])}%, "
                     f"B {pct(ds['mean_b'])}%, B - A {pp_ci(ds['diff'], ds['lo'], ds['hi'])}")

    # secondary metrics
    lines.append("Secondary metrics (seed-mean share; B - A paired: two-level bootstrap and Welch t; descriptive unless "
                 "the floor rule is active; expert = the expert run's rate on the same episodes, the reference):")
    res["secondary"] = {}
    for metric in SECONDARY_METRICS:
        MA_, MB_ = metric_matrix(runs_a, episodes, metric), metric_matrix(runs_b, episodes, metric)
        if MA_ is None or MB_ is None:
            lines.append(f"  {metric:<19} not available (column {METRIC_COLUMNS[metric]} missing in a run)")
            continue
        dm = paired_diff(MA_, MB_, boot=boot)
        tm = seed_t_test(seed_rates(MA_), seed_rates(MB_))
        sm = strata_analysis(MA_, MB_, d0, n_boot, seed)
        res["secondary"][metric] = {**dm, "ci_a": percentile_ci(boot.means(MA_, 0)),
                                    "ci_b": percentile_ci(boot.means(MB_, 1)), "per_seed_a": MA_.mean(axis=1),
                                    "per_seed_b": MB_.mean(axis=1), "welch": tm, "expert": xref.get(metric),
                                    "strata": sm["strata"], "worst_stratum_ci_upper_all": sm["worst_hi_all"],
                                    "worst_stratum_ci_upper_n30": sm["worst_hi_n30"]}
        lines.append(f"  {metric:<19} A {pct_ci(dm['mean_a'], *res['secondary'][metric]['ci_a'])}%  "
                     f"B {pct_ci(dm['mean_b'], *res['secondary'][metric]['ci_b'])}%  "
                     f"B - A {pp_ci(dm['diff'], dm['lo'], dm['hi'])}, p {fmt_p(dm['p'])}; "
                     f"Welch {pp_ci(tm['diff'], tm['lo'], tm['hi'])}, p {fmt_p(tm['p'])}")
        wa, wb = sm["worst_hi_all"], sm["worst_hi_n30"]
        worst_parts = ([f"n >= {MIN_STRATUM_N} {wb[0]} {100 * wb[1]:+.1f} pp"] if wb else []) + (
            [f"all {wa[0]} {100 * wa[1]:+.1f} pp"] if wa else [])
        lines.append(f"  {'':<19} " + (f"expert {pct(xref[metric])}%; " if xref.get(metric) is not None else "")
                     + "per seed A " + "/".join(pct(x, 0) for x in MA_.mean(axis=1)) + ", B "
                     + "/".join(pct(x, 0) for x in MB_.mean(axis=1))
                     + ("; worst stratum CI upper bound: " + ", ".join(worst_parts) if worst_parts else ""))
    if res["secondary"]:
        names = list(res["secondary"])
        adj2, _ = holm([res["secondary"][m]["p"] for m in names])
        adj3, _ = holm([combined_p(res["secondary"][m]["p"], res["secondary"][m]["welch"]["p"]) for m in names])
        lines.append("  Holm over the available secondary metrics (bootstrap p | max(bootstrap p, Welch p)): " + ", ".join(
            f"{m} {fmt_p(a)} | {fmt_p(b)}" for m, a, b in zip(names, adj2, adj3)))
        for m, a, b in zip(names, adj2, adj3):
            res["secondary"][m]["p_holm"], res["secondary"][m]["p_holm_combined"] = a, b

    # adoption-rule inputs (STEP3_PLAN_PROMPT.md 4: conditions 1-3; 4-5 need HARD / CONFIRM runs)
    full = [s_ for s_ in res["strata"] if s_["n"] > 0]
    big = [s_ for s_ in full if s_["n"] >= MIN_STRATUM_N]

    def worst(items, key, fn):
        return fn(items, key=lambda s_: s_[key]) if items else None
    w_all, w_big = worst(full, "hi", min), worst(big, "hi", min)
    w_all_rev, w_big_rev = worst(full, "lo", max), worst(big, "lo", max)
    c1_boot, c1_t = bool(d["lo"] > 0), bool(np.isfinite(t["lo"]) and t["lo"] > 0)
    r1_boot, r1_t = bool(d["hi"] < 0), bool(np.isfinite(t["hi"]) and t["hi"] < 0)
    res["adoption_inputs"] = {
        "ci_lower_b_minus_a": d["lo"], "ci_upper_b_minus_a": d["hi"],
        "welch_ci_lower_b_minus_a": t["lo"], "welch_ci_upper_b_minus_a": t["hi"], "welch_p": t["p"],
        "welch_test": t["test"],
        "cond1_bootstrap": c1_boot, "cond1_welch": c1_t, "cond1": c1_boot and c1_t,
        "reverse_cond1_bootstrap": r1_boot, "reverse_cond1_welch": r1_t, "reverse_cond1": r1_boot and r1_t,
        "share_from_predicted_classes": res.get("predicted_share", {}).get("share"),
        "cond3_strata": f"n >= {MIN_STRATUM_N}",
        "worst_stratum_ci_upper_n30": (w_big["name"], w_big["hi"]) if w_big else None,
        "worst_stratum_ci_upper_all": (w_all["name"], w_all["hi"]) if w_all else None,
        "reverse_worst_stratum_ci_lower_n30": (w_big_rev["name"], w_big_rev["lo"]) if w_big_rev else None,
        "reverse_worst_stratum_ci_lower_all": (w_all_rev["name"], w_all_rev["lo"]) if w_all_rev else None,
        "floor_rule_active": bool(floor),
    }
    lines.append("Adoption-rule inputs (B adopted only if (1) CI lower bound > 0, (2) share from predicted classes "
                 ">= 50%, (3) every stratum CI upper bound >= -10 pp, (4) HARD same direction, (5) CONFIRM):")
    lines.append(f"  (1) CI lower bound of B - A: bootstrap {100 * d['lo']:+.1f} pp, {_t_name(t)} "
                 f"{100 * t['lo']:+.1f} pp (upper: bootstrap {100 * d['hi']:+.1f}, t {100 * t['hi']:+.1f} pp) -> "
                 f"bootstrap > 0 {_yes(c1_boot)}, t > 0 {_yes(c1_t)}: condition (1) (both > 0) "
                 f"{'MET' if c1_boot and c1_t else 'NOT MET'}")
    ps = res.get("predicted_share")
    lines.append(f"  (2) share from predicted classes: " + (f"{pct(ps['share'])}% ({'+'.join(ps['classes'])})" if ps
                                                             else "not computed (no --predicted-classes or no classes)"))
    if w_all:
        lines.append(f"  (3) worst stratum CI upper bound (strata with n >= {MIN_STRATUM_N}, used): "
                     + (f"{w_big['name']} {100 * w_big['hi']:+.1f} pp (needs >= -10 pp)" if w_big
                        else f"no stratum with n >= {MIN_STRATUM_N}")
                     + f"; all strata (shown): {w_all['name']} {100 * w_all['hi']:+.1f} pp")
        lines.append(f"  reverse direction (A better than B): B - A CI upper bound: bootstrap {100 * d['hi']:+.1f} pp, "
                     f"t {100 * t['hi']:+.1f} pp (both need < 0: {'MET' if r1_boot and r1_t else 'NOT MET'}); highest "
                     f"stratum B - A CI lower bound (needs <= +10 pp): "
                     + (f"n >= {MIN_STRATUM_N} (used) {w_big_rev['name']} {100 * w_big_rev['lo']:+.1f} pp, " if w_big_rev
                        else f"no stratum with n >= {MIN_STRATUM_N}, ")
                     + f"all {w_all_rev['name']} {100 * w_all_rev['lo']:+.1f} pp")
    return lines, res


# ==============================================================================
# Experiment B: base / arm 2x2 split
# ==============================================================================
HYBRID_PREFIX = "HYBRID_EVENT "


def parse_hybrid_log(source):
    """HYBRID_EVENT lines of one episode log (a path, or an iterable of lines) -> dict with 'config' (first config
    event or None), 'phases' [(t, phase)], 'nav_done' ({t, trigger} of the first nav_done event or None),
    'arm_trigger' ({t, source} or None), 'events' (all, in order), 'n_bad' (unparsable event lines)."""
    if isinstance(source, str):
        if not os.path.exists(source):
            return None
        with open(source, errors="replace") as f:
            lines = f.read().splitlines()
    else:
        lines = list(source)
    out = {"config": None, "phases": [], "nav_done": None, "arm_trigger": None, "events": [], "n_bad": 0}
    for line in lines:
        k = line.find(HYBRID_PREFIX)
        if k < 0:
            continue
        try:
            ev = json.loads(line[k + len(HYBRID_PREFIX):].strip())
        except json.JSONDecodeError:
            out["n_bad"] += 1
            continue
        if not isinstance(ev, dict) or "event" not in ev:
            out["n_bad"] += 1
            continue
        out["events"].append(ev)
        kind = ev["event"]
        if kind == "config" and out["config"] is None:
            out["config"] = ev
        elif kind == "phase":
            out["phases"].append((float(ev["t"]), ev["phase"]))
        elif kind == "nav_done" and out["nav_done"] is None:
            out["nav_done"] = {"t": float(ev["t"]), "trigger": ev.get("trigger")}
        elif kind == "arm_trigger" and out["arm_trigger"] is None:
            out["arm_trigger"] = {"t": float(ev["t"]), "source": ev.get("source")}
    return out


def run_logs(run, episodes=None):
    """Parsed hybrid logs of a run by episode (None for a missing log)."""
    episodes = episodes if episodes is not None else [row["episode"] for row in run.rows]
    return {e: parse_hybrid_log(os.path.join(run.path, "logs", f"ep_{e:04d}.log")) for e in episodes}


def me_triggers(me_runs, episodes):
    """Shadow-FSM nav_done trigger of every ME seed x episode from the hybrid logs (S x E object matrix): 'threshold',
    'timeout', 'never' (no nav_done event) or 'no_log'."""
    out = []
    for run in me_runs:
        logs = run_logs(run, episodes)
        row = []
        for e in episodes:
            log = logs[e]
            key = "no_log" if log is None else ("never" if log["nav_done"] is None else log["nav_done"]["trigger"])
            row.append(key if key in ("threshold", "timeout", "never", "no_log") else "never")
        out.append(row)
    return np.array(out, dtype=object).reshape(len(me_runs), len(episodes))


def attribution(fail, me_f, em_f, n_boot=N_BOOT, seed=BOOT_SEED):
    """Attribution of the MM failures (seed x episode boolean matrices, S x E): base_only (ME fails, EM succeeds),
    arm_only, both, interaction (ME and EM both succeed); shares of the MM failures with two-level bootstrap CIs (the
    shares' seeds and episodes resampled). Returns dict(n_mm_failures, <category>: dict(count, share, ci, per_seed),
    base_cause, arm_cause)."""
    S, E = fail.shape
    cats = {"base_only": fail & me_f & ~em_f, "arm_only": fail & em_f & ~me_f, "both": fail & me_f & em_f,
            "interaction": fail & ~me_f & ~em_f}
    n_fail = int(fail.sum())
    out = {"n_mm_failures": n_fail, "n_episodes": E}
    boot = Bootstrap([S], E, n_boot, seed) if E else None
    den = boot.means(fail.astype(float)) if boot else np.array([])
    for name, m in cats.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            share_draws = boot.means(m.astype(float)) / den if boot else np.array([])
        out[name] = {"count": int(m.sum()), "share": m.sum() / n_fail if n_fail else np.nan,
                     "ci": percentile_ci(share_draws),
                     "per_seed": [float(m[s].sum() / fail[s].sum()) if fail[s].sum() else np.nan for s in range(S)]}
    out["base_cause"] = (cats["base_only"] | cats["both"]).sum() / n_fail if n_fail else np.nan
    out["arm_cause"] = (cats["arm_only"] | cats["both"]).sum() / n_fail if n_fail else np.nan
    return out


def _attribution_lines(att):
    lines = []
    for name in ("base_only", "arm_only", "both", "interaction"):
        a = att[name]
        lines.append(f"  {name:<12} {a['count']:>5}  {pct(a['share'])}% [{pct(a['ci'][0])}, {pct(a['ci'][1])}]")
    lines.append(f"  base cause (base_only + both) {pct(att['base_cause'])}%, arm cause (arm_only + both) "
                 f"{pct(att['arm_cause'])}%")
    return lines


def split2x2(ee_run, em_runs, me_runs, mm_runs, label="model", n_boot=N_BOOT, seed=BOOT_SEED, allow_incomplete=None):
    """Experiment B on the common episodes. em_runs / me_runs / mm_runs are lists with one run per seed, in the same
    seed order. The runs must pass check_integrity (refused otherwise, unless allow_incomplete; hybrid runs need
    --no-early-stop). Each contrast against EE: two-level bootstrap and the one-sample t on the per-seed differences
    (cell rate - EE rate); the Holm family of the 3 contrasts uses max(bootstrap p, t p). Returns (lines, result)."""
    all_runs = [ee_run, *em_runs, *me_runs, *mm_runs]
    warn = check_integrity(all_runs, f"experiment B ({label})", allow_incomplete)
    episodes, excluded = paired_episodes(all_runs)
    E, S = len(episodes), len(mm_runs)
    lines = [f"== Experiment B: base / arm 2x2 split, {label}, {S} seeds, {E} common episodes ==",
             f"EE = {ee_run.name}; EM = expert base + model arm; ME = model base + expert arm; MM = full model. "
             "Coupling by construction: the expert arm extends at the shadow FSM's nav_done (a 'parked' signal to "
             "the model base); the expert base commands exactly 0 after nav_done (a perfect stillness signal to the "
             "model arm): the single-swap cells overestimate their half."]
    lines += warn
    res = {"label": label, "n_episodes": E, "n_seeds": S, "ee": ee_run.name, "em": [r.name for r in em_runs],
           "me": [r.name for r in me_runs], "mm": [r.name for r in mm_runs], "integrity_warnings": warn,
           "harness_error_excluded": excluded}
    if excluded:
        lines.append(f"  harness_error episodes left out of every cell: {_ids(excluded)}")
    if E == 0 or S == 0:
        lines.append("  no common episodes / no seeds")
        return lines, res
    succ = lambda row: metric_value(row, "success")  # noqa: E731
    YEE = matrix([ee_run], episodes, succ)
    YEM, YME, YMM = (matrix(r, episodes, succ) for r in (em_runs, me_runs, mm_runs))
    boot = Bootstrap([1, S], E, n_boot, seed)
    res["success"] = {}
    for cell, Y, arm in (("EE", YEE, 0), ("EM", YEM, 1), ("ME", YME, 1), ("MM", YMM, 1)):
        res["success"][cell] = {"mean": Y.mean(), "ci": percentile_ci(boot.means(Y, arm)),
                                "per_seed": Y.mean(axis=1).tolist()}
    lines.append("Seed-mean success (two-level bootstrap 95% CI): " + "; ".join(
        f"{c} {pct_ci(v['mean'], *v['ci'])}%" for c, v in res["success"].items()))
    lines.append("  per seed: " + "; ".join(f"{c} " + "/".join(pct(x, 0) for x in v["per_seed"])
                                           for c, v in res["success"].items() if c != "EE"))
    # ME cell by shadow-FSM nav_done trigger: successes and failures
    trig = me_triggers(me_runs, episodes)
    res["me_trigger"] = {}
    for key in ("threshold", "timeout", "never", "no_log"):
        m = trig == key
        res["me_trigger"][key] = {"n": int(m.sum()), "success": int((m & (YME == 1)).sum()),
                                  "failure": int((m & (YME == 0)).sum())}
    lines.append("ME cell by the shadow FSM's nav_done trigger (seed x episode): " + "; ".join(
        f"{k} {v['n']} (success {v['success']}, failure {v['failure']}, "
        f"{pct(v['success'] / v['n']) if v['n'] else 'nan'}% success)" for k, v in res["me_trigger"].items() if v["n"]))
    lines.append("  NOTE: a timeout-triggered ME success does not show the model base parked: the expert arm reaches "
                 f"from wherever the base is after the shadow FSM's {BASE_NAV_TIMEOUT:g} s timeout (e.g. an M0-like "
                 "start already within reach); only a threshold trigger means the base matched the expert's parking pose")
    # attribution of the MM failures: all episodes, and without the M0-like episodes (|d0| < START_ON_RING)
    fail, me_f, em_f = YMM == 0, YME == 0, YEM == 0
    att = attribution(fail, me_f, em_f, n_boot, seed)
    res["attribution"] = att
    lines.append(f"Attribution of the {att['n_mm_failures']} MM failures (seed x episode; base cause = ME fails, arm "
                 f"cause = EM fails, interaction = ME and EM both succeed):")
    lines += _attribution_lines(att)
    d0 = np.array([_f(ee_run.by_episode[e], "d0") for e in episodes])
    keep = ~(np.abs(d0) < START_ON_RING)
    att_far = attribution(fail[:, keep], me_f[:, keep], em_f[:, keep], n_boot, seed + 1)
    res["attribution_no_m0like"] = att_far
    lines.append(f"Attribution without the {int((~keep).sum())} M0-like episodes (|d0| < {START_ON_RING} m, where an "
                 f"unmoved base can let the expert arm reach after the timeout): {att_far['n_mm_failures']} MM failures "
                 f"on {att_far['n_episodes']} episodes:")
    lines += _attribution_lines(att_far)
    ee_fail = np.broadcast_to(YEE == 0, fail.shape)
    lines.append(f"  of the MM failures, {int((fail & ee_fail).sum())} are on episodes the expert (EE) also fails")
    p_me, p_em, p_mm = YME.mean(), YEM.mean(), YMM.mean()
    res["independence"] = {"p_me_x_p_em": p_me * p_em, "p_mm": p_mm,
                           "per_seed": [(float(a * b), float(c)) for a, b, c in
                                        zip(YME.mean(axis=1), YEM.mean(axis=1), YMM.mean(axis=1))]}
    lines.append(f"Independence reference p_ME x p_EM = {pct(p_me * p_em)}% vs observed MM {pct(p_mm)}% (seed means); "
                 "per seed: " + ", ".join(f"{pct(a, 0)}/{pct(c, 0)}" for a, c in res["independence"]["per_seed"]))
    # paired differences against EE: bootstrap and the one-sample t; Holm over the 3 on max(bootstrap p, t p)
    lines.append(f"Cell - EE (paired, same episodes): two-level bootstrap and one-sample t on the per-seed differences "
                 f"(cell rate - EE rate, {S - 1} df); Holm over the 3 contrasts on max(bootstrap p, t p), reject at "
                 f"p_adj <= {ALPHA}:")
    res["diffs"] = {}
    for cell, Y in (("ME", YME), ("EM", YEM), ("MM", YMM)):
        dd = paired_diff(YEE, Y, boot=boot)
        dd["t"] = seed_t_test(seed_rates(YEE), seed_rates(Y))
        dd["p_used"] = combined_p(dd["p"], dd["t"]["p"])
        res["diffs"][cell] = dd
    adj, rej = holm([res["diffs"][c]["p_used"] for c in ("ME", "EM", "MM")])
    for c, a, r in zip(("ME", "EM", "MM"), adj, rej):
        dd = res["diffs"][c]
        dd["p_holm"], dd["reject"] = a, bool(r)
        lines.append(f"  {c} - EE: bootstrap {pp_ci(dd['diff'], dd['lo'], dd['hi'])}, p {fmt_p(dd['p'])}; "
                     f"t {pp_ci(dd['t']['diff'], dd['t']['lo'], dd['t']['hi'])}, p {fmt_p(dd['t']['p'])}; "
                     f"p used {fmt_p(dd['p_used'])}, Holm {fmt_p(a)}{' (significant)' if r else ''}")
    if S < 2:
        lines.append("  (t test undefined with < 2 seeds: the contrasts stay out of the Holm family)")
    base_sig, arm_sig = res["diffs"]["ME"]["reject"], res["diffs"]["EM"]["reject"]
    verdict = {(True, False): "only base", (False, True): "only arm", (True, True): "both",
               (False, False): "neither significant"}[(base_sig, arm_sig)]
    res["verdict"] = verdict
    lines.append(f"Verdict category (Holm-adjusted max(bootstrap p, t p) <= {ALPHA} of ME - EE and EM - EE): {verdict}")

    # hybrid log diagnostics
    lines.append("Hybrid log diagnostics:")
    lags, never_ext, no_nav = [], 0, 0
    for run in em_runs:
        logs = run_logs(run, episodes)
        has_ext = run.has("arm_extend_time")
        for e in episodes:
            log = logs[e]
            if log is None or log["nav_done"] is None:
                no_nav += 1
                continue
            ext = _f(run.by_episode[e], "arm_extend_time") if has_ext else np.nan
            if not np.isfinite(ext):
                never_ext += 1
                continue
            lags.append(ext - log["nav_done"]["t"])
    lags = np.array(lags)
    res["em_lag"] = {"n": int(lags.size), "never_extended": never_ext, "no_nav_done_in_log": no_nav}
    if lags.size:
        res["em_lag"].update(median=float(np.median(lags)), p5=float(np.percentile(lags, 5)),
                             p95=float(np.percentile(lags, 95)), min=float(lags.min()), max=float(lags.max()),
                             n_before_nav_done=int(np.sum(lags < 0)), n_over_0p2=int(np.sum(lags > NAV_LAG_MAX)))
        lines.append(f"  EM: model arm_extend_time - expert nav_done time (log): n {lags.size}, median "
                     f"{np.median(lags):.3f} s, p5 {np.percentile(lags, 5):.3f}, p95 {np.percentile(lags, 95):.3f}, "
                     f"min {lags.min():.3f}, max {lags.max():.3f}; before nav_done {int(np.sum(lags < 0))}, "
                     f"> {NAV_LAG_MAX} s {int(np.sum(lags > NAV_LAG_MAX))}")
    lines.append(f"  EM: never extended {never_ext}, no nav_done in the log {no_nav}")
    return lines, res


# ==============================================================================
# Descriptive analyses
# ==============================================================================
COVERAGE_EDGES = np.linspace(0.0, 1.0, 6)  # quintiles of the train leave-one-out distance distribution


def coverage_bins(Y, cov, n_boot=N_BOOT, seed=BOOT_SEED):
    """Seed-mean success per coverage-percentile bin (fixed edges 0, 0.2, ..., 1: the percentile already is
    relative to the train-LOO distances) with two-level bootstrap CIs."""
    cov = np.asarray(cov, dtype=float)
    which = np.clip(np.digitize(cov, COVERAGE_EDGES[1:-1]), 0, len(COVERAGE_EDGES) - 2)
    out = []
    for b in range(len(COVERAGE_EDGES) - 1):
        idx = np.flatnonzero((which == b) & np.isfinite(cov))
        if idx.size == 0:
            out.append({"bin": b, "n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        r = boot2(Y[:, idx], n_boot=n_boot, seed=seed + b)
        out.append({"bin": b, "n": int(idx.size), "mean": r["mean"], "lo": r["lo"], "hi": r["hi"],
                    "centre": float((COVERAGE_EDGES[b] + COVERAGE_EDGES[b + 1]) / 2)})
    return out


def group_runs(runs):
    """{(model, set): [runs sorted by seed]} for model runs (kind model), cell MM only."""
    groups = {}
    for run in runs:
        if run.meta["kind"] != "model":
            continue
        groups.setdefault((run.meta["model"], run.meta["set"]), []).append(run)
    for key in groups:
        groups[key].sort(key=lambda r: (r.meta["seed"] is None, r.meta["seed"] or 0))
    return groups


OFFLINE_CELL = ("all", "all", "all")   # (group, class, window) of the overall validation cell of an offline.json


def read_offline(path):
    """trained/<name>.offline.json (written by train_policy.py offline) -> {'val_loss': float, 'rmse': {...},
    'nrmse': {...}}. RMSE and nRMSE (RMSE / training label std) come from the overall validation cell (cells entry
    with group, class, window = all, all, all): outputs v and w, and tau_mean = the mean over the 7 joints of the
    per-joint torque value (for nRMSE equal to the file's nrmse_tau_mean). Missing parts are left out."""
    with open(path) as f:
        data = json.load(f)
    out = {"val_loss": float(data["val_loss"]) if data.get("val_loss") is not None else np.nan, "rmse": {},
           "nrmse": {}}
    cell = next((c for c in data.get("cells") or [] if isinstance(c, dict)
                 and (c.get("group"), c.get("class"), c.get("window")) == OFFLINE_CELL), None)
    for key in ("rmse", "nrmse"):
        values = (cell or {}).get(key) or {}
        for name in ("v", "w"):
            if isinstance(values.get(name), (int, float)):
                out[key][name] = float(values[name])
        taus = [float(v) for k, v in values.items() if k.startswith("tau") and isinstance(v, (int, float))]
        if taus:
            out[key]["tau_mean"] = float(np.mean(taus))
    return out


def parameter_count(path):
    """Number of weights + biases of a model file (bc_features format)."""
    import bc_features
    model = bc_features.load_model(path)
    return int(sum(np.asarray(w).size for w in model["weights"]) + sum(np.asarray(b).size for b in model["biases"]))


def describe(runs, trained_dir=TRAINED_DIR, n_boot=N_BOOT, seed=BOOT_SEED, allow_incomplete=None):
    """Descriptive part (no adoption decision): seed consistency, success vs coverage, offline vs closed loop, M1 vs
    M2 arm trigger, efficiency. runs: prepared runs. The seeds of every (model, set) group are paired, so each group
    must pass check_integrity (refused otherwise, listing every failing group, unless allow_incomplete; then
    harness_error episodes are left out). Returns (lines, result dict)."""
    lines, res = ["== Descriptive analyses =="], {"consistency": [], "coverage": [], "offline": [], "m1m2": [],
                                                    "efficiency": [], "integrity_warnings": []}
    groups = group_runs(runs)
    problems = [f"{model} {set_name}: {p}" for (model, set_name), rs in sorted(groups.items())
                for p in integrity_problems(rs)]
    if problems:
        allow = ALLOW_INCOMPLETE if allow_incomplete is None else allow_incomplete
        if not allow:
            raise SystemExit("describe: refused, the model runs fail the integrity checks:\n  " + "\n  ".join(problems)
                             + "\n(re-run / complete the runs, or pass --allow-incomplete to describe the common "
                               "episodes anyway, harness_error episodes left out)")
        res["integrity_warnings"] = [f"  WARNING (--allow-incomplete): {p}" for p in problems]
        lines += res["integrity_warnings"]
    # seed consistency
    lines.append("Seed consistency (models with >= 2 seeds): Cochran Q across seeds (chi2, S - 1 df), normalised seed "
                 "lottery psi = psi_raw / psi_indep (0 outcome fixed by the episode, 1 pure seed lottery):")
    for (model, set_name), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        eps, _ = paired_episodes(rs)
        Y = matrix(rs, eps, lambda row: metric_value(row, "success"))
        q, df, p = cochran_q(Y)
        psi = seed_lottery(Y)
        res["consistency"].append({"model": model, "set": set_name, "S": len(rs), "E": len(eps), "Q": q, "df": df,
                                   "p": p, **psi, "per_seed": Y.mean(axis=1).tolist()})
        lines.append(f"  {model} {set_name}: S {len(rs)}, E {len(eps)}, per-seed success "
                     + "/".join(pct(x, 0) for x in Y.mean(axis=1))
                     + f"; Q {q:.1f} (df {df}), p {fmt_p(p)}; psi_raw {psi['psi_raw']:.3f}, psi_indep "
                       f"{psi['psi_indep']:.3f}, psi {psi['psi']:.2f}")
    if not res["consistency"]:
        lines.append("  (no model with >= 2 seeds)")
    # success vs coverage
    lines.append("Success vs coverage percentile (seed-mean success per bin of the percentile 0-0.2-...-1, two-level "
                 "bootstrap 95% CI; logistic slope per unit percentile, CI from the two-level bootstrap):")
    for (model, set_name), rs in sorted(groups.items()):
        eps, _ = paired_episodes(rs)
        Y = matrix(rs, eps, lambda row: metric_value(row, "success"))
        for rep in ("world", "rel"):
            cov = np.array([rs[0].by_episode[e].get(f"ref_cov_pct_{rep}", np.nan) for e in eps], dtype=float)
            if not np.isfinite(cov).any():
                continue
            bins = coverage_bins(Y, cov, n_boot=n_boot, seed=seed)
            slope = logistic_slope(Y, cov, n_boot=min(n_boot, 1000), seed=seed)
            res["coverage"].append({"model": model, "set": set_name, "rep": rep, "bins": bins, "slope": slope})
            lines.append(f"  {model} {set_name} cov_pct_{rep}: " + "; ".join(
                f"{b['bin'] + 1}: {pct(b['mean'], 0)} [{pct(b['lo'], 0)}, {pct(b['hi'], 0)}] (n {b['n']})" for b in bins)
                + f"; slope {slope['slope']:+.2f} [{slope['lo']:+.2f}, {slope['hi']:+.2f}]")
    # offline vs closed loop (DEV)
    lines.append("Offline validation error vs DEV closed-loop success (one point per model x seed; RMSE / nRMSE of the "
                 "overall validation cell: v, w, tau_mean = mean over the 7 joints):")
    pts = []
    for (model, set_name), rs in sorted(groups.items()):
        if set_name != "dev":
            continue
        for run in rs:
            stem = f"{model}_s{run.meta['seed']}" if run.meta["seed"] is not None else model
            path = os.path.join(trained_dir, stem + ".offline.json")
            if not os.path.exists(path):
                lines.append(f"  {stem}: no {os.path.relpath(path, HERE)}")
                continue
            off = read_offline(path)
            rows = [row for row in run.rows if row.get("stop_reason") not in RUN_ERROR_REASONS]
            k = int(sum(metric_value(row, "success") for row in rows))
            n = len(rows)
            lo, hi = wilson(k, n)
            k5 = int(sum(metric_value(row, "reach_5cm") for row in rows))   # the floor-rule decision metric
            lo5, hi5 = wilson(k5, n)
            pts.append({"model": model, "seed": run.meta["seed"], "val_loss": off["val_loss"], "rmse": off["rmse"],
                        "nrmse": off["nrmse"], "success": k / n if n else np.nan, "lo": lo, "hi": hi,
                        "reach_5cm": k5 / n if n else np.nan, "lo5": lo5, "hi5": hi5})
            lines.append(f"  {stem}: val_loss {off['val_loss']:.4g}, DEV success {pct(k / n if n else np.nan)}% "
                         f"[{pct(lo)}, {pct(hi)}]"
                         + (", RMSE " + ", ".join(f"{a} {b:.3g}" for a, b in off["rmse"].items()) if off["rmse"] else "")
                         + (", nRMSE " + ", ".join(f"{a} {b:.3g}" for a, b in off["nrmse"].items())
                            if off["nrmse"] else ""))
    res["offline"] = pts
    if len(pts) >= 3:
        rho5, p5 = stats.spearmanr([q["val_loss"] for q in pts], [q["reach_5cm"] for q in pts])
        lines.append(f"  Spearman(val_loss, DEV reach_5cm) {rho5:+.2f}, p {fmt_p(p5)}, n {len(pts)} (descriptive)")
        rho, p = stats.spearmanr([q["val_loss"] for q in pts], [q["success"] for q in pts])
        res["offline_spearman"] = {"rho": float(rho), "p": float(p), "n": len(pts)}
        lines.append(f"  Spearman(val_loss, DEV success) {rho:+.2f}, p {fmt_p(p)}, n {len(pts)} (descriptive)")
    # M1 vs M2: parked_no_arm and arm_extended
    lines.append("M1 vs M2 (seed-mean share of the episodes of the mode; CIs two-level bootstrap within the mode; "
                 "difference with one shared seed resample):")
    for (model, set_name), rs in sorted(groups.items()):
        if set_name not in ("dev", "hard") or not all(getattr(r, "classified", False) for r in rs):
            continue
        eps, _ = paired_episodes(rs)
        modes = np.array([rs[0].by_episode[e].get("ref_mode") for e in eps])
        rows_out = {"model": model, "set": set_name}
        for metric, fn in (("parked_no_arm", lambda row: float(row["fclass"] == "parked_no_arm")),
                           ("arm_extended", lambda row: metric_value(row, "arm_extended"))):
            Y = matrix(rs, eps, fn)
            parts = {}
            for mode in ("M1", "M2"):
                idx = np.flatnonzero(modes == mode)
                if idx.size == 0:
                    continue
                r = boot2(Y[:, idx], n_boot=n_boot, seed=seed)
                parts[mode] = {"n": int(idx.size), "mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "idx": idx}
            if len(parts) == 2:
                rng = np.random.default_rng(seed)
                S = Y.shape[0]
                ws = _weights(rng, S, n_boot)
                w1, w2 = _weights(rng, parts["M1"]["n"], n_boot), _weights(rng, parts["M2"]["n"], n_boot)
                draws = (np.sum((ws @ Y[:, parts["M1"]["idx"]]) * w1, axis=1)
                         - np.sum((ws @ Y[:, parts["M2"]["idx"]]) * w2, axis=1))
                diff = parts["M1"]["mean"] - parts["M2"]["mean"]
                lo, hi = percentile_ci(draws)
                rows_out[metric] = {m: {k: v for k, v in p.items() if k != "idx"} for m, p in parts.items()}
                rows_out[metric]["diff"] = {"diff": diff, "lo": lo, "hi": hi, "p": boot_p(draws)}
                lines.append(f"  {model} {set_name} {metric}: M1 {pct_ci(parts['M1']['mean'], parts['M1']['lo'], parts['M1']['hi'])}% "
                             f"(n {parts['M1']['n']}), M2 {pct_ci(parts['M2']['mean'], parts['M2']['lo'], parts['M2']['hi'])}% "
                             f"(n {parts['M2']['n']}), M1 - M2 {pp_ci(diff, lo, hi)}")
        res["m1m2"].append(rows_out)
    # efficiency
    lines.append("Efficiency: policy step time (median over episodes of the per-episode mean / p99, ms) and parameters:")
    for (model, set_name), rs in sorted(groups.items()):
        means = [np.nanmedian([_f(row, "policy_ms_mean") for row in r.rows]) for r in rs]
        p99 = [np.nanmedian([_f(row, "policy_ms_p99") for row in r.rows]) for r in rs]
        stem = f"{model}_s{rs[0].meta['seed']}" if rs[0].meta["seed"] is not None else model
        path = os.path.join(trained_dir, stem + ".npz")
        n_par = parameter_count(path) if os.path.exists(path) else None
        res["efficiency"].append({"model": model, "set": set_name, "ms_mean": float(np.nanmedian(means)),
                                  "ms_p99": float(np.nanmedian(p99)), "params": n_par})
        lines.append(f"  {model} {set_name}: {np.nanmedian(means):.4f} ms (p99 {np.nanmedian(p99):.4f} ms), "
                     f"parameters {n_par if n_par is not None else 'n/a (no ' + os.path.relpath(path, HERE) + ')'}")
    return lines, res


# ==============================================================================
# Classifier validation
# ==============================================================================
def validate_classifier(expert_runs, zero_runs, label="v1.1 runs"):
    """Checks (i)-(v) of the classifier validation (runs prepared). Returns (lines, list of check dicts)."""
    lines = [f"== Classifier validation ({label}) =="]
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": passed, "detail": detail})
        lines.append(f"[{'PASS' if passed else ('FAIL' if passed is not None else 'INFO')}] {name}: {detail}")

    lines.append("Runs: expert " + ", ".join(r.name for r in expert_runs) + "; zero " + ", ".join(r.name for r in zero_runs))
    errs = harness_errors(expert_runs + zero_runs)
    add("no harness_error", not errs, f"harness_error episodes {errs}" if errs else "none")
    unclassified = [r.name for r in expert_runs + zero_runs if not getattr(r, "classified", False)]
    if unclassified:
        add("v1.1 classifier columns present", False, f"missing in {unclassified}: {CLASSIFIER_COLUMNS}")
    def ref_of(row):
        return {"align_req_deg": row.get("ref_align_req_deg"), "orient_req_deg": row.get("ref_orient_req_deg"),
                "navdone_ee_err": row.get("ref_navdone_ee_err")}

    # (i) expert failures -> circling
    fails = [(r.name, row) for r in expert_runs for row in r.rows if row["fclass"] != "success"]
    by = class_counts([row for _, row in fails])
    per_set = {r.name: sum(row["fclass"] != "success" for row in r.rows) for r in expert_runs}
    add("(i) expert failures classify as circling",
        all(row["fclass"] == "circling" for _, row in fails) if fails else None,
        f"{len(fails)} failures (14 expected over DEV + HARD + CONFIRM; per run {per_set}); classes {by}")
    for name, row in fails:
        if row["fclass"] != "circling":
            lines.append(f"    {name} ep {row['episode']}: {row['fclass']}, stop {row.get('stop_reason')}, ring min "
                         f"{_f(row, 'ring_dist_min'):.3f}, turn excess "
                         f"{_f(row, 'base_turn_total_deg') - required_turn_deg(row, ref_of(row)):.0f} deg")
    if fails:
        excess = [_f(row, "base_turn_total_deg") - required_turn_deg(row, ref_of(row)) for _, row in fails]
        lines.append(f"    expert failures: turn excess {min(excess):.0f}-{max(excess):.0f} deg, ring_dist_min "
                     f"{min(_f(r, 'ring_dist_min') for _, r in fails):.3f}-{max(_f(r, 'ring_dist_min') for _, r in fails):.3f} m")
    # (ii) zero policy -> no_start (starts at least START_ON_RING from the ring; closer: M0-like, reported separately)
    for run in zero_runs:
        zf = [row for row in run.rows if row["fclass"] != "success"]
        far = [row for row in zf if not row["start_on_ring"]]
        near = [row for row in zf if row["start_on_ring"]]
        add(f"(ii) {run.name}: failures starting >= {START_ON_RING} m from the ring classify as no_start",
            all(row["fclass"] == "no_start" for row in far) if far else None,
            f"{sum(row['fclass'] == 'no_start' for row in far)}/{len(far)} no_start; other classes "
            f"{class_counts([row for row in far if row['fclass'] != 'no_start'])}; successes "
            f"{len(run.rows) - len(zf)}")
        band = [row for row in far if row["start_near_ring"]]
        if band:
            lines.append(f"    of them {len(band)} start {START_ON_RING}-{RING_APPROACH} m from the ring (approached by "
                         f"construction; no_start is tested before the approach test): classes {class_counts(band)}")
        lines.append(f"    reported separately: {len(near)} failures start within {START_ON_RING} m of the ring (M0-like, "
                     f"never no_start): classes {class_counts(near)}; flag no_motion_near_ring "
                     f"{sum(bool(row.get('no_motion_near_ring')) for row in near)}")
        moving = [row for row in zf if not row["no_motion"]]
        if moving:
            lines.append(f"    zero-policy failures that moved (path >= {NO_START_PATH} m or turn >= {NO_START_TURN_DEG} "
                         f"deg): {len(moving)}, max path {max(_f(r, 'base_path_length') for r in moving):.3f} m, max turn "
                         f"{max(_f(r, 'base_turn_total_deg') for r in moving):.1f} deg")
    if not zero_runs:
        add("(ii) zero policy", None, "no zero-policy run")
    # (iii) expert successes: failure predicates that would fire, against the expected exceptions
    succ = [(r, row) for r in expert_runs for row in r.rows if row["fclass"] == "success"]
    names = ("crash", "never_approached", "no_start", "runaway", "no_progress", "circling", "left_or_overshot", "parked_no_arm",
             "extended_not_reached")
    fired = {k: 0 for k in names}
    hits, pna = [], []
    for r, row in succ:
        p = failure_predicates(row, ref_of(row))
        for k in names:
            if p[k]:
                fired[k] += 1
                hits.append((r.meta.get("set"), row["episode"], k, r.name, row))
        if p["parked_no_arm"]:
            pna.append(row)
    expected = [h for h in hits if EXPECTED_SUCCESS_FIRINGS.get((h[0], h[1])) == h[2]]
    unexpected = [h for h in hits if EXPECTED_SUCCESS_FIRINGS.get((h[0], h[1])) != h[2]]
    add("(iii) expert successes: no failure predicate fires except the expected exceptions", not unexpected,
        f"{len(succ)} successes; fired per predicate " + ", ".join(f"{k} {v}" for k, v in fired.items())
        + f"; expected exceptions {len(expected)}, unexpected {len(unexpected)}")

    def hit_text(h):
        set_name, episode, k, run_name, row = h
        extra = (f", turn excess {_f(row, 'base_turn_total_deg') - required_turn_deg(row, ref_of(row)):.0f} deg, "
                 f"parked_time {_f(row, 'parked_time'):.3f} s" if k == "circling" else "")
        return f"{run_name} ep {episode} {k}{extra}"
    lines.append(f"    expected exceptions (genuine circling before a late success: the expert circled until the "
                 f"{BASE_NAV_TIMEOUT:g} s navigation timeout, parked after it and still reached; "
                 + ", ".join(f"{s.upper()} {e} {k}" for (s, e), k in EXPECTED_SUCCESS_FIRINGS.items()) + "): "
                 + (", ".join(hit_text(h) for h in expected) if expected else "none fired"))
    analysed = {r.meta.get("set") for r in expert_runs}
    not_seen = [f"{s.upper()} {e} {k}" for (s, e), k in EXPECTED_SUCCESS_FIRINGS.items()
                if s in analysed and not any((h[0], h[1], h[2]) == (s, e, k) for h in expected)]
    if not_seen:
        lines.append(f"    expected exceptions that did not fire (informational): {', '.join(not_seen)}")
    if unexpected:
        lines.append("    UNEXPECTED: " + ", ".join(hit_text(h) for h in unexpected[:20])
                     + (f", ... ({len(unexpected)} in all)" if len(unexpected) > 20 else ""))
    succ = [row for _, row in succ]
    no_ext = [row for row in succ if not np.isfinite(_f(row, "arm_extend_time"))]
    lines.append(f"    successes with arm_extend_time NaN (EE never displaced > 0.10 m in the base frame): {len(no_ext)}"
                 f"/{len(succ)} ({pct(len(no_ext) / max(len(succ), 1))}%)"
                 + (f"; their navdone_ee_err median {np.median([_f(r, 'ref_navdone_ee_err') for r in no_ext]):.3f} m, "
                    f"range {min(_f(r, 'ref_navdone_ee_err') for r in no_ext):.3f}-"
                    f"{max(_f(r, 'ref_navdone_ee_err') for r in no_ext):.3f} m; ee_base_disp_max median "
                    f"{np.nanmedian([_f(r, 'ee_base_disp_max') for r in no_ext]):.3f} m" if no_ext else ""))
    n_short = {r.name: sum(bool(row.get("short_reach")) for row in r.rows) for r in expert_runs}
    lines.append(f"    short-reach episodes (expert hold posture at parking < {PARKED_NO_ARM_MIN_NAVDONE:.2f} m from the "
                 f"target: parked_no_arm cannot fire there): {n_short}")
    if pna:
        lines.append("    parked_no_arm fires on: " + ", ".join(
            f"ep {r['episode']} (min_error {_f(r, 'min_error'):.4f}, navdone_ee_err {_f(r, 'ref_navdone_ee_err'):.4f})"
            for r in pna[:20]))
    # (iv) expert arm_extend_time never before its recorded nav_done
    lag = np.array([_f(row, "arm_extend_time") - _f(row, "ref_navdone_time") for r in expert_runs for row in r.rows])
    ok = np.isfinite(lag)
    n_nan = int(np.sum(~ok))
    add("(iv) expert arm_extend_time >= recorded nav_done time", bool(ok.any()) and int(np.sum(lag[ok] < 0)) == 0,
        f"{int(np.sum(lag[ok] < 0))} of {int(ok.sum())} episodes extend before nav_done ({n_nan} without arm_extend_time "
        f"or nav_done)")
    # (v) lag distribution
    if ok.any():
        lv = lag[ok]
        add(f"(v) expert arm_extend_time - nav_done (reported; the plan's < {NAV_LAG_MAX} s cannot hold)", None,
            f"median {np.median(lv):.3f} s, p95 {np.percentile(lv, 95):.3f} s, max {np.max(lv):.3f} s, "
            f"> {NAV_LAG_MAX} s: {int(np.sum(lv > NAV_LAG_MAX))}/{lv.size}")
    else:
        add(f"(v) expert arm_extend_time - nav_done < {NAV_LAG_MAX} s", None, "no episode with both times")
    return lines, checks


def recorded_v11_columns(npz_path, n_obs, standoff):
    """Development aid: the classifier's v1.1 columns recomputed from a recorded expert episode over its first n_obs
    samples (ring distance, arm_extend_time / ee_base_disp_max with the full base orientation, parked_time with
    finite-difference base velocities). An approximation of evaluate_policy's v1.1 columns, for the precheck only."""
    from scipy.spatial.transform import Rotation
    with np.load(npz_path) as d:
        t = d["t"].astype(float)
        base_pos = d["base_pos"].astype(float)
        base_ori = d["base_ori"].astype(float)
        yaw = d["base_bearing"].astype(float)
        ee = d["ee_pos"].astype(float)
        target = d["target"].astype(float)
    n = int(min(len(t), n_obs))
    t, base_pos, base_ori, yaw, ee = t[:n], base_pos[:n], base_ori[:n], yaw[:n], ee[:n]
    ring = np.abs(np.linalg.norm(target[:2] - base_pos[:, :2], axis=1) - standoff)
    p_b = Rotation.from_quat(base_ori).inv().apply(ee - base_pos)
    disp = np.linalg.norm(p_b - p_b[0], axis=1)
    k = np.flatnonzero(disp > 0.10)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.001
    v = np.r_[0.0, np.linalg.norm(np.diff(base_pos[:, :2], axis=0), axis=1) / dt]
    w = np.r_[0.0, np.abs(ad.wrap(np.diff(yaw))) / dt]
    good = (ring < 0.20) & (v < 0.01) & (w < 0.02)  # evaluate_policy.PARK_RING_TOL, STILL_LIN, STILL_ANG
    need = round(1.0 / dt)
    parked_time, run_len = np.nan, 0
    for i, g in enumerate(good):
        run_len = run_len + 1 if g else 0
        if run_len >= need:
            parked_time = float(t[i])
            break
    return {"ring_dist_start": float(ring[0]), "ring_dist_min": float(ring.min()), "ring_dist_final": float(ring[-1]),
            "arm_extend_time": float(t[k[0]]) if k.size else np.nan, "ee_base_disp_max": float(disp.max()),
            "parked_time": parked_time, "precheck_samples": n}


def precheck_recorded(set_name, results_dir=RESULTS_DIR, out_dir=STEP3_DIR, data_dir=DATA_DIR):
    """Development check before the v1.1 runs exist: pseudo-v1.1 expert rows = results/expert_<set>/rollouts.csv (v1)
    + the classifier columns recomputed from the recorded expert episodes up to the v1 stop point (or the end of the
    recording when it is shorter), then the classifier validation checks (i), (iii)-(v). Returns (lines, checks)."""
    check_set_allowed(set_name)
    if set_name == "confirm":
        raise SystemExit("precheck-recorded reads the recorded episodes: not allowed on data/confirm")
    run = load_run(os.path.join(results_dir, f"expert_{set_name}"))
    ref = reference_for(run, out_dir, data_dir)
    set_dir = os.path.join(data_dir, set_name)
    index, _ = read_rows(os.path.join(set_dir, "index.csv"))
    files = {row["episode"]: row["file"] for row in index}
    truncated = 0
    for row in run.rows:
        n_obs = int(row["n_steps"]) + 1
        cols = recorded_v11_columns(os.path.join(set_dir, files[row["episode"]]), n_obs, ref[row["episode"]]["standoff"])
        truncated += cols["precheck_samples"] < n_obs
        row.update(cols)
    run.header = run.header + [c for c in ("ring_dist_start", "ring_dist_min", "ring_dist_final", "arm_extend_time",
                                           "ee_base_disp_max", "parked_time") if c not in run.header]
    run.meta["protocol"] = "v1 + recomputed classifier columns"
    prepare(run, out_dir, data_dir)
    lines, checks = validate_classifier([run], [], label=f"PRECHECK on the recorded {set_name.upper()} expert episodes")
    lines.insert(1, f"(recorded episode shorter than the v1 run in {truncated}/{len(run.rows)} episodes: columns over "
                    "the recording only; held_10s unknown)")
    near = sum(ref[e]["d0"] is not None and row["start_near_ring"] for e, row in run.by_episode.items())
    on = sum(row["start_on_ring"] for row in run.rows)
    lines.append(f"Prediction for zero_v11_{set_name}: {on} episodes start within {START_ON_RING} m of the ring "
                 f"(M0-like: never no_start, flag no_motion_near_ring), {near} within {RING_APPROACH} m (those from "
                 f"{START_ON_RING} m on are still no_start: the motion test comes before the approach test)")
    return lines, checks, run


# ==============================================================================
# Figures
# ==============================================================================
MERGED_STYLE = {  # colour (analyze_dataset SERIES / ink) + marker: identity never by colour alone
    "arrive_stay": (ad.SERIES[0], "o", True), "arrive_leave": (ad.SERIES[0], "o", False),
    "arrive": (ad.SERIES[0], "s", True), "base_ok_ee_not": (ad.SERIES[2], "^", True),
    "base_not": (ad.SERIES[1], "X", True), "crash": (ad.INK, "s", True),
}
ATTRIB_COLOURS = {"base_only": ad.SERIES[1], "arm_only": ad.SERIES[2], "both": "#eda100", "interaction": ad.SERIES[0]}
MODEL_ORDER = ("expert", "ego", "ego-novw", "raw", "ego-ols", "raw-ols")
MODEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def majority_merged(M):
    """Per episode (columns of a seed x episode matrix of merged classes): majority class (ties go to the worse class,
    order MERGED_WORST_FIRST) and its share of the seeds."""
    M = np.atleast_2d(M)
    out, share = [], []
    for col in M.T:
        values, counts = np.unique(col, return_counts=True)
        best = max(zip(values, counts), key=lambda vc: (vc[1], -(MERGED_WORST_FIRST.index(vc[0])
                                                                 if vc[0] in MERGED_WORST_FIRST else 99)))
        out.append(best[0])
        share.append(best[1] / len(col))
    return np.array(out, dtype=object), np.array(share)


def plot_failure_d0_theta(panels, out_dir, name="fig_step3_classes_d0_theta.png"):
    """Merged classes on the (d0, theta) plane, one panel per model: each episode drawn at its (d0, theta) with the
    seed-majority merged class (ties to the worse class); hollow circle = arrive & leave. Background as
    results/data_analysis/fig_d0_theta (grey: align turn > 160 deg; hatched: orient turn > 160 deg; lines: |d0| =
    pos_tol). panels: list of dict(title, d0, theta, classes, share)."""
    plt = _plt()
    n = len(panels)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    width = 3.5 if n == 1 else 7.0
    d0_all = np.concatenate([np.asarray(p["d0"], dtype=float) for p in panels])
    xg = np.linspace(min(d0_all.min(), -0.6) - 0.05, d0_all.max() + 0.05, 400)
    yg = np.linspace(-180, 180, 361)
    X, Y = np.meshgrid(xg, yg)
    align, orient = np.vectorize(ad.required_turns)(np.deg2rad(Y), X)
    align, orient = np.rad2deg(align), np.rad2deg(orient)
    with plt.rc_context(ad.REPORT_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(width, 2.6 * nrow + 0.4), layout="constrained", squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k >= n:
                ax.axis("off")
                continue
            p = panels[k]
            ax.contourf(X, Y, (align > 160).astype(float), levels=[0.5, 1.5], colors=[ad.GRID], zorder=0)
            ax.contourf(X, Y, ((orient > 160) & ~(align > 160)).astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=["////"], zorder=0)
            for x in (-ad.BASE.pos_tol, ad.BASE.pos_tol):
                ax.axvline(x, color=ad.MUTED, linewidth=0.6, zorder=1)
            classes = np.asarray(p["classes"], dtype=object)
            d0, th = np.asarray(p["d0"], dtype=float), np.asarray(p["theta"], dtype=float)
            for cls in (*MERGED_CLASSES, "arrive"):
                sel = classes == cls
                if not sel.any() and k > 0:
                    continue
                colour, marker, filled = MERGED_STYLE[cls]
                if cls == "arrive" and not sel.any():
                    continue
                ax.scatter(d0[sel], th[sel], s=9 if cls.startswith("arrive") else 14, marker=marker,
                           facecolors=colour if filled else "none", edgecolors=colour,
                           linewidths=0 if filled else 0.7, zorder=3 if cls.startswith("arrive") else 4,
                           label=f"{cls.replace('_', ' ')}")
            ax.set_xlim(xg[0], xg[-1])
            ax.set_ylim(-180, 180)
            ax.set_yticks([-180, -90, 0, 90, 180])
            ax.set_title(p["title"], color=ad.INK_2, loc="left")
            ax.set_xlabel(r"$d_0 = r - s(z)$ (m)")
            if k % ncol == 0:
                ax.set_ylabel(r"target bearing $\theta$ (deg)")
            ad.style_axes(ax)
        ad.finish(fig, out_dir, name, legend_ax=axes.flat[0], ncol=5)
    plt.close(fig)
    return os.path.join(out_dir, name)


def plot_split2x2(cells, attribution, out_dir, name="fig_step3_split2x2.png"):
    """Experiment B: seed-mean success per cell with CIs (left) and the attribution of the full model's failures
    (right, stacked). cells: {cell: (mean, lo, hi)}; attribution: {category: share}."""
    plt = _plt()
    order = [c for c in ("EE", "EM", "ME", "MM") if c in cells]
    with plt.rc_context(ad.REPORT_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.2), layout="constrained", width_ratios=[1.3, 1])
        x = np.arange(len(order))
        means = np.array([cells[c][0] for c in order]) * 100
        lo = np.array([cells[c][1] for c in order]) * 100
        hi = np.array([cells[c][2] for c in order]) * 100
        a1.bar(x, means, width=0.6, color=ad.MUTED, zorder=2)   # neutral: the legend colours belong to the attribution panel
        a1.errorbar(x, means, yerr=[np.maximum(means - lo, 0), np.maximum(hi - means, 0)], fmt="none",
                    ecolor=ad.INK_2, elinewidth=0.8, capsize=2, zorder=3)
        a1.set_xticks(x, order)
        a1.set_ylim(0, 100)
        a1.set_ylabel("success (%)")
        ad.style_axes(a1)
        bottom = 0.0
        for cat in ("base_only", "arm_only", "both", "interaction"):
            v = 100 * float(attribution.get(cat, 0.0) or 0.0)
            a2.bar([0], [v], bottom=[bottom], width=0.5, color=ATTRIB_COLOURS[cat], edgecolor=ad.SURFACE,
                   linewidth=1.0, label=cat.replace("_", " "), zorder=2)
            bottom += v
        a2.set_xticks([0], ["MM failures"])
        a2.set_xlim(-0.6, 0.6)
        a2.set_ylim(0, 100)
        a2.set_ylabel("share of failures (%)")
        ad.style_axes(a2)
        ad.finish(fig, out_dir, name, legend_ax=a2, ncol=4)
    plt.close(fig)
    return os.path.join(out_dir, name)


def plot_success_coverage(series, out_dir, name="fig_step3_success_coverage.png"):
    """Seed-mean success per coverage-percentile bin, world (left) and relative (right) coverage. series: list of
    dict(label, rep 'world'|'rel', bins [dict(centre, mean, lo, hi)])."""
    plt = _plt()
    labels = list(dict.fromkeys(s["label"] for s in series))
    with plt.rc_context(ad.REPORT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), layout="constrained", sharey=True)
        for ax, rep, title in ((axes[0], "world", "world-frame coverage (7-D)"),
                               (axes[1], "rel", "start-relative coverage (3-D)")):
            for s in series:
                if s["rep"] != rep:
                    continue
                k = labels.index(s["label"])
                bins = [b for b in s["bins"] if b["n"] > 0]
                x = np.array([b["centre"] for b in bins]) + (k - (len(labels) - 1) / 2) * 0.012
                m = np.array([b["mean"] for b in bins]) * 100
                lo = np.array([b["lo"] for b in bins]) * 100
                hi = np.array([b["hi"] for b in bins]) * 100
                ax.errorbar(x, m, yerr=[np.maximum(m - lo, 0), np.maximum(hi - m, 0)], fmt="-",
                            marker=MODEL_MARKERS[k % 8], ms=3.5, color=MODEL_COLOURS[k % 8], linewidth=1.2,
                            elinewidth=0.6, capsize=0, label=s["label"])
            ax.set_xlim(0, 1)
            ax.set_xticks(COVERAGE_EDGES)
            ax.set_xlabel("coverage percentile (train-LOO)")
            ax.set_title(title, color=ad.INK_2, loc="left")
            ad.style_axes(ax)
        axes[0].set_ylabel("seed-mean success (%)")
        axes[0].set_ylim(0, 100)
        ad.finish(fig, out_dir, name, legend_ax=axes[0] if axes[0].get_legend_handles_labels()[0] else axes[1], ncol=6)
    plt.close(fig)
    return os.path.join(out_dir, name)


def plot_offline_vs_closed(points, out_dir, name="fig_step3_offline_vs_closed.png"):
    """Offline validation loss (log axis) vs DEV closed loop, one point per model x seed: success (left) and the
    floor-rule decision metric reach_5cm (right), with Wilson CIs. points: dicts from describe()."""
    plt = _plt()
    models = list(dict.fromkeys(p["model"] for p in points))
    panels = [("success", "lo", "hi", "DEV success (%)"), ("reach_5cm", "lo5", "hi5", "DEV EE within 5 cm (%)")]
    with plt.rc_context(ad.REPORT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained", sharex=True)
        for ax, (key, klo, khi, label) in zip(axes, panels):
            for k, model in enumerate(models):
                ps = [p for p in points if p["model"] == model and key in p]
                if not ps:
                    continue
                x = np.array([p["val_loss"] for p in ps])
                y = np.array([p[key] for p in ps]) * 100
                lo = np.array([p[klo] for p in ps]) * 100
                hi = np.array([p[khi] for p in ps]) * 100
                ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt=MODEL_MARKERS[k % 8], ms=4,
                            color=MODEL_COLOURS[k % 8], elinewidth=0.6, capsize=0, label=f"{model} ({len(ps)})")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(plt.NullFormatter())   # overlapping 2x10^-2, 3x10^-2, ... labels
            ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if min(p["val_loss"] for p in points) / 1.5 <= t
                     <= max(p["val_loss"] for p in points) * 1.5]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.set_xlabel("offline validation loss (standardised MSE)")
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ad.style_axes(ax)
        ad.finish(fig, out_dir, name, legend_ax=axes[0], ncol=5)
    plt.close(fig)
    return os.path.join(out_dir, name)


def figures(runs, out_dir, desc=None, split=None, set_name="dev"):
    """All step-3 figures that the available runs allow. Returns a list of (figure, status) lines."""
    made, lines = [], []
    groups = group_runs(runs)
    panels = []
    expert = [r for r in runs if r.meta["kind"] == "expert" and r.meta["set"] == set_name
              and getattr(r, "classified", False) and r.meta.get("protocol") == "v1.1"]
    candidates = ([("expert", expert[:1])] if expert else []) + [
        (m, rs) for (m, s), rs in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0])
                                                                         if kv[0][0] in MODEL_ORDER else 99))
        if s == set_name]
    for model, rs in candidates:
        if not all(getattr(r, "classified", False) for r in rs):
            continue
        eps = common_episodes(rs)
        cls, share = majority_merged(class_matrix(rs, eps, "mclass"))
        panels.append({"title": f"{model} ({len(rs)} seed{'s' if len(rs) > 1 else ''}, {set_name.upper()})",
                       "d0": [rs[0].by_episode[e]["d0"] for e in eps],
                       "theta": [rs[0].by_episode[e]["theta_deg"] for e in eps], "classes": cls, "share": share})
    if panels:
        made.append(plot_failure_d0_theta(panels, out_dir))
    else:
        lines.append("fig_step3_classes_d0_theta: skipped (no classified v1.1 run on " + set_name + ")")
    if split is not None and split.get("success"):
        made.append(plot_split2x2({c: (v["mean"], *v["ci"]) for c, v in split["success"].items()},
                                  {k: v["share"] for k, v in split["attribution"].items() if isinstance(v, dict)},
                                  out_dir))
    else:
        lines.append("fig_step3_split2x2: skipped (no experiment-B runs)")
    if desc is not None and desc.get("coverage"):
        series = [{"label": f"{c['model']} {c['set']}", "rep": c["rep"], "bins": c["bins"]}
                  for c in desc["coverage"] if c["set"] == set_name]
        if series:
            made.append(plot_success_coverage(series, out_dir))
    else:
        lines.append("fig_step3_success_coverage: skipped (no model runs with coverage references)")
    if desc is not None and desc.get("offline"):
        made.append(plot_offline_vs_closed(desc["offline"], out_dir))
    else:
        lines.append("fig_step3_offline_vs_closed: skipped (no trained/*.offline.json matching a DEV run)")
    lines += [f"{os.path.relpath(p, HERE)} (+ .pdf)" for p in made]
    return lines


# ==============================================================================
# Report
# ==============================================================================
def _prepare_all(runs, out_dir, data_dir, lines):
    ok = []
    for run in runs:
        try:
            ok.append(prepare(run, out_dir, data_dir))
        except (SystemExit, FileNotFoundError, KeyError) as e:
            lines.append(f"  WARNING: {run.name} not analysed: {e}")
    return ok


def _find_seeds(results_dir, model, set_name, prefix=""):
    seeds = []
    for path in glob.glob(os.path.join(results_dir, f"{prefix}{model}_s*_{set_name}")):
        m = re.match(rf"^{re.escape(prefix + model)}_s(\d+)_{set_name}$", os.path.basename(path))
        if m and os.path.exists(os.path.join(path, "rollouts.csv")):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def expert_run_for(results_dir, set_name, out_dir=STEP3_DIR, data_dir=DATA_DIR):
    """The prepared expert_v11_<set> run (reference of experiment A's metrics on the same episodes), or None when it
    does not exist or cannot be prepared (e.g. CONFIRM without --allow-confirm)."""
    path = run_dir_for(results_dir, "expert", None, set_name)
    if (set_name == "confirm" and not ALLOW_CONFIRM) or not os.path.exists(os.path.join(path, "rollouts.csv")):
        return None
    try:
        return load_prepared(path, out_dir, data_dir)
    except SystemExit:
        return None


def run_split(results_dir, model, seeds, set_name, out_dir, data_dir, n_boot, seed, ee_run_dir=None):
    """Loads the experiment-B runs of a model; returns (lines, result or None, missing list)."""
    missing = []
    ee_path = ee_run_dir or run_dir_for(results_dir, None, None, set_name, cell="EE")
    if not os.path.exists(os.path.join(ee_path, "rollouts.csv")):
        alt = run_dir_for(results_dir, "expert", None, set_name)
        if ee_run_dir is None and os.path.exists(os.path.join(alt, "rollouts.csv")):
            ee_path = alt
        else:
            return [f"Experiment B {model} {set_name}: missing EE run {ee_path}"], None, [ee_path]
    em, me, mm, used = [], [], [], []
    for s in seeds:
        paths = [run_dir_for(results_dir, model, s, set_name, cell="EM"), run_dir_for(results_dir, model, s, set_name, cell="ME"),
                 run_dir_for(results_dir, model, s, set_name)]
        absent = [p for p in paths if not os.path.exists(os.path.join(p, "rollouts.csv"))]
        if absent:
            missing += absent
            continue
        a, b, c = (load_prepared(p, out_dir, data_dir) for p in paths)
        em.append(a)
        me.append(b)
        mm.append(c)
        used.append(s)
    if not used:
        return [f"Experiment B {model} {set_name}: no seed with all of EM, ME, MM ({len(missing)} folders missing)"], None, missing
    ee = load_prepared(ee_path, out_dir, data_dir)
    lines, res = split2x2(ee, em, me, mm, label=f"{model} {set_name} seeds {used}", n_boot=n_boot, seed=seed)
    base = run_dir_for(results_dir, "expert", None, set_name)
    if ee.meta["cell"] == "EE" and os.path.exists(os.path.join(base, "rollouts.csv")):
        ex = load_run(base)
        eps = common_episodes([ee, ex])
        same = sum(metric_value(ee.by_episode[e], "success") == metric_value(ex.by_episode[e], "success") for e in eps)
        res["ee_vs_expert"] = {"n": len(eps), "same_outcome": int(same)}
        lines.append(f"  EE check: {ee.name} and {ex.name} have the same outcome on {same}/{len(eps)} episodes"
                     + ("" if same == len(eps) else " (WARNING: the hybrid with expert halves differs from the expert)"))
    if missing:
        lines.append(f"  missing folders (seeds left out): {[os.path.relpath(p, HERE) for p in missing]}")
    return lines, res, missing


def write_report(results_dir=RESULTS_DIR, out_dir=STEP3_DIR, data_dir=DATA_DIR, trained_dir=TRAINED_DIR,
                 n_boot=N_BOOT, seed=BOOT_SEED, a_model="ego", b_model="ego-novw", predicted=("no_start", "left_or_overshot"),
                 with_figures=True):
    """results/step3/summary.txt with every part the available runs allow; missing parts are listed."""
    lines = [f"Step-3 closed-loop analysis (analyze_rollouts.py report, {time.strftime('%Y-%m-%d %H:%M')})",
             f"Bootstrap: {n_boot} draws, rng seed {seed}. Failure classes (precedence): {', '.join(FAILURE_CLASSES)}.",
             MERGED_DOC, ""]
    missing = []
    found = discover_runs(results_dir)
    lines.append("== Runs ==")
    old = [r for r in found if r.meta.get("protocol") in (None, "v1")]
    for run in old:  # protocol v1: no classifier columns; listed only (no reference needed)
        k, n = sum(metric_value(row, "success") for row in run.rows), len(run.rows)
        lo, hi = wilson(k, n)
        lines.append(f"{run.name} ({run.meta.get('protocol')}, {n} episodes): success {int(k)}/{n} = {pct(k / max(n, 1))}% "
                     f"[{pct(lo)}, {pct(hi)}] (protocol v1 run: not classified)")
    runs = _prepare_all([r for r in found if r not in old], out_dir, data_dir, lines)
    for run in runs:
        lines += class_summary_lines(run)
        for w in run.meta.get("warnings", []):
            lines.append(f"  WARNING: {w}")
    if not found:
        lines.append("  no run folders found")
    lines.append("")

    # references
    lines.append("== Episode references (cross-check with results/data_analysis) ==")
    for set_name in sorted({r.meta["set"] for r in runs if r.meta["set"]}):
        ref = load_reference(set_name, out_dir, data_dir)
        lines += cross_check_reference(set_name, ref, data_dir=data_dir)[0]
    if not runs:
        lines.append("  (no v1.1 run: no reference needed)")
    lines.append("")

    # validation
    expert = [r for r in runs if r.meta["kind"] == "expert" and r.meta.get("protocol") == "v1.1"]
    zero = [r for r in runs if r.meta["kind"] == "zero" and r.meta.get("protocol") == "v1.1"]
    if expert or zero:
        v_lines, _ = validate_classifier(expert, zero)
        lines += v_lines
    else:
        lines.append("== Classifier validation: missing (results/expert_v11_<set>, results/zero_v11_dev) ==")
        missing.append("classifier validation (expert_v11_*, zero_v11_dev)")
    for set_name in ("dev", "hard", "confirm"):
        if not any(r.meta["set"] == set_name for r in expert):
            missing.append(f"results/expert_v11_{set_name}")
    lines.append("")

    # experiment A (a refusal of the integrity checks, or a CONFIRM run without --allow-confirm, is listed, not fatal)
    for set_name in ("dev", "hard", "confirm"):
        sa, sb = _find_seeds(results_dir, a_model, set_name), _find_seeds(results_dir, b_model, set_name)
        if not sa or not sb:
            if set_name in ("dev", "hard"):
                missing.append(f"experiment A on {set_name} ({a_model} seeds {sa}, {b_model} seeds {sb})")
            continue
        try:
            ra, _ = arm_runs(results_dir, a_model, sa, set_name, out_dir, data_dir)
            rb, _ = arm_runs(results_dir, b_model, sb, set_name, out_dir, data_dir)
            a_lines, a_res = compare_arms(ra, rb, f"{a_model} {set_name}", f"{b_model} {set_name}", predicted, n_boot,
                                          seed, expert_run=expert_run_for(results_dir, set_name, out_dir, data_dir))
        except SystemExit as e:
            lines += [f"== Experiment A on {set_name}: not analysed ==", str(e), ""]
            missing.append(f"experiment A on {set_name} (refused, see above)")
            continue
        lines += a_lines + [""]
        write_outputs(out_dir, f"A_{a_model}_vs_{b_model}_{set_name}", a_lines, a_res)
    # experiment B
    split_res = None
    for model in (a_model, b_model):
        for set_name in ("dev",):
            seeds = _find_seeds(results_dir, model, set_name, prefix="B-EM_")
            if not seeds:
                continue
            try:
                b_lines, res, _ = run_split(results_dir, model, seeds, set_name, out_dir, data_dir, n_boot, seed)
            except SystemExit as e:
                lines += [f"== Experiment B {model} {set_name}: not analysed ==", str(e), ""]
                continue
            lines += b_lines + [""]
            if res is not None:
                split_res = res
                write_outputs(out_dir, f"B_{model}_{set_name}", b_lines, res)
    if split_res is None:
        missing.append("experiment B (results/B-EM_*/B-ME_* with matching MM runs)")
    # descriptive
    try:
        d_lines, desc = describe(runs, trained_dir, n_boot, seed)
    except SystemExit as e:
        d_lines, desc = ["== Descriptive analyses: not analysed ==", str(e)], None
        missing.append("descriptive part (refused, see above)")
    lines += d_lines + [""]
    if not group_runs(runs):
        missing.append("model runs (results/<model>_s<seed>_<set>) for the descriptive part")
    # figures
    if with_figures:
        lines.append("== Figures ==")
        lines += figures(runs, out_dir, desc, split_res)
        lines.append("")
    lines.append("== Missing parts ==")
    lines += [f"  {m}" for m in missing] or ["  none"]
    write_outputs(out_dir, "summary", lines)
    return lines


# ==============================================================================
# CLI
# ==============================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=RESULTS_DIR, help="folder with the run folders")
    parser.add_argument("--out", default=STEP3_DIR, help="output folder (and reference cache)")
    parser.add_argument("--data", default=DATA_DIR, help="folder with the evaluation sets (references)")
    parser.add_argument("--trained", default=TRAINED_DIR, help="folder with the trained models and offline metrics")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    parser.add_argument("--allow-confirm", action="store_true",
                        help="allow reading the recorded data/confirm episodes (references of CONFIRM runs); without "
                             "it, validate skips CONFIRM runs")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="analyse paired runs that fail the integrity checks (other set, incomplete episode set, "
                             "harness_error, hybrid without --no-early-stop) with warnings instead of refusing")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refs", help="build the per-episode references of sets and cross-check them")
    p.add_argument("--sets", nargs="+", default=["dev", "hard"])
    p.add_argument("--rebuild", action="store_true")
    p = sub.add_parser("classify", help="failure classes of run folders (writes classes_<run>.csv)")
    p.add_argument("runs", nargs="+")
    p = sub.add_parser("validate", help="classifier validation on the v1.1 expert and zero-policy runs")
    p.add_argument("--expert-runs", nargs="*", default=None)
    p.add_argument("--zero-runs", nargs="*", default=None)
    p = sub.add_parser("compare", help="experiment A: arm B vs arm A on the same episodes")
    p.add_argument("--a", default="ego")
    p.add_argument("--b", default="ego-novw")
    p.add_argument("--set", default="dev")
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--a-seeds", default=None)
    p.add_argument("--b-seeds", default=None)
    p.add_argument("--a-runs", nargs="*", default=None, help="explicit run folders of arm A (instead of --a/--seeds)")
    p.add_argument("--b-runs", nargs="*", default=None)
    p.add_argument("--predicted-classes", default="", help="comma-separated failure classes predicted to improve")
    p = sub.add_parser("split2x2", help="experiment B: base / arm 2x2 split")
    p.add_argument("--model", default="ego")
    p.add_argument("--set", default="dev")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--ee-run", default=None, help="EE run folder (default results/B-EE_<set>, else expert_v11_<set>)")
    p = sub.add_parser("describe", help="descriptive analyses of all model runs")
    p.add_argument("--sets", nargs="+", default=["dev", "hard"])
    p = sub.add_parser("figures", help="all step-3 figures the runs allow")
    p.add_argument("--set", default="dev")
    p = sub.add_parser("report", help="results/step3/summary.txt")
    p.add_argument("--a", default="ego")
    p.add_argument("--b", default="ego-novw")
    p.add_argument("--predicted-classes", default="no_start,left_or_overshot")
    p.add_argument("--no-figures", action="store_true")
    p = sub.add_parser("precheck-recorded", help="development check of the classifier on the recorded expert episodes")
    p.add_argument("--sets", nargs="+", default=["dev", "hard"])
    args = parser.parse_args(argv)
    out, data = args.out, args.data
    global ALLOW_CONFIRM, ALLOW_INCOMPLETE
    ALLOW_CONFIRM = args.allow_confirm
    ALLOW_INCOMPLETE = args.allow_incomplete

    if args.cmd == "refs":
        lines = ["== Episode references =="]
        for s in args.sets:
            check_set_allowed(s)
            start = time.time()
            ref = load_reference(s, out, data, rebuild=args.rebuild)
            lines.append(f"{s}: {len(ref)} episodes ({time.time() - start:.1f} s), cache {os.path.join(out, f'cache_ref_{s}.csv')}")
            lines += cross_check_reference(s, ref, data_dir=data)[0]
        print("\n".join(lines))
        write_outputs(out, "refs_check", lines)
    elif args.cmd == "classify":
        for path in args.runs:
            run = load_prepared(path, out, data)
            print("\n".join(class_summary_lines(run)))
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"classes_{run.name}.csv"), "w", newline="") as f:
                cols = ["episode", "success", "stop_reason", "fclass", "mclass", "start_on_ring", "start_near_ring",
                        "no_motion", "no_motion_near_ring", "short_reach", "d0", "theta_deg", "ref_mode"]
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(run.rows)
    elif args.cmd == "validate":
        res = args.results
        expert = args.expert_runs if args.expert_runs is not None else [
            os.path.join(res, f"expert_v11_{s}") for s in ("dev", "hard", "confirm")]
        zero = args.zero_runs if args.zero_runs is not None else [os.path.join(res, "zero_v11_dev")]
        # CONFIRM runs (folder name, else config.json) only with --allow-confirm: skipped with a note, not an abort
        skipped = [] if ALLOW_CONFIRM else [p for p in expert + zero if is_confirm_name(p)]
        missing = [p for p in expert + zero if p not in skipped and not os.path.exists(os.path.join(p, "rollouts.csv"))]
        for p in missing:
            print(f"WARNING: missing run {p}")
        er, zr = [], []
        for paths, runs_out in ((expert, er), (zero, zr)):
            for p in paths:
                if p in skipped or p in missing:
                    continue
                run = load_run(p)
                if run.meta["set"] == "confirm" and not ALLOW_CONFIRM:
                    skipped.append(p)
                    continue
                runs_out.append(prepare(run, out, data))
        for p in skipped:
            print(f"NOTE: skipped {p}: a CONFIRM run (pass --allow-confirm to include it)")
        lines, checks = validate_classifier(er, zr)
        if skipped:
            lines.append(f"skipped CONFIRM runs (no --allow-confirm): {skipped}")
        if missing:
            lines.append(f"missing runs: {missing}")
        print("\n".join(lines))
        write_outputs(out, "validate", lines, {"checks": checks, "missing": missing, "skipped_confirm": skipped})
    elif args.cmd == "compare":
        a_seeds = parse_seeds(args.a_seeds if args.a_seeds is not None else args.seeds)
        b_seeds = parse_seeds(args.b_seeds if args.b_seeds is not None else args.seeds)
        ra, ma = arm_runs(args.results, args.a, a_seeds, args.set, out, data, args.a_runs)
        rb, mb = arm_runs(args.results, args.b, b_seeds, args.set, out, data, args.b_runs)
        predicted = [c for c in args.predicted_classes.split(",") if c]
        label_a = args.a if not args.a_runs else "A"
        label_b = args.b if not args.b_runs else "B"
        set_name = ra[0].meta["set"] if ra else args.set
        lines, res = compare_arms(ra, rb, label_a, label_b, predicted, args.n_boot, args.boot_seed,
                                  expert_run=expert_run_for(args.results, set_name, out, data))
        if ma or mb:
            lines.append(f"missing run folders: {ma + mb}")
        print("\n".join(lines))
        write_outputs(out, f"A_{label_a}_vs_{label_b}_{args.set}", lines, res)
    elif args.cmd == "split2x2":
        lines, res, _ = run_split(args.results, args.model, parse_seeds(args.seeds), args.set, out, data, args.n_boot,
                                  args.boot_seed, args.ee_run)
        print("\n".join(lines))
        write_outputs(out, f"B_{args.model}_{args.set}", lines, res)
    elif args.cmd == "describe":
        lines = []
        runs = [r for r in discover_runs(args.results) if r.meta["set"] in args.sets
                and r.meta.get("protocol") not in (None, "v1")]
        runs = _prepare_all(runs, out, data, lines)
        d_lines, res = describe(runs, args.trained, args.n_boot, args.boot_seed)
        lines += d_lines
        print("\n".join(lines))
        write_outputs(out, "describe", lines, res)
    elif args.cmd == "figures":
        lines = []
        runs = _prepare_all([r for r in discover_runs(args.results) if r.meta.get("protocol") not in (None, "v1")],
                            out, data, lines)
        _, desc = describe(runs, args.trained, args.n_boot, args.boot_seed)
        split_res = None
        for model in ("ego", "ego-novw"):
            seeds = _find_seeds(args.results, model, args.set, prefix="B-EM_")
            if seeds:
                _, r, _ = run_split(args.results, model, seeds, args.set, out, data, args.n_boot, args.boot_seed)
                split_res = r or split_res
        lines += figures(runs, out, desc, split_res, args.set)
        print("\n".join(lines))
    elif args.cmd == "report":
        lines = write_report(args.results, out, data, args.trained, args.n_boot, args.boot_seed, args.a, args.b,
                             [c for c in args.predicted_classes.split(",") if c], not args.no_figures)
        print("\n".join(lines))
    elif args.cmd == "precheck-recorded":
        lines = []
        for s in args.sets:
            l_, _, _ = precheck_recorded(s, args.results, out, data)
            lines += l_ + [""]
        print("\n".join(lines))
        write_outputs(out, "precheck_recorded", lines)


if __name__ == "__main__":
    main()
