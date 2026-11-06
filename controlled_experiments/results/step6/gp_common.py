"""Integrity, shared C3 controls and frozen statistics for step 6.

No directory discovery is used: only explicitly named runs are opened.  In
particular this module never opens data/test or a sealed evaluation set.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent
SOURCE_HERE, SOURCE_TASK = HERE, TASK
sys.path.insert(0, str(TASK))
import analyze_rollouts as ar  # noqa: E402

OLD = "ego-novw-trig"
MODELS = {"G": "ego-novw-trig-chunk", "P": "ego-novw-trig-pd"}
SEEDS = {"dev": list(range(10)), "hard": list(range(5)), "confirm": list(range(100, 105))}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ar._clean(obj), indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


POST_HOC_NOTE = ("post-hoc re-run: the frozen plan is not in this copy (the code comments and report.txt were "
                 "translated after the experiment), so the freeze hashes are not checked; the frozen state is "
                 "branch feature/experimental-line-final (commit 71ea436)")


def post_hoc():
    """True in the cleaned copy of the experiments, where the frozen plan is absent."""
    return not (SOURCE_TASK / "STEP6_PLAN_PROMPT.md").exists()


def check_freeze():
    j = json.loads((SOURCE_HERE / "prereg_GP_frozen.json").read_text())
    if post_hoc():
        print("G/P " + POST_HOC_NOTE, flush=True)
        return j
    if sha(SOURCE_TASK / "STEP6_PLAN_PROMPT.md") != j["plan_sha256"]:
        raise RuntimeError("Step 6 frozen plan has changed")
    if sha(SOURCE_HERE / "prereg_GP_frozen.txt") != j["frozen_sha256"]:
        raise RuntimeError("Step 6 frozen record has changed")
    frozen = (SOURCE_HERE / "prereg_GP_frozen.txt").read_text()
    section = frozen[frozen.index(j["report_header"]):]
    report = (SOURCE_TASK / "report.txt").read_text()
    start = report.index(j["report_header"])
    if (hashlib.sha256(section.encode()).hexdigest() != j["section_sha256"]
            or report[start:start + len(section)] != section):
        raise RuntimeError("Step 6 frozen section in report.txt has changed")
    print("G/P frozen sha256: PASS", flush=True)
    return j


def configure_smoke_root(root):
    """Keep synthetic runs, references and outputs under the smoke boundary."""
    global TASK, HERE
    root = Path(root).resolve()
    if not root.is_relative_to(SOURCE_HERE / "smoke") or root == SOURCE_HERE / "smoke":
        raise ValueError("--smoke-root must be a child of results/step6/smoke")
    TASK, HERE = root, root / "results" / "step6"
    HERE.mkdir(parents=True, exist_ok=True)
    return TASK, HERE


def raw_runs(model, set_name, seeds):
    runs = [ar.load_run(str(TASK / "results" / f"{model}_s{s}_{set_name}")) for s in seeds]
    if any(r.meta.get("warnings") for r in runs):
        raise RuntimeError("Run path and configuration mismatch")
    return runs


def merged_old(runs):
    """Exactly D/F controls: replace only stalled early-stop rows."""
    result, notes = [], []
    for run in runs:
        stalled = sorted(r["episode"] for r in run.rows if r.get("stop_reason") == "stalled")
        if run.meta.get("no_early_stop") is True or not stalled:
            result.append(run)
            notes.append(f"{run.name}: original rows; no stalled replacement needed")
            continue
        candidates = [TASK / "results" / d / "nostall" / run.name for d in ("step3", "step5", "step6")]
        path = next((p for p in candidates if (p / "summary.txt").exists()), None)
        if path is None:
            raise RuntimeError(f"Missing stalled rerun: {run.name}")
        rr = ar.load_run(str(path))
        if (rr.meta.get("no_early_stop") is not True or sorted(rr.by_episode) != stalled
                or any(r.get("stop_reason") in ("policy_error", "harness_error") for r in rr.rows)):
            raise RuntimeError(f"Invalid stalled rerun: {path}")
        rows = [rr.by_episode.get(r["episode"], r) for r in run.rows]
        result.append(ar.Run(run.path, rows, run.header, run.config, dict(run.meta)))
        notes.append(f"{run.name}: replaced {len(stalled)} stalled rows from {path.relative_to(TASK)}")
    return result, notes


def prepare(runs):
    for run in runs:
        # Refuse implicit reference creation (which would open recorded data).
        if not (TASK / "results" / "step3" / f"cache_ref_{run.meta['set']}.csv").exists():
            raise RuntimeError("Missing frozen episode reference; implicit rebuild is not allowed")
        ar.prepare(run, out_dir=str(TASK / "results" / "step3"), data_dir=str(TASK / "data"))
    return runs


def difference(A, B, level=.95, boot=None):
    """F's paired_diff draws, with the preregistered CI level explicitly set."""
    d = ar.paired_diff(A, B, boot=boot)
    lo, hi = ar.percentile_ci(d["draws"], level=level)
    t = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B), level=level)
    return {"a": d["mean_a"], "b": d["mean_b"], "diff": d["diff"],
            "lo": lo, "hi": hi, "p_boot": d["p"], "level": level,
            "welch_lo": t["lo"], "welch_hi": t["hi"], "p_welch": t["p"],
            "per_seed_a": ar.seed_rates(A), "per_seed_b": ar.seed_rates(B)}


def dev_pass(j):
    d = j.get("dev") or {}
    return bool(d.get("condition_1") and d.get("condition_2"))


def hard_pass(j):
    return dev_pass(j) and bool((j.get("hard") or {}).get("condition_3"))


def verdict(j, experiment):
    if not j.get("dev"):
        return "PENDING DEV"
    if not dev_pass(j):
        return "KEEP C3"
    if not j.get("hard"):
        return "NEXT HARD"
    if not hard_pass(j):
        return "KEEP C3"
    if not j.get("confirm"):
        return "NEXT CONFIRM"
    return f"ADOPT {experiment}" if j["confirm"]["condition_4"] else "KEEP C3"


def pilot_choice(rates, baseline):
    """5 pp is inclusive; 10 pp stop threshold is strict (G5)."""
    means = {int(k): float(np.mean(v)) for k, v in rates.items()}
    if set(means) != {9, 27, 81}:
        raise ValueError("G5 requires deltas 9, 27 and 81")
    best = max(means.values())
    chosen = min(k for k, v in means.items() if best - v <= .05 + 1e-12)
    baseline = float(np.mean(baseline))
    return {"delta_star": chosen, "means": means, "c3_mean": baseline,
            "selected_mean": means[chosen], "continue": bool(means[chosen] >= baseline - .10 - 1e-12),
            "rule": "highest 2-seed mean; smallest delta within inclusive 5 pp; stop strictly below C3 minus 10 pp"}


def joint_choice(g_adopted, p_adopted, g_minus_p=None):
    if g_adopted and p_adopted:
        if g_minus_p is None:
            raise ValueError("both adopted requires DEV paired 95% G minus P")
        if g_minus_p["lo"] > 0:
            return "G"
        if g_minus_p["hi"] < 0:
            return "P"
        return "G"
    return "G" if g_adopted else "P" if p_adopted else "C3"
