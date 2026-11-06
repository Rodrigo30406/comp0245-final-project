"""Shared integrity checks and existing-run assembly for experiment F."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent
sys.path.insert(0, str(TASK))
import analyze_rollouts as ar  # noqa: E402

OLD = "ego-novw-trig"
NEW = "ego-novw-trig-dag1"
SEEDS = {"dev": list(range(10)), "hard": list(range(5)), "confirm": list(range(100, 105))}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


POST_HOC_NOTE = ("post-hoc re-run: the frozen plan is not in this copy (the code comments and report.txt were "
                 "translated after the experiment), so the freeze hashes are not checked; the frozen state is "
                 "branch feature/experimental-line-final (commit 71ea436)")


def check_freeze():
    """Check the plan and the exact frozen report section before every stage."""
    j = json.loads((HERE / "prereg_F_hashes.json").read_text())
    if not (TASK / "STEP5_DAGGER_PLAN_PROMPT.md").exists():
        print("F " + POST_HOC_NOTE, flush=True)
        return
    plan = (TASK / "STEP5_DAGGER_PLAN_PROMPT.md").read_bytes()
    amendment_path = HERE / "amendment2_baseline.json"
    expected_plan = j["plan_sha256"]
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        if amendment["previous_plan_sha256"] != expected_plan:
            raise RuntimeError("amendment 2 does not extend the frozen plan")
        paragraph = amendment["amendment2_text"].encode()
        if (plan.count(paragraph) != 1 or hashlib.sha256(plan.replace(paragraph, b"", 1)).hexdigest()
                != expected_plan):
            raise RuntimeError("plan changes exceed the approved descriptive amendment 2")
        expected_plan = amendment["plan_sha256"]
        for name, key in (("prereg_F_frozen.txt", "prereg_F_frozen_sha256"),
                          ("prereg_F_hashes.json", "prereg_F_hashes_sha256")):
            if sha(HERE / name) != amendment[key]:
                raise RuntimeError(f"original frozen record changed: {name}")
    if hashlib.sha256(plan).hexdigest() != expected_plan:
        raise RuntimeError("the frozen experiment F plan changed")
    report = (TASK / "report.txt").read_text()
    frozen = (HERE / "prereg_F_frozen.txt").read_text()
    frozen_section = frozen[frozen.index(j["report_header"]):]
    start = report.index(j["report_header"])
    section = report[start:start + len(frozen_section)]
    if hashlib.sha256(section.encode()).hexdigest() != j["section_sha256"]:
        raise RuntimeError("the frozen experiment F report section changed")
    implementation = HERE / "implementation_F_sha256.json"
    if implementation.exists():
        for name, digest in json.loads(implementation.read_text()).items():
            if sha(TASK / name) != digest:
                raise RuntimeError(f"experiment F implementation/input changed: {name}")
    print("F freeze hashes: PASS", flush=True)


def raw_runs(model, set_name, seeds):
    """Read existing result files only; never discover runs or regenerate references."""
    paths = [TASK / "results" / f"{model}_s{s}_{set_name}" for s in seeds]
    runs = [ar.load_run(str(p)) for p in paths]
    if any(r.meta.get("warnings") for r in runs):
        raise RuntimeError("run folder/config disagreement")
    return runs


def merged_old(runs):
    """Replace stalled rows using complete no-early-stop reruns, as in experiment D."""
    out, notes = [], []
    for run in runs:
        stalled = sorted(r["episode"] for r in run.rows if r.get("stop_reason") == "stalled")
        if run.meta.get("no_early_stop") is True or not stalled:
            out.append(run)
            notes.append(f"{run.name}: original rows; no stalled replacement needed")
            continue
        candidates = [TASK / "results" / "step3" / "nostall" / run.name, HERE / "nostall" / run.name]
        p = next((p for p in candidates if (p / "summary.txt").exists()), None)
        if p is None:
            raise RuntimeError(f"missing stalled rerun: {run.name}")
        rr = ar.load_run(str(p))
        if (rr.meta.get("no_early_stop") is not True or sorted(rr.by_episode) != stalled
                or any(r.get("stop_reason") in ("policy_error", "harness_error") for r in rr.rows)):
            raise RuntimeError(f"invalid stalled rerun: {p}")
        rows = [rr.by_episode.get(r["episode"], r) for r in run.rows]
        out.append(ar.Run(run.path, rows, run.header, run.config, dict(run.meta)))
        notes.append(f"{run.name}: replaced {len(stalled)} stalled rows from {p.relative_to(TASK)}")
    return out, notes


def prepare(runs):
    """Use the existing frozen episode references; refuse any implicit data access."""
    for run in runs:
        ref = TASK / "results" / "step3" / f"cache_ref_{run.meta['set']}.csv"
        if not ref.exists():
            raise RuntimeError(f"missing frozen reference: {ref}")
        ar.prepare(run)
    return runs


def clean(obj):
    return ar._clean(obj)


def write_json(path, obj):
    """Atomically write generated stage metadata, allowing deterministic resume."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
