"""Freeze the step-9 pre-registration: insert results/step9/prereg_step9_draft.md into report.txt (before the
'[References]' section) and write results/step9/prereg_step9_frozen.txt with the sha256 of the inserted text,
of report.txt and of everything the read-out depends on (plan, code, trained models, split, data indexes).

Refuses to run if any learner run on dev2 / confirm2 / test2 exists, so the freeze always precedes the learner
evaluation. Run once: python results/step9/freeze_step9.py
"""
import datetime
import glob
import hashlib
import os
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEP9 = os.path.join(TASK_DIR, "results", "step9")
REPORT = os.path.join(TASK_DIR, "report.txt")
DRAFT = os.path.join(STEP9, "prereg_step9_draft.md")
RECORD = os.path.join(STEP9, "prereg_step9_frozen.txt")
ANCHOR = "\n[References] 参考文献"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    if os.path.exists(RECORD):
        sys.exit(f"already frozen: {RECORD}")
    learner_runs = [p for p in glob.glob(os.path.join(STEP9, "*_s*_*2")) if os.path.isdir(p)
                    and os.path.basename(p).split("_s")[0] in ("L0", "L1", "L1-K0", "L2")]
    if learner_runs:
        sys.exit(f"learner runs already exist, refusing to freeze after the fact: {learner_runs[:3]}")
    section = open(DRAFT, encoding="utf-8").read().rstrip("\n") + "\n"
    report = open(REPORT, encoding="utf-8").read()
    if report.count(ANCHOR) != 1:
        sys.exit("References-section anchor (see ANCHOR) not found exactly once in report.txt")
    if section.splitlines()[0] in report:
        sys.exit("the step-9 pre-registration is already in report.txt")
    before_sha = hashlib.sha256(report.encode("utf-8")).hexdigest()
    new = report.replace(ANCHOR, "\n" + section + ANCHOR, 1)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(new)
    deps = [os.path.join(TASK_DIR, "STEP9_PLAN_PROMPT.md"),
            *sorted(glob.glob(os.path.join(TASK_DIR, "obstacle_task", "*.py"))),
            *sorted(glob.glob(os.path.join(TASK_DIR, "tests", "test_obstacle_*.py"))),
            os.path.join(STEP9, "run_step9_eval.sh"), os.path.join(STEP9, "step9_analysis.py"),
            os.path.join(TASK_DIR, "trained", "step9", "data_split.json"),
            *sorted(glob.glob(os.path.join(TASK_DIR, "trained", "step9", "*.npz"))),
            *[os.path.join(TASK_DIR, "data", "obstacle", s, f) for s in ("train2", "dev2", "confirm2")
              for f in ("index.csv", "scenarios.json")]]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [f"step-9 pre-registration frozen at {now} (UTC), before any learner run on dev2 / confirm2 / test2",
             f"section sha256 {hashlib.sha256(section.encode('utf-8')).hexdigest()}",
             f"report.txt sha256 before {before_sha}",
             f"report.txt sha256 after  {sha(REPORT)}", "dependencies:"]
    lines += [f"  {sha(p)}  {os.path.relpath(p, TASK_DIR)}" for p in deps]
    with open(RECORD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:4]), f"\n{len(deps)} dependencies recorded in {RECORD}")


if __name__ == "__main__":
    main()
