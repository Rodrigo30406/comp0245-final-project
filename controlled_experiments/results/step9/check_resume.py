"""End-to-end check of the collector's resume and determinism (with the noise stream): results/step9/smoke/resume_check.

1. collect pilot2n episodes 0-1 into a fresh folder;
2. plant a truncated ep_0003.npz (a file killed mid-write by some other tool) and a stale ep_0002.npz.tmp;
3. --resume --episodes 0-3: episodes 0-1 must be kept (not rewritten), 2-3 run, index.csv has 4 sorted rows;
4. every array and the meta of the 4 episodes must equal results/step9/smoke/pilot2n (same seed, noise streams);
5. --resume with a changed dataset setting (--noise-v) must be refused, and a fresh run into the folder too.
Writes evidence.json there (~1 min). Run from the task folder: python results/step9/check_resume.py
"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
OUT = TASK / "results/step9/smoke/resume_check"
REFERENCE = TASK / "results/step9/smoke/pilot2n"
BASE = [sys.executable, "-m", "obstacle_task.collect", "--set-name", "pilot2n", "--workers", "4", "--out-dir", str(OUT)]


def run(extra, log_name, check=True):
    with open(OUT.parent / f"resume_check_{log_name}.log", "w") as log:
        return subprocess.run(BASE + extra, cwd=TASK, stdout=log, stderr=subprocess.STDOUT, check=check,
                              timeout=500).returncode


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    run(["--episodes", "0-1"], "1_fresh")
    stamps = {ep: os.stat(OUT / f"ep_{ep:04d}.npz").st_mtime_ns for ep in (0, 1)}
    (OUT / "ep_0003.npz").write_bytes(b"PK\x03\x04 truncated")
    (OUT / "ep_0002.npz.tmp").write_bytes(b"stale")
    run(["--episodes", "0-3", "--resume"], "2_resume")
    kept = all(os.stat(OUT / f"ep_{ep:04d}.npz").st_mtime_ns == stamps[ep] for ep in (0, 1))
    with open(OUT / "index.csv", newline="") as f:
        index = [int(r["episode"]) for r in csv.DictReader(f)]
    identical = {}
    for ep in range(4):
        with np.load(OUT / f"ep_{ep:04d}.npz") as a, np.load(REFERENCE / f"ep_{ep:04d}.npz") as b:
            identical[ep] = (sorted(a.files) == sorted(b.files)
                             and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a.files))
    refused_setting = run(["--episodes", "0-3", "--resume", "--noise-v", "0.03"], "3_changed_setting", check=False)
    refused_fresh = run(["--episodes", "0-3"], "4_fresh_into_full_folder", check=False)
    report = {"check": "collector resume + determinism", "kept_saved_episodes_unchanged": kept,
              "index_episodes": index, "identical_to_pilot2n": identical,
              "changed_setting_exit_code": refused_setting, "fresh_into_nonempty_exit_code": refused_fresh}
    report["passed"] = (kept and index == [0, 1, 2, 3] and all(identical.values()) and refused_setting != 0
                        and refused_fresh != 0)
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
