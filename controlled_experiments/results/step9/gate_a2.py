"""Gate A2 (plan section 7): no-obstacle equivalence of the obstacle collector with collect_data.py.

Records the same 3 noisy A-to-B episodes (seed 909, random start poses, noise_frac 1.0) with collect_data.py and with
obstacle_task.collect --no-obstacle (zero obstacles, segment 0 skipped), each in fresh folders, and compares them:
every collect_data.LOG_FIELDS array and target bitwise (values and dtypes), segment == 1 everywhere, and every
collect_data meta key. Output and evidence.json under results/step9/smoke/a2 (rebuilt on every run, ~1 min).

Run from the task folder: python results/step9/gate_a2.py
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
from collect_data import LOG_FIELDS  # noqa: E402

OUT = TASK / "results/step9/smoke/a2"
COMMON = ["--seed", "909", "--n-episodes", "3", "--noise-frac", "1.0", "--workers", "3",
          "--init-radius", "2.0", "--init-yaw-range", "-180", "180"]


def main():
    for name in ("legacy", "obstacle"):
        shutil.rmtree(OUT / name, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    for name, entry in (("legacy", ["collect_data.py"]),
                        ("obstacle", ["-m", "obstacle_task.collect", "--set-name", "a2", "--no-obstacle"])):
        cmd = [sys.executable, *entry, *COMMON, "--out-dir", str(OUT / name)]
        with (OUT / f"{name}.log").open("w") as log:
            subprocess.run(cmd, cwd=TASK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=400)
    evidence = []
    for ep in range(3):
        with np.load(OUT / "legacy" / f"ep_{ep:04d}.npz") as old, np.load(OUT / "obstacle" / f"ep_{ep:04d}.npz") as new:
            mismatches = [key for key in (*LOG_FIELDS, "target")
                          if old[key].dtype != new[key].dtype or not np.array_equal(old[key], new[key])]
            old_meta, new_meta = json.loads(str(old["meta"])), json.loads(str(new["meta"]))
            meta_mismatches = [key for key in old_meta if new_meta.get(key) != old_meta[key]]
            evidence.append({"episode": ep, "steps": len(old["t"]), "status": old_meta["status"],
                             "noisy": old_meta["noise_v"] > 0, "mismatches": mismatches,
                             "meta_mismatches": meta_mismatches,
                             "segment_all_1": bool(np.all(new["segment"] == 1)),
                             "segment_dtype": str(new["segment"].dtype)})
    report = {"gate": "A2", "passed": all(not e["mismatches"] and not e["meta_mismatches"] and e["segment_all_1"]
                                          for e in evidence),
              "episodes": evidence, "fields": [*LOG_FIELDS, "target"], "command_args": COMMON}
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
