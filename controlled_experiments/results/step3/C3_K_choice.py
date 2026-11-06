"""Experiment C3, choice of K by the amended rule (report.txt "[Methodology] Step 3 continued, pre-registration
amendment 1", a post-hoc change written after the pilot diagnostic of K 64 / K 256 and before K 16's pilot run;
confirmed by the user 2026-10-30):
  K = the smallest K of {16, 64, 256} whose seed-0 calibration model
      (a) passes the offline gate (ii) of C3_probe.py (results/step3/C3_probe.json), and
      (b) extends the arm toward the target in > 30% of the data/pilot episodes in the EM cell (expert base, model arm,
          --no-early-stop; results/step3/C3_pilot/B-EM_ego-novw-trig-k<K>_s0_pilot; arm_extended_toward point estimate).
All three K must have both results. Gate (i) (FROZEN r >= 0.5) is dropped.

  python results/step3/C3_K_choice.py   -> results/step3/C3_K_choice.json / .txt
"""
import json
import os
import sys

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK)
import analyze_rollouts as ar  # noqa: E402

K_GRID = (16, 64, 256)
PILOT_MIN = 0.30


def main():
    probe = json.load(open(os.path.join(ar.STEP3_DIR, "C3_probe.json")))
    lines, rows, complete = ["C3 K choice (amended rule 1): smallest K with offline gate (ii) and pilot EM "
                             f"arm_extended_toward > {PILOT_MIN:.0%}"], {}, True
    for k in K_GRID:
        m = probe["models"].get(f"ego-novw-trig-k{k}_s0")
        p = os.path.join(ar.STEP3_DIR, "C3_pilot", f"B-EM_ego-novw-trig-k{k}_s0_pilot")
        if m is None or not os.path.exists(os.path.join(p, "rollouts.csv")):
            complete = False
            lines.append(f"  K {k}: missing {'probe' if m is None else os.path.relpath(p, TASK)}")
            continue
        run = ar.load_prepared(p)
        if run.meta.get("warnings") or ar.integrity_problems([run]):
            raise SystemExit(f"{p}: {run.meta.get('warnings')} {ar.integrity_problems([run])}")
        y = ar.metric_matrix([run], [r["episode"] for r in run.rows], "arm_extended_toward")[0]
        share = float(y.mean())
        ok = bool(m["gate_ii"] and share > PILOT_MIN)
        rows[k] = {"gate_ii": m["gate_ii"], "pilot_arm_extended_toward": share, "n": int(y.size), "passes": ok}
        lines.append(f"  K {k}: gate (ii) {'PASS' if m['gate_ii'] else 'FAIL'}; pilot EM arm_extended_toward "
                     f"{int(y.sum())}/{y.size} = {100 * share:.1f}% -> {'passes' if ok else 'fails'}")
    passing = sorted(k for k, r in rows.items() if r["passes"])
    chosen = passing[0] if complete and passing else None
    lines.append(f"  -> chosen K = {chosen}" if chosen is not None else
                 ("  -> no K chosen: grid incomplete" if not complete else "  -> no K passes: stop and propose"))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ar.STEP3_DIR, "C3_K_choice.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(ar.STEP3_DIR, "C3_K_choice.json"), "w") as f:
        json.dump({"rule": "amended rule 1", "grid": list(K_GRID), "complete": complete, "per_k": rows,
                   "passing": passing, "chosen_k": chosen}, f, indent=1)


if __name__ == "__main__":
    main()
