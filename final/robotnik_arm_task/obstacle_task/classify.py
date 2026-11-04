"""Failure classes of protocol v2.0 runs (obstacle task, plan section 4).

One class per episode of a run folder written by obstacle_task.evaluate. A success (the harness's success: a contact
inside a hold window does not undo it) is "success"; a harness_error is flagged, not classified (as step 3). A failed
episode gets the first matching class of
  crash      stop_reason tipped / invalid_action / policy_error (analyze_rollouts.CRASH_REASONS)
  collision  a robot-obstacle contact ended the run (column collision)
  no_pass    the base never passed the obstacle (passed_obstacle False), no contact
  then the A-to-B classes of analyze_rollouts.classify_row, unchanged, for the episodes that passed: no_start,
  runaway, no_progress, never_other, circling, left_or_overshot, parked_no_arm, extended_not_reached, approached_other.
The A-to-B predicates need three per-episode references, computed here from the set's recorded expert episodes (the
A-to-B caches are not used): navdone_ee_err (EE error at the expert's first nav_done step: its hold posture at the
dock), align_req_deg (the expert's in-place turns at the start and at the waypoint) and orient_req_deg (its turn at
the dock; counted only once the base has parked, as in v1.1).

Validation on an expert run (--validate-expert, like step 3): no harness_error; on every expert success no failure
predicate fires (the success is "unclassified"); every expert failure is listed with its class and evidence; the
arm never extends before the recorded nav_done. A failed validation raises.

CLI: python -m obstacle_task.classify RUN_DIR [--eval-set SET_DIR] [--validate-expert] writes RUN_DIR/classes.csv
(rollouts.csv + fclass + the references; validation: RUN_DIR/validate_expert.txt) and prints the class counts.
The set defaults to the run's config.json args.eval_set. data/test is refused, test2 needs --allow-test2.
"""
import argparse
from collections import Counter
import csv
import json
import os
import sys

import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import analyze_rollouts as baseline  # noqa: E402
from obstacle_task.collect import check_sealed  # noqa: E402
from obstacle_task.scenario import Scenario  # noqa: E402

FAILURE_CLASSES = ("crash", "collision", "no_pass", *baseline.FAILURE_CLASSES[1:])   # = the precedence
CLASSES = ("success", *FAILURE_CLASSES, "harness_error")
REFERENCE_COLUMNS = ("navdone_ee_err", "navdone_time", "align_req_deg", "orient_req_deg")
PREDICATES = ("crash", "collision", "no_pass", "never_approached", "no_start", "runaway", "no_progress", "circling",
              "left_or_overshot", "parked_no_arm", "extended_not_reached")


def _true(value):
    """True for the harness's bools and for typed or raw CSV values ('True', 1.0)."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "1.0")
    return bool(value == 1) if value is not None else False


def failure_predicates(row, ref):
    """Every failure predicate on one row (also on successes, for the validation): crash, collision, no_pass and
    analyze_rollouts.failure_predicates (needs ref: navdone_ee_err, align_req_deg, orient_req_deg)."""
    return {"crash": row.get("stop_reason") in baseline.CRASH_REASONS,
            "collision": _true(row.get("collision")) or row.get("stop_reason") == "collision",
            "no_pass": not _true(row.get("passed_obstacle")),
            **baseline.failure_predicates(row, ref or {})}


def classify_row(row, ref=None):
    """'success', a failure class (first match of FAILURE_CLASSES) or 'harness_error'; row is not modified."""
    if _true(row.get("success")):
        return "success"
    reason = row.get("stop_reason")
    if reason in baseline.RUN_ERROR_REASONS:
        return "harness_error"
    if reason in baseline.CRASH_REASONS:
        return "crash"
    if _true(row.get("collision")) or reason == "collision":
        return "collision"
    if not _true(row.get("passed_obstacle")):
        return "no_pass"
    return baseline.classify_row(row, ref or {})


def _angle(a, b):
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def required_turns(scenario):
    """The expert's nominal in-place turns (deg): (align = at the start + at the waypoint, orient = at the dock)."""
    sc = Scenario.from_dict(scenario) if isinstance(scenario, dict) else scenario
    init, dock = np.asarray(sc.init_xy), np.asarray(sc.dock_xy)
    points = [init, *sc.route(), dock]
    bearings = [float(np.arctan2(b[1] - a[1], b[0] - a[0])) for a, b in zip(points, points[1:])]
    align = _angle(bearings[0], sc.init_yaw) + sum(_angle(b, a) for a, b in zip(bearings, bearings[1:]))
    orient = _angle(sc.dock_bearing, bearings[-1])
    return float(np.rad2deg(align)), float(np.rad2deg(orient))


def episode_reference(path):
    """References of one recorded expert episode: navdone_ee_err, navdone_time and the required turns."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        done = np.flatnonzero(data["nav_done"])
        error = (float(np.linalg.norm(data["ee_pos"][done[0]].astype(float) - data["target"].astype(float)))
                 if len(done) else np.nan)
        navdone_time = float(data["t"][done[0]]) if len(done) and "t" in data else np.nan
    align, orient = required_turns(meta["scenario"])
    return {"navdone_ee_err": error, "navdone_time": navdone_time, "align_req_deg": align, "orient_req_deg": orient}


def load_reference(eval_dir, episodes=None, allow_test2=False):
    """{episode: reference} from the recorded episodes of an obstacle set (selected episodes only, if given)."""
    check_sealed(eval_dir, allow_test2)
    selected = None if episodes is None else set(episodes)
    with open(os.path.join(eval_dir, "index.csv"), newline="") as stream:
        index = list(csv.DictReader(stream))
    refs = {}
    for item in index:
        episode = int(item["episode"])
        if (selected is not None and episode not in selected) or not item.get("file"):
            continue
        refs[episode] = episode_reference(os.path.join(eval_dir, item["file"]))
    if selected is not None and selected - refs.keys():
        raise ValueError(f"episodes without a recorded reference: {sorted(selected - refs.keys())}")
    return refs


def _f(row, key):
    return baseline._f(row, key)


def validate_expert(rows):
    """Classifier validation on an expert run (rows classified, references joined). Returns (lines, passed)."""
    lines, passed = [], True
    errors = [row["episode"] for row in rows if row["fclass"] == "harness_error"]
    passed &= not errors
    lines.append(f"[{'PASS' if not errors else 'FAIL'}] no harness_error: {errors or 'none'}")
    successes = [row for row in rows if row["fclass"] == "success"]
    fired = {name: [] for name in PREDICATES}
    for row in successes:
        for name, value in failure_predicates(row, row).items():
            if value and name in fired:
                fired[name].append(row["episode"])
    unexpected = {name: eps for name, eps in fired.items() if eps}
    passed &= not unexpected
    lines.append(f"[{'PASS' if not unexpected else 'FAIL'}] {len(successes)} successes: no failure predicate fires "
                 f"(fired: {unexpected or 'none'})")
    failures = [row for row in rows if row["fclass"] in FAILURE_CLASSES]
    lines.append(f"[INFO] {len(failures)} failures, classes {dict(Counter(row['fclass'] for row in failures))}")
    for row in failures:
        excess = _f(row, "base_turn_total_deg") - baseline.required_turn_deg(row, row)
        lines.append(f"    ep {row['episode']}: {row['fclass']}, stop {row.get('stop_reason')}, collision "
                     f"{row.get('collision')} {row.get('collision_link') or ''} t {_f(row, 'collision_time'):.3f}, "
                     f"passed {row.get('passed_obstacle')}, min_error {_f(row, 'min_error'):.4f}, ring min "
                     f"{_f(row, 'ring_dist_min'):.3f} final {_f(row, 'ring_dist_final'):.3f}, turn excess "
                     f"{excess:.0f} deg, min_clearance {_f(row, 'min_clearance'):.3f}")
    lag = np.array([_f(row, "arm_extend_time") - _f(row, "navdone_time") for row in rows])
    ok = np.isfinite(lag)
    early = int(np.sum(lag[ok] < 0))
    passed &= early == 0
    lines.append(f"[{'PASS' if early == 0 else 'FAIL'}] arm_extend_time >= recorded nav_done time: {early} of "
                 f"{int(ok.sum())} extend earlier"
                 + (f"; lag median {np.median(lag[ok]):.3f} s, max {np.max(lag[ok]):.3f} s" if ok.any() else ""))
    return lines, bool(passed)


def classify_run(run_dir, eval_dir=None, out_path=None, validate=False, allow_test2=False):
    """Writes classes.csv (and validate_expert.txt); returns (classified rows, class counts, validation lines)."""
    check_sealed(run_dir, allow_test2)
    if eval_dir is None:
        with open(os.path.join(run_dir, "config.json")) as stream:
            eval_dir = json.load(stream).get("args", {}).get("eval_set")
        if not eval_dir:
            raise ValueError("config.json has no args.eval_set; supply --eval-set")
    if not os.path.isabs(eval_dir) and not os.path.isdir(eval_dir):
        eval_dir = os.path.join(TASK_DIR, eval_dir)
    check_sealed(eval_dir, allow_test2)   # before any result of the run is read
    rows, header = baseline.read_rows(os.path.join(run_dir, "rollouts.csv"))
    refs = load_reference(eval_dir, [int(row["episode"]) for row in rows], allow_test2)
    for row in rows:
        row.update(refs[int(row["episode"])])
        row["fclass"] = classify_row(row, row)
    counts = Counter(row["fclass"] for row in rows)
    lines = []
    if validate:
        lines, passed = validate_expert(rows)
        with open(os.path.join(run_dir, "validate_expert.txt"), "w") as stream:
            stream.write("\n".join([f"Classifier validation of {run_dir} (eval set {eval_dir})", *lines]) + "\n")
    out_path = out_path or os.path.join(run_dir, "classes.csv")
    fields = list(dict.fromkeys([*header, "fclass", *REFERENCE_COLUMNS]))
    with open(out_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if validate and not passed:
        raise ValueError("expert validation failed:\n" + "\n".join(lines))
    return rows, dict(counts), lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="run folder of obstacle_task.evaluate (rollouts.csv, config.json)")
    parser.add_argument("--eval-set", help="its evaluation set (default: config.json args.eval_set)")
    parser.add_argument("--out", help="output CSV (default RUN_DIR/classes.csv)")
    parser.add_argument("--validate-expert", action="store_true", help="validate the classifier on an expert run")
    parser.add_argument("--allow-test2", action="store_true", help="allow the sealed test2 set")
    args = parser.parse_args()
    try:
        rows, counts, lines = classify_run(args.run_dir, args.eval_set, args.out, args.validate_expert,
                                           args.allow_test2)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"{args.run_dir}: {len(rows)} episodes")
    for name in CLASSES:
        if counts.get(name):
            print(f"  {name}: {counts[name]}")
    for line in lines:
        print(line)
    print(f"Wrote {args.out or os.path.join(args.run_dir, 'classes.csv')}")


if __name__ == "__main__":
    main()
