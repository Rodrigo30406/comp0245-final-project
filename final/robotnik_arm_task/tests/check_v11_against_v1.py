"""Check that a protocol v1.1 run reproduces the v1 columns of a protocol v1 run, episode by episode.

Usage:
  python tests/check_v11_against_v1.py <v1_run_dir> <v11_run_dir> [--float-tol 0] [--allow-subset] [--require-compare]
  e.g. python tests/check_v11_against_v1.py results/expert_dev results/expert_v11_dev --require-compare
       python tests/check_v11_against_v1.py results/expert_dev results/tmp_harness/expert_dev20 --allow-subset

Protocol v1.1 must leave every v1 column of rollouts.csv exactly as v1 computes it (it only adds columns and keeps
simulating after the 1 s hold window). Both runs write their rows with the same code path (csv.DictWriter of the
same Python values), so the values are compared as the strings written in the CSV: identical results give identical
strings. --float-tol > 0 additionally accepts two numbers that differ by at most that much (not needed when both
runs were written by evaluate_policy.py).

Checked:
  - the reference run is protocol v1 and the other run protocol v1.1 (config.json "protocol"["PROTOCOL_VERSION"]),
    so that a swapped pair or a run compared with itself cannot pass;
  - no episode appears twice in either rollouts.csv;
  - the v1.1 header starts with the v1 header (the v1 columns are kept, in their order);
  - every episode of the v1.1 run is in the v1 run, and every episode of the v1 run is in the v1.1 run (with
    --allow-subset the v1.1 run may cover only part of the v1 episodes, e.g. a --limit run);
  - every v1 column of every episode is identical, except the columns that are not deterministic or that v1.1
    legitimately changes: wall_time, policy_ms_mean / p99 / max (timing), cmp_steps (the common length with the
    recording can grow, since the v1.1 trajectory covers the extended hold). error_message is compared too, except
    that an empty v1 value may become "extended hold: <error>" (an exception after the v1 stop point);
  - cmp_identical is True in the v1.1 run for every episode where it is present; with --require-compare it must
    be True in every compared episode (an empty value, i.e. no recorded comparison, fails).
Prints "N of M v1 episodes compared", then PASS or FAIL with the differing episodes and columns; exit status 0 on
PASS, 1 on FAIL.
"""
import argparse
import csv
import json
import os
import sys

EXCLUDED = ("wall_time", "policy_ms_mean", "policy_ms_p99", "policy_ms_max", "cmp_steps")
EXTENDED_HOLD_PREFIX = "extended hold: "  # error_message of an exception in the v1.1 extended window
V1_VERSION, V11_VERSION = "v1", "v1.1"    # required PROTOCOL_VERSION of the reference run and of the checked run
MAX_LISTED = 60                           # differences listed one by one
MAX_IDS = 20                              # episode ids listed in one problem line


def read_rollouts(run_dir):
    """(header, {episode: row}, duplicated episodes) of <run_dir>/rollouts.csv, values as the strings written in the
    file. For a duplicated episode the first row is kept."""
    with open(os.path.join(run_dir, "rollouts.csv"), newline="") as f:
        reader = csv.DictReader(f)
        rows, duplicates = {}, []
        for row in reader:
            ep = int(row["episode"])
            if ep in rows:
                duplicates.append(ep)
            else:
                rows[ep] = row
        return list(reader.fieldnames or []), rows, sorted(set(duplicates))


def protocol_version(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            return json.load(f).get("protocol", {}).get("PROTOCOL_VERSION", "?")
    except (OSError, ValueError):
        return "?"


def _equal(a, b, float_tol):
    if a == b:
        return True
    if float_tol > 0:
        try:
            return abs(float(a) - float(b)) <= float_tol
        except ValueError:
            return False
    return False


def values_equal(column, v1_value, v11_value, float_tol=0.0):
    """Whether the v1.1 value of a compared v1 column reproduces the v1 value (CSV strings)."""
    if column == "error_message":  # the v1 part had no error, the extended window ended with an exception
        return v1_value == v11_value or (v1_value == "" and v11_value.startswith(EXTENDED_HOLD_PREFIX))
    return _equal(v1_value, v11_value, float_tol)


def _ids(episodes):
    return f"{episodes[:MAX_IDS]}" + (f" (+{len(episodes) - MAX_IDS} more)" if len(episodes) > MAX_IDS else "")


def compare_runs(v1_dir, v11_dir, float_tol=0.0, allow_subset=False, require_compare=False):
    """Returns (passed, report lines)."""
    v1_header, v1_rows, v1_dups = read_rollouts(v1_dir)
    v11_header, v11_rows, v11_dups = read_rollouts(v11_dir)
    v1_version, v11_version = protocol_version(v1_dir), protocol_version(v11_dir)
    lines = [f"v1 run:   {v1_dir} (protocol {v1_version}, {len(v1_rows)} episodes)",
             f"v1.1 run: {v11_dir} (protocol {v11_version}, {len(v11_rows)} episodes)"]
    problems = []

    if v1_version != V1_VERSION or v11_version != V11_VERSION:
        problems.append(f"the reference run must be protocol {V1_VERSION} and the checked run protocol "
                        f"{V11_VERSION} (got {v1_version} and {v11_version})")
    for name, dups in (("v1", v1_dups), ("v1.1", v11_dups)):
        if dups:
            problems.append(f"{len(dups)} episodes appear more than once in the {name} run: {_ids(dups)}")
    if v11_header[:len(v1_header)] != v1_header:
        missing = [c for c in v1_header if c not in v11_header]
        problems.append("the v1.1 header does not start with the v1 header"
                        + (f" (missing columns: {', '.join(missing)})" if missing else " (order differs)"))
    compared_columns = [c for c in v1_header if c not in EXCLUDED and c in v11_header]
    extra = [c for c in v11_header if c not in v1_header]
    lines.append(f"Compared columns: {len(compared_columns)} v1 columns (excluded: {', '.join(EXCLUDED)}; "
                 f"error_message may change from empty to '{EXTENDED_HOLD_PREFIX}...'); "
                 f"{len(extra)} new columns in the v1.1 run")

    not_in_v1 = sorted(set(v11_rows) - set(v1_rows))
    if not_in_v1:
        problems.append(f"{len(not_in_v1)} episodes of the v1.1 run are not in the v1 run: {_ids(not_in_v1)}")
    not_in_v11 = sorted(set(v1_rows) - set(v11_rows))
    if not_in_v11 and not allow_subset:
        problems.append(f"{len(not_in_v11)} episodes of the v1 run are missing from the v1.1 run (incomplete run? "
                        f"use --allow-subset for a partial run): {_ids(not_in_v11)}")
    episodes = sorted(set(v11_rows) & set(v1_rows))
    if not episodes:
        problems.append("no common episode")

    diffs = []  # (episode, column, v1 value, v1.1 value)
    for ep in episodes:
        a, b = v1_rows[ep], v11_rows[ep]
        for col in compared_columns:
            if not values_equal(col, a[col], b[col], float_tol):
                diffs.append((ep, col, a[col], b[col]))
    if "cmp_identical" in v11_header:
        not_identical = [ep for ep in episodes if v11_rows[ep]["cmp_identical"] not in ("", "True")]
        no_compare = [ep for ep in episodes if v11_rows[ep]["cmp_identical"] == ""]
        n_cmp = len(episodes) - len(no_compare)
        lines.append(f"Recorded comparison in the v1.1 run: {n_cmp - len(not_identical)}/{n_cmp} episodes identical"
                     + (f", {len(no_compare)} episodes without a comparison" if no_compare else ""))
        if not_identical:
            problems.append(f"cmp_identical is not True in the v1.1 run for episodes {_ids(not_identical)}")
        if require_compare and no_compare:
            problems.append(f"--require-compare: no recorded comparison (cmp_identical empty) in the v1.1 run for "
                            f"{len(no_compare)} episodes: {_ids(no_compare)}")
    elif require_compare:
        problems.append("--require-compare: the v1.1 run has no cmp_identical column (run without "
                        "--compare-recorded)")

    if diffs:
        bad_episodes = sorted({d[0] for d in diffs})
        by_column = {}
        for _, col, _, _ in diffs:
            by_column[col] = by_column.get(col, 0) + 1
        problems.append(f"{len(diffs)} differing values in {len(bad_episodes)} episodes; columns: "
                        + ", ".join(f"{c} ({k})" for c, k in sorted(by_column.items(), key=lambda x: -x[1])))
        lines.append("Differences (episode, column, v1 value, v1.1 value):")
        for ep, col, x, y in diffs[:MAX_LISTED]:
            lines.append(f"  ep {ep:4d}  {col:<28} {x!r:>26}  {y!r:>26}")
        if len(diffs) > MAX_LISTED:
            lines.append(f"  ... {len(diffs) - MAX_LISTED} more")

    compared = f"{len(episodes)} of {len(v1_rows)} v1 episodes compared"
    lines.append(f"{compared}" + (f" ({len(not_in_v11)} not in the v1.1 run, --allow-subset)"
                                  if not_in_v11 and allow_subset else "")
                 + f"; differing episodes: {len({d[0] for d in diffs})}")
    passed = not problems
    if passed:
        lines.append(f"PASS: all {len(compared_columns)} v1 columns identical, {compared}"
                     + (f" (float tolerance {float_tol:g})" if float_tol > 0 else " (exact)")
                     + ("; cmp_identical True in every compared episode" if require_compare else ""))
    else:
        lines.append("FAIL:")
        lines.extend(f"  - {p}" for p in problems)
    return passed, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("v1_run_dir", help="run folder of protocol v1 (e.g. results/expert_dev)")
    parser.add_argument("v11_run_dir", help="run folder of protocol v1.1 on the same evaluation set")
    parser.add_argument("--float-tol", type=float, default=0.0,
                        help="accept numbers that differ by at most this much (default 0: exact strings)")
    parser.add_argument("--allow-subset", action="store_true",
                        help="accept a v1.1 run that covers only part of the v1 episodes (default: all are required)")
    parser.add_argument("--require-compare", action="store_true",
                        help="require cmp_identical True in every compared episode (runs with --compare-recorded)")
    args = parser.parse_args(argv)
    passed, lines = compare_runs(args.v1_run_dir, args.v11_run_dir, args.float_tol, args.allow_subset,
                                 args.require_compare)
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
