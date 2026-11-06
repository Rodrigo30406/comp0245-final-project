"""Complete synthetic PC analyses; every fixture is software verification only."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "results" / "step6"), str(TASK)]

import gp_smoke as fixture  # noqa: E402
from gp_common import ar, write_json  # noqa: E402
from evaluate_policy import PROTOCOL, ROW_COLUMNS  # noqa: E402
from PC_mechanisms import analyze, OLD, NEW, SEEDS, COUNTS  # noqa: E402
from pc_freeze import check_freeze  # noqa: E402

CASES = {
    "confirmed": (True, True, True, False),
    "bootstrap_not_positive": (False, True, True, False),
    "hard_nonpositive": (True, False, True, False),
    "confirm_nonpositive": (True, False, True, False),
    "risk_fail": (True, True, False, False),
    "mirror_worse": (False, False, True, True),
}


def create_fixture(case):
    if case not in CASES:
        raise ValueError(case)
    root = HERE / "smoke" / case
    for name, n in COUNTS.items():
        fixture.write_csv(root / "results" / "step3" / f"cache_ref_{name}.csv",
                          ar.REF_COLUMNS, fixture.reference_rows(n))
        specs = [(OLD, "C3", seed) for seed in SEEDS[name]]
        specs += [(NEW, "P", seed) for seed in SEEDS[name]]
        specs += [("expert", "expert", None)]
        for model, arm, seed in specs:
            stem = f"expert_v11_{name}" if seed is None else f"{model}_s{seed}_{name}"
            folder = root / "results" / stem
            baseline = round(n * (.8 if case == "mirror_worse" else .3))
            n_success = n if arm == "expert" else baseline
            if arm == "P":
                n_success = round(n * .9)
                if case == "mirror_worse":
                    n_success = round(n * .1)
                elif case == "bootstrap_not_positive":
                    n_success = baseline + 1
                elif case == "hard_nonpositive" and name == "hard":
                    n_success = baseline
                elif case == "confirm_nonpositive" and name == "confirm":
                    n_success = baseline
            rows = []
            for episode in range(n):
                outcome = ("success" if episode < n_success else
                           "crash" if case == "risk_fail" and arm == "P" else "extended_not_reached")
                rows.append(fixture.rollout_row(episode, outcome, arm))
            fixture.write_csv(folder / "rollouts.csv", ROW_COLUMNS, rows)
            config = {"args": {"policy": "expert" if arm == "expert" else "policies/bc_policy.py:make_policy",
                               "policy_arg": None if arm == "expert" else f"trained/{model}_s{seed}.npz",
                               "eval_set": f"synthetic/{name}", "no_early_stop": True},
                      "protocol": PROTOCOL, "eval_set_config": {"n_episodes": n},
                      "episodes": list(range(n)), "synthetic": True,
                      "purpose": "software smoke; never experimental evidence"}
            fixture.write_json(folder / "config.json", config)
            fixture.write_once(folder / "summary.txt", "SYNTHETIC SOFTWARE SMOKE ONLY\n")
    fixture.write_json(root / "fixture_manifest.json", {"synthetic": True, "case": case,
                       "counts": COUNTS, "seeds": SEEDS, "expected_conditions": CASES[case]})
    return root


def run(case):
    check_freeze()
    root = create_fixture(case)
    result = analyze(task_root=root, allow_confirm=True, smoke=True)
    decision = result["decision"]
    actual = tuple(decision[k] for k in ("condition_1", "condition_2", "condition_3", "mirror_worse"))
    if actual != CASES[case]:
        raise AssertionError(f"Synthetic numerical verification failed: {case}: {actual} != {CASES[case]}")
    if decision["confirmed"] != all(CASES[case][:3]):
        raise AssertionError("Synthetic conjunction/verdict mismatch")
    verification = {"case": case, "synthetic": True, "passed": True,
                    "actual_conditions": actual, "decision": decision,
                    "bootstrap_rule": result["bootstrap_rule"],
                    "combined_success_diff": result["combined"]["success"]["diff"],
                    "combined_bootstrap_ci": [result["combined"]["success"][key] for key in ("lo", "hi")],
                    "combined_welch_ci": [result["combined"]["success"][key] for key in ("welch_lo", "welch_hi")],
                    "combined_risk_diff": result["combined"]["risk"]["diff"]}
    write_json(root / "verification.json", verification)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    args = parser.parse_args(argv)
    run(args.case)


if __name__ == "__main__":
    main()
