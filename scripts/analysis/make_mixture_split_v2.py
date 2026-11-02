"""Mixture v2 splits (§6z): the geometry holdout of make_mixture_split.py, but each episode is mapped to its plan target by
the target coordinates instead of by `episode index % 1024`, so replacement collections (whose episodes are an arbitrary
subset of the plan targets) land on the right side. Expert failures stay in the split (the trainers drop them with
`keep_expert_failures: false`), so every episode of every dataset is accounted for.

  python scripts/analysis/make_mixture_split_v2.py OUTPUT.json PRIMARY_DATASET [EXTRA_DATASET ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.learning.dataset import inventory, extra_inventory, inventory_fingerprint, validate_partition  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "runs/training/mlp_stratified1024_torch_baseline/split.json"
PLAN = "experiment_specs/collection/stratified1024_plan.json"


def plan_index(targets):
    """target xyz -> plan target index, by exact coordinates (the collection specs copy the plan's targets verbatim)."""
    table = {tuple(np.round(t, 9)): i for i, t in enumerate(targets)}

    def lookup(target):
        key = tuple(np.round(np.asarray(target, dtype=float), 9))
        if key not in table:
            raise ValueError(f"target {target} is not a plan target")
        return table[key]
    return lookup


def build(primary, extras):
    primary = Path(primary).resolve(); extras = [str(Path(e).resolve()) for e in extras]
    plan = json.load(open(ROOT / PLAN)); lookup = plan_index(plan["targets"])
    held_out = {int(r["episode_id"].split("_")[1]) % 1024 for r in json.load(open(CLEAN))["validation"]}
    if len(held_out) != 205:
        raise ValueError("Clean validation split must hold out 205 geometries")
    saved = {"dataset": str(primary), "extra_datasets": extras}
    records = inventory(primary) + extra_inventory(saved)
    for r in records:  # the nominal dataset follows index % 1024 exactly; check that the coordinate rule agrees
        if r.get("dataset") is None and lookup(r["target"]) != int(r["episode_id"].split("_")[1]) % 1024:
            raise ValueError(f"primary episode {r['episode_id']} does not demonstrate plan target index % 1024")
    training = [r for r in records if lookup(r["target"]) not in held_out]
    validation = [r for r in records if lookup(r["target"]) in held_out]
    validate_partition(records, {"training": training, "validation": validation})
    failures = sum(1 for r in records if r.get("success") is not True)
    return {**saved, "plan": PLAN, "seed": None,
            "unit": "whole episode; geometry-only plan holdout per dataset, episode -> plan target by target coordinates (v2)",
            "expert_failures_in_split": failures,
            "note": "expert failures are kept in the split for completeness; trainers drop them with keep_expert_failures=false",
            "validation_fraction": len(validation) / len(records), "inventory_sha256": inventory_fingerprint(records),
            "training": training, "validation": validation}


if __name__ == "__main__":
    out = Path(sys.argv[1]); split = build(sys.argv[2], sys.argv[3:])
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(out, "w"), indent=1)
    ok_t = sum(1 for r in split["training"] if r.get("success") is True); ok_v = sum(1 for r in split["validation"] if r.get("success") is True)
    print(f"{out}: {len(split['training'])} training ({ok_t} expert successes) / {len(split['validation'])} validation ({ok_v}) over {1 + len(split['extra_datasets'])} datasets")
