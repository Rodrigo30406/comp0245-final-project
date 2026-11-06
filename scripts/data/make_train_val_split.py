"""Geometry-holdout split of one or more reaching datasets collected on the target grid.

Every episode is mapped to its grid cell by its target coordinates (the collection specs copy the grid targets
verbatim, so a re-collection whose episodes are an arbitrary subset of the grid lands on the right side).
An episode goes to validation when its target is one of the grid's held-out cells (`validation_indices`, 205 of the
1024 cells of the 16x8x8 grid), whatever its start pose: no target geometry is seen both in training and in
validation. Expert failures stay in the split (every episode is accounted for); the trainers drop them.

  python scripts/data/make_train_val_split.py OUTPUT.json PRIMARY_DATASET [EXTRA_DATASET ...]

The training mixture of the report:
  PRIMARY = the nominal-start DART collection (reaching_nominal_dart)
  EXTRAS  = the random-start DART collection (reaching_random_start_dart)
            and its expert-failure replacement (reaching_recollect_failures_dart)
Random-start validation bank: PRIMARY = reaching_validation_random_start_dart (only its 205 validation episodes are used).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robot_learning.data.splits import extra_inventory, inventory, inventory_fingerprint, validate_partition  # noqa: E402

TARGET_GRID = "experiment_specs/collection/target_grid_16x8x8.json"


def grid_lookup(targets):
    """target xyz -> index of its target-grid cell, by exact coordinates."""
    table = {tuple(np.round(t, 9)): i for i, t in enumerate(targets)}

    def lookup(target):
        key = tuple(np.round(np.asarray(target, dtype=float), 9))
        if key not in table:
            raise ValueError(f"target {target} is not a target-grid target")
        return table[key]
    return lookup


def build(primary, extras):
    grid = json.loads((ROOT / TARGET_GRID).read_text())
    lookup, held_out = grid_lookup(grid["targets"]), set(grid["validation_indices"])
    saved = {"dataset": str(Path(primary).resolve()), "extra_datasets": [str(Path(e).resolve()) for e in extras]}
    records = inventory(saved["dataset"]) + extra_inventory(saved)
    training = [r for r in records if lookup(r["target"]) not in held_out]
    validation = [r for r in records if lookup(r["target"]) in held_out]
    validate_partition(records, {"training": training, "validation": validation})
    return {**saved, "target_grid": TARGET_GRID, "seed": None,
            "unit": "whole episode; geometry holdout: an episode is validation when its target is a held-out grid cell",
            "expert_failures_in_split": sum(1 for r in records if r.get("success") is not True),
            "note": "expert failures are kept in the split for completeness; trainers drop them with keep_expert_failures=false",
            "validation_fraction": len(validation) / len(records), "inventory_sha256": inventory_fingerprint(records),
            "training": training, "validation": validation}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    split = build(sys.argv[2], sys.argv[3:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    ok = {k: sum(1 for r in split[k] if r.get("success") is True) for k in ("training", "validation")}
    print(f"{out}: {len(split['training'])} training ({ok['training']} expert successes) / "
          f"{len(split['validation'])} validation ({ok['validation']}) over {1 + len(split['extra_datasets'])} datasets")
