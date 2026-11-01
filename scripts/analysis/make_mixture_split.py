"""Build a mixture split: one primary dataset plus extra datasets, validation = the plan's held-out geometries.

Rule (same as runs/splits/e3_mixture_4096.json): episode i of every dataset demonstrates plan target i % 1024, so
an episode is validation when that plan target is in the clean validation set of the baseline split
(runs/training/mlp_stratified1024_torch_baseline/split.json, 205 of 1024 geometries). No episode of a held-out
geometry, from any start pose, is ever trained on.

    python scripts/analysis/make_mixture_split.py OUTPUT.json PRIMARY_DATASET EXTRA_DATASET [EXTRA ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.learning.dataset import inventory, extra_inventory, inventory_fingerprint, validate_partition  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "runs/training/mlp_stratified1024_torch_baseline/split.json"
PLAN = "experiment_specs/collection/stratified1024_plan.json"


def build(primary, extras):
    primary = Path(primary).resolve(); extras = [str(Path(e).resolve()) for e in extras]
    held_out = {int(r["episode_id"].split("_")[1]) % 1024 for r in json.load(open(CLEAN))["validation"]}
    if len(held_out) != 205:
        raise ValueError("Clean validation split must hold out 205 geometries")
    saved = {"dataset": str(primary), "extra_datasets": extras}
    records = inventory(primary) + extra_inventory(saved)
    training = [r for r in records if int(r["episode_id"].split("_")[1]) % 1024 not in held_out]
    validation = [r for r in records if int(r["episode_id"].split("_")[1]) % 1024 in held_out]
    validate_partition(records, {"training": training, "validation": validation})
    return {**saved, "plan": PLAN, "seed": None,
            "unit": "whole episode; geometry-only plan holdout applied per dataset (episode i -> plan target i % 1024)",
            "validation_fraction": len(validation) / len(records), "inventory_sha256": inventory_fingerprint(records),
            "training": training, "validation": validation}


if __name__ == "__main__":
    out = Path(sys.argv[1]); split = build(sys.argv[2], sys.argv[3:])
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(out, "w"), indent=1)
    print(f"{out}: {len(split['training'])} training / {len(split['validation'])} validation episodes over {1 + len(split['extra_datasets'])} datasets")
