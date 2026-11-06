"""Evaluation-bank split: every episode of a dataset on the validation side except the last few, which the explicit split
format needs on the training side (they are never evaluated). For the out-of-distribution banks of 3 Nov 2026
(`experiment_specs/collection/ood_*.json`): 210 collected, 5 held back, 205 evaluated, like the development validations.

  python scripts/data/make_bank_split.py DATASET_DIR OUTPUT.json [HOLDBACK=5]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robot_learning.data.splits import inventory, inventory_fingerprint, validate_partition  # noqa: E402


def build(dataset, holdback=5):
    dataset = Path(dataset).resolve(); records = inventory(dataset)
    if holdback < 1 or holdback >= len(records):
        raise ValueError("holdback must leave episodes on both sides")
    split = {"training": records[-holdback:], "validation": records[:-holdback]}
    validate_partition(records, split)
    return {"dataset": str(dataset), "extra_datasets": [], "target_grid": None, "seed": None,
            "unit": "evaluation bank: every episode is evaluated except the last HOLDBACK, parked on the training side; no model trains on this dataset",
            "expert_failures_in_split": sum(1 for r in records if r.get("success") is not True),
            "validation_fraction": len(split["validation"]) / len(records), "inventory_sha256": inventory_fingerprint(records), **split}


if __name__ == "__main__":
    out = Path(sys.argv[2]); split = build(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(split, open(out, "w"), indent=1)
    ok = sum(1 for r in split["validation"] if r.get("success") is True)
    print(f"{out}: {len(split['validation'])} evaluated ({ok} expert successes), {len(split['training'])} held back")
