"""Episode inventory of a collected dataset and episode-level train/validation splits.

A split is always made of whole episodes (never rows), records every episode of every dataset it uses (expert failures
included, so nothing is silently dropped), and carries a fingerprint of the inventory so a changed dataset is detected.
Trainers remove the expert failures afterwards with `drop_expert_failures` (rule: train and select on expert successes).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import dataset_schema
from .storage import read_json, sha256


def inventory(directory):
    """One validated record per episode of a complete dataset (raises on any incomplete or altered episode)."""
    directory = Path(directory).resolve()
    schema, expected = read_json(directory / "schema.json"), dataset_schema()
    if (schema["version"] != expected["version"] or schema["observation_names"] != expected["observation_names"]
            or schema["action_names"] != expected["action_names"]):
        raise ValueError("Dataset schema does not match the 21-observation / 11-action contract")
    scenarios = read_json(directory / "scenarios.json")
    if len(scenarios) < 2:
        raise ValueError("At least two complete episodes required")
    records, seen_ids, seen_conditions = [], set(), set()
    for scenario in scenarios:
        eid = scenario["episode_id"]
        if not isinstance(eid, str) or Path(eid).name != eid or eid in seen_ids:
            raise ValueError("Invalid or duplicate episode ID")
        seen_ids.add(eid)
        meta_path = directory / "episodes" / f"{eid}.json"
        meta = read_json(meta_path)
        if meta["status"] != "complete" or meta["episode_id"] != eid:
            raise ValueError(f"Incomplete episode: {eid}; no silent filtering")
        if meta["target"] != scenario["target"] or meta["scenario_id"] != scenario["scenario_id"]:
            raise ValueError(f"Scenario/metadata mismatch: {eid}")
        # The same physical start and goal must not appear twice (it could end up on both sides of a split).
        condition = json.dumps([scenario["target"], meta["actual_initial_observation"][:18]], sort_keys=True)
        if condition in seen_conditions:
            raise ValueError("Duplicate target/initial condition; group duplicates before splitting")
        seen_conditions.add(condition)
        if meta["data_file"] != f"{eid}.npz":
            raise ValueError("Unexpected episode data filename")
        digest = sha256(directory / "episodes" / meta["data_file"])
        if digest != meta["data_sha256"]:
            raise ValueError(f"Episode checksum mismatch: {eid}")
        records.append({"episode_id": eid, "scenario_id": scenario["scenario_id"], "target": scenario["target"],
                        "success": meta["success"], "termination": meta["termination"],
                        "rows": meta["recorded_transitions"], "npz_sha256": digest,
                        "metadata_sha256": sha256(meta_path)})
    return records


def extra_inventory(saved):
    """Records of a split's `extra_datasets`, each tagged with its directory so identical episode IDs never collide."""
    extras = saved.get("extra_datasets", [])
    if not isinstance(extras, list) or len(set(map(str, extras))) != len(extras):
        raise ValueError("extra_datasets must be a list of distinct dataset directories")
    records = []
    for extra in extras:
        directory = Path(extra).resolve()
        if str(directory) == str(Path(saved["dataset"]).resolve()):
            raise ValueError("extra_datasets must not repeat the primary dataset")
        records.extend({**r, "dataset": str(directory)} for r in inventory(directory))
    return records


def record_directory(record, default):
    """Directory an inventory record was read from (the primary dataset unless tagged)."""
    return Path(record["dataset"]) if "dataset" in record else Path(default)


def inventory_fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def split_episodes(records, validation_fraction, seed):
    """Uniform random episode holdout (used where every scenario is distinct, e.g. the obstacle sets)."""
    if not 0 < validation_fraction < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid validation fraction or seed")
    n = len(records)
    n_val = int(np.ceil(n * validation_fraction))
    if n_val < 1 or n_val >= n:
        raise ValueError("Split must leave at least one episode on each side")
    validation = set(np.random.default_rng(seed).permutation(n)[:n_val].tolist())
    return {name: [r for i, r in enumerate(records) if (i in validation) == is_val]
            for name, is_val in (("training", False), ("validation", True))}


def validate_partition(records, partition):
    """Reject missing, duplicated, overlapping or stale explicit split records."""
    if set(partition) != {"training", "validation"} or any(not partition[k] for k in partition):
        raise ValueError("Explicit split needs nonempty training and validation")
    key = lambda r: (r.get("dataset"), r["episode_id"])
    lookup = {key(r): r for r in records}
    chosen = partition["training"] + partition["validation"]
    ids = [key(r) for r in chosen]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError("Explicit split is overlapping, duplicated or incomplete")
    if any(lookup[key(r)] != r for r in chosen):
        raise ValueError("Explicit split contains stale or foreign records")
    return partition


def drop_expert_failures(partition):
    """Keep only the episodes whose expert run succeeded, on both sides of a validated split.

    Policies are trained and selected on expert successes only: a failed expert episode (navigation timeout, or the
    expert's own stop declaring 'closest pose reached' short of the target) would teach the failure. Failed scenarios
    are re-collected with another start pose / noise seed instead (see docs/01_data_collection.md).
    Returns (partition, counts)."""
    kept = {k: [r for r in v if r.get("success") is True] for k, v in partition.items()}
    counts = {k: {"kept": len(kept[k]), "dropped_expert_failures": len(partition[k]) - len(kept[k])} for k in partition}
    if any(not kept[k] for k in kept):
        raise ValueError("Dropping expert failures emptied one side of the split")
    return kept, counts
