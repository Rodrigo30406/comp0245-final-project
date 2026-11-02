"""Validated immutable episode inventory and deterministic episode-level split."""
import hashlib
import json
from pathlib import Path
import numpy as np
from robot_learning.data.storage import read_json, sha256
from .model import INPUT_NAMES, ACTION_NAMES, independent_actions


def inventory(directory):
    directory = Path(directory).resolve()
    schema = read_json(directory / "schema.json")
    if (schema['version'] != 2 or schema['observation_names'] != INPUT_NAMES or
        schema['action_names'] != ACTION_NAMES):
        raise ValueError("Dataset schema does not match baseline 21-input/11-stored-action contract")
    scenarios = read_json(directory / "scenarios.json")
    if len(scenarios) < 2:
        raise ValueError("At least two complete episodes required")
    records = []; seen_ids = set(); seen_conditions = set()
    for scenario in scenarios:
        eid = scenario['episode_id']
        if not isinstance(eid, str) or Path(eid).name != eid or eid in seen_ids:
            raise ValueError("Invalid or duplicate episode ID")
        seen_ids.add(eid)
        meta_path = directory / 'episodes' / f'{eid}.json'
        meta = read_json(meta_path)
        if meta['status'] != 'complete' or meta['episode_id'] != eid:
            raise ValueError(f"Incomplete episode: {eid}; no silent filtering")
        if meta['target'] != scenario['target'] or meta['scenario_id'] != scenario['scenario_id']:
            raise ValueError(f"Scenario/metadata mismatch: {eid}")
        # Same physical initial condition/goal must not cross splits under different IDs/seeds.
        condition = json.dumps([scenario['target'], meta['actual_initial_observation'][:18]], sort_keys=True)
        if condition in seen_conditions:
            raise ValueError("Duplicate target/initial condition; group duplicates before splitting")
        seen_conditions.add(condition)
        if meta['data_file'] != f'{eid}.npz':
            raise ValueError("Unexpected episode data filename")
        path = directory / 'episodes' / meta['data_file']
        digest = sha256(path)
        if digest != meta['data_sha256']:
            raise ValueError(f"Episode checksum mismatch: {eid}")
        records.append({'episode_id': eid, 'scenario_id': scenario['scenario_id'],
                        'target': scenario['target'], 'success': meta['success'],
                        'termination': meta['termination'], 'rows': meta['recorded_transitions'],
                        'npz_sha256': digest, 'metadata_sha256': sha256(meta_path)})
    return records


def extra_inventory(saved):
    """Records of the optional `extra_datasets` of an explicit split, each tagged with its directory.

    A split may add whole datasets (same schema and episode configuration) on top of its
    primary dataset; their records carry a 'dataset' key so identical episode IDs never collide.
    """
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
    """Directory an inventory record was read from (primary dataset unless tagged)."""
    return Path(record["dataset"]) if "dataset" in record else Path(default)


def split_episodes(records, validation_fraction, seed):
    if not 0 < validation_fraction < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid validation fraction or seed")
    n = len(records); n_val = int(np.ceil(n * validation_fraction))
    if n_val < 1 or n_val >= n:
        raise ValueError("Split must leave at least one episode on each side")
    order = np.random.default_rng(seed).permutation(n)
    val = set(order[:n_val].tolist())
    return {name: [r for i, r in enumerate(records) if (i in val) == is_val]
            for name, is_val in [('training', False), ('validation', True)]}


def load_episode(directory, record, window=None):
    path = Path(directory) / 'episodes' / f"{record['episode_id']}.npz"
    if sha256(path) != record['npz_sha256']:
        raise ValueError("Episode changed after inventory")
    with np.load(path, allow_pickle=False) as data:
        if window is not None:
            from .windows import validate_episode_sampling
            validate_episode_sampling(data, read_json(path.with_suffix('.json')), window)
        x = data['observation'].astype(np.float64)
        y = independent_actions(data['action']).astype(np.float64)
        phases = data['expert_phase'].copy()
    if (x.shape != (record['rows'], 21) or y.shape != (len(x), 9) or
        phases.shape != (len(x),) or len(x) == 0 or not np.isfinite(x).all()):
        raise ValueError("Invalid shapes, empty episode or non-finite inputs")
    target = np.asarray(record['target'], dtype=np.float32).astype(np.float64)
    if not np.all(x[:, -3:] == target):
        raise ValueError("Observations must contain the original scenario target")
    return x, y, phases


def inventory_fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def validate_partition(records, partition):
    """Reject missing, duplicated, overlapping or stale explicit split records."""
    if set(partition) != {'training', 'validation'} or any(not partition[k] for k in partition):
        raise ValueError('Explicit split needs nonempty training and validation')
    key = lambda r: (r.get('dataset'), r['episode_id'])
    lookup = {key(r): r for r in records}
    chosen = partition['training'] + partition['validation']
    ids = [key(r) for r in chosen]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError('Explicit split is overlapping, duplicated or incomplete')
    if any(lookup[key(r)] != r for r in chosen):
        raise ValueError('Explicit split contains stale or foreign records')
    return partition


def drop_expert_failures(partition):
    """Keep only the episodes whose expert run succeeded (record 'success' true) on both sides of a validated split.
    Rule of 2 Nov 2026: policies are trained and selected on expert successes only; failed scenarios are re-collected
    with another start pose / noise seed and discarded if they fail again. Returns (partition, counts)."""
    kept = {k: [r for r in v if r.get('success') is True] for k, v in partition.items()}
    counts = {k: {'kept': len(kept[k]), 'dropped_expert_failures': len(partition[k]) - len(kept[k])} for k in partition}
    if any(not kept[k] for k in kept):
        raise ValueError('Dropping expert failures emptied one side of the split')
    return kept, counts


def nested_partition(records, previous, fraction, seed):
    """Extend a frozen partition without moving any previous episode across sides."""
    lookup = {r['episode_id']: r for r in records}
    old = previous['training'] + previous['validation']
    if len({r['episode_id'] for r in old}) != len(old):
        raise ValueError('Previous partition overlaps')
    for record in old:
        current = lookup.get(record['episode_id'])
        if current is None or any(current[k] != record[k] for k in ['scenario_id', 'target']):
            raise ValueError('Nested datasets must contain identical previous scenarios')
    old_ids = {r['episode_id'] for r in old}
    added = [r for r in records if r['episode_id'] not in old_ids]
    count = int(np.ceil(len(records)*fraction)) - len(previous['validation'])
    if not 0 <= count <= len(added):
        raise ValueError('Cannot retain previous partition at requested fraction')
    rng = np.random.default_rng(np.random.SeedSequence([seed, len(old), len(records)]))
    extra_validation = {added[i]['episode_id'] for i in rng.permutation(len(added))[:count]}
    validation_ids = {r['episode_id'] for r in previous['validation']} | extra_validation
    return validate_partition(records, {name: [r for r in records if (r['episode_id'] in validation_ids)==is_val]
        for name,is_val in [('training',False),('validation',True)]})
