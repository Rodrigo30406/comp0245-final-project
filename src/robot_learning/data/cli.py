"""Collect reproducible expert datasets. Run with the RoboEnv Python environment."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from robot_learning.settings import PROJECT_ROOT, get_roboenv_root, load_project_env
from robot_learning.data.config import generate_scenarios, resolve_config
from robot_learning.data.storage import (
    provenance, read_json, sha256, snapshot_sources, summarize, utc_now, write_json,
)
from robot_learning.data.schema import dataset_schema
from robot_learning.data.scheduler import collect, validate_workers


def prepare(args):
    load_project_env()
    if args.roboenv_root:
        os.environ["ROBOENV_ROOT"] = str(Path(args.roboenv_root).expanduser().resolve())
    robo = Path(get_roboenv_root())
    current_provenance = provenance(PROJECT_ROOT, robo)
    if args.resume:
        if args.config or args.episodes is not None or args.max_steps is not None or args.gui:
            raise ValueError("Resume uses the saved configuration; do not supply config overrides")
        directory = Path(args.resume).expanduser().resolve()
        manifest = read_json(directory / "dataset.json")
        for filename in ("config.json", "scenarios.json", "schema.json"):
            if sha256(directory / filename) != manifest["file_hashes"][filename]:
                raise ValueError(f"Saved {filename} changed; create a new dataset instead")
        if read_json(directory / "provenance.json")["source_fingerprint"] != current_provenance["source_fingerprint"]:
            raise ValueError("Code, robot assets, packages or Python differ from the saved dataset; "
                             "restore that environment or create a new dataset")
        # A checkout may move without changing its content. Workers use the current root.
        manifest["last_roboenv_root"] = str(robo)
        cfg = read_json(directory / "config.json")
        validate_workers(args.workers, cfg)
        return directory, cfg, read_json(directory / "scenarios.json"), manifest, robo
    if not args.config:
        raise ValueError("Supply --config, or --resume for an existing dataset")
    source = Path(args.config).expanduser().resolve()
    requested = read_json(source)
    effective_request = json.loads(json.dumps(requested))
    if args.episodes is not None:
        effective_request["episodes"] = args.episodes
        targets = effective_request.get("targets")
        if targets is not None:
            # An explicit target list (the target-grid specs) is cut to its first N entries: the same prefix rule as
            # sampled scenarios. Growing it would need targets the spec does not contain.
            if args.episodes > len(targets):
                raise ValueError(f"--episodes {args.episodes} exceeds the {len(targets)} explicit targets of the spec")
            effective_request["targets"] = targets[:args.episodes]
    if args.max_steps is not None:
        effective_request["max_steps"] = args.max_steps
    if args.gui:
        effective_request["headless"] = False
    cfg = resolve_config(effective_request)
    validate_workers(args.workers, cfg)
    scenarios = generate_scenarios(cfg)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", cfg["name"]).strip("_") or "dataset"
    dataset_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + name + "_" + uuid.uuid4().hex[:8]
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else PROJECT_ROOT / "datasets"
    directory = output / dataset_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "episodes").mkdir()
    (directory / "logs").mkdir()
    shutil.copy2(source, directory / "config.requested.json")
    write_json(directory / "request.json", {"config_path": str(source), "argv": sys.argv,
                                           "effective_request": effective_request})
    write_json(directory / "config.json", cfg)
    write_json(directory / "scenarios.json", scenarios)
    write_json(directory / "schema.json", dataset_schema())
    write_json(directory / "provenance.json", current_provenance)
    snapshot_sources(directory, PROJECT_ROOT, robo)
    manifest = {"schema_version": dataset_schema()["version"], "dataset_id": dataset_id, "created_at": utc_now(),
                "status": "planned", "last_roboenv_root": str(robo),
                "file_hashes": {name: sha256(directory / name)
                                for name in ("config.json", "scenarios.json", "schema.json")}}
    write_json(directory / "dataset.json", manifest)
    write_json(directory / "summary.json", summarize(directory, scenarios, cfg))
    return directory, cfg, scenarios, manifest, robo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON experiment specification")
    parser.add_argument("--resume", help="Continue a saved dataset using its frozen configuration")
    parser.add_argument("--output-dir", help="Default: datasets/ at the project root")
    parser.add_argument("--roboenv-root", help="Override the asset root from .env / sibling detection")
    parser.add_argument("--episodes", type=int, help="Override the number of episodes")
    parser.add_argument("--max-steps", type=int, help="Override the global physics-step cap")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent episode processes (default: 1; headless only for >1)")
    parser.add_argument("--gui", action="store_true", help="Show episodes sequentially with PyBullet GUI")
    parser.add_argument("--plan-only", action="store_true", help="Freeze configuration/scenarios without running simulations")
    args = parser.parse_args()
    try:
        validate_workers(args.workers)
        directory, cfg, scenarios, manifest, robo = prepare(args)
        print(f"DATASET: {directory}", flush=True)
        if args.plan_only:
            print(f"Planned {len(scenarios)} episodes; resume this directory to collect them.")
            return 0
        return collect(directory, cfg, scenarios, manifest, robo, workers=args.workers)
    except KeyboardInterrupt:
        print("Interrupted. Resume the saved dataset to continue.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
