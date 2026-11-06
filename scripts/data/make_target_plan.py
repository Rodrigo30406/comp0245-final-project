"""Regenerate the stratified target plan of the report and compare it with the frozen copy.

The plan is one jittered target per cell of a 16x8x8 grid over the target box (x, y in [-2, 2] m, z in [0.3, 0.8] m)
and a geometry-only holdout of 205 cells (20 %), chosen before any episode exists (robot_learning.data.sampling).
The collection specs copy its 1024 targets verbatim (nominal start: once; random starts: three times), and
scripts/data/make_geometry_split.py reads its `validation_indices`.

  python scripts/data/make_target_plan.py            # regenerate and compare (prints OK or the differences)
  python scripts/data/make_target_plan.py OUT.json   # also write the regenerated plan
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robot_learning.data.sampling import stratified_grid  # noqa: E402

FROZEN = ROOT / "experiment_specs/collection/stratified1024_plan.json"
BOX = {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [0.3, 0.8]}
GRID, SEED, CANDIDATES = [16, 8, 8], 20261026, 32


if __name__ == "__main__":
    plan = stratified_grid(BOX, GRID, SEED, candidates=CANDIDATES)
    frozen = json.loads(FROZEN.read_text())
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(plan, indent=1))
    same = {k: plan[k] == frozen[k] for k in ("targets", "cells", "validation_indices", "training_indices")}
    print("OK: identical to the frozen plan" if all(same.values()) else f"DIFFERS from the frozen plan: {same}")
