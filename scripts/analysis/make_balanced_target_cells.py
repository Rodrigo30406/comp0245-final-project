"""Balanced coarser target grids for the data-size series (6 Nov 2026).

The random-subset series (training_episode_cap) cuts the fixed 16x8x8 design at random, so its small points have holes
and repeated regions and are compared with the only balanced design, the full one. Here coverage is the parameter: the
fine grid is grouped into coarse blocks and ONE training cell is drawn in each block, giving a balanced grid of N
targets; the trainer then keeps every training episode of those targets (nominal start + three random starts), via
the egocentric option training_target_cells. The validation cells (205) are never chosen and validation is unchanged.

Blocks (fine cells per block, x * y * z) are chosen so that every block contains a vertical pair of fine cells, and the
holdout keeps a training cell in every vertical pair, so each block has at least one training cell:
  128 targets: 2x2x2 blocks (coarse grid 8x4x4)
  256 targets: 2x1x2 blocks (8x8x4)
  512 targets: 1x1x2 blocks (16x8x4)

  python scripts/analysis/make_balanced_target_cells.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRID = "experiment_specs/collection/stratified1024_plan.json"
BLOCKS = {128: (2, 2, 2), 256: (2, 1, 2), 512: (1, 1, 2)}
SEED = 20261106


def balanced_cells(grid, block, rng):
    cells = np.array(grid["cells"])
    validation = set(grid["validation_indices"])
    coarse = cells // np.array(block)
    chosen = []
    for key in sorted({tuple(c) for c in coarse}):
        members = [i for i in np.flatnonzero(np.all(coarse == key, axis=1)) if i not in validation]
        if not members:
            raise ValueError(f"coarse block {key} has no training cell")
        chosen.append(int(rng.choice(members)))
    return sorted(chosen)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    grid = json.loads((ROOT / GRID).read_text())
    cells = np.array(grid["cells"])
    for n, block in BLOCKS.items():
        chosen = balanced_cells(grid, block, np.random.default_rng([SEED, n]))
        assert len(chosen) == n, (n, len(chosen))
        levels = {axis: np.bincount(cells[chosen][:, a] // block[a], minlength=[16, 8, 8][a] // block[a]).tolist()
                  for a, axis in enumerate("xyz")}
        path = out / f"balanced_cells_{n}.json"
        path.write_text(json.dumps({"target_grid": GRID, "targets": n, "block_xyz": block, "seed": SEED,
                                    "cells": chosen, "per_coarse_level": levels}, indent=1))
        print(f"{path}: {n} cells, one per {block} block; targets per coarse level {levels}")
