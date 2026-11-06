"""Stratified target plan: one jittered target per cell of a grid over the target box, and a geometry-only holdout.

The holdout is chosen from cell geometry alone, before any target is jittered or any episode is simulated, so no
outcome can leak into the choice. Used once to freeze experiment_specs/collection/stratified1024_plan.json
(grid 16x8x8 over x, y in [-2, 2] m and z in [0.3, 0.8] m, seed 20261026); scripts/data/make_target_plan.py
regenerates and checks it.
"""
import itertools
import numpy as np


def stratified_grid(box, grid, seed, candidates=32):
    """Balanced geometry-only holdout (20 % of the cells) for an even grid.

    Binary feasibility constraints ensure balanced axis levels, all octants, and
    at most one held-out cell in each adjacent vertical pair. Random objectives
    create candidate subsets; choose maximin normalized cell-center separation.
    No target jitter, robot state, command or rollout outcome enters selection.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csc_matrix
    dims = np.asarray(grid)
    if dims.shape != (3,) or not np.issubdtype(dims.dtype, np.integer) or np.any(dims < 2) or np.any(dims % 2):
        raise ValueError('Three positive even integer grid dimensions required')
    if type(seed) is not int or seed < 0 or type(candidates) is not int or candidates < 1:
        raise ValueError('Nonnegative seed and positive candidate count required')
    bounds = np.array([box[a] for a in 'xyz'], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Expected finite increasing XYZ bounds')
    cells = np.array(list(itertools.product(*(range(int(n)) for n in dims))))
    n = len(cells); nval = int(np.ceil(.2*n)); rows = [np.ones(n)]; lower = [nval]; upper = [nval]
    for axis, size in enumerate(dims):
        for level in range(size):
            rows.append((cells[:, axis] == level).astype(float))
            lower.append(nval//size); upper.append(int(np.ceil(nval/size)))
    octants = cells//(dims//2)
    for octant in itertools.product(range(2), repeat=3):
        rows.append(np.all(octants == octant, axis=1).astype(float)); lower.append(1); upper.append(nval)
    coarse = cells.copy(); coarse[:, 2] //= 2
    for cell in np.unique(coarse, axis=0):
        rows.append(np.all(coarse == cell, axis=1).astype(float)); lower.append(0); upper.append(1)
    matrix = csc_matrix(np.array(rows))
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 200]))
    best = None
    for _ in range(candidates):
        result = milp(rng.random(n), integrality=np.ones(n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper), options={'time_limit': 10.})
        if result.x is None:
            continue
        selected = np.flatnonzero(result.x > .5)
        values = matrix@np.isin(np.arange(n), selected).astype(float)
        if np.any(values < np.array(lower)-1e-7) or np.any(values > np.array(upper)+1e-7):
            raise ValueError('Invalid spatial holdout solver result')
        centers = (cells[selected]+.5)/dims
        distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)[np.triu_indices(nval, 1)]
        score = (float(distances.min()), float(distances.mean()))
        if best is None or score > best[0]:
            best = (score, selected)
    if best is None:
        raise ValueError('No feasible spatially balanced holdout')
    rng_targets = np.random.default_rng(np.random.SeedSequence([seed, n, 100]))
    unit = (cells+rng_targets.random((n, 3)))/dims
    targets = bounds[:, 0]+unit*(bounds[:, 1]-bounds[:, 0])
    selected = set(best[1].tolist())
    return {'targets': targets.tolist(), 'cells': cells.tolist(), 'grid': dims.tolist(),
            'training_indices': [i for i in range(n) if i not in selected],
            'validation_indices': sorted(selected), 'seed': seed, 'candidate_budget': candidates,
            'selection': 'balanced binary constraints then geometry-only maximin, normalized cell centers',
            'nested': False}
