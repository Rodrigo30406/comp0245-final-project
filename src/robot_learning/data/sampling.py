"""Outcome-independent spatial stratification for the 64-episode pilot."""
import itertools
import numpy as np


def stratified_pilot(box, seed, candidates=2000):
    """One jittered target per 4x4x4 cell; balanced 13-case spatial holdout.

    Holdout selection uses cell geometry only, before targets or outcomes. Each
    axis contributes 3 or 4 holdout cells per level, all octants are represented,
    and each coarse 4x4x2 cell retains at least one training target. Maximin
    selection uses box-normalized coordinates so height has equal priority.
    """
    if type(seed) is not int or seed < 0 or type(candidates) is not int or candidates < 1:
        raise ValueError('Nonnegative seed and positive candidate count required')
    bounds = np.array([box[a] for a in 'xyz'], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Expected increasing finite XYZ bounds')
    cells = np.array(list(itertools.product(range(4), repeat=3)))
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    best = None
    for _ in range(candidates):
        levels = [rng.permutation(np.r_[np.repeat(np.arange(4), 3), rng.integers(4)]) for _ in range(3)]
        selected = np.column_stack(levels)
        if len(np.unique(selected, axis=0)) != 13:
            continue
        if len(np.unique(selected//2, axis=0)) != 8:
            continue
        coarse = selected.copy(); coarse[:, 2] //= 2
        if len(np.unique(coarse, axis=0)) != 13:
            continue
        centers = (selected+.5)/4
        distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)[np.triu_indices(13, 1)]
        score = (float(distances.min()), float(distances.mean()))
        if best is None or score > best[0]:
            best = (score, [lookup[tuple(cell)] for cell in selected])
    if best is None:
        raise ValueError('No balanced holdout found; increase candidate budget')
    rng_targets = np.random.default_rng(np.random.SeedSequence([seed, 100]))
    unit = (cells+rng_targets.random((64, 3)))/4
    targets = bounds[:, 0]+unit*(bounds[:, 1]-bounds[:, 0])
    validation = sorted(best[1])
    return {'targets': targets.tolist(), 'cells': cells.tolist(),
            'validation_indices': validation,
            'training_indices': [i for i in range(64) if i not in validation],
            'seed': seed, 'grid': [4, 4, 4], 'candidate_budget': candidates,
            'selection': 'geometry-only balanced maximin on normalized cell centers'}


def stratified_grid(box, grid, seed, candidates=32):
    """Balanced geometry-only holdout for larger grids; retain the frozen pilot.

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
    if tuple(dims) == (4, 4, 4):
        return stratified_pilot(box, seed)
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
