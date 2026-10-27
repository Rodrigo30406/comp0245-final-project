"""Descriptive command tolerances; these do not certify control stability."""
import numpy as np
from .model import OUTPUT_NAMES

DEFAULT_TOLERANCES = [.05, .1, .2, .5, 1.]


def command_fidelity(truth, prediction, tolerances=DEFAULT_TOLERANCES):
    y = np.asarray(truth,dtype=float);p = np.asarray(prediction,dtype=float)
    tolerance = np.asarray(tolerances,dtype=float)
    if y.ndim!=2 or y.shape[1]!=9 or p.shape!=y.shape or not len(y):
        raise ValueError('Expected nonempty aligned (N,9) command arrays')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite commands')
    if tolerance.ndim!=1 or not len(tolerance) or not np.isfinite(tolerance).all() or np.any(tolerance<0) or np.any(np.diff(tolerance)<=0):
        raise ValueError('Expected increasing nonnegative physical tolerances')
    error=np.abs(p-y)
    return {'rows':len(y),'tolerances':tolerance.tolist(),
            'per_output':{name:{'unit':'rad/s' if j<2 else 'N*m',
                'p95_absolute_error':float(np.percentile(error[:,j],95)),
                'within_tolerance_fraction':[float(np.mean(error[:,j]<=t)) for t in tolerance]}
                for j,name in enumerate(OUTPUT_NAMES)},
            # Same numeric threshold, respective physical units per channel.
            'all_nine_within_tolerance_fraction':[float(np.mean(np.all(error<=t,axis=1))) for t in tolerance]}


def relative_command_fidelity(truth,prediction,training_scale,relative_tolerances=(.05,.1,.2,.5),floor_factors=(0.,.01,.05),zero_threshold=1e-6):
    """Relative fidelity on nonzero commands; zero-command diagnostics kept separate.

    Floors are descriptive fractions of training std, not calibrated control margins.
    """
    y=np.asarray(truth,dtype=float);p=np.asarray(prediction,dtype=float);scale=np.asarray(training_scale,dtype=float)
    command_fidelity(y,p)  # Shared shape/finiteness contract.
    if scale.shape!=(9,) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Invalid training scales')
    rt=np.asarray(relative_tolerances,dtype=float);floors=np.asarray(floor_factors,dtype=float)
    if (rt.ndim!=1 or not len(rt) or not np.isfinite(rt).all() or np.any(rt<0) or
        floors.ndim!=1 or not len(floors) or not np.isfinite(floors).all() or np.any(floors<0) or
        not np.isfinite(zero_threshold) or zero_threshold<0):
        raise ValueError('Invalid relative tolerances, floors or zero threshold')
    error=np.abs(p-y);active=np.abs(y)>zero_threshold
    out={}
    for j,name in enumerate(OUTPUT_NAMES):
        nonzero=active[:,j];zero=~nonzero
        out[name]={'nonzero_rows':int(nonzero.sum()),'zero_rows':int(zero.sum()),
            'nonzero_relative_within_fraction':[float(np.mean(error[nonzero,j]<=r*np.abs(y[nonzero,j]))) if nonzero.any() else None for r in rt],
            'zero_command_prediction_mean_abs':float(np.mean(np.abs(p[zero,j]))) if zero.any() else None,
            'zero_command_prediction_p95_abs':float(np.percentile(np.abs(p[zero,j]),95)) if zero.any() else None}
    stopped=~active[:,:2].any(axis=1)
    return {'relative_tolerances':rt.tolist(),'absolute_floor_factors_of_training_std':floors.tolist(),
            'zero_reference_threshold_physical_units':zero_threshold,'per_output':out,
            'base_stop_command_rows':int(stopped.sum()),
            'base_stop_prediction_mean_abs_by_side':np.mean(np.abs(p[stopped,:2]),axis=0).tolist() if stopped.any() else None,
            'mixed_profiles':[{'rtol':float(r),'atol_training_std_factor':float(f),
                'atol_physical_by_channel':(f*scale).tolist(),
                'per_output_within_fraction':np.mean(error<=f*scale+r*np.abs(y),axis=0).tolist(),
                'all_nine_within_fraction':float(np.mean(np.all(error<=f*scale+r*np.abs(y),axis=1)))}
                for f in floors for r in rt]}
